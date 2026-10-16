"""Train a patient-grouped TvsN classifier after paper-style ComBat correction.

ComBat is fit on source-training and target-adaptation samples only. Target-test
values are corrected using the learned target batch and labels are read only
for final scoring. Use ``--feature-scope shared`` for matched comparisons.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from combat_protocol import fit_apply_combat_partitions
from evaluation_protocol import classification_metrics, fit_with_validation
from experiment_protocol import make_patient_splits, ordered_common_gene_indices, patient_group_ids
from kpcada import load_expression_dataset
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer
from run_tvsn_experiment import choose_device, make_loader, set_seed, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--source-path", default="Data/TvsN/TCGA_BRCA_NvT_new.csv")
    parser.add_argument("--target-path", default="Data/TvsN/GSE45498_TvsN_official_200.csv")
    parser.add_argument("--exclude-source-label", action="append", default=["Unknown"])
    parser.add_argument("--variance-percentile", type=float, default=30.0)
    parser.add_argument("--kpca-gamma", type=float, default=0.2)
    parser.add_argument("--source-validation-size", type=float, default=0.2)
    parser.add_argument("--target-test-size", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=227)
    parser.add_argument("--model", default="resnet50d")
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output-root", default="results/tvsn")
    args = parser.parse_args()

    set_seed(args.seed)
    source_path = Path(args.source_path)
    target_path = Path(args.target_path)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not target_path.is_absolute():
        target_path = ROOT / target_path
    source_gxp, genes_s, ids_s, y_s = load_expression_dataset(str(source_path))
    target_gxp, genes_t, ids_t, y_t = load_expression_dataset(str(target_path))
    ids_s, ids_t = np.asarray(ids_s, dtype=str), np.asarray(ids_t, dtype=str)
    keep = ~np.isin(y_s, np.asarray(args.exclude_source_label))
    excluded_ids = ids_s[~keep].tolist()
    source_gxp, ids_s, y_s = source_gxp[:, keep], ids_s[keep], y_s[keep]
    if np.unique(y_s).size != 2:
        raise ValueError("ComBat TvsN runner requires exactly two source classes")
    source_groups, target_groups = patient_group_ids(ids_s), patient_group_ids(ids_t)
    source, target = source_gxp.T, target_gxp.T
    splits = make_patient_splits(
        y_s, source_ids=source_groups, target_ids=target_groups,
        source_validation_size=args.source_validation_size,
        target_test_size=args.target_test_size,
        random_state=args.seed, stratify_target=False,
    )
    source_train_raw = source[splits.source_train]
    variance = np.var(source_train_raw, axis=0)
    if not 0 <= args.variance_percentile < 100:
        raise ValueError("variance-percentile must be in [0, 100)")
    cutoff = np.percentile(variance, args.variance_percentile)
    source_keep = variance >= cutoff
    filtered_source_genes = np.asarray(genes_s, dtype=object)[source_keep]
    filtered_target_genes = np.asarray(genes_t, dtype=object)
    if len(set(filtered_source_genes.tolist())) != len(filtered_source_genes):
        raise ValueError("Source gene IDs must be unique before ComBat")
    if len(set(filtered_target_genes.tolist())) != len(filtered_target_genes):
        raise ValueError("Target gene IDs must be unique before ComBat")
    filtered_target_source_idx, filtered_target_idx, shared_genes = ordered_common_gene_indices(
        filtered_source_genes, filtered_target_genes)
    selected_source_idx = np.flatnonzero(source_keep)[filtered_target_source_idx]

    combat = fit_apply_combat_partitions(
        source[splits.source_train][:, selected_source_idx],
        source[splits.source_validation][:, selected_source_idx],
        target[splits.target_adapt][:, filtered_target_idx],
        target[splits.target_test][:, filtered_target_idx],
        shared_genes, shared_genes,
    )
    # ComBat uses only the paper's training-filtered, one-to-one shared genes.
    if not np.array_equal(combat.genes, shared_genes):
        raise RuntimeError("ComBat output gene order differs from the selected shared-gene order")
    matrices = [combat.source_train, combat.source_validation,
                combat.target_adaptation, combat.target_test]
    scaler = MinMaxScaler().fit(matrices[0])
    Xs_train, Xs_val, Xt_adapt, Xt_test = [scaler.transform(matrix) for matrix in matrices]

    encoder = LabelEncoder().fit(y_s[splits.source_train])
    ys_train = encoder.transform(y_s[splits.source_train])
    ys_val = encoder.transform(y_s[splits.source_validation])
    transformer = DomainAdaptiveImageTransformer(
        kpca_da_params={
            "n_components": 2, "kernel": "rbf", "kpca_gamma": args.kpca_gamma,
            "auto_gamma": True,
            "lambda_reg": 0.0, "lr": 0.001, "epochs": 0,
            "regularization_mode": "none", "kpca_random_state": args.seed,
        },
        pixels=(args.image_size, args.image_size),
    )
    transformer.fit_transform(Xs_train, Xt_adapt, shared_genes, shared_genes, plot=False)
    source_images = transformer.transform_source(Xs_train)
    validation_images = transformer.transform_source(Xs_val)
    target_images = transformer.transform_target(Xt_test)

    device = choose_device()
    model = timm.create_model(args.model, pretrained=not args.no_pretrained,
                              num_classes=len(encoder.classes_)).to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = make_loader(source_images, ys_train, preprocess, args.batch_size, shuffle=True)
    val_loader = make_loader(validation_images, ys_val, preprocess, args.batch_size)
    test_loader = make_loader(target_images, None, preprocess, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.classifier_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.scheduler_patience)
    training = fit_with_validation(
        model, train_loader, val_loader, torch.nn.CrossEntropyLoss(), optimizer,
        scheduler=scheduler, max_epochs=args.classifier_epochs,
        patience=args.early_stopping_patience, monitor="balanced_accuracy",
    )
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for (images,) in test_loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(logits.softmax(dim=1).cpu().tolist())
    positive = int(encoder.transform(["Tumor"])[0]) if "Tumor" in encoder.classes_ else None
    truth = encoder.transform(y_t[splits.target_test])
    metrics = classification_metrics(truth, predictions, labels=np.arange(len(encoder.classes_)),
                                      positive_label=positive)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / f"combat_seed{args.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_patients, target_patients = source_groups, target_groups
    metadata = {
        "arm": "combat", "seed": args.seed, "timestamp_utc": timestamp,
        "device": str(device), "source_file_sha256": sha256(source_path),
        "target_file_sha256": sha256(target_path), "source_file": str(source_path),
        "target_file": str(target_path), "source_excluded_sample_ids": excluded_ids,
        "source_patient_count": int(np.unique(source_patients).size),
        "target_patient_count": int(np.unique(target_patients).size),
        "source_train_n": int(len(splits.source_train)),
        "source_validation_n": int(len(splits.source_validation)),
        "target_adaptation_n": int(len(splits.target_adapt)),
        "target_test_n": int(len(splits.target_test)),
        "source_genes_retained": int(source_keep.sum()),
        "target_genes_retained": int(len(shared_genes)),
        "source_features_used": int(len(shared_genes)),
        "target_features_used": int(len(shared_genes)),
        "shared_genes_retained": int(len(shared_genes)),
        "label_classes": encoder.classes_.tolist(),
        "best_source_validation_epoch": training["best_epoch"],
        "best_source_validation_balanced_accuracy": training["best_validation_score"],
        "source_validation_history": training["history"],
        "test_metrics": metrics,
        "combat_fit": combat.fit_metadata,
        "config": {
            **vars(args), "source_validation_threshold": False,
            "clip_scaled_values": False, "feature_scope": "shared",
            "kpca_epochs": 0, "regularization_mode": "none",
            "classifier_lr": args.classifier_lr,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(run_dir / "predictions.npz", sample_ids=ids_t[splits.target_test],
        patient_ids=target_groups[splits.target_test].astype(str), y_true=truth,
        y_pred=np.asarray(predictions), probabilities=np.asarray(probabilities))
    np.savez_compressed(run_dir / "split_ids.npz",
        source_train=ids_s[splits.source_train], source_validation=ids_s[splits.source_validation],
        target_adaptation=ids_t[splits.target_adapt], target_test=ids_t[splits.target_test],
        source_train_patient=source_patients[splits.source_train].astype(str),
        source_validation_patient=source_patients[splits.source_validation].astype(str),
        target_adaptation_patient=target_patients[splits.target_adapt].astype(str),
        target_test_patient=target_patients[splits.target_test].astype(str))
    torch.save(training["model"].state_dict(), run_dir / "best_model.pt")
    with (run_dir / "fitted_preprocessing_and_map.pkl").open("wb") as handle:
        pickle.dump({"combat_estimates": combat.estimates, "scaler": scaler,
                     "transformer": transformer, "label_encoder": encoder,
                     "gene_ids": shared_genes}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(metadata, indent=2))
    print(f"Saved ComBat run artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
