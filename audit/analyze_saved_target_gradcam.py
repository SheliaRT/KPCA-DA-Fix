"""Generate post-hoc Grad-CAM gene rankings from one saved model run.

This utility only analyzes a finished checkpoint. It implements two distinct
paper descriptions: main-text top-k ranking from held-out target patients and
Supplement Note 4 thresholding from source-training patients. They use their
own fitted domain map and retained gene order; their outputs are not merged.
Labels only group post-hoc maps and never affect model selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import (ordered_common_gene_indices, patient_group_ids,
                                 patient_groups_from_mapping)
from kpcada import load_expression_dataset
from pyDeepInsightv3.gene_attribution import rank_gene_attributions


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_patient_ids(metadata: dict[str, Any], domain: str,
                       sample_ids: np.ndarray,
                       expected_patient_ids: np.ndarray) -> None:
    provenance = metadata.get("patient_mapping", {}).get(domain, {})
    if provenance.get("source") == "explicit sample-to-patient map":
        mapping_path = resolve(provenance["mapping_file"])
        if digest(mapping_path) != provenance["mapping_file_sha256"]:
            raise ValueError(f"{domain} patient-map hash differs from saved run metadata")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        observed = patient_groups_from_mapping(sample_ids, mapping).astype(str)
    else:
        observed = patient_group_ids(sample_ids).astype(str)
    if not np.array_equal(observed, expected_patient_ids):
        raise ValueError(f"{domain} patient IDs do not match saved sample IDs and mapping")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--analysis-policy",
                        choices=("main_text_top_k", "supplement_threshold"),
                        default="main_text_top_k",
                        help="main text: held-out target maps ranked top-k; supplement: source-training maps thresholded at 0.5")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        if args.analysis_policy == "main_text_top_k":
            sample_ids = split["target_test"].astype(str)
            expected_patient_ids = split["target_test_patient"].astype(str)
        else:
            sample_ids = split["source_train"].astype(str)
            expected_patient_ids = split["source_train_patient"].astype(str)

    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    preprocessor = fitted["preprocessor"]
    transformer = fitted["transformer"]
    encoder = fitted["label_encoder"]
    if args.analysis_policy == "main_text_top_k":
        expression_path = resolve(metadata["target_file"])
        expected_hash = metadata["target_file_sha256"]
        X_gxp, target_genes, all_ids, all_labels = load_expression_dataset(str(expression_path))
        if digest(expression_path) != expected_hash:
            raise ValueError("Target file hash differs from saved run metadata")
        verify_patient_ids(metadata, "target", sample_ids, expected_patient_ids)
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as predictions:
            if not np.array_equal(predictions["sample_ids"].astype(str), sample_ids):
                raise ValueError("Prediction sample IDs differ from the held-out target split")
            if not np.array_equal(predictions["patient_ids"].astype(str), expected_patient_ids):
                raise ValueError("Prediction patient IDs differ from the held-out target split")
            group_labels = predictions["y_true"].astype(int)
    else:
        expression_path = resolve(metadata["source_file"])
        expected_hash = metadata["source_file_sha256"]
        X_gxp, target_genes, all_ids, all_labels = load_expression_dataset(str(expression_path))
        if digest(expression_path) != expected_hash:
            raise ValueError("Source file hash differs from saved run metadata")
        verify_patient_ids(metadata, "source", sample_ids, expected_patient_ids)
        group_labels = None
    all_ids = np.asarray(all_ids, dtype=str)
    row_by_id = {sample_id: row for row, sample_id in enumerate(all_ids)}
    if any(sample_id not in row_by_id for sample_id in sample_ids):
        raise ValueError("Attribution sample ID is absent from its hashed expression matrix")
    rows = np.asarray([row_by_id[sample_id] for sample_id in sample_ids], dtype=int)
    group_labels_from_file = encoder.transform(all_labels[rows])
    if group_labels is not None and not np.array_equal(group_labels_from_file, group_labels):
        raise ValueError("Saved target labels do not match the hashed target matrix")
    group_labels = group_labels_from_file
    if args.analysis_policy == "main_text_top_k":
        X = preprocessor.transform_target(X_gxp[:, rows].T, target_genes)
        retained_genes = preprocessor.target_genes_.copy()
        if metadata["config"].get("feature_scope", "all") == "shared":
            _, target_shared_idx, shared = ordered_common_gene_indices(
                preprocessor.source_genes_, preprocessor.target_genes_)
            X = X[:, target_shared_idx]
            retained_genes = shared
        coords = transformer.coords_target()
        normalization = "mean raw Grad-CAM, then min-max normalize to [0,1]"
        selection = f"top {args.top_k} genes by normalized mean CAM score"
    else:
        X = preprocessor.transform_source(X_gxp[:, rows].T, target_genes)
        retained_genes = preprocessor.source_genes_.copy()
        if metadata["config"].get("feature_scope", "all") == "shared":
            source_shared_idx, _, shared = ordered_common_gene_indices(
                preprocessor.source_genes_, preprocessor.target_genes_)
            X = X[:, source_shared_idx]
            retained_genes = shared
        coords = transformer.coords_source()
        normalization = "mean raw source-training Grad-CAM, then min-max normalize to [0,1]"
        selection = "strict normalized mean CAM score > 0.5"
    if len(retained_genes) != len(coords) or X.shape[1] != len(coords):
        raise ValueError("Saved attribution gene order, intensities, and coordinates do not align")
    if args.analysis_policy == "main_text_top_k":
        images = transformer.transform_target(X)
    else:
        images = transformer.transform_source(X)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    model = timm.create_model(metadata["config"]["model"], pretrained=False,
                              num_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device,
                                     weights_only=True))
    model.eval()
    if not hasattr(model, "layer4"):
        raise ValueError("Expected the saved ResNet model to expose layer4 for Grad-CAM")
    target_layer = model.layer4[-1]
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    hook = target_layer.register_forward_hook(forward_hook)
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    inputs = torch.stack([preprocessing(image) for image in images]).float()
    cams_by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    try:
        for start in range(0, len(inputs), args.batch_size):
            stop = min(start + args.batch_size, len(inputs))
            batch = inputs[start:stop].to(device)
            class_ids = torch.as_tensor(group_labels[start:stop], dtype=torch.long, device=device)
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            chosen_scores = logits.gather(1, class_ids[:, None]).sum()
            chosen_scores.backward()
            activations = captured["activations"]
            gradients = captured["gradients"]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            raw_cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            raw_cam = F.interpolate(raw_cam, size=images.shape[1:3], mode="bilinear",
                                    align_corners=False)[:, 0]
            raw_np = raw_cam.detach().cpu().numpy()
            for offset, class_id in enumerate(class_ids.detach().cpu().tolist()):
                cams_by_class[int(class_id)].append(raw_np[offset])
    finally:
        hook.remove()

    class_results: dict[str, Any] = {}
    rows_out: list[dict[str, Any]] = []
    for class_id in sorted(cams_by_class):
        cam_batch = np.stack(cams_by_class[class_id])
        class_patient_ids = expected_patient_ids[group_labels == class_id]
        if args.analysis_policy == "main_text_top_k":
            selected = rank_gene_attributions(
                cam_batch, retained_genes, coords, top_k=args.top_k,
                normalize_mean_map=True,
            )
        else:
            selected = rank_gene_attributions(
                cam_batch, retained_genes, coords, threshold=0.5,
                normalize_mean_map=True,
            )
        class_name = str(encoder.classes_[class_id])
        class_results[class_name] = {
            "attribution_samples": int(len(cam_batch)),
            "attribution_patients": int(np.unique(class_patient_ids).size),
            "cam_aggregation_weighting": (
                "equal weight per sample; multiple samples from a patient contribute separately"
            ),
            "cam_normalization": normalization,
            "selection_policy": selection,
            "tie_policy": "stable retained-gene input order after descending score",
            "genes": selected,
        }
        rows_out.extend({"class": class_name, **row} for row in selected)

    default_name = ("target_gradcam_top_genes.json"
                    if args.analysis_policy == "main_text_top_k"
                    else "source_gradcam_threshold_genes.json")
    output = args.output or (run_dir / default_name)
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "run_dir": str(run_dir),
        "arm": metadata["arm"],
        "seed": metadata["seed"],
        "analysis_policy": args.analysis_policy,
        "expression_file_sha256": expected_hash,
        "attribution_sample_ids": sample_ids.tolist(),
        "attribution_patient_ids": expected_patient_ids.tolist(),
        "attribution_domain": ("target_test" if args.analysis_policy == "main_text_top_k"
                               else "source_train"),
        "feature_scope": metadata["config"].get("feature_scope", "all"),
        "gene_count": int(len(retained_genes)),
        "class_grouping": ("saved held-out target labels; post-hoc attribution only"
                           if args.analysis_policy == "main_text_top_k"
                           else "source-training labels; attribution is descriptive of fitted samples"),
        "target_labels_used_for_model_selection": False,
        "class_results": class_results,
    }
    if args.analysis_policy == "main_text_top_k":
        result["target_file_sha256"] = expected_hash
        result["target_test_ids"] = sample_ids.tolist()
    else:
        result["source_file_sha256"] = expected_hash
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "class", "rank", "gene_id", "feature_index", "score",
            "pixel_row", "pixel_column", "genes_at_pixel",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    print(json.dumps({"result": "passed", "json": str(output),
                      "csv": str(csv_path), "class_counts": {
                          name: values["attribution_samples"]
                          for name, values in class_results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
