"""Evaluate a source-validation-only threshold on saved TvsN checkpoints.

No models are retrained. The threshold is selected from source-validation
labels, then applied to the held-out target probabilities already saved by each
run. Target labels are used only for the final descriptive metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from evaluation_protocol import classification_metrics, select_binary_threshold
from experiment_protocol import ordered_common_gene_indices, patient_group_ids
from kpcada import load_expression_dataset


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_validation_probabilities(run_dir: Path, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source_path = resolve(metrics["source_file"])
    if digest(source_path) != metrics["source_file_sha256"]:
        raise ValueError("Source file hash differs from the saved run")
    source_gxp, source_genes, source_ids, source_labels = load_expression_dataset(str(source_path))
    excluded = set(metrics.get("source_excluded_sample_ids", []))
    keep = np.asarray([str(sample_id) not in excluded for sample_id in source_ids])
    source_gxp, source_ids, source_labels = source_gxp[:, keep], np.asarray(source_ids)[keep], source_labels[keep]
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        validation_ids = split["source_validation"].astype(str)
    row_by_id = {str(sample_id): i for i, sample_id in enumerate(source_ids)}
    if any(sample_id not in row_by_id for sample_id in validation_ids):
        raise ValueError("Saved source-validation IDs do not match the current source file")
    rows = np.asarray([row_by_id[sample_id] for sample_id in validation_ids], dtype=int)
    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    preprocessor = fitted["preprocessor"]
    transformer = fitted["transformer"]
    encoder = fitted["label_encoder"]
    X_val = preprocessor.transform_source(source_gxp[:, rows].T, source_genes)
    config = metrics["config"]
    if config.get("feature_scope", "all") == "shared":
        source_idx, _, _ = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_)
        X_val = X_val[:, source_idx]
    images = transformer.transform_source(X_val)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    model = timm.create_model(config["model"], pretrained=False,
                              num_classes=len(encoder.classes_)).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device,
                                     weights_only=True))
    model.eval()
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = torch.stack([preprocessing(image) for image in images]).float()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(tensor), config["batch_size"]):
            logits = model(tensor[start:start + config["batch_size"]].to(device))
            probabilities.extend(logits.softmax(dim=1).cpu().numpy())
    labels = encoder.transform(source_labels[rows])
    return labels, np.asarray(probabilities)


def evaluate(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    target_path = resolve(metrics["target_file"])
    if digest(target_path) != metrics["target_file_sha256"]:
        raise ValueError("Target file hash differs from the saved run")
    truth, validation_probabilities = source_validation_probabilities(run_dir, metrics)
    positive = int(metrics["test_metrics"]["positive_label"])
    negative = next(i for i in range(validation_probabilities.shape[1]) if i != positive)
    # Select the threshold before opening target labels.
    threshold = select_binary_threshold(
        truth, validation_probabilities[:, positive],
        positive_label=positive, negative_label=negative,
    )
    target_gxp, _, target_ids, target_labels = load_expression_dataset(str(target_path))
    del target_gxp
    target_ids = np.asarray(target_ids, dtype=str)
    target_by_id = {sample_id: i for i, sample_id in enumerate(target_ids)}
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        expected_test_ids = split["target_test"].astype(str)
        expected_patient_ids = split["target_test_patient"].astype(str)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as test:
        if not np.array_equal(test["sample_ids"].astype(str), expected_test_ids):
            raise ValueError("Saved target predictions differ from the held-out split IDs")
        if not np.array_equal(test["patient_ids"].astype(str), expected_patient_ids):
            raise ValueError("Saved target prediction patient IDs differ from the held-out split")
        if not np.array_equal(patient_group_ids(expected_test_ids).astype(str), expected_patient_ids):
            raise ValueError("Held-out target patient groups do not match the official sample IDs")
        if any(sample_id not in target_by_id for sample_id in expected_test_ids):
            raise ValueError("Held-out target IDs are missing from the hashed target file")
        target_rows = np.asarray([target_by_id[sample_id] for sample_id in expected_test_ids])
        encoder_classes = np.asarray(metrics["label_classes"])
        target_truth = np.asarray([
            int(np.flatnonzero(encoder_classes == label)[0]) for label in target_labels[target_rows]
        ])
        if not np.array_equal(test["y_true"], target_truth):
            raise ValueError("Saved target labels differ from the hashed target file")
        y_true = test["y_true"].copy()
        test_probabilities = test["probabilities"].copy()
    predictions = np.where(test_probabilities[:, positive] >= threshold["threshold"],
                           positive, negative)
    threshold_metrics = classification_metrics(
        y_true, predictions,
        labels=metrics["test_metrics"]["labels"], positive_label=positive,
    )
    return {
        "arm": metrics["arm"],
        "seed": metrics["seed"],
        "source_file_sha256": metrics["source_file_sha256"],
        "target_file_sha256": metrics["target_file_sha256"],
        "selection": {
            "rule": "maximize source-validation balanced accuracy",
            **threshold,
            "validation_samples": int(truth.size),
            "target_labels_used": False,
        },
        "argmax_target_metrics": metrics["test_metrics"],
        "source_threshold_target_metrics": threshold_metrics,
        "target_metrics_are_development_evidence": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = evaluate(args.baseline)
    adapted = evaluate(args.adapted)
    for key in ("seed", "source_file_sha256", "target_file_sha256"):
        if base[key] != adapted[key]:
            raise ValueError(f"Paired runs differ in {key}")
    result = {
        "valid_pair": True,
        "seed": base["seed"],
        "source_file_sha256": base["source_file_sha256"],
        "target_file_sha256": base["target_file_sha256"],
        "baseline": base,
        "adapted": adapted,
        "thresholded_adapted_minus_baseline": {
            metric: adapted["source_threshold_target_metrics"][metric]
                    - base["source_threshold_target_metrics"][metric]
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity")
        },
        "interpretation": "The threshold is chosen without target labels; target metrics are post-hoc development evidence and do not establish independent generalization.",
    }
    output = resolve(str(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
