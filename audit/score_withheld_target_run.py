"""Score one frozen run whose target-test labels were withheld during training.

The scorer validates the run's target hash and held-out sample/patient IDs,
then writes a separate, non-overwritable score artifact. It never changes the
checkpoint, predictions, or training metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "Scripts"))
from evaluation_protocol import classification_metrics
from kpcada import load_expression_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_run(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.npz"
    splits_path = run_dir / "split_ids.npz"
    for path in (metadata_path, predictions_path, splits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required run artifact is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target_test_labels_scored") is not False:
        raise ValueError("Run must explicitly record target_test_labels_scored=false")
    if metadata.get("test_metrics") is not None:
        raise ValueError("Run already contains target-test metrics")
    if not metadata.get("config", {}).get("withhold_target_test_labels"):
        raise ValueError("Run configuration does not record target-test label withholding")
    target_path = Path(metadata["target_file"])
    if not target_path.is_absolute():
        target_path = ROOT / target_path
    if sha256(target_path) != metadata.get("target_file_sha256"):
        raise ValueError("Target cohort hash differs from the frozen run metadata")
    model_path = run_dir / "best_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError("Frozen run checkpoint is missing")
    model_hash = sha256(model_path)

    output = run_dir / "target_test_score.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior target score: {output}")

    _, _, all_sample_ids, all_labels = load_expression_dataset(str(target_path))
    if all_labels is None:
        raise RuntimeError("Target labels were not returned by the label-reading loader")
    labels_by_sample = dict(zip(all_sample_ids.astype(str), all_labels.astype(str)))
    if len(labels_by_sample) != len(all_sample_ids):
        raise ValueError("Target sample IDs are not unique in the scoring cohort")

    with np.load(predictions_path, allow_pickle=False) as saved:
        if "y_true" in saved.files:
            raise ValueError("Withheld-label prediction artifact unexpectedly contains y_true")
        required = {"sample_ids", "patient_ids", "y_pred", "probabilities"}
        if not required.issubset(saved.files):
            raise ValueError(f"Prediction artifact is missing fields: {sorted(required - set(saved.files))}")
        sample_ids = saved["sample_ids"].astype(str)
        patient_ids = saved["patient_ids"].astype(str)
        y_pred = np.asarray(saved["y_pred"], dtype=int)
        probabilities = np.asarray(saved["probabilities"], dtype=float)

    with np.load(splits_path, allow_pickle=False) as splits:
        split_sample_ids = splits["target_test"].astype(str)
        split_patient_ids = splits["target_test_patient"].astype(str)
        adapt_patient_ids = splits["target_adaptation_patient"].astype(str)
    if not np.array_equal(sample_ids, split_sample_ids):
        raise ValueError("Predicted sample IDs do not match the frozen target-test split")
    if not np.array_equal(patient_ids, split_patient_ids):
        raise ValueError("Predicted patient IDs do not match the frozen target-test split")
    if np.intersect1d(patient_ids, adapt_patient_ids).size:
        raise ValueError("Target-test and adaptation patients overlap")
    missing = [sample for sample in sample_ids if sample not in labels_by_sample]
    if missing:
        raise ValueError(f"Target scoring cohort is missing held-out samples: {missing[:10]}")
    classes = [str(value) for value in metadata["label_classes"]]
    class_to_index = {label: index for index, label in enumerate(classes)}
    unseen = sorted({labels_by_sample[sample] for sample in sample_ids} - set(class_to_index))
    if unseen:
        raise ValueError(f"Held-out target labels are absent from source classes: {unseen}")
    y_true = np.asarray([class_to_index[labels_by_sample[sample]] for sample in sample_ids])
    if y_pred.shape != y_true.shape or probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Saved predictions have incompatible shapes")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Saved class probabilities are not finite normalized rows")
    if np.any(y_pred < 0) or np.any(y_pred >= len(classes)):
        raise ValueError("Saved predictions contain class indices outside the source label set")
    positive = class_to_index.get("Tumor") if len(classes) == 2 else None
    metrics = classification_metrics(
        y_true, y_pred, labels=np.arange(len(classes)), positive_label=positive
    )
    result = {
        "result": "target test scored once after model freeze",
        "run_dir": str(run_dir),
        "arm": metadata["arm"],
        "seed": metadata["seed"],
        "source_file_sha256": metadata["source_file_sha256"],
        "target_file_sha256": metadata["target_file_sha256"],
        "best_model_sha256": model_hash,
        "predictions_sha256": sha256(predictions_path),
        "split_ids_sha256": sha256(splits_path),
        "target_test_sample_ids": sample_ids.tolist(),
        "target_test_patient_ids": patient_ids.tolist(),
        "label_classes": classes,
        "target_test_metrics": metrics,
    }
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = score_run(args.run_dir)
    print(json.dumps({"result": "scored", "output": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
