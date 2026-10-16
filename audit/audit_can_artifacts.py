"""Inventory legacy CAN-named prediction artifacts without inferring provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]


def inspect(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = 0
    widths: set[int] = set()
    numeric_count = 0
    nonnumeric_values: list[str] = []
    minimum = float("inf")
    maximum = float("-inf")
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.reader(handle):
            rows += 1
            widths.add(len(row))
            for value in row:
                try:
                    number = float(value)
                except (ValueError, TypeError):
                    if len(nonnumeric_values) < 8:
                        nonnumeric_values.append(value)
                    continue
                numeric_count += 1
                minimum = min(minimum, number)
                maximum = max(maximum, number)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "rows_including_any_header": rows,
        "row_widths": sorted(widths),
        "numeric_values": numeric_count,
        "non_numeric_examples": nonnumeric_values,
        "numeric_min": minimum if numeric_count else None,
        "numeric_max": maximum if numeric_count else None,
        "contains_sample_ids": False,
    }


def read_scalar_vector(path: Path) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    headers: list[str] = []
    with path.open("r", newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 1:
                raise ValueError(f"Expected one-column vector in {path}:{row_number}")
            try:
                values.append(float(row[0]))
            except ValueError:
                if row_number != 1:
                    raise
                headers.append(row[0])
    return np.asarray(values, dtype=float), headers


def score_sidecars(path: Path) -> dict[str, object] | None:
    truth_path = path.parent / "true.csv"
    if not truth_path.is_file():
        return None
    truth, truth_header = read_scalar_vector(truth_path)
    scores, score_header = read_scalar_vector(path)
    if truth.size != scores.size:
        return {
            "truth_path": truth_path.relative_to(ROOT).as_posix(),
            "truth_rows": int(truth.size),
            "score_rows": int(scores.size),
            "row_counts_match": False,
        }
    labels = np.unique(truth)
    result: dict[str, object] = {
        "truth_path": truth_path.relative_to(ROOT).as_posix(),
        "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "score_rows": int(scores.size),
        "row_counts_match": True,
        "truth_header": truth_header,
        "score_header": score_header,
        "label_values": labels.tolist(),
        "row_order_assumption": "same-folder truth and score vectors are positionally paired, as in the archived scoring notebook; sample IDs are absent",
    }
    if set(labels.tolist()) == {0.0, 1.0}:
        truth_int = truth.astype(int)
        pred = (scores > 0.5).astype(int)
        result.update({
            "roc_auc": float(roc_auc_score(truth_int, scores)),
            "accuracy_at_archived_threshold_0_5": float(accuracy_score(truth_int, pred)),
            "balanced_accuracy_at_archived_threshold_0_5": float(balanced_accuracy_score(truth_int, pred)),
            "macro_f1_at_archived_threshold_0_5": float(f1_score(truth_int, pred, average="macro", zero_division=0)),
        })
    else:
        pred = np.rint(scores).astype(int)
        result.update({
            "score_interpreted_as_class_prediction": True,
            "accuracy": float(accuracy_score(truth.astype(int), pred)),
            "balanced_accuracy": float(balanced_accuracy_score(truth.astype(int), pred)),
            "macro_f1": float(f1_score(truth.astype(int), pred, average="macro", zero_division=0)),
        })
    return result


def main() -> int:
    files = sorted(path for path in (ROOT / "Data").rglob("*.csv")
                   if "can" in path.name.lower())
    artifacts = [inspect(path) for path in files]
    sidecars = [
        scored for path in files
        if path.name in {"pred_CAN.csv", "output_CAN.csv"}
        if (scored := score_sidecars(path)) is not None
    ]
    result = {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "sidecar_reconstructions": sidecars,
        "interpretation": (
            "These files are numeric CAN-named outputs only. No sample IDs, model "
            "checkpoint, executable CAN implementation, training configuration, "
            "or source-tree/environment fingerprint is present in these files. "
            "Metrics reconstructed against same-folder true.csv files assume positional "
            "order as the archived scoring notebook does; without sample IDs, that "
            "assumption and the paper cohort cannot be verified. These values are "
            "historical artifact diagnostics, not reproduced CAN results."
        ),
    }
    output = ROOT / "audit" / "can_artifact_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
