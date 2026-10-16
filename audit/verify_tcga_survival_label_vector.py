"""Check whether the legacy 250-row TCGA label vector encodes survival or TvsN.

The vector has no sample IDs. This audit compares its row order to the column
order of the accompanying TCGA expression matrix and TCGA sample-type codes.
It is a provenance diagnostic; it does not construct survival labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPRESSION = ROOT / "Data/Survival_TCGA_VS_Desmd/TCGA_BRCA_NvT_v2.csv"
LABELS = ROOT / "Data/Survival_TCGA_VS_Desmd/org_tcga_tvn.csv"
OUTPUT = ROOT / "audit/tcga_survival_label_vector_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_type(sample_id: str) -> str | None:
    parts = str(sample_id).split("-")
    if len(parts) < 4 or len(parts[3]) < 2:
        return None
    return parts[3][:2]


def audit(expression: Path = EXPRESSION, labels: Path = LABELS) -> dict[str, object]:
    with expression.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    sample_ids = header[1:]
    label_frame = pd.read_csv(labels)
    if label_frame.shape[1] != 1:
        raise ValueError(f"Expected a one-column label vector, got shape {label_frame.shape}")
    y = label_frame.iloc[:, 0].to_numpy()
    if len(sample_ids) != len(y):
        raise ValueError(f"Expression has {len(sample_ids)} samples but labels have {len(y)} rows")
    types = [sample_type(sample) for sample in sample_ids]
    expected = {"01": 0, "06": 0, "11": 1}
    comparisons = [expected.get(code) == label for code, label in zip(types, y)]
    by_type: dict[str, Counter] = defaultdict(Counter)
    mismatches = []
    for sample, code, label, matched in zip(sample_ids, types, y, comparisons):
        by_type[str(code)][str(label)] += 1
        if not matched:
            mismatches.append({"sample_id": sample, "sample_type": code, "label": str(label)})
    result: dict[str, object] = {
        "expression_path": str(expression.relative_to(ROOT)),
        "expression_sha256": sha256(expression),
        "label_path": str(labels.relative_to(ROOT)),
        "label_sha256": sha256(labels),
        "sample_count": len(sample_ids),
        "label_count": len(y),
        "sample_type_counts": dict(Counter(str(code) for code in types)),
        "label_counts": {str(label): int(count) for label, count in zip(*np.unique(y, return_counts=True))},
        "labels_by_sample_type": {key: dict(value) for key, value in by_type.items()},
        "tvsn_order_matches": int(sum(comparisons)),
        "tvsn_order_match_fraction": float(np.mean(comparisons)) if comparisons else None,
        "tvsn_mismatches": mismatches,
        "interpretation": (
            "The row-aligned vector encodes TCGA sample type (0 for primary/other tumor-coded types, 1 for normal), "
            "not a survival outcome. Do not use it as TCGA aggressive/indolent labels."
            if all(comparisons) else
            "The vector does not exactly match the expected TCGA sample-type coding; provenance remains unresolved."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
