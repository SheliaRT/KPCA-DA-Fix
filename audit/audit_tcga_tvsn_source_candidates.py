"""Inventory local TCGA tumor/normal matrices against the paper's cohort size.

The report is descriptive: sample-type barcodes and stored labels do not by
themselves establish the paper's inclusion criteria. It records input hashes,
column identity, labels and TCGA sample-type counts for reproducibility.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "Data/TvsN/TCGA_BRCA_NvT.csv",
    "Data/TvsN/TCGA_BRCA_NvT_new.csv",
    "Data/TvsN/TCGA_BRCA_NvTv1_filtered.csv",
    "Data/Survival_TCGA_VS_Desmd/TCGA_BRCA_NvT.csv",
    "Data/Survival_TCGA_VS_Desmd/TCGA_BRCA_NvT_v1.csv",
    "Data/Survival_TCGA_VS_Desmd/TCGA_BRCA_NvT_v2.csv",
)
OUTPUT = ROOT / "audit" / "tcga_tvsn_source_candidate_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_candidate(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        return {"path": relative_path, "available": False}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        sample_ids = [str(value).strip() for value in header[1:]]
        last_row: list[str] | None = None
        body_rows = 0
        inconsistent_rows = 0
        for row in reader:
            body_rows += 1
            if len(row) != len(header):
                inconsistent_rows += 1
            last_row = row
    if not last_row:
        raise ValueError(f"No data rows in {relative_path}")
    labels = [str(value).strip() for value in last_row[1:]]
    if len(labels) != len(sample_ids):
        raise ValueError(f"Last-row labels do not align to sample columns in {relative_path}")

    sample_types: Counter[str] = Counter()
    patient_ids: list[str | None] = []
    invalid_barcodes: list[str] = []
    for sample_id in sample_ids:
        parts = sample_id.split("-")
        if len(parts) >= 4 and parts[0] == "TCGA":
            patient_ids.append("-".join(parts[:3]))
            sample_types[parts[3][:2]] += 1
        else:
            patient_ids.append(None)
            invalid_barcodes.append(sample_id)
    eligible_indices = [
        index for index, sample_id in enumerate(sample_ids)
        if len(sample_id.split("-")) >= 4
        and sample_id.split("-")[0] == "TCGA"
        and sample_id.split("-")[3][:2] in {"01", "11"}
    ]
    eligible_labels = Counter(labels[index] for index in eligible_indices)
    return {
        "path": relative_path,
        "available": True,
        "sha256": sha256(path),
        "header_columns": len(header),
        "sample_columns": len(sample_ids),
        "body_rows_including_label_row": body_rows,
        "last_row_label_counts": dict(Counter(labels)),
        "tcga_sample_type_counts": dict(sorted(sample_types.items())),
        "eligible_type_01_or_11_samples": len(eligible_indices),
        "eligible_type_01_or_11_patient_count": len({patient_ids[i] for i in eligible_indices}),
        "eligible_type_01_or_11_stored_label_counts": dict(eligible_labels),
        "unique_patient_count_all_tcga_barcodes": len(set(patient_ids)),
        "duplicate_sample_id_count": len(sample_ids) - len(set(sample_ids)),
        "non_tcga_or_unparseable_sample_ids": invalid_barcodes,
        "rows_with_wrong_column_count": inconsistent_rows,
        "label_row_first_cell": last_row[0],
    }


def main() -> int:
    report = {
        "interpretation": (
            "Inventory only. Matching the manuscript's count is not proof of "
            "the same inclusion criteria or an authoritative cohort."
        ),
        "paper_reported_sample_count": 592,
        "candidates": [audit_candidate(value) for value in CANDIDATES],
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": "completed",
        "output": str(OUTPUT.relative_to(ROOT)),
        "candidates": [
            {
                "path": item["path"],
                "sample_columns": item.get("sample_columns"),
                "eligible_01_11": item.get("eligible_type_01_or_11_samples"),
                "labels": item.get("eligible_type_01_or_11_stored_label_counts"),
            }
            for item in report["candidates"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
