"""Verify candidate CSV-reader parity on real cohort formats, without training."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
sys.path.insert(0, str(ROOT / "audit"))
from benchmark_dataset_loader import read_fast
from kpcada import load_expression_dataset


def assert_equal_outputs(legacy: tuple, candidate: tuple, context: str) -> None:
    for index, (expected, actual) in enumerate(zip(legacy, candidate[:4])):
        if not np.array_equal(expected, actual):
            raise AssertionError(
                f"{context}: field {index} differs: "
                f"legacy shape {np.shape(expected)}, candidate shape {np.shape(actual)}"
            )


def main() -> int:
    started = time.perf_counter()
    target_path = ROOT / "Data" / "TvsN" / "GSE45498_TvsN_official_200.csv"
    legacy = load_expression_dataset(str(target_path), include_labels=True)
    candidate = read_fast(target_path, include_labels=True)
    assert_equal_outputs(legacy, candidate, "full corrected GEO matrix")
    unlabeled = read_fast(target_path, include_labels=False)
    if unlabeled[3] is not None or not np.array_equal(unlabeled[0], legacy[0]):
        raise AssertionError("GEO unlabeled mode returned labels or changed expression")

    source_path = ROOT / "Data" / "TvsN" / "TCGA_BRCA_NvT_new.csv"
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        feature_rows = []
        label_row = None
        for row in reader:
            if row[0].strip().lower() == "sample_type":
                label_row = row
                break
            if len(feature_rows) < 64:
                feature_rows.append(row)
    if label_row is None:
        raise AssertionError("Source CSV lacks its Sample_Type label row")
    with tempfile.TemporaryDirectory() as temporary_directory:
        fixture = Path(temporary_directory) / "tcga_real_format_subset.csv"
        with fixture.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(feature_rows)
            writer.writerow(label_row)
        source_legacy = load_expression_dataset(str(fixture), include_labels=True)
        source_candidate = read_fast(fixture, include_labels=True)
        assert_equal_outputs(source_legacy, source_candidate, "TCGA format subset")

    result = {
        "full_corrected_geo_exact_parity": True,
        "geo_unlabeled_mode_has_no_label_array": True,
        "tcga_format_subset_exact_parity": True,
        "source_fixture_genes": 64,
        "source_fixture_samples": int(len(header) - 1),
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
