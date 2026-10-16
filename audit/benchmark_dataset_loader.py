"""Benchmark a numeric-first CSV reader against the legacy repository loader.

This read-only preflight checks the wide gene-by-patient CSVs. It does not
train models or read target labels when loading the target matrix.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from kpcada import load_expression_dataset


def read_fast(path: Path, *, include_labels: bool) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]
]:
    start = time.perf_counter()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 2 or header[0].strip().lower() not in {"gene", "hugo_symbol"}:
            raise ValueError(f"Unexpected header in {path}")
        sample_ids = np.asarray(header[1:], dtype=object)
        gene_names: list[str] = []
        numeric_rows: list[np.ndarray] = []
        labels: np.ndarray | None = None
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{path}:{row_number} has {len(row)} fields; expected {len(header)}"
                )
            name = row[0].strip()
            if name.lower() in {"sample_type", "sample type", "label", "labels"}:
                if include_labels:
                    labels = np.asarray(row[1:], dtype=object)
                continue
            values = np.fromstring(",".join(row[1:]), sep=",", dtype=np.float64)
            if values.size != sample_ids.size or not np.isfinite(values).all():
                raise ValueError(f"Invalid numeric expression row {name!r} in {path}")
            gene_names.append(name)
            numeric_rows.append(values)
    if not gene_names:
        raise ValueError(f"No expression rows found in {path}")
    if include_labels and (labels is None or labels.size != sample_ids.size):
        raise ValueError(f"Missing or malformed label row in {path}")
    matrix = np.stack(numeric_rows, axis=0)
    digest = hashlib.sha256()
    digest.update(repr(matrix.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(matrix).view(np.uint8))
    for array in (np.asarray(gene_names, dtype=object), sample_ids):
        digest.update(repr(array.shape).encode("ascii"))
        digest.update("\0".join(map(str, array.tolist())).encode("utf-8"))
    return matrix, np.asarray(gene_names, dtype=object), sample_ids, labels, {
        "seconds": time.perf_counter() - start,
        "shape": list(matrix.shape),
        "sha256": digest.hexdigest(),
    }


def synthetic_parity() -> None:
    import tempfile

    fixture = "gene,S1,S2\nG1,1.25,2.5\nG2,4.0,8.0\nSample_Type,Normal,Tumor\n"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fixture.csv"
        path.write_text(fixture, encoding="utf-8")
        legacy = load_expression_dataset(str(path))
        fast = read_fast(path, include_labels=True)
        for got, expected in zip(fast[:4], legacy):
            if not np.array_equal(got, expected):
                raise AssertionError(f"Fast CSV reader differs from legacy loader: {got!r} != {expected!r}")
        unlabeled = read_fast(path, include_labels=False)
        if unlabeled[3] is not None or not np.array_equal(unlabeled[0], legacy[0]):
            raise AssertionError("Unlabeled loading changed expression or materialized labels")


def main() -> int:
    synthetic_parity()
    outputs = {}
    for name, include_labels in (
        ("TCGA_BRCA_NvT_new.csv", True),
        ("GSE45498_TvsN_official_200.csv", False),
    ):
        path = ROOT / "Data" / "TvsN" / name
        _, genes, samples, labels, stats = read_fast(path, include_labels=include_labels)
        stats.update({
            "genes": int(genes.size),
            "samples": int(samples.size),
            "labels_materialized": labels is not None,
        })
        outputs[name] = stats
    print(json.dumps({"synthetic_legacy_parity": True, "datasets": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
