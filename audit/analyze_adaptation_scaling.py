"""Label-free preflight for source-fit vs target-adaptation MinMax scaling."""

from __future__ import annotations

import json
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
sys.path.insert(0, str(ROOT / "audit"))
from benchmark_dataset_loader import read_fast
from experiment_protocol import (GeneAwareMinMax, make_patient_splits,
                                 patient_group_ids, patient_groups_from_mapping)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wait_for_job(job: str | None) -> None:
    if job is None:
        return
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.is_file():
        time.sleep(15)
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite job {job} failed; see {record.get('log')}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=46,
                        help="patient split seed for the adaptation-only diagnostic")
    parser.add_argument("--source-path", type=Path,
                        default=ROOT / "Data/TvsN/TCGA_BRCA_NvT_new.csv")
    parser.add_argument("--target-path", type=Path,
                        default=ROOT / "Data/TvsN/GSE45498_TvsN_official_200.csv")
    parser.add_argument("--target-patient-map", type=Path,
                        help="optional explicit sample-to-patient JSON map")
    parser.add_argument("--after-job", help="detached job that must finish before this diagnostic")
    parser.add_argument("--output", type=Path,
                        help="optional JSON output path (relative to repository root)")
    args = parser.parse_args()
    wait_for_job(args.after_job)
    source_path = args.source_path if args.source_path.is_absolute() else ROOT / args.source_path
    target_path = args.target_path if args.target_path.is_absolute() else ROOT / args.target_path
    patient_map_path = args.target_patient_map
    if patient_map_path is not None and not patient_map_path.is_absolute():
        patient_map_path = ROOT / patient_map_path
    source, source_genes, source_ids, source_labels, _ = read_fast(
        source_path, include_labels=True
    )
    target, target_genes, target_ids, target_labels, _ = read_fast(
        target_path, include_labels=False
    )
    if source_labels is None or target_labels is not None:
        raise AssertionError("Expected source labels only")
    keep_source = source_labels != "Unknown"
    source = source[:, keep_source]
    source_ids = source_ids[keep_source]
    source_labels = source_labels[keep_source]
    target_groups = patient_group_ids(target_ids)
    if patient_map_path is not None:
        target_mapping = json.loads(patient_map_path.read_text(encoding="utf-8"))
        target_groups = patient_groups_from_mapping(target_ids, target_mapping)
    splits = make_patient_splits(
        source_labels,
        source_ids=patient_group_ids(source_ids),
        target_ids=target_groups,
        random_state=args.seed,
    )
    source_train = source[:, splits.source_train].T
    target_adapt = target[:, splits.target_adapt].T
    preprocessing = GeneAwareMinMax(variance_percentile=30.0).fit(
        source_train, source_genes, target_adapt, target_genes
    )
    source_positions = {gene: i for i, gene in enumerate(preprocessing.source_genes_.tolist())}
    target_positions = {gene: i for i, gene in enumerate(target_genes.tolist())}
    shared = [gene for gene in preprocessing.shared_genes_.tolist()
              if gene in source_positions]
    src_idx = np.asarray([source_positions[g] for g in shared], dtype=int)
    tgt_idx = np.asarray([target_positions[g] for g in shared], dtype=int)
    target_values = target_adapt[:, tgt_idx]
    source_scaler = preprocessing.source_scaler_
    source_range = source_scaler.data_range_[src_idx]
    source_scaled = (
        target_values - source_scaler.data_min_[src_idx]
    ) / np.where(source_range == 0.0, 1.0, source_range)
    target_min = target_values.min(axis=0)
    target_range = np.ptp(target_values, axis=0)
    target_scaled = (target_values - target_min) / np.where(
        target_range == 0.0, 1.0, target_range
    )
    ratio = target_range / np.where(source_range == 0.0, np.nan, source_range)
    result = {
        "seed": args.seed,
        "source_path": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
        "source_sha256": sha256(source_path),
        "target_path": str(target_path.relative_to(ROOT) if target_path.is_relative_to(ROOT) else target_path),
        "target_sha256": sha256(target_path),
        "target_patient_map_path": (
            str(patient_map_path.relative_to(ROOT) if patient_map_path.is_relative_to(ROOT) else patient_map_path)
            if patient_map_path is not None else None
        ),
        "target_patient_map_sha256": sha256(patient_map_path) if patient_map_path is not None else None,
        "source_train_samples": int(splits.source_train.size),
        "target_adaptation_samples": int(splits.target_adapt.size),
        "target_adaptation_patients": int(len(set(target_groups[splits.target_adapt].tolist()))),
        "target_test_samples": int(splits.target_test.size),
        "target_test_patients": int(len(set(target_groups[splits.target_test].tolist()))),
        "shared_genes": int(len(shared)),
        "target_labels_materialized": False,
        "held_out_target_values_used": False,
        "target_labels_used_for_split": False,
        "source_fit_transform_on_target_adaptation": {
            "fraction_outside_unit_interval": float(np.mean((source_scaled < 0) | (source_scaled > 1))),
            "maximum": float(np.max(source_scaled)),
            "p99": float(np.quantile(source_scaled, 0.99)),
        },
        "adaptation_fit_transform_on_same_adaptation_rows": {
            "fraction_outside_unit_interval": float(np.mean((target_scaled < 0) | (target_scaled > 1))),
            "maximum": float(np.max(target_scaled)),
            "p99": float(np.quantile(target_scaled, 0.99)),
        },
        "target_to_source_per_gene_range_ratio": {
            "median": float(np.nanmedian(ratio)),
            "p10": float(np.nanquantile(ratio, 0.10)),
            "p90": float(np.nanquantile(ratio, 0.90)),
        },
        "scope_note": "Preprocessing diagnostic only. It does not evaluate target labels or establish a performance benefit.",
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
