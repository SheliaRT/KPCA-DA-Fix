"""Preflight the paper's train/adapt-only ComBat correction on the real TvsN data."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from combat_protocol import fit_apply_combat_partitions
from experiment_protocol import make_patient_splits, patient_group_ids
from kpcada import load_expression_dataset


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_abs_standardized_shift(source: np.ndarray, target: np.ndarray) -> float:
    source_var = np.var(source, axis=0, ddof=1)
    target_var = np.var(target, axis=0, ddof=1)
    denom = np.sqrt((source_var + target_var) / 2.0)
    valid = denom > 1e-12
    values = np.abs(source.mean(axis=0)[valid] - target.mean(axis=0)[valid]) / denom[valid]
    return float(values.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path", default="Data/TvsN/TCGA_BRCA_NvT_new.csv")
    parser.add_argument("--target-path", default="Data/TvsN/GSE45498_TvsN_official_200.csv")
    parser.add_argument("--exclude-source-label", action="append", default=["Unknown"])
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", default="audit/combat_readiness.json")
    args = parser.parse_args()
    source_path, target_path = resolve(args.source_path), resolve(args.target_path)
    source_gxp, source_genes, source_ids, source_labels = load_expression_dataset(str(source_path))
    target_gxp, target_genes, target_ids, _target_labels = load_expression_dataset(str(target_path))
    keep = ~np.isin(source_labels, np.asarray(args.exclude_source_label))
    source_gxp, source_ids, source_labels = source_gxp[:, keep], source_ids[keep], source_labels[keep]
    source_ids, target_ids = np.asarray(source_ids, dtype=str), np.asarray(target_ids, dtype=str)
    source_groups, target_groups = patient_group_ids(source_ids), patient_group_ids(target_ids)
    splits = make_patient_splits(
        source_labels, source_ids=source_groups, target_ids=target_groups,
        source_validation_size=0.2, target_test_size=0.3,
        random_state=args.seed, stratify_target=False,
    )
    source, target = source_gxp.T, target_gxp.T
    corrected = fit_apply_combat_partitions(
        source[splits.source_train], source[splits.source_validation],
        target[splits.target_adapt], target[splits.target_test],
        source_genes, target_genes,
    )
    raw_source_idx, raw_target_idx = np.flatnonzero(splits.source_train), np.flatnonzero(splits.target_adapt)
    raw_s = source[raw_source_idx][:, [np.flatnonzero(source_genes == gene)[0] for gene in corrected.genes]]
    raw_t = target[raw_target_idx][:, [np.flatnonzero(target_genes == gene)[0] for gene in corrected.genes]]
    result = {
        "result": "passed",
        "seed": args.seed,
        "source_file": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
        "target_file": str(target_path.relative_to(ROOT) if target_path.is_relative_to(ROOT) else target_path),
        "source_file_sha256": sha256(source_path),
        "target_file_sha256": sha256(target_path),
        "source_patient_count": int(np.unique(source_groups).size),
        "target_patient_count": int(np.unique(target_groups).size),
        "split_counts": corrected.fit_metadata,
        "shared_genes": int(corrected.genes.size),
        "target_test_batch_id": corrected.fit_metadata["target_batch_id"],
        "mean_abs_gene_standardized_shift_before": mean_abs_standardized_shift(raw_s, raw_t),
        "mean_abs_gene_standardized_shift_after": mean_abs_standardized_shift(
            corrected.source_train, corrected.target_adaptation),
        "corrected_values_finite": bool(all(np.isfinite(values).all() for values in (
            corrected.source_train, corrected.source_validation,
            corrected.target_adaptation, corrected.target_test))),
        "note": "ComBat was fit only on source train and target adaptation. This preflight does not train or score a classifier.",
    }
    out = resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
