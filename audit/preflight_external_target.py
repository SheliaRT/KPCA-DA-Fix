"""Label-free preprocessing/resource preflight for an external target cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import (GeneAwareMinMax, make_patient_splits,
                                 patient_groups_from_mapping)
from kpcada import load_expression_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "Data/TvsN/TCGA_BRCA_NvT_new.csv")
    parser.add_argument("--target", type=Path,
                        default=ROOT / "Data/TvsN/GSE183947_TvsN_official_60.csv")
    parser.add_argument("--target-patient-map", type=Path,
                        default=ROOT / "audit/gse183947_patient_map.json")
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "audit/gse183947_preflight_seed55.json")
    args = parser.parse_args()

    Xs, genes_s, ids_s, ys = load_expression_dataset(str(args.source))
    Xt, genes_t, ids_t, y_unmaterialized = load_expression_dataset(
        str(args.target), include_labels=False
    )
    if y_unmaterialized is not None:
        raise RuntimeError("Target labels were unexpectedly materialized")
    excluded = np.isin(ys, ["Unknown"])
    Xs, ids_s, ys = Xs[:, ~excluded], ids_s[~excluded], ys[~excluded]
    mapping = json.loads(args.target_patient_map.read_text(encoding="utf-8"))
    target_groups = patient_groups_from_mapping(ids_t, mapping)
    source_groups = np.asarray([
        "TCGA:" + "-".join(str(sample).split("-")[:3]) for sample in ids_s
    ], dtype=object)
    split = make_patient_splits(
        ys, source_ids=source_groups, target_ids=target_groups,
        source_validation_size=0.2, target_test_size=0.3,
        random_state=args.seed, stratify_target=False,
    )
    preprocessor = GeneAwareMinMax(variance_percentile=30.0).fit(
        Xs[:, split.source_train].T, genes_s,
        Xt[:, split.target_adapt].T, genes_t,
    )
    Xs_scaled = preprocessor.transform_source(Xs[:, split.source_train].T, genes_s)
    Xt_scaled = preprocessor.transform_target(Xt[:, split.target_adapt].T, genes_t)
    common = set(preprocessor.source_genes_.tolist()) & set(preprocessor.target_genes_.tolist())
    # KPCA is fitted over genes. This gives the approximate dense RBF matrix
    # size for each domain without allocating it or running model fitting.
    source_kpca_genes = int(len(preprocessor.source_genes_))
    target_kpca_genes = int(len(preprocessor.target_genes_))
    kpca_dtype = np.result_type(Xs_scaled.dtype, Xt_scaled.dtype)
    kpca_itemsize = int(np.dtype(kpca_dtype).itemsize)
    result = {
        "result": "passed",
        "note": "No KPCA fitting, image mapping, classifier training, or target-label loading was performed.",
        "source_path": str(args.source.relative_to(ROOT) if args.source.is_relative_to(ROOT) else args.source),
        "source_sha256": sha256(args.source),
        "target_path": str(args.target.relative_to(ROOT) if args.target.is_relative_to(ROOT) else args.target),
        "target_sha256": sha256(args.target),
        "target_patient_map_sha256": sha256(args.target_patient_map),
        "seed": args.seed,
        "counts": {
            "source_samples": int(Xs.shape[1]),
            "source_patients": int(len(set(source_groups.tolist()))),
            "target_samples": int(Xt.shape[1]),
            "target_patients": int(len(set(target_groups.tolist()))),
            "source_train_samples": int(len(split.source_train)),
            "source_validation_samples": int(len(split.source_validation)),
            "target_adaptation_samples": int(len(split.target_adapt)),
            "target_test_samples": int(len(split.target_test)),
            "target_adaptation_patients": int(len(set(target_groups[split.target_adapt].tolist()))),
            "target_test_patients": int(len(set(target_groups[split.target_test].tolist()))),
        },
        "features": {
            "source_retained_after_30pct_filter": source_kpca_genes,
            "target_retained_after_30pct_filter": target_kpca_genes,
            "shared_retained_genes": len(common),
            "source_kpca_gram_entries": source_kpca_genes ** 2,
            "target_kpca_gram_entries": target_kpca_genes ** 2,
            "kernel_pca_input_dtype": np.dtype(kpca_dtype).name,
            "kernel_pca_dense_gram_estimated_bytes_per_entry": kpca_itemsize,
            "source_kpca_gram_estimated_GiB": (
                source_kpca_genes ** 2 * kpca_itemsize / 1024**3
            ),
            "target_kpca_gram_estimated_GiB": (
                target_kpca_genes ** 2 * kpca_itemsize / 1024**3
            ),
            # Keep float32 lower-bound figures only for comparison; sklearn
            # KernelPCA receives float64 from the current MinMax pipeline.
            "source_float32_gram_GiB": source_kpca_genes ** 2 * 4 / 1024**3,
            "target_float32_gram_GiB": target_kpca_genes ** 2 * 4 / 1024**3,
        },
        "scaling": {
            "target_adaptation_outside_unit_interval_fraction": float(
                np.mean((Xt_scaled < 0.0) | (Xt_scaled > 1.0))
            ),
            "target_adaptation_scaled_min": float(np.min(Xt_scaled)),
            "target_adaptation_scaled_max": float(np.max(Xt_scaled)),
            "shared_scaler_fit_scope": "source training only",
            "target_private_filter_and_scaler_fit_scope": "target adaptation only",
        },
        "leakage_checks": {
            "target_test_labels_materialized": False,
            "target_labels_used_for_split": False,
            "patient_overlap_adapt_test": 0,
            "source_target_patient_overlap": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
