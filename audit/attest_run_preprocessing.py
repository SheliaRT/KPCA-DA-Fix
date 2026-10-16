"""Independently verify saved feature-filter/scaler fits and attest provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import (GeneAwareMinMax, ordered_common_gene_indices,
                                 patient_group_ids)
from kpcada import load_expression_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_run(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = metadata["config"]
    source_path = resolve(metadata["source_file"])
    target_path = resolve(metadata["target_file"])
    if sha256(source_path) != metadata["source_file_sha256"]:
        raise ValueError("Source input hash differs from the saved run")
    if sha256(target_path) != metadata["target_file_sha256"]:
        raise ValueError("Target input hash differs from the saved run")

    Xs_gxp, genes_source, ids_source, labels_source = load_expression_dataset(str(source_path))
    Xt_gxp, genes_target, ids_target, _labels_target = load_expression_dataset(str(target_path))
    ids_source = np.asarray(ids_source, dtype=str)
    ids_target = np.asarray(ids_target, dtype=str)
    labels_source = np.asarray(labels_source)
    keep_source = ~np.isin(labels_source, np.asarray(config["exclude_source_labels"]))
    ids_source = ids_source[keep_source]
    Xs = Xs_gxp[:, keep_source].T
    ys = labels_source[keep_source]
    Xt = Xt_gxp.T

    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        source_train_ids = split["source_train"].astype(str)
        target_adapt_ids = split["target_adaptation"].astype(str)
        map_source_ids = split["feature_map_source_train"].astype(str)
        map_target_ids = split["feature_map_target_adaptation"].astype(str)
        source_patient_ids = split["source_train_patient"].astype(str)
        target_patient_ids = split["target_test_patient"].astype(str)
        target_test_ids = split["target_test"].astype(str)
    source_rows = {sample_id: row for row, sample_id in enumerate(ids_source)}
    target_rows = {sample_id: row for row, sample_id in enumerate(ids_target)}
    if len(source_rows) != len(ids_source) or len(target_rows) != len(ids_target):
        raise ValueError("Input sample IDs are not unique")
    if any(value not in source_rows for value in source_train_ids):
        raise ValueError("Source-training split ID is absent from input")
    if any(value not in target_rows for value in target_adapt_ids):
        raise ValueError("Target-adaptation split ID is absent from input")
    train_rows = np.asarray([source_rows[value] for value in source_train_ids])
    adapt_rows = np.asarray([target_rows[value] for value in target_adapt_ids])
    expected_source_patients = patient_group_ids(source_train_ids).astype(str)
    expected_target_patients = patient_group_ids(target_test_ids).astype(str)
    if (not np.array_equal(source_patient_ids, expected_source_patients)
            or not np.array_equal(target_patient_ids, expected_target_patients)):
        raise ValueError("Saved split patient IDs do not match the built-in sample-ID parser")
    Xs_train = Xs[train_rows]
    Xt_adapt = Xt[adapt_rows]
    raw_arm = metadata["arm"] == "raw_deepinsight"

    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as stream:
        fitted = pickle.load(stream)
    preprocessor: GeneAwareMinMax = fitted["preprocessor"]
    if not np.array_equal(preprocessor.source_genes_original_, genes_source):
        raise ValueError("Saved source gene order differs from the input matrix")
    if not np.array_equal(preprocessor.target_genes_original_, genes_target):
        raise ValueError("Saved target gene order differs from the input matrix")
    expected_source_keep = GeneAwareMinMax._variance_keep_mask(
        Xs_train, float(config["variance_percentile"])
    )
    if not np.array_equal(preprocessor.source_keep_, expected_source_keep):
        raise ValueError("Saved source variance mask does not match deterministic rank filtering")
    source_min = Xs_train[:, expected_source_keep].min(axis=0)
    source_max = Xs_train[:, expected_source_keep].max(axis=0)
    if not (np.array_equal(preprocessor.source_scaler_.data_min_, source_min)
            and np.array_equal(preprocessor.source_scaler_.data_max_, source_max)):
        raise ValueError("Source scaler parameters do not match source-training values")

    source_retained = set(np.asarray(genes_source)[expected_source_keep].tolist())
    if raw_arm:
        expected_target_keep = np.asarray([gene in source_retained for gene in genes_target])
        if len(map_target_ids) != 0:
            raise ValueError("Raw DeepInsight run unexpectedly used target adaptation samples")
    else:
        private_idx = np.asarray([
            i for i, gene in enumerate(genes_target) if gene not in source_retained
        ], dtype=int)
        expected_target_keep = np.asarray([gene in source_retained for gene in genes_target])
        if private_idx.size:
            private_keep = GeneAwareMinMax._variance_keep_mask(
                Xt_adapt[:, private_idx], float(config["variance_percentile"])
            )
            expected_target_keep[private_idx] = private_keep
            retained_private = Xt_adapt[:, private_idx[private_keep]]
            scaler = preprocessor.target_private_scaler_
            if scaler is None or not (
                np.array_equal(scaler.data_min_, retained_private.min(axis=0))
                and np.array_equal(scaler.data_max_, retained_private.max(axis=0))
            ):
                raise ValueError("Target-private scaler does not match adaptation values")
    if not np.array_equal(preprocessor.target_keep_, expected_target_keep):
        raise ValueError("Saved target retention mask does not match permitted fit values")
    if not np.array_equal(map_source_ids, source_train_ids):
        raise ValueError("Map fit sample IDs differ from source-training IDs")
    if raw_arm and map_target_ids.size:
        raise ValueError("Raw map must not use target samples for fitting")
    source_map_genes = np.asarray(fitted["source_gene_ids_for_map"], dtype=str)
    target_map_genes = np.asarray(fitted["target_gene_ids_for_map"], dtype=str)
    if not np.array_equal(source_map_genes, target_map_genes):
        if config.get("feature_scope") != "all":
            raise ValueError("Shared-scope map gene order differs between domains")
    if config.get("feature_scope") == "shared":
        _, _, shared = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_
        )
        if not np.array_equal(source_map_genes, shared) or not np.array_equal(target_map_genes, shared):
            raise ValueError("Shared feature-scope map genes do not match the filtered intersection")

    policy = {
        "variance_filter": "stable rank, discard floor(percentile * feature_count); remove constants",
        "variance_percentile": float(config["variance_percentile"]),
        "variance_tie_break": "stable input gene order",
        "source_filter_fit_scope": "source training samples only",
        "target_private_filter_fit_scope": "not fitted" if raw_arm else "target adaptation samples only",
        "shared_gene_scaler_fit_scope": "source training samples only",
        "target_private_scaler_fit_scope": "not fitted" if raw_arm else "target adaptation samples only",
        "clip_scaled_values": bool(config["clip_scaled_values"]),
    }
    return {
        "result": "passed",
        "run_dir": str(run_dir),
        "run_metrics_sha256": sha256(metrics_path),
        "source_file_sha256": metadata["source_file_sha256"],
        "target_file_sha256": metadata["target_file_sha256"],
        "source_train_sample_count": len(source_train_ids),
        "target_adaptation_sample_count": len(target_adapt_ids),
        "source_features_retained": int(expected_source_keep.sum()),
        "target_features_retained": int(expected_target_keep.sum()),
        "saved_source_filter_matches_recomputed_rank_rule": True,
        "saved_target_filter_matches_permitted_fit_scope": True,
        "saved_scaler_extrema_match_fit_values": True,
        "representation_fit_sample_ids_verified": True,
        "patient_mapping": {
            "source": {
                "source": "built-in sample-ID parser",
                "mapping_file": None,
                "mapping_file_sha256": None,
            },
            "target": {
                "source": "built-in sample-ID parser",
                "mapping_file": None,
                "mapping_file_sha256": None,
            },
        },
        "policy": policy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    report = verify_run(run_dir)
    output = run_dir / "preprocessing_policy_attestation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": report["result"], "output": str(output),
                      "source_features_retained": report["source_features_retained"],
                      "target_features_retained": report["target_features_retained"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
