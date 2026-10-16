"""Rebuild a label-free, source-fitted ICA diagnostic from a saved TvsN run.

Fits FastICA only on source-training images and applies that unchanged
projection to held-out target images. No centroid adjustment or target-label
access occurs. Component-wise Wasserstein distances are marginal diagnostics,
not a multivariate Wasserstein distance and not comparable across runs whose
image coordinate systems differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from experiment_protocol import ordered_common_gene_indices, patient_group_ids
from kpcada import load_expression_dataset


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def summarize(source_ica: np.ndarray, target_ica: np.ndarray) -> dict[str, Any]:
    """Return unshifted centroid and explicitly marginal distance summaries."""
    if source_ica.ndim != 2 or target_ica.ndim != 2:
        raise ValueError("ICA coordinates must be two-dimensional arrays")
    if source_ica.shape[1] != 2 or target_ica.shape[1] != 2:
        raise ValueError("Expected exactly two ICA components")
    marginal = [float(wasserstein_distance(source_ica[:, i], target_ica[:, i]))
                for i in range(2)]
    return {
        "source_centroid": source_ica.mean(axis=0).tolist(),
        "target_centroid": target_ica.mean(axis=0).tolist(),
        "centroid_euclidean_distance": float(np.linalg.norm(
            source_ica.mean(axis=0) - target_ica.mean(axis=0))),
        "marginal_1d_wasserstein_by_component": marginal,
        "mean_marginal_1d_wasserstein": float(np.mean(marginal)),
        "euclidean_norm_of_marginal_1d_wasserstein": float(np.linalg.norm(marginal)),
        "metric_note": (
            "The last two values summarize the two marginal 1D distances; neither is "
            "a fully computed multivariate Wasserstein distance."
        ),
    }


def run(run_dir: Path, output: Path | None = None) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    source_path, target_path = resolve(metrics["source_file"]), resolve(metrics["target_file"])
    if digest(source_path) != metrics["source_file_sha256"]:
        raise ValueError("Source file hash differs from the saved run")
    if digest(target_path) != metrics["target_file_sha256"]:
        raise ValueError("Target file hash differs from the saved run")

    source_gxp, source_genes, source_ids, _ = load_expression_dataset(str(source_path))
    target_gxp, target_genes, target_ids, _ = load_expression_dataset(str(target_path))
    source_ids, target_ids = np.asarray(source_ids, dtype=str), np.asarray(target_ids, dtype=str)
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        source_train_ids = split["source_train"].astype(str)
        target_test_ids = split["target_test"].astype(str)
        expected_target_patients = split["target_test_patient"].astype(str)
    if not np.array_equal(patient_group_ids(target_test_ids).astype(str),
                          expected_target_patients):
        raise ValueError("Saved held-out target patient IDs disagree with GEO sample IDs")
    source_rows_by_id = {sample_id: i for i, sample_id in enumerate(source_ids)}
    target_rows_by_id = {sample_id: i for i, sample_id in enumerate(target_ids)}
    if any(sample_id not in source_rows_by_id for sample_id in source_train_ids):
        raise ValueError("A source-training ID is missing from the hashed source matrix")
    if any(sample_id not in target_rows_by_id for sample_id in target_test_ids):
        raise ValueError("A target-test ID is missing from the hashed target matrix")
    source_rows = np.asarray([source_rows_by_id[x] for x in source_train_ids], dtype=int)
    target_rows = np.asarray([target_rows_by_id[x] for x in target_test_ids], dtype=int)

    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    preprocessor, transformer = fitted["preprocessor"], fitted["transformer"]
    X_source = preprocessor.transform_source(source_gxp[:, source_rows].T, source_genes)
    X_target = preprocessor.transform_target(target_gxp[:, target_rows].T, target_genes)
    if metrics.get("config", {}).get("feature_scope", "all") == "shared":
        source_idx, target_idx, _ = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_)
        X_source, X_target = X_source[:, source_idx], X_target[:, target_idx]

    # Scalar images exactly represent the repeated grayscale RGB image before
    # classifier channel normalization. Drop only pixels constant in source
    # training; ICA cannot learn a component from a constant source feature.
    source_images = transformer.transform_source(X_source, img_format="scalar")
    target_images = transformer.transform_target(X_target, img_format="scalar")
    source_flat = source_images.reshape(len(source_images), -1).astype(np.float64)
    target_flat = target_images.reshape(len(target_images), -1).astype(np.float64)
    active = np.var(source_flat, axis=0) > 1e-12
    if int(active.sum()) < 2:
        raise ValueError("Fewer than two source-varying image pixels are available for ICA")
    source_flat, target_flat = source_flat[:, active], target_flat[:, active]

    ica = FastICA(n_components=2, random_state=42, max_iter=500,
                  whiten="unit-variance")
    convergence_warning = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        source_ica = ica.fit_transform(source_flat)
        target_ica = ica.transform(target_flat)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    diagnostics = summarize(source_ica, target_ica)
    result = {
        "run_dir": str(run_dir),
        "arm": metrics["arm"],
        "seed": metrics["seed"],
        "source_file_sha256": metrics["source_file_sha256"],
        "target_file_sha256": metrics["target_file_sha256"],
        "projection": {
            "algorithm": "sklearn FastICA",
            "fit_partition": "source training only",
            "target_partition": "held-out target test",
            "components": 2,
            "random_state": 42,
            "max_iter": 500,
            "whiten": "unit-variance",
            "input_representation": "scalar grayscale DeepInsight image; repeated RGB channels collapsed",
            "active_source_varying_pixels": int(active.sum()),
            "learned_component_shape": list(ica.components_.shape),
            "target_labels_used": False,
            "centroid_shift_applied": False,
            "convergence_warning": convergence_warning,
        },
        "source_train_ids": source_train_ids.tolist(),
        "target_test_ids": target_test_ids.tolist(),
        "target_test_patient_ids": expected_target_patients.tolist(),
        "diagnostics": diagnostics,
        "cross_run_comparison_caveat": (
            "ICA is fit separately for each run; absolute values should not be compared "
            "across runs with different pixel coordinate systems."
        ),
    }
    output_json = output or (run_dir / "source_fitted_ica_diagnostics.json")
    output_npz = output_json.with_suffix(".npz")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, source_ica=source_ica, target_ica=target_ica,
                        ica_mean=ica.mean_, ica_components=ica.components_,
                        ica_whitening=ica.whitening_,
                        source_train_ids=source_train_ids,
                        target_test_ids=target_test_ids,
                        target_test_patient_ids=expected_target_patients)
    result["raw_coordinates_npz"] = str(output_npz)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    output = args.output
    if output is not None and not output.is_absolute():
        output = ROOT / output
    result = run(run_dir, output)
    print(json.dumps({"result": "passed", "run_dir": result["run_dir"],
                      "diagnostics": result["diagnostics"],
                      "projection": result["projection"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
