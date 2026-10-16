"""Compare baseline and KPCA-DA image alignment in one label-free ICA frame.

Both saved runs must share a seed, cohort, split and preprocessing settings.
One feature-coordinate rotation and pixel scale are built from source-training
and target-adaptation gene embeddings across the pair; held-out target points
and labels are excluded from the frame. FastICA is fit once on source-training
images and applied unchanged to both held-out target image sets.
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
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with (run_dir / "fitted_preprocessing_and_map.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    with np.load(run_dir / "split_ids.npz", allow_pickle=False) as split:
        splits = {name: split[name].astype(str) for name in split.files}
    return metadata, fitted, splits


def verify_pair(base: dict[str, Any], adapted: dict[str, Any],
                split_base: dict[str, np.ndarray], split_adapted: dict[str, np.ndarray]) -> None:
    for key in ("seed", "source_file_sha256", "target_file_sha256"):
        if base.get(key) != adapted.get(key):
            raise ValueError(f"Paired saved runs differ in {key}")
    for key in ("source_train", "source_validation", "target_adaptation", "target_test",
                "source_train_patient", "source_validation_patient",
                "target_adaptation_patient", "target_test_patient"):
        if not np.array_equal(split_base[key], split_adapted[key]):
            raise ValueError(f"Paired saved runs differ in split IDs: {key}")
    common_keys = (
        "source_validation_size", "target_test_size", "variance_percentile",
        "feature_scope", "clip_scaled_values", "align_shared_genes",
        "anchor_shared_pixels", "image_size", "model", "classifier_epochs",
        "early_stopping_patience", "scheduler_patience", "classifier_lr",
        "weight_decay", "batch_size", "source_validation_threshold", "no_pretrained",
    )
    for key in common_keys:
        if base.get("config", {}).get(key) != adapted.get("config", {}).get(key):
            raise ValueError(f"Paired saved runs differ in common setting {key}")


def load_domains(metadata: dict[str, Any], fitted: dict[str, Any],
                 splits: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_path, target_path = resolve(metadata["source_file"]), resolve(metadata["target_file"])
    if sha256(source_path) != metadata["source_file_sha256"]:
        raise ValueError("Source expression hash differs from run metadata")
    if sha256(target_path) != metadata["target_file_sha256"]:
        raise ValueError("Target expression hash differs from run metadata")
    source_gxp, source_genes, source_ids, _ = load_expression_dataset(str(source_path))
    target_gxp, target_genes, target_ids, _ = load_expression_dataset(str(target_path))
    source_ids, target_ids = np.asarray(source_ids, dtype=str), np.asarray(target_ids, dtype=str)
    source_map = {sample_id: i for i, sample_id in enumerate(source_ids)}
    target_map = {sample_id: i for i, sample_id in enumerate(target_ids)}
    train_ids, adapt_ids, test_ids = (splits[name] for name in
                                      ("source_train", "target_adaptation", "target_test"))
    if any(x not in source_map for x in train_ids):
        raise ValueError("A source-training ID is absent from the hashed input")
    if any(x not in target_map for x in np.concatenate([adapt_ids, test_ids])):
        raise ValueError("A target ID is absent from the hashed input")
    train_rows = np.asarray([source_map[x] for x in train_ids], dtype=int)
    adapt_rows = np.asarray([target_map[x] for x in adapt_ids], dtype=int)
    test_rows = np.asarray([target_map[x] for x in test_ids], dtype=int)

    preprocessor = fitted["preprocessor"]
    X_source = preprocessor.transform_source(source_gxp[:, train_rows].T, source_genes)
    X_adapt = preprocessor.transform_target(target_gxp[:, adapt_rows].T, target_genes)
    X_test = preprocessor.transform_target(target_gxp[:, test_rows].T, target_genes)
    if metadata["config"].get("feature_scope", "all") == "shared":
        source_idx, target_idx, _ = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_)
        X_source = X_source[:, source_idx]
        X_adapt, X_test = X_adapt[:, target_idx], X_test[:, target_idx]
    return X_source, X_adapt, X_test, test_ids


def marginal_w1(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    values = [float(wasserstein_distance(source[:, i], target[:, i])) for i in range(2)]
    return {
        "marginal_1d_wasserstein_by_component": values,
        "mean_marginal_1d_wasserstein": float(np.mean(values)),
        "euclidean_norm_of_marginal_1d_wasserstein": float(np.linalg.norm(values)),
        "definition": "Two marginal one-dimensional Wasserstein distances; not a multivariate Wasserstein distance.",
    }


def run(base_dir: Path, adapted_dir: Path, output: Path) -> dict[str, Any]:
    base_meta, base_fit, base_split = read_run(base_dir)
    adapted_meta, adapted_fit, adapted_split = read_run(adapted_dir)
    if base_meta.get("arm") != "baseline" or adapted_meta.get("arm") != "kpcada":
        raise ValueError("Supply a baseline run followed by its KPCA-DA comparison run")
    verify_pair(base_meta, adapted_meta, base_split, adapted_split)

    base_source, base_adapt, base_test, test_ids = load_domains(
        base_meta, base_fit, base_split)
    da_source, da_adapt, da_test, da_test_ids = load_domains(
        adapted_meta, adapted_fit, adapted_split)
    if not np.array_equal(test_ids, da_test_ids):
        raise ValueError("Baseline and KPCA-DA target-test samples differ")
    if not (np.allclose(base_source, da_source, rtol=0, atol=1e-10)
            and np.allclose(base_adapt, da_adapt, rtol=0, atol=1e-10)
            and np.allclose(base_test, da_test, rtol=0, atol=1e-10)):
        raise ValueError("Paired runs do not share the same expression preprocessing")

    base_transformer, da_transformer = base_fit["transformer"], adapted_fit["transformer"]
    for gene_scope in ("source_genes_", "target_genes_"):
        if not np.array_equal(getattr(base_fit["preprocessor"], gene_scope),
                              getattr(adapted_fit["preprocessor"], gene_scope)):
            raise ValueError(f"Baseline and KPCA-DA retained gene order differs: {gene_scope}")
    base_kpca, da_kpca = base_transformer._kpca_da, da_transformer._kpca_da
    # KPCA-DA stores gene coordinates as genes x 2; rows are the points used
    # by the image transform and the common geometric frame.
    Z_source_base, Z_source_da = base_kpca.Z_s_, da_kpca.Z_s_
    Z_adapt_base, Z_adapt_da = base_kpca.Z_t_, da_kpca.Z_t_
    if Z_source_base.shape != Z_source_da.shape or not np.allclose(
            Z_source_base, Z_source_da, rtol=0, atol=1e-8):
        raise ValueError("The baseline and KPCA-DA source gene embeddings differ")
    if Z_adapt_base.shape != Z_adapt_da.shape:
        raise ValueError("Baseline and KPCA-DA target adaptation embedding shapes differ")

    # One frame is learned from source/train and target/adaptation coordinates;
    # test data cannot rotate or rescale it. Reuse the same pixel map for both arms.
    frame_points = np.vstack([Z_source_base, Z_adapt_base, Z_adapt_da])
    _, rotation = DomainAdaptiveImageTransformer._minimum_bounding_rectangle(frame_points)
    rotated_source = Z_source_base @ rotation.T
    rotated_adapt_base = Z_adapt_base @ rotation.T
    rotated_adapt_da = Z_adapt_da @ rotation.T
    frame_points_rotated = np.vstack([rotated_source, rotated_adapt_base, rotated_adapt_da])
    bounds = (frame_points_rotated.min(axis=0), frame_points_rotated.max(axis=0))
    pixels = base_transformer.pixels
    source_coords = base_transformer._dm(rotated_source, pixels, bounds)
    target_coords_base = base_transformer._dm(rotated_adapt_base, pixels, bounds)
    target_coords_da = base_transformer._dm(rotated_adapt_da, pixels, bounds)
    source_shared_idx, target_shared_idx, shared_genes = ordered_common_gene_indices(
        base_fit["preprocessor"].source_genes_, base_fit["preprocessor"].target_genes_)
    shared_pixel_displacement_base = np.linalg.norm(
        source_coords[source_shared_idx] - target_coords_base[target_shared_idx], axis=1)
    shared_pixel_displacement_da = np.linalg.norm(
        source_coords[source_shared_idx] - target_coords_da[target_shared_idx], axis=1)

    source_images = base_transformer._transform_helper(
        base_source, source_coords, img_format="scalar")
    target_images_base = base_transformer._transform_helper(
        base_test, target_coords_base, img_format="scalar")
    target_images_da = base_transformer._transform_helper(
        da_test, target_coords_da, img_format="scalar")
    source_flat = source_images.reshape(len(source_images), -1).astype(np.float64)
    target_base_flat = target_images_base.reshape(len(target_images_base), -1).astype(np.float64)
    target_da_flat = target_images_da.reshape(len(target_images_da), -1).astype(np.float64)
    active_pixels = np.var(source_flat, axis=0) > 1e-12
    if active_pixels.sum() < 2:
        raise ValueError("Common source frame has fewer than two varying pixels")
    source_flat = source_flat[:, active_pixels]
    target_base_flat = target_base_flat[:, active_pixels]
    target_da_flat = target_da_flat[:, active_pixels]
    ica = FastICA(n_components=2, random_state=42, max_iter=500,
                  whiten="unit-variance")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        source_ica = ica.fit_transform(source_flat)
        base_target_ica = ica.transform(target_base_flat)
        da_target_ica = ica.transform(target_da_flat)
    convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)

    # The layout contains gene rows; save the coordinate map itself for audit.
    coordinates = {
        "source_gene_pixels": source_coords,
        "baseline_target_gene_pixels": target_coords_base,
        "kpcada_target_gene_pixels": target_coords_da,
    }
    target_patients = base_split["target_test_patient"]
    if not np.array_equal(patient_group_ids(test_ids).astype(str), target_patients):
        raise ValueError("Saved target patient IDs disagree with sample identifiers")
    result = {
        "valid_paired_diagnostic": True,
        "seed": base_meta["seed"],
        "source_file_sha256": base_meta["source_file_sha256"],
        "target_file_sha256": base_meta["target_file_sha256"],
        "source_train_ids": base_split["source_train"].tolist(),
        "target_adaptation_ids": base_split["target_adaptation"].tolist(),
        "target_test_ids": test_ids.tolist(),
        "target_test_patient_ids": target_patients.tolist(),
        "target_labels_used": False,
        "target_test_used_to_fit_frame_or_ica": False,
        "frame": {
            "rotation_fit_points": "source gene embeddings + baseline and KPCA-DA target-adaptation gene embeddings",
            "pixel_bounds_fit_points": "same as rotation points; held-out target excluded",
            "common_pixels": list(pixels),
            "active_source_training_pixels": int(active_pixels.sum()),
            "target_gene_pixel_overlap_baseline_kpcada": float(np.mean(
                np.all(target_coords_base == target_coords_da, axis=1))),
            "shared_gene_count": int(len(shared_genes)),
            "shared_gene_same_pixel_fraction_baseline": float(np.mean(
                shared_pixel_displacement_base == 0)),
            "shared_gene_same_pixel_fraction_kpcada": float(np.mean(
                shared_pixel_displacement_da == 0)),
            "shared_gene_mean_pixel_displacement_baseline": float(
                shared_pixel_displacement_base.mean()),
            "shared_gene_mean_pixel_displacement_kpcada": float(
                shared_pixel_displacement_da.mean()),
        },
        "ica": {
            "algorithm": "FastICA",
            "fit_on": "source-training images under shared frame",
            "n_components": 2,
            "random_state": 42,
            "max_iter": 500,
            "whiten": "unit-variance",
            "convergence_warning": convergence_warning,
        },
        "baseline_diagnostics": marginal_w1(source_ica, base_target_ica),
        "kpcada_diagnostics": marginal_w1(source_ica, da_target_ica),
        "scope_note": (
            "This is a paired, source-fitted projection comparison under one feature-coordinate "
            "frame. It evaluates a diagnostic, not classifier performance; no target labels or "
            "centroid shifts are used."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"), source_ica=source_ica,
        baseline_target_ica=base_target_ica, kpcada_target_ica=da_target_ica,
        ica_mean=ica.mean_, ica_components=ica.components_, ica_whitening=ica.whitening_,
        **coordinates,
        source_train_ids=base_split["source_train"],
        target_adaptation_ids=base_split["target_adaptation"],
        target_test_ids=test_ids,
        target_test_patient_ids=target_patients,
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("kpcada_run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.baseline_run if args.baseline_run.is_absolute() else ROOT / args.baseline_run
    adapted = args.kpcada_run if args.kpcada_run.is_absolute() else ROOT / args.kpcada_run
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = run(base, adapted, output)
    print(json.dumps({
        "result": "passed",
        "output": str(output),
        "baseline_diagnostics": result["baseline_diagnostics"],
        "kpcada_diagnostics": result["kpcada_diagnostics"],
        "frame": result["frame"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
