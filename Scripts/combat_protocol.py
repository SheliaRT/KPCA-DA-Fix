"""Leakage-controlled ComBat fitting and application for two expression cohorts.

All matrices use samples x genes. The fitted ComBat model sees source-training
and target-adaptation samples only; source-validation and target-test matrices
are transformed with the corresponding already-seen batch IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

try:
    from .experiment_protocol import ordered_common_gene_indices
except ImportError:  # support direct execution/import from Scripts on sys.path
    from experiment_protocol import ordered_common_gene_indices


@dataclass(frozen=True)
class CombatPartitions:
    source_train: np.ndarray
    source_validation: np.ndarray
    target_adaptation: np.ndarray
    target_test: np.ndarray
    genes: np.ndarray
    estimates: dict[str, Any]
    fit_metadata: dict[str, Any]


def _validate_partition(X: np.ndarray, genes: Sequence[Hashable], name: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(X, dtype=float)
    names = np.asarray(genes, dtype=object)
    if matrix.ndim != 2 or names.ndim != 1 or matrix.shape[1] != names.size:
        raise ValueError(f"{name} must have shape (samples, len(genes))")
    if matrix.shape[0] == 0 or names.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if len(set(names.tolist())) != names.size:
        raise ValueError(f"{name} contains duplicate gene identifiers")
    return matrix, names


def _apply_fitted_batch(
    matrix: np.ndarray,
    batch_id: str,
    estimates: dict[str, Any],
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.empty_like(matrix, dtype=float)
    # This follows neuroCombatFromTraining's published adjustment equation.
    # The installed helper currently converts a one-element np.where result
    # with int(...), which raises under NumPy 2; use explicit scalar indexing.
    levels = np.asarray(estimates["batches"], dtype=str)
    matches = np.flatnonzero(levels == str(batch_id))
    if matches.size != 1:
        raise ValueError(f"Batch {batch_id!r} is not a unique fitted ComBat level")
    batch_index = int(matches[0])
    dat = matrix.T
    var_pooled = np.asarray(estimates["var.pooled"], dtype=float)
    stand_mean = (np.asarray(estimates["stand.mean"], dtype=float)[:, 0]
                  + np.asarray(estimates["mod.mean"], dtype=float).mean(axis=1))
    gamma = np.asarray(estimates["gamma.star"], dtype=float)[batch_index, :][:, None]
    delta = np.asarray(estimates["delta.star"], dtype=float)[batch_index, :][:, None]
    if np.any(var_pooled <= 0) or np.any(delta <= 0):
        raise ValueError("Fitted ComBat variances must be strictly positive")
    stand = stand_mean[:, None]
    bayesdata = (dat - stand) / np.sqrt(var_pooled)
    bayesdata = (bayesdata - gamma) / np.sqrt(delta)
    transformed = (bayesdata * np.sqrt(var_pooled)) + stand
    transformed = transformed.T
    transformed = np.asarray(transformed, dtype=float)
    if transformed.shape != matrix.shape or not np.isfinite(transformed).all():
        raise RuntimeError("ComBat returned invalid transformed values")
    return transformed


def fit_apply_combat_partitions(
    source_train: np.ndarray,
    source_validation: np.ndarray,
    target_adaptation: np.ndarray,
    target_test: np.ndarray,
    source_genes: Sequence[Hashable],
    target_genes: Sequence[Hashable],
    *,
    source_batch: str = "source",
    target_batch: str = "target",
) -> CombatPartitions:
    """Fit ComBat on source-train + target-adapt and transform all partitions.

    Only one-to-one shared genes are returned, in source-gene order. Source
    validation and target test are excluded from fitting and are transformed
    with source and target batch IDs respectively. No class labels are accepted.
    """
    if not source_batch or not target_batch or source_batch == target_batch:
        raise ValueError("Source and target batch IDs must be distinct non-empty strings")
    source_train, source_names = _validate_partition(source_train, source_genes, "source_train")
    source_validation, source_val_names = _validate_partition(
        source_validation, source_genes, "source_validation")
    target_adaptation, target_names = _validate_partition(
        target_adaptation, target_genes, "target_adaptation")
    target_test, target_test_names = _validate_partition(
        target_test, target_genes, "target_test")
    if not np.array_equal(source_names, source_val_names):
        raise ValueError("Source train and validation gene orders differ")
    if not np.array_equal(target_names, target_test_names):
        raise ValueError("Target adaptation and test gene orders differ")

    source_idx, target_idx, shared_genes = ordered_common_gene_indices(
        source_names, target_names)
    if not np.array_equal(source_names[source_idx], target_names[target_idx]):
        raise RuntimeError("Shared-gene row alignment failed")
    xs_train = source_train[:, source_idx]
    xs_val = source_validation[:, source_idx]
    xt_adapt = target_adaptation[:, target_idx]
    xt_test = target_test[:, target_idx]

    fit_data = np.concatenate((xs_train, xt_adapt), axis=0)
    batches = np.concatenate((
        np.full(xs_train.shape[0], source_batch, dtype=object),
        np.full(xt_adapt.shape[0], target_batch, dtype=object),
    ))
    combat_fit = neuroCombat(
        dat=fit_data.T,
        covars=pd.DataFrame({"batch": batches}),
        batch_col="batch",
    )
    corrected = np.asarray(combat_fit["data"], dtype=float).T
    if corrected.shape != fit_data.shape or not np.isfinite(corrected).all():
        raise RuntimeError("ComBat fitting returned invalid corrected values")
    estimates = combat_fit["estimates"]
    corrected_source_train = corrected[:xs_train.shape[0]]
    corrected_target_adapt = corrected[xs_train.shape[0]:]
    corrected_source_validation = _apply_fitted_batch(xs_val, source_batch, estimates)
    corrected_target_test = _apply_fitted_batch(xt_test, target_batch, estimates)

    return CombatPartitions(
        source_train=corrected_source_train,
        source_validation=corrected_source_validation,
        target_adaptation=corrected_target_adapt,
        target_test=corrected_target_test,
        genes=shared_genes,
        estimates=estimates,
        fit_metadata={
            "fit_sample_counts": {
                "source_train": int(xs_train.shape[0]),
                "target_adaptation": int(xt_adapt.shape[0]),
            },
            "excluded_from_fit": {
                "source_validation": int(xs_val.shape[0]),
                "target_test": int(xt_test.shape[0]),
            },
            "source_batch_id": source_batch,
            "target_batch_id": target_batch,
            "shared_gene_count": int(shared_genes.size),
            "gene_order_policy": "one-to-one shared genes in source order; first duplicate row retained",
            "heldout_transform": "neuroCombatFromTraining adjustment equation; local NumPy-2-compatible implementation",
            "target_labels_used": False,
        },
    )
