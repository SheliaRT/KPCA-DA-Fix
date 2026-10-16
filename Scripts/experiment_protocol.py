"""Leakage-resistant data preparation helpers for KPCA-DA experiments.

Arrays use samples x genes. Split raw patient rows before fitting filters,
scalers, embeddings, image maps, or augmentation. Target labels are optional
and should be omitted for unsupervised adaptation. By default the target split
uses patient groups only, without target-label stratification.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Hashable, Optional, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class PatientSplits:
    """Row indices for disjoint source train/validation and target partitions."""

    source_train: np.ndarray
    source_validation: np.ndarray
    target_adapt: np.ndarray
    target_test: np.ndarray

    def validate(self) -> None:
        for left, right in ((self.source_train, self.source_validation),
                            (self.target_adapt, self.target_test)):
            if np.unique(left).size != left.size or np.unique(right).size != right.size:
                raise ValueError("A partition contains a repeated row index")
            if np.intersect1d(left, right).size:
                raise ValueError("Patient partitions must be disjoint within a domain")


def patient_group_ids(sample_ids: Sequence[Hashable]) -> np.ndarray:
    """Normalize sample/accession IDs to patient groups for known cohorts.

    TCGA aliquots are grouped by the 12-character case barcode. GSE45498
    official IDs encode the numeric patient as ``GSE45498:<patient>:<GSM>``.
    Legacy titles are also recognized. Unrecognized IDs remain unique
    sample-level groups and should be reviewed before a patient-level claim.
    """
    normalized: list[str] = []
    for sample in np.asarray(sample_ids, dtype=object).tolist():
        value = str(sample).strip()
        tcga = re.match(r"^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", value)
        if tcga:
            normalized.append(f"TCGA:{tcga.group(1)}")
            continue
        official_geo = re.match(r"^GSE45498:(\d+):GSM\d+$", value)
        if official_geo:
            normalized.append(f"GSE45498:{official_geo.group(1)}")
            continue
        clean = re.sub(r"\.\d+$", "", value)
        geo = re.match(r"^(?:Normal_|TAS09\s+12\s+|TAS10-15-)(\d+)", clean)
        if geo:
            normalized.append(f"GSE45498:{geo.group(1)}")
        else:
            normalized.append(f"sample:{value}")
    return np.asarray(normalized, dtype=object)


def patient_groups_from_mapping(
    sample_ids: Sequence[Hashable], sample_to_patient: dict[str, Hashable]
) -> np.ndarray:
    """Resolve cohort sample IDs through an explicit authoritative mapping."""
    ids = np.asarray(sample_ids, dtype=object)
    missing = [str(value).strip() for value in ids
               if str(value).strip() not in sample_to_patient]
    if missing:
        raise ValueError(
            "Patient map lacks sample IDs: " + ", ".join(missing[:5])
        )
    return np.asarray([
        f"patient:{str(sample_to_patient[str(value).strip()]).strip()}"
        for value in ids
    ], dtype=object)


def make_patient_splits(
    y_source: Sequence[Any],
    y_target: Optional[Sequence[Any]] = None,
    *,
    source_ids: Optional[Sequence[Hashable]] = None,
    target_ids: Optional[Sequence[Hashable]] = None,
    target_sample_count: Optional[int] = None,
    source_validation_size: float = 0.2,
    target_test_size: float = 0.3,
    random_state: int = 42,
    stratify_target: bool = False,
) -> PatientSplits:
    """Split original, unaugmented patient rows by patient group.

    Source labels stratify the source train/validation partition. Under the
    unsupervised protocol, omit ``y_target`` and supply ``target_ids``; the
    target adaptation/test partition is drawn by patient group only. Set
    ``stratify_target=True`` only for a declared diagnostic split, in which
    case target labels are required. All rows in a supplied group stay
    together, including paired samples with different labels.
    """
    ys = np.asarray(y_source)
    yt = None if y_target is None else np.asarray(y_target)
    if ys.ndim != 1 or not ys.size:
        raise ValueError("Source labels must be a non-empty 1-D array")
    if yt is not None and (yt.ndim != 1 or not yt.size):
        raise ValueError("If provided, target labels must be a non-empty 1-D array")
    source_id_values = np.asarray(source_ids if source_ids is not None else np.arange(ys.size), dtype=object)
    if target_ids is not None:
        target_id_values = np.asarray(target_ids, dtype=object)
        target_rows = target_id_values.size
    elif yt is not None:
        target_rows = yt.size
        target_id_values = np.arange(target_rows, dtype=object)
    elif target_sample_count is not None and target_sample_count > 0:
        target_rows = int(target_sample_count)
        target_id_values = np.arange(target_rows, dtype=object)
    else:
        raise ValueError("Provide target_ids, target labels, or target_sample_count")
    if source_id_values.shape != ys.shape or target_id_values.shape != (target_rows,):
        raise ValueError("Patient ID arrays must match their sample rows")
    if yt is not None and yt.size != target_rows:
        raise ValueError("Target labels must match target sample rows")
    if source_ids is not None and target_ids is not None:
        overlap = set(source_id_values.tolist()) & set(target_id_values.tolist())
        if overlap:
            raise ValueError("Patient IDs overlap across source and target domains")
    for name, size in (("source_validation_size", source_validation_size),
                       ("target_test_size", target_test_size)):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} must be between zero and one")

    def split_groups(ids: np.ndarray, labels: Optional[np.ndarray], test_size: float,
                     seed: int, *, stratify: bool) -> tuple[np.ndarray, np.ndarray]:
        if stratify:
            if labels is None:
                raise ValueError("Labels are required when stratifying a split")
            n_splits = int(round(1.0 / test_size))
            if n_splits < 2 or not np.isclose(1.0 / n_splits, test_size):
                raise ValueError("Stratified grouped splits require test_size to be 1/n")
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed)
            left, right = next(splitter.split(np.zeros(labels.size), labels, groups=ids))
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=seed)
            left, right = next(splitter.split(np.zeros(ids.size), groups=ids))
        if set(ids[left].tolist()) & set(ids[right].tolist()):
            raise RuntimeError("Patient grouping failed to keep rows together")
        return np.asarray(left), np.asarray(right)

    source_train, source_validation = split_groups(
        source_id_values, ys, source_validation_size, random_state, stratify=True)
    target_adapt, target_test = split_groups(
        target_id_values, yt, target_test_size, random_state,
        stratify=stratify_target)
    result = PatientSplits(
        source_train=np.asarray(source_train),
        source_validation=np.asarray(source_validation),
        target_adapt=np.asarray(target_adapt),
        target_test=np.asarray(target_test),
    )
    result.validate()
    return result


def ordered_common_gene_indices(
    source_genes: Sequence[Hashable], target_genes: Sequence[Hashable]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one-to-one shared-gene indices in source order.

    Duplicate identifiers are resolved by retaining their first occurrence in
    each domain. The returned `genes` array is the authoritative row identity
    order for both matrices; callers should record this policy in run metadata.
    """
    source = np.asarray(source_genes, dtype=object)
    target = np.asarray(target_genes, dtype=object)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Gene identifiers must be one-dimensional")

    target_first: dict[Hashable, int] = {}
    for idx, gene in enumerate(target.tolist()):
        if gene not in target_first:
            target_first[gene] = idx

    source_first: set[Hashable] = set()
    source_idx: list[int] = []
    target_idx: list[int] = []
    common: list[Hashable] = []
    for idx, gene in enumerate(source.tolist()):
        if gene in source_first:
            continue
        source_first.add(gene)
        if gene in target_first:
            source_idx.append(idx)
            target_idx.append(target_first[gene])
            common.append(gene)
    if not common:
        raise ValueError("The domains have no shared gene identifiers")
    return (np.asarray(source_idx, dtype=int),
            np.asarray(target_idx, dtype=int),
            np.asarray(common, dtype=object))


class GeneAwareMinMax:
    """Fit permissible gene-wise scaling without reading target-test values.

    Shared genes use source-training min/max parameters. Target-only genes use
    target-adaptation min/max parameters because no source statistic exists for
    them. By default, values outside a fitted range remain outside [0, 1]; the
    optional clipping mode bounds transformed values for image pipelines whose
    input contract requires unit-range intensities. Each domain retains its own
    ordered gene matrix; shared identifiers can be matched later with
    :func:`ordered_common_gene_indices`.
    """

    def __init__(self, variance_percentile: float = 30.0,
                 clip_scaled_values: bool = False) -> None:
        if not 0.0 <= variance_percentile < 100.0:
            raise ValueError("variance_percentile must be in [0, 100)")
        self.variance_percentile = float(variance_percentile)
        self.clip_scaled_values = bool(clip_scaled_values)
        self.source_genes_: Optional[np.ndarray] = None
        self.target_genes_: Optional[np.ndarray] = None
        self.source_genes_original_: Optional[np.ndarray] = None
        self.target_genes_original_: Optional[np.ndarray] = None
        self.source_keep_: Optional[np.ndarray] = None
        self.target_keep_: Optional[np.ndarray] = None
        self.source_scaler_: Optional[MinMaxScaler] = None
        self.target_private_scaler_: Optional[MinMaxScaler] = None
        self.shared_genes_: Optional[np.ndarray] = None
        self._target_to_source: dict[Hashable, int] = {}
        self._target_private: np.ndarray = np.array([], dtype=int)

    @staticmethod
    def _check_matrix(X: np.ndarray, genes: Sequence[Hashable], name: str) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(X, dtype=float)
        names = np.asarray(genes, dtype=object)
        if matrix.ndim != 2 or names.ndim != 1 or matrix.shape[1] != names.size:
            raise ValueError(f"{name} must have shape (samples, len(genes))")
        if matrix.shape[0] == 0 or names.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return matrix, names

    @staticmethod
    def _variance_keep_mask(values: np.ndarray, percentile: float) -> np.ndarray:
        """Drop the requested low-variance fraction with stable tie handling.

        A threshold comparison such as ``variance >= percentile_cutoff`` can
        retain far more than the requested fraction when many genes tie at the
        cutoff (especially constant genes at zero). Rank selection makes the
        retained count deterministic: tied genes follow input order, constant
        genes are always removed, and the discarded count is rounded down.
        Constant genes count toward the requested discard fraction; if there
        are more constants than that fraction, all constants are removed and
        the retained fraction can be lower.
        """
        variance = np.var(np.asarray(values, dtype=float), axis=0)
        if variance.ndim != 1 or variance.size == 0 or not np.isfinite(variance).all():
            raise ValueError("Variance filtering requires finite, non-empty feature values")
        keep = variance > 0.0
        discard = int(np.floor(variance.size * percentile / 100.0))
        # Stable sorting makes the policy for tied variances explicit and
        # reproducible across runs and platforms.
        order = np.argsort(variance, kind="stable")
        keep[order[:discard]] = False
        return keep

    def fit(
        self,
        X_source_train: np.ndarray,
        source_genes: Sequence[Hashable],
        X_target_adapt: np.ndarray,
        target_genes: Sequence[Hashable],
    ) -> "GeneAwareMinMax":
        source, sg = self._check_matrix(X_source_train, source_genes, "X_source_train")
        target, tg = self._check_matrix(X_target_adapt, target_genes, "X_target_adapt")
        if len(set(sg.tolist())) != sg.size or len(set(tg.tolist())) != tg.size:
            raise ValueError("Resolve duplicate gene rows before fitting the scaler")

        self.source_genes_original_ = sg.copy()
        self.target_genes_original_ = tg.copy()
        self.source_keep_ = self._variance_keep_mask(
            source, self.variance_percentile
        )
        if not self.source_keep_.any():
            raise ValueError("Source variance filter removed every gene")
        self.source_genes_ = sg[self.source_keep_]
        self.source_scaler_ = MinMaxScaler().fit(source[:, self.source_keep_])
        source_positions = {gene: i for i, gene in enumerate(self.source_genes_.tolist())}

        shared = [gene for gene in tg.tolist() if gene in source_positions]
        private = [i for i, gene in enumerate(tg.tolist()) if gene not in source_positions]
        target_private = target[:, private] if private else np.empty((target.shape[0], 0))
        if private:
            keep_private_local = self._variance_keep_mask(
                target_private, self.variance_percentile
            )
            self._target_private = np.asarray(private, dtype=int)[keep_private_local]
            private_kept = target[:, self._target_private]
            self.target_private_scaler_ = MinMaxScaler().fit(private_kept)
        else:
            self._target_private = np.array([], dtype=int)
            self.target_private_scaler_ = None

        keep_target = np.zeros(tg.size, dtype=bool)
        for i, gene in enumerate(tg.tolist()):
            if gene in source_positions or i in set(self._target_private.tolist()):
                keep_target[i] = True
        if not keep_target.any():
            raise ValueError("Target gene policy retained no features")
        self.target_keep_ = keep_target
        self.target_genes_ = tg[keep_target]
        self.shared_genes_ = np.asarray(shared, dtype=object)
        self._target_to_source = source_positions
        return self

    def fit_source_shared(
        self,
        X_source_train: np.ndarray,
        source_genes: Sequence[Hashable],
        target_genes: Sequence[Hashable],
    ) -> "GeneAwareMinMax":
        """Fit a source-only scaler and retain only genes shared with target.

        This path supports an inductive, no-adaptation DeepInsight baseline:
        source training values determine filtering/scaling and the target gene
        identifiers determine the feature intersection, without reading target
        expression values or fitting target-private feature statistics.
        """
        source, sg = self._check_matrix(X_source_train, source_genes,
                                        "X_source_train")
        tg = np.asarray(target_genes, dtype=object)
        if tg.ndim != 1 or tg.size == 0:
            raise ValueError("target_genes must be a non-empty one-dimensional sequence")
        if len(set(sg.tolist())) != sg.size or len(set(tg.tolist())) != tg.size:
            raise ValueError("Resolve duplicate gene rows before fitting the scaler")

        source_keep = self._variance_keep_mask(source, self.variance_percentile)
        if not source_keep.any():
            raise ValueError("Source variance filter removed every gene")

        self.source_genes_original_ = sg.copy()
        self.target_genes_original_ = tg.copy()
        self.source_keep_ = source_keep
        self.source_genes_ = sg[source_keep]
        self.source_scaler_ = MinMaxScaler().fit(source[:, source_keep])
        source_retained = set(self.source_genes_.tolist())
        self.target_keep_ = np.asarray([gene in source_retained for gene in tg],
                                       dtype=bool)
        if not self.target_keep_.any():
            raise ValueError("No retained source genes are shared with target")
        self.target_genes_ = tg[self.target_keep_]
        self.shared_genes_ = self.target_genes_.copy()
        self._target_to_source = {
            gene: index for index, gene in enumerate(self.source_genes_.tolist())
        }
        self._target_private = np.array([], dtype=int)
        self.target_private_scaler_ = None
        return self

    def transform_source(self, X: np.ndarray, genes: Sequence[Hashable]) -> np.ndarray:
        self._require_fitted()
        matrix, names = self._check_matrix(X, genes, "source matrix")
        if not np.array_equal(names, self.source_genes_original_):
            raise ValueError("Source genes/order differ from the fitted input")
        scaled = self.source_scaler_.transform(matrix[:, self.source_keep_])
        return np.clip(scaled, 0.0, 1.0) if self.clip_scaled_values else scaled

    def transform_target(self, X: np.ndarray, genes: Sequence[Hashable]) -> np.ndarray:
        self._require_fitted()
        matrix, names = self._check_matrix(X, genes, "target matrix")
        if not np.array_equal(names, self.target_genes_original_):
            raise ValueError("Target genes/order differ from the fitted input")
        columns: list[np.ndarray] = []
        source_positions = {gene: i for i, gene in enumerate(self.source_genes_.tolist())}
        private_set = set(self._target_private.tolist())
        private_order = self._target_private.tolist()
        if private_order:
            private_scaled = self.target_private_scaler_.transform(matrix[:, private_order])
            private_scaled_by_index = {
                original_idx: private_scaled[:, local_idx]
                for local_idx, original_idx in enumerate(private_order)
            }
        for i, gene in enumerate(names.tolist()):
            if gene in source_positions:
                source_idx = source_positions[gene]
                columns.append(matrix[:, i] * self.source_scaler_.scale_[source_idx]
                               + self.source_scaler_.min_[source_idx])
            elif i in private_set:
                columns.append(private_scaled_by_index[i])
        scaled = np.column_stack(columns)
        return np.clip(scaled, 0.0, 1.0) if self.clip_scaled_values else scaled

    def transform_target_by_gene_id(
        self, X: np.ndarray, genes: Sequence[Hashable]
    ) -> np.ndarray:
        """Scale new target rows after aligning columns by gene identifier.

        Extra genes are ignored. Missing genes that survived target fitting
        raise an error because neither the frozen scaler nor pixel map defines
        their replacement values. Missing genes removed during fitting are
        filled with finite zeros and remain excluded from the output.
        """
        self._require_fitted()
        matrix, names = self._check_matrix(X, genes, "target matrix")
        if len(set(names.tolist())) != names.size:
            raise ValueError("New target rows must have unique gene identifiers")
        supplied = {gene: i for i, gene in enumerate(names.tolist())}
        required = set(self.target_genes_.tolist())
        missing = sorted(required - set(supplied), key=str)
        if missing:
            raise ValueError(
                "New target rows are missing genes required by the fitted map: "
                f"{missing[:10]}"
            )
        aligned = np.zeros((matrix.shape[0], self.target_genes_original_.size),
                           dtype=float)
        for column, gene in enumerate(self.target_genes_original_.tolist()):
            input_column = supplied.get(gene)
            if input_column is not None:
                aligned[:, column] = matrix[:, input_column]
        return self.transform_target(aligned, self.target_genes_original_)

    def _require_fitted(self) -> None:
        if self.source_scaler_ is None or self.source_keep_ is None:
            raise RuntimeError("Call fit before transforming data")


def transform_target_to_images(
    preprocessor: GeneAwareMinMax,
    transformer: Any,
    X: np.ndarray,
    genes: Sequence[Hashable],
    *,
    feature_scope: str = "all",
    img_format: str = "rgb",
) -> np.ndarray:
    """Apply a frozen target scaler and pixel map to new expression rows.

    Columns are matched by gene ID, then reordered to the exact fitted map.
    The function never refits scaling, KPCA coordinates, or pixel positions.
    """
    scaled = preprocessor.transform_target_by_gene_id(X, genes)
    mapped_genes = preprocessor.target_genes_.copy()
    if feature_scope == "shared":
        _, target_idx, shared_genes = ordered_common_gene_indices(
            preprocessor.source_genes_, preprocessor.target_genes_
        )
        scaled = scaled[:, target_idx]
        mapped_genes = shared_genes
    elif feature_scope != "all":
        raise ValueError("feature_scope must be 'all' or 'shared'")
    fitted_genes = np.asarray(transformer._genes_target, dtype=object)
    if not np.array_equal(mapped_genes, fitted_genes):
        raise ValueError("Fitted pixel-map gene IDs differ from the requested feature scope")
    return transformer.transform_target(scaled, img_format=img_format)
