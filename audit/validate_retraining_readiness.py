"""Run fast data/protocol checks before any classifier retraining.

This intentionally does not fit KPCA or a classifier. It validates the active
TvsN data path, patient splits, fitted preprocessing, and shared pixel frame.
Run after activating the repository-local environment:
    python audit/validate_retraining_readiness.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "Scripts"))

from experiment_protocol import (GeneAwareMinMax, make_patient_splits,
                                 ordered_common_gene_indices,
                                 patient_group_ids)
from kpcada import load_expression_dataset
from pyDeepInsightv3.image_transformer_new import DomainAdaptiveImageTransformer


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-path", default="Data/TvsN/TCGA_BRCA_NvT_new.csv")
parser.add_argument("--target-path", default="Data/TvsN/GSE45498_TvsN_official_200.csv")
parser.add_argument("--exclude-source-labels", nargs="*", default=["Unknown"])
parser.add_argument("--clip-scaled-values", action="store_true")
parser.add_argument("--output", default="audit/retraining_readiness.json")
args = parser.parse_args()

def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_path = resolve_path(args.source_path)
target_path = resolve_path(args.target_path)
source_gxs, source_genes, source_ids, source_labels = load_expression_dataset(str(source_path))
target_gxs, target_genes, target_ids, target_labels = load_expression_dataset(str(target_path))
source_ids = np.asarray(source_ids, dtype=str)
excluded_mask = np.isin(source_labels, np.asarray(args.exclude_source_labels))
excluded_source_ids = source_ids[excluded_mask].tolist()
if excluded_mask.any():
    source_gxs = source_gxs[:, ~excluded_mask]
    source_ids = source_ids[~excluded_mask]
    source_labels = source_labels[~excluded_mask]
if np.unique(source_labels).size != 2:
    raise ValueError(
        "TvsN readiness requires exactly two source classes after exclusions; "
        f"found {np.unique(source_labels).tolist()}"
    )
source = source_gxs.T
target = target_gxs.T

splits = make_patient_splits(
    source_labels,
    source_ids=patient_group_ids(source_ids),
    target_ids=patient_group_ids(target_ids),
    source_validation_size=0.2,
    target_test_size=0.3,
    random_state=42,
)
source_patient_ids = patient_group_ids(source_ids)
target_patient_ids = patient_group_ids(target_ids)
assert not (set(source_patient_ids[splits.source_train])
            & set(source_patient_ids[splits.source_validation]))
assert not (set(target_patient_ids[splits.target_adapt])
            & set(target_patient_ids[splits.target_test]))
assert len(splits.source_train) + len(splits.source_validation) == len(source_ids)
assert len(splits.target_adapt) + len(splits.target_test) == len(target_ids)

X_source_train = source[splits.source_train]
X_source_validation = source[splits.source_validation]
X_target_adaptation = target[splits.target_adapt]
X_target_test = target[splits.target_test]
preprocessor = GeneAwareMinMax(
    variance_percentile=30.0,
    clip_scaled_values=args.clip_scaled_values,
).fit(
    X_source_train,
    source_genes,
    X_target_adaptation,
    target_genes,
)
X_source_train_scaled = preprocessor.transform_source(X_source_train, source_genes)
X_source_validation_scaled = preprocessor.transform_source(X_source_validation, source_genes)
X_target_adaptation_scaled = preprocessor.transform_target(X_target_adaptation, target_genes)
X_target_test_scaled = preprocessor.transform_target(X_target_test, target_genes)
assert np.isfinite(X_source_train_scaled).all()
assert np.isfinite(X_source_validation_scaled).all()
assert np.isfinite(X_target_adaptation_scaled).all()
assert np.isfinite(X_target_test_scaled).all()
assert X_target_adaptation_scaled.shape[1] == len(preprocessor.target_genes_)
assert X_target_test_scaled.shape[1] == len(preprocessor.target_genes_)
assert np.array_equal(preprocessor.target_genes_, target_genes[preprocessor.target_keep_])
source_common, target_common, common_genes = ordered_common_gene_indices(
    preprocessor.source_genes_, preprocessor.target_genes_
)
assert np.array_equal(preprocessor.source_genes_[source_common], common_genes)
assert np.array_equal(preprocessor.target_genes_[target_common], common_genes)

def scaling_range_summary(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=float)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "below_zero_fraction": float(np.mean(values < 0.0)),
        "above_one_fraction": float(np.mean(values > 1.0)),
        "outside_unit_interval_fraction": float(np.mean((values < 0.0) | (values > 1.0))),
    }

shared_target_columns = np.isin(preprocessor.target_genes_, preprocessor.shared_genes_)

# Verify a shared point receives the same pixel coordinate across domains while
# distinct points remain distinct; this catches independent per-domain scaling.
mapper = DomainAdaptiveImageTransformer(kpca_da_params={"epochs": 0}, pixels=(31, 31))
mapper._xrot_source = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
mapper._xrot_target = np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 2.0]])
mapper._calculate_coords()
assert np.array_equal(mapper.coords_source()[1], mapper.coords_target()[1])
assert not np.array_equal(mapper.coords_source(), mapper.coords_target())

report = {
    "result": "passed",
    "note": "No KPCA fitting or classifier training was performed.",
    "source_file": str(source_path),
    "target_file": str(target_path),
    "source_file_sha256": sha256(source_path),
    "target_file_sha256": sha256(target_path),
    "source_excluded_samples": excluded_source_ids,
    "clip_scaled_values": args.clip_scaled_values,
    "preprocessing_policy": {
        "variance_filter": "stable rank, discard floor(percentile * feature_count); remove constants",
        "variance_percentile": 30.0,
        "variance_tie_break": "stable input gene order",
        "source_filter_fit_scope": "source training samples only",
        "target_private_filter_fit_scope": "target adaptation samples only",
        "shared_gene_scaler_fit_scope": "source training samples only",
        "target_private_scaler_fit_scope": "target adaptation samples only",
    },
    "source": {
        "samples": int(len(source_ids)),
        "patients": int(len(np.unique(source_patient_ids))),
        "genes": int(len(source_genes)),
        "train": int(len(splits.source_train)),
        "validation": int(len(splits.source_validation)),
        "genes_after_filter": int(len(preprocessor.source_genes_)),
    },
    "target": {
        "samples": int(len(target_ids)),
        "patients": int(len(np.unique(target_patient_ids))),
        "genes": int(len(target_genes)),
        "adaptation": int(len(splits.target_adapt)),
        "test": int(len(splits.target_test)),
        "genes_after_policy": int(len(preprocessor.target_genes_)),
    },
    "shared_genes_after_filtering": int(len(common_genes)),
    "training_only_scaling_ranges": {
        "source_training": scaling_range_summary(X_source_train_scaled),
        "source_validation": scaling_range_summary(X_source_validation_scaled),
        "target_adaptation_all_retained": scaling_range_summary(X_target_adaptation_scaled),
        "target_adaptation_shared_genes": scaling_range_summary(
            X_target_adaptation_scaled[:, shared_target_columns]
        ),
    },
    "seed": 42,
}
output_path = resolve_path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
