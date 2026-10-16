"""Validate and summarize a matched, patient-grouped TvsN run pair.

Example:
    python audit/compare_tvsn_runs.py results/tvsn/baseline_run results/tvsn/kpcada_run

The utility rejects legacy artifacts without patient-group manifests, mismatched
splits/configuration, or any patient overlap across train/validation/adaptation/test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Scripts"))
from evaluation_protocol import classification_metrics


SPLIT_ARRAYS = (
    "source_train_patient",
    "source_validation_patient",
    "target_adaptation_patient",
    "target_test_patient",
)
MATCHED_CONFIG_KEYS = (
    "source_validation_size", "target_test_size", "variance_percentile",
    "feature_scope", "image_size", "model", "classifier_epochs",
    "early_stopping_patience", "scheduler_patience", "classifier_lr",
    "weight_decay", "batch_size", "no_pretrained",
    "source_validation_threshold", "clip_scaled_values",
    "align_shared_genes", "anchor_shared_pixels",
    "class_weighted_loss",
)
CONFIG_DEFAULTS = {
    "feature_scope": "all",
    "source_validation_threshold": False,
    "clip_scaled_values": False,
    "align_shared_genes": False,
    "anchor_shared_pixels": False,
    "class_weighted_loss": False,
}


def config_value(config: dict[str, Any], key: str) -> Any:
    return config.get(key, CONFIG_DEFAULTS.get(key))


def patient_mapping_provenance_status(
    baseline_mapping: Any, adapted_mapping: Any
) -> str:
    """Validate recorded mapping provenance, allowing a verified legacy omission.

    Callers must still compare every saved patient split and the held-out
    patient manifest before emitting a valid paired comparison.
    """
    if baseline_mapping is not None and adapted_mapping is not None:
        if baseline_mapping != adapted_mapping:
            raise ValueError("Paired runs differ in sample-to-patient mapping provenance")
        return "mapping metadata matched"
    if baseline_mapping is None and adapted_mapping is None:
        return "mapping metadata not recorded; patient IDs verified from manifests"

    recorded_mapping = baseline_mapping if baseline_mapping is not None else adapted_mapping
    if not isinstance(recorded_mapping, dict) or any(
        not isinstance(recorded_mapping.get(domain), dict)
        or recorded_mapping[domain].get("source") != "built-in sample-ID parser"
        or recorded_mapping[domain].get("mapping_file") is not None
        for domain in ("source", "target")
    ):
        raise ValueError(
            "One paired run lacks mapping provenance and the recorded mapping is not "
            "the built-in sample-ID parser"
        )
    return (
        "mapping metadata missing from one legacy artifact; exact patient IDs and "
        "splits verified from both manifests"
    )


PREPROCESSING_POLICY_FIELDS = (
    "variance_filter", "variance_percentile", "variance_tie_break",
    "source_filter_fit_scope", "shared_gene_scaler_fit_scope",
    "clip_scaled_values",
)
TARGET_PRIVATE_POLICY_FIELDS = (
    "target_private_filter_fit_scope", "target_private_scaler_fit_scope",
)


def preprocessing_policy_signature(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return the auditable common preprocessing policy for paired runs.

    Historical run artifacts without an explicit policy cannot be silently
    treated as equivalent to runs produced by the ranked variance filter.
    Target-private handling is checked separately because it is intentionally
    not fitted by the inductive raw DeepInsight arm; in shared-only comparisons
    those target-private features are excluded from both image representations.
    """
    policy = metrics.get("preprocessing_policy")
    if not isinstance(policy, dict):
        raise ValueError(
            "Run artifact lacks explicit preprocessing_policy provenance; "
            "refresh the paired run before treating it as comparable"
        )
    missing = [key for key in PREPROCESSING_POLICY_FIELDS if key not in policy]
    if missing:
        raise ValueError(
            "Run preprocessing_policy is incomplete: " + ", ".join(missing)
        )
    return {key: policy[key] for key in PREPROCESSING_POLICY_FIELDS}


def verify_preprocessing_policy_match(
    baseline: dict[str, Any], adapted: dict[str, Any]
) -> dict[str, Any]:
    baseline_policy = preprocessing_policy_signature(baseline)
    adapted_policy = preprocessing_policy_signature(adapted)
    for metrics in (baseline, adapted):
        policy = metrics.get("preprocessing_policy", {})
        config = metrics.get("config", {})
        if ("variance_percentile" in config
                and float(policy["variance_percentile"])
                != float(config["variance_percentile"])):
            raise ValueError("Run preprocessing percentile differs from its saved configuration")
        if ("clip_scaled_values" in config
                and bool(policy["clip_scaled_values"])
                != bool(config["clip_scaled_values"])):
            raise ValueError("Run clipping policy differs from its saved configuration")
        if policy["source_filter_fit_scope"] != "source training samples only":
            raise ValueError("Source variance filter must be fit on source training samples only")
        if policy["shared_gene_scaler_fit_scope"] != "source training samples only":
            raise ValueError("Shared-gene scaling must be fit on source training samples only")

    if baseline_policy != adapted_policy:
        raise ValueError("Paired runs differ in preprocessing policy")

    feature_scope = baseline.get("feature_scope", baseline.get("config", {}).get("feature_scope", "all"))
    if feature_scope != "shared":
        for metrics in (baseline, adapted):
            arm = metrics.get("arm")
            expected_scope = (
                "not fitted" if arm == "raw_deepinsight"
                else "target adaptation samples only"
            )
            for key in TARGET_PRIVATE_POLICY_FIELDS:
                if metrics["preprocessing_policy"].get(key) != expected_scope:
                    raise ValueError(
                        f"Run preprocessing policy {key} must be {expected_scope!r} "
                        f"for arm {arm!r}"
                    )
        for key in TARGET_PRIVATE_POLICY_FIELDS:
            if baseline["preprocessing_policy"].get(key) != adapted["preprocessing_policy"].get(key):
                raise ValueError(f"Paired runs differ in preprocessing policy {key}")
    return baseline_policy


def verify_code_environment_match(
    baseline: dict[str, Any], adapted: dict[str, Any]
) -> dict[str, Any]:
    """Compare source/lock fingerprints, retaining compatibility with legacy runs."""
    fields = ("python_source_tree_sha256", "environment_lock_sha256")
    fingerprints: dict[str, Any] = {}
    field_status: dict[str, str] = {}
    for key in fields:
        base_value = baseline.get(key)
        adapted_value = adapted.get(key)
        if base_value is not None and adapted_value is not None and base_value != adapted_value:
            raise ValueError(f"Paired runs differ in code/environment fingerprint {key}")
        fingerprints[key] = base_value
        if base_value is None and adapted_value is None:
            field_status[key] = "not recorded"
        elif base_value is None or adapted_value is None:
            field_status[key] = "missing from one run"
        else:
            field_status[key] = "matched"
    if all(status == "not recorded" for status in field_status.values()):
        status = "not recorded in legacy run artifacts"
    elif all(value == "matched" for value in field_status.values()):
        status = "matched"
    else:
        status = "partial; one or both run artifacts lack fingerprints"
    return {
        "status": status,
        "field_status": field_status,
        **fingerprints,
    }


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics.get("preprocessing_policy"), dict):
        attestation_path = path / "preprocessing_policy_attestation.json"
        if attestation_path.is_file():
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            if (attestation.get("result") != "passed"
                    or attestation.get("run_metrics_sha256") != digest
                    or not isinstance(attestation.get("policy"), dict)):
                raise ValueError(f"Invalid preprocessing attestation in {attestation_path}")
            metrics["preprocessing_policy"] = attestation["policy"]
            if isinstance(attestation.get("patient_mapping"), dict):
                metrics["patient_mapping"] = attestation["patient_mapping"]
            metrics["preprocessing_policy_attestation_sha256"] = hashlib.sha256(
                attestation_path.read_bytes()
            ).hexdigest()
    with np.load(path / "split_ids.npz", allow_pickle=False) as archive:
        missing = sorted(set(SPLIT_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(
                f"{path} has no patient-level split manifest ({', '.join(missing)}); "
                "sample-level pilot results are not valid performance evidence."
            )
        splits = {name: archive[name].astype(str) for name in archive.files}
    return metrics, splits


def verify_partition_disjointness(splits: dict[str, np.ndarray]) -> None:
    for a, b in ((SPLIT_ARRAYS[0], SPLIT_ARRAYS[1]),
                 (SPLIT_ARRAYS[2], SPLIT_ARRAYS[3])):
        overlap = np.intersect1d(splits[a], splits[b])
        if overlap.size:
            raise ValueError(f"Patient leakage between {a} and {b}: {overlap[:5].tolist()}")


def verify_representation_fit_scope(metrics: dict[str, Any],
                                   splits: dict[str, np.ndarray]) -> None:
    """Check saved feature-map fit IDs against the declared protocol."""
    policy = metrics.get("representation_fit_policy")
    if policy is None:
        return  # Legacy verified pairs predate explicit feature-map manifests.
    required = {"feature_map_source_train", "feature_map_target_adaptation"}
    if not required.issubset(splits):
        raise ValueError("Run is missing feature-map fit sample IDs")
    if not np.array_equal(splits["feature_map_source_train"], splits["source_train"]):
        raise ValueError("Feature map was not fit from exactly the source-training rows")
    arm = metrics.get("arm")
    target_fit = splits["feature_map_target_adaptation"]
    if arm == "raw_deepinsight":
        if target_fit.size or policy.get("target_adaptation_expression_used"):
            raise ValueError("Raw DeepInsight baseline must not fit on target-adaptation values")
        geometry = metrics.get("geometry_diagnostics", {})
        if (geometry.get("target_expression_used_to_fit_map") is not False
                or geometry.get("target_adaptation_expression_used") is not False):
            raise ValueError("Raw DeepInsight geometry metadata contradicts source-only fitting")
        if metrics.get("feature_scope") != "shared":
            raise ValueError("Raw DeepInsight baseline must use the shared-gene scope")
    elif not np.array_equal(target_fit, splits["target_adaptation"]):
        raise ValueError("KPCA map fit IDs do not match target-adaptation rows")
    if policy.get("target_test_expression_used_for_fit") is not False:
        raise ValueError("Target-test values must not fit the feature map")
    if np.intersect1d(target_fit, splits["target_test"]).size:
        raise ValueError("Target-test samples appear in feature-map fitting IDs")


def patient_cluster_bootstrap(
    y_true: np.ndarray,
    y_baseline: np.ndarray,
    y_adapted: np.ndarray,
    patient_ids: np.ndarray,
    *,
    labels: list[Any],
    positive_label: Any,
    resamples: int,
    random_state: int,
) -> dict[str, Any]:
    """Bootstrap paired deltas by patient, retaining all rows per group."""
    patients = np.unique(patient_ids)
    rows_by_patient = {patient: np.flatnonzero(patient_ids == patient)
                       for patient in patients}
    rng = np.random.default_rng(random_state)
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity")
    deltas: dict[str, list[float]] | None = None
    for _ in range(resamples):
        sampled = rng.choice(patients, size=patients.size, replace=True)
        rows = np.concatenate([rows_by_patient[patient] for patient in sampled])
        try:
            base = classification_metrics(y_true[rows], y_baseline[rows],
                labels=labels, positive_label=positive_label)
            adapted = classification_metrics(y_true[rows], y_adapted[rows],
                labels=labels, positive_label=positive_label)
        except ValueError:
            continue
        if deltas is None:
            names = [name for name in metric_names if name in base and name in adapted]
            deltas = {name: [] for name in names}
        for name in names:
            deltas[name].append(adapted[name] - base[name])
    deltas = deltas or {}
    intervals = {
        name: ([float(x) for x in np.percentile(values, [2.5, 97.5])]
               if values else [None, None])
        for name, values in deltas.items()
    }
    return {
        "unit": "patient group; all rows from each sampled patient are retained",
        "resamples_requested": int(resamples),
        "resamples_used": int(min((len(values) for values in deltas.values()), default=0)),
        "random_state": int(random_state),
        "delta_95_percentile_interval": intervals,
        "scope_note": "Conditional on the fitted models and this held-out split; not uncertainty over training runs or new cohorts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("adapted", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    baseline, split_b = load_run(args.baseline)
    adapted, split_a = load_run(args.adapted)
    verify_partition_disjointness(split_b)
    verify_partition_disjointness(split_a)
    verify_representation_fit_scope(baseline, split_b)
    verify_representation_fit_scope(adapted, split_a)
    for key in ("seed", "source_file_sha256", "target_file_sha256"):
        if baseline.get(key) != adapted.get(key):
            raise ValueError(f"Paired runs differ in {key}")
    patient_mapping_provenance = patient_mapping_provenance_status(
        baseline.get("patient_mapping"), adapted.get("patient_mapping")
    )
    code_environment_signature = verify_code_environment_match(baseline, adapted)
    for key in MATCHED_CONFIG_KEYS:
        if (config_value(baseline.get("config", {}), key)
                != config_value(adapted.get("config", {}), key)):
            raise ValueError(f"Paired runs differ in matched setting {key}")
    preprocessing_signature = verify_preprocessing_policy_match(baseline, adapted)
    for name in SPLIT_ARRAYS:
        if not np.array_equal(split_b[name], split_a[name]):
            raise ValueError(f"Paired runs differ in patient split {name}")
    with np.load(args.baseline / "predictions.npz", allow_pickle=False) as bpred, \
            np.load(args.adapted / "predictions.npz", allow_pickle=False) as apred:
        required_prediction_fields = {"sample_ids", "patient_ids", "y_true"}
        if (not required_prediction_fields.issubset(bpred.files)
                or not required_prediction_fields.issubset(apred.files)):
            raise ValueError("Prediction artifacts must include sample IDs and patient group IDs")
        for name in ("sample_ids", "patient_ids", "y_true"):
            if not np.array_equal(bpred[name], apred[name]):
                raise ValueError(f"Paired runs differ in prediction manifest {name}")
        if not np.array_equal(bpred["sample_ids"].astype(str), split_b["target_test"].astype(str)):
            raise ValueError("Prediction sample IDs do not match the held-out target split")
        if not np.array_equal(bpred["patient_ids"].astype(str),
                              split_b["target_test_patient"].astype(str)):
            raise ValueError("Prediction patient IDs do not match the held-out patient groups")
        if args.bootstrap_resamples < 1:
            raise ValueError("bootstrap-resamples must be positive")
        bootstrap = patient_cluster_bootstrap(
            bpred["y_true"], bpred["y_pred"], apred["y_pred"],
            bpred["patient_ids"].astype(str),
            labels=baseline["test_metrics"]["labels"],
            positive_label=baseline["test_metrics"].get("positive_label"),
            resamples=args.bootstrap_resamples,
            random_state=int(baseline["seed"]),
        )

    bm = baseline["test_metrics"]
    am = adapted["test_metrics"]
    adapted_config = adapted.get("config", {})
    alignment_weight = adapted.get("alignment_weight")
    alignment_weight_provenance = "run metrics"
    if alignment_weight is None:
        if "alignment_weight" in adapted_config:
            alignment_weight = adapted_config["alignment_weight"]
            alignment_weight_provenance = "saved run configuration"
            if alignment_weight is None:
                raise ValueError("Adapted run has an explicitly null MMD alignment weight")
        else:
            # Before this field was configurable, the saved core objective was
            # mmd + lambda * regularizer, i.e. a unit MMD weight.
            alignment_weight = 1.0
            alignment_weight_provenance = "legacy implicit unit weight"
    result = {
        "valid_pair": True,
        "comparison_arm": adapted.get("arm", "unspecified"),
        "seed": baseline["seed"],
        "source_file_sha256": baseline["source_file_sha256"],
        "target_file_sha256": baseline["target_file_sha256"],
        "source_patient_count": baseline["source_patient_count"],
        "target_patient_count": baseline["target_patient_count"],
        "patient_mapping_provenance": patient_mapping_provenance,
        "target_test_samples": int(len(split_b["target_test"])),
        "target_test_patients": int(np.unique(split_b["target_test_patient"]).size),
        "matched_settings": {
            **{key: config_value(baseline.get("config", {}), key)
               for key in MATCHED_CONFIG_KEYS},
            "preprocessing_policy": preprocessing_signature,
            "code_environment": code_environment_signature,
        },
        "preprocessing_policies": {
            "baseline": baseline["preprocessing_policy"],
            "adapted": adapted["preprocessing_policy"],
        },
        "method_configurations": {
            "baseline": {
                "arm": baseline.get("arm"),
                "regularization_mode": baseline.get("config", {}).get("regularization_mode"),
                "selected_lambda_reg": baseline.get("selected_lambda_reg"),
                "alignment_weight": baseline.get("alignment_weight"),
                "effective_mmd_gamma": baseline.get("effective_mmd_gamma"),
            },
            "comparison": {
                "arm": adapted.get("arm"),
                "regularization_mode": adapted.get("config", {}).get("regularization_mode"),
                "kpca_epochs": adapted.get("config", {}).get("kpca_epochs"),
                "kpca_lr": adapted.get("config", {}).get("kpca_lr"),
                "kpca_gamma": adapted.get("config", {}).get("kpca_gamma"),
                "lambda_reg_setting": adapted.get("config", {}).get("lambda_reg"),
                "lambda_grid_setting": adapted.get("config", {}).get("lambda_grid"),
                "selected_lambda_reg": adapted.get("selected_lambda_reg"),
                "alignment_weight": float(alignment_weight),
                "alignment_weight_provenance": alignment_weight_provenance,
                "align_shared_genes": adapted.get("config", {}).get("align_shared_genes", False),
                "anchor_shared_pixels": adapted.get("config", {}).get("anchor_shared_pixels", False),
                "effective_mmd_gamma": adapted.get("effective_mmd_gamma"),
            },
        },
        "baseline": {
            "best_epoch": baseline["best_source_validation_epoch"],
            "source_validation_balanced_accuracy": baseline[
                "best_source_validation_balanced_accuracy"],
            "test_metrics": bm,
        },
        "adapted": {
            "best_epoch": adapted["best_source_validation_epoch"],
            "source_validation_balanced_accuracy": adapted[
                "best_source_validation_balanced_accuracy"],
            "test_metrics": am,
            "geometry_diagnostics": adapted.get("geometry_diagnostics"),
        },
        "test_metric_delta_adapted_minus_baseline": {
            key: float(am[key] - bm[key])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity")
            if key in am and key in bm
        },
        "patient_cluster_bootstrap": bootstrap,
        "interpretation": "One paired seed is a pilot; it does not establish paper-level performance.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
