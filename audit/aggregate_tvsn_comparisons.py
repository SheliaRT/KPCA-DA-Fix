"""Aggregate only verified, matched TvsN baseline/KPCA-DA pairs across seeds.

Example:
    .venv/bin/python audit/aggregate_tvsn_comparisons.py \
      audit/full_tcga_seed46_official_all_comparison.json \
      audit/full_tcga_seed47_official_all_comparison.json \
      --output audit/tvsn_official_all_seed_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "sensitivity", "specificity")


def summarize(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def effective_alignment_weight(method: dict[str, Any]) -> float:
    """Read the MMD weight, honoring the pre-field implementation's unit default.

    Early corrected-cohort runs predate the explicit ``alignment_weight`` field.
    The archived ``kpcada.py`` objective was ``mmd + lambda * regularizer``,
    which is exactly an effective MMD weight of 1.0. Later runs record it
    directly. Any other missing adaptation-method setting remains an error.
    """
    value = method.get("alignment_weight")
    return 1.0 if value is None else float(value)


def adaptation_method_signature(pair: dict[str, Any]) -> tuple[Any, ...]:
    methods = pair.get("method_configurations")
    if not isinstance(methods, dict) or not isinstance(methods.get("comparison"), dict):
        raise ValueError("Every comparison must record its adaptation method configuration")
    method = methods["comparison"]
    required = (
        "regularization_mode", "kpca_epochs", "kpca_lr", "kpca_gamma",
        "lambda_reg_setting", "lambda_grid_setting", "align_shared_genes",
        "anchor_shared_pixels",
    )
    missing = [key for key in required if key not in method]
    if missing:
        raise ValueError(
            "Every comparison must record adaptation settings for cross-seed checks; "
            f"missing {', '.join(missing)}. Refresh it with compare_tvsn_runs.py."
        )
    if not method["regularization_mode"]:
        raise ValueError("Every comparison must record its regularization mode")
    return (
        method["regularization_mode"], effective_alignment_weight(method),
        method["kpca_epochs"], method["kpca_lr"], method["kpca_gamma"],
        method["lambda_reg_setting"], method["lambda_grid_setting"],
        method["align_shared_genes"], method["anchor_shared_pixels"],
    )


def aggregate(paths: list[Path]) -> dict[str, Any]:
    pairs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(pair.get("valid_pair") is not True for pair in pairs):
        raise ValueError("Every input must be a valid paired-run comparison")
    seeds = [int(pair["seed"]) for pair in pairs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Comparison inputs must have unique seeds")
    reference = pairs[0]
    reference_method = adaptation_method_signature(reference)
    for pair in pairs[1:]:
        for key in ("source_file_sha256", "target_file_sha256", "matched_settings", "comparison_arm"):
            if pair.get(key) != reference.get(key):
                raise ValueError(f"Pairs do not share identical {key}; aggregate separately")
        if adaptation_method_signature(pair) != reference_method:
            raise ValueError(
                "Pairs use different adaptation methods or settings; aggregate separately"
            )

    arm_metrics: dict[str, Any] = {}
    metric_deltas: dict[str, Any] = {}
    for arm in ("baseline", "adapted"):
        arm_metrics[arm] = {}
        for metric in METRICS:
            values = [float(pair[arm]["test_metrics"][metric])
                      for pair in pairs if metric in pair[arm]["test_metrics"]]
            if len(values) == len(pairs):
                arm_metrics[arm][metric] = summarize(values)
        labels = reference[arm]["test_metrics"]["labels"]
        class_keys = [str(label) for label in labels]
        arm_metrics[arm]["sensitivity_by_class"] = {
            key: summarize([
                float(pair[arm]["test_metrics"]["sensitivity_by_class"][key])
                for pair in pairs
            ]) for key in class_keys
        }

    for metric in METRICS:
        deltas = [float(pair["test_metric_delta_adapted_minus_baseline"][metric])
                  for pair in pairs if metric in pair["test_metric_delta_adapted_minus_baseline"]]
        if len(deltas) == len(pairs):
            metric_deltas[metric] = {
                **summarize(deltas),
                "improved_seeds": int(sum(value > 0 for value in deltas)),
                "total_seeds": len(deltas),
            }
    for key in arm_metrics["baseline"]["sensitivity_by_class"]:
        deltas = [
            float(pair["adapted"]["test_metrics"]["sensitivity_by_class"][key]
                  - pair["baseline"]["test_metrics"]["sensitivity_by_class"][key])
            for pair in pairs
        ]
        metric_deltas[f"sensitivity_by_class:{key}"] = {
            **summarize(deltas),
            "improved_seeds": int(sum(value > 0 for value in deltas)),
            "total_seeds": len(deltas),
        }

    return {
        "valid_aggregation": True,
        "seeds": seeds,
        "pair_count": len(pairs),
        "comparison_arm": reference.get("comparison_arm", "unspecified"),
        "cohort_sha256": {
            "source": reference["source_file_sha256"],
            "target": reference["target_file_sha256"],
        },
        "matched_settings": reference["matched_settings"],
        "adaptation_configuration": {
            "regularization_mode": reference_method[0],
            "effective_alignment_weight": reference_method[1],
            "kpca_epochs": reference_method[2],
            "kpca_lr": reference_method[3],
            "kpca_gamma": reference_method[4],
            "lambda_reg_setting": reference_method[5],
            "lambda_grid_setting": reference_method[6],
            "align_shared_genes": reference_method[7],
            "anchor_shared_pixels": reference_method[8],
            "alignment_weight_note": (
                "Missing weights in early artifacts are interpreted as 1.0 because their "
                "archived objective was MMD + lambda * regularizer."
            ),
        },
        "arm_metrics_across_seeds": arm_metrics,
        "adapted_minus_baseline_across_seeds": metric_deltas,
        "interpretation": (
            "Descriptive variability across these fitted splits/seeds. It does not "
            "estimate uncertainty over new cohorts; do not select settings using "
            "these target metrics and then call the same runs an untouched test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparisons", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.comparisons)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
