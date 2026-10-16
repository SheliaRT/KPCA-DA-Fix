"""Queue a paired all-feature baseline/KPCA-DA comparison.

The selected classifier, source/target files and seed are shared by both arms.
The outer ``launch_detached.py`` worker and waiter keep the run detached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "Scripts" / "run_tvsn_experiment.py"
PYTHON = ROOT / ".venv" / "bin" / "python"


def wait_for_job(job: str) -> None:
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.exists():
        time.sleep(15)
    result = json.loads(marker.read_text(encoding="utf-8"))
    if result["exit_code"] != 0:
        raise RuntimeError(
            f"Required preceding job {job!r} failed with exit code "
            f"{result['exit_code']}; inspect {result['log']}"
        )
    print(f"Preceding job {job} completed successfully.", flush=True)


def run_arm(arm: str, seed: int, source: str, target: str,
            excluded_labels: list[str], clip_scaled_values: bool,
            model: str, no_pretrained: bool, feature_scope: str,
            regularization_mode: str, results_root: Path) -> Path:
    task_results = results_root / "tvsn"
    task_results.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in task_results.iterdir() if path.is_dir()}
    command = [
        str(PYTHON), str(RUNNER), "--arm", arm, "--seed", str(seed),
        "--source-path", source, "--target-path", target,
        "--exclude-source-labels", *excluded_labels,
        "--feature-scope", feature_scope, "--model", model,
        "--regularization-mode", regularization_mode,
        "--results-root", str(results_root.relative_to(ROOT)
                               if results_root.is_relative_to(ROOT) else results_root),
    ]
    if clip_scaled_values:
        command.append("--clip-scaled-values")
    if no_pretrained:
        command.append("--no-pretrained")
    print("Starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    created = sorted(
        (path for path in task_results.iterdir()
         if path.is_dir() and path.name not in before),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one new run artifact, found {[p.name for p in created]}")
    return created[0]


def validate_reusable_baseline(run_dir: Path, *, seed: int, arm: str,
                               model: str, feature_scope: str) -> Path:
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    metadata = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = metadata.get("config", {})
    expected = {
        "seed": seed,
        "arm": arm,
        "feature_scope": feature_scope,
        "model": model,
    }
    actual = {
        "seed": metadata.get("seed"),
        "arm": metadata.get("arm"),
        "feature_scope": metadata.get("feature_scope"),
        "model": config.get("model"),
    }
    if actual != expected:
        raise ValueError(f"Reusable baseline does not match requested pair: {actual} != {expected}")
    if not (run_dir / "preprocessing_policy_attestation.json").is_file() and not isinstance(
        metadata.get("preprocessing_policy"), dict
    ):
        raise ValueError("Reusable baseline lacks verified preprocessing provenance")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", required=True,
                        help="detached job whose completion gates this queued pair")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--source", default="Data/TvsN/TCGA_BRCA_NvT_new.csv")
    parser.add_argument("--target", default="Data/TvsN/GSE45498_TvsN_official_200.csv")
    parser.add_argument("--exclude-source-label", action="append", default=["Unknown"])
    parser.add_argument("--clip-scaled-values", action="store_true")
    parser.add_argument("--model", default="resnet50d",
                        help="matched classifier backbone (use resnet50 to test the paper's named architecture)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="initialize the matched classifier from random weights")
    parser.add_argument("--baseline-arm", choices=("baseline", "raw_deepinsight"),
                        default="baseline",
                        help="baseline representation; raw_deepinsight is the paper-style no-alignment arm")
    parser.add_argument("--feature-scope", choices=("all", "shared"), default="all")
    parser.add_argument("--regularization-mode", choices=("trace", "reference", "none"),
                        default="trace",
                        help="KPCA-DA regularizer; reference is a method revision, not the paper's trace term")
    parser.add_argument("--comparison")
    parser.add_argument("--results-root", type=Path,
                        help="isolated output root; defaults to a comparison-specific directory")
    parser.add_argument("--baseline-run-dir", type=Path,
                        help="reuse a completed baseline artifact instead of retraining it")
    args = parser.parse_args()

    comparison_value = args.comparison or (
        f"audit/full_tcga_seed{args.seed}_all"
        f"{'_clipped' if args.clip_scaled_values else ''}_comparison.json"
    )
    results_root = args.results_root or (
        ROOT / "results" / "audit_pairs" / Path(comparison_value).stem
    )
    if not results_root.is_absolute():
        results_root = ROOT / results_root

    wait_for_job(args.after_job)
    baseline = (
        validate_reusable_baseline(
            args.baseline_run_dir, seed=args.seed, arm=args.baseline_arm,
            model=args.model, feature_scope=args.feature_scope,
        )
        if args.baseline_run_dir is not None
        else run_arm(args.baseline_arm, args.seed, args.source, args.target,
                     args.exclude_source_label, args.clip_scaled_values,
                     args.model, args.no_pretrained, args.feature_scope,
                     args.regularization_mode,
                     results_root)
    )
    adapted = run_arm("kpcada", args.seed, args.source, args.target,
                      args.exclude_source_label, args.clip_scaled_values,
                      args.model, args.no_pretrained, args.feature_scope,
                      args.regularization_mode,
                      results_root)
    comparison = ROOT / comparison_value
    comparison.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(ROOT / "audit" / "compare_tvsn_runs.py"),
        str(baseline), str(adapted), "--output", str(comparison),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Validated matched pair; comparison saved to {comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
