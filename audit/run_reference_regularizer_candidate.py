"""Compare the audited reference-coordinate regularizer on corrected seed 46."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RESULTS = ROOT / "results" / "tvsn"
BASELINE = RESULTS / "baseline_seed46_20261016T175536Z"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="tvsn_seed46_source_threshold_eval")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--comparison", default="audit/full_tcga_seed46_reference_regularizer_comparison.json")
    args = parser.parse_args()
    marker = ROOT / "audit" / f"{args.after_job}.done"
    while not marker.exists():
        import time
        time.sleep(15)
    status = json.loads(marker.read_text(encoding="utf-8"))
    if status.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite {args.after_job} failed; see {status.get('log')}")
    before = {path.name for path in RESULTS.iterdir() if path.is_dir()}
    subprocess.run([
        str(PYTHON), "Scripts/run_tvsn_experiment.py", "--arm", "kpcada",
        "--seed", str(args.seed), "--feature-scope", "all",
        "--regularization-mode", "reference",
    ], cwd=ROOT, check=True)
    created = sorted(
        (path for path in RESULTS.iterdir() if path.is_dir()
         and path.name not in before and path.name.startswith(f"kpcada_seed{args.seed}_")),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one reference-regularized run, found {created}")
    if not BASELINE.is_dir():
        raise RuntimeError(f"Reference baseline artifact is missing: {BASELINE}")
    comparison = ROOT / args.comparison
    comparison.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(PYTHON), "audit/compare_tvsn_runs.py", str(BASELINE), str(created[0]),
        "--output", str(comparison),
    ], cwd=ROOT, check=True)
    print(f"Verified reference-regularized comparison: {comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
