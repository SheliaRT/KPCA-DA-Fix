"""Run a matched source-class-weighted TvsN baseline/KPCA-DA pair."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RESULTS = ROOT / "results" / "tvsn"


def wait_for(job: str) -> None:
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.exists():
        import time
        time.sleep(15)
    result = json.loads(marker.read_text(encoding="utf-8"))
    if result.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite {job} failed; see {result.get('log')}")


def run_arm(arm: str, seed: int, before: set[str]) -> Path:
    subprocess.run([
        str(PYTHON), "Scripts/run_tvsn_experiment.py", "--arm", arm,
        "--seed", str(seed), "--feature-scope", "all", "--class-weighted-loss",
    ], cwd=ROOT, check=True)
    created = sorted(
        (path for path in RESULTS.iterdir() if path.is_dir()
         and path.name not in before and path.name.startswith(f"{arm}_seed{seed}_")),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one new {arm} run for seed {seed}; found {created}")
    return created[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="anchored_pixels_seed48_pair")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--comparison", default="audit/full_tcga_seed49_official_all_weighted_comparison.json")
    args = parser.parse_args()
    wait_for(args.after_job)
    before = {path.name for path in RESULTS.iterdir() if path.is_dir()}
    baseline = run_arm("baseline", args.seed, before)
    before.add(baseline.name)
    adapted = run_arm("kpcada", args.seed, before)
    comparison = ROOT / args.comparison
    comparison.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(PYTHON), "audit/compare_tvsn_runs.py", str(baseline), str(adapted),
        "--output", str(comparison),
    ], cwd=ROOT, check=True)
    print(f"Verified class-weighted pair; comparison saved to {comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
