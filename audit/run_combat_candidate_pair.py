"""Run a matched shared-gene baseline versus ComBat TvsN comparison.

This queue is detached through ``launch_detached.py`` and waits for the
real-cohort ComBat readiness preflight before training either classifier.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def wait_for(job: str) -> None:
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.exists():
        import time
        time.sleep(15)
    status = json.loads(marker.read_text(encoding="utf-8"))
    if status.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite {job} failed; see {status.get('log')}")


def newest_run(arm: str, seed: int, before: set[str]) -> Path:
    results = ROOT / "results" / "tvsn"
    created = sorted(
        (path for path in results.iterdir() if path.is_dir()
         and path.name not in before and path.name.startswith(f"{arm}_seed{seed}_")),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one new {arm} artifact for seed {seed}; found {created}")
    return created[0]


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", action="append", default=[
        "combat_readiness_preflight", "tvsn_full_tcga_seed47_official_all_clipped"])
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--comparison", default="audit/combat_seed46_shared_comparison.json")
    args = parser.parse_args()
    for prerequisite in args.after_job:
        wait_for(prerequisite)
    results = ROOT / "results" / "tvsn"
    before = {path.name for path in results.iterdir() if path.is_dir()}
    run_command([
        str(PYTHON), "Scripts/run_tvsn_experiment.py", "--arm", "baseline",
        "--seed", str(args.seed), "--feature-scope", "shared",
    ])
    baseline = newest_run("baseline", args.seed, before)
    before.add(baseline.name)
    run_command([
        str(PYTHON), "Scripts/run_combat_tvsn_experiment.py",
        "--seed", str(args.seed),
    ])
    combat = newest_run("combat", args.seed, before)
    comparison = ROOT / args.comparison
    comparison.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        str(PYTHON), "audit/compare_tvsn_runs.py", str(baseline), str(combat),
        "--output", str(comparison),
    ])
    print(f"Validated matched baseline/ComBat pair; comparison saved to {comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
