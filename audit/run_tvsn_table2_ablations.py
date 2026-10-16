"""Run the manuscript's four TvsN KPCA component configurations on one split.

The four variants are kPCA-only, kPCA+MMD, kPCA+VML/trace-only, and full
KPCA-DA. This is a development ablation on the corrected local cohort, not a
paper reproduction or a basis for a performance claim until replicated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "Scripts" / "run_tvsn_experiment.py"
RESULTS = ROOT / "results" / "tvsn_table2_seed48"
SOURCE = "Data/TvsN/TCGA_BRCA_NvT_new.csv"
TARGET = "Data/TvsN/GSE45498_TvsN_official_200.csv"


def wait_for_job(job: str) -> None:
    marker = ROOT / "audit" / f"{job}.done"
    while not marker.exists():
        time.sleep(15)
    result = json.loads(marker.read_text(encoding="utf-8"))
    if result.get("exit_code") != 0:
        raise RuntimeError(
            f"Required preceding job {job!r} failed; inspect {result.get('log')}"
        )
    print(f"Preceding job {job} completed successfully.", flush=True)


def run_arm(name: str, arm: str, seed: int, extra: list[str]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in RESULTS.iterdir() if path.is_dir()}
    command = [
        str(PYTHON), str(RUNNER), "--arm", arm, "--seed", str(seed),
        "--task-name", "tvsn_table2_seed48", "--results-root", "results",
        "--source-path", SOURCE, "--target-path", TARGET,
        "--exclude-source-labels", "Unknown", "--feature-scope", "all",
        *extra,
    ]
    print(f"Starting {name}:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    created = sorted(
        (path for path in RESULTS.iterdir()
         if path.is_dir() and path.name not in before),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one artifact for {name}, found {created}")
    return created[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="pam50_bidirectional_seed46")
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    wait_for_job(args.after_job)
    baseline = run_arm("kpca_only", "baseline", args.seed, [])
    variants = (
        ("kpca_plus_mmd", ["--lambda-reg", "0", "--regularization-mode", "none"]),
        ("kpca_plus_vml_trace", ["--alignment-weight", "0", "--lambda-reg", "0.001",
                                  "--regularization-mode", "trace"]),
        ("full_kpcada", ["--regularization-mode", "trace"]),
    )
    for name, extra in variants:
        adapted = run_arm(name, "kpcada", args.seed, extra)
        output = ROOT / "audit" / f"tvsn_table2_{name}_seed{args.seed}.json"
        subprocess.run([
            str(PYTHON), str(ROOT / "audit" / "compare_tvsn_runs.py"),
            str(baseline), str(adapted), "--output", str(output),
        ], cwd=ROOT, check=True)
        print(f"Saved verified {name} comparison to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
