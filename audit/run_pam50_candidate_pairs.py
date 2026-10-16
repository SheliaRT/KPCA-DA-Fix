"""Run matched, patient-grouped multiclass PAM50 pairs after the TvsN queue.

This is a one-seed development screen in both manuscript directions. Each
baseline/KPCA-DA pair shares the same split and classifier recipe. The worker
is intended to run under launch_detached.py so its waiter records completion.
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
RESULTS = ROOT / "results"

TASKS = (
    ("pam50_smc_to_cptac", "Data/cptacvssmc/smc_labels.csv",
     "Data/cptacvssmc/cptac_labels.csv",
     "audit/pam50_smc_sample_patient.json",
     "audit/pam50_cptac_sample_patient.json"),
    ("pam50_cptac_to_smc", "Data/cptacvssmc/cptac_labels.csv",
     "Data/cptacvssmc/smc_labels.csv",
     "audit/pam50_cptac_sample_patient.json",
     "audit/pam50_smc_sample_patient.json"),
)


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


def run_arm(task: str, arm: str, seed: int, source: str, target: str,
            source_patient_map: str, target_patient_map: str) -> Path:
    task_dir = RESULTS / task
    task_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in task_dir.iterdir() if path.is_dir()}
    command = [
        str(PYTHON), str(RUNNER), "--arm", arm, "--seed", str(seed),
        "--task-name", task, "--allow-multiclass",
        "--source-path", source, "--target-path", target,
        "--source-patient-map", source_patient_map,
        "--target-patient-map", target_patient_map,
        "--exclude-source-labels", "Unknown", "--feature-scope", "all",
    ]
    print("Starting:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    created = sorted(
        (path for path in task_dir.iterdir()
         if path.is_dir() and path.name not in before),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one {task}/{arm} artifact, found {created}")
    return created[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="tvsn_seed46_target_gradcam")
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()

    wait_for_job(args.after_job)
    for task, source, target, source_map, target_map in TASKS:
        baseline = run_arm(task, "baseline", args.seed, source, target,
                           source_map, target_map)
        adapted = run_arm(task, "kpcada", args.seed, source, target,
                          source_map, target_map)
        comparison = ROOT / "audit" / f"{task}_seed{args.seed}_comparison.json"
        command = [
            str(PYTHON), str(ROOT / "audit" / "compare_tvsn_runs.py"),
            str(baseline), str(adapted), "--output", str(comparison),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"Validated matched pair; comparison saved to {comparison}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
