"""Queue source-validation threshold evaluation after the training queue."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="tvsn_procrustes_seed46_official_all")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output", default="audit/full_tcga_seed46_source_threshold.json")
    args = parser.parse_args()
    marker = ROOT / "audit" / f"{args.after_job}.done"
    while not marker.exists():
        import time
        time.sleep(15)
    state = json.loads(marker.read_text(encoding="utf-8"))
    if state.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite {args.after_job} failed; see {state.get('log')}")
    baseline = sorted((ROOT / "results" / "tvsn").glob(f"baseline_seed{args.seed}_*"),
                      key=lambda path: path.stat().st_mtime_ns)
    adapted = sorted((ROOT / "results" / "tvsn").glob(f"kpcada_seed{args.seed}_*"),
                     key=lambda path: path.stat().st_mtime_ns)
    if not baseline or not adapted:
        raise RuntimeError(f"Saved seed-{args.seed} baseline/KPCA-DA artifacts are missing")
    subprocess.run([
        str(PYTHON), "audit/evaluate_saved_source_threshold.py",
        str(baseline[0]), str(adapted[0]), "--output", args.output,
    ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
