"""Queue saved-checkpoint target Grad-CAM after classifier experiments."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-job", default="tvsn_seed46_reference_regularizer")
    parser.add_argument("--run-dir", default="results/tvsn/kpcada_seed46_20261016T180038Z")
    args = parser.parse_args()
    marker = ROOT / "audit" / f"{args.after_job}.done"
    while not marker.exists():
        import time
        time.sleep(15)
    state = json.loads(marker.read_text(encoding="utf-8"))
    if state.get("exit_code") != 0:
        raise RuntimeError(f"Prerequisite {args.after_job} failed; see {state.get('log')}")
    subprocess.run([
        str(PYTHON), "audit/analyze_saved_target_gradcam.py", args.run_dir,
    ], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
