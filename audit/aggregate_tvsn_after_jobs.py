"""Wait for detached comparison refreshes, then create a verified seed summary.

This utility is intended to run under ``launch_detached.py`` so the potentially
long bootstrap refreshes and their dependent aggregation remain non-blocking.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from aggregate_tvsn_comparisons import aggregate

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", help="detached job names whose .done markers to await")
    parser.add_argument("--comparisons", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    completions = {}
    for job in args.jobs:
        marker = AUDIT / f"{job}.done"
        while not marker.exists():
            time.sleep(15)
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if completion.get("exit_code") != 0:
            raise RuntimeError(f"Required job {job} failed: {completion}")
        completions[job] = completion

    paths = [path if path.is_absolute() else ROOT / path for path in args.comparisons]
    result = aggregate(paths)
    result["comparison_refresh_jobs"] = completions
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": "passed", "output": str(output_path), "seeds": result["seeds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
