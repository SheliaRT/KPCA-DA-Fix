"""Launch a long command with detached worker and completion waiter.

Example:
    python audit/launch_detached.py pilot_base_s42 -- \
        .venv/bin/python Scripts/run_tvsn_experiment.py --arm baseline --seed 42

Check `audit/pilot_base_s42.done` once the waiter notifies completion; output
is in `audit/pilot_base_s42.log` and the process exit code in `.status`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def worker(job: str, command: list[str]) -> int:
    log_path = AUDIT / f"{job}.log"
    status_path = AUDIT / f"{job}.status"
    done_path = AUDIT / f"{job}.done"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
            code = completed.returncode
        except Exception as exc:  # persist launch failures for the waiter/user
            log.write(f"\nLauncher error: {exc!r}\n")
            code = 127
    done_path.write_text(json.dumps({
        "job": job,
        "exit_code": code,
        "completed_utc": utc_now(),
        "log": str(log_path.relative_to(ROOT)),
    }, indent=2) + "\n", encoding="utf-8")
    status_path.write_text(f"{code}\n", encoding="utf-8")
    return code


def waiter(job: str) -> int:
    status_path = AUDIT / f"{job}.status"
    done_path = AUDIT / f"{job}.done"
    while not status_path.exists():
        time.sleep(15)
    payload = json.loads(done_path.read_text(encoding="utf-8"))
    notice = AUDIT / "job_completions.log"
    with notice.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return int(payload["exit_code"])


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "--worker":
        return worker(argv[2], argv[3:])
    if len(argv) == 3 and argv[1] == "--waiter":
        return waiter(argv[2])
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__, file=sys.stderr)
        return 2

    job = argv[1]
    command = argv[3:]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
        print("Job name may contain only letters, numbers, dot, underscore, and hyphen.", file=sys.stderr)
        return 2
    if not command:
        print("A command is required after --.", file=sys.stderr)
        return 2
    for suffix in ("log", "status", "done"):
        (AUDIT / f"{job}.{suffix}").unlink(missing_ok=True)

    worker_process = subprocess.Popen(
        [sys.executable, __file__, "--worker", job, *command],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
    )
    waiter_process = subprocess.Popen(
        [sys.executable, __file__, "--waiter", job],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True,
    )
    print(json.dumps({
        "job": job,
        "worker_pid": worker_process.pid,
        "waiter_pid": waiter_process.pid,
        "log": f"audit/{job}.log",
        "completion_marker": f"audit/{job}.done",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
