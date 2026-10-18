"""Wait for an existing local process and require its result artifact."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    while True:
        try:
            os.kill(args.pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            break
        time.sleep(15)
    if not args.result.is_file():
        raise SystemExit(f"Process {args.pid} exited without result artifact {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
