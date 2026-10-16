#!/bin/zsh
set +e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
LOG="$REPO_ROOT/audit/environment_setup.log"
STATUS="$REPO_ROOT/audit/environment_setup.status"
rm -f "$STATUS"
"$REPO_ROOT/.venv/bin/python" -m pip install --upgrade pip > "$LOG" 2>&1
PIP_STATUS=$?
if [ "$PIP_STATUS" -eq 0 ]; then
  "$REPO_ROOT/.venv/bin/python" -m pip install -r requirements-training.txt >> "$LOG" 2>&1
  PIP_STATUS=$?
fi
printf '%s\n' "$PIP_STATUS" > "$STATUS"
exit "$PIP_STATUS"
