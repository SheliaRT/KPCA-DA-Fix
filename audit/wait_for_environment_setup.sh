#!/bin/zsh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS="$REPO_ROOT/audit/environment_setup.status"
while [ ! -f "$STATUS" ]; do
  sleep 20
done
RESULT="$(cat "$STATUS")"
printf 'Environment dependency installation completed with exit code %s at %s\n' "$RESULT" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$REPO_ROOT/audit/environment_setup.done"
