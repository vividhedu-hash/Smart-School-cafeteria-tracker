#!/usr/bin/env bash
# One-command start: engine + dashboard. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python run.py "$@"
fi
exec python3 run.py "$@"
