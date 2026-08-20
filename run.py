"""
One-command launcher.

    ./run.sh
    python run.py

Starts the inference engine and the dashboard together, opens
http://localhost:8501, and stops both on Ctrl+C.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cafeteria.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
