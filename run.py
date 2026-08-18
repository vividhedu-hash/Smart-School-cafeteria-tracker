"""
One-click application launcher — Smart Cafeteria Waste Tracker.

Double-click  Start_Cafeteria_Tracker.command  (macOS)
 — OR —
Run in terminal:  python run.py

What it does:
  1. Detects Python / venv automatically (no manual activation needed)
  2. Starts the inference engine in the background
  3. Starts the Streamlit web dashboard
  4. Opens http://localhost:8501 in your default browser automatically

Camera permissions are handled by the browser when you visit the Training page.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import webbrowser
import threading
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent
DASHBOARD_URL = "http://localhost:8501"

# ── Find the right Python executable ─────────────────────────────────────────
def _find_python() -> str:
    """Return the venv python if available, else system python."""
    candidates = [
        PROJECT_ROOT / ".venv" / "bin" / "python",          # Unix/macOS
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


# ── Coloured terminal output ──────────────────────────────────────────────────
def _log(msg: str, colour: str = "") -> None:
    RESET  = "\033[0m"
    codes  = {"green": "\033[92m", "yellow": "\033[93m",
              "red": "\033[91m", "cyan": "\033[96m", "bold": "\033[1m"}
    prefix = codes.get(colour, "")
    print(f"{prefix}{msg}{RESET}", flush=True)


def _banner() -> None:
    _log("=" * 62, "bold")
    _log("  🍽️  Smart Cafeteria Waste Tracker", "bold")
    _log("=" * 62, "bold")
    _log(f"  Platform : {platform.system()} {platform.machine()}")
    _log(f"  Python   : {_find_python()}")
    _log(f"  Root     : {PROJECT_ROOT}")
    _log("=" * 62, "bold")


# ── Auto-open browser ─────────────────────────────────────────────────────────
def _open_browser_after_delay(delay: float = 4.0) -> None:
    def _fn():
        time.sleep(delay)
        _log(f"\n  🌐 Opening dashboard → {DASHBOARD_URL}", "green")
        webbrowser.open(DASHBOARD_URL)
    threading.Thread(target=_fn, daemon=True).start()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _banner()

    python = _find_python()
    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
    }

    # ── 1. Dashboard only — engine starts from Live Monitor so the camera
    # stays off until someone is actually watching.
    _log("\n  [1/2] Engine stays off until you click Start Engine on Live Monitor.", "cyan")

    # ── 2. Schedule browser open ──────────────────────────────────────────
    _log("  [2/2] Starting web dashboard (Ctrl+C to stop)…\n", "cyan")
    _open_browser_after_delay(delay=3.0)

    streamlit_cmd = [
        python, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.headless=true",
        "--server.port=8501",
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
    ]

    try:
        subprocess.run(streamlit_cmd, cwd=str(PROJECT_ROOT), env=env)
    except KeyboardInterrupt:
        _log("\n\n  Shutdown requested…", "yellow")
    finally:
        sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))
        try:
            from engine_ctl import stop_engine
            stop_engine()
        except Exception:
            pass
        (PROJECT_ROOT / "data" / "engine.pid").unlink(missing_ok=True)
        _log("  All services stopped.", "green")


if __name__ == "__main__":
    main()
