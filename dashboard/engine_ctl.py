"""
Dashboard ↔ engine process control.

Kept out of ``app.py`` so other Streamlit pages can start/stop the engine and
write heartbeats without importing the home page (which would re-run it).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DASHBOARD_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
PID_FILE = PROJECT_ROOT / "data" / "engine.pid"
HEARTBEAT_FILE = PROJECT_ROOT / "data" / "heartbeat"
_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def write_heartbeat() -> None:
    """Touch the heartbeat file and ping the engine API if it is up."""
    try:
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT_FILE.write_text(str(time.time()))
    except Exception:
        pass
    try:
        from engine_client import post_heartbeat
        post_heartbeat()
    except Exception:
        pass


def engine_is_alive() -> bool:
    """Return True if the engine subprocess is running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        PID_FILE.unlink(missing_ok=True)
        return False


def engine_pid() -> int | None:
    if not engine_is_alive():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def start_engine() -> int | None:
    """Launch ``cafeteria.main`` as a detached background process."""
    if engine_is_alive():
        return engine_pid()

    write_heartbeat()  # survive InsightFace load until Live Monitor polls

    python = str(_PYTHON if _PYTHON.exists() else Path(sys.executable))
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONUNBUFFERED": "1"}
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdio_path = log_dir / "engine_stdio.log"
    stdio = open(stdio_path, "ab", buffering=0)

    proc = subprocess.Popen(
        [python, "-m", "cafeteria.main"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=stdio,
        stderr=stdio,
        start_new_session=True,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))

    try:
        from engine_client import wait_for_api
        wait_for_api(timeout_s=1.5)
    except Exception:
        pass

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if engine_is_alive():
            return proc.pid
        time.sleep(0.1)
    return proc.pid if proc.poll() is None else None


def stop_engine() -> None:
    """Terminate the engine subprocess and release the camera."""
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pid = None

    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        for _ in range(25):
            try:
                os.kill(pid, 0)
                time.sleep(0.12)
            except ProcessLookupError:
                break
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    PID_FILE.unlink(missing_ok=True)

    try:
        subprocess.run(
            ["pkill", "-9", "-f", "cafeteria.main"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass

    (PROJECT_ROOT / "data" / "runtime_state.json").unlink(missing_ok=True)
