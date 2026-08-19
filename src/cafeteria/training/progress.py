"""
Training status published to disk so a UI can watch a run it does not own.

Training happens on a background thread. Streamlit session state cannot be
written from that thread, and a page reload would lose it anyway, so the
worker writes JSON here and any page reads it back.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

STATUS_FILENAME = "training_status.json"

_WRITE_LOCK = threading.Lock()


def status_path(project_root: str | Path) -> Path:
    return Path(project_root) / "data" / STATUS_FILENAME


def write_status(path: str | Path, **fields: Any) -> None:
    # [AI-CoLab: Verified by Antigravity] Writes background training progress JSON for UI polling
    target = Path(path)
    payload = {"updated_at": time.time(), **fields}
    try:
        with _WRITE_LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
            tmp.replace(target)
    except OSError:
        pass


def update_status(path: str | Path, **fields: Any) -> None:
    """Merge fields into the existing status (keeps epoch history intact)."""
    current = read_status(path)
    current.update(fields)
    write_status(path, **current)


def read_status(path: str | Path) -> dict:
    """Return the current status, or an empty dict when there is none."""
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def clear_status(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def is_running(status: Optional[dict], stale_after_seconds: float = 900.0) -> bool:
    """
    True when a run is in flight.

    A status left behind by a killed process is treated as finished once it
    stops being updated, so the UI can never be locked out permanently.
    """
    if not status or status.get("state") != "running":
        return False
    updated = status.get("updated_at")
    try:
        updated = float(updated)
    except (TypeError, ValueError):
        return False
    return (time.time() - updated) < stale_after_seconds
