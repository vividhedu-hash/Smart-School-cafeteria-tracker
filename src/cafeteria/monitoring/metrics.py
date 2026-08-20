"""
Runtime metrics tracking — latency history, FPS, dropped frames.
Writes the runtime_state.json file read by the Streamlit dashboard.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from cafeteria.utils.timing import RollingAverage
from cafeteria.utils.logging import get_logger

logger = get_logger("monitoring.metrics")


class MetricsCollector:
    """
    Collects and persists runtime performance metrics.

    Args:
        state_file:   Path to runtime_state.json.
        history_size: Number of samples to keep for rolling averages.
        write_interval_s: Minimum seconds between file writes.
    """

    def __init__(
        self,
        state_file: str | Path,
        history_size: int = 300,
        write_interval_s: float = 0.5,
    ) -> None:
        self._state_file = Path(state_file)
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_interval = write_interval_s
        self._last_write = 0.0

        self._plate_latency  = RollingAverage(maxlen=history_size)
        self._waste_latency  = RollingAverage(maxlen=history_size)
        self._face_latency   = RollingAverage(maxlen=history_size)
        self._e2e_latency    = RollingAverage(maxlen=history_size)
        self._camera_fps     = RollingAverage(maxlen=30)

        self._current_state: str = "IDLE"
        self._last_event: Optional[dict] = None
        self._live_face_match: Optional[dict] = None

    @property
    def live_face_match(self) -> Optional[dict]:
        return self._live_face_match

    def update_latencies(
        self,
        plate_ms: Optional[float] = None,
        waste_ms: Optional[float] = None,
        face_ms: Optional[float] = None,
        e2e_ms: Optional[float] = None,
    ) -> None:
        if plate_ms is not None:
            self._plate_latency.update(plate_ms)
        if waste_ms is not None:
            self._waste_latency.update(waste_ms)
        if face_ms is not None:
            self._face_latency.update(face_ms)
        if e2e_ms is not None:
            self._e2e_latency.update(e2e_ms)

    def update_fps(self, fps: float) -> None:
        self._camera_fps.update(fps)

    def set_state(self, state: str) -> None:
        self._current_state = state

    def set_last_event(self, event_dict: dict) -> None:
        self._last_event = event_dict

    def set_live_face_match(self, match: Optional[dict]) -> None:
        """Update real-time face match result (always-on recognition mode)."""
        self._live_face_match = match

    def snapshot(self, extra: Optional[dict] = None) -> dict:
        """In-memory runtime state (what the HTTP API serves)."""
        state = {
            "timestamp": time.time(),
            "state": self._current_state,
            "fps": round(self._camera_fps.mean(), 1),
            "latency": {
                "plate_ms": round(self._plate_latency.mean(), 1),
                "waste_ms": round(self._waste_latency.mean(), 1),
                "face_ms": round(self._face_latency.mean(), 1),
                "e2e_ms": round(self._e2e_latency.mean(), 1),
            },
            "last_event": self._last_event,
            "live_face_match": self._live_face_match,
        }
        if extra:
            state.update(extra)
        return state

    def write_state(self, extra: Optional[dict] = None) -> None:
        """Write runtime_state.json if enough time has passed (API fallback)."""
        now = time.time()
        if now - self._last_write < self._write_interval:
            return
        self._last_write = now
        state = self.snapshot(extra)

        try:
            tmp = self._state_file.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            # Atomic rename (works on all platforms)
            tmp.replace(self._state_file)
        except Exception as exc:
            logger.warning("Failed to write runtime state: %s", exc)


def read_runtime_state(state_file: str | Path) -> dict:
    """Read and parse runtime_state.json. Returns empty dict on failure."""
    try:
        with open(Path(state_file)) as f:
            return json.load(f)
    except Exception:
        return {}


def read_commands(commands_file: str | Path) -> Optional[dict]:
    """
    Read and consume a pending command from commands.json.

    Returns the command dict and deletes the file, or None if no command.
    """
    path = Path(commands_file)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            cmd = json.load(f)
        path.unlink(missing_ok=True)
        return cmd
    except Exception:
        return None


def write_command(commands_file: str | Path, command: dict) -> None:
    """Write a command for the inference engine to pick up."""
    with open(Path(commands_file), "w") as f:
        json.dump(command, f)
