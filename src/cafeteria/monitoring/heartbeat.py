"""Idle-shutdown rules for the inference engine camera.

The webcam is a exclusive macOS resource. The engine must release it as soon
as Live Monitor is no longer polling — otherwise enrollment (browser camera)
and the rest of the site cannot function.
"""
from __future__ import annotations

from typing import Optional

# Live Monitor and the enrollment wizard both heartbeat while they are open.
# 45s is long enough to survive a slow rerun, a tab switch, or a page load
# without the camera dying mid-task; the LED still drops within a minute of
# walking away. Startup grace covers a cold InsightFace load on CPU.
DEFAULT_IDLE_SECONDS = 45.0
STARTUP_GRACE_SECONDS = 90.0


def should_idle_shutdown(
    *,
    now: float,
    started_at: float,
    last_heartbeat: Optional[float],
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    startup_grace_seconds: float = STARTUP_GRACE_SECONDS,
) -> bool:
    """
    Return True when the engine should exit and release the camera.

    No Live Monitor heartbeat after the startup grace period → shut down.
    A stale heartbeat (user left Live Monitor) → shut down.
    """
    if now - started_at < startup_grace_seconds:
        return False
    if last_heartbeat is None:
        return True
    return (now - last_heartbeat) > idle_seconds
