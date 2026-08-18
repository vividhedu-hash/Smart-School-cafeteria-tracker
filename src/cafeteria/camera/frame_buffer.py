"""
Thread-safe bounded frame buffer.

Stores TimestampedFrame objects in a deque. The pipeline reads frames
from this buffer rather than blocking on the camera directly. A background
thread continuously fills the buffer from the camera.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Optional

from cafeteria.camera.base import CameraBase, TimestampedFrame
from cafeteria.utils.logging import get_logger

logger = get_logger("camera.frame_buffer")


class FrameBuffer:
    """
    Thread-safe bounded deque of TimestampedFrame objects.

    A background thread reads from the camera continuously.
    The pipeline calls `get_latest()` or `get_recent(n)` to access frames.

    Args:
        camera:        Open CameraBase instance.
        maxsize:       Maximum frames to keep in buffer.
        on_disconnect: Optional callback invoked when the camera disconnects.
    """

    def __init__(
        self,
        camera: CameraBase,
        maxsize: int = 5,
        on_disconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        self._camera = camera
        self._maxsize = maxsize
        self._on_disconnect = on_disconnect

        self._buf: Deque[TimestampedFrame] = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._total_frames: int = 0
        self._dropped_frames: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Thread control
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="FrameBufferThread",
            daemon=True,
        )
        self._thread.start()
        logger.debug("FrameBuffer capture thread started")

    def stop(self) -> None:
        """Stop the background capture thread and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.debug("FrameBuffer capture thread stopped")

    # ──────────────────────────────────────────────────────────────────────
    # Public accessors
    # ──────────────────────────────────────────────────────────────────────

    def get_latest(self) -> Optional[TimestampedFrame]:
        """Return the most recent frame, or None if buffer is empty."""
        with self._lock:
            return self._buf[-1] if self._buf else None

    def get_recent(self, n: int) -> list[TimestampedFrame]:
        """Return the last n frames (oldest first) without removing them."""
        with self._lock:
            frames = list(self._buf)
        return frames[-n:]

    def get_all(self) -> list[TimestampedFrame]:
        """Return a snapshot of the entire buffer (oldest first)."""
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        """Empty the frame buffer."""
        with self._lock:
            self._buf.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "dropped_frames": self._dropped_frames,
            "buffer_size": self.size(),
            "buffer_maxsize": self._maxsize,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Internal capture loop
    # ──────────────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Continuous capture loop running in background thread.

        On disconnect, invokes on_disconnect callback (which should attempt
        camera reconnect). The loop sleeps briefly and retries.
        """
        while not self._stop_event.is_set():
            frame = self._camera.read()

            if frame is None:
                self._dropped_frames += 1
                # Camera may have disconnected
                if not self._camera.is_open():
                    logger.warning("Camera disconnected — invoking disconnect callback")
                    if self._on_disconnect:
                        self._on_disconnect()
                    # Wait for reconnect or stop
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
                continue

            self._total_frames += 1
            with self._lock:
                self._buf.append(frame)


def build_camera(config) -> CameraBase:
    """
    Factory: create the correct camera type from config.

    Args:
        config: CameraSettings from settings.py

    Returns:
        An open-ready CameraBase instance (not yet opened).
    """
    from cafeteria.camera.webcam import WebcamCamera
    from cafeteria.camera.rtsp import RTSPCamera

    if config.mode == "rtsp":
        return RTSPCamera(
            url=str(config.source),
            fps=config.fps,
            camera_id="cam0",
            reconnect_delay=config.reconnect_delay_seconds,
            reconnect_max_attempts=config.reconnect_max_attempts,
        )
    else:
        return WebcamCamera(
            source=config.source,
            width=config.width,
            height=config.height,
            fps=config.fps,
            camera_id="cam0",
            reconnect_delay=config.reconnect_delay_seconds,
            reconnect_max_attempts=config.reconnect_max_attempts,
        )
