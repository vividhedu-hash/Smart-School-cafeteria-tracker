"""
Webcam camera implementation.

Supports:
  - macOS    → CAP_AVFOUNDATION (default)
  - Windows  → CAP_DSHOW
  - Linux    → CAP_V4L2
  - Auto-fallback to default backend if preferred fails

Automatic reconnect on disconnect.
"""
from __future__ import annotations

import platform
import time
from typing import Optional

import cv2
import numpy as np

from cafeteria.camera.base import CameraBase, CameraInfo, TimestampedFrame
from cafeteria.utils.timing import FPSCounter
from cafeteria.utils.logging import get_logger

logger = get_logger("camera.webcam")


def _get_backend() -> int:
    """Select the best OpenCV VideoCapture backend for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Windows":
        return cv2.CAP_DSHOW
    else:
        return cv2.CAP_V4L2


def _backend_name(backend: int) -> str:
    names = {
        cv2.CAP_AVFOUNDATION: "AVFoundation (macOS)",
        cv2.CAP_DSHOW: "DirectShow (Windows)",
        cv2.CAP_V4L2: "V4L2 (Linux)",
        cv2.CAP_ANY: "Auto",
    }
    return names.get(backend, str(backend))


class WebcamCamera(CameraBase):
    """
    OpenCV-based webcam camera.

    Args:
        source:                  Device index (int) or video file path (str).
        width, height:           Requested resolution.
        fps:                     Requested frame rate.
        camera_id:               Logical ID used in frame metadata.
        reconnect_delay:         Seconds to wait between reconnect attempts.
        reconnect_max_attempts:  Max reconnect attempts before giving up.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        camera_id: str = "cam0",
        reconnect_delay: float = 3.0,
        reconnect_max_attempts: int = 10,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self.reconnect_max_attempts = reconnect_max_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps_counter = FPSCounter(window=30)
        self._backend = _get_backend()
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
        self._dropped_frames: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # CameraBase interface
    # ──────────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the capture device. Tries preferred backend then falls back."""
        return self._open_with_fallback()

    def read(self) -> Optional[TimestampedFrame]:
        if self._cap is None or not self._cap.isOpened():
            self._dropped_frames += 1
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._dropped_frames += 1
            logger.debug("Frame read failed (source=%s)", self.source)
            return None

        self._frame_id += 1
        now_mono = time.monotonic()
        now_wall = time.time()
        self._fps_counter.tick()

        return TimestampedFrame(
            frame_id=self._frame_id,
            timestamp=now_mono,
            wall_time=now_wall,
            image=frame,
            camera_id=self.camera_id,
        )

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Camera closed (source=%s)", self.source)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def info(self) -> CameraInfo:
        return CameraInfo(
            camera_id=self.camera_id,
            mode="webcam",
            source=self.source,
            width=self._actual_width or self.width,
            height=self._actual_height or self.height,
            target_fps=self.fps,
            actual_fps=round(self._fps_counter.fps, 1),
            backend=_backend_name(self._backend),
            device_name=f"Webcam ({self.source})",
            is_connected=self.is_open(),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Reconnect logic
    # ──────────────────────────────────────────────────────────────────────

    def reconnect(self) -> bool:
        """Attempt to reconnect after a disconnect."""
        logger.warning("Attempting camera reconnect (source=%s)", self.source)
        self.close()
        for attempt in range(1, self.reconnect_max_attempts + 1):
            time.sleep(self.reconnect_delay)
            logger.info("Reconnect attempt %d/%d", attempt, self.reconnect_max_attempts)
            if self._open_with_fallback():
                logger.info("Camera reconnected successfully")
                return True
        logger.error("Camera reconnect failed after %d attempts", self.reconnect_max_attempts)
        return False

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _open_with_fallback(self) -> bool:
        """Try preferred backend; fall back to cv2.CAP_ANY if it fails."""
        for backend in [self._backend, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                self._configure_cap(cap)
                self._cap = cap
                self._log_camera_info()
                return True
            cap.release()
        logger.error("Could not open camera source=%s", self.source)
        return False

    def _configure_cap(self, cap: cv2.VideoCapture) -> None:
        """Apply resolution and FPS settings to an open capture."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Reduce internal buffer to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = cap.get(cv2.CAP_PROP_FPS)

    def _log_camera_info(self) -> None:
        info = self.info()
        logger.info(
            "Camera opened — source=%s  resolution=%dx%d  target_fps=%d  backend=%s",
            self.source,
            self._actual_width,
            self._actual_height,
            self.fps,
            _backend_name(self._backend),
        )
