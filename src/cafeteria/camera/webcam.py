"""
Webcam camera implementation.

Supports:
  - macOS    → CAP_AVFOUNDATION (default)
  - Windows  → CAP_DSHOW
  - Linux    → CAP_V4L2
  - Auto-fallback to default backend if preferred fails
  - Auto index probe (source: auto) and native resolution negotiation

Automatic reconnect on disconnect.
"""
from __future__ import annotations

import os
import platform
import time
from typing import Any, Optional

import cv2
import numpy as np

from cafeteria.camera.base import CameraBase, CameraInfo, TimestampedFrame
from cafeteria.camera.negotiate import (
    DEFAULT_MAX_CAPTURE_WIDTH,
    candidate_indices,
    is_auto_source,
    rank_capture_sizes,
)
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


def _grab_bgr(cap: cv2.VideoCapture, warmup: int = 3) -> Optional[np.ndarray]:
    """Read a real frame. CAP_PROP_* on many webcams is a lie until grab."""
    frame = None
    for _ in range(max(1, warmup)):
        ok, img = cap.read()
        if ok and img is not None and getattr(img, "size", 0) > 0:
            frame = img
    return frame


class WebcamCamera(CameraBase):
    """
    OpenCV-based webcam camera.

    Args:
        source:                  Device index, "auto", or video file path.
        width, height:           Requested resolution; 0 = camera native (capped).
        fps:                     Requested frame rate.
        camera_id:               Logical ID used in frame metadata.
        reconnect_delay:         Seconds to wait between reconnect attempts.
        reconnect_max_attempts:  Max reconnect attempts before giving up.
        max_capture_width:       Cap for auto/native 4K sensors.
    """

    def __init__(
        self,
        source: int | str = "auto",
        width: int = 0,
        height: int = 0,
        fps: int = 30,
        camera_id: str = "cam0",
        reconnect_delay: float = 3.0,
        reconnect_max_attempts: int = 10,
        max_capture_width: int = DEFAULT_MAX_CAPTURE_WIDTH,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self.reconnect_max_attempts = reconnect_max_attempts
        self.max_capture_width = int(max_capture_width) or DEFAULT_MAX_CAPTURE_WIDTH

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps_counter = FPSCounter(window=30)
        self._backend = _get_backend()
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
        self._dropped_frames: int = 0
        self._opened_index: Any = source
        self._virtual_cam: Optional[CameraBase] = None

    # ──────────────────────────────────────────────────────────────────────
    # CameraBase interface
    # ──────────────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Open the capture device. Tries preferred backend then falls back."""
        return self._open_with_fallback()

    def read(self) -> Optional[TimestampedFrame]:
        if self._virtual_cam is not None:
            return self._virtual_cam.read()

        if self._cap is None or not self._cap.isOpened():
            self._dropped_frames += 1
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._dropped_frames += 1
            logger.debug("Frame read failed (source=%s)", self._opened_index)
            return None

        self._frame_id += 1
        now_mono = time.monotonic()
        now_wall = time.time()
        self._fps_counter.tick()
        self._actual_width = int(frame.shape[1])
        self._actual_height = int(frame.shape[0])

        return TimestampedFrame(
            frame_id=self._frame_id,
            timestamp=now_mono,
            wall_time=now_wall,
            image=frame,
            camera_id=self.camera_id,
        )

    def close(self) -> None:
        if self._virtual_cam is not None:
            try:
                self._virtual_cam.close()
            except Exception:
                pass
            self._virtual_cam = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("Camera closed (source=%s)", self._opened_index)

    def is_open(self) -> bool:
        if self._virtual_cam is not None:
            return self._virtual_cam.is_open()
        return self._cap is not None and self._cap.isOpened()

    def info(self) -> CameraInfo:
        if self._virtual_cam is not None:
            return self._virtual_cam.info()
        return CameraInfo(
            camera_id=self.camera_id,
            mode="webcam",
            source=self._opened_index,
            width=self._actual_width or self.width,
            height=self._actual_height or self.height,
            target_fps=self.fps,
            actual_fps=round(self._fps_counter.fps, 1),
            backend=_backend_name(self._backend),
            device_name=f"Webcam ({self._opened_index})",
            is_connected=self.is_open(),
        )

    def reconnect(self) -> bool:
        """Attempt to reconnect after a disconnect."""
        if self._virtual_cam is not None:
            return self._virtual_cam.reconnect()
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

    def _open_with_fallback(self) -> bool:
        """Try indices × backends, then negotiate a capture size that actually sticks."""
        if os.environ.get("CAFETERIA_VIRTUAL_CAM") == "1" or str(self.source).lower() in ("virtual", "demo", "synthetic"):
            logger.info("Direct virtual cafeteria stream requested via environment/source")
            from cafeteria.camera.virtual import VirtualCafeteriaCamera
            self._virtual_cam = VirtualCafeteriaCamera(
                width=self.width or 1280,
                height=self.height or 720,
                fps=self.fps or 20,
                camera_id=self.camera_id,
            )
            return self._virtual_cam.open()

        if isinstance(self.source, str) and not is_auto_source(self.source) and not str(self.source).isdigit():
            # Video file / device path
            if self._open_one(self.source, [self._backend, cv2.CAP_ANY]):
                return True

        indices = candidate_indices(self.source)
        backends = [self._backend, cv2.CAP_ANY]
        for idx in indices:
            if self._open_one(idx, backends):
                return True

        logger.warning(
            "Could not open physical webcam (tried indices %s). "
            "Activating Virtual Cafeteria Camera (Demo Stream) fallback.",
            indices,
        )
        from cafeteria.camera.virtual import VirtualCafeteriaCamera
        self._virtual_cam = VirtualCafeteriaCamera(
            width=self.width or 1280,
            height=self.height or 720,
            fps=self.fps or 20,
            camera_id=self.camera_id,
        )
        return self._virtual_cam.open()

    def _open_one(self, source: Any, backends: list[int]) -> bool:
        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                cap.release()
                continue
            if self._negotiate(cap):
                self._cap = cap
                self._backend = backend
                self._opened_index = source
                self.source = source
                self._log_camera_info()
                return True
            cap.release()
        return False

    def _negotiate(self, cap: cv2.VideoCapture) -> bool:
        """[AI-CoLab: Cursor] Pick a mode the sensor actually delivers, not CAP_PROP fiction."""
        if platform.system() != "Darwin":
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        probe = _grab_bgr(cap, warmup=2)
        if probe is not None:
            native_h, native_w = int(probe.shape[0]), int(probe.shape[1])

        sizes = rank_capture_sizes(
            native_width=native_w,
            native_height=native_h,
            requested_width=self.width,
            requested_height=self.height,
            max_capture_width=self.max_capture_width,
        )

        chosen = None
        chosen_frame = None
        for w, h in sizes:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if self.fps and self.fps > 0:
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame = _grab_bgr(cap, warmup=3)
            if frame is None:
                continue
            ah, aw = int(frame.shape[0]), int(frame.shape[1])
            if aw < 160 or ah < 120:
                continue
            chosen = (aw, ah)
            chosen_frame = frame
            # Close enough to the request (or any working auto mode).
            if abs(aw - w) <= 32 and abs(ah - h) <= 32:
                break
            if self.width <= 0 or self.height <= 0:
                break

        if chosen is None or chosen_frame is None:
            logger.warning("Camera opened but produced no usable frames")
            return False

        self._actual_width, self._actual_height = chosen
        self._actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(self.fps or 0)
        return True

    def _log_camera_info(self) -> None:
        logger.info(
            "Camera opened — source=%s  resolution=%dx%d  target_fps=%d  backend=%s  "
            "requested=%sx%s  max_width=%d",
            self._opened_index,
            self._actual_width,
            self._actual_height,
            self.fps,
            _backend_name(self._backend),
            self.width or "auto",
            self.height or "auto",
            self.max_capture_width,
        )
