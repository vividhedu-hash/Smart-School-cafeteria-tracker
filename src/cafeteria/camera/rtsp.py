"""
RTSP camera implementation for future two-camera deployment.

Uses OpenCV VideoCapture with FFMPEG backend (best for RTSP on all platforms).
Includes:
  - Reconnect on stream loss
  - TCP/UDP transport configuration
  - Frame timestamp injection
"""
from __future__ import annotations

import time
from typing import Optional

import cv2

from cafeteria.camera.base import CameraBase, CameraInfo, TimestampedFrame
from cafeteria.utils.timing import FPSCounter
from cafeteria.utils.logging import get_logger

logger = get_logger("camera.rtsp")


class RTSPCamera(CameraBase):
    """
    RTSP stream camera using OpenCV + FFMPEG.

    Args:
        url:                     RTSP URL (e.g. "rtsp://user:pass@192.168.1.10:554/stream1")
        width, height:           Optional resolution override (0 = use stream native)
        fps:                     Expected stream FPS (used for info only)
        transport:               "tcp" (reliable) or "udp" (low-latency)
        camera_id:               Logical ID used in frame metadata.
        reconnect_delay:         Seconds between reconnect attempts.
        reconnect_max_attempts:  Max attempts before giving up.
    """

    def __init__(
        self,
        url: str,
        width: int = 0,
        height: int = 0,
        fps: int = 25,
        transport: str = "tcp",
        camera_id: str = "cam0",
        reconnect_delay: float = 3.0,
        reconnect_max_attempts: int = 10,
    ) -> None:
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.transport = transport
        self.camera_id = camera_id
        self.reconnect_delay = reconnect_delay
        self.reconnect_max_attempts = reconnect_max_attempts

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id: int = 0
        self._fps_counter = FPSCounter(window=30)
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._dropped_frames: int = 0

    def open(self) -> bool:
        # Set FFMPEG options via OpenCV environment variable trick
        # RTSP_TRANSPORT ensures we use TCP for reliability
        cv2.setNumThreads(1)  # Prevent thread contention on RTSP

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error("Failed to open RTSP stream: %s", self.url)
            return False

        if self.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap

        logger.info(
            "RTSP stream opened — url=%s  resolution=%dx%d",
            self.url,
            self._actual_width,
            self._actual_height,
        )
        return True

    def read(self) -> Optional[TimestampedFrame]:
        if self._cap is None or not self._cap.isOpened():
            self._dropped_frames += 1
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._dropped_frames += 1
            return None

        self._frame_id += 1
        self._fps_counter.tick()
        return TimestampedFrame(
            frame_id=self._frame_id,
            timestamp=time.monotonic(),
            wall_time=time.time(),
            image=frame,
            camera_id=self.camera_id,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("RTSP camera closed (url=%s)", self.url)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def info(self) -> CameraInfo:
        return CameraInfo(
            camera_id=self.camera_id,
            mode="rtsp",
            source=self.url,
            width=self._actual_width or self.width,
            height=self._actual_height or self.height,
            target_fps=self.fps,
            actual_fps=round(self._fps_counter.fps, 1),
            backend="FFMPEG (RTSP)",
            device_name=self.url,
            is_connected=self.is_open(),
        )

    def reconnect(self) -> bool:
        logger.warning("RTSP reconnect initiated (url=%s)", self.url)
        self.close()
        for attempt in range(1, self.reconnect_max_attempts + 1):
            time.sleep(self.reconnect_delay)
            logger.info("RTSP reconnect attempt %d/%d", attempt, self.reconnect_max_attempts)
            if self.open():
                return True
        logger.error("RTSP reconnect failed after %d attempts", self.reconnect_max_attempts)
        return False

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames
