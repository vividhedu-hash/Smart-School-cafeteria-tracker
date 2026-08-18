"""
Camera abstraction base class.

Concrete implementations: WebcamCamera, RTSPCamera.
Both implement this interface so the rest of the system is camera-agnostic.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TimestampedFrame:
    """A single captured frame with metadata."""
    frame_id: int
    timestamp: float       # time.monotonic()
    wall_time: float       # time.time() — for display
    image: np.ndarray      # BGR (H, W, 3)
    camera_id: str = "cam0"

    def age_seconds(self) -> float:
        """How old this frame is relative to now."""
        return time.monotonic() - self.timestamp


@dataclass
class CameraInfo:
    """Runtime information about a connected camera."""
    camera_id: str
    mode: str              # webcam | rtsp
    source: object         # int or str
    width: int
    height: int
    target_fps: int
    actual_fps: float = 0.0
    backend: str = ""
    device_name: str = ""
    is_connected: bool = False


class CameraBase(abc.ABC):
    """
    Abstract base class for all camera sources.

    Subclasses must implement:
        open()      — open the capture device
        read()      — return next TimestampedFrame or None
        close()     — release resources
        info()      — return CameraInfo
    """

    @abc.abstractmethod
    def open(self) -> bool:
        """
        Open the camera.

        Returns:
            True on success, False on failure.
        """

    @abc.abstractmethod
    def read(self) -> Optional[TimestampedFrame]:
        """
        Read the next frame.

        Returns:
            TimestampedFrame, or None if the frame could not be captured.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Release all camera resources."""

    @abc.abstractmethod
    def info(self) -> CameraInfo:
        """Return current camera metadata."""

    @abc.abstractmethod
    def is_open(self) -> bool:
        """Return True if the camera is currently open and operational."""

    def __enter__(self) -> "CameraBase":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
