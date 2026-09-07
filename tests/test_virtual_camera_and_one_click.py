"""
Tests for VirtualCafeteriaCamera, 1-click start fallback, and enroll camera stability.
"""
from __future__ import annotations

import os
import time
import numpy as np
import pytest

from cafeteria.camera.base import TimestampedFrame
from cafeteria.camera.virtual import VirtualCafeteriaCamera
from cafeteria.camera.webcam import WebcamCamera
from dashboard.enroll_cam import EnrollCamera, start_enroll_camera, stop_enroll_camera


def test_virtual_cafeteria_camera_lifecycle():
    cam = VirtualCafeteriaCamera(width=640, height=480, fps=15)
    assert cam.open() is True
    assert cam.is_open() is True

    info = cam.info()
    assert info.width == 640
    assert info.height == 480
    assert info.is_connected is True
    assert "Virtual" in info.device_name

    frame = cam.read()
    assert isinstance(frame, TimestampedFrame)
    assert frame.image is not None
    assert frame.image.shape == (480, 640, 3)
    assert frame.image.dtype == np.uint8
    assert frame.frame_id == 1

    frame2 = cam.read()
    assert frame2.frame_id == 2

    cam.close()
    assert cam.is_open() is False
    assert cam.read() is None


def test_webcam_camera_falls_back_to_virtual(monkeypatch):
    # Force candidate indices to invalid so hardware probe fails
    monkeypatch.setattr("cafeteria.camera.webcam.candidate_indices", lambda source: [9999])
    cam = WebcamCamera(source=9999, width=640, height=480, fps=15)
    
    # Should automatically fallback to VirtualCafeteriaCamera without error
    opened = cam.open()
    assert opened is True
    assert cam.is_open() is True

    info = cam.info()
    assert info.is_connected is True

    frame = cam.read()
    assert frame is not None
    assert frame.image.shape[2] == 3
    cam.close()


def test_start_enroll_camera_does_not_thrash_on_error(monkeypatch):
    sess_key = "test_stability_key"
    stop_enroll_camera(sess_key)

    # Force error condition
    monkeypatch.setattr("dashboard.enroll_cam.is_cloud", lambda: True)

    cam1 = start_enroll_camera(sess_key, source=0)
    time.sleep(0.05)
    status1 = cam1.status()
    assert status1.error is not None

    # Calling start_enroll_camera again must return the same instance, not thrash new threads
    cam2 = start_enroll_camera(sess_key, source=0)
    assert cam1 is cam2
    assert cam2.status().error is not None

    stop_enroll_camera(sess_key)
