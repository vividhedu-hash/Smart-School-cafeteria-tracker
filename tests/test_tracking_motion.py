"""
tests/test_tracking_motion.py — Unit tests for KalmanBoxTracker and motion estimation.
"""
import pytest
import numpy as np

from cafeteria.tracking.motion import (
    KalmanBoxTracker,
    box_iou,
    box_center_distance,
)


def test_box_iou():
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert box_iou(box1, box2) == pytest.approx(1.0)

    box3 = [100, 100, 150, 150]
    assert box_iou(box1, box3) == 0.0

    box4 = [30, 10, 70, 50]
    iou = box_iou(box1, box4)
    assert 0.0 < iou < 1.0


def test_box_center_distance():
    box1 = [0, 0, 10, 10]    # center (5, 5)
    box2 = [30, 40, 50, 60]  # center (40, 50)
    # dist: hypot(35, 45)
    expected = (35**2 + 45**2) ** 0.5
    assert box_center_distance(box1, box2) == pytest.approx(expected)


def test_kalman_tracker_prediction_and_velocity():
    tracker = KalmanBoxTracker(
        track_id=1,
        class_type="person",
        cx=100.0,
        cy=100.0,
        w=60.0,
        h=80.0,
        vx=20.0,  # Moving right at 20px/s
        vy=0.0,
    )

    # Initial bbox
    b0 = tracker.to_bbox()
    assert b0 == (70, 60, 130, 140)

    # Predict after dt=1.0s
    b1 = tracker.predict(dt=1.0)
    assert tracker.cx == pytest.approx(120.0)
    assert tracker.cy == pytest.approx(100.0)
    assert b1 == (90, 60, 150, 140)

    # Measurement update
    tracker.update([100, 60, 160, 140], confidence=0.95, dt=1.0)
    assert tracker.hits == 2
    assert tracker.time_since_update == 0
    assert tracker.speed > 0.0
