"""Walk-past bbox persistence — no flicker, no invented IDs."""
from __future__ import annotations

import numpy as np

from cafeteria.recognition.live_match import (
    box_iou,
    face_search_crop,
    face_size_px,
    persist_bbox,
    smooth_bbox,
)
from cafeteria.recognition.matcher import EmbeddingMatcher


def test_persist_bbox_holds_then_expires():
    box = [10, 20, 80, 120]
    held, missing = persist_bbox(None, box, 0, max_hold=3)
    assert held == box
    assert missing == 0
    held, missing = persist_bbox(held, None, missing, max_hold=3)
    assert held == box
    assert missing == 1
    held, missing = persist_bbox(held, None, missing, max_hold=3)
    held, missing = persist_bbox(held, None, missing, max_hold=3)
    held, missing = persist_bbox(held, None, missing, max_hold=3)
    assert held is None
    assert missing == 4


def test_smooth_bbox_averages():
    prev = [0, 0, 100, 100]
    curr = [10, 10, 110, 110]
    out = smooth_bbox(prev, curr, alpha=0.5)
    assert out == [5, 5, 105, 105]


def test_box_iou_overlap():
    a = [0, 0, 10, 10]
    b = [5, 5, 15, 15]
    assert 0.1 < box_iou(a, b) < 0.2
    assert box_iou(a, a) == 1.0
    assert box_iou(a, None) == 0.0


def test_face_size_px():
    assert face_size_px([10, 20, 58, 68]) == (48, 48)
    assert face_size_px(None) == (0, 0)


def test_face_search_crop_uses_lock_then_roi():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    crop, ox, oy = face_search_crop(img, held_bbox=[100, 80, 180, 180])
    assert ox == 56
    assert oy == 25
    assert crop.shape[1] == 168
    assert crop.shape[0] == 210
    roi_crop, rx, ry = face_search_crop(
        img, held_bbox=None, roi={"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.5}
    )
    assert (rx, ry) == (0, 0)
    assert roi_crop.shape == (240, 640, 3)


def test_matcher_default_threshold_is_0_52():
    m = EmbeddingMatcher("/nope")
    assert m._threshold == 0.52
