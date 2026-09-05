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
from cafeteria.recognition.walkpast import (
    WalkPastTracker,
    adaptive_pad_ratio,
    adaptive_smooth_alpha,
    lead_bbox,
    predict_bbox,
    same_track,
    update_velocity,
)


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


def test_predict_bbox_coasts_with_velocity():
    box = [100, 80, 160, 160]
    out = predict_bbox(box, vx=200.0, vy=0.0, dt=0.10, frame_w=640, frame_h=480)
    assert out == [120, 80, 180, 160]


def test_update_velocity_ema():
    prev = [0, 0, 40, 40]
    curr = [20, 0, 60, 40]
    vx, vy = update_velocity(prev, curr, dt=0.10, vx=0.0, vy=0.0, alpha=1.0)
    assert vx == 200.0
    assert vy == 0.0


def test_same_track_accepts_fast_walk_with_low_iou():
    prev = [40, 80, 100, 160]
    # jumped ~80px right — IoU is ~0, but along +x motion
    curr = [130, 82, 190, 162]
    assert box_iou(prev, curr) < 0.08
    assert same_track(prev, curr, vx=800.0, vy=0.0, dt=0.10) is True
    far = [400, 80, 460, 160]
    assert same_track(prev, far, vx=800.0, vy=0.0, dt=0.10) is False


def test_adaptive_follow_when_moving():
    assert adaptive_smooth_alpha(20) < adaptive_smooth_alpha(200)
    assert adaptive_pad_ratio(0.90, 200) > adaptive_pad_ratio(0.90, 0)


def test_lead_bbox_stretches_in_walk_direction():
    box = [100, 80, 160, 160]
    led = lead_bbox(box, vx=300.0, vy=0.0, frame_w=640, frame_h=480, lead_seconds=0.20)
    assert led[0] == 100
    assert led[2] > 160
    left = lead_bbox(box, vx=-300.0, vy=0.0, frame_w=640, frame_h=480, lead_seconds=0.20)
    assert left[0] < 100
    assert left[2] == 160


class _FakeMatch:
    def __init__(self, known=True):
        self.person_id = "p1" if known else None
        self.person_name = "Ada" if known else None
        self.similarity = 0.80 if known else 0.20
        self.is_known = known


def test_tracker_keeps_identity_through_blurry_frame():
    tracker = WalkPastTracker(identify_min=20, lock_min=10, hold_seconds=1.0)
    tracker.held_bbox = [40, 80, 120, 180]
    tracker.held_identity = {
        "person_id": "p1", "person_name": "Ada", "similarity": 0.80, "is_known": True
    }
    tracker.vx, tracker.vy = 200.0, 0.0
    tracker.last_update = 1.00
    tracker.held_until = 2.00

    detection = {
        "bbox": [56, 80, 136, 180],
        "embedding": np.ones(512, dtype=np.float32),
        "det_score": 0.55,
    }

    class _BlurMiss:
        calls = 0

        def match(self, _emb):
            self.calls += 1
            return _FakeMatch(known=False)

    matcher = _BlurMiss()
    out = tracker._on_hit(detection, now=1.08, frame_w=640, frame_h=480, matcher=matcher)
    assert out["is_known"] is True
    assert out["person_id"] == "p1"
    assert out["person_name"] == "Ada"
    assert matcher.calls == 1
    assert out["moving"] is True


def test_tracker_coasts_bbox_on_miss():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker = WalkPastTracker(hold_seconds=1.0)
    tracker.held_bbox = [100, 80, 160, 160]
    tracker.held_until = 10.0
    tracker.vx, tracker.vy = 250.0, 0.0
    tracker.last_update = 1.00
    tracker.held_identity = {
        "person_id": "p1", "person_name": "Ada", "similarity": 0.7, "is_known": True
    }

    def nobody(_small):
        return []

    out = tracker.step(img, nobody, 1.10, matcher=None)
    assert out is not None
    assert out["is_known"] is True
    assert out["bbox"][0] > 100
    assert out["moving"] is True


def test_infer_faster_while_moving():
    tracker = WalkPastTracker()
    standing = tracker.infer_interval(now=1.0)
    tracker.held_bbox = [40, 80, 120, 180]
    tracker.held_until = 10.0
    tracker.held_identity = {"is_known": True}
    locked_still = tracker.infer_interval(now=1.0)
    tracker.vx = 180.0
    moving = tracker.infer_interval(now=1.0)
    assert moving <= standing
    assert moving < locked_still
