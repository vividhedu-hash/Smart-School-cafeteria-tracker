"""Overlay and live_face_match accept both MatchResult objects and dicts."""
from __future__ import annotations

import numpy as np

from cafeteria.monitoring.overlay import draw_debug_overlay
from cafeteria.recognition.live_match import face_match_as_dict, face_match_field
from cafeteria.recognition.matcher import MatchResult


def test_draw_debug_overlay_accepts_dict_and_match_result():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    as_dict = {
        "person_id": "p1",
        "person_name": "Ada",
        "similarity": 0.91,
        "is_known": True,
    }
    as_obj = MatchResult(
        person_id="p1", person_name="Ada", similarity=0.91,
        is_known=True, candidates=[],
    )
    out_dict = draw_debug_overlay(frame, state="IDLE", fps=10.0, face_match=as_dict)
    out_obj = draw_debug_overlay(frame, state="IDLE", fps=10.0, face_match=as_obj)
    out_none = draw_debug_overlay(frame, state="IDLE", fps=10.0, face_match=None)
    assert out_dict.shape == frame.shape
    assert out_obj.shape == frame.shape
    assert out_none.shape == frame.shape


def test_face_match_field_dict_or_object():
    as_dict = {"is_known": True, "person_name": "Ada", "similarity": 0.5}
    as_obj = MatchResult(
        person_id="p1", person_name="Ada", similarity=0.5,
        is_known=True, candidates=[],
    )
    assert face_match_field(as_dict, "is_known") is True
    assert face_match_field(as_obj, "is_known") is True
    assert face_match_field(None, "is_known", False) is False


def test_face_match_as_dict_normalizes_both():
    as_obj = MatchResult(
        person_id="Bhavmanyu_24", person_name="Bhavmanyu",
        similarity=0.88, is_known=True, candidates=[],
    )
    d = face_match_as_dict(as_obj)
    assert d["is_known"] is True
    assert d["person_id"] == "Bhavmanyu_24"
    assert d["bbox"] is None
    assert d["approaching"] is False
    assert face_match_as_dict(d)["person_name"] == "Bhavmanyu"
    assert face_match_as_dict(None) is None


def test_face_match_as_dict_keeps_bbox():
    payload = {
        "person_id": None,
        "person_name": None,
        "similarity": 0.0,
        "is_known": False,
        "bbox": [10, 20, 110, 140],
        "det_score": 0.88,
        "approaching": True,
    }
    d = face_match_as_dict(payload)
    assert d["bbox"] == [10, 20, 110, 140]
    assert d["approaching"] is True
    assert d["det_score"] == 0.88


def test_scale_bbox_to_frame_doubles_when_upscaled():
    from cafeteria.recognition.live_match import scale_bbox_to_frame

    box = scale_bbox_to_frame((10, 20, 50, 80), src_w=640, src_h=360, dst_w=1280, dst_h=720)
    assert box == [20, 40, 100, 160]


def test_draw_debug_overlay_accepts_live_bbox():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    payload = {
        "person_id": "p1",
        "person_name": "Ada",
        "similarity": 0.91,
        "is_known": True,
        "bbox": [20, 10, 80, 90],
        "approaching": False,
    }
    out = draw_debug_overlay(frame, state="IDLE", fps=10.0, face_match=payload)
    assert out.shape == frame.shape
    # Corner brackets should paint some non-black pixels
    assert int(out.sum()) > 0
