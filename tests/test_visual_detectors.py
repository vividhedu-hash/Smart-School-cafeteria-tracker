"""Built-in OpenCV plate + waste detectors — real CV, no mocks."""
from __future__ import annotations

import cv2
import numpy as np

from cafeteria.detection.plate_detector import ModelNotFoundError, PlateDetector
from cafeteria.detection.visual import classify_waste_visual, detect_plates_visual
from cafeteria.detection.waste_detector import WasteDetector
from test_event_manager import make_manager


def _table_with_plate(food: bool = False, coverage: str = "empty") -> np.ndarray:
    img = np.full((480, 640, 3), 42, dtype=np.uint8)
    img[:] = (38, 42, 48)
    cv2.circle(img, (320, 360), 95, (228, 228, 232), -1)
    cv2.circle(img, (320, 360), 95, (170, 170, 176), 4)
    if coverage == "empty":
        return img
    if coverage == "low":
        cv2.circle(img, (300, 350), 18, (28, 90, 40), -1)
    elif coverage == "medium":
        cv2.ellipse(img, (320, 360), (55, 40), 20, 0, 360, (18, 70, 150), -1)
        cv2.circle(img, (290, 345), 22, (20, 110, 50), -1)
    else:
        cv2.ellipse(img, (320, 360), (78, 62), 10, 0, 360, (12, 55, 130), -1)
        cv2.circle(img, (280, 340), 28, (15, 90, 35), -1)
        cv2.circle(img, (350, 375), 24, (30, 40, 170), -1)
    return img


def test_visual_detects_white_plate_on_720p_frame():
    img = np.full((720, 1280, 3), 38, dtype=np.uint8)
    img[:] = (38, 42, 48)
    cv2.circle(img, (640, 520), 140, (228, 228, 232), -1)
    cv2.circle(img, (640, 520), 140, (170, 170, 176), 5)
    dets = detect_plates_visual(
        img, roi={"x1": 0.05, "y1": 0.40, "x2": 0.95, "y2": 1.0}, min_confidence=0.35
    )
    assert dets, "expected a plate on a 720p frame after downscale cascade"
    cx, cy = dets[0].center
    assert 520 < cx < 760
    assert 400 < cy < 640


def test_visual_detects_white_plate_on_dark_table():
    img = _table_with_plate()
    dets = detect_plates_visual(
        img, roi={"x1": 0.05, "y1": 0.40, "x2": 0.95, "y2": 1.0}, min_confidence=0.35
    )
    assert dets, "expected at least one plate detection"
    best = dets[0]
    assert best.confidence >= 0.40
    cx, cy = best.center
    assert 250 < cx < 390
    assert 300 < cy < 430


def test_visual_empty_plate_is_empty():
    crop = np.full((220, 220, 3), 230, dtype=np.uint8)
    result = classify_waste_visual(crop)
    assert result.label == "EMPTY"
    assert result.is_waste is False
    assert result.confidence > 0.5


def test_visual_food_plate_is_waste():
    crop = np.full((220, 220, 3), 230, dtype=np.uint8)
    cv2.ellipse(crop, (110, 110), (85, 70), 0, 0, 360, (20, 80, 40), -1)
    cv2.circle(crop, (80, 95), 30, (10, 50, 140), -1)
    result = classify_waste_visual(crop)
    assert result.is_waste is True
    assert result.label in ("LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE")
    assert result.confidence > 0.5


def test_visual_detects_plate_in_dim_light():
    img = np.full((720, 1280, 3), 18, dtype=np.uint8)
    cv2.circle(img, (640, 520), 140, (118, 118, 124), -1)
    cv2.circle(img, (640, 520), 140, (80, 80, 86), 5)
    dets = detect_plates_visual(
        img, roi={"x1": 0.05, "y1": 0.40, "x2": 0.95, "y2": 1.0}, min_confidence=0.35
    )
    assert dets, "CLAHE + adaptive pale mask should find a dim plate"
    cx, cy = dets[0].center
    assert 500 < cx < 780
    assert 390 < cy < 650


def test_visual_detects_dark_cafeteria_tray():
    img = np.full((480, 640, 3), 28, dtype=np.uint8)
    img[:] = (24, 26, 30)
    cv2.rectangle(img, (140, 280), (500, 450), (36, 110, 48), -1)
    cv2.rectangle(img, (140, 280), (500, 450), (20, 70, 30), 4)
    dets = detect_plates_visual(
        img, roi={"x1": 0.05, "y1": 0.40, "x2": 0.95, "y2": 1.0}, min_confidence=0.35
    )
    assert dets, "coloured tray on a dark table should be found"
    cx, cy = dets[0].center
    assert 200 < cx < 440
    assert 300 < cy < 450


def test_visual_waste_on_dark_tray_is_not_empty():
    crop = np.full((220, 220, 3), (40, 95, 50), dtype=np.uint8)
    cv2.ellipse(crop, (110, 110), (70, 55), 0, 0, 360, (18, 60, 150), -1)
    result = classify_waste_visual(crop)
    assert result.is_waste is True
    assert result.label != "EMPTY"


def test_plate_detector_loads_visual_when_weights_missing(tmp_path):
    det = PlateDetector(tmp_path / "missing.pt", allow_visual_fallback=True)
    det.load()
    assert det.is_loaded
    assert det.backend == "visual"
    assert det.visual_mode is True
    assert det.proxy_mode is False
    dets = det.detect(_table_with_plate())
    assert dets


def test_waste_detector_loads_visual_when_weights_missing(tmp_path):
    det = WasteDetector(tmp_path / "missing.pt", allow_visual_fallback=True)
    det.load()
    assert det.is_loaded
    assert det.backend == "visual"
    crop = np.full((180, 180, 3), 225, dtype=np.uint8)
    result = det.classify(crop)
    assert result.label == "EMPTY"


def test_missing_weights_still_raise_when_fallbacks_off(tmp_path):
    det = PlateDetector(
        tmp_path / "nope.pt",
        allow_coco_fallback=False,
        allow_visual_fallback=False,
    )
    try:
        det.load()
        assert False, "expected ModelNotFoundError"
    except ModelNotFoundError:
        assert det.is_loaded is False
        assert det.proxy_mode is False


def test_visual_backends_drive_a_real_event(tmp_path):
    """Loaded visual detectors are enough for EventManager to complete an event."""
    from cafeteria.camera.base import TimestampedFrame
    import time

    plate = PlateDetector(tmp_path / "p.pt", allow_visual_fallback=True)
    waste = WasteDetector(tmp_path / "w.pt", allow_visual_fallback=True)
    plate.load()
    waste.load()
    em, _sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)
    img = _table_with_plate(coverage="high")

    def _frame():
        return TimestampedFrame(
            frame_id=1,
            timestamp=time.monotonic(),
            wall_time=time.time(),
            image=img,
            camera_id="test",
        )

    event = None
    for _ in range(40):
        event = em.process_frame(_frame(), [_frame()])
        if event is not None:
            break
    assert event is not None
    assert event.waste_result is not None
    assert event.waste_result.is_waste
    assert event.plate_detected is True
