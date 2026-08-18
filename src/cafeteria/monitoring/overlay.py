"""Debug overlay for the live camera frame (kept out of main.py so tests stay light)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cafeteria.recognition.live_match import face_match_field


def _clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w - 1, int(x2)))
    y2 = max(y1 + 1, min(h - 1, int(y2)))
    return x1, y1, x2, y2


def draw_face_perimeter(frame, bbox, color, label: str | None = None) -> None:
    """Corner-bracket lock around a detected face (walk-up / walk-across scan)."""
    if not bbox or len(bbox) < 4:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    length = max(14, min(x2 - x1, y2 - y1) // 4)
    t = 2
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, t)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, t)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, t)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, t)
    if label:
        cv2.putText(
            frame, label, (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )


def draw_debug_overlay(
    image: np.ndarray,
    state: str,
    fps: float,
    plate_det=None,
    waste_result=None,
    face_match=None,
    latency_ms: float = 0.0,
    roi_cfg=None,
    debug: bool = True,
) -> np.ndarray:
    """Draw debug information on a copy of the frame."""
    if not debug:
        return image

    frame = image.copy()
    h, w = frame.shape[:2]

    if roi_cfg:
        p = roi_cfg.plate
        px1, py1, px2, py2 = (
            int(p.x1 * w), int(p.y1 * h),
            int(p.x2 * w), int(p.y2 * h)
        )
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 0), 1)
        cv2.putText(frame, "PLATE ROI", (px1 + 4, py1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    bbox = face_match_field(face_match, "bbox")
    if bbox:
        if face_match_field(face_match, "is_known"):
            color = (60, 220, 80)
            label = str(face_match_field(face_match, "person_name") or "IDENTIFIED")
        elif face_match_field(face_match, "approaching"):
            color = (220, 180, 40)
            label = "LOCKING"
        else:
            color = (0, 180, 255)
            label = "SCAN"
        draw_face_perimeter(frame, bbox, color, label)

    if plate_det:
        cv2.rectangle(frame,
                      (plate_det.x1, plate_det.y1),
                      (plate_det.x2, plate_det.y2),
                      (0, 255, 0), 2)
        cv2.putText(frame,
                    f"Plate {plate_det.confidence:.2f}",
                    (plate_det.x1, plate_det.y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if waste_result:
        waste_colour = {
            "EMPTY": (180, 180, 180),
            "LOW_WASTE": (0, 200, 255),
            "MEDIUM_WASTE": (0, 140, 255),
            "HIGH_WASTE": (0, 60, 255),
        }.get(waste_result.label, (255, 255, 255))
        cv2.putText(frame,
                    f"Waste: {waste_result.label}  ({waste_result.confidence:.2f})",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, waste_colour, 2)

    if face_match_field(face_match, "is_known"):
        name = face_match_field(face_match, "person_name") or "Unknown"
        sim = float(face_match_field(face_match, "similarity") or 0.0)
        cv2.putText(frame,
                    f"Person: {name}  ({sim:.2f})",
                    (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)

    state_colour = {
        "IDLE":               (80, 80, 80),
        "PLATE_DETECTED":     (0, 180, 0),
        "FOOD_ANALYSIS":      (0, 180, 200),
        "WASTE_EVENT":        (0, 100, 255),
        "FACE_CAPTURE":       (200, 100, 0),
        "FACE_RECOGNITION":   (200, 60, 0),
        "TRANSACTION_COMMIT": (0, 200, 100),
        "REVIEW_REQUIRED":    (200, 0, 200),
        "COOLDOWN":           (100, 100, 0),
        "ERROR":              (0, 0, 200),
    }.get(state, (80, 80, 80))

    cv2.rectangle(frame, (0, 0), (w, 30), state_colour, -1)
    cv2.putText(frame, f"STATE: {state}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.putText(frame,
                f"FPS: {fps:.1f}   Latency: {latency_ms:.0f} ms",
                (w - 260, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return frame
