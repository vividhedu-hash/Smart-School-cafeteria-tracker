"""
# [AI-CoLab: Verified by Antigravity] — Clean OpenCV fallback detector for plates & waste
Built-in OpenCV detectors used when no trained YOLO weights exist.


These are real computer-vision algorithms (not random labels, not mocked
YOLO). They let the live pipeline run on day one. Training a YOLOv8 plate
or waste model on cafeteria photos replaces this backend automatically.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cafeteria.detection.plate_detector import Detection
from cafeteria.detection.waste_detector import WasteResult

WASTE_LABELS = ("EMPTY", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE")

# Run the expensive circle/contour search at this width, then map boxes back.
# 1280×720 HoughCircles on the plate ROI was ~270 ms on the live engine.
ANALYZE_MAX_WIDTH = 512
# Skip Hough + Canny when the cheap bright-blob pass is this sure.
CASCADE_BLOB_CONF = 0.72


def _roi_pixels(roi: Optional[dict], width: int, height: int) -> tuple[int, int, int, int]:
    if not roi:
        return 0, 0, width, height
    x1 = int(max(0.0, min(1.0, float(roi.get("x1", 0.0)))) * width)
    y1 = int(max(0.0, min(1.0, float(roi.get("y1", 0.0)))) * height)
    x2 = int(max(0.0, min(1.0, float(roi.get("x2", 1.0)))) * width)
    y2 = int(max(0.0, min(1.0, float(roi.get("y2", 1.0)))) * height)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[Detection], iou_thresh: float = 0.40) -> list[Detection]:
    ordered = sorted(dets, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    for det in ordered:
        if all(_iou(det, kept) < iou_thresh for kept in keep):
            keep.append(det)
    return keep


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(x1 + 1, min(w, x2)))
    y2 = int(max(y1 + 1, min(h, y2)))
    return x1, y1, x2, y2


def _map_dets_to_frame(
    dets: list[Detection], scale: float, ox: int, oy: int
) -> list[Detection]:
    """Map boxes from the analysis image back to full-frame coordinates."""
    inv = 1.0 / scale if scale > 0 else 1.0
    mapped: list[Detection] = []
    for d in dets:
        mapped.append(Detection(
            label=d.label,
            confidence=d.confidence,
            x1=int(round(d.x1 * inv)) + ox,
            y1=int(round(d.y1 * inv)) + oy,
            x2=int(round(d.x2 * inv)) + ox,
            y2=int(round(d.y2 * inv)) + oy,
        ))
    return mapped


def detect_plates_visual(
    image: np.ndarray,
    roi: Optional[dict] = None,
    min_confidence: float = 0.35,
) -> list[Detection]:
    # [AI-CoLab: Verified by Antigravity] Downscaled ROI & bright-blob cascade optimization
    """
    Find plate/tray-like objects in a BGR frame using a cheap-to-expensive
    cascade: bright blobs, then Hough circles, then rounded contours.

    Work runs on a downscaled ROI so HoughCircles stays real-time. Boxes
    are returned in full-image coordinates.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    rx1, ry1, rx2, ry2 = _roi_pixels(roi, w, h)
    crop = image[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return []

    ch, cw = crop.shape[:2]
    scale = 1.0
    work = crop
    if cw > ANALYZE_MAX_WIDTH:
        scale = ANALYZE_MAX_WIDTH / float(cw)
        work = cv2.resize(
            crop,
            (ANALYZE_MAX_WIDTH, max(1, int(round(ch * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    wh, ww = work.shape[:2]
    min_area = max(400, int(wh * ww * 0.025))
    max_area = int(wh * ww * 0.85)

    blobs = _bright_blob_plates(work, 0, 0, min_area, max_area)
    blob_kept = [
        d for d in _nms(blobs)
        if d.confidence >= min_confidence and d.area >= min_area
    ]
    if blob_kept and blob_kept[0].confidence >= CASCADE_BLOB_CONF:
        mapped = _map_dets_to_frame(blob_kept, scale, rx1, ry1)
        mapped.sort(key=lambda d: d.confidence, reverse=True)
        return mapped[:3]

    candidates: list[Detection] = list(blobs)
    candidates.extend(_hough_plates(work, 0, 0, min_area, max_area))
    strong = [d for d in candidates if d.confidence >= 0.70]
    if not strong:
        candidates.extend(_contour_plates(work, 0, 0, min_area, max_area))

    frame_min_area = max(800, int(ch * cw * 0.025))
    kept = _map_dets_to_frame(_nms(candidates), scale, rx1, ry1)
    kept = [
        d for d in kept
        if d.confidence >= min_confidence and d.area >= frame_min_area
    ]
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept[:3]


def _hough_plates(
    crop: np.ndarray, ox: int, oy: int, min_area: int, max_area: int
) -> list[Detection]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    ch, cw = gray.shape[:2]
    min_r = max(18, int(min(ch, cw) * 0.08))
    max_r = max(min_r + 8, int(min(ch, cw) * 0.48))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, min(ch, cw) // 4),
        param1=90,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    out: list[Detection] = []
    if circles is None:
        return out
    for cx, cy, r in np.round(circles[0]).astype(int):
        area = int(np.pi * r * r)
        if area < min_area or area > max_area:
            continue
        x1, y1, x2, y2 = _clip_box(cx - r, cy - r, cx + r, cy + r, cw, ch)
        conf = float(np.clip(0.55 + (r / max(max_r, 1)) * 0.30, 0.55, 0.92))
        out.append(Detection(
            label="plate",
            confidence=round(conf, 3),
            x1=x1 + ox, y1=y1 + oy, x2=x2 + ox, y2=y2 + oy,
        ))
    return out


def _bright_blob_plates(
    crop: np.ndarray, ox: int, oy: int, min_area: int, max_area: int
) -> list[Detection]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    # White / cream / pale plates on a darker table.
    mask = ((val >= 145) & (sat <= 90)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _detections_from_mask(mask, crop, ox, oy, min_area, max_area, base_conf=0.62)


def _contour_plates(
    crop: np.ndarray, ox: int, oy: int, min_area: int, max_area: int
) -> list[Detection]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _detections_from_mask(closed, crop, ox, oy, min_area, max_area, base_conf=0.50)


def _detections_from_mask(
    mask: np.ndarray,
    crop: np.ndarray,
    ox: int,
    oy: int,
    min_area: int,
    max_area: int,
    base_conf: float,
) -> list[Detection]:
    ch, cw = crop.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue
        circularity = float(4.0 * np.pi * area / (peri * peri))
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 24 or bh < 24:
            continue
        aspect = bw / float(bh)
        # Circles / ellipses, or cafeteria trays that are roughly rectangular.
        tray_like = 0.55 <= aspect <= 2.2 and area >= min_area * 1.2
        if circularity < 0.35 and not tray_like:
            continue
        pad = int(0.04 * max(bw, bh))
        x1, y1, x2, y2 = _clip_box(x - pad, y - pad, x + bw + pad, y + bh + pad, cw, ch)
        shape_score = circularity if circularity >= 0.35 else 0.40
        conf = float(np.clip(base_conf + 0.35 * shape_score, 0.40, 0.93))
        out.append(Detection(
            label="plate",
            confidence=round(conf, 3),
            x1=x1 + ox, y1=y1 + oy, x2=x2 + ox, y2=y2 + oy,
        ))
    return out


def classify_waste_visual(plate_crop: np.ndarray) -> WasteResult:
    """
    Estimate leftover food from colour occupancy on a plate crop.

    Scores only the inner ellipse (not the table or plate rim) after a
    light CLAHE pass so cafeteria lighting changes move coverage less.
    """
    if plate_crop is None or plate_crop.size == 0 or plate_crop.ndim < 2:
        return WasteResult(
            label="EMPTY",
            confidence=0.0,
            all_scores={k: 0.0 for k in WASTE_LABELS},
            is_waste=False,
        )

    h, w = plate_crop.shape[:2]
    inner = plate_crop
    if h >= 16 and w >= 16:
        lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        inner = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (w // 2, h // 2),
        (max(1, int(w * 0.38)), max(1, int(h * 0.38))),
        0, 0, 360, 255, -1,
    )
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    colourful = (sat >= 42) & (val >= 35) & (val <= 245)
    dark_food = (sat >= 18) & (val < 115) & (val > 22)
    food = (colourful | dark_food) & (mask > 0)
    plate_pixels = int(np.count_nonzero(mask))
    coverage = float(np.count_nonzero(food) / plate_pixels) if plate_pixels else 0.0

    if coverage < 0.055:
        label = "EMPTY"
        conf = float(np.clip(0.92 - coverage * 4.0, 0.55, 0.97))
    elif coverage < 0.17:
        label = "LOW_WASTE"
        conf = float(np.clip(0.58 + (coverage - 0.055) / 0.115 * 0.28, 0.55, 0.90))
    elif coverage < 0.40:
        label = "MEDIUM_WASTE"
        conf = float(np.clip(0.58 + (coverage - 0.17) / 0.23 * 0.28, 0.55, 0.90))
    else:
        label = "HIGH_WASTE"
        conf = float(np.clip(0.60 + min(0.32, (coverage - 0.40) * 0.8), 0.60, 0.94))

    all_scores = {k: 0.04 for k in WASTE_LABELS}
    all_scores[label] = round(conf, 4)
    leftover = max(0.0, 1.0 - conf - 0.04 * 3)
    neighbours = [k for k in WASTE_LABELS if k != label]
    share = leftover / max(len(neighbours), 1)
    for k in neighbours:
        all_scores[k] = round(0.04 + share, 4)

    return WasteResult(
        label=label,
        confidence=round(conf, 4),
        all_scores=all_scores,
        is_waste=label != "EMPTY",
    )
