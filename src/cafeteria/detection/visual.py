"""
Built-in OpenCV detectors used when no trained YOLO weights exist.

These are real computer-vision algorithms (not random labels). They let the
live pipeline run on day one on any webcam. Training a YOLOv8 plate or waste
model on cafeteria photos replaces this backend automatically.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from cafeteria.detection.plate_detector import Detection
from cafeteria.detection.waste_detector import WasteResult

WASTE_LABELS = ("EMPTY", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE")

# Run the expensive circle/contour search at this width, then map boxes back.
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


def _clahe_bgr(image: np.ndarray) -> np.ndarray:
    """Lighting-invariant copy for geometry search (does not invent colour)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect_plates_visual(
    image: np.ndarray,
    roi: Optional[dict] = None,
    min_confidence: float = 0.35,
) -> list[Detection]:
    """
    Find plate/tray-like objects using a cheap-to-expensive cascade.

    Cues: adaptive pale ware, dark/coloured cafeteria trays, Hough circles,
    rounded/rectangular contours. Work runs on a downscaled, CLAHE-equalised
    ROI so cafeteria lighting (window vs overhead) does not kill detection.
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

    equalized = _clahe_bgr(work)
    wh, ww = work.shape[:2]
    min_area = max(400, int(wh * ww * 0.025))
    max_area = int(wh * ww * 0.85)

    blobs = _bright_blob_plates(equalized, 0, 0, min_area, max_area)
    blobs.extend(_bright_blob_plates(work, 0, 0, min_area, max_area))
    blob_kept = [
        d for d in _nms(blobs)
        if d.confidence >= min_confidence and d.area >= min_area
    ]
    if blob_kept and blob_kept[0].confidence >= CASCADE_BLOB_CONF:
        mapped = _map_dets_to_frame(blob_kept, scale, rx1, ry1)
        mapped.sort(key=lambda d: d.confidence, reverse=True)
        return mapped[:3]

    candidates: list[Detection] = list(blobs)
    candidates.extend(_tray_plates(work, 0, 0, min_area, max_area))
    candidates.extend(_hough_plates(equalized, 0, 0, min_area, max_area))
    strong = [d for d in candidates if d.confidence >= 0.70]
    if not strong:
        candidates.extend(_contour_plates(equalized, 0, 0, min_area, max_area))

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
    # Adaptive Canny-ish param1 from median so dim rooms still circle-find.
    median = float(np.median(blur))
    param1 = int(np.clip(median * 1.1, 50, 140))
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(40, min(ch, cw) // 4),
        param1=param1,
        param2=26,
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
    v_med = float(np.median(val))
    # Absolute cream-ware floor plus a relative lift so dim rooms still work
    # after CLAHE (median rises; pale ware stays above the table).
    pale_floor = max(118, min(168, v_med + 48))
    mask = ((val >= pale_floor) & (sat <= 100)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _detections_from_mask(mask, crop, ox, oy, min_area, max_area, base_conf=0.62)


def _tray_plates(
    crop: np.ndarray, ox: int, oy: int, min_area: int, max_area: int
) -> list[Detection]:
    """Dark or coloured cafeteria trays — the white-blob path misses these."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    v_med = float(np.median(val))
    # Tray sits between a dark table and a highlight; coloured plastic is sat.
    mid = (val > max(28.0, v_med * 0.45)) & (val < min(155.0, v_med + 70.0))
    coloured = sat >= 38
    mask = ((mid & coloured) | ((val < 130) & (sat >= 50) & (val > 35))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _detections_from_mask(
        mask, crop, ox, oy, min_area, max_area, base_conf=0.56, prefer_rect=True
    )


def _contour_plates(
    crop: np.ndarray, ox: int, oy: int, min_area: int, max_area: int
) -> list[Detection]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    med = float(np.median(blur))
    lo = int(max(20, 0.5 * med))
    hi = int(min(180, 1.5 * med + 20))
    edges = cv2.Canny(blur, lo, hi)
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
    prefer_rect: bool = False,
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
        fill = area / float(max(bw * bh, 1))
        tray_like = 0.55 <= aspect <= 2.4 and area >= min_area * 1.15 and fill >= 0.45
        if prefer_rect:
            if not (0.50 <= aspect <= 2.6 and fill >= 0.40 and circularity >= 0.18):
                continue
        elif circularity < 0.35 and not tray_like:
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
    Estimate leftover food from residual vs the plate's own colour.

    Rim annulus = empty-ware colour. Inner ellipse pixels that deviate in Lab
    (or have texture / chroma) are leftovers. This survives dark trays and
    cafeteria lighting better than a global saturation threshold.
    """
    if plate_crop is None or plate_crop.size == 0 or plate_crop.ndim < 2:
        return WasteResult(
            label="EMPTY",
            confidence=0.0,
            all_scores={k: 0.0 for k in WASTE_LABELS},
            is_waste=False,
        )

    h, w = plate_crop.shape[:2]
    if min(h, w) < 48:
        plate_crop = cv2.resize(plate_crop, (160, 160), interpolation=cv2.INTER_LINEAR)
        h, w = plate_crop.shape[:2]

    lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    inner = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    inner_mask = np.zeros((h, w), dtype=np.uint8)
    ring_mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(
        inner_mask, (cx, cy),
        (max(1, int(w * 0.40)), max(1, int(h * 0.40))),
        0, 0, 360, 255, -1,
    )
    cv2.ellipse(
        ring_mask, (cx, cy),
        (max(2, int(w * 0.48)), max(2, int(h * 0.48))),
        0, 0, 360, 255, -1,
    )
    cv2.ellipse(
        ring_mask, (cx, cy),
        (max(1, int(w * 0.40)), max(1, int(h * 0.40))),
        0, 0, 360, 0, -1,
    )

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)
    ring = ring_mask > 0
    if int(np.count_nonzero(ring)) < 30:
        ring = inner_mask > 0
    plate_l = float(np.median(L[ring]))
    plate_a = float(np.median(A[ring]))
    plate_b = float(np.median(B[ring]))
    dist = np.sqrt((L - plate_l) ** 2 + (A - plate_a) ** 2 + (B - plate_b) ** 2)

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    colourful = (sat >= 42) & (val >= 35) & (val <= 245)
    dark_food = (sat >= 18) & (val < 115) & (val > 22)
    residual = dist >= 16.0
    # Pale leftovers (rice, roti) on pale ware: local contrast, not chroma.
    # Ignore CLAHE tile flicker on an already-uniform empty plate.
    inner_px = inner_mask > 0
    l_std = float(np.std(L[inner_px])) if int(np.count_nonzero(inner_px)) else 0.0
    contrast = cv2.Laplacian(lab[:, :, 0], cv2.CV_32F, ksize=3)
    textured = (np.abs(contrast) >= 14.0) & (l_std >= 8.0)
    food = (residual | colourful | dark_food | textured) & inner_px
    plate_pixels = int(np.count_nonzero(inner_mask))
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
