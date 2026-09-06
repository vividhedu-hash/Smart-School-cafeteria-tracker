"""Normalize live face-match payloads and walk-past bbox tracking helpers."""
from __future__ import annotations

from typing import Any, Optional


def box_iou(a: Any, b: Any) -> float:
    """Intersection-over-union for [x1, y1, x2, y2] boxes. 0 if invalid."""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in list(a)[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in list(b)[:4]]
    except (TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def smooth_bbox(prev: Any, current: Any, alpha: float = 0.45) -> Optional[list[int]]:
    """Exponential moving average so the lock doesn't jitter while walking."""
    if current is None:
        return None
    try:
        curr = [float(v) for v in list(current)[:4]]
    except (TypeError, ValueError):
        return None
    if prev is None:
        return [int(round(v)) for v in curr]
    try:
        old = [float(v) for v in list(prev)[:4]]
    except (TypeError, ValueError):
        return [int(round(v)) for v in curr]
    a = max(0.0, min(1.0, float(alpha)))
    return [int(round(a * c + (1.0 - a) * p)) for p, c in zip(old, curr)]


def persist_bbox(
    previous: Optional[list[int]],
    current: Optional[list[int]],
    missing_frames: int,
    max_hold: int = 12,
) -> tuple[Optional[list[int]], int]:
    """
    Keep the last face box for a few frames when InsightFace briefly misses.

    Walking across the camera often drops a frame. Holding the box avoids
    flicker without inventing a detection after the person has left.
    """
    if current is not None:
        return current, 0
    if previous is None:
        return None, missing_frames + 1
    nxt = missing_frames + 1
    if nxt > max_hold:
        return None, nxt
    return previous, nxt


def face_size_px(bbox: Any) -> tuple[int, int]:
    """Return (width, height) in pixels, or (0, 0)."""
    if bbox is None:
        return 0, 0
    try:
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    except (TypeError, ValueError):
        return 0, 0
    return int(max(0, x2 - x1)), int(max(0, y2 - y1))


def scale_bbox_to_frame(
    bbox: Any,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> Optional[list[int]]:
    """Map a detection box from a resized inference image back to the full frame."""
    if bbox is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    except (TypeError, ValueError):
        return None
    sx = dst_w / src_w if src_w else 1.0
    sy = dst_h / src_h if src_h else 1.0
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]


def face_match_field(face_match: Any, name: str, default: Any = None) -> Any:
    """Read a field from a MatchResult-like object or a live_face_match dict."""
    if face_match is None:
        return default
    if isinstance(face_match, dict):
        return face_match.get(name, default)
    return getattr(face_match, name, default)


def face_search_crop(
    image: Any,
    held_bbox: Any = None,
    roi: Any = None,
    pad_ratio: float = 0.55,
    min_side: int = 80,
) -> tuple[Any, int, int]:
    # [AI-CoLab: Verified by Antigravity] Gated search window crop around held bbox or ROI
    """
    Return (crop, origin_x, origin_y) for gated face search.

    Prefer a padded window around the current lock. Otherwise use the
    configured face ROI. Origin is in full-image coordinates so boxes
    map back with a simple offset.
    """
    if image is None:
        return image, 0, 0
    try:
        h, w = int(image.shape[0]), int(image.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return image, 0, 0

    if held_bbox is not None:
        try:
            x1, y1, x2, y2 = [int(v) for v in list(held_bbox)[:4]]
        except (TypeError, ValueError):
            x1 = y1 = x2 = y2 = 0
        else:
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)
            if cx2 - cx1 >= min_side and cy2 - cy1 >= min_side:
                return image[cy1:cy2, cx1:cx2], cx1, cy1

    if roi:
        try:
            rx1 = int(max(0.0, min(1.0, float(roi.get("x1", 0.0)))) * w)
            ry1 = int(max(0.0, min(1.0, float(roi.get("y1", 0.0)))) * h)
            rx2 = int(max(0.0, min(1.0, float(roi.get("x2", 1.0)))) * w)
            ry2 = int(max(0.0, min(1.0, float(roi.get("y2", 1.0)))) * h)
        except (AttributeError, TypeError, ValueError):
            rx1, ry1, rx2, ry2 = 0, 0, w, h
        if rx2 > rx1 + 8 and ry2 > ry1 + 8:
            return image[ry1:ry2, rx1:rx2], rx1, ry1

    return image, 0, 0


def face_match_as_dict(face_match: Any) -> Optional[dict]:
    """
    Return a JSON-safe live_face_match dict, or None when there is no payload.

    Safe for both MatchResult objects and dicts written to runtime_state.json.
    """
    if face_match is None:
        return None
    similarity = face_match_field(face_match, "similarity", 0.0)
    try:
        similarity = float(similarity or 0.0)
    except (TypeError, ValueError):
        similarity = 0.0
    bbox = face_match_field(face_match, "bbox")
    if bbox is not None:
        try:
            bbox = [int(v) for v in list(bbox)[:4]]
            if len(bbox) != 4:
                bbox = None
        except (TypeError, ValueError):
            bbox = None
    return {
        "person_id": face_match_field(face_match, "person_id"),
        "person_name": face_match_field(face_match, "person_name"),
        "similarity": similarity,
        "is_known": bool(face_match_field(face_match, "is_known", False)),
        "bbox": bbox,
        "det_score": face_match_field(face_match, "det_score"),
        "approaching": bool(face_match_field(face_match, "approaching", False)),
        "infer_id": face_match_field(face_match, "infer_id"),
        "moving": bool(face_match_field(face_match, "moving", False)),
    }
