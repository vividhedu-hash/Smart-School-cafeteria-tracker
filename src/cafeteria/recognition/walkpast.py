"""
Walk-past face tracker — constant-velocity lock for a moving person.

InsightFace often misses 1–3 frames while someone walks. The previous lock
froze the box in place, searched a tight window, and dropped identity when
IoU fell. This tracker:

  - coasts the box with estimated velocity during misses
  - widens the search window in the direction of travel
  - falls back to the full face ROI if the gated crop is empty
  - keeps a confirmed identity through one or two blurry frames
  - infers more often while the person is moving, not less
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from cafeteria.recognition.live_match import (
    box_iou,
    face_search_crop,
    face_size_px,
    scale_bbox_to_frame,
    smooth_bbox,
)

GetFaces = Callable[[np.ndarray], list[dict]]
MatchFn = Callable[[np.ndarray], Any]


def bbox_center(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def translate_bbox(
    bbox: Any, dx: float, dy: float, frame_w: int, frame_h: int
) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    w, h = x2 - x1, y2 - y1
    nx1 = max(0.0, min(float(frame_w - 1), x1 + dx))
    ny1 = max(0.0, min(float(frame_h - 1), y1 + dy))
    nx2 = max(nx1 + 1.0, min(float(frame_w), nx1 + w))
    ny2 = max(ny1 + 1.0, min(float(frame_h), ny1 + h))
    return [int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))]


def predict_bbox(
    bbox: Any,
    vx: float,
    vy: float,
    dt: float,
    frame_w: int,
    frame_h: int,
) -> Optional[list[int]]:
    """Coast the lock using last known velocity (pixels / second)."""
    if bbox is None:
        return None
    dt = max(0.0, float(dt))
    return translate_bbox(bbox, vx * dt, vy * dt, frame_w, frame_h)


def update_velocity(
    prev: Any,
    curr: Any,
    dt: float,
    vx: float,
    vy: float,
    alpha: float = 0.55,
) -> tuple[float, float]:
    """EMA of box-center velocity, clamped to a fast walk-across."""
    if prev is None or curr is None or dt < 1e-3:
        return vx, vy
    pcx, pcy = bbox_center(prev)
    ccx, ccy = bbox_center(curr)
    inst_vx = (ccx - pcx) / dt
    inst_vy = (ccy - pcy) / dt
    inst_vx = max(-900.0, min(900.0, inst_vx))
    inst_vy = max(-900.0, min(900.0, inst_vy))
    a = max(0.0, min(1.0, float(alpha)))
    return a * inst_vx + (1.0 - a) * vx, a * inst_vy + (1.0 - a) * vy


def motion_speed(vx: float, vy: float) -> float:
    return float(math.hypot(vx, vy))


def same_track(
    prev: Any,
    curr: Any,
    vx: float = 0.0,
    vy: float = 0.0,
    dt: float = 0.08,
    min_iou: float = 0.08,
) -> bool:
    """
    True when `curr` is the same walking person as `prev`.

    Fast walk-across often yields IoU near zero even for the same face.
    Accept a detection whose center is near the motion-predicted center.
    """
    if prev is None or curr is None:
        return False
    if box_iou(prev, curr) >= min_iou:
        return True
    dt = max(1e-3, float(dt))
    predicted = predict_bbox(prev, vx, vy, dt, 10_000, 10_000)
    if predicted is None:
        return False
    pcx, pcy = bbox_center(predicted)
    ccx, ccy = bbox_center(curr)
    dist = math.hypot(pcx - ccx, pcy - ccy)
    pw, ph = face_size_px(prev)
    gate = max(0.6 * max(pw, ph), 28.0) + motion_speed(vx, vy) * dt * 0.6
    return dist <= gate


def adaptive_smooth_alpha(speed_px_s: float) -> float:
    """Trust the new box more when the person is moving quickly."""
    if speed_px_s >= 180:
        return 0.78
    if speed_px_s >= 80:
        return 0.62
    return 0.42


def adaptive_pad_ratio(base: float, speed_px_s: float) -> float:
    extra = 0.0
    if speed_px_s >= 60:
        extra = 0.25
    if speed_px_s >= 160:
        extra = 0.50
    return min(1.8, float(base) + extra)


def lead_bbox(
    bbox: Any,
    vx: float,
    vy: float,
    frame_w: int,
    frame_h: int,
    lead_seconds: float = 0.20,
) -> Optional[list[int]]:
    """Stretch the lock in the walk direction so the next frame stays in-window."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    lead_x = float(vx) * float(lead_seconds)
    lead_y = float(vy) * float(lead_seconds)
    if lead_x > 0:
        x2 = min(float(frame_w), x2 + lead_x)
    elif lead_x < 0:
        x1 = max(0.0, x1 + lead_x)
    if lead_y > 0:
        y2 = min(float(frame_h), y2 + lead_y)
    elif lead_y < 0:
        y1 = max(0.0, y1 + lead_y)
    if x2 <= x1 or y2 <= y1:
        return [int(v) for v in list(bbox)[:4]]
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _resize_for_infer(image: np.ndarray, infer_max: int) -> np.ndarray:
    import cv2
    h, w = image.shape[:2]
    if w <= infer_max:
        return image
    return cv2.resize(
        image,
        (infer_max, max(1, int(h * infer_max / w))),
        interpolation=cv2.INTER_LINEAR,
    )


@dataclass
class WalkPastTracker:
    # [AI-CoLab: Cursor] Constant-velocity lock so a walk-across does not drop ID.
    """Stateful lock + identity memory for the always-on face thread."""

    lock_min: int = 24
    identify_min: int = 40
    hold_seconds: float = 0.90
    infer_max_width: int = 640
    face_roi: dict = field(default_factory=dict)
    base_pad_ratio: float = 0.90

    held_bbox: Optional[list[int]] = None
    held_until: float = 0.0
    held_identity: Optional[dict] = None
    vx: float = 0.0
    vy: float = 0.0
    last_update: float = 0.0
    infer_id: int = 0

    def speed(self) -> float:
        return motion_speed(self.vx, self.vy)

    def infer_interval(self, now: float) -> float:
        locked = self.held_bbox is not None and now < self.held_until
        moving = self.speed() >= 70.0
        if moving or not locked:
            return 0.05
        if self.held_identity:
            return 0.10
        return 0.07

    def should_infer(self, now: float, last_infer_at: float) -> bool:
        return (now - last_infer_at) >= self.infer_interval(now)

    def _dt(self, now: float) -> float:
        if self.last_update <= 0:
            return 0.08
        return max(0.0, now - self.last_update)

    def search_crop(
        self, image: np.ndarray, now: float
    ) -> tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        pred = predict_bbox(
            self.held_bbox, self.vx, self.vy, self._dt(now), w, h
        )
        search_box = lead_bbox(pred or self.held_bbox, self.vx, self.vy, w, h)
        pad = adaptive_pad_ratio(self.base_pad_ratio, self.speed())
        return face_search_crop(
            image,
            held_bbox=search_box,
            roi=self.face_roi,
            pad_ratio=pad,
        )

    def detect(
        self,
        image: np.ndarray,
        get_faces: GetFaces,
        now: float,
    ) -> tuple[Optional[dict], int, int, tuple[int, int]]:
        """
        Run InsightFace on the motion window, then the full face ROI on a miss.

        Returns (largest_face or None, ox, oy, (search_w, search_h)).
        Face bbox is in the search-crop coordinate system (after any resize
        the caller maps with scale_bbox_to_frame).
        """
        search, ox, oy = self.search_crop(image, now)
        small = _resize_for_infer(search, self.infer_max_width)
        faces = get_faces(small)
        mapped = self._map_largest(faces, small, search, ox, oy)
        if mapped is not None:
            return mapped, ox, oy, (search.shape[1], search.shape[0])

        if self.held_bbox is None:
            return None, ox, oy, (search.shape[1], search.shape[0])

        # Gated window missed a walker — search the whole face ROI.
        fallback, fox, foy = face_search_crop(
            image, held_bbox=None, roi=self.face_roi, pad_ratio=self.base_pad_ratio
        )
        if fallback is search and fox == ox and foy == oy:
            return None, ox, oy, (search.shape[1], search.shape[0])
        small_f = _resize_for_infer(fallback, self.infer_max_width)
        faces_f = get_faces(small_f)
        mapped_f = self._map_largest(faces_f, small_f, fallback, fox, foy)
        if mapped_f is None:
            return None, fox, foy, (fallback.shape[1], fallback.shape[0])
        return mapped_f, fox, foy, (fallback.shape[1], fallback.shape[0])

    def _map_largest(
        self,
        faces: list[dict],
        small: np.ndarray,
        search: np.ndarray,
        ox: int,
        oy: int,
    ) -> Optional[dict]:
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
        raw_in_search = scale_bbox_to_frame(
            face["bbox"], small.shape[1], small.shape[0],
            search.shape[1], search.shape[0],
        )
        if raw_in_search is None:
            return None
        raw_bbox = [
            raw_in_search[0] + ox,
            raw_in_search[1] + oy,
            raw_in_search[2] + ox,
            raw_in_search[3] + oy,
        ]
        return {
            "bbox": raw_bbox,
            "embedding": face.get("embedding"),
            "det_score": float(face.get("det_score") or 0.0),
        }

    def step(
        self,
        image: np.ndarray,
        get_faces: GetFaces,
        now: float,
        matcher: Any = None,
    ) -> Optional[dict]:
        """Consume one camera frame; return a live_face_match dict or None."""
        h, w = image.shape[:2]
        detection = self.detect(image, get_faces, now)[0]
        if detection is None:
            return self._coast(now, w, h)
        return self._on_hit(detection, now, w, h, matcher)

    def _on_hit(
        self,
        detection: dict,
        now: float,
        frame_w: int,
        frame_h: int,
        matcher: Any,
    ) -> dict:
        raw_bbox = detection["bbox"]
        dt = self._dt(now)
        tracked = same_track(
            self.held_bbox, raw_bbox, self.vx, self.vy, dt=dt
        )
        if self.held_bbox is not None and not tracked:
            self.held_identity = None
            self.vx, self.vy = 0.0, 0.0

        speed = self.speed()
        bbox = smooth_bbox(
            self.held_bbox, raw_bbox, alpha=adaptive_smooth_alpha(speed)
        )
        if self.held_bbox is not None and bbox is not None:
            self.vx, self.vy = update_velocity(
                self.held_bbox, raw_bbox, max(dt, 0.04), self.vx, self.vy
            )
        self.held_bbox = bbox
        hold = self.hold_seconds
        if speed >= 80:
            hold = max(hold, 1.10)
        self.held_until = now + hold
        self.last_update = now
        self.infer_id += 1

        fw, fh = face_size_px(bbox)
        moving_now = speed >= 70.0
        identify_gate = self.identify_min
        if moving_now:
            identify_gate = max(self.lock_min + 4, self.identify_min - 12)
        approaching = fw < identify_gate or fh < identify_gate
        too_small = fw < self.lock_min or fh < self.lock_min
        det_score = float(detection.get("det_score") or 0.0)
        result = {
            "person_id": None,
            "person_name": None,
            "similarity": 0.0,
            "is_known": False,
            "bbox": bbox,
            "det_score": round(det_score, 3),
            "approaching": approaching or too_small,
            "infer_id": self.infer_id,
            "moving": moving_now,
        }

        emb = detection.get("embedding")
        sharp_enough = (not approaching) or (moving_now and det_score >= 0.50)
        can_identify = (
            matcher is not None
            and emb is not None
            and not too_small
            and sharp_enough
        )
        if can_identify:
            m = matcher.match(emb)
            if m.is_known:
                self.held_identity = {
                    "person_id": m.person_id,
                    "person_name": m.person_name,
                    "similarity": round(m.similarity, 3),
                    "is_known": True,
                }
            elif self.held_identity and tracked:
                # Motion blur: keep the last confirmed ID instead of flickering.
                pass
            else:
                self.held_identity = {
                    "person_id": m.person_id,
                    "person_name": m.person_name,
                    "similarity": round(m.similarity, 3),
                    "is_known": False,
                }

        if self.held_identity:
            result.update(self.held_identity)
            if result.get("is_known"):
                result["approaching"] = False

        return result

    def _coast(self, now: float, frame_w: int, frame_h: int) -> Optional[dict]:
        if self.held_bbox is None or now >= self.held_until:
            self.held_bbox = None
            self.held_identity = None
            self.vx = self.vy = 0.0
            return None
        coasted = predict_bbox(
            self.held_bbox, self.vx, self.vy, self._dt(now), frame_w, frame_h
        )
        self.held_bbox = coasted
        self.last_update = now
        ident = self.held_identity or {}
        return {
            "person_id": ident.get("person_id"),
            "person_name": ident.get("person_name"),
            "similarity": ident.get("similarity", 0.0),
            "is_known": bool(ident.get("is_known")),
            "bbox": coasted,
            "det_score": None,
            "approaching": not bool(ident.get("is_known")),
            "infer_id": self.infer_id,
            "moving": self.speed() >= 70.0,
        }
