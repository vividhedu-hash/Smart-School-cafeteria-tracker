"""
src/cafeteria/tracking/motion.py — High-performance 2D Kinematic State Estimator.

Maintains track state [cx, cy, w, h, vx, vy] with constant-velocity prediction,
measurement correction, and velocity clamping. Sub-millisecond execution, zero allocations in hot path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass
class KalmanBoxTracker:
    """
    2D constant-velocity kinematic Kalman filter for a single bounding box.
    State vector: [cx, cy, w, h, vx, vy]^T
    """
    track_id: int
    class_type: str  # "person" or "plate"
    cx: float
    cy: float
    w: float
    h: float
    vx: float = 0.0
    vy: float = 0.0
    time_since_update: int = 0
    hits: int = 1
    hit_streak: int = 1
    age: int = 1
    confidence: float = 1.0

    # Covariance diagonals (simplified fast diagonal formulation for ultra-low latency)
    _pos_var: float = 10.0
    _vel_var: float = 100.0

    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return (
            int(round(self.cx - half_w)),
            int(round(self.cy - half_h)),
            int(round(self.cx + half_w)),
            int(round(self.cy + half_h)),
        )

    def predict(self, dt: float = 1.0) -> Tuple[int, int, int, int]:
        """
        Advance state by dt according to constant-velocity model.
        """
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        self.age += 1
        self.time_since_update += 1
        self.hit_streak = 0
        # Expand uncertainty during coasting
        self._pos_var += 2.0 * dt
        return self.to_bbox()

    def update(
        self,
        bbox: Sequence[float],
        confidence: float,
        dt: float = 1.0,
        alpha_pos: float = 0.70,
        alpha_vel: float = 0.60,
    ) -> None:
        """
        Update state with an observed measurement [x1, y1, x2, y2].
        Applies adaptive Kalman/EMA smoothing based on speed.
        """
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        meas_w = max(1.0, x2 - x1)
        meas_h = max(1.0, y2 - y1)
        meas_cx = x1 + meas_w / 2.0
        meas_cy = y1 + meas_h / 2.0

        dt = max(1e-3, float(dt))
        # Instantaneous velocity
        inst_vx = (meas_cx - self.cx) / dt
        inst_vy = (meas_cy - self.cy) / dt

        # Clamp realistic human walking speeds (pixels/sec)
        inst_vx = max(-1200.0, min(1200.0, inst_vx))
        inst_vy = max(-1200.0, min(1200.0, inst_vy))

        # Speed-adaptive gain: trust observation more if moving briskly
        speed = math.hypot(inst_vx, inst_vy)
        if speed > 150.0:
            eff_alpha_pos = min(0.90, alpha_pos + 0.15)
            eff_alpha_vel = min(0.85, alpha_vel + 0.15)
        else:
            eff_alpha_pos = alpha_pos
            eff_alpha_vel = alpha_vel

        # Measurement correction
        self.cx = eff_alpha_pos * meas_cx + (1.0 - eff_alpha_pos) * self.cx
        self.cy = eff_alpha_pos * meas_cy + (1.0 - eff_alpha_pos) * self.cy
        self.w  = eff_alpha_pos * meas_w  + (1.0 - eff_alpha_pos) * self.w
        self.h  = eff_alpha_pos * meas_h  + (1.0 - eff_alpha_pos) * self.h

        # Velocity update
        self.vx = eff_alpha_vel * inst_vx + (1.0 - eff_alpha_vel) * self.vx
        self.vy = eff_alpha_vel * inst_vy + (1.0 - eff_alpha_vel) * self.vy

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = float(confidence)
        self._pos_var = 10.0

    @property
    def speed(self) -> float:
        """Magnitude of velocity vector in pixels per unit time."""
        return float(math.hypot(self.vx, self.vy))


def box_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    if interArea <= 0.0:
        return 0.0

    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea / (boxAArea + boxBArea - interArea))


def box_center_distance(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    """Euclidean distance between bounding box centroids."""
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0
    return float(math.hypot(cxA - cxB, cyA - cyB))
