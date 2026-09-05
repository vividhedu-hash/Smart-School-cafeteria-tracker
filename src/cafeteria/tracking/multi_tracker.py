"""
src/cafeteria/tracking/multi_tracker.py — High-rate Multi-Object Tracker (MOT).

Maintains active tracks for moving people and plates simultaneously.
Handles motion prediction, two-tier association, track coasting, and birth/death lifecycle.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from cafeteria.tracking.motion import KalmanBoxTracker, box_iou, box_center_distance


class MultiObjectTracker:
    """
    Real-time Multi-Object Tracker (SORT/ByteTrack design).

    Args:
        max_age: Maximum frames to coast an unobserved track before terminating.
        min_hits: Minimum hits before a track is considered confirmed.
        iou_threshold: Minimum IoU overlap for matching.
        dist_threshold: Maximum pixel distance for recovery matching when IoU is low (motion blur).
    """

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 2,
        iou_threshold: float = 0.20,
        dist_threshold: float = 120.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self._next_id: int = 1

    def step(
        self,
        detections: Sequence[Tuple[Sequence[float], float, str]],
        dt: float = 1.0,
    ) -> List[KalmanBoxTracker]:
        """
        Advance tracker by one frame.

        Args:
            detections: List of ((x1, y1, x2, y2), confidence, class_type)
            dt: Elapsed time in seconds or frame normalized time.

        Returns:
            List of confirmed active trackers.
        """
        # 1. Predict new positions for all existing trackers
        for trk in self.trackers:
            trk.predict(dt=dt)

        # 2. Separate detections by class_type ("person", "plate")
        # To ensure plates don't match people and vice versa
        by_class_dets: dict[str, list[Tuple[Sequence[float], float]]] = {}
        for bbox, conf, ctype in detections:
            by_class_dets.setdefault(ctype, []).append((bbox, conf))

        # We will collect updated and new trackers
        matched_trackers: set[int] = set()

        for ctype, dets in by_class_dets.items():
            class_trackers = [t for t in self.trackers if t.class_type == ctype]
            if not class_trackers or not dets:
                continue

            # Compute cost / distance matrix (IoU + distance fallback)
            cost_matrix = np.zeros((len(class_trackers), len(dets)), dtype=np.float32)
            for i, trk in enumerate(class_trackers):
                t_box = trk.to_bbox()
                for j, (d_box, _) in enumerate(dets):
                    iou = box_iou(t_box, d_box)
                    if iou >= self.iou_threshold:
                        cost_matrix[i, j] = 1.0 - iou
                    else:
                        dist = box_center_distance(t_box, d_box)
                        # Distance normalized penalty
                        if dist <= self.dist_threshold:
                            cost_matrix[i, j] = 1.0 + (dist / self.dist_threshold)
                        else:
                            cost_matrix[i, j] = 10.0  # Unmatchable

            # Greedy bipartite matching (fast, robust for N < 50 objects)
            used_trks = set()
            used_dets = set()

            # Flatten and sort matches by lowest cost
            matches = []
            for i in range(len(class_trackers)):
                for j in range(len(dets)):
                    if cost_matrix[i, j] < 2.0:
                        matches.append((cost_matrix[i, j], i, j))
            matches.sort(key=lambda x: x[0])

            for cost, trk_idx, det_idx in matches:
                if trk_idx in used_trks or det_idx in used_dets:
                    continue
                trk = class_trackers[trk_idx]
                d_box, d_conf = dets[det_idx]
                trk.update(d_box, d_conf, dt=dt)
                used_trks.add(trk_idx)
                used_dets.add(det_idx)
                matched_trackers.add(trk.track_id)

            # Unmatched detections become new tracks
            for j, (d_box, d_conf) in enumerate(dets):
                if j not in used_dets:
                    x1, y1, x2, y2 = [float(v) for v in d_box[:4]]
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    new_trk = KalmanBoxTracker(
                        track_id=self._next_id,
                        class_type=ctype,
                        cx=x1 + w / 2.0,
                        cy=y1 + h / 2.0,
                        w=w,
                        h=h,
                        confidence=d_conf,
                    )
                    self._next_id += 1
                    self.trackers.append(new_trk)

        # Handle any detection classes that had no existing trackers
        for ctype, dets in by_class_dets.items():
            existing = [t for t in self.trackers if t.class_type == ctype]
            if not existing:
                for d_box, d_conf in dets:
                    x1, y1, x2, y2 = [float(v) for v in d_box[:4]]
                    w = max(1.0, x2 - x1)
                    h = max(1.0, y2 - y1)
                    new_trk = KalmanBoxTracker(
                        track_id=self._next_id,
                        class_type=ctype,
                        cx=x1 + w / 2.0,
                        cy=y1 + h / 2.0,
                        w=w,
                        h=h,
                        confidence=d_conf,
                    )
                    self._next_id += 1
                    self.trackers.append(new_trk)

        # 3. Clean up dead trackers exceeding max_age
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 4. Return confirmed trackers (or newly confirmed)
        confirmed = [
            t for t in self.trackers
            if (t.hits >= self.min_hits or t.hit_streak >= 1) and t.time_since_update <= 2
        ]
        return confirmed

    def get_tracks_by_class(self, class_type: str) -> List[KalmanBoxTracker]:
        """Return all active confirmed tracks of a specific class."""
        return [t for t in self.trackers if t.class_type == class_type and t.time_since_update <= 2]
