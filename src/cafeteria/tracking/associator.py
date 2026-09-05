"""
src/cafeteria/tracking/associator.py — Spatio-Temporal Kinematic Person-Plate Associator.

Solves the multi-person, multi-plate moving tracking problem:
  - Binds moving people and their plates via horizontal alignment, vertical body hierarchy, and velocity correlation.
  - Maintains persistent PersonPlateSession across movement trajectories.
  - Collects multi-frame recognition votes and sharpest evidence crops.
  - Generates single debounced transaction event upon exit/completion.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from cafeteria.tracking.motion import KalmanBoxTracker


@dataclass
class PersonPlateSession:
    """
    Unified multi-frame session for a moving person carrying a plate.
    """
    session_id: str
    person_track_id: int
    plate_track_id: int

    # Current bounding boxes
    person_bbox: Tuple[int, int, int, int]
    plate_bbox: Tuple[int, int, int, int]

    # Kinematics
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    frame_count: int = 0

    # Trajectory history: list of ((cx_p, cy_p), (cx_l, cy_l), timestamp)
    trajectory: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = field(default_factory=list)

    # Recognition & Analysis
    face_identities: Dict[str, float] = field(default_factory=dict)  # name -> accumulated score
    face_votes: Dict[str, int] = field(default_factory=dict)         # name -> vote count
    waste_predictions: List[Tuple[str, float]] = field(default_factory=list)  # (status, conf)

    # Evidence selection
    best_evidence_frame: Optional[np.ndarray] = None
    best_sharpness: float = 0.0
    best_face_crop: Optional[np.ndarray] = None
    best_plate_crop: Optional[np.ndarray] = None

    # Lifecycle state
    is_committed: bool = False

    def update(
        self,
        person_track: KalmanBoxTracker,
        plate_track: KalmanBoxTracker,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Update session with new track positions and optional video frame."""
        now = time.time()
        self.person_bbox = person_track.to_bbox()
        self.plate_bbox = plate_track.to_bbox()
        self.velocity_x = (person_track.vx + plate_track.vx) / 2.0
        self.velocity_y = (person_track.vy + plate_track.vy) / 2.0
        self.last_updated = now
        self.frame_count += 1

        self.trajectory.append((
            (person_track.cx, person_track.cy),
            (plate_track.cx, plate_track.cy),
            now,
        ))
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)

        # Update evidence frame if current frame is sharp
        if frame is not None and frame.size > 0:
            px1, py1, px2, py2 = [max(0, v) for v in self.plate_bbox]
            fx1, fy1, fx2, fy2 = [max(0, v) for v in self.person_bbox]
            h, w = frame.shape[:2]

            # Estimate sharpness via Laplacian variance on the plate/face ROI
            crop = frame[min(py1, fy1):max(py2, fy2), min(px1, fx1):max(px2, fx2)]
            if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness > self.best_sharpness:
                    self.best_sharpness = sharpness
                    self.best_evidence_frame = frame.copy()
                    if fy2 <= h and fx2 <= w:
                        self.best_face_crop = frame[fy1:fy2, fx1:fx2].copy()
                    if py2 <= h and px2 <= w:
                        self.best_plate_crop = frame[py1:py2, px1:px2].copy()

    def record_face_match(self, person_id: str, person_name: str, score: float) -> None:
        """Accumulate multi-frame identity hypotheses."""
        self.face_identities[person_name] = self.face_identities.get(person_name, 0.0) + score
        self.face_votes[person_name] = self.face_votes.get(person_name, 0) + 1

    def record_waste_prediction(self, status: str, confidence: float) -> None:
        """Accumulate waste classification prediction."""
        self.waste_predictions.append((status, confidence))

    @property
    def top_identity(self) -> Tuple[Optional[str], float]:
        """Return the highest confidence identity from rolling votes."""
        if not self.face_votes:
            return None, 0.0
        best_name = max(self.face_votes.keys(), key=lambda n: self.face_identities[n])
        total_votes = self.face_votes[best_name]
        avg_score = self.face_identities[best_name] / max(1, total_votes)
        return best_name, avg_score

    @property
    def consensus_waste(self) -> Tuple[str, float]:
        """Return consensus waste status across the trajectory."""
        if not self.waste_predictions:
            return "UNKNOWN", 0.0
        counts: Dict[str, float] = {}
        for status, conf in self.waste_predictions:
            counts[status] = counts.get(status, 0.0) + conf
        best_status = max(counts.keys(), key=lambda s: counts[s])
        avg_conf = counts[best_status] / sum(1 for s, _ in self.waste_predictions if s == best_status)
        return best_status, avg_conf


class PersonPlateAssociator:
    """
    Spatio-Temporal Kinematic Association Engine for moving people and their plates.
    """

    def __init__(
        self,
        max_horizontal_offset_ratio: float = 0.50,  # Max allowable horizontal offset relative to person width
        min_vertical_separation: float = 10.0,       # Plate must be below face
        velocity_coherence_weight: float = 0.35,     # Weight given to matching motion vectors
        session_timeout_seconds: float = 2.5,        # Age before unobserved session expires
    ) -> None:
        self.max_h_offset = max_horizontal_offset_ratio
        self.min_v_sep = min_vertical_separation
        self.vel_weight = velocity_coherence_weight
        self.timeout_s = session_timeout_seconds
        self.sessions: Dict[str, PersonPlateSession] = {}
        self._session_counter: int = 1

    def associate(
        self,
        person_tracks: List[KalmanBoxTracker],
        plate_tracks: List[KalmanBoxTracker],
        frame: Optional[np.ndarray] = None,
    ) -> List[PersonPlateSession]:
        """
        Associate active person tracks with active plate tracks.

        Returns:
            List of active PersonPlateSession instances.
        """
        now = time.time()
        used_persons = set()
        used_plates = set()

        # 1. Update existing sessions if their constituent tracks are still active
        active_sessions: List[PersonPlateSession] = []
        for sid, sess in list(self.sessions.items()):
            p_trk = next((p for p in person_tracks if p.track_id == sess.person_track_id), None)
            l_trk = next((l for l in plate_tracks if l.track_id == sess.plate_track_id), None)

            if p_trk is not None and l_trk is not None:
                sess.update(p_trk, l_trk, frame=frame)
                used_persons.add(p_trk.track_id)
                used_plates.add(l_trk.track_id)
                active_sessions.append(sess)
            elif now - sess.last_updated > self.timeout_s:
                # Expired session
                del self.sessions[sid]

        # 2. Pair unmatched person tracks with unmatched plate tracks
        unmatched_persons = [p for p in person_tracks if p.track_id not in used_persons]
        unmatched_plates = [l for l in plate_tracks if l.track_id not in used_plates]

        for p in unmatched_persons:
            best_plate: Optional[KalmanBoxTracker] = None
            lowest_cost = float("inf")

            for l in unmatched_plates:
                if l.track_id in used_plates:
                    continue

                cost = self._compute_pairing_cost(p, l)
                if cost < lowest_cost and cost < 1.0:
                    lowest_cost = cost
                    best_plate = l

            if best_plate is not None:
                sid = f"SES-{self._session_counter:05d}"
                self._session_counter += 1
                new_session = PersonPlateSession(
                    session_id=sid,
                    person_track_id=p.track_id,
                    plate_track_id=best_plate.track_id,
                    person_bbox=p.to_bbox(),
                    plate_bbox=best_plate.to_bbox(),
                )
                new_session.update(p, best_plate, frame=frame)
                self.sessions[sid] = new_session
                used_plates.add(best_plate.track_id)
                active_sessions.append(new_session)

        return active_sessions

    def _compute_pairing_cost(self, p: KalmanBoxTracker, l: KalmanBoxTracker) -> float:
        """
        Multi-factor cost function to bind person to plate. Lower is better (< 1.0 is a match).
        """
        # Constraint 1: Vertical order — Face must be higher than plate (y is downwards in image coords)
        if l.cy <= p.cy + self.min_v_sep:
            return 999.0  # Physical violation: plate cannot be above head

        # Factor 1: Horizontal alignment (person holds plate centered with body)
        dx = abs(l.cx - p.cx)
        allowable_dx = max(p.w * (1.0 + self.max_h_offset), 80.0)
        h_cost = dx / allowable_dx
        if h_cost > 1.5:
            return 999.0

        # Factor 2: Vertical distance consistency (torso length bounds)
        dy = l.cy - p.cy
        # In typical view, plate is 0.8x to 3.5x face heights below face
        ideal_dy = p.h * 1.8
        v_cost = abs(dy - ideal_dy) / max(p.h * 2.5, 100.0)

        # Factor 3: Velocity vector coherence
        vel_cost = 0.0
        p_speed = p.speed
        l_speed = l.speed
        if p_speed > 30.0 and l_speed > 30.0:
            # Velocity cosine alignment
            dot = (p.vx * l.vx + p.vy * l.vy)
            mag = (p_speed * l_speed)
            cos_sim = dot / max(1e-5, mag)
            # If moving together, cos_sim ~ 1.0 -> 0 penalty; if opposite, cos_sim ~ -1.0 -> high penalty
            vel_cost = max(0.0, 1.0 - cos_sim)

        total_cost = (0.50 * h_cost) + (0.25 * v_cost) + (self.vel_weight * vel_cost)
        return float(total_cost)
