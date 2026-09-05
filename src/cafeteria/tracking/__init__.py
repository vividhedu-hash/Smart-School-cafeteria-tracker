"""
src/cafeteria/tracking — High-performance dynamic multi-object tracking and kinematic association.
"""
from cafeteria.tracking.motion import KalmanBoxTracker, box_iou, box_center_distance
from cafeteria.tracking.multi_tracker import MultiObjectTracker
from cafeteria.tracking.associator import PersonPlateAssociator, PersonPlateSession

__all__ = [
    "KalmanBoxTracker",
    "MultiObjectTracker",
    "PersonPlateAssociator",
    "PersonPlateSession",
    "box_iou",
    "box_center_distance",
]
