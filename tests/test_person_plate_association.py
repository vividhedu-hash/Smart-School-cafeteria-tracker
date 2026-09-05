"""
tests/test_person_plate_association.py — Tests for kinematic person-plate association.
"""
import pytest
import numpy as np

from cafeteria.tracking.motion import KalmanBoxTracker
from cafeteria.tracking.associator import PersonPlateAssociator


def test_person_plate_kinematic_pairing():
    associator = PersonPlateAssociator()

    # Person moving right: cx=200, cy=150, vx=50
    p_track = KalmanBoxTracker(
        track_id=10,
        class_type="person",
        cx=200.0,
        cy=150.0,
        w=80.0,
        h=100.0,
        vx=50.0,
        vy=0.0,
    )

    # Plate moving right with person: cx=205, cy=350, vx=48
    l_track = KalmanBoxTracker(
        track_id=20,
        class_type="plate",
        cx=205.0,
        cy=350.0,
        w=90.0,
        h=70.0,
        vx=48.0,
        vy=0.0,
    )

    # Synthetic frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    sessions = associator.associate([p_track], [l_track], frame=frame)
    assert len(sessions) == 1

    sess = sessions[0]
    assert sess.person_track_id == 10
    assert sess.plate_track_id == 20
    assert sess.frame_count == 1
    assert sess.velocity_x == pytest.approx(49.0)

    # Multi-frame consensus voting
    sess.record_face_match("p01", "Alice", 0.92)
    sess.record_face_match("p01", "Alice", 0.88)
    sess.record_waste_prediction("EMPTY", 0.95)

    name, score = sess.top_identity
    assert name == "Alice"
    assert score == pytest.approx(0.90)

    waste, w_conf = sess.consensus_waste
    assert waste == "EMPTY"
    assert w_conf == pytest.approx(0.95)


def test_rejection_of_invalid_kinematics():
    associator = PersonPlateAssociator()

    # Person at cy=350
    p_track = KalmanBoxTracker(
        track_id=1,
        class_type="person",
        cx=200.0,
        cy=350.0,
        w=80.0,
        h=100.0,
    )
    # Impossible: plate above head at cy=100
    l_track = KalmanBoxTracker(
        track_id=2,
        class_type="plate",
        cx=200.0,
        cy=100.0,
        w=80.0,
        h=60.0,
    )

    sessions = associator.associate([p_track], [l_track])
    # Should reject impossible physical vertical configuration
    assert len(sessions) == 0
