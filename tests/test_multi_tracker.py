"""
tests/test_multi_tracker.py — Tests for multi-target tracking with motion blur coasting.
"""
import pytest
from cafeteria.tracking.multi_tracker import MultiObjectTracker


def test_multi_tracker_creation_and_coasting():
    mot = MultiObjectTracker(max_age=5, min_hits=1)

    # Frame 1: Person at (100, 100), Plate at (110, 250)
    dets_f1 = [
        ((70, 60, 130, 140), 0.90, "person"),
        ((80, 220, 140, 280), 0.85, "plate"),
    ]
    tracks = mot.step(dets_f1, dt=0.1)
    assert len(tracks) == 2
    assert len(mot.get_tracks_by_class("person")) == 1
    assert len(mot.get_tracks_by_class("plate")) == 1

    person_id = mot.get_tracks_by_class("person")[0].track_id
    plate_id = mot.get_tracks_by_class("plate")[0].track_id

    # Frame 2: Objects moved right by 10 pixels
    dets_f2 = [
        ((80, 60, 140, 140), 0.92, "person"),
        ((90, 220, 150, 280), 0.88, "plate"),
    ]
    tracks2 = mot.step(dets_f2, dt=0.1)
    assert len(tracks2) == 2

    # Track IDs should be preserved
    p_track = mot.get_tracks_by_class("person")[0]
    l_track = mot.get_tracks_by_class("plate")[0]
    assert p_track.track_id == person_id
    assert l_track.track_id == plate_id
    assert p_track.hits == 2
    assert p_track.vx > 0.0  # moving right!

    # Frame 3: Temporary detection drop (e.g. motion blur)
    # Tracker should coast without losing track
    mot.step([], dt=0.1)
    # Both trackers should still be retained in memory
    assert len(mot.trackers) == 2
    assert mot.trackers[0].time_since_update == 1
