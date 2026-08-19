"""
Enrollment capture policy — forgiving on purpose.

The oval is guidance for the operator, not an acceptance test: a person who
never lines up perfectly must still be able to finish a scan. InsightFace,
not the Haar cascade, decides which stills are usable.
"""
from __future__ import annotations

from enroll_cam import (
    HOLD_SECONDS,
    RELAX_AFTER_SECONDS,
    SNAP_GAP,
    TARGET_FRAMES,
    capture_decision,
)


def test_aligned_face_snaps_after_a_short_hold():
    # [AI-CoLab: Verified by Antigravity] Verified enrollment capture policy unit tests
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=HOLD_SECONDS,
        face_visible=True,
        face_seen_for=0.2,
        since_last_snap=1.0,
    ) == "aligning"


def test_hold_shorter_than_the_threshold_waits():
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=HOLD_SECONDS / 2,
        face_visible=True,
        face_seen_for=0.2,
        since_last_snap=1.0,
    ) is None


def test_face_that_never_lines_up_is_still_captured():
    """The anti-stuck rule: a visible face gets captured once patience runs out."""
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=None,
        face_visible=True,
        face_seen_for=RELAX_AFTER_SECONDS + 0.1,
        since_last_snap=1.0,
    ) == "relaxed"


def test_relaxed_mode_waits_for_a_visible_face():
    """Patience alone is not enough — we never store faceless frames."""
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=None,
        face_visible=False,
        face_seen_for=RELAX_AFTER_SECONDS + 5.0,
        since_last_snap=1.0,
    ) is None


def test_manual_shutter_overrides_everything():
    assert capture_decision(
        manual=True,
        aligned_hold_seconds=None,
        face_visible=False,
        face_seen_for=None,
        since_last_snap=0.0,
    ) == "manual"


def test_burst_gap_is_respected_for_automatic_captures():
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=HOLD_SECONDS * 5,
        face_visible=True,
        face_seen_for=2.0,
        since_last_snap=SNAP_GAP / 2,
    ) is None


def test_no_face_yet_means_no_capture():
    assert capture_decision(
        manual=False,
        aligned_hold_seconds=None,
        face_visible=False,
        face_seen_for=None,
        since_last_snap=5.0,
    ) is None


def test_tuning_stays_gentle():
    """Guard the numbers the harsh version got wrong."""
    assert TARGET_FRAMES <= 6                 # a scan is a couple of seconds
    assert HOLD_SECONDS <= 0.25               # no statue-still requirement
    assert RELAX_AFTER_SECONDS <= 6.0         # patience runs out quickly
