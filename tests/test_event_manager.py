"""
Integration tests for the EventManager pipeline.

These use stub detectors (test-only — the runtime path never uses stubs)
to drive the real StateMachine + EventManager through complete waste events
and verify the honesty gates, duplicate prevention, and empty-plate handling.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from cafeteria.camera.base import TimestampedFrame
from cafeteria.detection.plate_detector import Detection
from cafeteria.detection.waste_detector import WasteResult
from cafeteria.pipeline.event_manager import EventManager
from cafeteria.pipeline.state_machine import State, StateMachine
from cafeteria.recognition.matcher import EmbeddingMatcher


# ──────────────────────────────────────────────────────────────────────────────
# Test stubs (tests only — never in the runtime path)
# ──────────────────────────────────────────────────────────────────────────────

class StubPlateDetector:
    """Deterministic plate detector: detects when told to."""

    def __init__(self):
        self.is_loaded = True
        self.plate_present = True
        self.call_count = 0

    def detect(self, image):
        self.call_count += 1
        if not self.plate_present:
            return []
        return [Detection(label="plate", confidence=0.91,
                          x1=100, y1=100, x2=400, y2=400)]


class StubWasteDetector:
    """Deterministic waste classifier."""

    def __init__(self, label="HIGH_WASTE", confidence=0.88):
        self.is_loaded = True
        self.label = label
        self.confidence = confidence

    def classify(self, plate_crop):
        return WasteResult(
            label=self.label,
            confidence=self.confidence,
            all_scores={self.label: self.confidence},
            is_waste=self.label != "EMPTY",
        )


def make_config():
    """Minimal config stub for EventManager."""
    return SimpleNamespace(
        event=SimpleNamespace(
            minimum_plate_presence_seconds=0.0,
            cooldown_seconds=0.05,
            timeout_seconds=1.0,
            best_frame_window=3,
        ),
        recognition=SimpleNamespace(
            frames_to_vote=2,
            minimum_face_size=40,
            similarity_threshold=0.52,
        ),
    )


def make_frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return TimestampedFrame(
        frame_id=1,
        timestamp=time.monotonic(),
        wall_time=time.time(),
        image=img,
        camera_id="test",
    )


def make_manager(plate=None, waste=None, tmp_path=None):
    sm = StateMachine(cooldown_seconds=0.05)
    matcher = EmbeddingMatcher(
        enrollment_dir=tmp_path or "/tmp/nonexistent_enrollment",
        similarity_threshold=0.45,
    )
    matcher.load_embeddings()  # 0 persons — face path bypassed
    em = EventManager(
        state_machine=sm,
        plate_detector=plate,
        waste_detector=waste,
        face_engine=None,
        matcher=matcher,
        config=make_config(),
    )
    return em, sm


def run_until_event(em, max_frames=50):
    """Pump frames through the pipeline until an event completes or limit hit."""
    for _ in range(max_frames):
        event = em.process_frame(make_frame(), [make_frame()])
        if event is not None:
            return event
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Honesty gates
# ──────────────────────────────────────────────────────────────────────────────

def test_no_models_means_no_pipeline(tmp_path):
    """Without both models, the pipeline must never start an event."""
    em, sm = make_manager(plate=None, waste=None, tmp_path=tmp_path)
    for _ in range(10):
        assert em.process_frame(make_frame(), []) is None
    assert sm.state == State.IDLE


def test_plate_only_is_not_enough(tmp_path):
    """Plate model without waste model must NOT start events (no fake FOOD_PRESENT)."""
    em, sm = make_manager(plate=StubPlateDetector(), waste=None, tmp_path=tmp_path)
    for _ in range(10):
        assert em.process_frame(make_frame(), []) is None
    assert sm.state == State.IDLE
    assert not em.pipeline_ready


def test_waste_only_is_not_enough(tmp_path):
    em, sm = make_manager(plate=None, waste=StubWasteDetector(), tmp_path=tmp_path)
    for _ in range(10):
        assert em.process_frame(make_frame(), []) is None
    assert sm.state == State.IDLE
    assert not em.pipeline_ready


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline (no face engine → UNKNOWN → REVIEW_REQUIRED)
# ──────────────────────────────────────────────────────────────────────────────

def test_full_event_completes_without_face_engine(tmp_path):
    """Plate + food + no face engine → exactly one REVIEW_REQUIRED event.

    Also regression-tests the WASTE_EVENT → FACE_RECOGNITION transition,
    which previously raised InvalidTransition and crashed the engine.
    """
    plate = StubPlateDetector()
    waste = StubWasteDetector(label="MEDIUM_WASTE", confidence=0.83)
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    event = run_until_event(em)
    assert event is not None, "Pipeline never completed an event"
    assert event.waste_result.label == "MEDIUM_WASTE"
    assert event.waste_result.confidence == pytest.approx(0.83)
    assert event.plate_detected is True
    assert event.plate_confidence == pytest.approx(0.91)
    assert event.status == "REVIEW_REQUIRED"  # no face → review
    assert sm.state == State.COOLDOWN


def test_empty_plate_creates_no_event(tmp_path):
    """EMPTY classification must return to IDLE without an event."""
    plate = StubPlateDetector()
    waste = StubWasteDetector(label="EMPTY", confidence=0.95)
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    event = run_until_event(em, max_frames=20)
    assert event is None
    assert sm.state == State.IDLE


def test_no_plate_stays_idle(tmp_path):
    plate = StubPlateDetector()
    plate.plate_present = False
    em, sm = make_manager(plate=plate, waste=StubWasteDetector(), tmp_path=tmp_path)

    event = run_until_event(em, max_frames=20)
    assert event is None
    assert sm.state == State.IDLE


def test_plate_disappearing_early_aborts_event(tmp_path):
    """Plate seen once then gone → back to IDLE, no event."""
    plate = StubPlateDetector()
    waste = StubWasteDetector()
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    # Frame 1: plate appears → PLATE_DETECTED
    em.process_frame(make_frame(), [])
    assert sm.state == State.PLATE_DETECTED

    # Plate vanishes before persistence confirmed
    plate.plate_present = False
    em.process_frame(make_frame(), [])
    assert sm.state == State.IDLE


# ──────────────────────────────────────────────────────────────────────────────
# Duplicate prevention
# ──────────────────────────────────────────────────────────────────────────────

def test_same_plate_does_not_create_second_event(tmp_path):
    """A plate that stays in view after an event must NOT retrigger."""
    plate = StubPlateDetector()
    waste = StubWasteDetector()
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    first = run_until_event(em)
    assert first is not None

    # Wait out cooldown, keep the same plate in view
    time.sleep(0.1)
    for _ in range(30):
        event = em.process_frame(make_frame(), [])
        assert event is None, "Duplicate event created for the same plate!"


def test_new_event_allowed_after_plate_leaves(tmp_path):
    """After the plate leaves for enough frames, a new plate starts a new event."""
    plate = StubPlateDetector()
    waste = StubWasteDetector()
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    first = run_until_event(em)
    assert first is not None

    time.sleep(0.1)  # cooldown expiry

    # Plate leaves for the required absent frames
    plate.plate_present = False
    for _ in range(EventManager.PLATE_ABSENT_FRAMES_REQUIRED + 2):
        em.process_frame(make_frame(), [])

    # New plate arrives → second event allowed
    plate.plate_present = True
    second = run_until_event(em)
    assert second is not None, "Second event blocked after plate absence"


def test_force_reset_clears_duplicate_guard(tmp_path):
    plate = StubPlateDetector()
    waste = StubWasteDetector()
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)

    run_until_event(em)
    em.force_reset()
    assert sm.state == State.IDLE
    assert em._awaiting_plate_absence is False


class BoomFaceEngine:
    is_loaded = True

    def get_largest_face(self, *args, **kwargs):
        raise AssertionError("main thread must not run InsightFace")


def test_live_tracker_identity_commits_without_inline_face(tmp_path):
    """Always-on lock is enough — do not stall 4s on a second InsightFace pass."""
    plate = StubPlateDetector()
    waste = StubWasteDetector(label="HIGH_WASTE", confidence=0.9)
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)
    em._face_engine = BoomFaceEngine()
    em._matcher._embeddings["person_01"] = np.zeros(512)
    em._matcher._names["person_01"] = "Ada"
    live = {
        "person_id": "person_01",
        "person_name": "Ada",
        "similarity": 0.81,
        "is_known": True,
        "bbox": [20, 20, 90, 90],
        "approaching": False,
        "infer_id": 4,
    }
    event = None
    for _ in range(40):
        event = em.process_frame(make_frame(), [make_frame()], live_face=live)
        if event is not None:
            break
    assert event is not None
    assert event.face_match is not None
    assert event.face_match.is_known is True
    assert event.face_match.person_id == "person_01"
    assert event.status == "AUTO_CONFIRMED"
