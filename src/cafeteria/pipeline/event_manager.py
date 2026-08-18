"""
Event manager — orchestrates the state machine, detection, and recognition
into the complete waste event pipeline.

This is the brain of the inference engine. It receives frames from the
FrameBuffer and drives the StateMachine through its lifecycle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cafeteria.camera.base import TimestampedFrame
from cafeteria.detection.plate_detector import Detection, PlateDetector
from cafeteria.detection.waste_detector import WasteDetector, WasteResult
from cafeteria.pipeline.state_machine import State, StateMachine
from cafeteria.recognition.face_engine import FaceEngine
from cafeteria.recognition.matcher import EmbeddingMatcher, MatchResult, TemporalVoter
from cafeteria.utils.image_quality import select_best_frame
from cafeteria.utils.logging import get_logger, EventCode, log_event
from cafeteria.utils.timing import Stopwatch

logger = get_logger("pipeline.event_manager")


@dataclass
class EventContext:
    """
    Holds all data accumulated during one waste event.

    Reset on every new event (IDLE → PLATE_DETECTED).
    """
    # Timing
    start_time: float = field(default_factory=time.time)
    event_start_monotonic: float = field(default_factory=time.monotonic)

    # Plate
    plate_detections: list[Detection] = field(default_factory=list)
    best_plate_detection: Optional[Detection] = None

    # Waste
    waste_result: Optional[WasteResult] = None

    # Face
    candidate_frames: list[np.ndarray] = field(default_factory=list)
    face_bboxes: list[Optional[tuple]] = field(default_factory=list)
    face_match: Optional[MatchResult] = None

    # Best evidence frame
    best_frame: Optional[np.ndarray] = None
    best_frame_score: float = 0.0

    # Latency
    stopwatch: Stopwatch = field(default_factory=Stopwatch)

    def event_age(self) -> float:
        return time.monotonic() - self.event_start_monotonic

    def latency_ms(self) -> float:
        return self.stopwatch.elapsed_ms()


class EventManager:
    """
    Drives the complete waste event pipeline.

    Args:
        state_machine:   StateMachine instance.
        plate_detector:  Loaded PlateDetector (or None if not available).
        waste_detector:  Loaded WasteDetector (or None if not available).
        face_engine:     Loaded FaceEngine (or None).
        matcher:         EmbeddingMatcher with loaded embeddings.
        config:          Settings object from config/settings.py.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        plate_detector: Optional[PlateDetector],
        waste_detector: Optional[WasteDetector],
        face_engine: Optional[FaceEngine],
        matcher: EmbeddingMatcher,
        config,
    ) -> None:
        self._sm = state_machine
        self._plate_detector = plate_detector
        self._waste_detector = waste_detector
        self._face_engine = face_engine
        self._matcher = matcher
        self._cfg = config

        self._ctx: Optional[EventContext] = None
        self._voter: Optional[TemporalVoter] = None

        # Duplicate event prevention:
        # After an event completes, the SAME plate lingering in view must not
        # trigger a second transaction. We require the plate to be absent for
        # PLATE_ABSENT_FRAMES_REQUIRED consecutive processed frames before a
        # new event may start.
        self._last_event_time: float = 0.0
        self._plate_first_seen: float = 0.0
        self._awaiting_plate_absence: bool = False
        self._plate_absent_frames: int = 0
        self._metrics = None  # optional MetricsCollector for live face updates
        # Throttle always-on face scan: run at most every N frames to stay ≤300ms
        self._idle_face_frame_count: int = 0

    PLATE_ABSENT_FRAMES_REQUIRED = 3

    @property
    def pipeline_ready(self) -> bool:
        """
        True only when BOTH real models are loaded.

        No trained waste model means the system cannot honestly classify
        waste, so no waste events are started. We never fabricate a
        FOOD_PRESENT result.
        """
        return bool(
            self._plate_detector and self._plate_detector.is_loaded
            and self._waste_detector and self._waste_detector.is_loaded
        )

    def set_metrics(self, metrics) -> None:
        """Attach a MetricsCollector for live face match updates."""
        self._metrics = metrics

    def process_frame(
        self,
        frame: TimestampedFrame,
        recent_frames: list[TimestampedFrame],
    ) -> Optional["CompletedEvent"]:
        """
        Process one frame through the state machine.

        Args:
            frame:         Current TimestampedFrame.
            recent_frames: Recent buffered frames for best-frame selection.

        Returns:
            CompletedEvent if a transaction should be created, else None.
        """
        sm = self._sm
        image = frame.image
        now = time.monotonic()

        # ── COOLDOWN: wait then return to IDLE ─────────────────────────────
        if sm.state == State.COOLDOWN:
            if sm.cooldown_expired():
                sm.transition(State.IDLE)
            return None

        # ── IDLE: gate on readiness, then check for a plate ─────────────────
        if sm.state == State.IDLE:
            if not self.pipeline_ready:
                # The waste pipeline is DISABLED until real plate + waste
                # models are loaded. We never fabricate detections.
                # Live face recognition in this mode is handled by the
                # threaded worker in main.py — never inline here, so the
                # frame loop is never blocked by InsightFace inference.
                return None

            # Both models loaded — normal plate-triggered flow
            _t0 = time.perf_counter()
            detections = self._plate_detector.detect(image)
            if self._metrics:
                self._metrics.update_latencies(
                    plate_ms=(time.perf_counter() - _t0) * 1000.0
                )

            # ── Duplicate prevention: same plate must leave first ──────────
            if self._awaiting_plate_absence:
                if detections:
                    self._plate_absent_frames = 0
                    return None
                self._plate_absent_frames += 1
                if self._plate_absent_frames < self.PLATE_ABSENT_FRAMES_REQUIRED:
                    return None
                self._awaiting_plate_absence = False
                self._plate_absent_frames = 0
                return None  # plate confirmed gone; next frame may start fresh

            if detections:
                # Minimum seconds-since-last-event guard (belt-and-braces on
                # top of the plate-absence hysteresis above).
                min_gap = self._cfg.event.cooldown_seconds
                if (self._last_event_time > 0.0
                        and time.time() - self._last_event_time < min_gap):
                    return None

                self._plate_first_seen = now
                self._ctx = EventContext()
                self._ctx.stopwatch.start()
                sm.transition(State.PLATE_DETECTED)
                log_event(logger, EventCode.PLATE_DETECTED,
                          f"confidence={detections[0].confidence:.2f}")
            return None

        # ── PLATE_DETECTED: verify persistence ─────────────────────────────
        if sm.state == State.PLATE_DETECTED:
            detections = (
                self._plate_detector.detect(image)
                if self._plate_detector and self._plate_detector.is_loaded
                else []
            )

            if not detections:
                # Plate disappeared before minimum presence
                sm.transition(State.IDLE)
                self._ctx = None
                return None

            # Update context
            self._ctx.plate_detections = detections
            self._ctx.best_plate_detection = detections[0]

            # Check minimum presence time
            presence = now - self._plate_first_seen
            min_presence = self._cfg.event.minimum_plate_presence_seconds
            if presence >= min_presence:
                sm.transition(State.FOOD_ANALYSIS)
            return None

        # ── FOOD_ANALYSIS: classify waste on plate crop ─────────────────────
        if sm.state == State.FOOD_ANALYSIS:
            if not self._waste_detector or not self._waste_detector.is_loaded:
                # Should be unreachable (IDLE gates on pipeline_ready), but if
                # the waste model was hot-removed mid-event, abort honestly —
                # we never fabricate a waste result.
                logger.error(
                    "Waste model unavailable mid-event — aborting event "
                    "(no fabricated results)."
                )
                sm.transition(State.ERROR)
                sm.force_idle()
                self._ctx = None
                return None

            det = self._ctx.best_plate_detection
            plate_crop = det.crop(image) if det else image

            try:
                _t0 = time.perf_counter()
                waste_result = self._waste_detector.classify(plate_crop)
                if self._metrics:
                    self._metrics.update_latencies(
                        waste_ms=(time.perf_counter() - _t0) * 1000.0
                    )
            except Exception as exc:
                logger.error("Waste classification error: %s", exc)
                sm.transition(State.ERROR)
                sm.force_idle()
                return None

            self._ctx.waste_result = waste_result

            if waste_result.is_empty:
                log_event(logger, EventCode.FOOD_ABSENT,
                          f"waste={waste_result.label}")
                sm.transition(State.IDLE)
                # Same empty plate must leave the frame before we analyse
                # again — prevents constant re-detection churn.
                self._awaiting_plate_absence = True
                self._plate_absent_frames = 0
                self._ctx = None
                return None

            log_event(logger, EventCode.FOOD_DETECTED,
                      f"waste={waste_result.label}  conf={waste_result.confidence:.2f}")
            sm.transition(State.WASTE_EVENT)
            return None

        # ── WASTE_EVENT: collect candidate frames ────────────────────────────
        if sm.state == State.WASTE_EVENT:
            self._ctx.candidate_frames.append(image.copy())
            self._ctx.face_bboxes.append(None)

            if (not self._face_engine or not self._face_engine.is_loaded
                    or self._matcher.enrolled_count == 0):
                # No face recognition available — go straight to commit
                self._finalize_best_frame(recent_frames)
                sm.transition(State.FACE_RECOGNITION)
                return None

            sm.transition(State.FACE_CAPTURE)
            self._voter = TemporalVoter(
                frames_to_vote=self._cfg.recognition.frames_to_vote,
            )
            return None

        # ── FACE_CAPTURE: gather face frames ────────────────────────────────
        if sm.state == State.FACE_CAPTURE:
            timeout = self._cfg.event.timeout_seconds
            if self._ctx.event_age() > timeout:
                # Timed out — commit with what we have
                log_event(logger, EventCode.FACE_UNKNOWN, "Face capture timed out")
                self._finalize_best_frame(recent_frames)
                sm.transition(State.FACE_RECOGNITION)
                return None

            # Detect face in current frame
            face = None
            if self._face_engine and self._face_engine.is_loaded:
                try:
                    _t0 = time.perf_counter()
                    face = self._face_engine.get_largest_face(
                        image,
                        min_size=self._cfg.recognition.minimum_face_size,
                    )
                    if self._metrics:
                        self._metrics.update_latencies(
                            face_ms=(time.perf_counter() - _t0) * 1000.0
                        )
                except Exception as exc:
                    logger.warning("Face detection error: %s", exc)

            if face:
                self._ctx.candidate_frames.append(image.copy())
                self._ctx.face_bboxes.append(face["bbox"])
                # Vote with this embedding
                match = self._matcher.match(face["embedding"])
                self._voter.add_vote(match)
                log_event(logger, EventCode.FACE_DETECTED,
                          f"det_score={face['det_score']:.2f}  "
                          f"match={match.person_id}  sim={match.similarity:.3f}")

            if self._voter and self._voter.is_ready():
                self._finalize_best_frame(recent_frames)
                sm.transition(State.FACE_RECOGNITION)

            return None

        # ── FACE_RECOGNITION: decide identity ────────────────────────────────
        if sm.state == State.FACE_RECOGNITION:
            if self._voter and len(self._voter._votes) > 0:
                match = self._voter.decide()
            else:
                match = MatchResult(
                    person_id=None, person_name=None,
                    similarity=0.0, is_known=False, candidates=[],
                )

            self._ctx.face_match = match

            if match.is_known:
                log_event(logger, EventCode.FACE_MATCHED,
                          f"person={match.person_id}  sim={match.similarity:.3f}")
                sm.transition(State.TRANSACTION_COMMIT)
            else:
                log_event(logger, EventCode.FACE_UNKNOWN,
                          f"best_sim={match.similarity:.3f}")
                sm.transition(State.REVIEW_REQUIRED)

            # Build completed event
            event = self._build_completed_event()
            sm.transition(State.COOLDOWN)
            self._last_event_time = time.time()
            # Duplicate prevention: the plate must leave the frame before a
            # new event may start (checked in IDLE after cooldown expires).
            self._awaiting_plate_absence = True
            self._plate_absent_frames = 0
            if self._metrics:
                self._metrics.update_latencies(
                    e2e_ms=event.processing_latency_ms
                )
            return event

        return None

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _finalize_best_frame(self, recent_frames: list[TimestampedFrame]) -> None:
        """Select the sharpest frame from candidates + recent buffer."""
        all_frames = [f.image for f in recent_frames] + self._ctx.candidate_frames
        all_bboxes = [None] * len(recent_frames) + self._ctx.face_bboxes

        if all_frames:
            idx, score = select_best_frame(all_frames, all_bboxes)
            self._ctx.best_frame = all_frames[idx]
            self._ctx.best_frame_score = score

    def _build_completed_event(self) -> "CompletedEvent":
        ctx = self._ctx
        sm = self._sm
        match = ctx.face_match

        # Determine final status
        if sm.state in (State.TRANSACTION_COMMIT,):
            if match and match.is_known:
                status = "AUTO_CONFIRMED"
            else:
                status = "REVIEW_REQUIRED"
        elif sm.state == State.REVIEW_REQUIRED:
            status = "REVIEW_REQUIRED"
        else:
            status = "REVIEW_REQUIRED"

        # Proxy plate mode: the detection came from the COCO stand-in model,
        # not a trained plate detector — force review and tag the reason.
        review_reason = None
        if getattr(self._plate_detector, "proxy_mode", False):
            status = "REVIEW_REQUIRED"
            review_reason = "PROXY_PLATE_MODE"

        return CompletedEvent(
            timestamp=ctx.start_time,
            plate_detected=ctx.best_plate_detection is not None,
            plate_confidence=(
                ctx.best_plate_detection.confidence
                if ctx.best_plate_detection else 0.0
            ),
            food_present=True,
            waste_result=ctx.waste_result,
            face_match=match,
            best_frame=ctx.best_frame,
            processing_latency_ms=ctx.latency_ms(),
            status=status,
            review_reason=review_reason,
        )

    def force_reset(self) -> None:
        """Emergency reset — called on camera disconnect or shutdown."""
        self._sm.force_idle()
        self._ctx = None
        self._voter = None
        self._awaiting_plate_absence = False
        self._plate_absent_frames = 0


@dataclass
class CompletedEvent:
    """All data from a completed waste event, ready for transaction creation."""
    timestamp: float
    plate_detected: bool
    plate_confidence: float
    food_present: bool
    waste_result: Optional[WasteResult]
    face_match: Optional[MatchResult]
    best_frame: Optional[np.ndarray]
    processing_latency_ms: float
    status: str  # AUTO_CONFIRMED | REVIEW_REQUIRED
    # Why the event was forced into review (e.g. "PROXY_PLATE_MODE"), if any
    review_reason: Optional[str] = None
