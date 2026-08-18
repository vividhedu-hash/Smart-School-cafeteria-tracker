"""
Explicit state machine for the cafeteria waste detection pipeline.

States and valid transitions are defined here.
The rest of the pipeline drives transitions using transition() — no ad-hoc
boolean flags allowed.
"""
from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

from cafeteria.utils.logging import get_logger, EventCode, log_event

logger = get_logger("pipeline.state_machine")


class InvalidTransition(ValueError):
    """Raised when an illegal state transition is attempted."""
    pass


class State(str, Enum):
    IDLE               = "IDLE"
    PLATE_DETECTED     = "PLATE_DETECTED"
    FOOD_ANALYSIS      = "FOOD_ANALYSIS"
    WASTE_EVENT        = "WASTE_EVENT"
    FACE_CAPTURE       = "FACE_CAPTURE"
    FACE_RECOGNITION   = "FACE_RECOGNITION"
    TRANSACTION_COMMIT = "TRANSACTION_COMMIT"
    REVIEW_REQUIRED    = "REVIEW_REQUIRED"
    COOLDOWN           = "COOLDOWN"
    ERROR              = "ERROR"


# Valid state transitions: {from_state: [allowed_to_states]}
VALID_TRANSITIONS: dict[State, list[State]] = {
    State.IDLE: [
        State.PLATE_DETECTED,
        State.ERROR,
    ],
    State.PLATE_DETECTED: [
        State.FOOD_ANALYSIS,
        State.IDLE,           # plate disappeared / confidence dropped
        State.ERROR,
    ],
    State.FOOD_ANALYSIS: [
        State.WASTE_EVENT,    # food present
        State.IDLE,           # plate empty — no waste event
        State.COOLDOWN,       # empty plate, skip cooldown path
        State.ERROR,
    ],
    State.WASTE_EVENT: [
        State.FACE_CAPTURE,
        State.FACE_RECOGNITION,    # no face engine / no enrollments — decide UNKNOWN
        State.TRANSACTION_COMMIT,  # bypass face if recognition disabled
        State.ERROR,
    ],
    State.FACE_CAPTURE: [
        State.FACE_RECOGNITION,
        State.TRANSACTION_COMMIT,  # timeout — commit with UNKNOWN
        State.ERROR,
    ],
    State.FACE_RECOGNITION: [
        State.TRANSACTION_COMMIT,
        State.REVIEW_REQUIRED,
        State.ERROR,
    ],
    State.TRANSACTION_COMMIT: [
        State.COOLDOWN,
        State.ERROR,
    ],
    State.REVIEW_REQUIRED: [
        State.COOLDOWN,
        State.ERROR,
    ],
    State.COOLDOWN: [
        State.IDLE,
        State.ERROR,
    ],
    State.ERROR: [
        State.IDLE,
    ],
}


class StateMachine:
    """
    Explicit finite state machine for the waste detection pipeline.

    Args:
        cooldown_seconds: Time to stay in COOLDOWN state before returning to IDLE.
    """

    def __init__(self, cooldown_seconds: float = 2.0) -> None:
        self._state: State = State.IDLE
        self._previous: Optional[State] = None
        self._entered_at: float = time.monotonic()
        self._cooldown_seconds = cooldown_seconds
        self._transition_count: int = 0

    @property
    def state(self) -> State:
        return self._state

    @property
    def previous_state(self) -> Optional[State]:
        return self._previous

    @property
    def time_in_state(self) -> float:
        """Seconds spent in the current state."""
        return time.monotonic() - self._entered_at

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def transition(self, new_state: State) -> None:
        """
        Perform a state transition.

        Args:
            new_state: Target state.

        Raises:
            ValueError: If the transition is not valid.
        """
        allowed = VALID_TRANSITIONS.get(self._state, [])
        if new_state not in allowed:
            raise InvalidTransition(
                f"Invalid transition: {self._state.value} → {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        log_event(
            logger,
            EventCode.STATE_TRANSITION,
            f"{self._state.value} → {new_state.value}",
            duration_s=f"{self.time_in_state:.3f}",
        )

        self._previous = self._state
        self._state = new_state
        self._entered_at = time.monotonic()
        self._transition_count += 1

    def force_idle(self) -> None:
        """Force a reset to IDLE (used for error recovery, shutdown)."""
        if self._state != State.IDLE:
            log_event(
                logger,
                EventCode.STATE_TRANSITION,
                f"FORCE RESET: {self._state.value} → IDLE",
            )
        self._previous = self._state
        self._state = State.IDLE
        self._entered_at = time.monotonic()

    def cooldown_expired(self) -> bool:
        """Return True if we've been in COOLDOWN long enough to go back to IDLE."""
        return (
            self._state == State.COOLDOWN
            and self.time_in_state >= self._cooldown_seconds
        )

    def is_idle(self) -> bool:
        return self._state == State.IDLE

    def is_processing(self) -> bool:
        """Return True if the pipeline is actively processing an event."""
        return self._state not in (State.IDLE, State.COOLDOWN, State.ERROR)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation for the runtime state file."""
        return {
            "state": self._state.value,
            "previous_state": self._previous.value if self._previous else None,
            "time_in_state_s": round(self.time_in_state, 3),
            "transition_count": self._transition_count,
        }
