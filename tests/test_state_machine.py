"""
Tests for the state machine.
"""
from __future__ import annotations

import time
import pytest
from cafeteria.pipeline.state_machine import State, StateMachine, InvalidTransition


def make_sm(cooldown=0.1):
    return StateMachine(cooldown_seconds=cooldown)


def test_initial_state():
    sm = make_sm()
    assert sm.state == State.IDLE
    assert sm.is_idle()


def test_valid_transition_sequence():
    sm = make_sm()
    sm.transition(State.PLATE_DETECTED)
    assert sm.state == State.PLATE_DETECTED
    sm.transition(State.FOOD_ANALYSIS)
    assert sm.state == State.FOOD_ANALYSIS
    sm.transition(State.WASTE_EVENT)
    sm.transition(State.FACE_CAPTURE)
    sm.transition(State.FACE_RECOGNITION)
    sm.transition(State.TRANSACTION_COMMIT)
    sm.transition(State.COOLDOWN)
    assert sm.state == State.COOLDOWN


def test_invalid_transition_raises():
    sm = make_sm()
    with pytest.raises(InvalidTransition):
        sm.transition(State.FOOD_ANALYSIS)  # Can't go IDLE → FOOD_ANALYSIS directly


def test_cooldown_expiry():
    sm = make_sm(cooldown=0.05)
    sm.transition(State.PLATE_DETECTED)
    sm.transition(State.FOOD_ANALYSIS)
    sm.transition(State.WASTE_EVENT)
    sm.transition(State.FACE_CAPTURE)
    sm.transition(State.FACE_RECOGNITION)
    sm.transition(State.REVIEW_REQUIRED)
    sm.transition(State.COOLDOWN)

    assert not sm.cooldown_expired()
    time.sleep(0.1)
    assert sm.cooldown_expired()


def test_force_idle():
    sm = make_sm()
    sm.transition(State.PLATE_DETECTED)
    sm.force_idle()
    assert sm.state == State.IDLE


def test_error_recovery():
    sm = make_sm()
    sm.transition(State.PLATE_DETECTED)
    sm.transition(State.ERROR)
    sm.force_idle()
    assert sm.state == State.IDLE


def test_idle_to_plate_detected():
    sm = make_sm()
    sm.transition(State.PLATE_DETECTED)
    assert not sm.is_idle()
    sm.transition(State.IDLE)  # Plate disappeared
    assert sm.is_idle()
