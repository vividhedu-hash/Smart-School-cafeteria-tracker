"""
Per-epoch progress extraction from Ultralytics trainer objects.

Attribute names differ per task and version, so every read is defensive —
progress reporting must never be the thing that kills a training run.
"""
from __future__ import annotations

from cafeteria.training.trainer import _attach_progress, _epoch_payload


class _FakeTrainer:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def mean(self):
        return self._value


def test_reads_epoch_and_metrics():
    # [AI-CoLab: Verified by Antigravity] Verified Ultralytics per-epoch payload extraction unit tests
    payload = _epoch_payload(_FakeTrainer(
        epoch=2, epochs=10,
        metrics={"metrics/accuracy_top1": 0.8123456, "fitness": 0.9},
    ))
    assert payload["epoch"] == 3
    assert payload["epochs"] == 10
    assert payload["metrics"]["metrics/accuracy_top1"] == 0.81235


def test_final_validation_pass_does_not_exceed_the_epoch_count():
    """A 2-epoch run must never report 'epoch 3 of 2'."""
    payload = _epoch_payload(_FakeTrainer(epoch=2, epochs=2))
    assert payload["epoch"] == 2


def test_loss_from_tensor_like_and_scalar():
    assert _epoch_payload(_FakeTrainer(tloss=_FakeTensor(0.4321)))["loss"] == 0.4321
    assert _epoch_payload(_FakeTrainer(tloss=1.5))["loss"] == 1.5


def test_unreadable_metrics_are_skipped_not_fatal():
    payload = _epoch_payload(_FakeTrainer(
        epoch=0, epochs=1, metrics={"good": 1.0, "bad": "not-a-number"},
    ))
    assert payload["metrics"] == {"good": 1.0}


def test_bare_trainer_object_yields_a_usable_payload():
    payload = _epoch_payload(_FakeTrainer())
    assert payload["epoch"] == 1
    assert payload["epochs"] == 0


def test_attach_progress_is_a_noop_without_a_callback():
    class _Model:
        def add_callback(self, *_args):
            raise AssertionError("should not register callbacks")

    _attach_progress(_Model(), None)


def test_callback_errors_do_not_propagate_into_training():
    registered = {}

    class _Model:
        def add_callback(self, event, fn):
            registered[event] = fn

    def _explode(_payload):
        raise RuntimeError("reporting bug")

    _attach_progress(_Model(), _explode)
    assert "on_train_epoch_end" in registered
    registered["on_train_epoch_end"](_FakeTrainer(epoch=0, epochs=1))


def test_attach_survives_a_model_that_rejects_callbacks():
    class _Model:
        def add_callback(self, *_args):
            raise TypeError("unsupported")

    _attach_progress(_Model(), lambda _p: None)
