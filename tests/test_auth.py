"""Operator PIN helpers — env, generated file, verify."""
from __future__ import annotations

import os

from cafeteria.auth import (
    PIN_ENV,
    ensure_operator_pin,
    resolve_operator_pin,
    verify_operator_pin,
)


def test_env_pin_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(PIN_ENV, "246801")
    status = resolve_operator_pin(tmp_path)
    assert status.configured is True
    assert status.source == "env"
    assert status.generated is False
    assert verify_operator_pin("246801", tmp_path) is True
    assert verify_operator_pin("000000", tmp_path) is False


def test_generates_file_pin_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv(PIN_ENV, raising=False)
    status = ensure_operator_pin(tmp_path)
    assert status.configured is True
    assert status.generated is True
    assert status.pin is not None
    assert len(status.pin) == 6
    assert status.path.exists()
    assert oct(status.path.stat().st_mode)[-3:] == "600"
    again = ensure_operator_pin(tmp_path)
    assert again.generated is False
    assert again.pin == status.pin
    assert verify_operator_pin(status.pin, tmp_path) is True
    assert verify_operator_pin("not-it", tmp_path) is False


def test_engine_token_persists(monkeypatch, tmp_path):
    monkeypatch.delenv("CAFETERIA_ENGINE_TOKEN", raising=False)
    from cafeteria.auth import ensure_engine_token
    a = ensure_engine_token(tmp_path)
    b = ensure_engine_token(tmp_path)
    assert a == b
    assert len(a) >= 16
    assert (tmp_path / "data" / ".engine_token").exists()
