"""Training status file — the UI's only view of a background training run."""
from __future__ import annotations

import time

from cafeteria.training.progress import (
    clear_status,
    is_running,
    read_status,
    status_path,
    update_status,
    write_status,
)


def test_status_path_lives_under_data(tmp_path):
    # [AI-CoLab: Verified by Antigravity] Verified background training progress IPC unit tests
    assert status_path(tmp_path) == tmp_path / "data" / "training_status.json"


def test_missing_status_reads_as_empty(tmp_path):
    assert read_status(tmp_path / "nope.json") == {}


def test_write_then_read_roundtrip(tmp_path):
    path = tmp_path / "data" / "training_status.json"
    write_status(path, state="running", task="waste", epoch=1, epochs=10)
    status = read_status(path)
    assert status["state"] == "running"
    assert status["task"] == "waste"
    assert status["epoch"] == 1
    assert "updated_at" in status


def test_update_merges_without_losing_fields(tmp_path):
    path = tmp_path / "data" / "training_status.json"
    write_status(path, state="running", task="waste", version="waste_v1", history=[{"epoch": 1}])
    update_status(path, epoch=2, history=[{"epoch": 1}, {"epoch": 2}])
    status = read_status(path)
    assert status["version"] == "waste_v1"
    assert status["epoch"] == 2
    assert len(status["history"]) == 2


def test_clear_status_removes_file(tmp_path):
    path = tmp_path / "data" / "training_status.json"
    write_status(path, state="done")
    clear_status(path)
    assert read_status(path) == {}


def test_is_running_only_for_fresh_running_status():
    assert is_running({"state": "running", "updated_at": time.time()}) is True
    assert is_running({"state": "done", "updated_at": time.time()}) is False
    assert is_running({}) is False
    assert is_running(None) is False


def test_stale_running_status_does_not_lock_the_ui():
    """A killed training process must not disable the start button forever."""
    stale = {"state": "running", "updated_at": time.time() - 5000}
    assert is_running(stale) is False


def test_corrupt_status_file_reads_as_empty(tmp_path):
    path = tmp_path / "training_status.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_status(path) == {}
