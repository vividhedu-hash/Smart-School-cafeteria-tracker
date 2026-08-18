"""
Tests for the model registry.
"""
from __future__ import annotations

import time
import pytest
from cafeteria.training.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models" / "registry.json")


def test_empty_registry(registry):
    assert registry.get_active("waste") is None
    assert registry.active_version_string("waste") == "none"
    assert registry.get_all_versions("waste") == []


def test_register_version(registry, tmp_path):
    weights = tmp_path / "best.pt"
    weights.touch()
    registry.register_version(
        task="waste", version="v001",
        weights_path=str(weights),
        metrics={"top1_accuracy": 0.95},
        config={"epochs": 50},
    )
    versions = registry.get_all_versions("waste")
    assert len(versions) == 1
    assert versions[0]["version"] == "v001"


def test_activate_version(registry, tmp_path):
    weights = tmp_path / "best.pt"
    weights.touch()
    registry.register_version("waste", "v001", str(weights))
    registry.register_version("waste", "v002", str(weights))

    result = registry.activate("waste", "v002")
    assert result is True
    assert registry.active_version_string("waste") == "v002"
    active = registry.get_active("waste")
    assert active["version"] == "v002"


def test_activate_nonexistent_returns_false(registry):
    result = registry.activate("waste", "v999")
    assert result is False


def test_next_version_increments(registry, tmp_path):
    w = tmp_path / "w.pt"
    w.touch()
    assert registry.next_version("waste") == "v001"
    registry.register_version("waste", "v001", str(w))
    assert registry.next_version("waste") == "v002"
    registry.register_version("waste", "v002", str(w))
    assert registry.next_version("waste") == "v003"


def test_registry_persistence(tmp_path):
    path = tmp_path / "models" / "registry.json"
    r1 = ModelRegistry(path)
    w = tmp_path / "w.pt"
    w.touch()
    r1.register_version("plate", "v001", str(w), metrics={"map50": 0.80})
    r1.activate("plate", "v001")

    # Create a new registry from same file — should persist
    r2 = ModelRegistry(path)
    assert r2.active_version_string("plate") == "v001"
    assert r2.get_active_weights("plate") == str(w)
