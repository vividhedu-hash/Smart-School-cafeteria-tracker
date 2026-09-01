"""Camera auto-index and native-resolution ranking — no hardware required."""
from __future__ import annotations

from cafeteria.camera.negotiate import (
    candidate_indices,
    cap_to_max,
    is_auto_size,
    is_auto_source,
    pick_verified_size,
    rank_capture_sizes,
)
from cafeteria.config.settings import CameraSettings


def test_auto_source_and_indices():
    assert is_auto_source("auto")
    assert is_auto_source("AUTO")
    assert is_auto_source(-1)
    assert is_auto_source(None)
    assert not is_auto_source(0)
    assert candidate_indices("auto") == [0, 1, 2, 3]
    assert candidate_indices(1)[0] == 1
    assert 0 in candidate_indices(1)


def test_auto_size():
    assert is_auto_size(0, 0)
    assert is_auto_size("auto", "auto")
    assert not is_auto_size(1280, 720)


def test_rank_auto_uses_native_capped_at_1080p():
    sizes = rank_capture_sizes(
        native_width=3840,
        native_height=2160,
        requested_width=0,
        requested_height=0,
        max_capture_width=1920,
    )
    assert sizes[0] == (1920, 1080)
    assert (1280, 720) in sizes
    assert all(w <= 1920 for w, _h in sizes)


def test_rank_explicit_request_comes_first():
    sizes = rank_capture_sizes(
        native_width=1920,
        native_height=1080,
        requested_width=1280,
        requested_height=720,
    )
    assert sizes[0] == (1280, 720)


def test_cap_to_max_keeps_aspect():
    assert cap_to_max(3840, 2160, 1920) == (1920, 1080)


def test_pick_verified_prefers_largest_under_cap():
    chosen = pick_verified_size(
        [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)],
        requested_width=0,
        requested_height=0,
        max_capture_width=1920,
    )
    assert chosen == (1920, 1080)


def test_camera_settings_coerce_auto_from_yaml_values():
    cfg = CameraSettings.model_validate({
        "source": "auto",
        "width": "auto",
        "height": "auto",
        "max_capture_width": 1920,
    })
    assert cfg.source == "auto"
    assert cfg.width == 0
    assert cfg.height == 0
    assert cfg.max_capture_width == 1920
