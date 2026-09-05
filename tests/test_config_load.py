"""Config YAML is actually loaded — not silent CWD pydantic defaults."""
from __future__ import annotations

import yaml

from cafeteria.config.settings import load_settings, reset_settings_cache


def test_load_settings_reads_project_yaml_not_cwd_defaults(tmp_path, monkeypatch):
    """Even from an empty CWD, the package-relative configs/config.yaml is used."""
    old_cache = None
    import cafeteria.config.settings as settings_mod
    old_cache = settings_mod._settings_cache
    reset_settings_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CAFETERIA_CONFIG", raising=False)
    try:
        cfg = load_settings()
        assert cfg.config_source is not None
        assert cfg.config_source.name == "config.yaml"
        assert "configs" in cfg.config_source.as_posix()
        # Values that differ from pydantic defaults in settings.py
        assert cfg.recognition.model_pack == "buffalo_s"
        assert list(cfg.recognition.det_size) == [320, 320]
        assert cfg.recognition.similarity_threshold == 0.52
        assert cfg.recognition.minimum_face_size == 24
        assert cfg.recognition.identify_face_size == 40
        assert cfg.recognition.bbox_hold_seconds == 0.90
        assert cfg.recognition.motion_pad_ratio == 0.90
        assert cfg.recognition.det_thresh == 0.40
        assert cfg.application.api_port == 8765
        assert cfg.camera.width == 0
        assert cfg.camera.height == 0
        assert cfg.camera.source == "auto"
        assert cfg.camera.max_capture_width == 1920
        assert cfg.models.plate.allow_coco_fallback is False
        assert cfg.models.plate.allow_visual_fallback is True
        assert cfg.models.waste.allow_visual_fallback is True
        assert cfg.inference.frame_skip == 2
        assert cfg.inference.debug_window is False
        assert cfg.event.cooldown_seconds == 2.0
    finally:
        settings_mod._settings_cache = old_cache


def test_explicit_config_path_is_honoured(tmp_path):
    import cafeteria.config.settings as settings_mod

    old = settings_mod._settings_cache
    cfg_file = tmp_path / "configs" / "config.yaml"
    cfg_file.parent.mkdir(parents=True)
    yaml.dump({
        "recognition": {"model_pack": "buffalo_s", "det_size": [320, 320]},
        "camera": {"width": 640, "height": 480},
        "models": {"plate": {"weights": "models/plate/best.pt", "allow_coco_fallback": False}},
    }, cfg_file.open("w"))
    reset_settings_cache()
    try:
        cfg = load_settings(config_path=cfg_file)
        assert cfg.camera.width == 640
        assert cfg.recognition.model_pack == "buffalo_s"
        assert cfg.config_source == cfg_file.resolve()
    finally:
        settings_mod._settings_cache = old
