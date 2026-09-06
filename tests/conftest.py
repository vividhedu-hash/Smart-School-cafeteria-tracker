"""
Test configuration and shared fixtures.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Isolate BLAS/OpenMP before numpy is imported — on some macOS versions
# `pytest tests/` otherwise hits a numpy FPE during collection.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")

import pytest

# Ensure src/ and dashboard/ are importable
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "dashboard"))

# Override project root for tests
os.environ["CAFETERIA_CONFIG"] = str(_project_root / "configs" / "config.yaml")
os.environ.pop("DATABASE_URL", None)


def pytest_ignore_collect(collection_path, config):
    """Never collect third-party tests (numpy/scipy in .venv crash collection on macOS)."""
    parts = set(Path(collection_path).parts)
    if parts & {".venv", "venv", "site-packages", "numpy", "scipy", "torch"}:
        return True
    return None


@pytest.fixture(scope="session")
def project_root():
    return _project_root


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """Return a Settings object pointing at a temporary directory."""
    from cafeteria.config.settings import load_settings, reset_settings_cache
    reset_settings_cache()
    # Use a temp dir so tests don't pollute the real database
    tmp = tmp_path_factory.mktemp("cafeteria_test")
    # Create minimal config pointing at tmp
    config_file = tmp / "config.yaml"
    import yaml
    raw = {
        "application": {"name": "Test", "debug": False, "log_level": "WARNING",
                         "runtime_state_path": "data/runtime_state.json",
                         "commands_path": "data/commands.json"},
        "storage": {
            "database": "database/test.db",
            "captures": "data/captures",
            "review_queue": "data/review_queue",
            "datasets": "data/datasets",
            "models": "models",
            "frames": "data/frames",
            "exports": "data/exports",
        },
        "recognition": {
            "embedding_dir": "data/enrollment",
            "similarity_threshold": 0.52,
            "frames_to_vote": 3,
            "minimum_face_size": 24,
            "identify_face_size": 48,
            "max_event_duration_seconds": 4.0,
            "model_pack": "buffalo_l",
            "det_size": [480, 480],
            "det_thresh": 0.42,
            "infer_max_width": 1280,
            "bbox_hold_frames": 12,
        },
        "event": {
            "minimum_plate_presence_seconds": 0.1,
            "cooldown_seconds": 0.2,
            "timeout_seconds": 2.0,
            "best_frame_window": 5,
        },
        "training": {
            "train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1,
            "random_seed": 42,
            "default_base_model": "yolov8n-cls.pt",
            "plate_base_model": "yolov8n.pt",
            "default_epochs": 2, "default_image_size": 64,
            "default_batch": 2, "output_base": "models",
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(raw, f)

    reset_settings_cache()
    settings = load_settings(config_path=config_file)
    # Point project_root at tmp
    settings.__dict__["project_root"] = tmp
    settings.ensure_directories()
    return settings


@pytest.fixture(scope="session")
def db(cfg):
    """Initialize test database."""
    from cafeteria.storage.database import init_db
    init_db(cfg.project_root / cfg.storage.database)
    return cfg.project_root / cfg.storage.database


@pytest.fixture
def db_session(db):
    """Provide a test database session, rolled back after each test."""
    from cafeteria.storage.database import get_session
    session = get_session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def dummy_frame():
    """Return a 640x480 dummy BGR frame."""
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)
