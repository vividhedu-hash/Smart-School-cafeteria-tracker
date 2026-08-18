"""
Tests for scripts/quickstart.py preflight/readiness logic.

No real YOLO training happens here — training paths are exercised only up to
the decision logic, using a temporary dataset directory and a stub config.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parent.parent


def _load_quickstart():
    spec = importlib.util.spec_from_file_location(
        "quickstart", _project_root / "scripts" / "quickstart.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qs = _load_quickstart()


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_stub_cfg(tmp_path: Path) -> SimpleNamespace:
    """Minimal cfg stub with only the fields quickstart uses."""
    (tmp_path / "data" / "datasets").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=tmp_path,
        storage=SimpleNamespace(datasets="data/datasets", models="models"),
        recognition=SimpleNamespace(embedding_dir="data/enrollment"),
        models=SimpleNamespace(
            plate=SimpleNamespace(weights="models/plate/best.pt"),
            waste=SimpleNamespace(weights="models/waste/best.pt"),
        ),
        camera=SimpleNamespace(source=0, mode="webcam"),
        training=SimpleNamespace(
            default_epochs=2,
            default_batch=2,
            default_base_model="yolov8n-cls.pt",
            plate_base_model="yolov8n.pt",
            default_image_size=64,
        ),
        device="cpu",
        application=SimpleNamespace(commands_path="data/commands.json"),
    )


def add_waste_images(tmp_path: Path, cls: str, n: int) -> None:
    """Write n tiny unique JPEGs into the waste dataset class dir."""
    import cv2
    d = tmp_path / "data" / "datasets" / "waste" / cls
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(cls) % (2**32))
    for i in range(n):
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(d / f"{i:04d}.jpg"), img)


def add_plate_data(tmp_path: Path, n_images: int, n_labels: int) -> None:
    import cv2
    img_dir = tmp_path / "data" / "datasets" / "plate" / "images"
    lbl_dir = tmp_path / "data" / "datasets" / "plate" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(n_images):
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"{i:05d}.jpg"), img)
        if i < n_labels:
            (lbl_dir / f"{i:05d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")


# ──────────────────────────────────────────────────────────────────────────────
# gather_report
# ──────────────────────────────────────────────────────────────────────────────

def test_report_empty_dataset(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    report = qs.gather_report(cfg)
    assert report["waste_total"] == 0
    assert report["plate_images"] == 0
    assert report["plate_labels"] == 0
    assert report["enrolled"] == []
    assert report["waste_model_ready"] is False
    assert report["plate_model_ready"] is False


def test_report_counts_waste_images(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_waste_images(tmp_path, "EMPTY", 3)
    add_waste_images(tmp_path, "HIGH_WASTE", 5)
    report = qs.gather_report(cfg)
    assert report["waste_counts"]["EMPTY"] == 3
    assert report["waste_counts"]["HIGH_WASTE"] == 5
    assert report["waste_total"] == 8


def test_report_counts_enrolled_persons(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    pdir = tmp_path / "data" / "enrollment" / "alice_01"
    pdir.mkdir(parents=True)
    np.save(str(pdir / "embedding.npy"), np.zeros(512))
    # person without embedding must NOT count
    (tmp_path / "data" / "enrollment" / "bob_02").mkdir(parents=True)

    report = qs.gather_report(cfg)
    assert report["enrolled"] == ["alice_01"]


def test_report_detects_trained_model_on_disk(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    w = tmp_path / "models" / "waste" / "best.pt"
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b"fake-weights")
    report = qs.gather_report(cfg)
    assert report["waste_model_ready"] is True
    assert report["plate_model_ready"] is False


def test_report_detects_registry_active_model(tmp_path):
    from cafeteria.training.registry import ModelRegistry
    cfg = make_stub_cfg(tmp_path)
    weights = tmp_path / "models" / "waste" / "v001" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fake")
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    registry.register_version(task="waste", version="v001",
                              weights_path=str(weights))
    registry.activate("waste", "v001")

    report = qs.gather_report(cfg)
    assert report["waste_model_ready"] is True
    assert "v001" in report["waste_model_detail"]


# ──────────────────────────────────────────────────────────────────────────────
# Trainability decisions
# ──────────────────────────────────────────────────────────────────────────────

def test_waste_blocked_below_minimum(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_waste_images(tmp_path, "EMPTY", qs.MIN_WASTE_TOTAL - 1)
    ok, why = qs.waste_trainable(qs.gather_report(cfg))
    assert ok is False
    assert str(qs.MIN_WASTE_TOTAL) in why


def test_waste_possible_at_minimum_with_warning(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_waste_images(tmp_path, "EMPTY", 2)
    add_waste_images(tmp_path, "HIGH_WASTE", 2)
    ok, why = qs.waste_trainable(qs.gather_report(cfg))
    assert ok is True
    assert "warning" in why  # below recommended per-class count


def test_plate_blocked_without_images(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    ok, why = qs.plate_trainable(qs.gather_report(cfg))
    assert ok is False
    assert "no plate images" in why


def test_plate_blocked_without_labels(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_plate_data(tmp_path, n_images=15, n_labels=0)
    ok, why = qs.plate_trainable(qs.gather_report(cfg))
    assert ok is False
    assert "0 YOLO label" in why
    assert "labels/" in why  # tells the user where labels go
    assert "DATA_COLLECTION_GUIDE.md" in why


def test_plate_possible_with_labelled_images(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_plate_data(tmp_path, n_images=qs.RECOMMENDED_PLATE_LABELLED,
                   n_labels=qs.RECOMMENDED_PLATE_LABELLED)
    ok, why = qs.plate_trainable(qs.gather_report(cfg))
    assert ok is True
    assert "warning" not in why


def test_plate_possible_below_recommended_with_warning(tmp_path):
    cfg = make_stub_cfg(tmp_path)
    add_plate_data(tmp_path, n_images=2, n_labels=2)
    ok, why = qs.plate_trainable(qs.gather_report(cfg))
    assert ok is True
    assert "warning" in why


# ──────────────────────────────────────────────────────────────────────────────
# main() — dry run and blocked-training exit codes (no YOLO involved)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def patched_cfg(tmp_path, monkeypatch):
    """Route quickstart.main() at a stub cfg without touching global settings."""
    cfg = make_stub_cfg(tmp_path)

    import cafeteria.config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: cfg)
    return cfg


def test_main_dry_run_exits_zero(patched_cfg, capsys):
    rc = qs.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PREFLIGHT" in out
    assert "dry run" in out
    assert "BLOCKED" in out  # empty dataset → both trainings blocked


def test_main_dry_run_never_trains_even_with_data(
    patched_cfg, tmp_path, capsys, monkeypatch,
):
    add_waste_images(tmp_path, "EMPTY", 8)
    add_plate_data(tmp_path, n_images=12, n_labels=12)

    def _boom(*a, **k):
        raise AssertionError("trainer must not run during --dry-run")

    import cafeteria.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "WasteModelTrainer", _boom)
    monkeypatch.setattr(trainer_mod, "PlateModelTrainer", _boom)

    rc = qs.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry run" in out
    assert "POSSIBLE" in out


def test_main_waste_only_blocked_is_user_error(patched_cfg, capsys):
    rc = qs.main(["--waste-only"])
    out = capsys.readouterr().out
    assert rc == 1  # explicit request that cannot be fulfilled → error exit
    assert "waste training was requested but is blocked" in out


def test_main_plate_only_blocked_is_user_error(patched_cfg, capsys):
    rc = qs.main(["--plate-only"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "plate training was requested but is blocked" in out


def test_main_skips_gracefully_when_nothing_trainable(patched_cfg, capsys):
    rc = qs.main([])
    out = capsys.readouterr().out
    assert rc == 0  # default run skips blocked tasks without failing
    assert "Skipping waste training" in out
    assert "Skipping plate training" in out
    assert "Nothing was trained" in out


def test_main_trains_waste_when_data_present(patched_cfg, tmp_path, capsys, monkeypatch):
    """With sufficient data, main() must call the trainer and activate the result."""
    add_waste_images(tmp_path, "EMPTY", 3)
    add_waste_images(tmp_path, "HIGH_WASTE", 3)

    calls = {}

    class StubResult:
        weights_path = tmp_path / "models" / "waste" / "v001" / "weights" / "best.pt"
        metrics = {"top1_accuracy": 0.9}
        config = {}
        dataset_stats = {}

    class StubTrainer:
        def __init__(self, **kw):
            calls["init"] = kw

        def train(self, **kw):
            calls["train"] = kw
            StubResult.weights_path.parent.mkdir(parents=True, exist_ok=True)
            StubResult.weights_path.write_bytes(b"fake")
            return StubResult()

    import cafeteria.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "WasteModelTrainer", StubTrainer)

    rc = qs.main(["--waste-only", "--epochs", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert calls["train"]["epochs"] == 1
    assert "Activated waste model v001" in out
    # Plate still missing → final status must honestly say not fully active
    assert "Pipeline NOT fully active" in out

    # Registry actually has the active version
    from cafeteria.training.registry import ModelRegistry
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    assert registry.active_version_string("waste") == "v001"


def test_main_trains_plate_when_labels_present(patched_cfg, tmp_path, capsys, monkeypatch):
    add_plate_data(tmp_path, n_images=12, n_labels=12)

    calls = {}

    class StubResult:
        weights_path = tmp_path / "models" / "plate" / "v001" / "weights" / "best.pt"
        metrics = {"map50": 0.7}
        config = {}
        dataset_stats = {}

    class StubTrainer:
        def __init__(self, **kw):
            calls["init"] = kw

        def train(self, **kw):
            calls["train"] = kw
            StubResult.weights_path.parent.mkdir(parents=True, exist_ok=True)
            StubResult.weights_path.write_bytes(b"fake")
            return StubResult()

    import cafeteria.training.trainer as trainer_mod
    monkeypatch.setattr(trainer_mod, "PlateModelTrainer", StubTrainer)

    rc = qs.main(["--plate-only", "--epochs", "2"])
    out = capsys.readouterr().out
    assert rc == 0
    assert calls["train"]["epochs"] == 2
    assert calls["train"]["data_yaml"].endswith("data.yaml")
    assert "Activated plate model v001" in out

    from cafeteria.training.registry import ModelRegistry
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    assert registry.active_version_string("plate") == "v001"
