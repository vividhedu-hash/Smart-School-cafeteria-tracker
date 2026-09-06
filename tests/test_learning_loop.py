"""Closed ML feedback loop — enqueue, promote, retrain policy."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from cafeteria.training.learning_loop import (
    LearningLoop,
    LoopConfig,
    _score_metrics,
    crop_plate_from_frame,
)
from cafeteria.training.registry import ModelRegistry
from cafeteria.detection.plate_detector import Detection


@pytest.fixture
def loop_env(tmp_path):
    datasets = tmp_path / "datasets"
    models = tmp_path / "models"
    registry = ModelRegistry(models / "registry.json")
    cfg = LoopConfig(
        enabled=True,
        auto_enqueue=True,
        min_confidence_to_enqueue=0.40,
        promote_auto_confirmed=True,
        auto_confirm_min_confidence=0.85,
        retrain_after_n_labels=3,
        min_images_per_class=2,
        auto_activate=True,
        auto_activate_min_accuracy=0.10,
        retrain_epochs=1,
    )
    loop = LearningLoop(
        project_root=tmp_path,
        datasets_dir=datasets,
        models_dir=models,
        registry=registry,
        config=cfg,
        device="cpu",
    )
    return loop, registry, datasets


def _plate_crop(color=(40, 180, 40)):
    img = np.full((120, 120, 3), 220, dtype=np.uint8)
    cv2.circle(img, (60, 60), 35, color, -1)
    return img


def test_enqueue_and_list_pending(loop_env):
    loop, _, _ = loop_env
    cand_id = loop.enqueue_plate_crop(
        _plate_crop(),
        predicted_label="MEDIUM_WASTE",
        confidence=0.72,
        transaction_id="TX-TEST1",
        auto_confirmed=False,
    )
    assert cand_id is not None
    pending = loop.list_pending()
    assert len(pending) == 1
    assert pending[0]["predicted_label"] == "MEDIUM_WASTE"
    assert Path(pending[0]["image_path"]).exists()


def test_enqueue_skips_low_confidence(loop_env):
    loop, _, _ = loop_env
    cand_id = loop.enqueue_plate_crop(
        _plate_crop(),
        predicted_label="LOW_WASTE",
        confidence=0.10,
        transaction_id="TX-LOW",
    )
    assert cand_id is None
    assert loop.list_pending() == []


def test_auto_confirm_promotes_into_dataset(loop_env):
    loop, _, datasets = loop_env
    cand_id = loop.enqueue_plate_crop(
        _plate_crop(),
        predicted_label="HIGH_WASTE",
        confidence=0.91,
        transaction_id="TX-AUTO",
        auto_confirmed=True,
    )
    assert cand_id is not None
    # Auto-promoted → no longer pending
    assert all(c["id"] != cand_id for c in loop.list_pending())
    high = list((datasets / "waste" / "HIGH_WASTE").glob("*.jpg"))
    assert len(high) == 1
    snap = loop.snapshot()
    assert snap["promoted_since_train"] == 1
    assert snap["labels_promoted_total"] == 1


def test_manual_promote_increments_counter(loop_env):
    loop, _, datasets = loop_env
    cand_id = loop.enqueue_plate_crop(
        _plate_crop((20, 40, 160)),
        predicted_label="LOW_WASTE",
        confidence=0.60,
        transaction_id="TX-MAN",
        auto_confirmed=False,
    )
    result = loop.promote(cand_id, "EMPTY", source="test", trigger_retrain=False)
    assert result["added"] == 1
    assert list((datasets / "waste" / "EMPTY").glob("*.jpg"))
    assert loop.snapshot()["promoted_since_train"] == 1


def test_can_retrain_requires_threshold_and_classes(loop_env):
    loop, _, _ = loop_env
    assert loop.can_retrain() is False

    # Seed two classes with enough unique images (hash-deduped)
    for label in ("EMPTY", "HIGH_WASTE"):
        for i in range(4):
            img = _plate_crop((10 + i * 40, 50 + i * 20 + (0 if label == "EMPTY" else 5), 100 + i * 15))
            path = loop._pending_dir / f"seed_{label}_{i}.jpg"
            cv2.imwrite(str(path), img)
            loop.promote_from_paths([path], label, trigger_retrain=False)

    assert loop.snapshot()["promoted_since_train"] >= 8
    assert loop.can_retrain() is True


def test_score_metrics_prefers_top1():
    assert _score_metrics({"metrics/accuracy_top1": 0.77}) == pytest.approx(0.77)
    assert _score_metrics({"top1_accuracy": 0.55}) == pytest.approx(0.55)
    assert _score_metrics({}) is None


def test_crop_plate_from_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[40:120, 50:150] = 200
    det = Detection(label="plate", confidence=0.9, x1=50, y1=40, x2=150, y2=120)
    crop = crop_plate_from_frame(frame, det)
    assert crop is not None
    assert crop.shape[0] == 80
    assert crop.shape[1] == 100


def test_activate_first_model_without_score(loop_env, monkeypatch):
    loop, registry, _ = loop_env

    class FakeResult:
        version = "v001"
        metrics = {}

    notified = []
    monkeypatch.setattr(loop, "_notify_engine", lambda v: notified.append(v))
    registry.register_version(
        task="waste", version="v001", weights_path="models/waste/v001/best.pt",
        metrics={},
    )
    info = loop._maybe_activate(FakeResult())
    assert info["activated"] is True
    assert info["reason"] == "first_model"
    assert registry.active_version_string("waste") == "v001"
    assert notified == ["v001"]


def test_activate_rejects_worse_model(loop_env, monkeypatch):
    loop, registry, _ = loop_env
    registry.register_version(
        task="waste", version="v001", weights_path="a.pt",
        metrics={"metrics/accuracy_top1": 0.90},
    )
    registry.activate("waste", "v001")
    registry.register_version(
        task="waste", version="v002", weights_path="b.pt",
        metrics={"metrics/accuracy_top1": 0.40},
    )

    class FakeResult:
        version = "v002"
        metrics = {"metrics/accuracy_top1": 0.40}

    monkeypatch.setattr(loop, "_notify_engine", lambda v: None)
    info = loop._maybe_activate(FakeResult())
    assert info["activated"] is False
    assert info["reason"] == "worse_than_active"
    assert registry.active_version_string("waste") == "v001"


def test_ingest_from_database(loop_env):
    loop, _, _ = loop_env
    res = loop.ingest_from_database(min_confidence=0.70, trigger_retrain=False)
    assert "ingested" in res
    assert "skipped" in res
    assert isinstance(res["ingested"], int)

