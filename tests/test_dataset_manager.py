"""
Tests for the dataset manager (no model dependencies).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES


@pytest.fixture
def dm(tmp_path):
    return DatasetManager(
        datasets_dir=tmp_path / "datasets",
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_seed=42,
    )


def _make_dummy_images(tmp_path: Path, n: int) -> list[Path]:
    """Create n dummy JPEG files."""
    from PIL import Image
    paths = []
    for i in range(n):
        p = tmp_path / f"img_{i:04d}.jpg"
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def test_waste_class_counts_empty(dm):
    counts = dm.waste_class_counts()
    assert all(v == 0 for v in counts.values())


def test_add_waste_images(dm, tmp_path):
    imgs = _make_dummy_images(tmp_path, 5)
    added, skipped = dm.add_waste_images("EMPTY", imgs)
    assert added == 5
    assert skipped == 0
    assert dm.waste_class_counts()["EMPTY"] == 5


def test_duplicate_detection(dm, tmp_path):
    imgs = _make_dummy_images(tmp_path, 3)
    dm.add_waste_images("LOW_WASTE", imgs)
    added, skipped = dm.add_waste_images("LOW_WASTE", imgs)  # Same files again
    assert added == 0
    assert skipped == 3


def test_invalid_class_raises(dm):
    with pytest.raises(ValueError, match="Unknown waste class"):
        dm.add_waste_images("INVALID_CLASS", [])


def test_prepare_yolo_cls_dataset(dm, tmp_path):
    # Add images across all classes
    for cls in WASTE_CLASSES:
        imgs = _make_dummy_images(tmp_path, 5)
        dm.add_waste_images(cls, imgs)

    out_dir = tmp_path / "split"
    stats = dm.prepare_yolo_cls_dataset(out_dir)

    # Check directories exist
    for split in ("train", "val", "test"):
        for cls in WASTE_CLASSES:
            split_cls_dir = out_dir / split / cls
            assert split_cls_dir.exists(), f"Missing: {split}/{cls}"

    assert stats["total"] > 0


def test_delete_waste_image(dm, tmp_path):
    imgs = _make_dummy_images(tmp_path, 2)
    dm.add_waste_images("HIGH_WASTE", imgs)
    images = dm.waste_images("HIGH_WASTE")
    assert len(images) == 2

    filename = images[0].name
    deleted = dm.delete_waste_image("HIGH_WASTE", filename)
    assert deleted is True
    assert dm.waste_class_counts()["HIGH_WASTE"] == 1


def test_total_waste_images(dm, tmp_path):
    imgs = _make_dummy_images(tmp_path, 3)
    dm.add_waste_images("EMPTY", imgs[:2])
    dm.add_waste_images("HIGH_WASTE", imgs[2:])
    assert dm.total_waste_images() == 3
