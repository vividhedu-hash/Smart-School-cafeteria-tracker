"""Plate label queue — capture, label, delete. Labels are only ever operator-accepted."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cafeteria.training.dataset_manager import DatasetManager


@pytest.fixture
def dm(tmp_path):
    return DatasetManager(
        datasets_dir=tmp_path / "datasets",
        train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
        random_seed=7,
    )


def _image(tmp_path, name="plate.jpg"):
    path = tmp_path / name
    Image.fromarray(
        np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
    ).save(path)
    return path


def test_unlabelled_queue_starts_empty(dm):
    # [AI-CoLab: Verified by Antigravity] Verified plate dataset labeling queue unit tests
    assert dm.plate_unlabelled_images() == []


def test_added_image_is_unlabelled_until_a_label_is_written(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path))
    queue = dm.plate_unlabelled_images()
    assert len(queue) == 1

    dm.write_plate_label(queue[0].stem, [(0.5, 0.5, 0.4, 0.3)])
    assert dm.plate_unlabelled_images() == []


def test_label_file_is_yolo_format(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path))
    stem = dm.plate_unlabelled_images()[0].stem
    path = dm.write_plate_label(stem, [(0.5, 0.25, 0.4, 0.3)])
    fields = path.read_text(encoding="utf-8").strip().split()
    assert fields[0] == "0"
    assert [round(float(v), 3) for v in fields[1:]] == [0.5, 0.25, 0.4, 0.3]


def test_out_of_range_boxes_are_clamped(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path))
    stem = dm.plate_unlabelled_images()[0].stem
    path = dm.write_plate_label(stem, [(1.4, -0.2, 2.0, 0.5)])
    values = [float(v) for v in path.read_text(encoding="utf-8").split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_zero_area_boxes_are_dropped(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path))
    stem = dm.plate_unlabelled_images()[0].stem
    path = dm.write_plate_label(stem, [(0.5, 0.5, 0.0, 0.0)])
    assert path.read_text(encoding="utf-8").strip() == ""


def test_delete_plate_image_removes_label_too(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path))
    image = dm.plate_unlabelled_images()[0]
    label = dm.write_plate_label(image.stem, [(0.5, 0.5, 0.4, 0.4)])

    assert dm.delete_plate_image(image.name) is True
    assert not image.exists()
    assert not label.exists()
    assert dm.delete_plate_image("does_not_exist.jpg") is False


def test_stats_track_the_labelling_backlog(dm, tmp_path):
    dm.add_plate_image(_image(tmp_path, "a.jpg"))
    dm.add_plate_image(_image(tmp_path, "b.jpg"))
    dm.write_plate_label(dm.plate_unlabelled_images()[0].stem, [(0.5, 0.5, 0.5, 0.5)])

    stats = dm.plate_dataset_stats()
    assert stats["images"] == 2
    assert stats["labels"] == 1
    assert stats["unlabelled"] == 1
