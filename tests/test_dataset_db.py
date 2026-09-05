"""
tests/test_dataset_db.py — Tests for database-backed DatasetImage repository.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cafeteria.storage.models import Base
from cafeteria.storage.repositories import DatasetImageRepository


@pytest.fixture
def mem_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_dataset_image_crud(mem_session):
    repo = DatasetImageRepository(mem_session)

    # Add images
    img1 = repo.add_image(
        task="plate",
        label="plate",
        image_path="/tmp/plate1.jpg",
        split="train",
        image_hash="hash1",
        bboxes_json='[[0.5, 0.5, 0.4, 0.4]]',
    )
    assert img1.id is not None
    assert img1.task == "plate"
    assert img1.split == "train"

    # Duplicate hash rejection per task
    img1_dup = repo.add_image(
        task="plate",
        label="plate",
        image_path="/tmp/plate1_copy.jpg",
        image_hash="hash1",
    )
    assert img1_dup.id == img1.id

    # Waste image
    img2 = repo.add_image(
        task="waste",
        label="LOW_WASTE",
        image_path="/tmp/waste1.jpg",
        split="val",
        image_hash="hash2",
    )
    assert img2.label == "LOW_WASTE"
    assert img2.split == "val"

    # Counts
    assert repo.count() == 2
    assert repo.count(task="plate") == 1
    assert repo.count(task="waste") == 1

    # Verification
    assert repo.verify(img1.id, label="plate") is True
    assert repo.get_by_id(img1.id).is_verified is True

    # Delete
    assert repo.delete(img2.id) is True
    assert repo.count(task="waste") == 0
