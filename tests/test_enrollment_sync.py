"""On-disk enrollments sync into the Person table without inventing people."""
from __future__ import annotations

import json

import numpy as np

from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.storage.repositories import PersonRepository


def test_sync_to_database_creates_missing_person(db_session, tmp_path):
    person_dir = tmp_path / "enroll_sync_01"
    (person_dir / "images").mkdir(parents=True)
    (person_dir / "meta.json").write_text(json.dumps({
        "person_id": "enroll_sync_01",
        "name": "Bhavmanyu",
        "image_count": 14,
    }))
    np.save(str(person_dir / "embedding.npy"), np.ones(512, dtype=np.float32))

    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    created = mgr.sync_to_database(db_session)
    db_session.flush()

    assert created == 1
    person = PersonRepository(db_session).get_by_person_id("enroll_sync_01")
    assert person is not None
    assert person.name == "Bhavmanyu"
    assert mgr.has_embedding("enroll_sync_01")

    # Second sync is idempotent
    assert mgr.sync_to_database(db_session) == 0


def test_sync_skips_empty_stub_directories(db_session, tmp_path):
    stub = tmp_path / "k"
    stub.mkdir()
    (stub / "meta.json").write_text(json.dumps({
        "person_id": "k",
        "name": "p",
        "image_count": 0,
    }))
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    assert mgr.sync_to_database(db_session) == 0
    db_session.flush()
    assert PersonRepository(db_session).get_by_person_id("k") is None


def test_sync_keeps_person_with_meta_history_even_if_images_missing(db_session, tmp_path):
    person_dir = tmp_path / "history_01"
    person_dir.mkdir()
    (person_dir / "meta.json").write_text(json.dumps({
        "person_id": "history_01",
        "name": "Historical",
        "image_count": 14,
    }))
    (person_dir / "samples.jsonl").write_text("{}\n")
    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    assert mgr.sync_to_database(db_session) == 1
    db_session.flush()
    person = PersonRepository(db_session).get_by_person_id("history_01")
    assert person is not None
    assert person.name == "Historical"
