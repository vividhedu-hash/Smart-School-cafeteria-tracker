"""On-disk enrollments sync into the Person table without inventing people."""
from __future__ import annotations

import json

import numpy as np

from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.storage.repositories import PersonRepository


def test_sync_to_database_creates_missing_person(db_session, tmp_path):
    person_dir = tmp_path / "Bhavmanyu_24"
    (person_dir / "images").mkdir(parents=True)
    (person_dir / "meta.json").write_text(json.dumps({
        "person_id": "Bhavmanyu_24",
        "name": "Bhavmanyu",
        "image_count": 14,
    }))
    np.save(str(person_dir / "embedding.npy"), np.ones(512, dtype=np.float32))

    mgr = EnrollmentManager(enrollment_dir=tmp_path)
    created = mgr.sync_to_database(db_session)
    db_session.flush()

    assert created == 1
    person = PersonRepository(db_session).get_by_person_id("Bhavmanyu_24")
    assert person is not None
    assert person.name == "Bhavmanyu"
    assert mgr.has_embedding("Bhavmanyu_24")

    # Second sync is idempotent
    assert mgr.sync_to_database(db_session) == 0
