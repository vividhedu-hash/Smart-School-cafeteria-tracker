"""First-run SQLite bootstrap: create file, health-check, sync gallery."""
from __future__ import annotations

import json

import numpy as np

from cafeteria.storage.bootstrap import ensure_runtime_storage, resolve_person_fk
from cafeteria.storage.database import db_health_check, get_session
from cafeteria.storage.repositories import PersonRepository, TransactionRepository


def test_ensure_runtime_storage_creates_sqlite_file(cfg, db):
    from types import SimpleNamespace

    from cafeteria.storage.database import init_db

    isolated = SimpleNamespace(
        project_root=cfg.project_root,
        storage=SimpleNamespace(database="database/isolated.db"),
        recognition=SimpleNamespace(embedding_dir="data/enrollment"),
    )
    db_path = cfg.project_root / isolated.storage.database
    try:
        report = ensure_runtime_storage(isolated, sync_enrollments=False)
        assert db_path.exists()
        assert report["created"] is True
        assert report["ok"] is True
        assert db_health_check() is True
        again = ensure_runtime_storage(isolated, sync_enrollments=False)
        assert again["created"] is False
        assert again["ok"] is True
    finally:
        init_db(db)


def test_bootstrap_syncs_ready_person_skips_empty_stub(cfg, db):
    root = cfg.project_root / cfg.recognition.embedding_dir
    ready = root / "ready_01"
    (ready / "images").mkdir(parents=True)
    (ready / "meta.json").write_text(json.dumps({
        "person_id": "ready_01",
        "name": "Ready Person",
        "image_count": 1,
    }))
    np.save(str(ready / "embedding.npy"), np.ones(512, dtype=np.float32))
    stub = root / "stub_k"
    stub.mkdir(parents=True)
    (stub / "meta.json").write_text(json.dumps({
        "person_id": "stub_k",
        "name": "p",
        "image_count": 0,
    }))

    report = ensure_runtime_storage(cfg)
    assert report["ok"] is True
    assert report["persons_synced"] == 1

    session = get_session()
    try:
        repo = PersonRepository(session)
        assert repo.get_by_person_id("ready_01") is not None
        assert repo.get_by_person_id("stub_k") is None
    finally:
        session.close()


def test_resolve_person_fk_creates_missing_row(db_session):
    fk = resolve_person_fk(db_session, "new_person", "New Person")
    db_session.flush()
    assert fk is not None
    person = PersonRepository(db_session).get_by_person_id("new_person")
    assert person is not None
    assert person.id == fk
    assert person.name == "New Person"


def test_update_status_sets_person_fk(db_session):
    tx_repo = TransactionRepository(db_session)
    tx = tx_repo.create(
        timestamp=0.0,
        plate_detected=True,
        food_present=True,
        waste_status="HIGH_WASTE",
        status="REVIEW_REQUIRED",
    )
    db_session.flush()
    assert tx_repo.update_status(
        tx.transaction_id,
        status="MANUALLY_CONFIRMED",
        person_id="alice_01",
        person_name="Alice",
    )
    db_session.flush()
    refreshed = tx_repo.get_by_id(tx.transaction_id)
    assert refreshed.person_id == "alice_01"
    assert refreshed.person_id_fk is not None
    person = PersonRepository(db_session).get_by_person_id("alice_01")
    assert person is not None
    assert refreshed.person_id_fk == person.id
