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


def test_supabase_url_normalisation():
    from cafeteria.storage.database import _normalise_supabase_url, _extract_host_port

    # 1. With standard port
    u1 = "postgresql://postgres:secret@db.qeanxgkbcljacserkked.supabase.co:5432/postgres"
    n1 = _normalise_supabase_url(u1)
    assert "aws-0-ap-south-1.pooler.supabase.com:5432" in n1
    assert "postgres.qeanxgkbcljacserkked" in n1

    # 2. Without port (common copy-paste)
    u2 = "postgresql://postgres:secret@db.qeanxgkbcljacserkked.supabase.co/postgres"
    n2 = _normalise_supabase_url(u2)
    assert "aws-0-ap-south-1.pooler.supabase.com:5432" in n2
    assert "postgres.qeanxgkbcljacserkked" in n2

    # 3. With transaction pooler port 6543
    u3 = "postgresql://postgres.qeanxgkbcljacserkked:secret@db.qeanxgkbcljacserkked.supabase.co:6543/postgres"
    n3 = _normalise_supabase_url(u3)
    assert "aws-0-ap-south-1.pooler.supabase.com:6543" in n3
    assert "postgres.qeanxgkbcljacserkked" in n3

    # 4. Legacy postgres:// scheme without port
    u4 = "postgres://postgres:secret@db.qeanxgkbcljacserkked.supabase.co/postgres"
    n4 = _normalise_supabase_url(u4)
    assert n4.startswith("postgresql://")
    assert "aws-0-ap-south-1.pooler.supabase.com:5432" in n4

    # 5. Non-supabase URL remains unchanged
    u5 = "postgresql://myuser:mypass@custom-db.internal:5432/production"
    assert _normalise_supabase_url(u5) == u5

    # 6. Host and port extraction
    host, port = _extract_host_port(n1)
    assert host == "aws-0-ap-south-1.pooler.supabase.com"
    assert port == 5432


def test_ensure_runtime_storage_fallback_on_unreachable_pg(cfg, db):
    from types import SimpleNamespace
    from cafeteria.storage.database import init_db

    fake_cfg = SimpleNamespace(
        project_root=cfg.project_root,
        storage=SimpleNamespace(
            database="database/isolated_fallback.db",
            database_url="postgresql://invalid_user:invalid_pass@192.0.2.1:5432/nonexistent",
        ),
        recognition=SimpleNamespace(embedding_dir="data/enrollment"),
    )
    try:
        report = ensure_runtime_storage(fake_cfg, sync_enrollments=False)
        assert report["ok"] is True
        assert db_health_check() is True
    finally:
        init_db(db)


def test_load_app_resilience(tmp_path, monkeypatch):
    from dashboard.boot import load_app
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@invalid.supabase.co/db")
    cfg = load_app()
    assert cfg is not None
    assert db_health_check() is True

