"""
Tests for the storage layer.
"""
from __future__ import annotations

import time
import pytest
from cafeteria.storage.repositories import (
    PersonRepository, TransactionRepository, ReviewRepository
)


def test_create_and_get_person(db_session):
    repo = PersonRepository(db_session)
    person = repo.create(person_id="test_p001", name="Test Person")
    db_session.flush()
    assert person.person_id == "test_p001"
    assert person.name == "Test Person"

    fetched = repo.get_by_person_id("test_p001")
    assert fetched is not None
    assert fetched.name == "Test Person"


def test_list_all_persons(db_session):
    repo = PersonRepository(db_session)
    repo.create(person_id="p_list_001", name="Alice")
    repo.create(person_id="p_list_002", name="Bob")
    db_session.flush()
    persons = repo.get_all_active()
    ids = {p.person_id for p in persons}
    assert "p_list_001" in ids
    assert "p_list_002" in ids


def test_create_transaction(db_session):
    repo = TransactionRepository(db_session)
    tx = repo.create(
        timestamp=time.time(),
        person_id="p_tx_001",
        person_name="Charlie",
        plate_detected=True,
        food_present=True,
        waste_status="HIGH_WASTE",
        plate_confidence=0.87,
        waste_confidence=0.91,
        face_confidence=0.75,
        status="AUTO_CONFIRMED",
        event_image_path=None,
        processing_latency_ms=123.4,
        review_reason=None,
    )
    db_session.flush()
    assert tx.transaction_id.startswith("TX-")
    assert tx.waste_status == "HIGH_WASTE"
    assert tx.status == "AUTO_CONFIRMED"


def test_count_transactions(db_session):
    repo = TransactionRepository(db_session)
    before = repo.count_total()
    repo.create(
        timestamp=time.time(),
        plate_detected=True, food_present=True,
        waste_status="EMPTY", status="AUTO_CONFIRMED",
    )
    db_session.flush()
    after = repo.count_total()
    assert after >= before + 1


def test_create_review_entry(db_session):
    tx_repo = TransactionRepository(db_session)
    rev_repo = ReviewRepository(db_session)

    tx = tx_repo.create(
        timestamp=time.time(),
        plate_detected=True, food_present=True,
        waste_status="MEDIUM_WASTE", status="REVIEW_REQUIRED",
    )
    db_session.flush()

    entry = rev_repo.create(
        transaction_id=tx.transaction_id,
        waste_status="MEDIUM_WASTE",
        waste_confidence=0.78,
        candidates=[{"person_id": "p001", "similarity": 0.40}],
        reason="Low confidence",
    )
    db_session.flush()
    assert entry.transaction_id == tx.transaction_id
    assert entry.is_resolved is False


def test_resolve_review(db_session):
    tx_repo = TransactionRepository(db_session)
    rev_repo = ReviewRepository(db_session)

    tx = tx_repo.create(
        timestamp=time.time(), plate_detected=True, food_present=True,
        waste_status="LOW_WASTE", status="REVIEW_REQUIRED",
    )
    db_session.flush()
    entry = rev_repo.create(
        transaction_id=tx.transaction_id,
        waste_status="LOW_WASTE",
        reason="test",
    )
    db_session.flush()

    rev_repo.resolve(entry.id, "p_resolved", "Dave", resolved_by="manual")
    db_session.flush()

    refreshed = rev_repo.get_by_id(entry.id)
    assert refreshed.is_resolved is True
    assert refreshed.resolved_person_id == "p_resolved"


def test_get_filtered_transactions(db_session):
    tx_repo = TransactionRepository(db_session)
    now = time.time()
    tx_repo.create(
        timestamp=now, plate_detected=True, food_present=True,
        waste_status="HIGH_WASTE", status="AUTO_CONFIRMED",
        person_id="filter_test_person",
    )
    db_session.flush()

    results = tx_repo.get_filtered(person_id="filter_test_person")
    assert any(r.person_id == "filter_test_person" for r in results)

    results_waste = tx_repo.get_filtered(waste_status="HIGH_WASTE")
    assert any(r.waste_status == "HIGH_WASTE" for r in results_waste)
