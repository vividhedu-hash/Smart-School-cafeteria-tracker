"""E2E: gated pipeline creates no DB rows; ready pipeline commits real ones."""
from __future__ import annotations

from cafeteria.detection.plate_detector import ModelNotFoundError, PlateDetector
from cafeteria.pipeline.transaction import TransactionEngine
from cafeteria.storage.repositories import ReviewRepository, TransactionRepository

from test_event_manager import (
    StubPlateDetector,
    StubWasteDetector,
    make_manager,
    run_until_event,
)


def test_gated_pipeline_creates_zero_transactions(db_session, tmp_path):
    before = TransactionRepository(db_session).count_total()
    em, sm = make_manager(plate=None, waste=None, tmp_path=tmp_path)
    assert run_until_event(em, max_frames=15) is None
    db_session.flush()
    assert TransactionRepository(db_session).count_total() == before


def test_ready_pipeline_commits_real_sqlite_row(db_session, tmp_path):
    plate = StubPlateDetector()
    waste = StubWasteDetector(label="HIGH_WASTE", confidence=0.9)
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)
    event = run_until_event(em)
    assert event is not None
    assert event.waste_result.label == "HIGH_WASTE"

    engine = TransactionEngine(
        captures_dir=tmp_path / "captures",
        review_queue_dir=tmp_path / "review",
    )
    tx_id = engine.commit(event)
    assert tx_id.startswith("TX-")

    db_session.expire_all()
    repo = TransactionRepository(db_session)
    tx = repo.get_by_id(tx_id)
    assert tx is not None
    assert tx.waste_status == "HIGH_WASTE"
    assert tx.plate_detected is True
    # No face engine → honest review, not a fabricated identity
    assert tx.status == "REVIEW_REQUIRED"
    review = ReviewRepository(db_session).get_unresolved()
    assert any(r.transaction_id == tx_id for r in review)


def test_proxy_plate_forces_review_not_auto_confirm(tmp_path):
    plate = StubPlateDetector()
    plate.proxy_mode = True
    waste = StubWasteDetector()
    em, sm = make_manager(plate=plate, waste=waste, tmp_path=tmp_path)
    event = run_until_event(em)
    assert event is not None
    assert event.review_reason == "PROXY_PLATE_MODE"
    assert event.status == "REVIEW_REQUIRED"


def test_missing_plate_weights_do_not_silently_load_coco(tmp_path):
    det = PlateDetector(
        tmp_path / "nope.pt",
        allow_coco_fallback=False,
        allow_visual_fallback=False,
    )
    try:
        det.load()
        assert False, "expected ModelNotFoundError"
    except ModelNotFoundError as exc:
        assert "Plate detector model not found" in str(exc)
        assert det.is_loaded is False
        assert det.proxy_mode is False
