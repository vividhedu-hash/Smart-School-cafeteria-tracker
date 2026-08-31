"""
Transaction creator — converts a CompletedEvent into a database record
and creates a ReviewEntry when identity is unknown.

Also feeds the closed ML loop: plate crops are queued for labeling /
auto-promoted when confidence is high, so the waste model can retrain.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

from cafeteria.pipeline.event_manager import CompletedEvent
from cafeteria.storage.bootstrap import resolve_person_fk
from cafeteria.storage.database import get_session
from cafeteria.storage.models import TransactionStatus
from cafeteria.storage.repositories import ReviewRepository, TransactionRepository
from cafeteria.utils.logging import get_logger, EventCode, log_event

logger = get_logger("pipeline.transaction")


class TransactionEngine:
    """
    Creates transactions and review entries from completed waste events.

    Args:
        captures_dir:     Base directory for saving evidence images.
        review_queue_dir: Directory for review queue images.
        learning_loop:    Optional LearningLoop for closed-loop training.
    """

    def __init__(
        self,
        captures_dir: str | Path,
        review_queue_dir: str | Path,
        learning_loop=None,
    ) -> None:
        self._captures_dir = Path(captures_dir)
        self._review_dir = Path(review_queue_dir)
        self._learning_loop = learning_loop
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        self._review_dir.mkdir(parents=True, exist_ok=True)

    def set_learning_loop(self, learning_loop) -> None:
        self._learning_loop = learning_loop

    def commit(self, event: CompletedEvent) -> str:
        """
        Commit a completed event to the database.

        Args:
            event: CompletedEvent from the EventManager.

        Returns:
            transaction_id string.
        """
        import cv2
        from datetime import datetime, timezone

        # Determine status. An event-level review_reason (e.g. PROXY_PLATE_MODE)
        # always forces the transaction into the review queue, even if the
        # face was recognised.
        if event.review_reason:
            status = TransactionStatus.REVIEW_REQUIRED
        elif event.face_match and event.face_match.is_known:
            status = TransactionStatus.AUTO_CONFIRMED
        else:
            status = TransactionStatus.REVIEW_REQUIRED

        review_reason = (
            None if status == TransactionStatus.AUTO_CONFIRMED
            else (event.review_reason or "Face not recognised")
        )

        # Save evidence image + plate crop for the ML loop
        event_image_path: Optional[str] = None
        plate_image_path: Optional[str] = None
        img_dir: Optional[Path] = None
        if event.best_frame is not None:
            dt = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
            date_str = dt.strftime("%Y%m%d")
            tx_id_short = uuid.uuid4().hex[:8].upper()
            img_dir = self._captures_dir / date_str / tx_id_short
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / "event.jpg"
            cv2.imwrite(str(img_path), event.best_frame)
            event_image_path = str(img_path)

        plate_crop = event.plate_crop
        if plate_crop is None and event.best_frame is not None and event.plate_detection is not None:
            try:
                plate_crop = event.plate_detection.crop(event.best_frame)
            except Exception:
                plate_crop = None
        if plate_crop is not None and getattr(plate_crop, "size", 0) > 0:
            if img_dir is None:
                dt = datetime.fromtimestamp(event.timestamp, tz=timezone.utc)
                img_dir = self._captures_dir / dt.strftime("%Y%m%d") / uuid.uuid4().hex[:8].upper()
                img_dir.mkdir(parents=True, exist_ok=True)
            plate_path = img_dir / "plate.jpg"
            if cv2.imwrite(str(plate_path), plate_crop):
                plate_image_path = str(plate_path)

        # Database write (atomic)
        session = get_session()
        try:
            tx_repo = TransactionRepository(session)
            rev_repo = ReviewRepository(session)

            matched_pid = event.face_match.person_id if event.face_match else None
            matched_name = event.face_match.person_name if event.face_match else None
            person_fk = resolve_person_fk(session, matched_pid, matched_name)

            tx = tx_repo.create(
                timestamp=event.timestamp,
                person_id=matched_pid,
                person_name=matched_name,
                person_id_fk=person_fk,
                plate_detected=event.plate_detected,
                food_present=event.food_present,
                waste_status=event.waste_result.label if event.waste_result else None,
                plate_confidence=event.plate_confidence,
                waste_confidence=event.waste_result.confidence if event.waste_result else None,
                face_confidence=(
                    event.face_match.similarity if event.face_match else None
                ),
                status=status,
                event_image_path=event_image_path,
                plate_image_path=plate_image_path,
                processing_latency_ms=event.processing_latency_ms,
                review_reason=review_reason,
            )

            tx_id = tx.transaction_id

            # Create review entry if unknown
            if status == TransactionStatus.REVIEW_REQUIRED:
                candidates = (
                    event.face_match.candidates if event.face_match else []
                )
                rev_repo.create(
                    transaction_id=tx_id,
                    event_image_path=event_image_path,
                    waste_status=event.waste_result.label if event.waste_result else None,
                    waste_confidence=event.waste_result.confidence if event.waste_result else None,
                    candidates=candidates,
                    reason=review_reason or "Face not recognised above threshold",
                )
                log_event(logger, EventCode.REVIEW_CREATED,
                          f"tx={tx_id}  waste={event.waste_result.label if event.waste_result else '?'}")

            session.commit()
            log_event(
                logger, EventCode.TRANSACTION_CREATED,
                f"tx={tx_id}",
                status=status,
                person=tx.person_id or "UNKNOWN",
                waste=tx.waste_status or "?",
                latency_ms=f"{event.processing_latency_ms:.0f}",
            )

            self._feed_learning_loop(
                event=event,
                tx_id=tx_id,
                status=status,
                event_image_path=event_image_path,
                plate_crop=plate_crop,
            )
            return tx_id

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _feed_learning_loop(
        self,
        *,
        event: CompletedEvent,
        tx_id: str,
        status: str,
        event_image_path: Optional[str],
        plate_crop,
    ) -> None:
        """Enqueue plate crop for the closed ML loop (never raises into commit)."""
        if self._learning_loop is None or event.waste_result is None:
            return
        try:
            bbox = None
            if event.plate_detection is not None:
                d = event.plate_detection
                bbox = (int(d.x1), int(d.y1), int(d.x2), int(d.y2))
            self._learning_loop.enqueue_plate_crop(
                plate_crop if plate_crop is not None else event.best_frame,
                predicted_label=event.waste_result.label,
                confidence=float(event.waste_result.confidence or 0.0),
                transaction_id=tx_id,
                event_image_path=event_image_path,
                auto_confirmed=(status == TransactionStatus.AUTO_CONFIRMED),
                plate_bbox=bbox,
            )
        except Exception as exc:
            logger.warning("ML loop enqueue failed for %s: %s", tx_id, exc)
