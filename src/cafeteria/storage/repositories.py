"""
Repository layer — all database read/write operations.

Each repository class wraps a SQLAlchemy Session and provides typed methods.
No raw SQL outside of this file.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from cafeteria.storage.models import (
    DatasetImage, ModelVersion, Person, ReviewEntry, Transaction,
    TransactionStatus, WasteStatus,
)
from cafeteria.utils.logging import get_logger

logger = get_logger("storage.repositories")


# ──────────────────────────────────────────────────────────────────────────────
# Person repository
# ──────────────────────────────────────────────────────────────────────────────

class PersonRepository:
    def __init__(self, session: Session) -> None:
        self._db = session

    def create(
        self,
        person_id: str,
        name: str,
        embedding_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Person:
        person = Person(
            id=str(uuid.uuid4()),
            person_id=person_id,
            name=name,
            embedding_path=embedding_path,
            notes=notes,
            enrolled_at=time.time(),
        )
        self._db.add(person)
        self._db.flush()
        logger.info("Created person: %s (%s)", person_id, name)
        return person

    def get_by_person_id(self, person_id: str) -> Optional[Person]:
        return (
            self._db.query(Person)
            .filter(Person.person_id == person_id, Person.is_active == True)
            .first()
        )

    def get_by_person_id_any(self, person_id: str) -> Optional[Person]:
        return (
            self._db.query(Person)
            .filter(Person.person_id == person_id)
            .first()
        )

    def get_all_active(self) -> list[Person]:
        return self._db.query(Person).filter(Person.is_active == True).all()

    def update_image_count(self, person_id: str, count: int) -> None:
        person = self.get_by_person_id(person_id)
        if person:
            person.image_count = count
            self._db.flush()

    def update_embedding_path(self, person_id: str, path: str) -> None:
        person = self.get_by_person_id(person_id)
        if person:
            person.embedding_path = path
            self._db.flush()

    def clear_embedding(self, person_id: str) -> None:
        person = self.get_by_person_id(person_id)
        if person:
            person.embedding_path = None
            self._db.flush()

    def update_name(self, person_id: str, name: str) -> bool:
        person = self.get_by_person_id(person_id)
        if not person:
            return False
        person.name = name.strip()
        self._db.flush()
        return True

    def delete(self, person_id: str) -> bool:
        person = self.get_by_person_id(person_id)
        if person:
            person.is_active = False
            person.embedding_path = None
            person.notes = (person.notes or "") + f"\ndeactivated_at={time.time()}"
            self._db.flush()
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Transaction repository
# ──────────────────────────────────────────────────────────────────────────────

class TransactionRepository:
    def __init__(self, session: Session) -> None:
        self._db = session

    def create(
        self,
        *,
        timestamp: float,
        person_id: Optional[str] = None,
        person_name: Optional[str] = None,
        plate_detected: bool = False,
        food_present: bool = False,
        waste_status: Optional[str] = None,
        plate_confidence: Optional[float] = None,
        waste_confidence: Optional[float] = None,
        face_confidence: Optional[float] = None,
        status: str = TransactionStatus.REVIEW_REQUIRED,
        event_image_path: Optional[str] = None,
        face_image_path: Optional[str] = None,
        plate_image_path: Optional[str] = None,
        processing_latency_ms: Optional[float] = None,
        review_reason: Optional[str] = None,
        person_id_fk: Optional[str] = None,
    ) -> Transaction:
        tx_id = f"TX-{uuid.uuid4().hex[:12].upper()}"
        tx = Transaction(
            id=str(uuid.uuid4()),
            transaction_id=tx_id,
            timestamp=timestamp,
            person_id_fk=person_id_fk,
            person_id=person_id,
            person_name=person_name,
            plate_detected=plate_detected,
            food_present=food_present,
            waste_status=waste_status,
            plate_confidence=plate_confidence,
            waste_confidence=waste_confidence,
            face_confidence=face_confidence,
            status=status,
            event_image_path=event_image_path,
            face_image_path=face_image_path,
            plate_image_path=plate_image_path,
            processing_latency_ms=processing_latency_ms,
            review_reason=review_reason,
            created_at=time.time(),
        )
        self._db.add(tx)
        self._db.flush()
        logger.info(
            "Created transaction %s — person=%s waste=%s status=%s",
            tx_id, person_id, waste_status, status,
        )
        return tx

    def get_by_id(self, transaction_id: str) -> Optional[Transaction]:
        return (
            self._db.query(Transaction)
            .filter(Transaction.transaction_id == transaction_id)
            .first()
        )

    def get_recent(self, limit: int = 50) -> list[Transaction]:
        return (
            self._db.query(Transaction)
            .order_by(Transaction.timestamp.desc())
            .limit(limit)
            .all()
        )

    def get_filtered(
        self,
        person_id: Optional[str] = None,
        waste_status: Optional[str] = None,
        status: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 200,
    ) -> list[Transaction]:
        q = self._db.query(Transaction)
        if person_id:
            q = q.filter(Transaction.person_id == person_id)
        if waste_status:
            q = q.filter(Transaction.waste_status == waste_status)
        if status:
            q = q.filter(Transaction.status == status)
        if start_ts:
            q = q.filter(Transaction.timestamp >= start_ts)
        if end_ts:
            q = q.filter(Transaction.timestamp <= end_ts)
        return q.order_by(Transaction.timestamp.desc()).limit(limit).all()

    def update_status(
        self,
        transaction_id: str,
        status: str,
        person_id: Optional[str] = None,
        person_name: Optional[str] = None,
    ) -> bool:
        tx = self.get_by_id(transaction_id)
        if tx:
            tx.status = status
            if person_id:
                tx.person_id = person_id
                from cafeteria.storage.bootstrap import resolve_person_fk
                tx.person_id_fk = resolve_person_fk(self._db, person_id, person_name)
            if person_name:
                tx.person_name = person_name
            self._db.flush()
            return True
        return False

    def count_total(self) -> int:
        return self._db.query(Transaction).count()

    def count_by_status(self, status: str) -> int:
        return self._db.query(Transaction).filter(Transaction.status == status).count()

    def count_by_waste(self, waste_status: str) -> int:
        return self._db.query(Transaction).filter(Transaction.waste_status == waste_status).count()

    def count_by_person(self, person_id: str) -> int:
        return self._db.query(Transaction).filter(Transaction.person_id == person_id).count()



# ──────────────────────────────────────────────────────────────────────────────
# Review queue repository
# ──────────────────────────────────────────────────────────────────────────────

class ReviewRepository:
    def __init__(self, session: Session) -> None:
        self._db = session

    def create(
        self,
        transaction_id: str,
        event_image_path: Optional[str] = None,
        waste_status: Optional[str] = None,
        waste_confidence: Optional[float] = None,
        candidates: Optional[list[dict]] = None,
        reason: Optional[str] = None,
    ) -> ReviewEntry:
        entry = ReviewEntry(
            id=str(uuid.uuid4()),
            transaction_id=transaction_id,
            created_at=time.time(),
            event_image_path=event_image_path,
            waste_status=waste_status,
            waste_confidence=waste_confidence,
            candidates_json=json.dumps(candidates or []),
            reason=reason,
        )
        self._db.add(entry)
        self._db.flush()
        return entry

    def get_unresolved(self) -> list[ReviewEntry]:
        return (
            self._db.query(ReviewEntry)
            .filter(ReviewEntry.is_resolved == False)
            .order_by(ReviewEntry.created_at.desc())
            .all()
        )

    def resolve(
        self,
        review_id: str,
        person_id: str,
        person_name: str,
        resolved_by: str = "manual",
    ) -> bool:
        entry = self._db.query(ReviewEntry).filter(ReviewEntry.id == review_id).first()
        if entry:
            entry.is_resolved = True
            entry.resolved_at = time.time()
            entry.resolved_person_id = person_id
            entry.resolved_person_name = person_name
            entry.resolved_by = resolved_by
            self._db.flush()
            return True
        return False

    def reject(self, review_id: str) -> bool:
        entry = self._db.query(ReviewEntry).filter(ReviewEntry.id == review_id).first()
        if entry:
            entry.is_resolved = True
            entry.resolved_at = time.time()
            entry.resolved_by = "rejected"
            self._db.flush()
            return True
        return False

    def count_unresolved(self) -> int:
        return self._db.query(ReviewEntry).filter(ReviewEntry.is_resolved == False).count()

    def get_by_id(self, review_id: str) -> Optional[ReviewEntry]:
        return self._db.query(ReviewEntry).filter(ReviewEntry.id == review_id).first()


# ──────────────────────────────────────────────────────────────────────────────
# Model version repository
# ──────────────────────────────────────────────────────────────────────────────

class ModelVersionRepository:
    def __init__(self, session: Session) -> None:
        self._db = session

    def create(
        self,
        task: str,
        version: str,
        weights_path: str,
        config_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        epochs: Optional[int] = None,
        map50: Optional[float] = None,
        map50_95: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        dataset_size: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> ModelVersion:
        mv = ModelVersion(
            id=str(uuid.uuid4()),
            task=task,
            version=version,
            weights_path=weights_path,
            config_path=config_path,
            metrics_path=metrics_path,
            trained_at=time.time(),
            epochs=epochs,
            map50=map50,
            map50_95=map50_95,
            precision=precision,
            recall=recall,
            dataset_size=dataset_size,
            notes=notes,
        )
        self._db.add(mv)
        self._db.flush()
        return mv

    def get_all(self, task: str) -> list[ModelVersion]:
        return (
            self._db.query(ModelVersion)
            .filter(ModelVersion.task == task)
            .order_by(ModelVersion.trained_at.desc())
            .all()
        )

    def get_active(self, task: str) -> Optional[ModelVersion]:
        return (
            self._db.query(ModelVersion)
            .filter(ModelVersion.task == task, ModelVersion.is_active == True)
            .first()
        )

    def activate(self, task: str, version: str) -> bool:
        # Deactivate all
        self._db.query(ModelVersion).filter(ModelVersion.task == task).update(
            {ModelVersion.is_active: False}
        )
        # Activate target
        mv = (
            self._db.query(ModelVersion)
            .filter(ModelVersion.task == task, ModelVersion.version == version)
            .first()
        )
        if mv:
            mv.is_active = True
            self._db.flush()
            return True
        return False

    def next_version_string(self, task: str) -> str:
        """Return the next version string e.g. 'v003'."""
        versions = self.get_all(task)
        return f"v{len(versions) + 1:03d}"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset image repository (Database-backed dataset management)
# ──────────────────────────────────────────────────────────────────────────────

class DatasetImageRepository:
    def __init__(self, session: Session) -> None:
        self._db = session

    def add_image(
        self,
        task: str,
        label: str,
        image_path: str,
        split: str = "train",
        image_hash: Optional[str] = None,
        bboxes_json: Optional[str] = None,
        source: str = "manual",
        is_verified: bool = False,
    ) -> DatasetImage:
        # Check duplicate hash per task if provided
        if image_hash:
            existing = self.get_by_hash(image_hash, task=task)
            if existing:
                return existing

        img = DatasetImage(
            id=str(uuid.uuid4()),
            task=task,
            label=label,
            split=split,
            image_path=image_path,
            image_hash=image_hash,
            bboxes_json=bboxes_json,
            source=source,
            is_verified=is_verified,
            created_at=time.time(),
        )
        self._db.add(img)
        self._db.flush()
        return img

    def get_by_id(self, image_id: str) -> Optional[DatasetImage]:
        return self._db.query(DatasetImage).filter(DatasetImage.id == image_id).first()

    def get_by_hash(self, image_hash: str, task: Optional[str] = None) -> Optional[DatasetImage]:
        q = self._db.query(DatasetImage).filter(DatasetImage.image_hash == image_hash)
        if task:
            q = q.filter(DatasetImage.task == task)
        return q.first()

    def get_counts(self, task: Optional[str] = None) -> dict[str, int]:
        """Return image counts grouped by label."""
        query = self._db.query(DatasetImage.label)
        if task:
            query = query.filter(DatasetImage.task == task)
        from sqlalchemy import func
        counts = (
            self._db.query(DatasetImage.label, func.count(DatasetImage.id))
            .filter(DatasetImage.task == task if task else True)
            .group_by(DatasetImage.label)
            .all()
        )
        return {label: count for label, count in counts}

    def count(self, task: Optional[str] = None, label: Optional[str] = None) -> int:
        q = self._db.query(DatasetImage)
        if task:
            q = q.filter(DatasetImage.task == task)
        if label:
            q = q.filter(DatasetImage.label == label)
        return q.count()

    def list_images(
        self,
        task: Optional[str] = None,
        split: Optional[str] = None,
        label: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DatasetImage]:
        q = self._db.query(DatasetImage)
        if task:
            q = q.filter(DatasetImage.task == task)
        if split:
            q = q.filter(DatasetImage.split == split)
        if label:
            q = q.filter(DatasetImage.label == label)
        return q.order_by(DatasetImage.created_at.desc()).offset(offset).limit(limit).all()

    def delete(self, image_id: str) -> bool:
        img = self.get_by_id(image_id)
        if img:
            self._db.delete(img)
            self._db.flush()
            return True
        return False

    def verify(
        self,
        image_id: str,
        label: Optional[str] = None,
        bboxes_json: Optional[str] = None,
    ) -> bool:
        img = self.get_by_id(image_id)
        if img:
            img.is_verified = True
            if label:
                img.label = label
            if bboxes_json is not None:
                img.bboxes_json = bboxes_json
            self._db.flush()
            return True
        return False
