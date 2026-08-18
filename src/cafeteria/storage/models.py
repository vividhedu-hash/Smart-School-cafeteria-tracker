"""
SQLAlchemy ORM models for the cafeteria database.

All timestamps stored as Unix epoch floats (UTC) for timezone safety.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Integer, String, Text, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations (stored as strings in DB for readability)
# ──────────────────────────────────────────────────────────────────────────────

class WasteStatus(str, Enum):
    EMPTY         = "EMPTY"
    LOW_WASTE     = "LOW_WASTE"
    MEDIUM_WASTE  = "MEDIUM_WASTE"
    HIGH_WASTE    = "HIGH_WASTE"
    UNKNOWN       = "UNKNOWN"


class TransactionStatus(str, Enum):
    AUTO_CONFIRMED    = "AUTO_CONFIRMED"
    REVIEW_REQUIRED   = "REVIEW_REQUIRED"
    MANUALLY_CONFIRMED = "MANUALLY_CONFIRMED"
    REJECTED          = "REJECTED"


# ──────────────────────────────────────────────────────────────────────────────
# Person (enrolled individuals)
# ──────────────────────────────────────────────────────────────────────────────

class Person(Base):
    __tablename__ = "persons"

    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    person_id     = Column(String, unique=True, nullable=False, index=True)  # e.g. "person_01"
    name          = Column(String, nullable=False)
    enrolled_at   = Column(Float, nullable=False, default=time.time)
    image_count   = Column(Integer, default=0)
    embedding_path = Column(String, nullable=True)  # path to .npy embedding file
    is_active     = Column(Boolean, default=True)
    notes         = Column(Text, nullable=True)

    transactions = relationship("Transaction", back_populates="person", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Person {self.person_id!r} name={self.name!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# Transaction (one per waste event)
# ──────────────────────────────────────────────────────────────────────────────

class Transaction(Base):
    __tablename__ = "transactions"

    id                    = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id        = Column(String, unique=True, nullable=False, index=True)
    timestamp             = Column(Float, nullable=False, index=True)  # event detection time

    # Person info
    person_id_fk          = Column(String, ForeignKey("persons.id"), nullable=True)
    person_id             = Column(String, nullable=True)  # denormalized for quick display
    person_name           = Column(String, nullable=True)

    # Detection results
    plate_detected        = Column(Boolean, default=False)
    food_present          = Column(Boolean, default=False)
    waste_status          = Column(String, nullable=True)  # WasteStatus value
    plate_confidence      = Column(Float, nullable=True)
    waste_confidence      = Column(Float, nullable=True)
    face_confidence       = Column(Float, nullable=True)   # similarity score

    # Status
    status                = Column(String, nullable=False,
                                   default=TransactionStatus.REVIEW_REQUIRED)

    # Evidence
    event_image_path      = Column(String, nullable=True)  # one-camera: combined image
    face_image_path       = Column(String, nullable=True)  # future: separate face cam
    plate_image_path      = Column(String, nullable=True)  # future: separate plate cam

    # Timing
    processing_latency_ms = Column(Float, nullable=True)
    created_at            = Column(Float, nullable=False, default=time.time)

    # Notes / reason for review
    review_reason         = Column(String, nullable=True)

    person = relationship("Person", back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_timestamp_status", "timestamp", "status"),
        Index("ix_transactions_person_id", "person_id"),
        Index("ix_transactions_waste_status", "waste_status"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction {self.transaction_id!r} "
            f"person={self.person_id!r} "
            f"waste={self.waste_status!r} "
            f"status={self.status!r}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Review queue entry
# ──────────────────────────────────────────────────────────────────────────────

class ReviewEntry(Base):
    __tablename__ = "review_queue"

    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(String, ForeignKey("transactions.transaction_id"),
                            nullable=False, unique=True)
    created_at     = Column(Float, nullable=False, default=time.time)
    resolved_at    = Column(Float, nullable=True)

    # Candidate identities returned by face recognition (stored as JSON string)
    candidates_json = Column(Text, nullable=True)  # [{"person_id": ..., "score": ...}, ...]

    # Review decision
    resolved_person_id   = Column(String, nullable=True)
    resolved_person_name = Column(String, nullable=True)
    resolved_by          = Column(String, nullable=True)  # "manual" or "auto"
    is_resolved          = Column(Boolean, default=False, index=True)

    event_image_path = Column(String, nullable=True)
    waste_status     = Column(String, nullable=True)
    waste_confidence = Column(Float, nullable=True)
    reason           = Column(String, nullable=True)

    def __repr__(self) -> str:
        return f"<ReviewEntry tx={self.transaction_id!r} resolved={self.is_resolved}>"


# ──────────────────────────────────────────────────────────────────────────────
# Model version registry
# ──────────────────────────────────────────────────────────────────────────────

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    task         = Column(String, nullable=False, index=True)    # "plate" | "waste"
    version      = Column(String, nullable=False)                # "v001"
    weights_path = Column(String, nullable=False)
    config_path  = Column(String, nullable=True)
    metrics_path = Column(String, nullable=True)
    is_active    = Column(Boolean, default=False, index=True)
    trained_at   = Column(Float, nullable=False, default=time.time)

    # Training summary (denormalized for display)
    epochs       = Column(Integer, nullable=True)
    map50        = Column(Float, nullable=True)
    map50_95     = Column(Float, nullable=True)
    precision    = Column(Float, nullable=True)
    recall       = Column(Float, nullable=True)
    dataset_size = Column(Integer, nullable=True)
    notes        = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_model_versions_task_version", "task", "version", unique=True),
    )

    def __repr__(self) -> str:
        return f"<ModelVersion task={self.task!r} version={self.version!r} active={self.is_active}>"
