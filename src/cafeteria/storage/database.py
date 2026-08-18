"""
Database engine setup and session factory.

Uses SQLite with WAL (Write-Ahead Logging) mode for concurrent read access
from the dashboard process without blocking the inference engine writes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from cafeteria.storage.models import Base
from cafeteria.utils.logging import get_logger

logger = get_logger("storage.database")

_engine = None
_SessionLocal = None


def _apply_pragmas(dbapi_connection, connection_record):
    """Apply SQLite performance and reliability PRAGMAs on every new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-32000")  # 32 MB cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA mmap_size=268435456")  # 256 MB memory-mapped I/O
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db(db_path: str | Path) -> None:
    """
    Initialize the SQLite database and create all tables.

    Safe to call multiple times (idempotent).

    Args:
        db_path: Path to the SQLite database file.
    """
    global _engine, _SessionLocal

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection_str = f"sqlite:///{db_path.as_posix()}"

    _engine = create_engine(
        connection_str,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
        echo=False,
        pool_pre_ping=True,
    )

    # Apply WAL mode and other pragmas on every connection
    event.listen(_engine, "connect", _apply_pragmas)

    # Create all tables
    Base.metadata.create_all(bind=_engine)

    _SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    logger.info("Database initialized at %s", db_path)


def get_session() -> Session:
    """
    Return a new database session.

    Caller is responsible for calling session.close() or using as context manager::

        with get_session() as session:
            ...
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()


def get_session_context() -> Generator[Session, None, None]:
    """
    Context-manager session for use in with-statements.

    Commits on success, rolls back on exception.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def db_health_check() -> bool:
    """Return True if the database is accessible."""
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False
