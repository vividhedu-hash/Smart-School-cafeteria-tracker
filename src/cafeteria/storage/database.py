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


def init_db(
    db_path: str | Path | None = None,
    database_url: Optional[str] = None,
) -> None:
    """
    Initialize the database engine and create all tables.
    Supports PostgreSQL (Supabase, RDS, Neon) and SQLite WAL mode.
    Safe to call multiple times (idempotent).

    Args:
        db_path: Optional path to SQLite database file.
        database_url: Optional full SQLAlchemy connection string
                      (e.g. postgresql://user:pass@host:5432/dbname).
                      Falls back to DATABASE_URL environment variable if set.
    """
    global _engine, _SessionLocal

    url = database_url
    if not url and db_path is None:
        url = os.environ.get("DATABASE_URL")

    if url and (url.startswith("postgresql://") or url.startswith("postgres://")):
        # Fix legacy postgres:// schema if returned by some providers like Heroku/Supabase
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]

        # If direct db.<ref>.supabase.co is used, translate to IPv4 Supavisor pooler
        # because cloud runtimes (Streamlit Cloud, AWS, GitHub) lack IPv6 egress.
        import re
        m = re.search(r"@db\.([a-z0-9]+)\.supabase\.co:(\d+)", url)
        if m:
            ref = m.group(1)
            port = m.group(2)
            prefix, rest = url.split("://", 1)
            if "@" in rest:
                userpass, hostdb = rest.split("@", 1)
                if ":" in userpass:
                    user, pwd = userpass.split(":", 1)
                    if not user.endswith("." + ref):
                        user = f"{user}.{ref}"
                    userpass = f"{user}:{pwd}"
                hostdb = hostdb.replace(
                    f"db.{ref}.supabase.co:{port}",
                    "aws-0-ap-south-1.pooler.supabase.com:5432",
                )
                url = f"{prefix}://{userpass}@{hostdb}"
                logger.info("Auto-adapted direct Supabase IPv6 URL to IPv4 Supavisor pooler")

        try:
            engine = create_engine(
                url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                connect_args={"connect_timeout": 10},
                echo=False,
            )
            with engine.connect() as conn:
                pass
            Base.metadata.create_all(bind=engine)
            logger.info("Database initialized with PostgreSQL connection pool")
        except Exception as exc:
            logger.error("PostgreSQL connection failed (%s); falling back to SQLite", exc)
            engine = None
    else:
        engine = None

    if engine is None:
        # SQLite fallback or explicit SQLite path
        target_path = Path(db_path) if db_path else Path("database/cafeteria.db")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        connection_str = f"sqlite:///{target_path.as_posix()}"

        engine = create_engine(
            connection_str,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
            echo=False,
            pool_pre_ping=True,
        )
        event.listen(engine, "connect", _apply_pragmas)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized at %s", target_path)

    _engine = engine
    _SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


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
