"""
Database engine setup and session factory.

Start-up order:
  1. Always boot on SQLite immediately (no network dependency → app always loads).
  2. Attempt PostgreSQL/Supabase connection in a background thread.
  3. If PG succeeds within 8 s, transparently swap the engine.

This makes the dashboard load on Streamlit Cloud even when Supabase is
temporarily unreachable, avoiding the OperationalError red-screen.
"""
from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from cafeteria.storage.models import Base
from cafeteria.utils.logging import get_logger

logger = get_logger("storage.database")

_engine = None
_SessionLocal = None
_pg_url: Optional[str] = None          # canonical PG URL after normalisation
_pg_connected: bool = False            # True once PG engine is live
_pg_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_pragmas(dbapi_connection, connection_record):
    """Apply SQLite performance and reliability PRAGMAs on every new connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-32000")   # 32 MB
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.execute("PRAGMA mmap_size=268435456") # 256 MB mmap
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _normalise_supabase_url(url: str) -> str:
    """
    Translate a direct Supabase host to the IPv4 Supavisor pooler.

    Cloud runtimes (Streamlit Cloud, AWS Lambda, GitHub Actions) only have
    IPv4 egress, so ``db.<ref>.supabase.co`` (IPv6-only) must become
    ``aws-0-ap-south-1.pooler.supabase.com:5432`` with the project-ref
    appended to the user name (``postgres.<ref>``).
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    m = re.search(r"@db\.([a-z0-9]+)\.supabase\.co:(\d+)", url)
    if m:
        ref, port = m.group(1), m.group(2)
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
            logger.info("Supabase URL normalised → IPv4 Supavisor pooler")
    return url


def _make_sqlite_engine(db_path: Optional[Path] = None):
    """Build and return a WAL-mode SQLite engine."""
    target = db_path if db_path else Path("database/cafeteria.db")
    target = Path(target)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(
            f"sqlite:///{target.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False,
            pool_pre_ping=True,
        )
        event.listen(eng, "connect", _apply_pragmas)
        Base.metadata.create_all(bind=eng)
        logger.info("SQLite engine ready at %s", target)
        return eng
    except OSError:
        # Read-only filesystem fallback (e.g. Streamlit Cloud /mount/src)
        tmp_target = Path("/tmp") / target.name
        tmp_target.parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(
            f"sqlite:///{tmp_target.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False,
            pool_pre_ping=True,
        )
        event.listen(eng, "connect", _apply_pragmas)
        Base.metadata.create_all(bind=eng)
        logger.info("SQLite engine ready at %s (fallback from read-only fs)", tmp_target)
        return eng


def _try_postgres_background(url: str) -> None:
    """
    Attempt PostgreSQL connection in a background thread.
    On success, swap the global engine transparently.
    """
    global _engine, _SessionLocal, _pg_connected
    try:
        pg_eng = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
            echo=False,
        )
        # Real connection test
        with pg_eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        Base.metadata.create_all(bind=pg_eng)

        with _pg_lock:
            _engine = pg_eng
            _SessionLocal = sessionmaker(
                bind=pg_eng, autocommit=False, autoflush=False, expire_on_commit=False
            )
            _pg_connected = True
        logger.info("PostgreSQL engine live — swapped from SQLite ✓")
    except Exception as exc:
        logger.warning("PostgreSQL connection failed (%s). Staying on SQLite.", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def init_db(
    db_path: str | Path | None = None,
    database_url: Optional[str] = None,
) -> None:
    """
    Initialise the database engine.

    Always starts on SQLite so the app loads instantly.
    If a PostgreSQL URL is provided (or found in DATABASE_URL env var),
    connection is attempted in a daemon thread; on success the engine swaps.

    Safe to call multiple times (idempotent).
    """
    global _engine, _SessionLocal, _pg_url

    # 1. Collect the PG URL (from arg → env)
    url = database_url
    if not url and db_path is None:
        url = os.environ.get("DATABASE_URL")

    if url and (url.startswith("postgresql://") or url.startswith("postgres://")):
        url = _normalise_supabase_url(url)
        _pg_url = url

    # 2. Spin up SQLite immediately so the app is never blocked by network
    # If PG is already connected, preserve it.
    if not _pg_connected:
        sqlite_path = Path(db_path) if db_path else None
        sqlite_eng = _make_sqlite_engine(sqlite_path)

        with _pg_lock:
            if not _pg_connected:
                _engine = sqlite_eng
                _SessionLocal = sessionmaker(
                    bind=sqlite_eng, autocommit=False, autoflush=False, expire_on_commit=False
                )

    # 3. Fire PG attempt in background (non-blocking, skipped in tests)
    if _pg_url and not _pg_connected and "PYTEST_CURRENT_TEST" not in os.environ:
        t = threading.Thread(
            target=_try_postgres_background, args=(_pg_url,), daemon=True
        )
        t.start()


def get_session() -> Session:
    """
    Return a new database session (SQLite or PostgreSQL, whichever is live).

    Caller is responsible for calling session.close() or using as context manager::

        with get_session() as session:
            ...
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    with _pg_lock:
        factory = _SessionLocal
    return factory()


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


def is_postgres() -> bool:
    """Return True if the live engine is PostgreSQL (not SQLite fallback)."""
    return _pg_connected
