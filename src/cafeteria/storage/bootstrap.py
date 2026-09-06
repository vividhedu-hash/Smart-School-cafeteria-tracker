"""
First-run storage bootstrap.

The schema lives in models.py; this module is what actually creates
``database/cafeteria.db``, health-checks it, and copies on-disk enrollments
into the Person table. Safe to call from the launcher, the engine, and every
dashboard page — idempotent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cafeteria.storage.database import db_health_check, get_session, init_db
from cafeteria.utils.logging import get_logger

logger = get_logger("storage.bootstrap")


def ensure_runtime_storage(cfg, *, sync_enrollments: bool = True) -> dict[str, Any]:
    """
    Create the SQLite file + tables, optionally sync the enrollment gallery.

    Args:
        cfg: Settings object (needs project_root, storage.database,
             recognition.embedding_dir).
        sync_enrollments: When True, ensure every on-disk person with images
            or an embedding has a Person row.

    Returns:
        Dict with db_path, created, ok, persons_synced.
    """
    db_url = getattr(cfg.storage, "database_url", None)
    if db_url:
        existed = True
        init_db(database_url=db_url)
        target_disp = db_url
    else:
        db_path = Path(cfg.project_root) / cfg.storage.database
        existed = db_path.exists()
        init_db(db_path=db_path)
        target_disp = str(db_path)

    ok = db_health_check()
    synced = 0
    if sync_enrollments:
        synced = _sync_enrollments(cfg)
    if not existed:
        logger.info("Created database at %s", target_disp)
    elif not ok:
        logger.error("Database health check failed at %s", target_disp)
    return {
        "db_path": target_disp,
        "created": not existed,
        "ok": ok,
        "persons_synced": synced,
    }


def _sync_enrollments(cfg) -> int:
    from cafeteria.recognition.enrollment import EnrollmentManager

    enrollment_dir = Path(cfg.project_root) / cfg.recognition.embedding_dir
    mgr = EnrollmentManager(enrollment_dir=enrollment_dir)
    session = get_session()
    try:
        created = mgr.sync_to_database(session)
        session.commit()
        if created:
            logger.info("Synced %d on-disk enrollment(s) into Person table", created)
        return created
    except Exception as exc:
        session.rollback()
        logger.warning("Enrollment sync skipped: %s", exc)
        return 0
    finally:
        session.close()


def resolve_person_fk(session, person_id: Optional[str], person_name: Optional[str] = None):
    """
    Return the Person.id for a person_id, creating a row if the identity is known
    but not yet in SQLite (e.g. enrolled after the last bootstrap).
    """
    if not person_id:
        return None
    from cafeteria.storage.repositories import PersonRepository

    repo = PersonRepository(session)
    person = repo.get_by_person_id(person_id)
    if person is None:
        person = repo.create(
            person_id=person_id,
            name=(person_name or person_id),
        )
    elif person_name and person.name in (None, "", person.person_id) and person_name != person_id:
        person.name = person_name
    return person.id
