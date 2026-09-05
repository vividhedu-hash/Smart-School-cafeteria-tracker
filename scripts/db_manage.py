"""
scripts/db_manage.py — Enterprise Database & Dataset Management CLI.

Usage:
    # 1. Test database connectivity (Supabase, PostgreSQL, SQLite)
    python scripts/db_manage.py --test-connection

    # 2. Display database health and dataset statistics
    python scripts/db_manage.py --stats

    # 3. Add an image directly to the database
    python scripts/db_manage.py --add-image /path/to/plate.jpg --task plate --label plate --split train

    # 4. Add a waste image with category
    python scripts/db_manage.py --add-image /path/to/food.jpg --task waste --label EMPTY --split train

    # 5. Bulk import a folder into the database
    python scripts/db_manage.py --import-dir data/datasets/waste/ --task waste
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from cafeteria.config.settings import load_settings
from cafeteria.storage.bootstrap import ensure_runtime_storage
from cafeteria.storage.database import db_health_check, get_session, init_db
from cafeteria.storage.repositories import (
    DatasetImageRepository, ModelVersionRepository, PersonRepository,
    ReviewRepository, TransactionRepository,
)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_connection(cfg) -> bool:
    print("\n🔍 Testing Database Connection…")
    db_url = getattr(cfg.storage, "database_url", None) or os.environ.get("DATABASE_URL")
    if db_url:
        print(f"  • Configured URL : {db_url.split('@')[-1] if '@' in db_url else db_url}")
    else:
        print(f"  • Configured DB  : {cfg.project_root / cfg.storage.database} (SQLite WAL)")

    report = ensure_runtime_storage(cfg)
    ok = db_health_check()
    if ok:
        print("  ✅ Database is connected and healthy!")
        print(f"  • Path / Host    : {report['db_path']}")
        print(f"  • Persons synced : {report.get('persons_synced', 0)}")
    else:
        print("  ❌ Database connection failed. Please check your credentials or path.")
    return ok


def print_stats(cfg) -> None:
    ensure_runtime_storage(cfg)
    session = get_session()
    try:
        p_repo = PersonRepository(session)
        t_repo = TransactionRepository(session)
        r_repo = ReviewRepository(session)
        d_repo = DatasetImageRepository(session)
        m_repo = ModelVersionRepository(session)

        print("\n📊 Database System Statistics:")
        print(f"  • Active Persons        : {len(p_repo.get_all_active())}")
        print(f"  • Total Transactions    : {len(t_repo.get_recent(limit=1000))}")
        print(f"  • Pending Review Items  : {r_repo.count_unresolved()}")
        print(f"  • Active Plate Model    : {m_repo.get_active('plate')}")
        print(f"  • Active Waste Model    : {m_repo.get_active('waste')}")

        print("\n🖼️ Dataset Images in Database:")
        plate_counts = d_repo.get_counts("plate")
        waste_counts = d_repo.get_counts("waste")
        print(f"  • Plate Detection Images : {d_repo.count(task='plate')} {plate_counts}")
        print(f"  • Waste Classifier Images: {d_repo.count(task='waste')} {waste_counts}")

        splits = {}
        for s in ("train", "val", "test"):
            splits[s] = len(d_repo.list_images(split=s, limit=10000))
        print(f"  • Dataset Splits         : {splits}")
    finally:
        session.close()


def add_single_image(
    cfg,
    image_path: str,
    task: str,
    label: str,
    split: str = "train",
    bboxes: str | None = None,
) -> None:
    p = Path(image_path).resolve()
    if not p.exists():
        print(f"❌ Error: Image file not found: {p}")
        sys.exit(1)

    ensure_runtime_storage(cfg)
    session = get_session()
    try:
        repo = DatasetImageRepository(session)
        h = _file_hash(p)
        item = repo.add_image(
            task=task,
            label=label,
            image_path=str(p),
            split=split,
            image_hash=h,
            bboxes_json=bboxes,
            source="manual",
            is_verified=True,
        )
        session.commit()
        print(f"✅ Image added to database!")
        print(f"   ID    : {item.id}")
        print(f"   Task  : {item.task} | Label: {item.label} | Split: {item.split}")
        print(f"   Path  : {item.image_path}")
    finally:
        session.close()


def bulk_import_dir(cfg, folder_path: str, task: str) -> None:
    root = Path(folder_path).resolve()
    if not root.exists():
        print(f"❌ Error: Directory not found: {root}")
        sys.exit(1)

    ensure_runtime_storage(cfg)
    session = get_session()
    added = 0
    skipped = 0
    try:
        repo = DatasetImageRepository(session)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        for img_file in root.rglob("*.*"):
            if img_file.suffix.lower() not in valid_exts:
                continue

            # Infer label from parent directory if task is waste (e.g. EMPTY, LOW_WASTE)
            if task == "waste":
                label = img_file.parent.name.upper()
                if label not in ("EMPTY", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE"):
                    label = "EMPTY"
            else:
                label = "plate"

            h = _file_hash(img_file)
            existing = repo.get_by_hash(h, task=task)
            if existing:
                skipped += 1
                continue

            repo.add_image(
                task=task,
                label=label,
                image_path=str(img_file.resolve()),
                split="train",
                image_hash=h,
                source="import",
                is_verified=True,
            )
            added += 1

        session.commit()
        print(f"\n✅ Import completed!")
        print(f"   Added   : {added} new images")
        print(f"   Skipped : {skipped} duplicates")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Manage cafeteria database and image datasets")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection")
    parser.add_argument("--stats", action="store_true", help="Print database statistics")
    parser.add_argument("--add-image", type=str, help="Path to image to add to database")
    parser.add_argument("--task", choices=["plate", "waste"], default="waste", help="Task type")
    parser.add_argument("--label", type=str, default="EMPTY", help="Class label")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Dataset split")
    parser.add_argument("--bboxes", type=str, help="Optional JSON string of bounding boxes")
    parser.add_argument("--import-dir", type=str, help="Import a directory of images")
    args = parser.parse_args()

    cfg = load_settings()

    if args.test_connection:
        test_connection(cfg)
    elif args.stats:
        print_stats(cfg)
    elif args.add_image:
        add_single_image(cfg, args.add_image, task=args.task, label=args.label, split=args.split, bboxes=args.bboxes)
    elif args.import_dir:
        bulk_import_dir(cfg, args.import_dir, task=args.task)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
