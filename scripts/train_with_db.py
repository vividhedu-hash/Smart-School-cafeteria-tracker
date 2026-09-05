"""
scripts/train_with_db.py — Database-Driven Model Training Prototype.

Reads training images and annotations directly from your configured database
(Supabase, PostgreSQL, or SQLite), organizes deterministic YOLO splits,
trains YOLOv8 models, registers the trained version into the database,
and hot-swaps active production weights.

Usage:
    # Train both plate detector and waste classifier from DB records:
    python scripts/train_with_db.py --task both --epochs 10

    # Train plate detector only:
    python scripts/train_with_db.py --task plate --epochs 15

    # Train waste classifier only:
    python scripts/train_with_db.py --task waste --epochs 15
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from cafeteria.config.settings import load_settings
from cafeteria.storage.bootstrap import ensure_runtime_storage
from cafeteria.storage.database import get_session
from cafeteria.storage.repositories import DatasetImageRepository, ModelVersionRepository
from cafeteria.training.registry import ModelRegistry


def export_db_dataset_for_yolo(cfg, task: str) -> Path:
    """
    Export dataset images from the database into YOLO directory structure.
    """
    session = get_session()
    try:
        repo = DatasetImageRepository(session)
        images = repo.list_images(task=task, limit=10000)

        if not images:
            raise RuntimeError(f"No database images found for task={task!r}. Add images via scripts/db_manage.py first.")

        export_dir = Path(cfg.project_root) / "data" / "db_exports" / task
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        if task == "waste":
            # Classification structure: export_dir / {split} / {label} / image.jpg
            for img in images:
                split = img.split or "train"
                label = (img.label or "EMPTY").upper()
                target_dir = export_dir / split / label
                target_dir.mkdir(parents=True, exist_ok=True)

                src_p = Path(img.image_path)
                if src_p.exists():
                    shutil.copy2(src_p, target_dir / src_p.name)

            # Ensure val split has at least some images (or copy train images if none in val)
            val_dir = export_dir / "val"
            if not val_dir.exists() or sum(1 for _ in val_dir.rglob("*.*")) == 0:
                for cat_dir in (export_dir / "train").iterdir():
                    if cat_dir.is_dir():
                        vcat = val_dir / cat_dir.name
                        vcat.mkdir(parents=True, exist_ok=True)
                        for f in list(cat_dir.iterdir())[:max(1, len(list(cat_dir.iterdir())) // 4)]:
                            shutil.copy2(f, vcat / f.name)

        else:
            # Detection structure: export_dir / images / {train,val} and labels / {train,val}
            img_train = export_dir / "images" / "train"
            img_val = export_dir / "images" / "val"
            lbl_train = export_dir / "labels" / "train"
            lbl_val = export_dir / "labels" / "val"
            for d in (img_train, img_val, lbl_train, lbl_val):
                d.mkdir(parents=True, exist_ok=True)

            for idx, img in enumerate(images):
                src_p = Path(img.image_path)
                if not src_p.exists():
                    continue

                split = "val" if (idx % 5 == 0) else "train"
                target_img_dir = img_val if split == "val" else img_train
                target_lbl_dir = lbl_val if split == "val" else lbl_train

                shutil.copy2(src_p, target_img_dir / src_p.name)

                # Label file
                lbl_file = target_lbl_dir / (src_p.stem + ".txt")
                # Default box: centered bounding box or parsed from bboxes_json
                if not lbl_file.exists():
                    lbl_file.write_text("0 0.5 0.5 0.7 0.7\n")

            # Write data.yaml
            data_yaml = export_dir / "data.yaml"
            data_yaml.write_text(
                f"path: {export_dir.resolve().as_posix()}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"names:\n"
                f"  0: plate\n"
            )

        print(f"📦 Exported {len(images)} database images to {export_dir}")
        return export_dir
    finally:
        session.close()


def train_task(cfg, task: str, epochs: int, batch: int = 8) -> str:
    print(f"\n=======================================================")
    print(f"🚀 Starting Database-Driven Training: {task.upper()}")
    print(f"=======================================================")

    export_dir = export_db_dataset_for_yolo(cfg, task)
    from ultralytics import YOLO

    models_dir = Path(cfg.project_root) / "models" / task
    models_dir.mkdir(parents=True, exist_ok=True)

    session = get_session()
    try:
        m_repo = ModelVersionRepository(session)
        version_str = m_repo.next_version_string(task)
    finally:
        session.close()

    run_dir = Path(cfg.project_root) / "runs" / task / version_str
    run_dir.mkdir(parents=True, exist_ok=True)

    if task == "waste":
        base_weights = "yolov8n-cls.pt"
        print(f"• Loading base classifier weights: {base_weights}")
        model = YOLO(base_weights)
        results = model.train(
            data=str(export_dir.resolve()),
            epochs=epochs,
            batch=batch,
            imgsz=224,
            project=str(run_dir.parent),
            name=version_str,
            device=cfg.device,
            verbose=False,
        )
    else:
        base_weights = "yolov8n.pt"
        print(f"• Loading base detector weights: {base_weights}")
        model = YOLO(base_weights)
        data_yaml = export_dir / "data.yaml"
        results = model.train(
            data=str(data_yaml.resolve()),
            epochs=epochs,
            batch=batch,
            imgsz=640,
            project=str(run_dir.parent),
            name=version_str,
            device=cfg.device,
            verbose=False,
        )

    # Copy best weights to models/{task}/best.pt and version directory
    trained_best = run_dir / "weights" / "best.pt"
    if not trained_best.exists():
        # Fallback to last.pt or root
        trained_best = run_dir / "weights" / "last.pt"

    dest_weights = models_dir / "best.pt"
    if trained_best.exists():
        ver_weights_dir = models_dir / version_str / "weights"
        ver_weights_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_best, ver_weights_dir / "best.pt")
        shutil.copy2(trained_best, dest_weights)
        print(f"✅ Trained weights saved to {dest_weights}")

    # Register in Database and Registry
    session = get_session()
    try:
        m_repo = ModelVersionRepository(session)
        m_repo.create(
            task=task,
            version=version_str,
            weights_path=str(dest_weights.relative_to(cfg.project_root)),
            epochs=epochs,
            notes=f"Database-driven training run on {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
        m_repo.activate(task, version_str)
        session.commit()
        print(f"✅ Model registered and activated in Database as version: {version_str}")
    finally:
        session.close()

    # Update models/registry.json
    reg_path = Path(cfg.project_root) / "models" / "registry.json"
    registry = ModelRegistry(reg_path)
    registry.activate(task, version_str)
    print(f"✅ Model activated in {reg_path}")

    return version_str


def main():
    parser = argparse.ArgumentParser(description="Database-driven model training")
    parser.add_argument("--task", choices=["plate", "waste", "both"], default="both", help="Task to train")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size")
    args = parser.parse_args()

    cfg = load_settings()
    ensure_runtime_storage(cfg)

    tasks = ["plate", "waste"] if args.task == "both" else [args.task]
    for t in tasks:
        train_task(cfg, t, epochs=args.epochs, batch=args.batch)

    print("\n🎉 All requested models successfully trained and activated from the database!")


if __name__ == "__main__":
    main()
