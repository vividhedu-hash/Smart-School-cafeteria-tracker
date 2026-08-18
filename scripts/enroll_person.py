"""
scripts/enroll_person.py — Command-line enrollment tool.

Usage:
    python scripts/enroll_person.py --id person_01 --name "Alice"

Captures images from webcam and generates embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import cv2
import numpy as np

from cafeteria.config.settings import load_settings
from cafeteria.recognition.face_engine import FaceEngine
from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import PersonRepository
from cafeteria.utils.logging import setup_logging, get_logger

logger = get_logger("scripts.enroll_person")


def main():
    parser = argparse.ArgumentParser(description="Enroll a person for face recognition")
    parser.add_argument("--id",   required=True, help="Person ID (e.g. person_01)")
    parser.add_argument("--name", required=True, help="Display name")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--images", type=int, default=10, help="Number of images to capture")
    args = parser.parse_args()

    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    setup_logging(log_level=cfg.application.log_level, log_dir=_project_root / "logs")
    init_db(cfg.project_root / cfg.storage.database)

    enrollment_mgr = EnrollmentManager(
        enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
    )

    print(f"\n=== Enrolling: {args.id} ({args.name}) ===")
    print(f"Target: {args.images} images")
    print("Controls: SPACE = capture | Q = quit and generate embeddings")
    print()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    captured = 0
    while captured < args.images:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Person: {args.id} ({args.name})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {captured}/{args.images}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture  Q=done",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = enrollment_mgr.save_image(args.id, frame)
            captured += 1
            print(f"  Saved: {path.name}  ({captured}/{args.images})")
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured == 0:
        print("No images captured. Exiting.")
        sys.exit(0)

    print(f"\nCaptured {captured} images. Generating embeddings…")

    # Load face engine and generate
    try:
        face_engine = FaceEngine(
            model_pack=cfg.recognition.model_pack,
            model_dir=cfg.project_root / "models" / "face",
            det_size=tuple(cfg.recognition.det_size),
            device=cfg.device,
        )
        face_engine.load()

        emgr = EnrollmentManager(
            enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
            face_engine=face_engine,
            min_face_size=cfg.recognition.minimum_face_size,
        )
        emb = emgr.generate_embedding(args.id, args.name)

        if emb is not None:
            # Update DB
            session = get_session()
            repo = PersonRepository(session)
            existing = repo.get_by_person_id(args.id)
            if not existing:
                repo.create(person_id=args.id, name=args.name,
                            embedding_path=str(emgr.embedding_path(args.id)))
            else:
                repo.update_embedding_path(args.id, str(emgr.embedding_path(args.id)))
                repo.update_image_count(args.id, captured)
            session.commit()
            session.close()
            print(f"\n✅ Enrollment complete! Embedding saved.")
            print(f"   Shape: {emb.shape}")
        else:
            print("❌ No valid faces found. Re-run with better lighting.")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
