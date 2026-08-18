"""
scripts/capture_dataset.py — Capture waste classification images from webcam.

Usage:
    python scripts/capture_dataset.py

Controls:
    SPACE = capture image
    N     = next category
    Q     = quit
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import cv2

from cafeteria.config.settings import load_settings
from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES


def main():
    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    dm = DatasetManager(cfg.project_root / cfg.storage.datasets)

    cat_idx = 0
    category = WASTE_CLASSES[cat_idx]

    cap = cv2.VideoCapture(cfg.camera.source if isinstance(cfg.camera.source, int) else 0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    print("\n=== Dataset Capture Tool ===")
    print("SPACE=capture  N=next category  Q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        counts = dm.waste_class_counts()
        display = frame.copy()

        # Overlay
        cv2.rectangle(display, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
        cv2.putText(display, f"Category: {category}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
        cv2.putText(display, f"Captured: {counts.get(category, 0)}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, "SPACE=capture  N=next  Q=quit",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        cv2.imshow("Dataset Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            import tempfile, numpy as np
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                added, _ = dm.add_waste_images(category, [Path(tmp.name)])
                Path(tmp.name).unlink(missing_ok=True)
            counts = dm.waste_class_counts()
            print(f"  Saved to {category}  (total: {counts[category]})")

        elif key in (ord("n"), ord("N")):
            cat_idx = (cat_idx + 1) % len(WASTE_CLASSES)
            category = WASTE_CLASSES[cat_idx]
            print(f"  → Category: {category}")

        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal counts:", dm.waste_class_counts())


if __name__ == "__main__":
    main()
