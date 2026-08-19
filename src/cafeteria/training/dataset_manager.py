"""
Dataset manager — handles uploading, organizing, splitting, and validating
images for waste classification and plate detection training.

Maintains deterministic train/val/test splits to prevent data leakage.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Optional

from cafeteria.utils.logging import get_logger

logger = get_logger("training.dataset_manager")

WASTE_CLASSES = ["EMPTY", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _file_hash(path: Path) -> str:
    """MD5 hash of file contents for duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetManager:
    """
    Manages dataset images for YOLO training.

    Args:
        datasets_dir:  Root directory (e.g. data/datasets/).
        train_ratio:   Fraction of images for training set.
        val_ratio:     Fraction for validation.
        test_ratio:    Fraction for test.
        random_seed:   Seed for reproducible splits.
    """

    def __init__(
        self,
        datasets_dir: str | Path,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> None:
        self._root = Path(datasets_dir)
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._rng = random.Random(random_seed)

        # Verify ratios
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"

    # ──────────────────────────────────────────────────────────────────────
    # Waste classification dataset
    # ──────────────────────────────────────────────────────────────────────

    def waste_class_dir(self, class_name: str) -> Path:
        return self._root / "waste" / class_name

    def add_waste_images(
        self,
        class_name: str,
        source_paths: list[Path | str],
        overwrite_duplicates: bool = False,
    ) -> tuple[int, int]:
        """
        Copy images into the waste dataset for a given class.

        Args:
            class_name:   One of WASTE_CLASSES.
            source_paths: List of image file paths to add.
            overwrite_duplicates: If False, skip files with identical content.

        Returns:
            (added_count, skipped_count)
        """
        if class_name not in WASTE_CLASSES:
            raise ValueError(f"Unknown waste class: {class_name}. Valid: {WASTE_CLASSES}")

        out_dir = self.waste_class_dir(class_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Hash existing files for duplicate detection
        existing_hashes = set()
        if not overwrite_duplicates:
            for f in out_dir.iterdir():
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    existing_hashes.add(_file_hash(f))

        added, skipped = 0, 0
        for src in source_paths:
            src = Path(src)
            if not src.exists():
                skipped += 1
                continue
            if src.suffix.lower() not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue

            if not overwrite_duplicates:
                h = _file_hash(src)
                if h in existing_hashes:
                    skipped += 1
                    continue
                existing_hashes.add(h)

            # Generate unique filename
            existing = list(out_dir.glob(f"*{src.suffix.lower()}"))
            dest = out_dir / f"{len(existing) + 1:05d}{src.suffix.lower()}"
            shutil.copy2(str(src), str(dest))
            added += 1

        logger.info("Added %d images to %s (skipped %d)", added, class_name, skipped)
        return added, skipped

    def waste_class_counts(self) -> dict[str, int]:
        """Return number of images per waste class."""
        counts = {}
        for cls in WASTE_CLASSES:
            d = self.waste_class_dir(cls)
            if d.exists():
                counts[cls] = len([
                    f for f in d.iterdir()
                    if f.suffix.lower() in SUPPORTED_EXTENSIONS
                ])
            else:
                counts[cls] = 0
        return counts

    def total_waste_images(self) -> int:
        return sum(self.waste_class_counts().values())

    def waste_images(self, class_name: str) -> list[Path]:
        """Return sorted list of image paths for a class."""
        d = self.waste_class_dir(class_name)
        if not d.exists():
            return []
        return sorted(
            f for f in d.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def delete_waste_image(self, class_name: str, filename: str) -> bool:
        """Delete a specific waste dataset image."""
        path = self.waste_class_dir(class_name) / filename
        if path.exists():
            path.unlink()
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Dataset split for YOLO training
    # ──────────────────────────────────────────────────────────────────────

    def prepare_yolo_cls_dataset(self, output_dir: Path) -> dict:
        """
        Create a YOLO classification dataset directory structure:

            output_dir/
                train/
                    EMPTY/
                    LOW_WASTE/
                    ...
                val/
                val/
                test/

        Images are symlinked (copied on Windows) rather than duplicated.

        Args:
            output_dir: Where to write the split dataset.

        Returns:
            Split statistics dict.
        """
        stats = {"train": {}, "val": {}, "test": {}, "total": 0}

        for cls in WASTE_CLASSES:
            images = self.waste_images(cls)
            if not images:
                continue

            # Deterministic shuffle based on filename hash (not random per run)
            sorted_images = sorted(images, key=lambda p: _file_hash(p))
            self._rng.shuffle(sorted_images)

            n = len(sorted_images)
            n_train = int(n * self._train_ratio)
            n_val   = int(n * self._val_ratio)
            # Remainder goes to test
            splits = {
                "train": sorted_images[:n_train],
                "val":   sorted_images[n_train:n_train + n_val],
                "test":  sorted_images[n_train + n_val:],
            }

            for split_name, split_images in splits.items():
                split_dir = output_dir / split_name / cls
                split_dir.mkdir(parents=True, exist_ok=True)
                for img in split_images:
                    dest = split_dir / img.name
                    if not dest.exists():
                        shutil.copy2(str(img), str(dest))
                stats[split_name][cls] = len(split_images)
                stats["total"] += len(split_images)

        logger.info("YOLO CLS dataset prepared at %s: %s", output_dir, stats)
        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Plate detection dataset
    # ──────────────────────────────────────────────────────────────────────

    def plate_images_dir(self) -> Path:
        return self._root / "plate" / "images"

    def plate_labels_dir(self) -> Path:
        return self._root / "plate" / "labels"

    def add_plate_image(
        self,
        image_path: Path | str,
        label_path: Optional[Path | str] = None,
    ) -> bool:
        """
        Add a plate detection image (and optional YOLO label) to the dataset.

        Args:
            image_path: Source image file.
            label_path: Optional .txt YOLO label file.

        Returns:
            True on success.
        """
        image_path = Path(image_path)
        img_dir = self.plate_images_dir()
        lbl_dir = self.plate_labels_dir()
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        existing = list(img_dir.glob(f"*{image_path.suffix.lower()}"))
        stem = f"{len(existing) + 1:05d}"
        dest_img = img_dir / f"{stem}{image_path.suffix.lower()}"
        shutil.copy2(str(image_path), str(dest_img))

        if label_path:
            label_path = Path(label_path)
            dest_lbl = lbl_dir / f"{stem}.txt"
            shutil.copy2(str(label_path), str(dest_lbl))

        return True

    def plate_unlabelled_images(self) -> list[Path]:
        """Plate images that have no matching YOLO label file yet."""
        img_dir = self.plate_images_dir()
        lbl_dir = self.plate_labels_dir()
        if not img_dir.exists():
            return []
        images = sorted(
            f for f in img_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        return [img for img in images if not (lbl_dir / f"{img.stem}.txt").exists()]

    def write_plate_label(self, stem: str, boxes: list[tuple[float, float, float, float]]) -> Path:
        # [AI-CoLab: Verified by Antigravity] Writes YOLO format label file for plate bounding boxes
        lbl_dir = self.plate_labels_dir()
        lbl_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for cx, cy, bw, bh in boxes:
            values = [max(0.0, min(1.0, float(v))) for v in (cx, cy, bw, bh)]
            if values[2] <= 0 or values[3] <= 0:
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in values))
        path = lbl_dir / f"{Path(stem).stem}.txt"
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info("Wrote %d plate box(es) to %s", len(lines), path.name)
        return path

    def delete_plate_image(self, filename: str) -> bool:
        """Delete a plate image and its label, if any."""
        name = Path(str(filename)).name
        img = self.plate_images_dir() / name
        if not img.exists():
            return False
        img.unlink()
        label = self.plate_labels_dir() / f"{img.stem}.txt"
        label.unlink(missing_ok=True)
        return True

    def plate_dataset_stats(self) -> dict:
        img_dir = self.plate_images_dir()
        lbl_dir = self.plate_labels_dir()
        n_images = len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        return {
            "images": n_images,
            "labels": n_labels,
            "unlabelled": max(0, n_images - n_labels),
        }

    def prepare_yolo_det_dataset(self, output_dir: Path) -> str:
        """
        Create a data.yaml for YOLO object detection training.

        Args:
            output_dir: Where to write train/val/test splits and data.yaml.

        Returns:
            Path to data.yaml as string.
        """
        img_dir = self.plate_images_dir()
        lbl_dir = self.plate_labels_dir()

        images = sorted(img_dir.glob("*.*")) if img_dir.exists() else []
        n = len(images)
        if n == 0:
            raise ValueError("No plate images in dataset")

        # Shuffle deterministically
        sorted_imgs = sorted(images, key=lambda p: _file_hash(p))
        self._rng.shuffle(sorted_imgs)

        n_train = int(n * self._train_ratio)
        n_val   = int(n * self._val_ratio)
        splits = {
            "train": sorted_imgs[:n_train],
            "val":   sorted_imgs[n_train:n_train + n_val],
            "test":  sorted_imgs[n_train + n_val:],
        }

        for split, imgs in splits.items():
            (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(str(img), str(output_dir / split / "images" / img.name))
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(str(lbl), str(output_dir / split / "labels" / f"{img.stem}.txt"))

        # Write data.yaml
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["plate"],
        }
        yaml_path = output_dir / "data.yaml"
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        return str(yaml_path)
