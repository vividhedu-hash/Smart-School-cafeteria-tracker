"""
scripts/train_from_web.py — Download public web images of plates and food,
auto-label bounding boxes, populate datasets, and train YOLO models.

Usage:
    python scripts/train_from_web.py --epochs 15
    python scripts/train_from_web.py --task plate --epochs 20
    python scripts/train_from_web.py --task waste --epochs 20
    python scripts/train_from_web.py --task both --epochs 15
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root / "dashboard"))

import cv2
import numpy as np

from cafeteria.config.settings import load_settings
from cafeteria.detection.visual import detect_plates_visual, classify_waste_visual
from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES
from cafeteria.training.registry import ModelRegistry
from cafeteria.training.trainer import PlateModelTrainer, WasteModelTrainer
from cafeteria.utils.logging import get_logger, setup_logging

logger = get_logger("scripts.train_from_web")

# Queries to search on Wikimedia Commons for diverse web images
WEB_QUERIES = [
    # EMPTY PLATES
    {"query": "empty white plate", "category": "EMPTY", "limit": 6},
    {"query": "empty ceramic plate", "category": "EMPTY", "limit": 6},
    {"query": "clean dinner plate", "category": "EMPTY", "limit": 6},

    # LOW WASTE (small crumbs / bone leftovers / small scrap)
    {"query": "plate with crumbs", "category": "LOW_WASTE", "limit": 6},
    {"query": "plate with food residue", "category": "LOW_WASTE", "limit": 6},
    {"query": "finished meal plate leftovers", "category": "LOW_WASTE", "limit": 6},

    # MEDIUM WASTE (half-eaten meal, moderate leftovers)
    {"query": "half eaten meal plate", "category": "MEDIUM_WASTE", "limit": 6},
    {"query": "leftovers on plate", "category": "MEDIUM_WASTE", "limit": 6},
    {"query": "sandwich on plate", "category": "MEDIUM_WASTE", "limit": 6},
    {"query": "portion of food on plate", "category": "MEDIUM_WASTE", "limit": 6},

    # HIGH WASTE (full cafeteria meal, full plate, buffet)
    {"query": "full plate food dinner", "category": "HIGH_WASTE", "limit": 6},
    {"query": "cafeteria meal plate", "category": "HIGH_WASTE", "limit": 6},
    {"query": "pasta on plate", "category": "HIGH_WASTE", "limit": 6},
    {"query": "indian thali food plate", "category": "HIGH_WASTE", "limit": 6},
]


def search_wikimedia_images(query: str, limit: int = 5) -> list[str]:
    """Search Wikimedia Commons API for images matching a query."""
    import urllib.parse

    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json&generator=search"
        f"&gsrnamespace=6&gsrsearch={urllib.parse.quote(query)}&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|size|mime"
    )
    req = urllib.request.Request(
        api_url,
        headers={"User-Agent": "SmartCafeteriaWaste/1.0 (academic; machine_learning)"},
    )
    urls = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for _, pdata in pages.items():
                info = pdata.get("imageinfo", [{}])[0]
                img_url = info.get("url")
                mime = info.get("mime", "")
                if img_url and "image/" in mime and not img_url.lower().endswith((".svg", ".tif", ".tiff", ".gif")):
                    urls.append(img_url)
    except Exception as exc:
        logger.warning("Wikimedia search failed for '%s': %s", query, exc)
    return urls


def download_image(url: str, timeout: float = 12.0) -> np.ndarray | None:
    """Fetch an image over HTTP and return as a BGR numpy array."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CafeteriaWasteTracker/1.0 (academic research; prototype)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            arr = np.asarray(bytearray(data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
    except Exception as exc:
        logger.warning("Download failed for %s: %s", url, exc)
        return None


def generate_augmented_samples(img: np.ndarray, num_variations: int = 3) -> list[np.ndarray]:
    """Generate subtle realistic augmentations (flips, slight rotations, brightness)."""
    samples = [img]
    h, w = img.shape[:2]

    # Horizontal flip
    samples.append(cv2.flip(img, 1))

    # Slight brightness adjustments
    for alpha, beta in [(1.15, 10), (0.85, -10)]:
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        samples.append(adjusted)

    # Slight rotation (+5 deg, -5 deg)
    for angle in [4.0, -4.0]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        samples.append(rotated)

    return samples[: num_variations + 1]


def populate_web_datasets(dm: DatasetManager) -> dict[str, int]:
    """Download images, apply visual detector for bbox auto-labels, and store in datasets."""
    print("\n🌐 [1/3] Searching and downloading web images of plates and food…")

    plate_added = 0
    waste_added = 0
    total_downloaded = 0

    for query_item in WEB_QUERIES:
        query = query_item["query"]
        category = query_item["category"]
        limit = query_item.get("limit", 5)

        print(f"  🔍 Query: '{query}' ({category})…")
        urls = search_wikimedia_images(query, limit=limit)
        if not urls:
            print("    ⚠️ No results from search API, moving to next.")
            continue

        for url_idx, url in enumerate(urls):
            total_downloaded += 1
            print(f"    [{url_idx+1}/{len(urls)}] Fetching: {url.split('/')[-1][:30]}…", end=" ")

            raw_img = download_image(url)
            if raw_img is None or raw_img.size == 0:
                print("❌ failed/skipped")
                continue

            # Resize to standard height while preserving aspect ratio
            h, w = raw_img.shape[:2]
            target_w = 640
            target_h = int(h * (target_w / w)) if w > 0 else 640
            resized = cv2.resize(raw_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Generate augmentations to give YOLO sufficient diversity
            variations = generate_augmented_samples(resized, num_variations=2)

            for var_idx, sample in enumerate(variations):
                sh, sw = sample.shape[:2]

                # 1. Add to Waste Classification Dataset
                temp_waste_path = Path(f"/tmp/waste_{total_downloaded}_{var_idx}.jpg")
                cv2.imwrite(str(temp_waste_path), sample)
                added_w, _ = dm.add_waste_images(category, [temp_waste_path])
                temp_waste_path.unlink(missing_ok=True)
                waste_added += added_w

                # 2. Detect Plate Bounding Box with Visual Cascade
                boxes = detect_plates_visual(sample)
                yolo_boxes = []

                if boxes:
                    for b in boxes:
                        bx1, by1, bx2, by2 = getattr(b, "x1", 0), getattr(b, "y1", 0), getattr(b, "x2", sw), getattr(b, "y2", sh)
                        bw = (bx2 - bx1) / sw
                        bh = (by2 - by1) / sh
                        bcx = (bx1 + bx2) / (2.0 * sw)
                        bcy = (by1 + by2) / (2.0 * sh)
                        yolo_boxes.append((bcx, bcy, bw, bh))
                else:
                    yolo_boxes.append((0.5, 0.5, 0.70, 0.70))

                # Save plate image & label
                temp_plate_img = Path(f"/tmp/plate_{total_downloaded}_{var_idx}.jpg")
                temp_plate_lbl = Path(f"/tmp/plate_{total_downloaded}_{var_idx}.txt")
                cv2.imwrite(str(temp_plate_img), sample)

                lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in yolo_boxes]
                temp_plate_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")

                if dm.add_plate_image(temp_plate_img, temp_plate_lbl):
                    plate_added += 1

                temp_plate_img.unlink(missing_ok=True)
                temp_plate_lbl.unlink(missing_ok=True)

            print(f"✅ added {len(variations)} samples")

    stats = {
        "plate_samples": plate_added,
        "waste_samples": waste_added,
        "waste_counts": dm.waste_class_counts(),
        "plate_stats": dm.plate_dataset_stats(),
    }
    print(f"\n📊 Dataset summary:")
    print(f"  • Plate images : {stats['plate_stats']['images']} (Labels: {stats['plate_stats']['labels']})")
    print(f"  • Waste images : {stats['waste_counts']}")
    return stats


def train_plate_detector(cfg, epochs: int) -> Path | None:
    """Train YOLOv8-det on plate detection dataset and register version."""
    print(f"\n🚀 [2/3] Training YOLOv8 Plate Detector ({epochs} epochs)…")
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    version = registry.next_version("plate")

    dm = DatasetManager(cfg.project_root / cfg.storage.datasets)
    split_dir = Path("/tmp/yolo_plate_split")
    if split_dir.exists():
        shutil.rmtree(split_dir)

    data_yaml = dm.prepare_yolo_det_dataset(split_dir)

    trainer = PlateModelTrainer(
        datasets_dir=cfg.project_root / cfg.storage.datasets,
        models_output_dir=cfg.project_root / cfg.storage.models,
        device=cfg.device,
    )

    result = trainer.train(
        base_model="yolov8n.pt",
        epochs=epochs,
        image_size=640,
        batch=8,
        version=version,
        data_yaml=data_yaml,
    )

    registry.register_version(
        task="plate",
        version=result.version,
        weights_path=str(result.weights_path),
        metrics=result.metrics,
        config=result.config,
    )
    registry.activate("plate", result.version)

    # Copy to default weights location models/plate/best.pt
    dest_best = cfg.project_root / "models" / "plate" / "best.pt"
    dest_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(result.weights_path), str(dest_best))

    print(f"✅ Plate Detector Training Complete!")
    print(f"   Version : {result.version}")
    print(f"   Weights : {dest_best}")
    print(f"   mAP50   : {result.metrics.get('map50', 'N/A')}")
    return dest_best


def train_waste_classifier(cfg, epochs: int) -> Path | None:
    """Train YOLOv8-cls on waste classification dataset and register version."""
    print(f"\n🚀 [3/3] Training YOLOv8 Waste Classifier ({epochs} epochs)…")
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    version = registry.next_version("waste")

    trainer = WasteModelTrainer(
        datasets_dir=cfg.project_root / cfg.storage.datasets,
        models_output_dir=cfg.project_root / cfg.storage.models,
        device=cfg.device,
    )

    result = trainer.train(
        base_model="yolov8n-cls.pt",
        epochs=epochs,
        image_size=224,
        batch=8,
        version=version,
    )

    registry.register_version(
        task="waste",
        version=result.version,
        weights_path=str(result.weights_path),
        metrics=result.metrics,
        config=result.config,
    )
    registry.activate("waste", result.version)

    # Copy to default weights location models/waste/best.pt
    dest_best = cfg.project_root / "models" / "waste" / "best.pt"
    dest_best.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(result.weights_path), str(dest_best))

    print(f"✅ Waste Classifier Training Complete!")
    print(f"   Version : {result.version}")
    print(f"   Weights : {dest_best}")
    print(f"   Top-1 Acc : {result.metrics.get('top1_accuracy', 'N/A')}")
    return dest_best


def main():
    parser = argparse.ArgumentParser(description="Train plate & waste detectors from web images")
    parser.add_argument("--task", choices=["plate", "waste", "both"], default="both",
                        help="Which model to train (default: both)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading images if dataset already populated")
    args = parser.parse_args()

    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    setup_logging(log_level=cfg.application.log_level, log_dir=_project_root / "logs")
    dm = DatasetManager(cfg.project_root / cfg.storage.datasets)

    if not args.skip_download:
        populate_web_datasets(dm)

    if args.task in ("plate", "both"):
        train_plate_detector(cfg, epochs=args.epochs)

    if args.task in ("waste", "both"):
        train_waste_classifier(cfg, epochs=args.epochs)

    print("\n🎉 ALL TRAINING COMPLETE! Live models are active in models/ and registered in registry.json.")


if __name__ == "__main__":
    main()
