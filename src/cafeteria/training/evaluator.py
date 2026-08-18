"""
Model evaluator — runs evaluation on the held-out test set and returns
real metrics (precision, recall, mAP, confusion matrix).

Clearly separates train / val / test metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from cafeteria.utils.logging import get_logger

logger = get_logger("training.evaluator")


class WasteModelEvaluator:
    """
    Evaluate a trained waste classification model on the test split.

    Args:
        weights_path:  Path to trained .pt model.
        dataset_dir:   Directory with test/ split (prepared by DatasetManager).
        device:        Compute device.
    """

    def __init__(
        self,
        weights_path: str | Path,
        dataset_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self._weights = Path(weights_path)
        self._dataset_dir = Path(dataset_dir)
        self._device = device

    def evaluate(self) -> dict:
        """
        Run evaluation on the test split.

        Returns:
            Dict with accuracy, per-class precision/recall/f1, confusion matrix.
        """
        from ultralytics import YOLO

        if not self._weights.exists():
            raise FileNotFoundError(f"Model weights not found: {self._weights}")

        test_dir = self._dataset_dir / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test split not found at: {test_dir}")

        model = YOLO(str(self._weights))
        results = model.val(
            data=str(self._dataset_dir),
            split="test",
            device=self._device,
            verbose=False,
        )

        metrics: dict = {
            "split": "test",
            "weights": str(self._weights),
        }

        try:
            if hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics["top1_accuracy"] = round(float(rd.get("metrics/accuracy_top1", 0)), 4)
                metrics["top5_accuracy"] = round(float(rd.get("metrics/accuracy_top5", 0)), 4)
        except Exception:
            pass

        # Per-class metrics via sklearn if available
        per_class = self._per_class_metrics(model, test_dir)
        if per_class:
            metrics["per_class"] = per_class

        logger.info("Evaluation complete: %s", metrics)
        return metrics

    def _per_class_metrics(self, model, test_dir: Path) -> Optional[dict]:
        """Compute per-class precision/recall/f1 using direct inference."""
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            import cv2

            y_true, y_pred = [], []
            class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])

            for cls_idx, cls_name in enumerate(class_names):
                cls_dir = test_dir / cls_name
                for img_path in cls_dir.glob("*.*"):
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    result = model.predict(img, verbose=False)
                    if result and result[0].probs is not None:
                        pred_idx = int(result[0].probs.top1)
                        pred_name = result[0].names.get(pred_idx, str(pred_idx))
                        y_true.append(cls_name)
                        y_pred.append(pred_name)

            if not y_true:
                return None

            report = classification_report(
                y_true, y_pred, labels=class_names, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=class_names).tolist()

            return {
                "class_names": class_names,
                "report": report,
                "confusion_matrix": cm,
            }
        except Exception as exc:
            logger.warning("Per-class metrics failed: %s", exc)
            return None


class PlateModelEvaluator:
    """
    Evaluate a trained YOLO detection model for plates.

    Args:
        weights_path:  Path to trained .pt model.
        data_yaml:     Path to data.yaml with test split.
        device:        Compute device.
    """

    def __init__(
        self,
        weights_path: str | Path,
        data_yaml: str | Path,
        device: str = "cpu",
    ) -> None:
        self._weights = Path(weights_path)
        self._data_yaml = Path(data_yaml)
        self._device = device

    def evaluate(self) -> dict:
        """
        Evaluate on the test split defined in data.yaml.

        Returns:
            Dict with mAP50, mAP50-95, precision, recall.
        """
        from ultralytics import YOLO

        if not self._weights.exists():
            raise FileNotFoundError(f"Model not found: {self._weights}")
        if not self._data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {self._data_yaml}")

        model = YOLO(str(self._weights))
        results = model.val(
            data=str(self._data_yaml),
            split="test",
            device=self._device,
            verbose=False,
        )

        metrics: dict = {
            "split": "test",
            "weights": str(self._weights),
        }
        try:
            b = results.box
            metrics["map50"]     = round(float(b.map50), 4)
            metrics["map50_95"]  = round(float(b.map), 4)
            metrics["precision"] = round(float(b.mp), 4)
            metrics["recall"]    = round(float(b.mr), 4)
        except Exception:
            pass

        logger.info("Plate model evaluation: %s", metrics)
        return metrics
