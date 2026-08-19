"""
Model trainer — wraps Ultralytics YOLO training with real metrics reporting.

Supports:
  - Waste classification (YOLOv8-cls)
  - Plate detection (YOLOv8-det)

Uses transfer learning from pretrained Ultralytics weights.
Never trains from random initialization.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from cafeteria.utils.logging import get_logger, EventCode, log_event

logger = get_logger("training.trainer")


def _epoch_payload(yolo_trainer) -> dict:
    """
    Best-effort snapshot of an in-flight Ultralytics epoch.

    Ultralytics exposes different attributes per task and version, so every
    read is guarded — progress reporting must never break a training run.
    """
    payload: dict = {}
    try:
        payload["epoch"] = int(getattr(yolo_trainer, "epoch", 0) or 0) + 1
        payload["epochs"] = int(getattr(yolo_trainer, "epochs", 0) or 0)
    except (TypeError, ValueError):
        pass

    metrics: dict = {}
    for key, value in dict(getattr(yolo_trainer, "metrics", None) or {}).items():
        try:
            metrics[str(key)] = round(float(value), 5)
        except (TypeError, ValueError):
            continue
    if metrics:
        payload["metrics"] = metrics

    loss = getattr(yolo_trainer, "tloss", None)
    if loss is not None:
        try:
            payload["loss"] = round(float(loss.mean()), 5)
        except (AttributeError, TypeError, ValueError):
            try:
                payload["loss"] = round(float(loss), 5)
            except (TypeError, ValueError):
                pass
    return payload


def _attach_progress(model, progress_callback: Optional[Callable[[dict], None]]) -> None:
    """Forward per-epoch progress to the caller, if it asked for it."""
    if progress_callback is None:
        return

    def _emit(yolo_trainer) -> None:
        try:
            progress_callback(_epoch_payload(yolo_trainer))
        except Exception as exc:  # a reporting bug must not kill training
            logger.debug("Progress callback failed: %s", exc)

    for event in ("on_train_epoch_end", "on_fit_epoch_end"):
        try:
            model.add_callback(event, _emit)
        except Exception as exc:
            logger.debug("Could not attach %s callback: %s", event, exc)


class TrainingResult:
    """Results from a completed training run."""

    def __init__(
        self,
        task: str,
        version: str,
        weights_path: Path,
        metrics: dict,
        config: dict,
        training_dir: Path,
        dataset_stats: dict,
    ) -> None:
        self.task = task
        self.version = version
        self.weights_path = weights_path
        self.metrics = metrics
        self.config = config
        self.training_dir = training_dir
        self.dataset_stats = dataset_stats
        self.trained_at = time.time()

    def save_manifest(self, path: Path) -> None:
        """Save a JSON manifest alongside the weights."""
        manifest = {
            "task": self.task,
            "version": self.version,
            "weights_path": str(self.weights_path),
            "metrics": self.metrics,
            "config": self.config,
            "dataset_stats": self.dataset_stats,
            "trained_at": self.trained_at,
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


class WasteModelTrainer:
    """
    Train a YOLO classification model for waste detection.

    Args:
        datasets_dir:      data/datasets/ root.
        models_output_dir: models/ root where versions are saved.
        device:            "cpu", "cuda:0", "mps", or "auto" (YOLO auto-select).
    """

    def __init__(
        self,
        datasets_dir: str | Path,
        models_output_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self._datasets_dir = Path(datasets_dir)
        self._models_dir = Path(models_output_dir) / "waste"
        self._device = device

    def train(
        self,
        *,
        base_model: str = "yolov8n-cls.pt",
        epochs: int = 50,
        image_size: int = 224,
        batch: int = 16,
        version: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> TrainingResult:
        """
        Run a complete training session.

        Args:
            base_model:         Pretrained YOLO model name (auto-downloaded).
            epochs:             Number of training epochs.
            image_size:         Input image size.
            batch:              Batch size (-1 = auto).
            version:            Version string e.g. "v001".
            progress_callback:  Called after each epoch with metrics dict.

        Returns:
            TrainingResult with paths and metrics.

        Raises:
            ValueError: If the dataset is empty.
            RuntimeError: If training fails.
        """
        from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES

        dm = DatasetManager(self._datasets_dir)
        counts = dm.waste_class_counts()
        total = sum(counts.values())

        if total < 4:
            raise ValueError(
                f"Need at least 4 waste images to train (found {total}). "
                "Upload more images via the Training page."
            )

        logger.info("Starting waste model training — version=%s  epochs=%d", version, epochs)
        log_event(logger, EventCode.MODEL_TRAINING_STARTED,
                  f"version={version}  epochs={epochs}  base={base_model}")

        # Prepare dataset split in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "split"
            dataset_stats = dm.prepare_yolo_cls_dataset(split_dir)

            # Train
            try:
                from ultralytics import YOLO
                model = YOLO(base_model)
                _attach_progress(model, progress_callback)

                # Use YOLO's built-in training
                results = model.train(
                    data=str(split_dir),
                    task="classify",
                    epochs=epochs,
                    imgsz=image_size,
                    batch=batch,
                    device=self._device,
                    project=str(self._models_dir),
                    name=version,
                    exist_ok=True,
                    verbose=True,
                    pretrained=True,
                )

                # Find best weights
                version_dir = self._models_dir / version
                best_pt = version_dir / "weights" / "best.pt"

                if not best_pt.exists():
                    # Try finding it in YOLO's default output
                    candidates = list(self._models_dir.rglob("best.pt"))
                    if candidates:
                        best_pt = candidates[-1]
                    else:
                        raise RuntimeError("Training completed but best.pt not found")

                # Extract metrics from results
                metrics = self._extract_metrics(results)
                metrics["dataset"] = counts
                metrics["dataset_stats"] = dataset_stats

                # Save manifest
                manifest_path = version_dir / "manifest.json"
                tr = TrainingResult(
                    task="waste",
                    version=version,
                    weights_path=best_pt,
                    metrics=metrics,
                    config={
                        "base_model": base_model,
                        "epochs": epochs,
                        "image_size": image_size,
                        "batch": batch,
                        "device": self._device,
                    },
                    training_dir=version_dir,
                    dataset_stats=dataset_stats,
                )
                tr.save_manifest(manifest_path)

                log_event(logger, EventCode.MODEL_TRAINING_COMPLETED,
                          f"version={version}  weights={best_pt}")
                return tr

            except Exception as exc:
                logger.error("Training failed: %s", exc)
                raise RuntimeError(f"Waste model training failed: {exc}") from exc

    def _extract_metrics(self, results) -> dict:
        """Extract metrics dict from Ultralytics results object."""
        metrics: dict = {}
        try:
            if hasattr(results, "results_dict"):
                rd = results.results_dict
                metrics["top1_accuracy"] = round(float(rd.get("metrics/accuracy_top1", 0)), 4)
                metrics["top5_accuracy"] = round(float(rd.get("metrics/accuracy_top5", 0)), 4)
            if hasattr(results, "box"):  # detection results
                b = results.box
                metrics["map50"]    = round(float(b.map50), 4) if hasattr(b, "map50") else None
                metrics["map50_95"] = round(float(b.map), 4)   if hasattr(b, "map") else None
        except Exception:
            pass
        return metrics


class PlateModelTrainer:
    """
    Train a YOLO object detection model for plate/tray detection.

    Args:
        datasets_dir:       data/datasets/ root.
        models_output_dir:  models/ root.
        device:             Compute device.
    """

    def __init__(
        self,
        datasets_dir: str | Path,
        models_output_dir: str | Path,
        device: str = "cpu",
    ) -> None:
        self._datasets_dir = Path(datasets_dir)
        self._models_dir = Path(models_output_dir) / "plate"
        self._device = device

    def train(
        self,
        *,
        base_model: str = "yolov8n.pt",
        epochs: int = 50,
        image_size: int = 640,
        batch: int = 16,
        version: str,
        data_yaml: str,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> TrainingResult:
        """
        Train plate detector from a YOLO detection dataset.

        Args:
            base_model: Pretrained YOLO detection model.
            epochs:     Training epochs.
            image_size: Detection input size.
            batch:      Batch size.
            version:    Version string.
            data_yaml:  Path to data.yaml file.
            progress_callback: Called after each epoch with a progress dict.

        Returns:
            TrainingResult.
        """
        logger.info("Starting plate model training — version=%s", version)
        log_event(logger, EventCode.MODEL_TRAINING_STARTED,
                  f"version={version}  task=plate  epochs={epochs}")

        try:
            from ultralytics import YOLO
            model = YOLO(base_model)
            _attach_progress(model, progress_callback)
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=image_size,
                batch=batch,
                device=self._device,
                project=str(self._models_dir),
                name=version,
                exist_ok=True,
                pretrained=True,
            )

            version_dir = self._models_dir / version
            best_pt = version_dir / "weights" / "best.pt"
            if not best_pt.exists():
                candidates = list(self._models_dir.rglob("best.pt"))
                if candidates:
                    best_pt = candidates[-1]
                else:
                    raise RuntimeError("Training completed but best.pt not found")

            metrics: dict = {}
            try:
                b = results.box
                metrics["map50"]    = round(float(b.map50), 4)
                metrics["map50_95"] = round(float(b.map), 4)
            except Exception:
                pass

            tr = TrainingResult(
                task="plate",
                version=version,
                weights_path=best_pt,
                metrics=metrics,
                config={
                    "base_model": base_model,
                    "epochs": epochs,
                    "image_size": image_size,
                    "batch": batch,
                },
                training_dir=version_dir,
                dataset_stats={},
            )
            manifest_path = version_dir / "manifest.json"
            tr.save_manifest(manifest_path)

            log_event(logger, EventCode.MODEL_TRAINING_COMPLETED,
                      f"version={version}  weights={best_pt}")
            return tr

        except Exception as exc:
            logger.error("Plate training failed: %s", exc)
            raise RuntimeError(f"Plate model training failed: {exc}") from exc
