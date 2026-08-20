"""
Closed ML feedback loop for waste classification.

Live path:
  detection → plate crop queued → operator confirms/corrects label
  → image enters data/datasets/waste/<CLASS>/ → retrain when enough
  → evaluate → activate if better → engine hot-swaps

This module never blocks the camera loop. Training runs on a daemon thread
outside the inference process (dashboard / CLI / Review page).
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cafeteria.training.dataset_manager import WASTE_CLASSES, DatasetManager
from cafeteria.training.progress import is_running, read_status, status_path, update_status, write_status
from cafeteria.training.registry import ModelRegistry
from cafeteria.utils.logging import get_logger

logger = get_logger("training.learning_loop")

_TRAIN_LOCK = threading.Lock()


@dataclass
class LoopConfig:
    enabled: bool = True
    auto_enqueue: bool = True
    min_confidence_to_enqueue: float = 0.40
    promote_auto_confirmed: bool = True
    auto_confirm_min_confidence: float = 0.85
    retrain_after_n_labels: int = 8
    min_images_per_class: int = 4
    auto_activate: bool = True
    auto_activate_min_accuracy: float = 0.50
    retrain_epochs: int = 15
    retrain_image_size: int = 224
    retrain_batch: int = 8


def loop_config_from_settings(training_cfg: Any) -> LoopConfig:
    """Build LoopConfig from TrainingSettings / YAML / namespace."""
    raw = getattr(training_cfg, "ml_loop", None)
    if raw is None and isinstance(training_cfg, dict):
        raw = training_cfg.get("ml_loop")
    if raw is None:
        return LoopConfig()
    if isinstance(raw, dict):
        return LoopConfig(**{
            k: raw[k] for k in LoopConfig.__dataclass_fields__ if k in raw
        })
    return LoopConfig(
        enabled=bool(getattr(raw, "enabled", True)),
        auto_enqueue=bool(getattr(raw, "auto_enqueue", True)),
        min_confidence_to_enqueue=float(
            getattr(raw, "min_confidence_to_enqueue", 0.40) or 0.40
        ),
        promote_auto_confirmed=bool(getattr(raw, "promote_auto_confirmed", True)),
        auto_confirm_min_confidence=float(
            getattr(raw, "auto_confirm_min_confidence", 0.85) or 0.85
        ),
        retrain_after_n_labels=int(getattr(raw, "retrain_after_n_labels", 8) or 8),
        min_images_per_class=int(getattr(raw, "min_images_per_class", 4) or 4),
        auto_activate=bool(getattr(raw, "auto_activate", True)),
        auto_activate_min_accuracy=float(
            getattr(raw, "auto_activate_min_accuracy", 0.50) or 0.50
        ),
        retrain_epochs=int(getattr(raw, "retrain_epochs", 15) or 15),
        retrain_image_size=int(getattr(raw, "retrain_image_size", 224) or 224),
        retrain_batch=int(getattr(raw, "retrain_batch", 8) or 8),
    )


class LearningLoop:
    """
    File-backed ML loop state under data/ml_loop/.

    Args:
        project_root: Repository root.
        datasets_dir: Path to data/datasets.
        models_dir: Path to models/.
        registry: ModelRegistry instance.
        config: LoopConfig thresholds / switches.
        device: Torch device for training.
        base_model: Ultralytics cls checkpoint name.
    """

    def __init__(
        self,
        project_root: str | Path,
        datasets_dir: str | Path,
        models_dir: str | Path,
        registry: ModelRegistry,
        config: Optional[LoopConfig] = None,
        device: str = "cpu",
        base_model: str = "yolov8n-cls.pt",
    ) -> None:
        self._root = Path(project_root)
        self._loop_dir = self._root / "data" / "ml_loop"
        self._pending_dir = self._loop_dir / "pending"
        self._state_path = self._loop_dir / "state.json"
        self._loop_dir.mkdir(parents=True, exist_ok=True)
        self._pending_dir.mkdir(parents=True, exist_ok=True)

        self._datasets_dir = Path(datasets_dir)
        self._models_dir = Path(models_dir)
        self._registry = registry
        self._cfg = config or LoopConfig()
        self._device = device
        self._base_model = base_model
        self._dm = DatasetManager(self._datasets_dir)

    # ── State ──────────────────────────────────────────────────────────────

    def _load_state(self) -> dict:
        if not self._state_path.exists():
            return {
                "promoted_since_train": 0,
                "last_train_at": None,
                "last_train_version": None,
                "last_activate_at": None,
                "last_activate_version": None,
                "candidates_enqueued": 0,
                "labels_promoted": 0,
            }
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"promoted_since_train": 0}

    def _save_state(self, state: dict) -> None:
        state["updated_at"] = time.time()
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self._state_path)

    def snapshot(self) -> dict:
        """Dashboard-friendly status of the learning loop."""
        state = self._load_state()
        pending = self.list_pending()
        counts = self._dm.waste_class_counts()
        train_status = read_status(status_path(self._root))
        return {
            "enabled": self._cfg.enabled,
            "pending_candidates": len(pending),
            "promoted_since_train": int(state.get("promoted_since_train") or 0),
            "retrain_after_n_labels": self._cfg.retrain_after_n_labels,
            "labels_promoted_total": int(state.get("labels_promoted") or 0),
            "candidates_enqueued_total": int(state.get("candidates_enqueued") or 0),
            "dataset_counts": counts,
            "dataset_total": sum(counts.values()),
            "ready_to_retrain": self.can_retrain(),
            "training_running": is_running(train_status),
            "last_train_version": state.get("last_train_version"),
            "last_activate_version": state.get("last_activate_version"),
            "active_waste": self._registry.active_version_string("waste"),
        }

    # ── Enqueue live crops ─────────────────────────────────────────────────

    def enqueue_plate_crop(
        self,
        plate_crop: np.ndarray,
        *,
        predicted_label: str,
        confidence: float,
        transaction_id: str,
        event_image_path: Optional[str] = None,
        auto_confirmed: bool = False,
        plate_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> Optional[str]:
        # [AI-CoLab: Verified by Antigravity] Live crop enqueue with auto-confirm promotion
        """
        Save a plate crop as a pending training candidate.

        High-confidence AUTO_CONFIRMED events can skip review and go straight
        into the waste dataset when promote_auto_confirmed is on.
        """
        if not self._cfg.enabled or not self._cfg.auto_enqueue:
            return None
        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
            return None
        if confidence < self._cfg.min_confidence_to_enqueue:
            return None
        if predicted_label not in WASTE_CLASSES:
            return None

        import cv2

        cand_id = f"{transaction_id}_{uuid.uuid4().hex[:6]}"
        img_path = self._pending_dir / f"{cand_id}.jpg"
        meta_path = self._pending_dir / f"{cand_id}.json"
        if not cv2.imwrite(str(img_path), plate_crop):
            return None

        meta = {
            "id": cand_id,
            "transaction_id": transaction_id,
            "predicted_label": predicted_label,
            "confidence": float(confidence),
            "event_image_path": event_image_path,
            "plate_bbox": list(plate_bbox) if plate_bbox else None,
            "image_path": str(img_path),
            "created_at": time.time(),
            "status": "pending",
            "confirmed_label": None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state = self._load_state()
        state["candidates_enqueued"] = int(state.get("candidates_enqueued") or 0) + 1
        self._save_state(state)
        logger.info(
            "ML loop enqueued crop %s label=%s conf=%.2f",
            cand_id, predicted_label, confidence,
        )

        if (
            auto_confirmed
            and self._cfg.promote_auto_confirmed
            and confidence >= self._cfg.auto_confirm_min_confidence
        ):
            self.promote(cand_id, predicted_label, source="auto_confirmed")
        return cand_id

    def list_pending(self) -> list[dict]:
        items: list[dict] = []
        for meta_path in sorted(self._pending_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("status") == "pending":
                items.append(meta)
        return items

    def get_candidate(self, cand_id: str) -> Optional[dict]:
        meta_path = self._pending_dir / f"{cand_id}.json"
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ── Promote confirmed labels ───────────────────────────────────────────

    def promote(
        self,
        cand_id: str,
        confirmed_label: str,
        *,
        source: str = "manual",
        trigger_retrain: bool = True,
    ) -> dict:
        """
        Move a pending crop into data/datasets/waste/<CLASS>/.

        Returns a result dict with added counts and whether retrain started.
        """
        if confirmed_label not in WASTE_CLASSES:
            raise ValueError(f"Invalid waste class: {confirmed_label}")

        meta = self.get_candidate(cand_id)
        if not meta:
            # Allow promoting from an arbitrary image path when cand_id is a file
            img = Path(cand_id)
            if img.exists() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                added, skipped = self._dm.add_waste_images(confirmed_label, [img])
                state = self._load_state()
                if added:
                    state["promoted_since_train"] = int(
                        state.get("promoted_since_train") or 0
                    ) + added
                    state["labels_promoted"] = int(state.get("labels_promoted") or 0) + added
                    self._save_state(state)
                retrain = self.maybe_retrain() if trigger_retrain else {"started": False}
                return {"added": added, "skipped": skipped, "retrain": retrain}
            raise FileNotFoundError(f"Unknown ML candidate: {cand_id}")

        img_path = Path(meta["image_path"])
        if not img_path.exists():
            raise FileNotFoundError(f"Crop missing for candidate {cand_id}")

        added, skipped = self._dm.add_waste_images(confirmed_label, [img_path])
        meta["status"] = "promoted"
        meta["confirmed_label"] = confirmed_label
        meta["promoted_at"] = time.time()
        meta["source"] = source
        (self._pending_dir / f"{cand_id}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        # Keep crop for audit; also copy into promoted archive
        archive = self._loop_dir / "promoted" / confirmed_label
        archive.mkdir(parents=True, exist_ok=True)
        if img_path.exists():
            shutil.copy2(str(img_path), str(archive / img_path.name))

        state = self._load_state()
        if added:
            state["promoted_since_train"] = int(state.get("promoted_since_train") or 0) + added
            state["labels_promoted"] = int(state.get("labels_promoted") or 0) + added
        self._save_state(state)
        logger.info(
            "ML loop promoted %s → %s (added=%d source=%s)",
            cand_id, confirmed_label, added, source,
        )

        retrain = self.maybe_retrain() if trigger_retrain else {"started": False}
        return {"added": added, "skipped": skipped, "retrain": retrain, "candidate": meta}

    def promote_from_paths(
        self,
        image_paths: list[str | Path],
        confirmed_label: str,
        *,
        trigger_retrain: bool = True,
    ) -> dict:
        """Promote one or more images (e.g. review evidence) into the dataset."""
        if confirmed_label not in WASTE_CLASSES:
            raise ValueError(f"Invalid waste class: {confirmed_label}")
        added, skipped = self._dm.add_waste_images(confirmed_label, image_paths)
        state = self._load_state()
        if added:
            state["promoted_since_train"] = int(state.get("promoted_since_train") or 0) + added
            state["labels_promoted"] = int(state.get("labels_promoted") or 0) + added
            self._save_state(state)
        retrain = self.maybe_retrain() if trigger_retrain else {"started": False}
        return {"added": added, "skipped": skipped, "retrain": retrain}

    # ── Retrain + activate ─────────────────────────────────────────────────

    def can_retrain(self) -> bool:
        if not self._cfg.enabled:
            return False
        if is_running(read_status(status_path(self._root))):
            return False
        state = self._load_state()
        if int(state.get("promoted_since_train") or 0) < self._cfg.retrain_after_n_labels:
            return False
        counts = self._dm.waste_class_counts()
        # Need at least two classes with enough images to learn a decision boundary
        ready_classes = sum(
            1 for n in counts.values() if n >= self._cfg.min_images_per_class
        )
        if ready_classes < 2:
            return False
        if self._dm.total_waste_images() < max(
            self._cfg.retrain_after_n_labels,
            self._cfg.min_images_per_class * 2,
        ):
            return False
        return True

    def maybe_retrain(self, *, force: bool = False) -> dict:
        """Start a background waste retrain if the policy says so."""
        if not force and not self.can_retrain():
            return {
                "started": False,
                "reason": "threshold_not_met",
                "snapshot": self.snapshot(),
            }
        if is_running(read_status(status_path(self._root))):
            return {"started": False, "reason": "already_running"}

        if not _TRAIN_LOCK.acquire(blocking=False):
            return {"started": False, "reason": "lock_busy"}

        version = self._registry.next_version("waste")
        progress = status_path(self._root)
        write_status(
            progress,
            state="running",
            task="waste",
            version=version,
            epoch=0,
            epochs=self._cfg.retrain_epochs,
            history=[],
            message="ML loop: preparing dataset split…",
            started_at=time.time(),
            source="learning_loop",
        )

        def _run() -> None:
            try:
                from cafeteria.training.trainer import WasteModelTrainer

                def _on_epoch(payload: dict) -> None:
                    current = read_status(progress)
                    history = list(current.get("history") or [])
                    entry = {"epoch": payload.get("epoch")}
                    if payload.get("loss") is not None:
                        entry["loss"] = payload["loss"]
                    for key, value in (payload.get("metrics") or {}).items():
                        if "accuracy_top1" in key:
                            entry["top1"] = value
                    if entry.get("epoch") and (
                        not history or history[-1].get("epoch") != entry["epoch"]
                    ):
                        history.append(entry)
                    elif history:
                        history[-1].update(entry)
                    update_status(
                        progress,
                        state="running",
                        epoch=payload.get("epoch", current.get("epoch")),
                        epochs=payload.get("epochs") or current.get("epochs"),
                        history=history[-300:],
                        message=(
                            f"ML loop epoch {payload.get('epoch', '?')} "
                            f"of {payload.get('epochs', self._cfg.retrain_epochs)}"
                        ),
                    )

                trainer = WasteModelTrainer(
                    datasets_dir=self._datasets_dir,
                    models_output_dir=self._models_dir,
                    device=self._device,
                )
                result = trainer.train(
                    base_model=self._base_model,
                    epochs=self._cfg.retrain_epochs,
                    image_size=self._cfg.retrain_image_size,
                    batch=self._cfg.retrain_batch,
                    version=version,
                    progress_callback=_on_epoch,
                )
                self._registry.register_version(
                    task=result.task,
                    version=result.version,
                    weights_path=str(result.weights_path),
                    metrics=result.metrics,
                    config=result.config,
                    dataset_stats=result.dataset_stats,
                )

                activate_info = self._maybe_activate(result)
                state = self._load_state()
                state["promoted_since_train"] = 0
                state["last_train_at"] = time.time()
                state["last_train_version"] = result.version
                if activate_info.get("activated"):
                    state["last_activate_at"] = time.time()
                    state["last_activate_version"] = result.version
                self._save_state(state)

                current = read_status(progress)
                write_status(
                    progress,
                    state="done",
                    task=result.task,
                    version=result.version,
                    weights_path=str(result.weights_path),
                    metrics=result.metrics,
                    epochs=self._cfg.retrain_epochs,
                    epoch=self._cfg.retrain_epochs,
                    history=current.get("history") or [],
                    message=(
                        "ML loop training finished."
                        + (" Activated." if activate_info.get("activated") else "")
                    ),
                    finished_at=time.time(),
                    activated=bool(activate_info.get("activated")),
                    activate_reason=activate_info.get("reason"),
                    source="learning_loop",
                )
            except Exception as exc:
                logger.exception("ML loop training failed")
                write_status(
                    progress,
                    state="error",
                    task="waste",
                    version=version,
                    message=str(exc),
                    finished_at=time.time(),
                    source="learning_loop",
                )
            finally:
                _TRAIN_LOCK.release()

        threading.Thread(target=_run, daemon=True, name="ml-loop-train").start()
        logger.info("ML loop started waste retrain %s", version)
        return {"started": True, "version": version}

    def _maybe_activate(self, result) -> dict:
        """Activate the new waste model when it beats the active one (or is first)."""
        if not self._cfg.auto_activate:
            return {"activated": False, "reason": "auto_activate_disabled"}

        metrics = dict(result.metrics or {})
        # Prefer top-1 accuracy from Ultralytics cls metrics when present
        new_score = _score_metrics(metrics)
        if new_score is None:
            # No usable score — activate first model only
            if self._registry.get_active("waste") is None:
                ok = self._registry.activate("waste", result.version)
                if ok:
                    self._notify_engine(result.version)
                return {
                    "activated": ok,
                    "reason": "first_model" if ok else "activate_failed",
                    "new_score": None,
                }
            return {"activated": False, "reason": "no_score", "new_score": None}

        if new_score < self._cfg.auto_activate_min_accuracy:
            return {
                "activated": False,
                "reason": "below_min_accuracy",
                "new_score": new_score,
            }

        active = self._registry.get_active("waste")
        if active is None:
            ok = self._registry.activate("waste", result.version)
            if ok:
                self._notify_engine(result.version)
            return {
                "activated": ok,
                "reason": "first_model" if ok else "activate_failed",
                "new_score": new_score,
            }

        old_score = _score_metrics(active.get("metrics") or {})
        if old_score is not None and new_score < old_score:
            return {
                "activated": False,
                "reason": "worse_than_active",
                "new_score": new_score,
                "old_score": old_score,
            }

        ok = self._registry.activate("waste", result.version)
        if ok:
            self._notify_engine(result.version)
        return {
            "activated": ok,
            "reason": "improved" if ok else "activate_failed",
            "new_score": new_score,
            "old_score": old_score,
        }

    def _notify_engine(self, version: str) -> None:
        try:
            from cafeteria.monitoring.metrics import write_command
            write_command(
                self._root / "data" / "commands.json",
                {"type": "activate_model", "task": "waste", "version": version},
            )
        except Exception as exc:
            logger.debug("Could not write activate command: %s", exc)


def _score_metrics(metrics: dict) -> Optional[float]:
    """Pick a single comparable score from trainer/evaluator metrics."""
    if not metrics:
        return None
    for key in (
        "metrics/accuracy_top1",
        "accuracy_top1",
        "top1_accuracy",
        "accuracy",
        "fitness",
    ):
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    # Nested Ultralytics results sometimes land under results_dict-like keys
    for key, value in metrics.items():
        if "accuracy_top1" in str(key).lower():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def crop_plate_from_frame(
    frame: Optional[np.ndarray],
    detection: Any,
) -> Optional[np.ndarray]:
    """Crop a plate Detection from a BGR frame; return None if impossible."""
    if frame is None or detection is None:
        return None
    try:
        if hasattr(detection, "crop"):
            crop = detection.crop(frame)
        else:
            x1, y1, x2, y2 = (
                int(detection.x1), int(detection.y1),
                int(detection.x2), int(detection.y2),
            )
            crop = frame[y1:y2, x1:x2]
        if crop is None or getattr(crop, "size", 0) == 0:
            return None
        return crop
    except Exception:
        return None
