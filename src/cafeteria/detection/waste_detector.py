"""
Waste classifier using Ultralytics YOLO classification model.

Classes: EMPTY | LOW_WASTE | MEDIUM_WASTE | HIGH_WASTE

If the model weights file is missing, the default is a real OpenCV
colour-occupancy classifier (labelled `backend=visual`). Training a
YOLOv8-cls model replaces this backend automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cafeteria.detection.plate_detector import ModelNotFoundError
from cafeteria.utils.logging import get_logger

logger = get_logger("detection.waste")

DEFAULT_CLASS_MAP = {
    0: "EMPTY",
    1: "LOW_WASTE",
    2: "MEDIUM_WASTE",
    3: "HIGH_WASTE",
}


@dataclass
class WasteResult:
    """Result of waste classification on a plate crop."""
    label: str          # e.g. "HIGH_WASTE"
    confidence: float   # model confidence ∈ [0, 1]
    all_scores: dict    # {label: confidence} for all classes
    is_waste: bool      # True for LOW/MEDIUM/HIGH

    @property
    def is_empty(self) -> bool:
        return self.label == "EMPTY"


class WasteDetector:
    """
    Real YOLO classification-based waste detector.

    Classifies a plate crop image into one of:
        EMPTY | LOW_WASTE | MEDIUM_WASTE | HIGH_WASTE

    Args:
        weights_path:  Path to trained YOLOv8-cls .pt file.
        confidence:    Minimum confidence to accept a classification.
        device:        Compute device ("cpu", "cuda:0", "mps").
        class_map:     Dict mapping class index → label string.
        allow_visual_fallback: Default True. If trained weights are missing,
                        use OpenCV colour occupancy so the live pipeline runs.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.55,
        device: str = "cpu",
        class_map: Optional[dict] = None,
        allow_visual_fallback: bool = True,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._confidence = confidence
        self._device = device
        self._class_map = class_map or DEFAULT_CLASS_MAP
        self._allow_visual_fallback = allow_visual_fallback
        self._model = None
        self._loaded = False
        self.visual_mode = False
        self.backend = "none"

    def load(self) -> None:
        """Load the YOLO classification model, or the OpenCV visual backend."""
        self.visual_mode = False
        self.backend = "none"
        self._model = None

        if not self._weights_path.exists():
            if self._allow_visual_fallback:
                self.visual_mode = True
                self.backend = "visual"
                self._loaded = True
                logger.info(
                    "Waste detector: OpenCV visual backend (no trained YOLO at %s). "
                    "Train a waste model later for higher accuracy.",
                    self._weights_path,
                )
                return
            raise ModelNotFoundError(
                f"Waste classifier model not found: {self._weights_path}\n"
                "Action required: Train a waste classification model first.\n"
                "  1. Open the dashboard → Training page\n"
                "  2. Upload waste images per category (EMPTY, LOW_WASTE, etc.)\n"
                "  3. Click START TRAINING\n"
                "  4. Activate the trained model\n"
                "(Built-in OpenCV visual classification is on by default. "
                "Set models.waste.allow_visual_fallback: false to disable it.)"
            )

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self._weights_path))
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            self._loaded = True
            self.backend = "yolo"
            logger.info(
                "Waste detector loaded — backend=yolo weights=%s device=%s",
                self._weights_path,
                self._device,
            )
        except Exception as exc:
            if self._allow_visual_fallback:
                logger.warning(
                    "YOLO waste load failed (%s) — falling back to OpenCV visual classifier.",
                    exc,
                )
                self.visual_mode = True
                self.backend = "visual"
                self._model = None
                self._loaded = True
                return
            raise RuntimeError(f"Failed to load waste detector: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(self, plate_crop: np.ndarray) -> WasteResult:
        """
        Classify a plate crop image.

        Args:
            plate_crop: BGR numpy array of the plate region.

        Returns:
            WasteResult with label, confidence, and per-class scores.

        Raises:
            RuntimeError: If model not loaded.
            ValueError:   If plate_crop is empty or invalid.
        """
        if not self._loaded:
            raise RuntimeError("WasteDetector not loaded. Call load() first.")

        if plate_crop is None or plate_crop.size == 0:
            raise ValueError("Empty plate crop passed to WasteDetector.classify()")

        if self.visual_mode:
            from cafeteria.detection.visual import classify_waste_visual
            return classify_waste_visual(plate_crop)

        results = self._model.predict(
            plate_crop,
            device=self._device,
            verbose=False,
        )

        if not results or results[0].probs is None:
            # No result — return UNKNOWN
            return WasteResult(
                label="UNKNOWN",
                confidence=0.0,
                all_scores={},
                is_waste=False,
            )

        probs = results[0].probs
        top_idx = int(probs.top1)
        top_conf = float(probs.top1conf)

        # Build all-class score dict
        all_scores = {}
        if probs.data is not None:
            for idx, score in enumerate(probs.data.tolist()):
                label = self._class_map.get(idx, str(idx))
                all_scores[label] = round(float(score), 4)

        # Use model's own class names if available
        if results[0].names:
            label = results[0].names.get(top_idx, self._class_map.get(top_idx, "UNKNOWN"))
        else:
            label = self._class_map.get(top_idx, "UNKNOWN")

        is_waste = label in ("FOOD_PRESENT", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE")

        return WasteResult(
            label=label,
            confidence=round(top_conf, 4),
            all_scores=all_scores,
            is_waste=is_waste,
        )

    def update_weights(self, new_weights_path: str | Path) -> None:
        """Hot-swap to a new model weights file."""
        self._weights_path = Path(new_weights_path)
        self._loaded = False
        self.load()
        logger.info("Waste detector weights updated to %s", self._weights_path)
