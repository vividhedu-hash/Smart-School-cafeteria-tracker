"""
Plate detector using Ultralytics YOLO.

Loads a real trained YOLO model (not a mock).
Returns bounding boxes and confidence scores.

If the model weights file is missing, raises ModelNotFoundError with
a clear message telling the user what to do.

Optional "proxy mode" (explicit opt-in via config
`models.plate.allow_coco_fallback: true`): when the trained weights are
missing, a pretrained COCO yolov8n model detects tableware-like classes
(bowl, cup, fork, knife, spoon, orange, dining table) as a stand-in.
Proxy detections are NOT a trained plate detector — the engine flags all
transactions produced this way for manual review.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cafeteria.utils.logging import get_logger

logger = get_logger("detection.plate")


class ModelNotFoundError(RuntimeError):
    """Raised when a model weights file does not exist."""


@dataclass
class Detection:
    """A single object detection result."""
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the image crop corresponding to this bounding box."""
        return image[self.y1:self.y2, self.x1:self.x2]


class PlateDetector:
    """
    Real YOLO-based plate / tray detector.

    Args:
        weights_path:   Path to the trained .pt file.
        confidence:     Minimum detection confidence threshold.
        iou:            NMS IOU threshold.
        device:         Compute device string ("cpu", "cuda:0", "mps").
        roi:            Optional normalized ROI dict {x1, y1, x2, y2}.
                        If given, only detections inside the ROI are returned.
        allow_coco_fallback: Explicit opt-in for PROXY MODE. If True and the
                        trained weights are missing, a pretrained COCO
                        yolov8n stands in for the plate detector. Default
                        False: missing weights raise ModelNotFoundError.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.55,
        iou: float = 0.50,
        device: str = "cpu",
        roi: Optional[dict] = None,
        allow_coco_fallback: bool = False,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._confidence = confidence
        self._iou = iou
        self._device = device
        self._roi = roi
        self._allow_coco_fallback = allow_coco_fallback
        self._model = None
        self._loaded = False
        self._coco_fallback = False
        self.proxy_mode = False

    def load(self) -> None:
        """
        Load the YOLO model. Called once at startup.

        Raises:
            ModelNotFoundError: If weights are missing and proxy mode is
                disabled (the default). We never silently pretend a COCO
                bowl/cup detector is a trained plate detector.
        """
        self._coco_fallback = False
        self.proxy_mode = False
        if not self._weights_path.exists():
            if not self._allow_coco_fallback:
                raise ModelNotFoundError(
                    f"Plate detector model not found: {self._weights_path}\n"
                    "Action required: Train a plate detection model first.\n"
                    "  1. Capture plate images: python scripts/capture_dataset.py\n"
                    "  2. Label them (YOLO format) and place in data/datasets/plate/\n"
                    "  3. Train via the dashboard Training page or scripts/train_waste_model.py\n"
                    "(To demo with a generic COCO model instead, set "
                    "models.plate.allow_coco_fallback: true in configs/config.yaml — "
                    "this is clearly labelled as PROXY mode, not real plate detection; "
                    "all transactions created in proxy mode are flagged for review.)"
                )
            logger.warning(
                "PLATE PROXY MODE: no trained model at %s — using pretrained "
                "COCO yolov8n.pt (bowls/cups/plates-ish objects). This is a "
                "demo proxy, NOT a trained plate detector. All transactions "
                "will be tagged PROXY_PLATE_MODE and require review.",
                self._weights_path
            )
            self._coco_fallback = True
            self.proxy_mode = True

        try:
            from ultralytics import YOLO
            if self._coco_fallback:
                self._model = YOLO("yolov8n.pt")
            else:
                self._model = YOLO(str(self._weights_path))
            # Warm-up inference on a dummy frame
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            self._loaded = True
            logger.info(
                "Plate detector loaded — weights=%s  device=%s",
                self._weights_path,
                self._device,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load plate detector: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_proxy(self) -> bool:
        """True when running the COCO demo proxy instead of a trained model."""
        return self.proxy_mode

    def detect(
        self,
        image: np.ndarray,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> list[Detection]:
        """
        Run plate detection on a BGR frame.

        Args:
            image:         BGR numpy array.
            frame_width:   Used for ROI pixel conversion (defaults to image width).
            frame_height:  Used for ROI pixel conversion (defaults to image height).

        Returns:
            List of Detection objects, filtered by confidence and ROI.
        """
        if not self._loaded:
            raise RuntimeError("PlateDetector not loaded. Call load() first.")

        h, w = image.shape[:2]
        fw = frame_width or w
        fh = frame_height or h

        predict_kwargs = {
            "conf": self._confidence,
            "iou": self._iou,
            "device": self._device,
            "verbose": False,
        }
        if self._coco_fallback:
            predict_kwargs["classes"] = [39, 41, 45, 46, 47, 49, 60]

        results = self._model.predict(image, **predict_kwargs)

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                label = result.names.get(cls_idx, str(cls_idx))
                if self._coco_fallback:
                    # Honest labelling: proxy detections are marked as such
                    label = "plate_proxy"
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                det = Detection(
                    label=label,
                    confidence=conf,
                    x1=max(0, x1),
                    y1=max(0, y1),
                    x2=min(w, x2),
                    y2=min(h, y2),
                )

                # ROI filter
                if self._roi and not self._inside_roi(det, fw, fh):
                    continue

                detections.append(det)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _inside_roi(self, det: Detection, fw: int, fh: int) -> bool:
        """Return True if the detection center is inside the plate ROI."""
        roi = self._roi
        rx1 = int(roi["x1"] * fw)
        ry1 = int(roi["y1"] * fh)
        rx2 = int(roi["x2"] * fw)
        ry2 = int(roi["y2"] * fh)
        cx, cy = det.center
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def update_weights(self, new_weights_path: str | Path) -> None:
        """Hot-swap to a new model weights file (used after model activation)."""
        self._weights_path = Path(new_weights_path)
        self._loaded = False
        self.load()
        logger.info("Plate detector weights updated to %s", self._weights_path)
