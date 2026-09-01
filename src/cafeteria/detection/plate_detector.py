"""
Plate detector using Ultralytics YOLO.

Loads a real trained YOLO model (not a mock).
Returns bounding boxes and confidence scores.

If the model weights file is missing, the default is a real OpenCV visual
detector (circles / bright plate blobs / tray contours). That is not YOLO
and is labelled `backend=visual` in runtime state.

Optional "proxy mode" (explicit opt-in via config
`models.plate.allow_coco_fallback: true`): a pretrained COCO yolov8n model
detects tableware-like classes as a stand-in. Proxy detections are NOT a
trained plate detector — the engine flags those transactions for review.

Set `allow_visual_fallback: false` (and coco fallback false) to raise
ModelNotFoundError instead of using the OpenCV backend.
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
                        yolov8n stands in for the plate detector.
        allow_visual_fallback: Default True. If trained weights are missing,
                        use the OpenCV visual plate detector so the live
                        pipeline still runs.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.55,
        iou: float = 0.50,
        device: str = "cpu",
        roi: Optional[dict] = None,
        allow_coco_fallback: bool = False,
        allow_visual_fallback: bool = True,
    ) -> None:
        self._weights_path = Path(weights_path)
        self._confidence = confidence
        self._iou = iou
        self._device = device
        self._roi = roi
        self._allow_coco_fallback = allow_coco_fallback
        self._allow_visual_fallback = allow_visual_fallback
        self._model = None
        self._loaded = False
        self._coco_fallback = False
        self.proxy_mode = False
        self.visual_mode = False
        self.backend = "none"

    def load(self) -> None:
        """
        Load the YOLO model. Called once at startup.

        Raises:
            ModelNotFoundError: If weights are missing and both visual and
                COCO fallbacks are disabled. We never silently pretend a
                COCO bowl/cup detector is a trained plate detector.
        """
        self._coco_fallback = False
        self.proxy_mode = False
        self.visual_mode = False
        self.backend = "none"
        self._model = None

        if not self._weights_path.exists():
            if self._allow_visual_fallback:
                self.visual_mode = True
                self.backend = "visual"
                self._loaded = True
                logger.info(
                    "Plate detector: OpenCV visual backend (no trained YOLO at %s). "
                    "Train a plate model later for higher accuracy.",
                    self._weights_path,
                )
                return
            if not self._allow_coco_fallback:
                raise ModelNotFoundError(
                    f"Plate detector model not found: {self._weights_path}\n"
                    "Action required: Train a plate detection model first.\n"
                    "  1. Capture plate images: python scripts/capture_dataset.py\n"
                    "  2. Label them (YOLO format) and place in data/datasets/plate/\n"
                    "  3. Train via the dashboard Training page or scripts/quickstart.py\n"
                    "(Built-in OpenCV visual detection is on by default. To disable it "
                    "set models.plate.allow_visual_fallback: false. To demo with a "
                    "generic COCO model instead, set allow_coco_fallback: true — "
                    "PROXY mode, not real plate detection; those transactions go to review.)"
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
            self.backend = "coco_proxy"

        try:
            from ultralytics import YOLO
            if self._coco_fallback:
                self._model = YOLO("yolov8n.pt")
            else:
                self._model = YOLO(str(self._weights_path))
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, device=self._device, verbose=False)
            self._loaded = True
            self.backend = "coco_proxy" if self._coco_fallback else "yolo"
            logger.info(
                "Plate detector loaded — backend=%s weights=%s device=%s",
                self.backend,
                self._weights_path,
                self._device,
            )
        except Exception as exc:
            if self._allow_visual_fallback:
                logger.warning(
                    "YOLO plate load failed (%s) — falling back to OpenCV visual detector.",
                    exc,
                )
                self.visual_mode = True
                self.backend = "visual"
                self.proxy_mode = False
                self._coco_fallback = False
                self._model = None
                self._loaded = True
                return
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

        if self.visual_mode:
            from cafeteria.detection.visual import detect_plates_visual
            return detect_plates_visual(
                image, roi=self._roi, min_confidence=min(self._confidence, 0.40)
            )

        h, w = image.shape[:2]
        fw = frame_width or w
        fh = frame_height or h

        long_side = max(h, w)
        imgsz = 640
        if (self._device.startswith("cuda") or self._device == "mps") and long_side >= 1600:
            imgsz = 736

        predict_kwargs = {
            "conf": self._confidence,
            "iou": self._iou,
            "device": self._device,
            "verbose": False,
            "imgsz": imgsz,
        }
        if self._device.startswith("cuda"):
            predict_kwargs["half"] = True
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
