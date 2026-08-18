"""
InsightFace / ArcFace face recognition engine.

Uses the `insightface` library with the buffalo_l or buffalo_s model pack.
Downloads weights on first use (~100 MB) from the InsightFace registry.
After that, entirely local.

Cross-platform: onnxruntime (CPU) on all platforms.
On macOS/Linux with onnxruntime-gpu, uses GPU automatically.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from cafeteria.utils.logging import get_logger

logger = get_logger("recognition.face_engine")


class FaceEngine:
    """
    Wrapper around InsightFace for face detection + embedding extraction.

    Args:
        model_pack:     InsightFace model name ("buffalo_l", "buffalo_s").
        model_dir:      Directory where InsightFace stores downloaded models.
        det_size:       Detection input size (width, height).
        device:         "cpu", "cuda:0", or "mps" — mapped to onnxruntime provider.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        model_dir: Optional[str | Path] = None,
        det_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        det_thresh: float = 0.5,
    ) -> None:
        self._model_pack = model_pack
        self._model_dir = Path(model_dir) if model_dir else Path("models/face")
        self._det_size = tuple(det_size)
        self._device = device
        self._det_thresh = float(det_thresh)
        self._app = None
        self._loaded = False

    def load(self) -> None:
        """
        Load InsightFace model. Downloads on first call if needed.

        Raises:
            RuntimeError: If insightface is not installed or download fails.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError:
            raise RuntimeError(
                "insightface is not installed.\n"
                "Install with: pip install insightface onnxruntime"
            )

        # Map device to onnxruntime provider
        providers = self._get_providers()

        self._model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading InsightFace model pack=%s  det_size=%s  det_thresh=%.2f  providers=%s",
            self._model_pack,
            self._det_size,
            self._det_thresh,
            providers,
        )

        self._app = FaceAnalysis(
            name=self._model_pack,
            root=str(self._model_dir),
            providers=providers,
        )
        self._app.prepare(
            ctx_id=0,
            det_size=self._det_size,
            det_thresh=self._det_thresh,
        )
        self._loaded = True

        logger.info("InsightFace loaded successfully (pack=%s)", self._model_pack)

    def _get_providers(self) -> list[str]:
        """Map device string to onnxruntime execution providers."""
        if self._device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self._device == "mps":
            # CoreML provider for Apple Silicon (if available)
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                if "CoreMLExecutionProvider" in available:
                    return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass
            return ["CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_faces(self, image: np.ndarray) -> list[dict]:
        """
        Detect all faces in a BGR image and return embeddings + bboxes.

        Args:
            image: BGR numpy array.

        Returns:
            List of dicts, each with:
                {
                    "bbox": (x1, y1, x2, y2),
                    "embedding": np.ndarray shape (512,),
                    "det_score": float,
                    "kps": np.ndarray,   # 5-point landmarks
                }
        """
        if not self._loaded:
            raise RuntimeError("FaceEngine not loaded. Call load() first.")

        import cv2
        # InsightFace expects BGR (same as OpenCV) — no conversion needed
        faces = self._app.get(image)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            embedding = face.normed_embedding  # Already L2-normalized

            results.append({
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "embedding": embedding,
                "det_score": float(face.det_score),
                "kps": face.kps,
            })

        return results

    def get_largest_face(
        self,
        image: np.ndarray,
        min_size: int = 60,
    ) -> Optional[dict]:
        """
        Return the largest detected face (by bounding-box area) meeting min_size.

        Args:
            image:    BGR frame.
            min_size: Minimum face width/height in pixels.

        Returns:
            Face dict or None if no qualifying face found.
        """
        faces = self.get_faces(image)
        if not faces:
            return None

        qualifying = []
        for face in faces:
            x1, y1, x2, y2 = face["bbox"]
            w, h = x2 - x1, y2 - y1
            if w >= min_size and h >= min_size:
                qualifying.append((w * h, face))

        if not qualifying:
            return None

        return max(qualifying, key=lambda t: t[0])[1]
