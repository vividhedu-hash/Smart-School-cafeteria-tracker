"""
Image store — manages saving and path generation for evidence images.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cafeteria.utils.logging import get_logger

logger = get_logger("media.image_store")


class ImageStore:
    """
    Handles evidence image persistence.

    Directory structure:
        captures/
            YYYYMMDD/
                {transaction_id}/
                    event.jpg
                    face.jpg    (future: separate face camera)
                    plate.jpg   (future: separate plate camera)

    Args:
        captures_dir: Root captures directory.
        quality:      JPEG quality (0-100).
    """

    def __init__(
        self,
        captures_dir: str | Path,
        quality: int = 92,
    ) -> None:
        self._root = Path(captures_dir)
        self._quality = quality
        self._root.mkdir(parents=True, exist_ok=True)

    def save_event_image(
        self,
        transaction_id: str,
        image: np.ndarray,
        timestamp: Optional[float] = None,
        filename: str = "event.jpg",
    ) -> Optional[str]:
        """
        Save an evidence image for a transaction.

        Args:
            transaction_id: Used for directory naming.
            image:          BGR numpy array.
            timestamp:      Event timestamp (Unix epoch). Defaults to now.
            filename:       Output filename.

        Returns:
            Absolute path string on success, None on failure.
        """
        import time
        ts = timestamp or time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        date_str = dt.strftime("%Y%m%d")

        out_dir = self._root / date_str / transaction_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            success = cv2.imwrite(str(out_path), image, encode_params)
            if success:
                logger.debug("Saved evidence image: %s", out_path)
                return str(out_path)
            else:
                logger.error("cv2.imwrite failed for path: %s", out_path)
                return None
        except Exception as exc:
            logger.error("Image save error: %s", exc)
            return None

    def save_debug_frame(
        self,
        frames_dir: str | Path,
        image: np.ndarray,
        filename: str = "latest.jpg",
    ) -> bool:
        """
        Overwrite the latest debug frame for dashboard display.

        Args:
            frames_dir: Directory for live frames.
            image:      BGR numpy array.
            filename:   Output filename (typically latest.jpg).

        Returns:
            True on success.
        """
        out_dir = Path(frames_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # lower quality for speed
            return cv2.imwrite(str(out_path), image, encode_params)
        except Exception as exc:
            logger.warning("Debug frame save error: %s", exc)
            return False

    def review_image_path(self, filename: str) -> Path:
        """Return path to a review queue image."""
        return Path(self._root).parent / "review_queue" / filename
