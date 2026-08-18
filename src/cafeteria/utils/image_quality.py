"""
Image quality metrics for selecting the best frame from a candidate set.

All functions operate on BGR numpy arrays (OpenCV native format).
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Optional


def laplacian_variance(image: np.ndarray) -> float:
    """
    Measure image sharpness via variance of the Laplacian operator.

    Higher = sharper. Blurry images have low variance because there are
    few strong edges. Typical "sharp" frames: > 100. "Blurry": < 50.

    Args:
        image: BGR image as numpy array.

    Returns:
        Float variance (higher = sharper).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def mean_brightness(image: np.ndarray) -> float:
    """
    Return mean pixel brightness in [0, 255].

    Used to detect over/under-exposed frames.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(gray.mean())


def brightness_ok(image: np.ndarray, lo: float = 30.0, hi: float = 230.0) -> bool:
    """Return True if the frame is neither too dark nor too bright."""
    b = mean_brightness(image)
    return lo <= b <= hi


def face_region_sharpness(
    image: np.ndarray,
    bbox: Optional[tuple[int, int, int, int]] = None,
) -> float:
    """
    Compute Laplacian variance restricted to a face bounding box (if given).

    Args:
        image: Full BGR frame.
        bbox:  (x1, y1, x2, y2) pixel bounding box, or None to use full image.

    Returns:
        Laplacian variance of the crop.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        if x2 > x1 and y2 > y1:
            image = image[y1:y2, x1:x2]
    return laplacian_variance(image)


def composite_quality_score(
    image: np.ndarray,
    face_bbox: Optional[tuple[int, int, int, int]] = None,
    sharpness_weight: float = 0.7,
    brightness_weight: float = 0.3,
) -> float:
    """
    Composite quality score combining sharpness and brightness.

    Returns a value in [0, ∞) where higher is better.
    Brightness component is 1.0 at 127.5 and decays toward 0 at 0 and 255.

    Args:
        image:             BGR frame.
        face_bbox:         Optional face crop for sharpness.
        sharpness_weight:  Contribution of Laplacian variance.
        brightness_weight: Contribution of brightness penalty.

    Returns:
        Composite score (higher = better quality).
    """
    sharpness = face_region_sharpness(image, face_bbox)

    # Brightness penalty: 1.0 at mid-grey, 0.0 at pure black/white
    b = mean_brightness(image)
    brightness_score = 1.0 - abs(b - 127.5) / 127.5

    return sharpness_weight * sharpness + brightness_weight * brightness_score * 100.0


def select_best_frame(
    frames: list[np.ndarray],
    face_bboxes: Optional[list[Optional[tuple[int, int, int, int]]]] = None,
) -> tuple[int, float]:
    """
    Select the index of the best-quality frame from a list.

    Args:
        frames:      List of BGR frames.
        face_bboxes: Optional per-frame face bounding boxes.

    Returns:
        (best_index, best_score)
    """
    if not frames:
        raise ValueError("frames list is empty")

    bboxes = face_bboxes or [None] * len(frames)
    scores = [composite_quality_score(f, b) for f, b in zip(frames, bboxes)]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]


def is_sharp_enough(image: np.ndarray, threshold: float = 40.0) -> bool:
    """Return True if the frame is considered sharp enough for recognition."""
    return laplacian_variance(image) >= threshold


# ── Aliases for test compatibility ──────────────────────────────────────────

def laplacian_sharpness_score(image: np.ndarray) -> float:
    """Alias for laplacian_variance — sharpness score."""
    return laplacian_variance(image)


def image_brightness(image: np.ndarray) -> float:
    """Return mean brightness in [0.0, 1.0]."""
    return mean_brightness(image) / 255.0
