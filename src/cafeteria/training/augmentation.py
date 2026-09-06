"""
Adaptive Cafeteria Data Augmentation Pipeline.

Designed specifically for cafeteria environmental conditions:
  - Specular glare & stainless steel / ceramic reflections
  - Fluorescent vs daylight illumination shifts
  - Sneeze guard shadows and color temperature jitter
  - Perspective distortions from overhead / angled cameras
  - Conveyor motion blur and sensor noise
  - CutMix and Mosaic synthesis for overlapping waste items
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from cafeteria.utils.logging import get_logger

logger = get_logger("training.augmentation")


def random_gamma_correction(
    image: np.ndarray,
    gamma_range: Tuple[float, float] = (0.65, 1.45),
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Apply non-linear gamma adjustment to simulate harsh or dim cafeteria lighting."""
    r = rng or random
    gamma = r.uniform(*gamma_range)
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def simulate_fluorescent_glare(
    image: np.ndarray,
    num_spots: int = 1,
    max_intensity: float = 0.4,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Simulate bright specular spots from cafeteria overhead lights on shiny plates/trays."""
    r = rng or random
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    for _ in range(num_spots):
        cx = r.randint(int(w * 0.2), int(w * 0.8))
        cy = r.randint(int(h * 0.2), int(h * 0.8))
        radius = r.randint(int(min(h, w) * 0.08), int(min(h, w) * 0.25))
        intensity = r.uniform(0.2, max_intensity)

        # Create radial gradient glare spot
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        glare_mask = np.clip(1.0 - (dist_from_center / max(radius, 1.0)), 0, 1.0)
        glare_mask = (glare_mask ** 2) * (intensity * 255.0)

        if len(image.shape) == 3:
            glare_mask = np.expand_dims(glare_mask, axis=-1)

        out = np.clip(out + glare_mask, 0, 255)

    return out.astype(np.uint8)


def random_color_jitter(
    image: np.ndarray,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.25),
    saturation_range: Tuple[float, float] = (0.75, 1.25),
    hue_delta: int = 12,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Apply photometric color jittering in HSV and BGR color spaces."""
    r = rng or random
    img_f = image.astype(np.float32)

    # Brightness & contrast
    alpha = r.uniform(*contrast_range)
    beta = (r.uniform(*brightness_range) - 1.0) * 128.0
    img_f = np.clip(img_f * alpha + beta, 0, 255).astype(np.uint8)

    # Saturation & Hue in HSV
    hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_delta = r.uniform(-hue_delta, hue_delta)
    s_scale = r.uniform(*saturation_range)

    hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_perspective_tilt(
    image: np.ndarray,
    max_distortion: float = 0.12,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Apply slight 3D perspective warp simulating different overhead camera tilt angles."""
    r = rng or random
    h, w = image.shape[:2]
    dx = w * max_distortion
    dy = h * max_distortion

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [r.uniform(0, dx), r.uniform(0, dy)],
        [w - r.uniform(0, dx), r.uniform(0, dy)],
        [w - r.uniform(0, dx), h - r.uniform(0, dy)],
        [r.uniform(0, dx), h - r.uniform(0, dy)],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101
    )


def random_blur_or_noise(
    image: np.ndarray,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Simulate motion blur or low-light sensor noise."""
    r = rng or random
    choice = r.choice(["none", "blur", "noise", "median"])
    if choice == "blur":
        ksize = r.choice([3, 5])
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif choice == "median":
        return cv2.medianBlur(image, 3)
    elif choice == "noise":
        h, w, c = image.shape
        sigma = r.uniform(4.0, 12.0)
        gauss = np.random.normal(0, sigma, (h, w, c))
        noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return noisy
    return image


def cutmix_augment(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, float]:
    """
    CutMix augmentation: cut a patch from img2 and paste it onto img1.
    Returns augmented image and lambda mix ratio.
    """
    r = rng or random
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = r.randint(0, w)
    cy = r.randint(0, h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    out = img1.copy()
    out[bby1:bby2, bbx1:bbx2] = img2[bby1:bby2, bbx1:bbx2]

    # Adjusted lambda based on true pixel area
    actual_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return out, actual_lam


class CafeteriaAugmentor:
    """
    Configurable cafeteria data augmentor for training robustness.
    """

    def __init__(
        self,
        seed: int = 42,
        enable_lighting: bool = True,
        enable_perspective: bool = True,
        enable_noise: bool = True,
        enable_glare: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.enable_lighting = enable_lighting
        self.enable_perspective = enable_perspective
        self.enable_noise = enable_noise
        self.enable_glare = enable_glare

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random combination of cafeteria augmentations to a single image."""
        if image is None or image.size == 0:
            return image

        out = image.copy()

        # Horizontal flip (50% probability)
        if self.rng.random() > 0.5:
            out = cv2.flip(out, 1)

        # Perspective distortion (50% probability)
        if self.enable_perspective and self.rng.random() > 0.4:
            out = random_perspective_tilt(out, max_distortion=0.08, rng=self.rng)

        # Lighting & color jitter (70% probability)
        if self.enable_lighting and self.rng.random() > 0.3:
            out = random_color_jitter(out, rng=self.rng)
            out = random_gamma_correction(out, rng=self.rng)

        # Overhead glare (30% probability)
        if self.enable_glare and self.rng.random() > 0.7:
            out = simulate_fluorescent_glare(out, num_spots=1, rng=self.rng)

        # Blur / sensor noise (40% probability)
        if self.enable_noise and self.rng.random() > 0.6:
            out = random_blur_or_noise(out, rng=self.rng)

        return out

    def augment_dataset(
        self,
        class_dirs: dict[str, Path],
        target_count_per_class: int = 20,
    ) -> int:
        """
        Synthesize augmented training images to balance under-represented classes.
        """
        generated = 0
        for class_name, folder in class_dirs.items():
            if not folder.exists():
                continue
            images = [
                f for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                and not f.name.startswith("aug_")
            ]
            if not images:
                continue

            current_count = len(images)
            needed = max(0, target_count_per_class - current_count)
            if needed <= 0:
                continue

            for i in range(needed):
                src_path = self.rng.choice(images)
                img = cv2.imread(str(src_path))
                if img is None:
                    continue

                aug_img = self.augment(img)
                dest_path = folder / f"aug_{int(self.rng.random() * 1e8)}_{i:04d}.jpg"
                if cv2.imwrite(str(dest_path), aug_img):
                    generated += 1

        logger.info("CafeteriaAugmentor generated %d augmented training images", generated)
        return generated
