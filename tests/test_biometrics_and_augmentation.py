"""
Tests for ISO/IEC biometric quality assessment, adaptive normalization,
multi-pose template matching, and cafeteria data augmentation.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from cafeteria.recognition.quality import (
    BiometricQualityAssessor,
    adaptive_normalize_face,
    compute_pose_adaptive_threshold,
)
from cafeteria.recognition.matcher import IdentityMatcher
from cafeteria.training.augmentation import (
    CafeteriaAugmentor,
    cutmix_augment,
    random_color_jitter,
    random_gamma_correction,
    random_perspective_tilt,
    simulate_fluorescent_glare,
)


def _synthetic_face_landmarks():
    # 5 standard facial landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
    return np.array([
        [40.0, 45.0],   # left eye
        [88.0, 45.0],   # right eye
        [64.0, 68.0],   # nose tip
        [46.0, 92.0],   # mouth left
        [82.0, 92.0],   # mouth right
    ], dtype=np.float32)


def test_biometric_quality_assessment():
    assessor = BiometricQualityAssessor()
    # Create synthetic face image
    face_img = np.zeros((128, 128, 3), dtype=np.uint8)
    face_img[30:100, 30:100] = 180  # bright center
    cv2.circle(face_img, (40, 45), 5, (255, 255, 255), -1)
    cv2.circle(face_img, (88, 45), 5, (255, 255, 255), -1)

    kps = _synthetic_face_landmarks()
    report = assessor.assess_quality(face_img, kps)

    assert report.ipd_pixels >= 45.0
    assert 0.0 <= report.overall_score <= 100.0
    assert report.pose is not None
    assert isinstance(report.passed, bool)
    assert report.coaching_hint != ""


def test_adaptive_normalize_face():
    # Test on dark image
    dark_face = np.full((112, 112, 3), 25, dtype=np.uint8)
    norm = adaptive_normalize_face(dark_face, target_luminance=128.0)

    assert norm.shape == dark_face.shape
    assert norm.mean() > dark_face.mean()


def test_compute_pose_adaptive_threshold():
    base_thresh = 0.40
    # Direct frontal face (yaw=0, pitch=0) should stay at base threshold
    t_front = compute_pose_adaptive_threshold(base_thresh, yaw=0.0, pitch=0.0)
    assert abs(t_front - base_thresh) < 1e-4

    # Angled face (yaw=25) should relax threshold slightly to adapt to pose degradation
    t_angled = compute_pose_adaptive_threshold(base_thresh, yaw=25.0, pitch=5.0)
    assert t_angled < base_thresh
    assert t_angled >= base_thresh * 0.70  # Never below floor


def test_identity_matcher_multi_pose_and_adaptation():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        p1_dir = root / "p1"
        p1_dir.mkdir(parents=True)

        # Create master embedding
        v1 = np.random.randn(512).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        np.save(p1_dir / "embedding.npy", v1)

        # Create multi-pose gallery: 2 exemplars (front + angled)
        v2 = v1 + np.random.randn(512) * 0.01
        v2 /= np.linalg.norm(v2)
        multi = np.stack([v1, v2], axis=0)
        np.save(p1_dir / "embeddings_multi.npy", multi)

        matcher = IdentityMatcher(
            enrollment_dir=root,
            similarity_threshold=0.40,
        )
        loaded = matcher.load_embeddings()
        assert loaded == 1
        assert "p1" in matcher.multi_embeddings

        # Test matching query close to v2 with yaw
        query = v2 + np.random.randn(512) * 0.005
        query /= np.linalg.norm(query)

        res = matcher.match(query, yaw=15.0, pitch=0.0)
        assert res.person_id == "p1"
        assert res.is_known is True

        # Test online adaptation
        adapted = matcher.adapt_template("p1", query, similarity=0.95)
        assert adapted is True

        # Test adaptation rejected if drift from initial master is too large
        drifted_query = np.random.randn(512).astype(np.float32)
        drifted_query /= np.linalg.norm(drifted_query)
        assert matcher.adapt_template("p1", drifted_query, similarity=0.95) is False


def test_cafeteria_augmentations():
    img = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Gamma
    g = random_gamma_correction(img, gamma_range=(0.7, 1.3))
    assert g.shape == img.shape

    # Glare
    gl = simulate_fluorescent_glare(img, num_spots=1)
    assert gl.shape == img.shape
    assert gl.max() >= img.max()

    # Color jitter
    cj = random_color_jitter(img)
    assert cj.shape == img.shape

    # Perspective
    pt = random_perspective_tilt(img, max_distortion=0.05)
    assert pt.shape == img.shape

    # Cutmix
    img2 = np.full((100, 100, 3), 50, dtype=np.uint8)
    cm, lam = cutmix_augment(img, img2)
    assert cm.shape == img.shape
    assert 0.0 <= lam <= 1.0

    # Pipeline
    augmentor = CafeteriaAugmentor(seed=123)
    out = augmentor.augment(img)
    assert out.shape == img.shape


def test_augment_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir) / "EMPTY"
        d.mkdir(parents=True)
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(d / "base.jpg"), img)

        augmentor = CafeteriaAugmentor(seed=42)
        count = augmentor.augment_dataset({"EMPTY": d}, target_count_per_class=5)
        assert count == 4
        all_imgs = list(d.glob("*.jpg"))
        assert len(all_imgs) == 5
