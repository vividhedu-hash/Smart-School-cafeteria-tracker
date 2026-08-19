"""
Health monitoring — system resource metrics and component status.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import psutil

from cafeteria.utils.logging import get_logger

logger = get_logger("monitoring.health")


def get_system_metrics() -> dict:
    """
    Return current system resource utilisation.

    Returns:
        Dict with cpu_percent, ram_mb, ram_percent, gpu_vram_mb (if available).
    """
    metrics: dict = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "ram_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu_vram_mb": None,
        "gpu_util_percent": None,
        "timestamp": time.time(),
    }

    # Try NVIDIA GPU
    try:
        import torch
        if torch.cuda.is_available():
            idx = 0
            metrics["gpu_vram_mb"] = (
                torch.cuda.memory_allocated(idx) / (1024 * 1024)
            )
            metrics["gpu_vram_total_mb"] = (
                torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
            )
    except Exception:
        pass

    return metrics


def component_status(
    camera_connected: bool,
    plate_detector_loaded: bool,
    waste_detector_loaded: bool,
    face_engine_loaded: bool,
    db_ok: bool,
) -> dict:
    """Return a dict of component statuses for the dashboard."""
    return {
        "camera": "OK" if camera_connected else "ERROR",
        "plate_detector": "OK" if plate_detector_loaded else "NOT_LOADED",
        "waste_detector": "OK" if waste_detector_loaded else "NOT_LOADED",
        "face_engine": "OK" if face_engine_loaded else "NOT_LOADED",
        "database": "OK" if db_ok else "ERROR",
    }


def compute_readiness(
    plate_model_loaded: bool,
    waste_model_loaded: bool,
    face_engine_loaded: bool,
    camera_connected: bool,
    enrolled_count: int,
    plate_proxy_mode: bool = False,
    plate_backend: str = "",
    waste_backend: str = "",
) -> dict:
    """
    Compute whether the pipeline is able to commit transactions.

    `ready` is True when the camera is connected AND a plate detector is
    loaded (YOLO, OpenCV visual, or explicit COCO proxy) AND a waste
    classifier is loaded (YOLO or OpenCV visual). Face / enrollment gaps
    do not block readiness — those events go to the review queue.

    Returns a JSON-serialisable dict for runtime_state.json.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    if not camera_connected:
        blocking.append("CAMERA NOT CONNECTED")
    if not plate_model_loaded:
        blocking.append("PLATE DETECTOR NOT LOADED — pipeline disabled")
    if not waste_model_loaded:
        blocking.append("WASTE CLASSIFIER NOT LOADED — pipeline disabled")

    if plate_proxy_mode:
        warnings.append(
            "PLATE PROXY MODE — COCO stand-in detector; all transactions "
            "flagged for review"
        )
    if plate_backend == "visual":
        warnings.append(
            "Plate detector is OpenCV visual (not a trained YOLO). "
            "Train a plate model for higher accuracy."
        )
    if waste_backend == "visual":
        warnings.append(
            "Waste classifier is OpenCV visual occupancy (not a trained YOLO). "
            "Train a waste model for higher accuracy."
        )
    if not face_engine_loaded:
        warnings.append("FACE ENGINE NOT LOADED — all transactions go to review")
    elif enrolled_count == 0:
        warnings.append("NO FACES ENROLLED — all transactions go to review")

    return {
        "ready": len(blocking) == 0,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "plate_model": plate_model_loaded,
        "waste_model": waste_model_loaded,
        "face_engine": face_engine_loaded,
        "camera": camera_connected,
        "enrolled_count": enrolled_count,
        "plate_proxy_mode": plate_proxy_mode,
        "waste_model_missing": not waste_model_loaded,
        "plate_backend": plate_backend or ("coco_proxy" if plate_proxy_mode else ""),
        "waste_backend": waste_backend,
    }
