"""
scripts/benchmark.py — Measure real inference latency on your hardware.

Usage:
    python scripts/benchmark.py

Reports:
  - Camera FPS
  - Plate detection latency
  - Waste classification latency
  - Face detection latency
  - End-to-end latency
  - RAM / GPU memory usage
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import numpy as np
import cv2
import psutil
import os

from cafeteria.config.settings import load_settings, describe_device, system_info
from cafeteria.utils.timing import Stopwatch

BENCHMARK_FRAMES = 50


def make_dummy_frame(h=720, w=1280):
    """Generate a realistic dummy frame (grey gradient with noise)."""
    frame = np.random.randint(80, 180, (h, w, 3), dtype=np.uint8)
    return frame


def bench_camera(cfg):
    print("\n─── Camera Benchmark ───────────────────────────────")
    from cafeteria.camera.webcam import WebcamCamera
    cam = WebcamCamera(
        source=cfg.camera.source,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fps=cfg.camera.fps,
    )
    if not cam.open():
        print("  SKIPPED — cannot open camera")
        return

    fps_samples = []
    start = time.monotonic()
    for _ in range(BENCHMARK_FRAMES):
        frame = cam.read()
        if frame:
            fps_samples.append(1)
    elapsed = time.monotonic() - start
    cam.close()

    actual_fps = len(fps_samples) / elapsed
    print(f"  Frames captured:  {len(fps_samples)}")
    print(f"  Elapsed:          {elapsed:.2f} s")
    print(f"  Actual FPS:       {actual_fps:.1f}")
    return actual_fps


def bench_plate_detector(cfg):
    print("\n─── Plate Detector Benchmark ────────────────────────")
    from cafeteria.training.registry import ModelRegistry
    from cafeteria.detection.plate_detector import PlateDetector, ModelNotFoundError

    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    weights = registry.get_active_weights("plate") or str(cfg.project_root / cfg.models.plate.weights)

    try:
        det = PlateDetector(weights_path=weights, device=cfg.device)
        det.load()
    except ModelNotFoundError:
        print("  SKIPPED — no plate model trained yet")
        return None

    frame = make_dummy_frame()
    latencies = []
    sw = Stopwatch()
    for _ in range(20):
        sw.start()
        det.detect(frame)
        sw.stop()
        latencies.append(sw.elapsed_ms())

    avg = sum(latencies) / len(latencies)
    print(f"  Mean latency:  {avg:.1f} ms")
    print(f"  Min:           {min(latencies):.1f} ms")
    print(f"  Max:           {max(latencies):.1f} ms")
    return avg


def bench_waste_detector(cfg):
    print("\n─── Waste Classifier Benchmark ──────────────────────")
    from cafeteria.training.registry import ModelRegistry
    from cafeteria.detection.waste_detector import WasteDetector, ModelNotFoundError as MNF

    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
    weights = registry.get_active_weights("waste") or str(cfg.project_root / cfg.models.waste.weights)

    try:
        det = WasteDetector(weights_path=weights, device=cfg.device)
        det.load()
    except (MNF, Exception) as e:
        print(f"  SKIPPED — {e!s:.60}")
        return None

    crop = make_dummy_frame(224, 224)
    latencies = []
    sw = Stopwatch()
    for _ in range(20):
        sw.start()
        det.classify(crop)
        sw.stop()
        latencies.append(sw.elapsed_ms())

    avg = sum(latencies) / len(latencies)
    print(f"  Mean latency:  {avg:.1f} ms")
    print(f"  Min:           {min(latencies):.1f} ms")
    print(f"  Max:           {max(latencies):.1f} ms")
    return avg


def bench_face_engine(cfg):
    print("\n─── Face Engine Benchmark ───────────────────────────")
    from cafeteria.recognition.face_engine import FaceEngine
    try:
        fe = FaceEngine(
            model_pack=cfg.recognition.model_pack,
            model_dir=cfg.project_root / "models" / "face",
            det_size=tuple(cfg.recognition.det_size),
            device=cfg.device,
        )
        fe.load()
    except Exception as e:
        print(f"  SKIPPED — {e!s:.60}")
        return None

    frame = make_dummy_frame()
    latencies = []
    sw = Stopwatch()
    for _ in range(10):
        sw.start()
        fe.get_faces(frame)
        sw.stop()
        latencies.append(sw.elapsed_ms())

    avg = sum(latencies) / len(latencies)
    print(f"  Mean latency:  {avg:.1f} ms")
    print(f"  Min:           {min(latencies):.1f} ms")
    print(f"  Max:           {max(latencies):.1f} ms")
    return avg


def bench_memory():
    print("\n─── Memory Usage ────────────────────────────────────")
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024 * 1024)
    total_ram = psutil.virtual_memory().total / (1024 * 1024)
    ram_pct = psutil.virtual_memory().percent
    print(f"  Process RAM:   {ram_mb:.0f} MB")
    print(f"  System RAM:    {ram_mb:.0f} / {total_ram:.0f} MB  ({ram_pct:.1f}%)")

    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            print(f"  VRAM:          {vram:.0f} / {vram_total:.0f} MB")
    except ImportError:
        pass


def main():
    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")

    print("=" * 55)
    print("  Smart Cafeteria Waste Tracker — Benchmark")
    print("=" * 55)

    info = system_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"  device: {cfg.device} ({describe_device(cfg.device)})")

    fps   = bench_camera(cfg)
    p_lat = bench_plate_detector(cfg)
    w_lat = bench_waste_detector(cfg)
    f_lat = bench_face_engine(cfg)
    bench_memory()

    print("\n─── Summary ─────────────────────────────────────────")
    print(f"  Camera FPS:         {fps or '—'}")
    print(f"  Plate latency:      {f'{p_lat:.1f} ms' if p_lat else '—'}")
    print(f"  Waste latency:      {f'{w_lat:.1f} ms' if w_lat else '—'}")
    print(f"  Face latency:       {f'{f_lat:.1f} ms' if f_lat else '—'}")
    print()

    if p_lat and w_lat:
        e2e = (p_lat or 0) + (w_lat or 0) + (f_lat or 0)
        print(f"  Estimated E2E:      {e2e:.1f} ms")
        meets = "✅ YES" if e2e < 1000 else "⚠️ EXCEEDS 1s target"
        print(f"  ≤ 1000 ms target:   {meets}")

    print("=" * 55)


if __name__ == "__main__":
    main()
