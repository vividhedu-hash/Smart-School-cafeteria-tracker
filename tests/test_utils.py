"""
Tests for utility modules: timing, image_quality, logging.
"""
from __future__ import annotations

import time
import numpy as np
import pytest


def test_stopwatch_elapsed():
    from cafeteria.utils.timing import Stopwatch
    sw = Stopwatch()
    sw.start()
    time.sleep(0.05)
    sw.stop()
    assert sw.elapsed_ms() >= 40.0  # allow generous margin


def test_stopwatch_lap():
    from cafeteria.utils.timing import Stopwatch
    sw = Stopwatch()
    sw.start()
    time.sleep(0.02)
    lap1 = sw.lap_ms()
    time.sleep(0.02)
    lap2 = sw.lap_ms()
    assert lap2 > 0


def test_fps_counter():
    from cafeteria.utils.timing import FPSCounter
    counter = FPSCounter(window=10)
    for _ in range(5):
        counter.tick()
        time.sleep(0.01)
    assert counter.fps > 0


def test_rolling_average():
    from cafeteria.utils.timing import RollingAverage
    avg = RollingAverage(maxlen=5)
    for i in [1, 2, 3, 4, 5]:
        avg.update(float(i))
    assert abs(avg.mean() - 3.0) < 0.001

    avg.update(10.0)
    # Now: [2, 3, 4, 5, 10], mean = 4.8
    assert abs(avg.mean() - 4.8) < 0.001


def test_laplacian_sharpness_score():
    from cafeteria.utils.image_quality import laplacian_sharpness_score
    # Blurry: all zeros
    blurry = np.zeros((100, 100, 3), dtype=np.uint8)
    # Sharp: noise
    sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    assert laplacian_sharpness_score(sharp) > laplacian_sharpness_score(blurry)


def test_select_best_frame():
    from cafeteria.utils.image_quality import select_best_frame
    frames = [
        np.zeros((100, 100, 3), dtype=np.uint8),            # black, score 0
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),  # noisy, high score
        np.ones((100, 100, 3), dtype=np.uint8) * 128,       # flat grey
    ]
    idx, score = select_best_frame(frames)
    assert idx == 1  # noisy frame should win
    assert score > 0


def test_image_brightness():
    from cafeteria.utils.image_quality import image_brightness
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert image_brightness(black) == 0.0
    assert image_brightness(white) == 1.0
