"""
Timing utilities — high-resolution wall-clock timers for latency measurement.
Cross-platform (uses time.perf_counter which is monotonic on all platforms).
"""
from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from typing import Deque, Generator, Optional


class Stopwatch:
    """
    Simple high-resolution stopwatch.

    Usage::

        sw = Stopwatch()
        sw.start()
        # ... work ...
        elapsed_ms = sw.elapsed_ms()
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> "Stopwatch":
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        """Stop and return elapsed time in seconds."""
        if self._start is None:
            raise RuntimeError("Stopwatch not started")
        self._end = time.perf_counter()
        return self._end - self._start

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds (does NOT stop the watch)."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return (end - self._start) * 1000.0

    def elapsed_s(self) -> float:
        """Return elapsed seconds (does NOT stop the watch)."""
        return self.elapsed_ms() / 1000.0

    def lap_ms(self) -> float:
        """Return time elapsed so far in milliseconds without stopping."""
        return self.elapsed_ms()

    def reset(self) -> "Stopwatch":
        self._start = None
        self._end = None
        return self


@contextmanager
def timed(label: str = "") -> Generator[Stopwatch, None, None]:
    """
    Context manager that measures elapsed time.

    Usage::

        with timed("plate inference") as sw:
            result = model(frame)
        print(f"Took {sw.elapsed_ms():.1f} ms")
    """
    sw = Stopwatch()
    sw.start()
    try:
        yield sw
    finally:
        sw.stop()


class RollingAverage:
    """
    Track a rolling average of the last N samples.
    Used for FPS, latency, etc.
    """

    def __init__(self, maxlen: int = 60) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)

    def update(self, value: float) -> None:
        self._buf.append(value)

    def mean(self) -> float:
        if not self._buf:
            return 0.0
        return sum(self._buf) / len(self._buf)

    def latest(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    def min(self) -> float:
        return min(self._buf) if self._buf else 0.0

    def max(self) -> float:
        return max(self._buf) if self._buf else 0.0

    def __len__(self) -> int:
        return len(self._buf)


class FPSCounter:
    """
    Measure actual frames-per-second using a rolling window.
    """

    def __init__(self, window: int = 30) -> None:
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def fps(self) -> float:
        """Return current FPS without updating."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
