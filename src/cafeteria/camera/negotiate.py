"""
Camera capability negotiation — no OpenCV required for the ranking logic.

Any-device rules:
  - ``source: auto`` tries webcam indices 0, 1, 2, 3 (first that opens wins).
  - ``width/height: auto`` (0) uses the camera's native mode, capped so a 4K
    laptop webcam stays real-time.
  - An explicit width×height is tried first, then a fallback ladder, then native.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

# Highest-first. 720p is the cafeteria sweet spot (faces + plates); 1080p is
# used when the device can feed it without dropping.
CAPTURE_LADDER: tuple[tuple[int, int], ...] = (
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (800, 600),
    (640, 480),
    (320, 240),
)

DEFAULT_MAX_CAPTURE_WIDTH = 1920
PROBE_INDEX_LIMIT = 4


def is_auto_source(source: Any) -> bool:
    if source is None:
        return True
    if isinstance(source, str) and source.strip().lower() in ("auto", "", "default"):
        return True
    if source == -1:
        return True
    return False


def is_auto_size(width: Any, height: Any) -> bool:
    def _zero(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and v.strip().lower() in ("auto", "", "0"):
            return True
        try:
            return int(v) <= 0
        except (TypeError, ValueError):
            return True

    return _zero(width) or _zero(height)


def candidate_indices(source: Any, limit: int = PROBE_INDEX_LIMIT) -> list[int]:
    """Webcam indices to try, in order."""
    if is_auto_source(source):
        return list(range(max(1, int(limit))))
    try:
        idx = int(source)
    except (TypeError, ValueError):
        return [0]
    if idx < 0:
        return list(range(max(1, int(limit))))
    # Prefer the configured index, then neighbours (unplugged default cam).
    neighbours = [i for i in range(max(1, int(limit))) if i != idx]
    return [idx] + neighbours


def cap_to_max(width: int, height: int, max_width: int = DEFAULT_MAX_CAPTURE_WIDTH) -> tuple[int, int]:
    """Keep aspect ratio; never request more than max_width."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return 1280, 720
    cap = int(max_width) if max_width else DEFAULT_MAX_CAPTURE_WIDTH
    if w <= cap:
        return w, h
    scale = cap / float(w)
    return cap, max(1, int(round(h * scale)))


def rank_capture_sizes(
    *,
    native_width: int = 0,
    native_height: int = 0,
    requested_width: int = 0,
    requested_height: int = 0,
    max_capture_width: int = DEFAULT_MAX_CAPTURE_WIDTH,
) -> list[tuple[int, int]]:
    """
    Ordered unique (w, h) to try on an open capture.

    Auto (requested 0): native (capped), then every ladder size the camera
    can plausibly do. Explicit request is tried first.
    """
    ordered: list[tuple[int, int]] = []

    def _add(w: int, h: int) -> None:
        if w < 160 or h < 120:
            return
        pair = cap_to_max(w, h, max_capture_width)
        if pair not in ordered:
            ordered.append(pair)

    auto = is_auto_size(requested_width, requested_height)
    native_w, native_h = int(native_width or 0), int(native_height or 0)

    if not auto:
        _add(int(requested_width), int(requested_height))

    if native_w >= 160 and native_h >= 120:
        _add(native_w, native_h)

    # Prefer HD/FHD from the ladder when the native sensor is at least that big
    # (or unknown — OpenCV often reports 0 until the first grab).
    for w, h in CAPTURE_LADDER:
        if native_w and native_h and (w > native_w + 16 or h > native_h + 16):
            continue
        _add(w, h)

    if not ordered:
        ordered.append((1280, 720))
    return ordered


def pick_verified_size(
    verified: Iterable[tuple[int, int]],
    *,
    requested_width: int = 0,
    requested_height: int = 0,
    max_capture_width: int = DEFAULT_MAX_CAPTURE_WIDTH,
) -> Optional[tuple[int, int]]:
    """Choose the largest verified mode that stays under the capture cap."""
    auto = is_auto_size(requested_width, requested_height)
    scored: list[tuple[int, tuple[int, int]]] = []
    for w, h in verified:
        if w < 160 or h < 120:
            continue
        pixels = w * h
        if w > max_capture_width + 16:
            continue
        if not auto:
            # Prefer closest to the request, then pixels.
            dw = abs(w - int(requested_width))
            dh = abs(h - int(requested_height))
            score = -(dw + dh) * 1000 + pixels
        else:
            score = pixels
        scored.append((score, (w, h)))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]
