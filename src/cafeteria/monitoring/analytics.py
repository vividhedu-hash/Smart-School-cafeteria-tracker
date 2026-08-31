"""
Empty-safe analytics helpers for the dashboard.

All functions accept zero rows / missing attributes and return empty
structures that Plotly and pandas can skip without crashing. They never
invent transactions or waste labels.
"""
from __future__ import annotations

import datetime
from typing import Any, Iterable, Optional


def waste_pie_payload(waste_counts: Optional[dict[str, int]]) -> tuple[list[str], list[int]]:
    """Return (labels, values) including only classes with count > 0."""
    if not waste_counts:
        return [], []
    labels = [k for k, v in waste_counts.items() if v and v > 0]
    values = [int(waste_counts[k]) for k in labels]
    return labels, values


def events_over_time_rows(transactions: Optional[Iterable[Any]]) -> list[dict]:
    """One {date, waste} row per transaction. Empty list if none."""
    rows: list[dict] = []
    for tx in transactions or []:
        ts = getattr(tx, "timestamp", None)
        if ts is None:
            continue
        try:
            day = datetime.datetime.fromtimestamp(float(ts)).date()
        except (TypeError, ValueError, OSError, OverflowError):
            continue
        rows.append({
            "date": day,
            "waste": getattr(tx, "waste_status", None) or "UNKNOWN",
        })
    return rows


def person_waste_rows(transactions: Optional[Iterable[Any]]) -> list[dict]:
    """Rows for non-empty waste events grouped by person. Empty if none."""
    rows: list[dict] = []
    for tx in transactions or []:
        waste = getattr(tx, "waste_status", None)
        if waste in (None, "EMPTY"):
            continue
        rows.append({
            "person": (
                getattr(tx, "person_name", None)
                or getattr(tx, "person_id", None)
                or "UNKNOWN"
            ),
            "waste": waste,
        })
    return rows


def status_counts(transactions: Optional[Iterable[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tx in transactions or []:
        status = getattr(tx, "status", None) or "UNKNOWN"
        counts[status] = counts.get(status, 0) + 1
    return counts


def latency_values(transactions: Optional[Iterable[Any]]) -> list[float]:
    out: list[float] = []
    for tx in transactions or []:
        v = getattr(tx, "processing_latency_ms", None)
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


PIPELINE_GATED_COPY = (
    "No waste events have been written to SQLite yet. The live pipeline runs "
    "on the built-in OpenCV visual backends (or on trained YOLO models once "
    "you activate them). Face recognition still runs live. No sample rows "
    "are shown."
)
