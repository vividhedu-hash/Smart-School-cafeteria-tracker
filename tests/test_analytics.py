"""Empty-safe analytics helpers — never invent rows."""
from __future__ import annotations

from types import SimpleNamespace

from cafeteria.monitoring.analytics import (
    events_over_time_rows,
    latency_values,
    person_waste_rows,
    status_counts,
    waste_pie_payload,
)


def test_waste_pie_payload_empty():
    assert waste_pie_payload({}) == ([], [])
    assert waste_pie_payload(None) == ([], [])
    assert waste_pie_payload({"EMPTY": 0, "HIGH_WASTE": 0}) == ([], [])


def test_waste_pie_payload_nonzero_only():
    labels, values = waste_pie_payload({"EMPTY": 2, "HIGH_WASTE": 0, "LOW_WASTE": 5})
    assert labels == ["EMPTY", "LOW_WASTE"]
    assert values == [2, 5]


def test_events_over_time_skips_invalid_timestamp():
    bad = SimpleNamespace(timestamp="not-a-time", waste_status="EMPTY")
    assert events_over_time_rows([bad]) == []


def test_events_over_time_rows():
    tx = SimpleNamespace(timestamp=1_700_000_000.0, waste_status="HIGH_WASTE")
    rows = events_over_time_rows([tx])
    assert len(rows) == 1
    assert rows[0]["waste"] == "HIGH_WASTE"


def test_person_waste_skips_empty():
    txs = [
        SimpleNamespace(person_name="Ada", person_id="p1", waste_status="EMPTY"),
        SimpleNamespace(person_name=None, person_id="p2", waste_status="LOW_WASTE"),
    ]
    rows = person_waste_rows(txs)
    assert len(rows) == 1
    assert rows[0]["person"] == "p2"


def test_status_and_latency_empty():
    assert status_counts([]) == {}
    assert latency_values([]) == []
    assert latency_values([SimpleNamespace(processing_latency_ms=None)]) == []
    assert latency_values([SimpleNamespace(processing_latency_ms=12.5)]) == [12.5]
    assert status_counts([SimpleNamespace(status="AUTO_CONFIRMED")]) == {
        "AUTO_CONFIRMED": 1
    }
