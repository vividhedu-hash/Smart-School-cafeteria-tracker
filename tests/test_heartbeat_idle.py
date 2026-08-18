"""Idle-shutdown: camera is released when Live Monitor stops heartbeating."""
from cafeteria.monitoring.heartbeat import should_idle_shutdown


def test_grace_period_keeps_engine_up_without_heartbeat():
    assert should_idle_shutdown(
        now=10.0, started_at=0.0, last_heartbeat=None,
        idle_seconds=8.0, startup_grace_seconds=25.0,
    ) is False


def test_no_heartbeat_after_grace_shuts_down():
    assert should_idle_shutdown(
        now=30.0, started_at=0.0, last_heartbeat=None,
        idle_seconds=8.0, startup_grace_seconds=25.0,
    ) is True


def test_fresh_heartbeat_keeps_engine_up():
    assert should_idle_shutdown(
        now=40.0, started_at=0.0, last_heartbeat=38.0,
        idle_seconds=8.0, startup_grace_seconds=25.0,
    ) is False


def test_stale_heartbeat_releases_camera():
    assert should_idle_shutdown(
        now=40.0, started_at=0.0, last_heartbeat=20.0,
        idle_seconds=8.0, startup_grace_seconds=25.0,
    ) is True
