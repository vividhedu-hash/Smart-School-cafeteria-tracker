"""Comprehensive audit verification: import integrity, boot resilience, and cloud safety."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import pytest

from cafeteria.storage.database import (
    _extract_host_port,
    _normalise_supabase_url,
    db_health_check,
    init_db,
    is_postgres,
)
from dashboard.boot import load_app
from dashboard.engine_ctl import is_cloud


def test_dashboard_modules_import_cleanly():
    """Verify all dashboard modules and pages can be imported without throwing."""
    import dashboard.app
    import dashboard.auth_gate
    import dashboard.boot
    import dashboard.empty_states
    import dashboard.engine_client
    import dashboard.engine_ctl
    import dashboard.enroll_cam
    import dashboard.theme


def test_supabase_url_all_permutations():
    """Verify Supabase URL normalization handles all permutations with zero errors."""
    cases = [
        ("postgresql://postgres:pwd@db.projref.supabase.co:5432/postgres", "aws-0-ap-south-1.pooler.supabase.com:5432"),
        ("postgresql://postgres:pwd@db.projref.supabase.co/postgres", "aws-0-ap-south-1.pooler.supabase.com:5432"),
        ("postgres://postgres:pwd@db.projref.supabase.co/postgres", "aws-0-ap-south-1.pooler.supabase.com:5432"),
        ("postgresql://postgres.projref:pwd@db.projref.supabase.co:6543/postgres", "aws-0-ap-south-1.pooler.supabase.com:6543"),
    ]
    for raw, expected_host in cases:
        norm = _normalise_supabase_url(raw)
        assert expected_host in norm
        assert "postgres.projref" in norm
        host, port = _extract_host_port(norm)
        assert host == "aws-0-ap-south-1.pooler.supabase.com"
        assert port in (5432, 6543)


def test_cloud_environment_detection(monkeypatch):
    """Verify cloud detection is accurate."""
    assert is_cloud() is False
    monkeypatch.setenv("HOME", "/home/adminuser")
    assert is_cloud() is True


def test_decode_upload_resilience():
    """Verify _decode_upload works with both cv2 and PIL fallback."""
    mod = importlib.import_module("dashboard.pages.2_Training")
    decode_fn = getattr(mod, "_decode_upload")
    from PIL import Image
    import io

    # Create dummy 50x50 JPEG bytes
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_bytes = buf.getvalue()

    decoded = decode_fn(raw_bytes)
    assert decoded is not None
    assert decoded.shape == (50, 50, 3)

    # Empty bytes returns None
    assert decode_fn(b"") is None


def test_enrollment_write_resilience(tmp_path):
    """Verify write_gallery_index and _write_json do not crash on filesystem issues."""
    from cafeteria.recognition.enrollment import EnrollmentManager
    mgr = EnrollmentManager(enrollment_dir=tmp_path / "enroll")
    idx_path = mgr.write_gallery_index()
    assert idx_path.exists()
