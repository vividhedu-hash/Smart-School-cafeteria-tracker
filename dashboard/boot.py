"""Shared dashboard bootstrap: settings + SQLite + enrollment sync."""
from __future__ import annotations

from pathlib import Path

from cafeteria.config.settings import load_settings
from cafeteria.storage.bootstrap import ensure_runtime_storage

_DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DASHBOARD_DIR.parent


def load_app():
    """Load YAML settings and make sure the database exists.

    Streamlit multipage scripts do not share ``app.py``'s ``@st.cache_resource``
    init, so every page must call this (or ``init_db``) before ``get_session``.
    """
    cfg = load_settings(config_path=PROJECT_ROOT / "configs" / "config.yaml")
    ensure_runtime_storage(cfg)
    return cfg
