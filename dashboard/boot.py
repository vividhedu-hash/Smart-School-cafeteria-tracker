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
    Guaranteed never to raise unhandled exceptions.
    """
    import os
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["DATABASE_URL", "CAFETERIA_OPERATOR_PIN", "STREAMLIT_SERVER_PORT"]:
                if key in st.secrets and not os.environ.get(key):
                    os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

    try:
        cfg = load_settings(config_path=PROJECT_ROOT / "configs" / "config.yaml")
    except Exception:
        from cafeteria.config.settings import Settings
        cfg = Settings(project_root=PROJECT_ROOT)

    try:
        ensure_runtime_storage(cfg)
    except Exception:
        pass

    return cfg
