"""Streamlit session gate — required on every dashboard page."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

_DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DASHBOARD_DIR.parent


def _sync_secrets() -> None:
    """Push st.secrets into os.environ before any auth logic runs."""
    try:
        for key in ("CAFETERIA_OPERATOR_PIN", "DATABASE_URL", "CAFETERIA_ENGINE_TOKEN"):
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


def require_login() -> None:
    """Block the page until the operator PIN is verified in this browser session."""
    _sync_secrets()  # must run before ensure_operator_pin reads os.environ

    from cafeteria.auth import ensure_operator_pin, verify_operator_pin

    if st.session_state.get("operator_ok"):
        with st.sidebar:
            st.caption("Operator session")
            if st.button("Lock dashboard", key="auth_lock"):
                st.session_state.operator_ok = False
                st.rerun()
        return

    status = ensure_operator_pin(PROJECT_ROOT)
    st.title("Operator login")
    st.caption("This dashboard is PIN-gated. It is not open on the LAN with zero auth.")

    if status.generated and status.pin:
        st.session_state["generated_pin"] = status.pin
        st.session_state["generated_pin_path"] = str(status.path or "cloud-memory")

    shown = st.session_state.get("generated_pin")
    if shown:
        path = st.session_state.get("generated_pin_path", "data/.operator_pin")
        st.warning(
            f"No `CAFETERIA_OPERATOR_PIN` was set, so a PIN was generated. "
            f"**PIN: `{shown}`** — enter it below. "
            "Set `CAFETERIA_OPERATOR_PIN` in Streamlit secrets to make it permanent."
        )
    elif status.source == "env":
        st.info("PIN is set via `CAFETERIA_OPERATOR_PIN`.")
    elif status.path:
        st.caption(f"PIN file: `{status.path}`")

    pin = st.text_input("Operator PIN", type="password", key="operator_pin_input")
    if st.button("Unlock", type="primary"):
        if verify_operator_pin(pin, PROJECT_ROOT):
            st.session_state.operator_ok = True
            st.session_state.pop("generated_pin", None)
            st.rerun()
        else:
            st.error("Wrong PIN.")
    st.stop()
