"""
Page 3 — Review Queue

Shows unresolved waste events where face was not recognized.
Allows manual confirmation or rejection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Review Queue", page_icon="🔍", layout="wide")

from auth_gate import require_login
require_login()

st.title("🔍 Review Queue")

from cafeteria.config.settings import load_settings
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import (
    PersonRepository, ReviewRepository, TransactionRepository
)
from empty_states import empty_state_html

cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
init_db(cfg.project_root / cfg.storage.database)

session = get_session()
rev_repo = ReviewRepository(session)
tx_repo  = TransactionRepository(session)
per_repo = PersonRepository(session)

unresolved = rev_repo.get_unresolved()
persons = per_repo.get_all_active()
n_tx = tx_repo.count_total()
session.close()

# ── Summary ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**{len(unresolved)} unresolved event(s)** awaiting review.")
with col2:
    if st.button("🔄 Refresh", key="refresh_review"):
        st.rerun()

if not unresolved:
    st.markdown(
        empty_state_html(
            icon="✅",
            title="Review queue is empty",
            body=(
                "There are no unresolved identity reviews. This is expected: "
                f"the live database currently has <b>{n_tx}</b> waste transaction(s). "
                "Reviews appear only when a real plate/waste event is captured "
                "and the face is unrecognized."
            ),
            steps=[
                "Train plate and waste models (Training page) so the pipeline can run.",
                "Start the engine and process a real tray event.",
                "Unknown faces from those events will land here for confirmation.",
            ],
        ),
        unsafe_allow_html=True,
    )
    st.stop()

person_map = {p.person_id: p.name for p in persons}
person_options = {"— Select —": None, **{f"{p.person_id} — {p.name}": p.person_id for p in persons}}

st.markdown("---")

# ── Review cards ──────────────────────────────────────────────────────────────
for entry in unresolved:
    import datetime
    ts = entry.created_at
    dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    with st.container():
        st.markdown(f"### 🔎 Review `{entry.transaction_id}`")
        img_col, info_col, action_col = st.columns([2, 2, 2])

        with img_col:
            if entry.event_image_path and Path(entry.event_image_path).exists():
                st.image(entry.event_image_path, caption="Event Image", use_container_width=True)
            else:
                st.markdown("🖼️ *Image not available*")

        with info_col:
            st.markdown(f"**Timestamp:** {dt}")
            st.markdown(f"**Waste Category:** `{entry.waste_status or '—'}`")
            st.markdown(f"**Confidence:** `{entry.waste_confidence or 0:.2f}`")
            st.markdown(f"**Reason:** {entry.reason or '—'}")

            # Candidates
            candidates = []
            if entry.candidates_json:
                try:
                    candidates = json.loads(entry.candidates_json)
                except Exception:
                    pass

            if candidates:
                st.markdown("**Possible Identities:**")
                for c in candidates[:3]:
                    pid = c.get("person_id", "—")
                    pname = c.get("person_name") or person_map.get(pid, pid)
                    sim = c.get("similarity", 0)
                    st.markdown(f"  - `{pid}` — {pname} ({sim:.3f})")

        with action_col:
            st.markdown("**Assign Identity:**")
            selected_label = st.selectbox(
                "Person",
                list(person_options.keys()),
                key=f"sel_person_{entry.id}",
            )
            selected_pid = person_options[selected_label]

            col_confirm, col_reject = st.columns(2)
            with col_confirm:
                if st.button(
                    "✅ Confirm",
                    key=f"confirm_{entry.id}",
                    disabled=selected_pid is None,
                    type="primary",
                ):
                    pname = person_map.get(selected_pid, selected_pid)
                    sess = get_session()
                    ReviewRepository(sess).resolve(
                        entry.id, selected_pid, pname, resolved_by="manual"
                    )
                    TransactionRepository(sess).update_status(
                        entry.transaction_id,
                        status="MANUALLY_CONFIRMED",
                        person_id=selected_pid,
                        person_name=pname,
                    )
                    sess.commit()
                    sess.close()
                    st.success(f"Confirmed: {selected_pid}")
                    st.rerun()

            with col_reject:
                if st.button("❌ Reject", key=f"reject_{entry.id}"):
                    sess = get_session()
                    ReviewRepository(sess).reject(entry.id)
                    TransactionRepository(sess).update_status(
                        entry.transaction_id, status="REJECTED"
                    )
                    sess.commit()
                    sess.close()
                    st.warning("Event rejected.")
                    st.rerun()

        st.markdown("---")
