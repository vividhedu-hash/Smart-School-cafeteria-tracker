"""
Page 3 — Review Queue

Shows unresolved waste events where face was not recognized.
Allows manual identity confirmation and waste-label correction that
feeds the closed ML training loop.
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

from theme import inject_css

inject_css()

st.title("🔍 Review Queue")

from cafeteria.config.settings import load_settings
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import (
    PersonRepository, ReviewRepository, TransactionRepository
)
from cafeteria.training.dataset_manager import WASTE_CLASSES
from cafeteria.training.learning_loop import LearningLoop, loop_config_from_settings
from cafeteria.training.registry import ModelRegistry
from empty_states import empty_state_html

cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
init_db(cfg.project_root / cfg.storage.database)

registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
ml_loop = LearningLoop(
    project_root=cfg.project_root,
    datasets_dir=cfg.project_root / cfg.storage.datasets,
    models_dir=cfg.project_root / cfg.storage.models,
    registry=registry,
    config=loop_config_from_settings(cfg.training),
    device=cfg.device,
    base_model=cfg.training.default_base_model,
)
loop_snap = ml_loop.snapshot()

session = get_session()
rev_repo = ReviewRepository(session)
tx_repo  = TransactionRepository(session)
per_repo = PersonRepository(session)

unresolved = rev_repo.get_unresolved()
persons = per_repo.get_all_active()
n_tx = tx_repo.count_total()
session.close()

# ── ML loop strip ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pending ML crops", loop_snap["pending_candidates"])
c2.metric("Labels since last train", f"{loop_snap['promoted_since_train']}/{loop_snap['retrain_after_n_labels']}")
c3.metric("Waste dataset images", loop_snap["dataset_total"])
c4.metric("Active waste model", loop_snap["active_waste"])
if loop_snap["ready_to_retrain"]:
    st.info(
        "Enough new labels to retrain. Confirming more waste labels will start a "
        "background waste model run automatically.",
        icon="🧠",
    )
elif loop_snap["training_running"]:
    st.warning("ML loop training is running — see the Training page for epoch progress.", icon="⏳")

pending_crops = ml_loop.list_pending()
if pending_crops:
    with st.expander(f"🧠 ML label queue — {len(pending_crops)} plate crop(s)", expanded=False):
        st.caption(
            "These crops came from live detections. Confirming the waste class "
            "adds them to the training dataset and may trigger a retrain."
        )
        for meta in pending_crops[:12]:
            img_p = Path(meta.get("image_path") or "")
            cols = st.columns([1.2, 2, 1.4, 1])
            with cols[0]:
                if img_p.exists():
                    st.image(str(img_p), width="stretch")
                else:
                    st.caption("crop missing")
            with cols[1]:
                st.markdown(
                    f"`{meta.get('id')}`  \n"
                    f"Predicted: **{meta.get('predicted_label')}** "
                    f"({float(meta.get('confidence') or 0):.2f})  \n"
                    f"TX: `{meta.get('transaction_id')}`"
                )
            with cols[2]:
                label = st.selectbox(
                    "Correct class",
                    WASTE_CLASSES,
                    index=WASTE_CLASSES.index(meta["predicted_label"])
                    if meta.get("predicted_label") in WASTE_CLASSES else 0,
                    key=f"ml_label_{meta['id']}",
                )
            with cols[3]:
                if st.button("Promote", key=f"ml_promote_{meta['id']}", type="primary"):
                    try:
                        result = ml_loop.promote(meta["id"], label, source="review_queue")
                        msg = f"Added {result['added']} to {label}."
                        if result.get("retrain", {}).get("started"):
                            msg += f" Retrain started ({result['retrain'].get('version')})."
                        st.success(msg)
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

# ── Summary ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**{len(unresolved)} unresolved identity review(s)**.")
with col2:
    if st.button("🔄 Refresh", key="refresh_review"):
        st.rerun()

if not unresolved and not pending_crops:
    st.markdown(
        empty_state_html(
            icon="✅",
            title="Review queue is empty",
            body=(
                "There are no unresolved identity reviews and no pending ML crops. "
                f"The live database currently has <b>{n_tx}</b> waste transaction(s). "
                "Reviews appear when a tray event is captured with an unrecognized face; "
                "plate crops appear after every waste event for labeling."
            ),
            steps=[
                "Start the engine and process a real tray event.",
                "Unknown faces land here for identity confirmation.",
                "Confirm waste labels on plate crops to grow the training set.",
            ],
        ),
        unsafe_allow_html=True,
    )
    st.stop()

if not unresolved:
    st.stop()

person_map = {p.person_id: p.name for p in persons}
person_options = {"— Select —": None, **{f"{p.person_id} — {p.name}": p.person_id for p in persons}}

st.markdown("---")

# ── Review cards ──────────────────────────────────────────────────────────────
for entry in unresolved:
    # [AI-CoLab: Verified by Antigravity] Review Queue with Closed-Loop ML Crop Labeling
    import datetime
    ts = entry.created_at
    dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    # Prefer plate crop for ML labeling when the transaction saved one
    plate_crop_path = None
    sess_lookup = get_session()
    try:
        tx = TransactionRepository(sess_lookup).get_by_id(entry.transaction_id)
        if tx and tx.plate_image_path and Path(tx.plate_image_path).exists():
            plate_crop_path = tx.plate_image_path
    finally:
        sess_lookup.close()

    with st.container():
        st.markdown(f"### 🔎 Review `{entry.transaction_id}`")
        img_col, info_col, action_col = st.columns([2, 2, 2])

        with img_col:
            if entry.event_image_path and Path(entry.event_image_path).exists():
                st.image(entry.event_image_path, caption="Event Image", width="stretch")
            else:
                st.markdown("🖼️ *Image not available*")
            if plate_crop_path:
                st.image(plate_crop_path, caption="Plate crop (ML)", width="stretch")

        with info_col:
            st.markdown(f"**Timestamp:** {dt}")
            st.markdown(f"**Waste Category:** `{entry.waste_status or '—'}`")
            st.markdown(f"**Confidence:** `{entry.waste_confidence or 0:.2f}`")
            st.markdown(f"**Reason:** {entry.reason or '—'}")

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

            waste_default = entry.waste_status if entry.waste_status in WASTE_CLASSES else WASTE_CLASSES[0]
            waste_label = st.selectbox(
                "Confirm waste label (feeds training)",
                WASTE_CLASSES,
                index=WASTE_CLASSES.index(waste_default),
                key=f"waste_label_{entry.id}",
            )
            feed_ml = st.checkbox(
                "Add plate crop to training set",
                value=True,
                key=f"feed_ml_{entry.id}",
            )

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
                    # Persist corrected waste status on the transaction
                    tx = TransactionRepository(sess).get_by_id(entry.transaction_id)
                    if tx is not None:
                        tx.waste_status = waste_label
                    sess.commit()
                    sess.close()

                    if feed_ml:
                        paths = []
                        if plate_crop_path:
                            paths.append(plate_crop_path)
                        elif entry.event_image_path and Path(entry.event_image_path).exists():
                            paths.append(entry.event_image_path)
                        # Prefer matching pending candidate for this transaction
                        matched = next(
                            (c for c in ml_loop.list_pending()
                             if c.get("transaction_id") == entry.transaction_id),
                            None,
                        )
                        try:
                            if matched:
                                result = ml_loop.promote(
                                    matched["id"], waste_label, source="identity_review"
                                )
                            elif paths:
                                result = ml_loop.promote_from_paths(
                                    paths, waste_label, trigger_retrain=True
                                )
                            else:
                                result = {"added": 0, "retrain": {}}
                            if result.get("retrain", {}).get("started"):
                                st.success(
                                    f"Confirmed {selected_pid}. Waste → {waste_label}. "
                                    f"Retrain started ({result['retrain'].get('version')})."
                                )
                            else:
                                st.success(
                                    f"Confirmed {selected_pid}. "
                                    f"Added {result.get('added', 0)} image(s) as {waste_label}."
                                )
                        except Exception as exc:
                            st.warning(f"Identity saved, but ML promote failed: {exc}")
                    else:
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
