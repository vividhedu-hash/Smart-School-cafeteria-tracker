"""
Page 4 — Transactions

Searchable, filterable table of all waste events with CSV export.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import datetime
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Transactions", page_icon="📋", layout="wide")

from auth_gate import require_login
require_login()

from theme import inject_css

inject_css()

st.title("📋 Transactions")

from boot import load_app
from cafeteria.storage.database import get_session
from cafeteria.storage.repositories import TransactionRepository, PersonRepository
from cafeteria.storage.models import WasteStatus, TransactionStatus
from empty_states import empty_state_html

cfg = load_app()

# ── Filters ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filters")

session = get_session()
persons = PersonRepository(session).get_all_active()
session.close()

person_filter_options = ["All"] + [f"{p.person_id} — {p.name}" for p in persons]
person_filter = st.sidebar.selectbox("Person", person_filter_options, key="tx_person")
waste_filter  = st.sidebar.selectbox(
    "Waste Category", ["All", "EMPTY", "LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE", "UNKNOWN"],
    key="tx_waste"
)
status_filter = st.sidebar.selectbox(
    "Status",
    ["All", "AUTO_CONFIRMED", "MANUALLY_CONFIRMED", "REVIEW_REQUIRED", "REJECTED"],
    key="tx_status"
)

date_from = date_to = None
if st.sidebar.checkbox("Filter by date range", value=False, key="tx_use_dates"):
    date_from = st.sidebar.date_input("From Date", key="tx_from")
    date_to   = st.sidebar.date_input("To Date",   key="tx_to")
limit     = st.sidebar.slider("Max rows", 10, 500, 100, key="tx_limit")

# ── Query ─────────────────────────────────────────────────────────────────────
session = get_session()
tx_repo = TransactionRepository(session)
total_all = tx_repo.count_total()

pid_filter = None
if person_filter != "All":
    pid_filter = person_filter.split(" — ")[0]

start_ts = datetime.datetime.combine(date_from, datetime.time.min).timestamp() if date_from else None
end_ts   = datetime.datetime.combine(date_to, datetime.time.max).timestamp() if date_to else None

transactions = tx_repo.get_filtered(
    person_id=pid_filter if pid_filter else None,
    waste_status=None if waste_filter == "All" else waste_filter,
    status=None if status_filter == "All" else status_filter,
    start_ts=start_ts,
    end_ts=end_ts,
    limit=limit,
)
session.close()

# ── Table ─────────────────────────────────────────────────────────────────────
if total_all == 0:
    st.markdown(
        empty_state_html(
            icon="📋",
            title="No transactions yet",
            body=(
                "The live SQLite database has zero waste events. "
                "Nothing is sampled or invented. Transactions appear only after "
                "the plate + waste pipeline runs on a real camera event."
            ),
            steps=[
                "Train plate and waste models on the Training page.",
                "Start the engine from Home, then process a real tray.",
                "Until those models exist, the engine stays in face-recognition-only mode.",
            ],
        ),
        unsafe_allow_html=True,
    )
    st.stop()

if not transactions:
    st.markdown(
        empty_state_html(
            icon="🔎",
            title="No rows match these filters",
            body=(
                f"The database has <b>{total_all}</b> transaction(s), but none match "
                "the current sidebar filters. Clear a filter to see real rows."
            ),
        ),
        unsafe_allow_html=True,
    )
    st.stop()

rows = []
for tx in transactions:
    dt = datetime.datetime.fromtimestamp(tx.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    rows.append({
        "Timestamp":     dt,
        "Transaction ID": tx.transaction_id,
        "Person":        tx.person_name or tx.person_id or "UNKNOWN",
        "Waste":         tx.waste_status or "—",
        "Face Conf":     f"{tx.face_confidence:.3f}" if tx.face_confidence else "—",
        "Plate Conf":    f"{tx.plate_confidence:.3f}" if tx.plate_confidence else "—",
        "Waste Conf":    f"{tx.waste_confidence:.3f}" if tx.waste_confidence else "—",
        "Status":        tx.status,
        "Latency (ms)":  f"{tx.processing_latency_ms:.0f}" if tx.processing_latency_ms else "—",
        "Has Image":     "✅" if tx.event_image_path else "❌",
    })

df = pd.DataFrame(rows)
st.markdown(f"**{len(rows)} transaction(s)**")
st.dataframe(df, width="stretch", height=500)

# ── Image viewer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### View Transaction Image")
selected_tx_id = st.text_input("Transaction ID", placeholder="TX-XXXXXXXXXX", key="view_tx")
if selected_tx_id:
    sess = get_session()
    tx = TransactionRepository(sess).get_by_id(selected_tx_id)
    sess.close()
    if tx:
        col1, col2 = st.columns([2, 3])
        with col1:
            if tx.event_image_path and Path(tx.event_image_path).exists():
                st.image(tx.event_image_path, caption="Event Evidence", width="stretch")
            else:
                st.info("No image saved.")
        with col2:
            st.json({
                "transaction_id": tx.transaction_id,
                "timestamp": datetime.datetime.fromtimestamp(tx.timestamp).isoformat(),
                "person_id": tx.person_id,
                "person_name": tx.person_name,
                "waste_status": tx.waste_status,
                "plate_confidence": tx.plate_confidence,
                "waste_confidence": tx.waste_confidence,
                "face_confidence": tx.face_confidence,
                "status": tx.status,
                "processing_latency_ms": tx.processing_latency_ms,
            })
    else:
        st.warning(f"Transaction '{selected_tx_id}' not found.")

# ── CSV Export ────────────────────────────────────────────────────────────────
st.markdown("---")
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
st.download_button(
    "⬇️ Export CSV",
    data=csv_buffer.getvalue(),
    file_name="cafeteria_transactions.csv",
    mime="text/csv",
)
