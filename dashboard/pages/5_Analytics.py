"""
Page 5 — Analytics

Charts and summary statistics derived from real transaction data.
No fake data — empty databases render a finished empty state, not a crash.
"""
from __future__ import annotations

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

from auth_gate import require_login
require_login()

st.title("📊 Analytics")

from cafeteria.config.settings import load_settings
from cafeteria.monitoring.analytics import (
    PIPELINE_GATED_COPY,
    events_over_time_rows,
    latency_values,
    person_waste_rows,
    status_counts as status_count_map,
    waste_pie_payload,
)
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import TransactionRepository, ReviewRepository
from empty_states import empty_state_html

cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
init_db(cfg.project_root / cfg.storage.database)

session = get_session()
tx_repo  = TransactionRepository(session)
rev_repo = ReviewRepository(session)

total = tx_repo.count_total()
unresolved = rev_repo.count_unresolved()
waste_counts = {
    "EMPTY":        tx_repo.count_by_waste("EMPTY"),
    "LOW_WASTE":    tx_repo.count_by_waste("LOW_WASTE"),
    "MEDIUM_WASTE": tx_repo.count_by_waste("MEDIUM_WASTE"),
    "HIGH_WASTE":   tx_repo.count_by_waste("HIGH_WASTE"),
}
all_tx = tx_repo.get_filtered(limit=1000)
session.close()

lat_vals = latency_values(all_tx)
avg_latency = (sum(lat_vals) / len(lat_vals)) if lat_vals else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Events",    total)
c2.metric("Waste Events",    sum(waste_counts[k] for k in ["LOW_WASTE", "MEDIUM_WASTE", "HIGH_WASTE"]))
c3.metric("Empty Plates",    waste_counts["EMPTY"])
c4.metric("Review Queue",    unresolved)
c5.metric("Avg Latency",     f"{avg_latency:.0f} ms")

if total == 0:
    st.markdown(
        empty_state_html(
            icon="📊",
            title="No analytics yet — this is real, not broken",
            body=PIPELINE_GATED_COPY,
            steps=[
                "Upload waste images and label plate photos on the Training page.",
                "Train and activate both models (or run <code>scripts/quickstart.py</code>).",
                "Charts appear from live SQLite rows after the first real event.",
            ],
        ),
        unsafe_allow_html=True,
    )
else:
    st.markdown("---")
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Waste Distribution")
        labels, values = waste_pie_payload(waste_counts)
        if labels:
            fig = px.pie(
                names=labels, values=values,
                color_discrete_map={
                    "EMPTY": "#94a3b8",
                    "LOW_WASTE": "#fcd34d",
                    "MEDIUM_WASTE": "#f97316",
                    "HIGH_WASTE": "#ef4444",
                },
                hole=0.4,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No waste-class counts yet.")

    with col_bar:
        st.subheader("Events Over Time")
        time_rows = events_over_time_rows(all_tx)
        if time_rows:
            df = pd.DataFrame(time_rows)
            daily = df.groupby(["date", "waste"]).size().reset_index(name="count")
            fig2 = px.bar(
                daily, x="date", y="count", color="waste",
                color_discrete_map={
                    "EMPTY": "#94a3b8", "LOW_WASTE": "#fcd34d",
                    "MEDIUM_WASTE": "#f97316", "HIGH_WASTE": "#ef4444",
                    "UNKNOWN": "#6b7280",
                },
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No dated events yet.")

    st.subheader("Waste Events by Person")
    person_rows = person_waste_rows(all_tx)
    if person_rows:
        df_person = pd.DataFrame(person_rows)
        fig3 = px.histogram(
            df_person, x="person", color="waste", barmode="stack",
            color_discrete_map={
                "LOW_WASTE": "#fcd34d",
                "MEDIUM_WASTE": "#f97316",
                "HIGH_WASTE": "#ef4444",
            },
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No waste events with identities yet.")

    st.markdown("---")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.subheader("Status Distribution")
        sc = status_count_map(all_tx)
        if sc:
            fig4 = px.bar(
                x=list(sc.keys()),
                y=list(sc.values()),
                color=list(sc.keys()),
                labels={"x": "Status", "y": "Count"},
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                showlegend=False,
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No status data yet.")

    with col_stat2:
        st.subheader("Processing Latency")
        if lat_vals:
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(
                x=lat_vals,
                nbinsx=30,
                marker_color="#38bdf8",
                name="Latency (ms)",
            ))
            fig5.update_layout(
                xaxis_title="Latency (ms)",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No latency data yet.")
