"""
Page 1 — Live Monitor

Real-time camera feed + always-on face recognition panel.

The whole view lives inside a single ``@st.fragment`` that re-runs on a timer.
The page script itself is not re-executed, so the feed no longer flickers or
blocks the sidebar the way a full-page ``st.rerun()`` loop did.
"""
from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from PIL import Image, ImageFile

# Allow loading partially-written JPEG files from the engine
ImageFile.LOAD_TRUNCATED_IMAGES = True


st.set_page_config(page_title="Live Monitor", page_icon="📹", layout="wide")

from auth_gate import require_login
require_login()

from boot import load_app
from cafeteria.recognition.live_match import face_match_as_dict
from engine_ctl import write_heartbeat, engine_is_alive, is_cloud, start_engine, stop_engine
from engine_client import fetch_state, fetch_frame_bytes
from theme import inject_css, status_strip

try:
    cfg = load_app()
except Exception as _boot_err:
    cfg = None

# Heartbeat immediately: the engine only keeps the camera while a page asks.
write_heartbeat()

# ── Refresh interval ─────────────────────────────────────────────────────────
# The engine writes state every 100 ms. Fragment reruns are cheap but not
# free — 350 ms keeps the feed fluid without saturating the browser socket.
REFRESH_SECONDS = 0.35
BOOT_PATIENCE_SECONDS = 75.0

inject_css()

st.markdown("""
<style>
.id-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155; border-radius: 20px;
    padding: 26px 22px; text-align: center; position: relative; overflow: hidden;
}
.id-card.known    { border-color: #22c55e; box-shadow: 0 0 40px rgba(34,197,94,0.18); }
.id-card.scanning { border-color: #3b82f6; box-shadow: 0 0 30px rgba(59,130,246,0.12); }
.id-card.unknown  { border-color: #f59e0b; box-shadow: 0 0 30px rgba(245,158,11,0.12); }
.id-card.locking  { border-color: #38bdf8; box-shadow: 0 0 30px rgba(56,189,248,0.16); }
.id-name { font-size: 2rem; font-weight: 800; color: #f1f5f9; margin: 12px 0 4px; letter-spacing: -0.02em; }
.id-sim  { font-size: 1rem; font-weight: 600; color: #22c55e; margin-bottom: 8px; }
.id-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-known    { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-scanning { background: #1e3a5f; color: #60a5fa; border: 1px solid #3b82f6; }
.badge-unknown  { background: #451a03; color: #fb923c; border: 1px solid #ea580c; }
.badge-locking  { background: #0c4a6e; color: #7dd3fc; border: 1px solid #38bdf8; }
.state-bar {
    border-radius: 10px; padding: 8px 18px; font-weight: 700;
    font-size: 1.05rem; text-align: center; color: #fff; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📹 Live Monitor")
st.caption(
    "Walk towards the camera with your face visible. A lock draws as soon as a face "
    "appears; the identity check runs once the face is close enough to be reliable. "
    "Side profiles report as unknown rather than guessing."
)

STATE_COLOURS = {
    "IDLE":               "#334155",
    "PLATE_DETECTED":     "#16a34a",
    "FOOD_ANALYSIS":      "#0891b2",
    "WASTE_EVENT":        "#d97706",
    "FACE_CAPTURE":       "#7c3aed",
    "FACE_RECOGNITION":   "#db2777",
    "TRANSACTION_COMMIT": "#059669",
    "REVIEW_REQUIRED":    "#dc2626",
    "COOLDOWN":           "#6b7280",
    "ERROR":              "#ef4444",
}


def _api_kwargs() -> dict:
    if cfg is None:
        return {"host": None, "port": None}
    return {
        "host": getattr(cfg.application, "api_host", None),
        "port": getattr(cfg.application, "api_port", None),
    }


def _enrollment_thumb(person_id: str) -> str:
    """Circular avatar for a matched person, or a neutral placeholder."""
    if cfg is None:
        return (
            '<div style="width:96px;height:96px;border-radius:50%;background:#1e293b;'
            'border:3px solid #22c55e;margin:0 auto 10px;display:flex;align-items:center;'
            'justify-content:center;font-size:2.5rem;">👤</div>'
        )
    enrollment_dir = cfg.project_root / cfg.recognition.embedding_dir / person_id
    photos: list[Path] = []
    for root in (enrollment_dir / "images", enrollment_dir):
        if not root.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            photos.extend(root.glob(ext))
        if photos:
            break
    for photo in sorted(photos):
        try:
            thumb = Image.open(photo)
            thumb.thumbnail((120, 120))
            buf = io.BytesIO()
            thumb.convert("RGB").save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return (
                f'<img src="data:image/jpeg;base64,{b64}" '
                'style="width:96px;height:96px;object-fit:cover;border-radius:50%;'
                'border:3px solid #22c55e;margin-bottom:10px;">'
            )
        except Exception:
            continue
    return (
        '<div style="width:96px;height:96px;border-radius:50%;background:#1e293b;'
        'border:3px solid #22c55e;margin:0 auto 10px;display:flex;align-items:center;'
        'justify-content:center;font-size:2.5rem;">👤</div>'
    )


def _render_identity(state_data: dict) -> None:
    live = face_match_as_dict(state_data.get("live_face_match"))

    if live is None:
        st.markdown("""
        <div class="id-card scanning">
            <div style="font-size:3rem">👁️</div>
            <div style="margin:14px 0 8px">
                <span class="scan-pulse"></span>
                <span style="color:#60a5fa;font-weight:600;">Waiting for someone to walk in…</span>
            </div>
            <span class="id-badge badge-scanning">WAITING</span>
        </div>
        """, unsafe_allow_html=True)
        return

    if live.get("is_known"):
        person_id = live.get("person_id", "")
        person_name = live.get("person_name") or person_id or "Unknown"
        sim_pct = int(float(live.get("similarity") or 0.0) * 100)
        st.markdown(f"""
        <div class="id-card known">
            {_enrollment_thumb(person_id)}
            <div class="id-name">{person_name}</div>
            <div class="id-sim">Match: {sim_pct}% confidence</div>
            <span class="id-badge badge-known">✓ IDENTIFIED</span>
        </div>
        """, unsafe_allow_html=True)
        try:
            from cafeteria.storage.database import get_session
            from cafeteria.storage.repositories import TransactionRepository
            session = get_session()
            repo = TransactionRepository(session)
            visits = repo.count_by_person(person_id) if hasattr(repo, "count_by_person") else None
            session.close()
            if visits is not None:
                st.metric("Total visits (this person)", visits)
        except Exception:
            pass
        return

    if live.get("approaching"):
        tracking = bool(live.get("moving"))
        hint = (
            "Tracking walk-past — identity holds while they move"
            if tracking
            else "Face found — walk closer to scan"
        )
        badge = "TRACKING" if tracking else "LOCKING"
        st.markdown(f"""
        <div class="id-card locking">
            <div style="font-size:3rem">🎯</div>
            <div style="margin:14px 0 8px">
                <span class="scan-pulse"></span>
                <span style="color:#7dd3fc;font-weight:600;">{hint}</span>
            </div>
            <span class="id-badge badge-locking">{badge}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    sim_pct = int(float(live.get("similarity") or 0.0) * 100)
    thresh = cfg.recognition.similarity_threshold if cfg else 0.52
    st.markdown(f"""
    <div class="id-card unknown">
        <div style="font-size:3rem">❓</div>
        <div class="id-name" style="color:#fb923c;">Unknown</div>
        <div style="color:#94a3b8;font-size:0.9rem;margin:6px 0 10px;">
            Best match: {sim_pct}% — below the {thresh:.2f} threshold
        </div>
        <span class="id-badge badge-unknown">NOT ENROLLED</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Enroll this person on the Training page to name them here.")


def _render_boot_card(alive_seconds: float) -> None:
    """Explain the wait instead of showing a blank screen while models load."""
    pct = min(0.95, max(0.05, alive_seconds / BOOT_PATIENCE_SECONDS))
    st.markdown(
        '<div class="glass-card">'
        '<div class="card-title">Engine is starting…</div>'
        '<div class="card-body">Opening the camera and loading the InsightFace model pack. '
        'The first start after a reboot is the slowest — later starts reuse the cached '
        'models.</div></div>',
        unsafe_allow_html=True,
    )
    st.progress(pct, text="Loading camera and face models")
    if alive_seconds > BOOT_PATIENCE_SECONDS:
        st.warning(
            "The engine has not reported state yet. Check `logs/engine_stdio.log`, "
            "then stop and start it again.",
            icon="⚠️",
        )
        if st.button("⏹ Stop engine", key="boot_stop"):
            stop_engine()
            st.rerun()


@st.fragment(run_every=REFRESH_SECONDS)
def live_view() -> None:
    # [AI-CoLab: Verified by Antigravity] Fragment-based self-refreshing live view preventing full page rerenders
    write_heartbeat()

    if not engine_is_alive():
        return

    st.session_state.setdefault("engine_started_at", time.time())
    state_data = fetch_state(
        cfg.project_root / cfg.application.runtime_state_path, **_api_kwargs()
    )
    status_strip(state_data, engine_alive=True)

    if not state_data:
        _render_boot_card(time.time() - st.session_state["engine_started_at"])
        return

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("🎥 Camera feed")
        frame_bytes = fetch_frame_bytes(
            cfg.project_root / cfg.storage.frames / "latest.jpg", **_api_kwargs()
        )
        if frame_bytes:
            try:
                st.image(Image.open(io.BytesIO(frame_bytes)).copy(), width="stretch")
            except Exception:
                st.info("Decoding the latest frame…")
        else:
            st.info("⏳ Waiting for the first camera frame…")

        state = state_data.get("state", "IDLE")
        st.markdown(
            f'<div class="state-bar" style="background:'
            f'{STATE_COLOURS.get(state, "#334155")}">STATE: {state}</div>',
            unsafe_allow_html=True,
        )

        latency = state_data.get("latency", {}) or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("FPS", f"{float(state_data.get('fps') or 0.0):.1f}")
        c2.metric("Face latency", f"{float(latency.get('face_ms') or 0):.0f} ms")
        c3.metric("End-to-end", f"{float(latency.get('e2e_ms') or 0):.0f} ms")

    with right:
        st.subheader("🔍 Who's here?")
        _render_identity(state_data)

        st.markdown("---")
        st.markdown("**Components**")
        for comp, status in (state_data.get("components", {}) or {}).items():
            icon = "✅" if status == "OK" else ("⚠️" if "NOT" in str(status) else "❌")
            st.markdown(f"{icon} `{comp}` — {status}")

        last_evt = state_data.get("last_event") or {}
        if last_evt:
            st.markdown("---")
            st.markdown("**Last transaction**")
            st.markdown(f"- Person: `{last_evt.get('person_id') or 'UNKNOWN'}`")
            st.markdown(f"- Waste: `{last_evt.get('waste_status', '—')}`")
            st.markdown(f"- Status: `{last_evt.get('status', '—')}`")
            st.markdown(f"- Latency: `{float(last_evt.get('latency_ms') or 0):.0f} ms`")


with st.sidebar:
    st.markdown("### Camera")
    if engine_is_alive():
        st.markdown(
            '<span class="state-badge status-ok">● ENGINE RUNNING</span>',
            unsafe_allow_html=True,
        )
        if st.button("⏹ Stop engine", key="sidebar_stop", width="stretch"):
            stop_engine()
            st.rerun()
    else:
        st.markdown(
            '<span class="state-badge status-error">● ENGINE STOPPED</span>',
            unsafe_allow_html=True,
        )
        if st.button("▶ Start engine", key="sidebar_start", width="stretch"):
            start_engine()
            st.session_state["engine_started_at"] = time.time()
            st.rerun()
    st.caption(
        "The camera stays on while this page or the enrollment scanner is open, "
        "and switches off about a minute after you leave."
    )

# ── Main view: Live Stream or Stopped Prompt ─────────────────────────────────
if engine_is_alive():
    live_view()
else:
    st.session_state.pop("engine_started_at", None)
    status_strip({}, engine_alive=False)
    st.info(
        "**The inference engine is currently STOPPED.**\n\n"
        "Click **▶ Start Engine** below to start real-time cafeteria detection, "
        "live camera video streaming, and face matching in 1 click.",
        icon="⏸️",
    )
    col_start, col_help = st.columns([1, 2])
    with col_start:
        if st.button("▶ Start Engine Now", type="primary", key="live_start_btn", use_container_width=True):
            start_engine()
            st.session_state["engine_started_at"] = time.time()
            st.rerun()
    with col_help:
        st.caption(
            "Launches the CV pipeline. Supports physical USB/built-in webcams on workstation, "
            "and automatically provisions the Virtual Cafeteria Stream on cloud containers."
        )
