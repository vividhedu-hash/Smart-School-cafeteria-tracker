"""
Page 1 — Live Monitor

Real-time camera feed + always-on face recognition panel.
Polls runtime_state.json every 300 ms via st.rerun().
"""
from __future__ import annotations

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

from cafeteria.config.settings import load_settings
from cafeteria.recognition.live_match import face_match_as_dict
from engine_ctl import write_heartbeat, engine_is_alive, start_engine
from engine_client import fetch_state, fetch_frame_bytes

cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")

# Heartbeat — camera stays on ONLY while this page is polling
write_heartbeat()

# ── Poll interval ─────────────────────────────────────────────────────────────
POLL_MS = 400  # ~2.5 Hz so the face lock feels live without freezing Streamlit


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.id-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 28px 24px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.id-card.known {
    border-color: #22c55e;
    box-shadow: 0 0 40px rgba(34,197,94,0.18);
}
.id-card.scanning {
    border-color: #3b82f6;
    box-shadow: 0 0 30px rgba(59,130,246,0.12);
}
.id-card.unknown {
    border-color: #f59e0b;
    box-shadow: 0 0 30px rgba(245,158,11,0.12);
}
.id-card.locking {
    border-color: #38bdf8;
    box-shadow: 0 0 30px rgba(56,189,248,0.16);
}
.badge-locking  { background: #0c4a6e; color: #7dd3fc; border: 1px solid #38bdf8; }
.id-name {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 12px 0 4px;
    letter-spacing: -0.02em;
}
.id-sim {
    font-size: 1rem;
    font-weight: 600;
    color: #22c55e;
    margin-bottom: 8px;
}
.id-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-known    { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.badge-scanning { background: #1e3a5f; color: #60a5fa; border: 1px solid #3b82f6; }
.badge-unknown  { background: #451a03; color: #fb923c; border: 1px solid #ea580c; }

.scan-pulse {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #3b82f6;
    margin-right: 6px;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}

.state-bar {
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 700;
    font-size: 1.1rem;
    text-align: center;
    color: #fff;
    margin-bottom: 12px;
}
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.metric-box {
    flex: 1; min-width: 100px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 10px 14px;
}
.metric-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.07em; }
.metric-value { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

st.title("📹 Live Monitor")
st.caption(
    "Walk into or across the camera with your face visible. A lock draws immediately; "
    "identity is scanned when the face is large enough. Profile/side views often fail — "
    "that is reported as unknown, not a guessed ID. Leave this page and the camera turns off."
)

# ── Read live state ───────────────────────────────────────────────────────────
engine_alive = engine_is_alive()

if not engine_alive:
    st.info(
        "⏸️ **The inference engine is currently STOPPED.**\n\n"
        "The camera is off. Click **▶ Start Engine** for a live feed and face match. "
        "Leave this page when you’re done — the camera turns off by itself."
    )
    if st.button("▶ Start Engine Now"):
        start_engine()
        st.rerun()
    st.stop()

state_data = fetch_state(
    cfg.project_root / cfg.application.runtime_state_path,
    host=getattr(cfg.application, "api_host", None),
    port=getattr(cfg.application, "api_port", None),
)


if not state_data:
    st.info("⏳ **Engine is starting** — waiting for the first camera frame and runtime state.")
    time.sleep(0.8)
    st.rerun()


# ── Layout: feed (left) | recognition (right) ────────────────────────────────
left, right = st.columns([3, 2], gap="large")

# ── LEFT: camera feed ─────────────────────────────────────────────────────────
with left:
    st.subheader("🎥 Camera Feed")
    latest_frame = cfg.project_root / cfg.storage.frames / "latest.jpg"
    frame_bytes = fetch_frame_bytes(
        latest_frame,
        host=getattr(cfg.application, "api_host", None),
        port=getattr(cfg.application, "api_port", None),
    )

    if frame_bytes:
        try:
            import io
            img = Image.open(io.BytesIO(frame_bytes)).copy()
            st.image(img, width='stretch')
        except Exception:
            st.info("Loading frame…")
    else:
        st.info("⏳ Waiting for first camera frame…")

    # Pipeline state bar
    state     = state_data.get("state", "IDLE")
    fps       = state_data.get("fps", 0.0)
    latency   = state_data.get("latency", {})

    state_colours = {
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
    colour = state_colours.get(state, "#334155")
    st.markdown(
        f'<div class="state-bar" style="background:{colour}">STATE: {state}</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("FPS", f"{fps:.1f}")
    with c2:
        st.metric("Face Latency", f"{latency.get('face_ms', 0):.0f} ms")
    with c3:
        st.metric("E2E Latency", f"{latency.get('e2e_ms', 0):.0f} ms")

# ── RIGHT: always-on face recognition panel ───────────────────────────────────
with right:
    st.subheader("🔍 Who's Here?")

    live = face_match_as_dict(state_data.get("live_face_match"))
    # #region agent log
    try:
        import json as _json
        _live_key = (
            live is None,
            bool(live and live.get("approaching")),
            bool(live and live.get("is_known")),
            bool(engine_is_alive()),
        )
        if st.session_state.get("_dbg_live_key") != _live_key:
            st.session_state["_dbg_live_key"] = _live_key
            with open("/Users/indian/Downloads/Adaptive signal project/India Lens /school project/smart-cafeteria-waste/.cursor/debug-e78165.log", "a") as _f:
                _f.write(_json.dumps({"sessionId":"e78165","timestamp":int(time.time()*1000),"location":"1_Live_Monitor.py","message":"who-panel","data":{"none": live is None,"approaching": bool(live and live.get("approaching")),"is_known": bool(live and live.get("is_known")),"sim": float((live or {}).get("similarity") or 0),"engine": bool(engine_is_alive())},"hypothesisId":"H4","runId":"pre-fix"})+"\n")
    except Exception:
        pass
    # #endregion

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

    elif live.get("is_known"):
        person_name = live.get("person_name") or live.get("person_id") or "Unknown"
        similarity  = live.get("similarity", 0.0)
        person_id   = live.get("person_id", "")
        sim_pct     = int(similarity * 100)

        # Try to load enrollment photo (images/ subdir or person dir)
        enrollment_dir = cfg.project_root / cfg.recognition.embedding_dir / person_id
        photo_html = ""
        photo_roots = [enrollment_dir / "images", enrollment_dir]
        photos = []
        for root in photo_roots:
            if not root.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                photos.extend(root.glob(ext))
            if photos:
                break
        photos = sorted(photos)
        if photos:
            try:
                thumb = Image.open(photos[0])
                thumb.thumbnail((120, 120))
                import io, base64
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                photo_html = (
                    f'<img src="data:image/jpeg;base64,{b64}" '
                    'style="width:96px;height:96px;object-fit:cover;'
                    'border-radius:50%;border:3px solid #22c55e;margin-bottom:10px;">'
                )
            except Exception:
                pass

        if not photo_html:
            photo_html = '<div style="width:96px;height:96px;border-radius:50%;background:#1e293b;border:3px solid #22c55e;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;font-size:2.5rem;">👤</div>'

        st.markdown(f"""
        <div class="id-card known">
            {photo_html}
            <div class="id-name">{person_name}</div>
            <div class="id-sim">Match: {sim_pct}% confidence</div>
            <span class="id-badge badge-known">✓ IDENTIFIED</span>
        </div>
        """, unsafe_allow_html=True)

        # Pull any DB data about the person
        st.markdown("")
        try:
            from cafeteria.storage.database import get_session
            from cafeteria.storage.repositories import TransactionRepository
            session = get_session()
            tx_repo = TransactionRepository(session)
            total_txns = tx_repo.count_by_person(person_id) if hasattr(tx_repo, "count_by_person") else None
            session.close()
            if total_txns is not None:
                st.metric("Total visits (this person)", total_txns)
        except Exception:
            pass

    elif live.get("approaching"):
        st.markdown("""
        <div class="id-card locking">
            <div style="font-size:3rem">🎯</div>
            <div style="margin:14px 0 8px">
                <span class="scan-pulse"></span>
                <span style="color:#7dd3fc;font-weight:600;">Face found — walk closer to scan</span>
            </div>
            <span class="id-badge badge-locking">LOCKING</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        similarity = live.get("similarity", 0.0)
        sim_pct    = int(similarity * 100)
        st.markdown(f"""
        <div class="id-card unknown">
            <div style="font-size:3rem">❓</div>
            <div class="id-name" style="color:#fb923c;">Unknown</div>
            <div style="color:#94a3b8;font-size:0.9rem;margin:6px 0 10px;">
                Best match: {sim_pct}% — below threshold
            </div>
            <span class="id-badge badge-unknown">NOT ENROLLED</span>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 Go to **Training** to enroll this person.")

    # Component status
    st.markdown("---")
    st.markdown("**Components**")
    comps = state_data.get("components", {})
    for comp, status in comps.items():
        icon = "✅" if status == "OK" else ("⚠️" if "NOT" in status else "❌")
        st.markdown(f"{icon} `{comp}` — {status}")

    # Last event
    last_evt = state_data.get("last_event") or {}
    if last_evt:
        st.markdown("---")
        st.markdown("**Last Transaction**")
        st.markdown(f"- Person: `{last_evt.get('person_id') or 'UNKNOWN'}`")
        st.markdown(f"- Waste: `{last_evt.get('waste_status', '—')}`")
        st.markdown(f"- Status: `{last_evt.get('status', '—')}`")
        st.markdown(f"- Latency: `{last_evt.get('latency_ms', 0):.0f} ms`")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
time.sleep(POLL_MS / 1000)
st.rerun()
