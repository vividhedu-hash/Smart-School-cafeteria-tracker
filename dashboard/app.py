"""
Smart Cafeteria Waste Tracker — Streamlit Dashboard

Entry point. Run with:
    streamlit run dashboard/app.py

The dashboard talks to the engine over HTTP (`http://127.0.0.1:8765`)
and falls back to:
    data/runtime_state.json
    data/commands.json
    database/cafeteria.db
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Make the src/ package importable ────────────────────────────────────────
_dashboard_dir = Path(__file__).resolve().parent
_project_root  = _dashboard_dir.parent
_src_dir       = _project_root / "src"
for p in [str(_src_dir), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cafeteria.config.settings import load_settings
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Smart Cafeteria Waste Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth_gate import require_login
require_login()

# ── Initialize settings + DB + engine control ────────────────────────────────
import time

from engine_ctl import (
    engine_is_alive as _engine_is_alive,
    start_engine as _start_engine,
    stop_engine as _stop_engine,
)
from engine_client import fetch_state

@st.cache_resource
def _init():
    cfg = load_settings(config_path=_project_root / "configs" / "config.yaml")
    from cafeteria.storage.database import init_db
    init_db(cfg.project_root / cfg.storage.database)
    # Do NOT auto-start engine; user explicitly controls lifecycle from website UI
    return cfg

cfg = _init()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #38bdf8; }

.state-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-weight: 700; font-size: 0.9rem; letter-spacing: 0.05em;
}
.status-ok    { background: #064e3b; color: #34d399; border: 1px solid #059669; }
.status-warn  { background: #451a03; color: #fb923c; border: 1px solid #ea580c; }
.status-error { background: #450a0a; color: #f87171; border: 1px solid #dc2626; }

/* Step cards */
.step-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 16px;
    padding: 20px 24px; margin: 8px 0;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.step-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 24px rgba(59,130,246,0.15);
}
.step-card-num {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.step-card-title { font-size: 1.05rem; font-weight: 600; color: #e2e8f0; margin: 6px 0 4px; }
.step-card-desc  { font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

/* Nav cards */
.nav-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px;
    padding: 16px; text-align: center;
    transition: all 0.2s;
}
.nav-card:hover { border-color: #6366f1; box-shadow: 0 0 20px rgba(99,102,241,0.2); }
.nav-card-icon  { font-size: 2rem; margin-bottom: 8px; }
.nav-card-title { font-weight: 600; color: #e2e8f0; font-size: 0.9rem; }
.nav-card-desc  { color: #64748b; font-size: 0.78rem; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Cafeteria Tracker")
    st.markdown("---")

    engine_alive = _engine_is_alive()
    st.markdown("**Engine Controls**")
    if engine_alive:
        st.markdown('<span class="state-badge status-ok">● ENGINE RUNNING</span>', unsafe_allow_html=True)
        st.caption("Camera LED is on. Stop Engine (or leave Live Monitor) to release it.")
        if st.button("⏹ Stop Engine", type="primary", use_container_width=True):
            _stop_engine()
            st.success("Engine terminated.")
            time.sleep(0.5)
            st.rerun()
    else:
        st.markdown('<span class="state-badge status-error">● ENGINE STOPPED</span>', unsafe_allow_html=True)
        st.caption("Starts the camera. Leave Live Monitor and it turns off within a few seconds.")
        if st.button("▶ Start Engine", use_container_width=True):
            _start_engine()
            st.success("Engine starting…")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    if engine_alive:
        # [AI-CoLab: Cursor] AppSettings already defines api_host/api_port. getattr
        # is only a belt-and-suspenders guard if an older config object is loaded.
        _api_h = getattr(cfg.application, "api_host", None)
        _api_p = getattr(cfg.application, "api_port", None)
        state_data = fetch_state(
            cfg.project_root / cfg.application.runtime_state_path,
            host=_api_h,
            port=_api_p,
        )
        if state_data.get("via") == "api":
            st.caption(f"Engine API: `http://{_api_h}:{_api_p}`")
        else:
            st.caption("Engine API not reachable — using `runtime_state.json` fallback.")
    else:
        state_data = {}


    if state_data and engine_alive:
        engine_state = state_data.get("state", "—")
        fps          = state_data.get("fps", 0.0)
        components   = state_data.get("components", {})

        cam_ok = components.get("camera", "ERROR") == "OK"
        cam_badge = (
            '<span class="state-badge status-ok">● ONLINE</span>'
            if cam_ok else
            '<span class="state-badge status-error">● OFFLINE</span>'
        )
        st.markdown(f"**Camera:** {cam_badge}", unsafe_allow_html=True)
        st.markdown(f"**FPS:** `{fps}`")
        st.markdown(f"**State:** `{engine_state}`")
        st.markdown("---")

        st.markdown("**Components**")
        for comp, status in components.items():
            icon = "✅" if status == "OK" else ("⚠️" if "NOT" in status else "❌")
            st.markdown(f"{icon} `{comp}`: {status}")

        active = state_data.get("active_models", {})
        if active:
            st.markdown("---")
            st.markdown("**Active Models**")
            st.markdown(f"🔷 Plate: `{active.get('plate', 'none')}`")
            st.markdown(f"🔶 Waste: `{active.get('waste', 'none')}`")

        enrolled = state_data.get("enrolled_persons", 0)
        st.markdown(f"**Enrolled:** `{enrolled}` person(s)")

    st.markdown("---")
    st.caption("Camera is on only while Live Monitor is open.")

# ── Home page content ─────────────────────────────────────────────────────────
st.title("🍽️ Smart Cafeteria Waste Tracker")
st.markdown(
    "AI-powered cafeteria waste analysis — real-time plate detection, "
    "face recognition, and food waste classification."
)

# Engine status banner
if not engine_alive:
    st.info(
        "⏸️ **The inference engine is currently STOPPED.**\n\n"
        "Click **▶ Start Engine** then open **Live Monitor** to use the camera. "
        "The camera turns off when you leave Live Monitor so Training can use it.",
        icon="ℹ️",
    )
    if st.button("▶ Start Engine Now"):
        _start_engine()
        st.rerun()
else:
    st.success("✅ Engine online — camera is on. Open Live Monitor, or Stop Engine to free the webcam.", icon="✅")

# ── System Readiness (honest status — no fake AI) ─────────────────────────────
st.markdown("### 🩺 System Readiness")

def _readiness() -> list[tuple[str, str, str]]:
    """Return [(component, status, detail)] from real files/state, not hopes."""
    rows: list[tuple[str, str, str]] = []

    from cafeteria.training.registry import ModelRegistry
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")

    # Plate model
    plate_active = registry.get_active_weights("plate")
    plate_default = cfg.project_root / cfg.models.plate.weights
    plate_proxy = bool(state_data.get("plate_proxy")) if state_data else False
    if plate_active and Path(plate_active).exists():
        rows.append(("Plate model", "READY", f"active: {registry.active_version_string('plate')}"))
    elif plate_default.exists():
        rows.append(("Plate model", "READY", str(plate_default.name)))
    elif plate_proxy:
        rows.append(("Plate model", "PROXY", "COCO demo proxy — NOT a trained plate detector"))
    else:
        rows.append(("Plate model", "MISSING", "train via Training page — pipeline disabled until then"))

    # Waste model
    waste_active = registry.get_active_weights("waste")
    waste_default = cfg.project_root / cfg.models.waste.weights
    if waste_active and Path(waste_active).exists():
        rows.append(("Waste model", "READY", f"active: {registry.active_version_string('waste')}"))
    elif waste_default.exists():
        rows.append(("Waste model", "READY", str(waste_default.name)))
    else:
        rows.append(("Waste model", "MISSING", "upload images + train via Training page"))

    # Face engine (model pack on disk)
    face_pack_dir = cfg.project_root / "models" / "face" / "models" / cfg.recognition.model_pack
    if face_pack_dir.exists() and any(face_pack_dir.glob("*.onnx")):
        rows.append(("Face engine", "READY", f"InsightFace {cfg.recognition.model_pack}"))
    else:
        rows.append(("Face engine", "PENDING", "downloads automatically on first engine start"))

    # Enrolled persons
    enroll_root = cfg.project_root / cfg.recognition.embedding_dir
    n_enrolled = 0
    if enroll_root.exists():
        n_enrolled = sum(
            1 for d in enroll_root.iterdir()
            if d.is_dir() and (d / "embedding.npy").exists()
        )
    rows.append((
        "Enrolled people",
        "READY" if n_enrolled > 0 else "NONE",
        f"{n_enrolled} person(s) with embeddings",
    ))

    # Camera (only meaningful while engine runs)
    if state_data:
        cam_ok = state_data.get("camera_connected", False)
        rows.append(("Camera", "READY" if cam_ok else "ERROR",
                     "connected" if cam_ok else "not connected"))
    else:
        rows.append(("Camera", "UNKNOWN", "start the engine to check"))

    return rows

_status_style = {
    "READY":   "status-ok",
    "PROXY":   "status-warn",
    "PENDING": "status-warn",
    "NONE":    "status-warn",
    "UNKNOWN": "status-warn",
    "MISSING": "status-error",
    "ERROR":   "status-error",
}

try:
    state_data = fetch_state(
        cfg.project_root / cfg.application.runtime_state_path,
        host=getattr(cfg.application, "api_host", None),
        port=getattr(cfg.application, "api_port", None),
    ) if engine_alive else {}

    _rows = _readiness()
    _cols = st.columns(len(_rows))
    for _col, (_name, _stat, _detail) in zip(_cols, _rows):
        with _col:
            _cls = _status_style.get(_stat, "status-warn")
            st.markdown(
                f'<div style="text-align:center">'
                f'<div style="font-size:0.8rem;color:#94a3b8;margin-bottom:4px">{_name}</div>'
                f'<span class="state-badge {_cls}">{_stat}</span>'
                f'<div style="font-size:0.72rem;color:#64748b;margin-top:6px">{_detail}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _pipeline_ok = all(
        s in ("READY",) for n, s, _ in _rows if n in ("Plate model", "Waste model")
    )
    if not _pipeline_ok:
        st.warning(
            "⚠️ **Waste pipeline is DISABLED** — it requires a trained plate model "
            "AND a trained waste model. Until both exist, the engine runs "
            "face-recognition-only mode and will not create waste transactions. "
            "No results are ever fabricated.",
            icon="⚠️",
        )
    if any(s == "PROXY" for _, s, _ in _rows):
        st.warning(
            "🟠 **PLATE PROXY MODE** — using a generic COCO model as a stand-in. "
            "Detections are labelled `plate_proxy` and are not real plate detection. "
            "Train a plate model for honest results.",
            icon="🟠",
        )
except Exception as _e:
    st.caption(f"Readiness check unavailable: {_e}")

st.markdown("---")


# ── Getting Started steps ─────────────────────────────────────────────────────
st.markdown("### 🚀 Getting Started")

steps_html = """
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; margin:8px 0 24px;">

  <div class="step-card">
    <div class="step-card-num">01</div>
    <div class="step-card-title">Enroll People</div>
    <div class="step-card-desc">
      Go to <b>🧠 Training → Face Enrollment</b>.<br>
      The camera opens automatically in your browser.<br>
      Follow the on-screen pose guide (front, left, right…).
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">02</div>
    <div class="step-card-title">Upload Waste Images</div>
    <div class="step-card-desc">
      Go to <b>🧠 Training → Waste Dataset</b>.<br>
      Drag &amp; drop plate photos into the correct waste category.<br>
      Aim for 50+ images per category.
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">03</div>
    <div class="step-card-title">Train the Model</div>
    <div class="step-card-desc">
      Go to <b>🧠 Training → Train Model</b>.<br>
      Click <b>START TRAINING</b>. The system fine-tunes<br>
      a YOLOv8 classifier on your dataset.
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">04</div>
    <div class="step-card-title">Monitor Live</div>
    <div class="step-card-desc">
      Go to <b>📹 Live Monitor</b>.<br>
      See the real-time annotated camera feed,<br>
      pipeline state, and live waste events.
    </div>
  </div>

</div>
"""
st.markdown(steps_html, unsafe_allow_html=True)

# ── Navigation grid ───────────────────────────────────────────────────────────
st.markdown("### 🗂️ Pages")
nav_html = """
<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:8px 0 24px;">
  <div class="nav-card"><div class="nav-card-icon">📹</div>
    <div class="nav-card-title">Live Monitor</div>
    <div class="nav-card-desc">Real-time feed &amp; state</div></div>
  <div class="nav-card"><div class="nav-card-icon">🧠</div>
    <div class="nav-card-title">Training</div>
    <div class="nav-card-desc">Enroll &amp; train models</div></div>
  <div class="nav-card"><div class="nav-card-icon">🔍</div>
    <div class="nav-card-title">Review Queue</div>
    <div class="nav-card-desc">Confirm unknown faces</div></div>
  <div class="nav-card"><div class="nav-card-icon">📋</div>
    <div class="nav-card-title">Transactions</div>
    <div class="nav-card-desc">Browse all events</div></div>
  <div class="nav-card"><div class="nav-card-icon">📊</div>
    <div class="nav-card-title">Analytics</div>
    <div class="nav-card-desc">Charts &amp; insights</div></div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ── Summary stats ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

try:
    from cafeteria.storage.database import get_session
    from cafeteria.storage.repositories import TransactionRepository, ReviewRepository
    session = get_session()
    tx_repo = TransactionRepository(session)
    rev_repo = ReviewRepository(session)

    with col1:
        st.metric("Total Transactions", tx_repo.count_total())
    with col2:
        st.metric("Review Queue", rev_repo.count_unresolved())
    with col3:
        n_waste = (
            tx_repo.count_by_waste("HIGH_WASTE")
            + tx_repo.count_by_waste("MEDIUM_WASTE")
            + tx_repo.count_by_waste("LOW_WASTE")
        )
        st.metric("Waste Events", n_waste)
    with col4:
        st.metric("Empty Plates", tx_repo.count_by_waste("EMPTY"))
    session.close()
except Exception as e:
    st.warning(f"Could not read database metrics: {e}")
