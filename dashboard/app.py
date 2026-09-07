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
    is_cloud as _is_cloud,
    start_engine as _start_engine,
    stop_engine as _stop_engine,
)
from engine_client import fetch_state

@st.cache_resource
def _init():
    from boot import load_app
    # Do NOT auto-start engine; user explicitly controls lifecycle from website UI
    return load_app()

cfg = None
try:
    cfg = _init()
except Exception:
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    from boot import load_app
    try:
        cfg = load_app()
    except Exception:
        cfg = None

# ── Shared visual language ───────────────────────────────────────────────────
from theme import inject_css, status_strip

inject_css()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍽️ Cafeteria Tracker")
    st.markdown("---")

    cloud_mode = _is_cloud()
    engine_alive = _engine_is_alive()
    st.markdown("**Engine Controls**")

    if cloud_mode:
        st.markdown('<span class="state-badge status-warn">☁️ CLOUD PREVIEW</span>', unsafe_allow_html=True)
        st.caption(
            "Dashboard is hosted in cloud mode. Real-time video detection and camera AI run on "
            "your local machine."
        )
        with st.expander("💻 Local Camera Setup", expanded=False):
            st.code("bash run.sh", language="bash")
            st.caption("Runs the local real-time inference engine and syncs with this Supabase database.")
    elif engine_alive:
        st.markdown('<span class="state-badge status-ok">● ENGINE RUNNING</span>', unsafe_allow_html=True)
        st.caption("Camera LED is on. Stop the engine to release it immediately.")
        if st.button("⏹ Stop Engine", type="primary", width="stretch"):
            _stop_engine()
            st.success("Engine terminated.")
            time.sleep(0.5)
            st.rerun()
    else:
        st.markdown('<span class="state-badge status-error">● ENGINE STOPPED</span>', unsafe_allow_html=True)
        st.caption(
            "Starts the camera. It releases itself about a minute after you leave "
            "Live Monitor and the enrollment scanner."
        )
        if st.button("▶ Start Engine", width="stretch"):
            _start_engine()
            st.success("Engine starting…")
            time.sleep(1)
            st.rerun()

    st.markdown("---")

    if engine_alive and cfg is not None:
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
    st.caption(
        "The camera runs while Live Monitor or the enrollment scanner is open. "
        "Both share the same feed — starting one no longer stops the other."
    )

# ── Home page content ─────────────────────────────────────────────────────────
st.title("🍽️ Smart Cafeteria Waste Tracker")
st.markdown(
    "AI-powered cafeteria waste analysis — real-time plate detection, "
    "face recognition, and food waste classification."
)

def _enrolled_count() -> int:
    # [AI-CoLab: Verified by Antigravity] Robust count of enrolled face profiles for top status strip
    if cfg is None:
        return 0
    try:
        from cafeteria.recognition.enrollment import EnrollmentManager
        mgr = EnrollmentManager(
            enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
        )
        return sum(1 for pid in mgr.list_enrolled() if mgr.has_embedding(pid))
    except Exception:
        return 0


status_strip(state_data, engine_alive=engine_alive, enrolled=_enrolled_count())

# Engine status banner
if cloud_mode:
    st.info(
        "☁️ **Cloud Management Dashboard Active**\n\n"
        "Your database, analytics, review queue, and model roster are fully operational. "
        "Real-time video inference and webcam tracking run on your physical machine.\n\n"
        "To start the live camera engine locally: run `bash run.sh` or double-click `Start_Cafeteria_Tracker.command`.",
        icon="☁️",
    )
elif not engine_alive:
    st.info(
        "**The inference engine is currently STOPPED.**\n\n"
        "Click **▶ Start Engine**, then open **Live Monitor** for the live feed. "
        "Enrollment and dataset capture borrow the same camera, so you can leave "
        "the engine running while you work.",
        icon="ℹ️",
    )
    if st.button("▶ Start Engine Now"):
        _start_engine()
        st.rerun()
else:
    st.success(
        "✅ Engine online — camera is on. Live Monitor, face enrollment, and dataset "
        "capture all share this feed.",
        icon="✅",
    )

# ── System Readiness (honest status — no fake AI) ─────────────────────────────
st.markdown("### 🩺 System Readiness")

def _readiness() -> list[tuple[str, str, str]]:
    """Return [(component, status, detail)] from real files/state, not hopes."""
    if cfg is None:
        return [("Engine", "UNAVAILABLE", "cloud preview mode — local deployment required")]
    rows: list[tuple[str, str, str]] = []

    from cafeteria.training.registry import ModelRegistry
    registry = ModelRegistry(cfg.project_root / "models" / "registry.json")

    plate_active = registry.get_active_weights("plate")
    plate_default = cfg.project_root / cfg.models.plate.weights
    plate_proxy = bool(state_data.get("plate_proxy")) if state_data else False
    plate_backend = str((state_data or {}).get("plate_backend") or "")
    waste_backend = str((state_data or {}).get("waste_backend") or "")

    if plate_active and Path(plate_active).exists():
        rows.append(("Plate model", "READY", f"YOLO {registry.active_version_string('plate')}"))
    elif plate_default.exists():
        rows.append(("Plate model", "READY", f"YOLO {plate_default.name}"))
    elif plate_proxy or plate_backend == "coco_proxy":
        rows.append(("Plate model", "PROXY", "COCO demo proxy — not a trained plate detector"))
    else:
        rows.append(("Plate model", "READY", "OpenCV visual — train YOLO later for higher accuracy"))

    waste_active = registry.get_active_weights("waste")
    waste_default = cfg.project_root / cfg.models.waste.weights
    if waste_active and Path(waste_active).exists():
        rows.append(("Waste model", "READY", f"YOLO {registry.active_version_string('waste')}"))
    elif waste_default.exists():
        rows.append(("Waste model", "READY", f"YOLO {waste_default.name}"))
    else:
        detail = "OpenCV visual — train YOLO later for higher accuracy"
        if waste_backend == "yolo":
            detail = "YOLO (engine)"
        rows.append(("Waste model", "READY", detail))

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

    # Database
    from cafeteria.storage.database import is_postgres
    if is_postgres():
        rows.append(("Database", "READY", "Supabase PostgreSQL connected"))
    else:
        rows.append(("Database", "READY", "SQLite (local / cloud preview)"))

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
    if cfg is not None:
        state_data = fetch_state(
            cfg.project_root / cfg.application.runtime_state_path,
            host=getattr(cfg.application, "api_host", None),
            port=getattr(cfg.application, "api_port", None),
        ) if engine_alive else {}
    else:
        state_data = {}

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

    _yolo_ok = all(
        "YOLO" in d for n, s, d in _rows if n in ("Plate model", "Waste model")
    )
    if not _yolo_ok and all(s == "READY" for n, s, _ in _rows if n in ("Plate model", "Waste model")):
        st.info(
            "The waste pipeline is **ON** using built-in OpenCV visual analysis "
            "(real computer vision on the camera frame — not mocked labels). "
            "Upload cafeteria photos on the Training page and train YOLO when "
            "you want higher accuracy.",
            icon="ℹ️",
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
    <div class="step-card-title">Start the engine</div>
    <div class="step-card-desc">
      Click <b>▶ Start Engine</b> in the sidebar, then open
      <b>📹 Live Monitor</b>. The camera stays on while that page is open.
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">02</div>
    <div class="step-card-title">Enroll faces</div>
    <div class="step-card-desc">
      Go to <b>🧠 Training → Face Enrollment</b>. The scan is automatic —
      there is a manual shutter and a photo upload if you would rather not
      wait. Unknown faces still create events; they land in the Review Queue.
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">03</div>
    <div class="step-card-title">Hold a plate in view</div>
    <div class="step-card-desc">
      Place a plate in the lower part of the camera view.
      Empty plates are ignored. Leftovers create one transaction
      with an evidence photo.
    </div>
  </div>

  <div class="step-card">
    <div class="step-card-num">04</div>
    <div class="step-card-title">Train YOLO (optional)</div>
    <div class="step-card-desc">
      Capture tray photos straight from the camera on <b>🧠 Training</b>, then
      hit <b>Start training</b> and watch the epochs stream in. Activating a
      version hot-swaps the running engine.
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
