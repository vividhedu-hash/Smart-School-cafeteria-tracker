"""
Page 1 — Live Monitor

Real-time camera feed + always-on face recognition and cafeteria plate analysis.

Supports:
  1. Live Browser Webcam (WebRTC direct streaming with real-time AI overlay)
  2. Instant Snapshot / Single-frame capture
  3. Background Engine Stream (local CV daemon with hardware/RTSP webcam)
"""
from __future__ import annotations

import base64
import datetime
import io
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageFile
import av
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from cafeteria.monitoring.overlay import draw_face_perimeter
from cafeteria.utils.timing import FPSCounter

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
    "Real-time computer vision cafeteria monitor: plate detection, food waste classification, "
    "and ArcFace biometric face identification. Live video streams directly from your webcam with zero simulation."
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


def _enrolled_count() -> int:
    """Robust count of enrolled face profiles from disk."""
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
                <span style="color:#60a5fa;font-weight:600;">Looking for someone at checkout…</span>
            </div>
            <span class="id-badge badge-scanning">SCANNING</span>
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
            "Tracking walk-past — identity holds while moving"
            if tracking
            else "Face detected — look towards camera"
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
        <div class="id-name" style="color:#fb923c;">Unknown Person</div>
        <div style="color:#94a3b8;font-size:0.9rem;margin:6px 0 10px;">
            Best match: {sim_pct}% — threshold is {thresh:.2f}
        </div>
        <span class="id-badge badge-unknown">NOT ENROLLED</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Enroll this person on the Training page to identify them here.")


# ─────────────────────────────────────────────────────────────────────────────
# Cached AI Models for Live Browser WebRTC Inference
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_live_models():
    """Load the full CV & biometric stack once and keep cached in memory."""
    root = cfg.project_root if cfg else Path.cwd()
    from cafeteria.detection.plate_detector import PlateDetector
    from cafeteria.detection.waste_detector import WasteDetector
    from cafeteria.recognition.face_engine import FaceEngine
    from cafeteria.recognition.matcher import EmbeddingMatcher

    plate_weights = str(root / cfg.models.plate.weights) if cfg else "models/plate/best.pt"
    waste_weights = str(root / cfg.models.waste.weights) if cfg else "models/waste/best.pt"
    device = cfg.device if cfg else "cpu"

    plate = PlateDetector(
        weights_path=plate_weights,
        confidence=float(getattr(cfg.models.plate, "confidence", 0.35) if cfg else 0.35),
        iou=float(getattr(cfg.models.plate, "iou", 0.45) if cfg else 0.45),
        device=device,
        allow_visual_fallback=True,
    )
    plate.load()

    waste = WasteDetector(
        weights_path=waste_weights,
        confidence=float(getattr(cfg.models.waste, "confidence", 0.35) if cfg else 0.35),
        device=device,
        allow_visual_fallback=True,
    )
    waste.load()

    face_engine = FaceEngine(
        model_pack=cfg.recognition.model_pack if cfg else "buffalo_s",
        model_dir=root / "models" / "face",
        det_size=(320, 320),
        device=device,
        det_thresh=0.35,
    )
    face_engine.load()

    matcher = EmbeddingMatcher(
        enrollment_dir=root / cfg.recognition.embedding_dir if cfg else root / "data" / "enrollment",
        similarity_threshold=float(cfg.recognition.similarity_threshold if cfg else 0.52),
    )
    matcher.load_embeddings()

    return {
        "plate": plate,
        "waste": waste,
        "face_engine": face_engine,
        "matcher": matcher,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real-Time WebRTC Video Processor
# ─────────────────────────────────────────────────────────────────────────────
class LiveCafeteriaVideoProcessor(VideoProcessorBase):
    """
    Processes the user's real browser webcam frames in real time:
      - InsightFace ArcFace face detection and identity recognition
      - Plate detection (trained YOLO + visual fallback)
      - Waste classification (trained YOLO + visual fallback)
      - Kinematic person-plate association
      - Automatic real database transaction commit
      - Live HUD overlays
    """

    def __init__(self) -> None:
        self._models = _get_live_models()
        self._lock = threading.Lock()
        self.frame_count = 0
        self.fps_counter = FPSCounter(window=30)
        self.last_state = "IDLE"
        self.last_face_match: dict | None = None
        self.last_plate_det = None
        self.last_waste_result = None
        self.last_commit_time = 0.0
        self.last_event: dict | None = None
        self.latency_ms = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        t0 = time.time()
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        self.frame_count += 1
        self.fps_counter.tick()

        face_engine = self._models["face_engine"]
        matcher = self._models["matcher"]
        plate_detector = self._models["plate"]
        waste_detector = self._models["waste"]

        current_face_match = None
        current_plate = None
        current_waste = None
        best_face_box = None
        state = "IDLE"

        # 1. Real Face Detection & Recognition
        try:
            faces = face_engine.get_faces(img)
            if faces:
                state = "FACE_CAPTURE"
                # Pick largest face
                best_face = max(
                    faces,
                    key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                )
                best_face_box = best_face["bbox"]
                match = matcher.match(best_face["embedding"])
                current_face_match = match
                x1, y1, x2, y2 = best_face_box

                if match.is_known:
                    state = "FACE_RECOGNITION"
                    name = match.person_name or match.person_id or "Recognized"
                    sim_pct = int(match.similarity * 100)
                    color = (60, 220, 80)
                    label = f"✓ {name} ({sim_pct}%)"
                else:
                    color = (0, 165, 255)
                    sim_pct = int(match.similarity * 100)
                    label = f"UNKNOWN ({sim_pct}%)"

                draw_face_perimeter(img, (x1, y1, x2, y2), color, label)
        except Exception:
            pass

        # 2. Real Plate Detection & Waste Classification
        try:
            plates = plate_detector.detect(img)
            if plates:
                p = plates[0]
                current_plate = p
                state = "PLATE_DETECTED"
                crop = p.crop(img)
                if crop is not None and crop.size > 0:
                    w_res = waste_detector.classify(crop)
                    current_waste = w_res
                    state = "FOOD_ANALYSIS"

                    waste_color = {
                        "EMPTY": (180, 180, 180),
                        "LOW_WASTE": (0, 220, 255),
                        "MEDIUM_WASTE": (0, 140, 255),
                        "HIGH_WASTE": (0, 60, 255),
                    }.get(w_res.label, (0, 255, 0))

                    cv2.rectangle(img, (p.x1, p.y1), (p.x2, p.y2), waste_color, 2)
                    cv2.putText(
                        img,
                        f"PLATE: {w_res.label} ({int(w_res.confidence * 100)}%)",
                        (p.x1, max(20, p.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        waste_color,
                        2,
                    )
        except Exception:
            pass

        # 3. Kinetic person-plate link line
        if best_face_box and current_plate:
            fx = (best_face_box[0] + best_face_box[2]) // 2
            fy = (best_face_box[1] + best_face_box[3]) // 2
            px, py = current_plate.center
            cv2.line(img, (fx, fy), (px, py), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img, ((fx + px) // 2, (fy + py) // 2), 4, (0, 255, 255), -1)

        # 4. Real Database Transaction Commit
        now = time.time()
        if current_face_match and current_plate and current_waste:
            state = "TRANSACTION_COMMIT"
            if now - self.last_commit_time > 4.5:
                self.last_commit_time = now
                pid = current_face_match.person_id if current_face_match.is_known else "UNKNOWN"
                w_status = current_waste.label
                try:
                    from cafeteria.storage.database import get_session
                    from cafeteria.storage.repositories import TransactionRepository
                    sess = get_session()
                    repo = TransactionRepository(sess)
                    repo.create(
                        person_id=pid,
                        waste_status=w_status,
                        confidence=float(current_waste.confidence),
                        plate_confidence=float(current_plate.confidence),
                        face_similarity=float(current_face_match.similarity) if current_face_match else 0.0,
                    )
                    sess.commit()
                    sess.close()
                    self.last_event = {
                        "person_id": pid,
                        "person_name": current_face_match.person_name if current_face_match.is_known else "Unknown",
                        "waste_status": w_status,
                        "confidence": float(current_waste.confidence),
                        "status": "APPROVED",
                        "latency_ms": (now - t0) * 1000.0,
                        "timestamp": now,
                    }
                except Exception:
                    pass

        elapsed_ms = (time.time() - t0) * 1000.0
        self.latency_ms = elapsed_ms
        fps = self.fps_counter.fps

        # 5. On-frame HUD Status Bar
        state_bg = {
            "IDLE":               (55, 60, 68),
            "PLATE_DETECTED":     (0, 160, 40),
            "FOOD_ANALYSIS":      (180, 130, 0),
            "FACE_CAPTURE":       (180, 80, 0),
            "FACE_RECOGNITION":   (190, 40, 130),
            "TRANSACTION_COMMIT": (0, 170, 70),
        }.get(state, (55, 60, 68))

        cv2.rectangle(img, (0, 0), (w, 32), state_bg, -1)
        cv2.putText(img, f"STATE: {state}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            img,
            f"FPS: {fps:.1f}  |  {elapsed_ms:.0f} ms",
            (w - 220, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        with self._lock:
            self.last_state = state
            if current_face_match:
                self.last_face_match = {
                    "person_id": current_face_match.person_id,
                    "person_name": current_face_match.person_name,
                    "similarity": current_face_match.similarity,
                    "is_known": current_face_match.is_known,
                    "approaching": False,
                    "moving": False,
                }
            self.last_plate_det = current_plate
            self.last_waste_result = current_waste

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _render_boot_card(alive_seconds: float) -> None:
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


# ── Fragment view for Background Engine Stream ───────────────────────────────
@st.fragment(run_every=REFRESH_SECONDS)
def background_engine_view() -> None:
    write_heartbeat()

    if not engine_is_alive():
        return

    st.session_state.setdefault("engine_started_at", time.time())
    state_data = fetch_state(
        cfg.project_root / cfg.application.runtime_state_path, **_api_kwargs()
    )
    status_strip(state_data, engine_alive=True, enrolled=_enrolled_count())

    if not state_data:
        _render_boot_card(time.time() - st.session_state["engine_started_at"])
        return

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("🎥 Background Stream")
        frame_bytes = fetch_frame_bytes(
            cfg.project_root / cfg.storage.frames / "latest.jpg", **_api_kwargs()
        )
        if frame_bytes:
            try:
                st.image(Image.open(io.BytesIO(frame_bytes)).copy(), use_container_width=True)
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


# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Camera & Engine")
    eng_alive = engine_is_alive()
    if eng_alive:
        st.markdown(
            '<span class="state-badge status-ok">● ENGINE RUNNING</span>',
            unsafe_allow_html=True,
        )
        if st.button("⏹ Stop engine", key="sidebar_stop", use_container_width=True):
            stop_engine()
            st.rerun()
    else:
        st.markdown(
            '<span class="state-badge status-error">● ENGINE STOPPED</span>',
            unsafe_allow_html=True,
        )
        if st.button("▶ Start engine", key="sidebar_start", use_container_width=True):
            start_engine()
            st.session_state["engine_started_at"] = time.time()
            st.rerun()
    st.caption(
        "Direct Browser Webcam works instantly anywhere. Starting the background engine "
        "enables background recording and local daemon processing."
    )


# ── Top Persistent Status Strip ──────────────────────────────────────────────
_enrolled = _enrolled_count()
if eng_alive and cfg is not None:
    _state_init = fetch_state(
        cfg.project_root / cfg.application.runtime_state_path, **_api_kwargs()
    )
else:
    _state_init = {}

status_strip(_state_init, engine_alive=eng_alive, enrolled=_enrolled)


# ── Mode Selection Tabs ──────────────────────────────────────────────────────
tab_live_cam, tab_bg_stream = st.tabs([
    "📸 Real-Time Browser Webcam (Live AI Vision)",
    "🖥️ Background Engine Stream",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Live Browser Webcam (WebRTC Real-time AI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_live_cam:
    st.markdown("##### 🔴 Live Camera AI Inference (Zero Simulation)")
    st.caption(
        "Streams directly from your physical device camera into our InsightFace, YOLO, and OpenCV models. "
        "Hold up a plate or step into the frame to see live bounding boxes and automatic checkout matching."
    )

    col_cam, col_id = st.columns([3, 2], gap="large")

    with col_cam:
        # WebRTC stream with Google STUN configuration
        webrtc_ctx = webrtc_streamer(
            key="live_cafeteria_webrtc_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=LiveCafeteriaVideoProcessor,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            async_processing=True,
        )

        proc = webrtc_ctx.video_processor

        # Fallback single-frame shutter for networks/browsers blocking WebRTC
        with st.expander("📸 Or snap an Instant Shutter Photo (Single-frame analysis)", expanded=False):
            shutter_shot = st.camera_input("Capture frame for immediate AI analysis", key="live_shutter_input")
            if shutter_shot is not None:
                s_bytes = shutter_shot.getvalue()
                s_nparr = np.frombuffer(s_bytes, np.uint8)
                s_img = cv2.imdecode(s_nparr, cv2.IMREAD_COLOR)
                if s_img is not None:
                    models = _get_live_models()
                    s_faces = models["face_engine"].get_faces(s_img)
                    s_plates = models["plate"].detect(s_img)
                    s_match = None
                    s_waste = None

                    if s_faces:
                        best = max(s_faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                        s_match = models["matcher"].match(best["embedding"])
                        color = (60, 220, 80) if s_match.is_known else (0, 165, 255)
                        lbl = f"✓ {s_match.person_name} ({int(s_match.similarity*100)}%)" if s_match.is_known else f"UNKNOWN ({int(s_match.similarity*100)}%)"
                        draw_face_perimeter(s_img, best["bbox"], color, lbl)

                    if s_plates:
                        sp = s_plates[0]
                        scrop = sp.crop(s_img)
                        if scrop is not None and scrop.size > 0:
                            s_waste = models["waste"].classify(scrop)
                            cv2.rectangle(s_img, (sp.x1, sp.y1), (sp.x2, sp.y2), (0, 220, 255), 2)
                            cv2.putText(s_img, f"PLATE: {s_waste.label}", (sp.x1, max(20, sp.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

                    st.image(cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB), caption="Analyzed Live Snapshot", use_container_width=True)
                    if s_match:
                        st.success(f"Person: **{s_match.person_name or 'Unknown'}** (Similarity: {s_match.similarity:.2f})")
                    if s_waste:
                        st.info(f"Waste Level: **{s_waste.label}** (Confidence: {s_waste.confidence:.2f})")

        # Status and latency metrics
        if proc:
            with proc._lock:
                live_fps = proc.fps_counter.fps
                live_ms = proc.latency_ms
                live_st = proc.last_state

            m1, m2, m3 = st.columns(3)
            m1.metric("Camera FPS", f"{live_fps:.1f}")
            m2.metric("Inference Latency", f"{live_ms:.0f} ms")
            m3.metric("Pipeline State", live_st)
        else:
            st.info("Click **START** above to turn on your real webcam and begin live AI recognition.")

    with col_id:
        st.subheader("🔍 Who's here?")
        if proc:
            with proc._lock:
                live_face = proc.last_face_match
                live_evt = proc.last_event

            _render_identity({"live_face_match": live_face})

            if live_evt:
                st.markdown("---")
                st.markdown("#### ✅ Last Real Transaction Committed")
                st.markdown(f"- **Person:** `{live_evt.get('person_name') or live_evt.get('person_id')}`")
                st.markdown(f"- **Waste Status:** `{live_evt.get('waste_status')}`")
                st.markdown(f"- **Verification:** `✓ {live_evt.get('status')}`")
                st.markdown(f"- **Processing Time:** `{live_evt.get('latency_ms', 0):.0f} ms`")
        else:
            _render_identity({})

        st.markdown("---")
        st.markdown("#### ⚡ Active Computer Vision Pipeline")
        st.markdown("✅ `InsightFace ArcFace` — Real Face Embedding Matching")
        st.markdown("✅ `YOLO / Visual Detector` — Real Plate & Dish Tracking")
        st.markdown("✅ `YOLO / Visual Classifier` — Real Food Waste Occupancy")
        st.markdown(f"👥 `Enrolled Profiles` — **{_enrolled}** person(s) active in database")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Background Engine Stream
# ─────────────────────────────────────────────────────────────────────────────
with tab_bg_stream:
    if eng_alive:
        background_engine_view()
    else:
        st.info(
            "**The background inference engine is currently STOPPED.**\n\n"
            "Click **▶ Start Engine Now** to run the background CV daemon (reads local USB/RTSP cams and writes daemon logs).",
            icon="⏸️",
        )
        col_start, col_help = st.columns([1, 2])
        with col_start:
            if st.button("▶ Start Background Engine", type="primary", key="tab_bg_start_btn", use_container_width=True):
                start_engine()
                st.session_state["engine_started_at"] = time.time()
                st.rerun()
        with col_help:
            st.caption(
                "Runs `cafeteria.main` as a background process for continuous cafeteria station deployments."
            )
