"""
Page 2 — Training & Dataset Management

Tabs:
  A. Face Enrollment  — automatic guided scan, forgiving, with manual override
  B. People           — roster, photos, rename, delete
  C. Waste Dataset    — capture from camera or upload, per class
  D. Plate Dataset    — capture plus assisted YOLO box labelling
  E. Train Model      — launch a run and watch real epoch progress
  F. Model Versions   — history and activation

The camera is shared, not stolen: while the engine is running this page
borrows its clean frames over HTTP instead of opening a second capture
device (macOS allows only one owner).
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_dashboard_dir = _project_root / "dashboard"
for p in [str(_project_root / "src"), str(_project_root), str(_dashboard_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Training & Enrollment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth_gate import require_login
require_login()

from theme import inject_css, status_strip

inject_css()

st.markdown("""
<style>
.wizard-steps { display:flex; align-items:center; gap:0; margin:1.2rem 0; }
.wizard-step  { display:flex; flex-direction:column; align-items:center; flex:1; }
.wizard-step-circle {
    width:40px; height:40px; border-radius:50%; display:flex;
    align-items:center; justify-content:center; font-weight:700; font-size:1rem;
    border:2px solid #334155; background:#0f172a; color:#64748b; transition:all .3s;
}
.wizard-step-circle.active { background:linear-gradient(135deg,#3b82f6,#8b5cf6); border-color:#3b82f6; color:#fff; box-shadow:0 0 20px rgba(59,130,246,.5); }
.wizard-step-circle.done   { background:#059669; border-color:#059669; color:#fff; }
.wizard-step-label { font-size:.72rem; margin-top:6px; color:#64748b; text-align:center; font-weight:500; }
.wizard-step-label.active  { color:#93c5fd; font-weight:600; }
.wizard-step-label.done    { color:#34d399; }
.wizard-step-line { flex:1; height:2px; background:#1e293b; margin:0 -1px; position:relative; top:-14px; }
.wizard-step-line.done { background:#059669; }

.person-card {
    background:linear-gradient(135deg,#1a2438,#0f172a); border:1px solid #2b3a55;
    border-radius:12px; padding:12px 16px; margin:4px 0;
    display:flex; align-items:center; gap:12px;
}
.person-card-name { font-weight:600; color:#e2e8f0; }
.person-card-meta { font-size:.78rem; color:#94a3b8; }
.badge-ready   { background:#064e3b; color:#34d399; border-radius:20px; padding:2px 8px; font-size:.75rem; font-weight:600; }
.badge-pending { background:#1c1917; color:#f59e0b; border-radius:20px; padding:2px 8px; font-size:.75rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Training & Dataset Management")

# ── Project imports ───────────────────────────────────────────────────────────
from cafeteria.config.settings import load_settings
from cafeteria.detection.visual import detect_plates_visual
from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES
from cafeteria.training.progress import (
    clear_status,
    is_running,
    read_status,
    status_path,
    update_status,
    write_status,
)
from cafeteria.training.registry import ModelRegistry
from cafeteria.monitoring.metrics import write_command
from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import PersonRepository
from engine_ctl import engine_is_alive, start_engine, stop_engine, write_heartbeat
from engine_client import fetch_raw_frame_bytes, fetch_state
from empty_states import empty_state_html
from enroll_cam import (
    TARGET_FRAMES,
    restart_enroll_camera,
    start_enroll_camera,
    stop_enroll_camera,
)

import inspect as _inspect


# [AI-CoLab: Cursor] Drop kwargs the target constructor does not accept. Needed
# because FaceEngine/EnrollmentManager signatures have drifted during collab.
def _construct(cls, **wanted):
    params = _inspect.signature(cls.__init__).parameters
    if any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**wanted)
    return cls(**{k: v for k, v in wanted.items() if k in params and k != "self"})


cfg      = load_settings(config_path=_project_root / "configs" / "config.yaml")
registry = ModelRegistry(cfg.project_root / "models" / "registry.json")
dm       = DatasetManager(
    cfg.project_root / cfg.storage.datasets,
    train_ratio=cfg.training.train_ratio,
    val_ratio=cfg.training.val_ratio,
    test_ratio=cfg.training.test_ratio,
    random_seed=cfg.training.random_seed,
)
TRAINING_STATUS_FILE = status_path(cfg.project_root)

init_db(cfg.project_root / cfg.storage.database)
enrollment_mgr = EnrollmentManager(
    enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
    audit_path=cfg.project_root / "data" / "audit" / "roster.jsonl",
)

_sync = get_session()
try:
    if hasattr(enrollment_mgr, "sync_to_database"):
        enrollment_mgr.sync_to_database(_sync)
        _sync.commit()
except Exception:
    _sync.rollback()
finally:
    _sync.close()
if hasattr(enrollment_mgr, "write_gallery_index"):
    enrollment_mgr.write_gallery_index()

_ENGINE_ALIVE = engine_is_alive()
_API = {
    "host": getattr(cfg.application, "api_host", None),
    "port": getattr(cfg.application, "api_port", None),
}
_RAW_FRAME_FALLBACK = cfg.project_root / cfg.storage.frames / "latest_raw.jpg"

status_strip(
    fetch_state(cfg.project_root / cfg.application.runtime_state_path, **_API)
    if _ENGINE_ALIVE else {},
    engine_alive=_ENGINE_ALIVE,
    enrolled=sum(
        1 for pid in enrollment_mgr.list_enrolled() if enrollment_mgr.has_embedding(pid)
    ),
)


def _engine_frame() -> bytes | None:
    """Clean frame from the running engine (no overlay drawn on it)."""
    return fetch_raw_frame_bytes(fallback_path=_RAW_FRAME_FALLBACK, **_API)


def _engine_command(cmd: dict) -> None:
    posted = False
    try:
        from engine_client import post_command
        posted = post_command(cmd)
    except Exception:
        posted = False
    if not posted:
        write_command(cfg.project_root / cfg.application.commands_path, cmd)


def _notify_matcher() -> None:
    _engine_command({"type": "reload_embeddings"})


@st.cache_resource(show_spinner=False)
def _face_engine():
    """InsightFace loaded once per dashboard process, not once per enrollment."""
    from cafeteria.recognition.face_engine import FaceEngine
    fe = _construct(
        FaceEngine,
        model_pack=cfg.recognition.model_pack,
        model_dir=cfg.project_root / "models" / "face",
        det_size=tuple(cfg.recognition.det_size),
        device=cfg.device,
        det_thresh=float(getattr(cfg.recognition, "det_thresh", 0.5)),
    )
    fe.load()
    return fe


def _rebuild_face_profile(pid: str, pname: str):
    emgr = _construct(
        EnrollmentManager,
        enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
        face_engine=_face_engine(),
        min_face_size=cfg.recognition.minimum_face_size,
        audit_path=cfg.project_root / "data" / "audit" / "roster.jsonl",
    )
    emb = emgr.generate_embedding(pid, pname)
    sess = get_session()
    repo = PersonRepository(sess)
    repo.update_image_count(pid, emgr.image_count(pid))
    if emb is None:
        repo.clear_embedding(pid)
    else:
        repo.update_embedding_path(pid, str(emgr.embedding_path(pid)))
    sess.commit()
    sess.close()
    _notify_matcher()
    return emb


def _decode_upload(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img if img is not None and img.size else None


def _add_images_to_waste_class(class_name: str, images: list[bytes]) -> int:
    """Persist raw JPEG/PNG bytes into a waste class folder."""
    added = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        paths: list[Path] = []
        for i, raw in enumerate(images):
            path = Path(tmpdir) / f"capture_{int(time.time() * 1000)}_{i}.jpg"
            path.write_bytes(raw)
            paths.append(path)
        if paths:
            added, _ = dm.add_waste_images(class_name, paths)
    return added


def _wizard_html(current_step: str) -> str:
    steps = [("person_select", "1", "Person"), ("capture", "2", "Scan"), ("embed", "3", "Finalize")]
    order = [s[0] for s in steps]
    cur   = order.index(current_step) if current_step in order else 0
    html  = '<div class="wizard-steps">'
    for i, (_key, num, label) in enumerate(steps):
        if i < cur:    cc, cl, nt = "done",   "done",   "✓"
        elif i == cur: cc, cl, nt = "active", "active", num
        else:          cc, cl, nt = "",       "",       num
        html += (f'<div class="wizard-step">'
                 f'<div class="wizard-step-circle {cc}">{nt}</div>'
                 f'<div class="wizard-step-label {cl}">{label}</div></div>')
        if i < len(steps) - 1:
            html += f'<div class="wizard-step-line {"done" if i < cur else ""}"></div>'
    return html + "</div>"


# =============================================================================
# TABS
# =============================================================================
tab_enroll, tab_people, tab_dataset, tab_plate, tab_train, tab_versions = st.tabs([
    "👤 Face Enrollment",
    "🗂️ People",
    "📦 Waste Dataset",
    "🍽️ Plate Dataset",
    "🚀 Train Model",
    "📋 Model Versions",
])

# =============================================================================
# TAB A — FACE ENROLLMENT
# =============================================================================
with tab_enroll:

    for key, val in [
        ("enroll_step",         "person_select"),
        ("enroll_pid",          ""),
        ("enroll_name",         ""),
        ("enroll_saved",        False),
        ("enroll_embed_status", None),
        ("enroll_source",       "engine" if _ENGINE_ALIVE else "direct"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — pick or create a person
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.enroll_step == "person_select":
        st.subheader("Step 1 — who are we scanning?")
        st.markdown(_wizard_html("person_select"), unsafe_allow_html=True)
        st.caption(
            "Pick someone, then look into the oval. The scan captures "
            f"{TARGET_FRAMES} frames automatically and there is a manual shutter "
            "if you would rather not wait."
        )

        col_new, col_existing = st.columns(2)

        with col_new:
            st.markdown("#### ➕ New person")
            new_pid  = st.text_input("Person ID",    placeholder="person_01", key="new_pid_input")
            new_name = st.text_input("Display name", placeholder="Alice Smith", key="new_name_input")
            if st.button("Start scan →", type="primary", key="btn_create_start"):
                if not new_pid.strip() or not new_name.strip():
                    st.info("Add an ID and a display name so we know who this profile belongs to.")
                else:
                    try:
                        pid = new_pid.strip()
                        enrollment_mgr.ensure_person_dir(pid)
                        enrollment_mgr.update_display_name(pid, new_name.strip())
                        sess = get_session()
                        repo = PersonRepository(sess)
                        if not repo.get_by_person_id(pid):
                            repo.create(person_id=pid, name=new_name.strip())
                            sess.commit()
                        sess.close()
                        st.session_state.update(
                            enroll_pid=pid,
                            enroll_name=new_name.strip(),
                            enroll_saved=False,
                            enroll_embed_status=None,
                            enroll_step="capture",
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with col_existing:
            st.markdown("#### 🔄 Existing person")
            sess        = get_session()
            all_persons = PersonRepository(sess).get_all_active()
            sess.close()
            if not all_persons:
                st.info("No people yet — create one on the left.")
            else:
                pmap      = {f"{p.person_id} — {p.name}": p for p in all_persons}
                sel_label = st.selectbox("Select person", list(pmap.keys()), key="exist_sel")
                sel_p     = pmap[sel_label]
                n_imgs    = enrollment_mgr.image_count(sel_p.person_id)
                badge     = ('<span class="badge-ready">✅ READY</span>'
                             if enrollment_mgr.has_embedding(sel_p.person_id) else
                             '<span class="badge-pending">⏳ PENDING</span>')
                st.markdown(f"Images: **{n_imgs}** &nbsp; {badge}", unsafe_allow_html=True)
                if st.button("Scan face →", key="btn_add_imgs"):
                    st.session_state.update(
                        enroll_pid=sel_p.person_id,
                        enroll_name=sel_p.name,
                        enroll_saved=False,
                        enroll_embed_status=None,
                        enroll_step="capture",
                    )
                    st.rerun()

        st.markdown("---")
        st.markdown("### 📋 Enrolled people")
        st.caption("To rename someone, delete photos, or remove a person, open the **People** tab.")
        sess  = get_session()
        all_p = PersonRepository(sess).get_all_active()
        sess.close()
        if not all_p:
            st.info("No people enrolled yet.")
        else:
            for p in all_p:
                n     = enrollment_mgr.image_count(p.person_id)
                has_e = enrollment_mgr.has_embedding(p.person_id)
                st.markdown(f"""
                <div class="person-card">
                    <div style="font-size:2rem">👤</div>
                    <div style="flex:1">
                        <div class="person-card-name">{p.name}</div>
                        <div class="person-card-meta">ID: {p.person_id} &nbsp;·&nbsp; {n} images</div>
                    </div>
                    <span class="{'badge-ready' if has_e else 'badge-pending'}">
                        {'✅ READY' if has_e else '⏳ PENDING'}</span>
                </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — automatic scan (forgiving) with manual shutter
    # ─────────────────────────────────────────────────────────────────────────
    elif st.session_state.enroll_step == "capture":
        pid   = st.session_state.enroll_pid
        pname = st.session_state.enroll_name
        session_key = f"enroll_{pid}"
        cam_src = cfg.camera.source if isinstance(cfg.camera.source, int) else 0
        borrowing = st.session_state.enroll_source == "engine" and _ENGINE_ALIVE

        st.subheader(f"Step 2 — scan **{pname}**")
        st.markdown(_wizard_html("capture"), unsafe_allow_html=True)

        if borrowing:
            st.caption(
                "Borrowing frames from the running engine, so the Live Monitor keeps working. "
                f"Look towards the camera — we take {TARGET_FRAMES} shots on our own."
            )
        else:
            st.caption(
                "Using the webcam directly from Python. "
                f"Look towards the camera — we take {TARGET_FRAMES} shots on our own."
            )

        def _provider():
            return _engine_frame() if borrowing else None

        cam = start_enroll_camera(
            session_key,
            source=int(cam_src),
            frame_provider=_provider if borrowing else None,
        )

        ctrl_snap, ctrl_use, ctrl_restart, ctrl_source = st.columns([1, 1, 1, 1.4])
        with ctrl_snap:
            if st.button("📸 Snap now", key="btn_snap_now", width="stretch"):
                cam.request_snap()
        with ctrl_use:
            if st.button("✓ Use these shots", key="btn_use_shots", width="stretch"):
                if not cam.finish_now():
                    st.info("No frames captured yet — press **Snap now** at least once.")
        with ctrl_restart:
            if st.button("↻ Restart scan", key="btn_restart_scan", width="stretch"):
                st.session_state.enroll_saved = False
                restart_enroll_camera(
                    session_key,
                    source=int(cam_src),
                    frame_provider=_provider if borrowing else None,
                )
                st.rerun()
        with ctrl_source:
            if borrowing:
                if st.button("Use webcam directly (stops engine)", key="btn_src_direct",
                             width="stretch"):
                    stop_enroll_camera(session_key)
                    stop_engine()
                    st.session_state.enroll_source = "direct"
                    st.rerun()
            elif _ENGINE_ALIVE:
                if st.button("Borrow the engine camera", key="btn_src_engine", width="stretch"):
                    stop_enroll_camera(session_key)
                    st.session_state.enroll_source = "engine"
                    st.rerun()
            else:
                if st.button("Start engine and borrow it", key="btn_src_start_engine",
                             width="stretch"):
                    stop_enroll_camera(session_key)
                    start_engine()
                    st.session_state.enroll_source = "engine"
                    st.rerun()

        @st.fragment(run_every=0.25)
        def _enroll_preview() -> None:
            if borrowing:
                write_heartbeat()  # keep the engine (and its camera) alive while scanning
            live = start_enroll_camera(
                session_key,
                source=int(cam_src),
                frame_provider=_provider if borrowing else None,
            )
            s = live.status()

            if s.error:
                st.warning(s.error, icon="📷")
                st.caption(
                    "Switch the capture source above, or upload photos further down — "
                    "either path finishes the enrollment."
                )
                return

            tone = "#34d399" if s.face_visible else "#7dd3fc"
            sub = {
                "relaxed": "Close enough — capturing now",
                "manual": "Captured from the manual shutter",
            }.get(s.mode, "Automatic capture, no need to hold perfectly still")
            st.markdown(
                f'<div class="coach-panel">'
                f'<div class="coach-headline" style="color:{tone}">{s.hint}</div>'
                f'<div class="coach-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

            if s.jpeg:
                st.image(s.jpeg, width=420)
            else:
                st.info("Opening the camera…")

            st.progress(min(s.count / max(s.target, 1), 1.0), text=f"{s.count} / {s.target} shots")

            if s.needs_help:
                st.info(
                    "No face detected yet. More light on your face usually fixes it — "
                    "otherwise press **Snap now** a few times, or upload photos below. "
                    "InsightFace makes the final call on which shots are usable.",
                    icon="💡",
                )

            if s.done and not st.session_state.enroll_saved:
                saved = 0
                for img in live.take_captures():
                    enrollment_mgr.save_image(pid, img, pose="front")
                    saved += 1
                stop_enroll_camera(session_key)
                st.session_state.enroll_saved = True
                if saved:
                    st.session_state.enroll_embed_status = "pending"
                    st.session_state.enroll_step = "embed"
                    st.rerun(scope="app")

        _enroll_preview()

        st.markdown("---")
        st.markdown("### Prefer to upload photos?")
        st.caption("Three or more front-facing photos in decent light are plenty.")
        uploads = st.file_uploader(
            "Face photos",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key=f"enroll_upload_{pid}",
        )
        if st.button("Save photos and finish enrollment", type="primary", key="btn_upload_enroll"):
            if not uploads:
                st.info("Add at least one photo first.")
            else:
                stop_enroll_camera(session_key)
                poses = ["front", "left", "right", "up", "down"]
                saved = 0
                for i, uf in enumerate(uploads):
                    img = _decode_upload(uf.getvalue())
                    if img is None:
                        continue
                    enrollment_mgr.save_image(pid, img, pose=poses[i % len(poses)])
                    saved += 1
                if saved < 1:
                    st.error("Those files could not be read as images.")
                else:
                    st.session_state.enroll_saved = True
                    st.session_state.enroll_embed_status = "pending"
                    st.session_state.enroll_step = "embed"
                    st.rerun()

        if st.button("← Back to people", key="back_to_select"):
            stop_enroll_camera(session_key)
            st.session_state.enroll_step = "person_select"
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — build the ArcFace profile
    # ─────────────────────────────────────────────────────────────────────────
    elif st.session_state.enroll_step == "embed":
        pid   = st.session_state.enroll_pid
        pname = st.session_state.enroll_name

        st.subheader(f"Enrolling **{pname}**")
        st.markdown(_wizard_html("embed"), unsafe_allow_html=True)
        st.caption("Turning those stills into an ArcFace profile — a few seconds.")

        if st.session_state.get("enroll_embed_status") == "pending":
            with st.spinner("Loading the face model and building the profile…"):
                try:
                    emb = _rebuild_face_profile(pid, pname)
                    st.session_state.enroll_embed_status = "ok" if emb is not None else "empty"
                except Exception as e:
                    st.session_state.enroll_embed_status = "error"
                    st.session_state.enroll_embed_error = str(e)
            st.rerun()

        status = st.session_state.get("enroll_embed_status")
        if status == "ok":
            st.success(
                f"{pname} is enrolled. The engine reloads the gallery immediately — "
                "open Live Monitor to see the match."
            )
        elif status == "empty":
            st.warning(
                "No usable face in those stills. Run the scan again with more light on "
                "your face, or upload a few clear photos.",
                icon="🙈",
            )
            if st.button("↻ Scan again", type="primary", key="btn_rescan"):
                st.session_state.update(enroll_saved=False, enroll_step="capture",
                                        enroll_embed_status=None)
                st.rerun()
        elif status == "error":
            st.error(f"Profile step failed: {st.session_state.get('enroll_embed_error', '')}")
            if st.button("Try again", type="primary", key="btn_gen_embed_retry"):
                st.session_state.enroll_embed_status = "pending"
                st.rerun()
        else:
            if st.button("Build profile", type="primary", key="btn_gen_embed"):
                st.session_state.enroll_embed_status = "pending"
                st.rerun()

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            if st.button("Scan someone else", key="btn_more"):
                st.session_state.update(
                    enroll_step="person_select", enroll_pid="", enroll_name="",
                    enroll_saved=False, enroll_embed_status=None,
                )
                st.rerun()
        with cb:
            if st.button("Back to people", key="btn_roster"):
                st.session_state.enroll_step = "person_select"
                st.rerun()

        images = enrollment_mgr.list_images(pid)
        if images:
            st.markdown(f"Photos on file ({len(images)}). Manage them in the **People** tab.")
            cols = st.columns(8)
            for i, ip in enumerate(images[:16]):
                pil = Image.open(ip)
                pil.thumbnail((80, 80))
                with cols[i % 8]:
                    st.image(pil, caption=ip.name, width="content")


# =============================================================================
# TAB B — People roster
# =============================================================================
with tab_people:
    st.subheader("People roster")
    st.caption(
        "Edit a display name, remove individual photos, or delete a person. "
        "Photos live under `data/enrollment/{id}/images/` — that folder is the "
        "sample set the face model reads. Removing photos clears the live "
        "profile until you rebuild it."
    )

    sess = get_session()
    roster = PersonRepository(sess).get_all_active()
    sess.close()

    if not roster:
        st.markdown(
            empty_state_html(
                icon="🗂️",
                title="No people on file",
                body="Enroll someone in Face Enrollment first. Their photos and embedding will show up here.",
                steps=[
                    "Open Face Enrollment and run a guided scan.",
                    "Come back here to review photos or remove a person.",
                ],
            ),
            unsafe_allow_html=True,
        )
    else:
        pmap = {f"{p.person_id}  —  {p.name}": p for p in roster}
        choice = st.selectbox("Person", list(pmap.keys()), key="people_sel")
        person = pmap[choice]
        pid = person.person_id
        n_imgs = enrollment_mgr.image_count(pid)
        has_emb = enrollment_mgr.has_embedding(pid)
        meta = enrollment_mgr.load_meta(pid) or {}
        stale = bool(meta.get("embedding_stale")) or (n_imgs > 0 and not has_emb)

        left, right = st.columns([2, 1], gap="large")
        with left:
            st.markdown(f"**ID:** `{pid}`")
            new_name = st.text_input("Display name", value=person.name, key=f"rename_{pid}")
            if st.button("Save name", key=f"btn_rename_{pid}"):
                try:
                    enrollment_mgr.update_display_name(pid, new_name)
                    sess = get_session()
                    PersonRepository(sess).update_name(pid, new_name.strip())
                    sess.commit()
                    sess.close()
                    _notify_matcher()
                    st.success("Name updated.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

            st.info(
                "Profile stale — rebuild before Live Monitor can match this person"
                if stale else
                ("Ready for Live Monitor" if has_emb else "No face profile yet")
            )

            st.markdown("**ML package (on disk)**")
            pdir = enrollment_mgr.person_dir(pid)
            st.code(
                "\n".join([
                    f"{pdir.relative_to(cfg.project_root)}/",
                    f"  images/            {n_imgs} JPEG/PNG sample(s)",
                    f"  embedding.npy      {'yes' if has_emb else 'missing (stale)'}",
                    f"  embedding.json     {'yes' if enrollment_mgr.embedding_json_path(pid).exists() else 'no'}",
                    f"  samples.jsonl      {'yes' if enrollment_mgr.samples_path(pid).exists() else 'no'}",
                    f"  meta.json          {'yes' if enrollment_mgr.meta_path(pid).exists() else 'no'}",
                    f"index.json           gallery manifest for all people",
                ]),
                language="text",
            )

            if stale and n_imgs > 0:
                if st.button("Rebuild face profile", type="primary", key=f"rebuild_{pid}"):
                    with st.spinner("Rebuilding the ArcFace profile from the photos on file…"):
                        try:
                            emb = _rebuild_face_profile(pid, new_name.strip() or person.name)
                            if emb is None:
                                st.warning("No usable face in the remaining photos.")
                            else:
                                st.success("Profile rebuilt.")
                        except Exception as e:
                            st.error(str(e))
                    st.rerun()

        with right:
            st.markdown("**Remove this person**")
            st.caption(
                "Deletes their photos and embedding from disk and deactivates the "
                "database row. Past transactions keep the name. Type the ID to confirm."
            )
            confirm = st.text_input("Type the person ID", key=f"del_confirm_{pid}", placeholder=pid)
            if st.button("Delete person permanently", key=f"btn_del_person_{pid}"):
                if confirm.strip() != pid:
                    st.error("ID did not match — nothing was deleted.")
                else:
                    sess = get_session()
                    PersonRepository(sess).delete(pid)
                    sess.commit()
                    sess.close()
                    enrollment_mgr.delete_person(pid)
                    _notify_matcher()
                    st.success(f"Removed {pid}.")
                    time.sleep(0.4)
                    st.rerun()

        st.markdown("---")
        st.markdown(f"**Photos ({n_imgs})**")
        images = enrollment_mgr.list_images(pid)
        if not images:
            st.info("No photos on file. Run Face Enrollment to add some.")
        else:
            selected: list[str] = []
            cols = st.columns(4)
            for i, ip in enumerate(images):
                with cols[i % 4]:
                    try:
                        st.image(str(ip), caption=ip.name, width="stretch")
                    except Exception:
                        st.caption(ip.name)
                    if st.checkbox("Delete", key=f"rm_{pid}_{ip.name}"):
                        selected.append(ip.name)
            if st.button("Delete selected photos", key=f"btn_del_imgs_{pid}"):
                if not selected:
                    st.info("Tick one or more photos first.")
                else:
                    result = enrollment_mgr.delete_images(pid, selected)
                    sess = get_session()
                    repo = PersonRepository(sess)
                    repo.update_image_count(pid, enrollment_mgr.image_count(pid))
                    repo.clear_embedding(pid)
                    sess.commit()
                    sess.close()
                    _notify_matcher()
                    st.success(
                        f"Deleted {len(result['removed'])} photo(s). "
                        "Rebuild the face profile to keep matching this person."
                    )
                    st.rerun()

    idx_path = enrollment_mgr.gallery_index_path()
    if idx_path.exists():
        with st.expander("Gallery index (what a trainer would load)"):
            st.code(idx_path.read_text(encoding="utf-8")[:4000], language="json")


# =============================================================================
# TAB C — Waste Dataset
# =============================================================================
with tab_dataset:
    st.subheader("📦 Waste classification dataset")
    st.markdown(
        "Photos you add here are what the waste classifier learns from. "
        "**25+ per category** gives a usable model; 50+ is better."
    )

    counts = dm.waste_class_counts()
    total = sum(counts.values())

    st.markdown("#### 📸 Capture from the camera")
    cam_col, act_col = st.columns([2, 1], gap="large")

    with cam_col:
        if _ENGINE_ALIVE:
            st.caption("Live frame from the engine — place a tray in view and grab it.")

            @st.fragment(run_every=0.5)
            def _dataset_live_frame() -> None:
                write_heartbeat()
                raw = _engine_frame()
                if raw:
                    st.session_state["dataset_frame"] = raw
                    st.image(raw, width="stretch")
                else:
                    st.info("Waiting for a frame from the engine…")

            _dataset_live_frame()
        else:
            st.caption(
                "The engine is stopped, so this uses your browser camera. "
                "Start the engine if you would rather grab frames from the tray camera."
            )
            shot = st.camera_input("Take a photo", key="dataset_browser_cam")
            if shot is not None:
                st.session_state["dataset_frame"] = shot.getvalue()
            if st.button("▶ Start engine and use the tray camera", key="btn_ds_start_engine"):
                start_engine()
                st.rerun()

    with act_col:
        st.markdown("**File it under**")
        frame = st.session_state.get("dataset_frame")
        if not frame:
            st.info("Grab or take a photo first.")
        for cls in WASTE_CLASSES:
            if st.button(f"➕ {cls}  ({counts.get(cls, 0)})", key=f"btn_cap_{cls}",
                         width="stretch", disabled=not frame):
                added = _add_images_to_waste_class(cls, [frame])
                if added:
                    st.success(f"Added 1 photo to {cls}.")
                else:
                    st.warning("That frame was a duplicate of one already on file.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 📁 Or upload a batch")
    col_cls, col_up = st.columns([1, 2])
    with col_cls:
        selected_class = st.selectbox("Category", WASTE_CLASSES, key="ds_class")
    with col_up:
        upload_files = st.file_uploader(
            f"Upload **{selected_class}** images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="ds_upload",
        )

    if upload_files and st.button("➕ Add to dataset", key="btn_add_ds", type="primary"):
        added = _add_images_to_waste_class(selected_class, [f.getvalue() for f in upload_files])
        st.success(f"Added **{added}** image(s) to {selected_class}.")
        st.rerun()

    st.markdown("---")
    counts = dm.waste_class_counts()
    total  = sum(counts.values())

    if total == 0:
        st.markdown(
            empty_state_html(
                icon="📦",
                title="No waste images yet",
                body=(
                    "The classifier trains only on photos you provide — nothing is "
                    "fabricated, so training stays blocked at 0 images."
                ),
                steps=[
                    "Grab a frame above and file it under EMPTY / LOW_WASTE / MEDIUM_WASTE / HIGH_WASTE.",
                    "Four images is the technical minimum; 25+ per category is the useful target.",
                ],
            ),
            unsafe_allow_html=True,
        )
    else:
        import pandas as pd
        st.bar_chart(
            pd.DataFrame({"Category": list(counts.keys()), "Images": list(counts.values())})
            .set_index("Category")
        )
        mc = st.columns(len(WASTE_CLASSES) + 1)
        for i, cls in enumerate(WASTE_CLASSES):
            mc[i].metric(cls, counts[cls])
        mc[-1].metric("Total", total)

        n_train = int(total * cfg.training.train_ratio)
        n_val   = int(total * cfg.training.val_ratio)
        c1, c2, c3 = st.columns(3)
        c1.metric("Train",      n_train, f"{cfg.training.train_ratio * 100:.0f}%")
        c2.metric("Validation", n_val,   f"{cfg.training.val_ratio * 100:.0f}%")
        c3.metric("Test",       total - n_train - n_val,
                  f"{cfg.training.test_ratio * 100:.0f}%")

    st.markdown("---")
    st.markdown("#### 🖼️ Review images")
    preview_class  = st.selectbox("Category to review", WASTE_CLASSES, key="preview_class")
    preview_images = dm.waste_images(preview_class)
    if not preview_images:
        st.info(f"No images for **{preview_class}** yet.")
    else:
        st.markdown(f"**{len(preview_images)} image(s)**")
        page_size = 12
        n_pages   = max(1, (len(preview_images) + page_size - 1) // page_size)
        page = (st.slider("Page", 1, n_pages, 1, key="preview_page") - 1) if n_pages > 1 else 0
        cols = st.columns(4)
        for i, ip in enumerate(preview_images[page * page_size:(page + 1) * page_size]):
            with cols[i % 4]:
                st.image(str(ip), width="stretch", caption=ip.name)
                if st.button("🗑️ Remove", key=f"del_ds_{preview_class}_{ip.name}"):
                    dm.delete_waste_image(preview_class, ip.name)
                    st.rerun()


# =============================================================================
# TAB D — Plate Dataset (capture + assisted labelling)
# =============================================================================
with tab_plate:
    st.subheader("🍽️ Plate detection dataset")
    plate_stats = dm.plate_dataset_stats()
    m1, m2, m3 = st.columns(3)
    m1.metric("Images", plate_stats["images"])
    m2.metric("Labelled", plate_stats["labels"])
    m3.metric("Awaiting labels", plate_stats["unlabelled"])

    st.caption(
        "A YOLO plate detector needs a box per plate. Capture photos here, then label "
        "them below — the built-in OpenCV detector proposes a box and you accept or "
        "adjust it. Nothing is written until you accept, so labels stay honest."
    )

    st.markdown("#### 📸 Add photos")
    pc_left, pc_right = st.columns([2, 1], gap="large")
    with pc_left:
        if _ENGINE_ALIVE:
            @st.fragment(run_every=0.5)
            def _plate_live_frame() -> None:
                write_heartbeat()
                raw = _engine_frame()
                if raw:
                    st.session_state["plate_frame"] = raw
                    st.image(raw, width="stretch")
                else:
                    st.info("Waiting for a frame from the engine…")

            _plate_live_frame()
        else:
            shot = st.camera_input("Take a plate photo", key="plate_browser_cam")
            if shot is not None:
                st.session_state["plate_frame"] = shot.getvalue()

    with pc_right:
        plate_frame = st.session_state.get("plate_frame")
        if st.button("➕ Add this frame", key="btn_add_plate_frame", width="stretch",
                     disabled=not plate_frame):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir) / f"plate_{int(time.time() * 1000)}.jpg"
                tmp.write_bytes(plate_frame)
                dm.add_plate_image(tmp)
            st.success("Added. Label it below.")
            st.rerun()

        plate_uploads = st.file_uploader(
            "Or upload plate photos",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="plate_upload",
        )
        if plate_uploads and st.button("➕ Add uploads", key="btn_add_plate_uploads",
                                       width="stretch"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, uf in enumerate(plate_uploads):
                    tmp = Path(tmpdir) / f"upload_{i}_{uf.name}"
                    tmp.write_bytes(uf.getvalue())
                    dm.add_plate_image(tmp)
            st.success(f"Added {len(plate_uploads)} image(s).")
            st.rerun()

    st.markdown("---")
    st.markdown("#### ✏️ Label the queue")
    unlabelled = dm.plate_unlabelled_images()
    if not unlabelled:
        if plate_stats["images"]:
            st.success("Every plate image is labelled — plate training is unblocked.")
        else:
            st.info("Add some plate photos above to start labelling.")
    else:
        target = unlabelled[0]
        st.markdown(f"**{len(unlabelled)} image(s) to go** — currently `{target.name}`")

        image = cv2.imread(str(target))
        if image is None:
            st.error(f"`{target.name}` could not be read.")
            if st.button("Remove this file", key="btn_drop_bad_plate"):
                dm.delete_plate_image(target.name)
                st.rerun()
        else:
            h, w = image.shape[:2]
            proposals = detect_plates_visual(
                image,
                roi={"x1": cfg.roi.plate.x1, "y1": cfg.roi.plate.y1,
                     "x2": cfg.roi.plate.x2, "y2": cfg.roi.plate.y2},
                min_confidence=0.30,
            )
            preview = image.copy()
            for det in proposals:
                cv2.rectangle(preview, (det.x1, det.y1), (det.x2, det.y2), (52, 211, 153), 3)

            lbl_left, lbl_right = st.columns([2, 1], gap="large")
            with lbl_left:
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    width="stretch",
                    caption=f"{len(proposals)} proposed box(es)",
                )
            with lbl_right:
                if proposals:
                    st.markdown("**Proposed boxes**")
                    for i, det in enumerate(proposals, start=1):
                        st.caption(
                            f"{i}. {det.width}×{det.height} px  ·  conf {det.confidence:.2f}"
                        )
                    if st.button("✅ Accept boxes", type="primary", key="btn_accept_boxes",
                                 width="stretch"):
                        boxes = [
                            (
                                ((d.x1 + d.x2) / 2) / w,
                                ((d.y1 + d.y2) / 2) / h,
                                d.width / w,
                                d.height / h,
                            )
                            for d in proposals
                        ]
                        dm.write_plate_label(target.stem, boxes)
                        st.rerun()
                else:
                    st.warning("No plate found by the visual detector.")

                st.markdown("**Whole frame is the plate**")
                if st.button("🔲 Label full frame", key="btn_label_full", width="stretch"):
                    dm.write_plate_label(target.stem, [(0.5, 0.5, 0.98, 0.98)])
                    st.rerun()

                if st.button("⏭️ Skip for now", key="btn_skip_plate", width="stretch"):
                    st.session_state["plate_skip"] = st.session_state.get("plate_skip", 0) + 1
                    st.info("Skipped images stay in the queue until labelled.")

                if st.button("🗑️ Delete image", key="btn_del_plate", width="stretch"):
                    dm.delete_plate_image(target.name)
                    st.rerun()


# =============================================================================
# TAB E — Train Model
# =============================================================================
with tab_train:
    st.subheader("🚀 Model training")

    task_options = {"Waste classification": "waste", "Plate detection": "plate"}
    task_label   = st.selectbox("Training task", list(task_options.keys()), key="train_task")
    task         = task_options[task_label]

    cc1, cc2 = st.columns(2)
    with cc1:
        base_model = st.text_input(
            "Base model",
            value=(cfg.training.default_base_model if task == "waste"
                   else cfg.training.plate_base_model),
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=500,
                                 value=cfg.training.default_epochs)
    with cc2:
        img_size = st.number_input("Image size", min_value=32, max_value=1280,
                                   value=224 if task == "waste" else 640, step=32)
        batch    = st.number_input("Batch size", min_value=1, max_value=512,
                                   value=cfg.training.default_batch)

    device_options = {
        "Auto":         cfg.device,
        "CPU":          "cpu",
        "GPU (cuda:0)": "cuda:0",
        "Apple MPS":    "mps",
    }
    device = device_options[st.selectbox("Device", list(device_options.keys()), key="train_device")]

    st.markdown("---")

    counts      = dm.waste_class_counts()
    total_ds    = sum(counts.values())
    plate_stats = dm.plate_dataset_stats()
    status      = read_status(TRAINING_STATUS_FILE)
    running     = is_running(status)

    can_train = True
    if task == "waste":
        if total_ds < 4:
            can_train = False
            st.warning(
                f"Waste training needs at least 4 labelled images — there are **{total_ds}**. "
                "Grab a few frames per category in the Waste Dataset tab; the classifier will "
                "not invent classes or use placeholder data.",
                icon="📦",
            )
        else:
            thin = [c for c, n in counts.items() if 0 < n < 25]
            st.info(
                f"Ready to train on **{total_ds}** image(s)."
                + (f" Below the recommended 25 per category: {', '.join(thin)}." if thin else ""),
                icon="✅",
            )
    else:
        st.markdown(
            f"**Plate dataset:** {plate_stats['images']} image(s), "
            f"{plate_stats['labels']} label(s), {plate_stats['unlabelled']} awaiting labels"
        )
        if plate_stats["images"] == 0 or plate_stats["labels"] == 0:
            can_train = False
            st.warning(
                "Plate training needs labelled images. Capture photos in the **Plate Dataset** "
                "tab and accept the proposed boxes — that writes real YOLO labels. A generic "
                "COCO model is never passed off as a plate detector.",
                icon="🍽️",
            )
        else:
            st.info(f"Ready to train on **{plate_stats['labels']}** labelled image(s).", icon="✅")

    def _launch_training() -> None:
        version = registry.next_version(task)
        write_status(
            TRAINING_STATUS_FILE,
            state="running", task=task, version=version,
            epoch=0, epochs=int(epochs), history=[],
            message="Preparing the dataset split…", started_at=time.time(),
        )

        def _on_epoch(payload: dict) -> None:
            current = read_status(TRAINING_STATUS_FILE)
            history = list(current.get("history") or [])
            entry = {"epoch": payload.get("epoch")}
            if payload.get("loss") is not None:
                entry["loss"] = payload["loss"]
            for key, value in (payload.get("metrics") or {}).items():
                if "accuracy_top1" in key:
                    entry["top1"] = value
                elif key.endswith("mAP50(B)") or "mAP50" in key:
                    entry.setdefault("map50", value)
            if entry.get("epoch") and (not history or history[-1].get("epoch") != entry["epoch"]):
                history.append(entry)
            elif history:
                history[-1].update(entry)
            update_status(
                TRAINING_STATUS_FILE,
                state="running",
                epoch=payload.get("epoch", current.get("epoch")),
                epochs=payload.get("epochs") or current.get("epochs"),
                history=history[-300:],
                message=f"Epoch {payload.get('epoch', '?')} of {payload.get('epochs', epochs)}",
            )

        def _run() -> None:
            try:
                if task == "waste":
                    from cafeteria.training.trainer import WasteModelTrainer
                    result = WasteModelTrainer(
                        datasets_dir=cfg.project_root / cfg.storage.datasets,
                        models_output_dir=cfg.project_root / cfg.storage.models,
                        device=device,
                    ).train(
                        base_model=base_model, epochs=int(epochs),
                        image_size=int(img_size), batch=int(batch), version=version,
                        progress_callback=_on_epoch,
                    )
                else:
                    from cafeteria.training.trainer import PlateModelTrainer
                    with tempfile.TemporaryDirectory(prefix="plate_split_") as tmp:
                        data_yaml = dm.prepare_yolo_det_dataset(Path(tmp))
                        result = PlateModelTrainer(
                            datasets_dir=cfg.project_root / cfg.storage.datasets,
                            models_output_dir=cfg.project_root / cfg.storage.models,
                            device=device,
                        ).train(
                            base_model=base_model, epochs=int(epochs),
                            image_size=int(img_size), batch=int(batch),
                            version=version, data_yaml=data_yaml,
                            progress_callback=_on_epoch,
                        )

                registry.register_version(
                    task=result.task, version=result.version,
                    weights_path=str(result.weights_path),
                    metrics=result.metrics, config=result.config,
                    dataset_stats=result.dataset_stats,
                )
                current = read_status(TRAINING_STATUS_FILE)
                write_status(
                    TRAINING_STATUS_FILE,
                    state="done", task=result.task, version=result.version,
                    weights_path=str(result.weights_path), metrics=result.metrics,
                    epochs=int(epochs), epoch=int(epochs),
                    history=current.get("history") or [],
                    message="Training finished.", finished_at=time.time(),
                )
            except Exception as exc:
                write_status(
                    TRAINING_STATUS_FILE,
                    state="error", task=task, version=version,
                    message=str(exc), finished_at=time.time(),
                )

        threading.Thread(target=_run, daemon=True, name="model-training").start()

    if st.button("🚀 Start training", key="btn_start_train", type="primary",
                 disabled=running or not can_train):
        _launch_training()
        st.rerun()

    st.markdown("---")

    @st.fragment(run_every=2.0)
    def _training_monitor() -> None:
        # [AI-CoLab: Verified by Antigravity] Fragment-based background training progress chart and status monitor
        status = read_status(TRAINING_STATUS_FILE)
        if not status:
            st.caption("No training run yet. Configure the options above and start one.")
            return

        state = status.get("state")
        version = status.get("version", "—")
        task_name = status.get("task", "—")

        if is_running(status):
            epoch = int(status.get("epoch") or 0)
            total_epochs = int(status.get("epochs") or 0) or 1
            st.markdown(
                f'<div class="glass-card"><div class="card-title">'
                f'Training {task_name} · {version}</div>'
                f'<div class="card-body">{status.get("message", "")}</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(
                min(max(epoch / total_epochs, 0.02), 1.0),
                text=f"Epoch {epoch} / {total_epochs}",
            )
            history = status.get("history") or []
            if len(history) > 1:
                import pandas as pd
                frame = pd.DataFrame(history).set_index("epoch")
                numeric = frame.select_dtypes("number")
                if not numeric.empty:
                    st.line_chart(numeric)
            st.caption("Leave this page if you like — the run keeps going and reports back here.")
            return

        if state == "done":
            st.success(f"✅ {task_name} model **{version}** finished training.")
            for key, value in (status.get("metrics") or {}).items():
                if isinstance(value, (int, float)):
                    st.markdown(f"- {key}: `{value}`")
            history = status.get("history") or []
            if len(history) > 1:
                import pandas as pd
                numeric = pd.DataFrame(history).set_index("epoch").select_dtypes("number")
                if not numeric.empty:
                    st.line_chart(numeric)
            act, dismiss = st.columns(2)
            with act:
                already_active = registry.active_version_string(task_name) == version
                if already_active:
                    st.info(f"{version} is already the active {task_name} model.")
                elif st.button("⚡ Activate this model", type="primary", key="btn_activate_done"):
                    registry.activate(task_name, version)
                    _engine_command({
                        "type": "activate_model", "task": task_name, "version": version,
                    })
                    st.success(f"{version} is active — a running engine hot-swaps within ~5 s.")
                    clear_status(TRAINING_STATUS_FILE)
                    st.rerun(scope="app")
            with dismiss:
                if st.button("Clear result", key="btn_clear_done"):
                    clear_status(TRAINING_STATUS_FILE)
                    st.rerun(scope="app")
            return

        if state == "error":
            st.error(f"Training failed: {status.get('message', 'unknown error')}")
            if st.button("Clear error", key="btn_clear_error"):
                clear_status(TRAINING_STATUS_FILE)
                st.rerun(scope="app")
            return

        st.caption("No training run in flight.")

    _training_monitor()


# =============================================================================
# TAB F — Model Versions
# =============================================================================
with tab_versions:
    import datetime
    st.subheader("📋 Model version history")
    for tname, tkey in [("Waste model", "waste"), ("Plate model", "plate")]:
        st.markdown(f"### {tname}")
        versions   = registry.get_all_versions(tkey)
        active_ver = registry.active_version_string(tkey)
        if not versions:
            st.markdown(
                empty_state_html(
                    icon="📋",
                    title=f"No {tname.lower()} versions",
                    body=(
                        "The registry is empty for this task. That is expected until "
                        "you train a real model — nothing is pre-loaded or faked."
                    ),
                    steps=[
                        "Collect a dataset in the Waste Dataset or Plate Dataset tab.",
                        "Train from the Train Model tab, or run scripts/quickstart.py.",
                    ],
                ),
                unsafe_allow_html=True,
            )
        else:
            for v in versions:
                is_active = v["version"] == active_ver
                badge     = "🟢 **ACTIVE**" if is_active else "⚪ archived"
                with st.expander(f"{v['version']}  {badge}", expanded=is_active):
                    dt = datetime.datetime.fromtimestamp(
                        v.get("trained_at", 0)
                    ).strftime("%Y-%m-%d %H:%M")
                    st.markdown(
                        f"**Trained:** {dt}  |  **Weights:** `{v.get('weights_path', '—')}`"
                    )
                    for mk, mv in v.get("metrics", {}).items():
                        if isinstance(mv, (int, float)):
                            st.markdown(f"  - {mk}: `{mv}`")
                    if not is_active:
                        if st.button(f"⚡ Activate {v['version']}",
                                     key=f"act_{tkey}_{v['version']}"):
                            registry.activate(tkey, v["version"])
                            _engine_command({
                                "type": "activate_model", "task": tkey, "version": v["version"],
                            })
                            st.success(f"Activated {v['version']}.")
                            st.rerun()
        st.markdown("---")
