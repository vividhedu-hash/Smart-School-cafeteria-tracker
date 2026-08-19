"""
Page 2 — Training & Dataset Management

Tabs:
  A. Face Enrollment  — enterprise guided wizard (MediaPipe FaceMesh component)
  B. Waste Dataset    — upload + preview
  C. Train Model      — configure + launch
  D. Model Versions   — history + activation
"""
from __future__ import annotations

import base64
import json
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
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(
    page_title="Training & Enrollment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from auth_gate import require_login
require_login()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.wizard-steps { display:flex; align-items:center; gap:0; margin:1.5rem 0; }
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
    background:linear-gradient(135deg,#1e293b,#0f172a); border:1px solid #334155;
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
from cafeteria.training.dataset_manager import DatasetManager, WASTE_CLASSES
from cafeteria.training.registry import ModelRegistry
from cafeteria.monitoring.metrics import write_command
from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.storage.database import init_db, get_session
from cafeteria.storage.repositories import PersonRepository
from cafeteria.utils.image_quality import laplacian_variance, brightness_ok
from engine_ctl import engine_is_alive, stop_engine
from empty_states import empty_state_html

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

init_db(cfg.project_root / cfg.storage.database)
enrollment_mgr = EnrollmentManager(
    enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
    audit_path=cfg.project_root / "data" / "audit" / "roster.jsonl",
)

if engine_is_alive():
    stop_engine()
    st.info(
        "Released the live camera so this page can use the webcam for enrollment. "
        "Start the engine again from Live Monitor when you want recognition."
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


def _engine_command(cmd: dict) -> None:
    posted = False
    try:
        from engine_client import post_command
        posted = post_command(cmd)
    except Exception:
        posted = False
    if not posted:
        write_command(
            cfg.project_root / cfg.application.commands_path,
            cmd,
        )


def _notify_matcher() -> None:
    _engine_command({"type": "reload_embeddings"})


def _rebuild_face_profile(pid: str, pname: str):
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
    emgr = _construct(
        EnrollmentManager,
        enrollment_dir=cfg.project_root / cfg.recognition.embedding_dir,
        face_engine=fe,
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

# ── Server-side face cascade (quality back-check) ─────────────────────────────
@st.cache_resource
def _load_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

face_cascade = _load_cascade()

def _has_face(bgr: np.ndarray) -> bool:
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return len(faces) > 0

# ── Enterprise face-capture component ─────────────────────────────────────────
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dashboard.enroll_cam import start_enroll_camera, stop_enroll_camera, TARGET_FRAMES


# ── Pose configuration ────────────────────────────────────────────────────────
POSE_CONFIGS = {
    "front": {"label": "FACE FORWARD", "icon": "😊", "hint": "Look straight at the camera"},
    "left":  {"label": "TURN LEFT",    "icon": "👈", "hint": "Turn your head left ~30°"},
    "right": {"label": "TURN RIGHT",   "icon": "👉", "hint": "Turn your head right ~30°"},
    "up":    {"label": "LOOK UP",      "icon": "👆", "hint": "Tilt head slightly upward"},
    "down":  {"label": "LOOK DOWN",    "icon": "👇", "hint": "Tilt head slightly downward"},
}
POSE_SEQUENCE = ["front", "left", "right", "up", "down"]
PHOTOS_PER_POSE = 1


def _b64_to_bgr(raw: str):
    if not raw:
        return None
    if "," in raw and raw.strip().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ── Wizard step HTML ──────────────────────────────────────────────────────────
def _wizard_html(current_step: str) -> str:
    steps = [("person_select", "1", "Person"), ("capture", "2", "Capture"), ("embed", "3", "Finalize")]
    order = [s[0] for s in steps]
    cur   = order.index(current_step) if current_step in order else 0
    html  = '<div class="wizard-steps">'
    for i, (key, num, label) in enumerate(steps):
        if i < cur:   cc, cl, nt = "done",   "done",   "✓"
        elif i == cur: cc, cl, nt = "active", "active", num
        else:          cc, cl, nt = "",       "",       num
        html += (f'<div class="wizard-step">'
                 f'<div class="wizard-step-circle {cc}">{nt}</div>'
                 f'<div class="wizard-step-label {cl}">{label}</div></div>')
        if i < len(steps) - 1:
            lc = "done" if i < cur else ""
            html += f'<div class="wizard-step-line {lc}"></div>'
    return html + "</div>"


# =============================================================================
# TABS
# =============================================================================
tab_enroll, tab_people, tab_dataset, tab_train, tab_versions = st.tabs([
    "👤 Face Enrollment",
    "🗂️ People",
    "📦 Waste Dataset",
    "🚀 Train Model",
    "📋 Model Versions",
])

# =============================================================================
# TAB A — FACE ENROLLMENT
# =============================================================================
with tab_enroll:

    # ── Session state initialisation ──────────────────────────────────────────
    for key, val in [
        ("enroll_step",     "person_select"),
        ("enroll_pid",      ""),
        ("enroll_name",     ""),
        ("enroll_pose_idx", 0),
        ("enroll_snaps",    {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Person Select / Create
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.enroll_step == "person_select":
        st.subheader("Step 1 — who are we scanning?")
        st.markdown(_wizard_html("person_select"), unsafe_allow_html=True)
        st.caption(
            "Pick someone, then look into the oval. Enrollment uses the same "
            "webcam as Live Monitor (not Chrome). Walk-past recognition is InsightFace."
        )

        col_new, col_existing = st.columns(2)

        with col_new:
            st.markdown("#### ➕ New Person")
            new_pid  = st.text_input("Person ID",    placeholder="person_01", key="new_pid_input")
            new_name = st.text_input("Display Name", placeholder="Alice Smith", key="new_name_input")
            if st.button("Start scan →", type="primary", key="btn_create_start"):
                if not new_pid.strip() or not new_name.strip():
                    st.info("Add an ID and a display name so we know who this profile belongs to.")
                else:
                    enrollment_mgr.ensure_person_dir(new_pid.strip())
                    enrollment_mgr.update_display_name(new_pid.strip(), new_name.strip())
                    try:
                        sess = get_session()
                        repo = PersonRepository(sess)
                        if not repo.get_by_person_id(new_pid.strip()):
                            repo.create(person_id=new_pid.strip(), name=new_name.strip())
                            sess.commit()
                        sess.close()
                        st.session_state.enroll_pid      = new_pid.strip()
                        st.session_state.enroll_name     = new_name.strip()
                        st.session_state.enroll_pose_idx = 0
                        st.session_state.enroll_snaps    = {}
                        st.session_state.enroll_batch_saved = False
                        st.session_state.enroll_embed_status = None
                        st.session_state.enroll_step     = "capture"
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with col_existing:
            st.markdown("#### 🔄 Existing Person")
            sess        = get_session()
            all_persons = PersonRepository(sess).get_all_active()
            sess.close()
            if not all_persons:
                st.info("No persons yet — create one on the left.")
            else:
                pmap      = {f"{p.person_id} — {p.name}": p for p in all_persons}
                sel_label = st.selectbox("Select Person", list(pmap.keys()), key="exist_sel")
                sel_p     = pmap[sel_label]
                n_imgs    = enrollment_mgr.image_count(sel_p.person_id)
                has_emb   = enrollment_mgr.has_embedding(sel_p.person_id)
                badge     = ('<span class="badge-ready">✅ READY</span>'
                             if has_emb else
                             '<span class="badge-pending">⏳ PENDING</span>')
                st.markdown(f"Images: **{n_imgs}** &nbsp; {badge}", unsafe_allow_html=True)
                if st.button("Scan face →", key="btn_add_imgs"):
                    st.session_state.enroll_pid      = sel_p.person_id
                    st.session_state.enroll_name     = sel_p.name
                    st.session_state.enroll_pose_idx = 0
                    st.session_state.enroll_snaps    = {}
                    st.session_state.enroll_batch_saved = False
                    st.session_state.enroll_embed_status = None
                    st.session_state.enroll_step     = "capture"
                    st.rerun()

        # ── Roster ────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Enrolled People")
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
                bc    = "badge-ready"   if has_e else "badge-pending"
                bt    = "✅ READY"      if has_e else "⏳ PENDING"
                st.markdown(f"""
                <div class="person-card">
                    <div style="font-size:2rem">👤</div>
                    <div style="flex:1">
                        <div class="person-card-name">{p.name}</div>
                        <div class="person-card-meta">ID: {p.person_id} &nbsp;·&nbsp; {n} images</div>
                    </div>
                    <span class="{bc}">{bt}</span>
                </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — OpenCV webcam capture (same camera as Live Monitor)
    # ─────────────────────────────────────────────────────────────────────────
    elif st.session_state.enroll_step == "capture":
        pid   = st.session_state.enroll_pid
        pname = st.session_state.enroll_name

        st.subheader(f"Step 2 — scan **{pname}**")
        st.markdown(_wizard_html("capture"), unsafe_allow_html=True)
        st.caption(
            "This uses the **same webcam as Live Monitor** (Python/OpenCV), not Chrome’s camera. "
            "Fit your face in the oval — we snap automatically in a few seconds. "
            "Walk-past identity on Live Monitor is still InsightFace ArcFace."
        )

        cam_src = cfg.camera.source if isinstance(cfg.camera.source, int) else 0
        session_key = f"enroll_{pid}"

        @st.fragment(run_every=0.15)
        def _enroll_preview():
            cam = start_enroll_camera(session_key, source=int(cam_src))
            jpeg, hint, n_caps, cam_done, cam_err = cam.snapshot()
            if cam_err:
                st.warning(cam_err)
                return
            st.markdown(f"**{hint}**")
            if jpeg:
                st.image(jpeg, use_container_width=False, width=420)
            else:
                st.info("Opening the webcam…")
            st.progress(min(n_caps / TARGET_FRAMES, 1.0), text=f"{n_caps} / {TARGET_FRAMES}")
            if cam_done and not st.session_state.get("enroll_batch_saved"):
                saved = 0
                for img in cam.take_captures():
                    enrollment_mgr.save_image(pid, img, pose="front")
                    st.session_state.enroll_snaps.setdefault("front", []).append(img.copy())
                    saved += 1
                stop_enroll_camera(session_key)
                st.session_state.enroll_batch_saved = True
                if saved >= 1:
                    st.session_state.enroll_embed_status = "pending"
                    st.session_state.enroll_step = "embed"
                    st.rerun()

        _enroll_preview()

        st.markdown("---")
        st.markdown("### Camera blocked? Upload photos instead")
        st.caption("Or drop 3+ face photos here. Front-facing, decent light is enough.")
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
                    data = np.frombuffer(uf.getvalue(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    pose_key = poses[i % len(poses)]
                    enrollment_mgr.save_image(pid, img, pose=pose_key)
                    st.session_state.enroll_snaps.setdefault(pose_key, []).append(img.copy())
                    saved += 1
                if saved < 1:
                    st.error("Those files could not be read as images.")
                else:
                    st.session_state.enroll_batch_saved = True
                    st.session_state.enroll_embed_status = "pending"
                    st.session_state.enroll_step = "embed"
                    st.rerun()

        if st.button("← Back to people", key="back_to_select"):
            stop_enroll_camera(f"enroll_{pid}")
            st.session_state.enroll_step = "person_select"
            st.rerun()

    elif st.session_state.enroll_step == "embed":
        pid    = st.session_state.enroll_pid
        pname  = st.session_state.enroll_name
        n_imgs = enrollment_mgr.image_count(pid)
        total_snaps = sum(len(v) for v in st.session_state.enroll_snaps.values())

        st.subheader(f"Enrolling **{pname}**")
        st.markdown(_wizard_html("embed"), unsafe_allow_html=True)
        st.caption("Turning those stills into an ArcFace profile — a few seconds.")

        def _run_embed():
            return _rebuild_face_profile(pid, pname)

        status = st.session_state.get("enroll_embed_status")
        if status == "pending":
            with st.spinner("Building your face profile… hang tight."):
                try:
                    emb = _run_embed()
                    st.session_state.enroll_embed_status = "ok" if emb is not None else "empty"
                    st.session_state.enroll_embed_shape = None if emb is None else tuple(emb.shape)
                except Exception as e:
                    st.session_state.enroll_embed_status = "error"
                    st.session_state.enroll_embed_error = str(e)
            st.rerun()

        status = st.session_state.get("enroll_embed_status")
        if status == "ok":
            st.success(f"{pname} is enrolled and ready for Live Monitor.")
        elif status == "empty":
            st.info("We couldn’t read a face in those stills. Try the scan again in a bit more light.")
        elif status == "error":
            st.info(f"Profile step hit a snag: {st.session_state.get('enroll_embed_error', '')}")
            if st.button("Try finishing the profile", type="primary", key="btn_gen_embed"):
                st.session_state.enroll_embed_status = "pending"
                st.rerun()
        else:
            if st.button("Finish profile", type="primary", key="btn_gen_embed"):
                st.session_state.enroll_embed_status = "pending"
                st.rerun()

        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            if st.button("Scan someone else", type="secondary", key="btn_more"):
                for k, v in [("enroll_step", "person_select"), ("enroll_pid", ""),
                              ("enroll_name", ""), ("enroll_pose_idx", 0),
                              ("enroll_snaps", {}), ("enroll_batch_saved", False),
                              ("enroll_embed_status", None)]:
                    st.session_state[k] = v
                st.rerun()
        with cb:
            if st.button("Back to people", key="btn_roster"):
                st.session_state.enroll_step = "person_select"
                st.rerun()

        images = enrollment_mgr.list_images(pid)
        if images:
            st.markdown(f"Photos on file ({len(images)}). Edit or delete them in the **People** tab.")
            cols = st.columns(min(len(images), 8))
            for i, ip in enumerate(images[:16]):
                pil = Image.open(ip)
                pil.thumbnail((80, 80))
                with cols[i % 8]:
                    st.image(pil, caption=ip.name, use_container_width=False)



# =============================================================================
# TAB — People roster (edit / delete)
# =============================================================================
with tab_people:
    st.subheader("People roster")
    st.caption(
        "Edit a display name, remove individual photos, or delete a person. "
        "Photos live under `data/enrollment/{id}/images/` — that folder is the "
        "training set the face model reads. Removing photos clears the live "
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

            status_label = (
                "Profile stale — rebuild before Live Monitor can match this person"
                if stale else
                ("Ready for Live Monitor" if has_emb else "No face profile yet")
            )
            st.info(status_label)

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
                    with st.spinner("Rebuilding ArcFace profile from remaining photos…"):
                        try:
                            emb = _rebuild_face_profile(pid, new_name.strip() or person.name)
                            if emb is None:
                                st.info("No usable face in the remaining photos.")
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
                        st.image(str(ip), caption=ip.name, use_container_width=True)
                    except Exception:
                        st.caption(ip.name)
                    if st.checkbox("Delete", key=f"rm_{pid}_{ip.name}"):
                        selected.append(ip.name)
            if st.button("Delete selected photos", type="secondary", key=f"btn_del_imgs_{pid}"):
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
# TAB B — Waste Dataset
# =============================================================================
with tab_dataset:
    st.subheader("📦 Waste Classification Dataset")
    st.markdown(
        "Upload plate / food photos to train the waste classifier. "
        "Aim for **50+ images per category** for good accuracy."
    )

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

    if upload_files and st.button("➕ Add to Dataset", key="btn_add_ds", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_paths = []
            for f in upload_files:
                tp = Path(tmpdir) / f.name
                tp.write_bytes(f.read())
                tmp_paths.append(tp)
            added, skipped = dm.add_waste_images(selected_class, tmp_paths)
            st.success(f"✅ Added **{added}** images (skipped {skipped} duplicates)")
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
                    "The waste classifier trains on photos you upload here. "
                    "Nothing is fabricated — with 0 images, training is blocked."
                ),
                steps=[
                    "Pick a category (EMPTY, LOW_WASTE, MEDIUM_WASTE, HIGH_WASTE).",
                    "Upload real plate photos, then click Add to Dataset.",
                    "Aim for 50+ images per category before training.",
                ],
            ),
            unsafe_allow_html=True,
        )
    else:
        import pandas as pd
        df = pd.DataFrame({"Category": list(counts.keys()), "Images": list(counts.values())})
        st.bar_chart(df.set_index("Category"))
        mc = st.columns(len(WASTE_CLASSES) + 1)
        for i, cls in enumerate(WASTE_CLASSES):
            mc[i].metric(cls, counts[cls])
        mc[-1].metric("Total", total)

        st.markdown("### Dataset Split")
        n_train = int(total * cfg.training.train_ratio)
        n_val   = int(total * cfg.training.val_ratio)
        n_test  = total - n_train - n_val
        c1, c2, c3 = st.columns(3)
        c1.metric("Train",      n_train, f"{cfg.training.train_ratio * 100:.0f}%")
        c2.metric("Validation", n_val,   f"{cfg.training.val_ratio * 100:.0f}%")
        c3.metric("Test",       n_test,  f"{cfg.training.test_ratio * 100:.0f}%")

    st.markdown("---")
    st.markdown("### Image Preview")
    preview_class  = st.selectbox("Preview Class", WASTE_CLASSES, key="preview_class")
    preview_images = dm.waste_images(preview_class)
    if not preview_images:
        st.info(f"No images for **{preview_class}** yet.")
    else:
        st.markdown(f"**{len(preview_images)} images**")
        page_size = 12
        n_pages   = max(1, (len(preview_images) + page_size - 1) // page_size)
        page = st.slider("Page", 1, n_pages, 1, key="preview_page") - 1
        cols = st.columns(4)
        for i, ip in enumerate(preview_images[page * page_size:(page + 1) * page_size]):
            with cols[i % 4]:
                st.image(str(ip), width=150, caption=ip.name)
                if st.button("🗑️", key=f"del_ds_{preview_class}_{ip.name}"):
                    dm.delete_waste_image(preview_class, ip.name)
                    st.rerun()


# =============================================================================
# TAB C — Train Model
# =============================================================================
with tab_train:
    st.subheader("🚀 Model Training")

    task_options = {"Waste Classification": "waste", "Plate Detection": "plate"}
    task_label   = st.selectbox("Training Task", list(task_options.keys()), key="train_task")
    task         = task_options[task_label]

    cc1, cc2 = st.columns(2)
    with cc1:
        base_model = st.text_input(
            "Base Model",
            value=(cfg.training.default_base_model if task == "waste"
                   else cfg.training.plate_base_model),
        )
        epochs = st.number_input("Epochs", min_value=1, max_value=500,
                                 value=cfg.training.default_epochs)
    with cc2:
        img_size = st.number_input("Image Size", min_value=32, max_value=1280,
                                   value=224 if task == "waste" else 640, step=32)
        batch    = st.number_input("Batch Size", min_value=1, max_value=512,
                                   value=cfg.training.default_batch)

    device_options = {
        "Auto":          cfg.device,
        "CPU":           "cpu",
        "GPU (cuda:0)":  "cuda:0",
        "Apple MPS":     "mps",
    }
    device = device_options[st.selectbox("Device", list(device_options.keys()), key="train_device")]

    for k, v in [("training_running", False), ("training_result", None), ("training_error", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown("---")

    counts       = dm.waste_class_counts()
    total_ds     = sum(counts.values())
    plate_stats  = dm.plate_dataset_stats()
    can_train    = True

    if task == "waste" and total_ds < 4:
        can_train = False
        st.markdown(
            empty_state_html(
                icon="🚫",
                title="Waste training is blocked",
                body=(
                    f"Need at least 4 labelled waste images. Current total: "
                    f"<b>{total_ds}</b>. Upload photos in the Waste Dataset tab — "
                    "the trainer will not invent classes or use placeholder data."
                ),
                steps=[
                    "Add images under Training → Waste Dataset.",
                    "Then return here and start waste training.",
                ],
            ),
            unsafe_allow_html=True,
        )
    if task == "plate":
        st.markdown(
            f"**Plate dataset:** {plate_stats['images']} image(s), "
            f"{plate_stats['labels']} label(s), "
            f"{plate_stats['unlabelled']} unlabelled"
        )
        if plate_stats["images"] == 0 or plate_stats["labels"] == 0:
            can_train = False
            st.markdown(
                empty_state_html(
                    icon="🍽️",
                    title="Plate training needs labelled images",
                    body=(
                        "No trained plate model exists yet, and there are no YOLO "
                        "labels to train from. This is not a crash — plate training "
                        "is gated until real labels exist. Do not use COCO as a plate model."
                    ),
                    steps=[
                        "Capture plate photos into <code>data/datasets/plate/images/</code>.",
                        "Add matching YOLO <code>.txt</code> labels in "
                        "<code>data/datasets/plate/labels/</code> "
                        "(<code>0 &lt;cx&gt; &lt;cy&gt; &lt;w&gt; &lt;h&gt;</code>, normalised 0–1).",
                        "See <code>docs/DATA_COLLECTION_GUIDE.md</code> for labelling.",
                        "When labels exist, train here or run "
                        "<code>PYTHONPATH=src .venv/bin/python scripts/quickstart.py --plate-only</code>.",
                    ],
                ),
                unsafe_allow_html=True,
            )
            st.code(
                "PYTHONPATH=src .venv/bin/python scripts/quickstart.py --plate-only\n"
                "\n"
                "# After labels exist, the Training page calls PlateModelTrainer:\n"
                "from pathlib import Path\n"
                "import tempfile\n"
                "from cafeteria.config.settings import load_settings\n"
                "from cafeteria.training.dataset_manager import DatasetManager\n"
                "from cafeteria.training.trainer import PlateModelTrainer\n"
                "\n"
                "cfg = load_settings()\n"
                "dm = DatasetManager(cfg.project_root / cfg.storage.datasets)\n"
                "with tempfile.TemporaryDirectory() as tmp:\n"
                "    yaml_path = dm.prepare_yolo_det_dataset(Path(tmp))\n"
                "    PlateModelTrainer(\n"
                "        datasets_dir=cfg.project_root / cfg.storage.datasets,\n"
                "        models_output_dir=cfg.project_root / cfg.storage.models,\n"
                "        device=cfg.device,\n"
                "    ).train(version='v001', data_yaml=yaml_path)",
                language="python",
            )

    if st.button("🚀 START TRAINING", key="btn_start_train", type="primary",
                 disabled=st.session_state.training_running or not can_train):
        st.session_state.training_running = True
        st.session_state.training_result  = None
        st.session_state.training_error   = None
        version = registry.next_version(task)

        def _run():
            try:
                if task == "waste":
                    from cafeteria.training.trainer import WasteModelTrainer
                    r = WasteModelTrainer(
                        datasets_dir=cfg.project_root / cfg.storage.datasets,
                        models_output_dir=cfg.project_root / cfg.storage.models,
                        device=device,
                    ).train(
                        base_model=base_model, epochs=int(epochs),
                        image_size=int(img_size), batch=int(batch), version=version,
                    )
                else:
                    from cafeteria.training.trainer import PlateModelTrainer
                    with tempfile.TemporaryDirectory(prefix="plate_split_") as tmp:
                        data_yaml = dm.prepare_yolo_det_dataset(Path(tmp))
                        r = PlateModelTrainer(
                            datasets_dir=cfg.project_root / cfg.storage.datasets,
                            models_output_dir=cfg.project_root / cfg.storage.models,
                            device=device,
                        ).train(
                            base_model=base_model, epochs=int(epochs),
                            image_size=int(img_size), batch=int(batch),
                            version=version, data_yaml=data_yaml,
                        )

                registry.register_version(
                    task=r.task, version=r.version, weights_path=str(r.weights_path),
                    metrics=r.metrics, config=r.config, dataset_stats=r.dataset_stats,
                )
                st.session_state.training_result = {
                    "version": r.version,
                    "weights_path": str(r.weights_path),
                    "metrics": r.metrics,
                }
            except Exception as e:
                st.session_state.training_error = str(e)
            finally:
                st.session_state.training_running = False

        threading.Thread(target=_run, daemon=True).start()
        st.rerun()

    if st.session_state.training_running:
        with st.spinner("Training in progress — this may take several minutes…"):
            time.sleep(2)
            st.rerun()

    if st.session_state.training_error:
        st.error(f"❌ Training failed: {st.session_state.training_error}")

    if st.session_state.training_result:
        r = st.session_state.training_result
        st.success(f"✅ Training complete! Version: **{r['version']}**")
        for k, v in r.get("metrics", {}).items():
            if isinstance(v, float):
                st.markdown(f"  - {k}: `{v:.4f}`")
        if st.button("⚡ Activate this Model", type="primary", key="btn_activate"):
            registry.activate(task, r["version"])
            _engine_command(
                {"type": "activate_model", "task": task, "version": r["version"]},
            )
            st.success(f"Model {r['version']} is now active!")
            st.session_state.training_result = None


# =============================================================================
# TAB D — Model Versions
# =============================================================================
with tab_versions:
    import datetime
    st.subheader("📋 Model Version History")
    for tname, tkey in [("Waste Model", "waste"), ("Plate Model", "plate")]:
        st.markdown(f"### {tname}")
        versions   = registry.get_all_versions(tkey)
        active_ver = registry.active_version_string(tkey)
        if not versions:
            st.markdown(
                empty_state_html(
                    icon="📋",
                    title=f"No {tname} versions",
                    body=(
                        "The registry is empty for this task. That is expected until "
                        "you train a real model — nothing is pre-loaded or faked."
                    ),
                    steps=[
                        "Collect a dataset in the Waste Dataset tab (or label plates on disk).",
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
                        f"**Trained:** {dt}  |  "
                        f"**Weights:** `{v.get('weights_path', '—')}`"
                    )
                    for mk, mv in v.get("metrics", {}).items():
                        if isinstance(mv, (int, float)):
                            st.markdown(f"  - {mk}: `{mv}`")
                    if not is_active:
                        if st.button(f"⚡ Activate {v['version']}",
                                     key=f"act_{tkey}_{v['version']}"):
                            registry.activate(tkey, v["version"])
                            _engine_command(
                                {"type": "activate_model", "task": tkey, "version": v["version"]},
                            )
                            st.success(f"Activated {v['version']}!")
                            st.rerun()
        st.markdown("---")
