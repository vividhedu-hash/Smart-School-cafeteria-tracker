"""
Smart Cafeteria Waste Tracker — Main Inference Engine

This is the background process that runs the CV pipeline.
It does NOT import Streamlit. The dashboard is a separate process.

Run with:
    python -m cafeteria.main

Or from the project root:
    python src/cafeteria/main.py

Controls (OpenCV debug window, if enabled):
    D  = toggle debug overlay
    Q  = quit
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

# ── Ensure project src is importable ────────────────────────────────────────
_here = Path(__file__).resolve().parent
_src  = _here.parent.parent   # src/
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import cv2
import numpy as np

from cafeteria.config.settings import (
    load_settings, describe_device, system_info
)
from cafeteria.camera.frame_buffer import FrameBuffer, build_camera
from cafeteria.detection.plate_detector import PlateDetector, ModelNotFoundError
from cafeteria.detection.waste_detector import WasteDetector
from cafeteria.recognition.face_engine import FaceEngine
from cafeteria.recognition.matcher import EmbeddingMatcher, match_field
from cafeteria.recognition.enrollment import EnrollmentManager
from cafeteria.recognition.live_match import (
    box_iou,
    face_size_px,
    persist_bbox,
    scale_bbox_to_frame,
    smooth_bbox,
)
from cafeteria.monitoring.overlay import draw_debug_overlay
from cafeteria.pipeline.state_machine import StateMachine
from cafeteria.pipeline.event_manager import EventManager
from cafeteria.pipeline.transaction import TransactionEngine
from cafeteria.storage.database import init_db
from cafeteria.monitoring.health import (
    get_system_metrics, component_status, compute_readiness
)
from cafeteria.monitoring.metrics import MetricsCollector, read_commands
from cafeteria.monitoring.heartbeat import should_idle_shutdown
from cafeteria.training.registry import ModelRegistry
from cafeteria.utils.logging import setup_logging, get_logger, EventCode, log_event
from cafeteria.utils.timing import FPSCounter

logger = get_logger("main")
_shutdown = False


def handle_signal(sig, frame):
    global _shutdown
    logger.info("Received signal %d — shutting down gracefully", sig)
    _shutdown = True


def run_engine() -> None:
    global _shutdown

    # ── Signals ─────────────────────────────────────────────────────────
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Settings ─────────────────────────────────────────────────────────
    cfg = load_settings()
    project_root = cfg.project_root

    setup_logging(
        log_level=cfg.application.log_level,
        log_dir=project_root / "logs",
    )

    log_event(logger, EventCode.APPLICATION_START, "Smart Cafeteria Waste Tracker")
    info = system_info()
    for k, v in info.items():
        logger.info("  %s: %s", k, v)

    # ── PID file (so the dashboard can see the engine regardless of how it
    #    was launched — run.py, CLI, or dashboard Start button) ─────────────
    pid_file = project_root / "data" / "engine.pid"
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))
    except Exception as exc:
        logger.warning("Could not write PID file: %s", exc)

    # ── Database ─────────────────────────────────────────────────────────
    db_path = project_root / cfg.storage.database
    init_db(db_path)

    # ── Metrics ──────────────────────────────────────────────────────────
    metrics = MetricsCollector(
        state_file=project_root / cfg.application.runtime_state_path,
        history_size=cfg.monitoring.metrics_history_size,
        write_interval_s=cfg.monitoring.state_write_interval_seconds,
    )

    # ── Model registry ───────────────────────────────────────────────────
    registry = ModelRegistry(project_root / "models" / "registry.json")

    # ── Detectors ────────────────────────────────────────────────────────
    plate_weights = registry.get_active_weights("plate") or str(
        project_root / cfg.models.plate.weights
    )
    waste_weights = registry.get_active_weights("waste") or str(
        project_root / cfg.models.waste.weights
    )

    plate_detector: PlateDetector | None = None
    waste_detector: WasteDetector | None = None

    try:
        plate_detector = PlateDetector(
            weights_path=plate_weights,
            confidence=cfg.models.plate.confidence,
            iou=cfg.models.plate.iou,
            device=cfg.device,
            roi={"x1": cfg.roi.plate.x1, "y1": cfg.roi.plate.y1,
                 "x2": cfg.roi.plate.x2, "y2": cfg.roi.plate.y2},
            allow_coco_fallback=cfg.models.plate.allow_coco_fallback,
        )
        plate_detector.load()
        if plate_detector.is_proxy:
            logger.warning(
                "PLATE PROXY MODE ACTIVE — COCO demo model, not a trained "
                "plate detector. Train a real model for production use."
            )
    except ModelNotFoundError as e:
        logger.warning("Plate detector not ready: %s", str(e).splitlines()[0])
    except Exception as e:
        logger.error("Plate detector load error: %s", e)

    try:
        waste_detector = WasteDetector(
            weights_path=waste_weights,
            confidence=cfg.models.waste.confidence,
            device=cfg.device,
        )
        waste_detector.load()
    except ModelNotFoundError as e:
        logger.warning("Waste detector not ready: %s", str(e).splitlines()[0])
    except Exception as e:
        logger.error("Waste detector load error: %s", e)

    # ── Face engine ──────────────────────────────────────────────────────
    face_engine: FaceEngine | None = None
    try:
        face_engine = FaceEngine(
            model_pack=cfg.recognition.model_pack,
            model_dir=project_root / "models" / "face",
            det_size=tuple(cfg.recognition.det_size),
            device=cfg.device,
            det_thresh=float(getattr(cfg.recognition, "det_thresh", 0.42) or 0.42),
        )
        face_engine.load()
    except Exception as e:
        logger.warning("Face engine not ready: %s", e)

    # ── Embeddings ───────────────────────────────────────────────────────
    matcher = EmbeddingMatcher(
        enrollment_dir=project_root / cfg.recognition.embedding_dir,
        similarity_threshold=cfg.recognition.similarity_threshold,
    )
    matcher.load_embeddings()

    # ── Pipeline ─────────────────────────────────────────────────────────
    state_machine = StateMachine(cooldown_seconds=cfg.event.cooldown_seconds)
    event_manager = EventManager(
        state_machine=state_machine,
        plate_detector=plate_detector,
        waste_detector=waste_detector,
        face_engine=face_engine,
        matcher=matcher,
        config=cfg,
    )
    event_manager.set_metrics(metrics)  # enables live face match in always-on mode
    tx_engine = TransactionEngine(
        captures_dir=project_root / cfg.storage.captures,
        review_queue_dir=project_root / cfg.storage.review_queue,
    )


    # ── Camera ───────────────────────────────────────────────────────────
    camera = build_camera(cfg.camera)
    if not camera.open():
        logger.error("Could not open camera. Check config.yaml camera settings.")
        log_event(logger, EventCode.CAMERA_DISCONNECTED, "Failed to open camera at startup")
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(1)

    cam_info = camera.info()
    log_event(
        logger, EventCode.CAMERA_CONNECTED,
        f"resolution={cam_info.width}x{cam_info.height}  "
        f"backend={cam_info.backend}"
    )

    frame_buffer = FrameBuffer(
        camera=camera,
        maxsize=cfg.camera.buffer_size,
        on_disconnect=lambda: camera.reconnect() if hasattr(camera, "reconnect") else None,
    )
    frame_buffer.start()

    fps_counter = FPSCounter(window=30)
    debug_mode = cfg.inference.debug_window
    debug_display = cfg.application.debug
    frame_count = 0
    last_registry_check = 0.0
    last_embedding_reload = 0.0
    current_plate_det = None
    current_waste_result = None
    current_face_match = None

    frames_dir = project_root / cfg.storage.frames
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ── Heartbeat watcher ─────────────────────────────────────────────────────
    # Live Monitor is the only page that heartbeats. Leave that page (or never
    # open it) and the engine exits so macOS can turn the camera LED off and
    # the Training wizard can use the browser webcam.
    _HEARTBEAT_FILE = project_root / "data" / "heartbeat"
    _engine_started_at = time.time()
    _last_heartbeat_check = 0.0

    # ── Dedicated face-inference thread (decouples grabbing from recognition) ──
    import threading
    _face_result_lock = threading.Lock()
    _face_result: dict | None = None           # latest live_face_match payload
    _face_inference_frame: np.ndarray | None = None  # frame queued for inference
    _face_frame_lock = threading.Lock()
    _face_thread_stop = threading.Event()
    _held_bbox = None
    _held_missing = 0
    _held_identity: dict | None = None

    def _face_inference_worker():
        """InsightFace walk-past lock: persist bbox, identify when the face is large enough."""
        nonlocal _face_result, _face_inference_frame
        nonlocal _held_bbox, _held_missing, _held_identity
        lock_min = int(getattr(cfg.recognition, "minimum_face_size", 24) or 24)
        identify_min = int(getattr(cfg.recognition, "identify_face_size", 48) or 48)
        max_hold = int(getattr(cfg.recognition, "bbox_hold_frames", 12) or 12)
        infer_max = int(getattr(cfg.recognition, "infer_max_width", 1280) or 1280)
        # [AI-CoLab: Cursor] Face worker is a hot path — no file/NDJSON debug probes here.
        while not _face_thread_stop.is_set():
            with _face_frame_lock:
                img = _face_inference_frame
                _face_inference_frame = None
            if img is None:
                time.sleep(0.005)
                continue
            try:
                import cv2 as _cv2
                _t0 = time.perf_counter()
                h, w = img.shape[:2]
                if w > infer_max:
                    small = _cv2.resize(
                        img,
                        (infer_max, int(h * infer_max / w)),
                        interpolation=_cv2.INTER_LINEAR,
                    )
                else:
                    small = img
                faces = face_engine.get_faces(small)
                result = None
                if faces:
                    face = max(
                        faces,
                        key=lambda f: (f["bbox"][2] - f["bbox"][0])
                        * (f["bbox"][3] - f["bbox"][1]),
                    )
                    raw_bbox = scale_bbox_to_frame(
                        face["bbox"], small.shape[1], small.shape[0], w, h
                    )
                    if _held_bbox is not None and box_iou(_held_bbox, raw_bbox) < 0.15:
                        _held_identity = None
                    bbox = smooth_bbox(_held_bbox, raw_bbox, alpha=0.45)
                    bbox, _held_missing = persist_bbox(bbox, bbox, 0, max_hold)
                    _held_bbox = bbox
                    fw, fh = face_size_px(bbox)
                    det_score = round(float(face["det_score"]), 3)
                    approaching = fw < identify_min or fh < identify_min
                    too_small = fw < lock_min or fh < lock_min
                    result = {
                        "person_id": None,
                        "person_name": None,
                        "similarity": 0.0,
                        "is_known": False,
                        "bbox": bbox,
                        "det_score": det_score,
                        "approaching": approaching,
                    }
                    if too_small:
                        result["approaching"] = True
                    elif not approaching:
                        m = matcher.match(face["embedding"])
                        result["person_id"] = m.person_id
                        result["person_name"] = m.person_name
                        result["similarity"] = round(m.similarity, 3)
                        result["is_known"] = m.is_known
                        _held_identity = {
                            "person_id": result["person_id"],
                            "person_name": result["person_name"],
                            "similarity": result["similarity"],
                            "is_known": result["is_known"],
                        }
                    else:
                        result["similarity"] = 0.0
                else:
                    _held_bbox, _held_missing = persist_bbox(
                        _held_bbox, None, _held_missing, max_hold
                    )
                    if _held_bbox is None:
                        _held_identity = None
                        result = None
                    else:
                        # Tracking hold only — do not invent a new identity.
                        result = {
                            "person_id": (_held_identity or {}).get("person_id"),
                            "person_name": (_held_identity or {}).get("person_name"),
                            "similarity": (_held_identity or {}).get("similarity", 0.0),
                            "is_known": bool((_held_identity or {}).get("is_known")),
                            "bbox": _held_bbox,
                            "det_score": None,
                            "approaching": not bool((_held_identity or {}).get("is_known")),
                        }
                face_ms = (time.perf_counter() - _t0) * 1000.0
                with _face_result_lock:
                    _face_result = result
                if metrics:
                    metrics.set_live_face_match(result)
                    metrics.update_latencies(face_ms=face_ms)
            except Exception as exc:
                logger.debug("Face thread error: %s", exc)

    _face_thread: threading.Thread | None = None

    def _pipeline_ready() -> bool:
        """Both real models loaded — full waste pipeline may run."""
        return (
            plate_detector is not None and getattr(plate_detector, "is_loaded", False)
            and waste_detector is not None and getattr(waste_detector, "is_loaded", False)
        )

    def _runtime_extra() -> dict:
        """Common runtime-state payload: components + readiness + models."""
        cam_ok = camera.is_open()
        plate_ok = plate_detector is not None and plate_detector.is_loaded
        waste_ok = waste_detector is not None and waste_detector.is_loaded
        face_ok = face_engine is not None and face_engine.is_loaded
        proxy = plate_detector is not None and getattr(plate_detector, "proxy_mode", False)
        readiness = compute_readiness(
            plate_model_loaded=plate_ok,
            waste_model_loaded=waste_ok,
            face_engine_loaded=face_ok,
            camera_connected=cam_ok,
            enrolled_count=matcher.enrolled_count,
            plate_proxy_mode=proxy,
        )
        return {
            "camera_connected": cam_ok,
            "components": component_status(
                camera_connected=cam_ok,
                plate_detector_loaded=plate_ok,
                waste_detector_loaded=waste_ok,
                face_engine_loaded=face_ok,
                db_ok=True,
            ),
            "ready": readiness["ready"],
            "blocking_reasons": readiness["blocking_reasons"],
            "readiness": readiness,
            "pipeline_ready": _pipeline_ready(),
            "plate_proxy": proxy,
            "plate_proxy_mode": proxy,
            "waste_model_missing": not waste_ok,
            "enrolled_persons": matcher.enrolled_count,
            "active_models": {
                "plate": registry.active_version_string("plate"),
                "waste": registry.active_version_string("waste"),
            },
            "similarity_threshold": cfg.recognition.similarity_threshold,
            "api": {
                "host": cfg.application.api_host,
                "port": cfg.application.api_port,
                "url": f"http://{cfg.application.api_host}:{cfg.application.api_port}",
            },
            "buffer": frame_buffer.stats,
        }

    def _ensure_face_thread(should_run: bool) -> None:
        """Start/stop the always-on face thread as pipeline readiness changes
        (e.g. after a model is trained and hot-swapped in)."""
        nonlocal _face_thread
        if should_run and _face_thread is None and face_engine and face_engine.is_loaded:
            _face_thread_stop.clear()
            _face_thread = threading.Thread(
                target=_face_inference_worker, daemon=True, name="face-inference"
            )
            _face_thread.start()
            logger.info("Face inference thread started (always-on mode)")
        elif not should_run and _face_thread is not None:
            _face_thread_stop.set()
            _face_thread.join(timeout=2.0)
            _face_thread = None
            logger.info("Face inference thread stopped — full pipeline active")

    _ensure_face_thread(not _pipeline_ready())

    api_server = None
    try:
        from cafeteria.api.server import start_engine_api
        from cafeteria.auth import ensure_engine_token
        api_server = start_engine_api(
            host=cfg.application.api_host,
            port=int(cfg.application.api_port),
            get_state=lambda: metrics.snapshot(_runtime_extra()),
            frame_path=frames_dir / "latest.jpg",
            heartbeat_path=_HEARTBEAT_FILE,
            commands_path=project_root / cfg.application.commands_path,
            on_command=lambda cmd: _handle_command(
                cmd, matcher, plate_detector, waste_detector, registry, project_root
            ),
            token=ensure_engine_token(project_root),
        )
        logger.info(
            "Dashboard should poll http://%s:%s/state (JSON files are fallback).",
            cfg.application.api_host,
            cfg.application.api_port,
        )
    except OSError as exc:
        logger.warning("Engine API did not bind (%s) — dashboard will use JSON files.", exc)

    logger.info("Engine running. Press D to toggle debug, Q to quit (in OpenCV window).")

    # ── Main loop ────────────────────────────────────────────────────────
    try:
        while not _shutdown:
            frame = frame_buffer.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            fps = fps_counter.tick()
            metrics.update_fps(fps)

            # ── Hot-swap check (every 5 s) ───────────────────────────────
            now = time.time()
            if now - last_registry_check > 5.0:
                last_registry_check = now
                _check_model_hotswap(registry, plate_detector, waste_detector, project_root)
                # Newly-trained models may flip us from face-only into full
                # pipeline mode (or back) without a restart.
                _ensure_face_thread(not _pipeline_ready())

            # ── Heartbeat check (every 1 s) ──────────────────────────────
            if now - _last_heartbeat_check > 1.0:
                _last_heartbeat_check = now
                last_beat = None
                if _HEARTBEAT_FILE.exists():
                    try:
                        last_beat = float(_HEARTBEAT_FILE.read_text().strip())
                    except Exception:
                        last_beat = None
                if should_idle_shutdown(
                    now=now,
                    started_at=_engine_started_at,
                    last_heartbeat=last_beat,
                ):
                    logger.info(
                        "Live Monitor idle — shutting down engine and releasing camera."
                    )
                    _shutdown = True

            # ── Reload embeddings if new enrollments (every 30 s) ────────
            if now - last_embedding_reload > 30.0:
                last_embedding_reload = now
                n = matcher.reload()
                if n > 0:
                    logger.debug("Reloaded %d embeddings", n)

            # ── Command processing ────────────────────────────────────────
            cmd = read_commands(project_root / cfg.application.commands_path)
            if cmd:
                _handle_command(cmd, matcher, plate_detector, waste_detector, registry, project_root)

            if _face_thread is not None:
                # ── Threaded always-on face scan (no plate model) ────────
                # Queue the latest frame for the face inference thread
                with _face_frame_lock:
                    _face_inference_frame = frame.image

                # Read latest result from the thread (non-blocking)
                with _face_result_lock:
                    current_face_match = _face_result

                # Still write metrics + debug frame every loop iteration
                metrics.set_state(state_machine.state.value)
                metrics.write_state(_runtime_extra())
                _write_debug_frame(frame.image, frames_dir, debug_display,
                                   state_machine.state.value, fps,
                                   current_plate_det, current_waste_result,
                                   current_face_match, cfg.roi)
                continue

            # ── Frame skip for plate detection ───────────────────────────
            run_inference = (frame_count % max(1, cfg.inference.frame_skip) == 0)
            if not run_inference and state_machine.is_idle():
                _write_debug_frame(frame.image, frames_dir, debug_display,
                                   state_machine.state.value, fps,
                                   current_plate_det, current_waste_result,
                                   current_face_match, cfg.roi)
                metrics.set_state(state_machine.state.value)
                metrics.write_state(_runtime_extra())
                continue

            # ── Event pipeline ───────────────────────────────────────────
            recent = frame_buffer.get_recent(cfg.event.best_frame_window)
            completed = event_manager.process_frame(frame, recent)

            # Update current display values for overlay
            if hasattr(event_manager, "_ctx") and event_manager._ctx:
                ctx = event_manager._ctx
                current_plate_det = ctx.best_plate_detection
                current_waste_result = ctx.waste_result
                current_face_match = ctx.face_match
            else:
                if state_machine.is_idle():
                    current_plate_det = None
                    current_waste_result = None
                    current_face_match = None

            # ── Commit completed event ────────────────────────────────────
            if completed is not None:
                tx_id = tx_engine.commit(completed)
                metrics.set_last_event({
                    "transaction_id": tx_id,
                    "timestamp": completed.timestamp,
                    "waste_status": completed.waste_result.label if completed.waste_result else None,
                    "person_id": completed.face_match.person_id if completed.face_match else None,
                    "status": completed.status,
                    "latency_ms": completed.processing_latency_ms,
                })

            # ── Metrics / state ──────────────────────────────────────────
            metrics.set_state(state_machine.state.value)
            metrics.write_state(_runtime_extra())

            # ── Debug display ─────────────────────────────────────────────
            _write_debug_frame(frame.image, frames_dir, debug_display,
                               state_machine.state.value, fps,
                               current_plate_det, current_waste_result,
                               current_face_match, cfg.roi)

            if debug_mode and debug_display:
                annotated = draw_debug_overlay(
                    frame.image,
                    state=state_machine.state.value,
                    fps=fps,
                    plate_det=current_plate_det,
                    waste_result=current_waste_result,
                    face_match=current_face_match,
                    latency_ms=0.0,
                    roi_cfg=cfg.roi,
                    debug=True,
                )
                cv2.imshow("Smart Cafeteria Waste Tracker (press Q to quit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    _shutdown = True
                elif key == ord("d") or key == ord("D"):
                    debug_display = not debug_display

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        logger.info("Shutting down...")
        _face_thread_stop.set()
        if _face_thread:
            _face_thread.join(timeout=2.0)
        if api_server is not None:
            try:
                api_server.shutdown()
            except Exception:
                pass
        frame_buffer.stop()
        camera.close()
        cv2.destroyAllWindows()
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        logger.info("Engine stopped.")




def _write_debug_frame(image, frames_dir, debug, state, fps, plate_det,
                       waste_result, face_match, roi_cfg):
    """Write annotated frame to data/frames/latest.jpg for dashboard."""
    annotated = draw_debug_overlay(
        image, state=state, fps=fps,
        plate_det=plate_det, waste_result=waste_result,
        face_match=face_match, latency_ms=0.0,
        roi_cfg=roi_cfg, debug=debug,
    )
    try:
        cv2.imwrite(str(frames_dir / "latest.jpg"), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
    except Exception:
        pass


def _check_model_hotswap(registry, plate_detector, waste_detector, project_root):
    """Check if a new model has been activated and hot-swap if needed."""
    if plate_detector:
        active = registry.get_active_weights("plate")
        if active:
            active_path = Path(active)
            if not active_path.is_absolute():
                active_path = project_root / active_path
            if active_path.exists() and active_path != Path(str(plate_detector._weights_path)):
                try:
                    plate_detector.update_weights(active_path)
                    logger.info("Plate detector hot-swapped to %s", active_path)
                except Exception as e:
                    logger.error("Plate hot-swap failed: %s", e)

    if waste_detector:
        active = registry.get_active_weights("waste")
        if active:
            active_path = Path(active)
            if not active_path.is_absolute():
                active_path = project_root / active_path
            if active_path.exists() and active_path != Path(str(waste_detector._weights_path)):
                try:
                    waste_detector.update_weights(active_path)
                    logger.info("Waste detector hot-swapped to %s", active_path)
                except Exception as e:
                    logger.error("Waste hot-swap failed: %s", e)


def _handle_command(cmd, matcher, plate_detector, waste_detector, registry, project_root):
    """Process a command written by the dashboard."""
    cmd_type = cmd.get("type")
    if cmd_type == "reload_embeddings":
        n = matcher.reload()
        logger.info("Command: reload_embeddings — loaded %d persons", n)
    elif cmd_type == "activate_model":
        task = cmd.get("task")
        version = cmd.get("version")
        if task and version:
            registry.activate(task, version)
            _check_model_hotswap(registry, plate_detector, waste_detector, project_root)


if __name__ == "__main__":
    run_engine()
