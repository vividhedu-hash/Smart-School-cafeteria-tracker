"""
Enrollment camera — automatic capture that is deliberately forgiving.

Two frame sources, picked by the caller:
  * ``source=<index>``   — open the webcam directly with OpenCV.
  * ``frame_provider``   — borrow clean frames from the running inference
    engine (``GET /raw.jpg``). macOS gives one process exclusive access to
    the camera, so borrowing is how enrollment works without stopping the
    engine.

Capture policy, in order of preference:
  1. Face is centred and a sensible size  → snap immediately.
  2. A face is visible but never lines up → after RELAX_AFTER_SECONDS snap
     anyway. InsightFace, not the Haar cascade, decides what is usable.
  3. Operator presses the manual button   → snap this frame, no questions.
Nothing here can leave the wizard waiting forever.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np


TARGET_FRAMES = 5

# Timing — short holds so a normal person is done in ~2 seconds.
HOLD_SECONDS = 0.10
SNAP_GAP = 0.12
MISS_TOLERANCE_SECONDS = 0.7    # a dropped detection must not reset the hold
RELAX_AFTER_SECONDS = 4.0       # face visible but fussy → start snapping
NO_FACE_HELP_SECONDS = 8.0      # nothing found → tell the operator, keep trying

# Geometry tolerances, in fractions of the frame. These are wide on purpose:
# the oval is a hint for the operator, not an acceptance test. Tight values are
# what made enrollment feel impossible to complete.
OVAL_CX, OVAL_CY = 0.50, 0.46
OVAL_RX, OVAL_RY = 0.22, 0.32
CENTER_TOLERANCE = 0.90         # fraction of the oval radius
FILL_MIN = 0.25
FILL_MAX = 2.20

# Haar on a full 720p frame costs ~700 ms, which makes the preview crawl and
# the coaching feel unresponsive. Detect on a small copy and scale the result.
DETECT_WIDTH = 480
PREVIEW_WIDTH = 720             # what the browser receives; captures stay full-res


def capture_decision(
    # [AI-CoLab: Verified by Antigravity] Pure decision function for adaptive auto-snap & manual camera enrollment
    *,
    manual: bool,
    aligned_hold_seconds: Optional[float],
    face_visible: bool,
    face_seen_for: Optional[float],
    since_last_snap: float,
) -> Optional[str]:
    """
    Decide whether to keep this frame, and why.

    Args:
        manual:               Operator pressed the shutter.
        aligned_hold_seconds: How long the face has been inside the oval,
                              None when it is not (or no longer) aligned.
        face_visible:         A face is in frame right now.
        face_seen_for:        Seconds since a face was first seen at all.
        since_last_snap:      Seconds since the previous capture.

    Returns:
        ``"manual"``, ``"aligning"``, ``"relaxed"``, or None to keep waiting.
    """
    if manual:
        return "manual"
    if since_last_snap < SNAP_GAP:
        return None
    if aligned_hold_seconds is not None and aligned_hold_seconds >= HOLD_SECONDS:
        return "aligning"
    if face_visible and face_seen_for is not None and face_seen_for >= RELAX_AFTER_SECONDS:
        return "relaxed"
    return None


@dataclass
class EnrollStatus:
    """Everything the Streamlit page needs to render one preview tick."""
    jpeg: Optional[bytes]
    hint: str
    count: int
    target: int
    done: bool
    error: Optional[str]
    face_visible: bool
    mode: str          # "aligning" | "relaxed" | "manual"
    elapsed: float
    needs_help: bool   # no face found for a while — surface the alternatives
    quality_score: float = 0.0
    pose_label: str = "Frontal"
    ipd_pixels: float = 0.0
    iso_compliant: bool = False


class EnrollCamera:
    def __init__(
        self,
        source: int = 0,
        width: int = 960,
        height: int = 720,
        frame_provider: Optional[Callable[[], Optional[bytes]]] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.hint = "Starting camera…"
        self.error: Optional[str] = None
        self.jpeg: Optional[bytes] = None
        self.captures: list[np.ndarray] = []
        self.done = False
        self.mode = "aligning"
        self.face_visible = False
        self.quality_score: float = 0.0
        self.pose_label: str = "Frontal"
        self.ipd_pixels: float = 0.0
        self.iso_compliant: bool = False
        self.started_at = time.monotonic()
        self._frame_provider = frame_provider
        self._first_face_at: Optional[float] = None
        # A counter, not a flag: two quick presses must yield two frames.
        self._snaps_requested = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        from cafeteria.recognition.quality import BiometricQualityAssessor
        self._quality_assessor = BiometricQualityAssessor()
        self._thread = threading.Thread(target=self._run, daemon=True, name="enroll-cam")
        self._thread.start()

    # ── Control surface used by the dashboard ────────────────────────────────

    def stop(self) -> None:
        self._stop.set()

    def request_snap(self) -> None:
        # [AI-CoLab: Verified by Antigravity] Thread-safe shutter counter increment for manual capture
        with self._lock:
            self._snaps_requested += 1

    def finish_now(self) -> bool:
        """Accept whatever has been captured so far. False if nothing yet."""
        with self._lock:
            if not self.captures:
                return False
            self.done = True
            self.hint = "Using the shots we have"
            return True

    def status(self) -> EnrollStatus:
        with self._lock:
            elapsed = time.monotonic() - self.started_at
            needs_help = (
                self._first_face_at is None
                and elapsed > NO_FACE_HELP_SECONDS
                and not self.captures
            )
            return EnrollStatus(
                jpeg=self.jpeg,
                hint=self.hint,
                count=len(self.captures),
                target=TARGET_FRAMES,
                done=self.done,
                error=self.error,
                face_visible=self.face_visible,
                mode=self.mode,
                elapsed=elapsed,
                needs_help=needs_help,
                quality_score=self.quality_score,
                pose_label=self.pose_label,
                ipd_pixels=self.ipd_pixels,
                iso_compliant=self.iso_compliant,
            )

    def take_captures(self) -> list[np.ndarray]:
        with self._lock:
            return [c.copy() for c in self.captures]

    # ── Frame acquisition ────────────────────────────────────────────────────

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.source), backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(self.source))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _run(self) -> None:
        if self._frame_provider is not None:
            self._loop(read=self._read_from_provider)
            return

        cap = self._open_capture()
        if cap is None:
            with self._lock:
                self.error = (
                    "Could not open the webcam from Python. Another app may be using it. "
                    "On macOS: System Settings → Privacy & Security → Camera → enable it "
                    "for your terminal / IDE, or upload photos below."
                )
                self.hint = "Camera not available"
            return

        def _read_from_webcam():
            ok, f = cap.read()
            return f if ok else None

        try:
            self._loop(read=_read_from_webcam)
        finally:
            cap.release()

    def _read_from_provider(self) -> Optional[np.ndarray]:
        if self._frame_provider is None:
            return None
        raw = self._frame_provider()
        if raw is None:
            return None
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── Capture loop ─────────────────────────────────────────────────────────

    def _loop(self, read: Callable[[], Optional[np.ndarray]]) -> None:
        borrowed = self._frame_provider is not None
        empty_reads = 0
        last_snap = 0.0
        good_since: Optional[float] = None
        last_good: Optional[float] = None

        while not self._stop.is_set():
            frame = read()
            if frame is None:
                empty_reads += 1
                if empty_reads > 50:
                    with self._lock:
                        self.error = (
                            "The engine stopped sending frames. Start the engine again, "
                            "or upload photos below."
                        )
                        self.hint = "No frames from the engine"
                    return
                time.sleep(0.05 if borrowed else 0.03)
                continue
            empty_reads = 0

            vis, aligned, face_found = self._overlay(frame)
            now = time.monotonic()

            with self._lock:
                self.face_visible = face_found
                if face_found and self._first_face_at is None:
                    self._first_face_at = now
                manual = self._snaps_requested > 0
                if manual:
                    self._snaps_requested -= 1
                first_face_at = self._first_face_at

            if aligned:
                last_good = now
                if good_since is None:
                    good_since = now
            elif last_good is not None and now - last_good <= MISS_TOLERANCE_SECONDS:
                pass  # brief miss — keep the hold alive
            else:
                good_since = None

            reason = capture_decision(
                manual=manual,
                aligned_hold_seconds=(now - good_since) if good_since is not None else None,
                face_visible=face_found,
                face_seen_for=(now - first_face_at) if first_face_at is not None else None,
                since_last_snap=now - last_snap,
            )

            if reason is not None:
                last_snap = now
                self._store_capture(frame, reason)

            self._publish_preview(vis, borrowed)

    def _store_capture(self, frame: np.ndarray, reason: str) -> None:
        # [AI-CoLab: Verified by Antigravity] Adaptive capture store supporting align and relax modes
        with self._lock:
            if len(self.captures) >= TARGET_FRAMES:
                return
            self.captures.append(frame.copy())
            n = len(self.captures)
            self.mode = reason
            if n >= TARGET_FRAMES:
                self.done = True
                self.hint = "Enrollment Complete"
            elif reason == "relaxed":
                self.hint = f"Capturing — {n}/{TARGET_FRAMES}"
            else:
                self.hint = f"Hold still — {n}/{TARGET_FRAMES}"

    def _publish_preview(self, vis: np.ndarray, borrowed: bool) -> None:
        ok_jpg, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok_jpg:
            with self._lock:
                self.jpeg = buf.tobytes()
        if borrowed:
            time.sleep(0.03)  # engine republishes ~5/s; a light yield is enough

    # ── Overlay / coaching ───────────────────────────────────────────────────

    def _detect_face(self, frame: np.ndarray) -> Optional[tuple[float, float, float]]:
        """
        Largest face as ``(cx, cy, width)`` in frame fractions, or None.

        Runs on a small grayscale copy — same detections, a fraction of the cost.
        """
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / float(w) if w > DETECT_WIDTH else 1.0
        small = (
            cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else frame
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32)
        )
        if not len(faces):
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        sw, sh = gray.shape[1], gray.shape[0]
        return (x + fw / 2) / sw, (y + fh / 2) / sh, fw / sw

    def _overlay(self, frame: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        face = self._detect_face(frame)
        aligned = False
        hint = "Look towards the camera"
        h, w = frame.shape[:2]

        quality_report = None
        if face is not None:
            fx, fy, fw = face
            fill = fw / (OVAL_RX * 2)
            centered = (
                abs(fx - OVAL_CX) < OVAL_RX * CENTER_TOLERANCE
                and abs(fy - OVAL_CY) < OVAL_RY * CENTER_TOLERANCE
            )
            size_ok = FILL_MIN < fill < FILL_MAX
            aligned = centered and size_ok

            # Evaluate ISO Biometric Quality
            bx1 = int(max(0, (fx - fw / 2) * w))
            by1 = int(max(0, (fy - fw / 2) * h))
            bx2 = int(min(w, (fx + fw / 2) * w))
            by2 = int(min(h, (fy + fw / 2) * h))
            quality_report = self._quality_assessor.evaluate(frame, [bx1, by1, bx2, by2])

            with self._lock:
                self.quality_score = quality_report.overall_score
                self.pose_label = quality_report.pose_category.value.replace("_", " ").title()
                self.ipd_pixels = quality_report.ipd_pixels
                self.iso_compliant = quality_report.passed

            if not size_ok and fill <= FILL_MIN:
                hint = "Come a bit closer"
            elif not size_ok:
                hint = "Back up slightly"
            elif not centered:
                hint = "Move into the oval"
            elif quality_report.coaching_hint:
                hint = quality_report.coaching_hint
            else:
                hint = "Got you — hold still"

        with self._lock:
            if not self.done and not self.captures:
                self.hint = hint
            n = len(self.captures)
            q_score = self.quality_score
            p_label = self.pose_label
            ipd_px = self.ipd_pixels
            iso_ok = self.iso_compliant

        # Preview is downscaled: the browser shows it at ~420 px wide, and a
        # full-res JPEG every tick is wasted bandwidth.
        if w > PREVIEW_WIDTH:
            ph = int(h * PREVIEW_WIDTH / w)
            vis = cv2.resize(frame, (PREVIEW_WIDTH, ph), interpolation=cv2.INTER_AREA)
        else:
            vis = frame.copy()
        vh, vw = vis.shape[:2]
        cx, cy = int(vw * OVAL_CX), int(vh * OVAL_CY)
        rx, ry = int(vw * OVAL_RX), int(vh * OVAL_RY)

        mask = np.zeros((vh, vw), np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        vis = np.where(mask[:, :, None] == 255, vis, (vis * 0.45).astype(np.uint8))

        colour = (
            (52, 211, 153) if (aligned and iso_ok)
            else (56, 189, 248) if (aligned or face is not None)
            else (100, 116, 139)
        )
        cv2.ellipse(vis, (cx, cy), (rx, ry), 0, 0, 360, colour, 3)

        # Apple Face ID style dynamic circular tick progress
        if n:
            sweep = int(360 * n / TARGET_FRAMES)
            cv2.ellipse(
                vis, (cx, cy), (rx + 10, ry + 10), 0, -90,
                -90 + sweep, (52, 211, 153), 4,
            )
        # Mirror only what the operator sees; saved frames stay un-flipped.
        flipped = cv2.flip(vis, 1)

        # Top Biometric HUD Bar
        if face is not None:
            hud_bg = (15, 23, 42)
            cv2.rectangle(flipped, (12, 12), (vw - 12, 42), hud_bg, -1)
            cv2.rectangle(flipped, (12, 12), (vw - 12, 42), (51, 65, 85), 1)
            status_dot = (52, 211, 153) if iso_ok else (245, 158, 11)
            cv2.circle(flipped, (26, 27), 4, status_dot, -1)
            hud_text = f"ISO 19794-5 QUALITY: {int(q_score)}% | IPD: {int(ipd_px)}px | POSE: {p_label.upper()}"
            cv2.putText(
                flipped, hud_text, (38, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA,
            )

        return flipped, aligned, face is not None


_SESSIONS: dict[str, EnrollCamera] = {}
_SESS_LOCK = threading.Lock()


def start_enroll_camera(
    session_key: str,
    source: int = 0,
    frame_provider: Optional[Callable[[], Optional[bytes]]] = None,
) -> EnrollCamera:
    with _SESS_LOCK:
        cam = _SESSIONS.get(session_key)
        if cam is None or cam.error:
            if cam is not None:
                cam.stop()
            cam = EnrollCamera(source=source, frame_provider=frame_provider)
            _SESSIONS[session_key] = cam
        return cam


def restart_enroll_camera(
    session_key: str,
    source: int = 0,
    frame_provider: Optional[Callable[[], Optional[bytes]]] = None,
) -> EnrollCamera:
    """Drop any existing session and start a fresh capture."""
    stop_enroll_camera(session_key)
    return start_enroll_camera(session_key, source=source, frame_provider=frame_provider)


def stop_enroll_camera(session_key: str) -> None:
    with _SESS_LOCK:
        cam = _SESSIONS.pop(session_key, None)
    if cam is not None:
        cam.stop()
