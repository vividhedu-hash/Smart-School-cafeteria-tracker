"""
Enrollment camera — same OpenCV webcam path as Live Monitor.

Chrome cannot use getUserMedia inside Streamlit's custom-component iframe
(no allow="camera"). Live Monitor already works because Python owns the
webcam. Enrollment uses that same camera so Chrome is just a viewer.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


TARGET_FRAMES = 8
HOLD_SECONDS = 0.28
SNAP_GAP = 0.14


class EnrollCamera:
    def __init__(self, source: int = 0, width: int = 960, height: int = 720) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.hint = "Starting camera…"
        self.error: Optional[str] = None
        self.jpeg: Optional[bytes] = None
        self.captures: list[np.ndarray] = []
        self.done = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="enroll-cam")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> tuple[Optional[bytes], str, int, bool, Optional[str]]:
        with self._lock:
            return self.jpeg, self.hint, len(self.captures), self.done, self.error

    def take_captures(self) -> list[np.ndarray]:
        with self._lock:
            return [c.copy() for c in self.captures]

    def _run(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(self.source), backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(self.source))
        if not cap.isOpened():
            with self._lock:
                self.error = (
                    "Could not open the webcam from Python. On macOS: System Settings → "
                    "Privacy & Security → Camera → enable it for Terminal (or Cursor)."
                )
                self.hint = "Camera unavailable"
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        good_since: Optional[float] = None
        last_snap = 0.0
        while not self._stop.is_set() and not self.done:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            vis, aligned = self._overlay(frame)
            now = time.monotonic()
            if aligned:
                if good_since is None:
                    good_since = now
                if now - good_since >= HOLD_SECONDS and now - last_snap >= SNAP_GAP:
                    with self._lock:
                        if len(self.captures) < TARGET_FRAMES:
                            self.captures.append(frame.copy())
                            last_snap = now
                            n = len(self.captures)
                            self.hint = f"Hold still — {n}/{TARGET_FRAMES}"
                            if n >= TARGET_FRAMES:
                                self.done = True
                                self.hint = "You’re in"
            else:
                good_since = None
            ok_jpg, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            if ok_jpg:
                with self._lock:
                    self.jpeg = buf.tobytes()
        cap.release()

    def _overlay(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        h, w = frame.shape[:2]
        cx, cy = w // 2, int(h * 0.46)
        rx, ry = int(w * 0.22), int(h * 0.32)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80)
        )
        aligned = False
        hint = "Fit your face in the oval"
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            fx, fy = x + fw / 2, y + fh / 2
            fill = fw / max(rx * 2, 1)
            centered = abs(fx - cx) < rx * 0.45 and abs(fy - cy) < ry * 0.45
            size_ok = 0.55 < fill < 1.25
            aligned = centered and size_ok
            if not size_ok and fill <= 0.55:
                hint = "Move a little closer"
            elif not size_ok:
                hint = "Move back a little"
            elif not centered:
                hint = "Center your face"
            else:
                hint = "Perfect — hold still"
        with self._lock:
            if not self.done:
                self.hint = hint

        vis = frame.copy()
        mask = np.zeros((h, w), np.uint8)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        dark = (vis * 0.38).astype(np.uint8)
        vis = np.where(mask[:, :, None] == 255, vis, dark)
        color = (52, 211, 153) if aligned else (56, 189, 248) if len(faces) else (100, 116, 139)
        cv2.ellipse(vis, (cx, cy), (rx, ry), 0, 0, 360, color, 3)
        n = len(self.captures)
        if n:
            cv2.ellipse(
                vis, (cx, cy), (rx + 10, ry + 10), 0, -90,
                -90 + int(360 * n / TARGET_FRAMES), (52, 211, 153), 4,
            )
        return vis, aligned


_SESSIONS: dict[str, EnrollCamera] = {}
_SESS_LOCK = threading.Lock()


def start_enroll_camera(session_key: str, source: int = 0) -> EnrollCamera:
    with _SESS_LOCK:
        cam = _SESSIONS.get(session_key)
        if cam is None or cam.error or cam.done:
            if cam is not None:
                cam.stop()
            cam = EnrollCamera(source=source)
            _SESSIONS[session_key] = cam
        return cam


def stop_enroll_camera(session_key: str) -> None:
    with _SESS_LOCK:
        cam = _SESSIONS.pop(session_key, None)
    if cam is not None:
        cam.stop()
