"""
Virtual Cafeteria Camera — Realistic dynamic cafeteria video feed.

Used when:
  - Running on Streamlit Cloud or headless Linux containers without physical webcams
  - Running in demo / simulation mode (CAFETERIA_VIRTUAL_CAM=1)
  - Hardware camera is missing, unplugged, or locked by another process

This feed generates a live cafeteria checkout stream with trays, plates, food items,
and arriving customers. Real OpenCV visual detectors, YOLO models, and FaceEngine
execute on this feed without mocks, enabling full 1-click cloud testing.
"""
from __future__ import annotations

import datetime
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cafeteria.camera.base import CameraBase, CameraInfo, TimestampedFrame
from cafeteria.utils.logging import get_logger
from cafeteria.utils.timing import FPSCounter

logger = get_logger("camera.virtual")


class VirtualCafeteriaCamera(CameraBase):
    """
    Virtual camera that procedurally generates or composites realistic
    cafeteria checkout frames at a specified frame rate and resolution.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 20,
        camera_id: str = "cam0",
        project_root: Optional[Path] = None,
    ) -> None:
        self.width = width if width and width >= 320 else 1280
        self.height = height if height and height >= 240 else 720
        self.fps = fps if fps and fps > 0 else 20
        self.camera_id = camera_id
        self._project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent

        self._is_open = False
        self._frame_id = 0
        self._fps_counter = FPSCounter(window=30)
        self._last_frame_mono = 0.0

        # Sample assets cache
        self._plate_samples: list[np.ndarray] = []
        self._face_samples: list[np.ndarray] = []
        self._loaded_assets = False

    def open(self) -> bool:
        """Initialize the virtual cafeteria camera and load sample assets."""
        self._load_sample_assets()
        self._is_open = True
        self._frame_id = 0
        self._last_frame_mono = time.monotonic()
        logger.info(
            "Virtual Cafeteria Camera active — %dx%d @ %d fps (assets: %d plates, %d faces)",
            self.width,
            self.height,
            self.fps,
            len(self._plate_samples),
            len(self._face_samples),
        )
        return True

    def is_open(self) -> bool:
        return self._is_open

    def close(self) -> None:
        self._is_open = False
        logger.info("Virtual Cafeteria Camera closed")

    def reconnect(self) -> bool:
        self.close()
        return self.open()

    def info(self) -> CameraInfo:
        return CameraInfo(
            camera_id=self.camera_id,
            mode="webcam",
            source="virtual_stream",
            width=self.width,
            height=self.height,
            target_fps=self.fps,
            actual_fps=round(self._fps_counter.fps, 1),
            backend="Virtual Stream",
            device_name="Virtual Cafeteria Camera (Demo Stream)",
            is_connected=self._is_open,
        )

    def read(self) -> Optional[TimestampedFrame]:
        if not self._is_open:
            return None

        # Pacing
        now_mono = time.monotonic()
        target_interval = 1.0 / max(1, self.fps)
        elapsed = now_mono - self._last_frame_mono
        if elapsed < target_interval:
            time.sleep(target_interval - elapsed)
            now_mono = time.monotonic()
        self._last_frame_mono = now_mono
        now_wall = time.time()

        self._frame_id += 1
        self._fps_counter.tick()

        frame = self._render_frame(self._frame_id, now_wall)

        return TimestampedFrame(
            frame_id=self._frame_id,
            timestamp=now_mono,
            wall_time=now_wall,
            image=frame,
            camera_id=self.camera_id,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _load_sample_assets(self) -> None:
        if self._loaded_assets:
            return
        self._loaded_assets = True

        # Load plate images from dataset
        plate_dir = self._project_root / "data" / "datasets" / "plate" / "images"
        if plate_dir.exists():
            for p in sorted(plate_dir.glob("*.jpg"))[:15]:
                try:
                    img = cv2.imread(str(p))
                    if img is not None and img.size > 0:
                        self._plate_samples.append(img)
                except Exception:
                    pass

        # Load face images from enrollment
        enroll_dir = self._project_root / "data" / "enrollment"
        if enroll_dir.exists():
            for person_p in enroll_dir.iterdir():
                if person_p.is_dir():
                    for f in (person_p / "images").glob("*.jpg"):
                        try:
                            img = cv2.imread(str(f))
                            if img is not None and img.size > 0:
                                self._face_samples.append(img)
                                if len(self._face_samples) >= 8:
                                    break
                        except Exception:
                            pass
                if len(self._face_samples) >= 8:
                    break

    def _render_frame(self, frame_num: int, now_wall: float) -> np.ndarray:
        h, w = self.height, self.width
        # Base countertop: warm slate grey cafeteria counter
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Subtle horizontal gradient
        for y in range(h):
            factor = y / float(h)
            b = int(45 + factor * 25)
            g = int(50 + factor * 25)
            r = int(58 + factor * 25)
            img[y, :] = (b, g, r)

        # Counter partition line / rim
        counter_y = int(h * 0.32)
        cv2.line(img, (0, counter_y), (w, counter_y), (35, 40, 48), 2)
        cv2.rectangle(img, (0, 0), (w, counter_y), (38, 42, 50), -1)

        # Simulation cycle: 140 frames per customer sequence (~7 seconds at 20fps)
        cycle_len = 140
        cycle_pos = frame_num % cycle_len
        cycle_idx = frame_num // cycle_len

        # Tray & Plate positioning in lower ROI (x: 20% to 80%, y: 35% to 92%)
        tray_x1 = int(w * 0.16)
        tray_y1 = int(h * 0.36)
        tray_x2 = int(w * 0.84)
        tray_y2 = int(h * 0.92)

        # Tray motion: arrives during 0..20, stays 21..115, leaves 116..139
        tray_offset_x = 0
        tray_alpha = 1.0
        if cycle_pos < 20:
            tray_offset_x = int((20 - cycle_pos) * (w * 0.04))
            tray_alpha = max(0.1, cycle_pos / 20.0)
        elif cycle_pos > 115:
            tray_offset_x = -int((cycle_pos - 115) * (w * 0.04))
            tray_alpha = max(0.0, 1.0 - (cycle_pos - 115) / 25.0)

        if tray_alpha > 0.05:
            # Draw Cafeteria Tray (rounded blue/charcoal composite)
            tx1 = tray_x1 + tray_offset_x
            tx2 = tray_x2 + tray_offset_x
            if tx2 > 0 and tx1 < w:
                cv2.rectangle(img, (tx1, tray_y1), (tx2, tray_y2), (70, 75, 88), -1)
                cv2.rectangle(img, (tx1, tray_y1), (tx2, tray_y2), (110, 118, 135), 3)
                # Tray inner rim
                cv2.rectangle(img, (tx1 + 10, tray_y1 + 10), (tx2 - 10, tray_y2 - 10), (55, 60, 70), 2)

                # Plate placement inside tray
                plate_cx = (tx1 + tx2) // 2
                plate_cy = (tray_y1 + tray_y2) // 2
                plate_rx = int((tx2 - tx1) * 0.28)
                plate_ry = int((tray_y2 - tray_y1) * 0.38)

                # Composite real plate or procedural ceramic plate
                self._draw_plate(img, plate_cx, plate_cy, plate_rx, plate_ry, cycle_idx, cycle_pos)

        # Face in upper ROI (x: 20% to 80%, y: 5% to 32%)
        # Face appears once tray is settled (frames 25..105)
        if 25 <= cycle_pos <= 105:
            self._draw_customer_face(img, cycle_idx, cycle_pos)

        # Ambient noise & realistic sensor flicker
        noise = np.random.randint(-2, 3, (h, w, 3), dtype=np.int16)
        clipped = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # On-screen camera status overlay
        ts_str = datetime.datetime.fromtimestamp(now_wall).strftime("%Y-%m-%d %H:%M:%S")
        hud_str = f"CAM01 LIVE | {ts_str} | FRAME #{frame_num}"
        cv2.putText(
            clipped,
            hud_str,
            (20, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 195, 210),
            1,
            cv2.LINE_AA,
        )
        return clipped

    def _draw_plate(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        rx: int,
        ry: int,
        cycle_idx: int,
        cycle_pos: int,
    ) -> None:
        """Render a realistic plate with food or sample asset."""
        h, w = img.shape[:2]
        if self._plate_samples:
            sample = self._plate_samples[cycle_idx % len(self._plate_samples)]
            pw = rx * 2
            ph = ry * 2
            if pw > 10 and ph > 10:
                resized = cv2.resize(sample, (pw, ph), interpolation=cv2.INTER_AREA)
                px1 = max(0, cx - rx)
                py1 = max(0, cy - ry)
                px2 = min(w, px1 + pw)
                py2 = min(h, py1 + ph)
                cw = px2 - px1
                ch = py2 - py1
                if cw > 0 and ch > 0:
                    # Circular mask to blend seamlessly into tray
                    mask = np.zeros((ch, cw), dtype=np.uint8)
                    cv2.ellipse(mask, (cw // 2, ch // 2), (cw // 2 - 2, ch // 2 - 2), 0, 0, 360, 255, -1)
                    patch = resized[:ch, :cw]
                    bg = img[py1:py2, px1:px2]
                    img[py1:py2, px1:px2] = np.where(mask[:, :, None] == 255, patch, bg)
                    # Plate border highlight
                    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (220, 225, 230), 2)
                    return

        # Fallback Stainless Steel Cafeteria Plate
        cv2.ellipse(img, (cx + 3, cy + 4), (rx, ry), 0, 0, 360, (28, 30, 36), -1)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (190, 195, 205), -1)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (220, 225, 235), 2)
        inner_rx = int(rx * 0.84)
        inner_ry = int(ry * 0.84)
        cv2.ellipse(img, (cx, cy), (inner_rx, inner_ry), 0, 0, 360, (160, 165, 175), -1)
        cv2.ellipse(img, (cx, cy), (inner_rx, inner_ry), 0, 0, 360, (130, 135, 145), 1)

    def _draw_customer_face(self, img: np.ndarray, cycle_idx: int, cycle_pos: int) -> None:
        """Render customer face in the upper face detection ROI."""
        h, w = img.shape[:2]
        walk_progress = (cycle_pos - 25) / 80.0  # 0.0 to 1.0
        face_cx = int(w * (0.35 + 0.30 * walk_progress))
        face_cy = int(h * 0.18)
        fw = int(w * 0.12)
        fh = int(h * 0.22)

        if self._face_samples:
            sample = self._face_samples[cycle_idx % len(self._face_samples)]
            resized = cv2.resize(sample, (fw, fh), interpolation=cv2.INTER_AREA)
            fx1 = max(0, face_cx - fw // 2)
            fy1 = max(0, face_cy - fh // 2)
            fx2 = min(w, fx1 + fw)
            fy2 = min(h, fy1 + fh)
            rw = fx2 - fx1
            rh = fy2 - fy1
            if rw > 0 and rh > 0:
                mask = np.zeros((rh, rw), dtype=np.uint8)
                cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2 - 2, rh // 2 - 2), 0, 0, 360, 255, -1)
                patch = resized[:rh, :rw]
                bg = img[fy1:fy2, fx1:fx2]
                img[fy1:fy2, fx1:fx2] = np.where(mask[:, :, None] == 255, patch, bg)
                return

        # Neutral silhouette if no enrollment photos exist
        cv2.ellipse(img, (face_cx, face_cy), (fw // 2, fh // 2), 0, 0, 360, (75, 85, 95), -1)
