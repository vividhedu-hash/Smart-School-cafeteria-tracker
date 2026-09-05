"""
Configuration system — loads configs/config.yaml, applies .env overrides,
and exposes a typed Settings object used throughout the application.

Device detection order: CUDA → MPS (Apple Silicon) → CPU
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Device detection (cross-platform)
# ──────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    """
    Returns the best available compute device string.

    Priority:
      1. CAFETERIA_DEVICE env-var override (e.g. "cpu", "cuda:0", "mps")
      2. CUDA (NVIDIA GPU)
      3. MPS  (Apple Silicon — macOS only)
      4. CPU  (universal fallback)
    """
    override = os.environ.get("CAFETERIA_DEVICE", "").strip().lower()
    if override:
        return override

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def detect_opencv_backend() -> int:
    """Return the best OpenCV VideoCapture backend for the current OS."""
    import cv2

    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Windows":
        return cv2.CAP_DSHOW
    else:  # Linux and everything else
        return cv2.CAP_V4L2


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic sub-models
# ──────────────────────────────────────────────────────────────────────────────

class AppSettings(BaseModel):
    name: str = "Smart Cafeteria Waste Tracker"
    debug: bool = True
    log_level: str = "INFO"
    runtime_state_path: str = "data/runtime_state.json"
    commands_path: str = "data/commands.json"
    api_host: str = "127.0.0.1"
    api_port: int = 8765


class CameraSettings(BaseModel):
    mode: str = "webcam"      # webcam | rtsp
    source: Any = "auto"      # auto | int index | RTSP URL
    width: int = 0            # 0 / auto = camera native (capped)
    height: int = 0
    fps: int = 30
    buffer_size: int = 5
    reconnect_delay_seconds: float = 3.0
    reconnect_max_attempts: int = 10
    max_capture_width: int = 1920  # 4K webcams are stepped down for realtime

    @field_validator("source", mode="before")
    @classmethod
    def coerce_source(cls, v: Any) -> Any:
        if v is None:
            return "auto"
        if isinstance(v, str):
            raw = v.strip()
            if raw.lower() in ("auto", "", "default"):
                return "auto"
            if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()):
                return int(raw)
            return raw
        return v

    @field_validator("width", "height", "fps", "max_capture_width", mode="before")
    @classmethod
    def coerce_auto_int(cls, v: Any) -> Any:
        if v is None:
            return 0
        if isinstance(v, str) and v.strip().lower() in ("auto", ""):
            return 0
        return v


class ROIBox(BaseModel):
    x1: float = 0.05
    y1: float = 0.00
    x2: float = 0.95
    y2: float = 1.00

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel coords."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )


class ROISettings(BaseModel):
    plate: ROIBox = Field(default_factory=lambda: ROIBox(x1=0.05, y1=0.40, x2=0.95, y2=1.00))
    face: ROIBox = Field(default_factory=lambda: ROIBox(x1=0.05, y1=0.00, x2=0.95, y2=0.55))


class ModelConfig(BaseModel):
    weights: str
    confidence: float = 0.55
    iou: float = 0.50
    classes: Optional[Any] = None
    # Plate only: explicit opt-in for COCO "proxy mode" when trained weights
    # are missing. Default False = do not silently use COCO-as-plate.
    allow_coco_fallback: bool = False
    # Default True: missing YOLO weights use the built-in OpenCV visual
    # detector so the live pipeline still runs. Set false to require training.
    allow_visual_fallback: bool = True


class ModelsSettings(BaseModel):
    plate: ModelConfig = Field(
        default_factory=lambda: ModelConfig(weights="models/plate/best.pt")
    )
    waste: ModelConfig = Field(
        default_factory=lambda: ModelConfig(weights="models/waste/best.pt")
    )


class RecognitionSettings(BaseModel):
    model_pack: str = "buffalo_l"
    similarity_threshold: float = 0.52
    frames_to_vote: int = 5
    minimum_face_size: int = 24
    identify_face_size: int = 40
    max_event_duration_seconds: float = 4.0
    embedding_dir: str = "data/enrollment"
    det_size: list[int] = Field(default_factory=lambda: [480, 480])
    det_thresh: float = 0.40
    infer_max_width: int = 640
    bbox_hold_frames: int = 12
    bbox_hold_seconds: float = 0.90
    motion_pad_ratio: float = 0.90


class EventSettings(BaseModel):
    minimum_plate_presence_seconds: float = 0.20
    cooldown_seconds: float = 2.0
    timeout_seconds: float = 4.0
    best_frame_window: int = 10


class StorageSettings(BaseModel):
    database: str = "database/cafeteria.db"
    database_url: Optional[str] = None  # e.g. postgresql://user:pass@host:5432/dbname (Supabase / Postgres)
    captures: str = "data/captures"
    review_queue: str = "data/review_queue"
    datasets: str = "data/datasets"
    models: str = "models"
    frames: str = "data/frames"
    exports: str = "data/exports"


class MlLoopSettings(BaseModel):
    """Closed-loop learning: live crops → labels → retrain → activate."""
    enabled: bool = True
    auto_enqueue: bool = True
    min_confidence_to_enqueue: float = 0.40
    promote_auto_confirmed: bool = True
    auto_confirm_min_confidence: float = 0.85
    retrain_after_n_labels: int = 8
    min_images_per_class: int = 4
    auto_activate: bool = True
    auto_activate_min_accuracy: float = 0.50
    retrain_epochs: int = 15
    retrain_image_size: int = 224
    retrain_batch: int = 8


class TrainingSettings(BaseModel):
    default_base_model: str = "yolov8n-cls.pt"
    plate_base_model: str = "yolov8n.pt"
    default_epochs: int = 50
    default_image_size: int = 640
    default_batch: int = 16
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    test_ratio: float = 0.10
    random_seed: int = 42
    output_base: str = "models"
    ml_loop: MlLoopSettings = Field(default_factory=MlLoopSettings)


class MonitoringSettings(BaseModel):
    state_write_interval_seconds: float = 0.5
    metrics_history_size: int = 300


class InferenceSettings(BaseModel):
    frame_skip: int = 2
    debug_window: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Root settings
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseModel):
    application: AppSettings = Field(default_factory=AppSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    roi: ROISettings = Field(default_factory=ROISettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    event: EventSettings = Field(default_factory=EventSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)

    # Runtime-resolved (not from YAML, set after load)
    device: str = Field(default_factory=detect_device)
    project_root: Path = Field(default_factory=Path.cwd)
    # Which YAML file these settings actually came from (None = pure defaults)
    config_source: Optional[Path] = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def ensure_directories(self) -> "Settings":
        """Create all required directories relative to project_root."""
        dirs = [
            self.storage.captures,
            self.storage.review_queue,
            self.storage.datasets,
            self.storage.frames,
            self.storage.exports,
            self.recognition.embedding_dir,
            "database",
            "logs",
            "models/face",
            "models/plate",
            "models/waste",
            "data/datasets/waste/EMPTY",
            "data/datasets/waste/LOW_WASTE",
            "data/datasets/waste/MEDIUM_WASTE",
            "data/datasets/waste/HIGH_WASTE",
            "data/datasets/plate/images",
            "data/datasets/plate/labels",
            "data/ml_loop/pending",
            "data/ml_loop/promoted",
        ]
        for d in dirs:
            (self.project_root / d).mkdir(parents=True, exist_ok=True)
        return self

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self.project_root / p


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

_settings_cache: Optional[Settings] = None


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Load settings from YAML, apply environment overrides, detect device.
    Cached after first call; pass config_path=None to use cache.
    """
    global _settings_cache

    if _settings_cache is not None:
        if config_path is None:
            return _settings_cache
        # Same file as the cached load → reuse cache (keeps every process
        # consistent no matter how many modules call load_settings(path)).
        if (_settings_cache.config_source is not None
                and Path(config_path).resolve() == _settings_cache.config_source):
            return _settings_cache

    # Determine config file
    if config_path is None:
        env_cfg = os.environ.get("CAFETERIA_CONFIG")
        if env_cfg:
            config_path = Path(env_cfg)
        else:
            # Look relative to CWD first, then relative to this package
            # (src/cafeteria/config/settings.py → project root is parents[3]).
            # The package-relative candidate guarantees the YAML is found no
            # matter which directory the engine was launched from.
            cwd = Path.cwd()
            pkg_root = Path(__file__).resolve().parents[3]
            candidates = [
                cwd / "configs" / "config.yaml",
                cwd.parent / "configs" / "config.yaml",
                pkg_root / "configs" / "config.yaml",
            ]
            for c in candidates:
                if c.exists():
                    config_path = c
                    break

    # Build raw dict from YAML
    raw: dict = {}
    config_found = config_path is not None and Path(config_path).exists()
    if config_found:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        import logging
        logging.getLogger("cafeteria.config").warning(
            "No config.yaml found (looked for %s) — using built-in defaults. "
            "Defaults may differ from configs/config.yaml (e.g. "
            "recognition.model_pack, det_size, camera settings).",
            config_path,
        )

    # Determine project root (directory that contains configs/)
    if config_found:
        project_root = Path(config_path).resolve().parent.parent
    else:
        project_root = Path.cwd()

    settings = Settings.model_validate({
        **raw,
        "project_root": project_root,
        "config_source": Path(config_path).resolve() if config_found else None,
    })
    _settings_cache = settings
    return settings


def get_settings() -> Settings:
    """Return cached settings (calls load_settings if not yet loaded)."""
    return load_settings()


def reset_settings_cache() -> None:
    """Clear the settings cache — used in tests."""
    global _settings_cache
    _settings_cache = None


def describe_device(device: str) -> str:
    """Return a human-readable description of the compute device."""
    if device.startswith("cuda"):
        try:
            import torch
            idx = int(device.split(":")[-1]) if ":" in device else 0
            name = torch.cuda.get_device_name(idx)
            mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            return f"CUDA — {name} ({mem_gb:.1f} GB VRAM)"
        except Exception:
            return f"CUDA GPU ({device})"
    elif device == "mps":
        return "Apple Silicon GPU (MPS)"
    else:
        cpu_count = os.cpu_count() or 1
        return f"CPU ({cpu_count} logical cores)"


def system_info() -> dict[str, str]:
    """Return a dict of system information for logging/display."""
    settings = get_settings()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "device": describe_device(settings.device),
        "project_root": str(settings.project_root),
    }
