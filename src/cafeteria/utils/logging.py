"""
Logging configuration for Smart Cafeteria Waste Tracker.

Uses Python's standard logging library with:
  - Rotating file handler (logs/app.log)
  - Coloured console output
  - Structured event codes for key pipeline events
  - Cross-platform (no ANSI on Windows unless colorama is present)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Event codes (logged as extra field for easy grepping)
# ──────────────────────────────────────────────────────────────────────────────

class EventCode:
    APPLICATION_START        = "APPLICATION_START"
    CAMERA_CONNECTED         = "CAMERA_CONNECTED"
    CAMERA_DISCONNECTED      = "CAMERA_DISCONNECTED"
    CAMERA_RECONNECTING      = "CAMERA_RECONNECTING"
    PLATE_DETECTED           = "PLATE_DETECTED"
    PLATE_ABSENT             = "PLATE_ABSENT"
    FOOD_DETECTED            = "FOOD_DETECTED"
    FOOD_ABSENT              = "FOOD_ABSENT"
    WASTE_EVENT              = "WASTE_EVENT"
    FACE_DETECTED            = "FACE_DETECTED"
    FACE_MATCHED             = "FACE_MATCHED"
    FACE_UNKNOWN             = "FACE_UNKNOWN"
    TRANSACTION_CREATED      = "TRANSACTION_CREATED"
    REVIEW_CREATED           = "REVIEW_CREATED"
    REVIEW_RESOLVED          = "REVIEW_RESOLVED"
    MODEL_TRAINING_STARTED   = "MODEL_TRAINING_STARTED"
    MODEL_TRAINING_COMPLETED = "MODEL_TRAINING_COMPLETED"
    MODEL_ACTIVATED          = "MODEL_ACTIVATED"
    APPLICATION_ERROR        = "APPLICATION_ERROR"
    STATE_TRANSITION         = "STATE_TRANSITION"


# ──────────────────────────────────────────────────────────────────────────────
# Colour support (safe on all platforms)
# ──────────────────────────────────────────────────────────────────────────────

_COLOURS_ENABLED = False

def _enable_colours() -> bool:
    """Return True if the terminal supports ANSI colours."""
    if platform.system() == "Windows":
        # Try to enable VT100 on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    # macOS / Linux: check if running in a real terminal
    return sys.stdout.isatty()


_COLOUR_MAP = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colour codes to levelname."""

    BASE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_colour: bool = True):
        super().__init__(fmt=self.BASE_FMT, datefmt=self.DATE_FMT)
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        if self._use_colour:
            colour = _COLOUR_MAP.get(record.levelname, "")
            record.levelname = f"{colour}{record.levelname}{_RESET}"
        return super().format(record)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

_configured = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    app_name: str = "cafeteria",
) -> logging.Logger:
    """
    Configure root logger once. Safe to call multiple times.

    Returns the application root logger.
    """
    global _configured, _COLOURS_ENABLED

    if _configured:
        return logging.getLogger(app_name)

    _COLOURS_ENABLED = _enable_colours()

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any default handlers
    root_logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    console.setFormatter(ColourFormatter(use_colour=_COLOURS_ENABLED))
    root_logger.addHandler(console)

    # File handler (rotating, 10 MB × 5 files)
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    # Silence noisy third-party libraries
    for noisy in ("ultralytics", "insightface", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger(app_name)


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger. Assumes setup_logging() was already called."""
    return logging.getLogger(f"cafeteria.{name}")


def log_event(logger: logging.Logger, code: str, message: str, **kwargs: Any) -> None:
    """Log a structured pipeline event."""
    extra_str = "  ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{code}] {message}  {extra_str}".rstrip())
