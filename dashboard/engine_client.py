"""Dashboard → engine HTTP client. JSON files are fallback only."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _DASHBOARD_DIR.parent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
TOKEN_HEADER = "X-Engine-Token"


def api_base_url(host: Optional[str] = None, port: Optional[int] = None) -> str:
    h = host or DEFAULT_HOST
    p = int(port or DEFAULT_PORT)
    return f"http://{h}:{p}"


def _engine_token() -> Optional[str]:
    env = os.environ.get("CAFETERIA_ENGINE_TOKEN", "").strip()
    if env:
        return env
    try:
        from cafeteria.auth import ensure_engine_token
        return ensure_engine_token(PROJECT_ROOT)
    except Exception:
        path = PROJECT_ROOT / "data" / ".engine_token"
        if path.exists():
            return path.read_text(encoding="utf-8").strip() or None
        return None


def _urlopen(url: str, data: Optional[bytes] = None, timeout: float = 0.6):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    token = _engine_token()
    if token:
        req.add_header(TOKEN_HEADER, token)
    return urllib.request.urlopen(req, timeout=timeout)


def api_health(host: Optional[str] = None, port: Optional[int] = None,
               timeout: float = 0.6) -> Optional[dict]:
    try:
        with _urlopen(f"{api_base_url(host, port)}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return None


def api_is_up(host: Optional[str] = None, port: Optional[int] = None) -> bool:
    payload = api_health(host, port)
    return bool(payload and payload.get("ok"))


def fetch_state(
    fallback_path: str | Path,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> dict:
    """Prefer live engine API; fall back to runtime_state.json."""
    try:
        with _urlopen(f"{api_base_url(host, port)}/state", timeout=0.5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                payload["via"] = "api"
                return payload
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        pass
    from cafeteria.monitoring.metrics import read_runtime_state
    data = read_runtime_state(fallback_path) or {}
    if data:
        data["via"] = "file"
    return data


def fetch_frame_bytes(
    fallback_path: str | Path,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> Optional[bytes]:
    """Prefer GET /frame.jpg; fall back to data/frames/latest.jpg."""
    try:
        with _urlopen(f"{api_base_url(host, port)}/frame.jpg", timeout=0.6) as resp:
            body = resp.read()
            if body:
                return body
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    path = Path(fallback_path)
    if path.exists():
        try:
            return path.read_bytes()
        except OSError:
            return None
    return None


def fetch_raw_frame_bytes(
    # [AI-CoLab: Verified by Antigravity] Fetch un-annotated raw JPEG frame for enrollment camera sharing
    host: Optional[str] = None,
    port: Optional[int] = None,
    fallback_path: Optional[str | Path] = None,
) -> Optional[bytes]:
    """
    Clean, un-annotated camera frame from the engine.

    Used by enrollment and dataset capture so they can share the camera the
    engine already owns instead of opening a second capture device.
    """
    try:
        with _urlopen(f"{api_base_url(host, port)}/raw.jpg", timeout=0.8) as resp:
            body = resp.read()
            if body:
                return body
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    if fallback_path:
        path = Path(fallback_path)
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                return None
    return None


def post_heartbeat(host: Optional[str] = None, port: Optional[int] = None) -> bool:
    try:
        with _urlopen(f"{api_base_url(host, port)}/heartbeat", data=b"{}", timeout=0.4):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def post_command(command: dict, host: Optional[str] = None,
                 port: Optional[int] = None) -> bool:
    body = json.dumps(command).encode("utf-8")
    try:
        with _urlopen(f"{api_base_url(host, port)}/command", data=body, timeout=1.0):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_api(host: Optional[str] = None, port: Optional[int] = None,
                 timeout_s: float = 8.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if api_is_up(host, port):
            return True
        time.sleep(0.2)
    return False
