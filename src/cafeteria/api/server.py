"""
Local engine HTTP API (stdlib — no extra web framework required).

The inference process owns camera + models and serves:

  GET  /health
  GET  /state
  GET  /frame.jpg
  POST /heartbeat
  POST /command

Dashboard is a client. JSON files remain as a fallback if the API is down.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from cafeteria.utils.logging import get_logger

logger = get_logger("api.server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class EngineAPIState:
    """Shared callbacks + paths for the HTTP handlers."""

    def __init__(
        self,
        *,
        get_state: Callable[[], dict],
        frame_path: Path,
        heartbeat_path: Path,
        commands_path: Path,
        on_command: Optional[Callable[[dict], None]] = None,
        token: Optional[str] = None,
    ) -> None:
        self.get_state = get_state
        self.frame_path = Path(frame_path)
        self.heartbeat_path = Path(heartbeat_path)
        self.commands_path = Path(commands_path)
        self.on_command = on_command
        self.token = token
        self.started_at = time.time()
        self.pid = os.getpid()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


def make_handler(api: EngineAPIState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("API %s", fmt % args)

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, code: int, payload: dict) -> None:
            self._send(code, _json_bytes(payload), "application/json")

        def _authorized(self) -> bool:
            if not api.token:
                return True
            got = (self.headers.get("X-Engine-Token") or "").strip()
            return secrets.compare_digest(got, api.token)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                self._send_json(200, {
                    "ok": True,
                    "service": "cafeteria-engine",
                    "pid": api.pid,
                    "uptime_s": round(time.time() - api.started_at, 1),
                })
                return
            if not self._authorized():
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            if path == "/state":
                try:
                    payload = api.get_state() or {}
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                payload.setdefault("ok", True)
                payload["api"] = True
                payload["engine_pid"] = api.pid
                self._send_json(200, payload)
                return
            if path in ("/frame", "/frame.jpg"):
                if not api.frame_path.exists():
                    self._send_json(404, {"ok": False, "error": "no frame yet"})
                    return
                try:
                    body = api.frame_path.read_bytes()
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send(200, body, "image/jpeg")
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path.rstrip("/") or "/"
            if not self._authorized():
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if path == "/heartbeat":
                try:
                    api.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                    api.heartbeat_path.write_text(str(time.time()), encoding="utf-8")
                except OSError as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True})
                return
            if path == "/command":
                try:
                    cmd = json.loads(raw.decode("utf-8") or "{}")
                    if not isinstance(cmd, dict):
                        raise ValueError("command must be a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    self._send_json(400, {"ok": False, "error": str(exc)})
                    return
                if api.on_command is not None:
                    try:
                        api.on_command(cmd)
                    except Exception as exc:
                        self._send_json(500, {"ok": False, "error": str(exc)})
                        return
                else:
                    try:
                        api.commands_path.parent.mkdir(parents=True, exist_ok=True)
                        api.commands_path.write_text(json.dumps(cmd), encoding="utf-8")
                    except OSError as exc:
                        self._send_json(500, {"ok": False, "error": str(exc)})
                        return
                self._send_json(200, {"ok": True, "type": cmd.get("type")})
                return
            self._send_json(404, {"ok": False, "error": "not found"})

    return Handler


def start_engine_api(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    get_state: Callable[[], dict],
    frame_path: Path,
    heartbeat_path: Path,
    commands_path: Path,
    on_command: Optional[Callable[[dict], None]] = None,
    token: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Bind and serve in a daemon thread. Raises OSError if the port is taken."""
    api = EngineAPIState(
        get_state=get_state,
        frame_path=frame_path,
        heartbeat_path=heartbeat_path,
        commands_path=commands_path,
        on_command=on_command,
        token=token,
    )
    server = ThreadingHTTPServer((host, int(port)), make_handler(api))
    server.daemon_threads = True
    thread = threading.Thread(
        target=server.serve_forever,
        name="engine-api",
        daemon=True,
    )
    thread.start()
    logger.info("Engine API listening on http://%s:%s", host, port)
    return server
