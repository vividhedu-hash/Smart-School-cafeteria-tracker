"""Engine HTTP API health + state (in-process server)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from cafeteria.api.server import start_engine_api


def test_engine_api_health_and_state(tmp_path):
    frame = tmp_path / "latest.jpg"
    frame.write_bytes(b"\xff\xd8\xff")  # tiny JPEG-ish
    hb = tmp_path / "heartbeat"
    cmd = tmp_path / "commands.json"
    received = {}

    def get_state():
        return {"state": "IDLE", "fps": 12.0, "live_face_match": None}

    def on_command(c):
        received.update(c)

    server = start_engine_api(
        host="127.0.0.1",
        port=0,
        get_state=get_state,
        frame_path=frame,
        heartbeat_path=hb,
        commands_path=cmd,
        on_command=on_command,
    )
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            health = json.loads(resp.read().decode())
        assert health["ok"] is True
        assert health["service"] == "cafeteria-engine"

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2) as resp:
            state = json.loads(resp.read().decode())
        assert state["state"] == "IDLE"
        assert state["api"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/heartbeat",
            data=b"{}",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert json.loads(resp.read().decode())["ok"] is True
        assert hb.exists()

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/command",
            data=json.dumps({"type": "reload_embeddings"}).encode(),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=2) as resp:
            assert json.loads(resp.read().decode())["ok"] is True
        assert received["type"] == "reload_embeddings"
    finally:
        server.shutdown()
        server.server_close()


def test_engine_api_rejects_missing_token(tmp_path):
    frame = tmp_path / "latest.jpg"
    frame.write_bytes(b"\xff\xd8\xff")
    server = start_engine_api(
        host="127.0.0.1",
        port=0,
        get_state=lambda: {"state": "IDLE"},
        frame_path=frame,
        heartbeat_path=tmp_path / "heartbeat",
        commands_path=tmp_path / "commands.json",
        token="secret-token",
    )
    port = server.server_address[1]
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=2)
            assert False, "expected 401"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        req = urllib.request.Request(f"http://127.0.0.1:{port}/state")
        req.add_header("X-Engine-Token", "secret-token")
        with urllib.request.urlopen(req, timeout=2) as resp:
            payload = json.loads(resp.read().decode())
        assert payload["state"] == "IDLE"
    finally:
        server.shutdown()
        server.server_close()
