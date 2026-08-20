"""One-command launcher helpers — no live camera or Streamlit required."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cafeteria.launcher import (
    dashboard_url,
    find_python,
    http_ok,
    looks_like_our_dashboard,
    parse_args,
    streamlit_command,
    wait_for_http,
    write_launcher_heartbeat,
)


def test_parse_args_defaults():
    # [AI-CoLab: Verified by Antigravity] Verified launcher CLI arguments and HTTP health checks
    args = parse_args([])
    assert args.stop is False
    assert args.dashboard_only is False
    assert args.no_browser is False
    assert args.port == 8501
    assert args.host == "127.0.0.1"


def test_parse_args_stop_and_port():
    args = parse_args(["--stop", "--port", "8599", "--no-browser"])
    assert args.stop is True
    assert args.port == 8599
    assert args.no_browser is True


def test_find_python_prefers_venv(tmp_path: Path):
    unix = tmp_path / ".venv" / "bin" / "python"
    unix.parent.mkdir(parents=True)
    unix.write_text("#!/bin/sh\n")
    unix.chmod(0o755)
    assert find_python(tmp_path) == unix


def test_looks_like_our_dashboard():
    assert looks_like_our_dashboard(
        "/app/.venv/bin/python -m streamlit run dashboard/app.py --server.port=8501"
    )
    assert not looks_like_our_dashboard("nginx: worker process")
    assert not looks_like_our_dashboard("python -m streamlit run some_other_app.py")


def test_streamlit_command_points_at_this_app(tmp_path: Path):
    python = tmp_path / "python"
    cmd = streamlit_command(python, tmp_path, "127.0.0.1", 8501)
    assert cmd[0] == str(python)
    assert "streamlit" in cmd
    assert "dashboard/app.py" in cmd
    assert "--server.port=8501" in cmd
    assert "--server.address=127.0.0.1" in cmd


def test_dashboard_url():
    assert dashboard_url() == "http://127.0.0.1:8501"
    assert dashboard_url("0.0.0.0", 9000) == "http://0.0.0.0:9000"


def test_wait_for_http_succeeds_on_local_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}/"
        assert http_ok(url)
        assert wait_for_http(url, timeout_s=2.0) is True
        assert wait_for_http("http://127.0.0.1:1/", timeout_s=0.4) is False
    finally:
        server.shutdown()
        server.server_close()


def test_write_launcher_heartbeat_creates_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "cafeteria.launcher._engine_ctl",
        lambda: (lambda: None, lambda: None, lambda: None),
    )
    write_launcher_heartbeat(tmp_path)
    path = tmp_path / "data" / "heartbeat"
    assert path.exists()
    assert float(path.read_text()) > 0
