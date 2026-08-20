"""
One-command start/stop for the inference engine and Streamlit dashboard.

    ./run.sh
    python run.py
    python -m cafeteria.launcher

Ctrl+C (or ``python run.py --stop``) shuts both down and releases the camera.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional, Sequence

# [AI-CoLab: Cursor] Launcher owns engine lifetime while it is running so a
# demo does not depend on clicking Start Engine after the browser opens.

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8501
HEARTBEAT_INTERVAL_SECONDS = 8.0
DASHBOARD_WAIT_SECONDS = 45.0
MIN_PYTHON = (3, 11)


class LauncherError(RuntimeError):
    """User-facing start/stop failure."""


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    return Path.cwd()


def find_python(root: Path) -> Path:
    """Prefer the project venv; otherwise a Python 3.11+ interpreter."""
    for candidate in (
        root / ".venv" / "bin" / "python",
        root / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return candidate
    return Path(_system_python())


def _system_python() -> str:
    for name in ("python3.11", "python3.12", "python3.13", "python3"):
        found = shutil.which(name)
        if found and _python_version_ok(found):
            return found
    if sys.version_info >= MIN_PYTHON:
        return sys.executable
    raise LauncherError(
        "Python 3.11 or newer is required. Install it, then run ./run.sh again."
    )


def _python_version_ok(executable: str) -> bool:
    try:
        out = subprocess.check_output(
            [executable, "-c", "import sys; print(sys.version_info[:2])"],
            text=True,
            timeout=8,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return False
    try:
        pair = ast.literal_eval(out)
        major, minor = int(pair[0]), int(pair[1])
        return (major, minor) >= MIN_PYTHON
    except (TypeError, ValueError, SyntaxError, IndexError):
        return False


def venv_has_app(python: Path, root: Path) -> bool:
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    try:
        result = subprocess.run(
            [str(python), "-c", "import streamlit, cafeteria"],
            cwd=str(root),
            env=env,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def ensure_install(root: Path) -> Path:
    """Create ``.venv`` and install the package if the app cannot be imported."""
    python = find_python(root)
    if python.exists() and venv_has_app(python, root):
        return python

    venv_dir = root / ".venv"
    if not (venv_dir / "bin" / "python").exists() and not (
        venv_dir / "Scripts" / "python.exe"
    ).exists():
        _log(f"Creating virtual environment at {venv_dir} …", "cyan")
        subprocess.check_call([_system_python(), "-m", "venv", str(venv_dir)])
        python = find_python(root)

    _log("Installing dependencies into .venv (first run only) …", "cyan")
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=str(root),
    )
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "-e", str(root)],
        cwd=str(root),
    )
    if not venv_has_app(python, root):
        raise LauncherError(
            "Install finished but Streamlit/cafeteria still cannot be imported. "
            "See the pip output above."
        )
    return python


def dashboard_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 400
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def wait_for_http(url: str, timeout_s: float = DASHBOARD_WAIT_SECONDS) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if http_ok(url):
            return True
        time.sleep(0.3)
    return False


def looks_like_our_dashboard(cmdline: str) -> bool:
    lowered = cmdline.lower()
    return "streamlit" in lowered and "dashboard/app.py" in lowered.replace("\\", "/")


def pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def cmdline_of(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def streamlit_command(
    python: Path,
    root: Path,
    host: str,
    port: int,
) -> list[str]:
    return [
        str(python),
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.headless=true",
        f"--server.port={port}",
        f"--server.address={host}",
        "--browser.gatherUsageStats=false",
        "--server.enableCORS=false",
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Start the cafeteria engine and dashboard together.",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop the engine and any dashboard this project started, then exit.",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Start the web UI without the camera engine.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Dashboard bind address (default {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Dashboard port (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not create a venv or pip-install; fail if the app is missing.",
    )
    return parser.parse_args(argv)


def _log(msg: str, colour: str = "") -> None:
    reset = "\033[0m"
    codes = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "bold": "\033[1m",
    }
    prefix = codes.get(colour, "")
    print(f"{prefix}{msg}{reset}", flush=True)


def _banner(python: Path, root: Path) -> None:
    import platform

    _log("=" * 62, "bold")
    _log("  Smart Cafeteria Waste Tracker", "bold")
    _log("=" * 62, "bold")
    _log(f"  Platform : {platform.system()} {platform.machine()}")
    _log(f"  Python   : {python}")
    _log(f"  Root     : {root}")
    _log("=" * 62, "bold")


def _engine_ctl():
    dashboard = str(project_root() / "dashboard")
    if dashboard not in sys.path:
        sys.path.insert(0, dashboard)
    from engine_ctl import start_engine, stop_engine, write_heartbeat

    return start_engine, stop_engine, write_heartbeat


def write_launcher_heartbeat(root: Path) -> None:
    path = root / "data" / "heartbeat"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass
    try:
        _, _, write_heartbeat = _engine_ctl()
        write_heartbeat()
    except Exception:
        pass


def _heartbeat_loop(root: Path, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        write_launcher_heartbeat(root)
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


def free_our_dashboard_port(host: str, port: int) -> None:
    """Stop a leftover dashboard from this project; refuse anyone else's port."""
    if not port_in_use(host, port):
        return

    ours = False
    for pid in pids_on_port(port):
        if looks_like_our_dashboard(cmdline_of(pid)):
            ours = True
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    if ours:
        deadline = time.time() + 5.0
        while time.time() < deadline and port_in_use(host, port):
            time.sleep(0.15)
        for pid in pids_on_port(port):
            if looks_like_our_dashboard(cmdline_of(pid)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
        time.sleep(0.2)

    if port_in_use(host, port):
        raise LauncherError(
            f"Port {port} is already in use. Stop the other process, or run:\n"
            f"  python run.py --stop\n"
            f"  python run.py --port {port + 1}"
        )


def stop_stack(root: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    try:
        _, stop_engine, _ = _engine_ctl()
        stop_engine()
    except Exception:
        pid_file = root / "data" / "engine.pid"
        pid_file.unlink(missing_ok=True)

    for pid in pids_on_port(port):
        if looks_like_our_dashboard(cmdline_of(pid)):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    time.sleep(0.4)
    for pid in pids_on_port(port):
        if looks_like_our_dashboard(cmdline_of(pid)):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    _log("  All services stopped.", "green")


def start_stack(
    root: Path,
    *,
    python: Path,
    host: str,
    port: int,
    start_engine_process: bool,
    open_browser: bool,
) -> int:
    url = dashboard_url(host, port)
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTHONUNBUFFERED": "1",
    }

    free_our_dashboard_port(host, port)

    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(root, heartbeat_stop),
        daemon=True,
        name="launcher-heartbeat",
    )

    if start_engine_process:
        _log("\n  [1/2] Starting inference engine (camera)…", "cyan")
        write_launcher_heartbeat(root)
        start_engine, _, _ = _engine_ctl()
        pid = start_engine()
        if pid:
            _log(f"        Engine pid {pid}", "green")
        else:
            _log(
                "        Engine did not stay up — dashboard will still open. "
                "Use ▶ Start Engine on the home page if the camera is free.",
                "yellow",
            )
        heartbeat_thread.start()
    else:
        _log("\n  [1/2] Engine skipped (--dashboard-only).", "cyan")

    _log(f"  [2/2] Starting dashboard → {url}", "cyan")
    cmd = streamlit_command(python, root, host, port)

    def _open_when_ready() -> None:
        if wait_for_http(url):
            _log(f"\n  Dashboard ready → {url}", "green")
            _log("  PIN gate is on the first page. Ctrl+C stops everything.\n", "green")
            if open_browser:
                webbrowser.open(url)
        else:
            _log(
                f"\n  Dashboard did not answer at {url} in time. "
                "Leave this window open and try the URL manually.",
                "yellow",
            )

    opener = threading.Thread(target=_open_when_ready, daemon=True)
    opener.start()

    try:
        completed = subprocess.run(cmd, cwd=str(root), env=env)
        return int(completed.returncode or 0)
    except KeyboardInterrupt:
        _log("\n\n  Shutdown requested…", "yellow")
        return 0
    finally:
        heartbeat_stop.set()
        stop_stack(root, host, port)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # [AI-CoLab: Verified by Antigravity] Unified stack launcher managing engine & Streamlit lifecycle
    args = parse_args(argv)
    root = project_root()

    if args.stop:
        _log("Stopping cafeteria engine and dashboard…", "yellow")
        stop_stack(root, args.host, args.port)
        return 0

    try:
        python = find_python(root) if args.skip_install else ensure_install(root)
        if args.skip_install and not venv_has_app(python, root):
            raise LauncherError(
                "App packages are missing. Re-run without --skip-install, or:\n"
                "  .venv/bin/pip install -e ."
            )
    except LauncherError as exc:
        _log(f"\n  {exc}", "red")
        return 1

    _banner(python, root)
    try:
        return start_stack(
            root,
            python=python,
            host=args.host,
            port=args.port,
            start_engine_process=not args.dashboard_only,
            open_browser=not args.no_browser,
        )
    except LauncherError as exc:
        _log(f"\n  {exc}", "red")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
