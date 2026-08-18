"""
Local operator authentication for the dashboard.

Not SSO / LDAP. A PIN stored outside git, overridable by env.

Resolution order:
  1. CAFETERIA_OPERATOR_PIN
  2. data/.operator_pin (created on first run if missing)
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PIN_ENV = "CAFETERIA_OPERATOR_PIN"
PIN_FILENAME = ".operator_pin"
ENGINE_TOKEN_ENV = "CAFETERIA_ENGINE_TOKEN"
ENGINE_TOKEN_FILENAME = ".engine_token"


@dataclass(frozen=True)
class OperatorPinStatus:
    configured: bool
    generated: bool
    pin: Optional[str]
    source: str  # "env" | "file" | "none"
    path: Optional[Path]


def _project_root() -> Path:
    env_root = os.environ.get("CAFETERIA_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    cfg = os.environ.get("CAFETERIA_CONFIG", "").strip()
    if cfg:
        return Path(cfg).resolve().parent.parent
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    return Path.cwd()


def operator_pin_path(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root else _project_root()
    return root / "data" / PIN_FILENAME


def _read_stored_pin(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        pin = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return pin or None


def _write_pin_file(path: Path, pin: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pin + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
        os.chmod(path.parent, 0o700)
    except OSError:
        pass


def resolve_operator_pin(project_root: Optional[Path] = None) -> OperatorPinStatus:
    """Return the configured PIN without creating a new one."""
    env_pin = os.environ.get(PIN_ENV, "").strip()
    if env_pin:
        return OperatorPinStatus(
            configured=True,
            generated=False,
            pin=env_pin,
            source="env",
            path=None,
        )
    path = operator_pin_path(project_root)
    stored = _read_stored_pin(path)
    if stored:
        return OperatorPinStatus(
            configured=True,
            generated=False,
            pin=stored,
            source="file",
            path=path,
        )
    return OperatorPinStatus(
        configured=False,
        generated=False,
        pin=None,
        source="none",
        path=path,
    )


def ensure_operator_pin(project_root: Optional[Path] = None) -> OperatorPinStatus:
    """
    Return a usable PIN, generating and storing one if nothing is configured.

    The generated PIN is a 6-digit number. It is written to data/.operator_pin
    (gitignored) with mode 0600.
    """
    existing = resolve_operator_pin(project_root)
    if existing.configured:
        return existing
    path = existing.path or operator_pin_path(project_root)
    pin = f"{secrets.randbelow(1_000_000):06d}"
    _write_pin_file(path, pin)
    return OperatorPinStatus(
        configured=True,
        generated=True,
        pin=pin,
        source="file",
        path=path,
    )


def engine_token_path(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root else _project_root()
    return root / "data" / ENGINE_TOKEN_FILENAME


def ensure_engine_token(project_root: Optional[Path] = None) -> str:
    """
    Shared secret for dashboard ↔ engine HTTP.

    Resolution: CAFETERIA_ENGINE_TOKEN, else data/.engine_token (created once).
    """
    env_token = os.environ.get(ENGINE_TOKEN_ENV, "").strip()
    if env_token:
        return env_token
    path = engine_token_path(project_root)
    existing = _read_stored_pin(path)
    if existing:
        return existing
    token = secrets.token_urlsafe(32)
    _write_pin_file(path, token)
    return token


def verify_operator_pin(candidate: str, project_root: Optional[Path] = None) -> bool:
    """Constant-time-ish compare against the configured PIN."""
    status = ensure_operator_pin(project_root)
    expected = status.pin or ""
    got = (candidate or "").strip()
    if not expected or not got:
        return False
    return secrets.compare_digest(got, expected)
