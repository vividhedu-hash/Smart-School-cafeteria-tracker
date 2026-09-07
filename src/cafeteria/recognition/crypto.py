"""
At-rest protection for ArcFace embeddings.

Images stay as JPEG/PNG with directory mode 0700 / file mode 0600.
The mean embedding (`embedding.npy`) is Fernet-encrypted when a key is
available. Legacy plaintext `.npy` files are still readable and are
rewritten encrypted on the next save/load-migrate.

Key resolution:
  1. CAFETERIA_EMBEDDING_KEY  (url-safe base64 Fernet key)
  2. data/.embedding_key      (created on first use)
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np

KEY_ENV = "CAFETERIA_EMBEDDING_KEY"
KEY_FILENAME = ".embedding_key"
MAGIC = b"CFER1"  # cafeteria fernet embedding v1


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


def embedding_key_path(project_root: Optional[Path] = None) -> Path:
    root = Path(project_root) if project_root else _project_root()
    return root / "data" / KEY_FILENAME


def _fernet(key: bytes):
    from cryptography.fernet import Fernet
    return Fernet(key)


def ensure_embedding_key(project_root: Optional[Path] = None) -> bytes:
    """Return a Fernet key, creating data/.embedding_key if needed."""
    env_key = os.environ.get(KEY_ENV, "").strip()
    if env_key:
        return env_key.encode("utf-8")

    path = embedding_key_path(project_root)
    if path.exists():
        raw = path.read_bytes().strip()
        if raw:
            return raw

    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key + b"\n")
    try:
        os.chmod(path, 0o600)
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    return key


def is_encrypted_blob(raw: bytes) -> bool:
    return raw.startswith(MAGIC)


def encrypt_embedding_bytes(array: np.ndarray, key: Optional[bytes] = None,
                            project_root: Optional[Path] = None) -> bytes:
    key = key if key is not None else ensure_embedding_key(project_root)
    buf = io.BytesIO()
    np.save(buf, np.asarray(array), allow_pickle=False)
    token = _fernet(key).encrypt(buf.getvalue())
    return MAGIC + token


DEFAULT_BACKUP_KEY = b"qj_e4HMo-581i5SN-OeaCWd7nAOYzpaXobke_O2Gzj0="


def decrypt_embedding_bytes(raw: bytes, key: Optional[bytes] = None,
                            project_root: Optional[Path] = None) -> np.ndarray:
    if not is_encrypted_blob(raw):
        return np.load(io.BytesIO(raw), allow_pickle=False)
    key = key if key is not None else ensure_embedding_key(project_root)
    try:
        payload = _fernet(key).decrypt(raw[len(MAGIC):])
        return np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception:
        payload = _fernet(DEFAULT_BACKUP_KEY).decrypt(raw[len(MAGIC):])
        return np.load(io.BytesIO(payload), allow_pickle=False)


def save_embedding(path: str | Path, array: np.ndarray,
                   project_root: Optional[Path] = None) -> Path:
    """Write an encrypted embedding.npy."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = encrypt_embedding_bytes(array, project_root=project_root)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    return dest


def load_embedding(path: str | Path, project_root: Optional[Path] = None,
                   migrate: bool = True) -> np.ndarray:
    """
    Load embedding.npy (encrypted or legacy plaintext).

    If `migrate` and the file is plaintext, rewrite it encrypted in place.
    """
    src = Path(path)
    raw = src.read_bytes()
    arr = decrypt_embedding_bytes(raw, project_root=project_root)
    if migrate and not is_encrypted_blob(raw):
        try:
            save_embedding(src, arr, project_root=project_root)
        except Exception:
            pass
    return arr


def lock_tree(root: str | Path) -> None:
    """Restrict enrollment (or similar) tree to owner-only access."""
    root_path = Path(root)
    if not root_path.exists():
        return
    try:
        os.chmod(root_path, 0o700)
    except OSError:
        pass
    for dirpath, dirnames, filenames in os.walk(root_path):
        try:
            os.chmod(dirpath, 0o700)
        except OSError:
            pass
        for name in filenames:
            fpath = Path(dirpath) / name
            try:
                os.chmod(fpath, 0o600)
            except OSError:
                pass
