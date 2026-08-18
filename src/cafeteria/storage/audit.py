"""Append-only JSONL audit log for roster / enrollment changes."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def append_audit(
    log_path: str | Path,
    *,
    action: str,
    person_id: Optional[str] = None,
    actor: str = "dashboard",
    **details: Any,
) -> None:
    """Append one JSON line. Never stores image bytes or embeddings."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "action": action,
        "person_id": person_id,
        "actor": actor,
        **{k: v for k, v in details.items() if v is not None},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
