"""
Model registry — manages model versioning and activation.

Reads/writes models/registry.json.
The inference engine polls this file and hot-swaps when the active version changes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from cafeteria.utils.logging import get_logger, EventCode, log_event

logger = get_logger("training.registry")


class ModelRegistry:
    """
    File-backed model registry.

    Schema (registry.json):
        {
            "plate": {
                "active_version": "v002",
                "versions": [
                    {
                        "version": "v001",
                        "weights_path": "models/plate/v001/weights/best.pt",
                        "trained_at": 1234567890.0,
                        "metrics": {...},
                        "is_active": false
                    },
                    ...
                ]
            },
            "waste": { ... }
        }

    Args:
        registry_path: Path to registry.json.
    """

    def __init__(self, registry_path: str | Path) -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {"plate": {"active_version": None, "versions": []},
                "waste": {"active_version": None, "versions": []}}

    def _save(self, data: dict) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._path)

    def register_version(
        self,
        task: str,
        version: str,
        weights_path: str,
        metrics: Optional[dict] = None,
        config: Optional[dict] = None,
        dataset_stats: Optional[dict] = None,
    ) -> None:
        """Add a new version to the registry (does NOT activate it)."""
        data = self._load()
        if task not in data:
            data[task] = {"active_version": None, "versions": []}

        entry = {
            "version": version,
            "weights_path": str(weights_path),
            "trained_at": time.time(),
            "metrics": metrics or {},
            "config": config or {},
            "dataset_stats": dataset_stats or {},
            "is_active": False,
        }

        # Remove existing entry for this version if present
        data[task]["versions"] = [
            v for v in data[task]["versions"] if v["version"] != version
        ]
        data[task]["versions"].append(entry)
        self._save(data)
        logger.info("Registered model version: task=%s  version=%s", task, version)

    def activate(self, task: str, version: str) -> bool:
        """
        Mark a version as active. The inference engine will detect this and hot-swap.

        Returns:
            True if version found and activated, False otherwise.
        """
        data = self._load()
        if task not in data:
            return False

        found = False
        for v in data[task]["versions"]:
            v["is_active"] = v["version"] == version
            if v["version"] == version:
                found = True

        if found:
            data[task]["active_version"] = version
            self._save(data)
            log_event(logger, EventCode.MODEL_ACTIVATED,
                      f"task={task}  version={version}")
            return True
        return False

    def get_active(self, task: str) -> Optional[dict]:
        """Return the active version entry for a task, or None."""
        data = self._load()
        active_ver = data.get(task, {}).get("active_version")
        if not active_ver:
            return None
        for v in data.get(task, {}).get("versions", []):
            if v["version"] == active_ver:
                return v
        return None

    def get_active_weights(self, task: str) -> Optional[str]:
        """Return the weights_path of the active version, or None."""
        entry = self.get_active(task)
        return entry["weights_path"] if entry else None

    def get_all_versions(self, task: str) -> list[dict]:
        """Return all versions for a task, newest first."""
        data = self._load()
        return sorted(
            data.get(task, {}).get("versions", []),
            key=lambda v: v.get("trained_at", 0),
            reverse=True,
        )

    def next_version(self, task: str) -> str:
        """Return the next version string (e.g. 'v003')."""
        versions = self.get_all_versions(task)
        return f"v{len(versions) + 1:03d}"

    def active_version_string(self, task: str) -> str:
        entry = self.get_active(task)
        return entry["version"] if entry else "none"
