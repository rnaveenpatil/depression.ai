"""Bounded, session-scoped task checkpoints for context recovery.

Only task facts are persisted; never store raw tool parameters, results or credentials.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

DEFAULT_DIR = Path.home() / ".agent" / "checkpoints"
MAX_BYTES = 8192


def _safe_id(value: str) -> str:
    return "".join(c for c in (value or "") if c.isalnum() or c in "-_")[:100] or "default"


class TaskCheckpointStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_DIR

    def _path(self, session_id: str) -> Path:
        return self.root / (_safe_id(session_id) + ".json")

    def save(self, session_id: str, checkpoint: Dict[str, Any]) -> None:
        data = json.dumps(checkpoint, ensure_ascii=True, default=str).encode("utf-8")
        if len(data) > MAX_BYTES:
            raise ValueError("checkpoint exceeds size limit")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        fd, tmp = tempfile.mkstemp(prefix=".checkpoint-", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(tmp, self._path(session_id))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def load(self, session_id: str, goal: str) -> Dict[str, Any] | None:
        try:
            path = self._path(session_id)
            if path.stat().st_size > MAX_BYTES:
                return None
            with path.open("r", encoding="utf-8") as handle:
                checkpoint = json.load(handle)
            if isinstance(checkpoint, dict) and checkpoint.get("goal") == goal:
                return checkpoint
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return None
