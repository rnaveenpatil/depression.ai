"""
Summary store — persists compaction summaries to disk, per session.

Layout:
    ~/.agent/summaries/
        <session_id>.jsonl     — one summary per line, newest last
        <session_id>.index.json — small index for fast "latest N" reads

Why separate from the session file:
    The session file is the FULL transcript (grows forever, one file).
    Summaries are a compact, append-only memory layer that survives
    restarts and can be re-injected when the context window fills.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DIR = os.path.expanduser("~/.agent/summaries")
FILE_MODE = 0o600


def _safe_id(session_id: str) -> str:
    return "".join(c for c in (session_id or "") if c.isalnum() or c in "-_") or "default"


class SummaryStore:
    """Reads and writes compaction summaries for one install."""

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or DEFAULT_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_safe_id(session_id)}.jsonl"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        session_id: str,
        summary: str,
        *,
        turns_summarized: int = 0,
        tokens_before: int = 0,
        tokens_after: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not summary or not summary.strip():
            return
        entry = {
            "ts": time.time(),
            "summary": summary.strip(),
            "turns": int(turns_summarized),
            "tokens_before": int(tokens_before),
            "tokens_after": int(tokens_after),
            "metadata": metadata or {},
        }
        path = self._path(session_id)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            try:
                os.chmod(path, FILE_MODE)
            except OSError:
                pass
        except Exception as exc:
            logger.warning("Failed to append summary for %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception as exc:
            logger.warning("Failed to read summaries for %s: %s", session_id, exc)
            return []
        if limit is not None:
            out = out[-limit:]
        return out

    def latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        rows = self.load(session_id, limit=1)
        return rows[0] if rows else None

    def clear(self, session_id: str) -> None:
        try:
            path = self._path(session_id)
            if path.exists():
                path.unlink()
        except Exception as exc:
            logger.warning("Failed to clear summaries for %s: %s", session_id, exc)


__all__ = ["SummaryStore", "DEFAULT_DIR"]