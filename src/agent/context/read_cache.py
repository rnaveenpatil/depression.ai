"""Session-scoped read cache for files.

Three layers:
    1. Content cache  — path -> (mtime, size, sha, content). Serves repeat
       reads from memory instead of hitting disk.
    2. Read index     — path -> (sha, tokens, last_seen). Survives
       compaction so the model knows what it has already seen.
    3. Slice store    — opaque token -> full original body for truncated
       reads. Lets the model zoom into the middle of a large file without
       re-reading it from disk.

All three are bounded. Writes invalidate the affected path in every layer.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_CONTENT_BUDGET_BYTES = 8 * 1024 * 1024      # 8 MB total content cache
DEFAULT_CONTENT_ENTRIES = 200
DEFAULT_INDEX_ENTRIES = 500
DEFAULT_SLICE_BUDGET_BYTES = 8 * 1024 * 1024        # 8 MB total slice store
DEFAULT_SLICE_ENTRIES = 50
STAT_CHECK_TTL = 1.0  # seconds; how long an mtime/size check is trusted


@dataclass
class CachedFile:
    path: str
    mtime: float
    size: int
    sha: str
    content: str
    tokens: int
    last_used: float
    hits: int = 0


@dataclass
class ReadIndexEntry:
    path: str
    sha: str
    tokens: int
    size: int
    first_seen: float
    last_seen: float
    hits: int = 0
    truncated: bool = False


@dataclass
class SliceEntry:
    token: str
    path: str
    sha: str
    content: str
    size: int
    created: float
    last_used: float


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _hash_str(s: str) -> str:
    return _hash_bytes(s.encode("utf-8"))


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ReadCache:
    """Thread-safe, bounded read cache. One instance per ContextManager."""

    def __init__(
        self,
        content_budget_bytes: int = DEFAULT_CONTENT_BUDGET_BYTES,
        content_entries: int = DEFAULT_CONTENT_ENTRIES,
        index_entries: int = DEFAULT_INDEX_ENTRIES,
        slice_budget_bytes: int = DEFAULT_SLICE_BUDGET_BYTES,
        slice_entries: int = DEFAULT_SLICE_ENTRIES,
    ):
        self._lock = threading.RLock()
        self._content: "OrderedDict[str, CachedFile]" = OrderedDict()
        self._content_bytes = 0
        self._content_budget = content_budget_bytes
        self._content_max = content_entries

        self._index: "OrderedDict[str, ReadIndexEntry]" = OrderedDict()
        self._index_max = index_entries

        self._slices: "OrderedDict[str, SliceEntry]" = OrderedDict()
        self._slice_bytes = 0
        self._slice_budget = slice_budget_bytes
        self._slice_max = slice_entries

        # (path -> (mtime, size, checked_at)) so we don't stat twice per second.
        self._stat_cache: Dict[str, Tuple[float, int, float]] = {}

    # ------------------------------------------------------------------
    # STAT
    # ------------------------------------------------------------------

    def _stat(self, path: str) -> Optional[Tuple[float, int]]:
        now = time.monotonic()
        cached = self._stat_cache.get(path)
        if cached is not None:
            mtime, size, checked = cached
            if now - checked < STAT_CHECK_TTL:
                return mtime, size
        try:
            st = os.stat(path)
        except OSError:
            return None
        self._stat_cache[path] = (st.st_mtime, st.st_size, now)
        return st.st_mtime, st.st_size

    def _invalidate_stat(self, path: str) -> None:
        self._stat_cache.pop(path, None)

    # ------------------------------------------------------------------
    # CONTENT CACHE
    # ------------------------------------------------------------------

    def _evict_content_locked(self) -> None:
        while (
            self._content_bytes > self._content_budget
            or len(self._content) > self._content_max
        ) and self._content:
            _, old = self._content.popitem(last=False)
            self._content_bytes -= len(old.content.encode("utf-8"))

    def get_content(self, path: str) -> Optional[Dict[str, Any]]:
        """Return cached content iff the file has not changed on disk."""
        with self._lock:
            entry = self._content.get(path)
            if entry is None:
                return None
            stat = self._stat(path)
            if stat is None:
                return None
            mtime, size = stat
            if mtime != entry.mtime or size != entry.size:
                # File changed under us. Drop it from every layer.
                self._content.pop(path, None)
                self._content_bytes -= len(entry.content.encode("utf-8"))
                self._index.pop(path, None)
                return None
            entry.last_used = time.time()
            entry.hits += 1
            self._content.move_to_end(path)
            return {
                "path": path,
                "content": entry.content,
                "sha": entry.sha,
                "tokens": entry.tokens,
                "size": entry.size,
                "cached": True,
                "hits": entry.hits,
            }

    def put_content(self, path: str, content: str, mtime: float, size: int) -> str:
        """Store content and return its sha."""
        sha = _hash_str(content)
        with self._lock:
            old = self._content.pop(path, None)
            if old is not None:
                self._content_bytes -= len(old.content.encode("utf-8"))
            entry = CachedFile(
                path=path,
                mtime=mtime,
                size=size,
                sha=sha,
                content=content,
                tokens=_estimate_tokens(content),
                last_used=time.time(),
            )
            self._content[path] = entry
            self._content_bytes += len(content.encode("utf-8"))
            self._content.move_to_end(path)
            self._evict_content_locked()
        return sha

    # ------------------------------------------------------------------
    # READ INDEX (survives compaction)
    # ------------------------------------------------------------------

    def record_read(
        self,
        path: str,
        sha: str,
        tokens: int,
        size: int,
        truncated: bool = False,
    ) -> None:
        with self._lock:
            now = time.time()
            prev = self._index.get(path)
            if prev is None:
                entry = ReadIndexEntry(
                    path=path, sha=sha, tokens=tokens, size=size,
                    first_seen=now, last_seen=now,
                    truncated=truncated,
                )
            else:
                prev.sha = sha
                prev.tokens = tokens
                prev.size = size
                prev.last_seen = now
                prev.hits += 1
                prev.truncated = prev.truncated or truncated
                entry = prev
                self._index.move_to_end(path)
            self._index[path] = entry
            while len(self._index) > self._index_max:
                self._index.popitem(last=False)

    def get_index(self) -> Dict[str, ReadIndexEntry]:
        with self._lock:
            return dict(self._index)

    def index_summary(self, max_entries: int = 40) -> str:
        """Compact human-readable list of files already read this session."""
        with self._lock:
            if not self._index:
                return ""
            entries = list(self._index.values())
            # Newest first, cap.
            entries.sort(key=lambda e: e.last_seen, reverse=True)
            entries = entries[:max_entries]
            lines = ["[Files already read this session]"]
            for e in entries:
                size_kb = max(1, e.size // 1024)
                trunc = " (truncated)" if e.truncated else ""
                lines.append(
                    f"  - {e.path} ({size_kb} KB, {e.tokens} tok, "
                    f"sha {e.sha}){trunc}"
                )
            lines.append(
                "Reuse these results. Only re-read if the file changed on "
                "disk or you need a section that was truncated "
                "(call read with the file path and offset/limit)."
            )
            return "\n".join(lines)

    # ------------------------------------------------------------------
    # SLICE STORE
    # ------------------------------------------------------------------

    def _evict_slices_locked(self) -> None:
        while (
            self._slice_bytes > self._slice_budget
            or len(self._slices) > self._slice_max
        ) and self._slices:
            _, old = self._slices.popitem(last=False)
            self._slice_bytes -= len(old.content.encode("utf-8"))

    def store_slice(self, path: str, sha: str, content: str) -> str:
        """Store the full body under a short token; return the token."""
        token = f"s_{sha[:8]}"
        with self._lock:
            old = self._slices.pop(token, None)
            if old is not None:
                self._slice_bytes -= len(old.content.encode("utf-8"))
            entry = SliceEntry(
                token=token,
                path=path,
                sha=sha,
                content=content,
                size=len(content.encode("utf-8")),
                created=time.time(),
                last_used=time.time(),
            )
            self._slices[token] = entry
            self._slice_bytes += entry.size
            self._slices.move_to_end(token)
            self._evict_slices_locked()
        return token

    def get_slice(
        self, token: str, offset: int = 0, limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._slices.get(token)
            if entry is None:
                return None
            entry.last_used = time.time()
            self._slices.move_to_end(token)
            body = entry.content
            n = len(body)
            if offset < 0:
                offset = 0
            if offset >= n:
                return {
                    "path": entry.path,
                    "sha": entry.sha,
                    "offset": offset,
                    "limit": 0,
                    "content": "",
                    "total_chars": n,
                }
            if limit is None:
                limit = n - offset
            end = min(n, offset + limit)
            return {
                "path": entry.path,
                "sha": entry.sha,
                "offset": offset,
                "limit": end - offset,
                "content": body[offset:end],
                "total_chars": n,
            }

    def get_slice_meta(self, token: str) -> Optional[SliceEntry]:
        with self._lock:
            return self._slices.get(token)

    # ------------------------------------------------------------------
    # INVALIDATION (on writes)
    # ------------------------------------------------------------------

    def invalidate(self, path: str) -> None:
        with self._lock:
            old = self._content.pop(path, None)
            if old is not None:
                self._content_bytes -= len(old.content.encode("utf-8"))
            self._index.pop(path, None)
            # Slice entries stay; they are historical snapshots. A write
            # creates a new sha, so old slices remain valid for the old sha.
            self._invalidate_stat(path)

    def clear(self) -> None:
        with self._lock:
            self._content.clear()
            self._index.clear()
            self._slices.clear()
            self._stat_cache.clear()
            self._content_bytes = 0
            self._slice_bytes = 0


__all__ = ["ReadCache"]