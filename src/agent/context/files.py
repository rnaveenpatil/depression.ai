"""
File Context - Tracks files referenced by the agent.

Responsibilities:
- Load file contents on demand (with size/token limits)
- Track read/write history
- Provide snippets for LLM prompts
- Detect and cache file metadata (mtime, size, hash)
"""

from __future__ import annotations

import os
import time
import hashlib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from agent.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_MAX_TOKENS = 8_000
DEFAULT_MAX_FILE_BYTES = 512 * 1024      # 512 KB per file
DEFAULT_MAX_FILES = 50                    # total files kept


@dataclass
class FileEntry:
    """A tracked file"""
    path: str                     # absolute or project-relative
    rel_path: str                 # relative to project root
    size: int
    mtime: float
    content: Optional[str] = None
    tokens: int = 0
    truncated: bool = False
    added_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    read_count: int = 0
    write_count: int = 0
    hash: Optional[str] = None
    binary: bool = False


class FileContext:
    """
    Manages which files the agent knows about.
    """

    def __init__(
        self,
        workspace: Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        include_contents: bool = True,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_files: int = DEFAULT_MAX_FILES,
    ):
        self.workspace = workspace
        self.max_tokens = max_tokens
        self.include_contents = include_contents
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files

        self._files: Dict[str, FileEntry] = {}   # key: normalized absolute path
        self._total_tokens = 0
        self._lock = asyncio.Lock()

        # Resolve project root
        self._root: Path = self._resolve_root()

    def _resolve_root(self) -> Path:
        try:
            if self.workspace and hasattr(self.workspace, "project_dir"):
                return Path(self.workspace.project_dir).resolve()
        except Exception:
            pass
        return Path.cwd().resolve()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def add_file(
        self,
        path: str,
        content: Optional[str] = None,
    ) -> bool:
        """
        Add or refresh a file in context.
        If `content` is None, the file is read from disk.
        """
        try:
            abs_path = self._normalize(path)
        except Exception as e:
            logger.debug(f"Invalid path {path}: {e}")
            return False

        async with self._lock:
            try:
                stat = abs_path.stat()
            except FileNotFoundError:
                logger.debug(f"File not found: {abs_path}")
                return False
            except Exception as e:
                logger.debug(f"Stat failed for {abs_path}: {e}")
                return False

            rel = self._rel(abs_path)
            entry = self._files.get(str(abs_path))

            # If unchanged and already cached, just bump accessed_at
            if entry and entry.mtime == stat.st_mtime and entry.size == stat.st_size:
                entry.accessed_at = time.time()
                entry.read_count += 1
                return True

            # Read content
            text, truncated, is_binary = "", False, False
            if content is not None:
                text = content
            elif self.include_contents:
                text, truncated, is_binary = await asyncio.to_thread(
                    self._read_file, abs_path
                )

            tokens = self._estimate_tokens(text)

            # Evict if we'd exceed max_files
            if entry is None and len(self._files) >= self.max_files:
                self._evict_oldest()

            new_entry = FileEntry(
                path=str(abs_path),
                rel_path=rel,
                size=stat.st_size,
                mtime=stat.st_mtime,
                content=text,
                tokens=tokens,
                truncated=truncated,
                binary=is_binary,
                hash=self._hash(text) if text else None,
                read_count=(entry.read_count + 1) if entry else 1,
                write_count=entry.write_count if entry else 0,
                added_at=entry.added_at if entry else time.time(),
                accessed_at=time.time(),
            )

            if entry:
                self._total_tokens -= entry.tokens
            self._files[str(abs_path)] = new_entry
            self._total_tokens += new_entry.tokens

            # Trim if over budget
            self._enforce_token_budget()

            return True

    async def get_content(self, path: str) -> Optional[str]:
        """Get cached content for a file (reads if not cached)"""
        abs_path = self._normalize(path)
        async with self._lock:
            entry = self._files.get(str(abs_path))
            if entry and entry.content is not None:
                entry.accessed_at = time.time()
                entry.read_count += 1
                return entry.content

        # Read fresh
        ok = await self.add_file(path)
        if not ok:
            return None
        entry = self._files.get(str(abs_path))
        return entry.content if entry else None

    async def remove_file(self, path: str) -> bool:
        abs_path = self._normalize(path)
        async with self._lock:
            entry = self._files.pop(str(abs_path), None)
            if entry:
                self._total_tokens -= entry.tokens
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._files.clear()
            self._total_tokens = 0

    async def count(self) -> int:
        return len(self._files)

    async def get_summary(self, max_files: int = 10) -> List[Dict[str, Any]]:
        """Return a compact list of tracked files"""
        async with self._lock:
            items = sorted(
                self._files.values(),
                key=lambda e: (-e.accessed_at, e.rel_path),
            )[:max_files]

            return [
                {
                    "path": e.rel_path,
                    "size": e.size,
                    "tokens": e.tokens,
                    "truncated": e.truncated,
                    "binary": e.binary,
                    "content": e.content if self.include_contents else None,
                }
                for e in items
            ]

    async def as_prompt(self, max_chars: int = 4000) -> str:
        """Render tracked files as an LLM-friendly string"""
        files = await self.get_summary(max_files=15)
        parts = []
        remaining = max_chars
        for f in files:
            header = f"--- {f['path']} ({f['size']} bytes) ---"
            body = f.get("content") or "(content not loaded)"
            block = f"{header}\n{body}\n"
            if len(block) > remaining:
                block = block[:remaining] + "\n… (truncated)\n"
            parts.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _normalize(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        if not p.is_absolute():
            p = (self._root / p)
        return p.resolve()

    def _rel(self, abs_path: Path) -> str:
        try:
            return str(abs_path.relative_to(self._root))
        except ValueError:
            return str(abs_path)

    def _read_file(self, path: Path) -> Tuple[str, bool, bool]:
        """Return (content, truncated, is_binary)"""
        try:
            with open(path, "rb") as f:
                raw = f.read(self.max_file_bytes + 1)
        except Exception as e:
            logger.debug(f"Read failed for {path}: {e}")
            return "", False, False

        # Binary check
        is_binary = b"\x00" in raw[:1024]
        truncated = len(raw) > self.max_file_bytes
        if truncated:
            raw = raw[: self.max_file_bytes]

        if is_binary:
            return "", truncated, True

        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            try:
                text = raw.decode("latin-1", errors="replace")
            except Exception:
                return "", truncated, True
        return text, truncated, False

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _evict_oldest(self) -> None:
        if not self._files:
            return
        oldest_key = min(
            self._files.keys(),
            key=lambda k: self._files[k].accessed_at,
        )
        entry = self._files.pop(oldest_key, None)
        if entry:
            self._total_tokens -= entry.tokens

    def _enforce_token_budget(self) -> None:
        while self._total_tokens > self.max_tokens and self._files:
            self._evict_oldest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": [
                {
                    "path": e.rel_path,
                    "size": e.size,
                    "tokens": e.tokens,
                    "read_count": e.read_count,
                    "write_count": e.write_count,
                    "truncated": e.truncated,
                    "binary": e.binary,
                }
                for e in self._files.values()
            ],
            "total_tokens": self._total_tokens,
        }