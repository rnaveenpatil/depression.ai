"""Bounded, ignore-aware filesystem traversal helpers.

The agent's inspection tools must not walk virtualenvs, dependency trees, or
build output by default.  This module centralizes that policy so listing,
globbing, and the pure-Python search fallback behave consistently.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Optional

DEFAULT_MAX_SECONDS = 5.0

IGNORED_DIR_NAMES = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "jspm_packages",
    "venv", ".venv", "env", ".tox", ".nox", ".eggs",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".cache", "site-packages",
    "dist", "build", "target",
    ".next", ".nuxt", ".svelte-kit", ".parcel-cache",
    ".idea", ".vscode",
    "coverage", "htmlcov", ".coverage",
    ".gradle", ".terraform", ".serverless", ".yarn", ".pnpm-store",
    "Pods", "DerivedData",
})


@dataclass
class WalkEntry:
    path: Path
    relative: str
    name: str
    is_dir: bool
    size: Optional[int]
    mtime: float


@dataclass
class WalkStats:
    scanned: int = 0
    yielded: int = 0
    truncated: bool = False
    reason: str = ""

    def stop(self, reason: str) -> None:
        self.truncated = True
        self.reason = reason


def is_ignored_dir(path: Path, name: str) -> bool:
    """Return True for dependency/build/VCS/virtualenv directories."""
    if name in IGNORED_DIR_NAMES:
        return True
    # Catch virtualenvs with arbitrary names (e.g. ``src/agent/depression``).
    try:
        if (path / "pyvenv.cfg").is_file():
            return True
    except OSError:
        pass
    return False


def _translate_glob(pattern: str) -> str:
    """Translate a glob (including ``**``) into an anchored regex."""
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            j = i
            while j < n and pattern[j] == "*":
                j += 1
            prev = pattern[i - 1] if i else ""
            nxt = pattern[j] if j < n else ""
            if (i == 0 or prev == "/") and (j >= n or nxt == "/"):
                if j < n and nxt == "/":
                    out.append("(?:.*/)?")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i = j
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                out.append(r"\[")
                i += 1
            else:
                body = pattern[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                i = j + 1
        else:
            out.append(re.escape(ch))
            i += 1
    return "^" + "".join(out) + "$"


@lru_cache(maxsize=256)
def _compiled_glob(pattern: str) -> re.Pattern[str]:
    return re.compile(_translate_glob(pattern))


def glob_match(relative: str, pattern: str) -> bool:
    """Match a normalized relative path against a glob pattern."""
    relative = relative.replace(os.sep, "/")
    pattern = pattern.replace("\\", "/")
    if relative.startswith("./"):
        relative = relative[2:]
    if pattern.startswith("./"):
        pattern = pattern[2:]
    if not pattern:
        return False
    return bool(_compiled_glob(pattern).match(relative))


def walk_paths(
    root: Path | str,
    *,
    recursive: bool = True,
    max_results: Optional[int] = None,
    max_seconds: Optional[float] = DEFAULT_MAX_SECONDS,
    max_depth: Optional[int] = None,
    accept: Optional[Callable[[str, str, bool], bool]] = None,
    stats: Optional[WalkStats] = None,
) -> Iterator[WalkEntry]:
    """Walk ``root`` breadth-first with ignore pruning and hard limits.

    ``accept(relative, name, is_dir)`` filters yielded entries only.  Ignored
    directories are never yielded or descended.  Directory symlinks are not
    followed, which also prevents traversal loops.
    """
    root_path = Path(root)
    state = stats if stats is not None else WalkStats()
    deadline = (
        time.monotonic() + max(0.05, float(max_seconds))
        if max_seconds is not None else None
    )
    queue: deque[tuple[Path, str, int]] = deque([(root_path, "", 0)])

    while queue:
        if deadline is not None and time.monotonic() >= deadline:
            state.stop("time_limit")
            return

        dirpath, rel_dir, depth = queue.popleft()
        try:
            with os.scandir(dirpath) as scan:
                entries = list(scan)
        except PermissionError:
            if dirpath == root_path:
                raise
            continue
        except OSError:
            if dirpath == root_path:
                raise
            continue

        entries.sort(key=lambda entry: entry.name)

        for entry in entries:
            if deadline is not None and time.monotonic() >= deadline:
                state.stop("time_limit")
                return

            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue

            if is_dir and is_ignored_dir(Path(entry.path), entry.name):
                continue

            state.scanned += 1
            if accept is None or accept(rel, entry.name, is_dir):
                try:
                    stat = entry.stat(follow_symlinks=False)
                    size = None if is_dir else stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size = None
                    mtime = 0.0
                yield WalkEntry(
                    path=Path(entry.path),
                    relative=rel,
                    name=entry.name,
                    is_dir=is_dir,
                    size=size,
                    mtime=mtime,
                )
                state.yielded += 1
                if max_results is not None and state.yielded >= max_results:
                    state.stop("max_results")
                    return

            if is_dir and recursive and (max_depth is None or depth < max_depth):
                queue.append((Path(entry.path), rel, depth + 1))
