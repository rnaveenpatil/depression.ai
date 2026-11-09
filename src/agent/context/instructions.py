"""Project instructions loader (AGENTS.md / CLAUDE.md).

Loads project-specific guidance once per workspace and exposes a compact,
fingerprinted block for the system prompt. This is the local equivalent of
opencode's ``AGENTS.md`` and Claude Code's ``CLAUDE.md`` support.

Rules:
    * nearest project file first, then parent, then ``.agent/`` and global.
    * explicit ``context.instructions`` file paths are appended.
    * total content is capped so a huge instruction file cannot crowd out the
      conversation.
    * reads are cached by (path, mtime, size) so the loop can fingerprint
      cheaply on every turn.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_FILES = ("AGENTS.md", "CLAUDE.md")
MAX_INSTRUCTION_BYTES = 16_000
_CACHE: Dict[Tuple[str, int, int], str] = {}
_CACHE_LIMIT = 64


def _read_cached(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = text
    return text


def _dedupe(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _candidate_files(project_dir: Optional[str]) -> List[Path]:
    root = Path(project_dir or ".").expanduser()
    try:
        root = root.resolve()
    except OSError:
        pass

    candidates: List[Path] = []
    # Nearest project instructions, then the parent (monorepo root).
    for base in (root, root.parent):
        for name in DEFAULT_FILES:
            candidates.append(base / name)
    for name in DEFAULT_FILES:
        candidates.append(root / ".agent" / name)

    home = Path.home()
    candidates.append(home / ".agent" / "AGENTS.md")
    candidates.append(home / ".config" / "agent" / "AGENTS.md")
    return _dedupe(candidates)


def _configured_files(
    project_dir: Optional[str], configured: Optional[Iterable[Any]]
) -> List[Path]:
    root = Path(project_dir or ".").expanduser()
    out: List[Path] = []
    for item in configured or []:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        out.append(path)
    return _dedupe(out)


def load_project_instructions(
    project_dir: Optional[str],
    configured: Optional[Iterable[Any]] = None,
    max_bytes: int = MAX_INSTRUCTION_BYTES,
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(text, metadata)`` for the active project's instructions.

    ``text`` is empty when no instruction file exists. ``metadata`` always
    contains a stable ``fingerprint`` so callers can invalidate cached system
    prompts when a file changes.
    """
    files = _candidate_files(project_dir) + _configured_files(
        project_dir, configured
    )
    files = _dedupe(files)

    chunks: List[str] = []
    used: List[str] = []
    skipped: List[str] = []
    total = 0
    limit = max(0, int(max_bytes))

    for path in files:
        if not path.is_file():
            continue
        text = _read_cached(path).strip()
        if not text:
            continue
        if total >= limit:
            skipped.append(str(path))
            continue
        remaining = limit - total
        if len(text.encode("utf-8", errors="replace")) > remaining:
            encoded = text.encode("utf-8", errors="replace")[:remaining]
            text = encoded.decode("utf-8", errors="ignore").rstrip()
            text += "\n... (truncated)"
        chunks.append(f"### {path}\n{text}")
        total += len(text)
        used.append(str(path))

    body = "\n\n".join(chunks)
    fingerprint = hashlib.sha1(
        (
            "\n".join(used)
            + f"\n{total}\n"
            + body
        ).encode("utf-8", errors="replace")
    ).hexdigest()
    return body, {
        "paths": used,
        "truncated": bool(skipped),
        "skipped": skipped,
        "bytes": total,
        "fingerprint": fingerprint,
    }


def clear_instruction_cache() -> None:
    """Drop cached file contents (used by tests and after config reloads)."""
    _CACHE.clear()


__all__ = [
    "DEFAULT_FILES",
    "MAX_INSTRUCTION_BYTES",
    "clear_instruction_cache",
    "load_project_instructions",
]
