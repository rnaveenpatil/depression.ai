"""
Shared output-path containment for tools that write bytes to a caller-supplied
location (browser screenshots, exports, ...).

Several tools accept an ``output_path``/``path`` straight from the model. When
nothing validates it, a screenshot request can overwrite ``~/.bashrc`` or any
other file the process can write, which turns a "take a picture" tool into an
arbitrary file-write primitive. ``agent.tools.filesystem`` already enforces
workspace containment for its own writes; this applies the same rule to the
tools that bypass it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# Absolute roots that stay writable regardless of the configured sandbox,
# matching the escape hatch in agent.tools.filesystem.
DEFAULT_ABS_ROOTS: tuple[str, ...] = ("/tmp", "/var/tmp")


class OutputPathError(ValueError):
    """Raised when a caller-supplied output path escapes the sandbox."""


def resolve_output_path(
    raw_path: str,
    root: Optional[os.PathLike[str] | str] = None,
    extra_roots: Optional[Iterable[str]] = None,
) -> Path:
    """Resolve ``raw_path`` and confirm it stays inside ``root``.

    Args:
        raw_path: the caller-supplied path (may be relative or absolute).
        root: sandbox root. Defaults to the current working directory.
        extra_roots: additional absolute roots that remain writable.

    Raises:
        OutputPathError: if the path is empty, or resolves outside every
            permitted root (e.g. via ``..`` or an absolute path).
    """
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise OutputPathError("path must be a non-empty string")

    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    candidate = Path(os.path.expanduser(raw_path.strip()))
    if not candidate.is_absolute():
        candidate = base / candidate

    # strict=False so we can validate destinations that do not exist yet.
    resolved = candidate.resolve(strict=False)

    try:
        resolved.relative_to(base)
        return resolved
    except ValueError:
        pass

    roots = list(DEFAULT_ABS_ROOTS) + [str(r) for r in (extra_roots or [])]
    for allowed in roots:
        try:
            resolved.relative_to(Path(allowed).resolve())
            return resolved
        except (ValueError, OSError):
            continue

    raise OutputPathError(f"output path escapes sandbox: {resolved}")


__all__ = ["resolve_output_path", "OutputPathError", "DEFAULT_ABS_ROOTS"]