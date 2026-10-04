"""
Single source of truth for the package version.

Every module that reports a version (the ``--version`` flag, the banner,
plugin compatibility checks, the MCP client handshake) reads from here
instead of hardcoding a literal. A literal in six places is how
``depression.ai`` 1.0.1 shipped with ``--version`` printing 1.0.0.

Resolution order:
  1. ``pyproject.toml`` — present in a source checkout and in the sdist.
     This must win over distribution metadata because a stale
     ``*.egg-info`` left over from a previous ``pip install -e`` reports
     the old version and would otherwise mask the real one.
  2. Installed distribution metadata — what a wheel install sees, since
     wheels do not ship ``pyproject.toml``.
  3. ``_FALLBACK`` — last resort so importing never explodes.
"""

from __future__ import annotations

from pathlib import Path

_FALLBACK = "1.0.2"

_DIST_NAME = "depression.ai"


def _from_metadata() -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version(_DIST_NAME)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _from_pyproject() -> str | None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    try:
        here = Path(__file__).resolve()
        # Bounded walk: src-layout is <root>/src/agent/_version.py, so three
        # levels up is the project root. Stopping early keeps an installed
        # copy from picking up an unrelated pyproject.toml further up.
        for parent in here.parents[:3]:
            candidate = parent / "pyproject.toml"
            if not candidate.is_file():
                continue
            data = tomllib.loads(candidate.read_text(encoding="utf-8"))
            project = data.get("project") or {}
            name = project.get("name")
            declared = project.get("version")
            if not declared or not name:
                continue
            if name.replace("_", "-").lower() == _DIST_NAME:
                return str(declared)
            return None  # a different project's pyproject; stop looking
    except Exception:
        return None
    return None


def get_version() -> str:
    """Return the running package version."""
    return _from_pyproject() or _from_metadata() or _FALLBACK


__version__ = get_version()
