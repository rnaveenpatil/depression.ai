"""Version consistency checks.

`depression.ai` 1.0.1 on PyPI shipped wheel metadata declaring 1.0.1 while
the code inside reported 1.0.0, because six modules each hardcoded their own
version literal. These tests make drift a hard failure.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import agent
from agent._version import get_version
from agent.config.config import Config
from agent.plugins.loader import AGENT_VERSION

_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_version_matches_pyproject():
    assert get_version() == _pyproject_version()


def test_every_version_surface_agrees():
    from agent.cli.ui import UI
    from agent.main import APP_VERSION

    ui = UI()
    banner = ui.print_banner
    assert callable(banner)
    assert agent.__version__ == APP_VERSION == AGENT_VERSION
    assert Config().version == get_version()


def test_no_hardcoded_version_literals_in_source():
    """No module may pin the agent's own version; they must call get_version().

    Sentinel defaults like "0.0.0" for third-party plugin manifests are
    legitimate and deliberately excluded.
    """
    offenders = []
    pattern = re.compile(r"""(?:^|[^\w.])["']v?\d+\.\d+\.\d+["']""")
    sentinel = re.compile(r"""["']v?0\.0\.0["']""")
    version_names = ("__version__", "APP_VERSION", "AGENT_VERSION")
    for path in (_ROOT / "src" / "agent").rglob("*.py"):
        if "depression" in path.parts:  # stray nested venv
            continue
        if path.name == "_version.py":
            continue  # the single documented fallback
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            stripped = line.strip()
            if stripped.startswith(("#", "*", '"""', "'''")):
                continue
            if not pattern.search(line) or sentinel.search(line):
                continue
            # Only flag literals that are actually *the agent's* version.
            if any(name in line for name in version_names) or "version=" in line.replace(" ", "") or "version:" in line:
                offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "hardcoded version literals found:\n" + "\n".join(offenders)


@pytest.mark.parametrize("value", ["1.2.3", "0.0.1", "10.20.30"])
def test_version_uses_pyproject_when_present(monkeypatch, value):
    """pyproject.toml wins when it is available (source checkout / sdist)."""
    import agent._version as mod

    monkeypatch.setattr(mod, "_from_metadata", lambda: value)
    assert mod.get_version() == _pyproject_version()


def test_version_falls_back_through_the_chain(monkeypatch):
    """Resolution order: pyproject -> metadata -> hardcoded fallback."""
    import agent._version as mod

    monkeypatch.setattr(mod, "_from_metadata", lambda: None)
    monkeypatch.setattr(mod, "_from_pyproject", lambda: None)
    assert mod.get_version() == mod._FALLBACK


def test_pyproject_wins_over_stale_installed_metadata(monkeypatch):
    """A leftover *.egg-info from `pip install -e` must not mask pyproject.

    This exact mismatch shipped 1.0.1 on PyPI: metadata said 1.0.1 while the
    code reported 1.0.0. pyproject.toml is the developer's source of truth.
    """
    import agent._version as mod

    monkeypatch.setattr(mod, "_from_metadata", lambda: "0.0.1-stale")
    assert mod.get_version() == _pyproject_version()


def test_pyproject_lookup_is_bounded(monkeypatch):
    """A wheel install has no pyproject.toml, so it must fall through."""
    import agent._version as mod

    monkeypatch.setattr(mod, "_from_metadata", lambda: "7.7.7")
    monkeypatch.setattr(mod, "_from_pyproject", lambda: None)
    assert mod.get_version() == "7.7.7"
