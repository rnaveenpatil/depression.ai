"""
Platform Module - OS / architecture detection and compatibility helpers.
"""

from __future__ import annotations

import os
import platform as _platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ======================================================================
# PLATFORM INFO
# ======================================================================

@dataclass(frozen=True)
class Platform:
    system: str          # "Linux" | "Darwin" | "Windows"
    machine: str         # "x86_64" | "arm64" | "aarch64" | ...
    python: str
    is_windows: bool
    is_macos: bool
    is_linux: bool
    is_wsl: bool
    is_docker: bool
    arch: str            # "amd64" | "arm64" | "x86"

    def __str__(self) -> str:
        return f"{self.system}/{self.arch} (py {self.python})"


def _detect_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


def _detect_docker() -> bool:
    try:
        return Path("/.dockerenv").exists() or "docker" in Path("/proc/1/cgroup").read_text()
    except Exception:
        return False


def _normalize_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("x86_64", "amd64"):
        return "amd64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("i386", "i686", "x86"):
        return "x86"
    if m.startswith("armv7"):
        return "armv7"
    return m


_platform_cache: Optional[Platform] = None


def get_platform() -> Platform:
    """Return a cached Platform object."""
    global _platform_cache
    if _platform_cache is not None:
        return _platform_cache

    system = _platform.system()
    machine = _platform.machine() or "unknown"
    _platform_cache = Platform(
        system=system,
        machine=machine,
        python=".".join(str(p) for p in sys.version_info[:3]),
        is_windows=(system == "Windows"),
        is_macos=(system == "Darwin"),
        is_linux=(system == "Linux"),
        is_wsl=_detect_wsl(),
        is_docker=_detect_docker(),
        arch=_normalize_arch(machine),
    )
    return _platform_cache


def is_windows() -> bool:
    return get_platform().is_windows


def is_macos() -> bool:
    return get_platform().is_macos


def is_linux() -> bool:
    return get_platform().is_linux


# ======================================================================
# SHELL
# ======================================================================

def get_shell() -> str:
    """Return the user's preferred shell."""
    if is_windows():
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL") or "/bin/bash"


def get_shell_args() -> list:
    """Get the arguments to pass a command to the shell."""
    if is_windows():
        return ["/c"]
    return ["-c"]


# ======================================================================
# DIRECTORIES
# ======================================================================

def get_home_dir() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def _xdg(var: str, fallback: Path) -> Path:
    value = os.environ.get(var)
    if value:
        return Path(os.path.expanduser(value)).resolve()
    return fallback


def get_config_dir(app_name: str = "agent") -> Path:
    """Platform-appropriate config directory."""
    if is_windows():
        return _xdg("APPDATA", get_home_dir() / "AppData" / "Roaming") / app_name
    if is_macos():
        return get_home_dir() / "Library" / "Application Support" / app_name
    return _xdg("XDG_CONFIG_HOME", get_home_dir() / ".config") / app_name


def get_cache_dir(app_name: str = "agent") -> Path:
    if is_windows():
        return _xdg("LOCALAPPDATA", get_home_dir() / "AppData" / "Local") / app_name / "cache"
    if is_macos():
        return get_home_dir() / "Library" / "Caches" / app_name
    return _xdg("XDG_CACHE_HOME", get_home_dir() / ".cache") / app_name


def get_data_dir(app_name: str = "agent") -> Path:
    if is_windows():
        return _xdg("APPDATA", get_home_dir() / "AppData" / "Roaming") / app_name
    if is_macos():
        return get_home_dir() / "Library" / "Application Support" / app_name
    return _xdg("XDG_DATA_HOME", get_home_dir() / ".local" / "share") / app_name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================================
# PATH UTILITIES
# ======================================================================

def normalize_path(path: str) -> str:
    """
    Normalize a path for the current platform:
    - expanduser (~)
    - expandvars ($HOME)
    - forward slashes on Unix, backslashes on Windows
    - resolve symlinks
    """
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    resolved = p.resolve()
    return str(resolved)


def which(cmd: str) -> Optional[str]:
    """Cross-platform `which`."""
    return shutil.which(cmd)


def has_command(cmd: str) -> bool:
    return which(cmd) is not None


def is_executable(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and os.access(p, os.X_OK)


# ======================================================================
# ENV
# ======================================================================

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def set_env(key: str, value: str, overwrite: bool = True) -> None:
    if overwrite or key not in os.environ:
        os.environ[key] = value


def unset_env(key: str) -> None:
    os.environ.pop(key, None)


def is_ci() -> bool:
    return any(os.environ.get(k) for k in (
        "CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS",
        "GITLAB_CI", "CIRCLECI", "TRAVIS", "JENKINS_URL",
        "TEAMCITY_VERSION", "BUILDKITE",
    ))


def is_tty() -> bool:
    return sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False


# ======================================================================
# TERMINAL
# ======================================================================

def terminal_size(default: tuple = (80, 24)) -> tuple:
    try:
        size = shutil.get_terminal_size(default)
        return (size.columns, size.lines)
    except Exception:
        return default


# ======================================================================
# PYTHON
# ======================================================================

def python_version_tuple() -> tuple:
    return sys.version_info[:3]


def python_at_least(major: int, minor: int = 0) -> bool:
    return sys.version_info >= (major, minor)


__all__ = [
    "Platform", "get_platform",
    "is_windows", "is_macos", "is_linux",
    "get_shell", "get_shell_args",
    "get_home_dir", "get_config_dir", "get_cache_dir", "get_data_dir", "ensure_dir",
    "normalize_path", "which", "has_command", "is_executable",
    "get_env", "set_env", "unset_env", "is_ci", "is_tty",
    "terminal_size",
    "python_version_tuple", "python_at_least",
]