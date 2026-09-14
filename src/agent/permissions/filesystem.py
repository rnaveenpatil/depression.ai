"""
Filesystem Policy - Path-based safety for read/write/delete operations.

Protects:
    - System directories (/etc, /sys, /proc, /boot, /usr)
    - User secrets (~/.ssh, ~/.aws, ~/.gnupg, ~/.config)
    - Any path outside the allowed project root (optional)
"""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.permissions.manager import (
    PermissionRequest,
    PermissionVerdict,
    Policy,
    RiskLevel,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# Directories that should NEVER be written to without explicit approval
SYSTEM_BLOCKED = [
    "/etc/", "/sys/", "/proc/", "/boot/", "/dev/",
    "/bin/", "/sbin/", "/lib/", "/lib64/", "/usr/",
    "/var/log/", "/private/etc/", "C:\\Windows\\",
]

# User-sensitive directories
USER_SENSITIVE = [
    "~/.ssh/", "~/.aws/", "~/.gnupg/", "~/.config/gcloud/",
    "~/.kube/", "~/.docker/", "~/.netrc", "~/.npmrc",
    "~/.pypirc", "~/.git-credentials", "~/.bash_history",
    "~/.zsh_history", "~/.sudo_as_admin_successful",
]

# Read is always allowed in these (even if blocked for write)
READ_ALLOWED_SYSTEM = ["/etc/hosts", "/etc/hostname", "/etc/os-release"]


class FilesystemPolicy(Policy):
    name = "filesystem"

    def __init__(self, config: Dict[str, Any], cwd: Optional[str] = None):
        cfg = config or {}
        self.cwd = Path(cwd or os.getcwd()).resolve()

        # User-configured safe/blocked paths
        self.safe_paths: List[str] = [
            self._expand(p) for p in cfg.get("safe_paths", ["~/", "./", "/tmp/"])
        ]
        self.blocked_paths: List[str] = [
            self._expand(p) for p in cfg.get("blocked_paths", [])
        ]
        self.confirm_writes: bool = cfg.get("confirm_file_writes", True)
        self.confirm_deletes: bool = cfg.get("confirm_deletes", True)

        # Combine built-in + user rules
        self._blocked = (
            [self._expand(p) for p in SYSTEM_BLOCKED]
            + [self._expand(p) for p in USER_SENSITIVE]
            + self.blocked_paths
        )

    # ------------------------------------------------------------------

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        # Only handle filesystem-like tools
        if not self._is_filesystem_op(request):
            return None

        action = request.action.lower()
        path_str = (
            request.params.get("path")
            or request.params.get("file")
            or request.params.get("filepath")
        )
        if not path_str:
            return None

        try:
            path = self._resolve(path_str)
        except Exception:
            return PermissionVerdict.deny(
                f"invalid path: {path_str}",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        # Reads: deny if path is in blocked system dirs (except read-allowed)
        if action in ("read", "list", "stat", "search"):
            if self._is_blocked(path) and not self._is_read_allowed(path):
                return PermissionVerdict.ask(
                    f"reading sensitive path: {path}",
                    risk=RiskLevel.MEDIUM,
                    policy=self.name,
                )
            # Reads inside project → allow
            if self._is_inside_project(path):
                return PermissionVerdict.allow(
                    "read inside project",
                    risk=RiskLevel.SAFE,
                    policy=self.name,
                )
            return None

        # Writes: deny system paths, ask for sensitive paths
        if action in ("write", "create", "edit", "append", "mkdir"):
            if self._is_blocked(path):
                return PermissionVerdict.deny(
                    f"writing to blocked path: {path}",
                    risk=RiskLevel.CRITICAL,
                    policy=self.name,
                )
            if self.confirm_writes and not self._is_inside_project(path):
                return PermissionVerdict.ask(
                    f"writing outside project: {path}",
                    risk=RiskLevel.MEDIUM,
                    policy=self.name,
                )
            return PermissionVerdict.allow(
                "write inside project",
                risk=RiskLevel.SAFE,
                policy=self.name,
            )

        # Deletes: always ask, deny sensitive
        if action in ("delete", "remove", "unlink", "rmdir", "rm"):
            if self._is_blocked(path):
                return PermissionVerdict.deny(
                    f"deleting blocked path: {path}",
                    risk=RiskLevel.CRITICAL,
                    policy=self.name,
                )
            if self.confirm_deletes:
                return PermissionVerdict.ask(
                    f"delete {path}",
                    risk=RiskLevel.HIGH,
                    policy=self.name,
                )
            return PermissionVerdict.allow(
                "delete allowed by config",
                risk=RiskLevel.MEDIUM,
                policy=self.name,
            )

        # Move/copy: treat like write to destination
        if action in ("move", "copy", "rename"):
            dest = request.params.get("dest") or request.params.get("destination")
            if dest:
                try:
                    dpath = self._resolve(str(dest))
                    if self._is_blocked(dpath):
                        return PermissionVerdict.deny(
                            f"writing to blocked dest: {dpath}",
                            risk=RiskLevel.CRITICAL,
                            policy=self.name,
                        )
                except Exception:
                    pass
            return PermissionVerdict.ask(
                f"{action} {path}",
                risk=RiskLevel.MEDIUM,
                policy=self.name,
            )

        return None

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _is_filesystem_op(self, request: PermissionRequest) -> bool:
        tool = request.tool.lower()
        if tool.startswith("mcp__filesystem__"):
            return True
        if tool in ("filesystem", "file", "fs"):
            return True
        if request.action.lower() in (
            "read", "write", "edit", "create", "delete",
            "remove", "unlink", "rmdir", "mkdir", "move",
            "copy", "rename", "list", "stat", "search",
        ):
            return True
        return False

    def _resolve(self, path_str: str) -> Path:
        p = Path(os.path.expanduser(path_str))
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()

    def _expand(self, path_str: str) -> str:
        return str(Path(os.path.expanduser(path_str)).resolve())

    def _is_blocked(self, path: Path) -> bool:
        pstr = str(path) + ("/" if path.is_dir() else "")
        for blocked in self._blocked:
            if pstr == blocked or pstr.startswith(blocked):
                return True
        return False

    def _is_read_allowed(self, path: Path) -> bool:
        pstr = str(path)
        return any(pstr.startswith(self._expand(r)) for r in READ_ALLOWED_SYSTEM)

    def _is_inside_project(self, path: Path) -> bool:
        try:
            path.relative_to(self.cwd)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        pass