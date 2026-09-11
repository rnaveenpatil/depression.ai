"""
Terminal Policy - Command safety for shell execution.

Classifies commands by risk:
    SAFE      → read-only commands (ls, cat, grep, git status)
    LOW       → dev commands (npm install, git commit, mkdir)
    MEDIUM    → system modification (chmod, chown, kill)
    HIGH      → package managers, network, docker
    CRITICAL  → rm -rf, sudo, curl|sh, fork bombs, disk ops
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from agent.permissions.manager import (
    PermissionRequest,
    PermissionVerdict,
    Policy,
    RiskLevel,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------- SAFE
SAFE_COMMANDS = {
    "ls", "pwd", "cat", "head", "tail", "wc", "echo", "printf",
    "grep", "egrep", "fgrep", "rg", "find", "which", "whoami",
    "date", "env", "printenv", "uname", "hostname", "df", "du",
    "stat", "file", "tree", "sort", "uniq", "cut", "tr", "awk",
    "sed", "diff", "cmp", "man", "help", "type", "id", "pstree",
    "ps", "top", "free", "uptime", "git", "gh", "sqlite3",
}

# ---------------------------------------------------------------- LOW
LOW_RISK_COMMANDS = {
    "mkdir", "touch", "cp", "mv", "ln", "npm", "pnpm", "yarn",
    "pip", "pip3", "poetry", "cargo", "go", "make", "cmake",
    "python", "python3", "node", "deno", "bun", "ruby", "php",
    "java", "javac", "kotlinc", "gcc", "g++", "clang", "rustc",
    "tsc", "eslint", "prettier", "black", "ruff", "flake8",
    "pytest", "jest", "vitest", "mypy", "ts-node", "uvicorn",
}

# ---------------------------------------------------------------- MEDIUM
MEDIUM_RISK_COMMANDS = {
    "chmod", "chown", "chgrp", "kill", "pkill", "killall",
    "systemctl", "service", "crontab", "at", "export", "source",
    "alias", "unalias", "ulimit", "nice", "renice",
}

# ---------------------------------------------------------------- HIGH
HIGH_RISK_COMMANDS = {
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "snap",
    "docker", "podman", "kubectl", "helm", "terraform",
    "curl", "wget", "scp", "rsync", "ssh", "nc", "netcat",
    "telnet", "nmap", "iptables", "ufw", "firewall-cmd",
    "mount", "umount", "fdisk", "parted", "mkfs",
}

# ---------------------------------------------------------------- CRITICAL
CRITICAL_PATTERNS = [
    r"\brm\s+(-[rRf]+\s+)*\S+",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bdd\s+if=.*of=/dev/",
    r"\bmkfs(\.\w+)?\b",
    r"\b:\(\)\s*\{.*\};:",       # fork bomb
    r">\s*/dev/sd[a-z]",
    r"\bchmod\s+(-R\s+)?777\s+/",
    r"\bchown\s+(-R\s+)?\S+\s+/",
    r"\bcurl\b.*\|\s*(ba)?sh",
    r"\bwget\b.*\|\s*(ba)?sh",
    r"\beval\b",
    r"\bexec\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\binit\s+0\b",
    r"\bhistory\s+-c\b",
    r"\bcrontab\s+-r\b",
]

# Dangerous patterns *inside* otherwise-safe commands
INLINE_DANGER = [
    r"[;&|]\s*rm\s+-[rRf]",
    r"[;&|]\s*sudo\b",
    r"[;&|]\s*shutdown\b",
    r"\$\(.*rm\s+-[rRf]",
    r"`.*rm\s+-[rRf]",
]


class TerminalPolicy(Policy):
    name = "terminal"

    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.allow_safe: bool = cfg.get("allow_safe_commands", True)
        self.auto_approve_low: bool = cfg.get("auto_approve_low_risk", False)
        self.block_network: bool = cfg.get("block_network_commands", False)

        self._critical_res = [re.compile(p) for p in CRITICAL_PATTERNS]
        self._inline_res = [re.compile(p) for p in INLINE_DANGER]

        # User overrides
        self.user_allowed: List[str] = list(cfg.get("allowed_commands", []))
        self.user_blocked: List[str] = list(cfg.get("blocked_commands", []))

    # ------------------------------------------------------------------

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        if not self._is_terminal_op(request):
            return None

        command = (
            request.params.get("command")
            or request.params.get("cmd")
            or request.params.get("shell")
        )
        if not command:
            return None

        # User blocklist
        for pat in self.user_blocked:
            if pat in command:
                return PermissionVerdict.deny(
                    f"command matches blocked pattern '{pat}'",
                    risk=RiskLevel.CRITICAL,
                    policy=self.name,
                )

        # User allowlist (short-circuits)
        for pat in self.user_allowed:
            if pat in command:
                return PermissionVerdict.allow(
                    f"command matches allowed pattern '{pat}'",
                    risk=RiskLevel.LOW,
                    policy=self.name,
                )

        risk = self._classify(command)

        # Critical → deny by default (unless allow_dangerous)
        if risk == RiskLevel.CRITICAL:
            return PermissionVerdict.ask(
                f"critical-risk command: {self._short(command)}",
                risk=RiskLevel.CRITICAL,
                policy=self.name,
            )

        if risk == RiskLevel.HIGH:
            if self.block_network and self._looks_networky(command):
                return PermissionVerdict.deny(
                    f"network command blocked: {self._short(command)}",
                    risk=RiskLevel.HIGH,
                    policy=self.name,
                )
            return PermissionVerdict.ask(
                f"high-risk command: {self._short(command)}",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        if risk == RiskLevel.MEDIUM:
            return PermissionVerdict.ask(
                f"medium-risk command: {self._short(command)}",
                risk=RiskLevel.MEDIUM,
                policy=self.name,
            )

        if risk == RiskLevel.LOW:
            if self.auto_approve_low:
                return PermissionVerdict.allow(
                    "low-risk auto-approved",
                    risk=RiskLevel.LOW,
                    policy=self.name,
                )
            return PermissionVerdict.ask(
                f"run: {self._short(command)}",
                risk=RiskLevel.LOW,
                policy=self.name,
            )

        # SAFE
        if self.allow_safe:
            return PermissionVerdict.allow(
                "safe read-only command",
                risk=RiskLevel.SAFE,
                policy=self.name,
            )
        return PermissionVerdict.ask(
            f"run: {self._short(command)}",
            risk=RiskLevel.SAFE,
            policy=self.name,
        )

    # ------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify(self, command: str) -> RiskLevel:
        # Check critical patterns first
        for r in self._critical_res:
            if r.search(command):
                return RiskLevel.CRITICAL
        for r in self._inline_res:
            if r.search(command):
                return RiskLevel.CRITICAL

        # Extract the primary command
        base = self._primary_command(command)

        if base in HIGH_RISK_COMMANDS:
            return RiskLevel.HIGH
        if base in MEDIUM_RISK_COMMANDS:
            return RiskLevel.MEDIUM
        if base in LOW_RISK_COMMANDS:
            return RiskLevel.LOW
        if base in SAFE_COMMANDS:
            # git can be dangerous with certain subcommands
            if base == "git":
                return self._classify_git(command)
            if base == "gh":
                return self._classify_gh(command)
            return RiskLevel.SAFE

        # Unknown command → treat as medium risk
        return RiskLevel.MEDIUM

    def _primary_command(self, command: str) -> str:
        try:
            parts = shlex.split(command, posix=True)
        except ValueError:
            parts = command.split()
        # Skip leading env vars like FOO=bar
        for p in parts:
            if "=" in p and not p.startswith("-"):
                continue
            return p.rsplit("/", 1)[-1]
        return ""

    def _classify_git(self, command: str) -> RiskLevel:
        # Safe git subcommands
        safe_git = {
            "status", "log", "diff", "show", "branch", "remote",
            "fetch", "blame", "config", "rev-parse", "describe",
            "tag", "ls-files", "ls-remote", "whatchanged", "shortlog",
        }
        # Destructive git subcommands
        destructive = {
            "reset", "clean", "rebase", "filter-branch",
            "update-ref", "gc", "prune",
        }
        try:
            parts = shlex.split(command, posix=True)
        except ValueError:
            return RiskLevel.MEDIUM
        for i, p in enumerate(parts):
            if p == "git" and i + 1 < len(parts):
                sub = parts[i + 1]
                if sub in destructive:
                    return RiskLevel.HIGH
                if sub == "push":
                    return RiskLevel.MEDIUM
                if sub in ("commit", "add", "merge", "checkout", "switch"):
                    return RiskLevel.LOW
                if sub in safe_git:
                    return RiskLevel.SAFE
                return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    def _classify_gh(self, command: str) -> RiskLevel:
        try:
            parts = shlex.split(command, posix=True)
        except ValueError:
            return RiskLevel.MEDIUM
        for i, p in enumerate(parts):
            if p == "gh" and i + 1 < len(parts):
                sub = parts[i + 1]
                if sub in ("repo", "pr", "issue", "release", "run", "workflow"):
                    return RiskLevel.LOW
                if sub in ("auth", "secret", "gpg-key"):
                    return RiskLevel.HIGH
                return RiskLevel.MEDIUM
        return RiskLevel.SAFE

    def _looks_networky(self, command: str) -> bool:
        return any(
            kw in command for kw in (
                "curl", "wget", "ssh", "scp", "nc ", "netcat",
                "telnet", "ftp", "sftp", "rsync",
            )
        )

    def _short(self, command: str, n: int = 80) -> str:
        s = command.strip().replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"

    def _is_terminal_op(self, request: PermissionRequest) -> bool:
        tool = request.tool.lower()
        if tool in ("terminal", "shell", "bash", "exec", "run_command"):
            return True
        if request.action.lower() in ("execute", "run", "shell", "exec"):
            return True
        # MCP filesystem-read style tools don't count
        return False

    def reset(self) -> None:
        pass