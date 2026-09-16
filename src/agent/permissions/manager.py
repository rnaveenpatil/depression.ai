"""
Permission Manager - Central authorization gate.

Default policy (this build):
    - ALLOW everything.
    - ASK only when the action looks like a delete/destroy/removal.
    - DENY .env file reads (hard-coded safety).

Flow:
    LLM  →  Tool/Command  →  PermissionManager  →  { Allow | Ask-if-delete | Deny-.env }
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import PermissionDeniedError

logger = get_logger(__name__)


# ======================================================================
# TYPES
# ======================================================================

class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"

    @property
    def is_terminal(self) -> bool:
        return self in (Decision.ALLOW, Decision.DENY)


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PermissionRequest:
    """A request to execute a tool or command."""
    tool: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""


@dataclass
class PermissionVerdict:
    """Result of a permission check."""
    decision: Decision
    reason: str = ""
    risk: RiskLevel = RiskLevel.SAFE
    policy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(cls, reason="", risk=RiskLevel.SAFE, policy=None) -> "PermissionVerdict":
        return cls(Decision.ALLOW, reason, risk, policy)

    @classmethod
    def deny(cls, reason="", risk=RiskLevel.HIGH, policy=None) -> "PermissionVerdict":
        return cls(Decision.DENY, reason, risk, policy)

    @classmethod
    def ask(cls, reason="", risk=RiskLevel.MEDIUM, policy=None) -> "PermissionVerdict":
        return cls(Decision.ASK, reason, risk, policy)


# ======================================================================
# DESTRUCTIVE-ACTION DETECTION
# ======================================================================

# Words that indicate a delete/destroy style operation. If a tool name or
# action string contains any of these, we ASK. Everything else is allowed.
_DELETE_TOKENS = (
    "delete", "remove", "rm ", "rmdir", "unlink", "drop", "truncate",
    "purge", "destroy", "wipe", "prune", "erase",
    "format", "kill", "terminate", "shutdown",
    "deletefile", "removefile", "remove_file", "delete_file",
)

# Shell-level destructive patterns inside `command` / `cmd` params.
_DELETE_SHELL_PATTERNS = [
    re.compile(r"\brm\b"),
    re.compile(r"\brmdir\b"),
    re.compile(r"\bunlink\b"),
    re.compile(r"\bshred\b"),
    re.compile(r"\btruncate\b"),
    re.compile(r"\bdd\b.*\bof="),
    re.compile(r"\bmkfs\b"),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f"),
    re.compile(r"\bkubectl\s+delete\b"),
    re.compile(r"\baws\s+s3\s+rm\b"),
    re.compile(r"\baws\s+s3api\s+delete"),
    re.compile(r"\baws\s+ec2\s+terminate"),
    re.compile(r"\baws\s+rds\s+delete"),
    re.compile(r"\bdocker\s+rm\b"),
    re.compile(r"\bdocker\s+rmi\b"),
    re.compile(r"\bdocker\s+system\s+prune\b"),
    re.compile(r"\bdocker\s+volume\s+rm\b"),
]


def _looks_destructive(tool: str, action: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (is_destructive, reason). Conservative but thorough."""
    tool_l = (tool or "").lower()
    action_l = (action or "").lower()
    combined = f"{tool_l} {action_l}"

    for tok in _DELETE_TOKENS:
        if tok in combined:
            return True, f"tool/action matches '{tok.strip()}'"

    # Check string params for shell-style destructive commands.
    for key in ("command", "cmd", "shell", "script", "input"):
        val = params.get(key)
        if isinstance(val, str) and val:
            for pat in _DELETE_SHELL_PATTERNS:
                if pat.search(val):
                    return True, f"{key} matches destructive pattern '{pat.pattern}'"

    # Git action strings like "reset_hard", "clean -f", "branch -D"
    git_action = params.get("action")
    if isinstance(git_action, str):
        ga = git_action.lower()
        if ga in ("delete", "remove", "reset", "clean", "drop", "prune",
                  "delete_branch", "deletebranch", "branch_delete",
                  "reset_hard", "reset-hard", "clean_force", "clean-force"):
            return True, f"git action '{ga}' is destructive"

    # Filesystem action strings
    fs_action = params.get("action")
    if isinstance(fs_action, str):
        fa = fs_action.lower()
        if fa in ("delete", "remove", "rm", "unlink", "rmdir", "purge", "wipe"):
            return True, f"filesystem action '{fa}' is destructive"

    return False, ""


# ======================================================================
# POLICY BASE
# ======================================================================

class Policy:
    """Base class for all policy modules."""
    name: str = "policy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def evaluate(self, request: PermissionRequest) -> Optional[PermissionVerdict]:
        return None


# ======================================================================
# MANAGER
# ======================================================================

class PermissionManager:
    """
    Default-allow permission manager.

    The only two situations that produce a non-ALLOW verdict:
      1. Destructive action detected → ASK
      2. .env / *.env file read      → DENY
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        ui: Any = None,
        input_handler: Any = None,
    ):
        cfg = config or {}
        self.config = cfg
        self.ui = ui
        self.input_handler = input_handler

        self.enabled: bool = cfg.get("enabled", True)
        self.mode: str = cfg.get("mode", "manual")   # kept for compat
        self.auto_approve: bool = cfg.get("auto_approve", False)
        self.allow_dangerous: bool = cfg.get("allow_dangerous", False)
        self.default_decision: Decision = Decision.ALLOW

        # No policy list is used — the manager decides directly.
        self.policies: List[Policy] = []

        # Session memory for "always allow this exact delete this session"
        self._session_allows: Dict[str, float] = {}
        self._session_denies: Dict[str, float] = {}
        self._session_ttl: float = cfg.get("session_ttl", 3600.0)

        # Statistics
        self.stats = {
            "allowed": 0,
            "denied": 0,
            "asked": 0,
            "auto_approved": 0,
            "user_approved": 0,
            "user_denied": 0,
        }

        # Confirmation callback — the TUI's modal hook goes here.
        self.confirm_callback: Optional[
            Callable[[PermissionRequest, PermissionVerdict], Awaitable[bool]]
        ] = None

        logger.info(
            "PermissionManager initialized — default ALLOW, "
            "ASK only on destructive actions, DENY .env reads"
        )

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def _parse_default(self, val: str) -> Decision:
        return Decision.ALLOW

    def _register_policies(self, cfg: Dict[str, Any]) -> None:
        # No policies. The manager handles everything directly.
        return

    def set_confirm_callback(
        self,
        callback: Callable[[PermissionRequest, PermissionVerdict], Awaitable[bool]],
    ) -> None:
        """Register the callback used to ask the user for confirmation."""
        self.confirm_callback = callback

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def check(
        self,
        tool: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        request = PermissionRequest(
            tool=tool,
            action=action,
            params=params or {},
            context=context or {},
            request_id=request_id or f"{tool}:{action}:{int(time.time()*1000)}",
        )
        verdict = await self.evaluate(request)

        if verdict.decision == Decision.ALLOW:
            self.stats["allowed"] += 1
            return True, verdict.reason or "allowed"

        if verdict.decision == Decision.DENY:
            self.stats["denied"] += 1
            raise PermissionDeniedError(
                f"Permission denied for {tool}.{action}: {verdict.reason}"
            )

        # ASK — consult the user via the injected callback.
        approved = await self._ask_user(request, verdict)
        if approved:
            self.stats["user_approved"] += 1
            self._remember_allow(request)
            return True, verdict.reason or "user approved"

        self.stats["user_denied"] += 1
        self._remember_deny(request)
        raise PermissionDeniedError(
            f"Permission denied by user for {tool}.{action}"
        )

    async def check_permission(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        action = self._infer_action(tool_name, params)
        try:
            return await self.check(tool_name, action, params, context)
        except PermissionDeniedError as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # EVALUATION — the entire decision logic
    # ------------------------------------------------------------------

    async def evaluate(self, request: PermissionRequest) -> PermissionVerdict:
        """
        Default allow. The only overrides:
          1. .env / *.env read → DENY
          2. destructive action → ASK
        """

        if not self.enabled:
            return PermissionVerdict.allow("permissions disabled")

        # Session cache — a previous "always allow this delete" wins.
        cached = self._check_session_cache(request)
        if cached is not None:
            return cached

        tool_l = (request.tool or "").lower()
        action_l = (request.action or "").lower()

        # --- .env protection (hard deny) -----------------------------
        if tool_l in ("read", "filesystem", "file") or action_l == "read":
            target = (
                request.params.get("path")
                or request.params.get("filePath")
                or request.params.get("file")
                or ""
            )
            base = os.path.basename(str(target))
            if base == ".env" or (base.startswith(".env.") and base != ".env.example"):
                return PermissionVerdict.deny(
                    f"default deny for .env: {target}",
                    risk=RiskLevel.HIGH,
                    policy="env_guard",
                )
            if base.endswith(".env") and base != ".env.example":
                return PermissionVerdict.deny(
                    f"default deny for {base}",
                    risk=RiskLevel.HIGH,
                    policy="env_guard",
                )

        # --- destructive action guard --------------------------------
        destructive, why = _looks_destructive(
            request.tool, request.action, request.params
        )
        if destructive:
            return PermissionVerdict.ask(
                f"destructive action detected ({why})",
                risk=RiskLevel.HIGH,
                policy="delete_guard",
            )

        # --- everything else is allowed ------------------------------
        self.stats["auto_approved"] += 1
        return PermissionVerdict.allow(
            "default allow",
            risk=RiskLevel.SAFE,
            policy="default_allow",
        )

    # ------------------------------------------------------------------
    # USER CONFIRMATION (only reached for destructive ASK verdicts)
    # ------------------------------------------------------------------

    async def _ask_user(
        self,
        request: PermissionRequest,
        verdict: PermissionVerdict,
    ) -> bool:
        self.stats["asked"] += 1

        if self.confirm_callback is not None:
            try:
                cb = self.confirm_callback
                try:
                    import threading as _th
                    logger.debug(
                        f"confirm_callback invoked on thread="
                        f"{_th.current_thread().name}"
                    )
                except Exception:
                    pass
                result = await cb(request, verdict)
                return bool(result)
            except Exception as e:
                logger.error(f"Confirm callback failed: {e}", exc_info=True)

        # Fall back to UI/input handler if wired
        if self.ui:
            self._render_prompt(request, verdict)

        if self.input_handler:
            try:
                if hasattr(self.input_handler, "get_confirmation"):
                    return await self.input_handler.get_confirmation(
                        f"Allow {request.tool}.{request.action}?", default=False,
                    )
                if hasattr(self.input_handler, "get_input"):
                    ans = await self.input_handler.get_input(prompt="Allow? [y/N] ")
                    return (ans or "").strip().lower() in ("y", "yes")
            except Exception as e:
                logger.warning(f"Input handler error: {e}")

        # No mechanism to ask → deny the destructive action.
        logger.warning(
            f"No confirmation mechanism available for "
            f"{request.tool}.{request.action} — denying destructive action."
        )
        return False

    def _render_prompt(
        self, request: PermissionRequest, verdict: PermissionVerdict
    ) -> None:
        if not self.ui:
            return
        try:
            from agent.cli.ui import Palette
            self.ui.print_warning(
                f"Permission requested: {request.tool}.{request.action}"
            )
            self.ui.print_info(f"  Risk: {verdict.risk.value.upper()}")
            if verdict.reason:
                self.ui.print_info(f"  Reason: {verdict.reason}")
            if request.params:
                for k, v in list(request.params.items())[:6]:
                    v_str = str(v)
                    if len(v_str) > 80:
                        v_str = v_str[:77] + "…"
                    self.ui.print_info(f"  {k}: {v_str}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # SESSION CACHE
    # ------------------------------------------------------------------

    def _cache_key(self, request: PermissionRequest) -> str:
        target = (
            request.params.get("path")
            or request.params.get("command")
            or request.params.get("cmd")
            or ""
        )
        return f"{request.tool}:{request.action}:{target}"

    def _check_session_cache(self, request: PermissionRequest) -> Optional[PermissionVerdict]:
        key = self._cache_key(request)
        now = time.time()

        until = self._session_allows.get(key)
        if until and until > now:
            return PermissionVerdict.allow(
                "previously approved this session",
                risk=RiskLevel.SAFE,
                policy="session_cache",
            )

        until = self._session_denies.get(key)
        if until and until > now:
            return PermissionVerdict.deny(
                "previously denied this session",
                risk=RiskLevel.HIGH,
                policy="session_cache",
            )

        return None

    def _remember_allow(self, request: PermissionRequest) -> None:
        self._session_allows[self._cache_key(request)] = time.time() + self._session_ttl

    def _remember_deny(self, request: PermissionRequest) -> None:
        self._session_denies[self._cache_key(request)] = time.time() + self._session_ttl

    # ------------------------------------------------------------------
    # RUNTIME CONTROL
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in ("manual", "auto", "deny"):
            raise ValueError(f"Invalid permission mode: {mode}")
        self.mode = mode
        logger.info(f"Permission mode set to: {mode}")

    def set_auto_approve(self, enabled: bool) -> None:
        self.auto_approve = bool(enabled)

    def add_allow_rule(self, pattern: str) -> None:
        self._session_allows[pattern] = time.time() + self._session_ttl

    def add_deny_rule(self, pattern: str) -> None:
        self._session_denies[pattern] = time.time() + self._session_ttl

    def reset_rules(self) -> None:
        self._session_allows.clear()
        self._session_denies.clear()

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "auto_approve": self.auto_approve,
            "allow_dangerous": self.allow_dangerous,
            "default": self.default_decision.value,
            "policies": [],
            "session_allows": len(self._session_allows),
            "session_denies": len(self._session_denies),
            "stats": dict(self.stats),
        }

    def _infer_action(self, tool: str, params: Dict[str, Any]) -> str:
        """Best-effort action inference."""
        t = tool.lower()
        if "read" in t or t in ("filesystem_read",):
            return "read"
        if "write" in t or "edit" in t:
            return "write"
        if "delete" in t or "remove" in t or t.startswith("rm") or t in ("del", "rmdir", "unlink"):
            return "delete"
        if "execute" in t or t == "terminal" or "shell" in t:
            return "execute"
        if t.startswith("mcp__"):
            return "mcp"
        if "git" in t:
            return "git"
        if "aws" in t or t.startswith("s3_") or t.startswith("ec2_"):
            return "aws"
        return "execute"