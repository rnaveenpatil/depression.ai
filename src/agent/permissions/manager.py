"""
Permission Manager - Central authorization gate.

Flow:
    LLM  →  Tool/Command  →  PermissionManager  →  { Safe | Dangerous }
                                                       │         │
                                                       ▼         ▼
                                                    Execute   Ask user

The manager asks each registered policy module (rules, filesystem, terminal,
aws, …) for a verdict, then either auto-approves, prompts the user, or denies.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
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
    action: str                     # e.g. "read", "write", "execute", "delete"
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
# POLICY BASE
# ======================================================================

class Policy(ABC_MARKER := object):
    """
    Base class for all policy modules (rules, filesystem, terminal, aws).
    Each policy inspects a PermissionRequest and returns an optional verdict.
    Returning None means "no opinion" — the next policy is consulted.
    """

    name: str = "policy"

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        return None


# ======================================================================
# MANAGER
# ======================================================================

class PermissionManager:
    """
    Orchestrates all policies and enforces the final decision.

    Modes:
        manual  — ask the user for anything not explicitly allowed
        auto    — auto-approve everything above SAFE (use with care)
        deny    — deny everything not explicitly allowed (paranoid mode)
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
        self.mode: str = cfg.get("mode", "manual")            # manual | auto | deny
        self.auto_approve: bool = cfg.get("auto_approve", False)
        self.allow_dangerous: bool = cfg.get("allow_dangerous", False)
        self.default_decision: Decision = self._parse_default(cfg.get("default", "ask"))

        # opencode defaults: most allow, doom_loop/external_directory ask
        self._opencode_defaults: Dict[str, str] = {
            "read": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "bash": "allow",
            "task": "allow",
            "skill": "allow",
            "question": "allow",
            "webfetch": "allow",
            "websearch": "allow",
            "lsp": "allow",
            "todowrite": "allow",
            "external_directory": "ask",
            "doom_loop": "ask",
        }

        # Registered policies (order matters — first non-None verdict wins)
        self.policies: List[Policy] = []
        self._register_policies(cfg)

        # Session memory: once approved in a session, don't re-ask
        self._session_allows: Dict[str, float] = {}
        self._session_denies: Dict[str, float] = {}
        self._session_ttl: float = cfg.get("session_ttl", 3600.0)

        # doom_loop tracking: last N tool calls
        from collections import deque
        self._recent_calls: deque = deque(maxlen=10)
        # workspace root for external_directory
        self._workspace_root: Optional[str] = cfg.get("cwd") or cfg.get("workspace_dir") or os.getcwd()

        # Statistics
        self.stats = {
            "allowed": 0,
            "denied": 0,
            "asked": 0,
            "auto_approved": 0,
            "user_approved": 0,
            "user_denied": 0,
        }

        # Confirmation callback (injected by CLI)
        self.confirm_callback: Optional[
            Callable[[PermissionRequest, PermissionVerdict], Awaitable[bool]]
        ] = None

        logger.info(
            f"PermissionManager initialized "
            f"(mode={self.mode}, auto_approve={self.auto_approve}, "
            f"policies={[p.name for p in self.policies]})"
        )

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def _parse_default(self, val: str) -> Decision:
        try:
            return Decision(val.lower())
        except ValueError:
            return Decision.ASK

    def _register_policies(self, cfg: Dict[str, Any]) -> None:
        """Instantiate and register all policy modules."""
        try:
            from agent.permissions.rules import RulesPolicy
            self.policies.append(RulesPolicy(cfg))
        except Exception as e:
            logger.debug(f"RulesPolicy not loaded: {e}")

        try:
            from agent.permissions.filesystem import FilesystemPolicy
            self.policies.append(FilesystemPolicy(cfg, cwd=cfg.get("cwd")))
        except Exception as e:
            logger.debug(f"FilesystemPolicy not loaded: {e}")

        try:
            from agent.permissions.terminal import TerminalPolicy
            self.policies.append(TerminalPolicy(cfg))
        except Exception as e:
            logger.debug(f"TerminalPolicy not loaded: {e}")

        try:
            from agent.permissions.aws import AWSPolicy
            self.policies.append(AWSPolicy(cfg))
        except Exception as e:
            logger.debug(f"AWSPolicy not loaded: {e}")

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
        """
        Main entry point.

        Returns:
            (allowed: bool, reason: str)
        """
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

        # Decision.ASK → consult user
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
        """
        Compatibility wrapper used by agent.py / loop.py.
        Derives action from tool_name and params.
        """
        action = self._infer_action(tool_name, params)
        try:
            return await self.check(tool_name, action, params, context)
        except PermissionDeniedError as e:
            return False, str(e)

    def _is_external_path(self, path_str: str) -> bool:
        try:
            p = Path(os.path.expanduser(path_str)).resolve()
            root = Path(self._workspace_root).resolve()
            try:
                p.relative_to(root)
                return False
            except ValueError:
                return True
        except Exception:
            return False

    def _check_doom_loop(self, request: PermissionRequest) -> Optional[PermissionVerdict]:
        # Track call signature for doom_loop (=3 identical tool calls)
        import json as _json
        try:
            sig = f"{request.tool}:{_json.dumps(request.params, sort_keys=True)}"
        except Exception:
            sig = f"{request.tool}:{str(request.params)}"
        self._recent_calls.append(sig)
        if len(self._recent_calls) >= 3:
            last3 = list(self._recent_calls)[-3:]
            if len(set(last3)) == 1:
                # check if permission doom_loop is allow via RulesPolicy
                # If not explicitly allowed, ask
                return PermissionVerdict.ask(
                    "doom_loop detected: same tool 3x",
                    risk=RiskLevel.HIGH,
                    policy="doom_loop",
                )
        return None

    def _default_for_tool(self, request: PermissionRequest) -> Optional[PermissionVerdict]:
        # opencode permissive defaults
        t = request.tool.lower()
        # map aliases to canonical permission key
        alias_map = {
            "read": "read", "write": "edit", "edit": "edit", "apply_patch": "edit",
            "patch": "edit", "glob": "glob", "grep": "grep", "search": "grep",
            "bash": "bash", "terminal": "bash", "shell": "bash",
            "task": "task", "skill": "skill", "question": "question",
            "webfetch": "webfetch", "websearch": "websearch", "web": "webfetch",
            "lsp": "lsp", "diagnostics": "lsp", "todowrite": "todowrite",
            "todoread": "todowrite", "todo": "todowrite",
        }
        key = alias_map.get(t, t)
        # opencode special: .env files denied for read even though default allow
        if key == "read":
            target = request.params.get("path") or request.params.get("filePath") or request.params.get("file") or ""
            base = os.path.basename(target)
            if base == ".env" or (base.startswith(".env.") and base != ".env.example"):
                return PermissionVerdict.deny(f"default deny for .env: {target}", risk=RiskLevel.HIGH, policy="default")
            if ".env" in base and base != ".env.example":
                # deny *.env
                import fnmatch as _fn
                if _fn.fnmatch(base, "*.env") or _fn.fnmatch(base, "*.env.*"):
                    if base != ".env.example":
                        return PermissionVerdict.deny(f"default deny for {base}", risk=RiskLevel.HIGH, policy="default")
        default_action = self._opencode_defaults.get(key)
        if default_action == "allow":
            return PermissionVerdict.allow(f"default allow for {key}", risk=RiskLevel.SAFE, policy="default")
        if default_action == "ask":
            return PermissionVerdict.ask(f"default ask for {key}", risk=RiskLevel.MEDIUM, policy="default")
        return None

    async def evaluate(self, request: PermissionRequest) -> PermissionVerdict:
        """
        Compute the verdict for a request without asking the user.
        """
        if not self.enabled:
            return PermissionVerdict.allow("permissions disabled")

        # Session memory shortcuts
        cached = self._check_session_cache(request)
        if cached is not None:
            return cached

        # Early default .env deny (opencode default) — must run before FilesystemPolicy allow
        early_env = self._default_for_tool(request)
        if early_env and early_env.decision == Decision.DENY and "env" in early_env.reason.lower():
            return early_env

        # doom_loop detection (before other policies)
        doom = self._check_doom_loop(request)
        if doom is not None:
            # If policy explicitly allows doom_loop, that will override after
            pass  # keep as fallback candidate

        # external_directory detection (if path outside workspace)
        ext_verdict: Optional[PermissionVerdict] = None
        for pkey in ("path", "filePath", "file", "filepath"):
            pval = request.params.get(pkey)
            if isinstance(pval, str) and pval and self._is_external_path(pval):
                # will be handled by RulesPolicy external_directory key; fallback ask
                ext_verdict = PermissionVerdict.ask(
                    f"external_directory: {pval}", risk=RiskLevel.MEDIUM, policy="external_directory"
                )
                break

        # Global auto-approve (--yolo) bypasses all checks
        if self.auto_approve:
            self.stats["auto_approved"] += 1
            return PermissionVerdict.allow(
                "auto_approve enabled (yolo)",
                risk=RiskLevel.MEDIUM,
                policy="global_auto",
            )

        # Mode shortcuts
        if self.mode == "deny":
            return PermissionVerdict.deny(
                "mode=deny (paranoid mode)",
                risk=RiskLevel.CRITICAL,
                policy="mode",
            )

        # Check doom_loop permission override (if explicitly allow, ignore doom)
        doom_allowed = False
        if doom is not None:
            # peek at RulesPolicy external check
            for p in self.policies:
                if p.name == "rules":
                    rules = getattr(p, "permission_rules", {})
                    doom_rules = rules.get("doom_loop") or rules.get("*")
                    if doom_rules:
                        # last rule wins
                        last_act = doom_rules[-1][1] if doom_rules else None
                        if last_act == "allow":
                            doom_allowed = True
                            doom = None
                        elif last_act == "deny":
                            doom = PermissionVerdict.deny("doom_loop denied by permission", risk=RiskLevel.HIGH, policy="rules")
                    break

        # Evaluate external_directory granular permission before generic policies
        # Build ext verdict from permission_rules if exists
        if ext_verdict is not None:
            for p in self.policies:
                if p.name == "rules":
                    rules = getattr(p, "permission_rules", {})
                    ext_rules = rules.get("external_directory") or rules.get("*")
                    if ext_rules:
                        # last matching rule wins — reuse same matching logic as rules
                        from agent.permissions.rules import _wildcard_match as _wm
                        target_path = None
                        for k in ("path", "filePath", "file", "filepath"):
                            if k in request.params:
                                target_path = request.params[k]
                                break
                        matched = None
                        for pat, act in ext_rules:
                            if pat == "*":
                                matched = act
                            elif target_path and _wm(target_path, pat):
                                matched = act
                        if matched == "allow":
                            ext_verdict = None  # allowed, clear
                        elif matched == "deny":
                            ext_verdict = PermissionVerdict.deny(f"external_directory denied: {target_path}", risk=RiskLevel.HIGH, policy="rules")
                        elif matched == "ask":
                            ext_verdict = PermissionVerdict.ask(f"external_directory: {target_path}", risk=RiskLevel.MEDIUM, policy="rules")
                    break

        # If external wants to deny, prioritize immediately
        if ext_verdict is not None and ext_verdict.decision == Decision.DENY:
            return ext_verdict

        # If doom still active and no explicit allow, doom takes precedence over safe allow
        doom_allowed = False
        if doom is not None:
            for p in self.policies:
                if p.name == "rules":
                    rules = getattr(p, "permission_rules", {})
                    doom_rules = rules.get("doom_loop") or rules.get("*")
                    if doom_rules:
                        last_act = doom_rules[-1][1] if doom_rules else None
                        if last_act == "allow":
                            doom_allowed = True
                            doom = None
                        elif last_act == "deny":
                            doom = PermissionVerdict.deny("doom_loop denied by permission", risk=RiskLevel.HIGH, policy="rules")
                    break

        pending_doom = doom if not doom_allowed else None
        pending_ext = ext_verdict

        # Ask each policy in order
        for policy in self.policies:
            try:
                verdict = await policy.evaluate(request)
            except Exception as e:
                logger.warning(f"Policy '{policy.name}' raised: {e}")
                continue

            if verdict is None:
                continue

            # If policy would allow but we have pending doom/external ask, prefer that ask/deny
            if verdict.decision == Decision.ALLOW and (pending_doom or pending_ext):
                # doom takes priority over external over allow
                if pending_doom:
                    if self.mode == "auto" and pending_doom.risk != RiskLevel.CRITICAL:
                        self.stats["auto_approved"] += 1
                        return PermissionVerdict.allow(f"auto doom_loop", risk=pending_doom.risk, policy="doom_loop")
                    return pending_doom
                if pending_ext:
                    if self.mode == "auto":
                        self.stats["auto_approved"] += 1
                        return PermissionVerdict.allow(f"auto external_directory", risk=pending_ext.risk, policy="external_directory")
                    return pending_ext

            # Policy made a decision
            if verdict.decision == Decision.ASK and self.mode == "auto":
                # auto mode upgrades ASK to ALLOW except for CRITICAL risk or explicit deny
                if verdict.risk != RiskLevel.CRITICAL or self.allow_dangerous:
                    # but if doom/external pending, keep ask
                    if pending_doom or pending_ext:
                        return pending_doom or pending_ext  # type: ignore
                    self.stats["auto_approved"] += 1
                    return PermissionVerdict.allow(
                        f"auto (policy={policy.name}, was ask)",
                        risk=verdict.risk,
                        policy=policy.name,
                    )
            return verdict

        # No policy had an opinion → use doom/ext defaults
        if pending_doom is not None:
            if self.mode == "auto" and pending_doom.risk != RiskLevel.CRITICAL:
                self.stats["auto_approved"] += 1
                return PermissionVerdict.allow(f"auto doom_loop", risk=pending_doom.risk, policy="doom_loop")
            return pending_doom
        if pending_ext is not None:
            if self.mode == "auto":
                self.stats["auto_approved"] += 1
                return PermissionVerdict.allow(f"auto external_directory", risk=pending_ext.risk, policy="external_directory")
            return pending_ext

        # opencode permissive defaults (pass full request for .env handling)
        default_by_tool = self._default_for_tool(request)
        if default_by_tool is not None:
            return default_by_tool

        # No policy had an opinion → default
        return PermissionVerdict(
            decision=self.default_decision,
            reason="no policy matched; using default",
            risk=RiskLevel.MEDIUM,
            policy="default",
        )

    # ------------------------------------------------------------------
    # USER CONFIRMATION
    # ------------------------------------------------------------------

    async def _ask_user(
        self,
        request: PermissionRequest,
        verdict: PermissionVerdict,
    ) -> bool:
        self.stats["asked"] += 1

        # Prefer the injected callback
        if self.confirm_callback is not None:
            try:
                return bool(await self.confirm_callback(request, verdict))
            except Exception as e:
                logger.error(f"Confirm callback failed: {e}")

        # Fall back to UI + input handler
        if self.ui:
            self._render_prompt(request, verdict)

        if self.input_handler:
            try:
                if hasattr(self.input_handler, "get_confirmation"):
                    return await self.input_handler.get_confirmation(
                        f"Allow {request.tool}.{request.action}?",
                        default=False,
                    )
                if hasattr(self.input_handler, "get_input"):
                    ans = await self.input_handler.get_input(prompt="Allow? [y/N] ")
                    return (ans or "").strip().lower() in ("y", "yes")
            except Exception as e:
                logger.warning(f"Input handler error: {e}")

        # Last resort: deny
        logger.warning(
            f"No confirmation mechanism available for {request.tool}.{request.action} — denying. "
            f"Risk: {verdict.risk.value}. Configure input_handler or set permissions.auto_approve=true"
        )
        return False

    def _render_prompt(
        self, request: PermissionRequest, verdict: PermissionVerdict
    ) -> None:
        if not self.ui:
            return
        try:
            from agent.cli.ui import Icons, Palette

            risk_colors = {
                RiskLevel.SAFE: Palette.SUCCESS,
                RiskLevel.LOW: Palette.INFO,
                RiskLevel.MEDIUM: Palette.WARNING,
                RiskLevel.HIGH: Palette.ERROR,
                RiskLevel.CRITICAL: Palette.ERROR,
            }
            color = risk_colors.get(verdict.risk, Palette.WARNING)

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
        # Group by tool + action + primary target (e.g. path/command)
        target = (
            request.params.get("path")
            or request.params.get("command")
            or request.params.get("cmd")
            or ""
        )
        return f"{request.tool}:{request.action}:{target}"

    def _check_session_cache(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
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
        self._session_allows[self._cache_key(request)] = (
            time.time() + self._session_ttl
        )

    def _remember_deny(self, request: PermissionRequest) -> None:
        self._session_denies[self._cache_key(request)] = (
            time.time() + self._session_ttl
        )

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
        logger.info(f"Auto-approve set to: {self.auto_approve}")

    def add_allow_rule(self, pattern: str) -> None:
        self._session_allows[pattern] = time.time() + self._session_ttl

    def add_deny_rule(self, pattern: str) -> None:
        self._session_denies[pattern] = time.time() + self._session_ttl

    def reset_rules(self) -> None:
        self._session_allows.clear()
        self._session_denies.clear()
        for policy in self.policies:
            if hasattr(policy, "reset"):
                policy.reset()

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "auto_approve": self.auto_approve,
            "allow_dangerous": self.allow_dangerous,
            "default": self.default_decision.value,
            "policies": [p.name for p in self.policies],
            "session_allows": len(self._session_allows),
            "session_denies": len(self._session_denies),
            "stats": dict(self.stats),
        }

    def _infer_action(self, tool: str, params: Dict[str, Any]) -> str:
        """Best-effort action inference for compatibility callers."""
        t = tool.lower()
        if "read" in t or t in ("filesystem_read",):
            return "read"
        if "write" in t or "edit" in t:
            return "write"
        if "delete" in t or "remove" in t or "rm" in t:
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