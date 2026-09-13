"""
Rules Policy - Pattern-based allow/deny rules from config.

Supports both legacy flat lists and opencode-compatible granular permission map:
  permission = {
    "edit": "deny",
    "bash": {"*": "ask", "git *": "allow", "rm *": "deny"},
    "read": {"*.env": "deny", "*.env.example": "allow", "*": "allow"},
  }
Last matching rule wins (opencode semantics).
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.permissions.manager import (
    Decision,
    PermissionRequest,
    PermissionVerdict,
    Policy,
    RiskLevel,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)

# Map opencode permission keys to internal tool aliases
PERMISSION_KEY_ALIASES: Dict[str, List[str]] = {
    "edit": ["edit", "write", "apply_patch", "patch", "filesystem"],
    "read": ["read", "filesystem"],
    "glob": ["glob", "filesystem"],
    "grep": ["grep", "search"],
    "bash": ["bash", "terminal", "shell"],
    "task": ["task"],
    "todowrite": ["todowrite", "todoread", "todo"],
    "todo": ["todo", "todowrite", "todoread"],
    "webfetch": ["webfetch", "web"],
    "websearch": ["websearch", "web"],
    "skill": ["skill"],
    "question": ["question"],
    "lsp": ["lsp", "diagnostics"],
    "external_directory": ["external_directory"],
    "doom_loop": ["doom_loop"],
}


def _expand_home(pattern: str) -> str:
    if pattern.startswith("~/") or pattern == "~":
        return os.path.expanduser(pattern)
    if pattern.startswith("$HOME/"):
        return os.path.expanduser("~" + pattern[5:])
    return pattern


def _wildcard_match(text: str, pattern: str) -> bool:
    pat = _expand_home(pattern)
    # also try basename for patterns like *.env matching /project/.env
    base = os.path.basename(text)
    return (
        fnmatch.fnmatch(text, pat)
        or fnmatch.fnmatch(text.lower(), pat.lower())
        or fnmatch.fnmatch(base, pat)
        or fnmatch.fnmatch(base.lower(), pat.lower())
        or fnmatch.fnmatch(text, f"*{pat}")
        or fnmatch.fnmatch(text.lower(), f"*{pat.lower()}")
    )


def _extract_target(request: PermissionRequest) -> str:
    # opencode granular rules match on different slices:
    #  bash -> raw command, read/edit -> file path, grep -> pattern,
    #  webfetch -> url, websearch -> query, skill -> name, etc.
    tool = request.tool.lower()
    params = request.params or {}
    if tool in ("bash", "terminal", "shell"):
        return params.get("command") or params.get("cmd") or ""
    if tool in ("read", "write", "edit", "apply_patch", "patch", "glob", "grep", "lsp"):
        return params.get("filePath") or params.get("path") or params.get("pattern") or params.get("file") or ""
    if tool in ("webfetch",):
        return params.get("url") or ""
    if tool in ("websearch", "search", "web"):
        return params.get("query") or params.get("url") or ""
    if tool in ("skill",):
        return params.get("name") or params.get("skill") or ""
    if tool in ("task",):
        return params.get("prompt") or params.get("description") or ""
    return params.get("path") or params.get("command") or params.get("query") or params.get("url") or ""


class RulesPolicy(Policy):
    name = "rules"

    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.allow_patterns: List[str] = list(cfg.get("allow", []))
        self.deny_patterns: List[str] = list(cfg.get("deny", []))
        self.ask_patterns: List[str] = list(cfg.get("ask", []))

        # Also accept structured rule lists
        for rule in cfg.get("rules", []):
            pattern = rule.get("pattern")
            action = rule.get("action", "allow").lower()
            if not pattern:
                continue
            if action == "allow":
                self.allow_patterns.append(pattern)
            elif action == "deny":
                self.deny_patterns.append(pattern)
            elif action == "ask":
                self.ask_patterns.append(pattern)

        # opencode permission map: cfg may contain keys like edit/read/bash etc.
        # Also handle nested under `permission` key
        perm_map: Dict[str, Any] = {}
        if "permission" in cfg and isinstance(cfg["permission"], dict):
            perm_map.update(cfg["permission"])
        # Also top-level keys that look like permission names
        for k, v in cfg.items():
            if k in PERMISSION_KEY_ALIASES and k not in perm_map:
                perm_map[k] = v
            # handle wildcard permission "*" globally
            if k == "*":
                perm_map[k] = v

        # Normalize: store ordered rules per permission key
        self.permission_rules: Dict[str, List[Tuple[str, str]]] = {}
        for key, val in perm_map.items():
            if isinstance(val, str):
                self.permission_rules[key] = [("*", val.lower())]
            elif isinstance(val, dict):
                # preserve insertion order; last wins
                rules: List[Tuple[str, str]] = []
                for pat, act in val.items():
                    if isinstance(act, str):
                        rules.append((pat, act.lower()))
                if rules:
                    self.permission_rules[key] = rules

        logger.debug(
            f"RulesPolicy: {len(self.allow_patterns)} allow, "
            f"{len(self.deny_patterns)} deny, {len(self.ask_patterns)} ask, "
            f"{len(self.permission_rules)} permission keys"
        )

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        # First: opencode granular permission map (last-rule-wins)
        granular = self._evaluate_granular(request)
        if granular is not None:
            return granular

        # Legacy flat lists (deny > ask > allow)
        signature = self._signature(request)

        for pattern in self.deny_patterns:
            if self._matches(pattern, signature):
                return PermissionVerdict.deny(
                    f"matched deny rule '{pattern}'",
                    risk=RiskLevel.HIGH,
                    policy=self.name,
                )

        for pattern in self.ask_patterns:
            if self._matches(pattern, signature):
                return PermissionVerdict.ask(
                    f"matched ask rule '{pattern}'",
                    risk=RiskLevel.MEDIUM,
                    policy=self.name,
                )

        for pattern in self.allow_patterns:
            if self._matches(pattern, signature):
                return PermissionVerdict.allow(
                    f"matched allow rule '{pattern}'",
                    risk=RiskLevel.SAFE,
                    policy=self.name,
                )

        return None

    def _evaluate_granular(self, request: PermissionRequest) -> Optional[PermissionVerdict]:
        tool = request.tool.lower()
        # Find which permission key(s) govern this tool
        governing_keys: List[str] = []
        for perm_key, aliases in PERMISSION_KEY_ALIASES.items():
            if tool in aliases or tool == perm_key:
                governing_keys.append(perm_key)
        # also check wildcard "*"
        if "*" in self.permission_rules:
            governing_keys.append("*")
        if not governing_keys:
            return None

        target = _extract_target(request)
        verdict: Optional[PermissionVerdict] = None

        for key in governing_keys:
            rules = self.permission_rules.get(key)
            if not rules:
                continue
            matched_action: Optional[str] = None
            matched_pat: Optional[str] = None
            for pat, act in rules:
                if pat == "*":
                    matched_action = act
                    matched_pat = pat
                elif _wildcard_match(target, pat) or _wildcard_match(f"{tool}:{target}", pat):
                    matched_action = act
                    matched_pat = pat
            if matched_action:
                if matched_action == "allow":
                    verdict = PermissionVerdict.allow(
                        f"permission {key}:{matched_pat} → allow", risk=RiskLevel.SAFE, policy=self.name
                    )
                elif matched_action == "deny":
                    verdict = PermissionVerdict.deny(
                        f"permission {key}:{matched_pat} → deny", risk=RiskLevel.HIGH, policy=self.name
                    )
                elif matched_action == "ask":
                    verdict = PermissionVerdict.ask(
                        f"permission {key}:{matched_pat} → ask", risk=RiskLevel.MEDIUM, policy=self.name
                    )
        return verdict

    # ------------------------------------------------------------------

    def _signature(self, request: PermissionRequest) -> str:
        """Build a canonical string for matching."""
        target = (
            request.params.get("path")
            or request.params.get("filePath")
            or request.params.get("command")
            or request.params.get("cmd")
            or request.params.get("bucket")
            or ""
        )
        return f"{request.tool}:{request.action}:{target}"

    def _matches(self, pattern: str, signature: str) -> bool:
        # Support both ':' and '*' style patterns
        if pattern == "*":
            return True
        return fnmatch.fnmatch(signature, pattern) or fnmatch.fnmatch(
            signature.lower(), pattern.lower()
        )

    def reset(self) -> None:
        # Config-sourced rules are immutable; only session rules live in manager
        pass