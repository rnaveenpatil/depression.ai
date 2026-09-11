"""
Rules Policy - Pattern-based allow/deny rules from config.

Rules use glob-style patterns:
    "terminal:execute:git *"        → allow all git commands
    "filesystem:write:/etc/*"       → deny writes to /etc
    "aws:s3_delete:*"               → ask before any S3 delete
"""

from __future__ import annotations

import fnmatch
from typing import Any, Dict, List, Optional

from agent.permissions.manager import (
    Decision,
    PermissionRequest,
    PermissionVerdict,
    Policy,
    RiskLevel,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


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

        logger.debug(
            f"RulesPolicy: {len(self.allow_patterns)} allow, "
            f"{len(self.deny_patterns)} deny, {len(self.ask_patterns)} ask"
        )

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        signature = self._signature(request)

        # Deny wins over allow
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

    # ------------------------------------------------------------------

    def _signature(self, request: PermissionRequest) -> str:
        """Build a canonical string for matching."""
        target = (
            request.params.get("path")
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