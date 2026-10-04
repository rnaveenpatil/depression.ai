"""
AWS Policy - Safety gate for AWS operations.

Handles both:
    - Direct AWS tool calls (aws_*, s3_*, ec2_*, iam_*, mcp__aws__*)
    - The unified `aws` tool (params: {service, operation, args})
    - Raw `aws` CLI commands run via the terminal/bash tool

Risk classification:
    SAFE      → describe-*, list-*, get-*  (read-only)
    LOW       → create-*, put-*, copy-*, start-*
    MEDIUM    → modify-*, update-*, tag-*
    HIGH      → delete-* (single resource), stop-*, terminate-* (single)
    CRITICAL  → anything with --force, all deletes with wildcards, IAM changes
"""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Optional

from agent.permissions.manager import (
    PermissionRequest,
    PermissionVerdict,
    Policy,
    RiskLevel,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


SAFE_AWS_PREFIXES = ("describe-", "list-", "get-", "head-", "ls")
LOW_AWS_PREFIXES = ("create-", "put-", "copy-", "start-", "enable-", "run-")
MEDIUM_AWS_PREFIXES = ("update-", "modify-", "tag-", "attach-", "detach-", "associate-")
HIGH_AWS_PREFIXES = ("delete-", "stop-", "terminate-", "remove-", "revoke-", "disassociate-")

CRITICAL_AWS_PATTERNS = [
    r"--force\b",
    r"--recursive\b.*\bs3\s+rm\b",
    r"\bs3\s+rm\b.*--recursive",
    r"iam\s+.*delete-",
    r"organizations\s+.*delete-",
    r"route53\s+.*delete-",
    r"ec2\s+.*delete-vpc",
    r"ec2\s+.*delete-subnet",
    r"ec2\s+.*delete-security-group",
    r"cloudtrail\s+.*delete-trail",
    r"kms\s+.*schedule-key-deletion",
    r"rds\s+.*delete-db-instance.*--skip-final-snapshot",
]


class AWSPolicy(Policy):
    name = "aws"

    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.enabled: bool = cfg.get("aws_enabled", True)
        self.read_only: bool = cfg.get("aws_read_only", False)
        self.require_confirm_delete: bool = cfg.get("aws_confirm_delete", True)
        self.allowed_profiles: List[str] = list(cfg.get("aws_allowed_profiles", []))
        self.allowed_regions: List[str] = list(cfg.get("aws_allowed_regions", []))
        self.denied_services: List[str] = list(cfg.get("aws_denied_services", []))

        self._critical_res = [re.compile(p) for p in CRITICAL_AWS_PATTERNS]

    async def evaluate(
        self, request: PermissionRequest
    ) -> Optional[PermissionVerdict]:
        if not self.enabled:
            return None

        op = self._extract_aws_op(request)
        if not op:
            return None

        service, action, args = op

        if service in self.denied_services:
            return PermissionVerdict.deny(
                f"AWS service '{service}' is blocked",
                risk=RiskLevel.CRITICAL,
                policy=self.name,
            )

        profile = self._extract_flag(args, "--profile")
        if profile and self.allowed_profiles and profile not in self.allowed_profiles:
            return PermissionVerdict.deny(
                f"AWS profile '{profile}' not in allowlist",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        region = self._extract_flag(args, "--region")
        if region and self.allowed_regions and region not in self.allowed_regions:
            return PermissionVerdict.deny(
                f"AWS region '{region}' not in allowlist",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        cmd_str = " ".join(args)
        for r in self._critical_res:
            if r.search(cmd_str):
                return PermissionVerdict.ask(
                    f"critical AWS operation: {service} {action}",
                    risk=RiskLevel.CRITICAL,
                    policy=self.name,
                )

        if action.startswith(SAFE_AWS_PREFIXES):
            return PermissionVerdict.allow(
                f"read-only AWS: {service} {action}",
                risk=RiskLevel.SAFE,
                policy=self.name,
            )

        if self.read_only:
            return PermissionVerdict.deny(
                f"read-only mode blocks AWS write: {service} {action}",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        if action.startswith(LOW_AWS_PREFIXES):
            return PermissionVerdict.ask(
                f"AWS write: {service} {action}",
                risk=RiskLevel.LOW,
                policy=self.name,
            )

        if action.startswith(MEDIUM_AWS_PREFIXES):
            return PermissionVerdict.ask(
                f"AWS modify: {service} {action}",
                risk=RiskLevel.MEDIUM,
                policy=self.name,
            )

        if action.startswith(HIGH_AWS_PREFIXES):
            if self._has_wildcard(cmd_str):
                return PermissionVerdict.ask(
                    f"AWS destructive with wildcard: {service} {action}",
                    risk=RiskLevel.CRITICAL,
                    policy=self.name,
                )
            if self.require_confirm_delete:
                return PermissionVerdict.ask(
                    f"AWS destructive: {service} {action}",
                    risk=RiskLevel.HIGH,
                    policy=self.name,
                )
            return PermissionVerdict.allow(
                f"AWS destructive (auto-approved): {service} {action}",
                risk=RiskLevel.HIGH,
                policy=self.name,
            )

        return PermissionVerdict.ask(
            f"AWS: {service} {action}",
            risk=RiskLevel.MEDIUM,
            policy=self.name,
        )

    # ------------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------------

    def _extract_aws_op(
        self, request: PermissionRequest
    ) -> Optional[tuple]:
        """Return (service, action, args) or None."""
        tool = (request.tool or "").lower()
        params = request.params or {}

        # Case 1: unified `aws` tool. Params carry service + operation.
        if tool == "aws":
            service = str(params.get("service") or "").strip().lower()
            operation = str(params.get("operation") or "").strip().lower()
            extra = list(params.get("args") or [])
            if service and operation:
                return service, operation, extra
            # Identity action
            if str(params.get("action") or "").lower() == "identity":
                return "sts", "get-caller-identity", []
            return None

        # Case 2: MCP AWS tool with explicit operation string.
        if tool.startswith("mcp__aws__"):
            operation = str(
                params.get("operation") or params.get("command") or ""
            ).strip()
            if operation:
                parts = operation.split(None, 1)
                if len(parts) == 2:
                    return parts[0].lower(), parts[1].lower(), []
            cleaned = tool.replace("mcp__aws__", "")
            parts = cleaned.split("_", 1)
            if len(parts) == 2:
                return parts[0].lower(), parts[1].replace("_", "-").lower(), []
            return cleaned.lower(), "unknown", []

        # Case 3: direct tool names.
        if tool.startswith("aws_") or tool.startswith("s3_") or tool.startswith("ec2_") or tool.startswith("iam_"):
            cleaned = tool.replace("aws_", "")
            parts = cleaned.split("_", 1)
            if len(parts) == 2:
                service, action = parts
                return service.lower(), action.replace("_", "-").lower(), []
            return cleaned.lower(), "unknown", []

        # Case 4: aws CLI via terminal/bash.
        command = (
            params.get("command")
            or params.get("cmd")
            or params.get("shell")
            or ""
        )
        if not command:
            return None

        try:
            parts = shlex.split(command, posix=True)
        except ValueError:
            return None

        for i, p in enumerate(parts):
            if p == "aws":
                rest = parts[i + 1:]
                break
        else:
            return None

        if not rest:
            return None

        service = None
        idx = 0
        while idx < len(rest):
            tok = rest[idx]
            if tok.startswith("--"):
                idx += 2
                continue
            service = tok
            idx += 1
            break

        if not service:
            return None

        action = rest[idx] if idx < len(rest) else "unknown"
        return service.lower(), action.lower(), rest[idx + 1:]

    def _extract_flag(self, args: List[str], flag: str) -> Optional[str]:
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    def _has_wildcard(self, s: str) -> bool:
        return "*" in s or "?" in s

    def reset(self) -> None:
        pass