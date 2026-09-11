"""
Permissions Module - Central authorization gate

Exports:
    PermissionManager   — Main permission orchestrator
    PermissionRequest   — A permission check request
    PermissionVerdict   — Result of a permission check
    Decision            — Allow/Deny/Ask enum
    RiskLevel           — Risk level enum
    Policy              — Base policy class
"""

from agent.permissions.manager import (
    PermissionManager,
    PermissionRequest,
    PermissionVerdict,
    Decision,
    RiskLevel,
    Policy,
)

__all__ = [
    "PermissionManager",
    "PermissionRequest",
    "PermissionVerdict",
    "Decision",
    "RiskLevel",
    "Policy",
]