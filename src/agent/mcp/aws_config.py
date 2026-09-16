"""
AWS MCP server configuration builder.

Reads AWS credentials from .env and builds a StdioTransport config that
passes them to the child process ONLY (never polluting the parent env).

The AWS MCP server runs `awslabs.core-mcp-server` via uvx, which uses
the standard AWS credential chain.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from agent.utils.logging import get_logger
from agent.utils.env_manager import get_aws_credentials

logger = get_logger(__name__)


DEFAULT_AWS_REGION = "ap-south-1"
AWS_MCP_PACKAGE = "awslabs.core-mcp-server@latest"


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def build_aws_mcp_config(
    region: Optional[str] = None,
    profile: Optional[str] = None,
    enabled: Optional[bool] = None,
    server_name: str = "aws",
) -> Optional[Dict[str, Any]]:
    """
    Return an MCP server config dict for the AWS MCP server, or None if
    credentials or tooling are missing.

    Credentials are placed in the child process env only. They are NEVER
    read from os.environ here (which the parent shares with every
    subprocess, including the user's shell).
    """
    creds = get_aws_credentials()
    access_key = creds.get("access_key")
    secret_key = creds.get("secret_key")
    resolved_region = region or creds.get("region") or DEFAULT_AWS_REGION

    if enabled is False:
        return None

    if not _uvx_available():
        logger.info(
            "uvx not found; AWS MCP server disabled. "
            "Install with: pip install uv (provides the uvx CLI)"
        )
        return None

    if not access_key or not secret_key:
        logger.info(
            "AWS credentials missing in .env; AWS MCP server disabled. "
            "Add them via the TUI /aws panel."
        )
        return None

    child_env: Dict[str, str] = {
        "AWS_ACCESS_KEY_ID": access_key,
        "AWS_SECRET_ACCESS_KEY": secret_key,
        "AWS_DEFAULT_REGION": resolved_region,
        "AWS_REGION": resolved_region,
        "FASTMCP_LOG_LEVEL": "ERROR",
    }
    if profile:
        child_env["AWS_PROFILE"] = profile

    return {
        "name": server_name,
        "transport": "stdio",
        "command": "uvx",
        "args": [AWS_MCP_PACKAGE],
        "env": child_env,
        "timeout": 90.0,
        "auto_reconnect": True,
        "max_reconnect_attempts": 3,
    }


def build_aws_cli_fallback_status() -> Dict[str, Any]:
    """
    Report whether the bash/terminal AWS CLI fallback is viable.

    Used by the agent bootstrap to decide what to put in the system prompt.
    """
    has_cli = _aws_cli_available()
    creds = get_aws_credentials()
    has_creds = bool(creds.get("access_key") and creds.get("secret_key"))
    return {
        "cli_available": has_cli,
        "credentials_present": has_creds,
        "region": creds.get("region") or DEFAULT_AWS_REGION,
        "usable": has_cli and has_creds,
    }


def install_aws_preset_into_config(
    mcp_config: Dict[str, Any],
    region: Optional[str] = None,
) -> bool:
    """
    Mutate an existing MCP config dict to add or update the AWS server.

    Returns True if the AWS server was added, False otherwise.
    Called from the TUI when the user saves AWS credentials.
    """
    server = build_aws_mcp_config(region=region)
    if server is None:
        return False

    servers = mcp_config.setdefault("servers", [])
    if not isinstance(servers, list):
        return False

    # Replace existing AWS server if present.
    servers[:] = [s for s in servers if s.get("name") != server["name"]]
    servers.append(server)
    mcp_config["enabled"] = True
    return True


__all__ = [
    "build_aws_mcp_config",
    "build_aws_cli_fallback_status",
    "install_aws_preset_into_config",
    "AWS_MCP_PACKAGE",
    "DEFAULT_AWS_REGION",
]