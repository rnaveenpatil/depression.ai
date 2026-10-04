"""
AWS MCP server configuration builder.

Reads AWS credentials from the canonical EnvManager (which is updated live
by the TUI when the user saves keys) and builds a StdioTransport config
that passes them to the child process ONLY — never into the parent env.

The AWS MCP server runs `awslabs.core-mcp-server` via uvx, which uses
the standard AWS credential chain.

Contract:
    * get_aws_env() is the ONLY credential source.
    * Credentials land in the returned dict's "env" field. Nothing here
      writes to os.environ.
    * On credential save, the TUI fires an event; the MCP client reloads
      this server (mcp_client.reload()) which respawns the child with the
      new env returned by this function.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Optional

from agent.utils.logging import get_logger
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


DEFAULT_AWS_REGION = "ap-south-1"
AWS_MCP_PACKAGE = "awslabs.core-mcp-server@latest"


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _resolve_creds() -> Dict[str, Optional[str]]:
    """
    Canonical credential read. Always goes through EnvManager so a save
    via the TUI is visible here immediately.
    """
    try:
        env = EnvManager.get().get_aws_env()
    except Exception:
        env = {}
    return {
        "access_key": env.get("AWS_ACCESS_KEY_ID"),
        "secret_key": env.get("AWS_SECRET_ACCESS_KEY"),
        "session_token": env.get("AWS_SESSION_TOKEN"),
        "region": env.get("AWS_DEFAULT_REGION"),
    }


def _resolve_region(
    explicit: Optional[str],
    from_creds: Optional[str],
) -> str:
    return explicit or from_creds or DEFAULT_AWS_REGION


def build_aws_mcp_config(
    region: Optional[str] = None,
    profile: Optional[str] = None,
    enabled: Optional[bool] = None,
    server_name: str = "aws",
) -> Optional[Dict[str, Any]]:
    """
    Return an MCP server config dict for the AWS MCP server, or None if
    credentials or tooling are missing.

    The child env dict is a fresh copy; nothing here mutates os.environ.
    """
    if enabled is False:
        return None

    creds = _resolve_creds()
    access_key = creds.get("access_key")
    secret_key = creds.get("secret_key")
    session_token = creds.get("session_token")
    resolved_region = _resolve_region(region, creds.get("region"))

    if not _uvx_available():
        logger.info(
            "uvx not found; AWS MCP server disabled. "
            "Install with: pip install uv (provides the uvx CLI)"
        )
        return None

    if not access_key or not secret_key:
        logger.info(
            "AWS credentials missing; AWS MCP server disabled. "
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
    if session_token:
        child_env["AWS_SESSION_TOKEN"] = session_token
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

    Reads from the canonical source so the TUI, system prompt, and this
    module always agree.
    """
    has_cli = _aws_cli_available()
    creds = _resolve_creds()
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
) -> Dict[str, Any]:
    """
    Mutate an existing MCP config dict to add or update the AWS server.

    Returns:
        {"installed": bool, "replaced": bool, "reason": str}
    """
    server = build_aws_mcp_config(region=region)
    if server is None:
        return {
            "installed": False,
            "replaced": False,
            "reason": "AWS MCP unavailable (missing creds or uvx)",
        }

    servers = mcp_config.setdefault("servers", [])
    if not isinstance(servers, list):
        return {
            "installed": False,
            "replaced": False,
            "reason": "'servers' is not a list",
        }

    existing = [s for s in servers if s.get("name") == server["name"]]
    servers[:] = [s for s in servers if s.get("name") != server["name"]]
    servers.append(server)
    mcp_config["enabled"] = True

    return {
        "installed": True,
        "replaced": bool(existing),
        "reason": "ok",
    }


__all__ = [
    "build_aws_mcp_config",
    "build_aws_cli_fallback_status",
    "install_aws_preset_into_config",
    "AWS_MCP_PACKAGE",
    "DEFAULT_AWS_REGION",
]