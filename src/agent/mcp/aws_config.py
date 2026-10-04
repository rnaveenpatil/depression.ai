"""
AWS MCP server configuration builder.

Credential sources (in order):
    1. EnvManager.get_aws_env() — writes to ~/.agent/env, 0600
    2. ~/.aws/credentials + ~/.aws/config — for users with existing AWS setup

The AWS MCP server runs `awslabs.core-mcp-server` via uvx and uses the
standard AWS credential chain.
"""

from __future__ import annotations

import configparser
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from agent.utils.logging import get_logger
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


DEFAULT_AWS_REGION = "us-east-1"
AWS_MCP_PACKAGE = "awslabs.core-mcp-server@latest"


def _uvx_available() -> bool:
    return shutil.which("uvx") is not None


def _aws_cli_available() -> bool:
    return shutil.which("aws") is not None


def _read_aws_file_credentials(profile: Optional[str] = None) -> Dict[str, str]:
    """Fallback: read ~/.aws/credentials + ~/.aws/config."""
    out: Dict[str, str] = {}
    section = profile or "default"

    creds_path = Path(os.path.expanduser("~/.aws/credentials"))
    cfg_path = Path(os.path.expanduser("~/.aws/config"))

    if creds_path.is_file():
        try:
            parser = configparser.RawConfigParser()
            parser.read(creds_path)
            if parser.has_section(section):
                for k, v in parser.items(section):
                    out[k] = v
        except Exception as exc:
            logger.debug("Could not read %s: %s", creds_path, exc)

    if cfg_path.is_file():
        try:
            parser = configparser.RawConfigParser()
            parser.read(cfg_path)
            cfg_section = section if section == "default" else f"profile {section}"
            if parser.has_section(cfg_section):
                for k, v in parser.items(cfg_section):
                    out.setdefault(k, v)
        except Exception as exc:
            logger.debug("Could not read %s: %s", cfg_path, exc)

    return out


def _resolve_creds(profile: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Canonical credential read. EnvManager first, ~/.aws/credentials second.
    """
    # 1. EnvManager
    try:
        env = EnvManager.get().get_aws_env()
    except Exception:
        env = {}

    access = env.get("AWS_ACCESS_KEY_ID")
    secret = env.get("AWS_SECRET_ACCESS_KEY")
    token = env.get("AWS_SESSION_TOKEN")
    region = env.get("AWS_DEFAULT_REGION")

    if access and secret:
        return {
            "access_key": access,
            "secret_key": secret,
            "session_token": token,
            "region": region,
            "source": "env_manager",
        }

    # 2. ~/.aws/credentials
    file_creds = _read_aws_file_credentials(profile)
    if file_creds.get("aws_access_key_id") and file_creds.get("aws_secret_access_key"):
        return {
            "access_key": file_creds["aws_access_key_id"],
            "secret_key": file_creds["aws_secret_access_key"],
            "session_token": file_creds.get("aws_session_token"),
            "region": file_creds.get("region"),
            "source": "aws_file",
        }

    return {
        "access_key": None,
        "secret_key": None,
        "session_token": None,
        "region": None,
        "source": "none",
    }


def _resolve_region(
    explicit: Optional[str],
    from_creds: Optional[str],
) -> str:
    return explicit or from_creds or DEFAULT_AWS_REGION


def has_aws_credentials(profile: Optional[str] = None) -> bool:
    """True if we can perform any AWS operation at all."""
    creds = _resolve_creds(profile)
    return bool(creds.get("access_key") and creds.get("secret_key"))


def build_aws_mcp_config(
    region: Optional[str] = None,
    profile: Optional[str] = None,
    enabled: Optional[bool] = None,
    server_name: str = "aws",
) -> Optional[Dict[str, Any]]:
    """
    Return an MCP server config dict for the AWS MCP server, or None if
    credentials or tooling are missing.
    """
    if enabled is False:
        return None

    if not _uvx_available():
        logger.info(
            "uvx not found; AWS MCP server disabled. "
            "Install with: pip install uv (provides the uvx CLI)"
        )
        return None

    creds = _resolve_creds(profile)
    access_key = creds.get("access_key")
    secret_key = creds.get("secret_key")
    session_token = creds.get("session_token")
    resolved_region = _resolve_region(region, creds.get("region"))

    if not access_key or not secret_key:
        logger.info(
            "AWS credentials missing (checked EnvManager and ~/.aws/credentials); "
            "AWS MCP server disabled."
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


def build_aws_cli_fallback_status(profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Report whether the AWS CLI fallback is viable.
    """
    has_cli = _aws_cli_available()
    creds = _resolve_creds(profile)
    has_creds = bool(creds.get("access_key") and creds.get("secret_key"))
    return {
        "cli_available": has_cli,
        "credentials_present": has_creds,
        "credentials_source": creds.get("source"),
        "region": creds.get("region") or DEFAULT_AWS_REGION,
        "usable": has_cli and has_creds,
    }


def install_aws_preset_into_config(
    mcp_config: Dict[str, Any],
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Mutate an MCP config dict to add or update the AWS server.

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


def rebuild_mcp_aws_servers(mcp_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force-refresh the AWS MCP server entry in an MCP config.

    Called after saving AWS credentials so a previously-disabled AWS
    server (disabled because creds were missing at boot) gets turned on.
    """
    return install_aws_preset_into_config(mcp_config, region=None)


__all__ = [
    "build_aws_mcp_config",
    "build_aws_cli_fallback_status",
    "install_aws_preset_into_config",
    "rebuild_mcp_aws_servers",
    "has_aws_credentials",
    "AWS_MCP_PACKAGE",
    "DEFAULT_AWS_REGION",
]