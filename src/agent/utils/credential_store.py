"""Single write bridge for credentials stored in the project .env file.

All TUI/runtime credential saves go through this module. It merges the
requested values into the existing file, writes once, mirrors the values
into ``os.environ``, and asks ``EnvManager`` to reload its cache/listeners.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from agent.utils.env_manager import (
    AWS_ENVS,
    EnvManager,
    credential_env_path,
    load_env_file,
    write_env_file,
)

LLM_ENVS = {
    "base_url": "DEPRESSION_BASE_URL",
    "api_key": "DEPRESSION_API_KEY",
    "model": "DEPRESSION_MODEL",
    "provider": "DEPRESSION_PROVIDER",
}


def save_credentials(
    updates: Mapping[str, Optional[str]],
    *,
    env_path: Optional[Path] = None,
    remove_empty: bool = False,
) -> Dict[str, str]:
    """Merge credential values into one .env file and refresh live state.

    ``None`` values are ignored. Empty strings are preserved unless
    ``remove_empty`` is true, which lets callers clear a key entirely.
    """
    target = env_path or credential_env_path()
    current = load_env_file(target)
    applied: Dict[str, str] = {}

    for key, value in updates.items():
        if value is None:
            continue
        if value == "" and remove_empty:
            current.pop(key, None)
        else:
            current[key] = value
        applied[key] = value

    write_env_file(current, target, strict=True, replace=True)

    for key, value in applied.items():
        if value == "" and remove_empty:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    EnvManager.get().reload()
    return applied


def save_llm_credentials(
    base_url: str,
    api_key: str,
    model: str,
    provider: str,
    *,
    env_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Persist the complete LLM connection in one write."""
    return save_credentials(
        {
            LLM_ENVS["base_url"]: base_url,
            LLM_ENVS["api_key"]: api_key,
            LLM_ENVS["model"]: model,
            LLM_ENVS["provider"]: provider,
        },
        env_path=env_path,
    )


def save_aws_credentials(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
    session_token: Optional[str] = None,
    *,
    env_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Persist the AWS credential set in one write."""
    return save_credentials(
        {
            AWS_ENVS["access_key"]: access_key,
            AWS_ENVS["secret_key"]: secret_key,
            AWS_ENVS["region"]: region,
            AWS_ENVS["session_token"]: session_token,
        },
        env_path=env_path,
        remove_empty=True,
    )


def save_mcp_secrets(
    secrets: Mapping[str, str],
    *,
    env_path: Optional[Path] = None,
    remove_empty: bool = False,
) -> Dict[str, str]:
    """Persist MCP secret values in one write.

    ``remove_empty`` lets callers remove keys that belonged to a deleted MCP
    server without introducing a second write path for the credential file.
    """
    return save_credentials(
        secrets, env_path=env_path, remove_empty=remove_empty,
    )


__all__ = [
    "credential_env_path",
    "save_aws_credentials",
    "save_credentials",
    "save_llm_credentials",
    "save_mcp_secrets",
]
