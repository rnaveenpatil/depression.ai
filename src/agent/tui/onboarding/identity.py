"""
Signed-in user identity, exposed exactly like every other credential.

Storage:
    1. `~/.agent/env` — the same global, 0600 file that holds the AWS keys
       and the provider API keys. Written through `EnvManager` so it stays
       consistent with the live process state.
    2. `os.environ` — mirrored live, so `bash`, MCP servers and any
       subprocess the agent spawns inherit it without knowing anything
       about onboarding.

Plus one identity line pushed into the conversation so the model knows who
it is talking to:

    DEPRESSION_USER_NAME=Patil
    DEPRESSION_USER_EMAIL=patil@gmail.com
    DEPRESSION_USER_UID=…

The Firebase refresh token is deliberately NOT exported: it stays in the
0600 `user_tokens.json` and is only refreshed through
`FirebaseGoogleAuth.refresh_tokens`.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent.tui.onboarding.profile import ProfileStore, UserProfile
from agent.utils.env_manager import (
    GLOBAL_ENV_PATH,
    EnvManager,
    load_env_file,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# ENV NAMES
# ======================================================================

ENV_UID = "DEPRESSION_USER_UID"
ENV_EMAIL = "DEPRESSION_USER_EMAIL"
ENV_NAME = "DEPRESSION_USER_NAME"
ENV_PHOTO = "DEPRESSION_USER_PHOTO"
ENV_PROVIDER = "DEPRESSION_USER_PROVIDER"
ENV_VERIFIED = "DEPRESSION_USER_EMAIL_VERIFIED"
ENV_SIGNED_IN = "DEPRESSION_USER_SIGNED_IN"
ENV_LAST_LOGIN = "DEPRESSION_USER_LAST_LOGIN"

IDENTITY_ENVS = (
    ENV_UID, ENV_EMAIL, ENV_NAME, ENV_PHOTO,
    ENV_PROVIDER, ENV_VERIFIED, ENV_SIGNED_IN, ENV_LAST_LOGIN,
)

IDENTITY_CONTEXT_PREFIX = "Signed-in user identity:"


# ======================================================================
# CONVERSIONS
# ======================================================================

def identity_env(profile: UserProfile) -> Dict[str, str]:
    return {
        ENV_UID: profile.uid or "",
        ENV_EMAIL: profile.email or "",
        ENV_NAME: profile.name or "",
        ENV_PHOTO: profile.photo_url or "",
        ENV_PROVIDER: profile.provider or "",
        ENV_VERIFIED: "true" if profile.email_verified else "false",
        ENV_SIGNED_IN: "true" if profile.signed_in else "false",
        ENV_LAST_LOGIN: str(int(profile.last_login_at or 0)),
    }


def identity_context(profile: UserProfile) -> str:
    data = {
        "name": profile.display_name,
        "email": profile.email,
        "uid": profile.uid,
        "provider": profile.provider,
        "email_verified": profile.email_verified,
        "signed_in_via_gmail": profile.signed_in,
        "last_login": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(profile.last_login_at)
        ) if profile.last_login_at else "",
        "env_vars": list(IDENTITY_ENVS),
        "source": "gmail sign-in (Firebase Auth)",
    }
    return (
        f"{IDENTITY_CONTEXT_PREFIX} this install belongs to the user below. "
        f"Use it to personalise replies (greet them by name, address them at "
        f"their email). It is read-only — never invent values for these fields. "
        f"```json\n{json.dumps(data, indent=2)}\n```"
    )


# ======================================================================
# PERSIST / EXPORT
# ======================================================================

def _rewrite_env_file(env_path: Path, env_vars: Mapping[str, str]) -> None:
    """Write the env file wholesale, owner-only."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{k}={v}\n" for k, v in sorted(env_vars.items()))
    env_path.write_text(body, encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass


def export_identity(
    profile: UserProfile,
    *,
    persist: bool = True,
    env_path: Optional[Any] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Write the identity into the global env file and mirror it live.

    Routed through EnvManager so:
        * the write is atomic (temp file + rename, 0600)
        * the live process env is updated
        * subscribers (MCP client, tools, panels) are notified via reload()
    """
    values = identity_env(profile)

    if persist:
        target_path = Path(env_path) if env_path is not None else Path(GLOBAL_ENV_PATH)

        try:
            if target_path == Path(GLOBAL_ENV_PATH):
                # Preferred: go through EnvManager so subscribers get notified.
                env_vars = dict(load_env_file(target_path))
                for key, value in values.items():
                    if value:
                        env_vars[key] = value
                    else:
                        env_vars.pop(key, None)
                _rewrite_env_file(target_path, env_vars)
                EnvManager.get().reload()
            else:
                # Explicit alt path — write it directly, then mirror into
                # this process. Never touches EnvManager.
                env_vars = dict(load_env_file(target_path))
                for key, value in values.items():
                    if value:
                        env_vars[key] = value
                    else:
                        env_vars.pop(key, None)
                _rewrite_env_file(target_path, env_vars)
        except Exception as exc:
            logger.warning("Could not persist user identity to %s: %s",
                           target_path, exc)

    target_env = os.environ if environ is None else environ
    for key, value in values.items():
        if value:
            target_env[key] = value
        else:
            target_env.pop(key, None)

    logger.info(
        "Identity exported: name=%s email=%s",
        values[ENV_NAME] or "-",
        values[ENV_EMAIL] or "-",
    )
    return values


def clear_identity(
    *,
    persist: bool = True,
    env_path: Optional[Any] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    if persist:
        target_path = Path(env_path) if env_path is not None else Path(GLOBAL_ENV_PATH)
        try:
            env_vars = dict(load_env_file(target_path))
            for key in IDENTITY_ENVS:
                env_vars.pop(key, None)
            _rewrite_env_file(target_path, env_vars)
            if target_path == Path(GLOBAL_ENV_PATH):
                EnvManager.get().reload()
        except Exception as exc:
            logger.warning("Could not clear user identity from %s: %s",
                           target_path, exc)

    target_env = os.environ if environ is None else environ
    for key in IDENTITY_ENVS:
        target_env.pop(key, None)


def load_identity(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    env = os.environ if environ is None else environ
    return {key: env.get(key, "") for key in IDENTITY_ENVS}


def identity_is_present(environ: Optional[Mapping[str, str]] = None) -> bool:
    return bool(load_identity(environ).get(ENV_EMAIL))


def sign_out(
    store: Optional[Any] = None,
    *,
    env_path: Optional[Any] = None,
) -> None:
    profile_store = store if store is not None else ProfileStore()
    try:
        profile_store.sign_out()
    except Exception as exc:
        logger.warning("Could not remove the stored profile: %s", exc)
    clear_identity(env_path=env_path)


# ======================================================================
# APPLY TO A LIVE SESSION
# ======================================================================

async def apply_identity(
    profile: UserProfile,
    context_manager: Any = None,
    *,
    export: bool = True,
    env_path: Optional[Any] = None,
) -> Dict[str, str]:
    values = export_identity(profile, env_path=env_path) if export else {}

    if context_manager is not None:
        try:
            if not _context_already_present(context_manager):
                await context_manager.add_system_message(
                    identity_context(profile), pinned=True
                )
        except Exception as exc:
            logger.warning("Could not add identity context: %s", exc)

    return values


def _context_already_present(context_manager: Any) -> bool:
    try:
        messages = getattr(context_manager, "messages", None) or []
    except Exception:
        return False
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        if isinstance(content, str) and content.startswith(IDENTITY_CONTEXT_PREFIX):
            return True
    return False


__all__ = [
    "ENV_EMAIL", "ENV_LAST_LOGIN", "ENV_NAME", "ENV_PHOTO",
    "ENV_PROVIDER", "ENV_SIGNED_IN", "ENV_UID", "ENV_VERIFIED",
    "IDENTITY_CONTEXT_PREFIX", "IDENTITY_ENVS",
    "apply_identity", "clear_identity", "export_identity",
    "identity_context", "identity_env", "identity_is_present",
    "load_identity", "sign_out",
]