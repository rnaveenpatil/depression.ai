"""
Secret redaction for tool output.

Any credential that appears in tool output (env dumps, cat .env, etc.) is
masked BEFORE it enters the context, the session, or the database.

This runs on the write path, so anything already persisted is untouched.
"""

from __future__ import annotations

import os
import re
from typing import Any

_SENSITIVE_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECRET_KEY",
    "DEPRESSION_API_KEY",
    "DEPRESSION_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "XAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
)

_PATTERNS = [
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"\b(?:sk|gsk|xai)-[A-Za-z0-9\-_]{20,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9\-_]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
]

REDACTED = "***REDACTED***"


def _known_secret_values() -> list[str]:
    out: list[str] = []
    for name in _SENSITIVE_ENV_VARS:
        val = os.environ.get(name)
        if val and len(val) >= 8:
            out.append(val)
    return out


def redact(value: Any) -> Any:
    """Return a copy of value with known secrets masked."""
    if isinstance(value, str):
        return _redact_str(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    return value


def _redact_str(text: str) -> str:
    for secret in _known_secret_values():
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pat in _PATTERNS:
        text = pat.sub(REDACTED, text)
    return text


__all__ = ["redact", "REDACTED"]