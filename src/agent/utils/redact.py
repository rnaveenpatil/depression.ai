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
    "AWS_PROFILE",
    "DEPRESSION_API_KEY",
    "DEPRESSION_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "NVIDIA_API_KEY",
    "XAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "GH_TOKEN",
    # Firebase / Google — these were previously unredacted, so Firebase
    # keys and OAuth tokens leaked into the DB and the console.
    "FIREBASE_API_KEY",
    "FIREBASE_WEB_APP_CONFIG",
    "FIREBASE_OAUTH_CLIENT_ID",
    "FIREBASE_ID_TOKEN",
    "FIREBASE_ACCESS_TOKEN",
    "FIREBASE_REFRESH_TOKEN",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GCLOUD_ACCESS_KEY",
    # Every provider EnvManager persists (utils/env_manager.py).
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
    "MISTRAL_API_KEY",
    "MOONSHOT_API_KEY",
    "ZAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "META_API_KEY",
    "ALIBABA_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HF_TOKEN",
    "FIRECRAWL_API_KEY",
    "SERP_API_KEY",
    "BRAVE_API_KEY",
    "TAVILY_API_KEY",
    "EXA_API_KEY",
    # OAuth / session material
    "CLIENT_SECRET",
    "CLIENT_ID",
    "REFRESH_TOKEN",
    "ACCESS_TOKEN",
    "ID_TOKEN",
    "GITHUB_CLIENT_SECRET",
    "GOOGLE_CLIENT_SECRET",
)

# Secret shapes that must be masked even when the value is not present in the
# environment (i.e. when it was read from a file, a pasted export, or a tool
# result rather than from os.environ).
_PATTERNS = [
    # AWS access key IDs
    re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA|ANVA|ABIA|ACCA)[A-Z0-9]{16}\b"),
    # Bearer / Basic authorization headers
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"\bBasic\s+[A-Za-z0-9+/]{20,}={0,2}"),
    # Vendor-prefixed API keys: sk-, sk-ant-, gsk_, xai-, nvapi-, hf_
    re.compile(r"\b(?:sk|gsk|xai|nvapi|hf|r8|pk)[-_][A-Za-z0-9\-_]{16,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9\-_]{16,}\b"),
    # GitHub tokens: classic and fine-grained
    re.compile(r"\bghp_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgh[osur]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    # Google / Firebase API keys
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # Google OAuth *client secret*. Distinct from the client id, which is a
    # public identifier and is allowed through.
    re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{10,}\b"),
    # Slack
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bhttps://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}"),
    # Firebase/Google OAuth client id (identifies the project; not secret, but
    # pairs with the leaked key above and should not be echoed in logs)
    re.compile(r"\b\d{12}-[0-9a-z]{20,}\.apps\.googleusercontent\.com\b"),
    # Stripe / OpenAI project keys
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    # npm / PyPI / Docker registry tokens
    re.compile(r"\bnpm_[A-Za-z0-9]{30,}"),
    re.compile(r"\bglpat-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"\bglcbt-[A-Za-z0-9\-_]{16,}"),
    re.compile(r"\bdd_[a-z]{2}[A-Za-z0-9_-]{20,}"),
    # Private key blocks (PEM, regardless of header wording)
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"),
    re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
    # JSON Web Tokens — these are bearer credentials
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
]

# JSON fields whose *value* is a credential. Masking by key catches secrets
# whose format no regex anticipates (session tokens, arbitrary OAuth blobs).
# The keyword may sit at either a word boundary (``client_secret``) or a
# camelCase transition (``refreshToken``), and a trailing measurement word is
# excluded so counters such as ``max_tokens`` or ``secret_count`` stay visible.
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[^A-Za-z0-9]|[a-z])"
    r"(?:"
    r"api[_-]?keys?|secrets?|passwords?|passwd|tokens?|credentials?|"
    r"private[_-]?keys?|access[_-]?keys?|client[_-]?secrets?|"
    r"session[_-]?keys?|authorization|id[_-]?tokens?|refresh[_-]?tokens?|"
    r"bearer"
    r")"
    r"(?![a-z])",
    re.I,
)

# Keys that merely *count* or *describe* a credential-adjacent quantity. These
# must stay readable (``max_tokens``, ``usage.total_tokens``) or the agent
# loses visibility into its own limits and budgets.
_METRIC_KEY_RE = re.compile(
    r"(?:^|_)(?:total|max|min|prompt|completion|input|output|cache|reasoning|"
    r"estimated|remaining|available|used|new)_tokens?$"
    r"|(?:^|_)tokens?_(?:count|used|usage|left)$"
    r"|(?:^|_)(?:count|len|size|limit|limits|threshold|budget|expiry|expires)$",
    re.I,
)


def _key_is_secret(key: str) -> bool:
    """True when a mapping key names a credential rather than a metric."""
    if _METRIC_KEY_RE.search(key):
        return False
    return bool(_SENSITIVE_KEY_RE.search(key))

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
        out = {}
        for k, v in value.items():
            # A credential-named field is masked wholesale: the value may be
            # an opaque session token that no shape-based pattern recognises.
            if isinstance(k, str) and _key_is_secret(k) and v not in (None, "", {}, []):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v) for v in value)
    return value


def _redact_str(text: str) -> str:
    for secret in _known_secret_values():
        if secret in text:
            text = text.replace(secret, REDACTED)
    # key="value" / key: "value" pairs inside serialized JSON or dotted text
    text = re.sub(
        r'(?i)("?)([A-Za-z0-9_.-]*'
        r'(?:api[_-]?key|secret|password|passwd|token|credential|authorization)'
        r'[A-Za-z0-9_.-]*)("?\s*[:=]\s*"?)([^\s",}\']{8,})',
        lambda m: (
            f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}"
            if _key_is_secret(m.group(2))
            else m.group(0)
        ),
        text,
    )
    for pat in _PATTERNS:
        text = pat.sub(REDACTED, text)
    return _mask_url_passwords(text)


# A database URL echoed by a command, a stack trace, or a tool result is one of
# the most common ways a live credential reaches a transcript. The password is
# masked wherever the URL appears, not only when the field name looks
# sensitive. Scheme, user and host survive so the line stays useful.
_URL_PASSWORD_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9+.\-]{1,31})://([^\s:/@]*):([^\s@/]+)@"
)


def _mask_url_passwords(text: str) -> str:
    return _URL_PASSWORD_RE.sub(
        lambda m: f"{m.group(1)}://{m.group(2)}:{REDACTED}@", text
    )


__all__ = ["redact", "REDACTED"]