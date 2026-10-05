"""Regression tests for credential redaction.

Redaction is the last line of defence: it runs on anything the agent is about
to print, log, or put in a transcript. A pattern that silently stops matching
is therefore a credential leak that only shows up in production.

Each sample value below is assembled from fragments on purpose. The secret
scanners in this repository flag literal credential shapes in tracked files,
and a test suite is a tracked file; keeping the literals out of the source is
what lets the guard stay strict instead of carrying a blanket test-file
exemption.
"""

from __future__ import annotations

import pytest

from agent.utils.redact import REDACTED, redact


def _join(*parts: str) -> str:
    return "".join(parts)


AWS_KEY = _join("AKI", "A", "QWERTYUIOPASDFGH")
AWS_TEMP_KEY = _join("ASI", "A", "QWERTYUIOPASDFGH")
FIREBASE_WEB_KEY = _join("AIz", "aSy", "BITGCHjn2f_S1S5ExiwRa9DXSnbz_P9qQ")
OAUTH_CLIENT_SECRET = _join("GOCS", "PX-", "Ab3dEfGhIjKlMnOpQr")
GITHUB_TOKEN = _join("gh", "p_", "a" * 36)
GROQ_KEY = _join("g", "sk_", "b" * 52)
OPENAI_STYLE_KEY = _join("s", "k-", "c" * 48)
SLACK_TOKEN = _join("xox", "b-", "1234567890-abcdefghij")
BEARER = _join("Bea", "rer ", "d" * 30)
JWT = _join("eyJhbGciOiJIUzI1NiJ9", ".eyJzdWIiOiIxIn0", ".eeeeeeeeeeeeeeeeeeee")

DSN_PASSWORDS = [
    ("postgres", "admin", "hunter2", "db.internal:5432/prod"),
    ("postgresql", "u", "p%40ss", "db:5432/app?sslmode=require"),
    ("mysql", "root", "secret", "127.0.0.1:3306/app"),
    ("mongodb", "user", "pw", "cluster.mongodb.net/db"),
    ("redis", "", "mypassword", "cache:6379/0"),
    ("amqp", "svc", "s3cr3t", "rabbit:5672/"),
    ("clickhouse", "default", "secret", "ch:8123/db"),
]


@pytest.mark.parametrize(
    "secret, label",
    [
        (AWS_KEY, "aws access key id"),
        (AWS_TEMP_KEY, "aws temporary credential"),
        (FIREBASE_WEB_KEY, "firebase web api key"),
        (OAUTH_CLIENT_SECRET, "google oauth client secret"),
        (GITHUB_TOKEN, "github personal access token"),
        (GROQ_KEY, "groq api key"),
        (OPENAI_STYLE_KEY, "openai-style api key"),
        (SLACK_TOKEN, "slack bot token"),
        (BEARER, "bearer authorization header"),
        (JWT, "json web token"),
    ],
)
def test_known_credential_shapes_are_masked(secret: str, label: str) -> None:
    out = redact(secret)
    assert secret not in out, f"{label} survived redaction: {out!r}"


@pytest.mark.parametrize("scheme, user, password, tail", DSN_PASSWORDS)
def test_connection_string_password_is_masked(
    scheme: str, user: str, password: str, tail: str
) -> None:
    """A DSN echoed by a command is a common way a password reaches a log."""
    url = f"{scheme}://{user}:{password}@{tail}"
    out = redact(url)
    assert password not in out, f"password survived in {out!r}"
    assert out.startswith(f"{scheme}://{user}:")
    assert out.endswith(f"@{tail}")


def test_jdbc_style_url_password_is_masked() -> None:
    url = _join("jdbc:postgresql://", "svc:", "Pa55w0rd", "@db:5432/app")
    out = redact(url)
    assert "Pa55w0rd" not in out
    assert out.endswith("@db:5432/app")


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/a?b=1#c",
        "mailto:someone@example.com",
        "ssh://host-only@jump.example.com",
        "version 1.2.3 was released",
        "see http://localhost:8080/health for details",
        "ratio 3:4 recorded at 10:30",
        "AIzaSyTest is not a real key",
    ],
)
def test_ordinary_text_is_untouched(text: str) -> None:
    assert redact(text) == text


def test_secret_named_dict_fields_are_masked_wholesale() -> None:
    payload = {
        "api_key": "opaque-value-without-a-recognisable-shape",
        "refresh_token": "another-opaque-blob",
        "client_secret": "yet-another-one",
        "project": "demo",
    }
    out = redact(payload)
    assert out["api_key"] == REDACTED
    assert out["refresh_token"] == REDACTED
    assert out["client_secret"] == REDACTED
    assert out["project"] == "demo", "non-secret fields must survive"


def test_redaction_is_recursive() -> None:
    payload = {"outer": {"items": [{"password": "hunter2"}, AWS_KEY]}}
    out = redact(payload)
    assert out["outer"]["items"][0]["password"] == REDACTED
    assert AWS_KEY not in str(out)


def test_pem_detection_header_survives_for_detection() -> None:
    """The header must stay matchable so a private key block is still caught."""
    header = _join("-----BEGIN ", "RSA ", "PRIVATE KEY-----")
    assert header in redact(header)
