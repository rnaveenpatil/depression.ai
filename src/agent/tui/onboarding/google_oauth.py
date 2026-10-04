"""
Gmail sign-in through Google's official *installed app* flow.

    pip install google-auth-oauthlib requests

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(
        "client_secrets.json",
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
    )
    credentials = flow.run_local_server(port=0)   # blocks until consent
    google_id_token = credentials.id_token

Google's `run_local_server()` brings its own loopback server, so nothing has to
be copied out of a browser window: the user clicks "Sign in with Google", the
browser comes up, consents, and the CLI is handed a Google ID token.

That Google token is then exchanged for a Firebase session
(`identitytoolkit/v1/accounts:signInWithIdp`), which is the same REST endpoint
the compat-SDK page uses — so both paths end at the same place, and the profile
that gets stored and exported is identical.

Two ways to supply the OAuth client
-----------------------------------
1. A `client_secrets.json` file (preferred): Google Cloud Console > APIs &
   Services > Credentials > your **Desktop app** client > Download JSON. Point at
   it with `FIREBASE_GOOGLE_CLIENT_SECRETS`, or drop it at
   `~/.config/depression/client_secrets.json` or `./client_secrets.json`.
   Desktop clients need no secret, so this just works.

2. A bare client id from `firebase.json`:

       {
         "project_id": "my-project",
         "api_key": [{ "current_key": "AIza…" }],
         "oauth_client": [
           { "client_id": "…:android:…", "client_type": 1 },
           { "client_id": "….apps.googleusercontent.com", "client_type": 3 }
         ]
       }

   The `client_type: 3` entry (the "Web client" Firebase creates for Auth) is
   used automatically; Android/iOS entries are skipped. Web clients normally
   want a client secret at the token endpoint — set
   `FIREBASE_OAUTH_CLIENT_SECRET` (or `oauth_client_secret` in the config) if
   Google answers `invalid_client`.

Nothing secret is stored by the CLI. Both `google_auth_oauthlib` and `requests`
are imported lazily: when they are missing the app falls back to the
dependency-free browser page in `welcome.py` instead of breaking.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from agent.utils.errors import AgentError
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

GOOGLE_SCOPES: tuple = (
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)

SIGN_IN_WITH_IDP_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp"

GOOGLE_PROVIDER = "google.com"
REQUEST_URI = "http://localhost"

ENV_CLIENT_SECRETS = "FIREBASE_GOOGLE_CLIENT_SECRETS"
ENV_CLIENT_SECRETS_ALT = "GOOGLE_CLIENT_SECRETS"
ENV_CLIENT_ID = "FIREBASE_OAUTH_CLIENT_ID"
ENV_CLIENT_SECRET = "FIREBASE_OAUTH_CLIENT_SECRET"

CLIENT_SECRETS_FILENAME = "client_secrets.json"

# Google's endpoints for an installed/public OAuth client.
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"


class GoogleOAuthError(AgentError):
    """The installed-app flow could not complete."""


# ======================================================================
# LAZY OPTIONAL IMPORTS
# ======================================================================

def _import_installed_app_flow() -> Any:
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GoogleOAuthError(
            "google-auth-oauthlib is not installed. "
            "Run: pip install google-auth-oauthlib requests"
        ) from exc
    return InstalledAppFlow


def _import_requests() -> Any:
    try:
        import requests
    except ImportError as exc:
        raise GoogleOAuthError(
            "requests is not installed. Run: pip install google-auth-oauthlib requests"
        ) from exc
    return requests


def oauthlib_available() -> bool:
    """True when both extra packages can be imported."""
    try:
        _import_installed_app_flow()
        _import_requests()
    except GoogleOAuthError:
        return False
    return True


# ======================================================================
# CLIENT SECRETS DISCOVERY
# ======================================================================

def find_client_secrets(
    path: Optional[Any] = None,
    *,
    environ: Optional[Dict[str, str]] = None,
) -> Optional[Path]:
    """
    Locate `client_secrets.json`.

    Order: explicit argument, `$FIREBASE_GOOGLE_CLIENT_SECRETS`,
    `$GOOGLE_CLIENT_SECRETS`, `~/.config/depression/`, then the cwd.
    """
    env = os.environ if environ is None else environ

    if path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_file() else None

    for name in (ENV_CLIENT_SECRETS, ENV_CLIENT_SECRETS_ALT):
        raw = str(env.get(name) or "").strip()
        if raw:
            candidate = Path(raw).expanduser()
            if candidate.is_file():
                return candidate
            logger.warning("%s points at a missing file: %s", name, candidate)
            return None

    for candidate in (
        Path.home() / ".config" / "depression" / CLIENT_SECRETS_FILENAME,
        Path.cwd() / CLIENT_SECRETS_FILENAME,
    ):
        if candidate.is_file():
            return candidate

    return None


# ======================================================================
# THE FLOW
# ======================================================================

class GoogleOAuthSignIn:
    """
    Blocking Google sign-in via `InstalledAppFlow`, Firebase exchange via REST.

    Injectable for tests:
      * `client_secrets` — path to the OAuth client file
      * `run_flow`      — replacement for `InstalledAppFlow.run_local_server`
      * `post`          — replacement for `requests.post`
    """

    def __init__(
        self,
        api_key: str,
        *,
        client_secrets: Optional[Any] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Sequence[str] = GOOGLE_SCOPES,
        timeout: float = 180.0,
        run_flow: Optional[Callable[..., Any]] = None,
        post: Optional[Callable[..., Any]] = None,
        port: int = 0,
        environ: Optional[Dict[str, str]] = None,
    ):
        env = os.environ if environ is None else environ
        self.api_key = (api_key or "").strip()
        self.scopes = tuple(scopes)
        self.timeout = float(timeout)
        self.port = int(port)
        self.client_secrets = find_client_secrets(client_secrets, environ=env)
        # A `client_secrets.json` always wins; otherwise build the config from
        # the project's OAuth client id (no file needed).
        self.client_id = "" if self.client_secrets else (
            (client_id or "").strip() or str(env.get(ENV_CLIENT_ID) or "").strip()
        )
        self.client_secret = (client_secret or "").strip() or str(
            env.get(ENV_CLIENT_SECRET) or ""
        ).strip()
        self._run_flow = run_flow
        self._post = post

    # -- capability ----------------------------------------------------

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) and self.has_client and oauthlib_available()

    @property
    def has_client(self) -> bool:
        """Either a client_secrets.json or a bare client id is enough."""
        return self.client_secrets is not None or bool(self.client_id)

    def client_config(self) -> Dict[str, Any]:
        """The dict shape `InstalledAppFlow.from_client_config` expects."""
        return {
            "installed": {
                "client_id": self.client_id,
                **({"client_secret": self.client_secret} if self.client_secret else {}),
                "auth_uri": AUTH_URI,
                "token_uri": TOKEN_URI,
                "redirect_uris": [REQUEST_URI],
            }
        }

    def describe(self) -> str:
        if not self.api_key:
            return "FIREBASE_API_KEY is not set"
        if not self.has_client:
            return (
                f"{CLIENT_SECRETS_FILENAME} not found and no OAuth client id "
                f"(set ${ENV_CLIENT_SECRETS} or ${ENV_CLIENT_ID})"
            )
        if not oauthlib_available():
            return "pip install google-auth-oauthlib requests"
        if self.client_secrets is not None:
            return f"Google OAuth via {self.client_secrets}"
        return f"Google OAuth via client {self.client_id[:24]}…"

    # -- steps ---------------------------------------------------------

    def google_id_token(self) -> str:
        """
        Run the installed-app flow and return Google's ID token.

        `run_local_server(port=0)` picks a free port, opens the browser and
        blocks until the consent screen is done.
        """
        if not self.has_client:
            raise GoogleOAuthError(
                f"No OAuth client. Either download a Desktop client's "
                f"{CLIENT_SECRETS_FILENAME} and set "
                f"${ENV_CLIENT_SECRETS}=/path/to/{CLIENT_SECRETS_FILENAME}, or put "
                f"the web client id in firebase.json / ${ENV_CLIENT_ID}."
            )

        if self._run_flow is not None:
            credentials = self._run_flow(self.client_secrets, self.scopes, self.port)
        else:
            installed_app_flow = _import_installed_app_flow()
            if self.client_secrets is not None:
                flow = installed_app_flow.from_client_secrets_file(
                    str(self.client_secrets),
                    scopes=list(self.scopes),
                )
            else:
                flow = installed_app_flow.from_client_config(
                    self.client_config(),
                    scopes=list(self.scopes),
                )
            logger.info("Opening browser for Google OAuth: %s", self._client_label())
            credentials = flow.run_local_server(port=self.port)

        id_token = str(getattr(credentials, "id_token", "") or "")
        if not id_token:
            raise GoogleOAuthError("Google returned credentials without an ID token.")
        return id_token

    def exchange_for_firebase(self, google_id_token: str) -> Dict[str, Any]:
        """Trade the Google ID token for a Firebase session (blocking)."""
        post = self._post if self._post is not None else _import_requests().post
        url = f"{SIGN_IN_WITH_IDP_URL}?key={self.api_key}"
        payload = {
            "postBody": f"id_token={google_id_token}&providerId={GOOGLE_PROVIDER}",
            "requestUri": REQUEST_URI,
            "returnSecureToken": True,
        }

        logger.info("Exchanging the Google ID token for a Firebase session")
        try:
            response = post(url, json=payload, timeout=self.timeout)
        except GoogleOAuthError:
            raise
        except Exception as exc:
            raise GoogleOAuthError(f"Could not reach Firebase: {exc}") from exc

        status = int(getattr(response, "status_code", 0) or 0)
        if not 200 <= status < 300:
            raise GoogleOAuthError(
                f"Firebase exchange failed ({status}): {_response_text(response)[:300]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise GoogleOAuthError("Firebase returned an unreadable response") from exc

        if not isinstance(data, dict) or not data.get("idToken"):
            raise GoogleOAuthError("Firebase returned no session token.")
        return data

    def _client_label(self) -> str:
        if self.client_secrets is not None:
            return str(self.client_secrets)
        return f"client_id={self.client_id}"

    def sign_in(self) -> Dict[str, Any]:
        """Full blocking flow: Google consent -> Firebase session."""
        if not self.api_key:
            raise GoogleOAuthError("FIREBASE_API_KEY is not set.")
        return self.exchange_for_firebase(self.google_id_token())

    async def sign_in_async(self) -> Dict[str, Any]:
        """`sign_in()` off the event loop — the flow blocks on the browser."""
        return await asyncio.to_thread(self.sign_in)


def _response_text(response: Any) -> str:
    text = str(getattr(response, "text", "") or "")
    return text or "<empty response>"


__all__ = [
    "AUTH_URI",
    "CLIENT_SECRETS_FILENAME",
    "ENV_CLIENT_ID",
    "ENV_CLIENT_SECRET",
    "ENV_CLIENT_SECRETS",
    "ENV_CLIENT_SECRETS_ALT",
    "TOKEN_URI",
    "GOOGLE_SCOPES",
    "REQUEST_URI",
    "SIGN_IN_WITH_IDP_URL",
    "GoogleOAuthError",
    "GoogleOAuthSignIn",
    "find_client_secrets",
    "oauthlib_available",
]