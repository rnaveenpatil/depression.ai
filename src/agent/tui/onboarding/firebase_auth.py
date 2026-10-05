"""
Gmail (Google) sign-in for the terminal, on top of Firebase Auth.

How it works
------------
1. The CLI opens a short-lived HTTP server bound to `localhost` on a random
   port and serves one page: a Firebase compat-SDK bootstrap using the
   project's *web* config (apiKey / authDomain / projectId / appId).
   It must be the *name* `localhost`, not `127.0.0.1`: Firebase matches the
   page origin against the project's authorized-domain list, which carries
   `localhost` and does not alias `127.0.0.1` to it.
2. The user's browser completes the normal Google consent screen, so Google
   sees an OAuth request from the authorized domain, never from the CLI.
3. The page posts the resulting Firebase `idToken` (+ `refreshToken`) back to
   the loopback URL, which the CLI consumes and the server shuts down.
4. The CLI verifies the token against Firebase's REST API
   (`identitytoolkit/v1/accounts:lookup`) and uses the authoritative
   `localId` / `email` / `displayName` / `photoUrl` it returns.

No client secret and no service-account key is needed or stored anywhere.
Everything here is plain `httpx` + stdlib so it works in a test harness and
in a headless box (where `sign_in_with_id_token` accepts a pasted token).
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import httpx

from agent.tui.onboarding.firebase_config import (
    FIREBASE_SDK_BASE,
    FirebaseConfig,
    load_firebase_config,
)
from agent.utils.errors import AgentError
from agent.utils.logging import get_logger
from agent.utils.redact import redact

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

IDENTITY_BASE = "https://identitytoolkit.googleapis.com/v1"
SECURE_TOKEN_BASE = "https://securetoken.googleapis.com/v1"

DEFAULT_TIMEOUT = 180.0
CALLBACK_PATH = "/callback"

GOOGLE_PROVIDER = "google.com"

_ERROR_HINTS = {
    "TOKEN_EXPIRED": "That token expired — sign in again.",
    "INVALID_ID_TOKEN": "Google did not return a valid Firebase token.",
    "INVALID_LOGIN_CREDENTIALS": "Google rejected the sign-in.",
    "INVALID_REFRESH_TOKEN": "The saved session expired — sign in again.",
    "USER_DISABLED": "This Firebase user is disabled.",
    "OPERATION_NOT_ALLOWED": (
        "Enable the Google provider in Firebase Console → Authentication → "
        "Sign-in method."
    ),
    "API_KEY_INVALID": "The Firebase API key is not valid for this project.",
    "CONFIGURATION_NOT_FOUND": "Firebase Auth is not initialised for this project.",
}


class OnboardingError(AgentError):
    """Raised when sign-in cannot be completed."""


# ======================================================================
# USER
# ======================================================================

@dataclass
class FirebaseUser:
    """The identity Firebase reports for a signed-in user."""

    uid: str
    email: str = ""
    name: str = ""
    photo_url: str = ""
    email_verified: bool = False
    provider: str = GOOGLE_PROVIDER
    id_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> Dict[str, Any]:
        """Payload for `ProfileStore.save_tokens` (0600 file)."""
        return {
            "uid": self.uid,
            "id_token": self.id_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "provider": self.provider,
        }

    @classmethod
    def from_lookup(
        cls,
        user: Dict[str, Any],
        *,
        id_token: str = "",
        refresh_token: str = "",
        expires_at: float = 0.0,
    ) -> "FirebaseUser":
        return cls(
            uid=str(user.get("localId", "") or ""),
            email=str(user.get("email", "") or ""),
            name=str(user.get("displayName", "") or ""),
            photo_url=str(user.get("photoUrl", "") or ""),
            email_verified=bool(user.get("emailVerified", False)),
            provider=GOOGLE_PROVIDER,
            id_token=id_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            raw=dict(user),
        )


# ======================================================================
# BROWSER PAGE
# ======================================================================

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{app_title} — sign in</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center;
    justify-content: center; background: #000; color: #aaffcc;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  .card {{
    width: min(440px, 92vw); border: 2px solid #00ff66; background: #031008;
    padding: 28px 24px; text-align: center; border-radius: 8px;
  }}
  h1 {{ font-size: 16px; letter-spacing: 3px; margin: 0 0 6px; color: #88ffbb; }}
  p {{ font-size: 13px; color: #3d8c5c; margin: 0 0 18px; }}
  button {{
    width: 100%; padding: 12px 16px; font: inherit; font-weight: 700;
    letter-spacing: 1px; cursor: pointer; color: #000; background: #00ff66;
    border: none; border-radius: 6px;
  }}
  button:hover {{ background: #88ffbb; }}
  button[disabled] {{ opacity: .5; cursor: progress; }}
  #msg {{ margin-top: 16px; font-size: 12px; min-height: 18px; color: #ffcc44;
          word-break: break-word; }}
  .ok {{ color: #00ff66; }}
  .err {{ color: #ff4466; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{app_title_upper}</h1>
    <p>Google account &mdash; you can close this tab once you are back in the terminal.</p>
    <button id="go" onclick="signIn()">SIGN IN WITH GOOGLE</button>
    <div id="msg">starting&hellip;</div>
  </div>
<script src="{sdk_base}/firebase-app-compat.js"></script>
<script src="{sdk_base}/firebase-auth-compat.js"></script>
<script>
  const CONFIG   = {config};
  const STATE    = {state};
  const CALLBACK = {callback};
  const msg = document.getElementById("msg");
  const btn = document.getElementById("go");

  function say(text, cls) {{ msg.textContent = text; msg.className = cls || ""; }}

  // Fail fast if the CDN is blocked — otherwise nothing on this page works
  // and the CLI just waits out its timeout.
  if (typeof firebase === "undefined" || !firebase.initializeApp) {{
    say("Firebase SDK failed to load — check your network / proxy.", "err");
    fetch(CALLBACK, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ state: STATE, error: "firebase-sdk-load-failed" }}),
    }}).catch(function () {{}});
  }} else {{
    main();
  }}

  function main() {{
    firebase.initializeApp(CONFIG);
    const auth = firebase.auth();
    auth.setPersistence(firebase.auth.Auth.Persistence.LOCAL).catch(function () {{}});

    if (!CONFIG.authDomain) {{
      say("authDomain missing — set FIREBASE_AUTH_DOMAIN.", "err");
    }}

    async function post(payload) {{
      payload.state = STATE;
      try {{
        const res = await fetch(CALLBACK, {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        return res.ok;
      }} catch (err) {{
        say("could not reach the CLI: " + err, "err");
        return false;
      }}
    }}

    async function deliver(result) {{
      const user = result && result.user;
      if (!user) {{
        const reason = (result && result.error && (result.error.message ||
          result.error.code)) || "no user returned from Google";
        await post({{ error: "auth/no-user: " + reason }});
        say("sign-in failed: " + reason, "err");
        btn.disabled = false;
        return;
      }}
      say("verifying\\u2026", "ok");
      try {{
        const idToken = await user.getIdToken();
        const refreshToken =
          user.refreshToken ||
          (user.stsTokenManager && user.stsTokenManager.refreshToken) || "";
        const ok = await post({{
          idToken: idToken,
          refreshToken: refreshToken,
          email: user.email || "",
          displayName: user.displayName || "",
          photoURL: user.photoURL || "",
          providerId: (result.credential && result.credential.providerId) || "google.com",
        }});
        if (ok) {{
          say("signed in — return to your terminal", "ok");
          btn.disabled = true;
        }}
      }} catch (err) {{
        await post({{ error: "token-delivery-failed: " + err }});
        say("could not deliver token: " + err, "err");
        btn.disabled = false;
      }}
    }}

    async function fail(err, report) {{
      const code = (err && err.code) || "auth/unknown";
      const message = code + ": " + ((err && err.message) || "sign-in failed");
      if (report !== false) {{
        try {{ await post({{ error: message }}); }} catch (e) {{}}
      }}
      say(message, "err");
      btn.disabled = false;
    }}

    async function signIn() {{
      btn.disabled = true;
      say("waiting for Google\\u2026");
      const provider = new firebase.auth.GoogleAuthProvider();
      provider.setCustomParameters({{ prompt: "select_account" }});
      try {{
        await deliver(await auth.signInWithPopup(provider));
      }} catch (err) {{
        const code = (err && err.code) || "";
        // Popups blocked or unsupported → full-page redirect.
        if (code === "auth/popup-blocked" ||
            code === "auth/popup-closed-by-user" ||
            code === "auth/operation-not-supported-in-this-environment" ||
            code === "auth/web-storage-unsupported") {{
          say("redirecting to Google\\u2026");
          try {{
            await auth.signInWithRedirect(provider);
          }} catch (redirectErr) {{
            await fail(redirectErr);
          }}
        }} else {{
          await fail(err);
        }}
      }}
    }}

    // Resume after a redirect. Only auto-start the popup when we did NOT
    // just come back from one — otherwise the page fires a popup before the
    // user has clicked anything.
    (async function resume() {{
      if (typeof auth.getRedirectResult !== "function") {{
        say("ready", "ok");
        return;
      }}
      try {{
        const result = await auth.getRedirectResult();
        if (result && result.user) {{
          btn.disabled = true;
          await deliver(result);
        }} else if (result && result.error) {{
          await fail(result.error);
        }} else {{
          say("ready", "ok");
        }}
      }} catch (err) {{
        await fail(err);
      }}
    }})();

    // Expose for the button's onclick.
    window.signIn = signIn;
  }}
</script>
</body>
</html>
"""

_OK_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{{background:#000;color:#00ff66;font-family:ui-monospace,Menlo,monospace;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;
font-size:15px;letter-spacing:1px}}</style></head>
<body>Signed in. You can close this tab and return to your terminal.</body></html>"""

_ERROR_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{{background:#000;color:#ff4466;font-family:ui-monospace,Menlo,monospace;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0;
font-size:15px;letter-spacing:1px;text-align:center;padding:24px}}</style></head>
<body>{message}</body></html>"""


def render_sign_in_page(
    config: FirebaseConfig,
    callback_url: str,
    state: str,
    app_title: str = "depression.ai",
) -> str:
    """The HTML page served to the browser (exposed for tests)."""
    return _PAGE_TEMPLATE.format(
        app_title=app_title,
        app_title_upper=app_title.upper(),
        sdk_base=FIREBASE_SDK_BASE,
        config=json.dumps(config.to_web_config()),
        state=json.dumps(state),
        callback=json.dumps(callback_url),
    )


# ======================================================================
# LOOPBACK CALLBACK SERVER
# ======================================================================

class _CallbackState:
    """Shared state between the server thread and the awaiting coroutine."""

    def __init__(self, state_token: str, html: str):
        self.state_token = state_token
        self.html = html
        self.done = threading.Event()
        self.payload: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.request_seen = threading.Event()


class _CallbackServer:
    """
    One-shot loopback HTTP server: serves the sign-in page, then captures the
    token the browser posts back. Bound to `localhost` on an ephemeral port.

    `localhost` rather than `127.0.0.1` on purpose: Firebase validates the
    page origin against the project's authorized domains, and that list names
    `localhost` without aliasing the literal IP. Resolving `localhost` still
    yields a loopback-only socket.
    """

    def __init__(self, html: str, state_token: str, host: str = "localhost", port: int = 0):
        self.state = _CallbackState(state_token, html)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._host = host
        self._port = port

    # -- lifecycle ----------------------------------------------------

    def start(self) -> str:
        handler = _make_handler(self.state)
        try:
            self._server = ThreadingHTTPServer((self._host, self._port), handler)
        except OSError as exc:
            raise OnboardingError(f"Could not open a local sign-in port: {exc}") from exc
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="firebase-signin",
            daemon=True,
        )
        self._thread.start()
        logger.info("Firebase sign-in page served at %s", self.url)
        return self.url

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else 0

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}/"

    @property
    def callback_url(self) -> str:
        return f"http://{self._host}:{self.port}{CALLBACK_PATH}"

    def wait_for_payload(self, timeout: float) -> Dict[str, Any]:
        if not self.state.done.wait(timeout):
            raise OnboardingError(
                f"Timed out after {int(timeout)}s waiting for the browser sign-in."
            )
        if self.state.error:
            raise OnboardingError(self.state.error)
        return dict(self.state.payload)

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()
        if thread is not None:
            thread.join(timeout=3.0)
        if server is not None:
            try:
                server.server_close()
            except Exception:
                pass

    def __enter__(self) -> "_CallbackServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _make_handler(state: _CallbackState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "depression-signin/1.0"
        protocol_version = "HTTP/1.1"

        # -- helpers -------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            # The request line can carry idToken/refreshToken as query
            # params, so it must never reach the log verbatim.
            logger.debug("signin-http %s", redact(fmt % args))

        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _valid_state(self, candidate: Any) -> bool:
            return bool(candidate) and secrets.compare_digest(str(candidate), state.state_token)

        def _finish(self, payload: Dict[str, Any]) -> None:
            if not self._valid_state(payload.get("state")):
                state.error = "Sign-in rejected: state mismatch."
            else:
                state.payload = payload
            state.request_seen.set()
            state.done.set()

        def _stop_soon(self) -> None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        # -- routes --------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"

            if route == "/":
                self._send(200, state.html)
                return
            if route == "/health":
                self._send(200, "ok", "text/plain; charset=utf-8")
                return
            if route == CALLBACK_PATH:
                query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                self._finish(query)
                message = query.get("error") or "Sign-in failed in the browser."
                self._send(200, _ERROR_PAGE.format(message=_escape(message)))
                self._stop_soon()
                return
            if route == "/favicon.ico":
                self._send(204, "")
                return
            self._send(404, "not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.rstrip("/") != CALLBACK_PATH:
                self._send(404, "not found", "text/plain; charset=utf-8")
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"

            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            self._finish(payload)

            if payload.get("error") and not payload.get("idToken"):
                self._send(200, _ERROR_PAGE.format(message=_escape(str(payload["error"])[:200])))
            else:
                self._send(200, _OK_PAGE)
            self._stop_soon()

    return Handler


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ======================================================================
# AUTH
# ======================================================================

class FirebaseGoogleAuth:
    """Firebase-backed Gmail sign-in."""

    def __init__(
        self,
        config: Optional[FirebaseConfig] = None,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        opener: Optional[Callable[[str], Any]] = None,
        timeout: float = 30.0,
        app_title: str = "depression.ai",
    ):
        self.config = config if config is not None else load_firebase_config()
        self._transport = transport
        self._open = opener if opener is not None else webbrowser.open
        self._timeout = timeout
        self.app_title = app_title

    # -- plumbing -----------------------------------------------------

    @property
    def is_configured(self) -> bool:
        return self.config.is_usable_for_google

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    def _require_config(self) -> None:
        if self.config.is_configured:
            return
        raise OnboardingError(
            "Firebase is not configured. Set FIREBASE_API_KEY and "
            "FIREBASE_AUTH_DOMAIN (or drop a firebase.json into "
            "~/.config/depression/). See docs/firebase-onboarding.md."
        )

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.is_success:
            return
        detail = ""
        try:
            body = response.json()
            detail = str((body.get("error") or {}).get("message") or body.get("error") or "")
        except Exception:
            detail = response.text[:200]
        hint = _ERROR_HINTS.get(detail.split(":")[0].strip(), "")
        message = f"Firebase request failed ({response.status_code}): {detail or 'unknown error'}"
        raise OnboardingError(f"{message} {hint}".strip())

    def _url(self, base: str, path: str) -> str:
        """Firebase REST endpoints are keyed by the web API key."""
        return f"{base}/{path}?key={self.config.api_key}"

    async def _post_json(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        async with self._client() as client:
            try:
                response = await client.post(url, json=body)
            except httpx.HTTPError as exc:
                raise OnboardingError(f"Could not reach Firebase: {exc}") from exc
        self._raise_for_error(response)
        try:
            return response.json()
        except Exception as exc:
            raise OnboardingError("Firebase returned an unreadable response") from exc

    async def _post_form(self, url: str, data: Dict[str, str]) -> Dict[str, Any]:
        async with self._client() as client:
            try:
                response = await client.post(url, data=data)
            except httpx.HTTPError as exc:
                raise OnboardingError(f"Could not reach Firebase: {exc}") from exc
        self._raise_for_error(response)
        try:
            return response.json()
        except Exception as exc:
            raise OnboardingError("Firebase returned an unreadable response") from exc

    # -- verification -------------------------------------------------

    async def verify_id_token(self, id_token: str) -> FirebaseUser:
        """Exchange an ID token for the authoritative Firebase user record."""
        self._require_config()
        id_token = (id_token or "").strip()
        if not id_token:
            raise OnboardingError("No ID token was provided.")

        body = await self._post_json(
            self._url(IDENTITY_BASE, "accounts:lookup"),
            {"idToken": id_token},
        )
        users = body.get("users") or []
        if not users:
            raise OnboardingError("Firebase did not return a user for this token.")
        expires_at = 0.0
        if isinstance(body.get("expiresIn"), (int, float)):
            expires_at = time.time() + float(body["expiresIn"])
        return FirebaseUser.from_lookup(users[0], id_token=id_token, expires_at=expires_at)

    # -- browser flow -------------------------------------------------

    def start_callback_server(self) -> tuple:
        """Start the loopback server; returns (server, state_token, url)."""
        token = secrets.token_urlsafe(24)
        html = render_sign_in_page(self.config, "", token, self.app_title)
        server = _CallbackServer(html, token)
        server.start()
        # The page needs the absolute callback URL, known only after bind.
        html = render_sign_in_page(self.config, server.callback_url, token, self.app_title)
        server.state.html = html
        return server, token, server.url

    async def sign_in_with_google(self, timeout: float = DEFAULT_TIMEOUT) -> FirebaseUser:
        """
        Open the browser, wait for the Google consent screen, verify the token.
        """
        self._require_config()
        server, _token, url = self.start_callback_server()
        try:
            logger.info("Opening browser for Google sign-in: %s", redact(url))
            try:
                self._open(url)
            except Exception as exc:
                raise OnboardingError(f"Could not open a browser: {exc}") from exc

            # The server runs on its own thread; keep the event loop free.
            payload = await asyncio.to_thread(server.wait_for_payload, timeout)
        finally:
            server.stop()

        error = payload.get("error")
        if error:
            raise OnboardingError(str(error)[:300])

        id_token = str(payload.get("idToken") or "")
        refresh_token = str(payload.get("refreshToken") or "")
        if not id_token:
            raise OnboardingError("The browser did not return a Firebase ID token.")

        user = await self.verify_id_token(id_token)
        user.refresh_token = refresh_token
        # Prefer the verified record, but keep a display name if Firebase has none.
        if not user.name:
            user.name = str(payload.get("displayName") or "")
        if not user.photo_url:
            user.photo_url = str(payload.get("photoURL") or "")
        return user

    async def sign_in_with_google_oauth(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        client_secrets: Optional[Any] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> FirebaseUser:
        """
        Sign in with `google-auth-oauthlib`'s installed-app flow.

        Only works with a Desktop OAuth client (`client_secrets.json`).
        `InstalledAppFlow.run_local_server()` owns its own loopback server, so
        the browser comes up on its own and nothing is copied out of it; the
        Google ID token is then exchanged for a Firebase session and verified
        exactly like the built-in page's token. Returns the same
        `FirebaseUser`, so every caller is unaware which path was used.
        """
        self._require_config()
        from agent.tui.onboarding.google_oauth import GoogleOAuthError, GoogleOAuthSignIn

        oauth = GoogleOAuthSignIn(
            self.config.api_key,
            client_secrets=client_secrets,
            client_id=client_id or self.config.oauth_client_id or None,
            client_secret=client_secret or self.config.oauth_client_secret or None,
            timeout=timeout,
        )
        if not oauth.is_available:
            raise OnboardingError(f"Google OAuth unavailable: {oauth.describe()}")

        try:
            session = await oauth.sign_in_async()
        except GoogleOAuthError as exc:
            raise OnboardingError(str(exc)) from exc

        return await self.sign_in_with_id_token(
            str(session.get("idToken") or ""),
            str(session.get("refreshToken") or ""),
        )

    async def sign_in_with_id_token(
        self,
        id_token: str,
        refresh_token: str = "",
    ) -> FirebaseUser:
        """Headless path: verify a token obtained elsewhere."""
        user = await self.verify_id_token(id_token)
        user.refresh_token = (refresh_token or "").strip()
        return user

    # -- token lifecycle ----------------------------------------------

    async def refresh_tokens(self, refresh_token: str) -> Dict[str, Any]:
        """Use a refresh token for a fresh ID token (standard Firebase REST)."""
        self._require_config()
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            raise OnboardingError("No refresh token stored — sign in again.")

        body = await self._post_form(
            self._url(SECURE_TOKEN_BASE, "token"),
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
        )
        return {
            "id_token": body.get("access_token", ""),
            "refresh_token": body.get("refresh_token") or refresh_token,
            "expires_at": time.time() + float(body.get("expires_in") or 3600),
            "uid": body.get("user_id", ""),
            "provider": GOOGLE_PROVIDER,
        }

    async def sign_out(self, id_token: str) -> None:
        """Best-effort remote sign-out; local files are the caller's job."""
        if not self.config.is_configured or not (id_token or "").strip():
            return
        try:
            await self._post_json(
                self._url(IDENTITY_BASE, "accounts:signOut"),
                {"idToken": id_token},
            )
        except OnboardingError as exc:
            logger.debug("Remote sign-out failed (ignored): %s", redact(str(exc)))


__all__ = [
    "CALLBACK_PATH",
    "DEFAULT_TIMEOUT",
    "FirebaseGoogleAuth",
    "FirebaseUser",
    "GOOGLE_PROVIDER",
    "OnboardingError",
    "render_sign_in_page",
]