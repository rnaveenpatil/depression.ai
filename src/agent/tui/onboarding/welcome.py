"""
First-run welcome page with Gmail (Google) sign-in.

`WelcomeScreen` is a `ModalScreen` pushed on top of the main TUI the first
time an install starts without a stored profile. It offers exactly one way
forward:

    * SIGN IN WITH GOOGLE — opens the browser (or the installed-app OAuth
      flow when available), completes Firebase Auth, stores the Gmail
      profile on disk, and mirrors it into the global env file.

There is no guest path and no skip: onboarding is a gate, not a suggestion.
Sign-in can fail (offline, closed browser, wrong config) — the screen stays
up and shows the error so the user can try again.

Everything is fail-soft at the process level: no unhandled exception ever
leaves this screen, so a broken onboarding config can never break the CLI
in a way that prevents a retry.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Mapping, Optional

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.containers import Center, Horizontal, Middle, Vertical
from textual.widgets import Button, Input, Static

from agent.tui.onboarding.firebase_auth import (
    DEFAULT_TIMEOUT,
    FirebaseGoogleAuth,
    FirebaseUser,
    OnboardingError,
)
from agent.tui.onboarding.firebase_config import FirebaseConfig, load_firebase_config
from agent.tui.onboarding.google_oauth import GoogleOAuthSignIn, oauthlib_available
from agent.tui.onboarding.identity import apply_identity, export_identity
from agent.tui.onboarding.profile import ProfileStore, UserProfile
from agent.tui.theme import (
    AMBER,
    BG,
    DIM,
    ERROR,
    GREEN,
    GREEN_DIM,
    GREEN_GLOW,
    MUTED,
    PANEL,
    TEXT,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


APP_TITLE = "depression.ai"
DEFAULT_COUNTDOWN = 0.0     # retained for API compat; no longer auto-skips


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _esc(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _one_line(text: Any, limit: int = 200) -> str:
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def user_to_profile(user: FirebaseUser, store: ProfileStore) -> UserProfile:
    """Persist a Firebase identity as the local profile."""
    return store.save(
        UserProfile(
            uid=user.uid,
            email=user.email,
            name=user.name,
            photo_url=user.photo_url,
            provider=user.provider,
            email_verified=user.email_verified,
            signed_in=True,
            metadata={
                "firebase": True,
                "gmail": user.email.lower().endswith("@gmail.com"),
            },
        )
    )


def can_show_tui() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except Exception:
        return False


def first_run_instructions(config: FirebaseConfig) -> str:
    """What to do when no Firebase project is wired up yet."""
    if config.is_usable_for_google:
        return ""
    return (
        "Google sign-in needs a Firebase project.\n\n"
        "Add the web config from Firebase Console → Project settings:\n\n"
        "  FIREBASE_API_KEY=…\n"
        "  FIREBASE_AUTH_DOMAIN=proj.firebaseapp.com\n"
        "  FIREBASE_PROJECT_ID=proj\n\n"
        "…or drop the JSON at ~/.config/depression/firebase.json\n"
        "See docs/firebase-onboarding.md"
    )


# ======================================================================
# SCREEN
# ======================================================================

class WelcomeScreen(ModalScreen[Optional[UserProfile]]):
    """One-shot welcome / login page. Sign-in is the only way forward."""

    TITLE = APP_TITLE
    SUB_TITLE = "first run"

    CSS = f"""
    Screen {{
        background: {BG};
        color: {TEXT};
        align: center middle;
    }}
    #welcome {{
        width: 76;
        max-width: 94%;
        height: auto;
        border: round {GREEN_DIM};
        background: {PANEL};
        padding: 1 2 1 2;
    }}
    #welcome-title {{
        color: {GREEN_GLOW};
        text-style: bold;
        width: 100%;
        content-align: center middle;
    }}
    #welcome-sub {{
        color: {MUTED};
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }}
    #welcome-body {{
        height: auto;
        color: {TEXT};
        margin: 0 0 1 0;
    }}
    #welcome-method {{
        height: auto;
        color: {DIM};
        content-align: center middle;
        margin-bottom: 1;
    }}
    #welcome-status {{
        height: auto;
        color: {AMBER};
        margin: 0 0 1 0;
        content-align: left middle;
    }}
    #welcome-buttons {{
        height: auto;
        margin: 0 0 1 0;
    }}
    #welcome-buttons Button {{
        width: 100%;
        height: 3;
        border: round {GREEN};
        background: transparent;
        color: {GREEN};
        text-style: bold;
    }}
    #welcome-buttons Button:hover, #welcome-buttons Button:focus {{
        background: {GREEN};
        color: {BG};
        border: round {GREEN};
    }}
    #welcome-buttons Button:disabled {{
        color: {MUTED};
        border: round {DIM};
        background: transparent;
    }}
    #welcome-input-row {{
        height: auto;
        margin: 0 0 1 0;
    }}
    #welcome-input-row.hidden {{ display: none; }}
    #welcome-input {{
        width: 1fr;
        height: 3;
        background: {BG};
        border: round {DIM};
        color: {TEXT};
    }}
    #welcome-input:focus {{ border: round {GREEN}; }}
    #welcome-hint {{
        height: 1;
        color: {DIM};
        content-align: center middle;
    }}
    """

    BINDINGS = [
        Binding("enter", "sign_in", "Sign in", priority=True),
        Binding("g", "sign_in", "Sign in", priority=True),
        Binding("t", "manual", "Paste token", priority=True),
        Binding("escape", "clear_status", "Clear message", priority=True),
    ]

    def __init__(
        self,
        *,
        config: Optional[FirebaseConfig] = None,
        store: Optional[ProfileStore] = None,
        auth: Optional[FirebaseGoogleAuth] = None,
        countdown: float = DEFAULT_COUNTDOWN,     # unused, kept for compat
        timeout: float = DEFAULT_TIMEOUT,
        env_path: Optional[Any] = None,
        oauth: Optional[GoogleOAuthSignIn] = None,
        enable_oauth: bool = True,
    ):
        super().__init__()
        self._env_path = env_path
        self.config = config if config is not None else load_firebase_config()
        self.store = store if store is not None else ProfileStore()
        self.auth = auth if auth is not None else FirebaseGoogleAuth(self.config)

        if not enable_oauth:
            self.oauth_sign_in: Optional[GoogleOAuthSignIn] = None
        elif oauth is not None:
            self.oauth_sign_in = oauth
        else:
            self.oauth_sign_in = (
                GoogleOAuthSignIn(
                    self.config.api_key,
                    client_id=self.config.oauth_client_id,
                    client_secret=self.config.oauth_client_secret,
                    timeout=float(timeout),
                )
                if oauthlib_available()
                else None
            )

        self.timeout = float(timeout)
        self._busy = False
        self._done = False

    # ------------------------------------------------------------------
    # LAYOUT
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Center():
            with Middle():
                with Vertical(id="welcome"):
                    yield Static(f"▌ {APP_TITLE.upper()}", id="welcome-title",
                                 markup=True)
                    yield Static("first run — sign in to continue",
                                 id="welcome-sub", markup=True)
                    yield Static(self._intro_text(), id="welcome-body",
                                 markup=True)
                    yield Static(self._method_pill(), id="welcome-method",
                                 markup=True)
                    yield Static("", id="welcome-status", markup=True)
                    with Horizontal(id="welcome-input-row"):
                        yield Input(
                            placeholder="paste a Firebase ID token, then Enter",
                            id="welcome-input",
                            password=True,
                        )
                    with Horizontal(id="welcome-buttons"):
                        yield Button("SIGN IN WITH GOOGLE  (G)", id="btn-google")
                        yield Button("PASTE TOKEN  (T)", id="btn-token")
                    yield Static("G sign in · T paste token · enter confirm",
                                 id="welcome-hint", markup=True)

    def _intro_text(self) -> str:
        if self.config.is_usable_for_google:
            return (
                "Sign in with your Google account to continue.\n"
                "Your name and email are stored on this machine only —\n"
                "nothing else is uploaded."
            )
        return f"[{AMBER}]{_esc(first_run_instructions(self.config))}[/]"

    def _method_pill(self) -> str:
        method = self._method_text()
        color = GREEN if self.config.is_usable_for_google else AMBER
        return f"[{color}]· sign-in path: {_esc(method)} ·[/]"

    def _method_text(self) -> str:
        oauth = self.oauth_sign_in
        if oauth is not None and oauth.is_available:
            return "Google OAuth (installed-app flow)"
        if oauth is not None:
            return f"browser ({oauth.describe()})"
        return "browser (Firebase popup)"

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._focus("#btn-google")
        if self.config.is_usable_for_google:
            self._status(
                f"[{GREEN}]Firebase ready — project "
                f"{_esc(self.config.project_id or 'unknown')}[/]"
            )
        else:
            self._status(f"[{AMBER}]Google sign-in unavailable — see notes above.[/]")
        # Keep focus cycling tidy.
        self.set_interval(0.25, self._ensure_focus)

    def _ensure_focus(self) -> None:
        if self._busy or self._done:
            return
        try:
            focused = self.app.focused
            if focused is None:
                self._focus("#btn-google")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _focus(self, selector: str) -> None:
        try:
            self.query_one(selector, Button).focus()
        except Exception:
            pass

    def _status(self, markup: str) -> None:
        try:
            self.query_one("#welcome-status", Static).update(markup)
        except Exception:
            pass

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        for button in self.query(Button):
            button.disabled = busy
        if message:
            self._status(message)

    def _finish(self, profile: Optional[UserProfile]) -> None:
        if self._done:
            return
        self._done = True
        try:
            self.dismiss(profile)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------

    def action_clear_status(self) -> None:
        """Escape only clears the status message — it cannot skip onboarding."""
        if self._busy:
            return
        self._status("")

    def action_manual(self) -> None:
        if self._busy:
            return
        try:
            row = self.query_one("#welcome-input-row", Horizontal)
            row.remove_class("hidden")
            field = self.query_one("#welcome-input", Input)
            field.focus()
        except Exception:
            return
        self._status(f"[{AMBER}]Paste a Firebase ID token, then press enter.[/]")

    async def action_sign_in(self) -> None:
        if self._busy:
            return
        if not self.auth.is_configured:
            self._status(
                f"[{ERROR}]Firebase is not configured — see the notes above.[/]"
            )
            return

        # Preferred: google-auth-oauthlib installed-app flow.
        if self.oauth_sign_in is not None and self.oauth_sign_in.is_available:
            if await self._run_sign_in(
                self.auth.sign_in_with_google_oauth(
                    self.timeout,
                    client_secrets=self.oauth_sign_in.client_secrets,
                    client_id=self.oauth_sign_in.client_id or None,
                    client_secret=self.oauth_sign_in.client_secret or None,
                )
            ):
                return
            if self._done:
                return
            self._status(
                f"[{AMBER}]OAuth sign-in failed — retrying in the browser…[/]"
            )
        await self._run_sign_in(self.auth.sign_in_with_google(self.timeout))

    async def action_submit_token(self) -> None:
        if self._busy:
            return
        try:
            field = self.query_one("#welcome-input", Input)
        except Exception:
            return
        token = (field.value or "").strip()
        field.value = ""
        if not token:
            self._status(f"[{ERROR}]Paste a Firebase ID token first.[/]")
            return
        await self._run_sign_in(self.auth.sign_in_with_id_token(token))

    async def _run_sign_in(self, coro: Any) -> bool:
        self._set_busy(True, f"[{AMBER}]Waiting for Google…[/]")
        try:
            user = await coro
        except asyncio.CancelledError:
            self._set_busy(False)
            raise
        except OnboardingError as exc:
            logger.warning("Google sign-in failed: %s", exc)
            self._set_busy(False, f"[{ERROR}]{_esc(_one_line(exc))}[/]")
            return False
        except Exception as exc:
            logger.error("Sign-in crashed: %s", exc, exc_info=True)
            self._set_busy(
                False,
                f"[{ERROR}]Sign-in failed: {_esc(_one_line(exc))}[/]",
            )
            return False

        try:
            if user.refresh_token:
                self.store.save_tokens(user.tokens)
            profile = user_to_profile(user, self.store)
            # Identity goes to the global env file, mirrored into os.environ,
            # and broadcast to EnvManager subscribers.
            export_identity(profile, env_path=self._env_path)
        except Exception as exc:
            logger.error("Could not store profile: %s", exc, exc_info=True)
            self._set_busy(
                False,
                f"[{ERROR}]Signed in, but saving the profile failed: "
                f"{_esc(_one_line(exc))}[/]",
            )
            return False

        self._status(
            f"[{GREEN}]Signed in as {_esc(profile.display_name)} "
            f"({_esc(profile.email) or 'no email'}).[/]"
        )
        await asyncio.sleep(0.4)
        self._finish(profile)
        return True

    # ------------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------------

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "btn-google":
            await self.action_sign_in()
        elif event.button.id == "btn-token":
            self.action_manual()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        await self.action_submit_token()


# ======================================================================
# FIRST-RUN GATE
# ======================================================================

ENV_ENABLED = "DEPRESSION_ONBOARDING"
ENV_TIMEOUT = "DEPRESSION_ONBOARDING_TIMEOUT"

DISABLED_VALUES = {"0", "off", "no", "false", "disabled", "none"}
FORCE_VALUES = {"force", "reset", "again"}


def _env_float(name: str, default: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def should_onboard(
    store: Optional[ProfileStore] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """
    The first-run gate.

    Shows the page only when: onboarding is not disabled, we are on a real
    terminal, the app is not headless, and no profile exists yet.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(ENV_ENABLED) or "").strip().lower()

    if raw in DISABLED_VALUES:
        return False
    if not can_show_tui():
        return False

    profile_store = store if store is not None else ProfileStore()

    if raw in FORCE_VALUES:
        return True

    return profile_store.is_new_user()


async def push_welcome_if_new_user(
    app: Any,
    *,
    store: Optional[ProfileStore] = None,
    config: Optional[FirebaseConfig] = None,
    countdown: Optional[float] = None,      # unused, kept for compat
    timeout: Optional[float] = None,
    context_manager: Any = None,
    env_path: Optional[Any] = None,
) -> Optional[UserProfile]:
    """
    Show the welcome page once, then keep the identity available forever.

    First run  -> the page is pushed; the profile it stores is exported to
                  the environment and handed back to the caller.
    Later runs -> no page; the stored profile is re-exported and injected
                  into the conversation so the agent knows who it is
                  talking to.

    Never raises — onboarding must not be able to break start-up.
    """
    try:
        profile_store = store if store is not None else ProfileStore()

        if should_onboard(profile_store) and not getattr(app, "is_headless", False):
            logger.info("First run detected — showing the welcome page")
            screen = WelcomeScreen(
                config=config if config is not None else load_firebase_config(),
                store=profile_store,
                timeout=(
                    timeout if timeout is not None
                    else _env_float(ENV_TIMEOUT, DEFAULT_TIMEOUT)
                ),
                env_path=env_path,
            )
            profile = await app.push_screen_wait(screen)
            if profile is not None:
                await apply_identity(profile, context_manager, env_path=env_path)
            return profile

        profile = profile_store.load()
        if profile is not None:
            logger.debug("Returning user: %s", profile.display_name)
            await apply_identity(profile, context_manager, env_path=env_path)
        return profile
    except Exception as exc:
        logger.warning("Onboarding skipped (%s) — continuing", exc)
        return None


__all__ = [
    "DEFAULT_COUNTDOWN",
    "ENV_ENABLED",
    "ENV_TIMEOUT",
    "WelcomeScreen",
    "can_show_tui",
    "first_run_instructions",
    "push_welcome_if_new_user",
    "should_onboard",
    "user_to_profile",
]