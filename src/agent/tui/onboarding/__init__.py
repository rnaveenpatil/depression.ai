"""
First-run TUI onboarding: welcome page + Gmail (Google) sign-in via Firebase.

`DepressionApp` pushes `WelcomeScreen` on start-up when this install has no
stored profile, so the page lives entirely under `agent/tui/`.

Public surface:

    load_firebase_config()          -> FirebaseConfig (env / file / explicit)
    ProfileStore                    -> reads/writes user.json + user_tokens.json
    FirebaseGoogleAuth              -> browser sign-in, ID-token verify, refresh
    GoogleOAuthSignIn               -> google-auth-oauthlib installed-app flow
    WelcomeScreen                   -> the modal page pushed on first launch
    should_onboard / push_welcome_if_new_user -> the first-run gate
    export_identity / apply_identity -> mirror the profile into env + context
"""

from __future__ import annotations

from agent.tui.onboarding.firebase_auth import (
    FirebaseGoogleAuth,
    FirebaseUser,
    OnboardingError,
    render_sign_in_page,
)
from agent.tui.onboarding.firebase_config import FirebaseConfig, load_firebase_config
from agent.tui.onboarding.google_oauth import (
    GoogleOAuthError,
    GoogleOAuthSignIn,
    find_client_secrets,
    oauthlib_available,
)
from agent.tui.onboarding.identity import (
    apply_identity,
    clear_identity,
    export_identity,
    identity_env,
    load_identity,
    sign_out,
)
from agent.tui.onboarding.profile import ProfileStore, UserProfile

__all__ = [
    "FirebaseConfig",
    "FirebaseGoogleAuth",
    "FirebaseUser",
    "GoogleOAuthError",
    "GoogleOAuthSignIn",
    "OnboardingError",
    "ProfileStore",
    "UserProfile",
    "apply_identity",
    "clear_identity",
    "export_identity",
    "find_client_secrets",
    "identity_env",
    "load_firebase_config",
    "load_identity",
    "oauthlib_available",
    "render_sign_in_page",
    "sign_out",
]