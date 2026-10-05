"""
Tests for the first-run TUI onboarding (Firebase / Gmail sign-in).

Nothing here touches the network: Firebase REST calls run through
httpx.MockTransport, and the loopback callback server is driven with a real
local request. Every test uses tmp_path so the developer's own
~/.local/share/depression/user.json is never read or written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from agent.tui.onboarding.firebase_auth import (
    FirebaseGoogleAuth,
    FirebaseUser,
    OnboardingError,
    render_sign_in_page,
)
from agent.tui.onboarding.firebase_config import (
    FirebaseConfig,
    _config_from_mapping,
    load_firebase_config,
)
from agent.tui.onboarding.google_oauth import (
    ENV_CLIENT_ID,
    ENV_CLIENT_SECRETS,
    ENV_CLIENT_SECRETS_ALT,
    GoogleOAuthError,
    GoogleOAuthSignIn,
    find_client_secrets,
    oauthlib_available,
)
from agent.tui.onboarding.identity import (
    ENV_EMAIL,
    ENV_LAST_LOGIN,
    ENV_NAME,
    ENV_PROVIDER,
    ENV_SIGNED_IN,
    ENV_UID,
    ENV_VERIFIED,
    IDENTITY_CONTEXT_PREFIX,
    IDENTITY_ENVS,
    apply_identity,
    clear_identity,
    export_identity,
    identity_context,
    identity_env,
    identity_is_present,
    load_identity,
    sign_out,
)
from agent.tui.onboarding.profile import (
    ProfileStore,
    UserProfile,
    guest_profile,
    profile_mode,
)
from agent.tui.onboarding.welcome import (
    WelcomeScreen,
    push_welcome_if_new_user,
    should_onboard,
    user_to_profile,
)


WEB_CONFIG = {
    "api_key": "AIzaTestKey1234567890",
    "auth_domain": "demo-project.firebaseapp.com",
    "project_id": "demo-project",
    "app_id": "1:1234567890:web:abcdef123456",
}

CONFIG = FirebaseConfig(**WEB_CONFIG)


# ======================================================================
# FIXTURES
# ======================================================================

@pytest.fixture
def store(tmp_path):
    return ProfileStore(data_dir=tmp_path)


def _lookup_handler(payload, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


GOOGLE_LOOKUP = {
    "users": [
        {
            "localId": "uid-42",
            "email": "patil@gmail.com",
            "displayName": "Patil",
            "photoUrl": "https://example.com/p.png",
            "emailVerified": True,
            "providerId": "google.com",
        }
    ]
}


def _auth(handler=None) -> FirebaseGoogleAuth:
    return FirebaseGoogleAuth(
        CONFIG,
        transport=httpx.MockTransport(handler) if handler else None,
        opener=lambda url: True,
    )


# ======================================================================
# CONFIG
# ======================================================================

def test_config_from_individual_env_vars(tmp_path):
    config = load_firebase_config(
        environ={
            "FIREBASE_API_KEY": "key-1",
            "FIREBASE_AUTH_DOMAIN": "p.firebaseapp.com",
            "FIREBASE_PROJECT_ID": "p",
        },
        config_dir=tmp_path,
    )
    assert config.is_configured
    assert config.is_usable_for_google
    assert config.to_web_config()["apiKey"] == "key-1"


def test_config_from_inline_json_env(tmp_path):
    config = load_firebase_config(
        environ={"DEPRESSION_FIREBASE_WEB_APP_CONFIG": json.dumps(WEB_CONFIG)},
        config_dir=tmp_path,
    )
    assert config.project_id == "demo-project"
    assert config.app_id == WEB_CONFIG["app_id"]


def test_config_from_file_accepts_camel_case_and_wrappers(tmp_path):
    (tmp_path / "firebase.json").write_text(json.dumps({"web": WEB_CONFIG}))
    config = load_firebase_config(environ={}, config_dir=tmp_path)
    assert config.api_key == WEB_CONFIG["api_key"]
    assert config.auth_domain == WEB_CONFIG["auth_domain"]


def test_config_accepts_snake_case_keys(tmp_path):
    config = load_firebase_config(
        explicit={"api_key": "abc", "auth_domain": "d.firebaseapp.com"}
    )
    assert config.is_usable_for_google
    assert config.api_key == "abc"


def test_missing_config_is_empty_and_safe(tmp_path):
    config = load_firebase_config(environ={}, config_dir=tmp_path)
    assert not config.is_configured
    assert config.to_web_config() == {}
    assert "not configured" in config.describe()


def test_describe_masks_api_key(tmp_path):
    text = load_firebase_config(
        environ={
            "FIREBASE_API_KEY": WEB_CONFIG["api_key"],
            "FIREBASE_PROJECT_ID": WEB_CONFIG["project_id"],
        },
        config_dir=tmp_path,
    ).describe()
    assert WEB_CONFIG["api_key"] not in text
    assert "demo-project" in text


def test_broken_config_file_does_not_raise(tmp_path):
    (tmp_path / "firebase.json").write_text("{not json")
    assert load_firebase_config(environ={}, config_dir=tmp_path).is_configured is False


# ======================================================================
# PROFILE STORE
# ======================================================================

def test_new_user_detection(store):
    assert store.is_new_user()
    assert store.exists() is False
    store.save(guest_profile())
    assert store.exists()
    assert store.is_new_user() is False


def test_profile_roundtrip(store):
    saved = store.save(
        UserProfile(uid="uid-1", email="patil@gmail.com", name="Patil", provider="google.com")
    )
    loaded = store.load()

    assert loaded is not None
    assert loaded.uid == "uid-1"
    assert loaded.email == "patil@gmail.com"
    assert loaded.name == "Patil"
    assert loaded.is_gmail
    assert loaded.display_name == "Patil"
    assert loaded.initials == "PA"
    assert saved.created_at == loaded.created_at


def test_repeat_login_bumps_counter_not_created_at(store):
    first = store.save(UserProfile(uid="uid-1", name="A"))
    second = store.save(UserProfile(uid="uid-1", name="A"))
    third = store.save(UserProfile(uid="uid-2", name="B"))

    assert second.login_count == 2
    assert second.created_at == first.created_at
    assert third.login_count == 1
    assert third.created_at >= first.created_at


def test_profile_display_name_falls_back(store):
    store.save(UserProfile(uid="uid-1"))
    assert store.load().display_name == "uid-1"


def test_tokens_are_written_owner_only(store):
    store.save(guest_profile())
    store.save_tokens({"id_token": "secret", "refresh_token": "r"})
    assert store.load_tokens()["id_token"] == "secret"
    assert profile_mode(store.token_path) == 0o600
    assert profile_mode(store.profile_path) == 0o600


def test_corrupt_profile_is_treated_as_new_user(store):
    store.profile_path.write_text("{oops")
    assert store.load() is None
    assert store.is_new_user()


def test_sign_out_removes_everything(store):
    store.save(guest_profile())
    store.save_tokens({"id_token": "x"})
    store.sign_out()

    assert store.exists() is False
    assert store.load_tokens() == {}


def test_google_user_becomes_profile(store):
    user = FirebaseUser(
        uid="uid-9",
        email="someone@gmail.com",
        name="Someone",
        provider="google.com",
        refresh_token="r",
    )
    profile = user_to_profile(user, store)

    assert profile.signed_in is True
    assert profile.metadata["gmail"] is True
    assert store.load().uid == "uid-9"


# ======================================================================
# SIGN-IN PAGE + CALLBACK SERVER
# ======================================================================

def test_sign_in_page_embeds_config_state_and_callback():
    html = render_sign_in_page(CONFIG, "http://127.0.0.1:5555/callback", "st-123")

    assert WEB_CONFIG["api_key"] in html
    assert WEB_CONFIG["auth_domain"] in html
    assert "st-123" in html
    assert "http://127.0.0.1:5555/callback" in html
    assert "GoogleAuthProvider" in html
    assert "signInWithPopup" in html


async def test_callback_server_receives_token():
    auth = _auth()
    server, state_token, url = auth.start_callback_server()
    try:
        assert url.startswith("http://localhost:")
        async with httpx.AsyncClient() as client:
            page = await client.get(url)
            response = await client.post(
                server.callback_url,
                json={"state": state_token, "idToken": "tok-123", "refreshToken": "ref-9"},
            )
        assert page.status_code == 200
        assert response.status_code == 200

        payload = server.wait_for_payload(timeout=5)
        assert payload["idToken"] == "tok-123"
        assert payload["refreshToken"] == "ref-9"
    finally:
        server.stop()


async def test_callback_server_rejects_foreign_state():
    auth = _auth()
    server, _token, _url = auth.start_callback_server()
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                server.callback_url, json={"state": "not-the-token", "idToken": "x"}
            )
        with pytest.raises(OnboardingError, match="state mismatch"):
            server.wait_for_payload(timeout=5)
    finally:
        server.stop()


async def test_callback_server_times_out():
    auth = _auth()
    server, _token, _url = auth.start_callback_server()
    try:
        with pytest.raises(OnboardingError, match="Timed out"):
            server.wait_for_payload(timeout=0.2)
    finally:
        server.stop()


# ======================================================================
# FIREBASE REST
# ======================================================================

async def test_verify_id_token_maps_user_record():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "users": [{
                "localId": "uid-42",
                "email": "patil@gmail.com",
                "displayName": "Patil",
                "photoUrl": "https://example.com/a.png",
                "emailVerified": True,
                "providerId": "google.com",
            }]
        })

    user = await _auth(handler).verify_id_token("tok-123")

    assert user.uid == "uid-42"
    assert user.email == "patil@gmail.com"
    assert user.name == "Patil"
    assert user.email_verified is True
    assert "accounts:lookup" in seen["url"]
    assert WEB_CONFIG["api_key"] in seen["url"]
    assert seen["body"] == {"idToken": "tok-123"}


async def test_verify_id_token_surfaces_firebase_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "TOKEN_EXPIRED"}})

    with pytest.raises(OnboardingError, match="TOKEN_EXPIRED"):
        await _auth(handler).verify_id_token("stale")


async def test_verify_id_token_requires_a_token():
    with pytest.raises(OnboardingError, match="No ID token"):
        await _auth().verify_id_token("   ")


async def test_unconfigured_auth_refuses_to_start():
    auth = FirebaseGoogleAuth(FirebaseConfig(), opener=lambda url: True)
    assert auth.is_configured is False
    with pytest.raises(OnboardingError, match="not configured"):
        await auth.sign_in_with_google(timeout=0.1)


async def test_refresh_tokens_returns_fresh_id_token():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "securetoken.googleapis.com" in str(request.url)
        return httpx.Response(200, json={
            "access_token": "new-id-token",
            "refresh_token": "ref-2",
            "expires_in": "3600",
            "user_id": "uid-42",
        })

    tokens = await _auth(handler).refresh_tokens("ref-1")

    assert tokens["id_token"] == "new-id-token"
    assert tokens["refresh_token"] == "ref-2"
    assert tokens["expires_at"] > 0


async def test_sign_in_with_id_token_verifies_locally():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"users": [{"localId": "uid-7", "email": "a@b.co"}]})

    user = await _auth(handler).sign_in_with_id_token("tok", refresh_token="ref")

    assert user.uid == "uid-7"
    assert user.refresh_token == "ref"


async def test_sign_in_reports_browser_failure():
    def _broken_opener(url: str):
        raise OSError("no display")

    auth = FirebaseGoogleAuth(CONFIG, opener=_broken_opener)
    with pytest.raises(OnboardingError, match="Could not open a browser"):
        await auth.sign_in_with_google(timeout=1)


async def test_sign_in_times_out_without_browser_response():
    auth = _auth()  # opener does nothing: nobody ever calls back
    with pytest.raises(OnboardingError, match="Timed out"):
        await auth.sign_in_with_google(timeout=0.3)


# ======================================================================
# FIRST-RUN GATE
# ======================================================================

def test_gate_skips_when_profile_exists(store, monkeypatch):
    store.save(guest_profile())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert should_onboard(store, environ={}) is False


def test_gate_shows_for_new_user(store, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert should_onboard(store, environ={}) is True


def test_gate_respects_env_switch(store, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert should_onboard(store, environ={"DEPRESSION_ONBOARDING": "off"}) is False
    assert should_onboard(store, environ={"DEPRESSION_ONBOARDING": "0"}) is False


def test_gate_needs_a_tty(store):
    assert should_onboard(store, environ={}) is False


def test_gate_force_ignores_existing_profile(store, monkeypatch):
    store.save(guest_profile())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert should_onboard(store, environ={"DEPRESSION_ONBOARDING": "force"}) is True


class _HostApp(App):
    def compose(self) -> ComposeResult:
        yield Static("host")


async def test_push_is_skipped_in_headless_mode(store):
    app = _HostApp()
    async with app.run_test() as pilot:
        assert await push_welcome_if_new_user(app, store=store, countdown=0) is None
        await pilot.pause()


async def test_push_is_skipped_when_disabled(store, monkeypatch):
    monkeypatch.setenv("DEPRESSION_ONBOARDING", "off")
    app = _HostApp()
    async with app.run_test() as pilot:
        assert await push_welcome_if_new_user(app, store=store, countdown=0) is None
        await pilot.pause()


# ======================================================================
# WELCOME SCREEN
# ======================================================================

async def test_welcome_screen_guest_saves_profile(store):
    app = _HostApp()
    async with app.run_test() as pilot:
        app.push_screen(WelcomeScreen(config=CONFIG, store=store, countdown=0))
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

    profile = store.load()
    assert profile is not None
    assert profile.signed_in is False


async def test_welcome_screen_escape_skips(store):
    app = _HostApp()
    async with app.run_test() as pilot:
        app.push_screen(WelcomeScreen(config=CONFIG, store=store, countdown=0))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert store.exists() is False


async def test_welcome_screen_signs_in_with_stub_auth(store):
    stub_user = FirebaseUser(
        uid="uid-77",
        email="newuser@gmail.com",
        name="New User",
        provider="google.com",
        refresh_token="ref-77",
    )

    class _StubAuth(FirebaseGoogleAuth):
        async def sign_in_with_google(self, timeout: float = 180.0) -> FirebaseUser:
            return stub_user

    app = _HostApp()
    async with app.run_test() as pilot:
        app.push_screen(
            WelcomeScreen(
                config=CONFIG,
                store=store,
                auth=_StubAuth(CONFIG),
                countdown=0,
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()
            if store.exists():
                break

    profile = store.load()
    assert profile is not None
    assert profile.uid == "uid-77"
    assert profile.name == "New User"
    assert store.load_tokens()["refresh_token"] == "ref-77"


async def test_welcome_screen_reports_failure(store):
    class _BrokenAuth(FirebaseGoogleAuth):
        async def sign_in_with_google(self, timeout: float = 180.0) -> FirebaseUser:
            raise OnboardingError("browser refused")

    app = _HostApp()
    async with app.run_test() as pilot:
        app.push_screen(
            WelcomeScreen(config=CONFIG, store=store, auth=_BrokenAuth(CONFIG), countdown=0)
        )
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        status = app.screen.query_one("#welcome-status", Static)
        assert "browser refused" in str(status.render())

    assert store.exists() is False

# ======================================================================
# IDENTITY -> ENV + CONTEXT
# ======================================================================

def _google_profile() -> UserProfile:
    return UserProfile(
        uid="uid-42",
        email="patil@gmail.com",
        name="Patil",
        provider="google.com",
        email_verified=True,
        signed_in=True,
    )


def test_identity_env_maps_every_field():
    values = identity_env(_google_profile())

    assert values[ENV_UID] == "uid-42"
    assert values[ENV_EMAIL] == "patil@gmail.com"
    assert values[ENV_NAME] == "Patil"
    assert values[ENV_PROVIDER] == "google.com"
    assert values[ENV_VERIFIED] == "true"
    assert values[ENV_SIGNED_IN] == "true"
    assert int(values[ENV_LAST_LOGIN]) > 0


def test_export_identity_writes_global_env_file_and_process(tmp_path):
    env_file = tmp_path / "env"
    env_file.write_text("AWS_DEFAULT_REGION=ap-south-1\n")

    export_identity(_google_profile(), env_path=env_file)

    written = env_file.read_text()
    assert "DEPRESSION_USER_EMAIL=patil@gmail.com" in written
    assert "DEPRESSION_USER_NAME=Patil" in written
    assert "AWS_DEFAULT_REGION=ap-south-1" in written  # other creds untouched
    assert os.environ[ENV_EMAIL] == "patil@gmail.com"

    clear_identity(env_path=env_file)


def test_export_identity_never_writes_the_refresh_token(tmp_path):
    env_file = tmp_path / "env"
    export_identity(_google_profile(), env_path=env_file)
    assert "refresh" not in env_file.read_text().lower()
    assert "id_token" not in env_file.read_text().lower()


def test_export_identity_skips_persistence_when_asked(tmp_path):
    env_file = tmp_path / "env"
    values = export_identity(_google_profile(), persist=False, env_path=env_file)
    assert values[ENV_EMAIL] == "patil@gmail.com"
    assert not env_file.exists()


def test_load_identity_round_trip():
    fake = {}
    export_identity(_google_profile(), persist=False, environ=fake)
    loaded = load_identity(environ=fake)

    assert loaded[ENV_EMAIL] == "patil@gmail.com"
    assert identity_is_present(environ=fake) is True


def test_clear_identity_removes_every_var():
    fake = {}
    export_identity(_google_profile(), persist=False, environ=fake)
    clear_identity(persist=False, environ=fake)

    assert load_identity(environ=fake) == {v: "" for v in IDENTITY_ENVS}
    assert identity_is_present(environ=fake) is False


def test_identity_context_names_the_user_and_the_env_vars():
    text = identity_context(_google_profile())

    assert text.startswith(IDENTITY_CONTEXT_PREFIX)
    assert "patil@gmail.com" in text
    assert ENV_EMAIL in text
    assert "gmail sign-in" in text


class _FakeContextManager:
    def __init__(self):
        self.messages = []

    async def add_system_message(self, content: str, pinned: bool = False):
        self.messages.append({"content": content, "pinned": pinned})
        return self.messages[-1]


async def test_apply_identity_exports_and_adds_context_once(tmp_path):
    env_file = tmp_path / "env"
    context = _FakeContextManager()
    profile = _google_profile()

    values = await apply_identity(profile, context, export=True)

    assert values[ENV_EMAIL] == "patil@gmail.com"
    assert len(context.messages) == 1
    assert context.messages[0]["pinned"] is True

    # A second launch must not repeat the block.
    await apply_identity(profile, context, export=True)
    assert len(context.messages) == 1

    clear_identity(env_path=env_file)


async def test_apply_identity_survives_a_broken_context_manager():
    class _Broken:
        async def add_system_message(self, content, pinned=False):
            raise RuntimeError("no session")

    values = await apply_identity(_google_profile(), _Broken(), export=False)
    assert values == {}
    clear_identity()


async def test_returning_user_is_reexported_without_the_page(store, tmp_path):
    env_file = tmp_path / "env"
    store.save(_google_profile())
    context = _FakeContextManager()

    app = _HostApp()
    async with app.run_test() as pilot:
        profile = await push_welcome_if_new_user(
            app, store=store, countdown=0, context_manager=context, env_path=env_file
        )
        await pilot.pause()

    assert profile is not None
    assert profile.email == "patil@gmail.com"
    assert len(context.messages) == 1  # identity handed to the agent
    assert "DEPRESSION_USER_EMAIL=patil@gmail.com" in env_file.read_text()
    clear_identity(env_path=env_file)


async def test_first_run_exports_identity_after_sign_in(store, tmp_path, monkeypatch):
    env_file = tmp_path / "env"
    monkeypatch.setenv("DEPRESSION_ONBOARDING", "force")

    stub_user = FirebaseUser(
        uid="uid-88",
        email="firstrun@gmail.com",
        name="First Run",
        provider="google.com",
    )

    class _StubAuth(FirebaseGoogleAuth):
        async def sign_in_with_google(self, timeout: float = 180.0) -> FirebaseUser:
            return stub_user

    app = _HostApp()
    async with app.run_test(size=(84, 30)) as pilot:
        app.push_screen(
            WelcomeScreen(
                config=CONFIG,
                store=store,
                auth=_StubAuth(CONFIG),
                countdown=0,
                env_path=env_file,
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()
            if store.exists():
                break

    from agent.utils.env_manager import load_env_file

    persisted = load_env_file(env_file)
    assert persisted[ENV_EMAIL] == "firstrun@gmail.com"
    assert persisted[ENV_NAME] == "First Run"
    clear_identity(env_path=env_file)


def test_sign_out_clears_profile_and_env(tmp_path):
    env_file = tmp_path / "env"
    store = ProfileStore(data_dir=tmp_path)
    store.save(_google_profile())
    store.save_tokens({"refresh_token": "secret"})
    export_identity(store.load(), env_path=env_file)

    sign_out(store, env_path=env_file)

    assert not store.exists()
    assert not store.token_path.exists()
    assert "DEPRESSION_USER_" not in env_file.read_text()
    assert ENV_EMAIL not in os.environ


# ======================================================================
# GOOGLE-AUTH-OAUTHLIB INSTALLED-APP FLOW
# ======================================================================

CLIENT_SECRETS_JSON = {
    "installed": {
        "client_id": "123.apps.googleusercontent.com",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"],
    }
}


class _StubResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _StubCredentials:
    def __init__(self, id_token="google-id-token"):
        self.id_token = id_token


def _write_client_secrets(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "client_secrets.json"
    path.write_text(json.dumps(CLIENT_SECRETS_JSON))
    return path


def test_oauthlib_available_reflects_the_installed_packages():
    # The extras are installed in CI-less environments too; the helper must
    # answer without raising either way.
    assert isinstance(oauthlib_available(), bool)


def test_find_client_secrets_prefers_the_env_var(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    found = find_client_secrets(environ={ENV_CLIENT_SECRETS: str(secrets_file)})
    assert found == secrets_file


def test_find_client_secrets_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(ENV_CLIENT_SECRETS, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRETS_ALT, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert find_client_secrets(environ={}) is None


def test_find_client_secrets_falls_back_to_the_config_dir(tmp_path, monkeypatch):
    secrets_file = _write_client_secrets(tmp_path / ".config" / "depression")
    monkeypatch.delenv(ENV_CLIENT_SECRETS, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRETS_ALT, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert find_client_secrets(environ={}) == secrets_file


def test_google_id_token_uses_installed_app_flow(tmp_path):
    """`run_local_server(port=0)` is what actually blocks on the browser."""
    secrets_file = _write_client_secrets(tmp_path)
    seen = {}

    class _StubFlow:
        @classmethod
        def from_client_secrets_file(cls, path, scopes):
            seen["path"] = path
            seen["scopes"] = list(scopes)
            return cls()

        def run_local_server(self, port=0, **kwargs):
            seen["port"] = port
            return _StubCredentials("google-token-1")

    oauth = GoogleOAuthSignIn("api-key", client_secrets=secrets_file)
    oauth._run_flow = None
    import agent.tui.onboarding.google_oauth as go

    original = go._import_installed_app_flow
    go._import_installed_app_flow = lambda: _StubFlow
    try:
        assert oauth.google_id_token() == "google-token-1"
    finally:
        go._import_installed_app_flow = original

    assert seen["path"] == str(secrets_file)
    assert seen["port"] == 0
    assert seen["scopes"] == [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]


def test_exchange_posts_the_documented_signinwithidp_payload(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    captured = {}

    def _post(url, json=None, timeout=None):
        captured.update(url=url, payload=json, timeout=timeout)
        return _StubResponse(200, {"idToken": "firebase-token", "refreshToken": "refresh-1"})

    oauth = GoogleOAuthSignIn("API-KEY", client_secrets=secrets_file, post=_post, timeout=42)
    data = oauth.exchange_for_firebase("google-token-1")

    assert data["idToken"] == "firebase-token"
    assert captured["url"] == (
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key=API-KEY"
    )
    assert captured["payload"] == {
        "postBody": "id_token=google-token-1&providerId=google.com",
        "requestUri": "http://localhost",
        "returnSecureToken": True,
    }
    assert captured["timeout"] == 42


def test_exchange_surfaces_firebase_errors(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    oauth = GoogleOAuthSignIn(
        "API-KEY",
        client_secrets=secrets_file,
        post=lambda *a, **k: _StubResponse(400, text="INVALID_IDP_RESPONSE"),
    )
    with pytest.raises(GoogleOAuthError) as excinfo:
        oauth.exchange_for_firebase("bad")
    assert "400" in str(excinfo.value)
    assert "INVALID_IDP_RESPONSE" in str(excinfo.value)


def test_exchange_requires_a_session_token(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    oauth = GoogleOAuthSignIn(
        "API-KEY", client_secrets=secrets_file, post=lambda *a, **k: _StubResponse(200, {})
    )
    with pytest.raises(GoogleOAuthError, match="no session token"):
        oauth.exchange_for_firebase("good")


def test_google_id_token_without_a_credentials_file(tmp_path):
    oauth = GoogleOAuthSignIn("API-KEY", client_secrets=tmp_path / "missing.json")
    with pytest.raises(GoogleOAuthError, match="client_secrets"):
        oauth.google_id_token()


def test_sign_in_requires_an_api_key(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    oauth = GoogleOAuthSignIn("", client_secrets=secrets_file)
    assert oauth.is_available is False
    with pytest.raises(GoogleOAuthError, match="FIREBASE_API_KEY"):
        oauth.sign_in()


def test_describe_explains_what_is_missing(tmp_path, monkeypatch):
    assert "FIREBASE_API_KEY" in GoogleOAuthSignIn("", client_secrets=tmp_path / "x.json").describe()
    assert "not found" in GoogleOAuthSignIn("k", client_secrets=tmp_path / "x.json").describe()


async def test_firebase_auth_sign_in_with_google_oauth(tmp_path, monkeypatch):
    secrets_file = _write_client_secrets(tmp_path)
    lookup = httpx.MockTransport(handler=_lookup_handler(GOOGLE_LOOKUP))
    auth = FirebaseGoogleAuth(CONFIG, transport=lookup, opener=lambda url: None)

    async def _fake_sign_in(self):
        return {
            "idToken": "google-id-token",
            "refreshToken": "refresh-oauth",
        }

    monkeypatch.setattr(
        "agent.tui.onboarding.google_oauth.GoogleOAuthSignIn.sign_in_async",
        _fake_sign_in,
    )

    user = await auth.sign_in_with_google_oauth(client_secrets=secrets_file)

    assert user.email == "patil@gmail.com"
    assert user.name == "Patil"
    assert user.refresh_token == "refresh-oauth"
    assert user.id_token == "google-id-token"


async def test_firebase_auth_reports_missing_oauth_prerequisites(tmp_path):
    auth = FirebaseGoogleAuth(CONFIG, transport=httpx.MockTransport(handler=_lookup_handler(GOOGLE_LOOKUP)),
                              opener=lambda url: None)
    with pytest.raises(OnboardingError, match="unavailable"):
        await auth.sign_in_with_google_oauth(client_secrets=tmp_path / "missing.json")


async def test_welcome_screen_uses_oauth_when_available(store, tmp_path, monkeypatch):
    secrets_file = _write_client_secrets(tmp_path)
    monkeypatch.setenv("DEPRESSION_ONBOARDING", "force")
    called = {"oauth": 0, "page": 0}

    class _StubAuth(FirebaseGoogleAuth):
        async def sign_in_with_google_oauth(
            self, timeout=180.0, client_secrets=None, client_id=None, client_secret=None
        ):
            called["oauth"] += 1
            assert client_secrets == secrets_file
            return FirebaseUser(
                uid="uid-oauth", email="oauth@gmail.com", name="OAuth User",
                provider="google.com", id_token="tok", refresh_token="refresh-oauth",
            )

        async def sign_in_with_google(self, timeout=180.0):
            called["page"] += 1
            raise AssertionError("must not fall back to the browser page")

    app = _HostApp()
    async with app.run_test(size=(84, 30)) as pilot:
        app.push_screen(
            WelcomeScreen(
                config=CONFIG,
                store=store,
                auth=_StubAuth(CONFIG),
                countdown=0,
                enable_oauth=True,
                oauth=GoogleOAuthSignIn("api-key", client_secrets=secrets_file),
                env_path=tmp_path / "env",
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()
            if store.exists():
                break

    assert called == {"oauth": 1, "page": 0}
    assert store.load().email == "oauth@gmail.com"
    clear_identity(env_path=tmp_path / "env")


async def test_welcome_screen_falls_back_when_oauth_fails(store, tmp_path, monkeypatch):
    secrets_file = _write_client_secrets(tmp_path)
    monkeypatch.setenv("DEPRESSION_ONBOARDING", "force")
    called = {"page": 0}

    class _StubAuth(FirebaseGoogleAuth):
        async def sign_in_with_google_oauth(
            self, timeout=180.0, client_secrets=None, client_id=None, client_secret=None
        ):
            raise OnboardingError("redirect_uri_mismatch")

        async def sign_in_with_google(self, timeout=180.0):
            called["page"] += 1
            return FirebaseUser(
                uid="uid-page", email="page@gmail.com", name="Page User",
                provider="google.com", id_token="tok", refresh_token="refresh-page",
            )

    app = _HostApp()
    async with app.run_test(size=(84, 30)) as pilot:
        app.push_screen(
            WelcomeScreen(
                config=CONFIG,
                store=store,
                auth=_StubAuth(CONFIG),
                countdown=0,
                oauth=GoogleOAuthSignIn("api-key", client_secrets=secrets_file),
                env_path=tmp_path / "env",
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
            if store.exists():
                break

    assert called["page"] == 1
    assert store.load().email == "page@gmail.com"
    clear_identity(env_path=tmp_path / "env")


async def test_welcome_screen_ignores_oauth_when_disabled(store, tmp_path, monkeypatch):
    monkeypatch.setenv("DEPRESSION_ONBOARDING", "force")
    secrets_file = _write_client_secrets(tmp_path)
    called = {"page": 0}

    class _StubAuth(FirebaseGoogleAuth):
        async def sign_in_with_google_oauth(
            self, timeout=180.0, client_secrets=None, client_id=None, client_secret=None
        ):
            raise AssertionError("oauth must stay off")

        async def sign_in_with_google(self, timeout=180.0):
            called["page"] += 1
            return FirebaseUser(
                uid="uid-page", email="only-browser@gmail.com", name="Only Browser",
                provider="google.com", id_token="tok", refresh_token="refresh-page",
            )

    app = _HostApp()
    async with app.run_test(size=(84, 30)) as pilot:
        app.push_screen(
            WelcomeScreen(
                config=CONFIG,
                store=store,
                auth=_StubAuth(CONFIG),
                countdown=0,
                enable_oauth=False,
                oauth=GoogleOAuthSignIn("api-key", client_secrets=secrets_file),
                env_path=tmp_path / "env",
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(12):
            await pilot.pause()
            if store.exists():
                break

    assert called["page"] == 1
    assert store.load().email == "only-browser@gmail.com"
    clear_identity(env_path=tmp_path / "env")


# ======================================================================
# FIREBASE CONSOLE PROJECT-SETTINGS EXPORT
# ======================================================================

CONSOLE_EXPORT = {
    "project_number": "123456789012",
    "project_id": "demo-project-000000",
    "storage_bucket": "demo-project-000000.firebasestorage.app",
    "client": [
        {
            "mobilesdk_app_id": "1:123456789012:android:845660f7f0469d382ce9fd",
            "android_client_info": {"package_name": "com.example.demoapp"},
            "oauth_client": [
                {
                    "client_id": "123456789012-mlsjpd19kcf4t3sattpkn2qi8r2f7b43"
                    ".apps.googleusercontent.com",
                    "client_type": 1,
                },
                {
                    "client_id": "123456789012-vb1c874ngo36nrr2vp90hkcaihte4dh1"
                    ".apps.googleusercontent.com",
                    "client_type": 3,
                },
            ],
            "api_key": [{"current_key": "AIzaFAKE-NOT-A-REAL-KEY"}],
        }
    ],
}


def test_console_export_yields_api_key_and_web_oauth_client(tmp_path):
    (tmp_path / "firebase.json").write_text(json.dumps(CONSOLE_EXPORT))
    cfg = load_firebase_config(environ={}, config_dir=tmp_path)

    assert cfg.api_key == "AIzaFAKE-NOT-A-REAL-KEY"
    assert cfg.project_id == "demo-project-000000"
    assert cfg.project_number == "123456789012"
    assert cfg.storage_bucket == "demo-project-000000.firebasestorage.app"
    # The web client (client_type 3), never the Android one.
    assert cfg.oauth_client_id == (
        "123456789012-vb1c874ngo36nrr2vp90hkcaihte4dh1.apps.googleusercontent.com"
    )
    assert cfg.is_usable_for_oauth is True
    # No Web app registered -> the browser page cannot be used yet.
    assert cfg.is_usable_for_google is False
    assert cfg.has_web_app is False


def test_console_export_via_inline_json():
    cfg = load_firebase_config(environ={"FIREBASE_WEB_APP_CONFIG": json.dumps(CONSOLE_EXPORT)})
    assert cfg.oauth_client_id.endswith(".apps.googleusercontent.com")
    assert cfg.is_usable_for_oauth is True


def test_oauth_client_id_from_env_alone():
    cfg = load_firebase_config(environ={
        "FIREBASE_API_KEY": "AIza-x",
        "FIREBASE_OAUTH_CLIENT_ID": "abc.apps.googleusercontent.com",
    })
    assert cfg.oauth_client_id == "abc.apps.googleusercontent.com"
    assert cfg.is_usable_for_oauth is True


def test_has_web_app_detects_the_web_app_id():
    cfg = FirebaseConfig(api_key="k", app_id="1:123456789012:web:abc123")
    assert cfg.has_web_app is True


def test_oauth_flow_runs_from_a_bare_client_id(tmp_path, monkeypatch):
    """No client_secrets.json anywhere — just the console's web client id."""
    monkeypatch.delenv(ENV_CLIENT_SECRETS, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRETS_ALT, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    cfg = _config_from_mapping(CONSOLE_EXPORT)
    oauth = GoogleOAuthSignIn(cfg.api_key, client_id=cfg.oauth_client_id)
    assert oauth.is_available is True
    assert oauth.client_secrets is None

    seen = {}

    class _StubFlow:
        @classmethod
        def from_client_config(cls, client_config, scopes):
            seen["client_id"] = client_config["installed"]["client_id"]
            seen["redirect_uris"] = client_config["installed"]["redirect_uris"]
            seen["scopes"] = list(scopes)
            return cls()

        def run_local_server(self, port=0, **kwargs):
            seen["port"] = port
            return _StubCredentials("google-token-2")

    import agent.tui.onboarding.google_oauth as go

    original = go._import_installed_app_flow
    go._import_installed_app_flow = lambda: _StubFlow
    try:
        assert oauth.google_id_token() == "google-token-2"
    finally:
        go._import_installed_app_flow = original

    assert seen["client_id"] == cfg.oauth_client_id
    assert seen["redirect_uris"] == ["http://localhost"]
    assert seen["port"] == 0
    assert seen["scopes"][0] == "openid"


def test_client_secrets_file_wins_over_the_bare_client_id(tmp_path):
    secrets_file = _write_client_secrets(tmp_path)
    oauth = GoogleOAuthSignIn(
        "API-KEY",
        client_secrets=secrets_file,
        client_id="ignored.apps.googleusercontent.com",
        environ={},
    )
    assert oauth.client_secrets == secrets_file
    assert oauth.client_id == ""


def test_describe_mentions_both_ways_to_configure(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_CLIENT_SECRETS, raising=False)
    monkeypatch.delenv(ENV_CLIENT_SECRETS_ALT, raising=False)
    monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    text = GoogleOAuthSignIn("API-KEY", environ={}).describe()
    assert "client_secrets.json" in text
    assert "FIREBASE_OAUTH_CLIENT_ID" in text


# ======================================================================
# REGRESSIONS: the three bugs that broke real browser sign-in
# ======================================================================
#
# 1. The loopback server bound `127.0.0.1`. Firebase matches the page origin
#    against the project's authorized-domain list, which names `localhost`
#    and does *not* alias the literal IP. The popup therefore stalled on
#    `__/auth/handler` and never reached Google's login page.
# 2. `_OK_PAGE` / `_ERROR_PAGE` embed raw CSS braces, so `.format(message=…)`
#    raised `KeyError: 'background'` — on the *success* path too — leaving the
#    browser's fetch() hanging forever.
# 3. The redirect-resume block guarded on `auth._getRedirectResult`, a private
#    method the compat SDK does not expose, so the whole block was dead code
#    and the popup-blocked fallback could never resume.

from agent.tui.onboarding.firebase_auth import (  # noqa: E402
    CALLBACK_PATH,
    _CallbackServer,
    _ERROR_PAGE,
    _OK_PAGE,
)


def test_callback_server_binds_localhost_not_the_literal_ip():
    """Firebase's authorized domains name `localhost`, never `127.0.0.1`."""
    server = _CallbackServer("<html></html>", "tok")
    try:
        assert server._host == "localhost"
        assert server._host != "127.0.0.1"
        # `localhost` must still resolve to a loopback-only socket.
        server.start()
        host = server._server.server_address[0]
        assert host in ("127.0.0.1", "::1"), host
    finally:
        server.stop()


def test_served_urls_use_localhost_so_firebase_accepts_the_origin():
    client = FirebaseGoogleAuth(_web_config(), opener=lambda url: None)
    server, _token, url = client.start_callback_server()
    try:
        assert url.startswith("http://localhost:")
        assert "127.0.0.1" not in url
        assert "127.0.0.1" not in server.callback_url
        assert server.callback_url.startswith("http://localhost:")
        assert server.callback_url.endswith(CALLBACK_PATH)
    finally:
        server.stop()


def test_result_pages_format_without_raising():
    """Regression: raw CSS braces used to raise KeyError on both templates."""
    assert "Signed in" in _OK_PAGE.format()
    out = _ERROR_PAGE.format(message="popup blocked")
    assert "popup blocked" in out
    # The doubled braces must collapse back to real CSS, not leak "{{".
    assert "{{" not in out and "}}" not in out
    assert "background:#000" in out


def test_success_page_is_served_verbatim(tmp_path):
    """The POST that carries a valid token must get a complete 200 response."""
    import httpx as _httpx

    client = FirebaseGoogleAuth(_web_config(), opener=lambda url: None)
    server, token, url = client.start_callback_server()
    try:
        response = _httpx.post(
            url.rstrip("/") + CALLBACK_PATH,
            json={"state": token, "idToken": "tok"},
            timeout=10,
        )
        assert response.status_code == 200
        assert "Signed in" in response.text
        assert server.wait_for_payload(5)["idToken"] == "tok"
    finally:
        server.stop()


def test_error_page_is_served_verbatim(tmp_path):
    import httpx as _httpx

    client = FirebaseGoogleAuth(_web_config(), opener=lambda url: None)
    server, token, url = client.start_callback_server()
    try:
        response = _httpx.post(
            url.rstrip("/") + CALLBACK_PATH,
            json={"state": token, "error": "auth/popup-blocked"},
            timeout=10,
        )
        assert response.status_code == 200
        assert "auth/popup-blocked" in response.text
        payload = server.wait_for_payload(5)
        assert payload.get("error") == "auth/popup-blocked"
    finally:
        server.stop()


def test_sign_in_page_does_not_depend_on_a_private_sdk_method():
    """Regression: `auth._getRedirectResult` is absent in the compat SDK."""
    html = render_sign_in_page(_web_config(), "http://localhost:1234/callback", "S")
    # No *call* to the private method may survive (prose mentioning it is fine).
    assert "auth._getRedirectResult" not in html
    assert 'typeof auth.getRedirectResult !== "function"' in html
    assert "auth.getRedirectResult()" in html


def test_sign_in_page_reports_a_failed_redirect_instead_of_hanging():
    """A throwing signInWithRedirect / failing resume must post an error."""
    html = render_sign_in_page(_web_config(), "http://localhost:1234/callback", "S")
    assert "catch (redirectErr)" in html
    assert "await fail(redirectErr)" in html
    # fail() itself posts, so the CLI is told rather than left waiting.
    assert "async function fail(err, report)" in html
    assert "await post({ error: message })" in html


def test_sign_in_page_handles_a_redirect_result_with_no_user():
    """getRedirectResult() can return a null user; that must not hang the CLI."""
    html = render_sign_in_page(_web_config(), "http://localhost:1234/callback", "S")
    assert "const user = result && result.user;" in html
    assert "if (!user) {" in html
    assert "auth/no-user" in html


def _web_config() -> FirebaseConfig:
    return FirebaseConfig(
        api_key="AIzaTest",
        auth_domain="depression-571e0.firebaseapp.com",
        project_id="depression-571e0",
        app_id="1:952406227868:web:44b56e6f1dbdec4d20e659",
    )
