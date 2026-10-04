"""
Firebase web-app configuration for Gmail (Google) sign-in.

Nothing secret lives in this module. A Firebase *web* config
(apiKey / authDomain / projectId / appId) is public by design — it ships
inside every web app that uses Firebase, and Firebase treats it as an
identifier, not a credential. What must never be committed to this repo
is a service-account JSON key.

Two ways in
------------
* The **browser page** needs a Firebase *Web app* (apiKey / authDomain /
  projectId / appId) because it boots the web SDK.
* The **google-auth-oauthlib flow** needs only the API key and a Google OAuth
  *Web client id* (`client_type: 3` in the console) — no Web app required.

Both are read from the same config, and the raw **Firebase Console >
Project settings > General export** is understood as-is, including its list
shapes:

    {
      "project_id": "my-project",
      "project_number": "605436333256",
      "storage_bucket": "my-project.firebasestorage.app",
      "api_key": [{ "current_key": "AIza…" }],
      "oauth_client": [
        { "client_id": "…android…", "client_type": 1 },
        { "client_id": "….apps.googleusercontent.com", "client_type": 3 }
      ]
    }

The `client_type: 3` entry is the web client used by the OAuth flow.

Resolution order (first hit wins):

    1. an explicit mapping passed by the caller
    2. FIREBASE_WEB_APP_CONFIG / DEPRESSION_FIREBASE_WEB_APP_CONFIG (inline JSON)
    3. FIREBASE_CONFIG_FILE / DEPRESSION_FIREBASE_CONFIG_FILE (path to a JSON file)
    4. individual env vars: FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN,
       FIREBASE_PROJECT_ID, FIREBASE_APP_ID, FIREBASE_MEASUREMENT_ID,
       FIREBASE_OAUTH_CLIENT_ID, FIREBASE_OAUTH_CLIENT_SECRET
    5. <config_dir>/firebase.json

Both camelCase (Firebase's own spelling) and snake_case keys are accepted,
and the file may either be the raw web config or wrap it:

    { "web": { "apiKey": "..." } }
    { "webAppConfig": { "apiKey": "..." } }
    { "api_key": "..." }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent.utils.logging import get_logger
from agent.utils.platform import get_config_dir

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

APP_NAME = "depression"

# Firebase compat SDK, served from Google's CDN. The web app config is
# embedded in the served page, exactly like a normal browser login.
FIREBASE_SDK_VERSION = "9.23.0"
FIREBASE_SDK_BASE = f"https://www.gstatic.com/firebasejs/{FIREBASE_SDK_VERSION}"

CONFIG_FILENAME = "firebase.json"

ENV_INLINE_JSON = (
    "DEPRESSION_FIREBASE_WEB_APP_CONFIG",
    "FIREBASE_WEB_APP_CONFIG",
)
ENV_CONFIG_FILE = (
    "DEPRESSION_FIREBASE_CONFIG_FILE",
    "FIREBASE_CONFIG_FILE",
)

ENV_KEYS: Dict[str, tuple] = {
    "api_key": ("DEPRESSION_FIREBASE_API_KEY", "FIREBASE_API_KEY"),
    "auth_domain": ("DEPRESSION_FIREBASE_AUTH_DOMAIN", "FIREBASE_AUTH_DOMAIN"),
    "project_id": ("DEPRESSION_FIREBASE_PROJECT_ID", "FIREBASE_PROJECT_ID"),
    "app_id": ("DEPRESSION_FIREBASE_APP_ID", "FIREBASE_APP_ID"),
    "measurement_id": (
        "DEPRESSION_FIREBASE_MEASUREMENT_ID",
        "FIREBASE_MEASUREMENT_ID",
    ),
    "project_number": ("FIREBASE_PROJECT_NUMBER",),
    "storage_bucket": ("FIREBASE_STORAGE_BUCKET",),
    "oauth_client_id": (
        "DEPRESSION_FIREBASE_OAUTH_CLIENT_ID",
        "FIREBASE_OAUTH_CLIENT_ID",
    ),
    "oauth_client_secret": (
        "DEPRESSION_FIREBASE_OAUTH_CLIENT_SECRET",
        "FIREBASE_OAUTH_CLIENT_SECRET",
    ),
}

# Accepted spellings for every field, in priority order.
FIELD_ALIASES: Dict[str, tuple] = {
    "api_key": ("apiKey", "api_key", "apikey", "current_key", "key"),
    "auth_domain": ("authDomain", "auth_domain", "authdomain"),
    "project_id": ("projectId", "project_id", "projectid"),
    "app_id": ("appId", "app_id", "appid"),
    "measurement_id": ("measurementId", "measurement_id", "measurementid"),
    "project_number": ("projectNumber", "project_number", "projectnumber"),
    "storage_bucket": ("storageBucket", "storage_bucket", "storagebucket"),
    "oauth_client_id": (
        "oauthClientId",
        "oauth_client_id",
        "oauth_client",  # console export: a *list* of client entries
        "oauthClientID",
        "clientId",
        "client_id",
    ),
    "oauth_client_secret": (
        "oauthClientSecret",
        "oauth_client_secret",
        "clientSecret",
        "client_secret",
    ),
}

# Keys whose value is a list in the console export.
_LIST_KEYS = ("api_key", "oauth_client")

# Firebase OAuth client_type -> usable by the CLI.
OAUTH_CLIENT_TYPE_WEB = 3

_WRAPPERS = ("web", "webAppConfig", "web_app_config", "firebase", "config")


# ======================================================================
# CONFIG
# ======================================================================

@dataclass(frozen=True)
class FirebaseConfig:
    """Resolved Firebase web-app config."""

    api_key: str = ""
    auth_domain: str = ""
    project_id: str = ""
    app_id: str = ""
    measurement_id: str = ""
    project_number: str = ""
    storage_bucket: str = ""
    oauth_client_id: str = ""
    oauth_client_secret: str = ""

    @property
    def is_configured(self) -> bool:
        """True when there is enough to talk to Firebase Auth."""
        return bool(self.api_key)

    @property
    def is_usable_for_google(self) -> bool:
        """The browser page also needs an authDomain to host the popup/iframe."""
        return self.is_configured and bool(self.auth_domain)

    @property
    def is_usable_for_oauth(self) -> bool:
        """The oauthlib flow needs the API key plus a Google OAuth client id."""
        return self.is_configured and bool(self.oauth_client_id)

    @property
    def has_web_app(self) -> bool:
        """True when the project has a *Web* app (the browser page needs one)."""
        return ":web:" in self.app_id or self.app_id.startswith("1:")

    def to_web_config(self) -> Dict[str, str]:
        """The `firebase.initializeApp({...})` payload."""
        web: Dict[str, str] = {}
        if self.api_key:
            web["apiKey"] = self.api_key
        if self.auth_domain:
            web["authDomain"] = self.auth_domain
        if self.project_id:
            web["projectId"] = self.project_id
        if self.app_id:
            web["appId"] = self.app_id
        if self.measurement_id:
            web["measurementId"] = self.measurement_id
        return web

    def to_dict(self) -> Dict[str, str]:
        return {
            "api_key": self.api_key,
            "auth_domain": self.auth_domain,
            "project_id": self.project_id,
            "app_id": self.app_id,
            "measurement_id": self.measurement_id,
            "project_number": self.project_number,
            "storage_bucket": self.storage_bucket,
            "oauth_client_id": self.oauth_client_id,
            "oauth_client_secret": self.oauth_client_secret,
        }

    def describe(self) -> str:
        """Log-safe one-liner (the API key is masked)."""
        if not self.is_configured:
            return "firebase: not configured"
        masked = f"{self.api_key[:6]}…{self.api_key[-4:]}" if len(self.api_key) > 12 else "…"
        return (
            f"firebase: project={self.project_id or '?'} "
            f"auth_domain={self.auth_domain or '?'} api_key={masked} "
            f"oauth_client={'yes' if self.oauth_client_id else 'no'}"
        )


EMPTY_CONFIG = FirebaseConfig()


# ======================================================================
# PARSING
# ======================================================================

def _unwrap(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap `{"web": {...}}` style containers."""
    for key in _WRAPPERS:
        inner = data.get(key)
        if isinstance(inner, Mapping):
            return inner
    return data


def _pick(data: Mapping[str, Any], field_name: str) -> str:
    for alias in FIELD_ALIASES[field_name]:
        value = data.get(alias)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _pick_list_entry(data: Mapping[str, Any], field_name: str) -> str:
    """
    Read a field from the console export's list shapes.

    `"api_key": [{ "current_key": "AIza…" }]` and
    `"oauth_client": [{ "client_id": "…", "client_type": 3 }, …]`.
    """
    for alias in FIELD_ALIASES[field_name]:
        entries = data.get(alias)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            if field_name == "oauth_client_id" and entry.get("client_type") not in (
                None,
                OAUTH_CLIENT_TYPE_WEB,
            ):
                # Android/iOS clients cannot serve a browser consent screen.
                continue
            nested = entry.get(alias)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
            value = _pick(entry, field_name)
            if value:
                return value
    return ""


def _flatten_console_export(data: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Pull the interesting fields out of the console's project-settings export,
    where everything hides under `client: [{ … }]`:

        {
          "project_id": "my-project",
          "client": [{
            "mobilesdk_app_id": "1:…:android:…",
            "api_key": [{"current_key": "AIza…"}],
            "oauth_client": [{"client_id": "…", "client_type": 3}]
          }]
        }

    Existing top-level values always win; nothing is overwritten.
    """
    flat: Dict[str, Any] = {key: value for key, value in data.items()}

    clients = data.get("client")
    entries = clients if isinstance(clients, list) else [clients]
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        for key, value in entry.items():
            if key in flat and isinstance(flat[key], list) and isinstance(value, list):
                flat[key] = list(flat[key]) + list(value)
            elif flat.get(key) in (None, "", {}):
                flat[key] = value
    return flat


def _config_from_mapping(data: Mapping[str, Any]) -> FirebaseConfig:
    inner = _flatten_console_export(_unwrap(data))
    return FirebaseConfig(
        api_key=_pick(inner, "api_key") or _pick_list_entry(inner, "api_key"),
        auth_domain=_pick(inner, "auth_domain"),
        project_id=_pick(inner, "project_id"),
        app_id=_pick(inner, "app_id"),
        measurement_id=_pick(inner, "measurement_id"),
        project_number=_pick(inner, "project_number"),
        storage_bucket=_pick(inner, "storage_bucket"),
        oauth_client_id=(
            _pick(inner, "oauth_client_id") or _pick_list_entry(inner, "oauth_client_id")
        ),
        oauth_client_secret=_pick(inner, "oauth_client_secret"),
    )


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.debug("Firebase config file unreadable (%s): %s", path, exc)
        return None
    try:
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("Firebase config file is not valid JSON (%s): %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def load_firebase_config(
    explicit: Optional[Mapping[str, Any]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    config_dir: Optional[Path] = None,
) -> FirebaseConfig:
    """
    Resolve the Firebase web config from the environment / disk.

    Never raises: a missing or broken config simply yields EMPTY_CONFIG so
    the caller can fall back to the offline path.
    """
    env = os.environ if environ is None else environ

    if explicit:
        cfg = _config_from_mapping(explicit)
        if cfg.is_configured:
            return cfg

    for name in ENV_INLINE_JSON:
        inline = (env.get(name) or "").strip()
        if not inline:
            continue
        try:
            data = json.loads(inline)
        except Exception as exc:
            logger.warning("%s is not valid JSON: %s", name, exc)
            continue
        if isinstance(data, dict):
            cfg = _config_from_mapping(data)
            if cfg.is_configured:
                return cfg

    for name in ENV_CONFIG_FILE:
        path_str = (env.get(name) or "").strip()
        if not path_str:
            continue
        data = _read_json_file(Path(os.path.expanduser(path_str)))
        if data:
            cfg = _config_from_mapping(data)
            if cfg.is_configured:
                return cfg

    env_values = {
        field: next(
            (str(env[name]).strip() for name in names if (env.get(name) or "").strip()),
            "",
        )
        for field, names in ENV_KEYS.items()
    }
    cfg = FirebaseConfig(**env_values)
    if cfg.is_configured:
        return cfg

    directory = Path(config_dir) if config_dir is not None else get_config_dir(APP_NAME)
    data = _read_json_file(directory / CONFIG_FILENAME)
    if data:
        return _config_from_mapping(data)

    return EMPTY_CONFIG


__all__ = [
    "APP_NAME",
    "CONFIG_FILENAME",
    "EMPTY_CONFIG",
    "FIREBASE_SDK_BASE",
    "FIREBASE_SDK_VERSION",
    "FirebaseConfig",
    "load_firebase_config",
]