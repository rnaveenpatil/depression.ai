"""
First-run user profile store.

Answers one question for the rest of the app: *who is using this install?*

The profile is written once, when the user signs in with Gmail on the
welcome page, and simply read on later launches. Location matches the
agent's own data dir (`get_data_dir("depression")`):

    Linux   ~/.local/share/depression/user.json
    macOS   ~/Library/Application Support/depression/user.json
    Windows %APPDATA%\\depression\\user.json

`user.json` holds the public identity (uid, name, email, photo, counters).
Tokens never go there — they live in `user_tokens.json`, written 0600.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from agent.utils.logging import get_logger
from agent.utils.platform import get_data_dir

logger = get_logger(__name__)


APP_NAME = "depression"
PROFILE_FILENAME = "user.json"
TOKENS_FILENAME = "user_tokens.json"

FILE_MODE = 0o600
DIR_MODE = 0o700


# ======================================================================
# PROFILE
# ======================================================================

@dataclass
class UserProfile:
    """Who is signed in on this machine."""

    uid: str
    email: str = ""
    name: str = ""
    photo_url: str = ""
    provider: str = "google.com"
    email_verified: bool = False
    signed_in: bool = True
    created_at: float = field(default_factory=time.time)
    last_login_at: float = field(default_factory=time.time)
    login_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.name or self.email or (self.uid[:12] if self.uid else "user")

    @property
    def is_gmail(self) -> bool:
        return self.email.lower().endswith("@gmail.com")

    @property
    def initials(self) -> str:
        source = self.display_name.strip()
        parts = [p for p in source.replace("_", " ").split() if p]
        if not parts:
            return "?"
        if len(parts) == 1:
            return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in (data or {}).items() if k in known}
        payload.setdefault("uid", "")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            payload["metadata"] = {}
        return cls(**payload)


def guest_profile(*_args: Any, **_kwargs: Any) -> UserProfile:
    """
    Identity for a local install that has not signed in with Google.

    This is a *real*, savable profile — not an error. It marks the install
    as owned by a local guest (``uid == "local-guest"``) while remaining
    explicitly signed out, so the first-run gate does not reappear and no
    Gmail-specific features (mail, calendar) are enabled for it.
    """
    return UserProfile(
        uid="local-guest",
        email="",
        name="Local user",
        photo_url="",
        provider="local",
        email_verified=False,
        signed_in=False,
        metadata={"local": True, "gmail": False},
    )


# ======================================================================
# STORE
# ======================================================================

class ProfileStore:
    """Reads/writes the on-disk identity for this install."""

    def __init__(self, data_dir: Optional[Path] = None, app_name: str = APP_NAME):
        self.app_name = app_name
        self._data_dir = Path(data_dir) if data_dir is not None else None

    # ------------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------------

    @property
    def data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        return get_data_dir(self.app_name)

    @property
    def profile_path(self) -> Path:
        return self.data_dir / PROFILE_FILENAME

    @property
    def token_path(self) -> Path:
        return self.data_dir / TOKENS_FILENAME

    # ------------------------------------------------------------------
    # FIRST-RUN DETECTION
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        return self.profile_path.is_file()

    def is_new_user(self) -> bool:
        return self.load() is None

    def load(self) -> Optional[UserProfile]:
        if not self.profile_path.is_file():
            return None
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("user.json is unreadable, treating as new user: %s", exc)
            return None
        if not isinstance(data, dict):
            return None
        profile = UserProfile.from_dict(data)
        if not profile.uid:
            logger.warning("user.json has no uid, treating as new user")
            return None
        return profile

    # ------------------------------------------------------------------
    # WRITES
    # ------------------------------------------------------------------

    def save(self, profile: UserProfile) -> UserProfile:
        """
        Persist the profile.

        `login_count` only increments when a *signed-in* profile replaces an
        existing profile for the same uid (i.e. a real re-login). Frequent
        saves from the same session don't inflate the counter.
        """
        existing = self.load()
        if existing is not None and existing.uid == profile.uid:
            profile.created_at = existing.created_at
            if profile.signed_in:
                profile.login_count = existing.login_count + 1
            else:
                profile.login_count = existing.login_count
            merged = dict(existing.metadata or {})
            merged.update(profile.metadata or {})
            profile.metadata = merged
        profile.last_login_at = time.time()

        _atomic_write_json(self.profile_path, profile.to_dict())
        logger.info(
            "Saved local profile for %s (%s)",
            profile.display_name, profile.provider,
        )
        return profile

    def delete(self) -> None:
        for path in (self.profile_path, self.token_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning("Could not remove %s: %s", path, exc)

    # ------------------------------------------------------------------
    # TOKENS
    # ------------------------------------------------------------------

    def load_tokens(self) -> Dict[str, Any]:
        if not self.token_path.is_file():
            return {}
        try:
            data = json.loads(self.token_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("user_tokens.json is unreadable: %s", exc)
            return {}
        return data if isinstance(data, dict) else {}

    def save_tokens(self, tokens: Dict[str, Any]) -> None:
        _atomic_write_json(self.token_path, dict(tokens or {}))

    def delete_tokens(self) -> None:
        try:
            self.token_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Could not remove %s: %s", self.token_path, exc)

    def sign_out(self) -> None:
        self.delete()


# ======================================================================
# HELPERS
# ======================================================================

def _atomic_write_json(path: Path, data: Dict[str, Any], mode: int = FILE_MODE) -> None:
    """
    Write JSON via a temp file + rename, owner-only.

    Directory mode is applied to the file's parent ONLY if the parent is
    not $HOME itself. This avoids accidentally chmod-ing the user's home.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Only tighten the parent dir if it's not the user's home / root.
    try:
        parent = path.parent.resolve()
        if parent not in (Path.home().resolve(), Path("/"), Path("/tmp")):
            current = stat.S_IMODE(parent.stat().st_mode)
            # Only tighten; never loosen.
            if (current & DIR_MODE) != DIR_MODE and current != 0o755:
                try:
                    os.chmod(parent, DIR_MODE)
                except Exception:
                    pass
    except Exception:
        pass

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.write("\n")
        try:
            os.chmod(tmp, mode)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
        raise


def profile_mode(path: Path) -> Optional[int]:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except Exception:
        return None


__all__ = [
    "APP_NAME",
    "DIR_MODE",
    "FILE_MODE",
    "PROFILE_FILENAME",
    "TOKENS_FILENAME",
    "ProfileStore",
    "UserProfile",
    "profile_mode",
]