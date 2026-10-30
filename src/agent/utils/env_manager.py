"""Environment file management for Depression.AI.

Handles reading/writing .env files for persistent credential storage.

Layout:
    <project>/.env          <- canonical LLM, AWS and MCP credentials
    ~/.agent/env            <- legacy credential fallback + signed-in identity

All credentials are saved to the project .env (mode 0600). Values in
~/.agent/env remain readable as a fallback for existing installs, but the
project .env always wins.

Public API:
    env_manager = EnvManager.get()
    env_manager.get_aws_env() -> dict       # canonical AWS env for subprocesses
    env_manager.subscribe(cb)               # notified on change

Credential writes are centralized in ``agent.utils.credential_store``.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional


ENV_FILE_NAME = ".env"
GLOBAL_ENV_DIR = Path.home() / ".agent"
GLOBAL_ENV_PATH = GLOBAL_ENV_DIR / "env"


# Provider to env var mappings
PROVIDER_API_KEY_ENVS = {
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zai": "ZAI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "meta": "META_API_KEY",
    "alibaba": "DASHSCOPE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_BASE_URL_ENVS = {
    "nvidia": "NVIDIA_BASE_URL",
    "deepseek": "DEEPSEEK_BASE_URL",
    "minimax": "MINIMAX_BASE_URL",
    "mistral": "MISTRAL_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "google": "GOOGLE_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "groq": "GROQ_BASE_URL",
    "moonshot": "MOONSHOT_BASE_URL",
    "zai": "ZAI_BASE_URL",
    "qwen": "QWEN_BASE_URL",
    "meta": "META_BASE_URL",
    "alibaba": "ALIBABA_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
}

AWS_ENVS = {
    "access_key": "AWS_ACCESS_KEY_ID",
    "secret_key": "AWS_SECRET_ACCESS_KEY",
    "region": "AWS_DEFAULT_REGION",
    "session_token": "AWS_SESSION_TOKEN",
}

SELECTED_MODEL_ENV = "DEPRESSION_SELECTED_MODEL"


# ======================================================================
# Low-level file helpers (kept for backward compat)
# ======================================================================

def default_env_path() -> Path:
    """Current-working-directory env file, evaluated at call time (not import)."""
    return Path.cwd() / ENV_FILE_NAME


def find_env_file(start_path: Optional[Path] = None) -> Path:
    """Find the nearest .env file walking up from start_path."""
    path = start_path or Path.cwd()
    for parent in [path, *path.parents]:
        env_file = parent / ENV_FILE_NAME
        if env_file.exists():
            return env_file
    return default_env_path()


def credential_env_path() -> Path:
    """Canonical credential store: the nearest project .env file."""
    return find_env_file()


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Load environment variables from a .env file."""
    path = env_path or find_env_file()
    result: Dict[str, str] = {}

    if not path.exists():
        return result

    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    except Exception:
        pass

    return result


def write_private_file(path: Path, content: str, mode: int = 0o600) -> None:
    """Write ``content`` to ``path`` so it is never briefly world-readable.

    ``write_text()`` then ``chmod()`` creates the file under the process
    umask first, so a credential file is observable as ``0644`` for a window.
    Writing to a ``0600`` temp file in the same directory and renaming it into
    place makes the secret unreadable to other users for its whole lifetime.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def write_env_file(
    env_vars: Dict[str, str],
    env_path: Optional[Path] = None,
    strict: bool = False,
) -> None:
    """Write environment variables to a .env file.

    Args:
        env_vars: keys to merge over whatever is already on disk.
        env_path: target file, defaults to ``find_env_file()``.
        strict: when True, filesystem failures raise ``OSError`` instead of
            being swallowed. Credential writes pass ``strict=True`` so the
            UI can report a failed save rather than reporting success while
            the old value silently remains on disk.
    """
    path = env_path or find_env_file()

    existing = load_env_file(path)
    existing.update(env_vars)

    lines = []
    for key, value in sorted(existing.items()):
        lines.append(f"{key}={value}")

    try:
        write_private_file(path, "\n".join(lines) + "\n", mode=0o600)
    except OSError:
        if strict:
            raise
    except Exception:
        if strict:
            raise


# ======================================================================
# Singleton EnvManager — the canonical source of truth
# ======================================================================

class EnvManager:
    """
    Process-wide environment manager.

    Responsibilities:
        * load persisted vars (project .env, with legacy ~/.agent/env fallback)
        * mirror them into os.environ so subprocesses see them
        * expose get_aws_env() as the one canonical AWS credential source
        * notify subscribers on change so running agents/TUI can reload
    """

    _instance: Optional["EnvManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: List[Callable[[Dict[str, str]], None]] = []
        self._generation = 0
        self._aws_cache: Dict[str, Optional[str]] = {}
        self._load_initial()

    # -- singleton --------------------------------------------------------

    @classmethod
    def get(cls) -> "EnvManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = EnvManager()
            return cls._instance

    # -- properties -------------------------------------------------------

    @property
    def generation(self) -> int:
        """Bumped on every change. Consumers can compare to detect updates."""
        return self._generation

    # -- loading ----------------------------------------------------------

    def _load_initial(self) -> None:
        """Merge legacy global + project env, then mirror into os.environ."""
        merged: Dict[str, str] = {}
        # The project .env is the canonical credential store. The legacy
        # user-scoped file remains a fallback for existing installs.
        merged.update(load_env_file(GLOBAL_ENV_PATH))
        merged.update(load_env_file(find_env_file()))

        for k, v in merged.items():
            os.environ.setdefault(k, v)

        self._aws_cache = self._read_aws_from_disk()

    def reload(self) -> None:
        """Re-read files, refresh os.environ, bump generation, notify."""
        self._load_initial()
        self._aws_cache = self._read_aws_from_disk()
        self._generation += 1
        self._notify()

    # -- subscribers ------------------------------------------------------

    def subscribe(self, cb: Callable[[Dict[str, str]], None]) -> None:
        self._subscribers.append(cb)

    def unsubscribe(self, cb: Callable[[Dict[str, str]], None]) -> None:
        try:
            self._subscribers.remove(cb)
        except ValueError:
            pass

    def _notify(self) -> None:
        snapshot = dict(self.get_aws_env())
        for cb in list(self._subscribers):
            try:
                cb(snapshot)
            except Exception:
                pass

    # -- AWS: canonical source --------------------------------------------

    def _read_aws_from_disk(self) -> Dict[str, Optional[str]]:
        """Prefer the project .env; fall back to the legacy global store."""
        creds: Dict[str, Optional[str]] = {
            "access_key": None,
            "secret_key": None,
            "region": None,
            "session_token": None,
        }

        for source in (find_env_file(), GLOBAL_ENV_PATH):
            data = load_env_file(source)
            if data.get(AWS_ENVS["access_key"]) and data.get(AWS_ENVS["secret_key"]):
                creds["access_key"] = data.get(AWS_ENVS["access_key"])
                creds["secret_key"] = data.get(AWS_ENVS["secret_key"])
                creds["region"] = data.get(AWS_ENVS["region"]) or "us-east-1"
                creds["session_token"] = data.get(AWS_ENVS["session_token"])
                return creds

        return creds

    def get_aws_credentials(self) -> Dict[str, Optional[str]]:
        """Return the current AWS credentials (access_key, secret_key, region, session_token)."""
        # Always re-read in case something outside changed the file.
        self._aws_cache = self._read_aws_from_disk()
        return dict(self._aws_cache)

    def get_aws_env(self) -> Dict[str, str]:
        """
        Canonical AWS env vars for subprocesses.
        This is what aws_helper, boto3 factories, and MCP launchers MUST use.
        """
        creds = self.get_aws_credentials()
        out: Dict[str, str] = {}
        if creds.get("access_key"):
            out["AWS_ACCESS_KEY_ID"] = creds["access_key"]
        if creds.get("secret_key"):
            out["AWS_SECRET_ACCESS_KEY"] = creds["secret_key"]
        if creds.get("region"):
            out["AWS_DEFAULT_REGION"] = creds["region"]
        if creds.get("session_token"):
            out["AWS_SESSION_TOKEN"] = creds["session_token"]
        return out

    def save_aws_credentials(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> None:
        """Persist AWS credentials through the single credential bridge."""
        from agent.utils.credential_store import save_aws_credentials

        save_aws_credentials(
            access_key=access_key,
            secret_key=secret_key,
            region=region,
            session_token=session_token,
        )

    # -- provider credentials (private user store by default) -------------

    def set_provider_credentials(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        from agent.utils.credential_store import save_credentials

        updates: Dict[str, Optional[str]] = {}
        if api_key is not None:
            key_env = PROVIDER_API_KEY_ENVS.get(provider.lower())
            if key_env:
                updates[key_env] = api_key
        if base_url is not None:
            url_env = PROVIDER_BASE_URL_ENVS.get(provider.lower())
            if url_env:
                updates[url_env] = base_url.rstrip("/")

        if updates:
            save_credentials(updates, env_path=env_path, remove_empty=True)

    def get_provider_credentials(
        self, provider: str, env_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        key_name = PROVIDER_API_KEY_ENVS.get(provider.lower(), "")
        url_name = PROVIDER_BASE_URL_ENVS.get(provider.lower(), "")
        return {
            "api_key": get_env_var(key_name, env_path),
            "base_url": get_env_var(url_name, env_path),
        }

    def set_selected_model(self, model: str, env_path: Optional[Path] = None) -> None:
        set_env_var(SELECTED_MODEL_ENV, model, env_path)
        os.environ[SELECTED_MODEL_ENV] = model

    def get_selected_model(self, env_path: Optional[Path] = None) -> Optional[str]:
        return get_env_var(SELECTED_MODEL_ENV, env_path)


# ======================================================================
# Module-level helpers — thin wrappers over the singleton for compat
# ======================================================================

def set_env_var(key: str, value: str, env_path: Optional[Path] = None) -> None:
    """Save a user-managed value through the single credential bridge."""
    from agent.utils.credential_store import save_credentials

    save_credentials({key: value}, env_path=env_path)


def get_env_var(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Use the project .env first; fall back to the legacy user store."""
    if env_path is not None:
        return load_env_file(env_path).get(key)
    project_data = load_env_file(find_env_file())
    if key in project_data:
        return project_data[key]
    return load_env_file(GLOBAL_ENV_PATH).get(key)


def set_provider_credentials(
    provider: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    env_path: Optional[Path] = None,
) -> None:
    EnvManager.get().set_provider_credentials(provider, api_key, base_url, env_path)


def get_provider_credentials(
    provider: str, env_path: Optional[Path] = None
) -> Dict[str, Optional[str]]:
    return EnvManager.get().get_provider_credentials(provider, env_path)


def set_selected_model(model: str, env_path: Optional[Path] = None) -> None:
    EnvManager.get().set_selected_model(model, env_path)


def get_selected_model(env_path: Optional[Path] = None) -> Optional[str]:
    return EnvManager.get().get_selected_model(env_path)


def set_aws_credentials(
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = None,
    env_path: Optional[Path] = None,  # kept for API compatibility; user store wins
    session_token: Optional[str] = None,
) -> None:
    from agent.utils.credential_store import save_aws_credentials

    save_aws_credentials(
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        session_token=session_token,
    )


def get_aws_credentials(env_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    return EnvManager.get().get_aws_credentials()


def get_aws_env() -> Dict[str, str]:
    """Canonical AWS env for subprocesses. Use this everywhere."""
    return EnvManager.get().get_aws_env()


def load_into_os_environ(env_path: Optional[Path] = None) -> None:
    """Load project .env values, with the legacy user store as fallback."""
    if env_path is not None:
        env_vars = load_env_file(env_path)
    else:
        env_vars = load_env_file(GLOBAL_ENV_PATH)
        env_vars.update(load_env_file(find_env_file()))
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


# Auto-load on import
EnvManager.get()