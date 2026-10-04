"""Environment file management for Depression.AI.

Handles reading/writing .env files for persistent credential storage.

Layout:
    ~/.agent/env            <- GLOBAL, user-scoped (AWS creds, global toggles)
    <project>/.env          <- PROJECT-scoped (project vars only, never creds)

Public API:
    env_manager = EnvManager.get()
    env_manager.save_aws_credentials(...)   # updates os.environ + notifies
    env_manager.get_aws_env() -> dict       # canonical AWS env for subprocesses
    env_manager.subscribe(cb)               # notified on change
"""
from __future__ import annotations

import os
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


def write_env_file(env_vars: Dict[str, str], env_path: Optional[Path] = None) -> None:
    """Write environment variables to a .env file."""
    path = env_path or find_env_file()

    existing = load_env_file(path)
    existing.update(env_vars)

    lines = []
    for key, value in sorted(existing.items()):
        lines.append(f"{key}={value}")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except Exception:
        pass


# ======================================================================
# Singleton EnvManager — the canonical source of truth
# ======================================================================

class EnvManager:
    """
    Process-wide environment manager.

    Responsibilities:
        * load persisted vars (global ~/.agent/env + project .env)
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
        """Merge global + project env, then mirror into os.environ."""
        merged: Dict[str, str] = {}
        # Global first (lower priority)
        merged.update(load_env_file(GLOBAL_ENV_PATH))
        # Project overrides for non-secret vars (but NOT AWS — see get_aws_env)
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
        """
        Prefer the PROJECT ``.env`` (where ``save_aws_credentials`` writes),
        falling back to the global file so pre-existing installs that only
        have ``~/.agent/env`` keep working.
        """
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
        """
        Persist AWS creds to the PROJECT ``.env`` (the checkout owns its own
        credentials), mirror them into os.environ, bump the generation
        counter, and notify subscribers.
        """
        # AWS creds are PROJECT-scoped: written next to the project's .env.
        aws_env_path = find_env_file()
        env_vars = load_env_file(aws_env_path)

        def _set(key: str, value: Optional[str]) -> None:
            if value is None:
                return
            if value == "":
                env_vars.pop(key, None)
            else:
                env_vars[key] = value

        _set(AWS_ENVS["access_key"], access_key)
        _set(AWS_ENVS["secret_key"], secret_key)
        _set(AWS_ENVS["region"], region)
        _set(AWS_ENVS["session_token"], session_token)

        write_env_file(env_vars, aws_env_path)

        # --- live process update (the actual bug #1 fix) ---
        def _mirror(key: str, value: Optional[str]) -> None:
            if value is None:
                return
            if value == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        _mirror(AWS_ENVS["access_key"], access_key)
        _mirror(AWS_ENVS["secret_key"], secret_key)
        _mirror(AWS_ENVS["region"], region)
        _mirror(AWS_ENVS["session_token"], session_token)

        self._aws_cache = self._read_aws_from_disk()
        self._generation += 1
        self._notify()

    # -- provider creds (project-scoped, unchanged behavior) -------------

    def set_provider_credentials(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        env_vars = load_env_file(env_path)
        if api_key is not None:
            key_env = PROVIDER_API_KEY_ENVS.get(provider.lower())
            if key_env:
                if api_key:
                    env_vars[key_env] = api_key
                else:
                    env_vars.pop(key_env, None)
        if base_url is not None:
            url_env = PROVIDER_BASE_URL_ENVS.get(provider.lower())
            if url_env:
                if base_url:
                    env_vars[url_env] = base_url.rstrip("/")
                else:
                    env_vars.pop(url_env, None)
        write_env_file(env_vars, env_path)
        # Mirror newly set provider vars into os.environ
        for k, v in env_vars.items():
            os.environ[k] = v
        self._generation += 1
        self._notify()

    def get_provider_credentials(
        self, provider: str, env_path: Optional[Path] = None
    ) -> Dict[str, Optional[str]]:
        env_vars = load_env_file(env_path)
        return {
            "api_key": env_vars.get(PROVIDER_API_KEY_ENVS.get(provider.lower(), "")),
            "base_url": env_vars.get(PROVIDER_BASE_URL_ENVS.get(provider.lower(), "")),
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
    env_vars = load_env_file(env_path)
    if value:
        env_vars[key] = value
    else:
        env_vars.pop(key, None)
    write_env_file(env_vars, env_path)
    if value:
        os.environ[key] = value
    else:
        os.environ.pop(key, None)


def get_env_var(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    env_vars = load_env_file(env_path)
    return env_vars.get(key)


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
    env_path: Optional[Path] = None,  # ignored for AWS — always global
    session_token: Optional[str] = None,
) -> None:
    EnvManager.get().save_aws_credentials(
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
    """Load .env variables into os.environ (for subprocesses and late-binding)."""
    env_vars = load_env_file(env_path)
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value


# Auto-load on import
EnvManager.get()