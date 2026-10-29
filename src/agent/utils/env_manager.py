"""Environment file management for Depression.AI.

Handles reading/writing .env files for persistent credential storage.

Layout:
    <project>/.env          <- single credential store for LLM, AWS and MCP
    ~/.agent/env            <- legacy user store (read-only fallback)

All new credentials are saved to the project .env with mode 0600. Existing
user-store values remain readable for compatibility; no old secrets are erased.

Public API:
    env_manager = EnvManager.get()
    env_manager.save_aws_credentials(...)   # updates os.environ + notifies
    env_manager.get_aws_env() -> dict       # canonical AWS env for subprocesses
    env_manager.subscribe(cb)               # notified on change
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

# User-scoped legacy keys retained as fallback until re-saved to project .env.
_LEGACY_AGENT_ENV_KEYS = (
    set(AWS_ENVS.values())
    | set(PROVIDER_API_KEY_ENVS.values())
    | set(PROVIDER_BASE_URL_ENVS.values())
    | {
        "DEPRESSION_API_KEY", "DEPRESSION_BASE_URL",
        "DEPRESSION_MODEL", "DEPRESSION_PROVIDER", SELECTED_MODEL_ENV,
    }
)


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
        # Current project credentials override legacy user-scoped values.
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
        """Read project AWS credentials first; legacy user store is fallback.

        An explicitly cleared or partial project credential must not silently
        revive a different AWS identity from the old user-scoped store.
        """
        credentials: Dict[str, Optional[str]] = {
            "access_key": None, "secret_key": None,
            "region": None, "session_token": None,
        }
        for source in (find_env_file(), GLOBAL_ENV_PATH):
            data = load_env_file(source)
            if not any(key in data for key in AWS_ENVS.values()):
                continue
            credentials["access_key"] = data.get(AWS_ENVS["access_key"]) or None
            credentials["secret_key"] = data.get(AWS_ENVS["secret_key"]) or None
            credentials["region"] = data.get(AWS_ENVS["region"]) or "us-east-1"
            credentials["session_token"] = data.get(AWS_ENVS["session_token"]) or None
            break
        return credentials

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
        """Persist AWS credentials to the project .env and update consumers."""
        aws_env_path = find_env_file()
        env_vars = load_env_file(aws_env_path)

        def _set(key: str, value: Optional[str]) -> None:
            if value is None:
                return
            # An empty value must shadow legacy global credentials.
            env_vars[key] = value

        _set(AWS_ENVS["access_key"], access_key)
        _set(AWS_ENVS["secret_key"], secret_key)
        _set(AWS_ENVS["region"], region)
        _set(AWS_ENVS["session_token"], session_token)

        # strict=True: a failed write must not be reported as a saved secret.
        write_env_file(env_vars, aws_env_path, strict=True)

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

    # -- provider credentials (project-scoped by default) ------------------

    def set_provider_credentials(
        self,
        provider: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        env_path = env_path or find_env_file()
        env_vars = load_env_file(env_path)
        if api_key is not None:
            key_env = PROVIDER_API_KEY_ENVS.get(provider.lower())
            if key_env:
                if api_key:
                    env_vars[key_env] = api_key
                else:
                    env_vars[key_env] = ""
        if base_url is not None:
            url_env = PROVIDER_BASE_URL_ENVS.get(provider.lower())
            if url_env:
                if base_url:
                    env_vars[url_env] = base_url.rstrip("/")
                else:
                    env_vars[url_env] = ""
        write_env_file(env_vars, env_path, strict=True)
        # Mirror newly set provider vars into os.environ
        for k, v in env_vars.items():
            os.environ[k] = v
        self._generation += 1
        self._notify()

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
    """Save user-managed values in the project .env unless a path is explicit."""
    target = env_path or find_env_file()
    env_vars = load_env_file(target)
    if value:
        env_vars[key] = value
    else:
        # Empty entries prevent reactivating an old user-store credential.
        env_vars[key] = ""
    write_env_file(env_vars, target, strict=True)
    os.environ[key] = value


def get_env_var(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Prefer the project .env; read legacy user values when absent."""
    if env_path is not None:
        return load_env_file(env_path).get(key)
    project_data = load_env_file(find_env_file())
    if key in project_data:
        return project_data[key]
    if key in _LEGACY_AGENT_ENV_KEYS or key.startswith("DEPRESSION_MCP_"):
        return load_env_file(GLOBAL_ENV_PATH).get(key)
    return None


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
    env_path: Optional[Path] = None,  # kept for API compatibility
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
    """Load project variables, with legacy user credentials as fallback."""
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