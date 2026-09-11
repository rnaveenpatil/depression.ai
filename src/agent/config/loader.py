"""
Config Loader - Layered Configuration Loading

Loads configuration from multiple sources in priority order:
    1. CLI arguments (highest)
    2. Environment variables (AGENT_* prefix)
    3. Project-local config (.agent/config.json)
    4. User config (~/.agent/config.json)
    5. Package defaults (lowest)

Supports:
- JSON, YAML, TOML file formats
- Environment variable interpolation (${VAR} and ${VAR:-default})
- Deep merging of nested dicts
- Schema validation
- Hot reload with file watching
- Dynamic LLM resolution (no hardcoded models)
- Adaptive temperature hooks
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from agent.utils.logging import get_logger
from agent.utils.errors import ConfigError
from agent.config.config import Config

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

ENV_PREFIX = "AGENT_"
USER_CONFIG_PATH = Path.home() / ".agent" / "config.json"
USER_CONFIG_DIR = Path.home() / ".agent"
PROJECT_CONFIG_NAME = ".agent/config.json"
DEFAULT_CONFIG_NAME = "config.json"

SUPPORTED_FORMATS = (".json", ".yaml", ".yml", ".toml")


# ======================================================================
# ENV INTERPOLATION
# ======================================================================

_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}"
)


def interpolate_env(value: Any) -> Any:
    """
    Recursively interpolate ${VAR} and ${VAR:-default} in strings.
    Also supports $VAR (bare form) for simple cases.
    """
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            var = match.group(1)
            default = match.group(2)
            env_val = os.environ.get(var)
            if env_val is not None:
                return env_val
            if default is not None:
                return default
            logger.warning(f"Unresolved env var in config: ${{{var}}}")
            return match.group(0)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


# ======================================================================
# FILE LOADING
# ======================================================================

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ConfigError(
            f"YAML config requires PyYAML. Install: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        try:
            import tomli
            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError:
            raise ConfigError(
                "TOML config requires tomli. Install: pip install tomli"
            )


def load_file(path: Path) -> Dict[str, Any]:
    """Load a config file based on its extension"""
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return _load_json(path)
        elif suffix in (".yaml", ".yml"):
            return _load_yaml(path)
        elif suffix == ".toml":
            return _load_toml(path)
        else:
            # Try JSON as fallback
            return _load_json(path)
    except Exception as e:
        raise ConfigError(f"Failed to load config from {path}: {e}")


# ======================================================================
# DEEP MERGE
# ======================================================================

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts. `override` wins on conflicts.
    Lists are replaced (not concatenated) unless both are dicts.
    """
    if not isinstance(base, dict):
        return override
    if not isinstance(override, dict):
        return override

    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ======================================================================
# ENV VAR PARSING
# ======================================================================

def _parse_env_value(raw: str) -> Any:
    """Parse an env var string into its proper Python type"""
    s = raw.strip()
    if not s:
        return ""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "null"):
        return None
    # JSON containers
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass
    # Numbers
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        pass
    return s


def load_env_vars(prefix: str = ENV_PREFIX) -> Dict[str, Any]:
    """
    Load environment variables with the given prefix.

    Mapping convention:
        AGENT_LLM__PROVIDER=openai       → {"llm": {"provider": "openai"}}
        AGENT_LLM__PARAMS__TEMPERATURE=0.5
        AGENT_UI__THEME=dark

    Double underscore `__` = nested level.
    """
    result: Dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]
        if not stripped:
            continue
        # Split by __ for nesting
        parts = stripped.lower().split("__")
        if not parts:
            continue

        value = _parse_env_value(raw)

        # Build nested dict
        cursor = result
        for p in parts[:-1]:
            if p not in cursor or not isinstance(cursor[p], dict):
                cursor[p] = {}
            cursor = cursor[p]
        cursor[parts[-1]] = value

    return result


# ======================================================================
# MAIN LOADER
# ======================================================================

class ConfigLoader:
    """
    Layered configuration loader with hot-reload support.
    """

    def __init__(self):
        self._config: Optional[Config] = None
        self._sources: List[Tuple[str, Path]] = []  # (label, path) for reload
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()
        self._reload_callbacks: List[Callable[[Config], None]] = []
        self._watch_interval: float = 2.0

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def load(
        self,
        config_path: Optional[Union[str, Path]] = None,
        extra: Optional[Dict[str, Any]] = None,
        init_if_missing: bool = False,
        validate: bool = True,
    ) -> Config:
        """
        Load configuration from all sources.

        Args:
            config_path: Optional explicit config file path (highest priority).
            extra: Optional dict of overrides (e.g., CLI args).
            init_if_missing: Create a default config file if none exists.
            validate: Validate the final config.

        Returns:
            A fully-merged Config object.
        """
        self._sources = []

        # Start with defaults
        merged: Dict[str, Any] = {}

        # 1. Package defaults (from config.py dataclass defaults)
        default_config = Config()
        merged = deep_merge(merged, default_config.to_dict(include_secrets=True))

        # 2. User config (~/.agent/config.json)
        if USER_CONFIG_PATH.exists():
            try:
                user_data = load_file(USER_CONFIG_PATH)
                merged = deep_merge(merged, user_data)
                self._sources.append(("user", USER_CONFIG_PATH))
                logger.debug(f"Loaded user config: {USER_CONFIG_PATH}")
            except Exception as e:
                logger.warning(f"Failed to load user config: {e}")

        # 3. Project config (.agent/config.json in cwd or ancestors)
        project_path = self._find_project_config()
        if project_path:
            try:
                project_data = load_file(project_path)
                merged = deep_merge(merged, project_data)
                self._sources.append(("project", project_path))
                logger.debug(f"Loaded project config: {project_path}")
            except Exception as e:
                logger.warning(f"Failed to load project config: {e}")

        # 4. Explicit path (highest file priority)
        if config_path:
            path = Path(config_path).expanduser()
            if not path.exists():
                if init_if_missing:
                    self._write_default_config(path)
                else:
                    raise ConfigError(f"Config file not found: {path}")
            else:
                explicit_data = load_file(path)
                merged = deep_merge(merged, explicit_data)
                self._sources.append(("explicit", path))
                logger.debug(f"Loaded explicit config: {path}")

        # 5. Environment variables
        env_data = load_env_vars()
        if env_data:
            merged = deep_merge(merged, env_data)
            logger.debug(f"Loaded env vars: {list(env_data.keys())}")

        # 6. Interpolate ${VAR} references
        merged = interpolate_env(merged)

        # 7. Extra overrides (CLI args) — highest priority
        if extra:
            merged = deep_merge(merged, extra)

        # 8. Handle missing user config creation
        if init_if_missing and not USER_CONFIG_PATH.exists():
            self._write_default_config(USER_CONFIG_PATH)

        # 9. Build Config object
        try:
            config = Config.from_dict(merged)
        except Exception as e:
            raise ConfigError(f"Failed to build config: {e}")

        # 10. Validate
        if validate:
            config.validate()

        # 11. Store & return
        self._config = config
        logger.info(
            f"Config loaded from {len(self._sources)} file source(s); "
            f"provider={config.llm.provider or '(auto)'} "
            f"model={config.llm.model or '(auto)'}"
        )
        return config

    def reload(self) -> Optional[Config]:
        """Reload config from all sources"""
        try:
            config = self.load()
            for cb in self._reload_callbacks:
                try:
                    cb(config)
                except Exception as e:
                    logger.error(f"Reload callback failed: {e}")
            return config
        except Exception as e:
            logger.error(f"Reload failed: {e}")
            return None

    def on_reload(self, callback: Callable[[Config], None]) -> None:
        """Register a reload callback"""
        self._reload_callbacks.append(callback)

    def start_watching(self, interval: float = 2.0) -> None:
        """Start watching config files for changes"""
        if self._watch_thread and self._watch_thread.is_alive():
            return
        self._watch_interval = interval
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop, daemon=True
        )
        self._watch_thread.start()
        logger.info("Config file watcher started")

    def stop_watching(self) -> None:
        """Stop watching config files"""
        self._watch_stop.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)
        logger.info("Config file watcher stopped")

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _find_project_config(self) -> Optional[Path]:
        """Walk up from cwd looking for .agent/config.json"""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / PROJECT_CONFIG_NAME
            if candidate.exists():
                return candidate
        return None

    def _write_default_config(self, path: Path) -> None:
        """Write a default config file to the given path"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            default = Config().to_dict(include_secrets=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2)
            logger.info(f"Wrote default config: {path}")
        except Exception as e:
            logger.error(f"Failed to write default config: {e}")

    def _watch_loop(self) -> None:
        """Watch config files for changes and reload"""
        mtimes: Dict[Path, float] = {}
        for _, path in self._sources:
            try:
                mtimes[path] = path.stat().st_mtime
            except Exception:
                pass

        while not self._watch_stop.is_set():
            self._watch_stop.wait(self._watch_interval)
            if self._watch_stop.is_set():
                break

            changed = False
            for _, path in self._sources:
                try:
                    mtime = path.stat().st_mtime
                    if mtimes.get(path) != mtime:
                        mtimes[path] = mtime
                        changed = True
                        logger.info(f"Config file changed: {path}")
                except Exception:
                    pass

            if changed:
                self.reload()


# ======================================================================
# MODULE-LEVEL API
# ======================================================================

_global_loader: Optional[ConfigLoader] = None
_global_config: Optional[Config] = None
_lock = threading.Lock()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    extra: Optional[Dict[str, Any]] = None,
    init_config: bool = False,
    validate: bool = True,
    cache: bool = True,
) -> Dict[str, Any]:
    """
    Main entry point used by main.py.

    Returns the config as a **dict** for backwards compatibility with
    existing code that expects a dict. Use `get_config()` to obtain
    the richer Config object.

    Args:
        config_path: Optional explicit config path.
        extra: CLI overrides.
        init_config: Create default config files if missing.
        validate: Validate on load.
        cache: Cache the loader globally.
    """
    global _global_loader, _global_config

    with _lock:
        if _global_loader is None or not cache:
            _global_loader = ConfigLoader()

        config_obj = _global_loader.load(
            config_path=config_path,
            extra=extra,
            init_if_missing=init_config,
            validate=validate,
        )
        _global_config = config_obj

    # Return as a dict for compatibility with older code paths
    return config_obj.to_dict(include_secrets=True)


def get_config() -> Optional[Config]:
    """Get the currently-loaded Config object"""
    return _global_config


def get_loader() -> Optional[ConfigLoader]:
    """Get the global loader instance"""
    return _global_loader


def reset_config() -> None:
    """Reset the global config (useful for tests)"""
    global _global_loader, _global_config
    with _lock:
        if _global_loader:
            _global_loader.stop_watching()
        _global_loader = None
        _global_config = None


# ======================================================================
# CONVENIENCE: DYNAMIC LLM HELPERS
# ======================================================================

def resolve_llm_from_config(
    config: Optional[Config] = None,
    registry: Any = None,
    prefer_quality: bool = False,
) -> Tuple[str, str]:
    """
    Convenience wrapper to resolve (provider, model) dynamically.
    Uses the LLM registry so no models are hardcoded.
    """
    cfg = config or get_config()
    if cfg is None:
        raise ConfigError("No config loaded")
    return cfg.resolve_llm(registry=registry, prefer_quality=prefer_quality)


def resolve_temperature_from_config(
    prompt: str,
    config: Optional[Config] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Convenience wrapper to resolve an adaptive temperature for a prompt.
    """
    cfg = config or get_config()
    if cfg is None:
        return 0.7
    return cfg.resolve_temperature(
        prompt=prompt,
        provider=provider,
        model=model,
        context=context,
    )


def resolve_temperature_with_reason_from_config(
    prompt: str,
    config: Optional[Config] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """Same as above, but returns (temperature, reason)"""
    cfg = config or get_config()
    if cfg is None:
        return 0.7, "no config"
    return cfg.resolve_temperature_with_reason(
        prompt=prompt, provider=provider, model=model, context=context,
    )


__all__ = [
    "ConfigLoader",
    "load_config",
    "get_config",
    "get_loader",
    "reset_config",
    "deep_merge",
    "load_env_vars",
    "load_file",
    "interpolate_env",
    "resolve_llm_from_config",
    "resolve_temperature_from_config",
    "resolve_temperature_with_reason_from_config",
    "USER_CONFIG_PATH",
    "USER_CONFIG_DIR",
]