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
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

from agent.utils.logging import get_logger
from agent.utils.errors import ConfigError
from agent.config.config import Config

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS — OpenCode compatible
# ======================================================================

ENV_PREFIX = "AGENT_"
USER_CONFIG_PATH = Path.home() / ".agent" / "config.json"
USER_CONFIG_DIR = Path.home() / ".agent"
# OpenCode primary locations
OPENCODE_USER_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
OPENCODE_USER_CONFIGC = Path.home() / ".config" / "opencode" / "opencode.jsonc"
PROJECT_CONFIG_NAME = ".agent/config.json"
DEFAULT_CONFIG_NAME = "config.json"

# OpenCode file precedence (later overrides earlier):
# 1 remote (.well-known/opencode) — skip (handled externally)
# 2 global (~/.config/opencode/opencode.json + ~/.agent/config.json)
# 3 custom (OPENCODE_CONFIG)
# 4 project (opencode.json + .agent/config.json)
# 5 .opencode dirs — agents/commands/plugins
# 6 inline (OPENCODE_CONFIG_CONTENT)
# 7 managed (/etc/opencode) — handled if present

SUPPORTED_FORMATS = (".json", ".jsonc", ".yaml", ".yml", ".toml")


# ======================================================================
# ENV INTERPOLATION
# ======================================================================

_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}"
)


def interpolate_env(value: Any) -> Any:
    """
    Recursively interpolate env/file vars:
      ${VAR} / ${VAR:-default}  (docker-compose style)
      {env:VAR}                (OpenCode style)
      {file:path}              (OpenCode style — file contents)
    """
    if isinstance(value, str):
        # 1. ${VAR} style
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
        value = _ENV_PATTERN.sub(repl, value)
        # 2. {env:VAR} OpenCode style
        def repl_env(m: re.Match) -> str:
            var = m.group(1)
            return os.environ.get(var, "")
        value = re.sub(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", repl_env, value)
        # 3. {file:path} OpenCode style
        def repl_file(m: re.Match) -> str:
            fpath = m.group(1).strip()
            try:
                p = Path(os.path.expanduser(fpath))
                if not p.is_absolute():
                    # relative to cwd (or project dir if known)
                    p = Path.cwd() / p
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                logger.debug(f"{{file:{fpath}}} read failed: {e}")
            return ""
        value = re.sub(r"\{file:([^}]+)\}", repl_file, value)
        return value
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


# ======================================================================
# FILE LOADING
# ======================================================================

def _strip_jsonc_comments(text: str) -> str:
    """Strip // and /* */ comments for JSONC compatibility — handles inline // outside strings."""
    # Remove /* block comments */
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove // line comments (inline) — naive but handles most configs: ignore // inside URLs (http://)
    out_lines = []
    for line in text.splitlines():
        # If line contains //, check if before // there's an odd number of quotes (inside string) -> keep
        # Simple: if // appears inside a quoted string, preserve
        idx = line.find("//")
        while idx != -1:
            # Check if // is inside string: count unescaped quotes before it
            before = line[:idx]
            # Count quotes not escaped
            dq = before.count('"') - before.count('\\"')
            # If even number, // is outside string → treat as comment
            if dq % 2 == 0:
                # But ignore http:// and https://
                if idx > 0 and line[idx-1] == ":":
                    # find next // after this
                    nxt = line.find("//", idx+2)
                    if nxt == -1:
                        break
                    idx = nxt
                    continue
                line = line[:idx]
                break
            else:
                nxt = line.find("//", idx+2)
                if nxt == -1:
                    break
                idx = nxt
        out_lines.append(line)
    return "\n".join(out_lines)

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Handle JSONC
    if path.suffix.lower() == ".jsonc":
        raw = _strip_jsonc_comments(raw)
        if not raw.strip():
            return {}
        return json.loads(raw)
    # Also tolerate JSONC in .json (comments) — resilient
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = _strip_jsonc_comments(raw)
        return json.loads(cleaned)


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
    """Load a config file based on its extension — handles JSONC via stripping."""
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    try:
        if suffix in (".json", ".jsonc"):
            data = _load_json(path)
            # Strip $schema keys (OpenCode style) — not part of our schema
            if isinstance(data, dict) and "$schema" in data:
                data = {k: v for k, v in data.items() if k != "$schema"}
            # Map OpenCode top-level keys to internal schema
            data = _normalize_opencode_config(data)
            return data if isinstance(data, dict) else {}
        elif suffix in (".yaml", ".yml"):
            return _load_yaml(path)
        elif suffix == ".toml":
            return _load_toml(path)
        else:
            return _load_json(path)
    except Exception as e:
        raise ConfigError(f"Failed to load config from {path}: {e}")


def _normalize_opencode_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Translate OpenCode opencode.json keys into depression internal schema."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    # model -> llm.model / llm.provider split
    if "model" in out and isinstance(out["model"], str):
        model_str = out["model"]
        out.setdefault("llm", {})
        if isinstance(out["llm"], dict):
            out["llm"].setdefault("model", model_str)
            if "/" in model_str and "provider" not in out["llm"]:
                out["llm"]["provider"] = model_str.split("/")[0]
    if "small_model" in out:
        out.setdefault("llm", {})
        if isinstance(out["llm"], dict):
            out["llm"]["small_model"] = out.pop("small_model")
    # permission (opencode) -> permissions.permission
    if "permission" in out and isinstance(out["permission"], dict):
        out.setdefault("permissions", {})
        if isinstance(out["permissions"], dict):
            out["permissions"].setdefault("permission", out["permission"])
    # instructions -> context.instructions
    if "instructions" in out:
        out.setdefault("context", {})
        if isinstance(out["context"], dict):
            out["context"]["instructions"] = out["instructions"]
    # watcher -> workspace.watcher
    if "watcher" in out:
        out.setdefault("workspace", {})
        if isinstance(out["workspace"], dict):
            out["workspace"]["watcher"] = out["watcher"]
    # formatter -> tools.formatter
    if "formatter" in out:
        out.setdefault("tools", {})
        if isinstance(out["tools"], dict):
            out["tools"]["formatter"] = out["formatter"]
    # lsp -> tools.lsp
    if "lsp" in out:
        out.setdefault("tools", {})
        if isinstance(out["tools"], dict):
            out["tools"]["lsp"] = out["lsp"]
    # mcp -> mcp (already compatible)
    # provider -> llm.provider_options
    if "provider" in out and isinstance(out["provider"], dict):
        out.setdefault("llm", {})
        if isinstance(out["llm"], dict):
            out["llm"].setdefault("provider_options", out["provider"])
    return out


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
        Load configuration from all sources — OpenCode precedence:

        1. Package defaults
        2. Global: ~/.config/opencode/opencode.json(c) + ~/.agent/config.json
        3. Custom: OPENCODE_CONFIG env var
        4. Project: ./opencode.json(c) + ./.agent/config.json (ancestors walk)
        5. Inline: OPENCODE_CONFIG_CONTENT
        6. Managed: /etc/opencode/opencode.json (if present)
        7. Env vars: AGENT_* / OPENCODE_*
        8. Extra (CLI overrides) — highest
        """
        self._sources = []
        merged: Dict[str, Any] = {}
        default_config = Config()
        merged = deep_merge(merged, default_config.to_dict(include_secrets=True))

        # 2. Global configs
        for gpath, label in [
            (OPENCODE_USER_CONFIG, "global-opencode"),
            (OPENCODE_USER_CONFIGC, "global-opencode-jsonc"),
            (USER_CONFIG_PATH, "global-agent"),
            (USER_CONFIG_DIR / "opencode.json", "global-agent-opencode"),
            (USER_CONFIG_DIR / "opencode.jsonc", "global-agent-opencode-jsonc"),
        ]:
            if gpath.exists():
                try:
                    gdata = load_file(gpath)
                    merged = deep_merge(merged, gdata)
                    self._sources.append((label, gpath))
                    logger.debug(f"Loaded {label}: {gpath}")
                except Exception as e:
                    logger.warning(f"Failed to load {label} {gpath}: {e}")

        # Managed (/etc/opencode/opencode.json) — low priority but above user project
        for mpath in [Path("/etc/opencode/opencode.json"), Path("/etc/opencode/opencode.jsonc")]:
            if mpath.exists():
                try:
                    mdata = load_file(mpath)
                    merged = deep_merge(merged, mdata)
                    self._sources.append(("managed", mpath))
                except Exception as e:
                    logger.debug(f"Managed config failed: {e}")

        # 3. Custom via OPENCODE_CONFIG
        opencode_custom = os.environ.get("OPENCODE_CONFIG")
        if opencode_custom:
            cp = Path(opencode_custom).expanduser()
            if cp.exists():
                try:
                    cdata = load_file(cp)
                    merged = deep_merge(merged, cdata)
                    self._sources.append(("custom-opencode", cp))
                except Exception as e:
                    logger.warning(f"OPENCODE_CONFIG load failed: {e}")

        # 4. Project configs (walk ancestors for both opencode.json and .agent)
        for proj_path, label in self._find_all_project_configs():
            try:
                pdata = load_file(proj_path)
                merged = deep_merge(merged, pdata)
                self._sources.append((label, proj_path))
                logger.debug(f"Loaded {label}: {proj_path}")
            except Exception as e:
                logger.warning(f"Failed to load {label} {proj_path}: {e}")

        # 4b. Explicit path (highest file priority)
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

        # 5. Inline OPENCODE_CONFIG_CONTENT
        inline = os.environ.get("OPENCODE_CONFIG_CONTENT")
        if inline:
            try:
                j = json.loads(_strip_jsonc_comments(inline))
                if isinstance(j, dict):
                    j = _normalize_opencode_config(j)
                    merged = deep_merge(merged, j)
                    logger.debug("Loaded OPENCODE_CONFIG_CONTENT")
            except Exception as e:
                logger.warning(f"OPENCODE_CONFIG_CONTENT parse failed: {e}")

        # 6. Environment variables (AGENT_* + OPENCODE_*)
        env_data = load_env_vars()
        # Also support OPENCODE_ prefix mapped to same structure
        opencode_env = load_env_vars(prefix="OPENCODE_")
        if opencode_env:
            # normalize opencode env keys: OPENCODE_MODEL -> llm.model etc.
            if "model" in opencode_env and "llm" not in opencode_env:
                opencode_env.setdefault("llm", {})["model"] = opencode_env.pop("model")
            env_data = deep_merge(env_data, opencode_env)
        if env_data:
            merged = deep_merge(merged, env_data)
            logger.debug(f"Loaded env vars: {list(env_data.keys())}")

        # 7. Interpolate ${VAR} / {env:} / {file:} references
        merged = interpolate_env(merged)

        # 8. Extra overrides (CLI args) — highest priority
        if extra:
            merged = deep_merge(merged, extra)

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
        """Walk up from cwd looking for .agent/config.json (legacy helper)."""
        for p, _ in self._find_all_project_configs():
            if p.name == "config.json" and ".agent" in str(p):
                return p
        return None

    def _find_all_project_configs(self) -> List[Tuple[Path, str]]:
        """Find all project configs walking up — returns in ancestor order (root→cwd so cwd wins)."""
        found: List[Tuple[Path, str]] = []
        current = Path.cwd()
        # Walk from root down so deeper overrides shallower (apply in order)
        ancestors = list(reversed([current, *current.parents]))
        for parent in ancestors:
            for name, label in [
                ("opencode.json", "project-opencode"),
                ("opencode.jsonc", "project-opencode-jsonc"),
                (".agent/config.json", "project-agent"),
                (".agent/opencode.json", "project-agent-opencode"),
            ]:
                cand = parent / name
                if cand.exists():
                    found.append((cand, label))
        return found

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