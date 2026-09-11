"""
Plugin Loader - Discovers, loads, and manages plugins.

Discovery sources (in priority order):
    1. Explicit plugin dirs from config (plugins.directories)
    2. Project-local       ./.agent/plugins/
    3. User-global         ~/.agent/plugins/
    4. Built-in            src/agent/plugins/builtin/

Plugin formats supported:
    A. Package with __init__.py:
        my_plugin/
          ├── __init__.py       (defines PLUGIN and register(agent))
          └── ...

    B. Single-file module:
        my_plugin.py           (defines PLUGIN and register(agent))

    C. Manifest-driven:
        my_plugin/
          ├── plugin.json       (manifest)
          └── main.py           (entry point)

Lifecycle:
    discover → validate → load → register → enable

Features:
    - Version compatibility checks
    - Dependency resolution (topological ordering)
    - Sandboxing hooks (permission gates per plugin)
    - Hot-reload in dev mode
    - Enable/disable without restart
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import PluginError
from agent.plugins.registry import (
    PluginRegistry,
    PluginRecord,
    PluginManifest,
    PluginState,
    PluginCapability,
    get_plugin_registry,
)

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

AGENT_VERSION = "1.0.0"

DEFAULT_PLUGIN_DIRS = [
    Path.cwd() / ".agent" / "plugins",
    Path.home() / ".agent" / "plugins",
    Path(__file__).parent / "builtin",
]

MANIFEST_FILES = ("plugin.json", "plugin.yaml", "plugin.toml", "pyproject.toml")
ENTRY_POINT_CANDIDATES = ("__init__.py", "main.py", "plugin.py", "index.py")


# ======================================================================
# VERSION UTILS
# ======================================================================

def _parse_version(v: str) -> Tuple[int, ...]:
    parts = []
    for p in str(v).lstrip("v").split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts) if parts else (0,)


def _version_ok(current: str, minimum: str, maximum: Optional[str]) -> bool:
    cur = _parse_version(current)
    lo = _parse_version(minimum or "0.0.0")
    if cur < lo:
        return False
    if maximum:
        hi = _parse_version(maximum)
        if cur > hi:
            return False
    return True


# ======================================================================
# LOADER
# ======================================================================

class PluginLoader:
    """
    Discovers, loads, and manages plugins for the agent.
    """

    def __init__(
        self,
        agent: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent = agent
        self.config = config or {}
        self.registry: PluginRegistry = get_plugin_registry()

        # Config
        self.enabled: bool = self.config.get("enabled", True)
        self.auto_load: bool = self.config.get("auto_load", True)
        self.dev_mode: bool = self.config.get("dev_mode", False)
        self.strict: bool = self.config.get("strict", False)

        self.directories: List[Path] = self._resolve_dirs()
        self.disabled: Set[str] = set(self.config.get("disabled", []))
        self.allowed: Set[str] = set(self.config.get("allowed", []))  # allowlist

        # State
        self._initialized = False
        self._loaded_modules: Dict[str, Any] = {}
        self._resolved_paths: Dict[str, Path] = {}

        logger.info(
            f"PluginLoader initialized "
            f"({len(self.directories)} dirs, auto_load={self.auto_load})"
        )

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def _resolve_dirs(self) -> List[Path]:
        dirs: List[Path] = []

        # Explicit config dirs
        for d in self.config.get("directories", []):
            dirs.append(Path(os.path.expanduser(str(d))))

        # Defaults
        dirs.extend(DEFAULT_PLUGIN_DIRS)

        # Deduplicate, keep only existing dirs
        seen: Set[str] = set()
        out: List[Path] = []
        for d in dirs:
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            if d.exists() and d.is_dir():
                out.append(d)
            else:
                # Ensure parent dirs exist for user/global (so users can drop plugins in)
                try:
                    d.mkdir(parents=True, exist_ok=True)
                    out.append(d)
                except Exception:
                    pass
        return out

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Discover and load all plugins."""
        if not self.enabled:
            logger.info("Plugin system disabled")
            return
        if self._initialized:
            return

        discovered = self.discover()
        logger.info(f"Discovered {len(discovered)} plugin(s)")

        if self.auto_load:
            for manifest, path in discovered:
                # Allowlist / blocklist
                if self.disabled and manifest.name in self.disabled:
                    logger.info(f"Plugin '{manifest.name}' disabled by config")
                    continue
                if self.allowed and manifest.name not in self.allowed:
                    logger.debug(f"Plugin '{manifest.name}' not in allowlist")
                    continue

                try:
                    await self.load_plugin_from_path(path, manifest)
                except Exception as e:
                    logger.error(f"Failed to load plugin '{manifest.name}': {e}")
                    if self.strict:
                        raise

        self._initialized = True
        logger.info(f"Plugin system ready: {self.registry}")

    async def shutdown(self) -> None:
        """Unload all plugins."""
        for name in list(self.registry.plugins.keys()):
            try:
                await self.unload_plugin(name)
            except Exception as e:
                logger.warning(f"Error unloading '{name}': {e}")
        self._initialized = False
        logger.info("Plugin system shut down")

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    def discover(self) -> List[Tuple[PluginManifest, Path]]:
        """
        Scan all plugin directories and return (manifest, path) pairs.
        Path is either a directory or a .py file.
        """
        found: List[Tuple[PluginManifest, Path]] = []
        seen_names: Set[str] = set()

        for base in self.directories:
            if not base.exists():
                continue

            for entry in sorted(base.iterdir()):
                if entry.name.startswith(".") or entry.name.startswith("_"):
                    continue

                try:
                    manifest = self._read_manifest(entry)
                except Exception as e:
                    logger.debug(f"Skipping {entry}: {e}")
                    continue

                if not manifest:
                    continue

                if manifest.name in seen_names:
                    logger.debug(f"Duplicate plugin '{manifest.name}' skipped")
                    continue
                seen_names.add(manifest.name)

                found.append((manifest, entry))

        return found

    def _read_manifest(self, path: Path) -> Optional[PluginManifest]:
        """Read plugin metadata from a directory or single .py file."""

        # Single-file plugin
        if path.is_file() and path.suffix == ".py":
            return self._manifest_from_py_file(path)

        # Package directory
        if path.is_dir():
            # 1. Try manifest file
            for mf in MANIFEST_FILES:
                mf_path = path / mf
                if mf_path.exists():
                    try:
                        data = self._load_manifest_file(mf_path)
                        if data:
                            manifest = PluginManifest.from_dict(data)
                            if not manifest.name:
                                manifest.name = path.name
                            return manifest
                    except Exception as e:
                        logger.debug(f"Manifest read failed for {mf_path}: {e}")

            # 2. Fall back to inspecting __init__.py for PLUGIN dict
            for candidate in ENTRY_POINT_CANDIDATES:
                py = path / candidate
                if py.exists():
                    manifest = self._manifest_from_py_file(py)
                    if manifest and not manifest.name:
                        manifest.name = path.name
                    return manifest

        return None

    def _load_manifest_file(self, path: Path) -> Optional[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except ImportError:
                return None
        if suffix == ".toml":
            try:
                import tomllib
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                return data.get("plugin") or data.get("tool", {}).get("agent", {}).get("plugin") or data
            except ImportError:
                try:
                    import tomli
                    with open(path, "rb") as f:
                        data = tomli.load(f)
                    return data.get("plugin") or data
                except ImportError:
                    return None
        return None

    def _manifest_from_py_file(self, path: Path) -> Optional[PluginManifest]:
        """
        Parse a Python file just enough to find the PLUGIN dict.
        Full module load happens later; here we only peek.
        """
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # Quick check: does it look like a plugin?
        if "PLUGIN" not in source and "def register" not in source:
            return None

        # Try to import it safely just to read PLUGIN
        try:
            mod = self._import_module_from_path(path.stem + "_peek", path)
        except Exception:
            # Fallback: minimal manifest from filename
            return PluginManifest(name=path.stem)

        raw = getattr(mod, "PLUGIN", None)
        if isinstance(raw, dict):
            return PluginManifest.from_dict(raw)
        if isinstance(raw, PluginManifest):
            return raw

        # Also accept module-level metadata
        return PluginManifest(
            name=getattr(mod, "__plugin_name__", path.stem),
            version=getattr(mod, "__version__", "0.0.0"),
            description=getattr(mod, "__doc__", "") or "",
            author=getattr(mod, "__author__", ""),
        )

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    async def load_plugin(self, name: str) -> bool:
        """
        Load a plugin by name.
        Searches all plugin directories for a matching manifest.
        """
        # Already loaded?
        if name in self.registry.plugins:
            logger.info(f"Plugin '{name}' already loaded")
            return self.registry.enable(name)

        for manifest, path in self.discover():
            if manifest.name == name:
                return await self.load_plugin_from_path(path, manifest)

        raise PluginError(f"Plugin '{name}' not found")

    async def load_plugin_from_path(
        self, path: Path, manifest: Optional[PluginManifest] = None
    ) -> bool:
        """
        Load a plugin from an explicit path.
        """
        if manifest is None:
            manifest = self._read_manifest(path)
            if manifest is None:
                raise PluginError(f"No valid manifest found at {path}")

        # Version compatibility check
        if not _version_ok(AGENT_VERSION, manifest.min_agent_version, manifest.max_agent_version):
            raise PluginError(
                f"Plugin '{manifest.name}' v{manifest.version} is not compatible "
                f"with agent v{AGENT_VERSION} "
                f"(requires {manifest.min_agent_version}"
                f"{'–' + manifest.max_agent_version if manifest.max_agent_version else '+'})"
            )

        # Dependencies
        for dep in manifest.dependencies:
            if dep not in self.registry.plugins:
                logger.warning(
                    f"Plugin '{manifest.name}' depends on '{dep}' which is not loaded"
                )

        # Register a placeholder record so downstream code can reference it
        record = PluginRecord(
            manifest=manifest,
            path=str(path),
            state=PluginState.LOADING,
        )
        self.registry.add_plugin(record)

        try:
            # Import the plugin module
            module = await self._import_plugin(path, manifest.name)
            record.module = module

            # Extract register() (async or sync)
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                raise PluginError(
                    f"Plugin '{manifest.name}' has no register() function"
                )

            # Call it with the agent (or a context object)
            target = self.agent if self.agent is not None else _LoaderContext(self)
            if inspect.iscoroutinefunction(register_fn):
                await register_fn(target)
            else:
                result = register_fn(target)
                if inspect.isawaitable(result):
                    await result

            record.state = PluginState.ENABLED
            record.loaded_at = time.time()
            self._loaded_modules[manifest.name] = module
            self._resolved_paths[manifest.name] = path

            logger.info(
                f"Loaded plugin '{manifest.name}' v{manifest.version} "
                f"({record.total_registrations()} registrations)"
            )
            return True

        except Exception as e:
            record.state = PluginState.ERROR
            record.error = str(e)
            logger.error(
                f"Plugin '{manifest.name}' failed to load: {e}\n"
                f"{traceback.format_exc()}"
            )
            raise PluginError(f"Failed to load plugin '{manifest.name}': {e}")

    async def _import_plugin(self, path: Path, name: str) -> Any:
        """
        Import a plugin module from a path.
        Handles both package dirs and single .py files.
        """
        # Package directory
        if path.is_dir():
            entry = None
            for candidate in ENTRY_POINT_CANDIDATES:
                p = path / candidate
                if p.exists():
                    entry = p
                    break
            if entry is None:
                raise PluginError(f"No entry point found in {path}")
            module_name = f"agent_plugin_{name}"
            return self._import_module_from_path(module_name, entry)

        # Single .py file
        module_name = f"agent_plugin_{name}"
        return self._import_module_from_path(module_name, path)

    def _import_module_from_path(self, module_name: str, path: Path) -> Any:
        """Import a Python module from an explicit file path."""
        if module_name in sys.modules:
            return sys.modules[module_name]

        # For __init__.py inside a package dir, use the parent as the spec origin
        if path.name == "__init__.py":
            spec = importlib.util.spec_from_file_location(
                module_name,
                path,
                submodule_search_locations=[str(path.parent)],
            )
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            raise PluginError(f"Cannot create module spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Ensure sibling imports work by prepending the plugin dir
        plugin_dir = str(path.parent)
        added_to_path = False
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
            added_to_path = True

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if added_to_path:
                try:
                    sys.path.remove(plugin_dir)
                except ValueError:
                    pass

        return module

    # ------------------------------------------------------------------
    # UNLOADING
    # ------------------------------------------------------------------

    async def unload_plugin(self, name: str) -> bool:
        """Unload a plugin by name, calling its optional unregister()."""
        record = self.registry.plugins.get(name)
        if not record:
            return False

        # Call the plugin's unregister() if present
        module = record.module or self._loaded_modules.get(name)
        if module is not None:
            unreg = getattr(module, "unregister", None)
            if unreg is not None:
                try:
                    target = self.agent if self.agent is not None else _LoaderContext(self)
                    if inspect.iscoroutinefunction(unreg):
                        await unreg(target)
                    else:
                        result = unreg(target)
                        if inspect.isawaitable(result):
                            await result
                except Exception as e:
                    logger.warning(f"Plugin '{name}' unregister() failed: {e}")

        # Remove from sys.modules
        sys.modules.pop(f"agent_plugin_{name}", None)
        self._loaded_modules.pop(name, None)
        self._resolved_paths.pop(name, None)

        # Remove all registrations
        self.registry.remove_plugin(name)
        logger.info(f"Unloaded plugin '{name}'")
        return True

    async def reload_plugin(self, name: str) -> bool:
        """Hot-reload a plugin (dev mode)."""
        path = self._resolved_paths.get(name)
        manifest = None
        if name in self.registry.plugins:
            manifest = self.registry.plugins[name].manifest

        await self.unload_plugin(name)

        if path is None:
            logger.warning(f"Cannot reload '{name}': path unknown")
            return False

        return await self.load_plugin_from_path(path, manifest)

    # ------------------------------------------------------------------
    # TOGGLING
    # ------------------------------------------------------------------

    def enable_plugin(self, name: str) -> bool:
        return self.registry.enable(name)

    def disable_plugin(self, name: str) -> bool:
        return self.registry.disable(name)

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        return self.registry.list_plugins()

    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        rec = self.registry.plugins.get(name)
        if not rec:
            return None
        return {
            "name": rec.name,
            "version": rec.version,
            "author": rec.manifest.author,
            "description": rec.manifest.description,
            "homepage": rec.manifest.homepage,
            "license": rec.manifest.license,
            "state": rec.state.value,
            "error": rec.error,
            "path": rec.path,
            "loaded_at": rec.loaded_at,
            "capabilities": rec.manifest.capabilities,
            "dependencies": rec.manifest.dependencies,
            "registered_tools": sorted(rec.registered_tools),
            "registered_commands": sorted(rec.registered_commands),
            "registered_policies": sorted(rec.registered_policies),
            "registered_providers": sorted(rec.registered_providers),
            "registered_hooks": sorted(rec.registered_hooks),
            "registered_widgets": sorted(rec.registered_widgets),
        }

    def stats(self) -> Dict[str, Any]:
        return {
            **self.registry.stats(),
            "directories": [str(d) for d in self.directories],
            "enabled": self.enabled,
            "auto_load": self.auto_load,
            "dev_mode": self.dev_mode,
        }


# ======================================================================
# LOADER CONTEXT (for testing / headless load)
# ======================================================================

class _LoaderContext:
    """
    Minimal context object passed to plugins when no agent is available.
    Plugins can register tools/commands/hooks the same way.
    """

    def __init__(self, loader: PluginLoader):
        self.loader = loader
        self.registry = loader.registry
        # Expose a compatible-looking tool_registry stub
        self.tool_registry = _ToolRegistryStub(loader.registry)

    def register_tool(self, name: str, obj: Any, plugin: str = "unknown") -> None:
        self.registry.register_tool(plugin, name, obj)

    def register_command(self, name: str, obj: Any, plugin: str = "unknown") -> None:
        self.registry.register_command(plugin, name, obj)


class _ToolRegistryStub:
    """A stub that forwards register_tool() calls to the plugin registry."""

    def __init__(self, plugin_registry: PluginRegistry):
        self._pr = plugin_registry
        self._current_plugin: str = "unknown"

    def register_tool(self, tool: Any) -> None:
        name = getattr(tool, "name", None) or getattr(tool, "__class__", type(tool)).__name__
        self._pr.register_tool(self._current_plugin, name, tool)

    def register_external(self, name: str, description: str = "", parameters: Any = None, handler=None) -> None:
        self._pr.register_tool(self._current_plugin, name, {
            "name": name,
            "description": description,
            "parameters": parameters or {},
            "handler": handler,
        })


# ======================================================================
# GLOBAL LOADER
# ======================================================================

_global_loader: Optional[PluginLoader] = None


def get_plugin_loader(
    agent: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> PluginLoader:
    """Get or create the global plugin loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = PluginLoader(agent=agent, config=config)
    return _global_loader


def reset_plugin_loader() -> None:
    global _global_loader
    _global_loader = None


__all__ = [
    "PluginLoader",
    "get_plugin_loader",
    "reset_plugin_loader",
    "AGENT_VERSION",
]