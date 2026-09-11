"""
Plugin Registry - Tracks installed plugins and their capabilities.

Responsibilities:
    - Maintain a manifest of every loaded plugin
    - Track metadata (name, version, author, capabilities)
    - Detect conflicts (two plugins registering the same tool/command)
    - Provide lookup: get_tool("jira"), get_policy("myrule")
    - Support enable/disable without unloading code
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# ENUMS & DATA MODELS
# ======================================================================

class PluginState(str, Enum):
    DISCOVERED = "discovered"
    LOADING = "loading"
    LOADED = "loaded"
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"
    UNLOADED = "unloaded"


class PluginCapability(str, Enum):
    TOOL = "tool"
    COMMAND = "command"
    POLICY = "policy"
    LLM_PROVIDER = "llm_provider"
    CONTEXT_SOURCE = "context_source"
    UI_WIDGET = "ui_widget"
    HOOK = "hook"


@dataclass
class PluginManifest:
    """Metadata declared by a plugin"""
    name: str
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    homepage: str = ""
    license: str = ""
    min_agent_version: str = "0.0.0"
    max_agent_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    python_requires: str = ">=3.9"
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class RegisteredItem:
    """A single thing a plugin registered"""
    kind: PluginCapability
    name: str
    plugin: str
    obj: Any
    registered_at: float = field(default_factory=time.time)


@dataclass
class PluginRecord:
    """Full record for a loaded plugin"""
    manifest: PluginManifest
    module: Any = None
    path: Optional[str] = None
    state: PluginState = PluginState.DISCOVERED
    error: Optional[str] = None
    loaded_at: float = 0.0
    # Everything this plugin contributed
    registered_tools: Set[str] = field(default_factory=set)
    registered_commands: Set[str] = field(default_factory=set)
    registered_policies: Set[str] = field(default_factory=set)
    registered_providers: Set[str] = field(default_factory=set)
    registered_hooks: Set[str] = field(default_factory=set)
    registered_widgets: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def is_active(self) -> bool:
        return self.state == PluginState.ENABLED

    def total_registrations(self) -> int:
        return (
            len(self.registered_tools)
            + len(self.registered_commands)
            + len(self.registered_policies)
            + len(self.registered_providers)
            + len(self.registered_hooks)
            + len(self.registered_widgets)
        )


# ======================================================================
# REGISTRY
# ======================================================================

class PluginRegistry:
    """
    Central registry for all plugins and their contributions.
    """

    def __init__(self):
        self.plugins: Dict[str, PluginRecord] = {}

        # Global maps: name → RegisteredItem (for fast lookup)
        self._tools: Dict[str, RegisteredItem] = {}
        self._commands: Dict[str, RegisteredItem] = {}
        self._policies: Dict[str, RegisteredItem] = {}
        self._providers: Dict[str, RegisteredItem] = {}
        self._hooks: Dict[str, List[RegisteredItem]] = {}
        self._widgets: Dict[str, RegisteredItem] = {}

        # Event listeners
        self._listeners: List[Callable[[str, Dict[str, Any]], None]] = []

    # ------------------------------------------------------------------
    # PLUGIN LIFECYCLE
    # ------------------------------------------------------------------

    def add_plugin(self, record: PluginRecord) -> None:
        """Register a plugin record (called by the loader)."""
        if record.name in self.plugins:
            logger.warning(f"Plugin '{record.name}' already registered; overwriting")
        self.plugins[record.name] = record
        self._emit("plugin_added", {"name": record.name, "version": record.version})

    def remove_plugin(self, name: str) -> bool:
        """Remove a plugin and all its registrations."""
        record = self.plugins.pop(name, None)
        if not record:
            return False

        # Remove tools
        for tool_name in record.registered_tools:
            self._tools.pop(tool_name, None)

        # Remove commands
        for cmd_name in record.registered_commands:
            self._commands.pop(cmd_name, None)

        # Remove policies
        for pol_name in record.registered_policies:
            self._policies.pop(pol_name, None)

        # Remove providers
        for prov_name in record.registered_providers:
            self._providers.pop(prov_name, None)

        # Remove widgets
        for w_name in record.registered_widgets:
            self._widgets.pop(w_name, None)

        # Remove hooks
        for hook_name in record.registered_hooks:
            self._hooks[hook_name] = [
                h for h in self._hooks.get(hook_name, [])
                if h.plugin != name
            ]

        self._emit("plugin_removed", {"name": name})
        logger.info(f"Plugin '{name}' removed from registry")
        return True

    def enable(self, name: str) -> bool:
        record = self.plugins.get(name)
        if not record:
            return False
        record.state = PluginState.ENABLED
        self._emit("plugin_enabled", {"name": name})
        return True

    def disable(self, name: str) -> bool:
        record = self.plugins.get(name)
        if not record:
            return False
        record.state = PluginState.DISABLED
        self._emit("plugin_disabled", {"name": name})
        return True

    def set_error(self, name: str, error: str) -> None:
        record = self.plugins.get(name)
        if record:
            record.state = PluginState.ERROR
            record.error = error

    # ------------------------------------------------------------------
    # REGISTRATION API (called by plugins via agent)
    # ------------------------------------------------------------------

    def register_tool(self, plugin: str, name: str, obj: Any) -> None:
        self._register(plugin, PluginCapability.TOOL, name, obj, self._tools)

    def register_command(self, plugin: str, name: str, obj: Any) -> None:
        self._register(plugin, PluginCapability.COMMAND, name, obj, self._commands)

    def register_policy(self, plugin: str, name: str, obj: Any) -> None:
        self._register(plugin, PluginCapability.POLICY, name, obj, self._policies)

    def register_provider(self, plugin: str, name: str, obj: Any) -> None:
        self._register(plugin, PluginCapability.LLM_PROVIDER, name, obj, self._providers)

    def register_widget(self, plugin: str, name: str, obj: Any) -> None:
        self._register(plugin, PluginCapability.UI_WIDGET, name, obj, self._widgets)

    def register_hook(self, plugin: str, event: str, callback: Callable) -> None:
        item = RegisteredItem(
            kind=PluginCapability.HOOK,
            name=event,
            plugin=plugin,
            obj=callback,
        )
        self._hooks.setdefault(event, []).append(item)
        record = self.plugins.get(plugin)
        if record:
            record.registered_hooks.add(event)
        logger.debug(f"Plugin '{plugin}' registered hook '{event}'")

    def _register(
        self,
        plugin: str,
        kind: PluginCapability,
        name: str,
        obj: Any,
        target: Dict[str, RegisteredItem],
    ) -> None:
        if name in target:
            existing = target[name]
            logger.warning(
                f"Plugin '{plugin}' tried to register {kind.value} '{name}' "
                f"but it's already registered by plugin '{existing.plugin}'"
            )
            return

        target[name] = RegisteredItem(
            kind=kind,
            name=name,
            plugin=plugin,
            obj=obj,
        )

        record = self.plugins.get(plugin)
        if record:
            if kind == PluginCapability.TOOL:
                record.registered_tools.add(name)
            elif kind == PluginCapability.COMMAND:
                record.registered_commands.add(name)
            elif kind == PluginCapability.POLICY:
                record.registered_policies.add(name)
            elif kind == PluginCapability.LLM_PROVIDER:
                record.registered_providers.add(name)
            elif kind == PluginCapability.UI_WIDGET:
                record.registered_widgets.add(name)

        logger.debug(f"Plugin '{plugin}' registered {kind.value} '{name}'")

    # ------------------------------------------------------------------
    # LOOKUP
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[Any]:
        item = self._tools.get(name)
        return item.obj if item else None

    def get_command(self, name: str) -> Optional[Any]:
        item = self._commands.get(name)
        return item.obj if item else None

    def get_policy(self, name: str) -> Optional[Any]:
        item = self._policies.get(name)
        return item.obj if item else None

    def get_provider(self, name: str) -> Optional[Any]:
        item = self._providers.get(name)
        return item.obj if item else None

    def get_widget(self, name: str) -> Optional[Any]:
        item = self._widgets.get(name)
        return item.obj if item else None

    def get_hooks(self, event: str) -> List[Callable]:
        return [h.obj for h in self._hooks.get(event, [])]

    # ------------------------------------------------------------------
    # LISTING
    # ------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": rec.name,
                "version": rec.version,
                "author": rec.manifest.author,
                "description": rec.manifest.description,
                "state": rec.state.value,
                "error": rec.error,
                "capabilities": rec.manifest.capabilities,
                "registrations": rec.total_registrations(),
                "loaded_at": rec.loaded_at,
            }
            for rec in self.plugins.values()
        ]

    def list_tools(self) -> List[str]:
        return sorted(self._tools.keys())

    def list_commands(self) -> List[str]:
        return sorted(self._commands.keys())

    def list_policies(self) -> List[str]:
        return sorted(self._policies.keys())

    def list_providers(self) -> List[str]:
        return sorted(self._providers.keys())

    def list_hooks(self) -> List[str]:
        return sorted(self._hooks.keys())

    def list_widgets(self) -> List[str]:
        return sorted(self._widgets.keys())

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "total_plugins": len(self.plugins),
            "enabled": sum(1 for p in self.plugins.values() if p.is_active),
            "disabled": sum(1 for p in self.plugins.values() if p.state == PluginState.DISABLED),
            "errors": sum(1 for p in self.plugins.values() if p.state == PluginState.ERROR),
            "tools": len(self._tools),
            "commands": len(self._commands),
            "policies": len(self._policies),
            "providers": len(self._providers),
            "widgets": len(self._widgets),
            "hooks": sum(len(v) for v in self._hooks.values()),
        }

    # ------------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------------

    def add_listener(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        self._listeners.append(fn)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        for fn in self._listeners:
            try:
                fn(event, data)
            except Exception as e:
                logger.warning(f"Registry listener failed: {e}")

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<PluginRegistry plugins={s['total_plugins']} "
            f"tools={s['tools']} commands={s['commands']} "
            f"policies={s['policies']} providers={s['providers']}>"
        )


# ======================================================================
# GLOBAL REGISTRY
# ======================================================================

_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_plugin_registry() -> None:
    global _global_registry
    _global_registry = None


__all__ = [
    "PluginRegistry",
    "PluginRecord",
    "PluginManifest",
    "PluginState",
    "PluginCapability",
    "RegisteredItem",
    "get_plugin_registry",
    "reset_plugin_registry",
]