"""
Plugins Module - Plugin discovery, loading, and management

Exports:
    PluginLoader         — Plugin discovery and loading
    PluginRegistry       — Plugin registry
    PluginRecord         — Plugin runtime record
    PluginManifest       — Plugin metadata
    PluginState          — Plugin state enum
    PluginCapability     — Plugin capability enum
    get_plugin_loader    — Get global loader
    get_plugin_registry  — Get global registry
    reset_plugin_loader  — Reset global loader
"""

from agent.plugins.loader import (
    PluginLoader,
    get_plugin_loader,
    reset_plugin_loader,
    AGENT_VERSION,
)
from agent.plugins.registry import (
    PluginRegistry,
    PluginRecord,
    PluginManifest,
    PluginState,
    PluginCapability,
    get_plugin_registry,
)

__all__ = [
    "PluginLoader",
    "get_plugin_loader",
    "reset_plugin_loader",
    "AGENT_VERSION",
    "PluginRegistry",
    "PluginRecord",
    "PluginManifest",
    "PluginState",
    "PluginCapability",
    "get_plugin_registry",
]