"""
CLI Agent - Advanced Agentic AI CLI

Top-level package exports.
"""

from agent.agent import Agent
from agent.config import Config, get_config, load_config
from agent.context import ContextManager
from agent.permissions import PermissionManager
from agent.session import SessionManager, Session, SessionState
from agent.storage import Database, Cache
from agent.project import WorkspaceManager
from agent.mcp import MCPClient
from agent.plugins import PluginLoader
from agent.tools import ToolRegistry, BaseTool
from agent.llm import get_llm_registry

__all__ = [
    "Agent",
    "Config",
    "get_config",
    "load_config",
    "ContextManager",
    "PermissionManager",
    "SessionManager",
    "Session",
    "SessionState",
    "Database",
    "Cache",
    "WorkspaceManager",
    "MCPClient",
    "PluginLoader",
    "ToolRegistry",
    "BaseTool",
    "get_llm_registry",
]

__version__ = "1.0.0"