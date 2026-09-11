"""
Tools Module - All agent capabilities.

Exports every built-in tool. Import from this module to register them
with the ToolRegistry.
"""

from agent.tools.registry import ToolRegistry, BaseTool
from agent.tools.terminal import TerminalTool
from agent.tools.filesystem import FileSystemTool
from agent.tools.search import SearchTool
from agent.tools.git import GitTool
from agent.tools.proccess import ProcessTool
from agent.tools.patch import PatchTool
from agent.tools.web import WebTool
from agent.tools.browser import BrowserTool
from agent.tools.task import TaskTool
from agent.tools.mcp import MCPTool
from agent.tools.diagnostics import DiagnosticsTool
from agent.tools.todo import TodoTool


__all__ = [
    "ToolRegistry",
    "BaseTool",
    "TerminalTool",
    "FileSystemTool",
    "SearchTool",
    "GitTool",
    "ProcessTool",
    "PatchTool",
    "WebTool",
    "BrowserTool",
    "TaskTool",
    "MCPTool",
    "DiagnosticsTool",
    "TodoTool",
]