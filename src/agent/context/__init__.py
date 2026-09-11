"""
Context Module - Conversation context management

Exports:
    ContextManager  — Central context orchestrator
    ContextMessage  — A message in the conversation
    ToolOutput      — A tool execution result
    ContextStats    — Context statistics
    FileContext     — File content management
    ProjectContext  — Project-level context
    Compactor       — Context compaction
"""

from agent.context.manager import ContextManager, ContextMessage, ToolOutput, ContextStats
from agent.context.files import FileContext
from agent.context.project import ProjectContext
from agent.context.compaction import Compactor

__all__ = [
    "ContextManager",
    "ContextMessage",
    "ToolOutput",
    "ContextStats",
    "FileContext",
    "ProjectContext",
    "Compactor",
]