"""
Agent Module - Core Agent Implementation.

Exports:
    Agent           -- Main orchestrator class
    AgentStatus     -- Agent status enum
    AgentContext    -- Runtime context dataclass
    Planner         -- Task planner
    AgentLoop       -- Execution loop
    SubAgentManager -- Sub-agent management
"""

from agent.agent.agent import Agent, AgentStatus, AgentContext
from agent.agent.planner import Planner, Plan, Task, TaskStatus, Priority
from agent.agent.loop import AgentLoop, LoopState, LoopContext
from agent.agent.subagent import (
    SubAgentManager,
    SubAgent,
    SubAgentRole,
    SubAgentConfig,
    BUILTIN_SUBAGENTS,
)

# Backcompat aliases for legacy imports.
SubAgentTask = SubAgent
SubAgentType = SubAgentRole
SubAgentStatus = str

__all__ = [
    "Agent",
    "AgentStatus",
    "AgentContext",
    "Planner",
    "Plan",
    "Task",
    "TaskStatus",
    "Priority",
    "AgentLoop",
    "LoopState",
    "LoopContext",
    "SubAgentManager",
    "SubAgent",
    "SubAgentRole",
    "SubAgentConfig",
    "BUILTIN_SUBAGENTS",
    "SubAgentTask",
    "SubAgentType",
    "SubAgentStatus",
]