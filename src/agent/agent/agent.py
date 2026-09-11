"""
Agent Core - The brain of the CLI Agent.

Orchestrates:
    - LLM provider registry (dynamic model selection)
    - Adaptive temperature (per-prompt complexity)
    - Tool registry (native + MCP + plugin tools)
    - Permission manager (safety gates)
    - Context manager (messages, files, project, compaction)
    - Session manager (persistence, history, state)
    - Storage (database + cache)
    - Planner, subagents, and the main loop
    - Plugin loader (extension system)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.utils.errors import (
    AgentError, ToolExecutionError, PermissionDeniedError,
    handle_exception, format_error, is_retryable,
)
from agent.utils.platform import get_platform

logger = get_logger(__name__)


# ======================================================================
# STATUS
# ======================================================================

class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING = "waiting"
    PAUSED = "paused"
    ERROR = "error"
    RECOVERING = "recovering"
    SHUTTING_DOWN = "shutting_down"


# ======================================================================
# AGENT CONTEXT
# ======================================================================

@dataclass
class AgentContext:
    session_id: str = ""
    conversation_id: str = ""
    turn_count: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    tool_calls: int = 0
    tasks_completed: int = 0
    start_time: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    tools_used: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ======================================================================
# AGENT
# ======================================================================

class Agent:
    """
    Main orchestrator. Every user query flows through `process_query`.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        session: Any,
        context_manager: Any,
        permission_manager: Any,
        workspace: Any,
        database: Any,
        ui: Any = None,
        input_handler: Any = None,
        mcp_client: Any = None,
        cache: Any = None,
        max_turns: int = 50,
        timeout: float = 300.0,
    ):
        # Dependencies
        self.config = config or {}
        self.session = session
        self.context_manager = context_manager
        self.permission_manager = permission_manager
        self.workspace = workspace
        self.database = database
        self.ui = ui
        self.input_handler = input_handler
        self.mcp_client = mcp_client
        self.cache = cache

        # Bounds
        self.max_turns = max_turns
        self.timeout = timeout

        # Subsystems (built in initialize())
        self.llm = None
        self.tool_registry = None
        self.planner = None
        self.loop = None
        self.subagent_manager = None
        self.plugin_loader = None

        # State
        self.status = AgentStatus.IDLE
        self.context = AgentContext()
        self.is_shutting_down = False
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.current_task: Optional[asyncio.Task] = None
        self._thinking_lock = asyncio.Lock()
        self._initialized = False

        # Metrics
        self.metrics = {
            "total_queries": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "total_tool_calls": 0,
            "success_rate": 1.0,
        }

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info("Initializing agent...")

        try:
            # 1. LLM registry — dynamic model selection
            from agent.llm.provider import get_llm_registry
            self.llm = get_llm_registry()

            # Resolve provider + model from config (may auto-select)
            cfg_obj = self._get_config_object()
            if cfg_obj and hasattr(cfg_obj, "resolve_llm"):
                provider, model = cfg_obj.resolve_llm(registry=self.llm)
                self.llm.set_model(model)
                logger.info(f"Agent using model: {provider}/{model}")
            else:
                # Fallback: pick the first available model
                models = self.llm.list_models()
                if models:
                    first = next((m for m in models if m.get("available")), None)
                    if first:
                        self.llm.set_model(first["id"])

            # 2. Tools
            from agent.tools.registry import ToolRegistry
            from agent.tools import (
                TerminalTool, FileSystemTool, SearchTool, GitTool,
                ProcessTool, PatchTool, WebTool, BrowserTool, TaskTool,
                MCPTool, DiagnosticsTool, TodoTool,
            )
            self.tool_registry = ToolRegistry(agent=self)

            tools_cfg = self.config.get("tools", {})
            tool_instances = [
                TerminalTool(self.workspace, tools_cfg),
                FileSystemTool(self.workspace),
                SearchTool(self.workspace),
                GitTool(self.workspace),
                ProcessTool(self.workspace),
                PatchTool(self.workspace),
                WebTool(tools_cfg),
                BrowserTool(tools_cfg.get("browser", {})),
                TaskTool(self),
                DiagnosticsTool(self.workspace),
                TodoTool(self.session),
            ]
            if self.mcp_client is not None:
                tool_instances.append(MCPTool(self.mcp_client))

            enabled = set(tools_cfg.get("enabled") or [])
            disabled = set(tools_cfg.get("disabled") or [])
            for tool in tool_instances:
                if enabled and tool.name not in enabled:
                    continue
                if tool.name in disabled:
                    continue
                await self.tool_registry.register_tool(tool)

            # 3. MCP tool bridging
            if self.mcp_client is not None:
                for tool_def in self.mcp_client.list_tools():
                    fn = tool_def["function"]
                    self.tool_registry.register_external(
                        name=fn["name"],
                        description=fn["description"],
                        parameters=fn["parameters"],
                        handler=lambda name=fn["name"], **kw: self.mcp_client.call_tool(name, kw),
                    )

            # 4. Planner
            from agent.agent.planner import Planner
            self.planner = Planner(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config.get("planner", {}),
            )

            # 5. Loop
            from agent.agent.loop import AgentLoop
            self.loop = AgentLoop(
                agent=self,
                llm=self.llm,
                tool_registry=self.tool_registry,
                planner=self.planner,
                config=self.config.get("loop", {}),
            )

            # 6. Subagents
            from agent.agent.subagent import SubAgentManager
            self.subagent_manager = SubAgentManager(
                agent=self, config=self.config.get("subagent", {}),
            )

            # 7. Plugins
            from agent.plugins.loader import get_plugin_loader
            self.plugin_loader = get_plugin_loader(
                agent=self, config=self.config.get("plugins", {}),
            )
            await self.plugin_loader.initialize()

            # 8. Restore state if it exists
            await self._load_state()

            # 9. Wire permission prompt into input handler
            if self.permission_manager and self.input_handler:
                async def _confirm(request, verdict):
                    msg = f"Allow {request.tool}.{request.action}? [{verdict.risk.value}]"
                    if hasattr(self.input_handler, "get_confirmation"):
                        return await self.input_handler.get_confirmation(msg, default=False)
                    return False
                self.permission_manager.set_confirm_callback(_confirm)

            self._initialized = True
            logger.info("Agent initialized")

        except Exception as e:
            logger.error(f"Agent init failed: {e}", exc_info=True)
            raise AgentError(f"Agent initialization failed: {e}")

    def _get_config_object(self):
        """Best-effort access to the live Config object."""
        try:
            from agent.config.loader import get_config
            return get_config()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a user query through the full agentic pipeline."""
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        async with self._thinking_lock:
            start = time.time()
            try:
                self.status = AgentStatus.THINKING
                self.context.turn_count += 1
                self.context.last_active = time.time()
                self.metrics["total_queries"] += 1

                # Record user message in context + session
                if self.context_manager:
                    await self.context_manager.add_user_message(query, context)
                if self.session:
                    self.session.add_user_message(query)

                # Compact if needed
                if self.context_manager and await self.context_manager.needs_compaction():
                    await self.context_manager.compact()

                # Get live context
                live_context = {}
                if self.context_manager:
                    live_context = await self.context_manager.get_context()

                # Resolve adaptive temperature
                temperature = self._resolve_temperature(query)

                # Run the loop
                result = await self.loop.run(
                    query=query,
                    context=live_context,
                    tool_outputs=(
                        self.context_manager.get_recent_tool_outputs(5)
                        if self.context_manager else []
                    ),
                    max_turns=self.max_turns,
                )

                # Save assistant response
                response_text = result.get("response", "")
                if self.context_manager and response_text:
                    await self.context_manager.add_assistant_message(response_text)
                if self.session and response_text:
                    self.session.add_assistant_message(response_text)

                # Update metrics
                elapsed = time.time() - start
                tokens = result.get("context", {}).get("tokens_used", 0)
                self.context.tokens_used += tokens
                self.metrics["total_tokens"] += tokens
                self.context.tool_calls += result.get("tool_calls", 0)
                self.metrics["total_tool_calls"] += result.get("tool_calls", 0)

                n = self.metrics["total_queries"]
                self.metrics["avg_response_time"] = (
                    (self.metrics["avg_response_time"] * (n - 1) + elapsed) / n
                )

                self.context.tasks_completed += 1 if result.get("success") else 0
                self.status = AgentStatus.IDLE

                # Persist
                if self.session:
                    self.session.set_status("idle")
                    await self._autosave()

                await self._trigger_event("on_query_complete", {
                    "query": query, "result": result, "duration": elapsed,
                })

                return {
                    "success": result.get("success", True),
                    "response": response_text,
                    "iteration": result.get("iteration", 0),
                    "tool_calls": result.get("tool_calls", 0),
                    "duration": elapsed,
                    "context": result.get("context", {}),
                }

            except PermissionDeniedError as e:
                self.status = AgentStatus.ERROR
                return {"success": False, "error": format_error(e), "denied": True}
            except Exception as e:
                self.status = AgentStatus.ERROR
                self.context.errors.append(str(e))
                logger.error(f"Query failed: {e}", exc_info=True)
                return {"success": False, "error": format_error(e)}

    # ------------------------------------------------------------------
    # TOOL EXECUTION (called by the loop)
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        require_permission: bool = True,
    ) -> Dict[str, Any]:
        """Execute a tool through the permission gate + registry."""
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        # Permission gate
        if require_permission and self.permission_manager:
            action = self._infer_action(tool_name, params)
            allowed, reason = await self.permission_manager.check_permission(
                tool_name=tool_name,
                params=params,
                context={"action": action},
            )
            if not allowed:
                return {
                    "success": False,
                    "error": reason or "permission denied",
                    "permission_denied": True,
                    "tool": tool_name,
                }

        # Execute
        try:
            result = await self.tool_registry.execute(tool_name, params)
        except Exception as e:
            wrapped = handle_exception(e, reraise=False)
            return {
                "success": False,
                "error": format_error(wrapped),
                "tool": tool_name,
            }

        # Record in context + session
        self.context.tools_used[tool_name] = self.context.tools_used.get(tool_name, 0) + 1
        if self.context_manager:
            await self.context_manager.add_tool_output(tool_name, params, result)
        if self.session:
            self.session.add_tool_message(
                content=json.dumps(result, default=str)[:10_000],
                tool_name=tool_name,
            )
        if self.database and self.session:
            try:
                await self.database.save_tool_output(
                    session_id=self.session.id,
                    tool=tool_name,
                    params=params,
                    result=result,
                    success=result.get("success", True),
                )
            except Exception as e:
                logger.debug(f"DB tool output save failed: {e}")

        await self._trigger_event("on_tool_executed", {
            "tool": tool_name, "params": params, "result": result,
        })
        return result

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _resolve_temperature(self, query: str) -> float:
        cfg_obj = self._get_config_object()
        try:
            if cfg_obj and hasattr(cfg_obj, "resolve_temperature"):
                return cfg_obj.resolve_temperature(
                    prompt=query,
                    provider=self.llm.get_current_provider() if self.llm else None,
                    model=self.llm.get_current_model() if self.llm else None,
                )
        except Exception as e:
            logger.debug(f"Temperature resolution failed: {e}")
        return 0.3

    def _infer_action(self, tool: str, params: Dict[str, Any]) -> str:
        t = tool.lower()
        if "read" in t:
            return "read"
        if "write" in t or "edit" in t:
            return "write"
        if "delete" in t or "remove" in t:
            return "delete"
        if t in ("terminal", "shell") or t.startswith("mcp__aws__"):
            if "aws" in t:
                return "aws"
            return "execute"
        if t.startswith("mcp__"):
            return "mcp"
        if "git" in t:
            return "git"
        if t == "aws" or t.startswith("aws_"):
            return "aws"
        return "execute"

    async def _autosave(self) -> None:
        if self.session and self.database:
            try:
                await self.database.save_session(self.session.to_dict())
            except Exception as e:
                logger.debug(f"Autosave failed: {e}")

    async def _load_state(self) -> None:
        if self.database and self.session:
            try:
                state = await self.database.load_agent_state(self.session.id)
                if state:
                    ctx = state.get("context") or {}
                    for k, v in ctx.items():
                        if hasattr(self.context, k):
                            setattr(self.context, k, v)
                    self.metrics.update(state.get("metrics") or {})
                    logger.debug("Restored agent state")
            except Exception as e:
                logger.debug(f"State restore failed: {e}")

    async def save_state(self) -> None:
        if self.database and self.session:
            try:
                await self.database.save_agent_state({
                    "session_id": self.session.id,
                    "context": self.context.__dict__,
                    "metrics": self.metrics,
                    "status": self.status.value,
                    "saved_at": time.time(),
                })
            except Exception as e:
                logger.debug(f"State save failed: {e}")

    # ------------------------------------------------------------------
    # STATUS / METRICS
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        current_model = None
        current_provider = None
        if self.llm:
            current_model = self.llm.get_current_model()
            current_provider = self.llm.get_current_provider()
        return {
            "status": self.status.value,
            "session_id": self.session.id if self.session else "",
            "turn_count": self.context.turn_count,
            "tasks_completed": self.context.tasks_completed,
            "tokens_used": self.context.tokens_used,
            "cost": self.context.cost,
            "uptime": time.time() - self.context.start_time,
            "model": current_model,
            "provider": current_provider,
            "metrics": dict(self.metrics),
            "tools_used": dict(self.context.tools_used),
            "tool_count": len(self.tool_registry.list_tools()) if self.tool_registry else 0,
            "errors": len(self.context.errors),
        }

    def is_working(self) -> bool:
        return self.status in (
            AgentStatus.THINKING, AgentStatus.PLANNING, AgentStatus.EXECUTING,
        )

    def is_healthy(self) -> bool:
        if self.llm and not self.llm.is_healthy():
            return False
        if len(self.context.errors) > 20:
            return False
        return True

    async def recover(self) -> bool:
        self.status = AgentStatus.RECOVERING
        try:
            if self.llm:
                await self.llm.reconnect_all()
            if self.loop:
                await self.loop.reset()
            self.context.errors = self.context.errors[-10:]
            self.status = AgentStatus.IDLE
            return True
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            self.status = AgentStatus.ERROR
            return False

    # ------------------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------------------

    def add_event_handler(self, event: str, handler: Callable) -> None:
        self.event_handlers.setdefault(event, []).append(handler)

    async def _trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        for handler in self.event_handlers.get(event, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Event handler failed for {event}: {e}")

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        self.is_shutting_down = True
        self.status = AgentStatus.SHUTTING_DOWN
        logger.info("Shutting down agent...")

        try:
            if self.current_task and not self.current_task.done():
                self.current_task.cancel()
                try:
                    await self.current_task
                except asyncio.CancelledError:
                    pass

            await self.save_state()

            if self.session and self.database:
                try:
                    await self.database.save_session(self.session.to_dict())
                except Exception:
                    pass

            if self.subagent_manager:
                await self.subagent_manager.shutdown()
            if self.plugin_loader:
                await self.plugin_loader.shutdown()
            if self.mcp_client:
                await self.mcp_client.shutdown()

            # Shutdown any tools that need it
            if self.tool_registry:
                for tool in self.tool_registry.tools.values():
                    if hasattr(tool, "shutdown"):
                        try:
                            await tool.shutdown()
                        except Exception:
                            pass

            if self.llm:
                await self.llm.reconnect_all()

            logger.info("Agent shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")