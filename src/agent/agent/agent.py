"""
Agent Core - The brain of the CLI Agent.

Orchestrates:
    - LLM provider registry (user-connected model)
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
from typing import Any, Callable, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.utils.errors import (
    AgentError,
    PermissionDeniedError,
    format_error,
    handle_exception,
)
from agent.utils.redact import redact
from agent.agent.subagent import SubAgentRole

logger = get_logger(__name__)


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


class Agent:
    """Main orchestrator. Every user query flows through `process_query`."""

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

        self.max_turns = max_turns
        self.timeout = timeout

        self.llm = None
        self.tool_registry = None
        self.planner = None
        self.loop = None
        self.subagent_manager = None
        self.plugin_loader = None

        self.status = AgentStatus.IDLE
        self.context = AgentContext()
        self.is_shutting_down = False
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.current_task: Optional[asyncio.Task] = None
        self._thinking_lock = asyncio.Lock()
        self._initialized = False

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
            from agent.llm.provider import get_llm_registry

            self.llm = get_llm_registry()

            # If the user has saved a runtime connection, it was already
            # installed by the TUI bootstrap; nothing to do here.
            # Otherwise the registry stays empty until /connect is used.
            if not self.llm.has_model():
                logger.warning(
                    "No model connected. The agent will run without a model "
                    "until the user pastes one via the TUI /connect panel."
                )

            # Tools
            from agent.tools.registry import ToolRegistry
            from agent.tools import (
                TerminalTool, FileSystemTool, SearchTool, GitTool,
                ProcessTool, PatchTool, WebTool, BrowserTool,
                RealtimeBrowserTool, TaskTool,
                MCPTool, DiagnosticsTool, TodoTool,
                BashTool, ReadTool, WriteTool, EditTool, ApplyPatchTool,
                GrepTool, GlobTool, WebFetchTool, WebSearchTool,
                TodoWriteTool, TodoReadTool, SkillTool, QuestionTool,
                LspTool, AWSHelperTool,
            )

            self.tool_registry = ToolRegistry(agent=self)
            tools_cfg = self.config.get("tools", {}) or {}
            tool_instances = [
                TerminalTool(self.workspace, tools_cfg),
                BashTool(self.workspace, tools_cfg),
                FileSystemTool(self.workspace),
                ReadTool(self.workspace),
                WriteTool(self.workspace),
                EditTool(self.workspace),
                ApplyPatchTool(self.workspace),
                SearchTool(self.workspace),
                GrepTool(self.workspace),
                GlobTool(self.workspace),
                GitTool(self.workspace),
                ProcessTool(self.workspace),
                PatchTool(self.workspace),
                WebTool(tools_cfg),
                WebFetchTool(tools_cfg),
                WebSearchTool(tools_cfg),
                BrowserTool(tools_cfg.get("browser", {})),
                RealtimeBrowserTool(tools_cfg.get("realtime_browser", {})),
                TaskTool(self),
                DiagnosticsTool(self.workspace),
                TodoTool(self.session),
                TodoWriteTool(self.session),
                TodoReadTool(self.session),
                SkillTool(self.workspace),
                QuestionTool(self.input_handler),
                LspTool(self.workspace, tools_cfg),
                AWSHelperTool(self, self.workspace),
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

            if self.mcp_client is not None:
                for tool_def in self.mcp_client.list_tools():
                    fn = tool_def["function"]
                    self.tool_registry.register_external(
                        name=fn["name"],
                        description=fn["description"],
                        parameters=fn["parameters"],
                        handler=lambda name=fn["name"], **kw: self.mcp_client.call_tool(name, kw),
                    )

            # Planner
            from agent.agent.planner import Planner
            self.planner = Planner(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config.get("planner", {}) or {},
                fallback_chain=getattr(self, "_fallback_chain", None),
            )

            # Loop
            from agent.agent.loop import AgentLoop
            self.loop = AgentLoop(
                agent=self,
                llm=self.llm,
                tool_registry=self.tool_registry,
                planner=self.planner,
                config=self.config.get("loop", {}) or {},
            )

            # Subagents
            from agent.agent.subagent import SubAgentManager
            self.subagent_manager = SubAgentManager(
                agent=self, config=self.config.get("subagent", {}) or {},
            )
            await self.subagent_manager.initialize()

            # Plugins
            from agent.plugins.loader import get_plugin_loader
            self.plugin_loader = get_plugin_loader(
                agent=self, config=self.config.get("plugins", {}) or {},
            )
            await self.plugin_loader.initialize()

            # Restore state
            await self._load_state()

            # Wire permission prompt into input handler ONLY if no callback set
            if (
                self.permission_manager
                and getattr(self.permission_manager, "confirm_callback", None) is None
                and self.input_handler
            ):
                async def _confirm(request, verdict):
                    msg = f"Allow {request.tool}.{request.action}? [{verdict.risk.value}]"
                    if hasattr(self.input_handler, "get_confirmation"):
                        return await self.input_handler.get_confirmation(msg, default=False)
                    return False
                self.permission_manager.set_confirm_callback(_confirm)

            self._initialized = True
            logger.info("Agent initialized")

        except Exception as e:
            logger.error("Agent init failed: %s", e, exc_info=True)
            raise AgentError(f"Agent initialization failed: {e}")

    def _get_config_object(self):
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
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        if self.subagent_manager:
            subagent_calls = self.subagent_manager.parse_subagent_invocation(query)
            if subagent_calls:
                return await self._handle_subagent_invocations(subagent_calls, context, query)

        return await self._run_query_pipeline(query, context, record_user=True)

    async def _run_query_pipeline(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        record_user: bool = True,
    ) -> Dict[str, Any]:
        """The single execution path shared by top-level queries and follow-ups."""
        async with self._thinking_lock:
            start = time.time()
            try:
                self.status = AgentStatus.THINKING
                self.context.turn_count += 1
                self.context.last_active = time.time()
                self.metrics["total_queries"] += 1

                if record_user and self.context_manager:
                    await self.context_manager.add_user_message(query, context)
                if record_user and self.session:
                    self.session.add_user_message(query)

                if self.context_manager and await self.context_manager.needs_compaction():
                    await self.context_manager.compact()

                live_context: Dict[str, Any] = {}
                if self.context_manager:
                    live_context = await self.context_manager.get_context()

                self._resolve_temperature(query)

                result = await self.loop.run(
                    query=query,
                    context=live_context,
                    tool_outputs=(
                        self.context_manager.get_recent_tool_outputs(5)
                        if self.context_manager else []
                    ),
                    max_turns=self.max_turns,
                )

                response_text = result.get("response", "")
                if self.context_manager and response_text:
                    await self.context_manager.add_assistant_message(response_text)
                if self.session and response_text:
                    self.session.add_assistant_message(response_text)

                elapsed = time.time() - start
                ctx_summary = result.get("context", {}) or {}
                tokens = ctx_summary.get("tokens_used", 0)
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

                if self.session:
                    self.session.set_status("idle")
                    await self._autosave()

                await self._trigger_event(
                    "on_query_complete",
                    {"query": query, "result": result, "duration": elapsed},
                )

                result_out = {
                    "success": result.get("success", True),
                    "response": response_text,
                    "iteration": result.get("iteration", 0),
                    "tool_calls": result.get("tool_calls", 0),
                    "duration": elapsed,
                    "context": ctx_summary,
                }
                if not result_out["success"] and result.get("error"):
                    result_out["error"] = str(result.get("error"))
                return result_out

            except PermissionDeniedError as e:
                self.status = AgentStatus.IDLE
                return {"success": False, "error": format_error(e), "denied": True}
            except Exception as e:
                self.status = AgentStatus.ERROR
                self.context.errors.append(str(e))
                logger.error("Query failed: %s", e, exc_info=True)
                return {"success": False, "error": format_error(e)}

    async def _handle_subagent_invocations(
        self,
        subagent_calls: List[Any],
        context: Optional[Dict[str, Any]],
        original_query: str,
    ) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []

        role_map = {
            "general": SubAgentRole.GENERAL,
            "explore": SubAgentRole.EXPLORE,
            "scout": SubAgentRole.SCOUT,
        }

        for agent_name, sub_query in subagent_calls:
            role = role_map.get(agent_name.lower())
            if not role:
                results.append(
                    {
                        "subagent": agent_name,
                        "success": False,
                        "error": (
                            f"Unknown subagent: {agent_name}. "
                            f"Available: {', '.join(role_map.keys())}"
                        ),
                    }
                )
                continue

            result = await self.subagent_manager.invoke_subagent(
                role=role, query=sub_query, context=context,
            )
            results.append(result)

        remaining_query = self.subagent_manager.strip_subagent_invocations(original_query)

        if remaining_query.strip():
            main_result = await self._run_query_pipeline(
                remaining_query, context, record_user=True,
            )
            results.append({"subagent": "main", "success": main_result.get("success", True),
                            "result": main_result.get("response", ""),
                            "main": main_result})

        combined_response = "\n\n".join(
            f"--- @{r.get('subagent', 'main')} ---\n"
            f"{r.get('result', r.get('error', ''))}"
            for r in results
        )

        return {
            "success": all(r.get("success", True) for r in results),
            "response": combined_response,
            "subagent_results": results,
            "iteration": 1,
            "tool_calls": sum(
                r.get("tool_calls", 0) for r in results if isinstance(r, dict)
            ),
            "context": {"tokens_used": 0},
        }

    # ------------------------------------------------------------------
    # TOOL EXECUTION
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        require_permission: bool = True,
    ) -> Dict[str, Any]:
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        if self.tool_registry and not self.tool_registry.has_tool(tool_name):
            from difflib import get_close_matches as _gcm
            available = list(self.tool_registry.tools or {}) + list(
                getattr(self.tool_registry, "_external", {}) or {}
            )
            suggestion = _gcm(tool_name, available, n=1)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            logger.warning("Unknown tool '%s' requested by model.%s", tool_name, hint)
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}.{hint}",
                "tool": tool_name,
            }

        if require_permission and self.permission_manager:
            action = self._infer_action(tool_name, params)
            try:
                allowed, reason = await self.permission_manager.check_permission(
                    tool_name=tool_name,
                    params=params,
                    context={"action": action},
                )
            except PermissionDeniedError as e:
                logger.warning("Permission denied for %s: %s", tool_name, e)
                return {
                    "success": False,
                    "error": str(e),
                    "permission_denied": True,
                    "tool": tool_name,
                }
            if not allowed:
                logger.warning("Permission denied for %s: %s", tool_name, reason)
                return {
                    "success": False,
                    "error": reason or "permission denied",
                    "permission_denied": True,
                    "tool": tool_name,
                }

        try:
            result = await self.tool_registry.execute(tool_name, params)
        except Exception as e:
            wrapped = handle_exception(e, reraise=False)
            logger.error("Tool %s execution failed: %s", tool_name, e, exc_info=True)
            return {
                "success": False,
                "error": format_error(wrapped),
                "tool": tool_name,
            }

        try:
            result = redact(result)
        except Exception:
            pass

        self.context.tools_used[tool_name] = self.context.tools_used.get(tool_name, 0) + 1
        if self.context_manager:
            try:
                await self.context_manager.add_tool_output(tool_name, params, result)
            except Exception as exc:
                logger.debug("add_tool_output failed: %s", exc)
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
                logger.debug("DB tool output save failed: %s", e)

        await self._trigger_event(
            "on_tool_executed",
            {"tool": tool_name, "params": params, "result": result},
        )
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
            logger.debug("Temperature resolution failed: %s", e)
        return 0.3

    def _infer_action(self, tool: str, params: Dict[str, Any]) -> str:
        t = tool.lower()
        if t.startswith("mcp__aws__") or t == "aws" or t.startswith("aws_"):
            return "aws"
        if t.startswith("mcp__"):
            return "mcp"
        if "read" in t or "list" in t or "search" in t or "grep" in t or "glob" in t:
            return "read"
        if "write" in t or "edit" in t or "patch" in t:
            return "write"
        if "delete" in t or "remove" in t:
            return "delete"
        if t in ("terminal", "shell", "bash", "process"):
            return "execute"
        if "git" in t:
            return "git"
        return "execute"

    async def _autosave(self) -> None:
        if self.session and self.database:
            try:
                await self.database.save_session(self.session.to_dict())
            except Exception as e:
                logger.debug("Autosave failed: %s", e)

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
                logger.debug("State restore failed: %s", e)

    async def save_state(self) -> None:
        if self.database and self.session:
            try:
                await self.database.save_agent_state(
                    {
                        "session_id": self.session.id,
                        "context": self.context.__dict__,
                        "metrics": self.metrics,
                        "status": self.status.value,
                        "saved_at": time.time(),
                    }
                )
            except Exception as e:
                logger.debug("State save failed: %s", e)

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
            logger.error("Recovery failed: %s", e)
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
                logger.error("Event handler failed for %s: %s", event, e)

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

            if self.tool_registry:
                for tool in self.tool_registry.tools.values():
                    if hasattr(tool, "shutdown"):
                        try:
                            await tool.shutdown()
                        except Exception:
                            pass

            logger.info("Agent shutdown complete")
        except Exception as e:
            logger.error("Shutdown error: %s", e)