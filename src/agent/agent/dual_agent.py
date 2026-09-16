"""
Dual Agent Architecture
=======================

Two primary agents with distinct roles:
- PlanAgent:  Read-only planning
- BuildAgent: Full tool access

Both agents share the LLM registry that the user connects via the TUI.
There is no built-in model catalog; the runtime connection is the sole
source of providers and the current model.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from agent.utils.logging import get_logger
from agent.utils.errors import (
    AgentError,
    PermissionDeniedError,
    format_error,
    handle_exception,
)
from agent.utils.redact import redact
from agent.agent.loop import AgentLoop
from agent.agent.planner import Planner
from agent.tools.registry import ToolRegistry, BaseTool
from agent.agent.subagent import SubAgentManager
from agent.plugins.loader import get_plugin_loader

logger = get_logger(__name__)


class AgentRole(str, Enum):
    PLAN = "plan"
    BUILD = "build"


# Tools a read-only agent is allowed to register.
READ_ONLY_TOOL_NAMES: Set[str] = {
    "read", "grep", "glob", "list", "search",
    "git", "webfetch", "websearch", "web", "diagnostics", "lsp",
    "todo", "todo_read", "todo_write", "skill", "question",
    # AWS reads are safe in plan mode; the aws helper rejects mutating
    # CLI operations when the plan role is active (see _infer_action).
    "aws", "mcp__aws__call_aws",
}


class BaseAgent:
    """Base class for PlanAgent and BuildAgent."""

    def __init__(
        self,
        role: AgentRole,
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
        self.role = role
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

        self.status = "idle"
        self.context: Dict[str, Any] = {
            "turn_count": 0,
            "tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        self.is_shutting_down = False
        self._shutdown_done = False
        self._initialized = False

        self.metrics = {
            "total_queries": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "total_tool_calls": 0,
        }

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info("Initializing %s agent...", self.role.value)

        from agent.llm.provider import get_llm_registry
        self.llm = get_llm_registry()

        llm_cfg = self.config.get("llm", {}) or {}
        keys = dict(llm_cfg.get("api_keys") or {})
        for provider_name, key in keys.items():
            if key:
                self.llm.set_api_key(provider_name, key)

        await self._resolve_role_model()
        await self._initialize_tools()

        if self.role == AgentRole.PLAN:
            self.planner = Planner(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config.get("planner", {}) or {},
                fallback_chain=getattr(self, "_fallback_chain", None),
            )

        self.loop = AgentLoop(
            agent=self,
            llm=self.llm,
            tool_registry=self.tool_registry,
            planner=self.planner,
            config=self.config.get("loop", {}) or {},
        )

        self.subagent_manager = SubAgentManager(
            agent=self, config=self.config.get("subagent", {}) or {},
        )
        await self.subagent_manager.initialize()

        self.plugin_loader = get_plugin_loader(
            agent=self, config=self.config.get("plugins", {}) or {},
        )
        await self.plugin_loader.initialize()

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
        logger.info(
            "%s agent initialized with model: %s",
            self.role.value,
            self._get_current_model(),
        )

    async def _resolve_role_model(self) -> None:
        """
        Use whatever model the runtime registry currently has. There is no
        hardcoded role-specific model and no hardcoded fallback chain. If
        the user hasn't connected yet, both agents share a None model and
        every query fails cleanly with "no model selected" until /connect.
        """
        current = None
        try:
            current = self.llm.get_current_model()
        except Exception:
            current = None

        if current:
            # Do not touch the registry's selection: the user chose it.
            self._fallback_chain = [current]
            logger.info(
                "%s agent using user-connected model: %s",
                self.role.value,
                current,
            )
        else:
            self._fallback_chain = []
            logger.warning(
                "%s agent has no model; connect via the TUI /connect panel "
                "before sending queries.",
                self.role.value,
            )

    def _get_current_model(self) -> str:
        if self.llm:
            provider = self.llm.get_current_provider() or "-"
            model = self.llm.get_current_model() or "-"
            return f"{provider}/{model}"
        return "unknown"

    def get_status(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "status": self.status,
            "model": self._get_current_model(),
            "tools": list(self.tool_registry.tools.keys()) if self.tool_registry else [],
            "metrics": self.metrics,
            "turn_count": self.context.get("turn_count", 0),
        }

    async def _initialize_tools(self) -> None:
        """Initialize role-specific tool registry."""
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
            FileSystemTool(self.workspace),
            ReadTool(self.workspace),
            WriteTool(self.workspace),
            EditTool(self.workspace),
            ApplyPatchTool(self.workspace),
            SearchTool(self.workspace),
            GrepTool(self.workspace),
            GlobTool(self.workspace),
            GitTool(self.workspace, read_only=(self.role == AgentRole.PLAN)),
            WebTool(tools_cfg),
            WebFetchTool(tools_cfg),
            WebSearchTool(tools_cfg),
            DiagnosticsTool(self.workspace),
            TodoTool(self.session),
            TodoWriteTool(self.session),
            TodoReadTool(self.session),
            SkillTool(self.workspace),
            QuestionTool(self.input_handler),
            LspTool(self.workspace, tools_cfg),
            TerminalTool(self.workspace, tools_cfg),
            BashTool(self.workspace, tools_cfg),
            ProcessTool(self.workspace),
            PatchTool(self.workspace),
            BrowserTool(tools_cfg.get("browser", {})),
            RealtimeBrowserTool(tools_cfg.get("realtime_browser", {})),
            TaskTool(self),
            AWSHelperTool(self, self.workspace),
        ]
        if self.mcp_client is not None:
            tool_instances.append(MCPTool(self.mcp_client))

        # Read-only agents never register write-capable tools.
        if self.role == AgentRole.PLAN:
            tool_instances = [t for t in tool_instances if t.name in READ_ONLY_TOOL_NAMES]

        enabled = set(tools_cfg.get("enabled") or [])
        disabled = set(tools_cfg.get("disabled") or [])

        for tool in tool_instances:
            if tool.name in disabled:
                continue
            if enabled and tool.name not in enabled:
                continue
            await self.tool_registry.register_tool(tool)

        if self.role == AgentRole.BUILD and self.mcp_client is not None:
            for tool_def in self.mcp_client.list_tools():
                fn = tool_def["function"]
                self.tool_registry.register_external(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=fn["parameters"],
                    handler=lambda name=fn["name"], **kw: self.mcp_client.call_tool(name, kw),
                )

    def _get_config_object(self):
        try:
            from agent.config.loader import get_config
            return get_config()
        except Exception:
            return None

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

        # Defence in depth: even if the model somehow emits a write tool name,
        # PlanAgent's registry will not contain it, so this branch is a
        # belt-and-braces check.
        if self.role == AgentRole.PLAN and tool_name not in READ_ONLY_TOOL_NAMES:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not allowed in plan mode (read-only)",
                "permission_denied": True,
                "tool": tool_name,
            }

        # AWS in plan mode: reject mutating CLI verbs.
        if self.role == AgentRole.PLAN and tool_name in ("aws", "mcp__aws__call_aws"):
            operation = ""
            if isinstance(params, dict):
                operation = str(
                    params.get("operation")
                    or params.get("service")
                    or ""
                ).lower()
            mutating_verbs = (
                "create", "delete", "put", "update", "modify", "remove",
                "start", "stop", "terminate", "reboot", "attach", "detach",
                "associate", "disassociate", "authorize", "revoke",
                "run-instances", "create-bucket", "delete-bucket",
                "put-object", "delete-object", "put-bucket-policy",
            )
            if any(v in operation for v in mutating_verbs):
                return {
                    "success": False,
                    "error": (
                        f"AWS operation '{operation}' is mutating and not "
                        "allowed in plan mode (read-only)"
                    ),
                    "permission_denied": True,
                    "tool": tool_name,
                }

        if require_permission and self.permission_manager:
            action = self._infer_action(tool_name, params)
            try:
                allowed, reason = await self.permission_manager.check_permission(
                    tool_name=tool_name,
                    params=params,
                    context={"action": action, "agent_role": self.role.value},
                )
            except PermissionDeniedError as e:
                return {
                    "success": False,
                    "error": str(e),
                    "permission_denied": True,
                    "tool": tool_name,
                }
            if not allowed:
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
            return {
                "success": False,
                "error": format_error(wrapped),
                "tool": tool_name,
            }

        # Redact credentials before the result is stored or sent to the model.
        try:
            result = redact(result)
        except Exception:
            pass

        self.context["tool_calls"] = self.context.get("tool_calls", 0) + 1
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

        return result

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

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        if not self._initialized:
            raise AgentError(
                f"{self.role.value} agent used before initialize() was called"
            )

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
        start = time.time()
        try:
            self.status = "thinking"
            self.context["turn_count"] = self.context.get("turn_count", 0) + 1
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
            self.context["tokens_used"] = self.context.get("tokens_used", 0) + tokens
            self.context["input_tokens"] = self.context.get("input_tokens", 0) + ctx_summary.get("input_tokens", 0)
            self.context["output_tokens"] = self.context.get("output_tokens", 0) + ctx_summary.get("output_tokens", 0)
            self.metrics["total_tokens"] += tokens
            self.metrics["total_tool_calls"] += result.get("tool_calls", 0)

            n = self.metrics["total_queries"]
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (n - 1) + elapsed) / n
            )

            self.status = "idle"
            if self.session:
                self.session.set_status("idle")

            result_out = {
                "success": result.get("success", True),
                "response": response_text,
                "iteration": result.get("iteration", 0),
                "tool_calls": result.get("tool_calls", 0),
                "duration": elapsed,
                "context": ctx_summary,
                "plan": result.get("plan"),
            }
            if not result_out["success"] and result.get("error"):
                result_out["error"] = str(result.get("error"))
            return result_out

        except PermissionDeniedError as e:
            self.status = "idle"
            return {"success": False, "error": format_error(e), "denied": True}
        except Exception as e:
            self.status = "error"
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
            results.append(
                {
                    "subagent": "main",
                    "success": main_result.get("success", True),
                    "result": main_result.get("response", ""),
                    "main": main_result,
                }
            )

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

    async def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self.is_shutting_down = True
        logger.info("Shutting down %s agent...", self.role.value)

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


class PlanAgent(BaseAgent):
    """Read-only architect."""

    def __init__(self, *args, **kwargs):
        super().__init__(AgentRole.PLAN, *args, **kwargs)

    async def create_plan(
        self, goal: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.planner:
            raise AgentError("Planner not initialized")
        plan = await self.planner.create_plan(
            goal=goal,
            context=context or {},
            constraints={
                "max_tasks": 20,
                "available_tools": list(self.tool_registry.tools.keys()),
                "role": "plan",
            },
        )
        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "tasks": [t.to_dict() for t in plan.tasks],
            "estimated_total_time": plan.metrics.get("estimated_total_time", 0),
        }

    def get_available_tools(self) -> List[str]:
        return list(self.tool_registry.tools.keys())


class BuildAgent(BaseAgent):
    """Full execution agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(AgentRole.BUILD, *args, **kwargs)

    def get_available_tools(self) -> List[str]:
        return list(self.tool_registry.tools.keys())


@dataclass
class AgentCoordinator:
    """Coordinates Plan and Build agents."""

    plan_agent: PlanAgent
    build_agent: BuildAgent
    config: Dict[str, Any]
    current_mode: str = "build"
    _shutdown_done: bool = False

    async def initialize(self) -> None:
        await self.plan_agent.initialize()
        await self.build_agent.initialize()

    def set_mode(self, mode: str) -> bool:
        if mode in ("plan", "build"):
            self.current_mode = mode
            return True
        return False

    def get_current_agent(self) -> BaseAgent:
        return self.plan_agent if self.current_mode == "plan" else self.build_agent

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        auto_execute: bool = False,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        effective_mode = mode or self.current_mode

        if effective_mode == "auto":
            return await self._process_auto(query, context, auto_execute, max_iterations)

        agent = self.plan_agent if effective_mode == "plan" else self.build_agent
        logger.info("Processing query with %s agent: %s", effective_mode, query[:100])

        if effective_mode == "plan":
            context = dict(context or {})
            context["build_agent_tools"] = self.build_agent.get_available_tools()

        result = await agent.process_query(query=query, context=context)

        if effective_mode == "plan" and auto_execute and result.get("success"):
            plan_data = result.get("response", "")
            if plan_data:
                return await self._execute_plan_with_build(plan_data, context)

        result["mode"] = effective_mode
        return result

    async def _execute_plan_with_build(
        self, plan: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        build_prompt = (
            f"Execute this plan step by step:\n\n{plan}\n\n"
            "For each task, call the appropriate tool(s). Report results after each step.\n"
            "Continue until all tasks are complete or you encounter blockers.\n"
            "Be specific about what you did and any errors encountered."
        )
        build_result = await self.build_agent.process_query(
            query=build_prompt, context=context,
        )
        result = {
            "success": build_result.get("success", True),
            "plan": plan,
            "execution": build_result.get("response", ""),
            "mode": "plan->build",
            "iterations": 1,
            "tool_calls": build_result.get("tool_calls", 0),
        }
        if not result["success"] and build_result.get("error"):
            result["error"] = build_result.get("error")
        return result

    async def _process_auto(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        auto_execute: bool = True,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        logger.info(
            "Processing query (auto mode): %s (max_iterations=%d)",
            query[:100],
            max_iterations,
        )

        current_context = dict(context or {})
        plan_data = ""
        execution_history: List[Dict[str, Any]] = []
        final_execution = ""
        first_error: Optional[str] = None

        for iteration in range(max_iterations):
            logger.info("=== Iteration %d/%d ===", iteration + 1, max_iterations)

            plan_loop = getattr(self.plan_agent, "loop", None)
            prev_planning = getattr(plan_loop, "enable_planning", True) if plan_loop else True

            if iteration == 0:
                plan_prompt = (
                    f"Create a detailed execution plan for: {query}\n\n"
                    f"Context: {json.dumps(current_context, default=str)}\n\n"
                    f"Available tools for Build agent: "
                    f"{', '.join(self.build_agent.get_available_tools())}\n\n"
                    "Output format: JSON with tasks array, each with description, "
                    "tools, dependencies, validation."
                )
            else:
                prev_results = "\n\n".join(
                    f"--- Iteration {i + 1} Results ---\n"
                    f"{h.get('execution', '')[:2000]}"
                    for i, h in enumerate(execution_history)
                )
                plan_prompt = (
                    f"Previous plan execution results:\n{prev_results}\n\n"
                    f"Original goal: {query}\n\n"
                    "Review the results above. Some tasks may have failed or be incomplete.\n"
                    "Create an UPDATED plan to complete the remaining work.\n"
                    "Focus on fixing failures and completing unfinished tasks.\n\n"
                    f"Available tools: {', '.join(self.build_agent.get_available_tools())}\n\n"
                    "Output format: JSON with tasks array (description, tools, "
                    "dependencies, validation)."
                )

            try:
                if plan_loop:
                    plan_loop.enable_planning = False
                plan_result = await self.plan_agent.process_query(
                    query=plan_prompt, context=current_context,
                )
            finally:
                if plan_loop:
                    plan_loop.enable_planning = prev_planning

            plan_data = plan_result.get("response", "")
            logger.info("Plan created (iteration %d): %d chars", iteration + 1, len(plan_data))

            if not auto_execute:
                return {
                    "success": True,
                    "plan": plan_data,
                    "execution": None,
                    "iteration": iteration + 1,
                    "message": "Plan created. Use execute_plan() to run it.",
                }

            build_prompt = (
                f"Execute this plan step by step:\n\n{plan_data}\n\n"
                "For each task, call the appropriate tool(s). Report results after each step.\n"
                "Continue until all tasks are complete or you encounter blockers.\n"
                "Be specific about what you did and any errors encountered."
            )
            build_result = await self.build_agent.process_query(
                query=build_prompt, context=current_context,
            )

            execution_result = build_result.get("response", "")
            execution_history.append(
                {
                    "iteration": iteration + 1,
                    "plan": plan_data,
                    "execution": execution_result,
                    "success": build_result.get("success", True),
                    "tool_calls": build_result.get("tool_calls", 0),
                }
            )
            final_execution = execution_result

            if build_result.get("success", True):
                logger.info("Plan completed successfully in %d iterations", iteration + 1)
                break

            if first_error is None and build_result.get("error"):
                first_error = build_result.get("error")
            if first_error is None and plan_result.get("error"):
                first_error = plan_result.get("error")

            current_context["previous_execution"] = execution_result
            current_context["iteration"] = iteration + 1

        overall_success = all(h.get("success", True) for h in execution_history)
        result: Dict[str, Any] = {
            "success": overall_success,
            "plan": plan_data,
            "execution": final_execution,
            "iterations": len(execution_history),
            "execution_history": execution_history,
            "tool_calls": sum(h.get("tool_calls", 0) for h in execution_history),
            "mode": "auto",
        }
        if not overall_success and first_error:
            result["error"] = first_error
        return result

    def _is_plan_complete(self, execution: str, plan: str) -> bool:
        execution_lower = (execution or "").lower()
        completion_indicators = [
            "completed", "finished", "done", "successfully",
            "all tasks", "complete", "✓", "✅",
        ]
        failure_indicators = [
            "failed", "error", "blocked", "cannot", "unable",
            "timeout", "permission denied", "not found",
        ]
        has_completion = any(ind in execution_lower for ind in completion_indicators)
        has_failure = any(ind in execution_lower for ind in failure_indicators)
        return has_completion and not has_failure

    async def execute_plan(
        self, plan: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return await self._execute_plan_with_build(plan, context)

    async def shutdown(self) -> None:
        if self._shutdown_done:
            return
        self._shutdown_done = True
        await self.plan_agent.shutdown()
        await self.build_agent.shutdown()

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "plan_agent": {
                "status": self.plan_agent.status,
                "model": self.plan_agent._get_current_model(),
                "tools": self.plan_agent.get_available_tools(),
            },
            "build_agent": {
                "status": self.build_agent.status,
                "model": self.build_agent._get_current_model(),
                "tools": self.build_agent.get_available_tools(),
            },
        }


async def create_dual_agent_system(
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
) -> AgentCoordinator:
    plan_agent = PlanAgent(
        config=config,
        session=session,
        context_manager=context_manager,
        permission_manager=permission_manager,
        workspace=workspace,
        database=database,
        ui=ui,
        input_handler=input_handler,
        mcp_client=mcp_client,
        cache=cache,
        max_turns=max_turns,
        timeout=timeout,
    )
    build_agent = BuildAgent(
        config=config,
        session=session,
        context_manager=context_manager,
        permission_manager=permission_manager,
        workspace=workspace,
        database=database,
        ui=ui,
        input_handler=input_handler,
        mcp_client=mcp_client,
        cache=cache,
        max_turns=max_turns,
        timeout=timeout,
    )
    coordinator = AgentCoordinator(
        plan_agent=plan_agent,
        build_agent=build_agent,
        config=config,
    )
    await coordinator.initialize()
    return coordinator


__all__ = [
    "AgentRole",
    "PlanAgent",
    "BuildAgent",
    "AgentCoordinator",
    "create_dual_agent_system",
    "READ_ONLY_TOOL_NAMES",
]