"""
Dual Agent Architecture - OpenCode Style (SWAPPED)
==================================================

Two primary agents with distinct roles:
- PlanAgent: Read-only planning, uses Groq GPT-OSS-120B (fast reasoning)
- BuildAgent: Full tool access, uses NVIDIA Nemotron 3 Ultra (LARGE CONTEXT WINDOW)

This mirrors OpenCode's architecture where:
- Plan agent analyzes code, creates plans, suggests changes (NO write tools)
- Build agent executes plans, makes changes, runs commands (ALL tools)

SWAPPED RATIONALE:
- Build agent needs LARGE CONTEXT WINDOW for:
  * Tool execution outputs (stdout/stderr)
  * File contents being read/written
  * Large codebases being analyzed
  * Multi-step execution history
  * NVIDIA Nemotron 3 Ultra has 1M context window
- Plan agent needs FAST REASONING for:
  * Structured plan generation
  * Dependency analysis
  * Task decomposition
  * Groq GPT-OSS-120B is excellent for this
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
    AgentError, ToolExecutionError, PermissionDeniedError,
    handle_exception, format_error, is_retryable,
)
from agent.agent.loop import AgentLoop
from agent.agent.planner import Planner
from agent.tools.registry import ToolRegistry, BaseTool
from agent.agent.subagent import SubAgentManager
from agent.plugins.loader import get_plugin_loader

logger = get_logger(__name__)


# ======================================================================
# AGENT ROLES
# ======================================================================

class AgentRole(str, Enum):
    PLAN = "plan"       # Read-only: analyze, plan, suggest
    BUILD = "build"     # Read-write: execute, modify, run


# ======================================================================
# PERMISSION RULESETS (OpenCode style)
# ======================================================================

# Tools that READ-ONLY agents can use (Plan agent - matches opencode's Plan agent)
# opencode Plan agent: grep, glob, list, bash, read, webfetch, websearch
PLAN_AGENT_TOOLS: Set[str] = {
    "filesystem",   # read, list, stat (read-only actions only)
    "search",       # grep, find (read-only)
    "git",          # status, log, diff, show, branch, blame, remote (read-only actions)
    "diagnostics",  # lint, typecheck (read-only)
    "web",          # search, fetch (read-only)
}

# Tools that BUILD agents can use (all tools - matches opencode's Build agent)
BUILD_AGENT_TOOLS: Set[str] = {
    "terminal",     # execute commands
    "filesystem",   # read, write, edit, delete, mkdir, move, copy
    "search",       # grep, find
    "git",          # all git operations
    "patch",        # apply patches
    "process",      # process management
    "web",          # search, fetch
    "browser",      # browser automation
    "task",         # spawn subagents
    "diagnostics",  # lint, typecheck
    "todo",         # task management
    "mcp",          # MCP tools
}

# Tool descriptions for LLM
TOOL_DESCRIPTIONS: Dict[str, str] = {
    "terminal": "Execute shell commands (bash, sh, zsh)",
    "filesystem": "Read, write, edit, list, search files",
    "search": "Search code with grep/rg/find",
    "git": "Git operations (status, log, diff, commit, push, etc.)",
    "patch": "Apply unified diff patches",
    "process": "Manage background processes",
    "web": "Web search and fetch",
    "browser": "Browser automation (playwright)",
    "task": "Spawn subagent for parallel work",
    "diagnostics": "Run linters, type checkers",
    "todo": "Manage task checklists",
    "mcp": "Model Context Protocol tools",
}


# ======================================================================
# BASE AGENT CLASS
# ======================================================================

class BaseAgent:
    """Base class for PlanAgent and BuildAgent"""
    
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
        
        # Subsystems
        self.llm = None
        self.tool_registry = None
        self.planner = None
        self.loop = None
        self.subagent_manager = None
        self.plugin_loader = None
        
        # State
        self.status = "idle"
        self.context = {"turn_count": 0, "tokens_used": 0}
        self.is_shutting_down = False
        self._initialized = False
        
        # Metrics
        self.metrics = {
            "total_queries": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "total_tool_calls": 0,
        }
    
    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info(f"Initializing {self.role.value} agent...")
        
        # 1. LLM Registry - role-specific model selection
        from agent.llm.provider import get_llm_registry
        self.llm = get_llm_registry()
        
        # Apply API keys
        llm_cfg = self.config.get("llm", {}) or {}
        keys = dict(llm_cfg.get("api_keys") or {})
        for provider_name, key in keys.items():
            if key:
                self.llm.set_api_key(provider_name, key)
        
        # Role-specific model resolution
        await self._resolve_role_model()
        
        # 2. Tools - role-specific tool registry
        await self._initialize_tools()
        
        # 3. Planner (only for Plan agent)
        if self.role == AgentRole.PLAN:
            self.planner = Planner(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config.get("planner", {}),
            )
        
        # 4. Loop
        self.loop = AgentLoop(
            agent=self,
            llm=self.llm,
            tool_registry=self.tool_registry,
            planner=self.planner,
            config=self.config.get("loop", {}),
        )
        
        # 5. Subagents (OpenCode style)
        self.subagent_manager = SubAgentManager(
            agent=self, config=self.config.get("subagent", {}),
        )
        await self.subagent_manager.initialize()
        
        # 6. Plugins
        self.plugin_loader = get_plugin_loader(
            agent=self, config=self.config.get("plugins", {}),
        )
        await self.plugin_loader.initialize()
        
        # 7. Permission callback
        if self.permission_manager and self.input_handler:
            async def _confirm(request, verdict):
                msg = f"Allow {request.tool}.{request.action}? [{verdict.risk.value}]"
                if hasattr(self.input_handler, "get_confirmation"):
                    return await self.input_handler.get_confirmation(msg, default=False)
                return False
            self.permission_manager.set_confirm_callback(_confirm)
        
        self._initialized = True
        logger.info(f"{self.role.value} agent initialized with model: {self._get_current_model()}")
    
    async def _resolve_role_model(self) -> None:
        """Resolve model based on agent role with fallback chains"""
        cfg_obj = self._get_config_object()
        
        if self.role == AgentRole.BUILD:
            # Build Agent: NVIDIA NIM - Nemotron 3 Ultra for complex reasoning/coding
            provider = "nvidia"
            model = "nvidia/nemotron-3-ultra-550b-a55b"
            fallback_chain = [
                "nvidia/nemotron-3-ultra-550b-a55b",      # Best for complex reasoning/coding (1M ctx)
                "nvidia/nemotron-3-super-120b-a12b",      # Balanced reasoning/coding (256K ctx)
                "nvidia/nemotron-3.5-lightning-30b-a3b",  # Fast execution, tool calls (1M ctx)
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",  # Lightweight reasoning
                "nvidia/nemotron-4-340b-instruct",        # Large model for complex tasks
                "openai/gpt-oss-20b",                     # Fallback: GPT-OSS on NVIDIA
            ]
            logger.info(f"Build agent using NVIDIA NIM (Nemotron 3 Ultra) with fallback chain ({len(fallback_chain)} models)")
        else:
            # Plan Agent: NVIDIA NIM - Nemotron 3.5 Lightning for fast planning
            provider = "nvidia"
            model = "nvidia/nemotron-3.5-lightning-30b-a3b"
            fallback_chain = [
                "nvidia/nemotron-3.5-lightning-30b-a3b",  # Fast planning, 1M ctx
                "nvidia/nemotron-3-super-120b-a12b",      # Balanced reasoning fallback
                "nvidia/nemotron-3-ultra-550b-a55b",      # Deep reasoning fallback
                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",  # Lightweight
                "openai/gpt-oss-20b",                     # Fallback: GPT-OSS
            ]
            logger.info(f"Plan agent using NVIDIA NIM (Nemotron 3.5 Lightning) with fallback chain ({len(fallback_chain)} models)")
        
        # Store fallback chain for retry logic
        self._fallback_chain = fallback_chain
        
        # Set the model
        if provider and hasattr(self.llm, 'providers') and provider in self.llm.providers:
            self.llm.set_model(model)
            logger.info(f"Set {self.role.value} agent model: {provider}/{model}")
    
    def _get_current_model(self) -> str:
        if self.llm:
            return f"{self.llm.get_current_provider()}/{self.llm.get_current_model()}"
        return "unknown"
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for display"""
        return {
            "role": self.role.value,
            "status": self.status,
            "model": self._get_current_model(),
            "tools": list(self.tool_registry.tools.keys()) if self.tool_registry else [],
            "metrics": self.metrics,
            "turn_count": self.context.get("turn_count", 0),
        }
    
    async def _initialize_tools(self) -> None:
        """Initialize role-specific tool registry (opencode-compatible)

        Plan agent: registers all tools but write/edit/bash are gated by
        PermissionManager (edit:deny, bash:deny) — matches opencode semantics
        where tool exists but is denied. This keeps schemas stable.
        """
        from agent.tools import (
            TerminalTool, FileSystemTool, SearchTool, GitTool,
            ProcessTool, PatchTool, WebTool, BrowserTool, TaskTool,
            MCPTool, DiagnosticsTool, TodoTool,
            BashTool, ReadTool, WriteTool, EditTool, ApplyPatchTool,
            GrepTool, GlobTool, WebFetchTool, WebSearchTool,
            TodoWriteTool, TodoReadTool, SkillTool, QuestionTool, LspTool,
        )
        
        self.tool_registry = ToolRegistry(agent=self)
        tools_cfg = self.config.get("tools", {})
        
        # All tools — permission layer + BaseAgent plan guard enforces read-only
        tool_instances = [
            FileSystemTool(self.workspace),
            ReadTool(self.workspace),
            WriteTool(self.workspace),
            EditTool(self.workspace),
            ApplyPatchTool(self.workspace),
            SearchTool(self.workspace),
            GrepTool(self.workspace),
            GlobTool(self.workspace),
            GitTool(self.workspace),
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
            TaskTool(self),
        ]
        
        if self.mcp_client is not None:
            tool_instances.append(MCPTool(self.mcp_client))
        
        # Opencode-style: plan still sees all tool schemas (permission denies writes)
        # but filter by global enabled/disabled only — not by PLAN_AGENT_TOOLS
        enabled = set(tools_cfg.get("enabled") or [])
        disabled = set(tools_cfg.get("disabled") or [])
        
        for tool in tool_instances:
            if tool.name in disabled:
                continue
            if enabled and tool.name not in enabled:
                continue
            await self.tool_registry.register_tool(tool)
        
        # Register external MCP tools
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
        """Execute a tool through permission gate + registry (opencode plan enforcement)"""
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}

        # Opencode Plan mode: deny writes/bash even if manager lacks explicit rule
        if self.role == AgentRole.PLAN:
            t = tool_name.lower()
            # deny all edit/write/patch and bash/terminal/process writes
            if t in ("write", "edit", "apply_patch", "filesystem", "patch", "process", "bash", "terminal", "shell", "git"):
                # for filesystem/git, only deny write-like actions
                if t in ("filesystem", "git"):
                    act = params.get("action", "").lower()
                    write_actions = ("write", "append", "edit", "delete", "mkdir", "move", "copy", "add", "commit", "push", "checkout", "reset", "stash")
                    if act in write_actions:
                        return {"success": False, "error": f"Action '{act}' not allowed in plan mode (read-only)", "permission_denied": True, "tool": tool_name}
                else:
                    return {"success": False, "error": f"Tool '{tool_name}' not allowed in plan mode (read-only)", "permission_denied": True, "tool": tool_name}
        
        # Permission gate
        if require_permission and self.permission_manager:
            action = self._infer_action(tool_name, params)
            allowed, reason = await self.permission_manager.check_permission(
                tool_name=tool_name,
                params=params,
                context={"action": action, "agent_role": self.role.value},
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
        
        # Record
        self.context["tool_calls"] = self.context.get("tool_calls", 0) + 1
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
        
        return result
    
    def _infer_action(self, tool: str, params: Dict[str, Any]) -> str:
        t = tool.lower()
        if "read" in t or "list" in t or "search" in t or "grep" in t:
            return "read"
        if "write" in t or "edit" in t or "patch" in t:
            return "write"
        if "delete" in t or "remove" in t:
            return "delete"
        if t in ("terminal", "shell", "process") or "bash" in t:
            return "execute"
        if t.startswith("mcp__"):
            return "mcp"
        if "git" in t:
            return "git"
        return "execute"
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process query through the agent loop with subagent support"""
        if self.is_shutting_down:
            return {"success": False, "error": "Agent shutting down"}
        
        # Check for @subagent invocations (OpenCode style)
        if self.subagent_manager:
            subagent_calls = self.subagent_manager.parse_subagent_invocation(query)
            if subagent_calls:
                return await self._handle_subagent_invocations(subagent_calls, context, query)
        
        start = time.time()
        try:
            self.status = "thinking"
            self.context["turn_count"] = self.context.get("turn_count", 0) + 1
            self.metrics["total_queries"] += 1
            
            # Record user message
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
            
            # Save response
            response_text = result.get("response", "")
            if self.context_manager and response_text:
                await self.context_manager.add_assistant_message(response_text)
            if self.session and response_text:
                self.session.add_assistant_message(response_text)
            
            # Update metrics
            elapsed = time.time() - start
            tokens = result.get("context", {}).get("tokens_used", 0)
            self.context["tokens_used"] = self.context.get("tokens_used", 0) + tokens
            self.context["input_tokens"] = self.context.get("input_tokens", 0) + result.get("context", {}).get("input_tokens", 0)
            self.context["output_tokens"] = self.context.get("output_tokens", 0) + result.get("context", {}).get("output_tokens", 0)
            self.metrics["total_tokens"] += tokens
            self.metrics["total_tool_calls"] += result.get("tool_calls", 0)
            
            n = self.metrics["total_queries"]
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (n - 1) + elapsed) / n
            )
            
            self.status = "idle"
            
            if self.session:
                self.session.set_status("idle")
            
            return {
                "success": result.get("success", True),
                "response": response_text,
                "iteration": result.get("iteration", 0),
                "tool_calls": result.get("tool_calls", 0),
                "duration": elapsed,
                "context": result.get("context", {}),
                "plan": result.get("plan"),
            }
            
        except PermissionDeniedError as e:
            self.status = "error"
            return {"success": False, "error": format_error(e), "denied": True}
        except Exception as e:
            self.status = "error"
            logger.error(f"Query failed: {e}", exc_info=True)
            return {"success": False, "error": format_error(e)}
    
    async def _handle_subagent_invocations(
        self,
        subagent_calls: List[tuple],
        context: Optional[Dict[str, Any]],
        original_query: str
    ) -> Dict[str, Any]:
        """Handle @subagent_name invocations in query"""
        from agent.agent.subagent import SubAgentRole
        
        results = []
        remaining_query = original_query
        
        for agent_name, sub_query in subagent_calls:
            # Map agent name to role
            role_map = {
                "general": SubAgentRole.GENERAL,
                "explore": SubAgentRole.EXPLORE,
                "scout": SubAgentRole.SCOUT,
            }
            
            role = role_map.get(agent_name.lower())
            if not role:
                results.append({
                    "subagent": agent_name,
                    "success": False,
                    "error": f"Unknown subagent: {agent_name}. Available: {', '.join(role_map.keys())}"
                })
                continue
            
            # Remove this invocation from remaining query
            remaining_query = remaining_query.replace(f"@{agent_name} {sub_query}", "").strip()
            
            # Invoke subagent
            result = await self.subagent_manager.invoke_subagent(
                role=role,
                query=sub_query,
                context=context,
            )
            results.append(result)
        
        # If there's remaining query after subagent invocations, process it normally
        if remaining_query:
            main_result = await self._process_main_query(remaining_query, context)
            results.append({"main": main_result})
        
        # Combine results
        combined_response = "\n\n".join([
            f"--- @{r.get('subagent', 'main')} ---\n{r.get('result', r.get('error', str(r.get('main', {}))))}"
            for r in results
        ])
        
        return {
            "success": all(r.get("success", True) for r in results),
            "response": combined_response,
            "subagent_results": results,
            "iteration": 1,
            "tool_calls": sum(r.get("tool_calls", 0) for r in results if isinstance(r, dict)),
            "context": {"tokens_used": 0},
        }
    
    async def _process_main_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process the main query after subagent invocations"""
        start = time.time()
        try:
            self.status = "thinking"
            self.context["turn_count"] = self.context.get("turn_count", 0) + 1
            self.metrics["total_queries"] += 1
            
            if self.context_manager:
                await self.context_manager.add_user_message(query, context)
            if self.session:
                self.session.add_user_message(query)
            
            if self.context_manager and await self.context_manager.needs_compaction():
                await self.context_manager.compact()
            
            live_context = {}
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
            tokens = result.get("context", {}).get("tokens_used", 0)
            self.context["tokens_used"] = self.context.get("tokens_used", 0) + tokens
            self.metrics["total_tokens"] += tokens
            self.metrics["total_tool_calls"] += result.get("tool_calls", 0)
            
            n = self.metrics["total_queries"]
            self.metrics["avg_response_time"] = (
                (self.metrics["avg_response_time"] * (n - 1) + elapsed) / n
            )
            
            self.status = "idle"
            
            return {
                "success": result.get("success", True),
                "response": response_text,
                "iteration": result.get("iteration", 0),
                "tool_calls": result.get("tool_calls", 0),
                "duration": elapsed,
                "context": result.get("context", {}),
            }
            
        except Exception as e:
            logger.error(f"Main query failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def shutdown(self) -> None:
        self.is_shutting_down = True
        logger.info(f"Shutting down {self.role.value} agent...")
        
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
        if self.llm:
            await self.llm.reconnect_all()


# ======================================================================
# PLAN AGENT - Read-only analysis and planning (uses Groq GPT-OSS-120B)
# ======================================================================

class PlanAgent(BaseAgent):
    """
    Plan Agent - Read-only architect (Groq GPT-OSS-120B for fast planning)
    
    Capabilities:
    - Analyze codebase
    - Create detailed plans
    - Suggest changes
    - Research and investigate
    
    Restrictions:
    - NO terminal/shell execution
    - NO file writes/edits
    - NO git mutations
    - NO process management
    - NO subagent spawning
    
    Model: Groq GPT-OSS-120B (fast reasoning, structured output)
    Fallback: GPT-OSS-20B → Llama-3.3-70B → Mixtral → Gemma → NVIDIA
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(AgentRole.PLAN, *args, **kwargs)
    
    async def create_plan(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a detailed execution plan"""
        if not self.planner:
            raise AgentError("Planner not initialized")
        
        plan = await self.planner.create_plan(
            goal=goal,
            context=context or {},
            constraints={
                "max_tasks": 20,
                "available_tools": list(self.tool_registry.tools.keys()),
                "role": "plan",
            }
        )
        
        return {
            "plan_id": plan.id,
            "goal": plan.goal,
            "tasks": [t.to_dict() for t in plan.tasks],
            "estimated_total_time": plan.metrics.get("estimated_total_time", 0),
        }
    
    def get_available_tools(self) -> List[str]:
        return list(self.tool_registry.tools.keys())


# ======================================================================
# BUILD AGENT - Full execution capabilities (uses NVIDIA Nemotron 3 Ultra - 1M context)
# ======================================================================

class BuildAgent(BaseAgent):
    """
    Build Agent - Full execution (NVIDIA Nemotron 3 Ultra for LARGE CONTEXT WINDOW)
    
    Capabilities:
    - Execute terminal commands
    - Read/write/edit files
    - Git operations (commit, push, etc.)
    - Process management
    - Spawn subagents
    - Run linters, tests
    - Browser automation
    
    Model: NVIDIA Nemotron 3 Ultra 550B (1M context window)
    Fallback: Super 120B (256K) → Lightning 30B (1M) → Nano 30B (256K) → Groq
    
    WHY NVIDIA FOR BUILD:
    - 1M context handles: tool outputs, file contents, execution history
    - Large codebases fit in context
    - Multi-step execution without context loss
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(AgentRole.BUILD, *args, **kwargs)
    
    def get_available_tools(self) -> List[str]:
        return list(self.tool_registry.tools.keys())


# ======================================================================
# AGENT COORDINATOR - Manages Plan + Build agents (OpenCode style)
# ======================================================================

@dataclass
class AgentCoordinator:
    """
    Coordinates Plan and Build agents (OpenCode style)
    
    Workflow (matching opencode):
    - Two primary agents: Plan (read-only) and Build (full tools)
    - User can switch between them explicitly
    - Subagents can be invoked with @syntax
    
    Modes:
    - "plan": Use Plan agent for analysis/planning
    - "build": Use Build agent for execution
    - "auto": Automatic Plan -> Build loop (legacy)
    """
    
    plan_agent: PlanAgent
    build_agent: BuildAgent
    config: Dict[str, Any]
    current_mode: str = "build"  # Default to build like opencode
    
    async def initialize(self) -> None:
        await self.plan_agent.initialize()
        await self.build_agent.initialize()
    
    def set_mode(self, mode: str) -> bool:
        """Switch between plan and build agents (like Tab in opencode)"""
        if mode in ("plan", "build"):
            self.current_mode = mode
            return True
        return False
    
    def get_current_agent(self) -> BaseAgent:
        """Get the currently active agent"""
        if self.current_mode == "plan":
            return self.plan_agent
        return self.build_agent
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        auto_execute: bool = False,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Process query with the current agent (or specified mode).
        
        Args:
            mode: "plan", "build", or "auto" (default: current_mode)
            auto_execute: If True and mode="plan", auto-execute with Build agent
            max_iterations: For "auto" mode only
        """
        effective_mode = mode or self.current_mode
        
        if effective_mode == "auto":
            return await self._process_auto(query, context, auto_execute, max_iterations)
        
        agent = self.plan_agent if effective_mode == "plan" else self.build_agent
        logger.info(f"Processing query with {effective_mode} agent: {query[:100]}...")
        
        # For plan agent, add context about available build tools
        if effective_mode == "plan":
            context = context or {}
            context["build_agent_tools"] = self.build_agent.get_available_tools()
        
        result = await agent.process_query(query=query, context=context)
        
        # If plan agent and auto_execute, run the plan with build agent
        if effective_mode == "plan" and auto_execute and result.get("success"):
            plan_data = result.get("response", "")
            if plan_data:
                return await self._execute_plan_with_build(plan_data, context)
        
        result["mode"] = effective_mode
        return result
    
    async def _execute_plan_with_build(self, plan: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a plan with the Build agent"""
        build_prompt = (
            f"Execute this plan step by step:\n\n{plan}\n\n"
            f"For each task, call the appropriate tool(s). Report results after each step.\n"
            f"Continue until all tasks are complete or you encounter blockers.\n"
            f"Be specific about what you did and any errors encountered."
        )
        
        build_result = await self.build_agent.process_query(
            query=build_prompt,
            context=context,
        )
        
        return {
            "success": build_result.get("success", True),
            "plan": plan,
            "execution": build_result.get("response", ""),
            "mode": "plan->build",
            "iterations": 1,
            "tool_calls": build_result.get("tool_calls", 0),
        }
    
    async def _process_auto(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        auto_execute: bool = True,
        max_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Legacy automatic Plan -> Build loop"""
        logger.info(f"Processing query (auto mode): {query[:100]}... (max_iterations={max_iterations})")
        
        current_context = context or {}
        plan_data = ""
        execution_history = []
        final_execution = ""
        
        for iteration in range(max_iterations):
            logger.info(f"=== Iteration {iteration + 1}/{max_iterations} ===")
            
            # Phase 1: Plan Agent creates/updates plan
            if iteration == 0:
                plan_prompt = (
                    f"Create a detailed execution plan for: {query}\n\n"
                    f"Context: {json.dumps(current_context)}\n\n"
                    f"Available tools for Build agent: {', '.join(self.build_agent.get_available_tools())}\n\n"
                    f"Output format: JSON with tasks array, each with description, tools, dependencies, validation."
                )
            else:
                prev_results = "\n\n".join([
                    f"--- Iteration {i+1} Results ---\n{exec_hist.get('execution', '')[:2000]}"
                    for i, exec_hist in enumerate(execution_history)
                ])
                plan_prompt = (
                    f"Previous plan execution results:\n{prev_results}\n\n"
                    f"Original goal: {query}\n\n"
                    f"Review the results above. Some tasks may have failed or be incomplete.\n"
                    f"Create an UPDATED plan to complete the remaining work.\n"
                    f"Focus on fixing failures and completing unfinished tasks.\n\n"
                    f"Available tools: {', '.join(self.build_agent.get_available_tools())}\n\n"
                    f"Output format: JSON with tasks array (description, tools, dependencies, validation)."
                )
            
            plan_result = await self.plan_agent.process_query(
                query=plan_prompt,
                context=current_context,
            )
            
            plan_data = plan_result.get("response", "")
            logger.info(f"Plan created (iteration {iteration + 1}): {len(plan_data)} chars")
            
            if not auto_execute:
                return {
                    "success": True,
                    "plan": plan_data,
                    "execution": None,
                    "iteration": iteration + 1,
                    "message": "Plan created. Use execute_plan() to run it.",
                }
            
            # Phase 2: Build Agent executes plan
            build_prompt = (
                f"Execute this plan step by step:\n\n{plan_data}\n\n"
                f"For each task, call the appropriate tool(s). Report results after each step.\n"
                f"Continue until all tasks are complete or you encounter blockers.\n"
                f"Be specific about what you did and any errors encountered."
            )
            
            build_result = await self.build_agent.process_query(
                query=build_prompt,
                context=current_context,
            )
            
            execution_result = build_result.get("response", "")
            execution_history.append({
                "iteration": iteration + 1,
                "plan": plan_data,
                "execution": execution_result,
                "success": build_result.get("success", True),
                "tool_calls": build_result.get("tool_calls", 0),
            })
            
            final_execution = execution_result
            
            if build_result.get("success", True) and self._is_plan_complete(execution_result, plan_data):
                logger.info(f"Plan completed successfully in {iteration + 1} iterations")
                break
            
            current_context["previous_execution"] = execution_result
            current_context["iteration"] = iteration + 1
        
        overall_success = all(h.get("success", True) for h in execution_history)
        
        return {
            "success": overall_success,
            "plan": plan_data,
            "execution": final_execution,
            "iterations": len(execution_history),
            "execution_history": execution_history,
            "tool_calls": sum(h.get("tool_calls", 0) for h in execution_history),
            "mode": "auto",
        }
    
    def _is_plan_complete(self, execution: str, plan: str) -> bool:
        execution_lower = execution.lower()
        completion_indicators = [
            "completed", "finished", "done", "successfully",
            "all tasks", "complete", "✓", "✅"
        ]
        failure_indicators = [
            "failed", "error", "blocked", "cannot", "unable",
            "timeout", "permission denied", "not found"
        ]
        
        has_completion = any(ind in execution_lower for ind in completion_indicators)
        has_failure = any(ind in execution_lower for ind in failure_indicators)
        
        return has_completion and not has_failure
    
    async def execute_plan(self, plan: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute an existing plan with Build agent"""
        return await self._execute_plan_with_build(plan, context)
    
    async def shutdown(self) -> None:
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


# ======================================================================
# FACTORY FUNCTION
# ======================================================================

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
    """Create Plan + Build agent system"""
    
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
    "PLAN_AGENT_TOOLS",
    "BUILD_AGENT_TOOLS",
]