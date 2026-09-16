"""
SubAgent Module - Role-based Subagents
======================================

Simple subagent system:
- general: Multi-step task agent with full tools (except todowrite)
- explore: Read-only codebase exploration (grep, glob, list, read, webfetch, websearch)
- scout:   Experimental docs/dependency research

Invoked via @agent_name syntax in prompts.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import SubAgentError
from agent.llm.provider import LLMProvider, Message
from agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class SubAgentRole(str, Enum):
    GENERAL = "general"
    EXPLORE = "explore"
    SCOUT = "scout"


# Tools that must never run in read-only subagents.
_WRITE_TOOL_NAMES = {
    "write", "edit", "apply_patch", "filesystem", "patch",
    "terminal", "bash", "process",
}


@dataclass
class SubAgentConfig:
    role: SubAgentRole
    name: str
    description: str
    model: str = ""
    temperature: float = 0.1
    allowed_tools: List[str] = field(default_factory=list)
    prompt_file: str = ""


BUILTIN_SUBAGENTS: Dict[SubAgentRole, SubAgentConfig] = {
    SubAgentRole.GENERAL: SubAgentConfig(
        role=SubAgentRole.GENERAL,
        name="general",
        description="General-purpose multi-step task agent. Full access minus todowrite.",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.3,
        allowed_tools=[
            "terminal", "filesystem", "search", "git", "patch", "process",
            "web", "browser", "diagnostics", "mcp",
        ],
    ),
    SubAgentRole.EXPLORE: SubAgentConfig(
        role=SubAgentRole.EXPLORE,
        name="explore",
        description=(
            "Read-only codebase exploration. Only grep, glob, list, read, "
            "webfetch, websearch, read-only git allowed."
        ),
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.1,
        allowed_tools=["search", "git", "web"],
    ),
    SubAgentRole.SCOUT: SubAgentConfig(
        role=SubAgentRole.SCOUT,
        name="scout",
        description="Experimental docs and dependency-source research agent.",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.1,
        allowed_tools=["search", "web"],
    ),
}


@dataclass
class SubAgent:
    id: str
    role: SubAgentRole
    config: SubAgentConfig
    parent_agent: Any
    created_at: float = field(default_factory=time.time)
    status: str = "idle"
    context: Dict[str, Any] = field(default_factory=dict)
    llm: Optional[LLMProvider] = None
    tool_registry: Optional[ToolRegistry] = None


class SubAgentManager:
    """
    Manages role-based subagents.

    Subagents are invoked via @agent_name syntax in user prompts.
    Each subagent gets its own model selection (isolated per call) and
    a restricted tool set.
    """

    _INVOCATION_RE = re.compile(r"@([A-Za-z_][A-Za-z0-9_]*)\b[ \t]*([^@]*)")

    def __init__(self, agent: Any, config: Dict[str, Any]):
        self.agent = agent
        self.config = config

        self.subagents: Dict[str, SubAgent] = {}
        self.subagent_configs: Dict[SubAgentRole, SubAgentConfig] = {
            role: SubAgentConfig(**vars(cfg)) for role, cfg in BUILTIN_SUBAGENTS.items()
        }

        for role, cfg in self.subagent_configs.items():
            user_cfg = config.get(role.value) or {}
            if "model" in user_cfg:
                cfg.model = user_cfg["model"]
            if "temperature" in user_cfg:
                cfg.temperature = user_cfg["temperature"]
            if "allowed_tools" in user_cfg:
                cfg.allowed_tools = list(user_cfg["allowed_tools"])

        self._model_lock = asyncio.Lock()
        self.llm_registry = None

        logger.info("SubAgentManager initialized with roles: %s", list(self.subagent_configs.keys()))

    async def initialize(self) -> None:
        from agent.llm.provider import get_llm_registry
        self.llm_registry = get_llm_registry()

        llm_cfg = self.agent.config.get("llm", {}) or {}
        keys = dict(llm_cfg.get("api_keys") or {})
        for provider_name, key in keys.items():
            if key:
                self.llm_registry.set_api_key(provider_name, key)

    def get_subagent_names(self) -> List[str]:
        return [cfg.name for cfg in self.subagent_configs.values()]

    def parse_subagent_invocation(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse @agent_name invocations. Returns [(name, query), ...] where query
        is the text between this invocation and the next @ or end-of-string.
        Only names present in subagent_configs are returned.
        """
        if not text:
            return []
        known = {cfg.name for cfg in self.subagent_configs.values()}
        matches = list(self._INVOCATION_RE.finditer(text))
        results: List[Tuple[str, str]] = []
        for idx, m in enumerate(matches):
            name = m.group(1)
            if name not in known:
                continue
            start = m.end(2)
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            query = text[start:end].strip()
            if query:
                results.append((name, query))
        return results

    def strip_subagent_invocations(self, text: str) -> str:
        """Remove all recognized @agent invocations from text."""
        known = {cfg.name for cfg in self.subagent_configs.values()}
        out_parts: List[str] = []
        last = 0
        for m in self._INVOCATION_RE.finditer(text or ""):
            if m.group(1) not in known:
                continue
            out_parts.append(text[last : m.start()])
            last = m.start()
            last = m.end(2)
        out_parts.append(text[last:])
        return " ".join(p.strip() for p in out_parts if p.strip())

    async def invoke_subagent(
        self,
        role: SubAgentRole,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if role not in self.subagent_configs:
            raise SubAgentError(f"Unknown subagent role: {role}")

        cfg = self.subagent_configs[role]
        subagent_id = f"{cfg.name}_{uuid.uuid4().hex[:8]}"
        logger.info("Invoking subagent %s (%s): %s", cfg.name, subagent_id, query[:100])

        subagent = SubAgent(
            id=subagent_id,
            role=role,
            config=cfg,
            parent_agent=self.agent,
            context=context or {},
        )

        async with self._model_lock:
            prev_model = None
            prev_provider = None
            model_switched = False
            try:
                if self.llm_registry is None:
                    await self.initialize()
                subagent.llm = self.llm_registry

                prev_model = subagent.llm.get_current_model()
                prev_provider = subagent.llm.get_current_provider()

                if cfg.model and cfg.model != prev_model:
                    result = subagent.llm.set_model(cfg.model)
                    if isinstance(result, dict) and not result.get("success", False):
                        logger.warning(
                            "Subagent %s could not set model %s: %s",
                            cfg.name,
                            cfg.model,
                            result.get("error"),
                        )
                    else:
                        model_switched = True

                subagent.tool_registry = await self._create_subagent_tools(cfg, subagent)
                self.subagents[subagent_id] = subagent
                subagent.status = "working"

                try:
                    system_prompt = self._build_subagent_system_prompt(cfg, context)
                    messages = [
                        Message(role="system", content=system_prompt),
                        Message(role="user", content=query),
                    ]
                    result_text = await self._run_subagent_loop(
                        subagent, messages, cfg.temperature
                    )
                    subagent.status = "completed"
                    return {
                        "success": True,
                        "subagent": cfg.name,
                        "result": result_text,
                        "subagent_id": subagent_id,
                    }
                except Exception as e:
                    subagent.status = "failed"
                    logger.error("Subagent %s failed: %s", subagent_id, e, exc_info=True)
                    return {
                        "success": False,
                        "subagent": cfg.name,
                        "error": str(e),
                        "subagent_id": subagent_id,
                    }
            finally:
                if model_switched and prev_model:
                    try:
                        subagent.llm.set_model(prev_model)
                        if prev_provider:
                            subagent.llm.set_provider(prev_provider)
                    except Exception as exc:
                        logger.debug("Failed to restore model '%s': %s", prev_model, exc)
                self.subagents.pop(subagent_id, None)

    async def _create_subagent_tools(
        self, cfg: SubAgentConfig, subagent: SubAgent
    ) -> ToolRegistry:
        from agent.tools import (
            TerminalTool, FileSystemTool, SearchTool, GitTool,
            ProcessTool, PatchTool, WebTool, BrowserTool,
            MCPTool, DiagnosticsTool,
            BashTool, ReadTool, WriteTool, EditTool, ApplyPatchTool,
            GrepTool, GlobTool, WebFetchTool, WebSearchTool,
        )

        registry = ToolRegistry(agent=subagent)
        workspace = self.agent.workspace
        tools_cfg = self.agent.config.get("tools", {}) or {}

        read_only = cfg.role in (SubAgentRole.EXPLORE, SubAgentRole.SCOUT)

        if any(k in cfg.allowed_tools for k in ("search", "grep", "glob")):
            await registry.register_tool(SearchTool(workspace))
            await registry.register_tool(GrepTool(workspace))
            await registry.register_tool(GlobTool(workspace))
            await registry.register_tool(ReadTool(workspace))

        if "git" in cfg.allowed_tools:
            try:
                await registry.register_tool(GitTool(workspace, read_only=read_only))
            except TypeError:
                await registry.register_tool(GitTool(workspace))

        if any(k in cfg.allowed_tools for k in ("web", "webfetch", "websearch")):
            await registry.register_tool(WebTool(tools_cfg))
            await registry.register_tool(WebFetchTool(tools_cfg))
            await registry.register_tool(WebSearchTool(tools_cfg))

        if "diagnostics" in cfg.allowed_tools or "lsp" in cfg.allowed_tools:
            await registry.register_tool(DiagnosticsTool(workspace))

        if cfg.role == SubAgentRole.GENERAL and not read_only:
            if any(k in cfg.allowed_tools for k in ("terminal", "bash")):
                await registry.register_tool(TerminalTool(workspace, tools_cfg))
                await registry.register_tool(BashTool(workspace, tools_cfg))
            if any(k in cfg.allowed_tools for k in ("filesystem", "read", "write", "edit")):
                await registry.register_tool(FileSystemTool(workspace, read_only=False))
                await registry.register_tool(WriteTool(workspace))
                await registry.register_tool(EditTool(workspace))
                await registry.register_tool(ApplyPatchTool(workspace))
            if "patch" in cfg.allowed_tools:
                await registry.register_tool(PatchTool(workspace))
            if "process" in cfg.allowed_tools:
                await registry.register_tool(ProcessTool(workspace))
            if "browser" in cfg.allowed_tools:
                await registry.register_tool(BrowserTool(tools_cfg.get("browser", {})))
            if "mcp" in cfg.allowed_tools and getattr(self.agent, "mcp_client", None):
                await registry.register_tool(MCPTool(self.agent.mcp_client))

        # Defensive filter — even if a caller overrides allowed_tools, never
        # register write-capable tools in read-only roles.
        if read_only:
            for name in list(registry.tools.keys()):
                if name in _WRITE_TOOL_NAMES:
                    registry.tools.pop(name, None)

        return registry

    def _build_subagent_system_prompt(
        self, cfg: SubAgentConfig, context: Optional[Dict[str, Any]]
    ) -> str:
        tools_desc = ", ".join(cfg.allowed_tools)
        base = (
            f"You are the {cfg.name} subagent.\n"
            f"{cfg.description}\n\n"
            f"Your available tools: {tools_desc}\n\n"
            "Guidelines:\n"
            "1. Think step by step and plan your actions.\n"
            "2. Use tools when necessary.\n"
            "3. Be concise and clear.\n"
            "4. Report progress and results after each tool call.\n"
            "5. If you encounter an error, explain it and suggest alternatives.\n\n"
            f"Context from parent agent: {json.dumps(context or {}, default=str)[:2000]}"
        )
        if cfg.role == SubAgentRole.EXPLORE:
            base += (
                "\n\nIMPORTANT: You are in READ-ONLY exploration mode.\n"
                "- You may ONLY read files, search code, inspect git status/log/diff.\n"
                "- You may NOT write, edit, delete, or execute shell commands.\n"
                "- Focus on understanding the codebase and answering questions."
            )
        elif cfg.role == SubAgentRole.SCOUT:
            base += (
                "\n\nIMPORTANT: You are a research agent for docs and dependencies.\n"
                "- Search the web for documentation, API references, changelogs.\n"
                "- Analyze dependency sources if needed.\n"
                "- Provide concise findings with sources."
            )
        return base

    async def _run_subagent_loop(
        self,
        subagent: SubAgent,
        messages: List[Message],
        temperature: float,
    ) -> str:
        max_turns = int(self.config.get("max_turns", 10))
        final_response = ""
        registry = subagent.tool_registry
        schemas = registry.get_schemas() if registry else []

        for _ in range(max_turns):
            response = await subagent.llm.complete_with_tools(
                messages=messages,
                tools=schemas,
                temperature=temperature,
                max_tokens=4000,
            )

            final_response = getattr(response, "content", "") or ""
            tool_calls = list(getattr(response, "tool_calls", []) or [])

            serialized_calls = (
                [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, default=str),
                        },
                    }
                    for tc in tool_calls
                ]
                if tool_calls
                else None
            )

            messages.append(
                Message(
                    role="assistant",
                    content=final_response,
                    tool_calls=serialized_calls,
                )
            )

            if not tool_calls:
                break

            for tc in tool_calls:
                try:
                    result = await registry.execute(tc.name, tc.arguments)
                    payload = json.dumps(result, default=str)[:5000]
                except Exception as e:
                    payload = f"Error: {e}"
                messages.append(
                    Message(role="tool", content=payload, tool_call_id=tc.id)
                )

        return final_response

    async def shutdown(self) -> None:
        self.subagents.clear()
        logger.info("SubAgentManager shutdown complete")


__all__ = [
    "SubAgentManager",
    "SubAgentRole",
    "SubAgentConfig",
    "SubAgent",
    "BUILTIN_SUBAGENTS",
]