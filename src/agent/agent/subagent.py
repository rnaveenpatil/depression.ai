"""
SubAgent Module - OpenCode Style Subagents
==========================================

Simple subagent system matching opencode:
- general: Multi-step task agent with full tools (except todowrite)
- explore: Read-only codebase exploration (grep, glob, list, read, webfetch, websearch)
- scout: Experimental docs/dependency research

Invoked via @agent_name syntax in prompts
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from agent.utils.logging import get_logger
from agent.utils.errors import SubAgentError
from agent.llm.provider import LLMProvider, Message
from agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class SubAgentRole(str, Enum):
    GENERAL = "general"      # Full tools for multi-step tasks
    EXPLORE = "explore"      # Read-only codebase exploration
    SCOUT = "scout"          # Experimental docs/dependency research


@dataclass
class SubAgentConfig:
    """Configuration for a subagent type"""
    role: SubAgentRole
    name: str
    description: str
    model: str = ""
    temperature: float = 0.1
    allowed_tools: List[str] = field(default_factory=list)
    prompt_file: str = ""


# Built-in subagent configurations matching opencode
BUILTIN_SUBAGENTS = {
    SubAgentRole.GENERAL: SubAgentConfig(
        role=SubAgentRole.GENERAL,
        name="general",
        description="General-purpose multi-step task agent. Full access minus todowrite.",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.3,
        allowed_tools=["terminal", "filesystem", "search", "git", "patch", "process", "web", "browser", "diagnostics", "mcp"],
    ),
    SubAgentRole.EXPLORE: SubAgentConfig(
        role=SubAgentRole.EXPLORE,
        name="explore",
        description="Read-only codebase exploration. Only grep, glob, list, bash, read, webfetch, websearch allowed.",
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
    """A subagent instance"""
    id: str
    role: SubAgentRole
    config: SubAgentConfig
    parent_agent: Any  # Reference to parent agent
    created_at: float = field(default_factory=time.time)
    status: str = "idle"
    context: Dict[str, Any] = field(default_factory=dict)
    llm: Optional[LLMProvider] = None
    tool_registry: Optional[ToolRegistry] = None


class SubAgentManager:
    """
    Simple SubAgent Manager matching opencode's approach.
    
    Subagents are invoked via @agent_name syntax in user prompts.
    Each subagent gets its own LLM instance and restricted tool set.
    """
    
    def __init__(self, agent: Any, config: Dict[str, Any]):
        self.agent = agent
        self.config = config
        
        # Subagent storage
        self.subagents: Dict[str, SubAgent] = {}
        self.subagent_configs: Dict[SubAgentRole, SubAgentConfig] = dict(BUILTIN_SUBAGENTS)
        
        # Apply config overrides
        for role, cfg in self.subagent_configs.items():
            if role.value in config:
                user_cfg = config[role.value]
                if "model" in user_cfg:
                    cfg.model = user_cfg["model"]
                if "temperature" in user_cfg:
                    cfg.temperature = user_cfg["temperature"]
                if "allowed_tools" in user_cfg:
                    cfg.allowed_tools = user_cfg["allowed_tools"]
        
        logger.info("SubAgentManager initialized with roles: %s", list(self.subagent_configs.keys()))
    
    async def initialize(self) -> None:
        """Initialize LLM registry for subagents"""
        from agent.llm.provider import get_llm_registry
        self.llm_registry = get_llm_registry()
        
        # Apply API keys
        llm_cfg = self.agent.config.get("llm", {}) or {}
        keys = dict(llm_cfg.get("api_keys") or {})
        for provider_name, key in keys.items():
            if key:
                self.llm_registry.set_api_key(provider_name, key)
    
    def get_subagent_names(self) -> List[str]:
        """Get list of available subagent names for @ syntax"""
        return [cfg.name for cfg in self.subagent_configs.values()]
    
    def parse_subagent_invocation(self, text: str) -> List[tuple]:
        """
        Parse @agent_name invocations from text.
        Returns list of (agent_name, query) tuples.
        """
        import re
        # Match @agent_name followed by text until next @ or end
        pattern = r'@(\w+)\s+([^@]+)'
        matches = re.findall(pattern, text)
        return [(name, query.strip()) for name, query in matches]
    
    async def invoke_subagent(
        self,
        role: SubAgentRole,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Invoke a subagent with a query.
        Creates a new subagent instance, runs the query, returns result.
        """
        if role not in self.subagent_configs:
            raise SubAgentError(f"Unknown subagent role: {role}")
        
        cfg = self.subagent_configs[role]
        subagent_id = f"{cfg.name}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Invoking subagent {cfg.name} ({subagent_id}) with query: {query[:100]}...")
        
        # Create subagent
        subagent = SubAgent(
            id=subagent_id,
            role=role,
            config=cfg,
            parent_agent=self.agent,
            context=context or {},
        )
        
        # Initialize subagent's LLM
        subagent.llm = self.llm_registry
        prev_model = self.llm_registry.get_current_model()
        prev_provider = self.llm_registry.get_current_provider()
        if cfg.model:
            subagent.llm.set_model(cfg.model)
        
        # Initialize subagent's tools
        subagent.tool_registry = await self._create_subagent_tools(cfg, subagent)
        
        self.subagents[subagent_id] = subagent
        subagent.status = "working"
        
        try:
            # Build system prompt
            system_prompt = self._build_subagent_system_prompt(cfg, context)
            
            # Run the query
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=query),
            ]
            
            # Execute with tools
            result = await self._run_subagent_loop(subagent, messages, cfg.temperature)
            
            subagent.status = "completed"
            
            return {
                "success": True,
                "subagent": cfg.name,
                "result": result,
                "subagent_id": subagent_id,
            }
            
        except Exception as e:
            subagent.status = "failed"
            logger.error(f"Subagent {subagent_id} failed: {e}")
            return {
                "success": False,
                "subagent": cfg.name,
                "error": str(e),
                "subagent_id": subagent_id,
            }
        finally:
            # Restore the parent session's model selection (shared singleton).
            if cfg.model and prev_model and self.llm_registry.get_current_model() != prev_model:
                try:
                    self.llm_registry.set_model(prev_model)
                    if prev_provider:
                        self.llm_registry.set_provider(prev_provider)
                except Exception as exc:
                    logger.debug(f"Failed to restore registry model '{prev_model}': {exc}")
            # Cleanup
            if subagent_id in self.subagents:
                del self.subagents[subagent_id]
    
    async def _create_subagent_tools(self, cfg: SubAgentConfig, subagent: SubAgent) -> ToolRegistry:
        """Create tool registry for subagent with allowed tools only (opencode compat)"""
        from agent.tools import (
            TerminalTool, FileSystemTool, SearchTool, GitTool,
            ProcessTool, PatchTool, WebTool, BrowserTool,
            MCPTool, DiagnosticsTool,
            BashTool, ReadTool, WriteTool, EditTool, ApplyPatchTool,
            GrepTool, GlobTool, WebFetchTool, WebSearchTool,
        )
        
        tool_registry = ToolRegistry(agent=subagent)
        tools_cfg = self.agent.config.get("tools", {})
        
        # Create tool instances based on allowed_tools (support both canonical and opencode names)
        if any(k in cfg.allowed_tools for k in ("search", "grep", "glob")):
            await tool_registry.register_tool(SearchTool(self.agent.workspace))
            await tool_registry.register_tool(GrepTool(self.agent.workspace))
            await tool_registry.register_tool(GlobTool(self.agent.workspace))
        if "git" in cfg.allowed_tools:
            # Explore agent gets read-only git
            read_only = (cfg.role == SubAgentRole.EXPLORE)
            await tool_registry.register_tool(GitTool(self.agent.workspace, read_only=read_only))
        if any(k in cfg.allowed_tools for k in ("web", "webfetch", "websearch")):
            await tool_registry.register_tool(WebTool(tools_cfg))
            await tool_registry.register_tool(WebFetchTool(tools_cfg))
            await tool_registry.register_tool(WebSearchTool(tools_cfg))
        if "diagnostics" in cfg.allowed_tools or "lsp" in cfg.allowed_tools:
            await tool_registry.register_tool(DiagnosticsTool(self.agent.workspace))
        
        # Full access tools (general agent only)
        if cfg.role == SubAgentRole.GENERAL:
            if any(k in cfg.allowed_tools for k in ("terminal", "bash")):
                await tool_registry.register_tool(TerminalTool(self.agent.workspace, tools_cfg))
                await tool_registry.register_tool(BashTool(self.agent.workspace, tools_cfg))
            if any(k in cfg.allowed_tools for k in ("filesystem", "read", "write", "edit")):
                await tool_registry.register_tool(FileSystemTool(self.agent.workspace, read_only=False))
                await tool_registry.register_tool(ReadTool(self.agent.workspace))
                await tool_registry.register_tool(WriteTool(self.agent.workspace))
                await tool_registry.register_tool(EditTool(self.agent.workspace))
                await tool_registry.register_tool(ApplyPatchTool(self.agent.workspace))
            if "patch" in cfg.allowed_tools:
                await tool_registry.register_tool(PatchTool(self.agent.workspace))
            if "process" in cfg.allowed_tools:
                await tool_registry.register_tool(ProcessTool(self.agent.workspace))
            if "browser" in cfg.allowed_tools:
                await tool_registry.register_tool(BrowserTool(tools_cfg.get("browser", {})))
            if "mcp" in cfg.allowed_tools and self.agent.mcp_client:
                await tool_registry.register_tool(MCPTool(self.agent.mcp_client))
            # todowrite is disabled for subagents by default (opencode) — keep if explicitly allowed
            if "todowrite" in cfg.allowed_tools or "todo" in cfg.allowed_tools:
                from agent.tools.compat import TodoWriteTool, TodoReadTool
                await tool_registry.register_tool(TodoWriteTool(None))
                await tool_registry.register_tool(TodoReadTool(None))
        
        return tool_registry
    
    def _build_subagent_system_prompt(self, cfg: SubAgentConfig, context: Optional[Dict]) -> str:
        """Build system prompt for subagent"""
        tools_desc = ", ".join(cfg.allowed_tools)
        
        base_prompt = f"""You are the {cfg.name} subagent.
{cfg.description}

Your available tools: {tools_desc}

Guidelines:
1. Think step by step and plan your actions
2. Use tools when necessary
3. Be concise and clear in your responses
4. Report progress and results after each tool call
5. If you encounter an error, explain it and suggest alternatives

Context from parent agent: {json.dumps(context or {}, default=str)[:2000]}"""
        
        if cfg.role == SubAgentRole.EXPLORE:
            base_prompt += """

IMPORTANT: You are in READ-ONLY exploration mode.
- You can ONLY read files, search code, check git status/log/diff
- You CANNOT write, edit, delete, or execute commands
- Focus on understanding the codebase and answering questions"""
        
        elif cfg.role == SubAgentRole.SCOUT:
            base_prompt += """

IMPORTANT: You are a research agent for docs and dependencies.
- Search web for documentation, API references, changelogs
- Analyze dependency sources if needed
- Provide concise findings with sources"""
        
        return base_prompt
    
    async def _run_subagent_loop(
        self,
        subagent: SubAgent,
        messages: List[Message],
        temperature: float
    ) -> str:
        """Run the subagent's thinking-acting loop"""
        max_turns = 10
        final_response = ""
        
        for turn in range(max_turns):
            # Get LLM response
            response = await subagent.llm.complete_with_tools(
                messages=messages,
                tools=subagent.tool_registry.get_schemas() if subagent.tool_registry else [],
                temperature=temperature,
                max_tokens=4000,
            )
            
            final_response = response.content or ""
            
            # Add assistant message
            messages.append(Message(
                role="assistant",
                content=final_response,
                tool_calls=[
                    {"id": tc.id, "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in response.tool_calls
                ] if response.tool_calls else None,
            ))
            
            # Execute tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    try:
                        result = await subagent.tool_registry.execute(
                            tool_call.name,
                            tool_call.arguments
                        )
                        messages.append(Message(
                            role="tool",
                            content=json.dumps(result, default=str)[:5000],
                            tool_call_id=tool_call.id,
                        ))
                    except Exception as e:
                        messages.append(Message(
                            role="tool",
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call.id,
                        ))
                continue
            
            # No tool calls - we're done
            break
        
        return final_response
    
    async def shutdown(self) -> None:
        """Cleanup"""
        for subagent_id in list(self.subagents.keys()):
            del self.subagents[subagent_id]
        logger.info("SubAgentManager shutdown complete")


__all__ = [
    "SubAgentManager",
    "SubAgentRole",
    "SubAgentConfig",
    "SubAgent",
    "BUILTIN_SUBAGENTS",
]