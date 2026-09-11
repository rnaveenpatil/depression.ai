"""
Context Manager - Central orchestrator for agent context.

Responsibilities:
- Maintains the live conversation context (messages, tool outputs, files)
- Tracks token usage against ContextConfig.max_tokens
- Triggers compaction when threshold is exceeded
- Provides retrieval APIs for the agent loop and planner
- Persists context snapshots to storage
"""

from __future__ import annotations

import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from agent.utils.logging import get_logger
from agent.utils.errors import ContextError
from agent.context.project import ProjectContext
from agent.context.files import FileContext
from agent.context.compaction import Compactor

logger = get_logger(__name__)


# ======================================================================
# DATA MODELS
# ======================================================================

@dataclass
class ContextMessage:
    """A single message in the conversation context"""
    role: str                   # "user" | "assistant" | "system" | "tool"
    content: str
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False        # pinned messages are never compacted


@dataclass
class ToolOutput:
    """A recorded tool execution output"""
    tool: str
    params: Dict[str, Any]
    result: Any
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    success: bool = True


@dataclass
class ContextStats:
    """Snapshot of context statistics"""
    total_tokens: int
    max_tokens: int
    usage_pct: float
    message_count: int
    tool_output_count: int
    file_count: int
    last_compaction: Optional[float]
    compaction_count: int


# ======================================================================
# CONTEXT MANAGER
# ======================================================================

class ContextManager:
    """
    Central context orchestrator.

    The manager holds three layers:
      1. `messages`   — conversation history (user / assistant / system)
      2. `tool_outputs` — recent tool results, bounded by recent_tool_outputs
      3. `files`      — file snippets referenced by the agent (via FileContext)
      4. `project`    — project-level summary (via ProjectContext)

    It automatically compacts when token usage crosses
    `ContextConfig.compaction_threshold` and shrinks down to
    `ContextConfig.compaction_target`.
    """

    def __init__(
        self,
        workspace: Any,
        session: Any,
        config: Optional[Dict[str, Any]] = None,
        llm: Any = None,
    ):
        self.workspace = workspace
        self.session = session
        self.config = config or {}
        self.llm = llm

        # Config-driven bounds
        self.max_tokens: int = self.config.get("max_tokens", 100_000)
        self.max_messages: int = self.config.get("max_messages", 200)
        self.compaction_threshold: float = self.config.get("compaction_threshold", 0.8)
        self.compaction_target: float = self.config.get("compaction_target", 0.5)
        self.recent_tool_outputs: int = self.config.get("recent_tool_outputs", 5)
        self.recent_messages: int = self.config.get("recent_messages", 20)
        self.include_file_contents: bool = self.config.get("include_file_contents", True)
        self.include_tool_outputs: bool = self.config.get("include_tool_outputs", True)
        self.enable_summarization: bool = self.config.get("enable_summarization", True)

        # Layer 1: messages
        self.messages: List[ContextMessage] = []
        # Layer 2: tool outputs
        self.tool_outputs: deque = deque(maxlen=max(20, self.recent_tool_outputs * 4))
        # Layer 3: file context
        self.file_context = FileContext(
            workspace=workspace,
            max_tokens=max(4_000, self.max_tokens // 8),
            include_contents=self.include_file_contents,
        )
        # Layer 4: project context
        self.project_context = ProjectContext(workspace=workspace)

        # Compactor
        self.compactor = Compactor(
            llm=llm,
            config={
                "target_ratio": self.compaction_target,
                "enable_summarization": self.enable_summarization,
            },
        )

        # State
        self._last_compaction: Optional[float] = None
        self._compaction_count: int = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"ContextManager initialized (max_tokens={self.max_tokens}, "
            f"threshold={self.compaction_threshold}, target={self.compaction_target})"
        )

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Warm up project + file context from the workspace"""
        try:
            await self.project_context.refresh()
            logger.debug("Project context initialized")
        except Exception as e:
            logger.warning(f"Project context init failed: {e}")

    async def save(self) -> None:
        """Persist context to the session (best-effort)"""
        try:
            if self.session and hasattr(self.session, "update_context"):
                self.session.update_context(self.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save context: {e}")

    # ------------------------------------------------------------------
    # MESSAGE MANAGEMENT
    # ------------------------------------------------------------------

    async def add_user_message(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        msg = ContextMessage(
            role="user",
            content=content,
            tokens=self._estimate_tokens(content),
            metadata=context or {},
        )
        return await self._add_message(msg)

    async def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        msg = ContextMessage(
            role="assistant",
            content=content,
            tokens=self._estimate_tokens(content),
            metadata=metadata or {},
        )
        return await self._add_message(msg)

    async def add_system_message(
        self,
        content: str,
        pinned: bool = False,
    ) -> ContextMessage:
        msg = ContextMessage(
            role="system",
            content=content,
            tokens=self._estimate_tokens(content),
            pinned=pinned,
        )
        return await self._add_message(msg)

    async def add_message(
        self,
        role: str,
        content: str,
        pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        msg = ContextMessage(
            role=role,
            content=content,
            tokens=self._estimate_tokens(content),
            pinned=pinned,
            metadata=metadata or {},
        )
        return await self._add_message(msg)

    async def _add_message(self, msg: ContextMessage) -> ContextMessage:
        async with self._lock:
            self.messages.append(msg)
            # Soft trim: drop oldest non-pinned if we exceeded max_messages
            if len(self.messages) > self.max_messages:
                self._trim_messages()
        return msg

    def _trim_messages(self) -> None:
        """Drop oldest non-pinned messages to stay under max_messages"""
        if len(self.messages) <= self.max_messages:
            return
        pinned = [m for m in self.messages if m.pinned]
        unpinned = [m for m in self.messages if not m.pinned]
        # Keep the newest unpinned messages
        keep = max(0, self.max_messages - len(pinned))
        unpinned = unpinned[-keep:] if keep else []
        # Restore original order
        self.messages = sorted(pinned + unpinned, key=lambda m: m.timestamp)

    # ------------------------------------------------------------------
    # TOOL OUTPUTS
    # ------------------------------------------------------------------

    async def add_tool_output(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
    ) -> ToolOutput:
        success = True
        if isinstance(result, dict):
            success = bool(result.get("success", True))

        rendered = self._render_tool_result(result)
        out = ToolOutput(
            tool=tool_name,
            params=params,
            result=result,
            tokens=self._estimate_tokens(rendered),
            success=success,
        )
        self.tool_outputs.append(out)
        return out

    def get_recent_tool_outputs(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        n = n or self.recent_tool_outputs
        recent = list(self.tool_outputs)[-n:]
        return [
            {
                "tool": t.tool,
                "params": t.params,
                "result": t.result,
                "success": t.success,
                "timestamp": t.timestamp,
            }
            for t in recent
        ]

    def _render_tool_result(self, result: Any) -> str:
        try:
            return json.dumps(result, default=str)
        except Exception:
            return str(result)

    # ------------------------------------------------------------------
    # FILE / PROJECT CONTEXT
    # ------------------------------------------------------------------

    async def add_file_context(
        self,
        path: str,
        content: Optional[str] = None,
    ) -> bool:
        return await self.file_context.add_file(path, content)

    async def get_project_context(self) -> Dict[str, Any]:
        return await self.project_context.get_summary()

    # ------------------------------------------------------------------
    # CONTEXT RETRIEVAL
    # ------------------------------------------------------------------

    async def get_context(
        self,
        include_project: bool = True,
        include_files: bool = True,
        include_tool_outputs: bool = True,
    ) -> Dict[str, Any]:
        """
        Return the current context as a dict suitable for LLM prompts.
        """
        ctx: Dict[str, Any] = {
            "messages": [
                {"role": m.role, "content": m.content}
                for m in self._effective_messages()
            ],
        }
        if include_tool_outputs and self.include_tool_outputs:
            ctx["tool_outputs"] = self.get_recent_tool_outputs()
        if include_files and self.include_file_contents:
            ctx["files"] = await self.file_context.get_summary()
        if include_project:
            ctx["project"] = await self.project_context.get_summary()
        return ctx

    def get_messages(
        self,
        limit: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        msgs = self.messages
        if not include_system:
            msgs = [m for m in msgs if m.role != "system"]
        if limit:
            msgs = msgs[-limit:]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def _effective_messages(self) -> List[ContextMessage]:
        """Messages that are currently 'live' for the LLM"""
        if len(self.messages) <= self.recent_messages:
            return list(self.messages)
        # Always include system/pinned + recent messages
        pinned = [m for m in self.messages if m.pinned or m.role == "system"]
        recent = self.messages[-self.recent_messages:]
        seen = set()
        combined = []
        for m in pinned + recent:
            key = (m.timestamp, m.role, m.content[:32])
            if key in seen:
                continue
            seen.add(key)
            combined.append(m)
        combined.sort(key=lambda m: m.timestamp)
        return combined

    # ------------------------------------------------------------------
    # TOKEN ACCOUNTING
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        msg_tokens = sum(m.tokens for m in self.messages)
        tool_tokens = sum(t.tokens for t in self.tool_outputs)
        return msg_tokens + tool_tokens

    def usage_pct(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.total_tokens() / self.max_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # COMPACTION
    # ------------------------------------------------------------------

    async def needs_compaction(self) -> bool:
        return self.usage_pct() >= self.compaction_threshold

    async def compact(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Compact the context by summarizing older messages.

        Called automatically from the agent loop when needs_compaction()
        returns True, or manually via /compact.
        """
        async with self._lock:
            before_tokens = self.total_tokens()
            if before_tokens == 0:
                return {"compacted": False, "reason": "empty context"}

            target_ratio = 0.35 if aggressive else self.compaction_target
            target_tokens = int(self.max_tokens * target_ratio)

            result = await self.compactor.compact(
                messages=self.messages,
                tool_outputs=list(self.tool_outputs),
                target_tokens=target_tokens,
                current_tokens=before_tokens,
            )

            if not result.get("compacted", False):
                return result

            # Replace messages with the compacted set
            self.messages = result["messages"]
            # Trim tool outputs
            keep = max(1, self.recent_tool_outputs)
            while len(self.tool_outputs) > keep:
                self.tool_outputs.popleft()

            self._last_compaction = time.time()
            self._compaction_count += 1

            after_tokens = self.total_tokens()
            logger.info(
                f"Context compacted: {before_tokens} → {after_tokens} tokens "
                f"({len(self.messages)} messages remain)"
            )

            return {
                "compacted": True,
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "saved_tokens": before_tokens - after_tokens,
                "messages_kept": len(self.messages),
                "summaries_created": result.get("summaries_created", 0),
            }

    async def compact_if_needed(self) -> Optional[Dict[str, Any]]:
        if await self.needs_compaction():
            return await self.compact()
        return None

    # ------------------------------------------------------------------
    # CLEARING
    # ------------------------------------------------------------------

    async def clear(self, keep_system: bool = True) -> None:
        """Clear the conversation context"""
        async with self._lock:
            if keep_system:
                self.messages = [m for m in self.messages if m.role == "system"]
            else:
                self.messages = []
            self.tool_outputs.clear()
            await self.file_context.clear()
        logger.info("Context cleared")

    # ------------------------------------------------------------------
    # STATS / SERIALIZATION
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict[str, Any]:
        stats = ContextStats(
            total_tokens=self.total_tokens(),
            max_tokens=self.max_tokens,
            usage_pct=self.usage_pct(),
            message_count=len(self.messages),
            tool_output_count=len(self.tool_outputs),
            file_count=await self.file_context.count(),
            last_compaction=self._last_compaction,
            compaction_count=self._compaction_count,
        )
        return stats.__dict__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tokens": m.tokens,
                    "timestamp": m.timestamp,
                    "pinned": m.pinned,
                    "metadata": m.metadata,
                }
                for m in self.messages
            ],
            "tool_outputs": [
                {
                    "tool": t.tool,
                    "params": t.params,
                    "result": t.result,
                    "success": t.success,
                    "timestamp": t.timestamp,
                }
                for t in self.tool_outputs
            ],
            "stats": {
                "total_tokens": self.total_tokens(),
                "usage_pct": self.usage_pct(),
                "compaction_count": self._compaction_count,
                "last_compaction": self._last_compaction,
            },
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        self.messages = [
            ContextMessage(
                role=m["role"],
                content=m["content"],
                tokens=m.get("tokens", 0),
                timestamp=m.get("timestamp", time.time()),
                pinned=m.get("pinned", False),
                metadata=m.get("metadata", {}),
            )
            for m in data.get("messages", [])
        ]
        self.tool_outputs.clear()
        for t in data.get("tool_outputs", []):
            self.tool_outputs.append(ToolOutput(
                tool=t["tool"],
                params=t.get("params", {}),
                result=t.get("result"),
                success=t.get("success", True),
                timestamp=t.get("timestamp", time.time()),
            ))