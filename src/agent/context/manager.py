"""Central, session-owned context for the agent runtime."""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.context.project import ProjectContext
from agent.context.files import FileContext
from agent.context.compaction import Compactor

logger = get_logger(__name__)


@dataclass
class ContextMessage:
    role: str
    content: str
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False


@dataclass
class ToolOutput:
    tool: str
    params: Dict[str, Any]
    result: Any
    tokens: int = 0
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    tool_call_id: Optional[str] = None


@dataclass
class ContextStats:
    total_tokens: int
    max_tokens: int
    usage_pct: float
    message_count: int
    tool_output_count: int
    file_count: int
    last_compaction: Optional[float]
    compaction_count: int


class ContextManager:
    """
    Single source of truth for messages, tool results and context state.

    Ownership:
        - messages[] is the canonical conversation log.
        - tool_outputs is a small ring buffer of recent tool results used
          for prompt hints. Tool results are ALSO recorded as `tool`
          messages in messages[]; the two stores are intentionally distinct
          and total_tokens() counts only messages[] to avoid double-counting.
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

        self.max_tokens = self.config.get("max_tokens", 100_000)
        self.max_messages = self.config.get("max_messages", 200)
        self.compaction_threshold = self.config.get("compaction_threshold", 0.8)
        self.compaction_target = self.config.get("compaction_target", 0.5)
        self.recent_tool_outputs = self.config.get("recent_tool_outputs", 5)
        self.recent_messages = self.config.get("recent_messages", 40)
        self.include_file_contents = self.config.get("include_file_contents", True)
        self.include_tool_outputs = self.config.get("include_tool_outputs", True)
        self.enable_summarization = self.config.get("enable_summarization", True)

        # Token counter (best effort; falls back to len//4)
        token_counter = None
        try:
            from agent.llm.rate_limiter import TokenCounter
            token_counter = TokenCounter
        except Exception:
            token_counter = None
        self._token_counter = token_counter

        self.messages: List[ContextMessage] = []
        self.tool_outputs: deque = deque(
            maxlen=max(20, self.recent_tool_outputs * 4)
        )

        self.file_context = FileContext(
            workspace=workspace,
            max_tokens=max(4_000, self.max_tokens // 8),
            include_contents=self.include_file_contents,
            token_counter=token_counter,
        )
        self.project_context = ProjectContext(workspace=workspace)
        self.compactor = Compactor(
            llm=llm,
            config={
                "target_ratio": self.compaction_target,
                "enable_summarization": self.enable_summarization,
            },
        )

        self._last_compaction: Optional[float] = None
        self._compaction_count = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        try:
            await self.project_context.refresh()
        except Exception as e:
            logger.warning("Project context init failed: %s", e)

    async def save(self) -> None:
        try:
            if self.session and hasattr(self.session, "update_context"):
                self.session.update_context(self.to_dict())
        except Exception as e:
            logger.warning("Failed to save context: %s", e)

    # ------------------------------------------------------------------
    # TOKEN ESTIMATION
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._token_counter is not None:
            try:
                model = (
                    self.llm.get_current_model()
                    if self.llm and hasattr(self.llm, "get_current_model")
                    else "default"
                )
                return self._token_counter.count_tokens(text, model or "default")
            except Exception:
                pass
        return max(1, len(text) // 4)

    def total_tokens(self) -> int:
        """
        Sum message tokens only. Tool outputs are duplicated as tool
        messages inside messages[], so counting the buffer would double
        the total and cause premature compaction.
        """
        return sum(m.tokens for m in self.messages)

    def usage_pct(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return min(1.0, self.total_tokens() / self.max_tokens)

    # ------------------------------------------------------------------
    # MESSAGE WRITES
    # ------------------------------------------------------------------

    async def _add_message(self, msg: ContextMessage) -> ContextMessage:
        async with self._lock:
            self.messages.append(msg)
            self._trim_messages()
        return msg

    async def add_message(
        self,
        role: str,
        content: str,
        pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        return await self._add_message(
            ContextMessage(
                role=role,
                content=content,
                tokens=self._estimate_tokens(content),
                metadata=metadata or {},
                pinned=pinned,
            )
        )

    async def add_user_message(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        # Agent owns the user-turn write. This guard only protects against
        # a duplicate write from the same code path (last message equals
        # the incoming content). It is intentionally NOT a general dedupe:
        # a user asking the same thing twice in a row must not be dropped.
        if (
            self.messages
            and self.messages[-1].role == "user"
            and self.messages[-1].content == content
        ):
            return self.messages[-1]
        return await self.add_message(
            "user", content, metadata=context or {}
        )

    async def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        if (
            self.messages
            and self.messages[-1].role == "assistant"
            and self.messages[-1].content == content
            and not metadata
        ):
            return self.messages[-1]
        return await self.add_message("assistant", content, metadata=metadata)

    async def add_system_message(
        self, content: str, pinned: bool = False
    ) -> ContextMessage:
        return await self.add_message("system", content, pinned=pinned)

    # ------------------------------------------------------------------
    # TOOL MESSAGES
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_call_ids(m: ContextMessage) -> set:
        return {
            str(x.get("id"))
            for x in (m.metadata.get("tool_calls") or [])
            if x.get("id")
        }

    def _trim_messages(self) -> None:
        """
        Cap the message list. Preserves pinned/system + the newest
        non-pinned messages. After trimming, normalize_messages() drops
        any orphan tool messages left behind.
        """
        if len(self.messages) <= self.max_messages:
            return

        pinned, unpinned = [], []
        for m in self.messages:
            if m.pinned or m.role == "system":
                pinned.append(m)
            else:
                unpinned.append(m)

        keep = max(0, self.max_messages - len(pinned))
        kept = unpinned[-keep:] if keep else []

        self.messages = sorted(
            pinned + kept, key=lambda m: m.timestamp
        )

        # Drop orphan tool messages (a `tool` with no preceding assistant
        # tool_call, or an assistant tool_call with missing results).
        try:
            from agent.context.runtime import normalize_messages
            normalize_messages(self)
        except Exception as e:
            logger.debug("normalize_messages after trim failed: %s", e)

    async def add_tool_output(
        self,
        tool_name: str,
        params: Dict[str, Any],
        result: Any,
        tool_call_id: Optional[str] = None,
    ) -> ToolOutput:
        rendered = self._render_tool_result(result)
        out = ToolOutput(
            tool=tool_name,
            params=params,
            result=result,
            tokens=self._estimate_tokens(rendered),
            success=bool(result.get("success", True)) if isinstance(result, dict) else True,
            tool_call_id=tool_call_id,
        )
        self.tool_outputs.append(out)
        return out

    def _render_tool_result(self, result: Any) -> str:
        try:
            return json.dumps(result, default=str)
        except Exception:
            return str(result)

    def get_recent_tool_outputs(self, n: Optional[int] = None):
        recent = list(self.tool_outputs)[-(n or self.recent_tool_outputs):]
        return [
            {
                "tool": t.tool,
                "params": t.params,
                "result": t.result,
                "success": t.success,
                "timestamp": t.timestamp,
                "tool_call_id": t.tool_call_id,
            }
            for t in recent
        ]

    # ------------------------------------------------------------------
    # FILE / PROJECT CONTEXT
    # ------------------------------------------------------------------

    async def add_file_context(
        self, path: str, content: Optional[str] = None
    ) -> bool:
        return await self.file_context.add_file(path, content)

    async def get_project_context(self) -> Dict[str, Any]:
        return await self.project_context.get_summary()

    # ------------------------------------------------------------------
    # EFFECTIVE MESSAGES (windowed view for the model)
    # ------------------------------------------------------------------

    def _effective_messages(self) -> List[ContextMessage]:
        """
        Return pinned + a windowed tail of non-pinned messages, keeping
        assistant tool_call groups intact. Group reconstruction walks
        backward from the tail and consumes whole groups in one step.
        """
        from agent.context.runtime import normalize_messages
        normalize_messages(self)

        msgs = list(self.messages)
        if len(msgs) <= self.recent_messages:
            return msgs

        pinned, unpinned = [], []
        for m in msgs:
            if m.pinned or m.role == "system":
                pinned.append(m)
            else:
                unpinned.append(m)

        selected: List[ContextMessage] = []
        i = len(unpinned) - 1
        while i >= 0 and len(selected) < self.recent_messages:
            m = unpinned[i]
            if m.role == "tool":
                tid = str(m.metadata.get("tool_call_id") or "")
                if (
                    i > 0
                    and unpinned[i - 1].role == "assistant"
                    and tid in self._tool_call_ids(unpinned[i - 1])
                ):
                    parent = unpinned[i - 1]
                    ids = self._tool_call_ids(parent)
                    # Collect the assistant + all matching tool results.
                    group: List[ContextMessage] = [parent]
                    j = i
                    while (
                        j < len(unpinned)
                        and unpinned[j].role == "tool"
                        and str(unpinned[j].metadata.get("tool_call_id")) in ids
                    ):
                        group.append(unpinned[j])
                        j += 1
                    if len(selected) + len(group) <= self.recent_messages:
                        selected[0:0] = group
                    # Advance past the whole group. -1 moves to the
                    # assistant we just consumed.
                    i -= len(group)
                    continue
            selected.insert(0, m)
            i -= 1

        return sorted(pinned + selected, key=lambda m: m.timestamp)

    async def get_context(
        self,
        include_project: bool = True,
        include_files: bool = True,
        include_tool_outputs: bool = True,
    ) -> Dict[str, Any]:
        from agent.context.runtime import get_model_messages

        ctx: Dict[str, Any] = {
            "messages": [m.__dict__ for m in self._effective_messages()],
            "model_messages": get_model_messages(self, self.recent_messages),
        }
        if include_tool_outputs and self.include_tool_outputs:
            ctx["tool_outputs"] = self.get_recent_tool_outputs()
        if include_files and self.include_file_contents:
            ctx["files"] = await self.file_context.get_summary()
        if include_project:
            ctx["project"] = await self.project_context.get_summary()
        return ctx

    def get_messages(
        self, limit: Optional[int] = None, include_system: bool = True
    ) -> List[Dict[str, Any]]:
        msgs = self._effective_messages()
        if not include_system:
            msgs = [m for m in msgs if m.role != "system"]
        if limit:
            msgs = msgs[-limit:]
        out = []
        for m in msgs:
            entry: Dict[str, Any] = {"role": m.role, "content": m.content}
            if m.metadata.get("tool_calls"):
                entry["tool_calls"] = m.metadata["tool_calls"]
            if m.role == "tool":
                entry["tool_call_id"] = m.metadata.get("tool_call_id")
                entry["name"] = m.metadata.get("name")
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # SESSION SEEDING
    # ------------------------------------------------------------------

    async def load_from_session(self, session: Any) -> int:
        """
        Seed messages[] from a Session's history. Called on resume so the
        model's context matches what the transcript shows the user.
        Returns the number of messages imported.
        """
        if session is None:
            return 0
        history = getattr(session, "history", None)
        if history is None:
            return 0

        entries = list(history.all()) if hasattr(history, "all") else list(history)
        if not entries:
            return 0

        # Keep the last N entries to respect the context window.
        entries = entries[-max(self.recent_messages * 2, 40):]

        async with self._lock:
            self.messages = []
            for e in entries:
                role = getattr(e, "role", "user")
                content = getattr(e, "content", "") or ""
                metadata: Dict[str, Any] = {}

                # Rehydrate tool-call metadata for assistant tool turns.
                if role == "assistant":
                    tcalls = getattr(e, "metadata", {}) or {}
                    if "tool_calls" in tcalls:
                        metadata["tool_calls"] = tcalls["tool_calls"]
                elif role == "tool":
                    metadata["tool_call_id"] = getattr(e, "tool_call_id", None)
                    metadata["name"] = getattr(e, "tool_name", None)

                self.messages.append(
                    ContextMessage(
                        role=role,
                        content=content,
                        tokens=getattr(e, "tokens", 0)
                        or self._estimate_tokens(content),
                        timestamp=getattr(e, "timestamp", time.time()),
                        metadata=metadata,
                        pinned=bool(getattr(e, "pinned", False)),
                    )
                )

            # Drop orphans so the first request is well-formed.
            try:
                from agent.context.runtime import normalize_messages
                normalize_messages(self)
            except Exception:
                pass

        logger.info(
            "Seeded context from session: %d message(s)", len(self.messages)
        )
        return len(self.messages)

    # ------------------------------------------------------------------
    # COMPACTION
    # ------------------------------------------------------------------

    async def needs_compaction(self) -> bool:
        return self.usage_pct() >= self.compaction_threshold

    async def compact(self, aggressive: bool = False) -> Dict[str, Any]:
        from agent.context.runtime import normalize_messages

        async with self._lock:
            normalize_messages(self)
            before = self.total_tokens()
            if not before:
                return {"compacted": False, "reason": "empty context"}

            target = int(
                self.max_tokens * (0.35 if aggressive else self.compaction_target)
            )
            result = await self.compactor.compact(
                messages=self.messages,
                tool_outputs=list(self.tool_outputs),
                target_tokens=target,
                current_tokens=before,
            )
            if not result.get("compacted"):
                return result

            self.messages = result["messages"]
            normalize_messages(self)

            while len(self.tool_outputs) > max(1, self.recent_tool_outputs):
                self.tool_outputs.popleft()

            self._last_compaction = time.time()
            self._compaction_count += 1
            return {
                "compacted": True,
                "before_tokens": before,
                "after_tokens": self.total_tokens(),
                "saved_tokens": before - self.total_tokens(),
                "messages_kept": len(self.messages),
                "summaries_created": result.get("summaries_created", 0),
            }

    async def compact_if_needed(self) -> Optional[Dict[str, Any]]:
        return await self.compact() if await self.needs_compaction() else None

    # ------------------------------------------------------------------
    # CLEAR / STATS / SERIALIZATION
    # ------------------------------------------------------------------

    async def clear(self, keep_system: bool = True) -> None:
        async with self._lock:
            self.messages = (
                [m for m in self.messages if m.role == "system"]
                if keep_system
                else []
            )
            self.tool_outputs.clear()
            await self.file_context.clear()

    async def get_stats(self) -> Dict[str, Any]:
        return ContextStats(
            total_tokens=self.total_tokens(),
            max_tokens=self.max_tokens,
            usage_pct=self.usage_pct(),
            message_count=len(self.messages),
            tool_output_count=len(self.tool_outputs),
            file_count=await self.file_context.count(),
            last_compaction=self._last_compaction,
            compaction_count=self._compaction_count,
        ).__dict__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.__dict__ for m in self.messages],
            "tool_outputs": [t.__dict__ for t in self.tool_outputs],
            "compaction_count": self._compaction_count,
            "last_compaction": self._last_compaction,
        }