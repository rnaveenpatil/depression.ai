"""Central, session-owned context for the agent runtime."""
from __future__ import annotations
import time, json, asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
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
    """Single source of truth for messages, tool results and context state."""
    def __init__(self, workspace: Any, session: Any, config: Optional[Dict[str, Any]] = None, llm: Any = None):
        self.workspace, self.session, self.config, self.llm = workspace, session, config or {}, llm
        self.max_tokens = self.config.get("max_tokens", 100_000)
        self.max_messages = self.config.get("max_messages", 200)
        self.compaction_threshold = self.config.get("compaction_threshold", 0.8)
        self.compaction_target = self.config.get("compaction_target", 0.5)
        self.recent_tool_outputs = self.config.get("recent_tool_outputs", 5)
        self.recent_messages = self.config.get("recent_messages", 40)
        self.include_file_contents = self.config.get("include_file_contents", True)
        self.include_tool_outputs = self.config.get("include_tool_outputs", True)
        self.enable_summarization = self.config.get("enable_summarization", True)
        self.messages: List[ContextMessage] = []
        self.tool_outputs = deque(maxlen=max(20, self.recent_tool_outputs * 4))
        self.file_context = FileContext(workspace=workspace, max_tokens=max(4_000, self.max_tokens // 8), include_contents=self.include_file_contents)
        self.project_context = ProjectContext(workspace=workspace)
        self.compactor = Compactor(llm=llm, config={"target_ratio": self.compaction_target, "enable_summarization": self.enable_summarization})
        self._last_compaction = None
        self._compaction_count = 0
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        try: await self.project_context.refresh()
        except Exception as e: logger.warning("Project context init failed: %s", e)

    async def save(self) -> None:
        try:
            if self.session and hasattr(self.session, "update_context"): self.session.update_context(self.to_dict())
        except Exception as e: logger.warning("Failed to save context: %s", e)

    def _estimate_tokens(self, text: str) -> int:
        if not text: return 0
        try:
            from agent.llm.rate_limiter import TokenCounter
            model = self.llm.get_current_model() if self.llm and hasattr(self.llm, "get_current_model") else "default"
            return TokenCounter.count_tokens(text, model or "default")
        except Exception: return max(1, len(text) // 4)

    async def _add_message(self, msg: ContextMessage) -> ContextMessage:
        async with self._lock:
            self.messages.append(msg)
            self._trim_messages()
        return msg

    async def add_message(self, role: str, content: str, pinned: bool = False, metadata: Optional[Dict[str, Any]] = None) -> ContextMessage:
        return await self._add_message(ContextMessage(role, content, self._estimate_tokens(content), metadata=metadata or {}, pinned=pinned))

    async def add_user_message(self, content: str, context: Optional[Dict[str, Any]] = None):
        return await self.add_message("user", content, metadata=context)

    async def add_assistant_message(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        return await self.add_message("assistant", content, metadata=metadata)

    async def add_system_message(self, content: str, pinned: bool = False):
        return await self.add_message("system", content, pinned=pinned)

    def _tool_call_ids(self, m: ContextMessage) -> set[str]:
        return {str(x.get("id")) for x in (m.metadata.get("tool_calls") or []) if x.get("id")}

    def _trim_messages(self) -> None:
        if len(self.messages) <= self.max_messages: return
        pinned = [m for m in self.messages if m.pinned or m.role == "system"]
        others = [m for m in self.messages if m not in pinned]
        keep = max(0, self.max_messages - len(pinned))
        kept = others[-keep:] if keep else []
        # Never keep an orphan tool result or an assistant tool call without all results.
        while kept and kept[0].role == "tool": kept.pop(0)
        while kept and kept[0].role == "assistant" and self._tool_call_ids(kept[0]):
            ids = self._tool_call_ids(kept[0]); results = [m for m in kept[1:] if m.role == "tool"]
            if not ids.issubset({str(m.metadata.get("tool_call_id")) for m in results}): kept.pop(0)
            else: break
        self.messages = sorted(pinned + kept, key=lambda m: m.timestamp)

    async def add_tool_output(self, tool_name: str, params: Dict[str, Any], result: Any, tool_call_id: Optional[str] = None):
        rendered = self._render_tool_result(result)
        out = ToolOutput(tool_name, params, result, self._estimate_tokens(rendered), success=bool(result.get("success", True)) if isinstance(result, dict) else True, tool_call_id=tool_call_id)
        self.tool_outputs.append(out)
        return out

    def _render_tool_result(self, result: Any) -> str:
        try: return json.dumps(result, default=str)
        except Exception: return str(result)

    async def add_tool_message(self, tool_name: str, result: Any, tool_call_id: str, params: Optional[Dict[str, Any]] = None):
        await self.add_tool_output(tool_name, params or {}, result, tool_call_id)
        return await self.add_message("tool", self._render_tool_result(result), metadata={"tool_call_id": tool_call_id, "name": tool_name})

    def get_recent_tool_outputs(self, n: Optional[int] = None):
        recent = list(self.tool_outputs)[-(n or self.recent_tool_outputs):]
        return [{"tool": t.tool, "params": t.params, "result": t.result, "success": t.success, "timestamp": t.timestamp, "tool_call_id": t.tool_call_id} for t in recent]

    async def add_file_context(self, path: str, content: Optional[str] = None): return await self.file_context.add_file(path, content)
    async def get_project_context(self): return await self.project_context.get_summary()

    def _effective_messages(self):
        from agent.context.runtime import normalize_messages
        normalize_messages(self)
        msgs = list(self.messages)
        if len(msgs) <= self.recent_messages: return msgs
        pinned = [m for m in msgs if m.pinned or m.role == "system"]
        others = [m for m in msgs if m not in pinned]
        selected = []
        i = len(others) - 1
        while i >= 0 and len(selected) < self.recent_messages:
            m = others[i]
            if m.role == "tool":
                tid = str(m.metadata.get("tool_call_id") or "")
                if i > 0 and others[i-1].role == "assistant" and tid in self._tool_call_ids(others[i-1]):
                    group = [others[i-1]]
                    j = i
                    ids = self._tool_call_ids(others[i-1])
                    while j < len(others) and others[j].role == "tool" and str(others[j].metadata.get("tool_call_id")) in ids:
                        group.append(others[j]); j += 1
                    if len(selected) + len(group) <= self.recent_messages: selected[0:0] = group
                    i -= 1
                    continue
            selected.insert(0, m); i -= 1
        return sorted(pinned + selected, key=lambda m: m.timestamp)

    async def get_context(self, include_project=True, include_files=True, include_tool_outputs=True):
        from agent.context.runtime import get_model_messages
        ctx = {"messages": [m.__dict__ for m in self._effective_messages()]}
        ctx["model_messages"] = get_model_messages(self, self.recent_messages)
        if include_tool_outputs and self.include_tool_outputs: ctx["tool_outputs"] = self.get_recent_tool_outputs()
        if include_files and self.include_file_contents: ctx["files"] = await self.file_context.get_summary()
        if include_project: ctx["project"] = await self.project_context.get_summary()
        return ctx

    def get_messages(self, limit=None, include_system=True):
        msgs = self._effective_messages()
        if not include_system: msgs = [m for m in msgs if m.role != "system"]
        return [{"role": m.role, "content": m.content, **({"tool_calls": m.metadata["tool_calls"]} if m.metadata.get("tool_calls") else {}), **({"tool_call_id": m.metadata["tool_call_id"], "name": m.metadata.get("name")} if m.role == "tool" else {})} for m in (msgs[-limit:] if limit else msgs)]

    def total_tokens(self): return sum(m.tokens for m in self.messages) + sum(t.tokens for t in self.tool_outputs)
    def usage_pct(self): return min(1.0, self.total_tokens() / self.max_tokens) if self.max_tokens > 0 else 0.0
    async def needs_compaction(self): return self.usage_pct() >= self.compaction_threshold

    async def compact(self, aggressive=False):
        from agent.context.runtime import normalize_messages
        async with self._lock:
            normalize_messages(self)
            before = self.total_tokens()
            if not before: return {"compacted": False, "reason": "empty context"}
            target = int(self.max_tokens * (0.35 if aggressive else self.compaction_target))
            result = await self.compactor.compact(messages=self.messages, tool_outputs=list(self.tool_outputs), target_tokens=target, current_tokens=before)
            if not result.get("compacted"): return result
            self.messages = result["messages"]
            normalize_messages(self)
            while len(self.tool_outputs) > max(1, self.recent_tool_outputs): self.tool_outputs.popleft()
            self._last_compaction, self._compaction_count = time.time(), self._compaction_count + 1
            return {"compacted": True, "before_tokens": before, "after_tokens": self.total_tokens(), "saved_tokens": before-self.total_tokens(), "messages_kept": len(self.messages), "summaries_created": result.get("summaries_created", 0)}

    async def compact_if_needed(self): return await self.compact() if await self.needs_compaction() else None
    async def clear(self, keep_system=True):
        async with self._lock:
            self.messages = [m for m in self.messages if m.role == "system"] if keep_system else []
            self.tool_outputs.clear(); await self.file_context.clear()
    async def get_stats(self):
        return ContextStats(self.total_tokens(), self.max_tokens, self.usage_pct(), len(self.messages), len(self.tool_outputs), await self.file_context.count(), self._last_compaction, self._compaction_count).__dict__
    def to_dict(self):
        return {"messages": [m.__dict__ for m in self.messages], "tool_outputs": [t.__dict__ for t in self.tool_outputs], "compaction_count": self._compaction_count, "last_compaction": self._last_compaction}
