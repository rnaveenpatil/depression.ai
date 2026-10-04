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
from agent.context.compaction import Compactor, _estimate_message_tokens
from agent.context.summary_store import SummaryStore

logger = get_logger(__name__)


# Cap the number of pinned (protected) messages. If more than this many
# accumulate, the newest N are kept and the rest are treated as normal
# messages so they can age out. [Bug 3]
MAX_PINNED_MESSAGES = 32


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


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def _is_compaction_summary(m: Any) -> bool:
    return bool((getattr(m, "metadata", {}) or {}).get("compacted"))


def _is_system_prompt(m: Any) -> bool:
    return bool((getattr(m, "metadata", {}) or {}).get("system_prompt"))


def _split_protected(messages: List[Any]) -> tuple[List[Any], List[Any]]:
    """
    Split into (protected, rest) using the *explicit* pinned flag only.

    [Bug 3] Before this, role=="system" was treated as protected too, so
    every plan-guidance message and every compaction summary became
    unkillable and eventually evicted the real conversation.
    """
    protected: List[Any] = []
    rest: List[Any] = []
    for m in messages:
        if bool(getattr(m, "pinned", False)):
            protected.append(m)
        else:
            rest.append(m)
    return protected, rest


def _drop_stale_summaries(messages: List[Any]) -> List[Any]:
    """
    Keep at most the newest compaction summary. [Bug 4]

    Older summaries describe turns that are no longer in the window; they
    are worse than useless — they read as current narrative.
    """
    summaries = [m for m in messages if _is_compaction_summary(m)]
    if len(summaries) <= 1:
        return messages
    newest = max(summaries, key=lambda m: getattr(m, "timestamp", 0))
    return [m for m in messages if not _is_compaction_summary(m) or m is newest]


def _cap_protected(protected: List[Any], cap: int) -> List[Any]:
    """Keep only the newest `cap` protected messages. [Bug 3]"""
    if len(protected) <= cap:
        return protected
    protected_sorted = sorted(protected, key=lambda m: getattr(m, "timestamp", 0))
    kept_ids = {id(m) for m in protected_sorted[-cap:]}
    return [m for m in protected if id(m) in kept_ids]


class ContextManager:
    """
    Single source of truth for messages, tool results and context state.

    Ownership:
        - messages[] is the ONLY conversation log fed to the model.
        - tool_outputs is a small ring buffer of recent tool results used
          for prompt hints. It never feeds the provider request directly;
          each tool result is ALSO present as a `tool` message.

    Long-term memory:
        - On compaction, the newest summary is written to a per-session
          JSONL file via `SummaryStore`.
        - On session resume, the newest N summaries from disk are
          re-injected as pinned system messages so prior context survives
          restarts.
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
        self.persist_summaries = self.config.get("persist_summaries", True)
        self.recover_summaries_on_load = self.config.get(
            "recover_summaries_on_load", 3
        )

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

        # [long-term memory] Persistent summary store, per session.
        self.summary_store = SummaryStore(
            root=self.config.get("summary_dir") or None
        )
        self._session_id = self._resolve_session_id()

    # ------------------------------------------------------------------
    # SESSION ID
    # ------------------------------------------------------------------

    def _resolve_session_id(self) -> str:
        try:
            if self.session is not None and getattr(self.session, "id", None):
                return str(self.session.id)
        except Exception:
            pass
        return "default"

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
                return int(self._token_counter.count_tokens(text, model or "default"))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _estimate_message_tokens_full(self, m: ContextMessage) -> int:
        return _estimate_message_tokens(m)

    def total_tokens(self) -> int:
        return sum(
            m.tokens or self._estimate_message_tokens_full(m)
            for m in self.messages
        )

    def extra_request_tokens(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> int:
        total = 0
        if system_prompt:
            total += self._estimate_tokens(system_prompt)
        if tool_schemas:
            try:
                total += self._estimate_tokens(json.dumps(tool_schemas, default=str))
            except Exception:
                total += 512
        if state_block:
            total += self._estimate_tokens(state_block)
        return total

    def real_request_tokens(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> int:
        return self.total_tokens() + self.extra_request_tokens(
            system_prompt, tool_schemas, state_block
        )

    def usage_pct(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> float:
        if self.max_tokens <= 0:
            return 0.0
        used = self.real_request_tokens(system_prompt, tool_schemas, state_block)
        return min(1.0, float(used) / float(self.max_tokens))

    # ------------------------------------------------------------------
    # MESSAGE WRITES
    # ------------------------------------------------------------------

    async def _add_message(self, msg: ContextMessage) -> ContextMessage:
        if not msg.tokens:
            msg.tokens = self._estimate_message_tokens_full(msg)
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        meta = dict(context or {})
        if metadata:
            meta.update(metadata)
        if (
            self.messages
            and self.messages[-1].role == "user"
            and self.messages[-1].content == content
        ):
            return self.messages[-1]
        return await self.add_message("user", content, metadata=meta)

    async def add_assistant_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        """
        Add an assistant message.

        [M2/N4] Dedup is checked against the last few assistant messages,
        not just the immediately previous one.
        """
        for m in reversed(self.messages[-8:]):
            if m.role != "assistant":
                if m.role == "user":
                    break
                continue
            if m.content == content and not metadata:
                return m
        return await self.add_message("assistant", content, metadata=metadata)

    async def add_system_message(
        self, content: str, pinned: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextMessage:
        meta = dict(metadata or {})
        return await self.add_message("system", content, pinned=pinned, metadata=meta)

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
        Cap the message list.

        [Bug 3] Only explicitly pinned messages are protected. System
        messages are normal citizens. Pinned set is capped. [N7] Repair
        is only run when we actually trimmed something.
        """
        if len(self.messages) <= self.max_messages:
            return

        protected, rest = _split_protected(self.messages)
        protected = _cap_protected(protected, MAX_PINNED_MESSAGES)

        keep = max(0, self.max_messages - len(protected))
        kept = rest[-keep:] if keep else []

        merged = protected + kept
        merged.sort(key=lambda m: getattr(m, "timestamp", 0))

        trimmed = len(self.messages) - len(merged)

        # [Bug 4] Ensure at most one compaction summary survives.
        merged = _drop_stale_summaries(merged)

        self.messages = merged

        if trimmed > 0:
            try:
                from agent.context.runtime import repair_context
                repairs = repair_context(self, fill_missing=True)
                if repairs:
                    logger.debug("trim repair: %d fix(es)", repairs)
            except Exception as e:
                logger.debug("repair_context after trim failed: %s", e)

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
            success=bool(result.get("success", True))
            if isinstance(result, dict) else True,
            tool_call_id=tool_call_id,
        )
        self.tool_outputs.append(out)
        return out

    def _render_tool_result(self, result: Any) -> str:
        try:
            return json.dumps(result, default=str)
        except Exception:
            return str(result)

    def get_recent_tool_outputs(
        self, n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
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
        from agent.context.runtime import repair_context
        repair_context(self, fill_missing=True)

        msgs = list(self.messages)
        if len(msgs) <= self.recent_messages:
            return _drop_stale_summaries(msgs)

        protected, rest = _split_protected(msgs)
        protected = _cap_protected(protected, MAX_PINNED_MESSAGES)

        selected: List[ContextMessage] = []
        i = len(rest) - 1
        while i >= 0 and len(selected) < self.recent_messages:
            m = rest[i]
            if m.role == "tool":
                tid = str(m.metadata.get("tool_call_id") or "")
                if (
                    i > 0
                    and rest[i - 1].role == "assistant"
                    and tid in self._tool_call_ids(rest[i - 1])
                ):
                    parent = rest[i - 1]
                    ids = self._tool_call_ids(parent)
                    group: List[ContextMessage] = [parent]
                    j = i
                    while (
                        j < len(rest)
                        and rest[j].role == "tool"
                        and str(rest[j].metadata.get("tool_call_id")) in ids
                    ):
                        group.append(rest[j])
                        j += 1
                    if len(selected) + len(group) <= self.recent_messages:
                        selected[0:0] = group
                    i -= len(group)
                    continue
            selected.insert(0, m)
            i -= 1

        combined = protected + selected
        combined.sort(key=lambda m: getattr(m, "timestamp", 0))
        return _drop_stale_summaries(combined)

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
    # SESSION SEEDING + PERSISTED SUMMARY RECOVERY
    # ------------------------------------------------------------------

    async def load_from_session(self, session: Any) -> int:
        if session is None:
            return 0

        # Refresh session id in case this manager was created before the
        # session was set.
        try:
            if getattr(session, "id", None):
                self._session_id = str(session.id)
        except Exception:
            pass

        history = getattr(session, "history", None)
        entries: List[Any] = []
        if history is not None:
            entries = list(history.all()) if hasattr(history, "all") else list(history)

        if entries:
            entries = entries[-max(self.recent_messages * 2, 40):]

        async with self._lock:
            self.messages = []
            for e in entries:
                role = getattr(e, "role", "user")
                content = getattr(e, "content", "") or ""
                metadata: Dict[str, Any] = {}

                if role == "assistant":
                    tcalls = getattr(e, "metadata", {}) or {}
                    if "tool_calls" in tcalls:
                        metadata["tool_calls"] = tcalls["tool_calls"]
                elif role == "tool":
                    metadata["tool_call_id"] = getattr(e, "tool_call_id", None)
                    metadata["name"] = getattr(e, "tool_name", None)

                msg = ContextMessage(
                    role=role,
                    content=content,
                    tokens=getattr(e, "tokens", 0)
                    or self._estimate_tokens(content),
                    timestamp=getattr(e, "timestamp", time.time()),
                    metadata=metadata,
                    pinned=bool(getattr(e, "pinned", False)),
                )
                if not msg.tokens:
                    msg.tokens = self._estimate_message_tokens_full(msg)
                self.messages.append(msg)

            try:
                from agent.context.runtime import repair_context
                repair_context(self, fill_missing=True)
            except Exception:
                pass

        logger.info("Seeded context from session: %d message(s)", len(self.messages))

        # [long-term memory] Pull persisted summaries back in.
        try:
            await self.load_persisted_summaries(
                limit=self.recover_summaries_on_load
            )
        except Exception as exc:
            logger.debug("load_persisted_summaries failed: %s", exc)

        return len(self.messages)

    async def load_persisted_summaries(self, limit: int = 3) -> int:
        """
        Inject the newest N summaries from disk as pinned system messages.

        Called on resume so the model sees prior-session memory even if
        the raw history has already been compacted away.
        """
        if limit <= 0:
            return 0
        try:
            rows = self.summary_store.load(self._session_id, limit=limit)
        except Exception as exc:
            logger.debug("summary load failed: %s", exc)
            return 0

        if not rows:
            return 0

        added = 0
        for row in rows:
            body = str(row.get("summary") or "").strip()
            if not body:
                continue
            text = f"[PRIOR CONTEXT — recovered from disk]\n{body}"
            try:
                await self.add_message(
                    role="system",
                    content=text,
                    pinned=True,
                    metadata={"compacted": True, "recovered": True},
                )
                added += 1
            except Exception as exc:
                logger.debug("inject summary failed: %s", exc)

        logger.info(
            "Re-injected %d persisted summary message(s) for session %s",
            added, self._session_id,
        )
        return added

    # ------------------------------------------------------------------
    # COMPACTION
    # ------------------------------------------------------------------

    async def needs_compaction(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> bool:
        return self.usage_pct(system_prompt, tool_schemas, state_block) >= self.compaction_threshold

    async def compact(
        self,
        aggressive: bool = False,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> Dict[str, Any]:
        from agent.context.runtime import repair_context

        async with self._lock:
            repair_context(self, fill_missing=True)
            before = self.total_tokens()
            if not before:
                return {"compacted": False, "reason": "empty context"}

            extra = self.extra_request_tokens(system_prompt, tool_schemas, state_block)
            target = int(
                self.max_tokens * (0.35 if aggressive else self.compaction_target)
            )
            result = await self.compactor.compact(
                messages=self.messages,
                tool_outputs=list(self.tool_outputs),
                target_tokens=target,
                current_tokens=before,
                extra_tokens=extra,
            )
            if not result.get("compacted"):
                return result

            self.messages = result["messages"]
            repair_context(self, fill_missing=True)
            self.messages = _drop_stale_summaries(self.messages)

            while len(self.tool_outputs) > max(1, self.recent_tool_outputs):
                self.tool_outputs.popleft()

            self._last_compaction = time.time()
            self._compaction_count += 1

            # [long-term memory] Persist the newest summary to disk so it
            # survives restarts and can be re-injected on resume.
            if self.persist_summaries:
                try:
                    newest_summary = None
                    for m in reversed(self.messages):
                        if _is_compaction_summary(m):
                            newest_summary = m
                            break
                    if newest_summary is not None:
                        body = str(getattr(newest_summary, "content", "") or "")
                        if body:
                            meta = getattr(newest_summary, "metadata", {}) or {}
                            self.summary_store.append(
                                self._session_id,
                                body,
                                turns_summarized=int(meta.get("summary_of", 0) or 0),
                                tokens_before=before,
                                tokens_after=self.total_tokens(),
                                metadata={
                                    "compaction_count": self._compaction_count,
                                },
                            )
                except Exception as exc:
                    logger.debug("summary persist failed: %s", exc)

            return {
                "compacted": True,
                "before_tokens": before,
                "after_tokens": self.total_tokens(),
                "saved_tokens": before - self.total_tokens(),
                "messages_kept": len(self.messages),
                "summaries_created": result.get("summaries_created", 0),
            }

    async def compact_if_needed(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if await self.needs_compaction(system_prompt, tool_schemas, state_block):
            return await self.compact(
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                state_block=state_block,
            )
        return None

    # ------------------------------------------------------------------
    # CLEAR / STATS / SERIALIZATION
    # ------------------------------------------------------------------

    async def clear(self, keep_system: bool = True) -> None:
        async with self._lock:
            self.messages = (
                [m for m in self.messages if m.role == "system"]
                if keep_system else []
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