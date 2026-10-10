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


# Cap the number of pinned (protected) messages.
MAX_PINNED_MESSAGES = 32

# Keep this many recent messages verbatim before compaction kicks in.
KEEP_VERBATIM = 20

# Trigger compaction when the message list reaches this size, not when the
# context window fills. Compact-on-schedule is cheaper than compact-on-
# overflow.
COMPACT_AT_MESSAGES = KEEP_VERBATIM + 10

# The running summary is capped in size — it is a summary, not a log.
RUNNING_SUMMARY_MAX_TOKENS = 400


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


def _is_running_summary(m: Any) -> bool:
    return bool((getattr(m, "metadata", {}) or {}).get("running_summary"))


def _split_protected(messages: List[Any]) -> tuple[List[Any], List[Any]]:
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
    Keep at most one compaction summary AND one running summary.
    """
    compactions = [m for m in messages if _is_compaction_summary(m)]
    runnings = [m for m in messages if _is_running_summary(m)]

    keep_ids = set()
    if compactions:
        newest = max(compactions, key=lambda m: getattr(m, "timestamp", 0))
        keep_ids.add(id(newest))
    if runnings:
        newest = max(runnings, key=lambda m: getattr(m, "timestamp", 0))
        keep_ids.add(id(newest))

    out: List[Any] = []
    for m in messages:
        if _is_compaction_summary(m) or _is_running_summary(m):
            if id(m) in keep_ids:
                out.append(m)
            continue
        out.append(m)
    return out


def _cap_protected(protected: List[Any], cap: int) -> List[Any]:
    if len(protected) <= cap:
        return protected
    protected_sorted = sorted(protected, key=lambda m: getattr(m, "timestamp", 0))
    kept_ids = {id(m) for m in protected_sorted[-cap:]}
    return [m for m in protected if id(m) in kept_ids]


# ----------------------------------------------------------------------
# TOOL RESULT STUBBING
# ----------------------------------------------------------------------

def _tool_outcome_stub(m: Any) -> str:
    """
    One-line summary of a tool message, used when compacting.

    Never contains the raw payload. Records what tool ran and whether it
    succeeded. The full payload is dropped; the summary text carries the
    *meaning*.
    """
    meta = getattr(m, "metadata", {}) or {}
    tool = meta.get("name") or "tool"
    try:
        payload = json.loads(getattr(m, "content", "") or "{}")
    except Exception:
        return f"{tool}: (unparseable result)"

    if isinstance(payload, dict):
        if payload.get("success") is False:
            err = str(payload.get("error") or "error")[:120]
            return f"{tool}: FAILED — {err}"
        # Success — record a hint about the shape.
        if "exit_code" in payload:
            code = payload.get("exit_code")
            return f"{tool}: exit {code}"
        if "bytes_written" in payload:
            return f"{tool}: wrote {payload.get('bytes_written')} bytes"
        if isinstance(payload.get("content"), str):
            lines = payload["content"].count("\n") + 1
            return f"{tool}: returned {lines} lines"
        return f"{tool}: success"
    return f"{tool}: done"


class ContextManager:
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
        self.recent_messages = self.config.get("recent_messages", KEEP_VERBATIM)
        self.include_file_contents = self.config.get("include_file_contents", True)
        self.include_tool_outputs = self.config.get("include_tool_outputs", True)
        self.enable_summarization = self.config.get("enable_summarization", True)
        self.persist_summaries = self.config.get("persist_summaries", True)
        self.recover_summaries_on_load = self.config.get(
            "recover_summaries_on_load", 3
        )

        # Running summary lives across compactions.
        self._running_summary: str = ""

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

        self.summary_store = SummaryStore(
            root=self.config.get("summary_dir") or None
        )
        self._session_id = self._resolve_session_id()

    # ------------------------------------------------------------------
    # SESSION / LIFECYCLE
    # ------------------------------------------------------------------

    def _resolve_session_id(self) -> str:
        try:
            if self.session is not None and getattr(self.session, "id", None):
                return str(self.session.id)
        except Exception:
            pass
        return "default"

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
        base = sum(
            m.tokens or self._estimate_message_tokens_full(m)
            for m in self.messages
        )
        if self._running_summary:
            base += self._estimate_tokens(self._running_summary)
        return base

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
        if len(self.messages) <= self.max_messages:
            return

        protected, rest = _split_protected(self.messages)
        protected = _cap_protected(protected, MAX_PINNED_MESSAGES)

        keep = max(0, self.max_messages - len(protected))
        kept = rest[-keep:] if keep else []

        merged = protected + kept
        merged.sort(key=lambda m: getattr(m, "timestamp", 0))

        trimmed = len(self.messages) - len(merged)
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
            "running_summary": self._running_summary,
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

        try:
            await self.load_persisted_summaries(
                limit=self.recover_summaries_on_load
            )
        except Exception as exc:
            logger.debug("load_persisted_summaries failed: %s", exc)

        return len(self.messages)

    async def load_persisted_summaries(self, limit: int = 3) -> int:
        if limit <= 0:
            return 0
        try:
            rows = self.summary_store.load(self._session_id, limit=limit)
        except Exception as exc:
            logger.debug("summary load failed: %s", exc)
            return 0

        if not rows:
            return 0

        # Newest summary becomes the running summary; older ones are
        # discarded because they overlap in time.
        newest = rows[-1]
        self._running_summary = str(newest.get("summary") or "").strip()

        if self._running_summary:
            await self.add_message(
                role="system",
                content=(
                    "[PRIOR CONTEXT — recovered from disk]\n"
                    + self._running_summary
                ),
                pinned=True,
                metadata={"running_summary": True, "recovered": True},
            )

        logger.info(
            "Recovered 1 persisted summary for session %s (kept oldest=%s)",
            self._session_id, bool(rows[:-1]),
        )
        return 1

    # ------------------------------------------------------------------
    # COMPACTION
    # ------------------------------------------------------------------

    async def needs_compaction(
        self,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> bool:
        # Trigger on schedule OR on overflow.
        if len(self.messages) >= COMPACT_AT_MESSAGES:
            return True
        return self.usage_pct(system_prompt, tool_schemas, state_block) >= self.compaction_threshold

    async def compact(
        self,
        aggressive: bool = False,
        system_prompt: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        state_block: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compact the message list.

        Tiered memory:
          - Keep the newest KEEP_VERBATIM messages verbatim.
          - Summarise everything older into a running summary.
          - Replace old tool payloads with one-line outcome stubs before
            they enter the summary text, so the summary is compact.
        """
        from agent.context.runtime import repair_context

        async with self._lock:
            repair_context(self, fill_missing=True)
            before = self.total_tokens()
            if not self.messages:
                return {"compacted": False, "reason": "empty context"}

            protected, rest = _split_protected(self.messages)
            protected = _cap_protected(protected, MAX_PINNED_MESSAGES)

            # Everything older than the verbatim window goes into the summary.
            if len(rest) <= KEEP_VERBATIM:
                return {"compacted": False, "reason": "below verbatim window"}

            to_summarise = rest[:-KEEP_VERBATIM]
            keep_verbatim = rest[-KEEP_VERBATIM:]

            # Build a compact, tool-stubbed digest of the older messages.
            digest_lines: List[str] = []
            for m in to_summarise:
                role = getattr(m, "role", "")
                if role == "tool":
                    digest_lines.append("  -> " + _tool_outcome_stub(m))
                    continue
                if role == "assistant":
                    meta = getattr(m, "metadata", {}) or {}
                    if meta.get("tool_calls"):
                        names = [
                            c.get("function", {}).get("name", "?")
                            for c in meta["tool_calls"]
                        ]
                        digest_lines.append(
                            f"AGENT called: {', '.join(names)}"
                        )
                        continue
                    text = (getattr(m, "content", "") or "")[:400]
                    if text.strip():
                        digest_lines.append(f"AGENT: {text}")
                    continue
                if role == "user":
                    text = (getattr(m, "content", "") or "")[:400]
                    if text.strip():
                        digest_lines.append(f"USER: {text}")
                    continue
                if role == "system":
                    # System guidance is injected per-turn; only the
                    # essential parts matter for the summary.
                    continue

            digest = "\n".join(digest_lines) if digest_lines else "(no substantive history)"

            # Summarise with the model. Include the previous running summary
            # so the new summary is cumulative.
            new_summary = await self._summarise_running(
                previous=self._running_summary,
                new_digest=digest,
            )
            if not new_summary:
                # Fall back to the previous summary plus a raw digest
                # truncated to fit.
                new_summary = (
                    (self._running_summary + "\n\n" if self._running_summary else "")
                    + digest[:1500]
                )

            self._running_summary = new_summary

            # Rebuild the message list: pinned + running summary + verbatim.
            summary_msg = ContextMessage(
                role="system",
                content=f"[Session summary so far]\n{new_summary}",
                tokens=self._estimate_tokens(new_summary),
                metadata={"running_summary": True},
                pinned=True,
            )

            merged: List[ContextMessage] = []
            merged.extend(protected)
            merged.append(summary_msg)
            merged.extend(keep_verbatim)
            merged.sort(key=lambda m: getattr(m, "timestamp", 0))

            self.messages = merged
            repair_context(self, fill_missing=True)
            self.messages = _drop_stale_summaries(self.messages)

            while len(self.tool_outputs) > max(1, self.recent_tool_outputs):
                self.tool_outputs.popleft()

            self._last_compaction = time.time()
            self._compaction_count += 1

            # Persist the newest summary.
            if self.persist_summaries and new_summary:
                try:
                    self.summary_store.append(
                        self._session_id,
                        new_summary,
                        turns_summarized=len(to_summarise),
                        tokens_before=before,
                        tokens_after=self.total_tokens(),
                        metadata={"compaction_count": self._compaction_count},
                    )
                except Exception as exc:
                    logger.debug("summary persist failed: %s", exc)

            return {
                "compacted": True,
                "before_tokens": before,
                "after_tokens": self.total_tokens(),
                "saved_tokens": before - self.total_tokens(),
                "messages_kept": len(self.messages),
                "summaries_created": 1,
            }

    async def _summarise_running(self, previous: str, new_digest: str) -> str:
        if not self.enable_summarization or not getattr(self.compactor, "llm", None):
            return ""
        try:
            from agent.llm.provider import Message
            prompt = (
                "You are maintaining a running summary of a coding session. "
                "Compress the following into a single running narrative.\n\n"
                "Rules:\n"
                "- Record: what the user asked, what was accomplished, what "
                "failed and why, what remains unresolved.\n"
                "- Do NOT quote tool arguments or outputs.\n"
                "- Do NOT repeat the plan.\n"
                "- Do NOT include pleasantries.\n"
                "- Keep it under 300 words. If there is a previous summary, "
                "fold it in — the new summary replaces, not appends.\n\n"
                f"Previous summary:\n{previous or '(none)'}\n\n"
                f"New turns to fold in:\n{new_digest}"
            )
            response = await self.compactor.llm.complete(
                messages=[
                    Message(role="system", content=prompt),
                    Message(role="user", content="Summarise."),
                ],
                temperature=0.2,
                max_tokens=500,
            )
            text = (getattr(response, "content", "") or "").strip()
            if len(text) > RUNNING_SUMMARY_MAX_TOKENS * 6:
                text = text[: RUNNING_SUMMARY_MAX_TOKENS * 6]
            return text
        except Exception as exc:
            logger.debug("running summary failed: %s", exc)
            return ""

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
            "running_summary": self._running_summary,
        }