"""
Compaction - Shrinks the context when it grows too large.

Strategy (multi-stage):
  1. Drop redundant tool outputs (keep the most recent N).
  2. Summarize older VALID message groups into a single summary message.
  3. If still over budget, hard-truncate the oldest non-pinned groups.

Whole groups only. Never splits an assistant / tool_call / tool_result
group. Groups that fail validation are collapsed to a single system marker
so they cannot corrupt the request.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.utils.logging import get_logger
from agent.llm.provider import Message

# Shared grouping / validation lives in runtime.py so runtime and
# compaction can never disagree about what a "valid group" is.
from agent.context.runtime import group_messages, validate_group

logger = get_logger(__name__)


SUMMARY_SYSTEM_PROMPT = """You are a context compaction engine.
Summarize the following conversation history into a dense, factual summary.
Preserve:
- User goals and constraints
- Decisions made and their rationale
- Key facts discovered (file paths, commands, errors, outputs)
- Current state and next steps
Omit:
- Verbose repetition
- Small talk
- Intermediate reasoning that led nowhere
Output only the summary, no preamble."""


# ----------------------------------------------------------------------
# TOKEN ESTIMATION
# ----------------------------------------------------------------------

def _tool_calls_of(message: Any) -> List[dict]:
    return list((getattr(message, "metadata", {}) or {}).get("tool_calls") or [])


def _estimate_message_tokens(message: Any) -> int:
    """
    Estimate a message's cost.

    Rules:
      - If `tokens` is present and covers tool_calls, use it.
      - Content is prose-ish: ~len/4.
      - Tool_calls JSON is dense: ~len/3.
      - Never return less than 1.
    """
    cached = int(getattr(message, "tokens", 0) or 0)
    calls = _tool_calls_of(message)
    calls_tokens = 0
    if calls:
        try:
            calls_json = json.dumps(calls, default=str)
        except Exception:
            calls_json = ""
        if calls_json:
            calls_tokens = max(1, len(calls_json) // 3)

    if cached:
        # Trust the cached count only if it's at least the tool_calls cost.
        if calls_tokens and cached < calls_tokens:
            return cached + calls_tokens
        return cached

    total = 0
    content = getattr(message, "content", "") or ""
    if content:
        total += max(1, len(content) // 4)
    total += calls_tokens
    return max(1, total)


def _estimate_group_tokens(group: List[Any]) -> int:
    return sum(_estimate_message_tokens(m) for m in group)


def _is_pinned_or_system(message: Any) -> bool:
    return bool(getattr(message, "pinned", False)) or getattr(message, "role", "") == "system"


def _sanitize_groups(groups: List[List[Any]]) -> List[List[Any]]:
    """
    Replace any group that fails validation with a single, valid marker
    message so the conversation sequence can never be broken.
    """
    out: List[List[Any]] = []
    for g in groups:
        if validate_group(g):
            out.append(g)
            continue

        # Try to repair: keep the first message only if it's not an
        # assistant with tool_calls.
        first = g[0] if g else None
        role = getattr(first, "role", "") if first is not None else ""
        content = getattr(first, "content", "") if first is not None else ""
        calls = _tool_calls_of(first) if first is not None else []

        try:
            from agent.context.manager import ContextMessage
        except Exception:
            ContextMessage = None  # type: ignore

        marker_text = (
            f"[{len(g)} invalid message(s) removed during compaction]"
            if len(g) > 1 else
            "[invalid assistant/tool sequence removed during compaction]"
        )

        if role == "assistant" and calls and ContextMessage is not None:
            # Drop tool_calls, keep any prose content if present.
            try:
                m = ContextMessage(
                    role="assistant",
                    content=content or marker_text,
                    metadata={},
                    tokens=max(1, len(content or marker_text) // 4),
                    timestamp=getattr(first, "timestamp", time.time()),
                )
                out.append([m])
                continue
            except Exception:
                pass

        if ContextMessage is not None:
            try:
                out.append([ContextMessage(
                    role="system",
                    content=marker_text,
                    metadata={"compacted": True, "invalid_group": True},
                    tokens=max(1, len(marker_text) // 4),
                    timestamp=time.time(),
                )])
                continue
            except Exception:
                pass

        # Absolute fallback
        class _Marker:
            __slots__ = ("role", "content", "tokens", "pinned", "metadata", "timestamp")
            def __init__(self) -> None:
                self.role = "system"
                self.content = marker_text
                self.tokens = max(1, len(marker_text) // 4)
                self.pinned = False
                self.metadata = {"compacted": True, "invalid_group": True}
                self.timestamp = time.time()
        out.append([_Marker()])
    return out


# ----------------------------------------------------------------------
# COMPACTOR
# ----------------------------------------------------------------------

class Compactor:
    """Context compaction engine."""

    def __init__(self, llm: Any = None, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.llm = llm
        self.target_ratio: float = cfg.get("target_ratio", 0.5)
        self.enable_summarization: bool = cfg.get("enable_summarization", True)
        self.min_messages_to_keep: int = cfg.get("min_messages_to_keep", 4)
        self.max_summary_tokens: int = cfg.get("max_summary_tokens", 800)
        self.max_tool_outputs_keep: int = cfg.get("max_tool_outputs_keep", 3)

        logger.info(
            "Compactor initialized (target_ratio=%s, summarization=%s)",
            self.target_ratio, self.enable_summarization,
        )

    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: List[Any],
        tool_outputs: List[Any],
        target_tokens: int,
        current_tokens: int,
        extra_tokens: int = 0,
    ) -> Dict[str, Any]:
        """
        extra_tokens = cost of everything the caller will also send:
        system prompt + tool schemas + runtime state block. Subtracted
        from the target so compaction frees enough room for those too.
        """
        if not messages:
            return {"compacted": False, "reason": "no messages"}

        stages: List[str] = []
        working = list(messages)

        if len(tool_outputs) > self.max_tool_outputs_keep:
            stages.append("tool_output_trim")

        # Sanitize before doing anything else so broken sequences can't
        # survive compaction.
        raw_groups = group_messages(working)
        sanitized_groups = _sanitize_groups(raw_groups)
        working = [m for g in sanitized_groups for m in g]

        effective_target = max(1, target_tokens - max(0, extra_tokens))

        if self.enable_summarization and self.llm is not None:
            try:
                working, created = await self._summarize_old_messages(
                    working, effective_target
                )
                if created:
                    stages.append("summarize")
            except Exception as e:
                logger.warning("Summarization stage failed: %s", e)

        est = self._estimate_tokens(working)
        if est > effective_target:
            working = self._hard_truncate(working, effective_target)
            stages.append("hard_truncate")

        if not stages:
            return {"compacted": False, "reason": "already within budget"}

        return {
            "compacted": True,
            "messages": working,
            "summaries_created": 1 if "summarize" in stages else 0,
            "stages_applied": stages,
        }

    # ------------------------------------------------------------------

    async def _summarize_old_messages(
        self,
        messages: List[Any],
        target_tokens: int,
    ) -> Tuple[List[Any], bool]:
        groups = group_messages(messages)

        pinned_groups: List[List[Any]] = []
        unpinned_groups: List[List[Any]] = []
        for g in groups:
            if all(_is_pinned_or_system(m) for m in g):
                pinned_groups.append(g)
            else:
                unpinned_groups.append(g)

        keep_recent = max(self.min_messages_to_keep, len(unpinned_groups) // 3)
        if keep_recent <= 0 or len(unpinned_groups) <= keep_recent:
            return messages, False

        old_groups = unpinned_groups[:-keep_recent]
        recent_groups = unpinned_groups[-keep_recent:]

        old_flat = [m for g in old_groups for m in g]
        if len(old_flat) < 3:
            return messages, False

        transcript = self._render_transcript(old_flat)
        summary_text = await self._generate_summary(transcript)

        try:
            from agent.context.manager import ContextMessage
        except Exception:
            ContextMessage = None  # type: ignore

        summary_content = f"[CONTEXT SUMMARY — earlier turns]\n{summary_text}"
        summary_tokens = max(1, len(summary_text) // 4) + 20

        summary_msg: Any
        if ContextMessage is not None:
            try:
                summary_msg = ContextMessage(
                    role="system",
                    content=summary_content,
                    tokens=summary_tokens,
                    pinned=True,
                    metadata={"compacted": True, "summary_of": len(old_flat)},
                )
            except Exception as e:
                logger.warning(
                    "ContextMessage construction failed (%s); using fallback", e
                )
                ContextMessage = None  # type: ignore
                summary_msg = None  # type: ignore
        else:
            summary_msg = None  # type: ignore

        if summary_msg is None:
            class _SummaryMessage:
                __slots__ = ("role", "content", "tokens", "pinned", "metadata", "timestamp")
                def __init__(self) -> None:
                    self.role = "system"
                    self.content = summary_content
                    self.tokens = summary_tokens
                    self.pinned = True
                    self.metadata: Dict[str, Any] = {
                        "compacted": True,
                        "summary_of": len(old_flat),
                    }
                    self.timestamp = time.time()
            summary_msg = _SummaryMessage()

        combined: List[Any] = []
        for g in pinned_groups:
            combined.extend(g)
        combined.append(summary_msg)
        for g in recent_groups:
            combined.extend(g)
        return combined, True

    async def _generate_summary(self, transcript: str) -> str:
        max_chars = self.max_summary_tokens * 4
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "\n… (truncated)"

        try:
            result = await self.llm.complete(
                messages=[
                    Message(role="system", content=SUMMARY_SYSTEM_PROMPT),
                    Message(role="user", content=transcript),
                ],
                temperature=0.1,
                max_tokens=self.max_summary_tokens,
            )
            text = getattr(result, "content", None) or str(result)
            text = text.strip()
            # Cap the fallback-safe summary too.
            if len(text) > self.max_summary_tokens * 4:
                text = text[: self.max_summary_tokens * 4] + "…"
            return text or self._heuristic_summary(transcript)
        except Exception as e:
            logger.warning("LLM summarization failed: %s", e)
            return self._heuristic_summary(transcript)

    # ------------------------------------------------------------------

    def _hard_truncate(self, messages: List[Any], target_tokens: int) -> List[Any]:
        groups = group_messages(messages)

        pinned_groups: List[List[Any]] = []
        unpinned_groups: List[List[Any]] = []
        for g in groups:
            if all(_is_pinned_or_system(m) for m in g):
                pinned_groups.append(g)
            else:
                unpinned_groups.append(g)

        must_keep_count = min(self.min_messages_to_keep, len(unpinned_groups))
        must_keep_groups = unpinned_groups[-must_keep_count:] if must_keep_count else []
        candidate_groups = (
            unpinned_groups[:-must_keep_count] if must_keep_count else list(unpinned_groups)
        )

        base: List[Any] = [m for g in pinned_groups for m in g] + [
            m for g in must_keep_groups for m in g
        ]
        budget = target_tokens - self._estimate_tokens(base)

        added_groups: List[List[Any]] = []
        for g in reversed(candidate_groups):
            cost = _estimate_group_tokens(g)
            if cost <= budget:
                added_groups.append(g)
                budget -= cost
            else:
                break

        added_groups.reverse()
        return (
            [m for g in pinned_groups for m in g]
            + [m for g in added_groups for m in g]
            + [m for g in must_keep_groups for m in g]
        )

    # ------------------------------------------------------------------

    def _heuristic_summary(self, transcript: str) -> str:
        lines = transcript.splitlines()
        user_lines = [l for l in lines if l.startswith("USER:")]
        assistant_previews = [l[:120] for l in lines if l.startswith("ASSISTANT:")][-5:]

        parts = ["Prior turns summarized (heuristic):"]
        if user_lines:
            parts.append("User asked about:")
            for l in user_lines[-8:]:
                parts.append(f"  - {l[5:].strip()[:160]}")
        if assistant_previews:
            parts.append("Assistant responses (preview):")
            for l in assistant_previews:
                parts.append(f"  - {l[10:].strip()}")
        text = "\n".join(parts)
        # Cap to the summary budget.
        cap = self.max_summary_tokens * 4
        if len(text) > cap:
            text = text[:cap] + "…"
        return text

    # ------------------------------------------------------------------

    def _render_transcript(self, messages: List[Any]) -> str:
        parts = []
        for m in messages:
            role = getattr(m, "role", "unknown").upper()
            content = getattr(m, "content", "") or ""
            if len(content) > 2000:
                content = content[:2000] + "…"
            calls = _tool_calls_of(m)
            if calls:
                try:
                    names = ", ".join(
                        c.get("function", {}).get("name", "?") for c in calls
                    )
                except Exception:
                    names = "?"
                parts.append(f"{role} [tool_calls: {names}]: {content}")
            else:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _estimate_tokens(self, messages: List[Any]) -> int:
        return sum(_estimate_message_tokens(m) for m in messages)