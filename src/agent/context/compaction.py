"""
Compaction - Shrinks the context when it grows too large.

Strategy (multi-stage):
  1. Drop redundant tool outputs (keep the most recent N).
  2. Summarize older messages into a single "summary" system message.
  3. If still over budget, hard-truncate the oldest non-pinned messages.

The summarization step uses the LLM when available; otherwise a
deterministic heuristic summary is produced.

Ordering is preserved throughout so assistant tool_calls always remain
adjacent to their tool results.
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.llm.provider import Message

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
            f"Compactor initialized (target_ratio={self.target_ratio}, "
            f"summarization={self.enable_summarization})"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def compact(
        self,
        messages: List[Any],
        tool_outputs: List[Any],
        target_tokens: int,
        current_tokens: int,
    ) -> Dict[str, Any]:
        """
        Compact messages + tool outputs down toward `target_tokens`.
        """
        if not messages:
            return {"compacted": False, "reason": "no messages"}

        stages: List[str] = []
        working = list(messages)

        # ---- Stage 1: trim tool outputs (recorded for callers) ----
        if len(tool_outputs) > self.max_tool_outputs_keep:
            stages.append("tool_output_trim")

        # ---- Stage 2: summarize old messages ----
        if self.enable_summarization and self.llm is not None:
            try:
                working, created = await self._summarize_old_messages(
                    working, target_tokens
                )
                if created:
                    stages.append("summarize")
            except Exception as e:
                logger.warning(f"Summarization stage failed: {e}")

        # ---- Stage 3: hard truncate if still over budget ----
        est = self._estimate_tokens(working)
        if est > target_tokens:
            working = self._hard_truncate(working, target_tokens)
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
    # PINNED / UNPINNED SPLIT
    # ------------------------------------------------------------------

    @staticmethod
    def _split_pinned(messages: List[Any]) -> tuple[List[Any], List[Any]]:
        """
        Return (pinned, unpinned). Pinned = `pinned=True` or system role.
        Single pass — no `in` list containment checks.
        """
        pinned: List[Any] = []
        unpinned: List[Any] = []
        for m in messages:
            if getattr(m, "pinned", False) or getattr(m, "role", "") == "system":
                pinned.append(m)
            else:
                unpinned.append(m)
        return pinned, unpinned

    # ------------------------------------------------------------------
    # STAGE 2: SUMMARIZATION
    # ------------------------------------------------------------------

    async def _summarize_old_messages(
        self,
        messages: List[Any],
        target_tokens: int,
    ) -> tuple[List[Any], bool]:
        """
        Split into (old, recent). Summarize `old` into a single message.
        Recent messages are preserved verbatim. Ordering preserved; the
        summary is inserted between pinned and recent.
        """
        pinned, unpinned = self._split_pinned(messages)

        keep_recent = max(self.min_messages_to_keep, len(unpinned) // 3)
        if keep_recent <= 0 or len(unpinned) <= keep_recent:
            return messages, False

        old = unpinned[:-keep_recent]
        recent = unpinned[-keep_recent:]

        if len(old) < 3:
            return messages, False

        transcript = self._render_transcript(old)
        summary_text = await self._generate_summary(transcript)

        try:
            from agent.context.manager import ContextMessage
        except Exception:
            ContextMessage = None  # type: ignore

        summary_content = f"[CONTEXT SUMMARY — earlier turns]\n{summary_text}"
        summary_tokens = self._estimate_str_tokens(summary_text) + 20

        if ContextMessage is not None:
            summary_msg = ContextMessage(
                role="system",
                content=summary_content,
                tokens=summary_tokens,
                pinned=True,
                metadata={"compacted": True, "summary_of": len(old)},
            )
        else:
            # Defensive fallback if manager import fails.
            class _Fallback:
                def __init__(self):
                    self.role = "system"
                    self.content = summary_content
                    self.tokens = summary_tokens
                    self.pinned = True
                    self.metadata = {"compacted": True, "summary_of": len(old)}
                    self.timestamp = time.time()
            summary_msg = _Fallback()

        # Preserve order: pinned + summary + recent. No re-sort.
        combined = list(pinned) + [summary_msg] + list(recent)
        return combined, True

    async def _generate_summary(self, transcript: str) -> str:
        """Ask the LLM for a summary; fall back to heuristic"""
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
            return text.strip() or self._heuristic_summary(transcript)
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._heuristic_summary(transcript)

    # ------------------------------------------------------------------
    # STAGE 3: HARD TRUNCATION
    # ------------------------------------------------------------------

    def _hard_truncate(self, messages: List[Any], target_tokens: int) -> List[Any]:
        """
        Drop oldest non-pinned messages until we're under budget.
        Always preserves pinned / system messages and the last few turns.
        No re-sort — messages keep their original order.
        """
        pinned, unpinned = self._split_pinned(messages)

        must_keep_count = min(self.min_messages_to_keep, len(unpinned))
        must_keep = unpinned[-must_keep_count:] if must_keep_count else []
        candidates = unpinned[:-must_keep_count] if must_keep_count else list(unpinned)

        result = list(pinned) + list(must_keep)
        budget = target_tokens - self._estimate_tokens(result)

        added: List[Any] = []
        for m in reversed(candidates):
            t = getattr(m, "tokens", 0) or self._estimate_str_tokens(
                getattr(m, "content", "")
            )
            if t <= budget:
                added.append(m)
                budget -= t
            else:
                break

        # Prepend the kept tail in original order.
        added.reverse()
        return list(pinned) + added + list(must_keep)

    # ------------------------------------------------------------------
    # HEURISTIC SUMMARY (fallback)
    # ------------------------------------------------------------------

    def _heuristic_summary(self, transcript: str) -> str:
        lines = transcript.splitlines()
        user_lines = [l for l in lines if l.startswith("USER:")]
        assistant_previews = [
            l[:120] for l in lines if l.startswith("ASSISTANT:")
        ][-5:]

        parts = ["Prior turns summarized (heuristic):"]
        if user_lines:
            parts.append("User asked about:")
            for l in user_lines[-8:]:
                parts.append(f"  - {l[5:].strip()[:160]}")
        if assistant_previews:
            parts.append("Assistant responses (preview):")
            for l in assistant_previews:
                parts.append(f"  - {l[10:].strip()}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _render_transcript(self, messages: List[Any]) -> str:
        parts = []
        for m in messages:
            role = getattr(m, "role", "unknown").upper()
            content = getattr(m, "content", "") or ""
            if len(content) > 2000:
                content = content[:2000] + "…"
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _estimate_tokens(self, messages: List[Any]) -> int:
        total = 0
        for m in messages:
            t = getattr(m, "tokens", 0)
            if not t:
                t = self._estimate_str_tokens(getattr(m, "content", ""))
            total += t
        return total

    def _estimate_str_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)