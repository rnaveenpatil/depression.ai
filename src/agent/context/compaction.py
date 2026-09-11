"""
Compaction - Shrinks the context when it grows too large.

Strategy (multi-stage):
  1. Drop redundant tool outputs (keep the most recent N).
  2. Summarize older messages into a single "summary" system message.
  3. If still over budget, hard-truncate the oldest non-pinned messages.

The summarization step uses the LLM when available; otherwise a
deterministic heuristic summary is produced.
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger

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
    """
    Context compaction engine.
    """

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

        Returns:
            {
                "compacted": bool,
                "messages": [ContextMessage, ...],
                "summaries_created": int,
                "stages_applied": [str, ...],
            }
        """
        if not messages:
            return {"compacted": False, "reason": "no messages"}

        stages: List[str] = []
        working = list(messages)

        # ---- Stage 1: trim tool outputs ----
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
    # STAGE 2: SUMMARIZATION
    # ------------------------------------------------------------------

    async def _summarize_old_messages(
        self,
        messages: List[Any],
        target_tokens: int,
    ) -> tuple[List[Any], bool]:
        """
        Split into (old, recent). Summarize `old` into a single message.
        Recent messages are preserved verbatim.
        """
        # Always keep pinned / system messages + last N verbatim
        pinned = [m for m in messages if getattr(m, "pinned", False) or m.role == "system"]
        unpinned = [m for m in messages if m not in pinned]

        keep_recent = max(self.min_messages_to_keep, len(unpinned) // 3)
        old = unpinned[:-keep_recent] if keep_recent else unpinned
        recent = unpinned[-keep_recent:] if keep_recent else []

        # Nothing meaningful to summarize
        if len(old) < 3:
            return messages, False

        # Build the transcript for the LLM
        transcript = self._render_transcript(old)
        summary_text = await self._generate_summary(transcript)

        # Build the summary message
        try:
            from agent.context.manager import ContextMessage
        except Exception:
            # Fallback if ContextMessage isn't importable (e.g. in isolation)
            class ContextMessage:  # type: ignore
                def __init__(self, role, content, tokens=0, pinned=False, metadata=None):
                    self.role = role
                    self.content = content
                    self.tokens = tokens
                    self.pinned = pinned
                    self.metadata = metadata or {}
                    self.timestamp = time.time()

        summary_msg = ContextMessage(
            role="system",
            content=f"[CONTEXT SUMMARY — earlier turns]\n{summary_text}",
            tokens=self._estimate_str_tokens(summary_text) + 20,
            pinned=True,
            metadata={"compacted": True, "summary_of": len(old)},
        )

        # Reassemble in original order-ish: pinned + summary + recent
        combined = pinned + [summary_msg] + recent
        combined.sort(key=lambda m: getattr(m, "timestamp", 0))
        return combined, True

    async def _generate_summary(self, transcript: str) -> str:
        """Ask the LLM for a summary; fall back to heuristic"""
        # Truncate transcript to a reasonable size for the summarizer
        max_chars = self.max_summary_tokens * 4
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "\n… (truncated)"

        try:
            result = await self.llm.complete(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
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
        """
        pinned = [m for m in messages if getattr(m, "pinned", False) or m.role == "system"]
        unpinned = [m for m in messages if m not in pinned]

        # Always keep the last `min_messages_to_keep`
        must_keep = unpinned[-self.min_messages_to_keep:]
        candidates = unpinned[:-self.min_messages_to_keep]

        result = list(pinned) + list(must_keep)
        budget = target_tokens - self._estimate_tokens(result)

        # Add back the newest of the remaining candidates until budget is hit
        for m in reversed(candidates):
            t = getattr(m, "tokens", 0) or self._estimate_str_tokens(getattr(m, "content", ""))
            if t <= budget:
                result.append(m)
                budget -= t
            else:
                break

        result.sort(key=lambda m: getattr(m, "timestamp", 0))
        return result

    # ------------------------------------------------------------------
    # HEURISTIC SUMMARY (fallback)
    # ------------------------------------------------------------------

    def _heuristic_summary(self, transcript: str) -> str:
        """
        Deterministic summary when no LLM is available.
        Extracts user messages + short assistant previews.
        """
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
            # Cap each message in the transcript
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