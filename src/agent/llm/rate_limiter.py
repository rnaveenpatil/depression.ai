"""
Rate limiter and token counter.

Provides:
    - TokenCounter  — deterministic, dependency-free token estimate
    - RateLimiter   — async token-bucket rate limiter (per-provider)

The token counter is a best-effort estimate. It splits on word boundaries
and punctuation, which tracks most modern subword tokenizers within ~20%.
It is not exact and is not meant to be — it exists so context accounting
reacts to large files and tool outputs before the provider's own limits
are hit.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Dict

from agent.utils.logging import get_logger

logger = get_logger(__name__)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

_MODEL_CORRECTION: Dict[str, float] = {}


class TokenCounter:
    """Deterministic, dependency-free token estimate."""

    @staticmethod
    def count_tokens(text: str, model: str = "default") -> int:
        if not text:
            return 0
        raw = len(_TOKEN_RE.findall(text))
        factor = _MODEL_CORRECTION.get(model, 1.0)
        return max(1, int(raw * factor))

    @staticmethod
    def count_messages(messages: list, model: str = "default") -> int:
        """Approximate the token cost of a chat request."""
        total = 0
        for m in messages or []:
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            total += TokenCounter.count_tokens(content or "", model)
            total += 4  # role + separator overhead per message

            calls = getattr(m, "tool_calls", None)
            if calls is None and isinstance(m, dict):
                calls = m.get("tool_calls")
            if calls:
                try:
                    total += TokenCounter.count_tokens(
                        json.dumps(calls, default=str), model
                    )
                except Exception:
                    pass
        return total

    @staticmethod
    def register_correction(model: str, factor: float) -> None:
        """Adjust the estimate for a specific model ID."""
        if factor > 0:
            _MODEL_CORRECTION[model] = float(factor)


class RateLimiter:
    """
    Simple token-bucket rate limiter. One bucket per key (typically a
    provider name). Acquire before each request; refills over time.
    """

    def __init__(self, rate_per_second: float = 4.0, burst: int = 8):
        self.rate = float(rate_per_second)
        self.burst = int(burst)
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str = "default", tokens: int = 1) -> None:
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.setdefault(
                key, {"tokens": float(self.burst), "ts": now}
            )
            elapsed = now - bucket["ts"]
            bucket["tokens"] = min(
                float(self.burst), bucket["tokens"] + elapsed * self.rate
            )
            bucket["ts"] = now

            if bucket["tokens"] < tokens:
                wait = (tokens - bucket["tokens"]) / self.rate
            else:
                wait = 0.0
                bucket["tokens"] -= tokens

        if wait > 0:
            await asyncio.sleep(wait)


__all__ = ["TokenCounter", "RateLimiter"]