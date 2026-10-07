"""Provider-response normalizer — the single boundary every LLM response crosses.

Every adapter (OpenAI-compatible, Anthropic, Ollama, Gemini, and any future
one) already converts its wire format into the framework's `LLMResponse`
type. This module is the *second* boundary: it takes that response and
turns it into one canonical shape the agent loop can consume without
ever thinking about which provider produced it.

    OpenAICompatible  →  tool_calls[i].function.{name,arguments}
    Anthropic         →  content[*].{type=="tool_use", name, input}
    Ollama            →  message.tool_calls[i].function.{name,arguments}
    Gemini            →  parts[*].{functionCall:{name,args}}
                                  │
                                  ▼
                        LLMResponse (framework type)
                                  │
                                  ▼
                            normalizer
                                  │
                                  ▼
                NormalizedResponse  ← the agent loop's ONLY input type

The normalizer is where these questions get answered once, so no caller
has to ask them again:

    * Which tool calls are valid (have a name), which have bad arguments
      (parse_error), and which have no id (synthesize one)?
    * Is the response text empty because the model chose to call a tool,
      or because it hit a content filter?
    * What does `finish_reason` mean in this framework's vocabulary?
    * How many tokens did this cost, split by kind, and in what units?
    * Did the provider report an error mid-response, and what code should
      the loop use to decide whether to retry?
    * If the provider reported reasoning/thinking tokens or text, where do
      they go?

Every caller downstream — the loop, the tool dispatcher, the cost panel,
the context window tracker — reads a `NormalizedResponse`. Never the raw
provider payload.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from agent.llm.provider import (
    FinishReason,
    LLMProvider,
    LLMResponse,
    Message,
    ProviderError,
    ProviderErrorCode,
    ToolCall,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# CANONICAL TYPES
# ======================================================================

@dataclass
class NormalizedToolCall:
    """
    One tool call, in the framework's canonical form.

    Every provider's tool call ends up here. Downstream code reads
    `name` and `arguments`; it never cares which provider produced it.

    Fields
    ------
    id              — always present. Synthesized when the provider
                      didn't supply one (Ollama, Gemini).
    name            — the function name. Never empty; calls without a
                      name are dropped during normalization.
    arguments       — always a dict. Malformed JSON becomes {} plus a
                      `parse_error` entry, and the loop re-prompts.
    parse_error     — set when arguments couldn't be decoded or weren't
                      an object. None on a clean call.
    raw             — the provider's original tool-call entry, kept for
                      debugging and for the transcript's expand view.
    """
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    parse_error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "name": self.name,
                             "arguments": dict(self.arguments)}
        if self.parse_error:
            d["parse_error"] = self.parse_error
        return d

    def to_tool_call(self) -> ToolCall:
        """Project back to the framework's base ToolCall type."""
        return ToolCall(id=self.id, name=self.name,
                        arguments=dict(self.arguments))


@dataclass
class NormalizedUsage:
    """
    Token accounting, in one shape regardless of provider.

    Fields are always ints; missing values are 0. `provider_raw` keeps
    the original dict for the cost panel to render extras like Ollama's
    duration counters or Anthropic's cache-creation split.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    # Only Anthropic reports this today.
    cache_creation_tokens: int = 0
    provider_raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def billable_input(self) -> int:
        """Input tokens excluding cache reads (already billed once)."""
        return max(0, self.prompt_tokens - self.cached_tokens)

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


@dataclass
class NormalizedError:
    """A provider-side error, already classified."""
    message: str
    code: str = ProviderErrorCode.UNKNOWN.value
    retryable: bool = False
    provider: str = ""
    model: str = ""
    status: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "code": self.code,
            "retryable": self.retryable,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
        }


@dataclass
class NormalizedResponse:
    """
    The one shape the agent loop consumes.

    Invariants the loop can rely on:
        * `text` is always a str (possibly "").
        * `tool_calls` is always a list; each entry has a non-empty
          `name` and a `dict` for `arguments`.
        * `finish_reason` is one of the FinishReason values.
        * `usage.total_tokens` is either the provider's number or the
          sum of prompt+completion; never 0 unless the provider truly
          reported nothing.
        * `error` is None on success, a NormalizedError on failure.
          A response may carry BOTH partial text and an error — the
          loop decides whether partial output is usable.
    """
    # Content
    text: str = ""
    reasoning_text: str = ""

    # Tools
    tool_calls: List[NormalizedToolCall] = field(default_factory=list)

    # Metadata
    provider: str = ""
    model: str = ""
    finish_reason: str = FinishReason.STOP.value
    usage: NormalizedUsage = field(default_factory=NormalizedUsage)
    error: Optional[NormalizedError] = None

    # Streaming / timing
    streamed: bool = False
    latency_ms: int = 0

    # Provider's original payload, for debugging.
    raw: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def has_text(self) -> bool:
        return bool(self.text)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def is_empty(self) -> bool:
        """No text, no tool calls, no error — provider returned nothing."""
        return not self.text and not self.tool_calls and self.error is None

    def to_message(self) -> Message:
        """
        Build the assistant message for the *next* turn.

        Tool calls are the framework's `ToolCall` type so providers that
        consume the message on the way out see a familiar shape.
        """
        return Message.assistant(
            content=self.text or None,
            tool_calls=[tc.to_tool_call() for tc in self.tool_calls] or None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "reasoning_text": self.reasoning_text,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "error": self.error.to_dict() if self.error else None,
            "streamed": self.streamed,
            "latency_ms": self.latency_ms,
        }


# ======================================================================
# NORMALIZER
# ======================================================================

class ResponseNormalizer:
    """
    Stateless converter from `LLMResponse` (framework type) to
    `NormalizedResponse` (canonical type).

    Every adapter already produces a well-formed `LLMResponse`, so this
    class is defensive rather than reparsing — it fixes up the handful
    of edge cases the adapters can't know about on their own (provider
    quirks in `finish_reason`, missing `id`s, usage totals that don't
    add up) and produces a shape the loop can trust.

    The class is stateless: keep one instance per process, call
    `normalize()` as many times as you like from any coroutine.
    """

    #: Provider names that sometimes send `finish_reason="tool_use"` even
    #: though the framework's vocabulary uses "tool_calls".
    _FINISH_ALIASES = {
        # OpenAI-compatible variants
        "tool_calls": FinishReason.TOOL_CALLS.value,
        "tool_use": FinishReason.TOOL_CALLS.value,
        "function_call": FinishReason.TOOL_CALLS.value,
        # Anthropic extras
        "end_turn": FinishReason.STOP.value,
        "stop_sequence": FinishReason.STOP.value,
        "pause_turn": FinishReason.STOP.value,
        # Ollama extras
        "max_tokens": FinishReason.LENGTH.value,
        "length": FinishReason.LENGTH.value,
        # Content-filter aliases
        "content_filter": FinishReason.CONTENT_FILTER.value,
        "safety": FinishReason.CONTENT_FILTER.value,
        "recitation": FinishReason.CONTENT_FILTER.value,
        "blocklist": FinishReason.CONTENT_FILTER.value,
        "refusal": FinishReason.CONTENT_FILTER.value,
        # Gemini's error-ish reason
        "malformed_function_call": FinishReason.ERROR.value,
    }

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINTS
    # ------------------------------------------------------------------

    def normalize(
        self,
        response: Union[LLMResponse, Dict[str, Any], Any],
        *,
        provider: str = "",
        model: str = "",
        started_at: Optional[float] = None,
    ) -> NormalizedResponse:
        """
        Convert `response` into a NormalizedResponse.

        Accepts:
            * A framework `LLMResponse` (the normal path).
            * A dict shaped like an LLMResponse (for tests).
            * Any object with the same attributes (duck-typed).

        `provider` and `model` override the response's own values when
        the caller knows better (e.g. for a fallback hop that changed
        the model).

        Never raises. A malformed input yields a NormalizedResponse with
        an error rather than a traceback, so the loop can decide what to
        do without a try/except around this call.
        """
        try:
            return self._normalize_inner(response, provider, model, started_at)
        except Exception as exc:
            logger.exception("Normalizer crashed on %r", type(response))
            return NormalizedResponse(
                provider=provider or "",
                model=model or "",
                finish_reason=FinishReason.ERROR.value,
                error=NormalizedError(
                    message=f"normalizer failure: {exc}",
                    code=ProviderErrorCode.UNKNOWN.value,
                    retryable=False,
                    provider=provider or "",
                    model=model or "",
                ),
                latency_ms=self._elapsed_ms(started_at),
            )

    def normalize_error(
        self,
        error: Union[BaseException, Dict[str, Any]],
        *,
        provider: str = "",
        model: str = "",
    ) -> NormalizedResponse:
        """
        Wrap a provider error into a NormalizedResponse.

        Use this when the adapter itself raised (a ProviderError from
        `chat()`), rather than returning a response. The loop then has
        one code path for "got a response" and "got an error".
        """
        if isinstance(error, ProviderError):
            return NormalizedResponse(
                provider=provider or error.provider or "",
                model=model or error.model or "",
                finish_reason=FinishReason.ERROR.value,
                error=NormalizedError(
                    message=str(error),
                    code=error.code,
                    retryable=error.retryable,
                    provider=error.provider or provider or "",
                    model=error.model or model or "",
                    status=error.status,
                ),
            )
        if isinstance(error, dict):
            return NormalizedResponse(
                provider=provider,
                model=model,
                finish_reason=FinishReason.ERROR.value,
                error=NormalizedError(
                    message=str(error.get("message") or error.get("error") or error),
                    code=str(error.get("code") or ProviderErrorCode.UNKNOWN.value),
                    retryable=bool(error.get("retryable")),
                    provider=str(error.get("provider") or provider),
                    model=str(error.get("model") or model),
                    status=error.get("status"),
                ),
            )
        return NormalizedResponse(
            provider=provider,
            model=model,
            finish_reason=FinishReason.ERROR.value,
            error=NormalizedError(
                message=str(error),
                code=ProviderErrorCode.UNKNOWN.value,
                retryable=False,
                provider=provider,
                model=model,
            ),
        )

    # ------------------------------------------------------------------
    # INNER PIPELINE
    # ------------------------------------------------------------------

    def _normalize_inner(
        self,
        response: Any,
        provider_override: str,
        model_override: str,
        started_at: Optional[float],
    ) -> NormalizedResponse:
        # Accept dicts and duck-typed objects uniformly.
        resp = _as_response_dict(response)

        provider = provider_override or str(resp.get("provider") or "")
        model = model_override or str(resp.get("model") or "")

        text = self._normalize_text(resp.get("content"))
        reasoning = self._normalize_text(resp.get("reasoning"))
        if not reasoning:
            reasoning = self._normalize_text(
                resp.get("reasoning_content") or resp.get("thinking")
            )

        tool_calls = self._normalize_tool_calls(resp.get("tool_calls") or [])

        finish = self._normalize_finish_reason(
            resp.get("finish_reason"),
            has_tools=bool(tool_calls),
            has_text=bool(text),
        )

        usage = self._normalize_usage(resp.get("usage") or {})

        streamed = bool(resp.get("streamed")) or bool(
            isinstance(resp.get("raw"), dict) and resp["raw"].get("streamed")
        )

        return NormalizedResponse(
            text=text,
            reasoning_text=reasoning,
            tool_calls=tool_calls,
            provider=provider,
            model=model,
            finish_reason=finish,
            usage=usage,
            error=None,
            streamed=streamed,
            latency_ms=self._elapsed_ms(started_at),
            raw=resp.get("raw") if isinstance(resp.get("raw"), dict) else None,
        )

    # ------------------------------------------------------------------
    # TEXT
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(content: Any) -> str:
        """
        Text is always a str.

        Content might arrive as a list of blocks (Anthropic if a caller
        didn't convert them, or a future provider that streams blocks).
        We extract text blocks, ignore non-text ones, and join.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (int, float, bool)):
            return str(content)
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    t = block.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
        if isinstance(content, dict):
            t = content.get("text")
            if isinstance(t, str):
                return t
            try:
                return json.dumps(content)
            except Exception:
                return str(content)
        return str(content)

    # ------------------------------------------------------------------
    # TOOL CALLS
    # ------------------------------------------------------------------

    def _normalize_tool_calls(
        self, raw_calls: Sequence[Any]
    ) -> List[NormalizedToolCall]:
        """
        Convert every tool-call shape into NormalizedToolCall.

        Rules (applied uniformly, whatever the provider):
            * Calls with no name are dropped — they can't be dispatched.
            * Missing ids are synthesized so pairing always works.
            * Arguments are coerced to a dict. A JSON string is parsed;
              a malformed one sets `parse_error` and yields {}.
            * The original entry is preserved under `raw`.
        """
        out: List[NormalizedToolCall] = []

        for idx, entry in enumerate(raw_calls or []):
            if entry is None:
                continue

            # Already normalized (idempotent).
            if isinstance(entry, NormalizedToolCall):
                out.append(entry)
                continue

            # Framework ToolCall.
            if isinstance(entry, ToolCall):
                args = dict(entry.arguments or {})
                if not entry.name:
                    logger.debug("Dropping ToolCall with no name (id=%r)", entry.id)
                    continue
                out.append(NormalizedToolCall(
                    id=entry.id or _synth_id(),
                    name=entry.name,
                    arguments=args,
                    parse_error=getattr(entry, "parse_error", None),
                    raw={"id": entry.id, "name": entry.name, "arguments": args},
                ))
                continue

            # Dict (any provider's shape).
            if isinstance(entry, dict):
                call = self._normalize_dict_tool_call(entry, idx)
                if call is not None:
                    out.append(call)
                continue

            # Duck-typed object with .name / .arguments.
            name = getattr(entry, "name", None)
            if name:
                args = getattr(entry, "arguments", {}) or {}
                args, parse_error = _coerce_arguments(args)
                out.append(NormalizedToolCall(
                    id=str(getattr(entry, "id", "") or _synth_id()),
                    name=str(name),
                    arguments=args,
                    parse_error=parse_error,
                    raw={"provider_obj": repr(entry)[:200]},
                ))
                continue

            logger.debug("Skipping unrecognized tool call: %r", entry)

        return out

    def _normalize_dict_tool_call(
        self, entry: Dict[str, Any], index: int
    ) -> Optional[NormalizedToolCall]:
        """Handle the union of {OpenAI, Anthropic, Ollama, Gemini} shapes."""

        # Gemini's functionCall part.
        fc_gem = entry.get("functionCall") or entry.get("function_call")
        if isinstance(fc_gem, dict) and fc_gem.get("name"):
            name = str(fc_gem["name"])
            args, parse_error = _coerce_arguments(fc_gem.get("args"))
            return NormalizedToolCall(
                id=str(entry.get("id") or _synth_id()),
                name=name,
                arguments=args,
                parse_error=parse_error,
                raw=entry,
            )

        # Anthropic's tool_use block (already flattened by the adapter,
        # but a raw block can still show up if a caller bypassed it).
        if entry.get("type") == "tool_use" and entry.get("name"):
            name = str(entry["name"])
            args, parse_error = _coerce_arguments(entry.get("input"))
            return NormalizedToolCall(
                id=str(entry.get("id") or _synth_id()),
                name=name,
                arguments=args,
                parse_error=parse_error,
                raw=entry,
            )

        # OpenAI / Ollama / Mistral: {"id","function":{"name","arguments"}}.
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name") or entry.get("name") or entry.get("tool")
        if not name:
            logger.debug("Dropping tool call with no name (entry=%r)", entry)
            return None

        raw_args = fn.get("arguments", entry.get("arguments",
                                                 entry.get("parameters", entry.get("input"))))
        args, parse_error = _coerce_arguments(raw_args)

        return NormalizedToolCall(
            id=str(entry.get("id") or fn.get("id") or _synth_id()),
            name=str(name),
            arguments=args,
            parse_error=parse_error,
            raw=entry,
        )

    # ------------------------------------------------------------------
    # FINISH REASON
    # ------------------------------------------------------------------

    def _normalize_finish_reason(
        self,
        raw: Any,
        *,
        has_tools: bool,
        has_text: bool,
    ) -> str:
        """
        Map anything a provider sends into FinishReason.

        Fallbacks, in order:
            1. Known alias (case-insensitive).
            2. If tool calls are present, treat as TOOL_CALLS.
            3. Otherwise STOP.
        """
        if raw is None or raw == "":
            return FinishReason.TOOL_CALLS.value if has_tools else FinishReason.STOP.value
        key = str(raw).strip().lower()
        # Some providers send "STOP" or "MAX_TOKENS" uppercase.
        if key in self._FINISH_ALIASES:
            return self._FINISH_ALIASES[key]
        # Unknown value — fall through to heuristics.
        if has_tools:
            return FinishReason.TOOL_CALLS.value
        if has_text:
            return FinishReason.STOP.value
        return FinishReason.STOP.value

    # ------------------------------------------------------------------
    # USAGE
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_usage(raw: Dict[str, Any]) -> NormalizedUsage:
        """
        Coerce every provider's usage dict into NormalizedUsage.

        Handles the shapes we've seen:
            OpenAI:     prompt_tokens / completion_tokens / total_tokens
            Anthropic:  input_tokens / output_tokens / cache_read_input_tokens
            Ollama:     prompt_eval_count / eval_count
            Gemini:     promptTokenCount / candidatesTokenCount / totalTokenCount
        """
        if not isinstance(raw, dict):
            return NormalizedUsage()

        prompt = _first_int(raw,
                            "prompt_tokens", "input_tokens",
                            "prompt_eval_count", "promptTokenCount")
        completion = _first_int(raw,
                                "completion_tokens", "output_tokens",
                                "eval_count", "candidatesTokenCount")
        total = _first_int(raw,
                           "total_tokens", "totalTokenCount")
        if total <= 0:
            total = prompt + completion

        cached = _first_int(raw,
                            "cached_tokens", "cache_read_input_tokens",
                            "cachedContentTokenCount")

        # Reasoning tokens appear under different names.
        reasoning = _first_int(raw,
                               "reasoning_tokens", "thoughtsTokenCount")
        if reasoning == 0:
            details = raw.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning = _first_int(details, "reasoning_tokens")
        if reasoning == 0:
            details = raw.get("output_tokens_details")
            if isinstance(details, dict):
                reasoning = _first_int(details, "reasoning_tokens")

        cache_creation = _first_int(raw, "cache_creation_tokens",
                                    "cache_creation_input_tokens")

        return NormalizedUsage(
            prompt_tokens=max(0, prompt),
            completion_tokens=max(0, completion),
            total_tokens=max(0, total),
            cached_tokens=max(0, cached),
            reasoning_tokens=max(0, reasoning),
            cache_creation_tokens=max(0, cache_creation),
            provider_raw=dict(raw),
        )

    # ------------------------------------------------------------------
    # MISC
    # ------------------------------------------------------------------

    @staticmethod
    def _elapsed_ms(started_at: Optional[float]) -> int:
        if not started_at:
            return 0
        try:
            return max(0, int((time.time() - started_at) * 1000))
        except Exception:
            return 0


# ======================================================================
# MERGING STREAMED CHUNKS
# ======================================================================

class StreamAccumulator:
    """
    Accumulate normalized streaming chunks into one NormalizedResponse.

    The provider's `stream()` yields many partial LLMResponses. This
    helper runs each through the normalizer, concatenates text, and
    keeps the final tool-call set. The loop calls `.finalize()` when the
    stream ends to get the same shape a non-streaming call would return.

    It's also safe to feed a single non-streaming response through
    — the result is that response, wrapped.
    """

    def __init__(self, normalizer: Optional[ResponseNormalizer] = None,
                 *, provider: str = "", model: str = ""):
        self._normalizer = normalizer or ResponseNormalizer()
        self._provider = provider
        self._model = model
        self._started_at = time.time()

        self._text_parts: List[str] = []
        self._reasoning_parts: List[str] = []
        self._tool_calls: List[NormalizedToolCall] = []
        self._finish_reason: str = FinishReason.STOP.value
        self._usage: NormalizedUsage = NormalizedUsage()
        self._error: Optional[NormalizedError] = None
        self._streamed: bool = False
        self._raw: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------

    def feed(self, chunk: Any) -> NormalizedResponse:
        """Ingest one raw provider chunk; return its normalized form."""
        n = self._normalizer.normalize(
            chunk,
            provider=self._provider,
            model=self._model,
        )
        if n.error is not None and self._error is None:
            # First error wins; later chunks are ignored.
            self._error = n.error
            return n
        if self._error is not None:
            return n  # already errored

        if n.text:
            self._text_parts.append(n.text)
        if n.reasoning_text:
            self._reasoning_parts.append(n.reasoning_text)

        # Tool calls are usually assembled by the adapter and arrive whole
        # on the final chunk. Multiple chunks carrying calls is allowed.
        for tc in n.tool_calls:
            self._tool_calls.append(tc)

        if n.finish_reason and n.finish_reason != FinishReason.STOP.value:
            self._finish_reason = n.finish_reason
        elif n.tool_calls:
            # Any chunk with tool calls marks the finish as tool_calls
            # unless a stronger reason arrives later.
            if self._finish_reason == FinishReason.STOP.value:
                self._finish_reason = FinishReason.TOOL_CALLS.value

        # Usage merges: later chunks may carry updated counters.
        self._usage = _merge_usage(self._usage, n.usage)

        if n.streamed:
            self._streamed = True
        if n.raw and self._raw is None:
            self._raw = n.raw

        return n

    # ------------------------------------------------------------------

    def finalize(self) -> NormalizedResponse:
        """Build the consolidated NormalizedResponse."""
        text = "".join(self._text_parts)
        reasoning = "".join(self._reasoning_parts)

        # If the final chunk didn't set TOOL_CALLS but we accumulated calls,
        # make sure the finish reason reflects reality.
        finish = self._finish_reason
        if self._tool_calls and finish == FinishReason.STOP.value:
            finish = FinishReason.TOOL_CALLS.value

        # If the accumulator saw no usage but a chunk did, take it.
        # (Already handled by _merge_usage above.)

        return NormalizedResponse(
            text=text,
            reasoning_text=reasoning,
            tool_calls=list(self._tool_calls),
            provider=self._provider,
            model=self._model,
            finish_reason=finish,
            usage=self._usage,
            error=self._error,
            streamed=self._streamed,
            latency_ms=int((time.time() - self._started_at) * 1000),
            raw=self._raw,
        )

    @property
    def has_output(self) -> bool:
        """True once any text or tool call has been accumulated."""
        return bool(self._text_parts or self._tool_calls)

    @property
    def errored(self) -> bool:
        return self._error is not None


# ======================================================================
# HELPERS
# ======================================================================

def _as_response_dict(response: Any) -> Dict[str, Any]:
    """
    View an LLMResponse (or any duck-typed object, or a dict) as a
    plain dict. Missing fields default to None so `.get()` works.
    """
    if isinstance(response, dict):
        return response
    if isinstance(response, LLMResponse):
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "usage": response.usage,
            "tool_calls": response.tool_calls,
            "finish_reason": response.finish_reason,
            "raw": response.raw,
        }
    # Generic duck typing.
    return {
        "content": getattr(response, "content", None),
        "model": getattr(response, "model", ""),
        "provider": getattr(response, "provider", ""),
        "usage": getattr(response, "usage", {}) or {},
        "tool_calls": getattr(response, "tool_calls", None) or [],
        "finish_reason": getattr(response, "finish_reason", None),
        "reasoning": getattr(response, "reasoning_content", None),
        "reasoning_content": getattr(response, "reasoning_content", None),
        "thinking": getattr(response, "thinking", None),
        "raw": getattr(response, "raw", None),
    }


def _synth_id() -> str:
    """Synthesize a stable-looking tool-call id."""
    return f"call_{uuid.uuid4().hex[:12]}"


def _coerce_arguments(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Return (args_dict, parse_error). args_dict is always a dict.

    Rules:
        None / ""             → ({}, None)
        dict                  → (dict, None)
        JSON object string    → (parsed, None)
        other string          → ({}, "invalid JSON ...")
        non-dict JSON         → ({}, "expected object, got list")
        other type            → ({}, "was <type>, not object or JSON string")
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}, None
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            return {}, f"invalid JSON arguments ({e.msg} at pos {e.pos})"
        if isinstance(parsed, dict):
            return parsed, None
        return {}, f"tool arguments parsed as {type(parsed).__name__}, expected object"
    if isinstance(raw, list):
        # Some providers send a positional array of arguments. We refuse
        # to guess at the parameter names — that's a schema mismatch.
        return {}, "tool arguments were a list, expected object"
    return {}, f"tool arguments were {type(raw).__name__}, not object or JSON string"


def _first_int(d: Dict[str, Any], *keys: str) -> int:
    """First key present in `d` with an int-coercible value."""
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return 0


def _merge_usage(
    base: NormalizedUsage, incoming: NormalizedUsage
) -> NormalizedUsage:
    """
    Merge two usage readings.

    Streaming providers often repeat cumulative counters rather than
    deltas, so the incoming value wins when it's larger. Deltas also
    work: if the new reading is smaller, we take the max — never lose
    information.
    """
    def _max(a: int, b: int) -> int:
        return max(a or 0, b or 0)

    raw = dict(base.provider_raw or {})
    raw.update(incoming.provider_raw or {})

    return NormalizedUsage(
        prompt_tokens=_max(base.prompt_tokens, incoming.prompt_tokens),
        completion_tokens=_max(base.completion_tokens, incoming.completion_tokens),
        total_tokens=_max(base.total_tokens, incoming.total_tokens),
        cached_tokens=_max(base.cached_tokens, incoming.cached_tokens),
        reasoning_tokens=_max(base.reasoning_tokens, incoming.reasoning_tokens),
        cache_creation_tokens=_max(base.cache_creation_tokens,
                                   incoming.cache_creation_tokens),
        provider_raw=raw,
    )


# ======================================================================
# MODULE-LEVEL CONVENIENCE
# ======================================================================

_shared: Optional[ResponseNormalizer] = None


def get_normalizer() -> ResponseNormalizer:
    """Process-wide shared normalizer (it's stateless)."""
    global _shared
    if _shared is None:
        _shared = ResponseNormalizer()
    return _shared


def normalize(response: Any, *, provider: str = "", model: str = "",
              started_at: Optional[float] = None) -> NormalizedResponse:
    """Shortcut: `get_normalizer().normalize(...)`."""
    return get_normalizer().normalize(
        response, provider=provider, model=model, started_at=started_at,
    )


def normalize_error(error: Any, *, provider: str = "",
                    model: str = "") -> NormalizedResponse:
    """Shortcut: `get_normalizer().normalize_error(...)`."""
    return get_normalizer().normalize_error(error, provider=provider, model=model)


__all__ = [
    "NormalizedToolCall",
    "NormalizedUsage",
    "NormalizedError",
    "NormalizedResponse",
    "ResponseNormalizer",
    "StreamAccumulator",
    "get_normalizer",
    "normalize",
    "normalize_error",
]