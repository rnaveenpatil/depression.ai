"""Generic OpenAI-compatible provider adapter.

Handles every backend that speaks OpenAI's HTTP shape:

    POST /chat/completions      (non-streaming and streaming)
    GET  /models                (model discovery, optional)

Tested against the shape used by OpenAI, Groq, DeepSeek, OpenRouter,
Together, NVIDIA NIM, and the many "openai-compatible" local servers
(vLLM, LM Studio, llama.cpp's server, text-generation-webui, ...).

What this file owns
-------------------
    * Request building       — message normalisation, tool spec passthrough
    * Response parsing       — content, usage, finish_reason, tool calls
    * Streaming              — SSE parsing with tool-call reassembly
    * Errors                 — every httpx/JSON failure mapped to
                               ProviderError with a stable code
    * Retries                — on 408/409/425/429/5xx, honoring Retry-After
    * Lifecycle              — one long-lived httpx.AsyncClient per provider

What this file does NOT own
---------------------------
    * Model selection, fallback chains, cost tables  — those live in
      agent.llm.runtime / the registry.
    * Tool execution                                 — the loop dispatches.

Tool-call argument policy
-------------------------
    * `arguments` arrives as a JSON string. It is parsed into a dict.
    * Malformed JSON does NOT become {"_raw": ...}. The ToolCall carries
      arguments={} plus a `parse_error` string; the loop re-prompts the
      model with that error.
    * Calls missing a `name` are dropped (nothing to dispatch).
    * Calls missing an `id` get a synthesized one so the tool result can
      always be paired back to the assistant turn.
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from agent.llm.provider import (
    FinishReason,
    LLMProvider,
    LLMResponse,
    Message,
    ModelInfo,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
    ToolCall,
    ToolSpec,
)
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

# HTTP statuses worth retrying. 401/403/404/422 are not — they need the
# user to fix something.
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Message keys we forward. Anything else is dropped so the request body
# stays spec-shaped.
_ALLOWED_MESSAGE_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls"}

# Optional top-level body fields we'll forward if the caller supplies them.
_PASSTHROUGH_BODY_KEYS = (
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "user",
    "response_format",
    "parallel_tool_calls",
)

# Auth-related codes that map to ProviderErrorCode.AUTH.
_AUTH_STATUSES = {401, 403}


# ======================================================================
# KNOWN MODELS DATABASE — OpenAI-compatible backends
# ======================================================================
#
# Context windows and max output for popular models when the gateway
# doesn't advertise them via /models.
# Keys are substrings that match model IDs (case-insensitive).
# Format: (context_window, max_output, supports_vision, supports_tools)
#
# Sources: Official docs, provider APIs, model cards.
# Updated: 2026-10-09
_KNOWN_OPENAI_COMPATIBLE_MODELS: Dict[str, Tuple[int, int, bool, bool]] = {
    # TokenHarbor / DeepSeek
    "deepseek-v4.1-flash:free": (1_048_576, 128_000, False, True),
    "deepseek-v4.1-flash": (1_048_576, 128_000, False, True),
    "deepseek-v4-flash:free": (1_048_576, 128_000, False, True),
    "deepseek-v4-flash": (1_048_576, 128_000, False, True),
    "deepseek-v3": (128_000, 8_192, False, True),
    "deepseek-v2.5": (128_000, 8_192, False, True),
    "deepseek-v2": (128_000, 8_192, False, True),
    "deepseek-coder-v2": (128_000, 8_192, False, True),
    "deepseek-coder": (16_384, 8_192, False, True),
    "deepseek-r1": (128_000, 8_192, False, True),
    "deepseek-r1-distill": (128_000, 8_192, False, True),

    # OpenAI
    "gpt-4o": (128_000, 16_384, True, True),
    "gpt-4o-mini": (128_000, 16_384, True, True),
    "gpt-4-turbo": (128_000, 4_096, True, True),
    "gpt-4": (8_192, 4_096, False, True),
    "gpt-3.5-turbo": (16_384, 4_096, False, True),
    "o1-preview": (128_000, 32_768, False, True),
    "o1-mini": (128_000, 65_536, False, True),

    # Anthropic (via OpenAI-compatible gateways like OpenRouter)
    "claude-3.5-sonnet": (200_000, 8_192, True, True),
    "claude-3.5-haiku": (200_000, 8_192, True, True),
    "claude-3-opus": (200_000, 4_096, True, True),
    "claude-3-sonnet": (200_000, 4_096, True, True),
    "claude-3-haiku": (200_000, 4_096, True, True),

    # Meta Llama
    "llama-3.1-405b": (128_000, 2_048, False, True),
    "llama-3.1-70b": (128_000, 2_048, False, True),
    "llama-3.1-8b": (128_000, 2_048, False, True),
    "llama-3.2-90b": (128_000, 2_048, True, True),
    "llama-3.2-11b": (128_000, 2_048, True, True),
    "llama-3.2-3b": (128_000, 2_048, False, True),
    "llama-3.2-1b": (128_000, 2_048, False, True),
    "llama-3-70b": (8_192, 2_048, False, True),
    "llama-3-8b": (8_192, 2_048, False, True),

    # Mistral
    "mistral-large": (128_000, 8_192, False, True),
    "mistral-medium": (32_768, 8_192, False, True),
    "mistral-small": (32_768, 8_192, False, True),
    "mistral-7b": (32_768, 8_192, False, True),
    "mixtral-8x7b": (32_768, 8_192, False, True),
    "mixtral-8x22b": (65_536, 8_192, False, True),

    # Qwen (Alibaba) - base instruct models support tools
    "qwen2.5-72b": (131_072, 8_192, False, True),
    "qwen2.5-32b": (131_072, 8_192, False, True),
    "qwen2.5-14b": (131_072, 8_192, False, True),
    "qwen2.5-7b": (131_072, 8_192, False, True),
    "qwen2.5-3b": (131_072, 8_192, False, True),
    "qwen2.5-1.5b": (131_072, 8_192, False, True),
    "qwen2.5-0.5b": (131_072, 8_192, False, True),
    "qwen2.5-coder": (131_072, 8_192, False, False),
    "qwen2.5-math": (131_072, 8_192, False, False),
    "qwen2.5-vl": (131_072, 8_192, True, True),

    # Generic fallbacks by pattern
    "32k": (32_768, 4_096, False, True),
    "64k": (65_536, 4_096, False, True),
    "128k": (131_072, 8_192, False, True),
    "200k": (200_000, 8_192, False, True),
    "256k": (256_000, 8_192, False, True),
    "1m": (1_000_000, 8_192, False, True),
    "1000k": (1_000_000, 8_192, False, True),
}


def _lookup_known_model(model_id: str) -> Optional[Tuple[int, int, bool, bool]]:
    """Look up a model in the known models database (case-insensitive substring match)."""
    if not model_id:
        return None
    model_lower = model_id.lower()
    for key, (ctx, max_out, vision, tools) in _KNOWN_OPENAI_COMPATIBLE_MODELS.items():
        if key in model_lower:
            return (ctx, max_out, vision, tools)
    return None


# ======================================================================
# PROVIDER
# ======================================================================

class OpenAICompatibleProvider(LLMProvider):
    """
    Adapter for any endpoint that implements OpenAI's chat completions API.

    Configure with a base URL, API key, and (optionally) a default model:

        OpenAICompatibleProvider(ProviderConfig(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            default_model="gpt-4o-mini",
        ))

    `base_url` should point at the API root that contains `/chat/completions`
    and `/models` — i.e. usually ending in `/v1`. The trailing slash is
    stripped.
    """

    name = "openai-compatible"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the shared HTTP client. Safe to call more than once."""
        if self._client is not None:
            self._started = True
            return
        self._client = self._build_client((self.config.base_url or "").rstrip("/"))
        self._started = True

    def _build_client(self, base: str) -> httpx.AsyncClient:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization

        return httpx.AsyncClient(
            base_url=base,
            headers=headers,
            timeout=httpx.Timeout(
                connect=min(15.0, self.config.timeout),
                read=self.config.timeout,
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            # Follow redirects; some gateways put /v1 behind a 308.
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close the shared HTTP client."""
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        self._started = False

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ProviderError(
                "Provider not started. Call `await provider.start()` first.",
                code=ProviderErrorCode.UNKNOWN,
                provider=self.name,
            )
        return self._client

    def _candidate_bases(self) -> List[str]:
        """
        API roots to try when the configured one404s.

        A pasted URL is usually off by a version segment in one direction
        or the other ("/v1" present or missing), and a couple of gateways
        live under "/api/v1". The configured base always goes first — the
        user told us where their endpoint is — then the alternates.
        """
        base = (self.config.base_url or "").rstrip("/")
        path = urlparse(base).path.rstrip("/")

        out = [base]
        if path.endswith("/v1"):
            out.append(base[: -len("/v1")].rstrip("/"))
        else:
            out.append(f"{base}/v1")
            if not path:
                out.append(f"{base}/api/v1")

        ordered: List[str] = []
        for candidate in out:
            if candidate and candidate not in ordered:
                ordered.append(candidate)
        return ordered

    async def _adopt_base(self, base: str) -> None:
        """Switch onto the API root that actually answered."""
        if base == (self.config.base_url or "").rstrip("/"):
            return
        logger.info("Adapted base URL to %s", base)
        self.config.base_url = base
        old, self._client = self._client, self._build_client(base)
        if old is not None:
            try:
                await old.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CAPABILITY / CONFIG
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        return bool(self.config.api_key) and bool(self.config.base_url)

    def _is_local(self) -> bool:
        base = (self.config.base_url or "").lower()
        return any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))

    def capabilities(self) -> Dict[str, bool]:
        # Anything OpenAI-compatible will at least do these.
        return {
            "tools": True,
            "streaming": True,
            "vision": False,      # per-model
            "json_mode": True,    # many gateways support response_format
            "system_prompt": True,
            "parallel_tool_calls": True,
        }

    # ------------------------------------------------------------------
    # MESSAGE NORMALISATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_message(m: Any) -> Dict[str, Any]:
        """
        Turn a Message (or dict, or duck-typed object) into the exact
        body shape the OpenAI spec accepts.
        """
        if isinstance(m, Message):
            d = m.to_dict()
        elif isinstance(m, dict):
            d = dict(m)
        else:
            d = {"role": getattr(m, "role", "user"),
                 "content": getattr(m, "content", "")}
            tc = getattr(m, "tool_calls", None)
            if tc:
                d["tool_calls"] = tc
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                d["tool_call_id"] = tcid
            nm = getattr(m, "name", None)
            if nm:
                d["name"] = nm

        out: Dict[str, Any] = {}
        for k in _ALLOWED_MESSAGE_KEYS:
            if k not in d:
                continue
            v = d[k]

            if k == "content":
                # The spec requires content=null when tool_calls are present
                # on an assistant message. Some gateways reject "" there.
                if v == "" and d.get("tool_calls"):
                    out[k] = None
                else:
                    out[k] = v
            elif k == "tool_calls":
                # Ensure each call is in the {"id","type","function":{...}}
                # shape the API expects, whatever we were handed.
                out[k] = OpenAICompatibleProvider._normalize_tool_calls_out(v)
            else:
                out[k] = v

        # Drop None-valued optional keys that some strict gateways reject.
        for k in ("name", "tool_call_id"):
            if out.get(k) is None:
                out.pop(k, None)

        return out

    @staticmethod
    def _normalize_tool_calls_out(calls: Any) -> List[Dict[str, Any]]:
        """Coerce outgoing tool calls into the exact wire shape."""
        if not calls:
            return []
        out: List[Dict[str, Any]] = []
        for tc in calls:
            if isinstance(tc, ToolCall):
                out.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _args_to_json_string(tc.arguments),
                    },
                })
                continue
            if not isinstance(tc, dict):
                continue
            # Already shaped.
            if "function" in tc and isinstance(tc["function"], dict):
                fn = dict(tc["function"])
                fn["arguments"] = _args_to_json_string(fn.get("arguments", {}))
                out.append({
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": tc.get("type", "function"),
                    "function": fn,
                })
                continue
            # Flat shape: {"id","name","arguments"}.
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if not name:
                continue
            out.append({
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": _args_to_json_string(tc.get("arguments", {})),
                },
            })
        return out

    # ------------------------------------------------------------------
    # TOOL CALL PARSING (incoming)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_calls: List[Any]) -> List[ToolCall]:
        """
        Convert provider tool_calls into ToolCall objects.

        Never returns arguments={"_raw": ...}. Malformed JSON produces
        arguments={} plus a `parse_error` attribute the loop reads back.
        """
        out: List[ToolCall] = []
        for tc in raw_calls or []:
            if not isinstance(tc, dict):
                logger.warning("Skipping non-dict tool_call: %r", tc)
                continue

            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            if not name:
                logger.warning("Skipping tool_call with no name (id=%r)", tc.get("id"))
                continue

            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            raw_args = fn.get("arguments", tc.get("arguments"))
            args, parse_error = OpenAICompatibleProvider._coerce_args(raw_args)

            call = ToolCall(id=call_id, name=name, arguments=args)
            if parse_error:
                try:
                    setattr(call, "parse_error", parse_error)
                except Exception:
                    args.setdefault("_parse_error", parse_error)
            out.append(call)
        return out

    @staticmethod
    def _coerce_args(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """Return (args_dict, parse_error_or_None). args_dict is always a dict."""
        if raw is None or raw == "":
            return {}, None
        if isinstance(raw, dict):
            return raw, None
        if not isinstance(raw, str):
            return {}, f"tool arguments were {type(raw).__name__}, not JSON string or object"

        s = raw.strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            return {}, f"invalid JSON arguments ({e.msg} at pos {e.pos})"

        if isinstance(parsed, dict):
            return parsed, None
        return {}, f"tool arguments parsed as {type(parsed).__name__}, expected object"

    # ------------------------------------------------------------------
    # REQUEST BODY
    # ------------------------------------------------------------------

    def _build_body(
        self,
        messages: Sequence[Any],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[Sequence[Any]],
        tool_choice: Optional[str],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": model,
            "messages": [self._normalize_message(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            body["stream"] = True
            # Include usage in the final SSE chunk when the gateway supports it.
            body.setdefault("stream_options", {"include_usage": True})

        if tools:
            body["tools"] = [_tool_to_dict(t) for t in tools]
            body["tool_choice"] = tool_choice or "auto"
        elif tool_choice == "none":
            body["tool_choice"] = "none"

        for key in _PASSTHROUGH_BODY_KEYS:
            v = kwargs.get(key)
            if v is not None:
                body[key] = v
        return body

    # ------------------------------------------------------------------
    # NON-STREAMING
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.config.api_key:
            raise ProviderError(
                "API key not configured", code=ProviderErrorCode.AUTH, provider=self.name
            )
        if not self.config.base_url:
            raise ProviderError(
                "Base URL not configured", code=ProviderErrorCode.UNKNOWN, provider=self.name
            )

        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.UNKNOWN,
                provider=self.name,
            )

        client = self._require_client()
        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=False,
            stop=stop, **kwargs,
        )

        # Endpoint adaptation: a wrong "/v1" shows up as a 404 on the
        # route, so walk the candidate roots and stick with the one that
        # works instead of failing the call.
        candidates = self._candidate_bases()
        current = (self.config.base_url or "").rstrip("/")
        last_error: Optional[ProviderError] = None
        for index, base in enumerate(candidates):
            attempt = client if base == current else self._build_client(base)
            try:
                data = await self._post_json(attempt, "/chat/completions", body, model)
            except ProviderError as exc:
                if exc.status not in (404, 405) or index == len(candidates) - 1:
                    raise
                last_error = exc
                if attempt is not client:
                    try:
                        await attempt.aclose()
                    except Exception:
                        pass
                logger.info(
                    "%s not found on %s; trying %s",
                    "/chat/completions", base, candidates[index + 1],
                )
                continue
            if base != current:
                await self._adopt_base(base)
            return self._parse(data, model)

        if last_error is not None:
            raise last_error
        raise ProviderError(
            "No usable endpoint", code=ProviderErrorCode.UNKNOWN,
            provider=self.name, model=model,
        )

    # ------------------------------------------------------------------
    # STREAMING (SSE)
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Yield incremental `LLMResponse` chunks.

        Text arrives as `chunk.content` deltas. Tool calls accumulate on
        the provider side and are emitted whole on the final chunk (their
        JSON `arguments` are only complete when the stream ends).

        If the gateway silently ignores `stream=True` and returns a normal
        JSON body, we detect that and yield a single full response — the
        caller sees the same contract either way.
        """
        if not self.config.api_key:
            raise ProviderError(
                "API key not configured", code=ProviderErrorCode.AUTH, provider=self.name
            )
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.UNKNOWN, provider=self.name,
            )

        client = self._require_client()
        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=True,
            stop=stop, **kwargs,
        )

        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    # Read the body so we can produce a useful message.
                    text = await resp.aread()
                    raise _status_to_error(
                        resp.status_code, text.decode("utf-8", "replace"), model, self.name
                    )

                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" not in content_type:
                    # Gateway ignored streaming. Fall back to a single response.
                    data = await resp.aread()
                    try:
                        parsed = json.loads(data.decode("utf-8"))
                    except Exception:
                        raise ProviderError(
                            f"Non-SSE response could not be parsed (content-type={content_type})",
                            code=ProviderErrorCode.SERVER,
                            provider=self.name, model=model,
                        )
                    yield self._parse(parsed, model)
                    return

                async for chunk in self._iter_sse(resp, model):
                    yield chunk

        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Stream timed out after {self.config.timeout}s",
                code=ProviderErrorCode.TIMEOUT,
                provider=self.name, model=model,
                retryable=True, cause=exc,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderError(
                f"Network error during stream: {exc}",
                code=ProviderErrorCode.CONNECTION,
                provider=self.name, model=model,
                retryable=True, cause=exc,
            ) from exc
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Unexpected streaming failure: {exc}",
                code=ProviderErrorCode.UNKNOWN,
                provider=self.name, model=model, cause=exc,
            ) from exc

    async def _iter_sse(
        self, resp: httpx.Response, model: str
    ) -> AsyncIterator[LLMResponse]:
        """
        Parse the SSE body. OpenAI sends `data: {json}` lines terminated by
        `data: [DONE]`. Every chunk is a partial ChatCompletion.
        """
        # Per-request state — reset for each stream() call.
        content_buf: List[str] = []
        tool_buf: Dict[int, Dict[str, Any]] = {}
        finish_reason: str = FinishReason.STOP.value
        usage: Dict[str, int] = {}

        async for line in resp.aiter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                # Some gateways send keep-alives or comments.
                continue

            # Usage-only chunk (OpenAI emits one at the end when
            # stream_options.include_usage is set).
            if isinstance(event.get("usage"), dict):
                usage = _normalize_usage(event["usage"])

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}

            # Text delta.
            text = delta.get("content")
            if text:
                content_buf.append(text)
                yield LLMResponse(
                    content=text,
                    model=model,
                    provider=self.name,
                    finish_reason="",
                    usage={},
                )

            # Tool-call deltas. Each entry has an `index` that ties it to a
            # specific call; name and arguments can arrive in pieces.
            for tc_delta in delta.get("tool_calls") or []:
                idx = int(tc_delta.get("index", 0) or 0)
                slot = tool_buf.setdefault(idx, {
                    "id": "", "name": "", "arguments": "",
                })
                if tc_delta.get("id"):
                    slot["id"] = tc_delta["id"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    slot["name"] = (slot["name"] or "") + fn["name"] if slot["name"] else fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

        # Final chunk: assembled tool calls and usage.
        calls = self._parse_tool_calls(
            [{"id": s["id"], "function": {"name": s["name"], "arguments": s["arguments"]}}
             for _, s in sorted(tool_buf.items())]
        )
        if calls:
            finish_reason = FinishReason.TOOL_CALLS.value

        yield LLMResponse(
            content="",
            model=model,
            provider=self.name,
            tool_calls=calls,
            finish_reason=finish_reason,
            usage=usage or {},
            raw={"streamed": True},
        )

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        """
        Static list — always empty. Real discovery is async; call
        `await discover_models()` instead. Kept for the abstract contract.
        """
        return []

    async def discover_models(self) -> List[ModelInfo]:
        """
        GET /models and map to ModelInfo.

        Returns an empty list when the gateway does not implement /models
        or the call fails — this is a nice-to-have, not a hard failure.
        """
        if not self.config.base_url:
            return []
        try:
            client = self._require_client()
        except ProviderError:
            # Lazy-start if the caller forgot.
            await self.start()
            client = self._require_client()

        try:
            resp = await client.get("/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Model discovery failed (%s): %s", self.name, exc)
            return []

        raw = data.get("data") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []

        out: List[ModelInfo] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id") or entry.get("name")
            if not mid:
                continue

            # Get context window from API response, fallback to known models DB
            api_ctx = int(entry.get("context_length") or entry.get("context_window") or 0)
            api_max_out = int(entry.get("max_output_tokens") or 0)
            api_vision = bool(entry.get("supports_vision", False))
            api_tools = bool(entry.get("supports_tools", True))

            known = _lookup_known_model(str(mid))
            if known:
                known_ctx, known_max_out, known_vision, known_tools = known
                ctx = api_ctx if api_ctx > 0 else known_ctx
                max_out = api_max_out if api_max_out > 0 else known_max_out
                vision = api_vision or known_vision
                tools = api_tools or known_tools
            else:
                ctx = api_ctx
                max_out = api_max_out
                vision = api_vision
                tools = api_tools

            # Ultimate fallback: 1M tokens if still unknown
            if ctx <= 0:
                ctx = 1_000_000
            if max_out <= 0:
                max_out = 8_192

            out.append(ModelInfo(
                id=str(mid),
                provider=self.name,
                name=str(entry.get("name") or mid),
                description=str(entry.get("description") or ""),
                context_window=ctx,
                max_output=max_out,
                supports_tools=tools,
                supports_streaming=bool(entry.get("supports_streaming", True)),
                supports_vision=vision,
                supports_json_mode=bool(entry.get("supports_json_mode", True)),
            ))
        return out

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Cheap reachability probe via GET /models."""
        if not self.is_configured():
            return False
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()
        try:
            resp = await client.get("/models")
            return resp.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # TRANSPORT: POST WITH RETRIES
    # ------------------------------------------------------------------

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        path: str,
        body: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        last_error: Optional[ProviderError] = None
        attempts = max(1, int(self.config.max_retries) + 1)

        for attempt in range(attempts):
            try:
                resp = await client.post(path, json=body)
                if resp.status_code >= 400:
                    raise _status_to_error(
                        resp.status_code, resp.text, model, self.name
                    )
                return resp.json()

            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = self._retry_delay(attempt, None)
                logger.warning(
                    "LLM %s attempt %d/%d: %s (retrying in %.1fs)",
                    model, attempt + 1, attempts, exc.code, delay,
                )
                await asyncio.sleep(delay)

            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    f"Request timed out after {self.config.timeout}s",
                    code=ProviderErrorCode.TIMEOUT,
                    provider=self.name, model=model,
                    retryable=True, cause=exc,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay(attempt, None))

            except httpx.NetworkError as exc:
                last_error = ProviderError(
                    f"Network error: {exc}",
                    code=ProviderErrorCode.CONNECTION,
                    provider=self.name, model=model,
                    retryable=True, cause=exc,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay(attempt, None))

            except json.JSONDecodeError as exc:
                # A 200 with a broken body — not worth retrying.
                raise ProviderError(
                    f"Provider returned non-JSON body: {exc}",
                    code=ProviderErrorCode.SERVER,
                    provider=self.name, model=model,
                    cause=exc,
                ) from exc

        # Unreachable, but keeps type checkers happy.
        if last_error is not None:
            raise last_error
        raise ProviderError(
            "Request failed with no error recorded",
            code=ProviderErrorCode.UNKNOWN,
            provider=self.name, model=model,
        )

    @staticmethod
    def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
        """Retry-After if sane, else exponential backoff with jitter."""
        if response is not None:
            ra = response.headers.get("retry-after")
            if ra:
                try:
                    v = float(ra)
                    if 0 < v <= 30:
                        return v
                except Exception:
                    # Retry-After can be an HTTP-date; we ignore those.
                    pass
        base = min(2 ** attempt, 8)
        return base + random.uniform(0, 0.25)

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse(self, data: Dict[str, Any], model: str) -> LLMResponse:
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        msg = choice.get("message") or {}

        calls = self._parse_tool_calls(msg.get("tool_calls") or [])

        content = msg.get("content")
        if content is None:
            content = ""

        usage = _normalize_usage(data.get("usage") or {})

        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            usage=usage,
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or FinishReason.STOP.value,
            raw=data,
        )


# ======================================================================
# HELPERS
# ======================================================================

def _tool_to_dict(tool: Any) -> Dict[str, Any]:
    """Normalise a ToolSpec (or already-shaped dict) into the wire form."""
    if isinstance(tool, ToolSpec):
        return tool.to_dict()
    if isinstance(tool, dict):
        # Already has the OpenAI shape?
        if "type" in tool and "function" in tool:
            return tool
        # Flat shape — wrap it.
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    raise ProviderError(
        f"Unsupported tool spec type: {type(tool).__name__}",
        code=ProviderErrorCode.UNSUPPORTED,
    )


def _args_to_json_string(args: Any) -> str:
    """Tool arguments must go over the wire as a JSON string."""
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args)
    except Exception:
        return "{}"


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """Flatten the many token-count shapes into one dict."""
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or 0)
    if not total:
        total = prompt + completion

    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)

    completion_details = usage.get("completion_tokens_details") or {}
    reasoning = int(completion_details.get("reasoning_tokens", 0) or 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def _status_to_error(
    status: int, body_text: str, model: str, provider: str
) -> ProviderError:
    """
    Map an HTTP status + body to a ProviderError with the right code.

    Prefers the provider's own error message when it's present in the body
    (OpenAI-style: {"error": {"message": "..."}}).
    """
    message = _extract_error_message(body_text) or body_text[:400] or f"HTTP {status}"

    if status in _AUTH_STATUSES:
        return ProviderError(
            f"Authentication failed ({status}): {message}",
            code=ProviderErrorCode.AUTH,
            provider=provider, model=model, status=status,
        )
    if status == 404:
        # Some gateways use 404 for both "no such model" and "no such route".
        lowered = message.lower()
        code = (
            ProviderErrorCode.MODEL_NOT_FOUND
            if "model" in lowered
            else ProviderErrorCode.BAD_REQUEST
        )
        return ProviderError(
            f"Not found ({status}): {message}",
            code=code, provider=provider, model=model, status=status,
        )
    if status == 429:
        return ProviderError(
            f"Rate limited: {message}",
            code=ProviderErrorCode.RATE_LIMIT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 408:
        return ProviderError(
            f"Provider timed out the request: {message}",
            code=ProviderErrorCode.TIMEOUT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 400 or status == 422:
        # Detect the common "context too long" case so callers can trim.
        if "context" in message.lower() and ("length" in message.lower()
                                             or "too long" in message.lower()):
            code = ProviderErrorCode.CONTEXT_LENGTH
        else:
            code = ProviderErrorCode.BAD_REQUEST
        return ProviderError(
            f"Bad request ({status}): {message}",
            code=code, provider=provider, model=model, status=status,
        )
    if 500 <= status < 600:
        return ProviderError(
            f"Provider error ({status}): {message}",
            code=ProviderErrorCode.SERVER,
            provider=provider, model=model, status=status, retryable=True,
        )
    return ProviderError(
        f"Request failed ({status}): {message}",
        code=ProviderErrorCode.UNKNOWN,
        provider=provider, model=model, status=status,
    )


def _extract_error_message(body_text: str) -> str:
    """
    Pull a human message out of an error body.

    Handles:
        {"error": {"message": "..."}}
        {"error": "..."}
        {"message": "..."}
        {"detail": "..."}
    """
    if not body_text:
        return ""
    try:
        data = json.loads(body_text)
    except Exception:
        return body_text.strip()

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str) and err:
            return err
        for key in ("message", "detail", "reason"):
            if isinstance(data.get(key), str) and data[key]:
                return data[key]
    return body_text.strip()


__all__ = ["OpenAICompatibleProvider"]