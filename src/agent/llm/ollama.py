"""Native Ollama provider for local models.

Talks directly to a running Ollama daemon over its HTTP API — not through
an OpenAI-compatible shim — so native streaming, tool calls, keep-alive
control, and Ollama-specific error shapes stay intact and get converted
into the framework's internal format at the boundary.

    Framework type           Ollama wire shape
    ────────────────         ──────────────────────────────────────────
    Message(role=system)     {"role":"system","content":...}
    Message(role=user)       {"role":"user","content":...}
    Message(role=assistant)  {"role":"assistant","content":...,
                              "tool_calls":[...]}
    Message(role=tool)       {"role":"tool","content":...} (no tool_call_id —
                              Ollama pairs by order, not by id)
    ToolCall                 {"function":{"name":..,"arguments":{...}}}
    ToolSpec                 {"type":"function","function":{...}}
    LLMResponse              {"message":{...},"done":true,"prompt_eval_count",
                              "eval_count","total_duration",...}

What this file owns
-------------------
    * Message conversion       — framework → Ollama's flat chat shape
    * Tool conversion          — ToolSpec → Ollama's function schema
    * Response conversion      — Ollama's message → LLMResponse
    * Streaming                — newline-delimited JSON (NDJSON), NOT SSE
    * Errors                   — HTTP errors and {"error": "..."} bodies
    * Retries                  — on connection failures and 5xx
    * Lifecycle                — one long-lived httpx.AsyncClient
    * Model discovery          — GET /api/tags, GET /api/show
    * Context window lookup    — from /api/show's model_info

What this file does NOT own
---------------------------
    * Model selection / fallback chains  — agent.llm.runtime
    * Cost tables                        — local models are free
    * Tool execution                     — the loop dispatches

Ollama API specifics worth knowing
----------------------------------
    * Default endpoint is http://localhost:11434. No API key needed.
    * Streaming is newline-delimited JSON: each line is a full JSON object
      with a `done` boolean on the last one. Not Server-Sent Events.
    * Tool calls come back in `message.tool_calls` as
      [{"function":{"name":..,"arguments":{dict}}}}]. Ollama does NOT
      return an `id`; we synthesize one so downstream pairing works.
    * Tool results are sent as role="tool" messages. Ollama matches them
      to the preceding assistant turn by order — there is no tool_call_id.
    * Some models emit tool calls as plain text (a JSON blob in `content`)
      instead of the native `tool_calls` field. We catch that shape as a
      fallback and convert it.
    * `options.num_ctx` sets the context window; Ollama silently truncates
      to the model's trained max if you ask for more.
    * `/api/show` returns the model's real context length under
      `model_info.<arch>.context_length`, plus parameter size and
      quantization.
    * `keep_alive` controls how long the model stays loaded in VRAM.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

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

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "5m"

# Ollama mostly returns 200 with an {"error":"..."} body. HTTP statuses
# that are worth retrying.
_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}

# Optional top-level body fields we forward when the caller supplies them.
_PASSTHROUGH_OPTION_KEYS = (
    "num_predict",       # max output tokens (Ollama's name for max_tokens)
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "temperature",
    "seed",
    "stop",
    "num_ctx",
    "num_batch",
    "num_gpu",
    "num_thread",
    "presence_penalty",
    "frequency_penalty",
)

# Fields under /api/show.model_info whose *suffix* identifies context
# length, regardless of the architecture prefix (llama, qwen2, mistral,
# phi3, gemma, ...). We match by suffix rather than a fixed key.
_CONTEXT_SUFFIXES = (".context_length", ".context_window", ".max_position_embeddings")

# Detects a JSON object that might be a text-form tool call emitted by a
# model that doesn't use Ollama's native tool_calls field. Very narrow on
# purpose: we only accept {"name": ..., "arguments": {...}} or
# {"tool": ..., "parameters": {...}} at the top level.
_TEXT_TOOL_RE = re.compile(
    r'\{\s*"(?:name|tool)"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{',
    re.DOTALL,
)


# ======================================================================
# PROVIDER
# ======================================================================

class OllamaProvider(LLMProvider):
    """
    Adapter for a local Ollama daemon.

    Configure with a base URL and (optionally) a default model:

        OllamaProvider(ProviderConfig(
            base_url="http://localhost:11434",
            default_model="qwen2.5:7b-instruct",
        ))

    No API key is required. `base_url` defaults to Ollama's standard
    localhost endpoint when omitted.
    """

    name = "ollama"

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        if not self.config.base_url:
            self.config.base_url = DEFAULT_BASE_URL
        # Persistent HTTP client so the daemon connection is reused.
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the shared HTTP client. Safe to call more than once."""
        if self._client is not None:
            self._started = True
            return

        base = (self.config.base_url or DEFAULT_BASE_URL).rstrip("/")
        # Local models can take a long time to load on first call and even
        # longer to generate. Use a generous read timeout; the caller can
        # still bound the whole request with asyncio.wait_for.
        self._client = httpx.AsyncClient(
            base_url=base,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(
                connect=10.0,
                read=max(self.config.timeout, 600.0),
                write=self.config.timeout,
                pool=self.config.timeout,
            ),
            follow_redirects=True,
        )
        self._started = True

    async def close(self) -> None:
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

    # ------------------------------------------------------------------
    # CAPABILITY / CONFIG
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        # No API key needed; a base URL is the only requirement.
        return bool(self.config.base_url)

    def _is_local(self) -> bool:
        base = (self.config.base_url or "").lower()
        return any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))

    def capabilities(self) -> Dict[str, bool]:
        return {
            "tools": True,         # native since Ollama 0.3+
            "streaming": True,
            "vision": True,        # model-dependent (llava, llama3.2-vision)
            "json_mode": True,     # via format:"json"
            "system_prompt": True,
            "parallel_tool_calls": False,
        }

    # ------------------------------------------------------------------
    # MESSAGE CONVERSION (framework → Ollama)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
        """
        Flatten framework messages into Ollama's flat chat shape.

        Ollama rules:
            * role="tool" carries no tool_call_id — pairing is by order.
            * assistant turns with tool_calls still need a `content` string
              (Ollama rejects missing content).
            * tool_call arguments are a dict, not a JSON string.
        """
        out: List[Dict[str, Any]] = []
        for raw in messages or []:
            m = _coerce_message(raw)
            role = (m.get("role") or "user").lower()

            if role == "tool":
                out.append({
                    "role": "tool",
                    "content": _stringify_content(m.get("content")),
                })
                continue

            if role == "assistant":
                entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": _stringify_content(m.get("content")) or "",
                }
                calls = _assistant_tool_calls_out(m.get("tool_calls") or [])
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue

            # system / user / anything else.
            out.append({
                "role": role,
                "content": _stringify_content(m.get("content")),
            })
        return out

    # ------------------------------------------------------------------
    # TOOL CONVERSION (framework → Ollama)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        """
        Ollama uses OpenAI's tool schema: {"type":"function","function":{
        "name","description","parameters"}}.
        """
        if not tools:
            return []
        out: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, ToolSpec):
                out.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.parameters
                                      or {"type": "object", "properties": {}},
                    },
                })
                continue
            if isinstance(t, dict):
                if "type" in t and "function" in t:
                    out.append(t)
                    continue
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                name = fn.get("name")
                if not name:
                    continue
                out.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": fn.get("description", "") or "",
                        "parameters": fn.get("parameters")
                                      or fn.get("input_schema")
                                      or {"type": "object", "properties": {}},
                    },
                })
                continue
            raise ProviderError(
                f"Unsupported tool spec type: {type(t).__name__}",
                code=ProviderErrorCode.UNSUPPORTED,
                provider="ollama",
            )
        return out

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
        options: Dict[str, Any] = {
            "temperature": temperature,
            # Ollama calls max_tokens "num_predict".
            "num_predict": int(max_tokens) if max_tokens and max_tokens > 0 else 4096,
        }
        for key in _PASSTHROUGH_OPTION_KEYS:
            v = kwargs.get(key)
            if v is not None and key != "temperature":
                options[key] = v

        body: Dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "stream": bool(stream),
            "options": options,
            # Keep the model loaded between turns. Caller can override.
            "keep_alive": kwargs.get("keep_alive") or DEFAULT_KEEP_ALIVE,
        }

        ollama_tools = self._convert_tools(tools)
        if ollama_tools:
            body["tools"] = ollama_tools
            # Ollama doesn't support "required"/"none" — "auto" is the only
            # mode it honours. We pass it through only when it's the default.
            if tool_choice == "required":
                # Best-effort: prepend an instruction the model tends to
                # follow, since Ollama can't force tool calls at the API.
                body["messages"].insert(0, {
                    "role": "system",
                    "content": "You must call one of the available tools.",
                })

        # Native JSON mode. If the caller passed response_format from the
        # OpenAI side, honour the intent.
        fmt = kwargs.get("format")
        if fmt is None and isinstance(kwargs.get("response_format"), dict):
            rf = kwargs["response_format"]
            if rf.get("type") == "json_object" or rf.get("type") == "json_schema":
                fmt = "json"
        if fmt is not None:
            body["format"] = fmt

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
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "No model specified and no default configured. "
                "Set ProviderConfig.default_model or pass model=.",
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=self.name,
            )

        if stop:
            kwargs.setdefault("stop", list(stop))

        client = self._require_client()
        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=False,
            **kwargs,
        )

        data = await self._post_json(client, "/api/chat", body, model)
        return self._parse(data, model)

    # ------------------------------------------------------------------
    # STREAMING (NDJSON)
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
        Yield incremental LLMResponse chunks.

        Ollama streams newline-delimited JSON. Each line is a full object:
            {"message":{"role":"assistant","content":"..."},"done":false}
        The final line has "done":true plus usage fields (prompt_eval_count,
        eval_count, total_duration, load_duration, ...).

        Tool calls arrive whole in one message's `tool_calls`, typically on
        the final chunk or on a chunk that has empty content.
        """
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "No model specified and no default configured",
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=self.name,
            )

        if stop:
            kwargs.setdefault("stop", list(stop))

        client = self._require_client()
        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=True,
            **kwargs,
        )

        content_buf: List[str] = []
        tool_calls_accum: List[ToolCall] = []
        finish_reason = FinishReason.STOP.value
        usage: Dict[str, int] = {}

        try:
            async with client.stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    raise _status_to_error(
                        resp.status_code,
                        raw.decode("utf-8", "replace"),
                        model, self.name,
                    )

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Ollama sometimes reports errors inline mid-stream.
                    if isinstance(event, dict) and event.get("error"):
                        raise _error_object_to_error(
                            str(event["error"]), model, self.name
                        )

                    msg = event.get("message") or {}
                    text = msg.get("content") or ""
                    if text:
                        content_buf.append(text)
                        yield LLMResponse(
                            content=text,
                            model=model,
                            provider=self.name,
                            finish_reason="",
                            usage={},
                        )

                    for tc in msg.get("tool_calls") or []:
                        parsed = _parse_ollama_tool_call(tc)
                        if parsed is not None:
                            tool_calls_accum.append(parsed)
                            finish_reason = FinishReason.TOOL_CALLS.value

                    if event.get("done"):
                        fr = event.get("done_reason")
                        if fr:
                            finish_reason = _map_done_reason(fr, finish_reason)
                        usage = _usage_from_event(event)
                        break

        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Ollama stream timed out after {self.config.timeout}s",
                code=ProviderErrorCode.TIMEOUT,
                provider=self.name, model=model,
                retryable=True, cause=exc,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {self.config.base_url}: {exc}",
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

        # Fallback: some models emit tool calls as a JSON blob in content
        # instead of using the native field. Recover them once, here.
        if not tool_calls_accum and content_buf:
            recovered = _recover_text_tool_calls("".join(content_buf))
            if recovered:
                tool_calls_accum = recovered
                finish_reason = FinishReason.TOOL_CALLS.value

        yield LLMResponse(
            content="",
            model=model,
            provider=self.name,
            tool_calls=tool_calls_accum,
            finish_reason=finish_reason,
            usage=usage,
            raw={"streamed": True},
        )

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        """Static list — live discovery is async. Use `discover_models()`."""
        return []

    async def discover_models(self) -> List[ModelInfo]:
        """
        GET /api/tags → installed models, then GET /api/show for the ones
        we care about (context length, parameter count, quantization).

        Failures are swallowed: a missing daemon yields an empty list, not
        an exception, so the caller can show an empty picker.
        """
        try:
            client = self._require_client()
        except ProviderError:
            try:
                await self.start()
            except Exception:
                return []
            client = self._require_client()

        try:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Ollama discovery failed: %s", exc)
            return []

        raw = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []

        out: List[ModelInfo] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model")
            if not name:
                continue
            info = ModelInfo(
                id=str(name),
                provider=self.name,
                name=str(name),
                description="",
                context_window=0,
                max_output=0,
                supports_tools=True,       # can't know without a probe; assume true
                supports_streaming=True,
                supports_vision=False,     # flipped below when we see a vision tag
                supports_json_mode=True,
            )
            # Details Ollama already provides on /api/tags.
            details = entry.get("details") or {}
            if isinstance(details, dict):
                family = str(details.get("family") or "").lower()
                if "llava" in family or "vision" in family:
                    info.supports_vision = True
                info.description = (
                    f"{details.get('parameter_size', '')} "
                    f"{details.get('quantization_level', '')}"
                ).strip()
            out.append(info)

        # Enrich with context length from /api/show. This is an extra
        # round-trip per model, so cap it — a picker with 40 local models
        # shouldn't do 40 requests.
        for info in out[:20]:
            try:
                ctx = await self._fetch_context_length(client, info.id)
                if ctx:
                    info.context_window = ctx
            except Exception:
                pass

        return out

    async def _fetch_context_length(
        self, client: httpx.AsyncClient, model: str
    ) -> int:
        """Read the model's real context window from /api/show."""
        resp = await client.post("/api/show", json={"model": model})
        resp.raise_for_status()
        data = resp.json()
        return _extract_context_length(data)

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Ping GET /api/tags. Fast and cheap."""
        if not self.is_configured():
            return False
        try:
            client = self._require_client()
        except ProviderError:
            try:
                await self.start()
            except Exception:
                return False
            client = self._require_client()
        try:
            resp = await client.get("/api/tags")
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
                data = resp.json()
                # Ollama frequently returns HTTP 200 with {"error": "..."}.
                if isinstance(data, dict) and data.get("error"):
                    raise _error_object_to_error(
                        str(data["error"]), model, self.name
                    )
                return data

            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Ollama %s attempt %d/%d: %s (retrying in %.1fs)",
                    model, attempt + 1, attempts, exc.code, delay,
                )
                await asyncio.sleep(delay)

            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    f"Ollama request timed out after {self.config.timeout}s",
                    code=ProviderErrorCode.TIMEOUT,
                    provider=self.name, model=model,
                    retryable=True, cause=exc,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay(attempt))

            except httpx.NetworkError as exc:
                last_error = ProviderError(
                    f"Could not reach Ollama at {self.config.base_url}: {exc}",
                    code=ProviderErrorCode.CONNECTION,
                    provider=self.name, model=model,
                    retryable=True, cause=exc,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                await asyncio.sleep(self._retry_delay(attempt))

            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"Ollama returned non-JSON body: {exc}",
                    code=ProviderErrorCode.SERVER,
                    provider=self.name, model=model,
                    cause=exc,
                ) from exc

        if last_error is not None:
            raise last_error
        raise ProviderError(
            "Request failed with no error recorded",
            code=ProviderErrorCode.UNKNOWN,
            provider=self.name, model=model,
        )

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        base = min(2 ** attempt, 8)
        return base + random.uniform(0, 0.25)

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse(self, data: Dict[str, Any], model: str) -> LLMResponse:
        """Convert an Ollama /api/chat response into our LLMResponse."""
        msg = data.get("message") or {}
        content = str(msg.get("content") or "")

        calls: List[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            parsed = _parse_ollama_tool_call(tc)
            if parsed is not None:
                calls.append(parsed)

        # Fallback: content might be a text-form tool call.
        if not calls and content:
            calls = _recover_text_tool_calls(content)

        done_reason = data.get("done_reason") or ("stop" if data.get("done") else "")
        if calls:
            finish_reason = FinishReason.TOOL_CALLS.value
        else:
            finish_reason = _map_done_reason(done_reason, FinishReason.STOP.value)

        usage = _usage_from_event(data)

        return LLMResponse(
            content=content,
            model=str(data.get("model") or model),
            provider=self.name,
            usage=usage,
            tool_calls=calls,
            finish_reason=finish_reason,
            raw=data,
        )

    # ------------------------------------------------------------------
    # CONTEXT WINDOW
    # ------------------------------------------------------------------

    async def context_window(self, model: Optional[str] = None) -> int:
        """
        Read the model's context length directly. Returns 0 when unknown.

        Useful for the context panel: local models have wildly different
        windows (4k on some llama2 builds, 128k on qwen2.5, 200k on newer
        Claude-adjacent quants) and guessing wrong breaks the token bar.
        """
        model = model or self.config.default_model
        if not model:
            return 0
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()
        try:
            return await self._fetch_context_length(client, model)
        except Exception:
            return 0


# ======================================================================
# HELPERS
# ======================================================================

def _coerce_message(m: Any) -> Dict[str, Any]:
    """Turn whatever the caller passed into a plain dict we can inspect."""
    if isinstance(m, Message):
        return m.to_dict()
    if isinstance(m, dict):
        return dict(m)
    d: Dict[str, Any] = {
        "role": getattr(m, "role", "user"),
        "content": getattr(m, "content", ""),
    }
    tc = getattr(m, "tool_calls", None)
    if tc:
        d["tool_calls"] = tc
    tcid = getattr(m, "tool_call_id", None)
    if tcid:
        d["tool_call_id"] = tcid
    nm = getattr(m, "name", None)
    if nm:
        d["name"] = nm
    return d


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, dict)):
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    return str(content)


def _assistant_tool_calls_out(calls: Any) -> List[Dict[str, Any]]:
    """
    Ollama expects assistant tool_calls as {"function":{"name","arguments"}}
    where arguments is a *dict* (not a JSON string).
    """
    if not calls:
        return []
    out: List[Dict[str, Any]] = []
    for tc in calls:
        if isinstance(tc, ToolCall):
            out.append({
                "function": {
                    "name": tc.name,
                    "arguments": dict(tc.arguments or {}),
                },
            })
            continue
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or tc.get("name")
        if not name:
            continue
        raw_args = fn.get("arguments", tc.get("arguments", {}))
        args: Any = raw_args
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({
            "function": {"name": str(name), "arguments": args},
        })
    return out


def _parse_ollama_tool_call(tc: Any) -> Optional[ToolCall]:
    """
    Convert one entry from Ollama's message.tool_calls into a ToolCall.

    Ollama shape:
        {"function": {"name": "get_weather", "arguments": {"city":"Paris"}}}

    No id is provided. We synthesize one so the framework can pair the
    result to the call.
    """
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    name = fn.get("name")
    if not name:
        return None

    raw_args = fn.get("arguments", tc.get("arguments"))
    args, parse_error = _coerce_args(raw_args)

    call_id = tc.get("id") or f"ollama_{uuid.uuid4().hex[:12]}"
    call = ToolCall(id=call_id, name=str(name), arguments=args)
    if parse_error:
        try:
            setattr(call, "parse_error", parse_error)
        except Exception:
            args.setdefault("_parse_error", parse_error)
    return call


def _coerce_args(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return (args_dict, parse_error). args_dict is always a dict."""
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
    return {}, f"tool arguments were {type(raw).__name__}, not object or JSON string"


def _recover_text_tool_calls(content: str) -> List[ToolCall]:
    """
    Some local models don't emit Ollama's native tool_calls — they write
    a JSON object into the text. Detect the common shape and lift it.

    Accepted shapes (top-level only):
        {"name": "...", "arguments": {...}}
        {"tool": "...", "parameters": {...}}

    Very narrow regex first, then a proper parse. Anything else is left
    alone — we do not want to convert prose into phantom tool calls.
    """
    if not content:
        return []
    if not _TEXT_TOOL_RE.search(content):
        return []

    # Find the outermost JSON object in the text.
    start = content.find("{")
    while start != -1:
        end = _find_matching_brace(content, start)
        if end == -1:
            return []
        chunk = content[start:end + 1]
        try:
            obj = json.loads(chunk)
        except Exception:
            start = content.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("tool")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(name, str) and name:
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                return [ToolCall(
                    id=f"text_{uuid.uuid4().hex[:12]}",
                    name=name,
                    arguments=args,
                )]
        start = content.find("{", start + 1)
    return []


def _find_matching_brace(text: str, start: int) -> int:
    """Find the index of the `}` matching the `{` at `start`."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _map_done_reason(done_reason: str, fallback: str) -> str:
    if not done_reason:
        return fallback
    r = done_reason.lower()
    if r in ("stop", "end_turn"):
        return FinishReason.STOP.value
    if r in ("length", "max_tokens"):
        return FinishReason.LENGTH.value
    if r in ("tool_calls", "tool_use"):
        return FinishReason.TOOL_CALLS.value
    if r in ("content_filter", "refusal"):
        return FinishReason.CONTENT_FILTER.value
    return fallback


def _usage_from_event(event: Dict[str, Any]) -> Dict[str, int]:
    """
    Ollama reports: prompt_eval_count, eval_count, and durations in
    nanoseconds. Map to our token dict; keep durations as extra keys so
    the cost panel can show them if it wants.
    """
    prompt = int(event.get("prompt_eval_count", 0) or 0)
    completion = int(event.get("eval_count", 0) or 0)
    total = prompt + completion

    usage: Dict[str, int] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }

    # Optional timing info — cheap to carry and useful for the cost panel.
    for key in ("total_duration", "load_duration",
                "prompt_eval_duration", "eval_duration"):
        v = event.get(key)
        if isinstance(v, (int, float)) and v:
            # Store as milliseconds under a namespaced key to avoid
            # clobbering token counts.
            usage[f"ms_{key}"] = int(v // 1_000_000)

    return usage


def _extract_context_length(show_data: Dict[str, Any]) -> int:
    """
    Pull the context window out of /api/show's model_info.

    The key is architecture-prefixed and varies by model family:
        llama.context_length
        qwen2.context_length
        mistral.context_length
        phi3.context_length
        gemma2.context_length
        ...
    We match by suffix rather than a fixed key so it works across all of
    them, including future architectures.
    """
    info = show_data.get("model_info") if isinstance(show_data, dict) else None
    if not isinstance(info, dict):
        return 0
    for key, value in info.items():
        if not isinstance(key, str):
            continue
        for suffix in _CONTEXT_SUFFIXES:
            if key.endswith(suffix):
                try:
                    n = int(value)
                    if n > 0:
                        return n
                except Exception:
                    pass
    # Fall back to the top-level parameter size hint if present.
    params = show_data.get("parameters") if isinstance(show_data, dict) else None
    if isinstance(params, str):
        # Look for "num_ctx N" set at model creation time.
        m = re.search(r"num_ctx\s+(\d+)", params)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return 0


def _status_to_error(
    status: int, body_text: str, model: str, provider: str
) -> ProviderError:
    """
    Map an HTTP status + body to a ProviderError.

    Ollama's error bodies are usually {"error":"..."}, but a reverse
    proxy in front of the daemon can return plain text or HTML.
    """
    message = _extract_error_message(body_text) or body_text[:400] or f"HTTP {status}"

    if status in (500, 502, 503, 504):
        return ProviderError(
            f"Ollama server error ({status}): {message}",
            code=ProviderErrorCode.SERVER,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 429:
        return ProviderError(
            f"Rate limited by Ollama: {message}",
            code=ProviderErrorCode.RATE_LIMIT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 408:
        return ProviderError(
            f"Ollama timed out the request: {message}",
            code=ProviderErrorCode.TIMEOUT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 404:
        # Ollama returns 404 with "model 'x' not found".
        code = (ProviderErrorCode.MODEL_NOT_FOUND
                if "model" in message.lower()
                else ProviderErrorCode.BAD_REQUEST)
        return ProviderError(
            f"Not found ({status}): {message}",
            code=code, provider=provider, model=model, status=status,
        )
    if status in (400, 422):
        return ProviderError(
            f"Bad request ({status}): {message}",
            code=ProviderErrorCode.BAD_REQUEST,
            provider=provider, model=model, status=status,
        )
    return ProviderError(
        f"Request failed ({status}): {message}",
        code=ProviderErrorCode.UNKNOWN,
        provider=provider, model=model, status=status,
    )


def _error_object_to_error(message: str, model: str, provider: str) -> ProviderError:
    """
    Ollama returns an HTTP 200 with {"error": "..."} for some failures,
    including "model requires more system memory" and
    "model 'x' not found, try pulling it first".

    Classify from the message text so the caller can react.
    """
    lowered = (message or "").lower()
    if "not found" in lowered and "model" in lowered:
        code = ProviderErrorCode.MODEL_NOT_FOUND
    elif "memory" in lowered or "out of memory" in lowered:
        code = ProviderErrorCode.SERVER
    elif "context" in lowered and ("length" in lowered or "too long" in lowered):
        code = ProviderErrorCode.CONTEXT_LENGTH
    elif "no such" in lowered or "unknown" in lowered:
        code = ProviderErrorCode.BAD_REQUEST
    else:
        code = ProviderErrorCode.UNKNOWN

    retryable = code == ProviderErrorCode.SERVER
    return ProviderError(
        message or "Ollama error",
        code=code,
        provider=provider, model=model,
        retryable=retryable,
    )


def _extract_error_message(body_text: str) -> str:
    """Ollama error bodies are {"error": "..."}."""
    if not body_text:
        return ""
    try:
        data = json.loads(body_text)
    except Exception:
        return body_text.strip()
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, str) and err:
            return err
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return body_text.strip()


__all__ = ["OllamaProvider"]