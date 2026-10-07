"""Native Anthropic API adapter for Claude models.

Handles the Anthropic Messages API (POST /v1/messages) directly — not
through an OpenAI-compatible shim — so Claude's distinct shapes stay
intact and get converted into the framework's internal format at the
boundary.

    Framework type         Anthropic wire shape
    ─────────────────      ──────────────────────────────────────────
    Message(role=system)   top-level "system" field (string or blocks)
    Message(role=user)     message with role="user"
    Message(role=assistant) message with role="assistant"
    ToolCall               content block: {"type":"tool_use", ...}
    Message(role=tool)     user turn with {"type":"tool_result", ...}
    ToolSpec               {"name","description","input_schema"}
    LLMResponse            {"content":[...], "usage":{...}, "stop_reason"}

What this file owns
-------------------
    * Message conversion       — internal → Anthropic's content-block form
    * Response conversion      — Anthropic's content blocks → LLMResponse
    * Streaming                — SSE parsing with tool_use reassembly
    * Errors                   — Anthropic error JSON mapped to ProviderError
    * Retries                  — on 429/5xx/overloaded, honoring Retry-After
    * Lifecycle                — one long-lived httpx.AsyncClient

What this file does NOT own
---------------------------
    * Model selection / fallback chains  — agent.llm.runtime
    * Cost tables                        — registry metadata
    * Tool execution                     — the loop dispatches

Anthropic API specifics worth knowing
-------------------------------------
    * System prompts are NOT messages. They go in the top-level `system`
      field. We hoist every role="system" Message into that field, joined
      by blank lines, and drop them from the messages array.
    * Every response is a list of content blocks. Blocks are one of:
      {"type":"text","text":...} and {"type":"tool_use","id":..,
      "name":..,"input":{...}}. Other block types (thinking, images in
      some versions) are tolerated and ignored for now.
    * `tool_result` blocks are NOT top-level messages — they must appear
      inside a role="user" turn. We rewrite every role="tool" Message
      into a user turn carrying a tool_result block, and merge adjacent
      ones so multiple tool results for one assistant turn arrive
      together (Anthropic requires this).
    * `max_tokens` is REQUIRED on every request. We always send a value.
    * Auth is `x-api-key`, not `Authorization: Bearer`.
    * Anthropic requires `anthropic-version` on every request.
"""

from __future__ import annotations

import asyncio
import json
import random
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

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

# Statuses worth retrying. 401/403/404/400/422 are not.
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Anthropic's `stop_reason` → our FinishReason.
_STOP_REASON_MAP = {
    "end_turn": FinishReason.STOP.value,
    "stop_sequence": FinishReason.STOP.value,
    "max_tokens": FinishReason.LENGTH.value,
    "tool_use": FinishReason.TOOL_CALLS.value,
    "pause_turn": FinishReason.STOP.value,   # rare, but pass through
    "refusal": FinishReason.CONTENT_FILTER.value,
}

# Anthropic error `type` → our ProviderErrorCode. Used when the body
# carries a structured error object.
_ERROR_TYPE_MAP = {
    "authentication_error": ProviderErrorCode.AUTH,
    "permission_error": ProviderErrorCode.AUTH,
    "invalid_request_error": ProviderErrorCode.BAD_REQUEST,
    "not_found_error": ProviderErrorCode.MODEL_NOT_FOUND,
    "rate_limit_error": ProviderErrorCode.RATE_LIMIT,
    "overloaded_error": ProviderErrorCode.SERVER,
    "api_error": ProviderErrorCode.SERVER,
    "timeout_error": ProviderErrorCode.TIMEOUT,
}

# Optional top-level body fields we forward when provided.
_PASSTHROUGH_BODY_KEYS = (
    "top_p",
    "top_k",
    "stop_sequences",
    "metadata",
)


# ======================================================================
# PROVIDER
# ======================================================================

class AnthropicProvider(LLMProvider):
    """
    Adapter for the Anthropic Messages API.

    Configure with an API key and, optionally, a base URL:

        AnthropicProvider(ProviderConfig(
            api_key="sk-ant-...",
            base_url="https://api.anthropic.com",
            default_model="claude-sonnet-4-5",
        ))

    `base_url` defaults to Anthropic's public endpoint when omitted. The
    default `max_tokens` Anthropic requires is applied when the caller
    doesn't supply one.
    """

    name = "anthropic"

    #: Anthropic requires this on every request. Kept as a class attribute
    #: so tests can override it without patching the module.
    api_version: str = ANTHROPIC_VERSION

    #: Anthropic refuses requests without max_tokens. This is our floor.
    default_max_tokens: int = 4096

    def __init__(self, config: Optional[Any] = None):
        super().__init__(config)
        if not self.config.base_url:
            self.config.base_url = DEFAULT_BASE_URL
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the shared HTTP client. Safe to call more than once."""
        if self._client is not None:
            self._started = True
            return

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-version": self.api_version,
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key

        base = (self.config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=base,
            headers=headers,
            timeout=httpx.Timeout(
                connect=min(15.0, self.config.timeout),
                read=self.config.timeout,
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
        return bool(self.config.api_key)

    def _is_local(self) -> bool:
        base = (self.config.base_url or "").lower()
        return any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))

    def capabilities(self) -> Dict[str, bool]:
        return {
            "tools": True,
            "streaming": True,
            "vision": True,
            "json_mode": False,       # no response_format; we use tool_choice
            "system_prompt": True,
            "parallel_tool_calls": True,   # Anthropic supports multiple tool_use blocks
        }

    # ------------------------------------------------------------------
    # MESSAGE CONVERSION (framework → Anthropic)
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: Sequence[Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Split incoming messages into (system_prompt, anthropic_messages).

        Rules:
            * Every role="system" becomes part of the top-level `system`
              string, joined by blank lines. It is removed from `messages`.
            * role="user" with str content      → {"role":"user", "content": str}
            * role="assistant" with str content → {"role":"assistant", "content": str}
            * role="assistant" with tool_calls  → {"role":"assistant",
                                                    "content":[{"type":"tool_use",...}, ...]}
            * role="tool"                       → {"role":"user",
                                                    "content":[{"type":"tool_result",...}]}
              and adjacent tool messages are merged into one user turn.
        """
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []

        for raw in messages or []:
            m = _coerce_message(raw)
            role = (m.get("role") or "user").lower()

            if role == "system":
                text = _stringify_content(m.get("content"))
                if text:
                    system_parts.append(text)
                continue

            if role == "user":
                converted.append({
                    "role": "user",
                    "content": _normalize_content_in(m.get("content")),
                })
                continue

            if role == "assistant":
                blocks = _assistant_to_blocks(m)
                # Anthropic rejects assistant turns with an empty content
                # list. Fall back to a single empty text block.
                if not blocks:
                    blocks = [{"type": "text", "text": ""}]
                converted.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or m.get("id") or "",
                    "content": _stringify_content(m.get("content")),
                }
                # Merge with a preceding user turn if it's also tool_results
                # — Anthropic requires all results for one assistant turn
                # to arrive in a single user message.
                if (converted and converted[-1]["role"] == "user"
                        and isinstance(converted[-1]["content"], list)
                        and all(b.get("type") == "tool_result"
                                for b in converted[-1]["content"])):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
                continue

            # Unknown role — treat as user to avoid dropping content.
            converted.append({
                "role": "user",
                "content": _normalize_content_in(m.get("content")),
            })

        # Anthropic requires the conversation to start with a user turn.
        # If it starts with an assistant turn, prepend an empty user turn.
        if converted and converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": [{"type": "text", "text": ""}]})

        system = "\n\n".join(system_parts)
        return system, converted

    # ------------------------------------------------------------------
    # TOOL CONVERSION (framework → Anthropic)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(tools: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        """
        Map ToolSpec (or already-shaped dicts) to Anthropic's tool format:

            {"name": ..., "description": ..., "input_schema": {...}}

        Note the field is `input_schema`, not `parameters`.
        """
        if not tools:
            return []
        out: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, ToolSpec):
                out.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                })
                continue
            if isinstance(t, dict):
                # Already Anthropic-shaped?
                if "input_schema" in t and "name" in t:
                    out.append(t)
                    continue
                # OpenAI-shaped {"type":"function","function":{...}}.
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                name = fn.get("name")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "description": fn.get("description", "") or "",
                    "input_schema": fn.get("parameters")
                                     or fn.get("input_schema")
                                     or {"type": "object", "properties": {}},
                })
                continue
            raise ProviderError(
                f"Unsupported tool spec type: {type(t).__name__}",
                code=ProviderErrorCode.UNSUPPORTED,
                provider="anthropic",
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
        system, converted = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "model": model,
            "messages": converted,
            # Anthropic requires max_tokens; never send 0/None.
            "max_tokens": int(max_tokens) if max_tokens and max_tokens > 0
                          else self.default_max_tokens,
            "temperature": temperature,
        }
        if system:
            body["system"] = system

        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools
            # tool_choice mapping: our framework uses "auto"/"none"/"required".
            if tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                # Anthropic expresses "no tools" by simply omitting them.
                # We keep them in the request but forbid the model from
                # choosing one.
                body["tool_choice"] = {"type": "none"}
            elif tool_choice == "auto" or tool_choice is None:
                body["tool_choice"] = {"type": "auto"}
            else:
                # Caller supplied a specific tool name.
                body["tool_choice"] = {"type": "tool", "name": str(tool_choice)}

        if stream:
            body["stream"] = True

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
                "Anthropic API key not configured",
                code=ProviderErrorCode.AUTH,
                provider=self.name,
            )
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.UNKNOWN,
                provider=self.name,
            )

        client = self._require_client()
        # `stop` maps to Anthropic's `stop_sequences`.
        if stop:
            kwargs.setdefault("stop_sequences", list(stop))

        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=False,
            **kwargs,
        )

        data = await self._post_json(client, "/v1/messages", body, model)
        return self._parse(data, model)

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
        Yield incremental LLMResponse chunks.

        Anthropic's SSE events we care about:
            message_start          — carries initial usage
            content_block_start    — begins a text or tool_use block
            content_block_delta    — text_delta or input_json_delta
            content_block_stop     — closes a block
            message_delta          — carries stop_reason and final usage
            message_stop           — end of stream
            error                  — an error mid-stream
            ping                   — keep-alive, ignored
        """
        if not self.config.api_key:
            raise ProviderError(
                "Anthropic API key not configured",
                code=ProviderErrorCode.AUTH,
                provider=self.name,
            )
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.UNKNOWN,
                provider=self.name,
            )

        client = self._require_client()
        if stop:
            kwargs.setdefault("stop_sequences", list(stop))

        body = self._build_body(
            messages=messages, model=model,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice, stream=True,
            **kwargs,
        )

        try:
            async with client.stream("POST", "/v1/messages", json=body) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    raise _status_to_error(
                        resp.status_code,
                        raw.decode("utf-8", "replace"),
                        model, self.name,
                    )

                content_type = (resp.headers.get("content-type") or "").lower()
                if "text/event-stream" not in content_type:
                    # Gateway ignored stream=True. Fall back to a single parse.
                    raw = await resp.aread()
                    try:
                        parsed = json.loads(raw.decode("utf-8"))
                    except Exception:
                        raise ProviderError(
                            f"Non-SSE streaming response (content-type={content_type}) "
                            "could not be parsed",
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
        Parse Anthropic's SSE body.

        Anthropic sends named events (`event: content_block_delta`) plus a
        `data:` JSON line for each. We ignore the `event:` name and dispatch
        on the `type` inside the JSON.
        """
        # Per-block accumulation state. Anthropic assigns each content
        # block an index at content_block_start; deltas refer back to it.
        tool_blocks: Dict[int, Dict[str, Any]] = {}
        stop_reason: str = FinishReason.STOP.value
        usage: Dict[str, int] = {}
        text_seen = False

        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "message_start":
                usage = _merge_usage(usage, (event.get("message") or {}).get("usage") or {})
                continue

            if etype == "content_block_start":
                idx = int(event.get("index", 0) or 0)
                block = event.get("content_block") or {}
                btype = block.get("type")
                if btype == "tool_use":
                    tool_blocks[idx] = {
                        "id": block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": block.get("name") or "",
                        "arguments": "",   # JSON string, built up from deltas
                    }
                # Text blocks start empty; deltas carry the content.
                continue

            if etype == "content_block_delta":
                idx = int(event.get("index", 0) or 0)
                delta = event.get("delta") or {}
                dtype = delta.get("type")

                if dtype == "text_delta":
                    text = delta.get("text") or ""
                    if text:
                        text_seen = True
                        yield LLMResponse(
                            content=text,
                            model=model,
                            provider=self.name,
                            finish_reason="",
                            usage={},
                        )
                    continue

                if dtype == "input_json_delta":
                    slot = tool_blocks.get(idx)
                    if slot is not None:
                        partial = delta.get("partial_json") or ""
                        if partial:
                            slot["arguments"] += partial
                    continue

                # thinking_delta / signature_delta / others: ignore for now.
                continue

            if etype == "content_block_stop":
                continue

            if etype == "message_delta":
                d = event.get("delta") or {}
                if d.get("stop_reason"):
                    stop_reason = _STOP_REASON_MAP.get(
                        d["stop_reason"], FinishReason.STOP.value
                    )
                if isinstance(event.get("usage"), dict):
                    usage = _merge_usage(usage, event["usage"])
                continue

            if etype == "message_stop":
                break

            if etype == "error":
                err = event.get("error") or {}
                raise _error_object_to_error(err, model, self.name)

            # ping / unknown events are ignored.

        # Emit assembled tool calls as the final chunk.
        calls: List[ToolCall] = []
        for _, slot in sorted(tool_blocks.items()):
            args, parse_error = self._coerce_args(slot.get("arguments"))
            call = ToolCall(id=slot["id"], name=slot["name"], arguments=args)
            if parse_error:
                try:
                    setattr(call, "parse_error", parse_error)
                except Exception:
                    args.setdefault("_parse_error", parse_error)
            calls.append(call)

        if calls:
            stop_reason = FinishReason.TOOL_CALLS.value

        yield LLMResponse(
            content="",
            model=model,
            provider=self.name,
            tool_calls=calls,
            finish_reason=stop_reason,
            usage=usage,
            raw={"streamed": True, "had_text": text_seen},
        )

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_models(self) -> List[ModelInfo]:
        """
        Static list — Anthropic's /v1/models endpoint requires a live call,
        so the sync method returns empty. Use `discover_models()` for the
        live catalog.
        """
        return []

    async def discover_models(self) -> List[ModelInfo]:
        """
        GET /v1/models and map to ModelInfo.

        Returns an empty list on any failure — model discovery is a
        convenience, not a hard requirement.
        """
        if not self.config.api_key:
            return []
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()

        try:
            resp = await client.get("/v1/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Anthropic model discovery failed: %s", exc)
            return []

        raw = data.get("data") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []

        out: List[ModelInfo] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id")
            if not mid:
                continue
            out.append(ModelInfo(
                id=str(mid),
                provider=self.name,
                name=str(entry.get("display_name") or mid),
                description=str(entry.get("description") or ""),
                context_window=int(entry.get("context_window") or 200_000),
                max_output=int(entry.get("max_output_tokens") or 8_192),
                supports_tools=True,
                supports_streaming=True,
                supports_vision=True,
                supports_json_mode=False,
            ))
        return out

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Cheap reachability probe. Anthropic has no /health; /v1/models works."""
        if not self.is_configured():
            return False
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()
        try:
            resp = await client.get("/v1/models")
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
                    "Anthropic %s attempt %d/%d: %s (retrying in %.1fs)",
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
                raise ProviderError(
                    f"Anthropic returned non-JSON body: {exc}",
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
    def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
        if response is not None:
            ra = response.headers.get("retry-after")
            if ra:
                try:
                    v = float(ra)
                    if 0 < v <= 30:
                        return v
                except Exception:
                    pass
        base = min(2 ** attempt, 8)
        return base + random.uniform(0, 0.25)

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse(self, data: Dict[str, Any], model: str) -> LLMResponse:
        """Convert an Anthropic Messages response into our LLMResponse."""
        content_blocks = data.get("content") or []
        text_parts: List[str] = []
        calls: List[ToolCall] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text") or ""))
            elif btype == "tool_use":
                cid = block.get("id") or f"toolu_{uuid.uuid4().hex[:12]}"
                name = str(block.get("name") or "")
                raw_input = block.get("input")
                args, parse_error = self._coerce_args(raw_input)
                call = ToolCall(id=cid, name=name, arguments=args)
                if parse_error:
                    try:
                        setattr(call, "parse_error", parse_error)
                    except Exception:
                        args.setdefault("_parse_error", parse_error)
                calls.append(call)
            # thinking / other block types: ignored for now.

        usage_raw = data.get("usage") or {}
        usage = _normalize_usage(usage_raw)

        stop_reason_raw = data.get("stop_reason") or "end_turn"
        finish_reason = _STOP_REASON_MAP.get(stop_reason_raw, FinishReason.STOP.value)
        if calls and finish_reason != FinishReason.TOOL_CALLS.value:
            finish_reason = FinishReason.TOOL_CALLS.value

        return LLMResponse(
            content="".join(text_parts),
            model=str(data.get("model") or model),
            provider=self.name,
            usage=usage,
            tool_calls=calls,
            finish_reason=finish_reason,
            raw=data,
        )

    @staticmethod
    def _coerce_args(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Anthropic hands tool inputs back as a parsed JSON object when
        non-streaming, and as an assembled JSON string when streaming.

        Return (args_dict, parse_error). args_dict is always a dict.
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
                return {}, f"invalid JSON tool input ({e.msg} at pos {e.pos})"
            if isinstance(parsed, dict):
                return parsed, None
            return {}, f"tool input parsed as {type(parsed).__name__}, expected object"
        return {}, f"tool input was {type(raw).__name__}, not object or JSON string"


# ======================================================================
# HELPERS
# ======================================================================

def _coerce_message(m: Any) -> Dict[str, Any]:
    """Turn whatever the caller passed into a plain dict we can inspect."""
    if isinstance(m, Message):
        return m.to_dict()
    if isinstance(m, dict):
        return dict(m)
    # Duck typing.
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
    """Coerce content to a string, preserving dicts/lists as JSON."""
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


def _normalize_content_in(content: Any) -> Any:
    """
    Normalize message content for the outgoing user turn.

    Anthropic accepts either a string or a list of blocks. We pass a list
    of text blocks through untouched, coerce strings to themselves, and
    JSON-encode anything else so no content is silently lost.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Assume already block-shaped; pass through.
        return content
    # Numbers, dicts, etc. — put them behind a text block.
    return [{"type": "text", "text": _stringify_content(content)}]


def _assistant_to_blocks(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert an assistant message into Anthropic content blocks.

    Text first (if any), then a tool_use block per requested call.
    """
    blocks: List[Dict[str, Any]] = []

    text = m.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for b in text:
            if isinstance(b, dict) and b.get("type"):
                blocks.append(b)
            else:
                s = _stringify_content(b)
                if s:
                    blocks.append({"type": "text", "text": s})

    calls = m.get("tool_calls") or []
    for tc in calls:
        if isinstance(tc, ToolCall):
            blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments or {},
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
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:12]}",
            "name": str(name),
            "input": args,
        })

    return blocksdef _merge_usage(existing: Dict[str, int], incoming: Dict[str, Any]) -> Dict[str, int]:
    """Anthropic reports usage twice (message_start and message_delta)."""
    out = dict(existing or {})
    for k, v in _normalize_usage(incoming).items():
        if v:
            out[k] = v
    return out


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """Flatten Anthropic's token counters into our dict shape."""
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cached_tokens": cache_read,
        "cache_creation_tokens": cache_creation,
        "reasoning_tokens": 0,
    }


def _status_to_error(
    status: int, body_text: str, model: str, provider: str
) -> ProviderError:
    """
    Map an HTTP status + body to a ProviderError.

    Anthropic bodies look like:
        {"type":"error","error":{"type":"invalid_request_error","message":"..."}}
    so we try the structured shape first and fall back to status.
    """
    err_obj: Dict[str, Any] = {}
    try:
        parsed = json.loads(body_text) if body_text else {}
        if isinstance(parsed, dict):
            err_obj = parsed.get("error") if isinstance(parsed.get("error"), dict) else {}
            if not err_obj and parsed.get("type") == "error":
                err_obj = parsed
    except Exception:
        err_obj = {}

    if err_obj:
        return _error_object_to_error(err_obj, model, provider, status=status)

    message = body_text[:400] or f"HTTP {status}"
    if status == 429:
        return ProviderError(
            f"Rate limited: {message}",
            code=ProviderErrorCode.RATE_LIMIT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status in (500, 502, 503, 504, 529):
        return ProviderError(
            f"Anthropic server error ({status}): {message}",
            code=ProviderErrorCode.SERVER,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 408:
        return ProviderError(
            f"Request timed out: {message}",
            code=ProviderErrorCode.TIMEOUT,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status in (401, 403):
        return ProviderError(
            f"Authentication failed ({status}): {message}",
            code=ProviderErrorCode.AUTH,
            provider=provider, model=model, status=status,
        )
    if status == 404:
        return ProviderError(
            f"Not found ({status}): {message}",
            code=ProviderErrorCode.MODEL_NOT_FOUND,
            provider=provider, model=model, status=status,
        )
    if status in (400, 422):
        lowered = message.lower()
        code = (ProviderErrorCode.CONTEXT_LENGTH
                if "prompt is too long" in lowered or "context" in lowered and "length" in lowered
                else ProviderErrorCode.BAD_REQUEST)
        return ProviderError(
            f"Bad request ({status}): {message}",
            code=code, provider=provider, model=model, status=status,
        )
    return ProviderError(
        f"Request failed ({status}): {message}",
        code=ProviderErrorCode.UNKNOWN,
        provider=provider, model=model, status=status,
    )


def _error_object_to_error(
    err: Dict[str, Any],
    model: str,
    provider: str,
    status: Optional[int] = None,
) -> ProviderError:
    """Build a ProviderError from Anthropic's structured error object."""
    etype = str(err.get("type") or "")
    message = str(err.get("message") or etype or "Anthropic error")
    code = _ERROR_TYPE_MAP.get(etype, ProviderErrorCode.UNKNOWN)
    retryable = code in (ProviderErrorCode.RATE_LIMIT, ProviderErrorCode.SERVER,
                         ProviderErrorCode.TIMEOUT)
    return ProviderError(
        message,
        code=code,
        provider=provider, model=model,
        status=status, retryable=retryable,
    )


__all__ = ["AnthropicProvider"]