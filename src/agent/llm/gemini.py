"""Native Google Gemini API adapter.

Talks directly to the Gemini `generateContent` / `streamGenerateContent`
endpoints — not through an OpenAI-compatible shim — so Gemini's distinct
shapes stay intact and get converted into the framework's internal
format at the boundary.

    Framework type           Gemini wire shape
    ────────────────         ──────────────────────────────────────────
    Message(role=system)     {"system_instruction": {"parts":[...]}}
    Message(role=user)       {"role":"user",  "parts":[{"text":...}]}
    Message(role=assistant)  {"role":"model", "parts":[{"text":...} |
                                                     {"functionCall":...}]}
    Message(role=tool)       {"role":"user",  "parts":[{"functionResponse":
                                                     {"name":..,"response":..}}]}
    ToolSpec                 {"function_declarations":[{"name","description",
                                                        "parameters":..}]}
    ToolCall                 part: {"functionCall":{"name":..,"args":{...}}}
    LLMResponse              candidate.content.parts + usageMetadata

What this file owns
-------------------
    * Message conversion       — framework → Gemini's parts-based content
    * Tool conversion          — ToolSpec → function_declarations
    * Response conversion      — Gemini candidates → LLMResponse
    * Streaming                — SSE-style, JSON array, or NDJSON
    * Errors                   — Google error JSON mapped to ProviderError
    * Retries                  — on 429/5xx, honoring Retry-After
    * Lifecycle                — one long-lived httpx.AsyncClient
    * Model discovery          — GET /v1beta/models

What this file does NOT own
---------------------------
    * Model selection / fallback chains  — agent.llm.runtime
    * Cost tables                        — registry metadata
    * Tool execution                     — the loop dispatches

Gemini API specifics worth knowing
----------------------------------
    * Auth is the `x-goog-api-key` header, not `Authorization: Bearer`.
    * The assistant role is called `"model"`, not `"assistant"`.
    * Content is a list of `parts`. A part is one of:
        {"text": "..."}                             — plain text
        {"functionCall": {"name":..,"args":{...}}}  — tool call
        {"functionResponse":{"name":..,"response":..}} — tool result
        {"inlineData": {...}}                       — images/files
    * System instructions are a top-level `system_instruction` field, not
      a message. It's shaped like a Content but has no role.
    * Tool results go in a **user** turn as `functionResponse` parts. The
      Gemini API has no dedicated "tool" role.
    * Function calls do NOT carry an id. Pairing is by name and by order.
      We synthesize an id for the framework and map it back at send time
      by matching on the function `name`.
    * `functionResponse.response` must be a JSON object (`{"result":..}`),
      not a bare string.
    * Streaming returns a JSON array of chunks when `alt=sse` is NOT set,
      and SSE when it is. We request SSE (`?alt=sse`) so the stream is
      line-oriented and cancellable.
    * Stop reasons live under `candidates[0].finishReason` as SCREAMING
      SNAKE constants: STOP, MAX_TOKENS, SAFETY, RECITATION, etc.
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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

# The current stable API version. Bump when Google promotes a new one.
API_VERSION = "v1beta"

# Statuses worth retrying. 400/401/403/404 are not.
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Gemini's finishReason → our FinishReason.
_FINISH_REASON_MAP = {
    "STOP": FinishReason.STOP.value,
    "MAX_TOKENS": FinishReason.LENGTH.value,
    "SAFETY": FinishReason.CONTENT_FILTER.value,
    "RECITATION": FinishReason.CONTENT_FILTER.value,
    "BLOCKLIST": FinishReason.CONTENT_FILTER.value,
    "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER.value,
    "SPII": FinishReason.CONTENT_FILTER.value,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR.value,
    "OTHER": FinishReason.STOP.value,
}

# Google's error `status` string → our ProviderErrorCode.
_ERROR_STATUS_MAP = {
    "INVALID_ARGUMENT": ProviderErrorCode.BAD_REQUEST,
    "FAILED_PRECONDITION": ProviderErrorCode.BAD_REQUEST,
    "OUT_OF_RANGE": ProviderErrorCode.BAD_REQUEST,
    "UNAUTHENTICATED": ProviderErrorCode.AUTH,
    "PERMISSION_DENIED": ProviderErrorCode.AUTH,
    "NOT_FOUND": ProviderErrorCode.MODEL_NOT_FOUND,
    "RESOURCE_EXHAUSTED": ProviderErrorCode.RATE_LIMIT,
    "INTERNAL": ProviderErrorCode.SERVER,
    "UNAVAILABLE": ProviderErrorCode.SERVER,
    "DEADLINE_EXCEEDED": ProviderErrorCode.TIMEOUT,
    "ABORTED": ProviderErrorCode.CANCELLED,
    "CANCELLED": ProviderErrorCode.CANCELLED,
}

# Optional generationConfig fields we forward when supplied.
_GENERATION_CONFIG_KEYS = (
    "topP",
    "topK",
    "candidateCount",
    "maxOutputTokens",
    "stopSequences",
    "presencePenalty",
    "frequencyPenalty",
    "responseMimeType",
    "responseSchema",
    "seed",
)

# Safety categories we accept filters for. Gemini rejects unknown names.
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)


# ======================================================================
# PROVIDER
# ======================================================================

class GeminiProvider(LLMProvider):
    """
    Adapter for the Google Gemini API.

    Configure with an API key and, optionally, a default model:

        GeminiProvider(ProviderConfig(
            api_key="AIza...",
            default_model="gemini-2.0-flash",
        ))

    `base_url` defaults to Google's public endpoint. `API_VERSION` is a
    module constant so it can be overridden without monkey-patching.

    Model IDs to use directly:
        gemini-2.0-flash
        gemini-2.0-flash-lite
        gemini-1.5-pro
        gemini-1.5-flash
        gemini-1.5-flash-8b
    """

    name = "gemini"

    #: Overridable so tests can pin a version.
    api_version: str = API_VERSION

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
        }
        if self.config.api_key:
            headers["x-goog-api-key"] = self.config.api_key

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
            "vision": True,           # inlineData parts
            "json_mode": True,        # responseMimeType: "application/json"
            "system_prompt": True,
            "parallel_tool_calls": True,
        }

    # ------------------------------------------------------------------
    # MESSAGE CONVERSION (framework → Gemini)
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: Sequence[Any]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Split incoming messages into (system_instruction, contents).

        Rules:
            * Every role="system" is joined into a single top-level
              `system_instruction` object. It is removed from `contents`.
            * role="user" or "tool" → contents entry with role="user".
            * role="assistant"      → contents entry with role="model".
            * role="tool" becomes a user turn with a `functionResponse`
              part. We must know the function's *name*; if the caller
              provided it we use it, else we look it up in a small map
              we maintain across the assistant turn.
            * Adjacent same-role turns are merged, since Gemini rejects
              consecutive turns with the same role.
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        # Maps a synthesized tool_call_id → function name, so a role="tool"
        # message can find its function name without a schema change.
        call_id_to_name: Dict[str, str] = {}

        for raw in messages or []:
            m = _coerce_message(raw)
            role = (m.get("role") or "user").lower()

            if role == "system":
                text = _stringify_content(m.get("content"))
                if text:
                    system_parts.append(text)
                continue

            if role == "user":
                entry = {
                    "role": "user",
                    "parts": _user_parts_in(m.get("content")),
                }
                _append_or_merge(contents, entry)
                continue

            if role == "assistant":
                parts: List[Dict[str, Any]] = []

                text = m.get("content")
                if text:
                    parts.append({"text": _stringify_content(text)})

                for tc in (m.get("tool_calls") or []):
                    call = _coerce_tool_call(tc)
                    if call is None:
                        continue
                    call_id_to_name[call.id] = call.name
                    parts.append({
                        "functionCall": {
                            "name": call.name,
                            "args": call.arguments or {},
                        },
                    })

                if not parts:
                    parts = [{"text": ""}]
                _append_or_merge(contents, {"role": "model", "parts": parts})
                continue

            if role == "tool":
                name = (
                    m.get("name")
                    or call_id_to_name.get(m.get("tool_call_id") or "", "")
                    or ""
                )
                # Gemini wants the response as a JSON object.
                response_obj = _to_response_object(m.get("content"))
                part = {
                    "functionResponse": {
                        "name": name,
                        "response": response_obj,
                    },
                }
                entry = {"role": "user", "parts": [part]}
                _append_or_merge(contents, entry)
                continue

            # Unknown role → treat as user to avoid dropping content.
            _append_or_merge(contents, {
                "role": "user",
                "parts": _user_parts_in(m.get("content")),
            })

        # Gemini needs the conversation to start with a user turn.
        if contents and contents[0]["role"] != "user":
            contents.insert(0, {"role": "user", "parts": [{"text": ""}]})

        system_instruction: Dict[str, Any] = {}
        if system_parts:
            system_instruction = {
                "parts": [{"text": "\n\n".join(system_parts)}],
            }

        return system_instruction, contents

    # ------------------------------------------------------------------
    # TOOL CONVERSION (framework → Gemini)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(
        tools: Optional[Sequence[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Gemini expects a single wrapper with a `function_declarations` list:

            [{"function_declarations": [{"name":..., "description":...,
                                          "parameters": {...}}]}]
        """
        if not tools:
            return []
        decls: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, ToolSpec):
                decls.append(_spec_to_declaration(t))
                continue
            if isinstance(t, dict):
                # Already Gemini-shaped?
                if "name" in t and ("parameters" in t or "description" in t) \
                        and "function" not in t:
                    decls.append(_sanitize_declaration(t))
                    continue
                # OpenAI-shaped {"type":"function","function":{...}}.
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                name = fn.get("name")
                if not name:
                    continue
                decls.append(_sanitize_declaration({
                    "name": name,
                    "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters")
                                  or fn.get("input_schema")
                                  or {"type": "object", "properties": {}},
                }))
                continue
            raise ProviderError(
                f"Unsupported tool spec type: {type(t).__name__}",
                code=ProviderErrorCode.UNSUPPORTED,
                provider="gemini",
            )
        return [{"function_declarations": decls}] if decls else []

    # ------------------------------------------------------------------
    # REQUEST BODY
    # ------------------------------------------------------------------

    def _build_body(
        self,
        messages: Sequence[Any],
        temperature: float,
        max_tokens: int,
        tools: Optional[Sequence[Any]],
        tool_choice: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        system_instruction, contents = self._convert_messages(messages)

        body: Dict[str, Any] = {"contents": contents}

        if system_instruction:
            body["system_instruction"] = system_instruction

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": int(max_tokens) if max_tokens and max_tokens > 0 else 4096,
        }
        for key in _GENERATION_CONFIG_KEYS:
            v = kwargs.get(key)
            if v is not None and key not in ("temperature", "maxOutputTokens"):
                generation_config[key] = v
        body["generationConfig"] = generation_config

        gemini_tools = self._convert_tools(tools)
        if gemini_tools:
            body["tools"] = gemini_tools
            # Gemini's tool_config controls mode:
            #   AUTO     — model decides (default)
            #   ANY      — must call a tool
            #   NONE     — must not call a tool
            mode = {
                "required": "ANY",
                "any": "ANY",
                "none": "NONE",
                "auto": "AUTO",
            }.get((tool_choice or "auto").lower())
            if mode and mode != "AUTO":
                body["tool_config"] = {"function_calling_config": {"mode": mode}}

        safety = kwargs.get("safety_settings")
        if isinstance(safety, list):
            body["safetySettings"] = safety

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
                "Gemini API key not configured",
                code=ProviderErrorCode.AUTH,
                provider=self.name,
            )
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=self.name,
            )

        if stop:
            kwargs.setdefault("stopSequences", list(stop))

        client = self._require_client()
        body = self._build_body(
            messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
            **kwargs,
        )

        path = f"/{self.api_version}/models/{model}:generateContent"
        data = await self._post_json(client, path, body, model)
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

        We request `?alt=sse` so the response is line-oriented
        Server-Sent Events. Each `data:` line is one chunk of the same
        shape as a non-streaming response — text deltas arrive as
        `candidates[0].content.parts[*].text`, and function calls arrive
        whole on the chunk that carries them.
        """
        if not self.config.api_key:
            raise ProviderError(
                "Gemini API key not configured",
                code=ProviderErrorCode.AUTH,
                provider=self.name,
            )
        model = model or self.config.default_model
        if not model:
            raise ProviderError(
                "Model not specified and no default configured",
                code=ProviderErrorCode.MODEL_NOT_FOUND,
                provider=self.name,
            )

        if stop:
            kwargs.setdefault("stopSequences", list(stop))

        client = self._require_client()
        body = self._build_body(
            messages=messages,
            temperature=temperature, max_tokens=max_tokens,
            tools=tools, tool_choice=tool_choice,
            **kwargs,
        )

        path = f"/{self.api_version}/models/{model}:streamGenerateContent"
        params = {"alt": "sse"}

        tool_calls_accum: List[ToolCall] = []
        finish_reason = FinishReason.STOP.value
        usage: Dict[str, int] = {}

        try:
            async with client.stream(
                "POST", path, json=body, params=params
            ) as resp:
                if resp.status_code >= 400:
                    raw = await resp.aread()
                    raise _status_to_error(
                        resp.status_code,
                        raw.decode("utf-8", "replace"),
                        model, self.name,
                    )

                content_type = (resp.headers.get("content-type") or "").lower()

                if "text/event-stream" not in content_type:
                    # The endpoint returned a JSON array instead. Read it,
                    # split it, and yield the chunks ourselves.
                    raw = await resp.aread()
                    for chunk in _split_json_array(raw.decode("utf-8", "replace")):
                        fr, tc, txt, us = self._parse_one(chunk, model)
                        if txt:
                            yield LLMResponse(
                                content=txt, model=model, provider=self.name,
                                finish_reason="", usage={},
                            )
                        tool_calls_accum.extend(tc)
                        if fr and fr != FinishReason.STOP.value:
                            finish_reason = fr
                        if us:
                            usage.update(us)
                else:
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        fr, tc, txt, us = self._parse_one(chunk, model)
                        if txt:
                            yield LLMResponse(
                                content=txt, model=model, provider=self.name,
                                finish_reason="", usage={},
                            )
                        tool_calls_accum.extend(tc)
                        if fr and fr != FinishReason.STOP.value:
                            finish_reason = fr
                        if us:
                            usage.update(us)

        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Gemini stream timed out after {self.config.timeout}s",
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

        if tool_calls_accum:
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
        GET /v1beta/models → the catalog visible to this API key.

        Filtered to entries that support `generateContent`, which is the
        ones we can actually use. `streamGenerateContent` support is
        inferred from `supportedGenerationMethods`.
        """
        if not self.config.api_key:
            return []
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()

        try:
            resp = await client.get(f"/{self.api_version}/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Gemini model discovery failed: %s", exc)
            return []

        raw = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []

        out: List[ModelInfo] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            methods = entry.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue

            # Model names come back as "models/gemini-2.0-flash"; the API
            # call path already includes "models/", so strip the prefix.
            full_name = str(entry.get("name") or "")
            mid = full_name.split("/", 1)[1] if "/" in full_name else full_name
            if not mid:
                continue

            out.append(ModelInfo(
                id=mid,
                provider=self.name,
                name=str(entry.get("displayName") or mid),
                description=str(entry.get("description") or ""),
                context_window=int(entry.get("inputTokenLimit") or 0),
                max_output=int(entry.get("outputTokenLimit") or 0),
                supports_tools=True,
                supports_streaming="streamGenerateContent" in methods,
                supports_vision=True,          # all current models accept images
                supports_json_mode=True,
            ))
        return out

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """Cheap reachability probe. /models requires auth and responds fast."""
        if not self.is_configured():
            return False
        try:
            client = self._require_client()
        except ProviderError:
            await self.start()
            client = self._require_client()
        try:
            resp = await client.get(f"/{self.api_version}/models")
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
                    "Gemini %s attempt %d/%d: %s (retrying in %.1fs)",
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
                    f"Gemini returned non-JSON body: {exc}",
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
        """Convert one Gemini response into our LLMResponse."""
        finish_reason, tool_calls, text, usage = self._parse_one(data, model)
        return LLMResponse(
            content=text,
            model=str(data.get("modelVersion") or model),
            provider=self.name,
            usage=usage,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=data,
        )

    def _parse_one(
        self, data: Dict[str, Any], model: str
    ) -> Tuple[str, List[ToolCall], str, Dict[str, int]]:
        """
        Extract (finish_reason, tool_calls, text, usage) from one chunk.

        Kept as a helper so the streaming path and the non-streaming path
        share exactly the same conversion logic.
        """
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(candidates, list) or not candidates:
            # Possible prompt-level block.
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                return (
                    FinishReason.CONTENT_FILTER.value, [], "",
                    _normalize_usage(data.get("usageMetadata") or {}),
                )
            return FinishReason.STOP.value, [], "", _normalize_usage({})

        candidate = candidates[0] or {}
        finish_raw = candidate.get("finishReason") or ""
        finish_reason = _FINISH_REASON_MAP.get(finish_raw, FinishReason.STOP.value)

        content = candidate.get("content") or {}
        parts = content.get("parts") if isinstance(content, dict) else None

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for part in parts or []:
            if not isinstance(part, dict):
                continue
            if "text" in part and part["text"]:
                text_parts.append(str(part["text"]))
            elif "functionCall" in part and isinstance(part["functionCall"], dict):
                fc = part["functionCall"]
                name = fc.get("name")
                if not name:
                    continue
                args = fc.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                tool_calls.append(ToolCall(
                    id=f"gemini_{uuid.uuid4().hex[:12]}",
                    name=str(name),
                    arguments=args,
                ))

        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS.value

        usage = _normalize_usage(data.get("usageMetadata") or {})

        return finish_reason, tool_calls, "".join(text_parts), usage


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


def _user_parts_in(content: Any) -> List[Dict[str, Any]]:
    """
    Build the parts list for a user turn.

    Accepts:
        str          → [{"text": str}]
        list of parts → passed through (must already be part-shaped)
        dict         → wrapped: if it's a {"text":...} or {"inlineData":...}
                       part we accept it, else JSON-encode as text.
        other        → [{"text": str(value)}]
    """
    if content is None:
        return [{"text": ""}]
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        out: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict) and ("text" in item or "inlineData" in item):
                out.append(item)
            else:
                out.append({"text": _stringify_content(item)})
        return out or [{"text": ""}]
    if isinstance(content, dict):
        if "text" in content or "inlineData" in content:
            return [content]
        return [{"text": _stringify_content(content)}]
    return [{"text": str(content)}]


def _coerce_tool_call(tc: Any) -> Optional[ToolCall]:
    """Coerce anything tool-call-shaped into a ToolCall."""
    if isinstance(tc, ToolCall):
        return tc
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    name = fn.get("name")
    if not name:
        return None
    raw_args = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        except Exception:
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    return ToolCall(
        id=tc.get("id") or f"gemini_{uuid.uuid4().hex[:12]}",
        name=str(name),
        arguments=raw_args,
    )


def _to_response_object(content: Any) -> Dict[str, Any]:
    """
    Gemini requires functionResponse.response to be a JSON object. If the
    caller passed a string (a JSON blob, a plain message, or the raw tool
    output), wrap or parse it so the wire shape is always an object.
    """
    if content is None:
        return {"result": None}
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        s = content.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"result": content}
    # Numbers, lists, booleans.
    return {"result": content}


def _append_or_merge(contents: List[Dict[str, Any]], entry: Dict[str, Any]) -> None:
    """
    Gemini rejects consecutive turns with the same role. If the previous
    entry has the same role, append the new parts to it.
    """
    if contents and contents[-1]["role"] == entry["role"]:
        contents[-1]["parts"].extend(entry["parts"])
        return
    contents.append(entry)


def _spec_to_declaration(spec: ToolSpec) -> Dict[str, Any]:
    return _sanitize_declaration({
        "name": spec.name,
        "description": spec.description or "",
        "parameters": spec.parameters or {"type": "object", "properties": {}},
    })


def _sanitize_declaration(decl: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gemini's schema dialect is an OpenAPI 3 subset. It rejects some JSON
    Schema keys that OpenAI accepts. Strip the common offenders so a tool
    that works with OpenAI doesn't break on Gemini.
    """
    params = decl.get("parameters")
    if isinstance(params, dict):
        decl = dict(decl)
        decl["parameters"] = _strip_unsupported_schema_keys(params)
    # Gemini rejects empty-string descriptions outright.
    if decl.get("description") == "":
        decl.pop("description", None)
    return decl


# Keys Gemini's schema validator refuses. Keeping it explicit means we
# don't accidentally strip a field Google later adds support for.
_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "additionalProperties",
    "definitions",
    "$defs",
    "examples",
    "const",
    "if",
    "then",
    "else",
    "not",
    "patternProperties",
    "propertyNames",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "prefixItems",
}


def _strip_unsupported_schema_keys(node: Any) -> Any:
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k in _UNSUPPORTED_SCHEMA_KEYS:
                continue
            out[k] = _strip_unsupported_schema_keys(v)
        return out
    if isinstance(node, list):
        return [_strip_unsupported_schema_keys(x) for x in node]
    return node


def _normalize_usage(usage: Dict[str, Any]) -> Dict[str, int]:
    """Flatten Gemini's usageMetadata into our token dict shape."""
    prompt = int(usage.get("promptTokenCount", 0) or 0)
    completion = int(usage.get("candidatesTokenCount", 0) or 0)
    total = int(usage.get("totalTokenCount", 0) or 0)
    if not total:
        total = prompt + completion

    cached = int(usage.get("cachedContentTokenCount", 0) or 0)
    # Gemini's "thoughtsTokenCount" is the closest analogue to reasoning
    # tokens. Keep the name consistent with the other providers.
    reasoning = int(usage.get("thoughtsTokenCount", 0) or 0)

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
    }


def _split_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Streaming without `alt=sse` returns a JSON array: `[{...},{...},...]`.

    A plain `json.loads` would need the whole body. We can't always get
    it (the body might be truncated on cancel), so fall back to a
    best-effort bracket walker when the simple parse fails.
    """
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
            return [c for c in data if isinstance(c, dict)]
        except Exception:
            return _walk_top_level_objects(text)
    # Sometimes the body is one object per line.
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _walk_top_level_objects(text: str) -> List[Dict[str, Any]]:
    """Extract every top-level {...} from a JSON-array-shaped body."""
    out: List[Dict[str, Any]] = []
    depth = 0
    in_str = False
    escape = False
    start = -1
    for i, c in enumerate(text):
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
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start != -1:
                chunk = text[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        out.append(obj)
                except Exception:
                    pass
                start = -1
    return out


def _status_to_error(
    status: int, body_text: str, model: str, provider: str
) -> ProviderError:
    """
    Google error bodies look like:
        {"error": {"code": 400, "message": "...", "status": "INVALID_ARGUMENT"}}
    We prefer the structured `status` string; fall back to the HTTP code.
    """
    err_obj: Dict[str, Any] = {}
    try:
        parsed = json.loads(body_text) if body_text else {}
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            err_obj = parsed["error"]
    except Exception:
        err_obj = {}

    if err_obj:
        return _error_object_to_error(err_obj, model, provider)

    message = body_text[:400] or f"HTTP {status}"
    if status in (500, 502, 503, 504):
        return ProviderError(
            f"Gemini server error ({status}): {message}",
            code=ProviderErrorCode.SERVER,
            provider=provider, model=model, status=status, retryable=True,
        )
    if status == 429:
        return ProviderError(
            f"Rate limited: {message}",
            code=ProviderErrorCode.RATE_LIMIT,
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
                if "token" in lowered and ("exceed" in lowered or "limit" in lowered)
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
    err: Dict[str, Any], model: str, provider: str
) -> ProviderError:
    """Build a ProviderError from Google's structured error object."""
    status_str = str(err.get("status") or "")
    message = str(err.get("message") or status_str or "Gemini error")
    code = _ERROR_STATUS_MAP.get(status_str, ProviderErrorCode.UNKNOWN)
    retryable = code in (ProviderErrorCode.RATE_LIMIT, ProviderErrorCode.SERVER,
                         ProviderErrorCode.TIMEOUT)
    # Context-length failures come through as INVALID_ARGUMENT with a
    # specific message rather than a distinct status.
    if code == ProviderErrorCode.BAD_REQUEST:
        lowered = message.lower()
        if "token" in lowered and ("exceed" in lowered or "limit" in lowered):
            code = ProviderErrorCode.CONTEXT_LENGTH
    return ProviderError(
        message,
        code=code,
        provider=provider, model=model,
        retryable=retryable,
        status=int(err.get("code") or 0) or None,
    )


__all__ = ["GeminiProvider"]