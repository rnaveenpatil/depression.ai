"""Generic OpenAI-compatible provider adapter.

Supports providers exposing POST /chat/completions with standard tool calls.
This lets users supply a provider/model/base URL/API key without adding a new
hard-coded provider class for every compatible gateway.

Tool call argument policy:
    * If the provider returns `arguments` as a JSON string, it is parsed.
    * Malformed JSON does NOT become {"_raw": ...}. Instead the ToolCall
      carries an empty arguments dict plus a `parse_error` field. The
      runtime's ToolRegistry validator will reject it and the loop will
      ask the model to re-emit the call with valid JSON.
    * Tool calls missing a name are dropped (they cannot be dispatched).
    * Missing ids are synthesized so the tool result can always be paired.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx

from agent.llm.provider import LLMProvider, LLMResponse, Message, ToolCall
from agent.utils.errors import LLMError
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# HTTP statuses that are worth retrying.
_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Fields on Message.to_dict() we pass through. Anything not listed is
# dropped to keep the request shape tight.
_ALLOWED_MESSAGE_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls"}


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    # ------------------------------------------------------------------
    # Message serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_message(m: Any) -> Dict[str, Any]:
        if isinstance(m, Message):
            d = m.to_dict()
        elif isinstance(m, dict):
            d = dict(m)
        else:
            # Best-effort duck typing for callers that pass a foreign type.
            d = {
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

        out: Dict[str, Any] = {}
        for k in _ALLOWED_MESSAGE_KEYS:
            if k not in d:
                continue
            v = d[k]
            # The OpenAI spec requires content=null (not "") when tool_calls
            # are present on an assistant message.
            if k == "content":
                if v == "" and d.get("tool_calls"):
                    out[k] = None
                elif v == "" and d.get("role") == "assistant" and not d.get("tool_calls"):
                    # Some gateways accept "", others prefer null. Keep ""
                    # for a plain assistant turn.
                    out[k] = ""
                else:
                    out[k] = v
            else:
                out[k] = v

        # Drop None-valued optional keys that some strict gateways reject.
        for k in ("name", "tool_call_id"):
            if out.get(k) is None:
                out.pop(k, None)

        return out

    # ------------------------------------------------------------------
    # Tool call parsing (the important part)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(raw_calls: List[Any]) -> List[ToolCall]:
        """
        Convert provider tool_calls into ToolCall objects.

        Never returns arguments={"_raw": ...}. Malformed JSON produces
        arguments={} plus a `parse_error` attribute the loop reads back.
        """
        out: List[ToolCall] = []
        for idx, tc in enumerate(raw_calls or []):
            if not isinstance(tc, dict):
                logger.warning("Skipping non-dict tool_call: %r", tc)
                continue

            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            if not name:
                logger.warning(
                    "Skipping tool_call with no name (id=%r)", tc.get("id")
                )
                continue

            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            if not tc.get("id"):
                logger.debug("Tool call missing id; synthesized %s", call_id)

            raw_args = fn.get("arguments", tc.get("arguments"))
            args, parse_error = OpenAICompatibleProvider._coerce_args(raw_args)

            call = ToolCall(id=call_id, name=name, arguments=args)
            # Attach parse_error as a plain attribute; ToolCall is a dataclass
            # so we can setattr freely. If a future version freezes it, we
            # fall back to embedding the error inside arguments.
            if parse_error:
                try:
                    setattr(call, "parse_error", parse_error)
                except Exception:
                    args.setdefault("_parse_error", parse_error)
            out.append(call)
        return out

    @staticmethod
    def _coerce_args(raw: Any) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Return (args_dict, parse_error). args_dict is always a dict.
        parse_error is a short human-readable string or None.
        """
        if raw is None or raw == "":
            return {}, None
        if isinstance(raw, dict):
            return raw, None
        if not isinstance(raw, str):
            # Provider sent an int/list/etc. Wrap it so the caller can see
            # the shape, but flag it as invalid for the schema.
            return {}, f"tool arguments were {type(raw).__name__}, not JSON string or object"

        # Trim and try to parse. Some gateways pad with whitespace or
        # append a trailing newline.
        s = raw.strip()
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError as e:
            # Truncated JSON is the most common failure. Give the model a
            # hint about how much was lost.
            return {}, f"invalid JSON arguments ({e.msg} at pos {e.pos})"

        if isinstance(parsed, dict):
            return parsed, None
        # Valid JSON but not an object: wrap it so downstream validation
        # reports a clear type error.
        return {}, f"tool arguments parsed as {type(parsed).__name__}, expected object"

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError("API key not configured")
        if not self.base_url:
            raise LLMError("Base URL not configured")
        if not model:
            raise LLMError("Model not configured")

        normalized = [self._normalize_message(m) for m in messages]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": normalized,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        for key in ("response_format", "top_p", "stop", "presence_penalty",
                    "frequency_penalty", "seed", "user"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last: Optional[Exception] = None
        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post("/chat/completions", json=payload)
                    response.raise_for_status()
                    return self._parse(response.json(), model)

                except httpx.HTTPStatusError as exc:
                    last = exc
                    status = exc.response.status_code

                    if status in _RETRYABLE_STATUSES:
                        delay = self._retry_delay(attempt, exc.response)
                        logger.warning(
                            "LLM %s attempt %d/%d: HTTP %d, retrying in %.1fs",
                            model, attempt + 1, self.max_retries, status, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    if status == 401:
                        raise LLMError(
                            "Authentication failed — check the API key "
                            f"for model '{model}'"
                        )
                    if status == 403:
                        raise LLMError(
                            f"Access forbidden for model '{model}' (403)"
                        )
                    if status == 404:
                        raise LLMError(
                            f"Model '{model}' not found at this endpoint (404)"
                        )
                    body = exc.response.text[:500]
                    raise LLMError(f"LLM API error {status} for '{model}': {body}")

                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last = exc
                    delay = self._retry_delay(attempt, getattr(exc, "response", None))
                    logger.warning(
                        "LLM %s attempt %d/%d: %s, retrying in %.1fs",
                        model, attempt + 1, self.max_retries,
                        type(exc).__name__, delay,
                    )
                    await asyncio.sleep(delay)

        raise LLMError(
            f"LLM request failed after {self.max_retries} attempts "
            f"for model '{model}': {last}"
        )

    # ------------------------------------------------------------------
    # Retry policy
    # ------------------------------------------------------------------

    @staticmethod
    def _retry_delay(attempt: int, response: Optional[httpx.Response]) -> float:
        # Honor Retry-After if present and sane.
        if response is not None:
            ra = response.headers.get("retry-after")
            if ra:
                try:
                    v = float(ra)
                    if 0 < v <= 30:
                        return v
                except Exception:
                    pass
        # Exponential backoff, capped.
        return min(2 ** attempt, 8)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse(self, data: Dict[str, Any], model: str) -> LLMResponse:
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        msg = choice.get("message") or {}

        calls = self._parse_tool_calls(msg.get("tool_calls") or [])

        content = msg.get("content")
        if content is None:
            content = ""

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens

        # Some providers (OpenAI, NIM) expose cached/detailed token counts.
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = int(details.get("cached_tokens", 0) or 0)

        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cached_tokens": cached_tokens,
            },
            tool_calls=calls,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        return []