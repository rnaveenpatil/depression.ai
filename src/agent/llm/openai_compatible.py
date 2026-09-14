"""Generic OpenAI-compatible provider adapter.

Supports providers exposing POST /chat/completions with standard tool calls.
This lets users supply a provider/model/base URL/API key without adding a new
hard-coded provider class for every compatible gateway.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
import httpx
from agent.llm.provider import LLMProvider, LLMResponse, Message, ToolCall
from agent.utils.errors import LLMError
from agent.utils.logging import get_logger

logger = get_logger(__name__)

class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    async def complete(self, messages: List[Any], model: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 4096, tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> LLMResponse:
        if not self.api_key:
            raise LLMError("API key not configured")
        if not self.base_url:
            raise LLMError("Base URL not configured")
        if not model:
            raise LLMError("Model not configured")
        normalized = [m.to_dict() if isinstance(m, Message) else m for m in messages]
        payload = {"model": model, "messages": normalized, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools; payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        for key in ("response_format", "top_p", "stop"):
            if key in kwargs: payload[key] = kwargs[key]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last: Optional[Exception] = None
        async with httpx.AsyncClient(base_url=self.base_url.rstrip("/"), headers=headers, timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post("/chat/completions", json=payload)
                    response.raise_for_status()
                    return self._parse(response.json(), model)
                except httpx.HTTPStatusError as exc:
                    last = exc; status = exc.response.status_code
                    if status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(min(2 ** attempt, 8)); continue
                    if status == 401: raise LLMError("Authentication failed — check the API key")
                    raise LLMError(f"LLM API error {status}: {exc.response.text[:500]}")
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last = exc; await asyncio.sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM request failed after {self.max_retries} attempts: {last}")

    def _parse(self, data: Dict[str, Any], model: str) -> LLMResponse:
        choice = (data.get("choices") or [{}])[0]; msg = choice.get("message") or {}
        calls: List[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}; raw = fn.get("arguments", "{}")
            try: args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError: args = {"_raw": raw}
            calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
        usage = data.get("usage") or {}
        return LLMResponse(content=msg.get("content") or "", model=model, provider=self.name, usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0), "total_tokens": usage.get("total_tokens", 0)}, tool_calls=calls, finish_reason=choice.get("finish_reason", "stop"), raw=data)

    def list_models(self) -> List[Dict[str, Any]]:
        return []
