"""
Groq Provider - GPT-OSS, Llama, Qwen, MiniMax via Groq Cloud.

Groq hosts open-weight models at https://api.groq.com/openai/v1
using an OpenAI-compatible API.

Featured models:
    - openai/gpt-oss-120b
    - openai/gpt-oss-20b
    - meta-llama/llama-3.3-70b-versatile
    - qwen/qwen3-27b
    - minimax/minimax-m2.7

API key resolution:
    1. config["api_key"]
    2. GROQ_API_KEY environment variable

Base URL: https://api.groq.com/openai/v1
Docs:     https://console.groq.com
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, List, Optional

import httpx

from agent.llm.provider import LLMProvider, LLMResponse, Message, ToolCall
from agent.utils.logging import get_logger
from agent.utils.errors import LLMError

logger = get_logger(__name__)


# ======================================================================
# MODEL DEFINITIONS
# ======================================================================

GROQ_MODELS: Dict[str, Dict[str, Any]] = {
    "openai/gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "provider": "groq",
        "description": "OpenAI's 117B open-weight model on Groq — extremely fast inference.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["production", "high reasoning", "agentic workflows"],
        "speed": "extremely fast",
        "quality": "very high",
        "parameters": "117B (5.1B active)",
    },
    "openai/gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B",
        "provider": "groq",
        "description": "OpenAI's 21B open-weight model — fast and efficient.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning"],
        "recommended_for": ["low latency", "high volume"],
        "speed": "extremely fast",
        "quality": "high",
        "parameters": "21B (3.6B active)",
    },
    "meta-llama/llama-3.3-70b-versatile": {
        "id": "meta-llama/llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B",
        "provider": "groq",
        "description": "Meta's Llama 3.3 70B — strong general-purpose model.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "structured_output"],
        "recommended_for": ["general use", "chat", "tool calling"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "70B",
    },
    "qwen/qwen3-27b": {
        "id": "qwen/qwen3-27b",
        "name": "Qwen 3.6 27B",
        "provider": "groq",
        "description": "Qwen's 27B model — great for coding and multilingual tasks.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "coding"],
        "recommended_for": ["coding", "multilingual"],
        "speed": "fast",
        "quality": "high",
        "parameters": "27B",
    },
    "minimax/minimax-m2.7": {
        "id": "minimax/minimax-m2.7",
        "name": "MiniMax M2.7",
        "provider": "groq",
        "description": "MiniMax's M2.7 model hosted on Groq.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling"],
        "recommended_for": ["general use"],
        "speed": "fast",
        "quality": "high",
    },
}


# ======================================================================
# PROVIDER
# ======================================================================

class GroqProvider(LLMProvider):
    """
    Groq provider for GPT-OSS, Llama, Qwen, MiniMax models.

    Uses the OpenAI-compatible endpoint at
    https://api.groq.com/openai/v1.

    Features:
    - Function calling / tool use
    - Extremely fast inference (Groq LPU)
    - Streaming support
    - Automatic retries with exponential backoff
    """

    name = "groq"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.api_key = (
            config.get("api_key")
            or os.environ.get("GROQ_API_KEY")
        )

        self.base_url = (
            config.get("base_url")
            or os.environ.get("GROQ_BASE_URL")
            or self.DEFAULT_BASE_URL
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._models = dict(GROQ_MODELS)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

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
            raise LLMError(
                "Groq API key not configured. "
                "Set GROQ_API_KEY environment variable."
            )

        model = model or "openai/gpt-oss-120b"

        normalized = [
            m.to_dict() if isinstance(m, Message) else m
            for m in messages
        ]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": normalized,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        stream = kwargs.get("stream", False)
        if stream:
            payload["stream"] = True

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                client = await self._get_client()

                if stream:
                    return await self._stream_response(client, payload, model)

                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(data, model)

            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                body = e.response.text[:500]

                if status == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Groq rate limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                elif status >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"Groq server error {status}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                elif status == 401:
                    raise LLMError("Groq authentication failed — check GROQ_API_KEY")
                elif status == 404:
                    raise LLMError(f"Groq model not found: {model}")
                else:
                    raise LLMError(f"Groq API error {status}: {body}")

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Groq timeout on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
                continue

            except Exception as e:
                last_error = e
                logger.error(f"Groq request failed: {e}")
                break

        raise LLMError(
            f"Groq request failed after {self.max_retries} attempts: {last_error}"
        )

    def _parse_response(self, data: Dict[str, Any], model: str) -> LLMResponse:
        choice = data.get("choices", [{}])[0] if data.get("choices") else {}
        message = choice.get("message", {}) or {}

        content = message.get("content", "") or ""

        tool_calls: List[ToolCall] = []
        for tc in (message.get("tool_calls") or []):
            fn = tc.get("function", {}) or {}
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=args,
            ))

        usage = data.get("usage", {}) or {}

        return LLMResponse(
            content=content,
            model=model,
            provider="groq",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        model: str,
    ) -> LLMResponse:
        content_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason = "stop"
        usage: Dict[str, int] = {}

        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[len("data:"):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                if data.get("usage"):
                    usage = data["usage"]

                for choice in data.get("choices", []):
                    delta = choice.get("delta", {}) or {}
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        acc = tool_calls_acc.setdefault(idx, {
                            "id": "", "name": "", "arguments": ""
                        })
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            acc["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["arguments"] += fn["arguments"]

        tool_calls: List[ToolCall] = []
        for idx in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_raw": acc["arguments"]}
            tool_calls.append(ToolCall(
                id=acc["id"] or f"call_{idx}",
                name=acc["name"],
                arguments=args,
            ))

        return LLMResponse(
            content="".join(content_parts),
            model=model,
            provider="groq",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=None,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {**meta, "available": self.is_available()}
            for meta in self._models.values()
        ]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def is_healthy(self) -> bool:
        return self.is_available()

    async def reconnect(self) -> bool:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        return True

    async def shutdown(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


__all__ = ["GroqProvider", "GROQ_MODELS"]
