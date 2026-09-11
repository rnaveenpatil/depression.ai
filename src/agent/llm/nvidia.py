"""
NVIDIA NIM Provider - Nemotron 3 family + NVIDIA-hosted models.

NVIDIA NIM (NVIDIA Inference Microservices) hosts a large catalog of
open-weight models at https://integrate.api.nvidia.com/v1 using an
OpenAI-compatible API.

Featured models:
    - nvidia/nemotron-3-ultra-550b-a55b        (550B/55B active — frontier reasoning)
    - nvidia/nemotron-3-super-120b-a12b        (120B/12B active — balanced)
    - nvidia/nemotron-3.5-lightning-30b-a3b    (30B/3B active  — fast agent execution)
    - nvidia/nemotron-3-nano-30b-a3b           (30B/3B active  — lightweight)
    - nvidia/llama-3.3-nemotron-super-49b-v1   (49B — high quality general)
    - meta/llama-3.3-70b-instruct               (Meta Llama 3.3 70B)
    - google/gemma-4-31b-it                    (Gemma 4 31B)
    - openai/gpt-oss-120b                      (OpenAI 120B open-weight)

API key resolution order (highest priority first):
    1. config["api_key"]
    2. NVIDIA_NIM_API_KEY environment variable
    3. NVIDIA_API_KEY environment variable
    4. HARDCODED_API_KEY                       (dev fallback — see below)
    5. NVIDIA_HARDCODED_API_KEY env variable

⚠️  SECURITY WARNING
    Hardcoding API keys is convenient for development but NOT safe
    for shared repos or production. Prefer NVIDIA_API_KEY env var.

Base URL: https://integrate.api.nvidia.com/v1
Docs:     https://build.nvidia.com
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
# ⚠️  HARDCODED API KEY (DEVELOPMENT FALLBACK)
# ======================================================================
#
# Set this only if you want a fallback key baked into the source.
# Leave as "" to rely solely on NVIDIA_API_KEY env var.
#
# NVIDIA keys typically start with "nvapi-".
#
# ⚠️  Never commit a real key to a public repository.
#
HARDCODED_API_KEY: str = ""

# Optional base URL override (proxy, on-prem NIM, etc.)
HARDCODED_BASE_URL: str = ""


# ======================================================================
# MODEL DEFINITIONS
# ======================================================================

NVIDIA_MODELS: Dict[str, Dict[str, Any]] = {
    # ---------------- Nemotron 3 family ----------------
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "id": "nvidia/nemotron-3-ultra-550b-a55b",
        "name": "Nemotron 3 Ultra",
        "provider": "nvidia",
        "description": (
            "NVIDIA's most capable open model. 550B total / 55B active "
            "MoE hybrid Mamba-Attention with 1M context. Built for long-running "
            "agentic workflows and frontier-level reasoning."
        ),
        "context_window": 1_000_000,
        "max_output": 32_768,
        "cost_input": 0.0,       # Free during NIM preview
        "cost_output": 0.0,
        "capabilities": [
            "text", "function_calling", "reasoning",
            "structured_output", "long_context",
        ],
        "recommended_for": [
            "frontier reasoning", "long-running agents",
            "complex planning", "multi-step synthesis",
        ],
        "speed": "medium",
        "quality": "highest",
        "parameters": "550B (55B active)",
        "architecture": "MoE Hybrid Mamba-Attention (LatentMoE + MTP)",
        "reasoning_levels": ["low", "medium", "high"],
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "id": "nvidia/nemotron-3-super-120b-a12b",
        "name": "Nemotron 3 Super",
        "provider": "nvidia",
        "description": (
            "Balanced 120B / 12B active MoE. Great quality-to-cost ratio "
            "for agentic reasoning and coding tasks."
        ),
        "context_window": 256_000,
        "max_output": 16_384,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["agentic reasoning", "coding", "general use"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "120B (12B active)",
        "architecture": "MoE",
        "reasoning_levels": ["low", "medium", "high"],
    },
    "nvidia/nemotron-3.5-lightning-30b-a3b": {
        "id": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "name": "Nemotron 3.5 Lightning",
        "provider": "nvidia",
        "description": (
            "Fast 30B / 3B active MoE built for the agent execution layer — "
            "tool calls, validation, and subagent work. Distilled from "
            "Nemotron 3 Ultra. Up to 1M context; ~256K practical on a single H100."
        ),
        "context_window": 1_000_000,
        "max_output": 16_384,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": [
            "text", "function_calling", "reasoning",
            "structured_output", "tool_use",
        ],
        "recommended_for": [
            "subagents", "high-volume tool loops",
            "code review", "validation steps",
        ],
        "speed": "extremely fast",
        "quality": "very good",
        "parameters": "30B (3B active)",
        "architecture": "Hybrid MoE (Mamba-2 + MoE + Attention)",
        "license": "OpenMDW-1.1",
    },
    "nvidia/nemotron-3-nano-30b-a3b": {
        "id": "nvidia/nemotron-3-nano-30b-a3b",
        "name": "Nemotron 3 Nano",
        "provider": "nvidia",
        "description": (
            "Lightweight 30B / 3B active MoE for efficient reasoning, math, "
            "and everyday tasks on modest hardware."
        ),
        "context_window": 256_000,
        "max_output": 16_384,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning"],
        "recommended_for": ["local inference", "quick reasoning", "math"],
        "speed": "very fast",
        "quality": "good",
        "parameters": "30B (3B active)",
        "architecture": "MoE",
    },
    "nvidia/llama-3.3-nemotron-super-49b-v1": {
        "id": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "name": "Llama 3.3 Nemotron Super 49B",
        "provider": "nvidia",
        "description": (
            "NVIDIA-tuned Llama 3.3 49B. Strong general-purpose model "
            "with improved reasoning and tool-calling."
        ),
        "context_window": 128_000,
        "max_output": 16_384,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "structured_output"],
        "recommended_for": ["general use", "chat", "tool calling"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "49B",
    },

    # ---------------- NVIDIA-hosted third-party models ----------------
    "meta/llama-3.3-70b-instruct": {
        "id": "meta/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B Instruct",
        "provider": "nvidia",
        "description": "Meta's Llama 3.3 70B instruction-tuned model.",
        "context_window": 128_000,
        "max_output": 8_192,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling"],
        "recommended_for": ["general use", "chat"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "70B",
    },
    "meta/llama-3.1-405b-instruct": {
        "id": "meta/llama-3.1-405b-instruct",
        "name": "Llama 3.1 405B Instruct",
        "provider": "nvidia",
        "description": "Meta's largest open Llama 3.1 model.",
        "context_window": 128_000,
        "max_output": 8_192,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling"],
        "recommended_for": ["complex reasoning"],
        "speed": "medium",
        "quality": "highest",
        "parameters": "405B",
    },
    "google/gemma-4-31b-it": {
        "id": "google/gemma-4-31b-it",
        "name": "Gemma 4 31B IT",
        "provider": "nvidia",
        "description": "Google's Gemma 4 31B — frontier reasoning for coding and agentic workflows.",
        "context_window": 128_000,
        "max_output": 8_192,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning"],
        "recommended_for": ["coding", "agentic workflows"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "31B",
    },
    "openai/gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B (NVIDIA)",
        "provider": "nvidia",
        "description": "OpenAI's 117B open-weight model hosted on NVIDIA NIM.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["production", "high reasoning"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "117B (5.1B active)",
    },
    "openai/gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B (NVIDIA)",
        "provider": "nvidia",
        "description": "OpenAI's 21B open-weight model hosted on NVIDIA NIM.",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["low latency", "high volume"],
        "speed": "very fast",
        "quality": "high",
        "parameters": "21B (3.6B active)",
    },
    "deepseek-ai/deepseek-r1": {
        "id": "deepseek-ai/deepseek-r1",
        "name": "DeepSeek R1",
        "provider": "nvidia",
        "description": "DeepSeek R1 reasoning model.",
        "context_window": 128_000,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "reasoning"],
        "recommended_for": ["complex reasoning", "math"],
        "speed": "medium",
        "quality": "very high",
        "parameters": "671B (37B active)",
    },
    "mistralai/mistral-large-2-instruct": {
        "id": "mistralai/mistral-large-2-instruct",
        "name": "Mistral Large 2",
        "provider": "nvidia",
        "description": "Mistral's large multilingual model.",
        "context_window": 128_000,
        "max_output": 8_192,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "structured_output"],
        "recommended_for": ["multilingual", "general use"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "123B",
    },
    "qwen/qwen3-coder-480b-a35b-instruct": {
        "id": "qwen/qwen3-coder-480b-a35b-instruct",
        "name": "Qwen3 Coder 480B",
        "provider": "nvidia",
        "description": "Qwen's code-specialized 480B MoE.",
        "context_window": 256_000,
        "max_output": 32_768,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "coding"],
        "recommended_for": ["coding", "refactoring", "code review"],
        "speed": "fast",
        "quality": "very high",
        "parameters": "480B (35B active)",
    },
}


# ======================================================================
# PROVIDER
# ======================================================================

class NVIDIAProvider(LLMProvider):
    """
    NVIDIA NIM provider for Nemotron 3 family + third-party models.

    Uses the OpenAI-compatible endpoint at
    https://integrate.api.nvidia.com/v1.

    Features:
    - Function calling / tool use
    - Structured outputs (JSON mode + schema)
    - Reasoning budget control (Nemotron 3 family)
    - Streaming support
    - Automatic retries with exponential backoff
    - Hardcoded API key fallback (development)
    """

    name = "nvidia"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # ------------------------------------------------------------------
        # API key resolution (highest priority first)
        # ------------------------------------------------------------------
        #   1. config["api_key"]                     — runtime override
        #   2. NVIDIA_NIM_API_KEY env var            — recommended (extension)
        #   3. NVIDIA_API_KEY env var                — NVIDIA standard
        #   4. HARDCODED_API_KEY module constant     — dev fallback
        #   5. NVIDIA_HARDCODED_API_KEY env var      — env fallback
        # ------------------------------------------------------------------
        self.api_key = (
            config.get("api_key")
            or os.environ.get("NVIDIA_NIM_API_KEY")
            or os.environ.get("NVIDIA_API_KEY")
            or HARDCODED_API_KEY
            or os.environ.get("NVIDIA_HARDCODED_API_KEY")
        )

        if self.api_key and self.api_key == HARDCODED_API_KEY and HARDCODED_API_KEY:
            logger.warning(
                "NVIDIA NIM is using a HARDCODED API key. "
                "Set NVIDIA_API_KEY in your environment for production use."
            )

        # ------------------------------------------------------------------
        # Base URL resolution
        # ------------------------------------------------------------------
        self.base_url = (
            config.get("base_url")
            or os.environ.get("NVIDIA_BASE_URL")
            or HARDCODED_BASE_URL
            or self.DEFAULT_BASE_URL
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._models = dict(NVIDIA_MODELS)

    # ------------------------------------------------------------------
    # HTTP CLIENT
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or lazily create the HTTP client"""
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

    # ------------------------------------------------------------------
    # COMPLETION
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
        """Send a chat completion request to NVIDIA NIM"""
        if not self.api_key:
            raise LLMError(
                "NVIDIA NIM API key not configured. "
                "Options:\n"
                "  1. export NVIDIA_API_KEY=nvapi-...\n"
                "  2. Set HARDCODED_API_KEY in src/agent/llm/nvidia.py\n"
                "  3. Use the CLI command: /apikey nvidia <key>"
            )

        # Default to Nemotron 3.5 Lightning (best balance)
        model = model or "nvidia/nemotron-3.5-lightning-30b-a3b"

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

        # Tool calling
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Structured outputs
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        # Reasoning budget (Nemotron 3 family)
        if "reasoning_budget" in kwargs:
            payload["reasoning_budget"] = kwargs["reasoning_budget"]
        if "reasoning_effort" in kwargs:
            effort = kwargs["reasoning_effort"]
            if effort in ("low", "medium", "high"):
                payload["reasoning_effort"] = effort

        # Sampling
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        # Streaming
        stream = kwargs.get("stream", False)
        if stream:
            payload["stream"] = True

        # Retry loop
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
                    logger.warning(f"NVIDIA NIM rate limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                elif status >= 500:
                    wait = 2 ** attempt
                    logger.warning(f"NVIDIA NIM server error {status}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                    continue
                elif status == 401:
                    raise LLMError(
                        "NVIDIA NIM authentication failed — check NVIDIA_API_KEY "
                        "or the HARDCODED_API_KEY in nvidia.py"
                    )
                elif status == 404:
                    raise LLMError(f"NVIDIA NIM model not found: {model}")
                else:
                    raise LLMError(f"NVIDIA NIM API error {status}: {body}")

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"NVIDIA NIM timeout on attempt {attempt + 1}")
                await asyncio.sleep(2 ** attempt)
                continue

            except Exception as e:
                last_error = e
                logger.error(f"NVIDIA NIM request failed: {e}")
                break

        raise LLMError(
            f"NVIDIA NIM request failed after {self.max_retries} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # RESPONSE PARSING
    # ------------------------------------------------------------------

    def _parse_response(self, data: Dict[str, Any], model: str) -> LLMResponse:
        """Convert a NIM response into our unified LLMResponse"""
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
            provider="nvidia",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    # ------------------------------------------------------------------
    # STREAMING
    # ------------------------------------------------------------------

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
        model: str,
    ) -> LLMResponse:
        """Stream a response and accumulate chunks into a single LLMResponse"""
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
            provider="nvidia",
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw=None,
        )

    # ------------------------------------------------------------------
    # MODEL LISTING
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        """Return all NVIDIA NIM models defined in this provider"""
        return [
            {**meta, "available": self.is_available()}
            for meta in self._models.values()
        ]

    # ------------------------------------------------------------------
    # AVAILABILITY / HEALTH
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if an NVIDIA API key is configured"""
        return bool(self.api_key)

    def is_healthy(self) -> bool:
        """Quick availability check"""
        return self.is_available()

    async def reconnect(self) -> bool:
        """Reset the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        return True

    async def shutdown(self) -> None:
        """Close the HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


# ======================================================================
# HELPERS
# ======================================================================

def nvidia_models() -> Dict[str, Dict[str, Any]]:
    """Return the static definition of NVIDIA NIM models"""
    return dict(NVIDIA_MODELS)


def set_hardcoded_api_key(key: str) -> None:
    """
    Set the hardcoded API key at runtime.
    Useful for programmatic configuration in trusted environments.
    """
    global HARDCODED_API_KEY
    HARDCODED_API_KEY = key
    logger.info("NVIDIA hardcoded API key updated at runtime")


__all__ = [
    "NVIDIAProvider",
    "NVIDIA_MODELS",
    "nvidia_models",
    "HARDCODED_API_KEY",
    "HARDCODED_BASE_URL",
    "set_hardcoded_api_key",
]