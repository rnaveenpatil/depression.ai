"""
LLM Provider Registry - Central Model Management

This module is the SINGLE SOURCE OF TRUTH for all available models.
No other file should hardcode model names.

Providers registered:
    - xai      : Grok family (grok-4.6, grok-4.5, grok-4-1-fast-*, grok-code-fast-1, ...)
    - groq     : GPT-OSS 120B / 20B (Ultra-fast inference on Groq LPUs)
    - nvidia   : Nemotron 3 family (Ultra, Super, Lightning, Nano) + third-party models
    - openai   : GPT-4o, GPT-4o Mini
    - anthropic: Claude 3.5 Sonnet, Haiku
    - gpt-oss  : Local GPT-OSS deployment (vLLM/llama.cpp)
    - local    : Ollama / llama.cpp (Llama, Mixtral, etc.)

Provides:
    - Dynamic model discovery across providers
    - Unified model listing interface
    - Model selection and switching
    - API key management (env vars + hardcoded fallbacks)
    - Provider health checking
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum

from agent.utils.logging import get_logger
from agent.utils.errors import LLMError, ConfigError

logger = get_logger(__name__)


# ======================================================================
# BASE INTERFACES
# ======================================================================

@dataclass
class Message:
    """A chat message"""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class ToolCall:
    """A tool call requested by the model"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    name: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 120.0)
        self.max_retries = config.get("max_retries", 3)

    @abstractmethod
    async def complete(
        self,
        messages: List[Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a completion request"""
        ...

    @abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """Return a list of available models"""
        ...

    def is_available(self) -> bool:
        """Check if this provider is properly configured"""
        return bool(self.api_key) or self._is_local()

    def is_healthy(self) -> bool:
        """Check if the provider is responsive"""
        return self.is_available()

    async def reconnect(self) -> bool:
        """Attempt to reconnect (for providers that need it)"""
        return True

    async def shutdown(self) -> None:
        """Clean up resources"""
        pass

    def _is_local(self) -> bool:
        return self.name in ("local", "gpt-oss")


# ======================================================================
# MODEL CATALOG ENTRIES
# ======================================================================

MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    # ==================== xAI / Grok ====================
    "grok-4.6": {
        "provider": "xai",
        "name": "Grok 4.6",
        "description": "Flagship model for code and agentic tasks, minimal hallucinations",
        "context_window": 500_000,
        "max_output": 32_768,
        "cost_input": 2.00,
        "cost_output": 6.00,
        "capabilities": ["text", "vision", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["coding", "agentic tasks", "complex reasoning"],
        "speed": "fast",
        "quality": "highest",
    },
    "grok-4.5": {
        "provider": "xai",
        "name": "Grok 4.5",
        "description": "Frontier model for coding, agentic tasks, and knowledge work",
        "context_window": 500_000,
        "max_output": 32_768,
        "cost_input": 2.00,
        "cost_output": 6.00,
        "capabilities": ["text", "vision", "function_calling", "reasoning"],
        "recommended_for": ["coding", "agentic tasks"],
        "speed": "fast",
        "quality": "highest",
    },
    "grok-4-1-fast-reasoning": {
        "provider": "xai",
        "name": "Grok 4.1 Fast (Reasoning)",
        "description": "Fast reasoning model with 2M context",
        "context_window": 2_000_000,
        "max_output": 32_768,
        "cost_input": 0.20,
        "cost_output": 0.50,
        "capabilities": ["text", "vision", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["fast reasoning", "large context"],
        "speed": "very fast",
        "quality": "very high",
    },
    "grok-4-1-fast-non-reasoning": {
        "provider": "xai",
        "name": "Grok 4.1 Fast (Non-Reasoning)",
        "description": "Fast model without extended reasoning",
        "context_window": 2_000_000,
        "max_output": 32_768,
        "cost_input": 0.20,
        "cost_output": 0.50,
        "capabilities": ["text", "vision", "function_calling", "structured_output"],
        "recommended_for": ["quick tasks", "large context"],
        "speed": "very fast",
        "quality": "very good",
    },
    "grok-code-fast-1": {
        "provider": "xai",
        "name": "Grok Code Fast 1",
        "description": "Code-optimized model for agentic coding workflows",
        "context_window": 256_000,
        "max_output": 32_768,
        "cost_input": 0.20,
        "cost_output": 1.50,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["coding", "code review", "refactoring"],
        "speed": "very fast",
        "quality": "high",
    },
    "grok-3": {
        "provider": "xai",
        "name": "Grok 3",
        "description": "Previous generation flagship",
        "context_window": 131_072,
        "max_output": 16_384,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "capabilities": ["text", "function_calling", "structured_output"],
        "recommended_for": ["general use"],
        "speed": "medium",
        "quality": "very high",
    },
    "grok-3-mini": {
        "provider": "xai",
        "name": "Grok 3 Mini",
        "description": "Compact reasoning model",
        "context_window": 131_072,
        "max_output": 16_384,
        "cost_input": 0.30,
        "cost_output": 0.50,
        "capabilities": ["text", "function_calling", "reasoning", "structured_output"],
        "recommended_for": ["quick reasoning"],
        "speed": "fast",
        "quality": "good",
    },

    # ==================== Groq (hosts GPT-OSS) ====================
    "openai/gpt-oss-120b": {
        "provider": "groq",
        "name": "GPT-OSS 120B (Groq)",
        "description": "OpenAI's 117B open-weight model on Groq LPUs — high reasoning, production-ready",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.15,
        "cost_output": 0.75,
        "capabilities": [
            "text", "function_calling", "reasoning",
            "structured_output", "code_execution",
        ],
        "recommended_for": ["production", "high reasoning", "agentic tasks"],
        "speed": "very fast",
        "quality": "very high",
        "parameters": "117B (5.1B active)",
        "reasoning_levels": ["low", "medium", "high"],
        "harmony_format": True,
    },
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "name": "GPT-OSS 20B (Groq)",
        "description": "OpenAI's 21B open-weight model on Groq LPUs — extremely low latency",
        "context_window": 131_072,
        "max_output": 32_768,
        "cost_input": 0.10,
        "cost_output": 0.50,
        "capabilities": [
            "text", "function_calling", "reasoning",
            "structured_output", "code_execution",
        ],
        "recommended_for": ["low latency", "high volume", "quick tasks"],
        "speed": "extremely fast",
        "quality": "high",
        "parameters": "21B (3.6B active)",
        "reasoning_levels": ["low", "medium", "high"],
        "harmony_format": True,
    },

    # ==================== NVIDIA NIM (Nemotron 3 family) ====================
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "provider": "nvidia",
        "name": "Nemotron 3 Ultra",
        "description": "NVIDIA's most capable open model — 550B/55B active, 1M context, MoE Mamba-Attention",
        "context_window": 1_000_000,
        "max_output": 32_768,
        "cost_input": 0.0,
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
        "architecture": "MoE Hybrid Mamba-Attention",
        "reasoning_levels": ["low", "medium", "high"],
    },
    "nvidia/nemotron-3-super-120b-a12b": {
        "provider": "nvidia",
        "name": "Nemotron 3 Super",
        "description": "Balanced 120B/12B active MoE for agentic reasoning and coding",
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
    },
    "nvidia/nemotron-3.5-lightning-30b-a3b": {
        "provider": "nvidia",
        "name": "Nemotron 3.5 Lightning",
        "description": "Fast 30B/3B active MoE for agent execution, tool calls, and subagents",
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
    },
    "nvidia/nemotron-3-nano-30b-a3b": {
        "provider": "nvidia",
        "name": "Nemotron 3 Nano",
        "description": "Lightweight 30B/3B active MoE for efficient reasoning on modest hardware",
        "context_window": 256_000,
        "max_output": 16_384,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling", "reasoning"],
        "recommended_for": ["local inference", "quick reasoning", "math"],
        "speed": "very fast",
        "quality": "good",
        "parameters": "30B (3B active)",
    },
    "nvidia/llama-3.3-nemotron-super-49b-v1": {
        "provider": "nvidia",
        "name": "Llama 3.3 Nemotron Super 49B",
        "description": "NVIDIA-tuned Llama 3.3 49B for general chat and tool calling",
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
    "meta/llama-3.3-70b-instruct": {
        "provider": "nvidia",
        "name": "Llama 3.3 70B (NVIDIA)",
        "description": "Meta's Llama 3.3 70B hosted on NVIDIA NIM",
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
    "google/gemma-4-31b-it": {
        "provider": "nvidia",
        "name": "Gemma 4 31B (NVIDIA)",
        "description": "Google's Gemma 4 31B for coding and agentic workflows",
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
    "deepseek-ai/deepseek-r1": {
        "provider": "nvidia",
        "name": "DeepSeek R1 (NVIDIA)",
        "description": "DeepSeek R1 reasoning model hosted on NVIDIA NIM",
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
        "provider": "nvidia",
        "name": "Mistral Large 2 (NVIDIA)",
        "description": "Mistral's large multilingual model hosted on NVIDIA NIM",
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
        "provider": "nvidia",
        "name": "Qwen3 Coder 480B (NVIDIA)",
        "description": "Qwen's code-specialized 480B MoE hosted on NVIDIA NIM",
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

    # ==================== OpenAI ====================
    "gpt-4o": {
        "provider": "openai",
        "name": "GPT-4o",
        "description": "Most capable multimodal model",
        "context_window": 128_000,
        "max_output": 16_384,
        "cost_input": 2.50,
        "cost_output": 10.00,
        "capabilities": ["text", "vision", "function_calling", "json_mode"],
        "recommended_for": ["complex reasoning", "coding"],
        "speed": "fast",
        "quality": "highest",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "name": "GPT-4o Mini",
        "description": "Fast and affordable small model",
        "context_window": 128_000,
        "max_output": 16_384,
        "cost_input": 0.15,
        "cost_output": 0.60,
        "capabilities": ["text", "vision", "function_calling", "json_mode"],
        "recommended_for": ["quick tasks", "high-volume"],
        "speed": "very fast",
        "quality": "good",
    },

    # ==================== Anthropic ====================
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Sonnet",
        "description": "Most intelligent Claude model for coding",
        "context_window": 200_000,
        "max_output": 8_192,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "capabilities": ["text", "vision", "function_calling"],
        "recommended_for": ["coding", "analysis"],
        "speed": "fast",
        "quality": "highest",
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Haiku",
        "description": "Fastest Claude model",
        "context_window": 200_000,
        "max_output": 8_192,
        "cost_input": 1.00,
        "cost_output": 5.00,
        "capabilities": ["text", "vision", "function_calling"],
        "recommended_for": ["quick tasks"],
        "speed": "very fast",
        "quality": "very good",
    },

    # ==================== Local ====================
    "llama-3.1-70b": {
        "provider": "local",
        "name": "Llama 3.1 70B",
        "description": "Meta's open-source model, runs locally",
        "context_window": 128_000,
        "max_output": 4_096,
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": ["text", "function_calling"],
        "recommended_for": ["privacy", "offline use"],
        "speed": "medium",
        "quality": "very good",
    },
}


# ======================================================================
# PROVIDER REGISTRY
# ======================================================================

class LLMProviderRegistry:
    """
    Central registry for all LLM providers and models.

    This is the single source of truth for:
    - Which providers are available
    - Which models each provider offers
    - API key management
    - Current model selection
    """

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self._models: Dict[str, List[Dict[str, Any]]] = {}
        self._current_provider: Optional[str] = None
        self._current_model: Optional[str] = None
        self._api_keys: Dict[str, str] = {}
        self._lock = asyncio.Lock()

        # Load API keys from environment
        self._load_env_keys()

        # Register built-in providers
        self._register_builtin_providers()

        logger.info(
            f"LLM Provider Registry initialized with "
            f"{len(self.providers)} providers, {len(MODEL_METADATA)} models"
        )

    def _load_env_keys(self) -> None:
        """Load API keys from environment variables"""
        key_mapping = {
            "xai":       ["XAI_API_KEY", "GROK_API_KEY"],
            "openai":    ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "google":    ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "groq":      ["GROQ_API_KEY"],
            "gpt-oss":   ["GPT_OSS_API_KEY"],
            "nvidia":    ["NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"],
        }
        for provider, env_vars in key_mapping.items():
            for env_var in env_vars:
                key = os.environ.get(env_var)
                if key:
                    self._api_keys[provider] = key
                    logger.debug(f"Loaded API key for {provider} from {env_var}")
                    break

    def _register_builtin_providers(self) -> None:
        """Register all built-in providers"""
        from agent.llm.groq import GroqProvider
        from agent.llm.nvidia import NVIDIAProvider

        provider_classes: List[Tuple[str, Type[LLMProvider]]] = [
            ("groq", GroqProvider),
            ("nvidia", NVIDIAProvider),
        ]

        for name, cls in provider_classes:
            try:
                config = {
                    "api_key": self._api_keys.get(name),
                    "base_url": self._get_default_base_url(name),
                }
                provider = cls(config)

                # If the provider has a hardcoded key and we don't have one
                # from env, adopt the hardcoded key.
                if not config["api_key"] and getattr(provider, "api_key", None):
                    self._api_keys[name] = provider.api_key
                    logger.debug(
                        f"Adopted hardcoded API key from {name} provider"
                    )

                self.providers[name] = provider
                logger.debug(f"Registered provider: {name}")
            except Exception as e:
                logger.warning(f"Failed to register provider {name}: {e}")

    def _get_default_base_url(self, provider: str) -> Optional[str]:
        """Get default base URL for a provider"""
        defaults = {
            "xai": "https://api.x.ai/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "gpt-oss": "http://localhost:8000/v1",
            "local": "http://localhost:11434/v1",
            "groq": "https://api.groq.com/openai/v1",
            "nvidia": "https://integrate.api.nvidia.com/v1",
        }
        return defaults.get(provider)

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Return all available models grouped by provider.

        Format:
            {
                "xai":    [ {id, name, description, ...}, ... ],
                "groq":   [ {id, name, ...}, ... ],
                "nvidia": [ {id, name, ...}, ... ],
                ...
            }
        """
        result: Dict[str, List[Dict[str, Any]]] = {}

        for model_id, metadata in MODEL_METADATA.items():
            provider = metadata.get("provider", "unknown")
            if provider not in result:
                result[provider] = []

            model_entry = {
                "id": model_id,
                "model": model_id,
                **metadata,
                "available": self._is_model_available(provider, model_id),
            }
            result[provider].append(model_entry)

        return result

    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return a flat list of models, optionally filtered by provider"""
        all_models = self.list_all_models()
        if provider:
            return all_models.get(provider, [])
        return [m for models in all_models.values() for m in models]

    def list_providers(self) -> List[str]:
        """Return list of provider names"""
        return list(self.providers.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific model"""
        if model_id in MODEL_METADATA:
            metadata = MODEL_METADATA[model_id]
            return {
                "id": model_id,
                "model": model_id,
                **metadata,
                "available": self._is_model_available(
                    metadata.get("provider", ""), model_id
                ),
            }
        return None

    def is_model_available(self, provider: str, model: str) -> bool:
        """Check if a specific model is available"""
        return self._is_model_available(provider, model)

    def _is_model_available(self, provider: str, model: str) -> bool:
        """Internal availability check"""
        if provider not in self.providers:
            return False

        p = self.providers[provider]

        # For local providers, check if the service is reachable
        if provider in ("local", "gpt-oss"):
            return p.is_available()

        # For API providers, check if we have an API key
        return bool(self._api_keys.get(provider) or p.api_key)

    # ------------------------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------------------------

    def set_model(self, model_id: str) -> Dict[str, Any]:
        """
        Switch to a specific model.

        Returns:
            {"success": bool, "provider": str, "model": str, "error": str}
        """
        info = self.get_model_info(model_id)
        if not info:
            return {
                "success": False,
                "error": f"Model '{model_id}' not found in registry",
            }

        provider = info.get("provider")
        if provider not in self.providers:
            return {
                "success": False,
                "error": f"Provider '{provider}' not available",
            }

        if not self._is_model_available(provider, model_id):
            return {
                "success": False,
                "error": f"Model '{model_id}' not available (check API key or service)",
            }

        self._current_provider = provider
        self._current_model = model_id

        logger.info(f"Switched to model: {provider}/{model_id}")

        return {
            "success": True,
            "provider": provider,
            "model": model_id,
        }

    def set_provider(self, provider_name: str) -> bool:
        """Switch to a provider (keeping current model if compatible)"""
        if provider_name not in self.providers:
            return False

        self._current_provider = provider_name

        # If current model doesn't belong to this provider, pick a default
        if self._current_model:
            info = self.get_model_info(self._current_model)
            if info and info.get("provider") != provider_name:
                models = self.list_models(provider_name)
                if models:
                    self._current_model = models[0]["id"]

        return True

    def set_api_key(self, provider: str, key: str) -> None:
        """Set API key for a provider"""
        self._api_keys[provider] = key
        if provider in self.providers:
            self.providers[provider].api_key = key
        logger.info(f"API key set for provider: {provider}")

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get the API key for a provider"""
        return self._api_keys.get(provider)

    def get_current_model(self) -> Optional[str]:
        """Get the current model ID"""
        return self._current_model

    def get_current_provider(self) -> Optional[str]:
        """Get the current provider name"""
        return self._current_provider

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
        """
        Send a completion request using the current or specified model.
        """
        model_id = model or self._current_model
        if not model_id:
            raise LLMError("No model selected. Use set_model() first.")

        info = self.get_model_info(model_id)
        if not info:
            raise LLMError(f"Unknown model: {model_id}")

        provider_name = info.get("provider")
        provider = self.providers.get(provider_name)
        if not provider:
            raise LLMError(f"Provider not available: {provider_name}")

        return await provider.complete(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

    async def complete_with_tools(
        self,
        messages: List[Any],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> LLMResponse:
        """Convenience method for tool-calling completions"""
        return await self.complete(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=tools,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Check if at least one provider is healthy"""
        return any(p.is_healthy() for p in self.providers.values())

    async def reconnect_all(self) -> None:
        """Attempt to reconnect all providers"""
        for name, provider in self.providers.items():
            try:
                await provider.reconnect()
                logger.debug(f"Reconnected provider: {name}")
            except Exception as e:
                logger.warning(f"Failed to reconnect {name}: {e}")


# ======================================================================
# GLOBAL REGISTRY
# ======================================================================

_registry: Optional[LLMProviderRegistry] = None


def get_llm_registry() -> LLMProviderRegistry:
    """Get or create the global LLM provider registry"""
    global _registry
    if _registry is None:
        _registry = LLMProviderRegistry()
    return _registry


def reset_llm_registry() -> None:
    """Reset the global registry (for tests)"""
    global _registry
    _registry = None


__all__ = [
    "LLMProvider",
    "LLMProviderRegistry",
    "LLMResponse",
    "Message",
    "ToolCall",
    "MODEL_METADATA",
    "get_llm_registry",
    "reset_llm_registry",
]