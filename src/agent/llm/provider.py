"""
LLM Provider Registry - User-Driven Model Management

This module is the SINGLE SOURCE OF TRUTH for the currently active model.

Design:
    The registry starts empty. The user connects a model through the TUI
    (base URL + API key + model ID), which installs a runtime provider via
    agent.llm.runtime.configure_runtime_provider(). Until that happens,
    the registry has no providers and no models.

    Every query before the first /connect returns a clean "no model
    selected" error. That is deliberate: no hardcoded providers, no
    catalog of models the user cannot reach, no accidental fallback.

Provides:
    - Registry lifecycle (empty at boot; populated by runtime.install)
    - Model info lookup against the live catalog
    - Runtime model selection
    - API key management
    - Fallback chain execution on provider failure
    - Health checking
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.utils.logging import get_logger
from agent.utils.errors import LLMError

logger = get_logger(__name__)


# ======================================================================
# BASE INTERFACES
# ======================================================================

@dataclass
class Message:
    """A chat message"""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Optional[str]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
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
        return False


# ======================================================================
# MODEL CATALOG
# ======================================================================
#
# Starts empty. agent.llm.runtime fills it in when the user connects.
# Reading code (loop._compute_cost, provider.get_model_info, ...) still
# works — it just returns empty/zero values until a model is connected.

MODEL_METADATA: Dict[str, Dict[str, Any]] = {}


# ======================================================================
# PROVIDER REGISTRY
# ======================================================================

class LLMProviderRegistry:
    """
    Central registry for the active provider and model.

    Begins empty. agent.llm.runtime owns the entry point that installs
    a runtime provider when the user connects via the TUI.
    """

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self._models: Dict[str, List[Dict[str, Any]]] = {}
        self._current_provider: Optional[str] = None
        self._current_model: Optional[str] = None
        self._api_keys: Dict[str, str] = {}
        self._fallback_chain: List[str] = []
        self._lock = asyncio.Lock()

        logger.info("LLM Provider Registry initialized (empty; connect via /connect)")

    # ------------------------------------------------------------------
    # RUNTIME INSTALLATION
    # ------------------------------------------------------------------

    def install_provider(
        self,
        name: str,
        provider: LLMProvider,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Install a runtime provider. Called by agent.llm.runtime when the
        user connects a model. Idempotent — replaces any existing entry
        with the same name.
        """
        self.providers[name] = provider
        if api_key:
            self._api_keys[name] = api_key
            provider.api_key = api_key

        if model:
            self._current_provider = name
            self._current_model = model
            if metadata:
                MODEL_METADATA[model] = metadata
            else:
                MODEL_METADATA.setdefault(model, {
                    "provider": name,
                    "name": model,
                    "description": "User-connected runtime model",
                    "context_window": 200_000,
                    "max_output": 16_384,
                    "cost_input": 0.0,
                    "cost_output": 0.0,
                    "capabilities": ["text", "function_calling"],
                    "recommended_for": ["agentic tasks"],
                    "speed": "provider-dependent",
                    "quality": "provider-dependent",
                })

        logger.info(
            "Installed runtime provider %r (model=%s)", name, model or "(none)"
        )

    def uninstall_provider(self, name: str) -> bool:
        """Remove a runtime provider."""
        provider = self.providers.pop(name, None)
        if provider is None:
            return False
        self._api_keys.pop(name, None)
        if self._current_provider == name:
            self._current_provider = None
            self._current_model = None
        return True

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return all available models grouped by provider."""
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

        if provider in ("local", "gpt-oss"):
            return p.is_available()

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

        if self._current_model:
            info = self.get_model_info(self._current_model)
            if info and info.get("provider") != provider_name:
                models = self.list_models(provider_name)
                if models:
                    self._current_model = models[0]["id"]

        return True

    def set_fallback_chain(self, models: List[str]) -> None:
        """
        Set an ordered fallback chain. When complete() fails on the current
        model, the next model in this chain is tried. An empty chain means
        no fallback.
        """
        self._fallback_chain = [m for m in (models or []) if m]

    def get_fallback_chain(self) -> List[str]:
        return list(self._fallback_chain)

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
        Send a completion request. On failure, walks the fallback chain
        (if any) in order until one model succeeds.
        """
        primary = model or self._current_model
        if not primary:
            raise LLMError(
                "No model selected. Connect a model via the /connect panel first."
            )

        candidates: List[str] = [primary]
        for fallback in self._fallback_chain:
            if fallback and fallback not in candidates:
                candidates.append(fallback)

        last_error: Optional[Exception] = None
        for candidate in candidates:
            info = self.get_model_info(candidate)
            if not info:
                last_error = LLMError(f"Unknown model: {candidate}")
                continue
            provider_name = info.get("provider")
            provider = self.providers.get(provider_name)
            if not provider:
                last_error = LLMError(f"Provider not available: {provider_name}")
                continue

            try:
                return await provider.complete(
                    messages=messages,
                    model=candidate,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    **kwargs,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Model %s failed (%s); trying next fallback",
                    candidate,
                    exc,
                )
                continue

        if last_error is not None:
            raise LLMError(f"All models failed. Last error: {last_error}")
        raise LLMError("No usable model for completion.")

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

    def has_model(self) -> bool:
        """True if a current model is selected and reachable."""
        if not self._current_model:
            return False
        info = self.get_model_info(self._current_model)
        if not info:
            return False
        return info.get("provider") in self.providers

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