"""Bridge TUI-selected providers into the central LLM registry."""
from __future__ import annotations
from typing import Any
from agent.llm.openai_compatible import OpenAICompatibleProvider


def configure_runtime_provider(registry: Any, provider_name: str, model_id: str, api_key: str, base_url: str) -> bool:
    """Register/update an OpenAI-compatible provider selected at runtime.

    Existing native providers are left intact; unknown providers become a
    generic OpenAI-compatible adapter. This is the key piece that turns the
    TUI form from a cosmetic selector into a real runtime connection.
    """
    if not api_key or not base_url or not model_id:
        return False
    provider = registry.providers.get(provider_name)
    if provider is None or not isinstance(provider, OpenAICompatibleProvider):
        provider = OpenAICompatibleProvider({"api_key": api_key, "base_url": base_url, "timeout": 120.0, "max_retries": 3})
        registry.providers[provider_name] = provider
    else:
        provider.api_key = api_key
        provider.base_url = base_url
    registry._api_keys[provider_name] = api_key
    # Registry.complete expects model metadata. Runtime models are deliberately
    # added without replacing the static catalog.
    existing = registry.MODEL_METADATA if hasattr(registry, "MODEL_METADATA") else None
    try:
        from agent.llm import provider as provider_module
        provider_module.MODEL_METADATA.setdefault(model_id, {
            "provider": provider_name,
            "name": model_id,
            "description": "User-configured runtime model",
            "context_window": 200_000,
            "max_output": 16_384,
            "cost_input": 0.0,
            "cost_output": 0.0,
            "capabilities": ["text", "function_calling"],
            "recommended_for": ["agentic tasks"],
            "speed": "provider-dependent",
            "quality": "provider-dependent",
        })
    except Exception:
        return False
    registry._current_provider = provider_name
    registry._current_model = model_id
    return True
