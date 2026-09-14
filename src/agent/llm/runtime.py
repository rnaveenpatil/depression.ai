"""Bridge TUI-selected providers into the central LLM registry."""
from __future__ import annotations
from typing import Any
from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.anthropic import AnthropicProvider


def configure_runtime_provider(registry: Any, provider_name: str, model_id: str, api_key: str, base_url: str) -> bool:
    if not api_key or not base_url or not model_id:
        return False
    provider = registry.providers.get(provider_name)
    adapter_type = AnthropicProvider if provider_name == "anthropic" else OpenAICompatibleProvider
    if provider is None or not isinstance(provider, adapter_type):
        provider = adapter_type({"api_key": api_key, "base_url": base_url, "timeout": 120.0, "max_retries": 3})
        registry.providers[provider_name] = provider
    else:
        provider.api_key = api_key; provider.base_url = base_url
    registry._api_keys[provider_name] = api_key
    from agent.llm import provider as provider_module
    provider_module.MODEL_METADATA.setdefault(model_id, {
        "provider": provider_name, "name": model_id, "description": "User-configured runtime model",
        "context_window": 200_000, "max_output": 16_384,
        "cost_input": 0.0, "cost_output": 0.0,
        "capabilities": ["text", "function_calling"], "recommended_for": ["agentic tasks"],
        "speed": "provider-dependent", "quality": "provider-dependent",
    })
    registry._current_provider = provider_name; registry._current_model = model_id
    return True
