"""Runtime LLM connection selected by the TUI.

A user can supply any OpenAI-compatible Base URL, API key and model ID.
The runtime connection is kept separate from the built-in provider catalog.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.utils.env_manager import get_env_var, set_env_var, load_into_os_environ

RUNTIME_PROVIDER = "custom"
ENV_BASE_URL = "DEPRESSION_BASE_URL"
ENV_API_KEY = "DEPRESSION_API_KEY"
ENV_MODEL = "DEPRESSION_MODEL"
ENV_PROVIDER = "DEPRESSION_PROVIDER"


def load_runtime_config() -> Dict[str, str]:
    load_into_os_environ()
    return {
        "base_url": get_env_var(ENV_BASE_URL) or os.getenv(ENV_BASE_URL, ""),
        "api_key": get_env_var(ENV_API_KEY) or os.getenv(ENV_API_KEY, ""),
        "model": get_env_var(ENV_MODEL) or os.getenv(ENV_MODEL, ""),
        "provider": get_env_var(ENV_PROVIDER) or RUNTIME_PROVIDER,
    }


def save_runtime_config(base_url: str, api_key: str, model: str) -> None:
    base_url, api_key, model = base_url.strip().rstrip("/"), api_key.strip(), model.strip()
    if not base_url or not api_key or not model:
        raise ValueError("base_url, api_key and model are required")
    for key, value in ((ENV_BASE_URL, base_url), (ENV_API_KEY, api_key),
                       (ENV_MODEL, model), (ENV_PROVIDER, RUNTIME_PROVIDER)):
        set_env_var(key, value)
    os.environ.update({ENV_BASE_URL: base_url, ENV_API_KEY: api_key,
                       ENV_MODEL: model, ENV_PROVIDER: RUNTIME_PROVIDER})


def configure_runtime_provider(registry: Any, base_url: str, api_key: str,
                               model: str) -> OpenAICompatibleProvider:
    """Install the user endpoint as the active runtime provider."""
    save_runtime_config(base_url, api_key, model)
    provider = OpenAICompatibleProvider({
        "api_key": api_key.strip(), "base_url": base_url.strip().rstrip("/"),
        "timeout": 120.0, "max_retries": 3,
    })
    registry.providers[RUNTIME_PROVIDER] = provider
    registry._api_keys[RUNTIME_PROVIDER] = api_key.strip()
    registry._current_provider = RUNTIME_PROVIDER
    registry._current_model = model.strip()
    # Existing registry completion code resolves a model through metadata.
    # This entry is created only for the user-selected runtime model.
    from agent.llm import provider as provider_module
    provider_module.MODEL_METADATA[model.strip()] = {
        "provider": RUNTIME_PROVIDER, "name": model.strip(),
        "description": "User-configured runtime model", "context_window": 200_000,
        "max_output": 16_384, "cost_input": 0.0, "cost_output": 0.0,
        "capabilities": ["text", "function_calling"],
        "recommended_for": ["agentic tasks"], "speed": "provider-dependent",
        "quality": "provider-dependent",
    }
    return provider
