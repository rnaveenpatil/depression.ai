"""
LLM module — the provider contract, the concrete adapters, and the runtime registry.

Public surface:

    Contract
        LLMProvider           — abstract base every adapter subclasses
        LLMResponse           — one completion result
        Message               — one chat turn
        ToolCall              — a tool invocation requested by the model
        ToolSpec              — a tool the model is allowed to call
        ModelInfo             — static metadata for a model
        ProviderConfig        — the settings a provider needs to run
        ProviderError         — the standard provider failure type
        ProviderErrorCode     — stable error codes
        FinishReason          — why a completion stopped

    Concrete adapters
        OpenAICompatibleProvider — OpenAI, Groq, DeepSeek, OpenRouter,
                                   Together, NVIDIA NIM, local vLLM, ...
        AnthropicProvider        — Claude Messages API
        OllamaProvider           — local models via the Ollama daemon
        GeminiProvider           — Google Gemini generateContent

    Registry (from agent.llm.runtime)
        LLMProviderRegistry   — holds the active provider and model
        get_llm_registry      — process-wide registry accessor
        reset_llm_registry    — reset the registry (tests)

    Normalization (from agent.llm.normalizer)
        NormalizedResponse    — the one shape the agent loop consumes
        ResponseNormalizer    — provider response → NormalizedResponse
        StreamAccumulator     — collapse streaming chunks into one response
"""

from agent.llm.provider import (
    FinishReason,
    LLMProvider,
    LLMResponse,
    Message,
    ModelInfo,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
    Role,
    ToolCall,
    ToolSpec,
)

from agent.llm.openai_compatible import OpenAICompatibleProvider
from agent.llm.anthropic import AnthropicProvider
from agent.llm.ollama import OllamaProvider
from agent.llm.gemini import GeminiProvider

from agent.llm.normalizer import (
    NormalizedError,
    NormalizedResponse,
    NormalizedToolCall,
    NormalizedUsage,
    ResponseNormalizer,
    StreamAccumulator,
    get_normalizer,
    normalize,
    normalize_error,
)

# The registry lives in runtime.py (it owns the empty-at-boot policy and
# the install path). Re-export it here so `from agent.llm import
# get_llm_registry` keeps working for existing call sites.
from agent.llm.runtime import (
    LLMProviderRegistry,
    get_llm_registry,
    reset_llm_registry,
)


__all__ = [
    # ---- Contract types ----
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSpec",
    "ModelInfo",
    "ProviderConfig",
    "ProviderError",
    "ProviderErrorCode",
    "FinishReason",
    "Role",

    # ---- Concrete adapters ----
    "OpenAICompatibleProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GeminiProvider",

    # ---- Normalizer ----
    "NormalizedResponse",
    "NormalizedToolCall",
    "NormalizedUsage",
    "NormalizedError",
    "ResponseNormalizer",
    "StreamAccumulator",
    "get_normalizer",
    "normalize",
    "normalize_error",

    # ---- Registry ----
    "LLMProviderRegistry",
    "get_llm_registry",
    "reset_llm_registry",
]