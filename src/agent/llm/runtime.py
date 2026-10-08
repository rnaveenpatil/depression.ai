"""Runtime LLM connection — the only source of providers.

The user supplies a Base URL, API key, and model ID. This module
determines which adapter to instantiate, discovers the model's real
capabilities, and installs the provider into the registry.

    User input             Runtime decides
    ─────────────          ────────────────────────────────────────────
    base URL               which API family is this?
    api key                → openai-compatible / anthropic / ollama / gemini
    model                  → instantiate the matching adapter
                           → discover real capabilities + context window
                           → install into the registry with accurate metadata

Design rules
------------
    * The runtime is the sole source of providers — there is no built-in
      catalog. If the user hasn't connected, the registry is empty.
    * Capabilities are *discovered*, not assumed. When discovery fails,
      conservative fallbacks are applied based on the API family — never
      the same 200k + function-calling metadata for every model.
    * The stored env layout is unchanged (DEPRESSION_BASE_URL etc.) so
      existing installs keep working.
    * Every provider gets `await provider.start()` called at install time,
      so the first real request doesn't pay the client-construction cost.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from agent.llm.normalizer import NormalizedResponse, normalize
from agent.llm.provider import (
    LLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
)
from agent.llm.openai_compatible import _lookup_known_model
from agent.utils.env_manager import get_env_var, set_env_var, load_into_os_environ
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# CONSTANTS
# ======================================================================

RUNTIME_PROVIDER = "custom"

ENV_BASE_URL = "DEPRESSION_BASE_URL"
ENV_API_KEY = "DEPRESSION_API_KEY"
ENV_MODEL = "DEPRESSION_MODEL"
ENV_PROVIDER = "DEPRESSION_PROVIDER"       # the adapter family, e.g. "anthropic"

# Adapter families the runtime can pick from.
FAMILY_OPENAI = "openai-compatible"
FAMILY_ANTHROPIC = "anthropic"
FAMILY_OLLAMA = "ollama"
FAMILY_GEMINI = "gemini"

_FAMILIES = (FAMILY_OPENAI, FAMILY_ANTHROPIC, FAMILY_OLLAMA, FAMILY_GEMINI)

# Hostname fragments that identify a family. Matched case-insensitively
# against the base URL's netloc.
_HOST_HINTS: Tuple[Tuple[str, str], ...] = (
    ("anthropic.com",         FAMILY_ANTHROPIC),
    ("generativelanguage.googleapis.com", FAMILY_GEMINI),
    ("googleapis.com",        FAMILY_GEMINI),   # catch other Google hosts
    ("localhost",             FAMILY_OLLAMA),   # most common local host
    ("127.0.0.1",             FAMILY_OLLAMA),
    ("0.0.0.0",               FAMILY_OLLAMA),
)

# Model-name fragments that identify a family when the host doesn't.
_MODEL_HINTS: Tuple[Tuple[str, str], ...] = (
    ("claude",  FAMILY_ANTHROPIC),
    ("gemini",  FAMILY_GEMINI),
    ("gpt-oss", FAMILY_OLLAMA),   # the specific model the user mentioned
)

# When discovery fails, we fall back to these. Deliberately modest —
# the model might be tiny, so we don't want to claim 200k of context
# that doesn't exist.
_FAMILY_FALLBACKS: Dict[str, Dict[str, Any]] = {
    FAMILY_OPENAI: {
        "context_window": 8_192,
        "max_output": 4_096,
        "supports_tools": True,       # most OpenAI-compatible gateways do
        "supports_streaming": True,
        "supports_vision": False,
        "supports_json_mode": True,
    },
    FAMILY_ANTHROPIC: {
        "context_window": 200_000,
        "max_output": 4_096,
        "supports_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "supports_json_mode": False,
    },
    FAMILY_OLLAMA: {
        "context_window": 4_096,      # safe floor; almost all local models have at least this
        "max_output": 2_048,
        "supports_tools": True,       # Ollama 0.3+ supports native tools
        "supports_streaming": True,
        "supports_vision": False,
        "supports_json_mode": True,
    },
    FAMILY_GEMINI: {
        "context_window": 32_768,     # oldest Gemini models started here
        "max_output": 2_048,
        "supports_tools": True,
        "supports_streaming": True,
        "supports_vision": True,
        "supports_json_mode": True,
    },
}

# Fail fast: one hung request should not stall the whole turn for two
# minutes. Matches the documented "30s x 2 retries" behaviour.
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 2


# ======================================================================
# MODEL CATALOG
# ======================================================================
#
# Starts empty — there is no built-in model list anywhere in this
# codebase. `install_provider()` fills it in when the user connects, and
# everything downstream (cost accounting, context limits, model pickers)
# reads it. Until then those callers simply see empty/zero values.

#: Context window assumed when the connected model's is unknown or tiny.
<<<<<<< HEAD
DEFAULT_CONTEXT_WINDOW = 1_000_000
=======
DEFAULT_CONTEXT_WINDOW = 1_048_576
>>>>>>> f2aabb6 (finaly fixed)

#: model id -> metadata (context_window, cost_input, capabilities, ...)
MODEL_METADATA: Dict[str, Dict[str, Any]] = {}


# ======================================================================
# PROVIDER REGISTRY
# ======================================================================

class LLMProviderRegistry:
    """
    Process-wide holder for the active provider, model, and model catalog.

    Begins empty. `agent.llm.runtime` owns the install path — connecting
    (TUI / CLI / persisted env) calls `install_provider()`; nothing else
    ever puts a provider in here.

    Note the two provider names in play:

        install name  — "custom", the key in `providers` and what
                        `get_current_provider()` returns.
        family        — the adapter family the metadata was built from
                        ("openai-compatible", "anthropic", ...). It is
                        what `MODEL_METADATA[model]["provider"]` and the
                        model-list group key say.

    `_model_owner` maps model id -> install name so lookups work with
    either spelling.
    """

    def __init__(self) -> None:
        self.providers: Dict[str, LLMProvider] = {}
        self._model_owner: Dict[str, str] = {}
        self._current_provider: Optional[str] = None
        self._current_model: Optional[str] = None
        self._api_keys: Dict[str, str] = {}
        self._fallback_chain: List[str] = []

        logger.info("LLM registry initialized (empty; connect a model first)")

    # ------------------------------------------------------------------
    # INSTALL / REMOVE
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
        Push a runtime provider into the registry.

        Called by this module's `configure_runtime_provider*()`. Idempotent:
        reinstalling the same name replaces the entry.
        """
        self.providers[name] = provider
        if api_key:
            self._api_keys[name] = api_key
            provider.api_key = api_key

        if model:
            self._current_provider = name
            self._current_model = model
            self._model_owner[model] = name
            if metadata:
                MODEL_METADATA[model] = dict(metadata)
            else:
                MODEL_METADATA[model] = {
                    "provider": name,
                    "name": model,
                    "description": "User-connected runtime model",
                    "context_window": DEFAULT_CONTEXT_WINDOW,
                    "max_output": 4_096,
                    "cost_input": 0.0,
                    "cost_output": 0.0,
                    "capabilities": ["text", "function_calling"],
                    "recommended_for": ["agentic tasks"],
                    "speed": "provider-dependent",
                    "quality": "provider-dependent",
                }

        logger.info(
            "Installed runtime provider %r (model=%s)", name, model or "(none)"
        )

    def uninstall_provider(self, name: str) -> bool:
        """Remove a runtime provider and everything registered for it."""
        provider = self.providers.pop(name, None)
        self._api_keys.pop(name, None)
        if provider is None:
            return False
        for model in [
            m for m, owner in self._model_owner.items() if owner == name
        ]:
            self._model_owner.pop(model, None)
            MODEL_METADATA.pop(model, None)
        if self._current_provider == name:
            self._current_provider = None
            self._current_model = None
        return True

    # ------------------------------------------------------------------
    # LOOKUP HELPERS
    # ------------------------------------------------------------------

    def _provider_for(self, model_id: str) -> Optional[LLMProvider]:
        """Resolve the installed provider serving `model_id`, if any."""
        owner = self._model_owner.get(model_id)
        if owner and owner in self.providers:
            return self.providers[owner]

        family = (MODEL_METADATA.get(model_id) or {}).get("provider")
        if not family:
            return None
        if family in self.providers:
            return self.providers[family]
        for provider in self.providers.values():
            if getattr(provider, "name", "") == family:
                return provider
        return None

    def _family_of(self, model_id: str) -> str:
        meta = MODEL_METADATA.get(model_id) or {}
        return str(meta.get("provider") or self._model_owner.get(model_id) or "")

    # ------------------------------------------------------------------
    # MODEL DISCOVERY
    # ------------------------------------------------------------------

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """All known models grouped by provider family."""
        result: Dict[str, List[Dict[str, Any]]] = {}
        for model_id in MODEL_METADATA:
            entry = self.get_model_info(model_id)
            if entry is None:
                continue
            result.setdefault(self._family_of(model_id), []).append(entry)
        return result

    async def list_all_models_async(self) -> Dict[str, List[Dict[str, Any]]]:
        """Async twin of `list_all_models()` (the CLI prefers this one)."""
        return self.list_all_models()

    def list_models(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Flat list of models, optionally filtered by family or install name."""
        if not provider:
            return [
                m
                for models in self.list_all_models().values()
                for m in models
            ]
        return [
            entry
            for model_id in MODEL_METADATA
            if provider in (self._family_of(model_id), self._model_owner.get(model_id))
            for entry in [self.get_model_info(model_id)]
            if entry is not None
        ]

    def list_providers(self) -> List[str]:
        """Installed provider names (the keys of `providers`)."""
        return list(self.providers.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Detailed info for one model, or None if it isn't registered."""
        meta = MODEL_METADATA.get(model_id)
        if meta is None:
            return None
        owner = self._model_owner.get(model_id)
        return {
            "id": model_id,
            "model": model_id,
            **meta,
            "provider": self._family_of(model_id),
            "installed_as": owner,
            "available": self.is_model_available(self._family_of(model_id), model_id),
        }

    def is_model_available(self, provider: str, model: str) -> bool:
        """
        Is `model` installed and its provider usable?

        `provider` may be either the install name ("custom") or the
        family the model's metadata was built from.
        """
        meta = MODEL_METADATA.get(model)
        if not meta:
            return False
        owner = self._model_owner.get(model)
        if provider and provider not in (owner, meta.get("provider")):
            return False
        active = self.providers.get(owner or "") or self._provider_for(model)
        if active is None:
            return False
        return bool(active.is_configured())

    # ------------------------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------------------------

    def set_model(self, model_id: str) -> Dict[str, Any]:
        """Switch to a registered model."""
        info = self.get_model_info(model_id)
        if not info:
            return {"success": False, "error": f"Model '{model_id}' not found in registry"}
        if not info["available"]:
            return {
                "success": False,
                "error": f"Model '{model_id}' not available (connect the provider first)",
            }
        owner = info.get("installed_as") or info.get("provider")
        self._current_provider = owner
        self._current_model = model_id
        logger.info("Switched to model: %s/%s", owner, model_id)
        return {"success": True, "provider": owner, "model": model_id}

    def set_provider(self, provider_name: str) -> bool:
        """Switch provider by install name or by family."""
        target = provider_name if provider_name in self.providers else None
        if target is None:
            target = next(
                (
                    name
                    for name, p in self.providers.items()
                    if getattr(p, "name", "") == provider_name
                ),
                None,
            )
        if target is None:
            target = next(
                (
                    owner
                    for model, owner in self._model_owner.items()
                    if MODEL_METADATA.get(model, {}).get("provider") == provider_name
                    and owner in self.providers
                ),
                None,
            )
        if target is None:
            return False

        self._current_provider = target
        if not self._current_model or self._model_owner.get(self._current_model) != target:
            self._current_model = next(
                (
                    model
                    for model, owner in self._model_owner.items()
                    if owner == target
                ),
                self._current_model,
            )
        return True

    def set_fallback_chain(self, models: List[str]) -> None:
        """Ordered alternates tried when the primary model fails."""
        self._fallback_chain = [m for m in (models or []) if m]

    def get_fallback_chain(self) -> List[str]:
        return list(self._fallback_chain)

    def set_api_key(self, provider: str, key: str) -> None:
        """Store an API key for an installed provider."""
        self._api_keys[provider] = key
        installed = self.providers.get(provider)
        if installed is not None:
            installed.api_key = key
        logger.info("API key set for provider: %s", provider)

    def get_api_key(self, provider: str) -> Optional[str]:
        return self._api_keys.get(provider)

    def get_current_model(self) -> Optional[str]:
        return self._current_model

    def get_current_provider(self) -> Optional[str]:
        return self._current_provider

    def get_context_limit(self) -> int:
        """Context window for the active model, with a sane floor."""
        meta = MODEL_METADATA.get(self._current_model or "") or {}
        try:
            window = int(meta.get("context_window") or 0)
        except (TypeError, ValueError):
            window = 0
        return window if window >= 2048 else DEFAULT_CONTEXT_WINDOW

    # ------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[Sequence[Any]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        Send a completion through the active provider.

        The adapter's `LLMResponse` is passed through the normalizer on
        the way out, so callers always get a `NormalizedResponse`.

        On a retryable failure the fallback chain (if any) is walked in
        order; the last error is re-raised when every candidate fails.
        """
        primary = model or self._current_model
        if not primary:
            raise ProviderError(
                "No model selected. Connect a model first (TUI /connect, "
                "or set DEPRESSION_BASE_URL / DEPRESSION_MODEL).",
                code=ProviderErrorCode.MODEL_NOT_FOUND,
            )

        candidates = [primary]
        for fallback in self._fallback_chain:
            if fallback and fallback not in candidates:
                candidates.append(fallback)

        request: Dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        request.update(kwargs)

        last_error: Optional[BaseException] = None
        for candidate in candidates:
            provider = self._provider_for(candidate)
            if provider is None:
                last_error = ProviderError(
                    f"No installed provider for model: {candidate}",
                    code=ProviderErrorCode.MODEL_NOT_FOUND,
                    model=candidate,
                )
                continue
            started_at = time.time()
            try:
                response = await provider.chat(messages, model=candidate, **request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Model %s failed (%s); trying next fallback",
                    candidate, exc,
                )
                continue

            # This is the adapters → normalizer → loop boundary: whatever
            # the adapter produced is re-shaped here once, so the loop
            # only ever sees NormalizedResponse.
            return normalize(
                response,
                provider=getattr(provider, "name", "") or "",
                model=candidate,
                started_at=started_at,
            )

        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderError(
            f"All models failed. Last error: {last_error}",
            code=ProviderErrorCode.UNKNOWN,
            model=primary,
            cause=last_error if isinstance(last_error, BaseException) else None,
        )

    async def complete_with_tools(
        self,
        messages: Sequence[Any],
        tools: Sequence[Any],
        model: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """
        `complete()` with a tool list — the shape the agent loop uses.

        Both methods return a `NormalizedResponse`: guaranteed str text,
        tool calls with ids + dict arguments, canonical finish_reason,
        and usage that always sums correctly.
        """
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

    def has_model(self) -> bool:
        """True if a model is selected and its provider is installed."""
        if not self._current_model:
            return False
        return self._provider_for(self._current_model) is not None

    def is_healthy(self) -> bool:
        """True when at least one installed provider can make a call."""
        if not self.providers:
            return False
        return any(p.is_configured() for p in self.providers.values())

    async def reconnect_all(self) -> None:
        """Best-effort restart of every installed provider."""
        for name, provider in self.providers.items():
            try:
                if not provider.started:
                    await provider.start()
                logger.debug("Reconnected provider: %s", name)
            except Exception as exc:
                logger.warning("Failed to reconnect %s: %s", name, exc)


# ======================================================================
# GLOBAL REGISTRY
# ======================================================================

_registry: Optional[LLMProviderRegistry] = None


def get_llm_registry() -> LLMProviderRegistry:
    """Get (or lazily create) the process-wide registry."""
    global _registry
    if _registry is None:
        _registry = LLMProviderRegistry()
    return _registry


def reset_llm_registry() -> None:
    """Drop the process-wide registry (tests use this for isolation)."""
    global _registry
    _registry = None


# ======================================================================
# DETECTION RESULT
# ======================================================================

@dataclass
class RuntimeSpec:
    """What we figured out about the user's input before creating anything."""
    family: str
    base_url: str
    api_key: str
    model: str
    provider_config: ProviderConfig
    notes: List[str] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.family == FAMILY_OLLAMA or any(
            h in self.base_url.lower()
            for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
        )


# ======================================================================
# CONFIG PERSISTENCE (unchanged env layout)
# ======================================================================

def load_runtime_config() -> Dict[str, str]:
    """
    Read the persisted connection from the env file (or os.environ).

    Returns a dict with keys: base_url, api_key, model, provider.
    Empty strings when a value is missing. `provider` defaults to the
    family we last wrote, or "custom" if we can't infer one.
    """
    load_into_os_environ()
    return {
        "base_url": get_env_var(ENV_BASE_URL) or os.getenv(ENV_BASE_URL, ""),
        "api_key":  get_env_var(ENV_API_KEY)  or os.getenv(ENV_API_KEY, ""),
        "model":    get_env_var(ENV_MODEL)    or os.getenv(ENV_MODEL, ""),
        "provider": get_env_var(ENV_PROVIDER) or os.getenv(ENV_PROVIDER, ""),
    }


def save_runtime_config(
    base_url: str,
    api_key: str,
    model: str,
    family: Optional[str] = None,
) -> Dict[str, str]:
    """
    Persist the connection. `family` records which adapter was chosen,
    so a later launch can reinstall the same one without re-detecting.

    Returns the values that were written.
    """
    base_url = (base_url or "").strip().rstrip("/")
    api_key = (api_key or "").strip()
    model = (model or "").strip()

    if not base_url or not model:
        raise ValueError("base_url and model are required")

    # Ollama doesn't need an API key; everything else does.
    if not api_key and family != FAMILY_OLLAMA:
        raise ValueError("api_key is required for this provider")

    values = {
        ENV_BASE_URL: base_url,
        ENV_API_KEY: api_key,
        ENV_MODEL: model,
        ENV_PROVIDER: family or RUNTIME_PROVIDER,
    }
    for key, value in values.items():
        set_env_var(key, value)
    os.environ.update(values)
    return values


# ======================================================================
# FAMILY DETECTION
# ======================================================================

def detect_family(base_url: str, model: str, api_key: str = "") -> str:
    """
    Pick the adapter family from the URL and model name.

    Order:
        1. An explicit host hint (anthropic.com, localhost, ...).
        2. A model-name hint (claude-*, gemini-*, gpt-oss:*).
        3. Default to OpenAI-compatible, which is the widest umbrella.

    Never raises. A garbage URL just yields the default family.
    """
    host = _safe_host(base_url)
    haystack = f"{host} {base_url or ''}".lower()

    for fragment, family in _HOST_HINTS:
        if fragment in haystack:
            return family

    model_l = (model or "").lower()
    for fragment, family in _MODEL_HINTS:
        if model_l.startswith(fragment) or fragment in model_l:
            return family

    # Fall through to the widest umbrella.
    return FAMILY_OPENAI


def _safe_host(url: str) -> str:
    """Extract the netloc from a URL, tolerating a missing scheme."""
    if not url:
        return ""
    candidate = url
    if "://" not in candidate:
        candidate = "http://" + candidate
    try:
        return (urlparse(candidate).netloc or "").lower()
    except Exception:
        return ""


# Endpoint suffixes users paste verbatim. Every adapter appends its own
# path ("/chat/completions", "/v1/messages", "/api/chat", ...), so a
# pasted endpoint has to come back off or the request doubles up.
_PASTED_ENDPOINT_SUFFIXES: Tuple[str, ...] = (
    "/chat/completions",
    "/v1/messages",
    "/messages",
    "/api/chat",
    "/api/generate",
    "/completions",
    "/models",
)


def normalize_base_url(base_url: str, family: str) -> str:
    """
    Turn whatever the user pasted into a URL the adapter can build on.

    People paste a bare host without a scheme, a whole endpoint
    (".../chat/completions"), or a versioned prefix the adapter will add
    again itself (".../v1" for Anthropic). Adapters only append their own
    path, so the connection is normalized here — that way the same paste
    works no matter which shape it arrived in.

    Family rules:
        openai-compatible  — a bare origin gets "/v1", because essentially
                             every OpenAI-compatible server serves there
        anthropic         — drop a trailing "/v1" (adapter posts /v1/messages)
        gemini            — drop "/v1" or "/v1beta" (adapter posts /{version}/models)
        ollama            — drop "/v1" (that is Ollama's OpenAI-compat layer;
                             the adapter posts /api/chat)

    Never raises; a garbage URL just comes back tidied.
    """
    url = (base_url or "").strip()
    if not url:
        return ""
    url = url.split("#", 1)[0].split("?", 1)[0].strip()
    if "://" not in url:
        url = "https://" + url.lstrip("/")
    url = url.rstrip("/")

    # 1. Strip a pasted endpoint (longest suffix first).
    lowered = url.lower()
    stripped_endpoint = False
    for suffix in _PASTED_ENDPOINT_SUFFIXES:
        if lowered.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            stripped_endpoint = True
            break

    # 2. Family-specific version handling.
    if family == FAMILY_ANTHROPIC and url.endswith("/v1"):
        url = url[: -len("/v1")].rstrip("/")
    elif family == FAMILY_GEMINI:
        for version in ("/v1beta", "/v1"):
            if url.endswith(version):
                url = url[: -len(version)].rstrip("/")
                break
    elif family == FAMILY_OLLAMA and url.endswith("/v1"):
        url = url[: -len("/v1")].rstrip("/")
    elif family == FAMILY_OPENAI:
        path = urlparse(url).path.rstrip("/")
        # Only when the user gave us a bare origin: if they pasted the
        # endpoint itself, they have already told us where it lives.
        if path in ("",) and not stripped_endpoint:
            url = f"{url}/v1"

    return url


# ======================================================================
# PROVIDER CONSTRUCTION
# ======================================================================

def _build_provider_config(
    family: str, base_url: str, api_key: str, model: str
) -> ProviderConfig:
    """Build a ProviderConfig with sensible defaults per family."""
    return ProviderConfig(
        api_key=api_key or None,
        base_url=base_url or None,
        default_model=model or None,
        timeout=_DEFAULT_TIMEOUT,
        max_retries=_DEFAULT_MAX_RETRIES,
    )


def _instantiate_provider(spec: RuntimeSpec) -> LLMProvider:
    """
    Import and construct the adapter for this family.

    Imports are lazy so a missing optional dependency for one family
    (say, a broken httpx install for Gemini) doesn't break the others.
    """
    cfg = spec.provider_config

    if spec.family == FAMILY_ANTHROPIC:
        from agent.llm.anthropic import AnthropicProvider
        return AnthropicProvider(cfg)

    if spec.family == FAMILY_OLLAMA:
        from agent.llm.ollama import OllamaProvider
        return OllamaProvider(cfg)

    if spec.family == FAMILY_GEMINI:
        from agent.llm.gemini import GeminiProvider
        return GeminiProvider(cfg)

    # Default: OpenAI-compatible.
    from agent.llm.openai_compatible import OpenAICompatibleProvider
    return OpenAICompatibleProvider(cfg)


def plan_runtime(
    base_url: str,
    api_key: str,
    model: str,
    *,
    family: Optional[str] = None,
) -> RuntimeSpec:
    """
    Work out which adapter to use, without constructing it.

    Public so the TUI can show "this looks like Anthropic" before the
    user commits, or before a slow network call.

    `family` short-circuits detection when the caller already knows
    (e.g. re-applying a saved connection).
    """
    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()
    model = (model or "").strip()

    chosen = (family or "").strip().lower() or None
    if chosen and chosen not in _FAMILIES:
        chosen = None

    notes: List[str] = []

    if not chosen:
        chosen = detect_family(base_url, model, api_key)
        notes.append(f"detected family: {chosen}")

    # Adapt whatever the user pasted into a URL the adapter can build on,
    # so a bare host, a full endpoint, or a versioned prefix all work.
    normalized = normalize_base_url(base_url, chosen)
    if normalized != base_url:
        notes.append(f"base URL adapted → {normalized}")
        base_url = normalized

    cfg = _build_provider_config(chosen, base_url, api_key, model)
    spec = RuntimeSpec(
        family=chosen,
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider_config=cfg,
        notes=notes,
    )
    if spec.is_local:
        notes.append("local endpoint")
    return spec


# ======================================================================
# CAPABILITY DISCOVERY
# ======================================================================

# Import known models lookup functions from provider modules.
def _lookup_known_model_for_family(family: str, model_id: str) -> Optional[Tuple[int, int, bool, bool]]:
    """
    Look up a model in the known models database for the given family.
    
    Returns: (context_window, max_output, supports_vision, supports_tools)
    or None if not found.
    """
    if not model_id:
        return None
    
    if family == FAMILY_OPENAI:
        try:
            from agent.llm.openai_compatible import _lookup_known_model
            return _lookup_known_model(model_id)
        except ImportError:
            pass
    elif family == FAMILY_GEMINI:
        try:
            from agent.llm.gemini import _lookup_known_gemini_model
            result = _lookup_known_gemini_model(model_id)
            if result:
                ctx, max_out = result
                return (ctx, max_out, True, True)
        except ImportError:
            pass
    elif family == FAMILY_ANTHROPIC:
        try:
            from agent.llm.anthropic import _lookup_known_anthropic_model
            result = _lookup_known_anthropic_model(model_id)
            if result:
                ctx, max_out = result
                return (ctx, max_out, True, True)
        except ImportError:
            pass
    elif family == FAMILY_OLLAMA:
        try:
            from agent.llm.ollama import _lookup_known_ollama_model
            return _lookup_known_ollama_model(model_id)
        except ImportError:
            pass
    
    return None


async def discover_capabilities(
    provider: LLMProvider,
    model: str,
    family: str,
) -> Dict[str, Any]:
    """
    Ask the provider what this specific model can do.

    Order of precedence:
        1. The provider's `discover_models()` result, if the model is in it.
        2. The provider's `model_info(model)` result, if implemented.
        3. Known models database for the family (OpenAI-compatible, Gemini, Anthropic, Ollama).
        4. Conservative defaults for the family.

    Never raises — a failure yields the family fallbacks with a note.

    Returns a dict with keys:
        context_window, max_output,
        supports_tools, supports_streaming, supports_vision, supports_json_mode,
        description, source
    where source is one of "discovered", "model_info", "known_db", "fallback".
    """
    fallback = dict(_FAMILY_FALLBACKS.get(family, _FAMILY_FALLBACKS[FAMILY_OPENAI]))
    fallback.setdefault("description", "")
    fallback["source"] = "fallback"

    # 1. Try discover_models() if the provider implements it.
    discover = getattr(provider, "discover_models", None)
    if callable(discover):
        try:
            models = await discover()
        except Exception as exc:
            logger.debug("discover_models() failed for %s: %s", family, exc)
            models = None

        if isinstance(models, list) and models:
            match = _match_model(models, model)
            if match is not None:
                return {
                    "context_window": int(match.context_window or 0) or fallback["context_window"],
                    "max_output": int(match.max_output or 0) or fallback["max_output"],
                    "supports_tools": bool(match.supports_tools),
                    "supports_streaming": bool(match.supports_streaming),
                    "supports_vision": bool(match.supports_vision),
                    "supports_json_mode": bool(match.supports_json_mode),
                    "description": str(match.description or ""),
                    "source": "discovered",
                }

    # 2. Try model_info() if the provider implements it.
    try:
        info = provider.model_info(model)
    except Exception:
        info = None
    if isinstance(info, ModelInfo):
        return {
            "context_window": int(info.context_window or 0) or fallback["context_window"],
            "max_output": int(info.max_output or 0) or fallback["max_output"],
            "supports_tools": bool(info.supports_tools),
            "supports_streaming": bool(info.supports_streaming),
            "supports_vision": bool(info.supports_vision),
            "supports_json_mode": bool(info.supports_json_mode),
            "description": str(info.description or ""),
            "source": "model_info",
        }

    # 3. Try known models database for the family.
<<<<<<< HEAD
    known = _lookup_known_model_for_family(family, model)
    if known:
        ctx, max_out, vision, tools = known
        return {
            "context_window": ctx if ctx > 0 else 1_000_000,
            "max_output": max_out if max_out > 0 else 8_192,
            "supports_tools": tools,
            "supports_streaming": True,
            "supports_vision": vision,
            "supports_json_mode": True,
            "description": "Known model from built-in database",
            "source": "known_db",
        }
=======
    if family == FAMILY_OPENAI:
        known = _lookup_known_model(model)
        if known:
            ctx, max_out, vision, tools = known
            return {
                "context_window": ctx if ctx > 0 else 1_000_000,
                "max_output": max_out if max_out > 0 else 8_192,
                "supports_tools": tools,
                "supports_streaming": True,
                "supports_vision": vision,
                "supports_json_mode": True,
                "description": "Known model from built-in database",
                "source": "known_db",
            }
>>>>>>> f2aabb6 (finaly fixed)

    # 4. Family fallbacks.
    return fallback


def _match_model(models: List[ModelInfo], wanted: str) -> Optional[ModelInfo]:
    """
    Find the ModelInfo matching `wanted`.

    Matches exactly first, then by suffix after a '/' or ':' so
    "qwen2.5:7b" matches "qwen2.5:7b-instruct" and "models/gemini-2.0-flash"
    matches "gemini-2.0-flash".
    """
    if not wanted:
        return None
    wanted_l = wanted.lower()
    for m in models:
        if m.id.lower() == wanted_l:
            return m
    # Suffix / prefix match.
    for m in models:
        mid = m.id.lower()
        if mid.endswith(":" + wanted_l) or wanted_l.endswith(mid):
            return m
        # Last path segment.
        if mid.rsplit("/", 1)[-1] == wanted_l:
            return m
    return None


# ======================================================================
# INSTALL
# ======================================================================

def configure_runtime_provider(
    registry: Any,
    base_url: str,
    api_key: str,
    model: str,
    *,
    family: Optional[str] = None,
) -> LLMProvider:
    """
    Install the user's endpoint as the active runtime provider.

    Detection → instantiation → capability discovery → registry install.
    Persists the connection on success.

    The provider's `start()` is scheduled but not awaited here, because
    this function is called from the TUI's synchronous event path. The
    async variant `configure_runtime_provider_async()` awaits it.
    """
    spec = plan_runtime(base_url, api_key, model, family=family)
    provider = _instantiate_provider(spec)

    # Capability discovery is async; on the sync path we run it inline
    # via asyncio.run() only if there's no running loop. Otherwise we
    # fall back to conservative defaults and let the async variant
    # refresh them.
    capabilities = _sync_capabilities(provider, model, spec.family)

    _install(
        registry=registry,
        provider=provider,
        spec=spec,
        capabilities=capabilities,
    )

    save_runtime_config(spec.base_url, spec.api_key, spec.model, family=spec.family)

    # Kick off start() so the shared HTTP client exists before the first
    # real call. Fire-and-forget is fine here: providers treat a missing
    # client as "start lazily".
    _schedule_start(provider)

    return provider


async def configure_runtime_provider_async(
    registry: Any,
    base_url: str,
    api_key: str,
    model: str,
    *,
    family: Optional[str] = None,
) -> LLMProvider:
    """
    Async variant that awaits capability discovery and provider start.
    Preferred from async call sites (the TUI's async handlers).
    """
    spec = plan_runtime(base_url, api_key, model, family=family)
    provider = _instantiate_provider(spec)

    try:
        await provider.start()
    except Exception as exc:
        logger.debug("provider.start() failed: %s", exc)

    capabilities = await discover_capabilities(provider, model, spec.family)

    _install(
        registry=registry,
        provider=provider,
        spec=spec,
        capabilities=capabilities,
    )

    save_runtime_config(spec.base_url, spec.api_key, spec.model, family=spec.family)
    return provider


def _sync_capabilities(
    provider: LLMProvider, model: str, family: str
) -> Dict[str, Any]:
    """
    Capability lookup on a synchronous code path.

    If there's no running loop, we run the async discovery to completion.
    If there IS a running loop (the common case from Textual), we can't
    block on it — return the family fallbacks and let the async caller
    refresh them.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to run discovery synchronously.
        try:
            return asyncio.run(discover_capabilities(provider, model, family))
        except Exception as exc:
            logger.debug("sync capability discovery failed: %s", exc)

    # We're inside a running loop; use the fallbacks.
    fallback = dict(_FAMILY_FALLBACKS.get(family, _FAMILY_FALLBACKS[FAMILY_OPENAI]))
    fallback.setdefault("description", "")
    fallback["source"] = "fallback"
    return fallback


def _schedule_start(provider: LLMProvider) -> None:
    """Best-effort: schedule provider.start() without blocking."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        try:
            asyncio.run(provider.start())
        except Exception as exc:
            logger.debug("provider.start() failed: %s", exc)
        return

    try:
        loop.create_task(provider.start())
    except Exception:
        # If the provider has to be started manually later, its own
        # `_require_client()` will lazily start it. No harm done.
        pass


def _install(
    *,
    registry: Any,
    provider: LLMProvider,
    spec: RuntimeSpec,
    capabilities: Dict[str, Any],
) -> None:
    """Push the provider and its accurate metadata into the registry."""
    metadata = _metadata_from_capabilities(spec, capabilities)
    registry.install_provider(
        name=RUNTIME_PROVIDER,
        provider=provider,
        api_key=spec.api_key or None,
        model=spec.model,
        metadata=metadata,
    )


def _metadata_from_capabilities(
    spec: RuntimeSpec, capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the registry metadata from discovered capabilities.

    This is the whole point of the rewrite: context window, max output,
    and capability flags come from discovery, not from a hardcoded
    constant. The `source` key records where they came from so the UI
    can show "discovered" vs "assumed".
    """
    caps: List[str] = ["text"]
    if capabilities.get("supports_tools"):
        caps.append("function_calling")
    if capabilities.get("supports_vision"):
        caps.append("vision")
    if capabilities.get("supports_streaming"):
        caps.append("streaming")
    if capabilities.get("supports_json_mode"):
        caps.append("json_mode")

    return {
        "provider": spec.family,
        "name": spec.model,
        "description": capabilities.get("description") or _describe(spec),
        "context_window": int(capabilities.get("context_window") or 0),
        "max_output": int(capabilities.get("max_output") or 0),
        "cost_input": 0.0,
        "cost_output": 0.0,
        "capabilities": caps,
        "recommended_for": ["agentic tasks"],
        "speed": "provider-dependent",
        "quality": "provider-dependent",
        # New: record where the numbers came from.
        "capability_source": capabilities.get("source", "fallback"),
        "family": spec.family,
    }


def _describe(spec: RuntimeSpec) -> str:
    if spec.is_local:
        return "Local model via Ollama"
    if spec.family == FAMILY_ANTHROPIC:
        return "Anthropic Claude model"
    if spec.family == FAMILY_GEMINI:
        return "Google Gemini model"
    return "User-configured runtime model"


# ======================================================================
# RE-APPLY A SAVED CONNECTION
# ======================================================================

def apply_persisted_runtime(registry: Any) -> bool:
    """
    Reinstall a previously saved connection at start-up.

    Uses the persisted family when present, so detection doesn't run a
    second time and the same adapter is reinstalled even if the URL
    alone would be ambiguous.
    """
    runtime = load_runtime_config()
    base_url = runtime.get("base_url") or ""
    api_key = runtime.get("api_key") or ""
    model = runtime.get("model") or ""
    family = runtime.get("provider") or None

    if not base_url or not model:
        return False
    if not api_key and family != FAMILY_OLLAMA:
        return False
    if family not in _FAMILIES:
        family = None  # force detection

    try:
        configure_runtime_provider(
            registry, base_url, api_key, model, family=family,
        )
        return True
    except ProviderError:
        return False
    except Exception as exc:
        logger.warning("Could not re-apply saved runtime: %s", exc)
        return False


async def apply_persisted_runtime_async(registry: Any) -> bool:
    """Async variant of `apply_persisted_runtime`."""
    runtime = load_runtime_config()
    base_url = runtime.get("base_url") or ""
    api_key = runtime.get("api_key") or ""
    model = runtime.get("model") or ""
    family = runtime.get("provider") or None

    if not base_url or not model:
        return False
    if not api_key and family != FAMILY_OLLAMA:
        return False
    if family not in _FAMILIES:
        family = None

    try:
        await configure_runtime_provider_async(
            registry, base_url, api_key, model, family=family,
        )
        return True
    except ProviderError:
        return False
    except Exception as exc:
        logger.warning("Could not re-apply saved runtime: %s", exc)
        return False


# ======================================================================
# EXPORTS
# ======================================================================

__all__ = [
    # Constants
    "RUNTIME_PROVIDER",
    "ENV_BASE_URL",
    "ENV_API_KEY",
    "ENV_MODEL",
    "ENV_PROVIDER",
    "FAMILY_OPENAI",
    "FAMILY_ANTHROPIC",
    "FAMILY_OLLAMA",
    "FAMILY_GEMINI",
    # Model catalog
    "MODEL_METADATA",
    "DEFAULT_CONTEXT_WINDOW",
    # Registry
    "LLMProviderRegistry",
    "get_llm_registry",
    "reset_llm_registry",
    # Config
    "load_runtime_config",
    "save_runtime_config",
    # Detection / planning
    "RuntimeSpec",
    "detect_family",
    "normalize_base_url",
    "plan_runtime",
    "discover_capabilities",
    # Install
    "configure_runtime_provider",
    "configure_runtime_provider_async",
    "apply_persisted_runtime",
    "apply_persisted_runtime_async",
]