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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from agent.llm.provider import (
    LLMProvider,
    ModelInfo,
    ProviderConfig,
    ProviderError,
    ProviderErrorCode,
)
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

# Reasonable defaults for the adapter constructor.
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_MAX_RETRIES = 2


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
    base_url = (base_url or "").strip().rstrip("/")
    api_key = (api_key or "").strip()
    model = (model or "").strip()

    chosen = (family or "").strip().lower() or None
    if chosen and chosen not in _FAMILIES:
        chosen = None

    notes: List[str] = []

    if not chosen:
        chosen = detect_family(base_url, model, api_key)
        notes.append(f"detected family: {chosen}")

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
        3. Conservative defaults for the family.

    Never raises — a failure yields the family fallbacks with a note.

    Returns a dict with keys:
        context_window, max_output,
        supports_tools, supports_streaming, supports_vision, supports_json_mode,
        description, source
    where source is one of "discovered", "model_info", "fallback".
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

    # 3. Family fallbacks.
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

    save_runtime_config(base_url, api_key, model, family=spec.family)

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

    save_runtime_config(base_url, api_key, model, family=spec.family)
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
    # Config
    "load_runtime_config",
    "save_runtime_config",
    # Detection / planning
    "RuntimeSpec",
    "detect_family",
    "plan_runtime",
    "discover_capabilities",
    # Install
    "configure_runtime_provider",
    "configure_runtime_provider_async",
    "apply_persisted_runtime",
    "apply_persisted_runtime_async",
]