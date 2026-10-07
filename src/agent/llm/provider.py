"""
LLM Provider — the common contract every model backend must satisfy.

Purpose
-------
This module defines the *interface* between the agent and any LLM. Every
concrete backend (OpenAI-compatible, Anthropic, Ollama, a local model, a
test double, ...) subclasses `LLMProvider` and implements `chat()`.

What lives here
---------------
    * Message          — a single chat turn (role, content, tool calls)
    * ToolCall         — a tool invocation requested by the model
    * ToolSpec         — a tool the model is allowed to call
    * ModelInfo        — static metadata for a model
    * ProviderConfig   — the settings a provider needs to run
    * ProviderError    — the standard exception type for provider failures
    * LLMProvider      — the abstract base class

What does NOT live here
-----------------------
    * Any provider-specific HTTP logic (OpenAI, Anthropic, Ollama, ...)
    * The registry / current-model selection (agent.llm.runtime)
    * Model catalogs, pricing tables, cost computation
    * Fallback chains, health checks across providers
    * The empty-at-boot policy — that belongs to the registry

Design notes
------------
    * One async method, `chat()`, is the entire request surface. Streaming
      is expressed as an optional keyword (`stream=True`) that yields
      `chat()` results, not as a second abstract method — this keeps the
      contract small.
    * Every provider reports its own capabilities via `capabilities()`.
      Callers must consult that instead of assuming tool support.
    * Errors are raised as `ProviderError` with a stable `code` field so
      callers can react without string matching.
    * Lifecycle is explicit: `start()` / `close()` are awaited, both are
      idempotent, and neither is required to do anything.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union


# ======================================================================
# ENUMS
# ======================================================================

class Role(str, Enum):
    """Chat roles understood by every provider."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Why a completion stopped."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ProviderErrorCode(str, Enum):
    """Stable error codes for cross-provider handling."""
    AUTH = "auth"                   # bad/missing API key
    RATE_LIMIT = "rate_limit"       # 429 or equivalent
    TIMEOUT = "timeout"             # network / read timeout
    CONNECTION = "connection"       # DNS, TCP, TLS failure
    BAD_REQUEST = "bad_request"     # 4xx other than auth/rate limit
    SERVER = "server"               # 5xx from the provider
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_LENGTH = "context_length"
    UNSUPPORTED = "unsupported"     # provider can't do what was asked
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# ======================================================================
# MESSAGES AND TOOLS
# ======================================================================

@dataclass
class ToolCall:
    """A tool invocation the model asked for."""
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}


@dataclass
class Message:
    """
    One chat turn.

    `content` is `Optional[str]` because a tool-calling assistant turn may
    carry no text at all — only `tool_calls`.
    """
    role: str
    content: Optional[str] = None
    name: Optional[str] = None                # tool/function name (for role=tool)
    tool_call_id: Optional[str] = None        # for role=tool
    # ToolCall objects, or the dict forms the context layer and the loop
    # serialize into message metadata. Canonicalized in __post_init__.
    tool_calls: Optional[List[Any]] = None

    def __post_init__(self) -> None:
        if not self.tool_calls:
            return
        self.tool_calls = [
            tc for tc in (_coerce_tool_call(t) for t in self.tool_calls)
            if tc is not None
        ]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _stringify_args(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM.value, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=Role.USER.value, content=content)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> "Message":
        return cls(role=Role.ASSISTANT.value, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "Message":
        return cls(role=Role.TOOL.value, content=content,
                   tool_call_id=tool_call_id, name=name)


@dataclass
class ToolSpec:
    """
    A tool the model is allowed to call.

    `parameters` is a JSON-schema object describing the arguments. It is
    provider-agnostic; concrete providers convert it to their own shape.
    """
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


def _coerce_tool_call(raw: Any) -> Optional[ToolCall]:
    """
    Canonicalize one entry of `Message.tool_calls`.

    Three shapes arrive here:
      * `ToolCall`                              — already canonical
      * `{"id","type","function":{...}}`        — OpenAI wire form, which is
        exactly what `context.runtime.add_tool_call()` and the loop write
        into message metadata
      * `{"name", "arguments": {...}}`          — the plain form

    Anything unrecognised is dropped instead of exploding later inside an
    adapter, because a malformed history entry must not kill the request.
    """
    if isinstance(raw, ToolCall):
        return raw
    if not isinstance(raw, dict):
        return None

    fn = raw.get("function")
    fn = fn if isinstance(fn, dict) else {}
    name = fn.get("name") or raw.get("name") or raw.get("tool") or ""
    if not isinstance(name, str) or not name:
        return None

    args: Any = fn.get("arguments", raw.get("arguments", raw.get("parameters")))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}

    call_id = raw.get("id") or fn.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{uuid.uuid4().hex[:12]}"
    return ToolCall(id=call_id, name=name, arguments=args)


def _stringify_args(args: Any) -> str:
    """Providers want tool arguments as a JSON string, not a dict."""
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args or {})
    except Exception:
        return "{}"


# ======================================================================
# MODEL / CAPABILITY / CONFIG
# ======================================================================

@dataclass
class ModelInfo:
    """Static metadata a provider reports for a model it serves."""
    id: str
    provider: str
    name: str = ""
    description: str = ""
    context_window: int = 0
    max_output: int = 0
    supports_tools: bool = False
    supports_streaming: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "name": self.name or self.id,
            "description": self.description,
            "context_window": self.context_window,
            "max_output": self.max_output,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "supports_vision": self.supports_vision,
            "supports_json_mode": self.supports_json_mode,
        }


@dataclass
class ProviderConfig:
    """
    Everything a provider needs to run. Deliberately flat and JSON-safe.

    `extra` is the escape hatch for provider-specific settings (region,
    project, deployment name, ...) without polluting this contract.
    """
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 2
    default_model: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ProviderConfig":
        data = data or {}
        known = {
            "api_key", "base_url", "organization", "timeout",
            "max_retries", "default_model",
        }
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**kwargs)
        cfg.extra.update(extra)
        return cfg


# ======================================================================
# ERRORS
# ======================================================================

class ProviderError(Exception):
    """
    The single exception type every provider raises.

    `code` is one of ProviderErrorCode; `retryable` says whether the
    caller should try the same request again (rate limit, timeout, 5xx)
    versus fail fast (auth, bad request, model not found).
    """

    def __init__(
        self,
        message: str,
        *,
        code: Union[str, ProviderErrorCode] = ProviderErrorCode.UNKNOWN,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retryable: bool = False,
        status: Optional[int] = None,
        cause: Optional[BaseException] = None,
    ):
        super().__init__(message)
        self.code = code.value if isinstance(code, ProviderErrorCode) else str(code)
        self.provider = provider
        self.model = model
        self.retryable = retryable
        self.status = status
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": str(self),
            "code": self.code,
            "provider": self.provider,
            "model": self.model,
            "retryable": self.retryable,
            "status": self.status,
        }

    def __repr__(self) -> str:
        parts = [f"code={self.code!r}"]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.model:
            parts.append(f"model={self.model!r}")
        if self.status:
            parts.append(f"status={self.status}")
        return f"ProviderError({str(self)!r}, {', '.join(parts)})"


# ======================================================================
# RESPONSE
# ======================================================================

@dataclass
class LLMResponse:
    """A single completion result."""
    content: str = ""
    model: str = ""
    provider: str = ""
    finish_reason: str = FinishReason.STOP.value
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """Turn this response into an assistant message for the next turn."""
        return Message.assistant(
            content=self.content or None,
            tool_calls=self.tool_calls or None,
        )


# ======================================================================
# THE PROVIDER CONTRACT
# ======================================================================

class LLMProvider(ABC):
    """
    Abstract base every model backend implements.

    Minimum contract
    ----------------
        name           — short identifier, e.g. "openai", "anthropic"
        chat()         — the one request method
        list_models()  — what this provider can serve (may be static)
        model_info()   — metadata for one model (optional override)

    Lifecycle
    ---------
        `start()` is awaited once before the first request; `close()` is
        awaited at shutdown. Both are idempotent and default to no-ops.

    Capabilities
    ------------
        `capabilities()` returns a dict describing what this provider can
        do as a whole; per-model flags live on `ModelInfo`.
    """

    #: Short provider identifier, lowercase. Subclasses must set this.
    name: str = "base"

    def __init__(self, config: Optional[Union[ProviderConfig, Dict[str, Any]]] = None):
        if isinstance(config, ProviderConfig):
            self.config = config
        else:
            self.config = ProviderConfig.from_dict(config)
        self._started = False

    # ------------------------------------------------------------------
    # REQUIRED
    # ------------------------------------------------------------------

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        tools: Optional[Sequence[ToolSpec]] = None,
        tool_choice: Optional[str] = None,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send one chat request and return the response.

        Must raise `ProviderError` (never a raw HTTP/JSON exception) so
        callers have a single failure type to handle.

        `tool_choice` values are provider-agnostic hints:
            None         — provider default
            "auto"       — model decides
            "none"       — no tools
            "required"   — must call at least one tool
        """
        ...

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Return the models this provider serves. May be static."""
        ...

    # ------------------------------------------------------------------
    # OPTIONAL OVERRIDES
    # ------------------------------------------------------------------

    def model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Look up one model. Default: search `list_models()`."""
        for info in self.list_models():
            if info.id == model_id:
                return info
        return None

    def capabilities(self) -> Dict[str, bool]:
        """
        Provider-wide feature flags. Callers should still check the
        per-model flags on `ModelInfo` before relying on a feature.
        """
        return {
            "tools": True,
            "streaming": False,
            "vision": False,
            "json_mode": False,
            "system_prompt": True,
        }

    def is_configured(self) -> bool:
        """
        True when this provider has everything it needs to make a call
        (typically an API key, or a reachable local endpoint).
        """
        return bool(self.config.api_key) or self._is_local()

    def _is_local(self) -> bool:
        """Override for providers that talk to localhost (Ollama, vLLM...)."""
        return False

    # ------------------------------------------------------------------
    # CONFIG SHORTHANDS
    # ------------------------------------------------------------------
    # Call sites (and the registry) talk about `provider.api_key` /
    # `provider.base_url` without reaching into `.config`. These are
    # plain pass-throughs to the ProviderConfig — no extra state.

    @property
    def api_key(self) -> Optional[str]:
        return self.config.api_key

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self.config.api_key = value

    @property
    def base_url(self) -> Optional[str]:
        return self.config.base_url

    @base_url.setter
    def base_url(self, value: Optional[str]) -> None:
        self.config.base_url = value

    @property
    def timeout(self) -> float:
        return self.config.timeout

    @timeout.setter
    def timeout(self, value: float) -> None:
        self.config.timeout = float(value)

    @property
    def max_retries(self) -> int:
        return self.config.max_retries

    @max_retries.setter
    def max_retries(self, value: int) -> None:
        self.config.max_retries = int(value)

    @property
    def default_model(self) -> Optional[str]:
        return self.config.default_model

    @default_model.setter
    def default_model(self, value: Optional[str]) -> None:
        self.config.default_model = value

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Prepare the provider (open an HTTP session, warm a connection...).
        Default: no-op. Safe to call more than once.
        """
        self._started = True

    async def close(self) -> None:
        """
        Release resources. Default: no-op. Safe to call more than once.
        """
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    # ------------------------------------------------------------------
    # STREAMING — optional, expressible in terms of chat()
    # ------------------------------------------------------------------

    async def stream(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> AsyncIterator[LLMResponse]:
        """
        Optional streaming interface.

        The default implementation calls `chat()` once and yields the
        whole response as a single chunk. Providers that support real
        streaming override this. Callers that only need incremental text
        can consume `chunk.content`; tool calls arrive on the final chunk.
        """
        response = await self.chat(messages, **kwargs)
        yield response

    # ------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """
        A cheap reachability probe. Default: whether we're configured.
        Providers with a real ping endpoint (models list, etc.) override.
        """
        return self.is_configured()

    # ------------------------------------------------------------------
    # CONVENIENCE
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Alias for `chat()`. Kept for call sites that predate the rename
        and for readability in loops that think in terms of "completions".
        """
        return await self.chat(messages, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} name={self.name!r} "
            f"configured={self.is_configured()} started={self._started}>"
        )


__all__ = [
    "Role",
    "FinishReason",
    "ProviderErrorCode",
    "Message",
    "ToolCall",
    "ToolSpec",
    "ModelInfo",
    "ProviderConfig",
    "ProviderError",
    "LLMResponse",
    "LLMProvider",
]