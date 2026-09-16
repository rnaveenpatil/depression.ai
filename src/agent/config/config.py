"""
Configuration Module - Schema, Defaults, Validation, and Dynamic LLM Resolution

KEY FEATURES:
- LLM models are NOT hardcoded — they are pulled dynamically from the
  LLM provider registry (src/agent/llm/provider.py).
- Temperature is resolved dynamically based on prompt complexity/task type.
- Full schema validation, environment variable interpolation, and layered
  merging of config sources.

TEMPERATURE TABLE:
    Trivial (greetings)     → 0.2
    Simple (short Q)        → 0.3
    Moderate                → 0.5
    Complex (reasoning)     → 0.2
    Creative                → 0.8
    Precise (code/math)     → 0.1
    Explanation             → 0.3
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from agent.utils.logging import get_logger
from agent.utils.errors import ConfigError

logger = get_logger(__name__)


# ======================================================================
# ENUMS
# ======================================================================

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Theme(str, Enum):
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"


class EditingMode(str, Enum):
    EMACS = "emacs"
    VI = "vi"


class PermissionMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    DENY = "deny"


class TaskComplexity(str, Enum):
    """Task complexity tiers used for adaptive temperature"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CREATIVE = "creative"
    PRECISE = "precise"
    EXPLANATION = "explanation"


# ======================================================================
# DYNAMIC TEMPERATURE RESOLVER
# ======================================================================

class TemperatureResolver:
    """
    Adaptive temperature resolver.

    Temperature mapping:
        Trivial (greetings)     → 0.2
        Simple (short Q)        → 0.3
        Moderate                → 0.5
        Complex (reasoning)     → 0.2
        Creative                → 0.8
        Precise (code/math)     → 0.1
        Explanation             → 0.3

    Returns (temperature, reason) so the caller can log/trace the decision.
    """

    COMPLEXITY_TEMPS: Dict[str, float] = {
        TaskComplexity.TRIVIAL.value:      0.2,
        TaskComplexity.SIMPLE.value:       0.3,
        TaskComplexity.MODERATE.value:     0.5,
        TaskComplexity.COMPLEX.value:      0.2,
        TaskComplexity.CREATIVE.value:     0.8,
        TaskComplexity.PRECISE.value:      0.1,
        TaskComplexity.EXPLANATION.value:  0.3,
    }

    DEFAULT_TEMP = 0.3

    # ------------------------------------------------------------------
    # Keyword signals
    # ------------------------------------------------------------------

    TRIVIAL_KEYWORDS = {
        "hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
        "yes", "no", "bye", "goodbye", "yo", "sup", "cheers",
    }

    SIMPLE_KEYWORDS = {
        "what is", "who is", "when", "where", "list", "show", "print",
        "how many", "count", "define",
    }

    EXPLANATION_KEYWORDS = {
        "explain", "explanation", "teach me", "walk me through",
        "how does", "how do", "how can", "why does", "why is", "why do",
        "what is the difference", "what's the difference", "difference between",
        "describe", "help me understand", "clarify", "elaborate",
        "tell me about", "give me an overview", "overview of",
        "in simple terms", "for beginners", "tutorial", "guide me",
        "concept of", "meaning of", "purpose of", "intuition behind",
        "breakdown of", "break down", "unpack", "deep dive",
        "compare and contrast", "pros and cons", "use cases",
        "when should i use", "when to use", "best practices",
    }
    STRONG_EXPLANATION_PHRASES = {
        "explain", "teach me", "walk me through", "help me understand",
        "in simple terms", "for beginners", "elaborate", "deep dive",
    }

    CREATIVE_KEYWORDS = {
        "write", "story", "poem", "creative", "brainstorm", "imagine",
        "design", "generate ideas", "suggest names", "marketing", "slogan",
        "compose", "draft", "narrative", "fiction", "article", "blog",
        "song", "lyrics", "joke", "riddle", "caption", "tagline",
    }

    PRECISE_KEYWORDS = {
        "code", "function", "bug", "fix", "error", "debug", "refactor",
        "math", "calculate", "compute", "algorithm", "sql", "query",
        "json", "yaml", "regex", "test", "unit test", "assert",
        "prove", "derive", "solve", "translate", "summarize",
        "extract", "parse", "convert", "format",
    }

    COMPLEX_KEYWORDS = {
        "analyze", "analyse", "architecture", "plan", "strategy", "compare",
        "evaluate", "review", "audit", "migrate", "system design",
        "optimize", "investigate", "diagnose", "research", "assess",
        "trade-off", "tradeoff", "trade-offs", "design pattern",
    }

    TERMINAL_KEYWORDS = {
        "run", "execute", "shell", "bash", "terminal", "command", "cmd",
        "install", "npm", "pip", "apt", "brew", "git commit", "git push",
        "mkdir", "rm ", "cp ", "mv ", "ls ", "cat ", "grep", "find ",
        "chmod", "chown", "docker", "kubectl", "systemctl", "curl",
        "wget", "ssh", "scp", "rsync", "tar", "zip", "unzip",
    }

    # Detectors
    CODE_FENCE_RE = re.compile(r"```")
    CODE_TOKEN_RE = re.compile(
        r"\b(def|class|function|const|let|var|import|from|return|"
        r"async|await|yield|public|private|static)\b"
    )
    SHELL_PROMPT_RE = re.compile(r"^\s*[\$>#]\s+", re.MULTILINE)
    SHELL_CMD_RE = re.compile(
        r"\b(sudo|apt-get|yum|brew|npm|yarn|pnpm|pip|pip3|cargo|go get|"
        r"docker|kubectl|make|cmake|gcc|g\+\+|python3?\s|node\s|"
        r"git\s+(clone|commit|push|pull|checkout|branch|merge))\b"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.override: Optional[float] = cfg.get("override")
        self.min_temp = cfg.get("min", 0.0)
        self.max_temp = cfg.get("max", 1.2)
        self.tier_temps: Dict[str, float] = {
            **self.COMPLEXITY_TEMPS,
            **cfg.get("tier_temps", {}),
        }
        self.provider_bias: Dict[str, float] = cfg.get("provider_bias", {})
        self.model_bias: List[Tuple[str, float]] = cfg.get("model_bias", [
            (r"o1|o3", -0.05),
            (r"deepseek-coder", -0.05),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Classify the prompt into a TaskComplexity tier.
        Returns (tier, reason).
        """
        if not prompt or not prompt.strip():
            return TaskComplexity.SIMPLE.value, "empty prompt"

        text = prompt.lower()
        words = text.split()
        word_count = len(words)

        has_code_fence = bool(self.CODE_FENCE_RE.search(prompt))
        has_code_token = bool(self.CODE_TOKEN_RE.search(text))
        has_shell_prompt = bool(self.SHELL_PROMPT_RE.search(prompt))
        has_shell_cmd = bool(self.SHELL_CMD_RE.search(text))
        has_question = "?" in prompt
        has_multiple_questions = prompt.count("?") >= 2
        has_bullets = bool(re.search(r"(^|\n)\s*[-*•]\s", prompt))
        long_prompt = word_count > 80

        # 1. Explicit hint
        if context:
            hint = context.get("task_type") or context.get("complexity")
            if hint:
                hint = str(hint).lower()
                if hint in self.tier_temps:
                    return hint, f"explicit hint: {hint}"

        # 2. Trivial → 0.2
        if word_count <= 3 and any(w in self.TRIVIAL_KEYWORDS for w in words):
            return TaskComplexity.TRIVIAL.value, "greeting/short acknowledgment"

        # 3. Explanation → 0.3
        if any(p in text for p in self.STRONG_EXPLANATION_PHRASES):
            return TaskComplexity.EXPLANATION.value, "explanation phrase detected"
        if any(kw in text for kw in self.EXPLANATION_KEYWORDS):
            if not has_code_fence or has_question:
                return TaskComplexity.EXPLANATION.value, "explanation keyword detected"

        # 4. Creative → 0.9
        if any(kw in text for kw in self.CREATIVE_KEYWORDS):
            return TaskComplexity.CREATIVE.value, "creative keyword detected"

        # 5. Terminal / shell → 0.1 (treated as precise)
        if has_shell_prompt or has_shell_cmd:
            return TaskComplexity.PRECISE.value, "shell command detected"
        if any(kw in text for kw in self.TERMINAL_KEYWORDS):
            return TaskComplexity.PRECISE.value, "terminal keyword detected"

        # 6. Precise (code/math) → 0.1
        if has_code_fence or has_code_token:
            return TaskComplexity.PRECISE.value, "code detected"
        if any(kw in text for kw in self.PRECISE_KEYWORDS):
            if long_prompt or has_multiple_questions:
                return TaskComplexity.COMPLEX.value, "precise + complex"
            return TaskComplexity.PRECISE.value, "precise keyword detected"

        # 7. Complex reasoning → 0.4
        if any(kw in text for kw in self.COMPLEX_KEYWORDS):
            return TaskComplexity.COMPLEX.value, "complex keyword detected"
        if long_prompt and (has_bullets or has_multiple_questions):
            return TaskComplexity.COMPLEX.value, "long structured prompt"

        # 8. Simple → 0.3
        if any(kw in text for kw in self.SIMPLE_KEYWORDS):
            return TaskComplexity.SIMPLE.value, "simple keyword detected"

        # 9. Moderate (default) → 0.5
        if word_count <= 10 and has_question:
            return TaskComplexity.SIMPLE.value, "short question"
        if word_count <= 30:
            return TaskComplexity.MODERATE.value, "moderate length"

        return TaskComplexity.MODERATE.value, "default → moderate"

    def resolve(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """Resolve the temperature for a given prompt. Returns (temp, reason)."""
        if self.override is not None:
            return self._clamp(self.override), "explicit override"

        if not self.enabled:
            return self.DEFAULT_TEMP, f"adaptive disabled (default {self.DEFAULT_TEMP})"

        tier, tier_reason = self.classify(prompt, context)
        base = self.tier_temps.get(tier, self.DEFAULT_TEMP)

        bias = 0.0
        reasons = [f"tier={tier} ({tier_reason})"]

        if provider:
            b = self.provider_bias.get(provider.lower(), 0.0)
            if b:
                bias += b
                reasons.append(f"provider_bias={b:+.2f}")

        if model:
            for pattern, b in self.model_bias:
                if re.search(pattern, model, re.IGNORECASE):
                    if b:
                        bias += b
                        reasons.append(f"model_bias={b:+.2f}")
                    break

        final = self._clamp(base + bias)
        reasons.append(f"base={base:.2f} bias={bias:+.2f} → {final:.2f}")
        return final, " | ".join(reasons)

    def _clamp(self, temp: float) -> float:
        return round(max(self.min_temp, min(self.max_temp, temp)), 2)


# ======================================================================
# CONFIG SECTIONS
# ======================================================================

@dataclass
class LLMParamsConfig:
    """
    Model generation parameters.
    temperature = None → adaptive (resolved per prompt).
    temperature = float → fixed override.
    """
    temperature: Optional[float] = None   # None = adaptive
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.temperature is not None:
            if not 0.0 <= self.temperature <= 2.0:
                raise ConfigError(
                    f"temperature must be 0.0–2.0, got {self.temperature}"
                )
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ConfigError(f"top_p must be 0.0–1.0, got {self.top_p}")
        if self.top_k < 0:
            raise ConfigError(f"top_k must be >= 0, got {self.top_k}")


@dataclass
class AdaptiveTemperatureConfig:
    """Adaptive temperature configuration"""
    enabled: bool = True
    override: Optional[float] = None
    min: float = 0.0
    max: float = 1.2
    tier_temps: Dict[str, float] = field(default_factory=dict)
    provider_bias: Dict[str, float] = field(default_factory=dict)
    model_bias: List[Tuple[str, float]] = field(default_factory=list)

    def to_resolver(self) -> TemperatureResolver:
        return TemperatureResolver({
            "enabled": self.enabled,
            "override": self.override,
            "min": self.min,
            "max": self.max,
            "tier_temps": self.tier_temps,
            "provider_bias": self.provider_bias,
            "model_bias": self.model_bias,
        })


@dataclass
class LLMConfig:
    """
    LLM provider and model configuration.

    IMPORTANT: `provider` and `model` are placeholders.
    The real list of available models is fetched dynamically from the
    LLM registry (src/agent/llm/provider.py) at runtime.
    """
    provider: str = ""     # empty → resolver picks default from registry
    model: str = ""        # empty → resolver picks default from registry
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    params: LLMParamsConfig = field(default_factory=LLMParamsConfig)
    adaptive_temperature: AdaptiveTemperatureConfig = field(
        default_factory=AdaptiveTemperatureConfig
    )
    extra_headers: Dict[str, str] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    auto_select_model: bool = True
    preferred_models: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.timeout <= 0:
            raise ConfigError("llm.timeout must be > 0")
        if self.max_retries < 0:
            raise ConfigError("llm.max_retries must be >= 0")
        self.params.validate()


@dataclass
class SessionConfig:
    auto_save: bool = True
    save_interval: float = 30.0
    max_sessions: int = 100
    max_messages_per_session: int = 1000
    default_name: str = "session"
    storage_dir: str = "~/.agent/sessions"


@dataclass
class ContextConfig:
    max_tokens: int = 100_000
    max_messages: int = 200
    compaction_threshold: float = 0.8
    compaction_target: float = 0.5
    enable_summarization: bool = True
    include_file_contents: bool = True
    include_tool_outputs: bool = True
    recent_tool_outputs: int = 5
    recent_messages: int = 20


@dataclass
class PermissionRuleConfig:
    pattern: str
    action: str = "allow"


@dataclass
class PermissionsConfig:
    # OpenCode efficient default: allow-all, only ask for external/doom_loop; manual auto handled via manager defaults
    enabled: bool = True
    mode: str = "manual"
    auto_approve: bool = False
    allow_dangerous: bool = False
    default: str = "allow"
    rules: List[PermissionRuleConfig] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    confirm_file_writes: bool = False  # OpenCode: not ask by default
    confirm_shell_commands: bool = False
    permission: Dict[str, Any] = field(default_factory=dict)  # opencode granular map
    safe_paths: List[str] = field(default_factory=lambda: ["~/", "./", "/tmp/"])
    blocked_paths: List[str] = field(default_factory=lambda: [
        "/etc/", "/sys/", "/proc/", "~/.ssh/", "~/.aws/"
    ])


@dataclass
class ToolsConfig:
    # OpenCode-style: every tool is available by default. Only the tools listed
    # here are registered; an empty list would mean "all tools" in legacy paths,
    # so keep the full catalog explicit. Restrict with `disabled`/`enabled`.
    enabled: List[str] = field(default_factory=lambda: [
        "terminal", "filesystem", "read", "write", "edit", "apply_patch",
        "search", "grep", "glob", "git", "web", "webfetch", "websearch",
        "diagnostics", "todo", "todowrite", "todoread", "skill", "question",
        "lsp", "bash", "process", "patch", "browser", "task",
    ])
    disabled: List[str] = field(default_factory=list)
    timeout: float = 60.0
    max_output_size: int = 100_000
    shell: str = "/bin/bash"
    shell_timeout: float = 60.0
    allow_network: bool = True
    allow_git_write: bool = True
    max_file_size: int = 10 * 1024 * 1024


@dataclass
class PlannerConfig:
    enabled: bool = True
    max_tasks_per_plan: int = 20
    max_subtasks_per_task: int = 10
    max_depth: int = 3
    enable_parallel: bool = True
    max_parallel_tasks: int = 5
    adaptive_planning: bool = True
    replan_on_failure: bool = True
    max_replans: int = 3


@dataclass
class SubAgentConfig:
    enabled: bool = True
    max_subagents: int = 10
    max_depth: int = 3
    default_timeout: float = 60.0
    enable_parallel: bool = True
    enable_communication: bool = True
    subagent_model: Optional[str] = None


@dataclass
class LoopConfig:
    max_iterations: int = 50
    max_tool_calls_per_iteration: int = 5
    max_history_length: int = 20
    enable_planning: bool = True
    enable_caching: bool = True
    timeout: float = 300.0
    enable_self_reflection: bool = True
    enable_auto_retry: bool = True


@dataclass
class UIConfig:
    theme: str = "dark"
    color: bool = True
    show_spinner: bool = True
    show_status_bar: bool = True
    show_token_count: bool = True
    show_cost: bool = False
    compact: bool = False
    editing_mode: str = "emacs"
    width: Optional[int] = None
    animate: bool = True
    stream_output: bool = True
    stream_delay: float = 0.005
    show_thinking: bool = True


@dataclass
class StorageConfig:
    path: str = "~/.agent/agent.db"
    cache_dir: str = "~/.agent/cache"
    history_file: str = "~/.agent/input_history"
    max_cache_size: int = 500 * 1024 * 1024
    enable_cache: bool = True
    enable_db: bool = True
    backup_interval: float = 3600.0


@dataclass
class WorkspaceConfig:
    path: str = "."
    auto_scan: bool = True
    scan_depth: int = 5
    ignore_patterns: List[str] = field(default_factory=lambda: [
        ".git", "node_modules", "__pycache__", "*.pyc",
        ".venv", "venv", "dist", "build", ".DS_Store",
        "*.log", "*.tmp", ".cache"
    ])
    respect_gitignore: bool = True
    max_file_size: int = 1024 * 1024


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    console: bool = True


@dataclass
class MCPConfig:
    enabled: bool = False
    servers: List[Dict[str, Any]] = field(default_factory=list)
    timeout: float = 30.0


@dataclass
class PluginsConfig:
    enabled: bool = True
    directories: List[str] = field(default_factory=lambda: ["~/.agent/plugins"])
    auto_load: bool = True
    disabled: List[str] = field(default_factory=list)


@dataclass
class ServerConfig:
    port: int = 4096
    hostname: str = "127.0.0.1"
    mdns: bool = False
    mdnsDomain: str = "opencode.local"
    cors: List[str] = field(default_factory=list)

@dataclass
class SnapshotConfig:
    enabled: bool = True

@dataclass
class WatcherConfig:
    ignore: List[str] = field(default_factory=lambda: ["node_modules/**", "dist/**", ".git/**"])

@dataclass
class AdvancedConfig:
    enable_streaming: bool = True
    enable_function_calling: bool = True
    enable_json_mode: bool = True
    enable_vision: bool = True
    enable_computer_use: bool = False
    prompt_cache: bool = False
    beta_features: List[str] = field(default_factory=list)
    # OpenCode compat
    snapshot: bool = True
    share: str = "manual"  # manual|auto|disabled
    autoupdate: bool = True
    shell: str = "/bin/bash"
    formatter: Any = False
    lsp: Any = False
    compaction: Dict[str, Any] = field(default_factory=lambda: {"auto": True, "prune": False, "reserved": 10000})
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    instructions: List[str] = field(default_factory=list)
    disabled_providers: List[str] = field(default_factory=list)
    enabled_providers: List[str] = field(default_factory=list)
    model: str = ""  # top-level alias for llm.model
    small_model: str = ""
    permission: Dict[str, Any] = field(default_factory=dict)
    server: ServerConfig = field(default_factory=ServerConfig)
    snapshot_config: SnapshotConfig = field(default_factory=SnapshotConfig)


# ======================================================================
# ROOT CONFIG
# ======================================================================

@dataclass
class Config:
    """
    Root configuration object.

    Dynamic LLM behavior:
      - `llm.provider` and `llm.model` may be empty.
      - `resolve_llm()` uses the LLM registry to pick a default model if
        empty, or validates the given model against the registry.
      - `resolve_temperature()` uses TemperatureResolver for adaptive temps.
    """
    version: str = "1.0.0"
    debug: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    subagent: SubAgentConfig = field(default_factory=SubAgentConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    # Runtime cache (not serialized)
    _resolved_provider: Optional[str] = field(default=None, repr=False, compare=False)
    _resolved_model: Optional[str] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errors: List[str] = []

        for section_name in ("llm",):
            section = getattr(self, section_name)
            if hasattr(section, "validate"):
                try:
                    section.validate()
                except ConfigError as e:
                    errors.append(f"{section_name}: {e}")

        if self.permissions.mode not in [m.value for m in PermissionMode]:
            errors.append(
                f"permissions.mode must be one of "
                f"{[m.value for m in PermissionMode]}"
            )
        if self.ui.theme not in [t.value for t in Theme]:
            errors.append(f"ui.theme must be one of {[t.value for t in Theme]}")
        if self.ui.editing_mode not in [e.value for e in EditingMode]:
            errors.append(
                f"ui.editing_mode must be one of "
                f"{[e.value for e in EditingMode]}"
            )
        if self.logging.level not in [l.value for l in LogLevel]:
            errors.append(
                f"logging.level must be one of "
                f"{[l.value for l in LogLevel]}"
            )
        if self.subagent.max_depth > self.planner.max_depth:
            errors.append(
                f"subagent.max_depth ({self.subagent.max_depth}) cannot exceed "
                f"planner.max_depth ({self.planner.max_depth})"
            )

        if errors:
            raise ConfigError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    # ------------------------------------------------------------------
    # DYNAMIC LLM RESOLUTION (all models come from llm/provider.py)
    # ------------------------------------------------------------------

    def resolve_llm(
        self,
        registry: Any = None,
        prefer_quality: bool = False,
    ) -> Tuple[str, str]:
        """
        Resolve (provider, model) using the LLM registry.

        The registry lives in `src/agent/llm/provider.py` and is the single
        source of truth for available models. NOTHING is hardcoded here.

        Behavior:
          1. If both provider and model are set → validate against registry.
          2. If provider set, model empty → pick best model from that provider.
          3. If both empty → pick best available model from any provider.
        """
        if registry is None:
            try:
                from agent.llm.provider import get_llm_registry
                registry = get_llm_registry()
            except Exception as e:
                logger.debug(f"LLM registry unavailable: {e}")
                registry = None

        provider = self.llm.provider or ""
        model = self.llm.model or ""

        # Case 1: Both set → validate against registry
        if provider and model:
            if registry and hasattr(registry, "is_model_available"):
                if registry.is_model_available(provider, model):
                    self._resolved_provider = provider
                    self._resolved_model = model
                    return provider, model
                logger.warning(
                    f"Configured model {provider}/{model} not found in registry; "
                    f"auto-selecting."
                )
            else:
                self._resolved_provider = provider
                self._resolved_model = model
                return provider, model

        # Case 2: Auto-select from registry
        selected = self._auto_select_model(registry, provider, prefer_quality)
        if selected:
            self._resolved_provider, self._resolved_model = selected
            logger.info(f"Auto-selected model: {selected[0]}/{selected[1]}")
            return selected

        # Case 3: No registry → ask registry for whatever it can give us
        if registry is None:
            raise ConfigError(
                "No LLM registry available. Implement "
                "agent.llm.provider.get_llm_registry() to provide models."
            )

        raise ConfigError(
            "LLM registry returned no usable models. "
            "Check that at least one provider is configured."
        )

    def _auto_select_model(
        self,
        registry: Any,
        provider_hint: str,
        prefer_quality: bool,
    ) -> Optional[Tuple[str, str]]:
        """
        Pick the best model dynamically from the registry.
        The registry format is intentionally flexible:

            { "provider_name": [ {id, name, quality, speed, ...}, ... ], ... }
        or
            [ {provider, id, name, quality, ...}, ... ]
        """
        if registry is None:
            return None

        # Preferred list first
        for pref in self.llm.preferred_models:
            if "/" in pref:
                p, m = pref.split("/", 1)
                if hasattr(registry, "is_model_available") and registry.is_model_available(p, m):
                    return p, m

        # Fetch all models from the registry
        all_models: Any = None
        for method in ("list_all_models", "list_models", "get_models"):
            if hasattr(registry, method):
                try:
                    all_models = getattr(registry, method)()
                    break
                except Exception as e:
                    logger.debug(f"{method}() failed: {e}")
                    continue

        if not all_models:
            return None

        # Normalize to List[(provider, model_id, meta)]
        candidates: List[Tuple[str, str, Dict]] = []
        if isinstance(all_models, dict):
            for prov, models in all_models.items():
                if provider_hint and prov != provider_hint:
                    continue
                for m in models:
                    if isinstance(m, str):
                        candidates.append((prov, m, {}))
                        continue
                    mid = m.get("id") or m.get("model") or m.get("name")
                    if mid:
                        candidates.append((prov, mid, m))
        elif isinstance(all_models, list):
            for m in all_models:
                if isinstance(m, str):
                    prov = provider_hint or "default"
                    candidates.append((prov, m, {}))
                    continue
                prov = m.get("provider", provider_hint or "default")
                if provider_hint and prov != provider_hint:
                    continue
                mid = m.get("id") or m.get("model") or m.get("name")
                if mid:
                    candidates.append((prov, mid, m))

        if not candidates:
            return None

        # Score each candidate
        def score(item: Tuple[str, str, Dict]) -> float:
            prov, mid, meta = item
            s = 0.0

            quality = (meta.get("quality") or "").lower()
            quality_scores = {
                "highest": 10, "very high": 8, "high": 7,
                "very good": 6, "good": 5, "medium": 4, "low": 2,
            }
            s += quality_scores.get(quality, 5)

            speed = (meta.get("speed") or "").lower()
            speed_scores = {
                "very fast": 5, "fast": 4, "medium": 3, "slow": 1,
            }
            s += speed_scores.get(speed, 2)

            ctx = meta.get("context_window") or 0
            if isinstance(ctx, int):
                s += min(5, ctx / 50_000)

            caps = set(meta.get("capabilities", []) or [])
            if "function_calling" in caps:
                s += 3
            if "vision" in caps:
                s += 1
            if "json_mode" in caps:
                s += 1

            if prefer_quality and quality in ("highest", "very high"):
                s *= 1.5

            cost_in = meta.get("cost_input") or 0
            cost_out = meta.get("cost_output") or 0
            if isinstance(cost_in, (int, float)):
                s -= cost_in * 0.05
            if isinstance(cost_out, (int, float)):
                s -= cost_out * 0.02

            return s

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        return best[0], best[1]

    # ------------------------------------------------------------------
    # DYNAMIC TEMPERATURE
    # ------------------------------------------------------------------

    def resolve_temperature(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Resolve the temperature for a given prompt.

        Priority:
          1. If `llm.params.temperature` is not None → return it (fixed).
          2. Otherwise → use TemperatureResolver (adaptive).
        """
        if self.llm.params.temperature is not None:
            return self.llm.params.temperature

        resolver = self.llm.adaptive_temperature.to_resolver()
        provider = provider or self._resolved_provider or self.llm.provider
        model = model or self._resolved_model or self.llm.model

        temp, reason = resolver.resolve(
            prompt=prompt,
            provider=provider,
            model=model,
            context=context,
        )
        logger.debug(f"Temperature resolved: {temp} ({reason})")
        return temp

    def resolve_temperature_with_reason(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """Same as resolve_temperature but returns the reason too."""
        if self.llm.params.temperature is not None:
            return self.llm.params.temperature, "fixed config override"
        resolver = self.llm.adaptive_temperature.to_resolver()
        provider = provider or self._resolved_provider or self.llm.provider
        model = model or self._resolved_model or self.llm.model
        return resolver.resolve(prompt, provider, model, context)

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        data = self._to_dict_internal(asdict(self))
        if not include_secrets:
            data = self._redact_secrets(data)
        return data

    @staticmethod
    def _to_dict_internal(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: Config._to_dict_internal(v) for k, v in d.items()}
        if isinstance(d, list):
            return [Config._to_dict_internal(v) for v in d]
        if isinstance(d, Enum):
            return d.value
        return d

    @staticmethod
    def _redact_secrets(d: Dict[str, Any]) -> Dict[str, Any]:
        SECRET_KEYS = {"api_key", "api_keys", "token", "password", "secret"}
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = Config._redact_secrets(v)
            elif k.lower() in SECRET_KEYS and v:
                if isinstance(v, str):
                    result[k] = _mask(v)
                elif isinstance(v, dict):
                    result[k] = {kk: _mask(vv) for kk, vv in v.items()}
                else:
                    result[k] = v
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Reconstruct a Config from a plain dict."""
        def build(section_cls, section_data):
            if section_data is None or not isinstance(section_data, dict):
                return section_cls()

            kwargs = {}
            for f in dc_fields(section_cls):
                if f.name not in section_data:
                    continue
                val = section_data[f.name]
                nested_cls = _resolve_nested_dataclass(section_cls, f.name)
                if nested_cls is not None and isinstance(val, dict):
                    kwargs[f.name] = build(nested_cls, val)
                else:
                    kwargs[f.name] = val
            return section_cls(**kwargs)

        return build(cls, data)


# ======================================================================
# HELPERS
# ======================================================================

def _mask(value: str) -> str:
    if not value or not isinstance(value, str):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "…" + value[-4:]


def _resolve_nested_dataclass(parent_cls: Any, field_name: str) -> Optional[type]:
    """Best-effort resolution of a nested dataclass type."""
    try:
        for f in dc_fields(parent_cls):
            if f.name == field_name:
                t = f.type
                if isinstance(t, str):
                    return globals().get(t)
                return t if isinstance(t, type) else None
    except Exception:
        return None
    return None


__all__ = [
    "Config",
    "LLMConfig",
    "LLMParamsConfig",
    "AdaptiveTemperatureConfig",
    "TemperatureResolver",
    "SessionConfig",
    "ContextConfig",
    "PermissionsConfig",
    "PermissionRuleConfig",
    "ToolsConfig",
    "PlannerConfig",
    "SubAgentConfig",
    "LoopConfig",
    "UIConfig",
    "StorageConfig",
    "WorkspaceConfig",
    "LoggingConfig",
    "MCPConfig",
    "PluginsConfig",
    "AdvancedConfig",
    "TaskComplexity",
    "LogLevel",
    "Theme",
    "EditingMode",
    "PermissionMode",
]