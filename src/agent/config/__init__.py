"""
Config Module - Schema, Defaults, Validation, and Dynamic LLM Resolution

KEY FEATURES:
- LLM models are NOT hardcoded — they are pulled dynamically from the
  LLM provider registry (src/agent/llm/provider.py).
- Temperature is resolved dynamically based on prompt complexity/task type.
- Full schema validation, environment variable interpolation, and layered
  merging of config sources.

Exports:
    Config                      — Root configuration object
    LLMConfig                   — LLM provider and model settings
    LLMParamsConfig             — Generation parameters
    AdaptiveTemperatureConfig   — Adaptive temperature settings
    TemperatureResolver         — Adaptive temperature resolver
    TaskComplexity              — Task complexity enum
    SessionConfig               — Session settings
    ContextConfig               — Context settings
    PermissionsConfig           — Permission settings
    ToolsConfig                 — Tool settings
    PlannerConfig               — Planner settings
    SubAgentConfig              — Sub-agent settings
    LoopConfig                  — Loop settings
    UIConfig                    — UI settings
    StorageConfig               — Storage settings
    WorkspaceConfig             — Workspace settings
    LoggingConfig               — Logging settings
    MCPConfig                   — MCP settings
    PluginsConfig               — Plugin settings
    AdvancedConfig              — Advanced feature flags
    LogLevel                    — Log level enum
    Theme                       — UI theme enum
    EditingMode                 — Editing mode enum
    PermissionMode              — Permission mode enum
    load_config                 — Load configuration from all sources
    get_config                  — Get current config object
    get_loader                  — Get config loader instance
    reset_config                — Reset global config
    resolve_llm_from_config     — Resolve provider/model dynamically
    resolve_temperature_from_config — Resolve adaptive temperature
"""

from agent.config.config import (
    Config,
    LLMConfig,
    LLMParamsConfig,
    AdaptiveTemperatureConfig,
    TemperatureResolver,
    SessionConfig,
    ContextConfig,
    PermissionsConfig,
    PermissionRuleConfig,
    ToolsConfig,
    PlannerConfig,
    SubAgentConfig,
    LoopConfig,
    UIConfig,
    StorageConfig,
    WorkspaceConfig,
    LoggingConfig,
    MCPConfig,
    PluginsConfig,
    AdvancedConfig,
    TaskComplexity,
    LogLevel,
    Theme,
    EditingMode,
    PermissionMode,
)

from agent.config.loader import (
    ConfigLoader,
    load_config,
    get_config,
    get_loader,
    reset_config,
    deep_merge,
    load_env_vars,
    load_file,
    interpolate_env,
    resolve_llm_from_config,
    resolve_temperature_from_config,
    resolve_temperature_with_reason_from_config,
    USER_CONFIG_PATH,
    USER_CONFIG_DIR,
)

__all__ = [
    # Config classes
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
    # Enums
    "TaskComplexity",
    "LogLevel",
    "Theme",
    "EditingMode",
    "PermissionMode",
    # Loader functions
    "ConfigLoader",
    "load_config",
    "get_config",
    "get_loader",
    "reset_config",
    "deep_merge",
    "load_env_vars",
    "load_file",
    "interpolate_env",
    "resolve_llm_from_config",
    "resolve_temperature_from_config",
    "resolve_temperature_with_reason_from_config",
    # Paths
    "USER_CONFIG_PATH",
    "USER_CONFIG_DIR",
]