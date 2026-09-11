"""
Utils Module - Cross-cutting utilities.

Exports:
    Errors:      AgentError and all subclasses (including AWS errors)
    Logging:     get_logger, setup_logging, get_recent_logs
    Platform:    OS/arch detection and compatibility helpers
"""

from agent.utils.errors import (
    # Base
    AgentError,
    # Config
    ConfigError,
    # Input
    InputError,
    # LLM
    LLMError,
    LLMRateLimitError,
    LLMAuthError,
    LLMTimeoutError,
    # Tools
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    # Permissions
    PermissionError as AgentPermissionError,
    PermissionDeniedError,
    # Context
    ContextError,
    # Session
    SessionError,
    # Storage
    StorageError,
    # MCP
    MCPError,
    MCPConnectionError,
    MCPTimeoutError,
    # Project
    WorkspaceError,
    GitError,
    # Planning
    PlanningError,
    ExecutionError,
    # Subagent
    SubAgentError,
    # Loop
    LoopError,
    TimeoutError as AgentTimeoutError,
    # Plugins
    PluginError,
    # AWS
    AWSError,
    AWSAuthError,
    AWSPermissionError,
    AWSNotFoundError,
    AWSValidationError,
    AWSThrottlingError,
    AWSServiceError,
    AWSNetworkError,
    AWSConfigError,
    # Helpers
    is_retryable,
    format_error,
    handle_exception,
)

from agent.utils.logging import (
    get_logger,
    setup_logging,
    get_recent_logs,
    set_log_level,
)

from agent.utils.platform import (
    Platform,
    get_platform,
    is_windows,
    is_macos,
    is_linux,
    get_shell,
    get_home_dir,
    get_config_dir,
    get_cache_dir,
    get_data_dir,
    which,
    normalize_path,
)

__all__ = [
    # Errors
    "AgentError", "ConfigError", "InputError",
    "LLMError", "LLMRateLimitError", "LLMAuthError", "LLMTimeoutError",
    "ToolError", "ToolNotFoundError", "ToolExecutionError", "ToolTimeoutError",
    "AgentPermissionError", "PermissionDeniedError",
    "ContextError", "SessionError", "StorageError",
    "MCPError", "MCPConnectionError", "MCPTimeoutError",
    "WorkspaceError", "GitError",
    "PlanningError", "ExecutionError", "SubAgentError", "LoopError",
    "AgentTimeoutError", "PluginError",
    "AWSError", "AWSAuthError", "AWSPermissionError", "AWSNotFoundError",
    "AWSValidationError", "AWSThrottlingError", "AWSServiceError",
    "AWSNetworkError", "AWSConfigError",
    "is_retryable", "format_error", "handle_exception",
    # Logging
    "get_logger", "setup_logging", "get_recent_logs", "set_log_level",
    # Platform
    "Platform", "get_platform", "is_windows", "is_macos", "is_linux",
    "get_shell", "get_home_dir", "get_config_dir", "get_cache_dir",
    "get_data_dir", "which", "normalize_path",
]