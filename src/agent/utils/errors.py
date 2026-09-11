"""
Errors Module - All custom exceptions for the agent.

Includes:
    - Base agent errors
    - Config / Input / LLM / Tool / Permission / Context / Session / Storage
    - MCP / Workspace / Git / Planning / SubAgent / Loop / Plugin
    - AWS errors with automatic classification from botocore responses
    - Retryability helpers
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# BASE
# ======================================================================

class AgentError(Exception):
    """Base class for all agent errors."""

    code: str = "agent_error"
    retryable: bool = False

    def __init__(self, message: str = "", **details):
        self.message = message or self.__class__.__name__
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


# ======================================================================
# CONFIG / INPUT
# ======================================================================

class ConfigError(AgentError):
    code = "config_error"


class InputError(AgentError):
    code = "input_error"


# ======================================================================
# LLM
# ======================================================================

class LLMError(AgentError):
    code = "llm_error"


class LLMRateLimitError(LLMError):
    code = "llm_rate_limit"
    retryable = True


class LLMAuthError(LLMError):
    code = "llm_auth_error"


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    retryable = True


# ======================================================================
# TOOLS
# ======================================================================

class ToolError(AgentError):
    code = "tool_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"


class ToolExecutionError(ToolError):
    code = "tool_execution_error"


class ToolTimeoutError(ToolError):
    code = "tool_timeout"
    retryable = True


# ======================================================================
# PERMISSIONS
# ======================================================================

class PermissionError(AgentError):
    """Base permission error. Note: distinct from builtins.PermissionError."""
    code = "permission_error"


class PermissionDeniedError(PermissionError):
    code = "permission_denied"


# ======================================================================
# CONTEXT / SESSION / STORAGE
# ======================================================================

class ContextError(AgentError):
    code = "context_error"


class SessionError(AgentError):
    code = "session_error"


class StorageError(AgentError):
    code = "storage_error"


# ======================================================================
# MCP
# ======================================================================

class MCPError(AgentError):
    code = "mcp_error"


class MCPConnectionError(MCPError):
    code = "mcp_connection_error"
    retryable = True


class MCPTimeoutError(MCPError):
    code = "mcp_timeout"
    retryable = True


# ======================================================================
# PROJECT
# ======================================================================

class WorkspaceError(AgentError):
    code = "workspace_error"


class GitError(AgentError):
    code = "git_error"


# ======================================================================
# PLANNING / EXECUTION / SUBAGENT / LOOP
# ======================================================================

class PlanningError(AgentError):
    code = "planning_error"


class ExecutionError(AgentError):
    code = "execution_error"


class SubAgentError(AgentError):
    code = "subagent_error"


class LoopError(AgentError):
    code = "loop_error"


class TimeoutError(AgentError):  # noqa: A001 - shadows builtin by design
    code = "timeout"
    retryable = True


# ======================================================================
# PLUGINS
# ======================================================================

class PluginError(AgentError):
    code = "plugin_error"


# ======================================================================
# AWS ERRORS
# ======================================================================

class AWSError(AgentError):
    """Base AWS error."""
    code = "aws_error"
    service: Optional[str] = None
    operation: Optional[str] = None
    aws_code: Optional[str] = None

    def __init__(
        self,
        message: str = "",
        service: Optional[str] = None,
        operation: Optional[str] = None,
        aws_code: Optional[str] = None,
        **details,
    ):
        super().__init__(message, **details)
        self.service = service
        self.operation = operation
        self.aws_code = aws_code

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "service": self.service,
            "operation": self.operation,
            "aws_code": self.aws_code,
        })
        return d


class AWSAuthError(AWSError):
    """Invalid, missing, or expired credentials."""
    code = "aws_auth_error"


class AWSPermissionError(AWSError):
    """IAM policy denies the action."""
    code = "aws_permission_error"


class AWSNotFoundError(AWSError):
    """Resource not found."""
    code = "aws_not_found"


class AWSValidationError(AWSError):
    """Invalid parameters or request."""
    code = "aws_validation_error"


class AWSThrottlingError(AWSError):
    """Rate limit exceeded."""
    code = "aws_throttling"
    retryable = True


class AWSServiceError(AWSError):
    """AWS-side error (5xx, internal)."""
    code = "aws_service_error"
    retryable = True


class AWSNetworkError(AWSError):
    """Network / connection issue."""
    code = "aws_network_error"
    retryable = True


class AWSConfigError(AWSError):
    """Missing or invalid configuration (region, profile, etc.)."""
    code = "aws_config_error"


# ======================================================================
# AWS CLASSIFICATION
# ======================================================================

# Map AWS error codes → our exception classes
AWS_ERROR_MAP = {
    # Auth
    "InvalidClientTokenId": AWSAuthError,
    "UnrecognizedClientException": AWSAuthError,
    "AuthFailure": AWSAuthError,
    "ExpiredToken": AWSAuthError,
    "ExpiredTokenException": AWSAuthError,
    "InvalidAccessKeyId": AWSAuthError,
    "SignatureDoesNotMatch": AWSAuthError,
    "TokenRefreshRequired": AWSAuthError,
    "InvalidSignatureException": AWSAuthError,
    "CredentialsError": AWSAuthError,
    "NoCredentialsError": AWSAuthError,
    "PartialCredentialsError": AWSAuthError,
    "SSOTokenLoadError": AWSAuthError,
    # Permission
    "AccessDenied": AWSPermissionError,
    "AccessDeniedException": AWSPermissionError,
    "UnauthorizedOperation": AWSPermissionError,
    "AuthorizationError": AWSPermissionError,
    "OperationNotPermitted": AWSPermissionError,
    "OptInRequired": AWSPermissionError,
    "SubscriptionRequiredException": AWSPermissionError,
    "AccessDeniedError": AWSPermissionError,
    # Not found
    "NoSuchBucket": AWSNotFoundError,
    "NoSuchKey": AWSNotFoundError,
    "NoSuchEntity": AWSNotFoundError,
    "ResourceNotFoundException": AWSNotFoundError,
    "NotFoundException": AWSNotFoundError,
    "QueueDoesNotExist": AWSNotFoundError,
    "TableNotFoundException": AWSNotFoundError,
    "DBInstanceNotFound": AWSNotFoundError,
    "ParameterNotFound": AWSNotFoundError,
    "ClusterNotFoundException": AWSNotFoundError,
    "InvalidVpcID.NotFound": AWSNotFoundError,
    "InvalidInstanceID.NotFound": AWSNotFoundError,
    "StackNotFoundException": AWSNotFoundError,
    "FunctionNotFound": AWSNotFoundError,
    "ResourceNotFoundException": AWSNotFoundError,
    # Validation
    "ValidationException": AWSValidationError,
    "ValidationError": AWSValidationError,
    "InvalidParameterValue": AWSValidationError,
    "InvalidParameterCombination": AWSValidationError,
    "MissingParameter": AWSValidationError,
    "InvalidRequest": AWSValidationError,
    "MalformedPolicyDocument": AWSValidationError,
    "InvalidInput": AWSValidationError,
    "InvalidQueryParameter": AWSValidationError,
    "BucketAlreadyExists": AWSValidationError,
    "BucketAlreadyOwnedByYou": AWSValidationError,
    "EntityAlreadyExists": AWSValidationError,
    "InvalidBucketName": AWSValidationError,
    "InvalidClientTokenId": AWSAuthError,  # duplicate guard
    # Throttling
    "Throttling": AWSThrottlingError,
    "ThrottlingException": AWSThrottlingError,
    "ThrottledException": AWSThrottlingError,
    "RequestLimitExceeded": AWSThrottlingError,
    "TooManyRequestsException": AWSThrottlingError,
    "RequestThrottled": AWSThrottlingError,
    "RequestThrottledException": AWSThrottlingError,
    "SlowDown": AWSThrottlingError,
    "LimitExceededException": AWSThrottlingError,
    "ProvisionedThroughputExceededException": AWSThrottlingError,
    # Service / internal
    "InternalError": AWSServiceError,
    "InternalFailure": AWSServiceError,
    "ServiceUnavailable": AWSServiceError,
    "ServiceException": AWSServiceError,
    "RequestTimeout": AWSServiceError,
    "RequestTimeoutException": AWSServiceError,
    "Unavailable": AWSServiceError,
    "PriorRequestNotComplete": AWSServiceError,
    "DependencyTimeout": AWSServiceError,
    "DependencyTimeoutException": AWSServiceError,
}

# Environment / config errors surfaced by botocore
AWS_CONFIG_CODES = {
    "NoRegionError",
    "RegionDisabledException",
    "ProfileNotFound",
    "ConfigParseError",
    "UnknownCredentialError",
    "PartialCredentialsError",
    "EndpointConnectionError",
}

# Errors whose message implies network issues
AWS_NETWORK_HINTS = [
    "Could not connect to the endpoint URL",
    "Connection refused",
    "Failed to establish a new connection",
    "Read timed out",
    "Connect timeout",
    "timed out",
    "Name or service not known",
    "Temporary failure in name resolution",
    "SSL",
    "unreachable",
]


def classify_aws_error(exc: Exception) -> AWSError:
    """
    Convert a botocore / boto3 / CLI exception into one of our AWSError
    subclasses, extracting service, operation, and AWS error code.
    """
    # If it's already one of ours, return as-is
    if isinstance(exc, AWSError):
        return exc

    aws_code = None
    service = None
    operation = None
    http_status = None
    request_id = None
    message = str(exc)

    # ---- botocore ClientError ----
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error", {}) or {}
        aws_code = err.get("Code") or err.get("code")
        message = err.get("Message") or err.get("message") or message
        meta = response.get("ResponseMetadata", {}) or {}
        http_status = meta.get("HTTPStatusCode")
        request_id = meta.get("RequestId")
        op = response.get("OperationName")
        svc = response.get("ServiceName")
        if op:
            operation = op
        if svc:
            service = svc

    # ---- CLI errors often embed the code in the message ----
    if not aws_code:
        m = re.search(r"An error occurred \(([A-Za-z0-9_.]+)\)", message)
        if m:
            aws_code = m.group(1)

    # ---- Determine class ----
    cls = AWSError

    if aws_code:
        if aws_code in AWS_CONFIG_CODES:
            cls = AWSConfigError
        elif aws_code in AWS_ERROR_MAP:
            cls = AWS_ERROR_MAP[aws_code]
        else:
            # Heuristic by prefix
            lower = aws_code.lower()
            if "throttl" in lower or "limit" in lower or "slowdown" in lower:
                cls = AWSThrottlingError
            elif "access" in lower or "denied" in lower or "unauthor" in lower:
                cls = AWSPermissionError
            elif "notfound" in lower or "nosuch" in lower:
                cls = AWSNotFoundError
            elif "auth" in lower or "credential" in lower or "token" in lower or "signature" in lower:
                cls = AWSAuthError
            elif "valid" in lower or "invalid" in lower or "malformed" in lower:
                cls = AWSValidationError
            elif "internalfailure" in lower or "internalerror" in lower or "unavailable" in lower:
                cls = AWSServiceError

    # ---- HTTP status fallbacks ----
    if cls is AWSError and http_status:
        if http_status in (401,):
            cls = AWSAuthError
        elif http_status == 403:
            cls = AWSPermissionError
        elif http_status == 404:
            cls = AWSNotFoundError
        elif http_status == 429:
            cls = AWSThrottlingError
        elif 500 <= http_status < 600:
            cls = AWSServiceError

    # ---- Network heuristics ----
    if cls is AWSError:
        msg_lower = message.lower()
        if any(h.lower() in msg_lower for h in AWS_NETWORK_HINTS):
            cls = AWSNetworkError

    # ---- Config heuristics ----
    if cls is AWSError:
        msg_lower = message.lower()
        if "region" in msg_lower and "specify" in msg_lower:
            cls = AWSConfigError
        elif "profile" in msg_lower and "not found" in msg_lower:
            cls = AWSConfigError

    # ---- Final fallback ----
    if cls is AWSError:
        cls = AWSServiceError

    err = cls(
        message=message,
        service=service,
        operation=operation,
        aws_code=aws_code,
    )
    err.details.update({
        "http_status": http_status,
        "request_id": request_id,
    })
    return err


# ======================================================================
# RETRYABILITY
# ======================================================================

RETRYABLE_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_KEYWORDS = (
    "timeout", "timed out", "connection reset", "connection refused",
    "temporarily unavailable", "try again", "throttl", "rate limit",
    "service unavailable", "internal error", "server error",
)


def is_retryable(exc: Exception) -> bool:
    """Decide whether an exception is worth retrying."""
    if isinstance(exc, AgentError):
        if exc.retryable:
            return True
        if isinstance(exc, AWSThrottlingError):
            return True
        status = exc.details.get("http_status") if isinstance(exc, AWSError) else None
        if status in RETRYABLE_HTTP_STATUS:
            return True
        return False

    # HTTPX / requests errors
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and code in RETRYABLE_HTTP_STATUS:
            return True

    # Message heuristics
    msg = str(exc).lower()
    return any(k in msg for k in RETRYABLE_KEYWORDS)


# ======================================================================
# FORMATTING / HANDLING
# ======================================================================

def format_error(exc: Exception, include_details: bool = False) -> str:
    """Format an exception into a human-readable string."""
    if isinstance(exc, AgentError):
        parts = [f"{exc.__class__.__name__}: {exc.message}"]
        if isinstance(exc, AWSError):
            bits = []
            if exc.service:
                bits.append(f"service={exc.service}")
            if exc.operation:
                bits.append(f"op={exc.operation}")
            if exc.aws_code:
                bits.append(f"code={exc.aws_code}")
            if bits:
                parts.append("(" + ", ".join(bits) + ")")
        if include_details and exc.details:
            parts.append(f"details={exc.details}")
        return " ".join(parts)

    return f"{type(exc).__name__}: {exc}"


def handle_exception(exc: Exception, reraise: bool = True) -> Optional[AgentError]:
    """
    Normalize any exception into an AgentError.
    If `reraise` is True, raises the wrapped error; otherwise returns it.
    """
    if isinstance(exc, AgentError):
        wrapped = exc
    else:
        # Detect AWS exceptions (botocore / boto3 / CLI)
        name = type(exc).__name__
        module = getattr(type(exc), "__module__", "")
        if (
            "botocore" in module
            or "boto3" in module
            or name in ("ClientError", "BotoCoreError", "NoCredentialsError",
                        "NoRegionError", "ProfileNotFound", "EndpointConnectionError")
        ):
            wrapped = classify_aws_error(exc)
        else:
            wrapped = AgentError(str(exc), original_type=name)

    if reraise:
        raise wrapped
    return wrapped


__all__ = [
    "AgentError", "ConfigError", "InputError",
    "LLMError", "LLMRateLimitError", "LLMAuthError", "LLMTimeoutError",
    "ToolError", "ToolNotFoundError", "ToolExecutionError", "ToolTimeoutError",
    "PermissionError", "PermissionDeniedError",
    "ContextError", "SessionError", "StorageError",
    "MCPError", "MCPConnectionError", "MCPTimeoutError",
    "WorkspaceError", "GitError",
    "PlanningError", "ExecutionError", "SubAgentError", "LoopError",
    "TimeoutError", "PluginError",
    "AWSError", "AWSAuthError", "AWSPermissionError", "AWSNotFoundError",
    "AWSValidationError", "AWSThrottlingError", "AWSServiceError",
    "AWSNetworkError", "AWSConfigError",
    "classify_aws_error", "is_retryable", "format_error", "handle_exception",
]