"""
MCP Module - Model Context Protocol Integration

Exports:
    MCPClient         — High-level MCP client
    MCPPresets        — Curated server presets
    MCPServerConfig   — Server configuration
    MCPTool           — Exposed tool
    MCPResource       — Exposed resource
    MCPPrompt         — Exposed prompt
    MCPTransport      — Transport type enum
    MCPState          — Connection state enum
    StdioTransport    — Local stdio transport
    HTTPTransport     — HTTP/SSE transport
    CloudTransport    — Cloud transport

AWS helpers:
    build_aws_mcp_config         — Build AWS server config from .env
    build_aws_cli_fallback_status — Report CLI fallback availability
    install_aws_preset_into_config — Add AWS server to an MCP config
"""

from agent.mcp.client import (
    MCPClient,
    MCPPresets,
    MCPServerConfig,
    MCPTool,
    MCPResource,
    MCPPrompt,
    MCPTransport,
    MCPState,
    StdioTransport,
    HTTPTransport,
    CloudTransport,
    cloud_mcp_server,
    stdio_mcp_server,
    sse_mcp_server,
    http_mcp_server,
)

from agent.mcp.aws_config import (
    build_aws_mcp_config,
    build_aws_cli_fallback_status,
    install_aws_preset_into_config,
    AWS_MCP_PACKAGE,
    DEFAULT_AWS_REGION,
)

__all__ = [
    "MCPClient",
    "MCPPresets",
    "MCPServerConfig",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPTransport",
    "MCPState",
    "StdioTransport",
    "HTTPTransport",
    "CloudTransport",
    "cloud_mcp_server",
    "stdio_mcp_server",
    "sse_mcp_server",
    "http_mcp_server",
    "build_aws_mcp_config",
    "build_aws_cli_fallback_status",
    "install_aws_preset_into_config",
    "AWS_MCP_PACKAGE",
    "DEFAULT_AWS_REGION",
]