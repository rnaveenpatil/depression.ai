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
]