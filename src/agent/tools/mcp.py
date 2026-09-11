"""
MCP Tool - Bridge to the MCP client.

Exposes MCP tools to the tool registry so the LLM can invoke them
with the same interface as native tools.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class MCPTool(BaseTool):
    name = "mcp"
    description = "List, inspect, or call tools exposed by connected MCP servers."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "call", "info"]},
            "tool": {"type": "string", "description": "Tool name for call/info"},
            "arguments": {"type": "object", "description": "Arguments for call"},
            "server": {"type": "string", "description": "Filter by server"},
        },
        "required": ["action"],
    }
    timeout = 120.0

    def __init__(self, mcp_client: Any = None):
        self.mcp_client = mcp_client

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.mcp_client is None:
            return {"success": False, "error": "MCP client not available"}

        action = params.get("action", "list")
        if action == "list":
            server = params.get("server")
            tools = self.mcp_client.list_tools()
            if server:
                tools = [t for t in tools if t["function"]["name"].startswith(f"mcp__{server}__")]
            return {
                "success": True,
                "tools": [
                    {"name": t["function"]["name"], "description": t["function"]["description"]}
                    for t in tools
                ],
            }
        if action == "info":
            name = params.get("tool")
            if not name:
                return {"success": False, "error": "info requires 'tool'"}
            schema = self.mcp_client.get_tool_schema(name)
            if not schema:
                return {"success": False, "error": f"Unknown MCP tool: {name}"}
            return {"success": True, "schema": schema}
        if action == "call":
            name = params.get("tool")
            args = params.get("arguments", {})
            if not name:
                return {"success": False, "error": "call requires 'tool'"}
            return await self.mcp_client.call_tool(name, args)
        return {"success": False, "error": f"Unknown action: {action}"}