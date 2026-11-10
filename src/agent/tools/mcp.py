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
    description = (
        "Search, list, inspect, or call tools exposed by connected MCP servers. "
        "Large MCP catalogs are hidden until you search; matching tools are "
        "activated automatically so you can call them directly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "list", "call", "info"],
            },
            "query": {
                "type": "string",
                "description": "Keywords for search (e.g. 'jira issue create')",
            },
            "tool": {"type": "string", "description": "Tool name for call/info"},
            "arguments": {"type": "object", "description": "Arguments for call"},
            "server": {"type": "string", "description": "Filter by server"},
            "limit": {
                "type": "integer",
                "description": "Max results for list/search (default 10, max 50)",
            },
        },
        "required": ["action"],
    }
    timeout = 120.0

    read_only = False
    mutating = True
    category = "misc"

    def __init__(self, mcp_client: Any = None):
        self.mcp_client = mcp_client

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.mcp_client is None:
            return {"success": False, "error": "MCP client not available"}

        action = params.get("action", "search")
        server = params.get("server")
        limit = params.get("limit", 10)
        query = params.get("query", "")
        total = len(getattr(self.mcp_client, "tools", {}) or {})

        if action in ("search", "list"):
            results = self.mcp_client.search_tools(
                query=query if action == "search" else "",
                server=server,
                limit=limit,
            )
            activated: List[str] = []
            if action == "search" and query:
                count = min(3, max(1, int(limit or 10)))
                activated = self.mcp_client.activate_tools(
                    [item["name"] for item in results[:count]]
                )
            return {
                "success": True,
                "tools": results,
                "total_tools": total,
                "activated": activated,
                "note": (
                    "Matching tools are now directly callable. Use action=info "
                    "to reveal one tool's full schema."
                ),
            }

        if action == "info":
            name = params.get("tool")
            if not name:
                return {"success": False, "error": "info requires 'tool'"}
            schema = self.mcp_client.get_tool_schema(name)
            if not schema:
                return {"success": False, "error": f"Unknown MCP tool: {name}"}
            activated = self.mcp_client.activate_tool(name)
            return {
                "success": True,
                "schema": schema,
                "activated": activated,
                "total_tools": total,
            }

        if action == "call":
            name = params.get("tool")
            args = params.get("arguments", {})
            if not name:
                return {"success": False, "error": "call requires 'tool'"}
            resolved = self.mcp_client.activate_tool(name)
            if not resolved:
                return {"success": False, "error": f"Unknown MCP tool: {name}"}
            return await self.mcp_client.call_tool(resolved, args)

        return {"success": False, "error": f"Unknown action: {action}"}