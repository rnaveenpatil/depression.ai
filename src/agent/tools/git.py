"""
Git Tool - Wrapper around git commands for the agent.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class GitTool(BaseTool):
    name = "git"
    description = "Inspect and manipulate git repositories (status, diff, log, add, commit, branch)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status", "diff", "log", "show", "branch",
                    "add", "commit", "checkout", "pull", "push",
                    "stash", "reset", "blame", "remote",
                ],
            },
            "path": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
            "ref": {"type": "string"},
            "staged": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "default": 20},
        },
        "required": ["action"],
    }
    timeout = 60.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        self.cwd = (
            str(workspace.get_project_dir()) if workspace else os.getcwd()
        )

    async def _git(self, *args: str, timeout: float = 30.0) -> Dict[str, Any]:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "error": "git command timed out"}
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "").lower()
        args: List[str] = []

        if action == "status":
            args = ["status", "--short", "--branch"]
        elif action == "diff":
            args = ["diff"]
            if params.get("staged"):
                args.append("--staged")
            if params.get("path"):
                args.extend(["--", params["path"]])
        elif action == "log":
            args = ["log", f"--max-count={int(params.get('limit', 20))}", "--oneline"]
            if params.get("path"):
                args.extend(["--", params["path"]])
        elif action == "show":
            args = ["show", params.get("ref", "HEAD")]
        elif action == "branch":
            args = ["branch", "-a"]
        elif action == "add":
            args = ["add"] + (params.get("args") or ["."])
        elif action == "commit":
            msg = params.get("message", "")
            if not msg:
                return {"success": False, "error": "commit requires 'message'"}
            args = ["commit", "-m", msg]
        elif action == "checkout":
            ref = params.get("ref") or (params.get("args") or [None])[0]
            if not ref:
                return {"success": False, "error": "checkout requires 'ref'"}
            args = ["checkout", ref]
        elif action == "pull":
            args = ["pull"] + (params.get("args") or [])
        elif action == "push":
            args = ["push"] + (params.get("args") or [])
        elif action == "stash":
            args = ["stash"] + (params.get("args") or [])
        elif action == "reset":
            args = ["reset"] + (params.get("args") or [])
        elif action == "blame":
            if not params.get("path"):
                return {"success": False, "error": "blame requires 'path'"}
            args = ["blame", params["path"]]
        elif action == "remote":
            args = ["remote", "-v"]
        else:
            return {"success": False, "error": f"Unknown action: {action}"}

        result = await self._git(*args, timeout=self.timeout)
        result["action"] = action
        result["args"] = args
        return result