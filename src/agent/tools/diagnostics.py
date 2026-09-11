"""
Diagnostics Tool - Errors, lint, and compiler diagnostics.

Runs linters / type checkers / compilers and returns structured diagnostics.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class DiagnosticsTool(BaseTool):
    name = "diagnostics"
    description = "Run linters, type checkers, and compilers to surface errors and warnings."
    parameters = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": ["auto", "ruff", "flake8", "mypy", "pyright", "eslint", "tsc", "cargo", "go", "pylint"],
                "default": "auto",
            },
            "path": {"type": "string", "description": "File or directory to analyze"},
            "args": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "number", "default": 60},
        },
        "required": [],
    }
    timeout = 120.0

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        self.cwd = str(workspace.get_project_dir()) if workspace else os.getcwd()

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        tool = params.get("tool", "auto")
        if tool == "auto":
            tool = self._detect_tool()

        cmd = self._build_command(tool, params)
        if not cmd:
            return {"success": False, "error": f"No diagnostics backend available for '{tool}'"}

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(params.get("timeout", 60))
            )
        except asyncio.TimeoutError:
            return {"success": False, "error": "diagnostics timed out"}
        except FileNotFoundError:
            return {"success": False, "error": f"Command not found: {cmd[0]}"}

        return {
            "success": proc.returncode in (0, 1),  # most linters exit 1 on findings
            "tool": tool,
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    def _detect_tool(self) -> str:
        if (os.path.isdir(os.path.join(self.cwd, "src")) or
                any(f.endswith(".py") for f in os.listdir(self.cwd) if os.path.isfile(os.path.join(self.cwd, f)))):
            if shutil.which("ruff"):
                return "ruff"
            if shutil.which("flake8"):
                return "flake8"
            if shutil.which("mypy"):
                return "mypy"
        if os.path.exists(os.path.join(self.cwd, "tsconfig.json")):
            return "tsc"
        if os.path.exists(os.path.join(self.cwd, "package.json")):
            return "eslint"
        if os.path.exists(os.path.join(self.cwd, "Cargo.toml")):
            return "cargo"
        if os.path.exists(os.path.join(self.cwd, "go.mod")):
            return "go"
        return "ruff"

    def _build_command(self, tool: str, params: Dict[str, Any]) -> Optional[List[str]]:
        path = params.get("path") or "."
        extra = params.get("args") or []

        table = {
            "ruff": ["ruff", "check", path, "--output-format", "concise"],
            "flake8": ["flake8", path],
            "mypy": ["mypy", path],
            "pyright": ["pyright", path],
            "pylint": ["pylint", path],
            "eslint": ["npx", "eslint", path],
            "tsc": ["npx", "tsc", "--noEmit"],
            "cargo": ["cargo", "check"],
            "go": ["go", "vet", path],
        }
        cmd = table.get(tool)
        if cmd is None:
            return None
        return cmd + list(extra)