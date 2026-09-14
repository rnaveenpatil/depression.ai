"""
Terminal Tool - Execute shell commands with safety, timeout, and streaming.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class TerminalTool(BaseTool):
    name = "terminal"
    description = (
        "Execute a shell command and return its stdout/stderr. "
        "Use for running builds, tests, git, package managers, and CLI tools."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
            "cwd": {"type": "string", "description": "Working directory (optional)"},
            "timeout": {"type": "number", "description": "Timeout seconds", "default": 60},
            "shell": {"type": "boolean", "description": "Run via shell", "default": True},
            "env": {"type": "object", "description": "Extra env vars"},
        },
        "required": ["command"],
    }
    timeout = 120.0

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        cfg = config or {}
        self.default_cwd = cfg.get("cwd") or (
            str(workspace.get_project_dir()) if workspace else os.getcwd()
        )
        self.default_shell = cfg.get("shell", "/bin/bash")
        self.max_output = cfg.get("max_output_size", 100_000)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = params.get("command", "").strip()
        if not command:
            return {"success": False, "error": "Empty command"}

        cwd = params.get("cwd") or self.default_cwd
        timeout = float(params.get("timeout", 60))
        use_shell = bool(params.get("shell", True))
        extra_env = params.get("env") or {}

        # Validate cwd — allow /tmp/opencode (OpenCode uses /tmp for tests) and allowed_extra_dirs
        try:
            if self.workspace:
                # Permit /tmp and system temp without boundary check
                if cwd.startswith("/tmp") or cwd.startswith(os.path.expanduser("~/.cache")):
                    cwd = str(Path(cwd).resolve())
                else:
                    cwd = str(self.workspace.assert_inside_workspace(cwd))
            else:
                cwd = str(Path(cwd).expanduser().resolve())
                if not Path(cwd).exists():
                    Path(cwd).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # Fallback to workspace dir instead of hard fail (more efficient UX)
            logger.warning(f"Invalid cwd {cwd}: {e}, falling back to project_dir")
            cwd = str(self.workspace.get_project_dir()) if self.workspace else os.getcwd()

        env = {**os.environ, **{str(k): str(v) for k, v in extra_env.items()}}

        t0 = time.time()
        try:
            if use_shell:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            else:
                parts = shlex.split(command)
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "command": command,
                }

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")

            truncated = False
            if len(out) > self.max_output:
                out = out[: self.max_output]
                truncated = True
            if len(err) > self.max_output:
                err = err[: self.max_output]
                truncated = True

            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "stdout": out,
                "stderr": err,
                "duration": time.time() - t0,
                "truncated": truncated,
                "command": command,
            }

        except FileNotFoundError as e:
            return {"success": False, "error": f"Command not found: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}