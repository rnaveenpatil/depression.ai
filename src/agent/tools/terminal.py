"""
Terminal Tool - Execute shell commands with safety, timeout, and streaming.

Long-running commands (dev servers, watchers, `flutter run`, etc.) are
detected and refused. Use the `process` tool to launch them in the
background instead — otherwise the TUI blocks forever waiting for a
process that never exits.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# Substrings that indicate a command will not exit on its own. If any of
# these appear in the command string, the tool refuses to run it and
# points the model at the `process` tool instead.
_LONG_RUNNING_PATTERNS: Tuple[str, ...] = (
    # Flutter / Dart
    "flutter run",
    "flutter run -d",
    "flutter run --",
    "dart run",
    # Node dev servers
    "npm run dev",
    "npm run start",
    "npm start",
    "yarn dev",
    "yarn start",
    "pnpm dev",
    "pnpm start",
    "bun dev",
    "bun run dev",
    # Common web frameworks
    "vite",
    "webpack serve",
    "webpack-dev-server",
    "next dev",
    "next start",
    "nuxt dev",
    "astro dev",
    "remix dev",
    "svelte-kit dev",
    # Python servers
    "uvicorn",
    "gunicorn",
    "flask run",
    "python -m http.server",
    "python3 -m http.server",
    "python -m flask run",
    "hypercorn",
    "daphne",
    # Watchers
    "nodemon",
    "watchmedo",
    "tsc --watch",
    "tsc -w",
    "cargo watch",
    "npm run watch",
    "yarn watch",
    # Generic
    " serve ",
    " dev-server",
    " dev_server",
    "tail -f",
    "watch ",
)


def _is_long_running(command: str) -> Optional[str]:
    """Return the matched pattern if the command looks long-running."""
    lowered = f" {command.lower()} "
    for pattern in _LONG_RUNNING_PATTERNS:
        if pattern in lowered:
            return pattern.strip()
    return None


class TerminalTool(BaseTool):
    name = "terminal"
    description = (
        "Execute a short-lived shell command and return its stdout/stderr. "
        "Use for builds, tests, linters, git, package managers, and any CLI "
        "tool that exits on its own. "
        "DO NOT use for servers or watchers (flutter run, npm run dev, vite, "
        "uvicorn, nodemon, etc.) — those never exit and will block the UI. "
        "Use the `process` tool with action='start' for long-running commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run. Must exit on its own within "
                    "`timeout` seconds. Long-running servers should use "
                    "the `process` tool instead."
                ),
            },
            "cwd": {
                "type": "string",
                "description": "Working directory (optional)",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout seconds (default 60)",
                "default": 60,
            },
            "shell": {
                "type": "boolean",
                "description": "Run via shell (default true)",
                "default": True,
            },
            "env": {
                "type": "object",
                "description": "Extra env vars",
            },
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
        extra = cfg.get("allowed_extra_dirs") or []
        self.allowed_extra_dirs = [
            str(Path(p).expanduser().resolve()) for p in extra
        ]
        self._system_temp_roots = tuple(
            {"/tmp", "/var/tmp", os.path.expanduser("~/.cache")}
        )

    # ------------------------------------------------------------------
    # PATH RESOLUTION
    # ------------------------------------------------------------------

    def _is_allowed_extra(self, path: str) -> bool:
        return any(
            path == root or path.startswith(root.rstrip("/") + "/")
            for root in self.allowed_extra_dirs
        )

    def _is_system_temp(self, path: str) -> bool:
        return any(
            path == root or path.startswith(root.rstrip("/") + "/")
            for root in self._system_temp_roots
        )

    def _resolve_cwd(self, cwd: str) -> Optional[str]:
        """Return a validated cwd, or None if the path is rejected."""
        try:
            if self.workspace:
                if self._is_system_temp(cwd) or self._is_allowed_extra(cwd):
                    p = Path(cwd).expanduser().resolve()
                    if not p.exists():
                        p.mkdir(parents=True, exist_ok=True)
                    return str(p)
                return str(self.workspace.assert_inside_workspace(cwd))

            p = Path(cwd).expanduser().resolve()
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            return str(p)
        except Exception as e:
            logger.warning("Rejected cwd %r: %s", cwd, e)
            return None

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = (params.get("command") or "").strip()
        if not command:
            return {"success": False, "error": "Empty command"}

        # Refuse long-running commands. Running `flutter run` or a dev
        # server here would block until the process exits (never) and
        # freeze the UI. Point the caller at the process tool instead.
        matched = _is_long_running(command)
        if matched:
            logger.info(
                "Refused long-running command in terminal tool: %r", command[:120]
            )
            return {
                "success": False,
                "error": (
                    f"This command looks long-running ('{matched}' detected). "
                    f"The `terminal` tool waits for the process to exit, which "
                    f"for a server is never. Use the `process` tool with "
                    f"action='start' instead."
                ),
                "long_running": True,
                "suggested_tool": "process",
                "suggested_params": {
                    "action": "start",
                    "command": command,
                    "cwd": params.get("cwd"),
                },
            }

        raw_cwd = params.get("cwd") or self.default_cwd
        cwd = self._resolve_cwd(raw_cwd)
        if cwd is None:
            return {
                "success": False,
                "error": f"Invalid cwd: {raw_cwd!r} (outside workspace and not allowlisted)",
                "command": command,
            }

        timeout = float(params.get("timeout", 60))
        use_shell = bool(params.get("shell", True))
        extra_env = params.get("env") or {}

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
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except Exception:
                    pass
                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "command": command,
                    "hint": (
                        "If this command is a server or watcher, use the "
                        "`process` tool with action='start' instead."
                    ),
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


__all__ = ["TerminalTool"]