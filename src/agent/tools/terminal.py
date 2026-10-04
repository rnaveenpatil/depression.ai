"""
Terminal Tool - Execute shell commands with safety, timeout, and streaming.

Long-running commands (dev servers, watchers, `flutter run`, etc.) are
detected and refused. Use the `process` tool to launch them in the
background instead — otherwise the TUI blocks forever waiting for a
process that never exits.

Results are always a dict with `success`. On failure the dict includes
`exit_code`, `stdout`, `stderr`, `command`, `cwd`, and `recoverable`.

Env policy:
    * os.environ is copied, then overlayed with EnvManager.get_aws_env(),
      then overlayed with per-call env. This keeps subprocesses in sync
      with credentials saved mid-session.
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
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


# Substrings that indicate a command will not exit on its own.
_LONG_RUNNING_PATTERNS: Tuple[str, ...] = (
    "flutter run", "dart run",
    "npm run dev", "npm run start", "npm start",
    "yarn dev", "yarn start",
    "pnpm dev", "pnpm start",
    "bun dev", "bun run dev",
    "vite", "webpack serve", "webpack-dev-server",
    "next dev", "next start",
    "nuxt dev", "astro dev", "remix dev", "svelte-kit dev",
    "uvicorn", "gunicorn", "flask run",
    "python -m http.server", "python3 -m http.server",
    "python -m flask run", "hypercorn", "daphne",
    "nodemon", "watchmedo",
    "tsc --watch", "tsc -w",
    "cargo watch", "npm run watch", "yarn watch",
    " serve ", " dev-server", " dev_server",
    "tail -f", "watch ",
)


def _is_long_running(command: str) -> Optional[str]:
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
            "cwd": {"type": "string", "description": "Working directory (optional)"},
            "timeout": {"type": "number", "description": "Timeout seconds (default 60)", "default": 60},
            "shell": {"type": "boolean", "description": "Run via shell (default true)", "default": True},
            "env": {"type": "object", "description": "Extra env vars"},
        },
        "required": ["command"],
    }
    timeout = 120.0

    read_only = False
    mutating = True
    category = "run"

    def __init__(self, workspace: Any = None, config: Optional[Dict[str, Any]] = None):
        self.workspace = workspace
        cfg = config or {}
        self.default_cwd = cfg.get("cwd") or (
            str(workspace.get_project_dir()) if workspace else os.getcwd()
        )
        self.default_shell = cfg.get("shell", "/bin/bash")
        self.max_output = int(cfg.get("max_output_size", 100_000))
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
    # ENV
    # ------------------------------------------------------------------

    def _build_env(self, extra: Dict[str, Any]) -> Dict[str, str]:
        env = {k: str(v) for k, v in os.environ.items()}
        try:
            env.update(EnvManager.get().get_aws_env())
        except Exception:
            pass
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(b: bytes, cap: int) -> Tuple[str, bool]:
        s = b.decode(errors="replace")
        if len(s) > cap:
            return s[:cap], True
        return s, False

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = (params.get("command") or "").strip()
        if not command:
            return {"success": False, "error": "Empty command",
                    "recoverable": True}

        matched = _is_long_running(command)
        if matched:
            logger.info("Refused long-running command: %r", command[:120])
            return {
                "success": False,
                "error": (
                    f"This command looks long-running ('{matched}' detected). "
                    f"The `terminal` tool waits for the process to exit, which "
                    f"for a server is never. Use the `process` tool with "
                    f"action='start' instead."
                ),
                "long_running": True,
                "recoverable": True,
                "suggested_tool": "process",
                "suggested_params": {
                    "action": "start",
                    "command": command,
                    "cwd": params.get("cwd"),
                },
                "command": command,
            }

        raw_cwd = params.get("cwd") or self.default_cwd
        cwd = self._resolve_cwd(raw_cwd)
        if cwd is None:
            return {
                "success": False,
                "error": f"Invalid cwd: {raw_cwd!r} (outside workspace and not allowlisted)",
                "recoverable": False,
                "command": command,
            }

        timeout = float(params.get("timeout", 60))
        use_shell = bool(params.get("shell", True))
        env = self._build_env(params.get("env") or {})

        t0 = time.time()
        proc = None
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
                # Capture whatever was produced before killing.
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                partial_out, partial_err = b"", b""
                try:
                    partial_out, partial_err = await asyncio.wait_for(
                        proc.communicate(), timeout=2
                    )
                except Exception:
                    try:
                        if proc.stdout:
                            partial_out = await asyncio.wait_for(
                                proc.stdout.read(), timeout=0.5
                            )
                    except Exception:
                        pass
                    try:
                        if proc.stderr:
                            partial_err = await asyncio.wait_for(
                                proc.stderr.read(), timeout=0.5
                            )
                    except Exception:
                        pass

                out_s, out_trunc = self._decode(partial_out, self.max_output)
                err_s, err_trunc = self._decode(partial_err, self.max_output)

                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "timed_out": True,
                    "recoverable": True,
                    "command": command,
                    "cwd": cwd,
                    "duration": time.time() - t0,
                    "stdout": out_s,
                    "stderr": err_s,
                    "truncated": out_trunc or err_trunc,
                    "exit_code": None,
                    "hint": (
                        "If this command is a server or watcher, use the "
                        "`process` tool with action='start' instead."
                    ),
                }

            out, out_trunc = self._decode(stdout, self.max_output)
            err, err_trunc = self._decode(stderr, self.max_output)
            exit_code = proc.returncode

            return {
                "success": exit_code == 0,
                "exit_code": exit_code,
                "stdout": out,
                "stderr": err,
                "duration": time.time() - t0,
                "truncated": out_trunc or err_trunc,
                "command": command,
                "cwd": cwd,
                "recoverable": exit_code != 0,
            }

        except FileNotFoundError as e:
            return {"success": False, "error": f"Command not found: {e}",
                    "recoverable": True, "command": command, "cwd": cwd}
        except PermissionError as e:
            return {"success": False, "error": f"Permission denied: {e}",
                    "recoverable": False, "command": command, "cwd": cwd}
        except Exception as e:
            return {"success": False, "error": str(e),
                    "recoverable": True, "command": command, "cwd": cwd}


__all__ = ["TerminalTool"]