"""
Process Tool - Start, list, inspect, and manage long-running background processes.

Lifecycle:
    * start  -> spawns a child process; assigns a short tool-managed id
    * status -> refreshes OS state; never reports a stale returncode
    * logs   -> returns captured stdout/stderr (bounded, byte-capped)
    * wait   -> waits for exit, or returns timed_out=True with partial output
    * stop   -> graceful terminate, then kill after timeout
    * kill   -> immediate SIGKILL

The registry keeps exited processes around so their logs can still be read
until the user explicitly stops them or the tool shuts down.

Env policy:
    Child env = os.environ + EnvManager.get_aws_env() + optional per-call
    env. This keeps long-running servers in sync with credentials saved
    mid-session (AWS, etc.).
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger
from agent.utils.env_manager import EnvManager

logger = get_logger(__name__)


# Maximum number of stdout/stderr lines retained per process.
_MAX_BUFFER_LINES = 5000

# Byte cap for what the logs action returns in a single call.
_MAX_LOG_BYTES = 32 * 1024

# How long we wait after terminate() before escalating to kill().
_TERMINATE_GRACE_SECONDS = 5.0


@dataclass
class ManagedProcess:
    id: str
    command: str
    pid: int
    proc: asyncio.subprocess.Process
    cwd: str = ""
    started_at: float = field(default_factory=time.time)
    exited_at: Optional[float] = None
    exit_code: Optional[int] = None
    stdout: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    stderr: Deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_BUFFER_LINES))
    _pumps: List[asyncio.Task] = field(default_factory=list)
    _watcher: Optional[asyncio.Task] = None

    @property
    def running(self) -> bool:
        return self.proc.returncode is None

    def refresh(self) -> None:
        """Pull fresh returncode from the OS-backed Process object."""
        rc = self.proc.returncode
        if rc is not None and self.exit_code is None:
            self.exit_code = rc
            self.exited_at = time.time()

    def status(self) -> str:
        self.refresh()
        if self.running:
            return "running"
        return f"exited({self.exit_code})"


class ProcessTool(BaseTool):
    name = "process"
    description = (
        "Start, list, inspect, and stop background processes "
        "(dev servers, watchers, etc.). Use this for anything that does "
        "not exit on its own. Returns lifecycle state and captured output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "list", "status", "stop", "kill", "logs", "wait"],
            },
            "command": {"type": "string", "description": "For start"},
            "cwd": {"type": "string"},
            "env": {
                "type": "object",
                "description": "Extra env vars for the child process",
            },
            "process_id": {
                "type": "string",
                "description": "Tool-managed process id returned by 'start' (not the OS pid).",
            },
            "lines": {"type": "integer", "default": 100},
            "timeout": {"type": "number", "default": 30},
        },
        "required": ["action"],
    }
    timeout = 999999.0

    read_only = False
    mutating = True
    category = "proc"

    def __init__(self, workspace: Any = None):
        self.workspace = workspace
        try:
            if workspace is not None:
                self.cwd = str(Path(str(workspace.get_project_dir())).resolve())
            else:
                self.cwd = str(Path.cwd().resolve())
        except Exception:
            self.cwd = os.getcwd()
        self.processes: Dict[str, ManagedProcess] = {}

    # ------------------------------------------------------------------
    # ENV
    # ------------------------------------------------------------------

    def _build_env(self, extra: Optional[Dict[str, Any]]) -> Dict[str, str]:
        env = {k: str(v) for k, v in os.environ.items()}
        try:
            env.update(EnvManager.get().get_aws_env())
        except Exception:
            pass
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env

    # ------------------------------------------------------------------
    # ENTRY
    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "")
        if action == "start":
            return await self._start(params)
        if action == "list":
            return self._list()
        if action == "status":
            return self._status(params)
        if action in ("stop", "kill"):
            return await self._stop(params, force=(action == "kill"))
        if action == "logs":
            return self._logs(params)
        if action == "wait":
            return await self._wait(params)
        return {"success": False, "error": f"Unknown action: {action}",
                "recoverable": True}

    # ------------------------------------------------------------------
    # START
    # ------------------------------------------------------------------

    async def _start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        command = (params.get("command") or "").strip()
        if not command:
            return {"success": False, "error": "start requires 'command'",
                    "recoverable": True}

        raw_cwd = params.get("cwd") or self.cwd
        try:
            cwd = str(Path(raw_cwd).expanduser().resolve())
        except Exception as e:
            return {"success": False, "error": f"Invalid cwd: {e}",
                    "recoverable": False}
        if not Path(cwd).exists():
            return {"success": False, "error": f"cwd does not exist: {cwd}",
                    "recoverable": False}

        env = self._build_env(params.get("env"))

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return {"success": False, "error": f"Command not found: {e}",
                    "recoverable": True, "command": command, "cwd": cwd}
        except PermissionError as e:
            return {"success": False, "error": f"Permission denied: {e}",
                    "recoverable": False, "command": command, "cwd": cwd}
        except Exception as e:
            return {"success": False, "error": str(e),
                    "recoverable": True, "command": command, "cwd": cwd}

        pid = str(uuid.uuid4())[:8]
        mp = ManagedProcess(
            id=pid, command=command, pid=proc.pid or 0, proc=proc, cwd=cwd,
        )
        self.processes[pid] = mp

        async def pump(stream: Any, buf: Deque[str], tag: str) -> None:
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    buf.append(line.decode(errors="replace").rstrip())
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("process pump %s error: %s", tag, e)

        mp._pumps = [
            asyncio.create_task(pump(proc.stdout, mp.stdout, "out")),
            asyncio.create_task(pump(proc.stderr, mp.stderr, "err")),
        ]

        async def watch() -> None:
            """Background watcher: records exit code as soon as it happens."""
            try:
                rc = await proc.wait()
                mp.exit_code = rc
                mp.exited_at = time.time()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("process watcher error: %s", e)

        mp._watcher = asyncio.create_task(watch())

        return {
            "success": True,
            "process_id": pid,
            "os_pid": proc.pid,
            "command": command,
            "cwd": cwd,
            "status": "running",
            "note": "Use process_id (not os_pid) with the 'process' tool.",
        }

    # ------------------------------------------------------------------
    # LIST / STATUS
    # ------------------------------------------------------------------

    def _list(self) -> Dict[str, Any]:
        out = []
        for mp in self.processes.values():
            mp.refresh()
            out.append({
                "id": mp.id,
                "os_pid": mp.pid,
                "command": mp.command,
                "cwd": mp.cwd,
                "status": mp.status(),
                "running": mp.running,
                "exit_code": mp.exit_code,
                "started_at": mp.started_at,
                "exited_at": mp.exited_at,
                "uptime": (
                    (mp.exited_at or time.time()) - mp.started_at
                    if mp.exited_at else time.time() - mp.started_at
                ),
            })
        return {"success": True, "processes": out, "count": len(out)}

    def _get(self, params: Dict[str, Any]) -> Optional[ManagedProcess]:
        pid = params.get("process_id")
        if not pid:
            return None
        return self.processes.get(pid)

    def _status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
                "recoverable": True,
                "suggestion": "Call action='list' to see live process ids.",
            }
        mp.refresh()
        return {
            "success": True,
            "id": mp.id,
            "os_pid": mp.pid,
            "command": mp.command,
            "cwd": mp.cwd,
            "status": mp.status(),
            "running": mp.running,
            "exit_code": mp.exit_code,
            "started_at": mp.started_at,
            "exited_at": mp.exited_at,
            "uptime": (
                (mp.exited_at or time.time()) - mp.started_at
                if mp.exited_at else time.time() - mp.started_at
            ),
            "stdout_lines": len(mp.stdout),
            "stderr_lines": len(mp.stderr),
        }

    # ------------------------------------------------------------------
    # STOP / KILL
    # ------------------------------------------------------------------

    async def _stop(
        self, params: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
                "recoverable": True,
                "suggestion": "Call action='list' to see live process ids.",
            }

        if mp.running:
            try:
                if force:
                    mp.proc.kill()
                else:
                    mp.proc.terminate()
            except ProcessLookupError:
                pass
            except Exception as e:
                logger.debug("stop signal failed: %s", e)

            try:
                await asyncio.wait_for(mp.proc.wait(), timeout=_TERMINATE_GRACE_SECONDS)
            except asyncio.TimeoutError:
                try:
                    mp.proc.kill()
                    await asyncio.wait_for(mp.proc.wait(), timeout=2)
                except Exception:
                    pass

        mp.refresh()

        # Cancel pumps and watcher. Pumps may already be done naturally if
        # the process exited on its own and closed its pipes.
        for t in list(mp._pumps):
            t.cancel()
        for t in list(mp._pumps):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if mp._watcher is not None:
            mp._watcher.cancel()
            try:
                await mp._watcher
            except (asyncio.CancelledError, Exception):
                pass

        # Remove from the live registry.
        self.processes.pop(mp.id, None)

        return {
            "success": True,
            "id": mp.id,
            "os_pid": mp.pid,
            "command": mp.command,
            "killed": force,
            "exit_code": mp.exit_code,
            "status": "stopped",
        }

    # ------------------------------------------------------------------
    # LOGS
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_lines(lines: List[str], max_bytes: int) -> str:
        text = "\n".join(lines)
        data = text.encode("utf-8")
        if len(data) <= max_bytes:
            return text
        # Keep the TAIL (most recent), and drop a marker at the top.
        tail = data[-max_bytes:].decode("utf-8", errors="replace")
        return f"… [truncated {len(data) - max_bytes} bytes] …\n{tail}"

    def _logs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
                "recoverable": True,
            }
        mp.refresh()
        n = int(params.get("lines", 100))
        n = max(1, min(n, _MAX_BUFFER_LINES))

        stdout_text = self._cap_lines(list(mp.stdout)[-n:], _MAX_LOG_BYTES)
        stderr_text = self._cap_lines(list(mp.stderr)[-n:], _MAX_LOG_BYTES)

        return {
            "success": True,
            "id": mp.id,
            "os_pid": mp.pid,
            "command": mp.command,
            "status": mp.status(),
            "running": mp.running,
            "exit_code": mp.exit_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

    # ------------------------------------------------------------------
    # WAIT
    # ------------------------------------------------------------------

    async def _wait(self, params: Dict[str, Any]) -> Dict[str, Any]:
        mp = self._get(params)
        if not mp:
            return {
                "success": False,
                "error": f"Process {params.get('process_id')!r} not found",
                "recoverable": True,
            }
        timeout = float(params.get("timeout", 30))
        try:
            code = await asyncio.wait_for(mp.proc.wait(), timeout=timeout)
            mp.refresh()
            return {
                "success": True,
                "id": mp.id,
                "os_pid": mp.pid,
                "exit_code": code,
                "status": mp.status(),
                "stdout": self._cap_lines(list(mp.stdout)[-100:], _MAX_LOG_BYTES),
                "stderr": self._cap_lines(list(mp.stderr)[-100:], _MAX_LOG_BYTES),
            }
        except asyncio.TimeoutError:
            mp.refresh()
            return {
                "success": False,
                "error": f"process still running after {timeout}s",
                "timed_out": True,
                "recoverable": True,
                "id": mp.id,
                "os_pid": mp.pid,
                "status": mp.status(),
                "stdout": self._cap_lines(list(mp.stdout)[-100:], _MAX_LOG_BYTES),
                "stderr": self._cap_lines(list(mp.stderr)[-100:], _MAX_LOG_BYTES),
            }

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Idempotent; safe to call multiple times."""
        ids = list(self.processes.keys())
        for pid in ids:
            try:
                await self._stop({"process_id": pid}, force=True)
            except Exception as e:
                logger.debug("process shutdown error for %s: %s", pid, e)
        self.processes.clear()


__all__ = ["ProcessTool"]