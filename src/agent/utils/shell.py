"""Shared async shell execution helpers.

Every child is detached from the TUI terminal (``stdin=DEVNULL`` plus a new
session) so interactive commands cannot steal keystrokes or put the terminal
into raw mode. Timeouts and cancellation terminate the whole process group,
not just the shell that spawned it.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

OutputCallback = Callable[[str, str], None]

SIGTERM = signal.SIGTERM
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


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


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    truncated: bool = False
    duration: float = 0.0
    pid: Optional[int] = None


def is_long_running(command: str) -> Optional[str]:
    """Return the matched pattern when *command* is likely to stay alive."""
    lowered = f" {command.lower()} "
    for pattern in _LONG_RUNNING_PATTERNS:
        if pattern in lowered:
            return pattern.strip()
    return None


def signal_process_group(proc: Any, sig: int = SIGTERM) -> None:
    """Signal the child's whole process group when possible."""
    if proc is None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, sig)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


async def terminate_process(proc: Any, *, force: bool = False) -> None:
    """Terminate *proc* and its descendants, escalating to SIGKILL."""
    if proc is None:
        return
    if not force:
        signal_process_group(proc, SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
    # Always SIGKILL the group on timeout/cancel: the shell may exit before
    # its children, and those children still hold the TUI's terminal.
    signal_process_group(proc, SIGKILL)
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
    except Exception:
        pass


async def _pump(
    stream: Any,
    tag: str,
    cap: int,
    chunks: List[str],
    on_output: Optional[OutputCallback],
) -> bool:
    """Drain *stream*, emitting and retaining lines up to *cap* characters."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    used = 0
    truncated = False

    def emit(line: str) -> None:
        nonlocal used, truncated
        line = line.rstrip("\r")
        if used < cap:
            if on_output is not None:
                try:
                    on_output(tag, line)
                except Exception:
                    pass
            piece = line[: cap - used]
            chunks.append(piece + "\n")
            used += len(piece) + 1
            if len(piece) < len(line):
                truncated = True
        else:
            truncated = True

    while True:
        raw = await stream.read(8192)
        if not raw:
            break
        pending += decoder.decode(raw)
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            emit(line)
    pending += decoder.decode(b"", final=True)
    if pending:
        emit(pending)
    return truncated


async def run_command(
    command: str,
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = 60.0,
    max_output: int = 100_000,
    shell: bool = True,
    on_output: Optional[OutputCallback] = None,
) -> CommandResult:
    """Run *command* without attaching it to the caller's terminal.

    ``timeout=None`` waits indefinitely (until cancellation). Output is
    drained continuously so a chatty child cannot fill its pipe and stall.
    """
    started = time.monotonic()
    kwargs: Dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
        "env": env,
        "start_new_session": True,
    }
    if shell:
        proc = await asyncio.create_subprocess_shell(command, **kwargs)
    else:
        proc = await asyncio.create_subprocess_exec(*shlex.split(command), **kwargs)

    out_chunks: List[str] = []
    err_chunks: List[str] = []
    pumps = [
        asyncio.create_task(_pump(proc.stdout, "stdout", max_output, out_chunks, on_output)),
        asyncio.create_task(_pump(proc.stderr, "stderr", max_output, err_chunks, on_output)),
    ]

    timed_out = False
    try:
        if timeout is None:
            await proc.wait()
        else:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        await terminate_process(proc)
    except asyncio.CancelledError:
        await terminate_process(proc, force=True)
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)
        raise
    finally:
        if proc.returncode is None:
            signal_process_group(proc, SIGKILL)

    truncated = False
    try:
        results = await asyncio.wait_for(asyncio.gather(*pumps), timeout=2.0)
        truncated = any(results)
    except asyncio.TimeoutError:
        for task in pumps:
            task.cancel()
        await asyncio.gather(*pumps, return_exceptions=True)

    return CommandResult(
        stdout="".join(out_chunks),
        stderr="".join(err_chunks),
        exit_code=proc.returncode,
        timed_out=timed_out,
        truncated=truncated,
        duration=time.monotonic() - started,
        pid=proc.pid,
    )


__all__ = [
    "CommandResult",
    "SIGKILL",
    "SIGTERM",
    "is_long_running",
    "run_command",
    "signal_process_group",
    "terminate_process",
]
