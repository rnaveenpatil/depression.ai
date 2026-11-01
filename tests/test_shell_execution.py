"""Regression tests for detached shell execution shared by TUI and tools."""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from agent.tools import ProcessTool
from agent.tui.app import DepressionApp
from agent.utils.shell import is_long_running, run_command


class _Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def get_project_dir(self) -> Path:
        return self.root

    def assert_inside_workspace(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        p.relative_to(self.root)
        return p


def _stdin_probe() -> str:
    return (
        f"{shlex.quote(sys.executable)} -c "
        "'import sys; print(repr(sys.stdin.read()))'"
    )


@pytest.mark.asyncio
async def test_run_command_does_not_inherit_stdin():
    result = await run_command(_stdin_probe(), timeout=10)

    assert result.exit_code == 0
    assert result.stdout.strip() == "''"


@pytest.mark.asyncio
async def test_timeout_kills_whole_process_group():
    if os.name != "posix":
        pytest.skip("process-group assertions are POSIX-only")

    result = await run_command(
        "bash -c 'sleep 100 & echo $!; wait'", timeout=0.5
    )

    assert result.timed_out is True
    child_pid = int(result.stdout.strip().splitlines()[0])
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not os.path.exists(f"/proc/{child_pid}"):
            break
        await asyncio.sleep(0.05)
    assert not os.path.exists(f"/proc/{child_pid}")


def test_long_running_detection_is_shared():
    assert is_long_running("flutter run -d chrome") == "flutter run"
    assert is_long_running("printf hello") is None


@pytest.mark.asyncio
async def test_tui_run_detaches_stdin_and_streams_output():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._run_shell(_stdin_probe())
        worker = app._user_shell_worker
        for _ in range(100):
            if worker is None or not getattr(worker, "is_running", False):
                break
            await pilot.pause(0.02)
        await pilot.pause(0.05)

        rendered = " ".join(
            str(node.render())
            for node in app.query_one("#transcript").children
        )
        assert "''" in rendered


@pytest.mark.asyncio
async def test_tui_stop_cancels_long_running_shell():
    app = DepressionApp(coordinator=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._run_shell("tail -f /dev/null")
        worker = app._user_shell_worker
        for _ in range(100):
            if worker is not None and getattr(worker, "is_running", False):
                break
            await pilot.pause(0.02)
        assert worker is not None and worker.is_running

        app._request_agent_stop()
        task = worker._task
        for _ in range(150):
            if task.done():
                break
            await pilot.pause(0.02)
        assert task.done()


@pytest.mark.asyncio
async def test_process_tool_detaches_stdin(tmp_path):
    tool = ProcessTool(_Workspace(tmp_path))
    result = await tool.execute({
        "action": "start",
        "command": _stdin_probe(),
    })
    assert result["success"] is True, result
    process_id = result["process_id"]

    for _ in range(100):
        status = await tool.execute({
            "action": "status", "process_id": process_id,
        })
        if not status["running"]:
            break
        await asyncio.sleep(0.02)

    logs = await tool.execute({
        "action": "logs", "process_id": process_id, "lines": 20,
    })
    assert "''" in logs["stdout"]
    await tool.shutdown()
