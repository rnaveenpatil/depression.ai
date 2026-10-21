"""Regression tests for built-in tool routing, aliases, and safety."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import (
    ApplyPatchTool,
    EditTool,
    GitTool,
    ProcessTool,
    ReadTool,
    TerminalTool,
    TodoTool,
    WebTool,
    WriteTool,
)
from agent.tools.registry import ToolRegistry, ToolStats


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


def _registry(*tools) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in tools:
        reg.tools[tool.name] = tool
        reg.stats.setdefault(tool.name, ToolStats())
    return reg


@pytest.mark.asyncio
async def test_feature_intent_keeps_core_tools(tmp_path):
    ws = _Workspace(tmp_path)
    reg = _registry(
        TerminalTool(ws, {}),
        ReadTool(ws),
        WriteTool(ws),
        EditTool(ws),
        ApplyPatchTool(ws),
        GitTool(ws),
        WebTool({}),
        TodoTool(None),
    )

    names = {s["function"]["name"] for s in reg.select_for_task("feature")}

    assert {"terminal", "read", "write", "edit", "apply_patch", "git",
            "web", "todo"} <= names


@pytest.mark.asyncio
async def test_core_tool_metadata(tmp_path):
    ws = _Workspace(tmp_path)
    reg = _registry(
        ReadTool(ws), WriteTool(ws), ApplyPatchTool(ws), GitTool(ws)
    )

    assert reg.is_read_only("read") is True
    assert reg.is_mutating("write") is True
    assert reg.is_mutating("apply_patch") is True
    assert reg.get_category("git") == "vcs"


@pytest.mark.asyncio
async def test_alias_arguments_pass_registry_validation(tmp_path):
    ws = _Workspace(tmp_path)
    reg = _registry(ReadTool(ws), WriteTool(ws), ApplyPatchTool(ws))

    assert reg.validate_arguments("read", {"path": "a.txt"}) is None
    assert reg.validate_arguments(
        "write", {"path": "a.txt", "content": "x"}
    ) is None
    assert reg.validate_arguments(
        "apply_patch", {"patch": "*** Begin Patch\n*** End Patch\n"}
    ) is None


@pytest.mark.asyncio
async def test_apply_patch_update_accepts_hunk_header(tmp_path):
    ws = _Workspace(tmp_path)
    (tmp_path / "a.txt").write_text("old\n")
    tool = ApplyPatchTool(ws)

    result = await tool.execute({
        "patchText": (
            "*** Begin Patch\n"
            "*** Update File: a.txt\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
        )
    })

    assert result["success"] is True, result
    assert (tmp_path / "a.txt").read_text() == "new\n"


@pytest.mark.asyncio
async def test_process_rejects_cwd_outside_workspace(tmp_path):
    tool = ProcessTool(_Workspace(tmp_path))

    result = await tool.execute({
        "action": "start",
        "command": "pwd",
        "cwd": "/etc",
    })

    assert result["success"] is False
    assert "Invalid cwd" in result["error"]
