"""Regression coverage for red/green file edit previews."""
import pytest

from agent.tui.widgets.diff_view import compute_line_diff, render_diff
from agent.tui.widgets.tool_call import ToolCallWidget
from agent.tui.app import DepressionApp
from agent.tools.filesystem import FileSystemTool


def test_add_and_delete_empty_snapshots_render_with_colors():
    added = ToolCallWidget("filesystem", {"action": "write", "path": "new.txt"})
    added._result = {"success": True, "before": "", "after": "hello", "path": "new.txt"}
    assert "[#00ff66]+ hello[/]" in added._render_result()

    removed = ToolCallWidget("filesystem", {"action": "write", "path": "old.txt"})
    removed._result = {"success": True, "before": "goodbye", "after": "", "path": "old.txt"}
    assert "[#ff4466]- goodbye[/]" in removed._render_result()


def test_patch_multifile_and_partial_failure_are_visible():
    widget = ToolCallWidget("patch", {"action": "apply"})
    widget._result = {
        "success": True,
        "results": [
            {"success": True, "path": "one.txt", "before": "a", "after": "b"},
            {"success": True, "path": "two.txt", "before": "", "after": "new"},
        ],
    }
    output = widget._render_result()
    assert "one.txt" in output and "two.txt" in output
    assert "+ b" in output and "- a" in output and "+ new" in output


def test_reads_do_not_become_green_file_edits():
    payload = {"success": True, "content": "original", "path": "a.txt"}
    shaped = DepressionApp._shape_diff_payload("filesystem", {"action": "read"}, payload)
    assert "before" not in shaped and "after" not in shaped
    widget = ToolCallWidget("filesystem", {"action": "read"})
    widget._result = payload
    assert "+ original" not in widget._render_result()


def test_diff_omission_is_explicit():
    widget = ToolCallWidget("filesystem", {"action": "write"})
    widget._result = {"success": True, "diff_omitted": True, "diff_omitted_reason": "large file"}
    assert "Diff preview omitted" in widget._render_result()


def test_diff_markup_is_escaped_and_change_counts_are_correct():
    markup = render_diff("[red]old[/]", "[green]new[/]", path="[red]file[/].py")
    assert r"\[red]file\[/]" in markup
    assert r"\[green]new\[/]" in markup
    changes = compute_line_diff("same\nold\n", "same\nnew\n")
    assert [line.kind for line in changes] == ["ctx", "del", "add"]


def test_large_line_count_uses_bounded_preview():
    large = "\n".join(str(i) for i in range(4000))
    output = render_diff(large, large + "\nextra", path="large.txt")
    assert "Diff preview omitted" in output


@pytest.mark.asyncio
async def test_filesystem_create_and_clear_contain_real_snapshots(tmp_path):
    tool = FileSystemTool()
    tool.base_cwd = tmp_path
    create = await tool.execute({"action": "write", "path": "empty.txt", "content": "hello"})
    assert create["success"] is True
    assert create["before"] == "" and create["after"] == "hello"
    clear = await tool.execute({"action": "write", "path": "empty.txt", "content": ""})
    assert clear["success"] is True
    assert clear["before"] == "hello" and clear["after"] == ""
