"""Regression tests for the FileSystemTool write/append path.

Guards the bug where the write helper called ``open(target, mode, "wb")``:
the *third positional* argument of ``open()`` is ``buffering`` (an int), so
every write raised ``TypeError: 'str' object cannot be interpreted as an
integer`` and surfaced as ``Failed to write <path>: ...``.
"""

from __future__ import annotations

import json

import pytest

from agent.tools.filesystem import FileSystemTool


class _Workspace:
    def __init__(self, root):
        self.root = root

    def get_project_dir(self):
        return str(self.root)


@pytest.fixture()
def tool(tmp_path):
    return FileSystemTool(workspace=_Workspace(tmp_path))


async def test_write_creates_file_without_typeerror(tool, tmp_path):
    result = await tool.execute(
        {"action": "write", "path": "button.html", "content": "<button>Go</button>"}
    )

    assert result["success"] is True, result
    assert "error" not in result
    assert result["created"] is True
    assert result["bytes_written"] == len("<button>Go</button>")
    assert (tmp_path / "button.html").read_text(encoding="utf-8") == "<button>Go</button>"


async def test_write_overwrite_and_append(tool, tmp_path):
    await tool.execute({"action": "write", "path": "a.txt", "content": "one\n"})
    overwrite = await tool.execute(
        {"action": "write", "path": "a.txt", "content": "two\n"}
    )
    assert overwrite["success"] is True
    assert overwrite["created"] is False

    append = await tool.execute({"action": "append", "path": "a.txt", "content": "three\n"})
    assert append["success"] is True
    assert append["mode"] == "append"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two\nthree\n"


async def test_write_unicode_payload_is_utf8(tool, tmp_path):
    payload = "caf\u00e9 \u2014 \u00fcber \u2705 \U0001f600"
    result = await tool.execute({"action": "write", "path": "u.txt", "content": payload})

    assert result["success"] is True, result
    assert (tmp_path / "u.txt").read_text(encoding="utf-8") == payload


async def test_write_reports_diff_and_rejects_non_string(tool):
    created = await tool.execute({"action": "write", "path": "d.txt", "content": "x"})
    assert created["success"] is True
    assert created["before"] == ""
    assert created["after"] == "x"

    bad = await tool.execute({"action": "write", "path": "d.txt", "content": 123})
    assert bad["success"] is False
    assert json.loads(json.dumps(bad))  # result stays JSON-serialisable like the tool contract
