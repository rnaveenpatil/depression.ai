"""Regression coverage for bounded filesystem discovery and reads."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent.tools.compat import GlobTool, ReadTool
from agent.tools.filesystem import FileSystemTool
from agent.tools import search as search_module
from agent.tools.search import SearchTool


class _Workspace:
    def __init__(self, root: Path):
        self.root = root

    def get_project_dir(self) -> str:
        return str(self.root)


@pytest.mark.asyncio
async def test_read_tool_reads_past_default_32kb_cap(tmp_path):
    lines = [f"{i:04d} " + ("x" * 40) for i in range(1200)]
    (tmp_path / "large.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert (tmp_path / "large.txt").stat().st_size > 32 * 1024

    result = await ReadTool(_Workspace(tmp_path)).execute({
        "filePath": "large.txt",
        "offset": 1198,
        "limit": 2,
    })

    assert result["success"] is True, result
    assert result["content"].splitlines() == lines[1198:1200]
    assert result["truncated"] is False


@pytest.mark.asyncio
async def test_recursive_list_skips_ignored_and_virtualenv_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("junk")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("junk")
    venv = tmp_path / "src" / "custom-env"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "lib").mkdir()
    (venv / "lib" / "junk.py").write_text("junk")

    result = await FileSystemTool(_Workspace(tmp_path)).execute({
        "action": "list",
        "path": ".",
        "recursive": True,
        "max_files": 100,
    })

    paths = {entry["path"] for entry in result["entries"]}
    assert "src/app.py" in paths
    assert not any(
        path.startswith(("node_modules/", ".git/", "src/custom-env/"))
        for path in paths
    )

    shallow = await FileSystemTool(_Workspace(tmp_path)).execute({
        "action": "list",
        "path": ".",
        "pattern": "src/*.py",
        "recursive": False,
    })
    assert [entry["path"] for entry in shallow["entries"]] == ["src/app.py"]


@pytest.mark.asyncio
async def test_glob_skips_ignored_dirs_caps_and_preserves_depth(tmp_path):
    (tmp_path / "root.py").write_text("root")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "nested.py").write_text("nested")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("junk")
    venv = tmp_path / "custom-env"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "junk.py").write_text("junk")

    tool = GlobTool(_Workspace(tmp_path))
    recursive = await tool.execute({"pattern": "**/*.py", "max_results": 2})
    assert recursive["success"] is True, recursive
    assert recursive["count"] == 2
    assert recursive["truncated"] is True
    assert not any("node_modules" in path or "custom-env" in path
                   for path in recursive["matches"])

    shallow = await tool.execute({"pattern": "*.py"})
    assert shallow["matches"] == ["root.py"]


@pytest.mark.asyncio
async def test_python_search_uses_shared_ignore_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: None)
    (tmp_path / "app.py").write_text("needle\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("needle\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("needle\n")
    venv = tmp_path / "custom-env"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / "junk.py").write_text("needle\n")

    result = await SearchTool(_Workspace(tmp_path)).execute({
        "query": "needle",
        "path": str(tmp_path),
        "max_results": 10,
    })

    paths = {Path(match["path"]).relative_to(tmp_path).as_posix()
             for match in result["matches"]}
    assert paths == {"app.py", "src/lib.py"}
    assert result["backend"] == "python"


@pytest.mark.asyncio
async def test_append_after_sha_matches_final_file_without_diff(tmp_path):
    tool = FileSystemTool(_Workspace(tmp_path))
    created = await tool.execute({
        "action": "write", "path": "a.txt", "content": "one\n",
        "include_diff": False,
    })
    assert created["success"] is True
    appended = await tool.execute({
        "action": "append", "path": "a.txt", "content": "two\n",
        "include_diff": False,
    })
    assert appended["success"] is True
    assert appended["after_sha256"] == hashlib.sha256(b"one\ntwo\n").hexdigest()


@pytest.mark.asyncio
async def test_write_omits_diff_for_large_existing_file(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("x" * (60 * 1024), encoding="utf-8")
    result = await FileSystemTool(_Workspace(tmp_path)).execute({
        "action": "write",
        "path": "large.txt",
        "content": "replacement\n",
        "include_diff": True,
    })
    assert result["success"] is True, result
    assert result["diff_omitted"] is True
    assert "before" not in result
