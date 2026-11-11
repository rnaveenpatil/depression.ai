"""Regression coverage for tool safety and lifecycle hardening."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.permissions.aws import AWSPolicy
from agent.permissions.manager import PermissionRequest
from agent.tools.aws_helper import AWSHelperTool
from agent.tools.browser import BrowserTool
from agent.tools.compat import ApplyPatchTool, EditTool, GlobTool, SkillTool, WebFetchTool, WebSearchTool
from agent.tools.diagnostics import DiagnosticsTool
from agent.tools.git import GitTool
from agent.tools.patch import PatchTool
from agent.tools.path_guard import OutputPathError, resolve_output_path
from agent.tools.proccess import ProcessTool
from agent.tools.realtime_browser import RealtimeBrowserTool
from agent.tools.registry import BaseTool, ToolRegistry
from agent.tools.search import SearchTool
from agent.tools.task import TaskTool
from agent.tools.todo import TodoTool
from agent.tools import search as search_module
from agent.tools import terminal as terminal_module


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def get_project_dir(self) -> str:
        return str(self.root)

    def assert_inside_workspace(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        p.relative_to(self.root)
        return p


@pytest.mark.asyncio
async def test_task_tool_uses_subagent_manager_instead_of_parent(tmp_path):
    class Manager:
        def __init__(self):
            self.calls = []

        async def invoke_subagent(self, role, prompt, context):
            self.calls.append((role.value, prompt, context))
            return {"success": True, "result": f"did {prompt}"}

    manager = Manager()
    tool = TaskTool(SimpleNamespace(subagent_manager=manager))
    spawned = await tool.execute({"action": "spawn", "prompt": "work"})
    assert spawned["success"] is True, spawned
    waited = await tool.execute({"action": "wait", "task_id": spawned["task_id"], "timeout": 2})
    assert waited == {
        "success": True,
        "id": spawned["task_id"],
        "result": "did work",
        "error": None,
    }
    assert manager.calls[0][0] == "general"
    await tool.shutdown()


@pytest.mark.asyncio
async def test_task_tool_refuses_parent_agent_without_manager():
    tool = TaskTool(SimpleNamespace())
    result = await tool.execute({"action": "spawn", "prompt": "work"})
    assert result["success"] is False
    assert "refusing" in result["error"].lower()


@pytest.mark.asyncio
async def test_skill_tool_rejects_traversal_and_absolute_paths(tmp_path):
    (tmp_path / ".skills").mkdir()
    (tmp_path / ".skills" / "safe.md").write_text("safe content", encoding="utf-8")
    tool = SkillTool(Workspace(tmp_path))

    ok = await tool.execute({"name": "safe"})
    assert ok["success"] is True
    assert "safe content" in ok["content"]

    for name in ("../secret", "/etc/passwd", "a/b"):
        result = await tool.execute({"name": name})
        assert result["success"] is False
        assert "invalid skill name" in result["error"]


@pytest.mark.asyncio
async def test_git_rejects_option_injection_and_secret_paths(tmp_path):
    tool = GitTool(Workspace(tmp_path))
    called = []

    async def fake_git(*args, timeout=30.0):
        called.append(args)
        return {"success": True, "stdout": "", "stderr": "", "exit_code": 0}

    tool._git = fake_git  # type: ignore[assignment]

    result = await tool.execute({"action": "show", "ref": "--output=/tmp/pwn"})
    assert result["success"] is False
    result = await tool.execute({"action": "add", "args": ["--force"]})
    assert result["success"] is False
    result = await tool.execute({"action": "add", "args": [".env"]})
    assert result["success"] is False
    assert called == []

    result = await tool.execute({"action": "log", "limit": 99999})
    assert result["success"] is True
    assert called[-1][1] == "--max-count=1000"


@pytest.mark.asyncio
async def test_diagnostics_rejects_fix_output_and_outside_paths(tmp_path):
    tool = DiagnosticsTool(Workspace(tmp_path))

    for args in (["--fix"], ["--output-file", "/tmp/out"], ["--plugin", "evil"]):
        result = await tool.execute({"tool": "ruff", "args": args})
        assert result["success"] is False, args

    result = await tool.execute({"tool": "ruff", "path": "/etc"})
    assert result["success"] is False
    assert "workspace" in result["error"].lower()

    cmd = tool._build_command("eslint", str(tmp_path), [])
    assert "--no-install" in cmd
    assert tool._validate_extra_args(["--follow-imports", "skip"]) == [
        "--follow-imports", "skip",
    ]
    with pytest.raises(ValueError):
        tool._validate_extra_args(["--config", "/tmp/evil.ini"])


@pytest.mark.asyncio
async def test_aws_blocks_endpoint_and_global_flag_injection(tmp_path):
    tool = AWSHelperTool(None, Workspace(tmp_path))
    result = await tool.execute({
        "action": "call",
        "service": "s3",
        "operation": "ls",
        "args": ["--endpoint-url", "http://127.0.0.1"],
    })
    assert result["success"] is False
    assert "global option" in result["error"]

    result = await tool.execute({
        "action": "call",
        "service": "s3 --debug",
        "operation": "ls",
    })
    assert result["success"] is False

    result = await tool.execute({
        "action": "call",
        "mcp_tool": "mcp__aws__call_aws",
        "mcp_arguments": {"EndpointUrl": "http://127.0.0.1"},
    })
    assert result["success"] is False
    assert "endpoint" in result["error"].lower()

    policy = AWSPolicy({"aws_allowed_endpoints": []})
    verdict = await policy.evaluate(PermissionRequest(
        tool="aws",
        action="call",
        params={
            "service": "s3",
            "operation": "ls",
            "args": ["--endpoint-url", "http://127.0.0.1"],
        },
    ))
    assert verdict is not None
    assert verdict.decision.value == "deny"


@pytest.mark.asyncio
async def test_patch_rejects_empty_old_and_redacts_secret_preview(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=one\n", encoding="utf-8")
    tool = PatchTool(Workspace(tmp_path))

    result = await tool.execute({
        "action": "apply",
        "patches": [{"path": "a.txt", "old": "", "new": "prefix\n"}],
    })
    assert result["success"] is False
    assert (tmp_path / "a.txt").read_text() == "hello\n"

    preview = await tool.execute({
        "action": "preview",
        "patches": [{"path": ".env", "old": "SECRET=one", "new": "SECRET=two"}],
    })
    assert preview["success"] is True
    assert preview["results"][0]["redacted"] is True
    assert "one" not in preview["results"][0]["before"]
    assert (tmp_path / ".env").read_text() == "SECRET=one\n"


@pytest.mark.asyncio
async def test_patch_batch_is_all_or_nothing_and_rejects_binary(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    tool = PatchTool(Workspace(tmp_path))

    result = await tool.execute({
        "action": "apply",
        "patches": [
            {"path": "a.txt", "old": "a", "new": "changed"},
            {"path": "b.txt", "old": "missing", "new": "x"},
        ],
    })
    assert result["success"] is False
    assert (tmp_path / "a.txt").read_text() == "a\n"

    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00")
    result = await tool.execute({
        "action": "apply",
        "patches": [{"path": "bin.dat", "old": "x", "new": "y"}],
    })
    assert result["success"] is False
    assert (tmp_path / "bin.dat").read_bytes() == b"\xff\xfe\x00"


@pytest.mark.asyncio
async def test_apply_patch_refuses_overwrite_and_directory_delete(tmp_path):
    (tmp_path / "exists.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "dir").mkdir()
    tool = ApplyPatchTool(Workspace(tmp_path))

    add = await tool.execute({
        "patchText": "*** Begin Patch\n*** Add File: exists.txt\n+new\n*** End Patch\n",
    })
    assert add["success"] is False
    assert (tmp_path / "exists.txt").read_text() == "old\n"

    delete = await tool.execute({
        "patchText": "*** Begin Patch\n*** Delete File: dir\n*** End Patch\n",
    })
    assert delete["success"] is False
    assert (tmp_path / "dir").is_dir()


@pytest.mark.asyncio
async def test_apply_patch_marker_batch_rolls_back_on_failure(tmp_path):
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("two\n", encoding="utf-8")
    tool = ApplyPatchTool(Workspace(tmp_path))

    result = await tool.execute({
        "patchText": (
            "*** Begin Patch\n"
            "*** Update File: a.txt\n"
            "@@\n"
            "-one\n"
            "+ONE\n"
            "*** Update File: b.txt\n"
            "@@\n"
            "-missing\n"
            "+changed\n"
            "*** End Patch\n"
        ),
    })
    assert result["success"] is False
    assert (tmp_path / "a.txt").read_text() == "one\n"
    assert result["results"][0]["rolled_back"] is True


@pytest.mark.asyncio
async def test_unified_patch_preserves_crlf_line_endings(tmp_path):
    (tmp_path / "c.txt").write_bytes(b"one\r\ntwo\r\n")
    tool = PatchTool(Workspace(tmp_path))

    result = await tool.execute({
        "action": "apply",
        "diff": (
            "--- a/c.txt\n"
            "+++ b/c.txt\n"
            "@@ -1,1 +1,1 @@\n"
            "-one\n"
            "+ONE\n"
        ),
    })
    assert result["success"] is True, result
    assert (tmp_path / "c.txt").read_bytes() == b"ONE\r\ntwo\r\n"


@pytest.mark.asyncio
async def test_search_and_glob_skip_secret_files_and_reject_outside_root(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle\n", encoding="utf-8")
    (tmp_path / "prod.env").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr(search_module.shutil, "which", lambda _name: None)

    search = await SearchTool(Workspace(tmp_path)).execute({"query": "needle"})
    names = {Path(m["path"]).name for m in search["matches"]}
    assert names == {"app.py"}

    outside = await SearchTool(Workspace(tmp_path)).execute({"query": "needle", "path": "/etc"})
    assert outside["success"] is False

    globbed = await GlobTool(Workspace(tmp_path)).execute({"pattern": "**/*", "path": "."})
    assert ".env" not in globbed["matches"]
    assert "prod.env" not in globbed["matches"]

    outside_glob = await GlobTool(Workspace(tmp_path)).execute({"pattern": "/etc/**"})
    assert outside_glob["success"] is False


def test_path_guard_refuses_sensitive_destinations(tmp_path):
    assert resolve_output_path("shot.png", tmp_path).name == "shot.png"
    for bad in (".env", ".bashrc", ".git/hooks/pre-commit", ".ssh/config"):
        with pytest.raises(OutputPathError):
            resolve_output_path(bad, tmp_path)


def test_browser_and_realtime_reject_file_and_private_urls(tmp_path):
    for tool in (
        BrowserTool({"allow_network": True, "sandbox_root": str(tmp_path)}),
        RealtimeBrowserTool({"allow_network": True, "sandbox_root": str(tmp_path)}),
    ):
        with pytest.raises(ValueError):
            tool._checked_url("file:///etc/passwd")
        from agent.tools.web import BlockedRequestError
        with pytest.raises(BlockedRequestError):
            tool._checked_url("http://127.0.0.1:8080")


@pytest.mark.asyncio
async def test_browser_shutdown_resets_lifecycle_state():
    class Browser:
        async def close(self):
            pass

    class Playwright:
        async def stop(self):
            pass

    tool = BrowserTool({"allow_network": True})
    tool._browser = Browser()
    tool._page = object()
    tool._pw = Playwright()
    await tool.shutdown()
    assert tool._browser is None
    assert tool._page is None
    assert tool._pw is None


@pytest.mark.asyncio
async def test_todo_accepts_completed_alias_and_is_mutating(tmp_path):
    tool = TodoTool(None)
    tool.items.clear()
    added = await tool.execute({"action": "add", "title": "task"})
    updated = await tool.execute({
        "action": "update",
        "todo_id": added["todo_id"],
        "status": "completed",
    })
    assert updated["item"]["status"] == "completed"
    listed = await tool.execute({"action": "list"})
    assert listed["items"][0]["status"] == "completed"
    assert tool.mutating is True


@pytest.mark.asyncio
async def test_edit_rejects_empty_old_string(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    result = await EditTool(Workspace(tmp_path)).execute({
        "filePath": "a.txt",
        "oldString": "",
        "newString": "prefix",
    })
    assert result["success"] is False
    assert (tmp_path / "a.txt").read_text() == "hello\n"


def test_terminal_aws_credentials_are_opt_in(monkeypatch):
    class FakeEnvManager:
        @classmethod
        def get(cls):
            return cls()

        def get_aws_env(self):
            return {
                "AWS_ACCESS_KEY_ID": "AKIA_TEST",
                "AWS_SECRET_ACCESS_KEY": "secret",
            }

    monkeypatch.setattr(terminal_module, "EnvManager", FakeEnvManager)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_INHERITED")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "inherited")
    tool = terminal_module.TerminalTool(Workspace(Path.cwd()))

    default_env = tool._build_env({})
    assert "AWS_ACCESS_KEY_ID" not in default_env
    assert "AWS_SECRET_ACCESS_KEY" not in default_env

    opted_in = tool._build_env({}, include_aws=True)
    assert opted_in["AWS_ACCESS_KEY_ID"] == "AKIA_TEST"


@pytest.mark.asyncio
async def test_process_drains_huge_lines_without_deadlock(tmp_path):
    tool = ProcessTool(Workspace(tmp_path))
    started = await tool.execute({
        "action": "start",
        "command": "python -c \"print('x'*100000); print('END')\"",
    })
    assert started["success"] is True, started
    try:
        waited = await tool.execute({
            "action": "wait",
            "process_id": started["process_id"],
            "timeout": 10,
        })
        assert waited["success"] is True, waited
        assert "line truncated" in waited["stdout"]
        assert "END" in waited["stdout"]
    finally:
        await tool.execute({"action": "stop", "process_id": started["process_id"]})
        await tool.shutdown()


@pytest.mark.asyncio
async def test_webfetch_and_websearch_validate_numeric_params():
    fetch = WebFetchTool({})
    result = await fetch.execute({"url": "https://example.com", "maxChars": "lots"})
    assert result["success"] is False
    result = await fetch.execute({"url": "https://example.com", "extractMode": "bogus"})
    assert result["success"] is False

    search = WebSearchTool({})
    result = await search.execute({"query": "x", "count": "many"})
    assert result["success"] is False


async def _external_should_not_run(**_kwargs):
    raise AssertionError("duplicate external tool should not execute")


@pytest.mark.asyncio
async def test_registry_can_hide_aliases_without_disabling_execution():
    class Dummy(BaseTool):
        name = "dummy"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, params):
            return {"success": True, "params": params}

    reg = ToolRegistry()
    await reg.register_tool(Dummy())
    reg.register_external(
        "dummy",
        handler=_external_should_not_run,
        description="duplicate",
    )
    assert [s["function"]["name"] for s in reg.get_schemas()] == ["dummy"]
    assert reg.hide_tool("dummy") is True
    assert reg.get_schemas() == []
    result = await reg.execute("dummy", {})
    assert result["success"] is True
    assert result["params"] == {}
