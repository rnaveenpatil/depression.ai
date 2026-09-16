"""Plan/Build agent role verification.

Tests the two-agent (Plan/Build) workflow against OpenCode-style semantics
using REAL code paths: BaseAgent.execute_tool guard, AgentLoop, ToolRegistry,
file-backed tools, ContextManager, PermissionManager, and Plan/Task objects —
with a scripted LLM standing in for the model.

  - PLAN agent: builds a plan, reads files, and is denied every write/bash/
    process mutation even though the permission rules allow it.
  - BUILD agent: has full tool access (read/write/edit/bash/filesystem/git).
  - Schema divergence is pinned so regressions are caught: both agents expose
    the same tool schemas; read-only enforcement is role-based, not schema-based.
"""
from __future__ import annotations

import subprocess
import json
from pathlib import Path

import pytest

from agent.agent.dual_agent import AgentCoordinator, AgentRole, BuildAgent, PlanAgent
from agent.agent.loop import AgentLoop
from agent.agent.planner import Plan, Task
from agent.context.manager import ContextManager
from agent.permissions.manager import PermissionManager
from agent.project.workspace import WorkspaceManager
from agent.tools.compat import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
)
from agent.tools.filesystem import FileSystemTool
from agent.tools.git import GitTool
from agent.tools.registry import ToolRegistry
from agent.tools.terminal import TerminalTool
from agent.tools.todo import TodoTool
from agent.tools.compat import TodoReadTool, TodoWriteTool


class _FakeSession:
    id = "verify-session"

    def add_user_message(self, *a, **k):
        pass

    def add_assistant_message(self, *a, **k):
        pass

    def add_tool_message(self, *a, **k):
        pass

    def set_status(self, *a, **k):
        pass

    def update_context(self, *a, **k):
        pass


class _FakeToolCall:
    _n = 0

    def __init__(self, name, arguments):
        _FakeToolCall._n += 1
        self.id = f"call_{name}_{_FakeToolCall._n}"
        self.name = name
        self.arguments = arguments


class _MockResponse:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {"prompt_tokens": 25, "completion_tokens": 25, "total_tokens": 50}


class _MockLLM:
    def __init__(self, script):
        self.script = list(script)
        self.labels = []

    async def complete_with_tools(self, messages=None, tools=None, temperature=None, max_tokens=None):
        assert self.script, "MockLLM script exhausted"
        step = self.script.pop(0)
        self.labels.append(step["label"])
        return _MockResponse(
            content=step.get("content"),
            tool_calls=[_FakeToolCall(c["name"], c["arguments"]) for c in step.get("tool_calls", [])],
        )

    async def complete(self, messages=None, **kw):
        return _MockResponse(content="fallback-final")

    def get_current_model(self):
        return "vendor/test-model"

    def get_current_provider(self):
        return "test"


class _FakePlanner:
    def __init__(self):
        self.calls = 0
        self.plan = None

    async def create_plan(self, goal, context=None, constraints=None):
        self.calls += 1
        self.plan = Plan(
            id="plan-e2e",
            goal=goal,
            tasks=[
                Task(id="t-inspect", description="inspect the relevant files first"),
                Task(id="t-draft", description="draft the implementation steps"),
            ],
        )
        return self.plan


def _make_workspace(tmp_path):
    ws = WorkspaceManager(
        workspace_dir=str(tmp_path / ".agent"),
        project_dir=str(tmp_path),
        config={"enforce_boundaries": True, "auto_scan": False},
    )
    return ws


async def _make_agent(ws, role, script, with_planner):
    pm = PermissionManager({"mode": "manual", "*": "allow", "default": "allow"}, ui=None, input_handler=None)
    session = _FakeSession()
    cm = ContextManager(
        workspace=ws,
        session=session,
        config={"recent_messages": 40, "max_messages": 200},
        llm=_MockLLM([]),
    )
    if role == AgentRole.PLAN:
        agent = PlanAgent(
            config={"llm": {}},
            session=session,
            context_manager=cm,
            permission_manager=pm,
            workspace=ws,
            database=None,
        )
    else:
        agent = BuildAgent(
            config={"llm": {}},
            session=session,
            context_manager=cm,
            permission_manager=pm,
            workspace=ws,
            database=None,
        )

    registry = ToolRegistry(agent=agent)
    for tool in (
        ReadTool(ws), WriteTool(ws), EditTool(ws), GrepTool(ws), GlobTool(ws),
        FileSystemTool(ws), GitTool(ws), TerminalTool(ws, {}), BashTool(ws, {}),
        TodoTool(session), TodoWriteTool(session), TodoReadTool(session),
    ):
        await registry.register_tool(tool)

    llm = _MockLLM(script)
    planner = _FakePlanner() if with_planner else None
    loop = AgentLoop(
        agent,
        llm,
        registry,
        planner,
        config={
            "enable_planning": bool(with_planner),
            "max_iterations": 10,
            "max_tool_calls": 5,
            "max_history_length": 40,
            "enable_caching": False,
            "timeout": 60,
        },
    )
    agent.llm = llm
    agent.tool_registry = registry
    agent.planner = planner
    agent.loop = loop
    agent.subagent_manager = None
    agent.plugin_loader = None
    return agent, planner


def _notes(tmp_path):
    (tmp_path / "notes.txt").write_text("hello world\n")


@pytest.mark.asyncio
async def test_plan_agent_is_read_only_even_when_permissions_allow(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _notes(tmp_path)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [], with_planner=True)
    nt = str(tmp_path / "notes.txt")

    read = await plan.execute_tool("read", {"filePath": nt})
    assert read["success"] and "hello world" in str(read.get("content", ""))

    for tool, params in [
        ("write", {"filePath": nt, "content": "x"}),
        ("edit", {"filePath": nt, "oldString": "a", "newString": "b"}),
        ("apply_patch", {"patch": "--- a\n+++ b\n"}),
        ("bash", {"command": "echo hi"}),
        ("terminal", {"command": "echo hi"}),
        ("process", {"command": "ls"}),
        ("filesystem", {"action": "write", "path": nt, "content": "x"}),
        ("filesystem", {"action": "append", "path": nt, "content": "y"}),
        ("git", {"action": "commit", "message": "nope"}),
    ]:
        res = await plan.execute_tool(tool, params)
        assert res.get("permission_denied") is True and not res.get("success"), (tool, res)

    fs_read = await plan.execute_tool("filesystem", {"action": "read", "path": nt})
    assert fs_read["success"]

    git_status = await plan.execute_tool("git", {"action": "status"})
    assert git_status["success"]

    grep = await plan.execute_tool("grep", {"pattern": "hello", "path": str(tmp_path)})
    assert grep["success"]

    glob_res = await plan.execute_tool("glob", {"pattern": "*.txt", "path": str(tmp_path)})
    assert glob_res["success"]


@pytest.mark.asyncio
async def test_plan_agent_loop_builds_plan_reads_but_cannot_write(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _notes(tmp_path)

    nt = str(tmp_path / "notes.txt")
    script = [
        {"label": "plan-read", "content": "", "tool_calls": [{"name": "read", "arguments": {"filePath": nt}}]},
        {"label": "plan-write", "content": "", "tool_calls": [{"name": "write", "arguments": {"filePath": str(tmp_path / "hack.txt"), "content": "HACKED"}}]},
        {"label": "plan-final", "content": "Done; only reading was permitted.", "tool_calls": []},
    ]
    plan, planner = await _make_agent(ws, AgentRole.PLAN, script, with_planner=True)

    q1 = "analyze the project and create a multi-step plan for a new feature across several files"
    r1 = await plan.process_query(q1)
    assert r1["success"] and "hello world" in r1.get("response", "")
    assert planner.calls == 1
    assert plan.loop.current_plan is not None and plan.loop.current_plan.goal == q1
    # single read tool call -> fast-path (one LLM round-trip)
    assert r1["tool_calls"] == 1
    assert plan.llm.labels == ["plan-read"]

    r2 = await plan.process_query("implement everything now")
    assert r2["success"]
    assert r2["tool_calls"] == 1
    assert not (tmp_path / "hack.txt").exists()


@pytest.mark.asyncio
async def test_build_agent_has_full_tool_access(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _notes(tmp_path)

    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)
    nt = str(tmp_path / "notes.txt")

    for tool, params in [
        ("read", {"filePath": nt}),
        ("write", {"filePath": str(tmp_path / "built.txt"), "content": "built-ok"}),
        ("edit", {"filePath": str(tmp_path / "built.txt"), "oldString": "built-ok", "newString": "built-edited"}),
        ("bash", {"command": "echo bash-ok"}),
        ("terminal", {"command": "echo term-ok"}),
        ("filesystem", {"action": "write", "path": str(tmp_path / "fs.txt"), "content": "fs-ok"}),
        ("git", {"action": "status"}),
    ]:
        res = await build.execute_tool(tool, params)
        assert res["success"], (tool, res)

    assert (tmp_path / "built.txt").read_text() == "built-edited"
    assert (tmp_path / "fs.txt").read_text() == "fs-ok"


@pytest.mark.asyncio
async def test_build_agent_loop_read_write_final(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _notes(tmp_path)

    script = [
        {"label": "bw-read", "content": "inspecting first", "tool_calls": [{"name": "read", "arguments": {"filePath": str(tmp_path / "notes.txt")}}]},
        {"label": "bw-write", "content": "", "tool_calls": [{"name": "write", "arguments": {"filePath": str(tmp_path / "built2.txt"), "content": "from loop"}}]},
        {"label": "bw-final", "content": "Created built2.txt", "tool_calls": []},
    ]
    build, _ = await _make_agent(ws, AgentRole.BUILD, script, with_planner=False)

    r = await build.process_query("create a file from the notes content")
    assert r["success"]
    assert build.llm.labels == ["bw-read", "bw-write", "bw-final"]
    assert r["tool_calls"] == 2
    assert (tmp_path / "built2.txt").read_text() == "from loop"


@pytest.mark.asyncio
async def test_role_tool_schemas_are_identical_guarded_at_execution(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [], with_planner=True)
    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)

    # Both roles expose the same schema surface (opencode keeps schemas stable;
    # mutation is denied at execution time for the plan role).
    assert set(plan.tool_registry.tools) == set(build.tool_registry.tools)
    for name in ("read", "write", "bash", "git", "filesystem"):
        assert name in plan.tool_registry.tools
    assert "write" in plan.tool_registry.tools  # "few read-only tools" filter is unused by design


@pytest.mark.asyncio
async def test_auto_mode_single_iteration_minimal_llm_calls(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    # Plan phase must emit its plan in ONE LLM call (loop-internal planning is
    # suppressed) -> no nested Planner call. Build phase finishes via the fast
    # path in ONE call.
    plan, plan_planner = await _make_agent(
        ws, AgentRole.PLAN,
        [{"label": "plan-think", "content": "PLAN: write hello.txt", "tool_calls": []}],
        with_planner=True,
    )
    build, _ = await _make_agent(
        ws, AgentRole.BUILD,
        [{"label": "build-write", "content": "",
          "tool_calls": [{"name": "write", "arguments": {
              "filePath": str(tmp_path / "hello.txt"), "content": "hello"}}]}],
        with_planner=False,
    )

    coord = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    r = await coord.process_query("create a file hello.txt with the text hello",
                                  mode="auto", auto_execute=True,
                                  max_iterations=3)

    assert r["success"] is True
    assert r["iterations"] == 1  # completed work must not be re-planned/re-run
    assert (tmp_path / "hello.txt").read_text() == "hello"
    assert plan_planner.calls == 0  # nested planner URL/LLM call removed from auto plan phase
    assert plan.llm.labels == ["plan-think"]
    assert build.llm.labels == ["build-write"]
    assert plan.loop.enable_planning is True  # restored after the auto phase


@pytest.mark.asyncio
async def test_todo_tools_share_store_across_actions(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)

    for name in ("todo", "todowrite", "todoread"):
        assert name in build.tool_registry.tools, name

    r = await build.execute_tool("todowrite", {"todos": [
        {"content": "task one", "status": "in_progress", "priority": "high"},
        {"content": "task two", "status": "pending", "priority": "medium"},
    ]})
    assert r["success"], r
    assert r["count"] == 2

    r = await build.execute_tool("todoread", {})
    assert r["success"] and r["count"] == 2
    by_title = {t["content"]: t for t in r["todos"]}
    assert by_title["task one"]["status"] == "in_progress"

    tid = by_title["task one"]["id"]
    r = await build.execute_tool("todo", {"action": "complete", "todo_id": tid})
    assert r["success"]

    r = await build.execute_tool("todoread", {})
    by_title = {t["content"]: t for t in r["todos"]}
    assert by_title["task one"]["status"] == "completed"

    r = await build.execute_tool("todo", {"action": "summary"})
    assert r["success"] and r["stats"].get("done") == 1

    r = await build.execute_tool("todowrite", {"todos": "not-a-list"})
    assert r["success"] is False and "array" in r["error"]


@pytest.mark.asyncio
async def test_loop_runs_bash_and_tracks_telemetry(tmp_path):
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    script = [
        {"label": "loop-bash", "content": "inspecting", "tool_calls": [
            {"name": "bash", "arguments": {"command": "mkdir -p out && echo wrote > out/flag.txt"}}]},
        {"label": "loop-final", "content": "Done creating flag.", "tool_calls": []},
    ]
    build, _ = await _make_agent(ws, AgentRole.BUILD, script, with_planner=False)

    r = await build.process_query("create a flag file with bash")
    assert r["success"] is True
    assert build.llm.labels == ["loop-bash", "loop-final"]
    assert r["tool_calls"] == 1
    assert (tmp_path / "out" / "flag.txt").read_text().startswith("wrote")

    s = r["context"]
    assert s["iteration"] == 2
    assert s["total_actions"] == 1 and s["total_observations"] == 1 and s["errors"] == 0


@pytest.mark.asyncio
async def test_tool_call_messages_serialize_spec_compliant(tmp_path):
    """Assistant tool-call messages must serialize content=null and tool
    results must carry a matching tool_call_id (strict NIM/vLLM gateways
    reject "" content on tool_calls)."""
    from agent.agent.loop import AgentLoop
    from agent.context.runtime import add_tool_call, add_tool_result, get_model_messages
    from agent.llm.provider import Message

    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)

    cm = build.context_manager
    await cm.add_user_message("make a file")
    call = _FakeToolCall("bash", {"command": "echo hi"})
    await add_tool_call(cm, "", [call])
    await add_tool_result(cm, call, {"success": True, "stdout": "hi\n"})

    msgs = get_model_messages(cm)
    roles = [m.role for m in msgs]
    assert "assistant" in roles and "tool" in roles

    assistant = next(m for m in msgs if m.role == "assistant" and m.tool_calls)
    assert assistant.content is None, assistant  # null on the wire
    assert assistant.to_dict()["content"] is None
    calls = assistant.tool_calls
    assert calls[0]["type"] == "function" and calls[0]["function"]["name"] == "bash"
    assert json.dumps(calls[0], ensure_ascii=False)  # JSON-safe

    tool = next(m for m in msgs if m.role == "tool")
    assert tool.tool_call_id == assistant.tool_calls[0]["id"]
    assert tool.to_dict()["tool_call_id"] == assistant.tool_calls[0]["id"]
    assert tool.to_dict()["content"]  # tool result content present


@pytest.mark.asyncio
async def test_auto_mode_propagates_llm_error(tmp_path):
    """A failed build request must surface its real error in the result so the
    TUI shows the cause instead of the generic 'agent request failed'."""
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [
        {"label": "plan-1", "content": "PLAN: run build", "tool_calls": []},
        {"label": "plan-2", "content": "PLAN: retry build", "tool_calls": []},
    ], with_planner=True)
    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)  # empty -> _think raises

    coord = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    r = await coord.process_query("do the work", mode="auto", auto_execute=True,
                                  max_iterations=2)

    assert r["success"] is False
    assert "error" in r, r
    assert "MockLLM" in r["error"] or "script exhausted" in r["error"], r["error"]

@pytest.mark.asyncio
async def test_plan_build_mode_propagates_llm_error(tmp_path):
    """Same as auto mode: plan->build must surface the real build error, not a
    bare short-circuited success."""
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    plan, _ = await _make_agent(ws, AgentRole.PLAN, [
        {"label": "plan-1", "content": "PLAN: run build", "tool_calls": []},
    ], with_planner=True)
    build, _ = await _make_agent(ws, AgentRole.BUILD, [], with_planner=False)

    coord = AgentCoordinator(plan_agent=plan, build_agent=build, config={})
    r = await coord.process_query("do the work", mode="plan->build")

    assert r["success"] is False
    assert "error" in r, r
    assert "MockLLM" in r["error"] or "script exhausted" in r["error"], r["error"]


@pytest.mark.asyncio
async def test_unknown_tool_name_surfaces_tool_not_found_not_permission_deny(tmp_path):
    """A misspelled/invented tool name (e.g. `terminal.delete`) must never be
    permission-gated into a spooky 'permission denied by user' — the registry
    is the source of truth and should tell the model the tool does not exist
    so it can retry with the real name."""
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    build, _ = await _make_agent(ws, AgentRole.BUILD, [
        {"label": "go", "content": "done", "tool_calls": []},
    ], with_planner=True)

    r = await build.execute_tool("terminal.delete", {"command": "rm foo"})
    assert r["success"] is False
    assert "Tool not found" in r["error"] and "terminal" in r["error"], r
    assert not r.get("permission_denied"), r

    ok = await build.execute_tool("terminal", {"command": "echo hi"})
    assert ok["success"] and "hi" in ok.get("stdout", "")


@pytest.mark.asyncio
async def test_permission_ask_verdict_can_be_approved(tmp_path):
    """An 'ask' verdict must be answerable (the TUI modal now does this).
    Without a confirm mechanism it auto-denies (the old bug); with one it
    runs the tool and remembers the approval for the session."""
    ws = _make_workspace(tmp_path)
    await ws.initialize()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    build, _ = await _make_agent(ws, AgentRole.BUILD, [
        {"label": "go", "content": "done", "tool_calls": []},
    ], with_planner=False)

    # Config forces everything (incl. terminal) -> ask via the opencode
    # global permission shorthand, matching the reported TUI failure.
    pm = PermissionManager(
        {"mode": "manual", "default": "allow", "permission": {"*": "ask"}},
        ui=None, input_handler=None,
    )
    build.permission_manager = pm

    # Without any confirm mechanism the ask silently denies (the TUI bug).
    denied = await build.execute_tool("terminal", {"command": "echo nope"})
    assert denied["success"] is False
    assert "denied by user" in denied.get("error", ""), denied

    # With a confirm callback (what the TUI modal now provides) it approves.
    calls = []
    async def _yes(request, verdict):
        calls.append(request.tool)
        return True
    pm.set_confirm_callback(_yes)
    ok = await build.execute_tool("terminal", {"command": "echo hi"})
    assert ok["success"] and "hi" in ok.get("stdout", "")
    assert calls == ["terminal"]

    # Session memory: the same call is auto-approved without asking again.
    calls.clear()
    ok2 = await build.execute_tool("terminal", {"command": "echo hi"})
    assert ok2["success"] and calls == []
