"""Regression tests for AgentLoop verification, safety, and run isolation."""
from __future__ import annotations

import subprocess
import time
from types import SimpleNamespace

import pytest

from agent.agent.loop import (
    AgentLoop,
    _bash_fallback_edit,
    _bash_fallback_write,
    _looks_destructive_call,
)
from agent.llm.provider import LLMResponse, ToolCall


class FakeContext:
    def __init__(self):
        self.messages = []

    async def add_message(self, role, content, metadata=None, pinned=False, **kwargs):
        self.messages.append(
            SimpleNamespace(
                role=role,
                content=content,
                metadata=metadata or {},
                pinned=pinned,
            )
        )

    async def add_system_message(self, content, pinned=True, **kwargs):
        await self.add_message("system", content, {"pinned": pinned}, pinned=pinned)

    async def add_assistant_message(self, content, metadata=None, **kwargs):
        await self.add_message("assistant", content, metadata or {})


class FakeWorkspace:
    def __init__(self, project_dir):
        self.project_dir = str(project_dir)

    async def list_files(self, max_files=10):
        return []


class FakeAgent:
    def __init__(self, tool_impl, workspace=None, permission_manager=None):
        self.context_manager = FakeContext()
        self.workspace = workspace
        self.permission_manager = permission_manager
        self.tool_impl = tool_impl
        self.executed = []

    async def execute_tool(self, name, args):
        self.executed.append((name, args))
        return self.tool_impl(name, args)


class FakeRegistry:
    def __init__(self, read_only=("read",), tools=("read", "write", "bash", "delete")):
        self._read_only = set(read_only)
        self.tools = {name: object() for name in tools}
        self.mutation_epoch = 0

    def list_tools(self):
        return list(self.tools)

    def has_tool(self, name):
        return name in self.tools

    def is_read_only(self, name):
        return name in self._read_only

    def get_schemas(self):
        return [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in self.tools
        ]

    def select_for_task(self, intent):
        return self.get_schemas()


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.model = "test/model"

    def get_current_model(self):
        return self.model

    def get_current_provider(self):
        return "test"

    async def complete_with_tools(self, messages=None, **kwargs):
        self.calls += 1
        if not self.responses:
            return LLMResponse(content="done", model=self.model, provider="test")
        return self.responses.pop(0)

    async def complete(self, messages=None, **kwargs):
        self.calls += 1
        if not self.responses:
            return LLMResponse(content="final summary", model=self.model, provider="test")
        return self.responses.pop(0)


def tool_response(*calls):
    return LLMResponse(
        content="",
        model="test/model",
        provider="test",
        tool_calls=list(calls),
    )


def text_response(text):
    return LLMResponse(content=text, model="test/model", provider="test")


def make_loop(responses, tool_impl=None, **config):
    def default_tool(name, args):
        return {"success": True, "tool": name, "content": "ok"}

    agent = FakeAgent(tool_impl or default_tool)
    llm = ScriptedLLM(responses)
    loop = AgentLoop(
        agent,
        llm,
        FakeRegistry(),
        None,
        {
            "enable_intent_classification": False,
            "enable_planning": False,
            "enable_streaming": False,
            "require_demo_offer": False,
            "verification_mode": "batch",
            **config,
        },
    )
    return loop, agent, llm


@pytest.mark.asyncio
async def test_batch_verification_stops_after_all_targets_are_read():
    loop, _agent, llm = make_loop([
        tool_response(
            ToolCall(id="w1", name="write", arguments={"filePath": "a.txt", "content": "a"}),
            ToolCall(id="w2", name="write", arguments={"filePath": "b.txt", "content": "b"}),
        ),
        text_response("done"),
        tool_response(
            ToolCall(id="r1", name="read", arguments={"filePath": "a.txt"}),
            ToolCall(id="r2", name="read", arguments={"filePath": "b.txt"}),
        ),
        text_response("all verified"),
    ])

    result = await loop.run("fix both files")

    assert result["success"] is True, result
    assert loop.context.batch_verify_attempts == 1
    assert "UNVERIFIED" not in result["response"]
    assert llm.calls == 4


@pytest.mark.asyncio
async def test_batch_verification_reports_remaining_targets_after_budget():
    loop, _agent, _llm = make_loop(
        [
            tool_response(
                ToolCall(
                    id="w1",
                    name="write",
                    arguments={"filePath": "a.txt", "content": "a"},
                ),
                ToolCall(
                    id="w2",
                    name="write",
                    arguments={"filePath": "b.txt", "content": "b"},
                ),
            ),
            text_response("done"),
            text_response("done"),
            text_response("done"),
            text_response("done"),
            text_response("done"),
        ],
        max_batch_verify_attempts=4,
    )

    result = await loop.run("fix the file")

    assert result["success"] is False, result
    assert "UNVERIFIED" in result["response"]
    assert "a.txt" in result["response"]
    assert result["unverified_targets"] == ["a.txt", "b.txt"]
    assert loop.context.batch_verify_attempts == 4


@pytest.mark.asyncio
async def test_plan_task_read_clears_only_the_target_it_checked():
    loop, agent, _llm = make_loop([
        tool_response(
            ToolCall(id="w1", name="write", arguments={"filePath": "a.txt", "content": "a"}),
            ToolCall(id="w2", name="write", arguments={"filePath": "b.txt", "content": "b"}),
        ),
        tool_response(
            ToolCall(id="r1", name="read", arguments={"filePath": "a.txt"}),
        ),
        text_response("TASK COMPLETE: one file checked"),
        tool_response(
            ToolCall(id="r2", name="read", arguments={"filePath": "b.txt"}),
        ),
        text_response("TASK COMPLETE: both files checked"),
    ])
    task = SimpleNamespace(
        id="t1",
        description="change both files",
        dependencies=[],
        metadata={},
        created_at=time.time(),
        status=None,
        result=None,
        actual_time=0.0,
    )
    plan = SimpleNamespace(get_task_by_id=lambda _id: None)

    result = await loop._execute_task_turn(
        task=task,
        plan=plan,
        system_prompt="system",
    )

    assert result["success"] is True, result
    assert not (task.metadata or {}).get("unverified")
    assert loop.context.unverified_mutations == []
    # Plan-task turns are local; no orphan tool results in canonical context.
    assert not [m for m in agent.context_manager.messages if m.role == "tool"]


def test_fallback_edit_fails_when_old_string_is_missing(tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    command = _bash_fallback_edit(
        {"filePath": str(target), "oldString": "missing", "newString": "x"}
    )["command"]

    proc = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True, text=True)

    assert proc.returncode != 0
    assert target.read_text(encoding="utf-8") == "hello"


def test_fallback_write_treats_heredoc_marker_as_literal_content(tmp_path):
    target = tmp_path / "note.txt"
    payload = "hello\nAGENT_EOF\ntouch injected.txt\n"
    command = _bash_fallback_write(
        {"filePath": str(target), "content": payload}
    )["command"]

    proc = subprocess.run(command, shell=True, cwd=tmp_path, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert target.read_text(encoding="utf-8") == payload
    assert not (tmp_path / "injected.txt").exists()


def test_fallback_write_rejects_path_escape(tmp_path):
    loop, agent, _llm = make_loop([], tool_impl=lambda name, args: {"success": True})
    agent.workspace = FakeWorkspace(tmp_path)

    assert loop._fallback_path_allowed("write", {"filePath": "inside.txt"}) is True
    assert loop._fallback_path_allowed("write", {"filePath": "/etc/passwd"}) is False


@pytest.mark.asyncio
async def test_max_turns_does_not_leak_into_the_next_run():
    loop, _agent, _llm = make_loop([text_response("ok")], max_iterations=7)

    await loop.run("hello", max_turns=2)
    assert loop.max_iterations == 7

    await loop.run("hello again")
    assert loop.max_iterations == 7


@pytest.mark.asyncio
async def test_system_prompt_is_replaced_when_fingerprint_changes(tmp_path):
    loop, agent, _llm = make_loop([], project_context_ttl=0)
    agent.workspace = FakeWorkspace(tmp_path / "alpha")

    await loop._ensure_system_prompt()
    first = [
        m for m in agent.context_manager.messages
        if m.role == "system" and m.metadata.get("system_prompt")
    ]
    assert len(first) == 1
    assert "alpha" in first[0].content

    agent.workspace.project_dir = str(tmp_path / "beta")
    await loop._ensure_system_prompt()

    systems = [
        m for m in agent.context_manager.messages
        if m.role == "system" and m.metadata.get("system_prompt")
    ]
    assert len(systems) == 1
    assert "beta" in systems[0].content
    assert "alpha" not in systems[0].content


@pytest.mark.asyncio
async def test_repeated_identical_bash_calls_are_not_blocked():
    loop, agent, _llm = make_loop([
        tool_response(ToolCall(id="b1", name="bash", arguments={"command": "echo hi"})),
        tool_response(ToolCall(id="b2", name="bash", arguments={"command": "echo hi"})),
        text_response("done"),
    ], verification_mode="off")

    result = await loop.run("run the check twice")

    assert result["success"] is True, result
    assert [name for name, _args in agent.executed] == ["bash", "bash"]


def test_destructive_scan_ignores_file_content_but_checks_shell_commands():
    assert _looks_destructive_call("write", {"filePath": "x", "content": "rm -rf /"}) is False
    assert _looks_destructive_call("bash", {"command": "rm -rf /"}) is True


@pytest.mark.asyncio
async def test_permission_denial_is_recorded_in_telemetry():
    loop, agent, _llm = make_loop([])
    assert agent.permission_manager is None

    result = await loop._act_one(
        ToolCall(id="d1", name="delete", arguments={"path": "x"}),
        persist=False,
    )

    assert result["result"]["permission_denied"] is True
    assert loop.context.actions_taken[-1]["permission_denied"] is True
    assert loop.completed_tool_calls[-1].id == "d1"


@pytest.mark.asyncio
async def test_plan_task_mutations_do_not_pollute_global_verification():
    loop, _agent, _llm = make_loop([])

    await loop._act(
        [ToolCall(id="w1", name="write", arguments={"filePath": "a.txt", "content": "a"})],
        persist=False,
        track_verification=False,
    )

    assert loop.context.unverified_mutations == []
    assert loop.context.batch_changed_targets == []


def test_loop_status_reports_verification_nudges_as_a_count():
    loop, _agent, _llm = make_loop([])
    loop.context.verification_nudges = 2

    assert loop.get_loop_status()["verification_nudges"] == 2


@pytest.mark.asyncio
async def test_simple_shell_task_does_not_trigger_verification():
    loop, agent, llm = make_loop([
        tool_response(ToolCall(id="b1", name="bash", arguments={"command": "ls"})),
        text_response("Listed the files."),
    ])

    result = await loop.run("list the files")

    assert result["success"] is True, result
    assert llm.calls == 2
    assert loop.context.batch_verify_attempts == 0
    assert loop.context.unverified_mutations == []
    assert [name for name, _args in agent.executed] == ["bash"]


@pytest.mark.asyncio
async def test_simple_write_is_auto_verified_without_extra_llm_turn():
    loop, agent, llm = make_loop([
        tool_response(
            ToolCall(
                id="w1",
                name="write",
                arguments={"filePath": "hello.txt", "content": "hi"},
            )
        ),
        text_response("Created hello.txt."),
    ])

    result = await loop.run("create hello.txt")

    assert result["success"] is True, result
    assert llm.calls == 2
    assert loop.context.batch_verify_attempts == 0
    assert loop.context.unverified_mutations == []
    assert ("read", {"filePath": "hello.txt"}) in agent.executed
    assert loop.context.metadata["auto_verified_targets"] == ["hello.txt"]


@pytest.mark.asyncio
async def test_user_stop_produces_a_summary_without_another_llm_call():
    loop, _agent, llm = make_loop([])

    async def fake_think():
        loop.should_stop = True
        return {"response": "partial work", "tool_calls": []}

    loop._think = fake_think
    result = await loop.run("do a long task")

    assert result["success"] is False
    assert result["error"] == "stopped by user"
    assert "Stopped as requested" in result["response"]
    assert llm.calls == 0


def test_stop_disables_batch_verification():
    loop, _agent, _llm = make_loop([])
    loop.context.unverified_mutations = [
        {"tool": "write", "target": "a.txt"}
    ]
    loop.should_stop = True

    assert loop._should_batch_verify("done") is False


def test_successful_reads_are_not_counted_as_task_progress():
    loop, _agent, _llm = make_loop([])
    assert loop._made_progress([
        {"tool": "read", "result": {"success": True, "content": "file data"}}
    ]) is False
    assert loop._made_progress([
        {"tool": "grep", "result": {"success": True, "content": "match"}}
    ]) is False


def test_mutations_still_count_as_progress_but_noops_do_not():
    loop, _agent, _llm = make_loop([])
    assert loop._made_progress([
        {"tool": "write", "result": {"success": True, "changed": True}}
    ]) is True
    assert loop._made_progress([
        {"tool": "write", "result": {"success": True, "changed": False}}
    ]) is False
    assert loop._made_progress([
        {"tool": "read", "cached": True, "result": {"success": True}}
    ]) is False


@pytest.mark.asyncio
async def test_repeated_reads_stop_without_indefinite_loop():
    calls = [
        tool_response(ToolCall(id=f"r{i}", name="read", arguments={"filePath": f"file{i}.py"}))
        for i in range(12)
    ]
    loop, agent, _llm = make_loop(
        calls, no_progress_streak_cap=6, max_iterations=30
    )
    result = await loop.run("inspect these files")
    assert loop.context.metadata.get("stopped_on_no_progress") is True
    assert len(agent.executed) <= 6
    assert result["success"] is False


@pytest.mark.asyncio
async def test_greeting_does_not_expose_tools_or_invoke_file_reads():
    loop, agent, _llm = make_loop([text_response("Hello!")])
    result = await loop.run("hello")
    assert result["success"] is True
    assert agent.executed == []
    assert loop.context.metadata["direct_greeting"] is True
    assert loop._get_available_tools() == []


@pytest.mark.asyncio
async def test_coding_request_keeps_all_tools_available():
    loop, _agent, _llm = make_loop([text_response("ready")])
    await loop.run("fix the parser bug")
    assert loop.context.metadata["direct_greeting"] is False
    assert loop._get_available_tools()


@pytest.mark.asyncio
async def test_identical_read_batches_stop_early():
    calls = [
        tool_response(ToolCall(id=f"repeat{i}", name="read",
                               arguments={"filePath": "same.py"}))
        for i in range(12)
    ]
    loop, agent, _llm = make_loop(calls, no_progress_streak_cap=12)
    result = await loop.run("inspect the same file")
    assert loop.context.metadata.get("stopped_on_no_progress") is True
    assert len(agent.executed) <= 3
    assert result["success"] is False


@pytest.mark.asyncio
async def test_read_only_answer_stops_after_one_observation():
    loop, agent, llm = make_loop([
        tool_response(ToolCall(id="read-one", name="read",
                               arguments={"filePath": "target.py"})),
        text_response("The function returns the requested value."),
    ])
    result = await loop.run("explain target.py")
    assert result["success"] is True
    assert llm.calls == 2
    assert len(agent.executed) == 1
