"""Regression tests for LLM-led intent and safe workflow decisions."""
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.tui.plan_parser import parse_plan_from_text


def make_loop():
    return AgentLoop(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(tools={}),
        None,
        {},
    )


def test_plan_priority_comes_from_explicit_annotation_only():
    plan = parse_plan_from_text(
        "- [ ] Critical optional task\n"
        "- [ ] Fix bug [priority: high]\n"
        "- [ ] Nice to have [priority: low]\n"
    )
    assert plan is not None
    assert [entry.priority for entry in plan.entries] == [
        "medium", "high", "low"
    ]
    assert plan.entries[1].content == "Fix bug"


@pytest.mark.asyncio
async def test_batch_verification_checks_files_commands_and_bounds_retries():
    loop = make_loop()
    loop.verification_mode = "batch"
    loop.max_batch_verify_attempts = 3
    loop.context.unverified_mutations = [
        {"tool": "edit", "target": "src/main.py", "call_id": "1"},
        {"tool": "bash", "target": "cmd::pytest -q", "call_id": "2"},
    ]
    messages = []
    async def capture(message):
        messages.append(message)
    loop.agent.context_manager = SimpleNamespace(add_system_message=capture)
    await loop._inject_batch_verification_guidance()
    assert len(messages) == 1
    guidance = messages[0].lower()
    assert "src/main.py" in guidance
    assert "cmd::pytest -q" in guidance
    assert "command" in guidance and "exit codes" in guidance
    assert "fix only" in guidance
    assert "round 1/3" in guidance
