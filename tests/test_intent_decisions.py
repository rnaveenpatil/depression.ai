"""Regression tests for LLM-led intent and safe workflow decisions."""
from types import SimpleNamespace

import pytest

from agent.agent.loop import AgentLoop
from agent.llm.provider import LLMResponse
from agent.tui.plan_parser import parse_plan_from_text


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(
            content=next(self.responses),
            model="test",
            provider="test",
        )


def make_loop(responses):
    return AgentLoop(
        SimpleNamespace(),
        FakeModel(responses),
        SimpleNamespace(tools={}),
        None,
        {"enable_intent_classification": True, "enable_planning": True},
    )


@pytest.mark.asyncio
async def test_question_with_edit_request_is_action_and_planned():
    loop = make_loop([
        '{"intent":"feature","needs_plan":true,'
        '"wants_launch":false,"requires_action":true}'
    ])
    intent = await loop._classify_intent(
        "How can you improve this project? Implement the changes too."
    )
    assert intent == "feature"
    assert loop.context.metadata["requires_action"] is True
    assert await loop._should_plan("Improve this project", intent) is True
    assert loop.llm.calls == 1


@pytest.mark.asyncio
async def test_show_code_does_not_mean_launch():
    loop = make_loop([
        '{"intent":"question","needs_plan":false,'
        '"wants_launch":false,"requires_action":false}'
    ])
    intent = await loop._classify_intent("Show me the source code")
    assert intent == "question"
    assert loop.context.user_asked_to_show is False
    assert await loop._should_plan("Show me the source code", intent) is False


@pytest.mark.asyncio
async def test_single_word_provider_falls_back_to_model_plan_decision():
    loop = make_loop(["feature", "YES"])
    intent = await loop._classify_intent("Make the project changes")
    assert intent == "feature"
    assert await loop._should_plan("Make the project changes", intent) is True
    assert loop.llm.calls == 2


@pytest.mark.asyncio
async def test_invalid_classification_does_not_disable_execution():
    loop = make_loop(["unstructured model response", "INVALID"])
    assert await loop._classify_intent("Build a new feature") == "unknown"
    assert await loop._should_plan("Build a new feature") is False


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
    loop = make_loop([])
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
