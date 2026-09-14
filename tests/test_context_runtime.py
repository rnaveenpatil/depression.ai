from types import SimpleNamespace

from agent.context.runtime import get_model_messages, normalize_messages


def test_tool_call_and_result_are_atomic():
    ctx = SimpleNamespace(messages=[
        SimpleNamespace(role="system", content="sys", metadata={}, pinned=True),
        SimpleNamespace(role="user", content="do it", metadata={}, pinned=False),
        SimpleNamespace(role="assistant", content="", metadata={"tool_calls":[{"id":"call-1","type":"function","function":{"name":"read","arguments":"{}"}}]}, pinned=False),
        SimpleNamespace(role="tool", content="ok", metadata={"tool_call_id":"call-1","name":"read"}, pinned=False),
        SimpleNamespace(role="assistant", content="done", metadata={}, pinned=False),
    ])
    messages = get_model_messages(ctx, limit=3)
    assert [m.role for m in messages] == ["system", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0]["id"] == "call-1"
    assert messages[2].tool_call_id == "call-1"


def test_orphan_tool_result_is_removed():
    ctx = SimpleNamespace(messages=[
        SimpleNamespace(role="system", content="sys", metadata={}, pinned=True),
        SimpleNamespace(role="tool", content="orphan", metadata={"tool_call_id":"missing"}, pinned=False),
    ])
    normalize_messages(ctx)
    assert [m.role for m in ctx.messages] == ["system"]
