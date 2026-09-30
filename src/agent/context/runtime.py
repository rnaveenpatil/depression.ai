"""Runtime helpers for message integrity.

The ContextManager is the only persistent conversation store. This module
adapts its ContextMessage records to provider Message objects and
guarantees that assistant tool calls and their tool results stay as
atomic groups through trimming, compaction, and provider serialization.
"""
from __future__ import annotations

import json
from typing import Any, List

from agent.llm.provider import Message


def _tool_calls(message: Any) -> List[dict]:
    calls = (getattr(message, "metadata", {}) or {}).get("tool_calls") or []
    return list(calls)


def _result_ids_for(messages: List[Any]) -> set:
    out: set = set()
    for m in messages:
        if getattr(m, "role", "") != "tool":
            continue
        tid = str((getattr(m, "metadata", {}) or {}).get("tool_call_id") or "")
        if tid:
            out.add(tid)
    return out


def normalize_messages(context_manager: Any) -> None:
    """
    Repair the message list after trimming or compaction.

    Policy:
      - An assistant message with tool_calls is kept REGARDLESS of whether
        every tool result is present. Losing the assistant turn is worse
        than losing a tool result — the model would forget it ever tried.
      - A tool result whose tool_call_id has no matching assistant call is
        dropped (providers reject orphan tool messages).
      - A tool result whose tool_call_id belongs to an assistant call that
        IS present is kept.
    """
    messages = list(getattr(context_manager, "messages", []) or [])
    if not messages:
        return

    # First pass: collect the set of tool_call ids that any assistant message
    # in this list references.
    referenced_ids: set = set()
    for m in messages:
        if getattr(m, "role", "") == "assistant":
            for c in _tool_calls(m):
                cid = c.get("id")
                if cid:
                    referenced_ids.add(str(cid))

    valid: List[Any] = []
    for m in messages:
        role = getattr(m, "role", "")
        if role == "tool":
            tid = str((getattr(m, "metadata", {}) or {}).get("tool_call_id") or "")
            if tid and tid in referenced_ids:
                valid.append(m)
            # else: orphan tool result — drop it
            continue
        # assistant (with or without tool_calls), user, system — keep
        valid.append(m)

    context_manager.messages = valid


def _to_provider_message(cm: Any) -> Message:
    metadata = getattr(cm, "metadata", {}) or {}
    kwargs: dict[str, Any] = {"role": cm.role, "content": cm.content}
    if cm.role == "assistant" and metadata.get("tool_calls"):
        # OpenAI-compatible spec: assistant messages carrying tool_calls must
        # have "content": null (strict gateways like NIM/vLLM reject "").
        kwargs["tool_calls"] = metadata["tool_calls"]
        kwargs["content"] = None
    if cm.role == "tool":
        kwargs["name"] = metadata.get("name")
        kwargs["tool_call_id"] = metadata.get("tool_call_id")
    return Message(**kwargs)


def get_model_messages(context_manager: Any, limit: int = 40) -> List[Message]:
    """
    Build provider messages without splitting a tool-call group.

    An "atomic group" is an assistant message carrying tool_calls plus all
    of its adjacent tool result messages. Groups are always kept whole or
    dropped whole.
    """
    normalize_messages(context_manager)
    messages = list(getattr(context_manager, "messages", []) or [])

    system = [m for m in messages if getattr(m, "role", "") == "system"]
    non_system = [m for m in messages if getattr(m, "role", "") != "system"]

    groups: list[list[Any]] = []
    i = 0
    while i < len(non_system):
        m = non_system[i]
        if getattr(m, "role", "") == "assistant" and _tool_calls(m):
            group = [m]
            ids = {str(c.get("id")) for c in _tool_calls(m) if c.get("id")}
            j = i + 1
            while j < len(non_system) and getattr(non_system[j], "role", "") == "tool":
                tid = str((getattr(non_system[j], "metadata", {}) or {}).get("tool_call_id") or "")
                if tid in ids:
                    group.append(non_system[j])
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([m])
            i += 1

    selected: list[Any] = []
    for group in reversed(groups):
        if len(selected) + len(group) > max(1, limit):
            break
        selected[0:0] = group

    return [_to_provider_message(m) for m in system + selected]


async def add_tool_call(
    context_manager: Any, content: str | None, tool_calls: List[Any]
) -> None:
    payload = []
    for call in tool_calls:
        payload.append({
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, default=str),
            },
        })
    await context_manager.add_message(
        role="assistant",
        content=content or "",
        metadata={"tool_calls": payload},
    )


async def add_tool_result(context_manager: Any, call: Any, result: Any) -> None:
    await context_manager.add_message(
        role="tool",
        content=json.dumps(result, default=str),
        metadata={"tool_call_id": call.id, "name": call.name},
    )