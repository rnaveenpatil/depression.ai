"""Runtime helpers for message integrity.

The ContextManager is the only persistent conversation store. This module
adapts its ContextMessage records to provider Message objects and
guarantees that assistant tool calls and their tool results stay as
atomic, VALID groups through trimming, compaction, and provider
serialization.

Invariants enforced here (all required by OpenAI/Anthropic-compatible APIs):
    * Every assistant message with `tool_calls` is followed by exactly one
      `tool` message per declared `tool_call_id`, in the same order.
    * Tool results only attach to the immediately preceding assistant call.
    * No orphan `tool` messages are ever emitted to the provider.
    * Trimming and grouping operate on whole, valid groups atomically.
    * The newest conversational turn is never dropped, even if it exceeds
      the requested window. [Bug 2]
    * [R5] A tool-call group is never sliced. Slicing it would separate
      the assistant tool_calls from their tool results and the provider
      would reject the request with HTTP 400 "last message must have
      role = user". If the newest group is larger than the count cap, the
      entire group is kept.
"""
from __future__ import annotations

import json
from typing import Any, List

from agent.llm.provider import Message
from agent.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_TOOL_RESULT_BYTE_CAP = 8 * 1024


def _calculate_tool_result_cap(context_window: int) -> int:
    """Calculate dynamic tool result byte cap based on model's context window."""
    if context_window <= 0:
        return DEFAULT_TOOL_RESULT_BYTE_CAP

    if context_window < 32_768:
        return 8 * 1024      # 8 KB
    elif context_window < 131_072:
        return 16 * 1024     # 16 KB
    elif context_window < 500_000:
        return 64 * 1024     # 64 KB
    elif context_window < 1_000_000:
        return 128 * 1024    # 128 KB
    else:
        cap = int(context_window * 4 * 0.02)
        return min(512 * 1024, max(128 * 1024, cap))


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _tool_calls(message: Any) -> List[dict]:
    return list((getattr(message, "metadata", {}) or {}).get("tool_calls") or [])


def _tool_call_ids(message: Any) -> List[str]:
    out: List[str] = []
    for c in _tool_calls(message):
        cid = c.get("id")
        if cid:
            out.append(str(cid))
    return out


def _is_assistant_with_calls(m: Any) -> bool:
    return getattr(m, "role", "") == "assistant" and bool(_tool_calls(m))


def _tool_result_id(m: Any) -> str:
    return str((getattr(m, "metadata", {}) or {}).get("tool_call_id") or "")


def _truncate_payload(payload: Any, cap: int = DEFAULT_TOOL_RESULT_BYTE_CAP) -> Any:
    if not isinstance(payload, dict):
        try:
            payload = {"success": True, "result": payload}
        except Exception:
            return {"success": False, "error": "unserializable tool result"}
    try:
        encoded = json.dumps(payload, default=str)
    except Exception:
        return {"success": False, "error": "tool result not JSON-serializable"}
    if len(encoded.encode("utf-8")) <= cap:
        return payload

    trimmed = dict(payload)
    big_keys = sorted(
        (k for k, v in trimmed.items() if isinstance(v, str)),
        key=lambda k: len(trimmed[k]),
        reverse=True,
    )
    # Target half the cap per field, so a few big fields together still fit.
    target_bytes = max(256, cap // 2)

    for k in big_keys:
        s = trimmed[k]
        if len(s.encode("utf-8")) <= target_bytes:
            continue

        # [R7] Slice by CHARACTERS, never bytes. A byte slice can land in
        # the middle of a multi-byte UTF-8 sequence and errors="replace"
        # would insert U+FFFD into the text the model reads. Work in
        # characters and shrink from the front until the byte length fits.
        max_chars = max(64, target_bytes // 4)  # conservative: ≤ 4 bytes/char
        tail = s[-max_chars:] if len(s) > max_chars else s
        while len(tail.encode("utf-8")) > target_bytes and len(tail) > 1:
            # Drop 1/8 of the front each pass; converges quickly.
            drop = max(1, len(tail) // 8)
            tail = tail[drop:]
        trimmed[k] = tail
        trimmed["_truncated"] = True
        try:
            if len(json.dumps(trimmed, default=str).encode("utf-8")) <= cap:
                return trimmed
        except Exception:
            pass
    return trimmed


# ----------------------------------------------------------------------
# Grouping (used by runtime and compaction)
# ----------------------------------------------------------------------

def group_messages(messages: List[Any]) -> List[List[Any]]:
    groups: List[List[Any]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if _is_assistant_with_calls(m):
            declared = _tool_call_ids(m)
            group = [m]
            j = i + 1
            seen = set()
            while j < n and getattr(messages[j], "role", "") == "tool":
                tid = _tool_result_id(messages[j])
                if tid and tid in declared and tid not in seen:
                    group.append(messages[j])
                    seen.add(tid)
                j += 1
            groups.append(group)
            i = j
        else:
            groups.append([m])
            i += 1
    return groups


def validate_group(group: List[Any]) -> bool:
    if not group:
        return True
    first = group[0]
    if not _is_assistant_with_calls(first):
        if getattr(first, "role", "") == "tool":
            return False
        return True
    declared = _tool_call_ids(first)
    results = [m for m in group[1:] if getattr(m, "role", "") == "tool"]
    return [_tool_result_id(m) for m in results] == declared


# ----------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------

def repair_context(context_manager: Any, *, fill_missing: bool = True) -> int:
    messages = list(getattr(context_manager, "messages", []) or [])
    if not messages:
        return 0

    repairs = 0
    out: List[Any] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        role = getattr(m, "role", "")

        if role == "tool":
            tid = _tool_result_id(m)
            prev = out[-1] if out else None
            if (
                prev is not None
                and _is_assistant_with_calls(prev)
                and tid in _tool_call_ids(prev)
            ):
                out.append(m)
            else:
                repairs += 1
            i += 1
            continue

        if _is_assistant_with_calls(m):
            declared = _tool_call_ids(m)
            results_by_id: dict[str, Any] = {}
            j = i + 1
            while j < n and getattr(messages[j], "role", "") == "tool":
                tid = _tool_result_id(messages[j])
                if tid in declared and tid not in results_by_id:
                    results_by_id[tid] = messages[j]
                j += 1

            missing = [cid for cid in declared if cid not in results_by_id]

            if missing and not fill_missing:
                try:
                    m.metadata = dict(getattr(m, "metadata", {}) or {})
                    m.metadata.pop("tool_calls", None)
                except Exception:
                    pass
                out.append(m)
                repairs += 1
                i = j
                continue

            out.append(m)
            for cid in declared:
                if cid in results_by_id:
                    out.append(results_by_id[cid])
                else:
                    out.append(_synthetic_tool_result(m, cid))
                    repairs += 1
            i = j
            continue

        out.append(m)
        i += 1

    if repairs:
        logger.debug("repair_context: %d repair(s) applied", repairs)
    context_manager.messages = out
    return repairs


def _synthetic_tool_result(assistant_msg: Any, tool_call_id: str) -> Any:
    try:
        from agent.context.manager import ContextMessage
    except Exception:
        ContextMessage = None  # type: ignore

    payload = {
        "success": False,
        "error": "tool result missing (dropped during trimming)",
        "recoverable": True,
        "suggestion": "Re-issue the tool call if the result is still needed.",
    }
    content = json.dumps(payload, default=str)
    meta = {
        "tool_call_id": tool_call_id,
        "name": "unknown",
        "synthetic": True,
    }
    if ContextMessage is not None:
        try:
            return ContextMessage(
                role="tool",
                content=content,
                metadata=meta,
                tokens=max(1, len(content) // 4),
                timestamp=getattr(assistant_msg, "timestamp", 0) + 1e-6,
            )
        except Exception:
            pass

    class _Synthetic:
        __slots__ = ("role", "content", "tokens", "pinned", "metadata", "timestamp")
        def __init__(self) -> None:
            self.role = "tool"
            self.content = content
            self.tokens = max(1, len(content) // 4)
            self.pinned = False
            self.metadata = meta
            self.timestamp = getattr(assistant_msg, "timestamp", 0) + 1e-6
    return _Synthetic()


def normalize_messages(context_manager: Any) -> None:
    repair_context(context_manager, fill_missing=True)


# ----------------------------------------------------------------------
# Provider serialization
# ----------------------------------------------------------------------

def _to_provider_message(cm: Any) -> Message:
    metadata = getattr(cm, "metadata", {}) or {}
    kwargs: dict[str, Any] = {"role": cm.role, "content": cm.content}
    if cm.role == "assistant" and metadata.get("tool_calls"):
        kwargs["tool_calls"] = metadata["tool_calls"]
        # OpenAI-compatible APIs want content=None for tool-call-only turns.
        kwargs["content"] = None
    if cm.role == "tool":
        kwargs["name"] = metadata.get("name")
        kwargs["tool_call_id"] = metadata.get("tool_call_id")
    return Message(**kwargs)


def _estimate_msg_tokens(cm: Any) -> int:
    try:
        from agent.context.compaction import _estimate_message_tokens
        return _estimate_message_tokens(cm)
    except Exception:
        content = getattr(cm, "content", "") or ""
        return max(1, len(content) // 4)


def _interleave_systems_and_selected(
    messages: List[Any], selected: List[Any]
) -> List[Any]:
    """
    Return messages in the correct order for the provider.

    [M1] System messages are interleaved by timestamp with the selected
    tail, not hoisted to the front. Hoisting made stale plan-guidance
    system messages read like they were current framing.
    """
    combined = list(messages) + list(selected)
    combined.sort(key=lambda m: getattr(m, "timestamp", 0))
    seen = set()
    out: List[Any] = []
    for m in combined:
        if id(m) in seen:
            continue
        seen.add(id(m))
        out.append(m)
    return out


def get_model_messages(context_manager: Any, limit: int = 40_000) -> List[Message]:
    """
    Build provider messages from a token-budgeted tail of the conversation.

    [Bug 2] The count branch (limit < 500) never drops the newest turn.
    [R5]    It also never SLICES the newest group. Slicing a tool-call
            group separates the assistant tool_calls from their tool
            results, and the provider rejects the resulting array with
            HTTP 400 "last message must have role = user". If the newest
            group exceeds the count cap, the whole group is kept.
    """
    repair_context(context_manager, fill_missing=True)
    messages = list(getattr(context_manager, "messages", []) or [])

    system = [m for m in messages if getattr(m, "role", "") == "system"]
    non_system = [m for m in messages if getattr(m, "role", "") != "system"]

    groups = group_messages(non_system)

    # Back-compat: small limit == message count.
    if limit < 500:
        cap = max(1, limit)
        selected: List[Any] = []
        for g in reversed(groups):
            if len(selected) + len(g) > cap:
                if selected:
                    break
                # [R5] Never drop the newest turn AND never slice a group.
                # Keep the entire newest group even if it exceeds cap. The
                # provider's context window is the real limit; the loop
                # has a compaction path for genuine overflow.
                selected[0:0] = list(g)
                break
            selected[0:0] = g
        return [
            _to_provider_message(m)
            for m in _interleave_systems_and_selected(system, selected)
        ]

    # Token-budget walk.
    system_tokens = sum(_estimate_msg_tokens(m) for m in system)
    budget = max(0, limit - system_tokens)
    selected = []
    used = 0
    for g in reversed(groups):
        cost = sum(_estimate_msg_tokens(m) for m in g)
        if used + cost > budget:
            if not selected:
                # Same rule as above: keep the whole newest group. If it
                # does not fit the budget, the provider's own context
                # window will trim on its side, but at least the group
                # arrives intact.
                selected[0:0] = list(g)
            break
        selected[0:0] = g
        used += cost
        if used >= budget:
            break

    return [
        _to_provider_message(m)
        for m in _interleave_systems_and_selected(system, selected)
    ]


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


async def add_tool_result(
    context_manager: Any,
    call: Any,
    result: Any,
    context_window: int = 0,
) -> None:
    cap = _calculate_tool_result_cap(context_window) if context_window > 0 else DEFAULT_TOOL_RESULT_BYTE_CAP
    payload = _truncate_payload(result, cap=cap)
    await context_manager.add_message(
        role="tool",
        content=json.dumps(payload, default=str),
        metadata={"tool_call_id": call.id, "name": call.name},
    )