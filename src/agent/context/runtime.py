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
"""
from __future__ import annotations

import json
from typing import Any, List

from agent.llm.provider import Message
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# Byte cap for a single tool result payload stored in context.
TOOL_RESULT_BYTE_CAP = 8 * 1024


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


def _truncate_payload(payload: Any, cap: int = TOOL_RESULT_BYTE_CAP) -> Any:
    """
    Truncate a tool result payload by BYTES (multibyte safe), preserving
    the shape {success, ...}. Long fields are shortened with a marker.
    """
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

    # Preserve the shape; cap the largest string values.
    trimmed = dict(payload)
    big_keys = sorted(
        (k for k, v in trimmed.items() if isinstance(v, str)),
        key=lambda k: len(trimmed[k]),
        reverse=True,
    )
    for k in big_keys:
        s = trimmed[k]
        if len(s.encode("utf-8")) <= cap // 2:
            continue
        trimmed[k] = s.encode("utf-8")[-(cap // 2):].decode("utf-8", errors="replace")
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
    """
    Group messages so an assistant with tool_calls and its ADJACENT
    tool results are always in the same group.

    Groups are always internally VALID or the assistant's tool_calls are
    stripped (see validate_group). Orphan tool messages become singletons
    (and are dropped by repair_context).
    """
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
    """True iff the group is a valid provider sequence."""
    if not group:
        return True
    first = group[0]
    if not _is_assistant_with_calls(first):
        # A bare assistant / user / system / tool message is only valid if
        # it's not followed by tool results it can't claim.
        if getattr(first, "role", "") == "tool":
            return False
        return True
    declared = _tool_call_ids(first)
    results = [m for m in group[1:] if getattr(m, "role", "") == "tool"]
    return [ _tool_result_id(m) for m in results ] == declared


# ----------------------------------------------------------------------
# Repair
# ----------------------------------------------------------------------

def repair_context(context_manager: Any, *, fill_missing: bool = True) -> int:
    """
    Repair the message list in place. Returns the number of repairs made.

    Policy:
      - Every assistant tool_call must have exactly one adjacent tool result.
      - Missing results are FILLED with a synthetic structured error
        (fill_missing=True, the default) or the assistant's tool_calls are
        stripped (fill_missing=False).
      - Orphan tool results are dropped.
      - Pinned messages and system messages are never dropped.
    """
    messages = list(getattr(context_manager, "messages", []) or [])
    if not messages:
        return 0

    repairs = 0
    out: List[Any] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        role = getattr(m, "role", "")

        # Orphan tool result — drop.
        if role == "tool":
            # Only valid if the previous appended message was the assistant
            # that declared this id. Otherwise: drop.
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

        # Assistant with tool_calls — ensure full set of results.
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
                # Strip tool_calls metadata; degrade to a plain assistant msg.
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
                    # Synthetic filler
                    out.append(_synthetic_tool_result(m, cid))
                    repairs += 1
            i = j
            continue

        # Anything else
        out.append(m)
        i += 1

    if repairs:
        logger.debug("repair_context: %d repair(s) applied", repairs)
    context_manager.messages = out
    return repairs


def _synthetic_tool_result(assistant_msg: Any, tool_call_id: str) -> Any:
    """Build a ContextMessage-shaped filler for a missing tool result."""
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

    # Minimal fallback with the same attribute surface.
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


# Back-compat alias — the old name is still imported by other modules.
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
        # OpenAI-compatible: assistant carrying tool_calls must have null content.
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


def get_model_messages(context_manager: Any, limit: int = 40_000) -> List[Message]:
    """
    Build provider messages from a token-budgeted tail of the conversation.

    `limit` is a TOKEN budget. Backward-compat: values < 500 are treated
    as a message count instead (old callers passed max_history_length=40).
    """
    # Repair once so grouping is guaranteed valid.
    repair_context(context_manager, fill_missing=True)
    messages = list(getattr(context_manager, "messages", []) or [])

    system = [m for m in messages if getattr(m, "role", "") == "system"]
    non_system = [m for m in messages if getattr(m, "role", "") != "system"]

    groups = group_messages(non_system)

    # Back-compat: small limit == message count.
    if limit < 500:
        selected: List[Any] = []
        for g in reversed(groups):
            if len(selected) + len(g) > max(1, limit):
                break
            selected[0:0] = g
        return [_to_provider_message(m) for m in system + selected]

    # Token-budget walk.
    system_tokens = sum(_estimate_msg_tokens(m) for m in system)
    budget = max(0, limit - system_tokens)
    selected = []
    used = 0
    for g in reversed(groups):
        cost = sum(_estimate_msg_tokens(m) for m in g)
        if used + cost > budget and selected:
            break
        selected[0:0] = g
        used += cost
        if used >= budget:
            break

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
    """
    Persist a tool result. Payload is truncated by bytes to a single cap so
    plan mode and free-form mode produce identical-sized messages.
    """
    payload = _truncate_payload(result)
    await context_manager.add_message(
        role="tool",
        content=json.dumps(payload, default=str),
        metadata={"tool_call_id": call.id, "name": call.name},
    )