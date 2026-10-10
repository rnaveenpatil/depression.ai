"""Agent execution loop.

Flow: session context -> LLM -> tool calls -> permission/execution -> tool
results -> canonical context -> LLM, repeated until the model returns a final
answer. AgentLoop owns execution telemetry only; ContextManager owns history.

Consistency guarantees enforced here:
    * The tool list the model sees == the tools the runtime can execute
      (single source: ToolRegistry.get_schemas(), narrowed by intent).
    * Every assistant tool_call gets exactly one tool result before the next
      LLM turn (no orphan tool_call_ids).
    * Tool failures are injected back as structured guidance so the agent
      can self-correct instead of stalling. The guidance survives the API
      sanitizer (converted to user, not dropped).
    * Success is only reported when a tool output in this session proves it.
    * Cache is invalidated whenever a mutating tool runs.
    * System prompt is rebuilt when its fingerprint changes and is tagged
      with metadata so the loop can detect it.
    * The loop NEVER writes the final assistant turn — that is the caller's
      job so each user query produces exactly one assistant message.
    * [Permission gate] Every tool call is dispatched through the AGENT's
      execute_tool(), which runs the permission manager. The registry's
      raw execute_safe is only used when the agent has no execute_tool.
    * [Retry policy] Empty/incomplete turns are retried once with a
      doubled output budget and a nudge. Failed tool calls are retried
      once via a bash/terminal fallback with transformed args.
    * [Task completion] The agent is instructed to finish the whole task.
      It stops only for destructive approval, real ambiguity, or a
      concrete blocker.
    * [API contract] Every request sent to the LLM is sanitized: it ends
      in a `user` or `tool` message, and every assistant turn that carries
      tool_calls has matching tool results.
    * [Token budget] Tool schemas travel in `tools=[...]` only.
    * [Fast tools] Multiple read-only tool calls in one turn execute
      concurrently.
    * [Tool result cap] Tool results are capped by a byte budget scaled
      from the model's context window.
    * [Plan surface] Whenever a checklist appears in the model's reply
      OR the planner produces one, an `on_plan_rendered` event fires so
      the UI can render it.
    * [Show the work] When the user says run/launch/show, the agent runs
      the artifact and prints the real output in the reply.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.agent.planner import Plan, Planner, TaskStatus
from agent.llm.provider import Message, ToolCall, FinishReason
from agent.llm.runtime import MODEL_METADATA, LLMProviderRegistry
from agent.tools.registry import ToolRegistry, INTENT_CATEGORIES
from agent.utils.errors import TimeoutError
from agent.utils.logging import get_logger
from agent.utils.redact import redact
from agent.context.runtime import add_tool_call, add_tool_result, get_model_messages

logger = get_logger(__name__)


PLAN_DRIVEN_MIN_TASKS = 3
PLAN_TASK_MAX_ATTEMPTS = 3

_QUESTION_PREFIXES = (
    "what", "why", "how", "when", "where", "who", "which",
    "explain", "describe", "tell me", "show me",
)

SYSTEM_PROMPT_MARKER = "system_prompt"

_UNVERIFIED_MARKER = re.compile(
    r"^\s*UNVERIFIED\s*:\s*(.+?)\s*$",
    re.MULTILINE,
)
_UNVERIFIED_MAX_POS = 200

TOTAL_FAILURE_CAP = 15
PER_TOOL_FAILURE_CAP = 5
SAME_ARGS_FAILURE_CAP = 2

# Detect "run/launch/show me" style requests so the system prompt can
# demand an actual demonstration, not a description.
_LAUNCH_INTENT_PATTERNS = (
    re.compile(r"\b(run|launch|start|open|execute|serve)\b", re.I),
    re.compile(r"\bshow\s+me\b", re.I),
    re.compile(r"\bin\s+(?:the\s+)?browser\b", re.I),
    re.compile(r"\bsee\s+it\b", re.I),
)


# ----------------------------------------------------------------------
# Bash/terminal fallback transforms
# ----------------------------------------------------------------------

def _sh_single(s: str) -> str:
    return str(s).replace("'", "'\\''")


def _fallback_error_command(msg: str) -> Dict[str, Any]:
    return {"command": f"echo {_sh_single('FALLBACK ERROR: ' + msg)} >&2; exit 1"}


def _bash_fallback_write(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    content = args.get("content")
    if content is None:
        content = args.get("text") or args.get("data") or ""
    if not path:
        return _fallback_error_command(
            "write called without filePath/path; cannot fall back to bash."
        )
    return {
        "command": (
            f"cat > '{_sh_single(path)}' << 'AGENT_EOF'\n{content}\nAGENT_EOF"
        )
    }


def _bash_fallback_read(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    if not path:
        return _fallback_error_command(
            "read called without filePath/path; cannot fall back to bash."
        )
    return {"command": f"cat '{_sh_single(path)}'"}


def _bash_fallback_list(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path") or args.get("dir") or "."
    return {"command": f"ls -la '{_sh_single(path)}'"}


def _bash_fallback_edit(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("filePath") or args.get("path") or args.get("filename") or ""
    old_str = args.get("oldString") or args.get("old_text") or ""
    new_str = args.get("newString") or args.get("new_text") or ""
    if not path:
        return _fallback_error_command(
            "edit called without filePath/path; cannot fall back to bash."
        )
    if not old_str:
        return _fallback_error_command(
            "edit called without oldString; cannot fall back to bash."
        )
    script = (
        "import sys,pathlib;"
        "p=pathlib.Path(sys.argv[1]);"
        "s=p.read_text();"
        "old=sys.argv[2];new=sys.argv[3];"
        "sys.exit(0 if old not in s else (p.write_text(s.replace(old,new,1)) or 0))"
    )
    return {
        "command": (
            f"python3 -c {_sh_single(script)} "
            f"'{_sh_single(path)}' "
            f"'{_sh_single(old_str)}' "
            f"'{_sh_single(new_str)}'"
        )
    }


def _bash_fallback_glob(args: Dict[str, Any]) -> Dict[str, Any]:
    root = args.get("path") or args.get("dir") or "."
    pattern = args.get("pattern") or args.get("glob") or "*"
    return {
        "command": (
            f"find '{_sh_single(root)}' -name '{_sh_single(pattern)}' -type f"
        )
    }


def _bash_fallback_grep(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern") or args.get("query") or ""
    root = args.get("path") or args.get("dir") or "."
    if not pattern:
        return _fallback_error_command(
            "grep called without pattern; cannot fall back to bash."
        )
    return {
        "command": f"grep -rn -- '{_sh_single(pattern)}' '{_sh_single(root)}'"
    }


def _bash_fallback_patch(args: Dict[str, Any]) -> Dict[str, Any]:
    patch_file = (
        args.get("patchFile") or args.get("patch_file")
        or args.get("path") or ""
    )
    if not patch_file:
        return _fallback_error_command(
            "apply_patch called without patchFile/path; cannot fall back to bash."
        )
    return {"command": f"patch -p1 < '{_sh_single(patch_file)}'"}


def _bash_fallback_filesystem(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path") or args.get("dir") or "."
    return {"command": f"ls -la '{_sh_single(path)}'"}


TOOL_FALLBACKS: Dict[str, List[tuple]] = {
    "write": [("bash", _bash_fallback_write), ("terminal", _bash_fallback_write)],
    "edit": [("bash", _bash_fallback_edit), ("terminal", _bash_fallback_edit)],
    "apply_patch": [("bash", _bash_fallback_patch), ("terminal", _bash_fallback_patch)],
    "read": [("bash", _bash_fallback_read), ("terminal", _bash_fallback_read)],
    "list": [("bash", _bash_fallback_list), ("terminal", _bash_fallback_list)],
    "glob": [("bash", _bash_fallback_glob), ("terminal", _bash_fallback_glob)],
    "grep": [("bash", _bash_fallback_grep), ("terminal", _bash_fallback_grep)],
    "filesystem": [
        ("bash", _bash_fallback_filesystem),
        ("terminal", _bash_fallback_filesystem),
    ],
    "patch": [("bash", _bash_fallback_patch), ("terminal", _bash_fallback_patch)],
}


# ----------------------------------------------------------------------
# Destructive-call detection
# ----------------------------------------------------------------------
_DESTRUCTIVE_TOOL_TOKENS = (
    "delete", "remove", "unlink", "destroy", "purge", "wipe", "erase",
    "rmdir", "shred", "truncate", "drop",
)

_DESTRUCTIVE_ACTION_TOKENS = (
    "delete", "remove", "reset", "clean", "drop", "prune",
    "destroy", "purge", "wipe", "erase", "truncate",
    "force", "overwrite", "hard", "kill", "terminate",
)

_DESTRUCTIVE_SHELL_PATTERNS = (
    re.compile(r"\brm\b"),
    re.compile(r"\brmdir\b"),
    re.compile(r"\bunlink\b"),
    re.compile(r"\bshred\b"),
    re.compile(r"\btruncate\b"),
    re.compile(r"\bdd\b.*\bof="),
    re.compile(r"\bmkfs\b"),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+clean\s+-[a-z]*f"),
    re.compile(r"\bgit\s+push\s+--force\b"),
    re.compile(r"\bkubectl\s+delete\b"),
    re.compile(r"\baws\s+s3\s+rm\b"),
    re.compile(r"\baws\s+s3api\s+delete"),
    re.compile(r"\baws\s+ec2\s+terminate"),
    re.compile(r"\baws\s+rds\s+delete"),
    re.compile(r"\bdocker\s+rm\b"),
    re.compile(r"\bdocker\s+rmi\b"),
    re.compile(r"\bdocker\s+system\s+prune\b"),
    re.compile(r"\bdocker\s+volume\s+rm\b"),
    re.compile(r"\bfind\b.*\s-delete\b"),
    re.compile(r"curl\b.*\|\s*(?:ba)?sh\b"),
    re.compile(r"wget\b.*\|\s*(?:ba)?sh\b"),
)

_DEMO_OFFER_PATTERNS = (
    re.compile(r"\bwould you like me to\b", re.I),
    re.compile(r"\bwant me to (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bshall i (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bi can (run|show|launch|demo|open|execute)\b", re.I),
    re.compile(r"\bready (for me )?to (run|show|launch|demo)\b", re.I),
)

_USER_DECLINED_PATTERNS = (
    re.compile(r"\bno\b\s*,?\s*(thanks|thank you|thx)\b", re.I),
    re.compile(r"\bdon'?t\b.*\b(run|launch|show)\b", re.I),
    re.compile(r"\bnot?\b.*\b(run|launch|show)\b", re.I),
    re.compile(r"\bjust\b.*\b(explain|tell me|describe)\b", re.I),
)


def _looks_destructive_call(name: str, args: Dict[str, Any]) -> bool:
    name_l = (name or "").lower()

    for tok in _DESTRUCTIVE_TOOL_TOKENS:
        if tok in name_l:
            return True

    action = ""
    if isinstance(args, dict):
        for k in ("action", "operation", "op", "verb"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                action = v.strip().lower()
                break
    if not action:
        for tok in _DESTRUCTIVE_ACTION_TOKENS:
            if tok in name_l:
                action = tok
                break
    if action:
        for tok in _DESTRUCTIVE_ACTION_TOKENS:
            if tok in action:
                return True

    if isinstance(args, dict):
        for key, val in args.items():
            if not isinstance(val, str) or not val:
                continue
            for pat in _DESTRUCTIVE_SHELL_PATTERNS:
                if pat.search(val):
                    return True

    return False


class LoopState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    THINKING = "thinking"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    EVALUATING = "evaluating"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LoopContext:
    iteration: int = 0
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_action_time: float = field(default_factory=time.time)
    tokens_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_iteration_limit: bool = False
    intent: str = "unknown"

    unverified_mutations: List[Dict[str, Any]] = field(default_factory=list)
    verification_nudges: int = 0
    user_declined_demo: bool = False
    user_asked_to_show: bool = False

    def add_observation(self, observation: Dict[str, Any]) -> None:
        self.observations.append(observation)
        self.metadata["last_observation_time"] = time.time()

    def add_action(self, action: Dict[str, Any]) -> None:
        self.actions_taken.append(action)
        self.last_action_time = time.time()
        self.metadata["last_action_time"] = self.last_action_time

    def get_summary(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_actions": len(self.actions_taken),
            "total_observations": len(self.observations),
            "tokens_used": self.tokens_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "cost": self.cost,
            "errors": len(self.errors),
            "duration": time.time() - self.start_time,
            "hit_iteration_limit": self.hit_iteration_limit,
            "intent": self.intent,
            "unverified_mutations": len(self.unverified_mutations),
        }


class AgentLoop:
    def __init__(
        self,
        agent: Any,
        llm: LLMProviderRegistry,
        tool_registry: ToolRegistry,
        planner: Planner,
        config: Dict[str, Any],
    ):
        self.agent = agent
        self.llm = llm
        self.tool_registry = tool_registry
        self.planner = planner
        self.config = config

        self.max_iterations = config.get("max_iterations", 50)
        self.max_tool_calls_per_iteration = config.get("max_tool_calls", 5)
        self.max_history_length = config.get("max_history_length", 40)
        self.enable_planning = config.get("enable_planning", True)
        self.enable_caching = config.get("enable_caching", True)
        self.default_timeout = config.get("timeout", 999999)
        self.project_ctx_ttl = config.get("project_context_ttl", 30.0)
        self.enable_plan_driven = config.get("enable_plan_driven", True)
        self.enable_intent_classification = config.get("enable_intent_classification", True)
        self.enable_qa_verification = config.get("enable_qa_verification", True)
        self.reserve_output_tokens = int(config.get("reserve_output_tokens", 4000))
        self.require_permission_for_destructive = bool(
            config.get("require_permission_for_destructive", True)
        )
        self.require_verification = bool(config.get("require_verification", True))
        self.max_verification_nudges = int(config.get("max_verification_nudges", 1))
        self.require_demo_offer = bool(config.get("require_demo_offer", True))
        self.retry_tool_timeout = bool(config.get("retry_tool_timeout", False))
        self.enable_parallel_reads = bool(config.get("enable_parallel_reads", True))

        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan: Optional[Plan] = None
        self.pending_tool_calls: List[ToolCall] = []
        self.completed_tool_calls: List[ToolCall] = []

        self.performance_history: deque = deque(maxlen=100)
        self.tool_execution_times: Dict[str, List[float]] = {}

        self.tool_result_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_epoch: int = 0

        self.event_handlers: Dict[str, List[Callable]] = {}
        self.should_stop = False
        self.is_paused = False

        self._project_ctx: Optional[Dict[str, Any]] = None
        self._project_ctx_ts = 0.0
        self._system_prompt_cache: Optional[str] = None
        self._system_prompt_fp: Optional[tuple] = None

        self._checklist_rendered = False
        self._last_checklist_signature: Optional[tuple] = None

    @property
    def model_context(self):
        return self.agent.context_manager

    # ------------------------------------------------------------------
    # API CONTRACT: message sanitizer
    # ------------------------------------------------------------------

    def _sanitize_messages_for_api(
        self,
        messages: List[Message],
    ) -> List[Message]:
        if not messages:
            return messages

        out: List[Message] = []
        i = 0
        n = len(messages)

        while i < n:
            m = messages[i]
            role = getattr(m, "role", None)

            if role == "assistant":
                tcs = getattr(m, "tool_calls", None) or []
                if not tcs:
                    out.append(m)
                    i += 1
                    continue

                declared_ids: List[str] = []
                for tc in tcs:
                    if isinstance(tc, dict):
                        cid = tc.get("id")
                    else:
                        cid = getattr(tc, "id", None)
                    if cid:
                        declared_ids.append(cid)

                out.append(m)
                j = i + 1
                found: Dict[str, Message] = {}
                while j < n and getattr(messages[j], "role", None) == "tool":
                    cid = getattr(messages[j], "tool_call_id", None)
                    if cid in declared_ids and cid not in found:
                        found[cid] = messages[j]
                    j += 1

                for cid in declared_ids:
                    if cid in found:
                        out.append(found[cid])
                    else:
                        out.append(
                            Message(
                                role="tool",
                                content=json.dumps({
                                    "success": False,
                                    "error": (
                                        "tool result missing from history; "
                                        "the previous attempt was interrupted"
                                    ),
                                }),
                                tool_call_id=cid,
                            )
                        )

                i = j
                continue

            if role == "tool":
                i += 1
                continue

            out.append(m)
            i += 1

        if out and getattr(out[-1], "role", None) == "system":
            last = out[-1]
            out[-1] = Message(
                role="user",
                content=getattr(last, "content", "") or "Continue.",
            )

        if out and getattr(out[-1], "role", None) == "assistant":
            out.append(
                Message(
                    role="user",
                    content="Continue from here. Answer the user or call a tool.",
                )
            )

        return out

    # ------------------------------------------------------------------
    # REGISTRY CAPABILITY PROBES
    # ------------------------------------------------------------------

    def _is_read_only(self, name: str) -> bool:
        probe = getattr(self.tool_registry, "is_read_only", None)
        if not callable(probe):
            return False
        try:
            return bool(probe(name))
        except Exception:
            logger.debug("is_read_only(%s) probe failed", name, exc_info=True)
            return False

    def _registry_epoch(self) -> int:
        epoch = getattr(self.tool_registry, "mutation_epoch", 0)
        try:
            return int(epoch)
        except (TypeError, ValueError):
            return 0

    def _registry_list_tools(self) -> List[str]:
        probe = getattr(self.tool_registry, "list_tools", None)
        if callable(probe):
            try:
                return list(probe())
            except Exception:
                logger.debug("list_tools probe failed", exc_info=True)
        tools = getattr(self.tool_registry, "tools", None)
        if isinstance(tools, dict):
            return list(tools)
        return []

    def _registry_has_tool(self, name: str) -> bool:
        probe = getattr(self.tool_registry, "has_tool", None)
        if callable(probe):
            try:
                return bool(probe(name))
            except Exception:
                logger.debug("has_tool(%s) probe failed", name, exc_info=True)
        return name in self._registry_list_tools()

    def _registry_get_schemas(self) -> List[Dict[str, Any]]:
        for attr, with_intent in (("select_for_task", True), ("get_schemas", False)):
            probe = getattr(self.tool_registry, attr, None)
            if not callable(probe):
                continue
            try:
                raw = probe(self.context.intent) if with_intent else probe()
                schemas = list(raw)
                if not schemas:
                    logger.warning(
                        "Tool registry returned zero schemas via %s (intent=%s)",
                        attr, self.context.intent,
                    )
                return schemas
            except Exception:
                logger.debug("%s probe failed", attr, exc_info=True)
        logger.error("No schema provider found on tool registry")
        return []

    async def _registry_execute(self, name: str, args: Dict[str, Any]) -> Any:
        agent_exec = getattr(self.agent, "execute_tool", None)
        if callable(agent_exec):
            try:
                result = agent_exec(name, args)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as exc:
                logger.error(
                    "agent.execute_tool(%s) failed: %s", name, exc, exc_info=True
                )
                return {
                    "success": False,
                    "tool": name,
                    "error": str(exc),
                    "recoverable": True,
                }

        probe = getattr(self.tool_registry, "execute_safe", None)
        if not callable(probe):
            probe = getattr(self.tool_registry, "execute", None)
        if not callable(probe):
            return {
                "success": False,
                "tool": name,
                "error": "no tool executor available",
                "recoverable": True,
            }
        try:
            result = probe(name, args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            logger.error("registry execute(%s) failed: %s", name, exc, exc_info=True)
            return {
                "success": False,
                "tool": name,
                "error": str(exc),
                "recoverable": True,
            }

    # ==================================================================
    # PUBLIC ENTRY
    # ==================================================================

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tool_outputs: Optional[List[Any]] = None,
        max_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.context = LoopContext()
        self.current_plan = None
        self.pending_tool_calls = []
        self.completed_tool_calls = []
        self.tool_result_cache.clear()
        self._cache_epoch = self._registry_epoch()
        self.should_stop = False
        self.is_paused = False
        self._project_ctx = None
        self._project_ctx_ts = 0.0
        self._system_prompt_cache = None
        self._system_prompt_fp = None
        self._checklist_rendered = False
        self._last_checklist_signature = None

        if max_turns is not None:
            self.max_iterations = max_turns

        try:
            self.state = LoopState.INITIALIZING

            self.context.user_asked_to_show = any(
                p.search(query) for p in _LAUNCH_INTENT_PATTERNS
            )

            intent = await self._classify_intent(query)
            self.context.intent = intent
            logger.info("Classified intent: %s", intent)

            self.context.user_declined_demo = any(
                p.search(query) for p in _USER_DECLINED_PATTERNS
            )

            if intent in ("question", "ambiguous"):
                logger.info("Skipping planning (intent=%s)", intent)
            elif self.enable_planning and await self._should_plan(query, intent=intent):
                await self._create_plan(query, context, intent=intent)

            if (
                self.enable_plan_driven
                and self.current_plan is not None
                and len(self.current_plan.tasks) >= PLAN_DRIVEN_MIN_TASKS
            ):
                plan_result = await self._run_plan_driven(query, context)
                if plan_result is not None:
                    return plan_result

            return await self._run_free_form(query)

        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            self.context.errors.append(str(exc))
            self.state = LoopState.ERROR
            self._update_metrics()
            return self._result(False, error=str(exc))
        except Exception as exc:
            logger.error("Loop failed: %s", exc, exc_info=True)
            self.context.errors.append(str(exc))
            self.state = LoopState.ERROR
            self._update_metrics()
            return self._result(False, error=str(exc))

    # ==================================================================
    # INTENT CLASSIFICATION
    # ==================================================================

    async def _classify_intent(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return "ambiguous"

        if not self.enable_intent_classification:
            return "unknown"

        lowered = q.lower()
        if any(lowered.startswith(p) for p in _QUESTION_PREFIXES) and len(q.split()) < 40:
            return "question"
        for kw in ("refactor", "clean up", "rename"):
            if kw in lowered:
                return "refactor"
        for kw in ("fix", "bug", "error", "crash", "broken", "failing"):
            if kw in lowered:
                return "bugfix"
        for kw in ("add ", "implement", "create", "build", "support for"):
            if kw in lowered:
                return "feature"
        for kw in ("analyze", "inspect", "review", "audit"):
            if kw in lowered:
                return "analysis"

        try:
            self.context.llm_calls += 1
            response = await self.llm.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Classify the user's request into exactly one "
                            "category: feature, bugfix, refactor, analysis, "
                            "question, or ambiguous. Reply with only the "
                            "category word, lowercase."
                        ),
                    ),
                    Message(role="user", content=query),
                ],
                temperature=0.0,
                max_tokens=32,
            )
            raw = (getattr(response, "content", "") or "").strip().lower()
            word = raw.split()[0].strip(".,:;!?") if raw else ""
            if word in (
                "feature", "bugfix", "refactor", "analysis",
                "question", "ambiguous",
            ):
                return word
        except Exception as exc:
            logger.debug("Intent classification failed: %s", exc)
        return "unknown"

    # ==================================================================
    # FREE-FORM LOOP
    # ==================================================================

    async def _run_free_form(self, query: str) -> Dict[str, Any]:
        while not self.should_stop and self.context.iteration < self.max_iterations:
            while self.is_paused:
                await asyncio.sleep(0.1)

            self.context.iteration += 1
            self.state = LoopState.THINKING

            thought = await self._think()

            if thought.get("tool_calls"):
                self.state = LoopState.ACTING
                results = await self._act(thought["tool_calls"])

                self.state = LoopState.OBSERVING
                await self._observe(results)

                await self._inject_failure_guidance(results)
                self._maybe_invalidate_cache()

                self.state = LoopState.EVALUATING
                if not await self._evaluate(results):
                    break

                fast = await self._maybe_fast_path(thought, results)
                if fast is not None:
                    self.state = LoopState.STOPPED
                    self._update_metrics()
                    return self._result(True, fast)

                continue

            response_text = (thought.get("response") or "").strip()

            if (
                self.require_verification
                and self.context.unverified_mutations
                and self.context.verification_nudges < self.max_verification_nudges
                and self.context.iteration < self.max_iterations
            ):
                if not self._declared_unverified(response_text):
                    self.context.verification_nudges += 1
                    logger.info(
                        "Refusing final answer: %d unverified mutation(s) "
                        "(nudge %d/%d)",
                        len(self.context.unverified_mutations),
                        self.context.verification_nudges,
                        self.max_verification_nudges,
                    )
                    await self._inject_verification_guidance()
                    continue

                self.context.metadata["unverified_declared"] = True

            if not response_text:
                response_text = await self._generate_final_response()

            response_text = await self._ensure_demo_offer(response_text)

            self.state = LoopState.STOPPED
            self._update_metrics()
            return self._result(True, response_text)

        self.context.hit_iteration_limit = self.context.iteration >= self.max_iterations
        self.state = LoopState.STOPPED
        final = await self._generate_final_response()
        final = await self._ensure_demo_offer(final)
        self._update_metrics()
        success = bool(final and final.strip())
        return self._result(success, final,
                            error=None if success else "empty final response")

    # ==================================================================
    # PLAN-DRIVEN LOOP
    # ==================================================================

    async def _run_plan_driven(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        plan = self.current_plan
        if plan is None:
            return None

        logger.info(
            "Plan-driven execution starting: %d task(s) (intent=%s)",
            len(plan.tasks), self.context.intent,
        )

        try:
            system_prompt = await self._get_system_prompt()
        except Exception as exc:
            logger.warning("Could not build system prompt for plan mode: %s", exc)
            return None

        task_attempts: Dict[str, int] = {}
        stalled = False

        while not plan.is_complete():
            if self.should_stop:
                stalled = True
                break

            ready = plan.get_pending_tasks()
            if not ready:
                remaining = [
                    t for t in plan._all_tasks() if t.status == TaskStatus.PENDING
                ]
                if not remaining:
                    break
                self.planner._resolve_deadlock(plan)
                if not plan.get_pending_tasks():
                    for t in remaining:
                        t.status = TaskStatus.SKIPPED
                        t.error = "unresolved dependency"
                    break
                continue

            task = ready[0]
            self.context.iteration += 1

            if self.context.iteration > self.max_iterations:
                stalled = True
                break

            attempt = task_attempts.get(task.id, 0) + 1
            task_attempts[task.id] = attempt

            if attempt > PLAN_TASK_MAX_ATTEMPTS:
                task.status = TaskStatus.FAILED
                task.error = "exceeded max attempts"
                stalled = True
                break

            turn_result = await self._execute_task_turn(
                task=task, plan=plan, system_prompt=system_prompt,
            )

            if turn_result is None:
                stalled = True
                break

            if turn_result.get("success"):
                await self._trigger_event(
                    "on_plan_task_complete",
                    {"task_id": task.id, "summary": turn_result.get("summary", "")},
                )
                self._refresh_plan_metadata()
                continue

            if turn_result.get("retry"):
                task.status = TaskStatus.PENDING
                self._refresh_plan_metadata()
                continue

            task.status = TaskStatus.FAILED
            task.error = turn_result.get("error") or "unknown"
            self._refresh_plan_metadata()
            await self._trigger_event(
                "on_plan_task_failed",
                {"task_id": task.id, "error": task.error},
            )
            stalled = True
            break

        failed = [
            t for t in plan._all_tasks()
            if t.status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
        ]
        all_done = plan.is_complete()

        if all_done or stalled:
            summary = await self._summarize_plan(plan, system_prompt)
            summary = await self._ensure_demo_offer(summary)
            self.state = LoopState.STOPPED
            self._update_metrics()

            if failed:
                lines = [
                    "⚠️ Completed with unresolved items:",
                    *[f"- {t.description}: {t.error or 'failed'}" for t in failed],
                    "",
                    summary,
                ]
                return self._result(
                    False,
                    "\n".join(lines),
                    error=f"{len(failed)} task(s) failed or skipped",
                )

            return self._result(True, summary)

        return None

    async def _execute_task_turn(
        self,
        task: Any,
        plan: Plan,
        system_prompt: str,
    ) -> Optional[Dict[str, Any]]:
        dep_summaries: List[str] = []
        for dep_id in task.dependencies:
            dep = plan.get_task_by_id(dep_id)
            if dep is not None and dep.result is not None:
                snippet = str(dep.result)[:300]
                dep_summaries.append(f"- {dep.description}: {snippet}")

        task_prompt = f"Execute task {task.id}: {task.description}\n\n"
        if dep_summaries:
            task_prompt += "Results from prerequisite tasks:\n"
            task_prompt += "\n".join(dep_summaries) + "\n\n"
        qa = (task.metadata or {}).get("validation")
        if qa:
            task_prompt += f"QA criterion (must pass to complete): {qa}\n\n"
        task_prompt += (
            "Use tools as needed. VERIFY your work before declaring completion: "
            "if you wrote or edited a file, read it back; if you ran a fix, "
            "re-run the failing command and quote the output.\n"
            "When the task is complete and verified, respond with a short "
            "summary prefixed by 'TASK COMPLETE:' that includes the exact "
            "evidence (command output or diff).\n"
            "If the task cannot be completed, respond with 'TASK FAILED: <reason>'.\n"
            "If the task cannot be verified for a concrete reason, respond "
            "with 'UNVERIFIED: <reason>' on its own first line.\n"
            "Do not describe the plan — execute this task."
        )

        messages: List[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task_prompt),
        ]

        local_turns = 0
        max_local_turns = self.max_tool_calls_per_iteration * 2
        local_unverified: List[Dict[str, Any]] = []

        while local_turns < max_local_turns:
            local_turns += 1

            messages = self._sanitize_messages_for_api(messages)

            try:
                self.context.llm_calls += 1
                response = await self.llm.complete_with_tools(
                    messages=messages,
                    tools=self._get_available_tools(),
                    temperature=0.4,
                    max_tokens=self.reserve_output_tokens,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Plan task turn failed: %s", exc)
                return {"success": False, "retry": True}

            self._record_usage(getattr(response, "usage", None))

            content = (getattr(response, "content", "") or "").strip()
            tool_calls = list(getattr(response, "tool_calls", []) or [])[
                : self.max_tool_calls_per_iteration
            ]

            if not tool_calls:
                upper = content.upper()
                if upper.startswith("TASK FAILED"):
                    return {"success": False, "retry": True, "error": content}

                if (
                    self.require_verification
                    and local_unverified
                    and not self._declared_unverified(content)
                    and local_turns < max_local_turns
                ):
                    names = [m["tool"] for m in local_unverified]
                    messages.append(Message(role="assistant", content=content))
                    messages.append(Message(
                        role="user",
                        content=(
                            f"You modified state via {names} but have not "
                            f"verified the result. Read the changed artifact "
                            f"back or rerun the affected command and quote "
                            f"the output. If it cannot be verified, start "
                            f"your reply with 'UNVERIFIED: <reason>'."
                        ),
                    ))
                    continue

                summary = content
                if "TASK COMPLETE" in upper:
                    summary = content.split("TASK COMPLETE", 1)[1].lstrip(": ").strip()
                    if not summary:
                        summary = task.description

                if qa and self.enable_qa_verification:
                    verdict = await self._verify_task_qa(task, summary, qa)
                    if not verdict.get("passed", True):
                        return {
                            "success": False,
                            "retry": True,
                            "error": f"QA failed: {verdict.get('reason', '')}",
                        }

                task.status = TaskStatus.COMPLETED
                task.result = summary or task.description
                task.actual_time = time.time() - task.created_at
                if self._declared_unverified(content):
                    task.metadata["unverified"] = True
                self._update_todo_status(task, "done")
                self._refresh_plan_metadata()
                return {"success": True, "summary": task.result}

            serialized = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, default=str),
                    },
                }
                for tc in tool_calls
            ]
            messages.append(
                Message(role="assistant", content=content, tool_calls=serialized)
            )

            results = await self._act(tool_calls)
            await self._observe(results)
            self._maybe_invalidate_cache()

            for x in results:
                r = x.get("result") or {}
                tool = x.get("tool", "")
                if not r.get("success"):
                    continue
                if self._is_read_only(tool):
                    local_unverified.clear()
                else:
                    local_unverified.append({"tool": tool, "time": time.time()})

            by_id = {r.get("tool_call_id"): r for r in results}
            for tc in tool_calls:
                r = by_id.get(tc.id)
                if r is None:
                    payload = {
                        "success": False,
                        "error": "internal: no result produced for this tool call",
                        "recoverable": False,
                    }
                else:
                    payload = r.get("result") or {}
                    if not isinstance(payload, dict):
                        payload = {"success": True, "result": payload}
                try:
                    payload = redact(payload)
                except Exception:
                    pass
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(payload, default=str),
                        tool_call_id=tc.id,
                    )
                )

        return {
            "success": False,
            "retry": True,
            "error": "task exceeded local tool-call budget",
        }

    async def _verify_task_qa(
        self,
        task: Any,
        summary: str,
        qa: str,
    ) -> Dict[str, Any]:
        try:
            self.context.llm_calls += 1
            response = await self.llm.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "You are a strict verifier. Reply PASS or "
                            "FAIL: <short reason>. Only PASS if the reported "
                            "result contains concrete evidence (command output, "
                            "diff, file content) proving the QA criterion."
                        ),
                    ),
                    Message(
                        role="user",
                        content=(
                            f"Task: {task.description}\n"
                            f"QA criterion: {qa}\n"
                            f"Reported result: {summary}\n\n"
                            "Did the result satisfy the QA criterion?"
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=100,
            )
            self._record_usage(getattr(response, "usage", None))
            raw = (getattr(response, "content", "") or "").strip()
            upper = raw.upper()
            if upper.startswith("PASS"):
                return {"passed": True, "reason": ""}
            if upper.startswith("FAIL"):
                reason = raw.split(":", 1)[1].strip() if ":" in raw else raw
                return {"passed": False, "reason": reason}
            return {"passed": False, "reason": f"unrecognized verdict: {raw[:80]}"}
        except Exception as exc:
            logger.warning("QA verification failed for task %s: %s", task.id, exc)
            return {"passed": False, "reason": f"verifier error: {exc}"}

    async def _summarize_plan(self, plan: Plan, system_prompt: str) -> str:
        completed = [
            t for t in plan._all_tasks() if t.status == TaskStatus.COMPLETED
        ]
        failed = [
            t for t in plan._all_tasks()
            if t.status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
        ]

        bullets = []
        for t in completed:
            snippet = str(t.result or "").strip().splitlines()
            snippet = snippet[0] if snippet else ""
            if len(snippet) > 120:
                snippet = snippet[:117] + "…"
            bullets.append(f"- {t.description}: {snippet}")
        for t in failed:
            bullets.append(f"- {t.description}: {t.error or 'failed'}")

        body = "\n".join(bullets) if bullets else "(no tasks executed)"

        try:
            self.context.llm_calls += 1
            response = await self.llm.complete(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(
                        role="user",
                        content=(
                            "Write a short closing summary: 2–4 sentences "
                            "describing what you actually did, why, and "
                            "anything the user should know. Cite the concrete "
                            "evidence (command output, diff, or file content) "
                            "that proves each major step. End with a one-line "
                            "offer to demonstrate the result if the task "
                            "produced something runnable. Do not repeat the "
                            "step list. Do not claim unverified success.\n\n"
                            f"Plan: {plan.goal}\n\n"
                            f"Task outcomes:\n{body}"
                        ),
                    ),
                ],
                temperature=0.3,
                max_tokens=800,
            )
            summary = (getattr(response, "content", "") or "").strip()
            self._record_usage(getattr(response, "usage", None))
            if summary:
                return summary
        except Exception as exc:
            logger.debug("Plan summary LLM call failed: %s", exc)

        return f"Plan '{plan.goal}' completed.\n\n{body}"

    # ==================================================================
    # RESULT
    # ==================================================================

    def _result(
        self,
        success: bool,
        response: str = "",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "success": success,
            "response": response,
            "iteration": self.context.iteration,
            "tool_calls": len(self.completed_tool_calls),
            "context": self.context.get_summary(),
            "plan": self.current_plan.to_dict() if self.current_plan else None,
        }
        if error:
            out["error"] = error
        return out

    # ==================================================================
    # SYSTEM PROMPT
    # ==================================================================

    async def _system_prompt_fingerprint(self) -> tuple:
        try:
            tools = tuple(sorted(self._registry_list_tools()))
        except Exception:
            tools = ()
        try:
            p = getattr(self.agent, "permission_manager", None)
            perms = (bool(getattr(p, "enabled", True)),
                     bool(getattr(p, "auto_approve", False)))
        except Exception:
            perms = (True, False)
        try:
            from agent.utils.env_manager import EnvManager
            env_gen = EnvManager.get().generation
        except Exception:
            env_gen = 0
        try:
            ws = getattr(self.agent, "workspace", None)
            ws_path = str(getattr(ws, "project_dir", "")) if ws else ""
        except Exception:
            ws_path = ""
        try:
            model = self.llm.get_current_model()
        except Exception:
            model = None
        try:
            mcp = getattr(self.agent, "mcp_client", None)
            mcp_count = len(mcp.list_tools()) if mcp is not None else 0
        except Exception:
            mcp_count = 0
        return (tools, perms, env_gen, ws_path, model, mcp_count, self.context.intent)

    async def _get_system_prompt(self) -> str:
        fp = await self._system_prompt_fingerprint()
        if self._system_prompt_cache is not None and fp == self._system_prompt_fp:
            return self._system_prompt_cache

        project = await self._get_project_context()
        perms = self._get_permissions_context()
        env_block = self._build_environment_block()
        aws_block = self._build_aws_guidance()

        # Concise. Every line earns its place. Roughly 40% smaller than
        # the previous version, same behavioural coverage.
        base_prompt = """
        You are an AI CLI agent (depression.ai). Complete the user's task; do not describe it.

UNDERSTAND THE USER
- Question → answer, do not edit.
- Fix → reproduce, find cause, fix, rerun the failing command.
- Build/Create → plan briefly, implement, test, show.
- Show/Run/Launch → run it, show the real output.
- Analyze → inspect, report; do not modify unless asked.

PLAN, THEN ACT
- For nontrivial tasks, work the loop: understand → plan → call a tool →
  observe the result → decide the next step → repeat until done.
- Inspect only what blocks the next action. After ~6 reads/greps, act.
- A plan with no edits is not progress. Do not re-read, re-plan, or narrate.

TOOLS
- Most specific tool first. Use bash/terminal for short terminating
  commands; if any tool fails twice, switch to bash or terminal.
- On error: read it, fix the argument, retry once. Never repeat the exact
  same failing call. If it still fails, use a different tool.
- Never bypass permissions. If a destructive call is denied, stop and ask.


RUN AND SHOW
If the user says run / launch / start / open / show me / in the browser,
you MUST actually run the artifact and paste the real output in your
reply. Do not describe how to run it — run it.
- CLI / script → bash, paste stdout + exit code
- Web UI / dev server → use the process tool to start it, wait for the
  URL line, then use the browser tool to open that URL and capture the
  page title or a short text excerpt as evidence
- API → curl, paste status + body
- File → show the path and the first lines

VERIFY BEFORE CLAIMING
- After any change to state, read it back or rerun the affected command.
- Quote the actual output as evidence.
- Never say "fixed" / "done" / "success" without evidence in this session.
- If truly unverifiable, start your reply with: UNVERIFIED: <reason>

SHOW THE USER
- If the task produced something runnable, run it and show the output.
- End work tasks with one line: "Want me to run it and show you?"
- Skip the offer if the user already asked, already declined, or the
  result is a pure answer.

DON'T STOP MID-TASK
- Finish the whole change, verify it, then show it.
- Only stop for: destructive approval, real ambiguity, or a concrete
  blocker only the user can resolve.

FINAL RESPONSE
1. WHAT changed  2. WHY  3. EVIDENCE  4. Anything unverified  5. Offer
Short. No narration. No tool-by-tool log.


"""

        self._system_prompt_cache = (
            base_prompt
            + env_block
            + aws_block
            + "\n\n## Project\n"
            + json.dumps(project, indent=2, default=str)
            + "\n\n## Permissions\n"
            + json.dumps(perms, indent=2, default=str)
        ).strip()
        self._system_prompt_fp = fp
        return self._system_prompt_cache

    def _build_environment_block(self) -> str:
        import shutil as _shutil
        package_managers = []
        for pm in ("npm", "yarn", "pnpm", "bun", "pip", "uv", "cargo", "go"):
            if _shutil.which(pm):
                package_managers.append(pm)

        if not package_managers and not _shutil.which("git"):
            return ""

        lines = ["\n## Environment"]
        if package_managers:
            lines.append(
                "Package managers on PATH: " + ", ".join(package_managers) + "."
            )
        if _shutil.which("git"):
            lines.append("Git is available.")
        return "\n".join(lines)

    def _build_aws_guidance(self) -> str:
        try:
            mcp_client = getattr(self.agent, "mcp_client", None)
            mcp_aws_tools: List[str] = []
            if mcp_client is not None:
                try:
                    for t in mcp_client.list_tools():
                        name = t.get("function", {}).get("name", "")
                        if name.startswith("mcp__aws__") and self._registry_has_tool(name):
                            mcp_aws_tools.append(name)
                except Exception:
                    mcp_aws_tools = []

            from agent.mcp.aws_config import build_aws_cli_fallback_status
            fallback = build_aws_cli_fallback_status()

            if not mcp_aws_tools and not fallback.get("usable"):
                return ""

            lines = ["\n## AWS"]
            if mcp_aws_tools:
                lines.append(
                    "AWS MCP tools available: "
                    + ", ".join(sorted(mcp_aws_tools))
                    + ". Use the `aws` tool for any AWS operation."
                )
            else:
                lines.append(
                    "AWS CLI + credentials available. Use the `aws` tool, "
                    "or `aws <service> <operation>` via `bash`. Credentials "
                    "are injected automatically — never print them."
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("AWS guidance build failed: %s", exc)
            return ""

    async def _get_project_context(self) -> Dict[str, Any]:
        now = time.time()
        if self._project_ctx is not None and now - self._project_ctx_ts < self.project_ctx_ttl:
            return self._project_ctx

        workspace = getattr(self.agent, "workspace", None)
        if workspace is not None:
            try:
                files = await workspace.list_files(max_files=10)
            except Exception as exc:
                logger.debug("list_files failed: %s", exc)
                files = []
            ctx = {
                "project_path": str(getattr(workspace, "project_dir", "")),
                "files": files,
            }
        else:
            ctx = {"project_path": "", "files": []}

        self._project_ctx = ctx
        self._project_ctx_ts = now
        return ctx

    def _get_permissions_context(self) -> Dict[str, Any]:
        p = getattr(self.agent, "permission_manager", None)
        if not p:
            return {"enabled": True, "auto_approve": False}
        return {
            "enabled": getattr(p, "enabled", True),
            "auto_approve": getattr(p, "auto_approve", False),
        }

    # ==================================================================
    # PLANNING
    # ==================================================================

    async def _should_plan(self, q: str, intent: str = "unknown") -> bool:
        if intent in ("question", "ambiguous"):
            return False
        if intent == "analysis" and len(q.split()) < 10:
            return False

        nontrivial = len(q.split()) > 12 or any(
            x in q.lower() for x in ("plan", "steps", "multiple", "several")
        )

        if intent in ("feature", "bugfix", "refactor", "analysis"):
            return nontrivial

        return len(q.split()) > 20 or any(
            x in q.lower() for x in ("plan", "steps", "multiple", "several")
        )

    async def _create_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        intent: str = "unknown",
    ) -> None:
        planner = getattr(self, "planner", None)
        if planner is None:
            return
        try:
            self.state = LoopState.PLANNING
            plan_context = dict(context or {})
            plan_context["intent"] = intent
            self.current_plan = await planner.create_plan(
                goal=query,
                context=plan_context,
                constraints=self._get_plan_constraints(intent=intent),
            )
            self._mirror_plan_to_todos(self.current_plan)
            self._checklist_rendered = True

            await self._trigger_event(
                "on_plan_created",
                {
                    "plan_id": self.current_plan.id,
                    "goal": self.current_plan.goal,
                    "tasks": [
                        {"id": t.id, "description": t.description}
                        for t in self.current_plan.tasks
                    ],
                    "count": len(self.current_plan.tasks),
                },
            )
            # Also fire the unified render event so the sidebar Todo
            # panel picks it up.
            await self._trigger_event(
                "on_plan_rendered",
                {
                    "entries": [
                        {
                            "content": t.description,
                            "status": "in_progress" if i == 0 else "pending",
                        }
                        for i, t in enumerate(self.current_plan.tasks)
                    ],
                    "source": "planner",
                },
            )
        except Exception as exc:
            logger.warning("Planning failed: %s", exc)
            self.current_plan = None

    def _refresh_plan_metadata(self) -> None:
        if self.current_plan is None:
            self.context.metadata.pop("plan", None)
            return
        try:
            self.context.metadata["plan"] = self.current_plan.to_dict()
        except Exception:
            pass

    def _mirror_plan_to_todos(self, plan: Plan) -> None:
        try:
            from agent.tools.todo import TodoTool, TodoItem
            import uuid as _uuid
        except Exception:
            return

        session = getattr(self.agent, "session", None)
        sid = TodoTool._session_key(session)
        store = TodoTool._strong_keys.setdefault(sid, {})
        store.clear()

        for idx, task in enumerate(plan.tasks):
            status = "in_progress" if idx == 0 else "pending"
            tid = task.id or str(_uuid.uuid4())[:8]
            item = TodoItem(
                id=tid,
                title=task.description,
                status=status,
                priority=getattr(task.priority, "value", 3),
            )
            store[tid] = item
            task.metadata["todo_id"] = tid

    def _update_todo_status(self, task: Any, status: str) -> None:
        try:
            from agent.tools.todo import TodoTool
        except Exception:
            return
        todo_id = (task.metadata or {}).get("todo_id")
        if not todo_id:
            return
        session = getattr(self.agent, "session", None)
        sid = TodoTool._session_key(session)
        store = TodoTool._strong_keys.get(sid) or {}
        item = store.get(todo_id)
        if item is None:
            return
        item.status = status

    def _get_plan_constraints(self, intent: str = "unknown") -> Dict[str, Any]:
        return {
            "max_tasks": 20,
            "timeout": self.default_timeout,
            "available_tools": self._registry_list_tools(),
            "intent": intent,
        }

    # ==================================================================
    # CONTEXT LIMIT / COMPACTION
    # ==================================================================

    def _context_limit(self) -> int:
        try:
            fn = getattr(self.llm, "get_context_limit", None)
            if callable(fn):
                return int(fn())
        except Exception:
            pass
        try:
            from agent.llm.runtime import MODEL_METADATA as _MM, DEFAULT_CONTEXT_WINDOW as _DCW
            model = self.llm.get_current_model()
            meta = _MM.get(model, {}) if model else {}
            w = int(meta.get("context_window") or 0)
            return w if w >= 2048 else _DCW
        except Exception:
            try:
                from agent.llm.runtime import DEFAULT_CONTEXT_WINDOW as _DCW
                return _DCW
            except Exception:
                return 8000

    async def _maybe_compact_before_call(
        self,
        system_prompt: str,
        tool_schemas: List[Dict[str, Any]],
        state_block: str,
    ) -> None:
        try:
            mc = self.model_context
            if not hasattr(mc, "needs_compaction"):
                return
            if await mc.needs_compaction(
                system_prompt=system_prompt,
                tool_schemas=tool_schemas,
                state_block=state_block,
            ):
                await mc.compact(
                    system_prompt=system_prompt,
                    tool_schemas=tool_schemas,
                    state_block=state_block,
                )
        except Exception as exc:
            logger.debug("Pre-call compaction failed: %s", exc)

    # ==================================================================
    # THINK
    # ==================================================================

    async def _think(self) -> Dict[str, Any]:
        await self._ensure_system_prompt()

        system_prompt = await self._get_system_prompt()
        tool_schemas = self._get_available_tools()

        state_parts: Dict[str, Any] = {"state": await self._get_state_context()}
        if self.current_plan and self.current_plan.get_pending_tasks():
            state_parts["plan"] = self._get_plan_context()
        state_block = json.dumps(state_parts, default=str)

        await self._maybe_compact_before_call(system_prompt, tool_schemas, state_block)

        messages = get_model_messages(self.model_context, self.max_history_length)
        messages = self._sanitize_messages_for_api(messages)
        messages.append(Message(role="system", content="Execution state: " + state_block))
        messages = self._sanitize_messages_for_api(messages)

        self.context.llm_calls += 1
        response = await self.llm.complete_with_tools(
            messages=messages,
            tools=tool_schemas,
            temperature=0.2,
            max_tokens=self.reserve_output_tokens,
        )

        calls = list(getattr(response, "tool_calls", []) or [])
        content = getattr(response, "content", None) or ""
        finish_reason = getattr(response, "finish_reason", None) or FinishReason.STOP.value

        empty_arg_calls = any(
            not (getattr(tc, "arguments", None) or {}) for tc in calls
        )
        should_retry = (
            (not calls and not content.strip())
            or empty_arg_calls
            or (
                not calls
                and finish_reason not in (
                    FinishReason.STOP.value,
                    FinishReason.TOOL_CALLS.value,
                )
            )
        )

        if should_retry and self.context.iteration < self.max_iterations:
            logger.warning(
                "Incomplete turn (finish_reason=%s, calls=%d, empty_args=%s); retrying once",
                finish_reason, len(calls), empty_arg_calls,
            )
            retry_messages = list(messages)
            retry_messages.append(
                Message(
                    role="user",
                    content=(
                        "Your previous response was incomplete. Either provide "
                        "a full answer now, or re-emit the tool call with the "
                        "complete JSON object matching the schema — every "
                        "required field must be present."
                    ),
                )
            )
            retry_messages = self._sanitize_messages_for_api(retry_messages)

            self.context.llm_calls += 1
            response = await self.llm.complete_with_tools(
                messages=retry_messages,
                tools=tool_schemas,
                temperature=0.1,
                max_tokens=self.reserve_output_tokens * 2,
            )
            calls = list(getattr(response, "tool_calls", []) or [])
            content = getattr(response, "content", None) or ""
            finish_reason = getattr(response, "finish_reason", None) or FinishReason.STOP.value

        if len(calls) > self.max_tool_calls_per_iteration:
            logger.info(
                "Capping tool calls from %d to %d",
                len(calls), self.max_tool_calls_per_iteration,
            )
            calls = calls[: self.max_tool_calls_per_iteration]

        self._record_usage(getattr(response, "usage", None))

        # Always try to extract a checklist from the model's text and
        # fire the render event. This is how the todo panel appears for
        # short tasks where no structured plan was created.
        if content and not calls:
            self._maybe_emit_checklist(content)

        if calls:
            await add_tool_call(self.model_context, content, calls)
        elif content.strip():
            await self.model_context.add_assistant_message(content)

        self.context.tool_calls.extend(calls)
        return {"response": content, "tool_calls": calls}

    def _maybe_emit_checklist(self, content: str) -> None:
        """Parse a checklist out of the model's text and fire the render event."""
        if self._checklist_rendered:
            return
        try:
            from agent.tui.plan_parser import parse_plan_from_text
            parsed = parse_plan_from_text(content)
            if not (parsed and parsed.is_valid):
                return
            signature = tuple(
                (e.content[:80], e.status) for e in parsed.entries
            )
            if signature == self._last_checklist_signature:
                return
            self._last_checklist_signature = signature
            self._checklist_rendered = True
            entries = parsed.to_list()
            # Fire both events. `on_plan_updated` is the legacy path;
            # `on_plan_rendered` is the unified path the sidebar uses.
            asyncio.create_task(self._trigger_event(
                "on_plan_updated",
                {
                    "entries": entries,
                    "count": parsed.count,
                    "raw": content,
                    "render": True,
                },
            ))
            asyncio.create_task(self._trigger_event(
                "on_plan_rendered",
                {"entries": entries, "source": "reply"},
            ))
        except Exception as exc:
            logger.debug("Checklist parse failed: %s", exc)

    async def _ensure_system_prompt(self) -> None:
        messages = getattr(self.model_context, "messages", None) or []
        has_real_prompt = any(
            getattr(m, "role", None) == "system"
            and (getattr(m, "metadata", {}) or {}).get(SYSTEM_PROMPT_MARKER)
            for m in messages
        )
        if not has_real_prompt:
            await self.model_context.add_message(
                role="system",
                content=await self._get_system_prompt(),
                pinned=True,
                metadata={SYSTEM_PROMPT_MARKER: True},
            )

    # ==================================================================
    # USAGE / COST
    # ==================================================================

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, int]:
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if isinstance(usage, dict):
            d = usage
        else:
            d = getattr(usage, "__dict__", None) or {}
            if not d:
                d = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
        prompt = int(d.get("prompt_tokens", 0) or 0)
        completion = int(d.get("completion_tokens", 0) or 0)
        total = int(d.get("total_tokens", 0) or 0) or (prompt + completion)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    def _record_usage(self, usage: Any) -> None:
        u = self._normalize_usage(usage)
        self.context.tokens_used += u["total_tokens"]
        self.context.input_tokens += u["prompt_tokens"]
        self.context.output_tokens += u["completion_tokens"]
        self.context.cost += self._compute_cost(u)

    def _compute_cost(self, usage: Dict[str, int]) -> float:
        try:
            model = self.llm.get_current_model()
        except Exception:
            model = None
        meta = MODEL_METADATA.get(model, {}) if model else {}
        input_rate = meta.get("cost_input", 0.0) or 0.0
        output_rate = meta.get("cost_output", 0.0) or 0.0
        if not input_rate and not output_rate:
            return 0.0
        return (
            (usage.get("prompt_tokens", 0) / 1_000_000) * input_rate
            + (usage.get("completion_tokens", 0) / 1_000_000) * output_rate
        )

    # ==================================================================
    # TOOL EXPOSURE
    # ==================================================================

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        return self._registry_get_schemas()

    async def _get_state_context(self) -> Dict[str, Any]:
        return {
            "iteration": self.context.iteration,
            "intent": self.context.intent,
            "unverified": len(self.context.unverified_mutations),
        }

    def _get_plan_context(self) -> Dict[str, Any]:
        if not self.current_plan:
            return {}
        return {
            "plan_id": self.current_plan.id,
            "goal": self.current_plan.goal,
            "completion": self.current_plan.get_completion_percentage(),
            "pending_tasks": [t.to_dict() for t in self.current_plan.get_pending_tasks()],
        }

    # ==================================================================
    # FAST PATH
    # ==================================================================

    async def _maybe_fast_path(
        self,
        thought: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> Optional[str]:
        if self.context.iteration != 1:
            return None
        calls = thought.get("tool_calls") or []
        if len(calls) != 1 or len(results) != 1:
            return None
        if thought.get("response"):
            return None

        tool_name = getattr(calls[0], "name", "")
        if tool_name in ("question",):
            return None

        if not self._is_read_only(tool_name):
            return None

        if self.context.intent in ("feature", "bugfix", "refactor", "analysis"):
            return None

        outcome = results[0].get("result") or {}
        if not isinstance(outcome, dict) or not outcome.get("success", False):
            return None

        out = (
            outcome.get("content")
            or outcome.get("output")
            or outcome.get("result")
            or outcome.get("text")
            or ""
        )
        if not out and outcome.get("path"):
            if outcome.get("bytes_written") is not None:
                out = f"Wrote {outcome['bytes_written']} bytes to {outcome['path']}"
            else:
                out = f"Result written to {outcome['path']}"

        if isinstance(out, str) and out.strip():
            return out.strip()
        return None

    # ==================================================================
    # ACT
    # ==================================================================

    async def _act(self, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        parallel_batch: List[ToolCall] = []
        sequential_batch: List[ToolCall] = []

        for call in calls:
            name = getattr(call, "name", "")
            if (
                self.enable_parallel_reads
                and self._is_read_only(name)
                and not _looks_destructive_call(name, getattr(call, "arguments", {}) or {})
            ):
                parallel_batch.append(call)
            else:
                sequential_batch.append(call)

        if parallel_batch:
            parallel_results = await asyncio.gather(
                *[self._act_one(call) for call in parallel_batch],
                return_exceptions=False,
            )
            results.extend(parallel_results)

        for call in sequential_batch:
            r = await self._act_one(call)
            results.append(r)

        return results

    async def _act_one(self, call: ToolCall) -> Dict[str, Any]:
        call_id = getattr(call, "id", None) or f"call_{int(time.time()*1e6)}"
        name = getattr(call, "name", "?")
        args = getattr(call, "arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}

        if self._is_duplicate_failing_call(name, args):
            logger.info("Refusing duplicate failing call: %s", name)
            result = {
                "success": False,
                "error": (
                    f"Refusing to repeat the same failing call to '{name}' "
                    f"with identical arguments. Change the arguments or "
                    f"take a different action."
                ),
                "recoverable": True,
                "invalid_arguments": True,
            }
            await self._safe_add_tool_result(call, result)
            return {
                "tool": name, "tool_call_id": call_id,
                "result": result, "success": False, "params": args,
            }

        if (
            self.require_permission_for_destructive
            and _looks_destructive_call(name, args)
        ):
            pm = getattr(self.agent, "permission_manager", None)
            if pm is None:
                logger.warning(
                    "Refusing destructive call to %s: no permission manager",
                    name,
                )
                result = {
                    "success": False,
                    "error": (
                        f"Refused destructive call to '{name}': no permission "
                        f"manager is configured on this agent. Deletion and "
                        f"removal require an explicit permission check."
                    ),
                    "recoverable": False,
                    "permission_denied": True,
                }
                await self._safe_add_tool_result(call, result)
                return {
                    "tool": name, "tool_call_id": call_id,
                    "result": result, "success": False, "params": args,
                }

        try:
            key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            cache_entry = None
            if self.enable_caching and self._is_read_only(name):
                cache_entry = self.tool_result_cache.get(key)

            if (
                cache_entry
                and time.time() - cache_entry["timestamp"] < 60
                and cache_entry["result"].get("success", False)
            ):
                result = cache_entry["result"]
                await self._safe_add_tool_result(call, result)
                self.completed_tool_calls.append(call)
                self.context.add_action({
                    "tool": name, "tool_call_id": call_id,
                    "params": args, "result": result,
                    "time": 0.0, "cached": True,
                })
                if result.get("success", False) and self._is_read_only(name):
                    self._clear_unverified_mutations(name)
                await self._trigger_event("on_tool_executed", {
                    "tool": name, "tool_call_id": call_id,
                    "params": args, "result": result, "cached": True,
                })
                return {
                    "tool": name, "tool_call_id": call_id,
                    "result": result, "cached": True, "params": args,
                }

            start = time.time()
            result = await self._registry_execute(name, args)
            elapsed = time.time() - start

            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            try:
                result = redact(result)
            except Exception:
                pass

            if result.get("permission_denied"):
                logger.info(
                    "Tool %s denied by permission manager: %s",
                    name, result.get("error"),
                )

            if (
                not result.get("success", False)
                and not result.get("permission_denied", False)
                and name in TOOL_FALLBACKS
            ):
                fb = await self._try_fallback(name, args, result)
                if fb is not None and fb.get("success", False):
                    logger.info("Fallback succeeded for %s", name)
                    result = fb
                    result["fallback_from"] = name

            self.tool_execution_times.setdefault(name, []).append(elapsed)
            self.context.add_action({
                "tool": name, "tool_call_id": call_id,
                "params": args, "result": result, "time": elapsed,
            })
            await self._safe_add_tool_result(call, result)

            if result.get("success", False):
                self._clear_failure_signature(name, args)
                self.context.metadata.get("tool_failures", {}).pop(name, None)
                if self.enable_caching and self._is_read_only(name):
                    self.tool_result_cache[key] = {
                        "result": result, "timestamp": time.time(),
                    }
                self._track_mutation_or_read(name, args, call_id, result)

            self.completed_tool_calls.append(call)
            await self._trigger_event("on_tool_executed", {
                "tool": name, "tool_call_id": call_id,
                "params": args, "result": result,
                "execution_time": elapsed,
            })
            return {
                "tool": name, "tool_call_id": call_id,
                "result": result, "execution_time": elapsed, "params": args,
            }

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc, exc_info=True)
            result = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "recoverable": True,
                "suggestion": "Inspect the error and retry with corrected arguments.",
            }
            await self._safe_add_tool_result(call, result)
            await self._trigger_event("on_tool_executed", {
                "tool": name, "tool_call_id": call_id,
                "params": args, "result": result, "error": str(exc),
            })
            return {
                "tool": name, "tool_call_id": call_id,
                "result": result, "error": str(exc), "success": False, "params": args,
            }

    async def _try_fallback(
        self,
        name: str,
        args: Dict[str, Any],
        failed: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        fallbacks = TOOL_FALLBACKS.get(name, [])
        if not fallbacks:
            return None

        for fb_tool, transformer in fallbacks:
            if not self._registry_has_tool(fb_tool):
                continue
            try:
                fb_args = transformer(args)
            except Exception as exc:
                logger.debug("Fallback transformer for %s failed: %s", name, exc)
                continue
            if not fb_args or "command" not in fb_args:
                continue
            try:
                fb_result = await self._registry_execute(fb_tool, fb_args)
            except Exception as exc:
                logger.debug("Fallback %s failed: %s", fb_tool, exc)
                continue
            if not isinstance(fb_result, dict):
                fb_result = {"success": True, "result": fb_result}
            if fb_result.get("success", False):
                return fb_result
        return None

    # ------------------------------------------------------------------
    # Verification gate
    # ------------------------------------------------------------------

    def _track_mutation_or_read(
        self,
        name: str,
        args: Dict[str, Any],
        call_id: str,
        result: Dict[str, Any],
    ) -> None:
        if not self.require_verification:
            return

        if self._is_read_only(name):
            self._clear_unverified_mutations(name)
            return

        self.context.unverified_mutations.append({
            "tool": name,
            "args": args,
            "call_id": call_id,
            "time": time.time(),
        })
        logger.debug(
            "Recorded unverified mutation: %s (%d pending)",
            name, len(self.context.unverified_mutations),
        )

    def _clear_unverified_mutations(self, by_tool: str) -> None:
        if not self.context.unverified_mutations:
            return
        self.context.unverified_mutations.clear()

    async def _inject_verification_guidance(self) -> None:
        pending = self.context.unverified_mutations
        if not pending:
            return
        names = sorted({m["tool"] for m in pending})
        count = len(pending)
        msg = (
            f"You have {count} unverified change(s) to state via: "
            f"{', '.join(names)}.\n"
            f"Before answering, verify each change: read the changed "
            f"section back, or rerun the affected command, and quote the "
            f"actual output as evidence.\n"
            f"If verification is genuinely impossible, begin your reply "
            f"with:\n    UNVERIFIED: <concrete reason>"
        )
        try:
            await self.model_context.add_system_message(msg)
        except Exception as exc:
            logger.debug("add_system_message (verification) failed: %s", exc)

    def _declared_unverified(self, text: str) -> bool:
        if not text:
            return False
        m = _UNVERIFIED_MARKER.search(text)
        if not m:
            return False
        return m.start() < _UNVERIFIED_MAX_POS

    # ------------------------------------------------------------------
    # Failure signature bookkeeping
    # ------------------------------------------------------------------

    def _call_signature(self, name: str, args: Dict[str, Any]) -> Optional[str]:
        try:
            return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        except Exception:
            return None

    def _is_duplicate_failing_call(self, name: str, args: Dict[str, Any]) -> bool:
        sig = self._call_signature(name, args)
        if sig is None:
            return False
        seen = self.context.metadata.setdefault("failing_signatures", {})
        return seen.get(sig, 0) >= SAME_ARGS_FAILURE_CAP

    def _record_failure_signature(self, name: str, args: Dict[str, Any]) -> None:
        sig = self._call_signature(name, args)
        if sig is None:
            return
        seen = self.context.metadata.setdefault("failing_signatures", {})
        seen[sig] = seen.get(sig, 0) + 1

    def _clear_failure_signature(self, name: str, args: Dict[str, Any]) -> None:
        sig = self._call_signature(name, args)
        if sig is None:
            return
        seen = self.context.metadata.get("failing_signatures")
        if seen and sig in seen:
            del seen[sig]

    def _maybe_invalidate_cache(self) -> None:
        current = self._registry_epoch()
        if current != self._cache_epoch:
            if self.tool_result_cache:
                logger.debug(
                    "Invalidating tool result cache (mutation epoch %d -> %d)",
                    self._cache_epoch, current,
                )
            self.tool_result_cache.clear()
            self._cache_epoch = current

    async def _safe_add_tool_result(
        self, call: ToolCall, result: Dict[str, Any]
    ) -> None:
        try:
            context_window = self._context_limit()
            await add_tool_result(
                self.model_context,
                call,
                result,
                context_window=context_window,
            )
        except Exception as inner:
            logger.error(
                "add_tool_result failed for %s: %s",
                getattr(call, "id", "?"), inner, exc_info=True,
            )

    async def _observe(self, results: List[Dict[str, Any]]) -> None:
        for x in results:
            r = x.get("result") or {}
            self.context.add_observation({
                "timestamp": time.time(),
                "tool": x.get("tool"),
                "tool_call_id": x.get("tool_call_id"),
                "success": bool(r.get("success", False)),
                "error": x.get("error") or r.get("error"),
                "execution_time": x.get("execution_time", 0),
                "result": r,
            })

    async def _inject_failure_guidance(self, results: List[Dict[str, Any]]) -> None:
        for x in results:
            r = x.get("result") or {}
            if r.get("success", False):
                continue

            tool = x.get("tool", "?")
            args = x.get("params") or {}
            self._record_failure_signature(tool, args)

            err = r.get("error") or x.get("error") or "unknown error"
            recoverable = bool(r.get("recoverable", True))
            suggestion = r.get("suggestion") or (
                "Re-read the tool schema and retry with corrected arguments."
            )
            invalid = r.get("invalid_arguments", False)
            permission_denied = bool(r.get("permission_denied"))

            logger.warning(
                "TOOL FAILURE: %s | args=%s | err=%s | recoverable=%s",
                tool,
                json.dumps(args, default=str)[:300],
                str(err)[:300],
                recoverable,
            )

            guidance = "Do NOT retry with the same arguments."
            if permission_denied:
                guidance = (
                    "This call was DENIED by the permission system. Do not "
                    "try to work around it (no alternate tools, no shell "
                    "tricks). Explain to the user what was blocked and ask "
                    "how they would like to proceed."
                )
            elif not recoverable:
                guidance = (
                    "This failure is NOT recoverable. Do not retry this tool "
                    "with these arguments. Choose a different approach or stop "
                    "and explain the blocker to the user."
                )
            else:
                guidance += (
                    "  Try a different tool — bash or terminal usually work "
                    "as a fallback for file, search, and edit operations."
                )

            msg = (
                f"Tool '{tool}' failed.\n"
                f"Error: {err}\n"
                f"Arguments you sent: {json.dumps(args, default=str)}\n"
                + ("This was an argument-validation failure. "
                   if invalid else "")
                + ("This was a PERMISSION DENIAL. "
                   if permission_denied else "")
                + f"Recoverable: {recoverable}\n"
                + f"Suggestion: {suggestion}\n"
                + guidance
            )
            try:
                await self.model_context.add_system_message(msg)
            except Exception as exc:
                logger.debug("add_system_message (failure guidance) failed: %s", exc)

    async def _evaluate(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True

        per_tool = self.context.metadata.setdefault("tool_failures", {})
        total = int(self.context.metadata.get("total_tool_failures", 0) or 0)
        for x in results:
            r = x.get("result") or {}
            tool = x.get("tool", "?")
            if not r.get("success", False):
                per_tool[tool] = per_tool.get(tool, 0) + 1
                total += 1
        self.context.metadata["total_tool_failures"] = total

        if total >= TOTAL_FAILURE_CAP:
            logger.warning(
                "Stopping loop: %d total tool failures across the session", total
            )
            return False

        for tool, count in per_tool.items():
            if count >= PER_TOOL_FAILURE_CAP:
                marker = f"stop_directive_sent:{tool}"
                if not self.context.metadata.get(marker):
                    self.context.metadata[marker] = True
                    try:
                        await self.model_context.add_system_message(
                            f"You have failed to use '{tool}' {count} times. "
                            f"Stop calling it. State the concrete blocker "
                            f"to the user and complete your reply without "
                            f"another call to '{tool}'."
                        )
                    except Exception:
                        pass
                    logger.warning(
                        "Stop directive issued for tool '%s' (%d failures)",
                        tool, count,
                    )
                if count >= PER_TOOL_FAILURE_CAP + 2:
                    logger.warning(
                        "Stopping loop: tool '%s' failed %d times", tool, count
                    )
                    return False

        signature = tuple(
            sorted(
                (
                    r.get("tool"),
                    str((r.get("result") or {}).get("error",
                                                    r.get("error", "")))[:120],
                )
                for r in results
                if not (r.get("result") or {}).get("success", False)
            )
        )
        if not signature:
            self.context.metadata.pop("last_failure_signature", None)
            return True

        if self.context.metadata.get("last_failure_signature") == signature:
            self.context.metadata["repeat_failures"] = (
                self.context.metadata.get("repeat_failures", 1) + 1
            )
        else:
            self.context.metadata["repeat_failures"] = 1
        self.context.metadata["last_failure_signature"] = signature

        if self.context.metadata["repeat_failures"] >= 3:
            logger.warning("Stopping loop: same failure signature repeated 3 times")
            return False
        return True

    # ==================================================================
    # DEMO OFFER NUDGE
    # ==================================================================

    def _response_offers_demo(self, text: str) -> bool:
        if not text or not text.strip():
            return False
        # If the user asked to see the work, no offer is required — the
        # run already happened.
        if self.context.user_asked_to_show:
            return True
        if not self.completed_tool_calls:
            return True
        if self.context.intent == "question":
            return True
        if self.context.user_declined_demo:
            return True
        for pat in _DEMO_OFFER_PATTERNS:
            if pat.search(text):
                return True
        return False

    async def _ensure_demo_offer(self, text: str) -> str:
        if not self.require_demo_offer:
            return text
        if not text or not text.strip():
            return text
        if self._response_offers_demo(text):
            return text

        target = None
        for action in reversed(self.context.actions_taken):
            r = action.get("result") or {}
            if r.get("success") and not self._is_read_only(action.get("tool", "")):
                target = action.get("tool")
                break

        if target:
            offer = (
                "\n\nI've finished the changes. Want me to run it and show "
                "you the output?"
            )
        else:
            offer = (
                "\n\nWant me to show you the result — the diff, file contents, "
                "or command output?"
            )
        return text.rstrip() + offer

    # ==================================================================
    # FINAL RESPONSE
    # ==================================================================

    async def _generate_final_response(self) -> str:
        messages = get_model_messages(self.model_context, self.max_history_length)
        messages = self._sanitize_messages_for_api(messages)
        if not any(getattr(m, "role", None) == "system" for m in messages):
            messages.insert(0, Message(
                role="system",
                content=await self._get_system_prompt(),
            ))

        if self.context.unverified_mutations:
            pending_names = sorted(
                {m["tool"] for m in self.context.unverified_mutations}
            )
            extra = (
                f" WARNING: {len(self.context.unverified_mutations)} change(s) "
                f"via {', '.join(pending_names)} are still UNVERIFIED. Either "
                f"state the concrete evidence you already have, or begin the "
                f"response with 'UNVERIFIED: <reason>'. Do not claim success "
                f"without evidence."
            )
        else:
            extra = ""

        # If the user asked to see the work, demand that the reply contain
        # the actual output, not a description.
        if self.context.user_asked_to_show:
            extra += (
                " The user asked to see the result. Your reply MUST contain "
                "the real output: the command's stdout/stderr, the URL and "
                "startup line for a server, the browser page title, or the "
                "file contents. Do not describe how to run it — show that "
                "you already ran it."
            )

        messages.append(Message(
            role="user",
            content=(
                "Provide the final response to the user. Structure it as: "
                "1) WHAT changed, 2) WHY, 3) EVIDENCE (concrete command "
                "output, diff, or file content — only what proves the "
                "result), 4) anything still unverified or blocked, 5) a "
                "one-line offer to run/show the result when applicable. "
                "Do not claim unverified success." + extra
            ),
        ))
        messages = self._sanitize_messages_for_api(messages)

        try:
            self.context.llm_calls += 1
            response = await self.llm.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=self.reserve_output_tokens,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Final response generation failed: %s", exc)
            return ""

        content = getattr(response, "content", "") or ""
        self._record_usage(getattr(response, "usage", None))
        return content

    # ==================================================================
    # METRICS / LIFECYCLE
    # ==================================================================

    def _update_metrics(self) -> None:
        self.performance_history.append({
            "timestamp": time.time(),
            "duration": time.time() - self.context.start_time,
            "iterations": self.context.iteration,
            "tool_calls": len(self.completed_tool_calls),
            "tokens": self.context.tokens_used,
            "llm_calls": self.context.llm_calls,
            "errors": len(self.context.errors),
            "intent": self.context.intent,
            "unverified_mutations": len(self.context.unverified_mutations),
            "verification_nudges": self.context.verification_nudges,
        })

    async def reset(self) -> None:
        self.context = LoopContext()
        self.current_plan = None
        self.pending_tool_calls = []
        self.completed_tool_calls = []
        self.tool_result_cache.clear()
        self._cache_epoch = self._registry_epoch()
        self.should_stop = False
        self.is_paused = False
        self.state = LoopState.IDLE
        self._system_prompt_cache = None
        self._system_prompt_fp = None
        self._checklist_rendered = False
        self._last_checklist_signature = None

    def pause(self) -> None:
        self.is_paused = True
        self.state = LoopState.PAUSED

    def resume(self) -> None:
        self.is_paused = False
        self.state = LoopState.THINKING

    def stop(self) -> None:
        self.should_stop = True
        self.state = LoopState.STOPPED

    def get_loop_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "iteration": self.context.iteration,
            "max_iterations": self.max_iterations,
            "has_plan": self.current_plan is not None,
            "plan_completion": (
                self.current_plan.get_completion_percentage()
                if self.current_plan else 0
            ),
            "tool_calls_completed": len(self.completed_tool_calls),
            "tool_calls_pending": len(self.pending_tool_calls),
            "actions_taken": len(self.context.actions_taken),
            "observations": len(self.context.observations),
            "errors": len(self.context.errors),
            "duration": time.time() - self.context.start_time,
            "intent": self.context.intent,
            "llm_calls": self.context.llm_calls,
            "unverified_mutations": len(self.context.unverified_mutations),
            "verification_nudges": self.context.verification_nudges,
            "total_tool_failures": int(
                self.context.metadata.get("total_tool_failures", 0) or 0
            ),
        }

    def add_event_handler(self, event: str, handler: Callable) -> None:
        self.event_handlers.setdefault(event, []).append(handler)

    async def _trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        for handler in self.event_handlers.get(event, []):
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error("Event handler failed: %s", exc)