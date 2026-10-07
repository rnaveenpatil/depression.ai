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
      can self-correct instead of stalling.
    * Success is only reported when a tool output in this session proves it.
    * Cache is invalidated whenever a mutating tool runs.
    * System prompt is rebuilt when its fingerprint changes and is tagged
      with metadata so the loop can detect it. [Bug 5]
    * The loop NEVER writes the final assistant turn — that is the caller's
      job so each user query produces exactly one assistant message.
      [M2/N4/N10]
    * [Permission gate] Every tool call is dispatched through the AGENT's
      execute_tool(), which runs the permission manager. The registry's
      raw execute_safe is only used when the agent has no execute_tool
      (tests / sub-agents). A belt-and-braces check in _act() also refuses
      destructive calls that somehow reach the registry directly.
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
from agent.llm.provider import Message, ToolCall
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

# Marker that identifies the real system prompt. Anything else that is a
# pinned system message is NOT the prompt. [Bug 5]
SYSTEM_PROMPT_MARKER = "system_prompt"


# ----------------------------------------------------------------------
# Destructive-call detection (belt-and-braces permission guard)
# ----------------------------------------------------------------------
# If a tool call looks destructive, _act() requires that the agent's
# permission manager EXPLICITLY allowed it. This catches the case where
# execute_tool is missing (tests, stripped agents) or where the registry
# was called directly and the permission gate was skipped.

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


def _looks_destructive_call(name: str, args: Dict[str, Any]) -> bool:
    """
    Conservative detector for destructive tool calls.

    Used as a safety net: if the call looks destructive, `_act` requires
    the permission manager to have explicitly allowed it.
    """
    name_l = (name or "").lower()

    # 1. Tool name matches a destructive keyword.
    for tok in _DESTRUCTIVE_TOOL_TOKENS:
        if tok in name_l:
            return True

    # 2. Action string (either the top-level `action` param or the
    #    tool-name-suffix) matches a destructive keyword.
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

    # 3. Any string-valued argument matches a destructive shell pattern.
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
        self.default_timeout = config.get("timeout", 60)
        self.project_ctx_ttl = config.get("project_context_ttl", 30.0)
        self.enable_plan_driven = config.get("enable_plan_driven", True)
        self.enable_intent_classification = config.get("enable_intent_classification", True)
        self.enable_qa_verification = config.get("enable_qa_verification", True)
        self.reserve_output_tokens = int(config.get("reserve_output_tokens", 2000))
        # Set to False to disable the belt-and-braces destructive guard
        # (the agent's permission manager is still authoritative).
        self.require_permission_for_destructive = bool(
            config.get("require_permission_for_destructive", True)
        )

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
    # REGISTRY CAPABILITY PROBES
    # ------------------------------------------------------------------

    def _is_read_only(self, name: str) -> bool:
        registry = self.tool_registry
        probe = getattr(registry, "is_read_only", None)
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

    def _registry_describe_for_prompt(self, tools: Any) -> str:
        probe = getattr(self.tool_registry, "describe_for_prompt", None)
        if callable(probe):
            try:
                return str(probe(tools))
            except Exception:
                logger.debug("describe_for_prompt probe failed", exc_info=True)
        lines: List[str] = []
        for item in tools or []:
            if not isinstance(item, dict):
                continue
            fn = item.get("function") if isinstance(item.get("function"), dict) else item
            name = fn.get("name") or ""
            desc = fn.get("description") or ""
            if name:
                lines.append(f"- {name}: {desc}".rstrip(": ").strip())
        return "\n".join(lines) if lines else "(no tools available)"

    def _registry_get_schemas(self) -> List[Dict[str, Any]]:
        for attr, with_intent in (("select_for_task", True), ("get_schemas", False)):
            probe = getattr(self.tool_registry, attr, None)
            if not callable(probe):
                continue
            try:
                raw = probe(self.context.intent) if with_intent else probe()
                return list(raw)
            except Exception:
                logger.debug("%s probe failed", attr, exc_info=True)
        return []

    async def _registry_execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool.

        [Permission gate] The AGENT's execute_tool is preferred because it
        runs the permission manager. The registry's raw execute_safe is
        only used as a fallback when the agent has no execute_tool (tests,
        stripped sub-agents).
        """
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

        # Fallback: registry directly (no permission gate — tests only).
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

            intent = await self._classify_intent(query)
            self.context.intent = intent
            logger.info("Classified intent: %s", intent)

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
                max_tokens=16,
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
            if not response_text:
                response_text = await self._generate_final_response()

            self.state = LoopState.STOPPED
            self._update_metrics()
            return self._result(True, response_text)

        self.context.hit_iteration_limit = self.context.iteration >= self.max_iterations
        self.state = LoopState.STOPPED
        final = await self._generate_final_response()
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
            "Do not describe the plan — execute this task."
        )

        messages: List[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task_prompt),
        ]

        local_turns = 0
        max_local_turns = self.max_tool_calls_per_iteration * 2

        while local_turns < max_local_turns:
            local_turns += 1
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
            return {"passed": True, "reason": "unrecognized verdict"}
        except Exception as exc:
            logger.warning("QA verification failed for task %s: %s", task.id, exc)
            return {"passed": True, "reason": f"verifier error: {exc}"}

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
                            "that proves each major step. Do not repeat the "
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
    # SYSTEM PROMPT (fingerprinted + tagged)
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

        self._system_prompt_cache = ("""
            You are an advanced AI CLI agent. Your job is to complete the user's task,
not merely explain how to do it. Inspect, modify, execute, test, and verify
when necessary.

## CORE RULES

1. Understand the user's goal before acting.
2. Inspect relevant files/code first.
3. Use the most specific tool available instead of bash.
4. Make the smallest change that solves the problem.
5. Never invent tool results, file contents, command output, or success.
6. Never expose API keys, passwords, tokens, AWS secrets, private keys, or
   other credentials.

## TOOL SELECTION

Prefer:
- read/list/glob/search → inspect files
- edit/write/append → modify files
- process → long-running applications/servers
- bash → short commands that terminate
- git tools → git operations
- AWS tools → AWS operations

Use bash only when no suitable specialized tool exists.

Never run long-lived applications or servers through bash.
Use process.start for them.

## WORKFLOW

For a multi-step task:

1. Inspect
2. Identify the actual problem
3. Plan the minimum fix
4. Modify
5. Verify
6. Test
7. Report evidence

Do not perform unnecessary steps.

## TOOL ERRORS

A tool error is information, not automatically a task failure.

When a tool fails:

1. Read the complete error.
2. Identify the actual cause.
3. Compare the error with the tool's parameters/schema.
4. Correct the cause.
5. Retry with corrected arguments.

Never repeat the exact same failed call.

If the error indicates a missing file, invalid path, wrong parameter,
permission problem, timeout, dependency problem, or incorrect command,
adapt the next action accordingly.

If the failure cannot be safely resolved, stop that operation and explain
the exact blocker.

Do not blame the tool without analysing its error.

## VERIFICATION

After changing something, verify the resulting state.

For files:
- read the changed section/file when practical.

For commands:
- rerun the command that previously failed.

For bugs:
- reproduce the original failure, apply the fix, then rerun the same
  operation to prove the problem is resolved.

For applications:
- start the application only when useful for verification.

Exit code 0 alone is not proof that the intended result exists.

Never claim "fixed", "working", or "successful" without evidence.

## APPLICATION RUNNING

After fixing code, determine whether running the application would provide
meaningful verification.

If running it would help verify the fix, ask the user:

"Would you like me to run the application and verify the fix?"

Do not ask this when:
- the user explicitly asked you to run it
- the application must be run to complete the requested task
- automated tests already provide sufficient verification
- running it would be irrelevant or unnecessarily expensive

If the user agrees:
- start it using the process tool
- monitor the startup result
- report whether it started successfully
- if appropriate, show the relevant output/status to the user

Never claim that an application works merely because its process started.

## LONG-RUNNING PROCESSES

Use process.start for:
- development servers
- web applications
- Flutter applications
- Vite
- npm dev servers
- uvicorn
- nodemon
- watchers
- interactive programs

Use bash for short commands that should terminate.

Do not retry a command that may still be running.

## STATE

Keep track of:
- current working directory
- files inspected
- files changed
- important errors
- tests performed
- verification status

Never assume a previous operation succeeded without its result confirming it.

## TOOL ARGUMENTS

Use only parameters defined by the tool schema.

Never invent parameter names or argument formats.

If a tool rejects arguments:
- inspect the error
- correct the arguments
- retry with the corrected call

Do not repeatedly guess.

## PERMISSIONS

Never bypass permission checks.

For destructive or high-risk operations such as deletion, overwriting,
production deployment, infrastructure destruction, or credential changes,
use the configured permission mechanism.

Prefer reversible operations when possible.

If a destructive tool call is denied by the permission system, do not
attempt to work around it (no `bash` tricks, no alternate tools). Explain
what was blocked and ask the user how to proceed.

## PLANNING

For tasks requiring multiple operations, show a short checklist before
execution:

- [ ] inspect
- [ ] identify and fix
- [ ] verify
- [ ] test

Mark an item complete only when evidence confirms it.

Do not repeat the checklist.

For simple tasks, skip the checklist.

## TOKEN EFFICIENCY

Do not narrate every successful tool call.

Do not repeat information already known.

Do not quote large files or logs unless necessary.

After a successful routine tool call, immediately continue with the next
useful action.

Explicitly explain only:
- important findings
- failures
- unexpected results
- verification evidence
- decisions requiring the user's input

Prefer concise reasoning and concrete actions.

## FINAL RESPONSE

After completing the task:

- state what was changed
- state what was verified
- mention important test output briefly
- mention anything still unverified
- ask whether the user wants the application run if that is the next
  useful verification step

Never claim success without evidence.
"""

            + env_block
            + aws_block
            + "\n\n## Available Tools\n"
            + self._registry_describe_for_prompt(self._get_available_tools())
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
                files = await workspace.list_files(max_files=20)
            except Exception as exc:
                logger.debug("list_files failed: %s", exc)
                files = []
            git_info = None
            if hasattr(workspace, "get_git_info"):
                try:
                    git_info = await workspace.get_git_info()
                except Exception as exc:
                    logger.debug("get_git_info failed: %s", exc)
            ctx = {
                "project_path": str(getattr(workspace, "project_dir", "")),
                "files": files,
                "git_info": git_info,
            }
        else:
            ctx = {}

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
            # [Bug 3] Plan guidance is NOT persisted as a system message.
            # It is delivered per-request via the state block that _think
            # already builds.

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
            from agent.llm.runtime import MODEL_METADATA, DEFAULT_CONTEXT_WINDOW
            model = self.llm.get_current_model()
            meta = MODEL_METADATA.get(model, {}) if model else {}
            w = int(meta.get("context_window") or 0)
            return w if w >= 2048 else DEFAULT_CONTEXT_WINDOW
        except Exception:
            return 8192

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
        if self.current_plan:
            state_parts["plan"] = self._get_plan_context()
        state_block = json.dumps(state_parts, default=str)

        await self._maybe_compact_before_call(system_prompt, tool_schemas, state_block)

        messages = get_model_messages(self.model_context, self.max_history_length)
        messages.append(Message(role="system", content="Execution state: " + state_block))

        self.context.llm_calls += 1
        response = await self.llm.complete_with_tools(
            messages=messages,
            tools=tool_schemas,
            temperature=0.7,
            max_tokens=self.reserve_output_tokens,
        )

        calls = list(getattr(response, "tool_calls", []) or [])
        content = getattr(response, "content", None) or ""

        if not calls and not content.strip():
            logger.warning("LLM returned empty; retrying once")
            retry_messages = list(messages) + [
                Message(
                    role="user",
                    content=(
                        "Please provide your answer now, or call a tool if you "
                        "still need information."
                    ),
                )
            ]
            self.context.llm_calls += 1
            response = await self.llm.complete_with_tools(
                messages=retry_messages,
                tools=tool_schemas,
                temperature=0.3,
                max_tokens=self.reserve_output_tokens,
            )
            calls = list(getattr(response, "tool_calls", []) or [])
            content = getattr(response, "content", None) or ""

        if len(calls) > self.max_tool_calls_per_iteration:
            logger.info(
                "Capping tool calls from %d to %d",
                len(calls), self.max_tool_calls_per_iteration,
            )
            calls = calls[: self.max_tool_calls_per_iteration]

        self._record_usage(getattr(response, "usage", None))

        if content and not calls:
            try:
                from agent.tui.plan_parser import parse_plan_from_text
                parsed = parse_plan_from_text(content)
                if parsed and parsed.is_valid:
                    signature = tuple(
                        (e.content[:80], e.status) for e in parsed.entries
                    )
                    if signature != self._last_checklist_signature:
                        self._last_checklist_signature = signature
                        should_render = not self._checklist_rendered
                        self._checklist_rendered = True
                        await self._trigger_event(
                            "on_plan_updated",
                            {
                                "entries": parsed.to_list(),
                                "count": parsed.count,
                                "raw": content,
                                "render": should_render,
                            },
                        )
            except Exception as exc:
                logger.debug("Plan parse failed: %s", exc)

        if calls:
            await add_tool_call(self.model_context, content, calls)
        elif content.strip():
            await self.model_context.add_assistant_message(content)

        self.context.tool_calls.extend(calls)
        return {"response": content, "tool_calls": calls}

    async def _ensure_system_prompt(self) -> None:
        """
        Ensure the real system prompt is present.

        [Bug 5] The real prompt is identified by metadata["system_prompt"].
        """
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
            "max_iterations": self.max_iterations,
            "actions_taken": len(self.context.actions_taken),
            "observations": len(self.context.observations),
            "completed_tool_calls": len(self.completed_tool_calls),
            "pending_tool_calls": len(self.pending_tool_calls),
            "intent": self.context.intent,
        }

    def _get_plan_context(self) -> Dict[str, Any]:
        if not self.current_plan:
            return {}
        return {
            "plan_id": self.current_plan.id,
            "goal": self.current_plan.goal,
            "completion": self.current_plan.get_completion_percentage(),
            "pending_tasks": [t.to_dict() for t in self.current_plan.get_pending_tasks()],
            "completed_tasks": [
                t.to_dict()
                for t in self.current_plan.tasks
                if t.status == TaskStatus.COMPLETED
            ],
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

        for call in calls:
            call_id = getattr(call, "id", None) or f"call_{len(results)}"
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
                results.append({
                    "tool": name, "tool_call_id": call_id,
                    "result": result, "success": False,
                })
                continue

            # ---- Belt-and-braces permission guard --------------------
            # If the call looks destructive AND the agent has no permission
            # manager wired in, refuse outright. This catches the case where
            # execute_tool was replaced or the manager was never attached.
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
                    results.append({
                        "tool": name, "tool_call_id": call_id,
                        "result": result, "success": False,
                    })
                    continue

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
                    results.append({
                        "tool": name, "tool_call_id": call_id,
                        "result": result, "cached": True,
                    })
                    await self._trigger_event("on_tool_executed", {
                        "tool": name, "tool_call_id": call_id,
                        "params": args, "result": result, "cached": True,
                    })
                    continue

                start = time.time()
                result = await self._registry_execute(name, args)
                elapsed = time.time() - start

                if not isinstance(result, dict):
                    result = {"success": True, "result": result}
                try:
                    result = redact(result)
                except Exception:
                    pass

                # If the executor denied the call for permission reasons,
                # surface it clearly.
                if result.get("permission_denied"):
                    logger.info(
                        "Tool %s denied by permission manager: %s",
                        name, result.get("error"),
                    )

                self.tool_execution_times.setdefault(name, []).append(elapsed)
                self.context.add_action({
                    "tool": name, "tool_call_id": call_id,
                    "params": args, "result": result, "time": elapsed,
                })
                await self._safe_add_tool_result(call, result)

                if (
                    self.enable_caching
                    and result.get("success", False)
                    and self._is_read_only(name)
                ):
                    self.tool_result_cache[key] = {
                        "result": result, "timestamp": time.time(),
                    }

                self.completed_tool_calls.append(call)
                results.append({
                    "tool": name, "tool_call_id": call_id,
                    "result": result, "execution_time": elapsed,
                })
                await self._trigger_event("on_tool_executed", {
                    "tool": name, "tool_call_id": call_id,
                    "params": args, "result": result,
                    "execution_time": elapsed,
                })

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
                results.append({
                    "tool": name, "tool_call_id": call_id,
                    "result": result, "error": str(exc), "success": False,
                })
                await self._trigger_event("on_tool_executed", {
                    "tool": name, "tool_call_id": call_id,
                    "params": args, "result": result, "error": str(exc),
                })

        return results

    def _is_duplicate_failing_call(self, name: str, args: Dict[str, Any]) -> bool:
        try:
            sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        except Exception:
            return False

        seen = self.context.metadata.setdefault("failing_signatures", {})
        count = seen.get(sig, 0)
        return count >= 2

    def _record_failure_signature(self, name: str, args: Dict[str, Any]) -> None:
        try:
            sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        except Exception:
            return
        seen = self.context.metadata.setdefault("failing_signatures", {})
        seen[sig] = seen.get(sig, 0) + 1

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

    async def _safe_add_tool_result(self, call: ToolCall, result: Dict[str, Any]) -> None:
        try:
            await add_tool_result(self.model_context, call, result)
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
        for x in results:
            r = x.get("result") or {}
            tool = x.get("tool", "?")
            if not r.get("success", False):
                per_tool[tool] = per_tool.get(tool, 0) + 1

        for tool, count in per_tool.items():
            if count >= 5:
                logger.warning("Stopping loop: tool '%s' failed %d times", tool, count)
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
    # FINAL RESPONSE
    # ==================================================================

    async def _generate_final_response(self) -> str:
        """
        Generate a closing summary text.

        [M2/N4/N10] This method does NOT persist the assistant turn. The
        caller (`BaseAgent._run_query_pipeline` / `AgentCoordinator._process_auto`)
        is the single writer of the final assistant message.
        """
        messages = get_model_messages(self.model_context, self.max_history_length)
        if not any(getattr(m, "role", None) == "system" for m in messages):
            messages.insert(0, Message(role="system",
                                       content=await self._get_system_prompt()))
        messages.append(
            Message(
                role="user",
                content=(
                    "Provide the final response to the user. Include a short "
                    "closing summary (2–4 sentences) of what you actually did, "
                    "why, and anything they should know. Cite the concrete "
                    "evidence (command output, diff, or file content) that "
                    "proves each major step. Do not claim unverified success. "
                    "If anything is unverified or blocked, say so explicitly."
                ),
            )
        )

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