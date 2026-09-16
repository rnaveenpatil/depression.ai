"""Agent execution loop.

Flow: session context -> LLM -> tool calls -> permission/execution -> tool
results -> canonical context -> LLM, repeated until the model returns a final
answer. AgentLoop owns execution telemetry only; ContextManager owns history.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.agent.planner import Plan, Planner, TaskStatus
from agent.llm.provider import LLMProvider, Message, ToolCall, MODEL_METADATA
from agent.tools.registry import ToolRegistry
from agent.utils.errors import TimeoutError
from agent.utils.logging import get_logger
from agent.utils.redact import redact
from agent.context.runtime import add_tool_call, add_tool_result, get_model_messages

logger = get_logger(__name__)


# Minimum number of tasks for a plan to trigger plan-driven execution.
PLAN_DRIVEN_MIN_TASKS = 3

# Maximum attempts per task before we drop back to free-form mode.
PLAN_TASK_MAX_ATTEMPTS = 3


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
        llm: LLMProvider,
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

        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan: Optional[Plan] = None
        self.pending_tool_calls: List[ToolCall] = []
        self.completed_tool_calls: List[ToolCall] = []

        self.performance_history: deque = deque(maxlen=100)
        self.tool_execution_times: Dict[str, List[float]] = {}
        self.tool_result_cache: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.should_stop = False
        self.is_paused = False

        self._project_ctx: Optional[Dict[str, Any]] = None
        self._project_ctx_ts = 0.0
        self._system_prompt_cache: Optional[str] = None

    @property
    def model_context(self):
        return self.agent.context_manager

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
        self.should_stop = False
        self.is_paused = False
        self._project_ctx = None
        self._project_ctx_ts = 0.0
        self._system_prompt_cache = None

        if max_turns is not None:
            self.max_iterations = max_turns

        try:
            self.state = LoopState.INITIALIZING

            intent = "unknown"
            if self.enable_intent_classification:
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

    # ------------------------------------------------------------------
    # INTENT CLASSIFICATION
    # ------------------------------------------------------------------

    async def _classify_intent(self, query: str) -> str:
        try:
            response = await self.llm.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "Classify the user's request into exactly one "
                            "category:\n"
                            "  feature   — add new functionality\n"
                            "  bugfix    — fix broken behavior\n"
                            "  refactor  — restructure without behavior change\n"
                            "  analysis  — investigate/explain existing code\n"
                            "  question  — factual question, no action needed\n"
                            "  ambiguous — request is unclear or underspecified\n"
                            "Reply with only the category word, lowercase."
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
            logger.debug("Intent classifier returned unexpected %r", raw)
            return "unknown"
        except Exception as exc:
            logger.debug("Intent classification failed: %s", exc)
            return "unknown"

    # ------------------------------------------------------------------
    # FREE-FORM LOOP
    # ------------------------------------------------------------------

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
        return self._result(success, final, error=None if success else "empty final response")

    # ------------------------------------------------------------------
    # PLAN-DRIVEN LOOP
    # ------------------------------------------------------------------

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
            len(plan.tasks),
            self.context.intent,
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
                logger.info("Plan-driven execution stopped by signal")
                stalled = True
                break

            ready = plan.get_pending_tasks()
            if not ready:
                remaining = [
                    t for t in plan._all_tasks() if t.status == TaskStatus.PENDING
                ]
                if not remaining:
                    break
                logger.warning(
                    "Plan-driven: deadlock with %d pending task(s)",
                    len(remaining),
                )
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
                logger.warning("Plan-driven: iteration cap reached")
                stalled = True
                break

            attempt = task_attempts.get(task.id, 0) + 1
            task_attempts[task.id] = attempt

            if attempt > PLAN_TASK_MAX_ATTEMPTS:
                logger.warning(
                    "Plan-driven: task %s exceeded %d attempts; aborting plan",
                    task.id,
                    PLAN_TASK_MAX_ATTEMPTS,
                )
                task.status = TaskStatus.FAILED
                task.error = "exceeded max attempts"
                stalled = True
                break

            logger.info(
                "Plan task %s (%d/%d): %s",
                task.id,
                attempt,
                PLAN_TASK_MAX_ATTEMPTS,
                task.description[:80],
            )

            turn_result = await self._execute_task_turn(
                task=task,
                plan=plan,
                system_prompt=system_prompt,
            )

            if turn_result is None:
                stalled = True
                break

            if turn_result.get("success"):
                await self._trigger_event(
                    "on_plan_task_complete",
                    {"task_id": task.id, "summary": turn_result.get("summary", "")},
                )
                continue

            if turn_result.get("retry"):
                task.status = TaskStatus.PENDING
                continue

            task.status = TaskStatus.FAILED
            task.error = turn_result.get("error") or "unknown"
            await self._trigger_event(
                "on_plan_task_failed",
                {"task_id": task.id, "error": task.error},
            )
            stalled = True
            break

        completed = sum(
            1 for t in plan._all_tasks() if t.status == TaskStatus.COMPLETED
        )
        total = len(plan._all_tasks())
        all_done = plan.is_complete()

        logger.info(
            "Plan-driven execution finished: %d/%d tasks completed",
            completed,
            total,
        )

        if all_done or stalled:
            summary = await self._summarize_plan(plan, system_prompt)
            self.state = LoopState.STOPPED
            self._update_metrics()
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
            "Use tools as needed. When the task is complete, respond with a "
            "short summary of what you did prefixed by 'TASK COMPLETE:'. "
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
                response = await self.llm.complete_with_tools(
                    messages=messages,
                    tools=self._get_available_tools(),
                    temperature=0.4,
                    max_tokens=2000,
                )
            except Exception as exc:
                logger.error("Plan task turn failed: %s", exc)
                return {"success": False, "retry": True}

            self._record_usage(getattr(response, "usage", None))

            content = (getattr(response, "content", "") or "").strip()
            tool_calls = list(getattr(response, "tool_calls", []) or [])

            if not tool_calls:
                upper = content.upper()
                if upper.startswith("TASK FAILED"):
                    return {
                        "success": False,
                        "retry": True,
                        "error": content,
                    }

                summary = content
                if "TASK COMPLETE" in upper:
                    summary = content.split("TASK COMPLETE", 1)[1].lstrip(": ").strip()
                    if not summary:
                        summary = task.description

                if qa and self.enable_qa_verification:
                    verdict = await self._verify_task_qa(task, summary, qa)
                    if not verdict.get("passed", True):
                        logger.warning(
                            "QA failed for task %s: %s",
                            task.id,
                            verdict.get("reason", "no reason"),
                        )
                        return {
                            "success": False,
                            "retry": True,
                            "error": f"QA failed: {verdict.get('reason', '')}",
                        }

                task.status = TaskStatus.COMPLETED
                task.result = summary or task.description
                task.actual_time = time.time() - task.created_at
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

            for tc, res in zip(tool_calls, results):
                payload = res.get("result") or {}
                if not isinstance(payload, dict):
                    payload = {"success": True, "result": payload}
                try:
                    payload = redact(payload)
                except Exception:
                    pass
                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(payload, default=str)[:5000],
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
            response = await self.llm.complete(
                messages=[
                    Message(
                        role="system",
                        content=(
                            "You are a strict verifier. Given a task, its QA "
                            "criterion, and the reported result, decide if the "
                            "QA criterion was actually satisfied.\n"
                            "Reply exactly:\n"
                            "  PASS\n"
                            "or\n"
                            "  FAIL: <short reason>"
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
            logger.debug("QA verifier returned unexpected %r", raw)
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
            bullets.append(f"- ✓ {t.description}: {snippet}")

        for t in failed:
            bullets.append(f"- ✗ {t.description}: {t.error or 'failed'}")

        body = "\n".join(bullets) if bullets else "(no tasks executed)"

        try:
            response = await self.llm.complete(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(
                        role="user",
                        content=(
                            "Summarize the following plan execution for the user. "
                            "State what was completed, what failed, and any "
                            "follow-ups. Do not add unverified claims.\n\n"
                            f"Plan: {plan.goal}\n\n"
                            f"Task outcomes:\n{body}"
                        ),
                    ),
                ],
                temperature=0.3,
                max_tokens=1200,
            )
            summary = (getattr(response, "content", "") or "").strip()
            self._record_usage(getattr(response, "usage", None))
            if summary:
                try:
                    await self.model_context.add_assistant_message(summary)
                except Exception:
                    pass
                return summary
        except Exception as exc:
            logger.debug("Plan summary LLM call failed: %s", exc)

        fallback = f"Plan '{plan.goal}' completed.\n\n{body}"
        try:
            await self.model_context.add_assistant_message(fallback)
        except Exception:
            pass
        return fallback

    # ------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # SYSTEM PROMPT
    # ------------------------------------------------------------------

    async def _get_system_prompt(self) -> str:
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache
        project = await self._get_project_context()
        perms = self._get_permissions_context()
        tool_guide = self._build_tool_guide()
        environment = self._build_environment_block()
        aws_guidance = self._build_aws_guidance()

        self._system_prompt_cache = (
            "You are an advanced AI CLI agent. Your mission is to complete the "
            "user's request accurately and verifiably.\n\n"
            "## Core Responsibilities\n"
            "1. **Understand before acting**: Read relevant files and search the "
            "codebase before making changes.\n"
            "2. **Verify every claim**: Never state something is done, fixed, or "
            "working unless a tool result proves it.\n"
            "3. **Report honestly**: If something failed, say it failed. If "
            "something is uncertain, say it's uncertain.\n"
            "4. **Use the right tool**: Prefer a purpose-built tool over "
            "`bash` whenever one exists. The Tool Selection Guide below "
            "tells you which to pick.\n\n"
            "## Workflow\n"
            "### Phase 1: Explore\n"
            "1. Identify which files are relevant to the request.\n"
            "2. Read them using the `read` or `filesystem` tool.\n"
            "3. Search for related code with `grep` or `search`.\n"
            "4. Do NOT modify anything yet.\n\n"
            "### Phase 2: Act\n"
            "1. Make the smallest change that solves the problem.\n"
            "2. Use `edit` or `write` for file changes, not `bash`.\n"
            "3. Run the relevant verification (tests, linter, build).\n\n"
            "### Phase 3: Verify\n"
            "1. Re-read the modified files to confirm the change landed.\n"
            "2. Run the command that proves the fix works.\n"
            "3. Report the actual command output, not a summary of it.\n\n"
            + tool_guide
            + "\n\n"
            + environment
            + aws_guidance
            + "\n\n## Output Format\n"
            "When you finish, structure your response as:\n"
            "### What I did\n"
            "- Bullet list of changes with file paths\n"
            "### Evidence\n"
            "- Exact command run and its result\n"
            "### Status\n"
            "SUCCESS | PARTIAL | FAILED — one line, with reason if not SUCCESS\n\n"
            "## Anti-Patterns to Avoid\n"
            "- **Claiming success without evidence**: If you didn't run a command "
            "that proves it, you can't claim it.\n"
            "- **Editing files you haven't read**: Always read first.\n"
            "- **Running destructive commands without checking**: `rm`, `git reset`, "
            "`git checkout .` — confirm before executing.\n"
            "- **Running long-lived servers in `bash`/`terminal`**: This blocks the "
            "UI. Use the `process` tool instead.\n"
            "- **Inventing tool names**: If a tool doesn't exist, the registry will "
            "say so. Use only tools from the list below.\n"
            "- **Printing credentials**: Never `echo`, `cat`, or print API keys, "
            "AWS credentials, or tokens. The runtime injects them where needed.\n\n"
            "## Available Tools\n"
            + self._get_tools_description()
            + "\n\n## Project Context\n"
            + json.dumps(project, indent=2, default=str)
            + "\n\n## Permissions\n"
            + json.dumps(perms, indent=2, default=str)
        ).strip()
        return self._system_prompt_cache

    def _build_tool_guide(self) -> str:
        return (
            "## Tool Selection Guide\n"
            "Use the RIGHT tool. Do not default to `bash` for everything.\n\n"
            "**Reading and searching**\n"
            "- Read a specific file: `read` (NOT `bash cat`)\n"
            "- Find files by name/glob: `glob` (NOT `bash find` or `bash ls`)\n"
            "- Search file contents: `grep` (NOT `bash grep` or `bash rg`)\n"
            "- List a directory: `filesystem` with action='list'\n\n"
            "**Editing**\n"
            "- Replace an exact string in a file: `edit` (NOT `bash sed`)\n"
            "- Create or overwrite a file: `write` (NOT `bash echo >`)\n"
            "- Apply a unified diff or multi-file patch: `apply_patch`\n"
            "- Preview a patch without applying: `apply_patch` with action='preview'\n\n"
            "**Running commands**\n"
            "- One-shot commands that exit (tests, build, lint, git, curl): `bash`\n"
            "- Long-running servers (dev servers, watchers, `flutter run`, "
            "`npm run dev`, `vite`, `uvicorn`, `gunicorn`, `nodemon`): "
            "**use the `process` tool, NOT `bash`**. `bash` will block until "
            "the process exits, which for a server is never.\n\n"
            "**Process tool cheat-sheet**\n"
            "- Start a server: `process` with action='start' and command='<cmd>'\n"
            "- Check it's running: `process` with action='list'\n"
            "- Read its output: `process` with action='logs' and process_id='<id>'\n"
            "- Stop it: `process` with action='stop' and process_id='<id>'\n"
            "- Wait for a short command to finish: `process` with action='wait'\n\n"
            "**Web and browser**\n"
            "- Fetch a URL (static HTML): `webfetch`\n"
            "- Search the web: `websearch`\n"
            "- Interact with a page (click, type, JS): `realtime_browser`\n\n"
            "**Planning and tracking**\n"
            "- Track multi-step work: `todowrite` (once at the start) and "
            "`todoread` (to check status)\n"
            "- Ask the user a question: `question` (do NOT guess)\n\n"
            "**Rules of thumb**\n"
            "- Prefer a specific tool over `bash` whenever one exists.\n"
            "- If a command will not exit on its own, use `process start`.\n"
            "- If you are about to run a shell command, pause and ask: is "
            "there a purpose-built tool for this? Usually yes."
        )

    def _build_environment_block(self) -> str:
        """
        Describe the runtime environment so the model knows what's
        available: AWS credentials, package managers, the filesystem.
        """
        lines = ["## Environment\n"]
        lines.append(
            "You are running inside an agent runtime with a working "
            "directory, filesystem access, and a set of tools. The user's "
            "credentials for external services are stored securely and "
            "injected into tool calls where needed — you do not need to "
            "ask for them."
        )

        # Detect a few things about the environment so the model doesn't
        # waste turns probing.
        import shutil as _shutil
        import os as _os
        package_managers = []
        for pm in ("npm", "yarn", "pnpm", "bun", "pip", "uv", "cargo", "go"):
            if _shutil.which(pm):
                package_managers.append(pm)
        if package_managers:
            lines.append(
                "\nPackage managers available on PATH: "
                + ", ".join(package_managers)
                + "."
            )

        if _shutil.which("aws"):
            lines.append(
                "\nThe AWS CLI (`aws`) is installed. You can run AWS "
                "commands directly via `bash` when the `aws` tool or MCP "
                "path is unavailable."
            )
        if _shutil.which("git"):
            lines.append("\nGit is available for version control operations.")

        return "\n".join(lines)

    def _build_aws_guidance(self) -> str:
        """
        Tell the model whether AWS is available and how to reach it.
        Covers three paths: MCP server, unified `aws` helper, and raw CLI.
        """
        try:
            mcp_client = getattr(self.agent, "mcp_client", None)
            from agent.mcp.aws_config import build_aws_cli_fallback_status
            fallback = build_aws_cli_fallback_status()

            mcp_aws_tools = []
            if mcp_client is not None:
                try:
                    mcp_aws_tools = [
                        t["function"]["name"]
                        for t in mcp_client.list_tools()
                        if t["function"]["name"].startswith("mcp__aws__")
                    ]
                except Exception:
                    mcp_aws_tools = []

            header = "\n## AWS access\n"
            lines: List[str] = []

            if mcp_aws_tools:
                lines.append(
                    "The AWS MCP server is CONNECTED. Prefer MCP tools for "
                    "structured, audited calls: "
                    + ", ".join(sorted(mcp_aws_tools))
                    + "."
                )
            elif fallback.get("usable"):
                lines.append(
                    "The AWS MCP server is not available, but the AWS CLI "
                    "and credentials are. Use the `aws` tool, or run "
                    "`aws <service> <operation>` through `bash`."
                )
            else:
                lines.append(
                    "No AWS credentials or CLI are configured yet. If the "
                    "user asks for AWS work, tell them to open the TUI "
                    "`/aws` panel and save their credentials first."
                )

            lines.append(
                "\n**Credentials**\n"
                "- AWS access keys are stored securely by the runtime and "
                "injected into `aws` tool calls and `bash` subprocesses "
                "automatically. You do NOT need to ask the user for them.\n"
                "- NEVER print, echo, or `cat` credentials. The runtime "
                "redacts them from tool output before it reaches you, but "
                "do not try.\n"
                "- To check which AWS account you are acting as, call the "
                "`aws` tool with action='identity'. That runs "
                "`sts get-caller-identity` and returns the account ID, "
                "user ARN, and region.\n"
                "- To list available AWS services you can drive, call the "
                "`aws` tool with action='list_services'.\n"
            )

            lines.append(
                "\n**How to run AWS operations**\n"
                "1. Preferred: `aws` tool with action='call', service='s3', "
                "operation='list-buckets' (structured result, redacted "
                "credentials).\n"
                "2. Fallback: `bash` with `aws s3 ls` (used when the aws "
                "tool reports MCP or helper unavailable).\n"
                "3. For long-running AWS operations (e.g. `aws s3 sync` on "
                "a large bucket), use `process` with action='start' and "
                "poll with `logs`.\n"
            )

            return header + "\n".join(lines)
        except Exception as exc:
            logger.debug("AWS guidance build failed: %s", exc)
            return (
                "\n## AWS access\n"
                "AWS credentials are stored securely by the runtime. Use "
                "the `aws` tool for any AWS operation; it handles "
                "credentials, MCP routing, and CLI fallback.\n"
            )

    def _get_tools_description(self) -> str:
        lines = []
        for name, tool in self.tool_registry.tools.items():
            lines.append(
                f"- {name}: {tool.description}\n"
                f"  Parameters: {json.dumps(tool.parameters, default=str)}"
            )
        return "\n".join(lines)

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

    # ------------------------------------------------------------------
    # PLANNING
    # ------------------------------------------------------------------

    async def _should_plan(self, q: str, intent: str = "unknown") -> bool:
        if intent in ("question", "ambiguous"):
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
            self.context.metadata["plan"] = self.current_plan.to_dict()
            await self.model_context.add_system_message(
                "Execution plan guidance:\n"
                + json.dumps(self.current_plan.to_dict(), indent=2, default=str)
            )
        except Exception as exc:
            logger.warning("Planning failed: %s", exc)
            self.current_plan = None

    def _get_plan_constraints(self, intent: str = "unknown") -> Dict[str, Any]:
        return {
            "max_tasks": 20,
            "timeout": self.default_timeout,
            "available_tools": list(self.tool_registry.tools.keys()),
            "intent": intent,
        }

    # ------------------------------------------------------------------
    # THINK
    # ------------------------------------------------------------------

    async def _think(self) -> Dict[str, Any]:
        await self._ensure_system_prompt()

        messages = get_model_messages(self.model_context, self.max_history_length)
        messages.append(
            Message(
                role="system",
                content="Current execution state: "
                + json.dumps(await self._get_state_context(), default=str),
            )
        )
        if self.current_plan:
            messages.append(
                Message(
                    role="system",
                    content="Plan state: "
                    + json.dumps(self._get_plan_context(), default=str),
                )
            )

        response = await self.llm.complete_with_tools(
            messages=messages,
            tools=self._get_available_tools(),
            temperature=0.7,
            max_tokens=2000,
        )

        calls = list(getattr(response, "tool_calls", []) or [])
        content = getattr(response, "content", None) or ""

        if not calls and not content.strip():
            logger.warning("LLM returned empty content and no tool calls; retrying once")
            retry_messages = list(messages) + [
                Message(
                    role="user",
                    content="Please provide your answer now, or call a tool if you still need information.",
                )
            ]
            response = await self.llm.complete_with_tools(
                messages=retry_messages,
                tools=self._get_available_tools(),
                temperature=0.3,
                max_tokens=2000,
            )
            calls = list(getattr(response, "tool_calls", []) or [])
            content = getattr(response, "content", None) or ""

        self._record_usage(getattr(response, "usage", None))

        if calls:
            await add_tool_call(self.model_context, content, calls)
        else:
            await self.model_context.add_assistant_message(content)

        self.context.tool_calls.extend(calls)
        return {"response": content, "tool_calls": calls}

    async def _ensure_system_prompt(self) -> None:
        messages = getattr(self.model_context, "messages", None) or []
        has_pinned = any(
            getattr(m, "role", None) == "system"
            and getattr(m, "pinned", True)
            and not str(getattr(m, "content", "")).startswith("Current execution state:")
            and not str(getattr(m, "content", "")).startswith("Plan state:")
            for m in messages
        )
        if not has_pinned:
            await self.model_context.add_system_message(
                await self._get_system_prompt(), pinned=True
            )

    def _record_usage(self, usage: Any) -> None:
        u = usage or {}
        if not isinstance(u, dict):
            u = getattr(u, "__dict__", {}) or {}
        prompt = int(u.get("prompt_tokens", 0) or 0)
        completion = int(u.get("completion_tokens", 0) or 0)
        total = int(u.get("total_tokens", 0) or 0) or (prompt + completion)
        self.context.tokens_used += total
        self.context.input_tokens += prompt
        self.context.output_tokens += completion
        self.context.cost += self._compute_cost(u)

    def _compute_cost(self, usage: Any) -> float:
        usage = usage or {}
        if not isinstance(usage, dict):
            usage = getattr(usage, "__dict__", {}) or {}
        try:
            model = self.llm.get_current_model()
        except Exception:
            model = None
        meta = MODEL_METADATA.get(model, {}) if model else {}
        input_rate = meta.get("cost_input", 0.0) or 0.0
        output_rate = meta.get("cost_output", 0.0) or 0.0
        if not input_rate and not output_rate:
            return 0.0
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        return (
            (prompt_tokens / 1_000_000) * input_rate
            + (completion_tokens / 1_000_000) * output_rate
        )

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for name, tool in self.tool_registry.tools.items()
        ]

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

    # ------------------------------------------------------------------
    # FAST PATH
    # ------------------------------------------------------------------

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
        if getattr(calls[0], "name", "") in ("question",):
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
        if isinstance(out, (dict, list)):
            return json.dumps(out, ensure_ascii=False, default=str)
        return None

    # ------------------------------------------------------------------
    # ACT
    # ------------------------------------------------------------------

    async def _act(self, calls: List[ToolCall]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for call in list(calls)[: self.max_tool_calls_per_iteration]:
            try:
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                cache_entry = self.tool_result_cache.get(key) if self.enable_caching else None

                if (
                    cache_entry
                    and time.time() - cache_entry["timestamp"] < 60
                    and cache_entry["result"].get("success", False)
                ):
                    result = cache_entry["result"]
                    await add_tool_result(self.model_context, call, result)
                    self.completed_tool_calls.append(call)
                    self.context.add_action(
                        {
                            "tool": call.name,
                            "tool_call_id": call.id,
                            "params": call.arguments,
                            "result": result,
                            "time": 0.0,
                            "cached": True,
                        }
                    )
                    results.append(
                        {
                            "tool": call.name,
                            "tool_call_id": call.id,
                            "result": result,
                            "cached": True,
                        }
                    )
                    await self._trigger_event(
                        "on_tool_executed",
                        {
                            "tool": call.name,
                            "tool_call_id": call.id,
                            "params": call.arguments,
                            "result": result,
                            "cached": True,
                        },
                    )
                    continue

                start = time.time()
                result = await self.agent.execute_tool(call.name, call.arguments)
                elapsed = time.time() - start

                if not isinstance(result, dict):
                    result = {"success": True, "result": result}

                try:
                    result = redact(result)
                except Exception:
                    pass

                self.tool_execution_times.setdefault(call.name, []).append(elapsed)
                self.context.add_action(
                    {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "params": call.arguments,
                        "result": result,
                        "time": elapsed,
                    }
                )
                await add_tool_result(self.model_context, call, result)

                if self.enable_caching and result.get("success", False):
                    self.tool_result_cache[key] = {
                        "result": result,
                        "timestamp": time.time(),
                    }

                self.completed_tool_calls.append(call)
                results.append(
                    {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "result": result,
                        "execution_time": elapsed,
                    }
                )
                await self._trigger_event(
                    "on_tool_executed",
                    {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "params": call.arguments,
                        "result": result,
                        "execution_time": elapsed,
                    },
                )

            except Exception as exc:
                logger.error("Tool %s failed: %s", getattr(call, "name", "?"), exc, exc_info=True)
                result = {"success": False, "error": str(exc)}
                try:
                    await add_tool_result(self.model_context, call, result)
                except Exception as inner:
                    logger.debug("add_tool_result failed: %s", inner)
                results.append(
                    {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "result": result,
                        "error": str(exc),
                        "success": False,
                    }
                )
                await self._trigger_event(
                    "on_tool_executed",
                    {
                        "tool": call.name,
                        "tool_call_id": call.id,
                        "params": call.arguments,
                        "result": result,
                        "error": str(exc),
                    },
                )

        return results

    async def _observe(self, results: List[Dict[str, Any]]) -> None:
        for x in results:
            r = x.get("result") or {}
            self.context.add_observation(
                {
                    "timestamp": time.time(),
                    "tool": x.get("tool"),
                    "tool_call_id": x.get("tool_call_id"),
                    "success": bool(r.get("success", False)),
                    "error": x.get("error") or r.get("error"),
                    "execution_time": x.get("execution_time", 0),
                    "result": r,
                }
            )

    async def _evaluate(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True

        signature = tuple(
            sorted(
                (
                    r.get("tool"),
                    str((r.get("result") or {}).get("error", r.get("error", "")))[:120],
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

    # ------------------------------------------------------------------
    # FINAL RESPONSE
    # ------------------------------------------------------------------

    async def _generate_final_response(self) -> str:
        messages = get_model_messages(self.model_context, self.max_history_length)
        if not any(getattr(m, "role", None) == "system" for m in messages):
            messages.insert(0, Message(role="system", content=await self._get_system_prompt()))
        messages.append(
            Message(
                role="user",
                content=(
                    "Provide the final response to the user's original request. "
                    "Use the Output Format from the system prompt "
                    "(### What I did / ### Evidence / ### Status). "
                    "State what was actually completed and mention any remaining issue; "
                    "do not claim unverified success."
                ),
            )
        )

        try:
            response = await self.llm.complete(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception as exc:
            logger.error("Final response generation failed: %s", exc)
            return ""

        content = getattr(response, "content", "") or ""
        self._record_usage(getattr(response, "usage", None))
        if content:
            try:
                await self.model_context.add_assistant_message(content)
            except Exception as exc:
                logger.debug("add_assistant_message failed: %s", exc)
        return content

    # ------------------------------------------------------------------
    # METRICS / LIFECYCLE
    # ------------------------------------------------------------------

    def _update_metrics(self) -> None:
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "duration": time.time() - self.context.start_time,
                "iterations": self.context.iteration,
                "tool_calls": len(self.completed_tool_calls),
                "tokens": self.context.tokens_used,
                "errors": len(self.context.errors),
                "intent": self.context.intent,
            }
        )

    async def reset(self) -> None:
        self.context = LoopContext()
        self.current_plan = None
        self.pending_tool_calls = []
        self.completed_tool_calls = []
        self.should_stop = False
        self.is_paused = False
        self.state = LoopState.IDLE
        self._system_prompt_cache = None

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
                self.current_plan.get_completion_percentage() if self.current_plan else 0
            ),
            "tool_calls_completed": len(self.completed_tool_calls),
            "tool_calls_pending": len(self.pending_tool_calls),
            "actions_taken": len(self.context.actions_taken),
            "observations": len(self.context.observations),
            "errors": len(self.context.errors),
            "duration": time.time() - self.context.start_time,
            "intent": self.context.intent,
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