"""
Planner Module - Task Decomposition and Planning.

Hierarchical task decomposition, dependency management, and adaptive
re-planning.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agent.utils.logging import get_logger
from agent.utils.errors import PlanningError, ExecutionError
from agent.llm.provider import LLMProvider, Message
from agent.tools.registry import ToolRegistry

logger = get_logger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RETRY = "retry"


class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    OPTIONAL = 4


@dataclass
class Task:
    description: str
    id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List["Task"] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    estimated_time: Optional[float] = None
    actual_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "dependencies": list(self.dependencies),
            "subtasks": [t.to_dict() for t in self.subtasks],
            "tool_calls": self.tool_calls,
            "estimated_time": self.estimated_time,
            "actual_time": self.actual_time,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        task = cls(
            description=data["description"],
            id=data.get("id", ""),
            status=TaskStatus(data.get("status", "pending")),
            priority=Priority(data.get("priority", 2)),
            dependencies=list(data.get("dependencies", []) or []),
            tool_calls=list(data.get("tool_calls", []) or []),
            estimated_time=data.get("estimated_time"),
            actual_time=data.get("actual_time"),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )
        task.subtasks = [cls.from_dict(st) for st in data.get("subtasks", [])]
        return task


@dataclass
class Plan:
    goal: str
    tasks: List[Task]
    id: str = ""
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())

    def _all_tasks(self) -> List[Task]:
        out: List[Task] = []
        stack = list(self.tasks)
        while stack:
            t = stack.pop()
            out.append(t)
            stack.extend(t.subtasks)
        return out

    def get_pending_tasks(self) -> List[Task]:
        all_tasks = self._all_tasks()
        completed_ids = {t.id for t in all_tasks if t.status == TaskStatus.COMPLETED}
        ready: List[Task] = []
        for task in all_tasks:
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=lambda t: t.priority.value)

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        for task in self._all_tasks():
            if task.id == task_id:
                return task
        return None

    def get_completion_percentage(self) -> float:
        all_tasks = self._all_tasks()
        if not all_tasks:
            return 0.0
        completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(all_tasks)) * 100

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self._all_tasks())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics,
            "context": self.context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        return cls(
            goal=data["goal"],
            tasks=[Task.from_dict(t) for t in data["tasks"]],
            id=data.get("id", ""),
            status=data.get("status", "active"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            completed_at=data.get("completed_at"),
            metrics=data.get("metrics", {}),
            context=data.get("context", {}),
            context_extra=data.get("context_extra", {}) if False else {},
        )


class Planner:
    """Task planner with hierarchical decomposition and dependency resolution."""

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        config: Dict[str, Any],
        fallback_chain: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        self.fallback_chain = fallback_chain

        self.max_tasks_per_plan = config.get("max_tasks_per_plan", 20)
        self.max_subtasks_per_task = config.get("max_subtasks_per_task", 10)
        self.enable_parallel = config.get("enable_parallel", True)
        self.max_parallel_tasks = config.get("max_parallel_tasks", 5)
        self.adaptive_planning = config.get("adaptive_planning", True)

        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
        self.execution_context: Dict[str, Any] = {}
        self.cached_plans: Dict[str, Plan] = {}

        self.metrics = {
            "plans_created": 0,
            "tasks_decomposed": 0,
            "replans": 0,
            "avg_planning_time": 0.0,
        }

        logger.info("Planner initialized")

    # ------------------------------------------------------------------
    # PLAN CREATION
    # ------------------------------------------------------------------

    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        start_time = time.time()
        try:
            planning_context = self._prepare_planning_context(goal, context, constraints)
            plan_data = await self._generate_plan(goal, planning_context)
            plan = await self._structure_plan(goal, plan_data, constraints)
            self._validate_dependencies(plan)
            await self._estimate_resources(plan)

            self.current_plan = plan
            self.plan_history.append(plan)
            self.metrics["plans_created"] += 1

            planning_time = time.time() - start_time
            n = self.metrics["plans_created"]
            self.metrics["avg_planning_time"] = (
                (self.metrics["avg_planning_time"] * (n - 1) + planning_time) / n
            )
            logger.info("Created plan with %d tasks in %.2fs", len(plan.tasks), planning_time)
            return plan
        except Exception as e:
            logger.error("Planning failed: %s", e, exc_info=True)
            raise PlanningError(f"Failed to create plan: {e}")

    async def _generate_plan(self, goal: str, context: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = self._get_planning_system_prompt()
        user_prompt = self._get_planning_user_prompt(goal, context)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
        try:
            response = await self.llm.complete(
                messages=messages, temperature=0.3, max_tokens=2000,
            )
            content = getattr(response, "content", "") or ""
        except Exception as e:
            logger.error("Planner LLM call failed: %s", e)
            return self._create_fallback_plan(goal)

        try:
            return self._parse_plan_response(content)
        except Exception as e:
            logger.error("Failed to parse plan: %s", e)
            return self._create_fallback_plan(goal)

    def _get_planning_system_prompt(self) -> str:
        return (
            "You are an expert task planner. Decompose complex goals into "
            "executable tasks with clear dependencies and priorities.\n\n"
            "Principles:\n"
            "1. Break the goal into discrete, actionable tasks.\n"
            "2. Identify dependencies between tasks.\n"
            "3. Assign priorities (critical, high, medium, low, optional).\n"
            "4. Suggest tools for each task.\n"
            "5. Consider parallel execution opportunities.\n"
            "6. Include verification steps.\n\n"
            "Output ONLY JSON in this shape:\n"
            "{\n"
            '  "tasks": [\n'
            "    {\n"
            '      "id": "task_1",\n'
            '      "description": "...",\n'
            '      "priority": "high",\n'
            '      "dependencies": [],\n'
            '      "subtasks": [],\n'
            '      "tools": [],\n'
            '      "estimated_time": 30,\n'
            '      "validation": "..."\n'
            "    }\n"
            "  ],\n"
            '  "parallel_groups": [],\n'
            '  "estimated_total_time": 300\n'
            "}"
        )

    def _get_planning_user_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        available_tools = [t.name for t in self.tool_registry.tools.values()]
        return (
            f"Goal: {goal}\n\n"
            f"Available Tools: {', '.join(available_tools)}\n\n"
            f"Additional Context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Create a detailed plan to achieve this goal."
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return candidate[start : i + 1]
        return None

    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        block = self._extract_json_block(response)
        if block:
            try:
                return json.loads(block)
            except json.JSONDecodeError as e:
                logger.warning("JSON decode failed, using fallback parser: %s", e)
        return self._create_fallback_plan_data(response)

    def _create_fallback_plan_data(self, response: str) -> Dict[str, Any]:
        tasks = re.split(r"\d+\.\s*|•\s*|\n- ", response or "")
        plan_data: Dict[str, Any] = {"tasks": [], "parallel_groups": [], "estimated_total_time": 300}
        for i, task_desc in enumerate(tasks):
            if task_desc.strip():
                plan_data["tasks"].append(
                    {
                        "id": f"task_{i + 1}",
                        "description": task_desc.strip(),
                        "priority": "medium",
                        "dependencies": [],
                        "subtasks": [],
                        "tools": [],
                        "estimated_time": 30,
                        "validation": "Verify task completion",
                    }
                )
        return plan_data

    def _create_fallback_plan(self, goal: str) -> Dict[str, Any]:
        return {
            "tasks": [
                {
                    "id": "task_1",
                    "description": f"Analyze goal: {goal}",
                    "priority": "critical",
                    "dependencies": [],
                    "subtasks": [],
                    "tools": ["analyze"],
                    "estimated_time": 30,
                    "validation": "Goal understood",
                },
                {
                    "id": "task_2",
                    "description": "Execute primary tasks",
                    "priority": "high",
                    "dependencies": ["task_1"],
                    "subtasks": [],
                    "tools": ["execute"],
                    "estimated_time": 60,
                    "validation": "Tasks completed",
                },
                {
                    "id": "task_3",
                    "description": "Verify results",
                    "priority": "medium",
                    "dependencies": ["task_2"],
                    "subtasks": [],
                    "tools": ["verify"],
                    "estimated_time": 30,
                    "validation": "Results verified",
                },
            ],
            "parallel_groups": [],
            "estimated_total_time": 120,
        }

    def _coerce_priority(self, value: Any) -> Priority:
        if isinstance(value, Priority):
            return value
        if isinstance(value, int):
            try:
                return Priority(value)
            except ValueError:
                return Priority.MEDIUM
        if isinstance(value, str):
            try:
                return Priority[value.upper()]
            except KeyError:
                return Priority.MEDIUM
        return Priority.MEDIUM

    async def _structure_plan(
        self,
        goal: str,
        plan_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]],
    ) -> Plan:
        tasks: List[Task] = []

        def build(data: Dict[str, Any]) -> Task:
            subtasks = [build(st) for st in (data.get("subtasks") or [])][
                : self.max_subtasks_per_task
            ]
            return Task(
                description=data.get("description", "(unnamed task)"),
                id=data.get("id", "") or str(uuid.uuid4())[:8],
                priority=self._coerce_priority(data.get("priority", "medium")),
                dependencies=list(data.get("dependencies", []) or []),
                subtasks=subtasks,
                tool_calls=[{"tool": t} for t in (data.get("tools") or [])],
                estimated_time=data.get("estimated_time"),
                metadata={"validation": data.get("validation", "")},
            )

        for task_data in plan_data.get("tasks", [])[: self.max_tasks_per_plan]:
            tasks.append(build(task_data))

        return Plan(
            goal=goal,
            tasks=tasks,
            context=plan_data.get("context", {}) or {},
            metrics={
                "parallel_groups": plan_data.get("parallel_groups", []) or [],
                "estimated_total_time": plan_data.get("estimated_total_time", 0),
                "constraints": constraints or {},
            },
        )

    def _prepare_planning_context(
        self,
        goal: str,
        context: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        planning_context: Dict[str, Any] = {
            "goal": goal,
            "available_tools": list(self.tool_registry.tools.keys()),
            "max_tasks": self.max_tasks_per_plan,
            "enable_parallel": self.enable_parallel,
            "context": context or {},
            "constraints": constraints or {},
        }
        if self.current_plan:
            planning_context["previous_plan"] = {
                "tasks_completed": len(
                    [t for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]
                ),
                "previous_goal": self.current_plan.goal,
            }
        return planning_context

    def _validate_dependencies(self, plan: Plan) -> None:
        all_tasks = plan._all_tasks()
        task_ids = {t.id for t in all_tasks}
        for task in all_tasks:
            invalid = [dep for dep in task.dependencies if dep not in task_ids]
            if invalid:
                logger.warning("Task %s has invalid dependencies: %s", task.id, invalid)
            task.dependencies = [dep for dep in task.dependencies if dep in task_ids]

    async def _estimate_resources(self, plan: Plan) -> None:
        for task in plan._all_tasks():
            if not task.estimated_time:
                task.estimated_time = (
                    30 + len(task.subtasks) * 15 + len(task.dependencies) * 10
                )

    # ------------------------------------------------------------------
    # PLAN-DRIVEN EXECUTION (used by AgentLoop)
    # ------------------------------------------------------------------

    async def execute_task_with_llm(
        self,
        task: Task,
        plan: Plan,
        loop_context: Dict[str, Any],
        turn_executor: Callable[[List[Message], List[Dict[str, Any]], float], Any],
        system_prompt: str,
    ) -> Dict[str, Any]:
        """
        Execute a single task with the LLM.

        `turn_executor` is a callable supplied by the loop with the signature:
            async (messages, tools, temperature) -> LLMResponse

        The loop owns the tool registry and the message plumbing, so the
        planner stays model- and tool-agnostic here.

        Returns:
            {"success": bool, "summary": str, "error": Optional[str], "turns": int}
        """
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()

        dep_summaries = []
        for dep_id in task.dependencies:
            dep = plan.get_task_by_id(dep_id)
            if dep and dep.result:
                dep_summaries.append(f"- {dep.description}: {str(dep.result)[:300]}")

        task_prompt = (
            f"Execute task {task.id}: {task.description}\n\n"
        )
        if dep_summaries:
            task_prompt += (
                "Results from prerequisite tasks:\n"
                + "\n".join(dep_summaries)
                + "\n\n"
            )
        if task.metadata.get("validation"):
            task_prompt += f"Validation criteria: {task.metadata['validation']}\n\n"
        task_prompt += (
            "Use tools as needed. When the task is complete, respond with a "
            "short summary of what you did prefixed by 'TASK COMPLETE:'. "
            "If the task cannot be completed, respond with 'TASK FAILED: <reason>'.\n"
            "Do not describe the plan — execute this task."
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task_prompt),
        ]

        try:
            response = await turn_executor(messages, [], 0.4)
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.retry_count += 1
            return {
                "success": False,
                "summary": "",
                "error": str(exc),
                "turns": 0,
            }

        content = (getattr(response, "content", "") or "").strip()
        tool_calls = list(getattr(response, "tool_calls", []) or [])

        # If the LLM still wants tools, we hand off to the loop's own act path.
        # The loop calls us with turn_executor bound to its think-only path,
        # so we return a "needs tools" signal and let the loop drive.
        if tool_calls:
            return {
                "success": False,
                "summary": "",
                "error": "task requires tool execution",
                "turns": 1,
                "needs_tools": True,
                "pending_tool_calls": tool_calls,
                "assistant_text": content,
                "messages": messages,
            }

        upper = content.upper()
        if upper.startswith("TASK FAILED"):
            task.status = TaskStatus.FAILED
            task.error = content
            return {
                "success": False,
                "summary": content,
                "error": content,
                "turns": 1,
            }

        if "TASK COMPLETE" in upper:
            summary = content.split("TASK COMPLETE", 1)[1].lstrip(": ").strip()
            task.status = TaskStatus.COMPLETED
            task.result = summary or content
            task.actual_time = time.time() - task.created_at
            return {
                "success": True,
                "summary": summary or content,
                "error": None,
                "turns": 1,
            }

        # LLM didn't use the marker — treat as complete with the raw content.
        task.status = TaskStatus.COMPLETED
        task.result = content or "(no summary)"
        task.actual_time = time.time() - task.created_at
        return {
            "success": True,
            "summary": task.result,
            "error": None,
            "turns": 1,
        }

    # ------------------------------------------------------------------
    # LEGACY EXECUTION PATH (kept for callers that still use it)
    # ------------------------------------------------------------------

    async def execute_plan(
        self,
        plan: Plan,
        executor: Callable,
        progress_callback: Optional[Callable] = None,
        max_steps: int = 10_000,
    ) -> Dict[str, Any]:
        if not plan:
            raise PlanningError("No plan to execute")

        start_time = time.time()
        self.current_plan = plan
        results: Dict[str, Any] = {}
        failed_tasks: List[Task] = []

        try:
            steps = 0
            while not plan.is_complete():
                steps += 1
                if steps > max_steps:
                    logger.error("Plan execution exceeded %d steps; aborting", max_steps)
                    break

                ready_tasks = plan.get_pending_tasks()
                if not ready_tasks:
                    remaining = [t for t in plan._all_tasks() if t.status == TaskStatus.PENDING]
                    if not remaining:
                        break
                    logger.warning(
                        "Deadlock: %d pending tasks with unmet dependencies",
                        len(remaining),
                    )
                    self._resolve_deadlock(plan)
                    if not plan.get_pending_tasks():
                        for t in remaining:
                            t.status = TaskStatus.SKIPPED
                            t.error = "skipped due to unresolved dependency"
                        break
                    continue

                if self.enable_parallel and len(ready_tasks) > 1:
                    await self._execute_parallel_tasks(
                        ready_tasks[: self.max_parallel_tasks],
                        executor,
                        results,
                        progress_callback,
                    )
                else:
                    await self._execute_single_task(
                        ready_tasks[0], executor, results, progress_callback
                    )

                if progress_callback:
                    await progress_callback(plan.get_completion_percentage(), plan)

            execution_time = time.time() - start_time
            all_tasks = plan._all_tasks()
            success = all(t.status == TaskStatus.COMPLETED for t in all_tasks)
            failed_tasks = [t for t in all_tasks if t.status == TaskStatus.FAILED]

            plan.metrics.update(
                {
                    "execution_time": execution_time,
                    "success": success,
                    "tasks_completed": len(
                        [t for t in all_tasks if t.status == TaskStatus.COMPLETED]
                    ),
                    "tasks_failed": len(failed_tasks),
                    "success_rate": (
                        sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
                        / len(all_tasks)
                        if all_tasks
                        else 0.0
                    ),
                }
            )
            if success:
                plan.status = "completed"
                plan.completed_at = time.time()

            logger.info(
                "Plan execution %s in %.2fs",
                "succeeded" if success else "failed",
                execution_time,
            )
            return {
                "success": success,
                "plan_id": plan.id,
                "metrics": plan.metrics,
                "results": results,
                "failed_tasks": [t.to_dict() for t in failed_tasks],
            }
        except Exception as e:
            logger.error("Plan execution failed: %s", e, exc_info=True)
            raise ExecutionError(f"Failed to execute plan: {e}")

    async def _execute_single_task(
        self,
        task: Task,
        executor: Callable,
        results: Dict[str, Any],
        progress_callback: Optional[Callable],
    ) -> None:
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()

        try:
            if task.subtasks:
                subtask_results = []
                for subtask in task.subtasks:
                    await self._execute_single_task(subtask, executor, results, progress_callback)
                    subtask_results.append(subtask.result)
                task.result = subtask_results
            else:
                task.result = await executor(task)

            task.status = TaskStatus.COMPLETED
            task.actual_time = time.time() - task.created_at
            results[task.id] = {
                "success": True,
                "result": task.result,
                "time": task.actual_time,
            }
            logger.debug("Task %s completed", task.id)
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRY
                logger.warning(
                    "Task %s failed, retrying (%d/%d)",
                    task.id,
                    task.retry_count,
                    task.max_retries,
                )
                await asyncio.sleep(min(2 ** task.retry_count, 30))
                await self._execute_single_task(task, executor, results, progress_callback)
            else:
                task.status = TaskStatus.FAILED
                results[task.id] = {
                    "success": False,
                    "error": task.error,
                    "retries": task.retry_count,
                }
                logger.error(
                    "Task %s failed after %d retries: %s",
                    task.id,
                    task.retry_count,
                    task.error,
                )

    async def _execute_parallel_tasks(
        self,
        tasks: List[Task],
        executor: Callable,
        results: Dict[str, Any],
        progress_callback: Optional[Callable],
    ) -> None:
        await asyncio.gather(
            *[
                self._execute_single_task(t, executor, results, progress_callback)
                for t in tasks
            ],
            return_exceptions=True,
        )

    def _resolve_deadlock(self, plan: Plan) -> bool:
        all_tasks = plan._all_tasks()
        by_id = {t.id: t for t in all_tasks}
        changed = False
        for task in all_tasks:
            if task.status != TaskStatus.PENDING:
                continue
            unresolved = [
                dep
                for dep in task.dependencies
                if dep not in by_id
                or by_id[dep].status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            ]
            if unresolved:
                task.dependencies = [d for d in task.dependencies if d not in unresolved]
                changed = True
                logger.warning(
                    "Removed unresolvable deps %s from task %s", unresolved, task.id
                )
        return changed

    async def re_plan(
        self,
        plan: Plan,
        failed_tasks: List[Task],
        new_context: Optional[Dict[str, Any]] = None,
    ) -> Plan:
        self.metrics["replans"] += 1
        failure_analysis = self._analyze_failures(failed_tasks)
        remaining_goal = f"{plan.goal} - Adjust based on failures: {failure_analysis}"

        context = {
            "original_plan": plan.to_dict(),
            "failed_tasks": [t.to_dict() for t in failed_tasks],
            "failure_analysis": failure_analysis,
            "completed_tasks": [
                t.to_dict() for t in plan.tasks if t.status == TaskStatus.COMPLETED
            ],
            "new_context": new_context or {},
        }
        new_plan = await self.create_plan(remaining_goal, context)
        new_plan.metadata["original_plan_id"] = plan.id
        new_plan.metadata["replan_reason"] = failure_analysis
        plan.status = "superseded"
        logger.info("Replanned with %d tasks", len(new_plan.tasks))
        return new_plan

    def _analyze_failures(self, failed_tasks: List[Task]) -> str:
        if not failed_tasks:
            return "No failures to analyze"
        failure_types: Dict[str, int] = defaultdict(int)
        details = []
        for task in failed_tasks:
            err = str(task.error).lower()
            if "timeout" in err:
                failure_types["timeout"] += 1
            elif "permission" in err:
                failure_types["permission"] += 1
            elif "tool" in err:
                failure_types["tool_error"] += 1
            else:
                failure_types["unknown"] += 1
            details.append(f"Task {task.id}: {task.error}")
        analysis = f"Failed {len(failed_tasks)} tasks: "
        analysis += ", ".join(f"{k}: {v}" for k, v in failure_types.items())
        analysis += ". Details: " + "; ".join(details[:3])
        return analysis

    async def optimize_plan(self, plan: Plan) -> Plan:
        parallel_groups = self._find_parallel_groups(plan)
        optimized_tasks = self._optimize_task_structure(plan.tasks)
        return Plan(
            goal=plan.goal,
            tasks=optimized_tasks,
            context=plan.context,
            metrics={
                "original_tasks": len(plan.tasks),
                "optimized_tasks": len(optimized_tasks),
                "parallel_groups": len(parallel_groups),
            },
        )

    def _find_parallel_groups(self, plan: Plan) -> List[List[str]]:
        graph: Dict[str, Set[str]] = defaultdict(set)
        for task in plan._all_tasks():
            for dep in task.dependencies:
                graph[dep].add(task.id)
                graph[task.id].add(dep)

        groups: List[List[str]] = []
        remaining = {t.id for t in plan._all_tasks()}
        by_id = {t.id: t for t in plan._all_tasks()}

        while remaining:
            first_id = next(iter(remaining))
            group = [first_id]
            remaining.discard(first_id)
            for tid in list(remaining):
                task = by_id[tid]
                if any(dep in group for dep in task.dependencies):
                    continue
                if any(tid in graph[g] for g in group):
                    continue
                group.append(tid)
                remaining.discard(tid)
            groups.append(group)
        return groups

    def _optimize_task_structure(self, tasks: List[Task]) -> List[Task]:
        merged: List[Task] = []
        i = 0
        while i < len(tasks):
            if (
                i + 1 < len(tasks)
                and (tasks[i].estimated_time or 0) < 10
                and tasks[i].status == TaskStatus.PENDING
                and tasks[i + 1].status == TaskStatus.PENDING
            ):
                merged.append(
                    Task(
                        description=f"{tasks[i].description} then {tasks[i + 1].description}",
                        dependencies=list(
                            dict.fromkeys(tasks[i].dependencies + tasks[i + 1].dependencies)
                        ),
                        priority=min(tasks[i].priority.value, tasks[i + 1].priority.value),
                        estimated_time=(tasks[i].estimated_time or 0)
                        + (tasks[i + 1].estimated_time or 0),
                    )
                )
                i += 2
            else:
                merged.append(tasks[i])
                i += 1
        return merged

    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        if self.current_plan and self.current_plan.id == plan_id:
            return self._get_plan_status_dict(self.current_plan)
        for plan in self.plan_history:
            if plan.id == plan_id:
                return self._get_plan_status_dict(plan)
        return None

    def _get_plan_status_dict(self, plan: Plan) -> Dict[str, Any]:
        return {
            "id": plan.id,
            "goal": plan.goal,
            "status": plan.status,
            "completion": plan.get_completion_percentage(),
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "priority": t.priority.name,
                    "time": t.actual_time or t.estimated_time,
                }
                for t in plan.tasks
            ],
            "metrics": plan.metrics,
        }

    async def save_plan(self, plan: Plan, storage: Any) -> None:
        await storage.save_plan(plan.id, plan.to_dict())

    async def load_plan(self, plan_id: str, storage: Any) -> Optional[Plan]:
        data = await storage.load_plan(plan_id)
        return Plan.from_dict(data) if data else None