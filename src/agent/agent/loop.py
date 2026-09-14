"""Core agent execution loop.

The loop follows the canonical agent cycle:
LLM -> tool call -> permission/execution -> tool result -> LLM.
Tool calls and their results are persisted in the same message history that is
sent to the next model request. Execution telemetry remains separate from the
conversation messages.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.agent.planner import Plan, Planner, Task, TaskStatus
from agent.llm.provider import LLMProvider, Message, ToolCall
from agent.tools.registry import ToolRegistry
from agent.utils.errors import TimeoutError
from agent.utils.logging import get_logger

logger = get_logger(__name__)


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
    """Runtime telemetry plus the single canonical model message history."""

    iteration: int = 0
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_action_time: float = field(default_factory=time.time)
    tokens_used: int = 0
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.metadata["last_message_time"] = time.time()

    def add_observation(self, observation: Dict[str, Any]) -> None:
        self.observations.append(observation)
        self.metadata["last_observation_time"] = time.time()

    def add_action(self, action: Dict[str, Any]) -> None:
        self.actions_taken.append(action)
        self.last_action_time = time.time()
        self.metadata["last_action_time"] = self.last_action_time

    def get_recent_messages(self, n: int = 10) -> List[Message]:
        return self.messages[-n:]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_messages": len(self.messages),
            "total_actions": len(self.actions_taken),
            "total_observations": len(self.observations),
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "errors": len(self.errors),
            "duration": time.time() - self.start_time,
        }


class AgentLoop:
    """Thinking -> acting -> observing loop with reliable tool feedback."""

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
        self.max_history_length = config.get("max_history_length", 20)
        self.enable_planning = config.get("enable_planning", True)
        self.enable_caching = config.get("enable_caching", True)
        self.default_timeout = config.get("timeout", 60)

        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan: Optional[Plan] = None
        self.pending_tool_calls: List[ToolCall] = []
        self.completed_tool_calls: List[ToolCall] = []

        self.performance_history: deque = deque(maxlen=100)
        self.tool_execution_times: Dict[str, List[float]] = {}
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.tool_result_cache: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.should_stop = False
        self.is_paused = False

        logger.info("Agent loop initialized")

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        tool_outputs: Optional[List[Dict[str, Any]]] = None,
        max_turns: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            self.state = LoopState.INITIALIZING
            self.should_stop = False
            self.is_paused = False
            if max_turns:
                self.max_iterations = max_turns

            await self._prepare_context(query, context, tool_outputs)

            if self.enable_planning and await self._should_plan(query):
                await self._create_plan(query, context)

            self.state = LoopState.THINKING
            while not self.should_stop and self.context.iteration < self.max_iterations:
                try:
                    while self.is_paused:
                        await asyncio.sleep(0.1)

                    self.context.iteration += 1
                    self.state = LoopState.THINKING
                    thought = await self._think()

                    if thought.get("stop"):
                        break

                    if thought.get("need_plan"):
                        await self._create_plan(query, context)
                        continue

                    tool_calls = thought.get("tool_calls", [])
                    if not tool_calls:
                        # Persist normal assistant output too. This makes the
                        # conversation history complete for later turns.
                        self._record_assistant_response(thought.get("response"), [])
                        return {
                            "success": True,
                            "response": thought.get("response", ""),
                            "iteration": self.context.iteration,
                            "context": self.context.get_summary(),
                        }

                    self.state = LoopState.ACTING
                    results = await self._act(tool_calls)
                    self.state = LoopState.OBSERVING
                    await self._observe(results)
                    self.state = LoopState.EVALUATING

                    if not await self._evaluate(results):
                        break

                except TimeoutError as exc:
                    logger.warning("Loop iteration timed out: %s", exc)
                    await self._handle_timeout()
                except Exception as exc:
                    logger.error("Loop iteration failed: %s", exc, exc_info=True)
                    await self._handle_error(exc)
                    if self.context.iteration >= self.max_iterations:
                        break
                    if len(self.context.errors) > 5:
                        break

            self.state = LoopState.STOPPED
            final_response = await self._generate_final_response()
            self._update_metrics()

            return {
                "success": True,
                "response": final_response,
                "iteration": self.context.iteration,
                "tool_calls": len(self.completed_tool_calls),
                "context": self.context.get_summary(),
                "plan": self.current_plan.to_dict() if self.current_plan else None,
            }
        except Exception as exc:
            self.state = LoopState.ERROR
            logger.error("Loop failed: %s", exc, exc_info=True)
            return {"success": False, "error": str(exc), "iteration": self.context.iteration}

    async def _prepare_context(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        tool_outputs: Optional[List[Dict[str, Any]]],
    ) -> None:
        # System must precede user content in the canonical history.
        self.context.add_message(Message(role="system", content=await self._get_system_prompt()))
        self.context.add_message(Message(role="user", content=query))

        if tool_outputs:
            for output in tool_outputs:
                self.context.add_message(
                    Message(role="tool", content=json.dumps(output, default=str))
                )

        self.context.metadata.update(
            {"query": query, "start_time": time.time(), "context": context or {}}
        )

    async def _get_system_prompt(self) -> str:
        tools_description = self._get_tools_description()
        project_context = await self._get_project_context()
        permissions_context = self._get_permissions_context()
        return f"""
You are an advanced AI CLI agent.

Available tools:
{tools_description}

Project Context:
{json.dumps(project_context, indent=2, default=str)}

Permissions:
{json.dumps(permissions_context, indent=2, default=str)}

Rules:
1. Use tools when necessary.
2. Inspect the result of every tool call before deciding the next action.
3. Never claim an action succeeded unless the tool result confirms it.
4. Respect permissions and security constraints.
5. Continue until the user's task is complete or clarification is required.
""".strip()

    def _get_tools_description(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}\n  Parameters: {json.dumps(tool.parameters, indent=2)}"
            for name, tool in self.tool_registry.tools.items()
        )

    async def _get_project_context(self) -> Dict[str, Any]:
        if self.agent and hasattr(self.agent, "workspace"):
            return {
                "project_path": str(self.agent.workspace.project_dir),
                "files": await self.agent.workspace.list_files(max_files=20),
                "git_info": (
                    await self.agent.workspace.get_git_info()
                    if hasattr(self.agent.workspace, "get_git_info")
                    else None
                ),
            }
        return {}

    def _get_permissions_context(self) -> Dict[str, Any]:
        if self.agent and hasattr(self.agent, "permission_manager"):
            return {
                "enabled": self.agent.permission_manager.enabled,
                "auto_approve": self.agent.permission_manager.auto_approve,
            }
        return {"enabled": True, "auto_approve": False}

    async def _should_plan(self, query: str) -> bool:
        q = query.lower()
        return any(
            (
                len(query.split()) > 20,
                "plan" in q,
                "steps" in q,
                "multiple" in q,
                "need to" in q,
                "several" in q,
            )
        )

    async def _create_plan(self, query: str, context: Optional[Dict[str, Any]]) -> None:
        """Create a plan as guidance; do not execute arbitrary planner tool calls.

        The LLM remains responsible for inspecting the workspace, selecting the
        actual tool calls, observing their results, and deciding when each task
        is complete. This prevents the planner from bypassing the normal
        permission -> execution -> tool-result feedback path.
        """
        try:
            self.state = LoopState.PLANNING
            self.current_plan = await self.planner.create_plan(
                goal=query,
                context=context or {},
                constraints=self._get_plan_constraints(),
            )
            self.context.metadata["plan"] = self.current_plan.to_dict()
            self.context.add_message(
                Message(
                    role="system",
                    content=(
                        "Execution plan created. Treat it as guidance. "
                        "Complete tasks through normal tool calls and verify results.\n"
                        + json.dumps(self.current_plan.to_dict(), indent=2, default=str)
                    ),
                )
            )
            logger.info("Created plan with %d tasks", len(self.current_plan.tasks))
        except Exception as exc:
            logger.error("Planning failed: %s", exc)
            self.current_plan = None

    def _get_plan_constraints(self) -> Dict[str, Any]:
        return {
            "max_tasks": 20,
            "timeout": self.default_timeout,
            "available_tools": list(self.tool_registry.tools.keys()),
        }

    async def _think(self) -> Dict[str, Any]:
        messages = list(self.context.get_recent_messages(self.max_history_length))
        state_context = await self._get_state_context()
        messages.append(
            Message(role="system", content=f"Current state: {json.dumps(state_context, default=str)}")
        )
        if self.current_plan:
            messages.append(
                Message(
                    role="system",
                    content=f"Plan context: {json.dumps(self._get_plan_context(), default=str)}",
                )
            )

        response = await self.llm.complete_with_tools(
            messages=messages,
            tools=self._get_available_tools(),
            temperature=0.7,
            max_tokens=2000,
        )

        usage = getattr(response, "usage", {}) or {}
        if hasattr(usage, "total_tokens"):
            self.context.tokens_used += usage.total_tokens
        elif isinstance(usage, dict):
            self.context.tokens_used += usage.get("total_tokens", 0)

        tool_calls = list(getattr(response, "tool_calls", []) or [])
        content = getattr(response, "content", None)

        # Critical invariant: every model response that contains tool calls is
        # persisted before execution, so the following tool messages can be
        # correlated to the exact assistant tool-call message.
        self._record_assistant_response(content, tool_calls)
        self.context.tool_calls.extend(tool_calls)

        return {
            "response": content,
            "tool_calls": tool_calls,
            "stop": getattr(response, "finish_reason", "stop") == "stop" and not tool_calls,
        }

    def _record_assistant_response(
        self, content: Optional[str], tool_calls: List[ToolCall]
    ) -> None:
        payload = []
        for call in tool_calls:
            payload.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, default=str),
                    },
                }
            )
        self.context.add_message(
            Message(
                role="assistant",
                content=content or "",
                tool_calls=payload or None,
            )
        )

    def _record_tool_result(
        self, tool_call: ToolCall, result: Dict[str, Any]
    ) -> None:
        # This is the missing feedback edge in the old implementation.
        self.context.add_message(
            Message(
                role="tool",
                name=tool_call.name,
                tool_call_id=tool_call.id,
                content=json.dumps(result, default=str),
            )
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
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "completed_tool_calls": len(self.completed_tool_calls),
            "pending_tool_calls": len(self.pending_tool_calls),
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
                t.to_dict() for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED
            ],
        }

    def _get_cache_key(self, messages: List[Message]) -> str:
        return str(hash("|".join(f"{m.role}:{m.content[:200]}" for m in messages)))

    async def _act(self, tool_calls: List[ToolCall]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for tool_call in list(tool_calls)[: self.max_tool_calls_per_iteration]:
            try:
                cache_key = f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True, default=str)}"
                if self.enable_caching and cache_key in self.tool_result_cache:
                    cached = self.tool_result_cache[cache_key]
                    if time.time() - cached.get("timestamp", 0) < 60:
                        result = cached["result"]
                        item = {"tool": tool_call.name, "result": result, "cached": True}
                        results.append(item)
                        self._record_tool_result(tool_call, result)
                        continue

                start = time.time()
                result = await self.agent.execute_tool(
                    tool_name=tool_call.name,
                    params=tool_call.arguments,
                )
                elapsed = time.time() - start
                if not isinstance(result, dict):
                    result = {"success": True, "result": result}

                self.tool_execution_times.setdefault(tool_call.name, []).append(elapsed)
                self.context.add_action(
                    {
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "params": tool_call.arguments,
                        "result": result,
                        "time": elapsed,
                    }
                )
                if self.enable_caching and result.get("success", False):
                    self.tool_result_cache[cache_key] = {
                        "result": result,
                        "timestamp": time.time(),
                    }

                self.completed_tool_calls.append(tool_call)
                self._record_tool_result(tool_call, result)
                results.append(
                    {
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "result": result,
                        "execution_time": elapsed,
                    }
                )
            except Exception as exc:
                logger.error("Tool execution failed: %s", exc, exc_info=True)
                error_result = {"success": False, "error": str(exc)}
                self._record_tool_result(tool_call, error_result)
                self.pending_tool_calls.append(tool_call)
                results.append(
                    {
                        "tool": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "result": error_result,
                        "error": str(exc),
                        "success": False,
                    }
                )
        return results

    async def _observe(self, results: List[Dict[str, Any]]) -> None:
        for item in results:
            tool_result = item.get("result") or {}
            observation = {
                "timestamp": time.time(),
                "tool": item.get("tool"),
                "tool_call_id": item.get("tool_call_id"),
                "success": bool(tool_result.get("success", False)),
                "error": item.get("error") or tool_result.get("error"),
                "execution_time": item.get("execution_time", 0),
                "result": tool_result,
            }
            self.context.add_observation(observation)

    async def _evaluate(self, results: List[Dict[str, Any]]) -> bool:
        failures = [r for r in results if not (r.get("result") or {}).get("success", False)]
        if failures and len(failures) == len(results):
            logger.warning("All tools failed; giving the model another turn to recover")
            return self.context.iteration < self.max_iterations
        if self.context.iteration >= self.max_iterations:
            return False
        # A plan is guidance, not an execution shortcut. The model decides when
        # the user's actual goal is complete after seeing the tool results.
        if self._has_final_answer():
            return False
        return True

    def _has_final_answer(self) -> bool:
        return any(
            action.get("result", {}).get("final_answer", False)
            for action in self.context.actions_taken
        )

    async def _execute_plan_task(self, task: Task) -> None:
        """Mark a task as active without bypassing the normal tool loop.

        Kept for compatibility with callers. Actual work must go through the
        model-generated tool calls in _act().
        """
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()
        self.context.metadata["active_plan_task"] = task.id

    async def _handle_timeout(self) -> None:
        self.context.errors.append("Timeout")
        self.context.add_message(
            Message(role="system", content="A tool/model step timed out. Reassess and retry safely.")
        )

    async def _handle_error(self, error: Exception) -> None:
        self.context.errors.append(str(error))
        self.context.add_message(
            Message(role="system", content=f"Execution error: {error}. Reassess using available tool results.")
        )

    async def _generate_final_response(self) -> str:
        if self.context.messages:
            for message in reversed(self.context.messages):
                if message.role == "assistant" and message.content and not message.tool_calls:
                    return message.content

        messages = list(self.context.get_recent_messages(10))
        messages.append(
            Message(role="user", content="Summarize what was accomplished and provide the final response.")
        )
        response = await self.llm.complete(messages)
        content = getattr(response, "content", "")
        self.context.add_message(Message(role="assistant", content=content))
        return content

    def _update_metrics(self) -> None:
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "duration": time.time() - self.context.start_time,
                "iterations": self.context.iteration,
                "tool_calls": len(self.completed_tool_calls),
                "tokens": self.context.tokens_used,
                "errors": len(self.context.errors),
            }
        )
        if self.current_plan:
            self.planner.metrics["last_plan_execution"] = {
                "duration": time.time() - self.context.start_time,
                "tasks_completed": sum(
                    1 for task in self.current_plan.tasks if task.status == TaskStatus.COMPLETED
                ),
                "success": self.current_plan.is_complete(),
            }

    async def reset(self) -> None:
        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan = None
        self.pending_tool_calls = []
        self.completed_tool_calls = []
        self.should_stop = False
        self.is_paused = False

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
            "plan_completion": self.current_plan.get_completion_percentage() if self.current_plan else 0,
            "tool_calls_completed": len(self.completed_tool_calls),
            "tool_calls_pending": len(self.pending_tool_calls),
            "actions_taken": len(self.context.actions_taken),
            "observations": len(self.context.observations),
            "errors": len(self.context.errors),
            "duration": time.time() - self.context.start_time,
        }

    def add_event_handler(self, event: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self.event_handlers.setdefault(event, []).append(handler)

    async def _trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        for handler in self.event_handlers.get(event, []):
            try:
                await handler(data)
            except Exception as exc:
                logger.error("Event handler failed: %s", exc)
