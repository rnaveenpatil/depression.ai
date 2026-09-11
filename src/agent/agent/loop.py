"""
Agent Loop Module - Core Execution Loop
Handles the thinking-acting-observing cycle with advanced features
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from agent.utils.logging import get_logger
from agent.utils.errors import LoopError, TimeoutError
from agent.llm.provider import LLMProvider, Message, ToolCall
from agent.tools.registry import ToolRegistry
from agent.agent.planner import Planner, Plan, Task

logger = get_logger(__name__)

class LoopState(Enum):
    """States of the agent loop"""
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
    """Context maintained across loop iterations"""
    iteration: int = 0
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: List[Dict] = field(default_factory=list)
    actions_taken: List[Dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_action_time: float = field(default_factory=time.time)
    tokens_used: int = 0
    cost: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the context"""
        self.messages.append(message)
        self.metadata['last_message_time'] = time.time()
    
    def add_observation(self, observation: Dict) -> None:
        """Add an observation"""
        self.observations.append(observation)
        self.metadata['last_observation_time'] = time.time()
    
    def add_action(self, action: Dict) -> None:
        """Add an action"""
        self.actions_taken.append(action)
        self.metadata['last_action_time'] = time.time()
        self.last_action_time = time.time()
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get the n most recent messages"""
        return self.messages[-n:]
    
    def get_summary(self) -> Dict:
        """Get a summary of the loop context"""
        return {
            'iteration': self.iteration,
            'total_messages': len(self.messages),
            'total_actions': len(self.actions_taken),
            'total_observations': len(self.observations),
            'tokens_used': self.tokens_used,
            'cost': self.cost,
            'errors': len(self.errors),
            'duration': time.time() - self.start_time
        }

class AgentLoop:
    """
    Advanced Agent Execution Loop with:
    - Thinking-Acting-Observing cycle
    - Tool calling and orchestration
    - Plan execution and monitoring
    - Adaptive behavior
    - Error recovery
    - Performance optimization
    """
    
    def __init__(
        self,
        agent: Any,  # Avoid circular import
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        planner: Planner,
        config: Dict[str, Any]
    ):
        self.agent = agent
        self.llm = llm
        self.tool_registry = tool_registry
        self.planner = planner
        
        # Configuration
        self.config = config
        self.max_iterations = config.get('max_iterations', 50)
        self.max_tool_calls_per_iteration = config.get('max_tool_calls', 5)
        self.max_history_length = config.get('max_history_length', 20)
        self.enable_planning = config.get('enable_planning', True)
        self.enable_caching = config.get('enable_caching', True)
        self.default_timeout = config.get('timeout', 60)
        
        # State
        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan: Optional[Plan] = None
        self.pending_tool_calls: List[ToolCall] = []
        self.completed_tool_calls: List[ToolCall] = []
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        self.tool_execution_times: Dict[str, List[float]] = {}
        
        # Caching
        self.response_cache: Dict[str, Dict] = {}
        self.tool_result_cache: Dict[str, Dict] = {}
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Control flags
        self.should_stop = False
        self.is_paused = False
        
        logger.info("Agent loop initialized")
    
    async def run(
        self,
        query: str,
        context: Dict[str, Any] = None,
        tool_outputs: List[Dict] = None,
        max_turns: int = None
    ) -> Dict[str, Any]:
        """
        Run the agent loop for a given query
        
        Args:
            query: User query
            context: Additional context
            tool_outputs: Previous tool outputs
            max_turns: Maximum turns for this run
            
        Returns:
            Result dictionary with response and metadata
        """
        try:
            # Initialize
            self.state = LoopState.INITIALIZING
            self.should_stop = False
            self.is_paused = False
            
            if max_turns:
                self.max_iterations = max_turns
            
            # Prepare context
            await self._prepare_context(query, context, tool_outputs)
            
            # Determine if planning is needed
            if self.enable_planning and await self._should_plan(query):
                self.state = LoopState.PLANNING
                await self._create_plan(query, context)
            
            # Main loop
            self.state = LoopState.THINKING
            
            while not self.should_stop and self.context.iteration < self.max_iterations:
                try:
                    # Check for pause
                    while self.is_paused:
                        await asyncio.sleep(0.1)
                    
                    # Increment iteration
                    self.context.iteration += 1
                    
                    # THINK: Generate next action
                    self.state = LoopState.THINKING
                    thought_result = await self._think()
                    
                    # Check if we're done
                    if thought_result.get('stop', False):
                        break
                    
                    # Check if we need to plan
                    if thought_result.get('need_plan', False):
                        self.state = LoopState.PLANNING
                        await self._create_plan(query, context)
                        continue
                    
                    # Get tool calls from thought
                    tool_calls = thought_result.get('tool_calls', [])
                    
                    if not tool_calls:
                        # No tool calls, just respond
                        return {
                            'success': True,
                            'response': thought_result.get('response', ''),
                            'iteration': self.context.iteration,
                            'context': self.context.get_summary()
                        }
                    
                    # ACT: Execute tool calls
                    self.state = LoopState.ACTING
                    results = await self._act(tool_calls)
                    
                    # OBSERVE: Process results
                    self.state = LoopState.OBSERVING
                    await self._observe(results)
                    
                    # EVALUATE: Check if we should continue
                    self.state = LoopState.EVALUATING
                    should_continue = await self._evaluate(results)
                    
                    if not should_continue:
                        break
                    
                except TimeoutError as e:
                    logger.warning(f"Loop iteration timed out: {e}")
                    await self._handle_timeout()
                    
                except Exception as e:
                    logger.error(f"Loop iteration failed: {e}", exc_info=True)
                    await self._handle_error(e)
                    
                    # Check if we should continue
                    if self.context.iteration >= self.max_iterations:
                        break
                    if len(self.context.errors) > 5:
                        break
            
            # Finalize
            self.state = LoopState.STOPPED
            
            # Generate final response
            final_response = await self._generate_final_response()
            
            # Update metrics
            self._update_metrics()
            
            return {
                'success': True,
                'response': final_response,
                'iteration': self.context.iteration,
                'tool_calls': len(self.completed_tool_calls),
                'context': self.context.get_summary(),
                'plan': self.current_plan.to_dict() if self.current_plan else None
            }
            
        except Exception as e:
            self.state = LoopState.ERROR
            logger.error(f"Loop failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'iteration': self.context.iteration
            }
    
    async def _prepare_context(
        self,
        query: str,
        context: Dict,
        tool_outputs: List[Dict]
    ) -> None:
        """Prepare the loop context"""
        # Add user query
        self.context.add_message(Message(
            role="user",
            content=query
        ))
        
        # Add system context
        system_prompt = await self._get_system_prompt()
        self.context.add_message(Message(
            role="system",
            content=system_prompt
        ))
        
        # Add previous tool outputs
        if tool_outputs:
            for output in tool_outputs:
                self.context.add_message(Message(
                    role="assistant",
                    content=f"Tool output: {json.dumps(output, default=str)}"
                ))
        
        # Set metadata
        self.context.metadata.update({
            'query': query,
            'start_time': time.time(),
            'context': context or {}
        })
    
    async def _get_system_prompt(self) -> str:
        """Get the system prompt with full context"""
        # Get available tools
        tools_description = self._get_tools_description()
        
        # Get project context
        project_context = await self._get_project_context()
        
        # Get permissions context
        permissions_context = self._get_permissions_context()
        
        return f"""
        You are an advanced AI CLI agent with the following capabilities:
        
        {tools_description}
        
        Project Context:
        {json.dumps(project_context, indent=2)}
        
        Permissions:
        {json.dumps(permissions_context, indent=2)}
        
        Follow these guidelines:
        1. Think step by step and plan your actions
        2. Use tools when necessary
        3. Be concise and clear in your responses
        4. Ask for clarification when needed
        5. Consider security and permissions
        6. Provide explanations for your actions
        """
    
    def _get_tools_description(self) -> str:
        """Get a description of all available tools"""
        tools = []
        for name, tool in self.tool_registry.tools.items():
            tools.append(
                f"- {name}: {tool.description}\n"
                f"  Parameters: {json.dumps(tool.parameters, indent=2)}"
            )
        return "\n".join(tools)
    
    async def _get_project_context(self) -> Dict:
        """Get the project context"""
        if self.agent and hasattr(self.agent, 'workspace'):
            return {
                'project_path': str(self.agent.workspace.project_dir),
                'files': await self.agent.workspace.list_files(max_files=20),
                'git_info': await self.agent.workspace.get_git_info() if hasattr(self.agent.workspace, 'get_git_info') else None
            }
        return {}
    
    def _get_permissions_context(self) -> Dict:
        """Get the permissions context"""
        if self.agent and hasattr(self.agent, 'permission_manager'):
            return {
                'enabled': self.agent.permission_manager.enabled,
                'auto_approve': self.agent.permission_manager.auto_approve
            }
        return {'enabled': True, 'auto_approve': False}
    
    async def _should_plan(self, query: str) -> bool:
        """Determine if a task requires planning"""
        # Simple heuristic: check if query is complex
        complexity_indicators = [
            len(query.split()) > 20,
            'plan' in query.lower(),
            'steps' in query.lower(),
            'multiple' in query.lower(),
            'need to' in query.lower(),
            'several' in query.lower()
        ]
        
        return any(complexity_indicators)
    
    async def _create_plan(self, query: str, context: Dict) -> None:
        """Create a plan for the current task"""
        try:
            self.state = LoopState.PLANNING
            
            # Create plan
            plan = await self.planner.create_plan(
                goal=query,
                context=context or {},
                constraints=self._get_plan_constraints()
            )
            
            self.current_plan = plan
            
            # Add plan to context
            self.context.metadata['plan'] = plan.to_dict()
            
            logger.info(f"Created plan with {len(plan.tasks)} tasks")
            
            # Execute first task immediately
            if plan.tasks:
                first_task = plan.get_pending_tasks()[0] if plan.get_pending_tasks() else None
                if first_task:
                    await self._execute_plan_task(first_task)
                    
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Continue without plan
            self.current_plan = None
    
    def _get_plan_constraints(self) -> Dict:
        """Get constraints for planning"""
        return {
            'max_tasks': 20,
            'timeout': self.default_timeout,
            'available_tools': list(self.tool_registry.tools.keys())
        }
    
    async def _think(self) -> Dict[str, Any]:
        """Think and decide on the next action"""
        try:
            # Get conversation history
            messages = self.context.get_recent_messages(self.max_history_length)
            
            # Add current state context
            state_context = await self._get_state_context()
            messages.append(Message(
                role="system",
                content=f"Current state: {json.dumps(state_context, default=str)}"
            ))
            
            # Add plan context if available
            if self.current_plan:
                plan_context = self._get_plan_context()
                messages.append(Message(
                    role="system",
                    content=f"Plan context: {json.dumps(plan_context, default=str)}"
                ))
            
            # Get available tools for the LLM
            tools = self._get_available_tools()
            
            # Call LLM
            response = await self.llm.complete_with_tools(
                messages=messages,
                tools=tools,
                temperature=0.7,
                max_tokens=2000
            )
            
            # Update token usage
            if hasattr(response, 'usage'):
                usage = response.usage
                if hasattr(usage, 'total_tokens'):
                    self.context.tokens_used += usage.total_tokens
                elif isinstance(usage, dict):
                    self.context.tokens_used += usage.get('total_tokens', 0)
            
            # Process response
            result = {
                'response': response.content if hasattr(response, 'content') else None,
                'tool_calls': response.tool_calls if hasattr(response, 'tool_calls') else [],
                'stop': response.finish_reason == 'stop' if hasattr(response, 'finish_reason') else False
            }
            
            # Cache the result
            if self.enable_caching:
                cache_key = self._get_cache_key(messages)
                self.response_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Thinking failed: {e}")
            
            # If planning is enabled, try to recover
            if self.enable_planning and self.current_plan:
                return {
                    'response': "Let me continue with the plan",
                    'tool_calls': [],
                    'need_plan': False
                }
            
            raise
    
    def _get_available_tools(self) -> List[Dict]:
        """Get available tools for the LLM"""
        tools = []
        
        for name, tool in self.tool_registry.tools.items():
            tools.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.parameters
                }
            })
        
        return tools
    
    async def _get_state_context(self) -> Dict:
        """Get current state context"""
        return {
            'iteration': self.context.iteration,
            'max_iterations': self.max_iterations,
            'actions_taken': len(self.context.actions_taken),
            'observations': len(self.context.observations),
            'current_plan': self.current_plan.to_dict() if self.current_plan else None,
            'completed_tool_calls': len(self.completed_tool_calls),
            'pending_tool_calls': len(self.pending_tool_calls)
        }
    
    def _get_plan_context(self) -> Dict:
        """Get plan execution context"""
        if not self.current_plan:
            return {}
        
        return {
            'plan_id': self.current_plan.id,
            'goal': self.current_plan.goal,
            'completion': self.current_plan.get_completion_percentage(),
            'pending_tasks': [
                t.to_dict() for t in self.current_plan.get_pending_tasks()
            ],
            'completed_tasks': [
                t.to_dict() for t in self.current_plan.tasks
                if t.status == TaskStatus.COMPLETED
            ]
        }
    
    def _get_cache_key(self, messages: List[Message]) -> str:
        """Generate a cache key for the response"""
        # Simple hash of messages
        content = "|".join([f"{m.role}:{m.content[:100]}" for m in messages])
        return str(hash(content))
    
    async def _act(self, tool_calls: List[ToolCall]) -> List[Dict]:
        """Execute tool calls"""
        results = []
        
        # Limit tool calls
        tool_calls = tool_calls[:self.max_tool_calls_per_iteration]
        
        for tool_call in tool_calls:
            try:
                # Check cache first
                cache_key = f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
                if self.enable_caching and cache_key in self.tool_result_cache:
                    cache_result = self.tool_result_cache[cache_key]
                    # Check if cache is still valid (within 60 seconds)
                    if time.time() - cache_result.get('timestamp', 0) < 60:
                        results.append({
                            'tool': tool_call.name,
                            'result': cache_result['result'],
                            'cached': True
                        })
                        continue
                
                # Execute tool
                start_time = time.time()
                result = await self.agent.execute_tool(
                    tool_name=tool_call.name,
                    params=tool_call.arguments
                )
                execution_time = time.time() - start_time
                
                # Track performance
                if tool_call.name not in self.tool_execution_times:
                    self.tool_execution_times[tool_call.name] = []
                self.tool_execution_times[tool_call.name].append(execution_time)
                
                # Add to context
                self.context.add_action({
                    'tool': tool_call.name,
                    'params': tool_call.arguments,
                    'result': result,
                    'time': execution_time
                })
                
                # Cache result
                if self.enable_caching and result.get('success', False):
                    self.tool_result_cache[cache_key] = {
                        'result': result,
                        'timestamp': time.time()
                    }
                
                # Add to completed calls
                self.completed_tool_calls.append(tool_call)
                
                results.append({
                    'tool': tool_call.name,
                    'result': result,
                    'execution_time': execution_time
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append({
                    'tool': tool_call.name,
                    'error': str(e),
                    'success': False
                })
                
                # Add to pending for retry
                self.pending_tool_calls.append(tool_call)
        
        return results
    
    async def _observe(self, results: List[Dict]) -> None:
        """Process and observe tool execution results"""
        for result in results:
            observation = {
                'timestamp': time.time(),
                'tool': result.get('tool'),
                'success': result.get('result', {}).get('success', False),
                'error': result.get('error'),
                'execution_time': result.get('execution_time', 0)
            }
            
            if result.get('result'):
                observation['result'] = result['result']
            
            self.context.add_observation(observation)
    
    async def _evaluate(self, results: List[Dict]) -> bool:
        """Evaluate if the loop should continue"""
        # Check if any tool failed
        failures = [r for r in results if not r.get('result', {}).get('success', False)]
        
        if failures:
            # If all failed, maybe stop
            if len(failures) == len(results):
                logger.warning("All tools failed, stopping loop")
                return False
        
        # Check if we have reached iteration limit
        if self.context.iteration >= self.max_iterations:
            logger.info(f"Reached max iterations: {self.max_iterations}")
            return False
        
        # Check if plan is complete
        if self.current_plan and self.current_plan.is_complete():
            logger.info("Plan completed")
            return False
        
        # Check if we have a final answer
        if self._has_final_answer():
            return False
        
        return True
    
    def _has_final_answer(self) -> bool:
        """Check if we have a final answer"""
        # Check if any action produced a final answer
        for action in self.context.actions_taken:
            if action.get('result', {}).get('final_answer', False):
                return True
        
        return False
    
    async def _execute_plan_task(self, task: Task) -> None:
        """Execute a task from the plan"""
        try:
            # Update task status
            task.status = TaskStatus.IN_PROGRESS
            
            # Execute the task
            result = await self.agent.execute_tool(
                tool_name=task.tool_calls[0]['tool'] if task.tool_calls else 'execute',
                params={'task': task.description}
            )
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.result = result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            # Add to plan context
            if self.current_plan:
                self.current_plan.metrics['failed_tasks'] = \
                    self.current_plan.metrics.get('failed_tasks', 0) + 1
    
    async def _handle_timeout(self) -> None:
        """Handle timeout in the loop"""
        self.context.errors.append("Timeout")
        
        # Add a timeout message
        self.context.add_message(Message(
            role="system",
            content="Timeout occurred. Attempting to continue."
        ))
        
        # If we have a plan, move to next task
        if self.current_plan:
            next_tasks = self.current_plan.get_pending_tasks()
            if next_tasks:
                await self._execute_plan_task(next_tasks[0])
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors in the loop"""
        self.context.errors.append(str(error))
        
        # Add error to context
        self.context.add_message(Message(
            role="system",
            content=f"Error occurred: {str(error)}. Attempting to recover."
        ))
        
        # If we have a plan, continue with next task
        if self.current_plan:
            next_tasks = self.current_plan.get_pending_tasks()
            if next_tasks:
                await self._execute_plan_task(next_tasks[0])
    
    async def _generate_final_response(self) -> str:
        """Generate the final response"""
        # If we have a response from the last iteration, use it
        if self.context.messages:
            last_message = self.context.messages[-1]
            if last_message.role == 'assistant':
                return last_message.content
        
        # Otherwise, generate a summary
        summary_prompt = "Summarize what was accomplished and provide a final response."
        messages = self.context.get_recent_messages(10)
        messages.append(Message(role="user", content=summary_prompt))
        
        response = await self.llm.complete(messages)
        return response.content
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        self.performance_history.append({
            'timestamp': time.time(),
            'duration': time.time() - self.context.start_time,
            'iterations': self.context.iteration,
            'tool_calls': len(self.completed_tool_calls),
            'tokens': self.context.tokens_used,
            'errors': len(self.context.errors)
        })
        
        # Update planner metrics
        if self.current_plan:
            self.planner.metrics['last_plan_execution'] = {
                'duration': time.time() - self.context.start_time,
                'tasks_completed': len([t for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]),
                'success': self.current_plan.is_complete()
            }
    
    async def reset(self) -> None:
        """Reset the loop state"""
        self.state = LoopState.IDLE
        self.context = LoopContext()
        self.current_plan = None
        self.pending_tool_calls = []
        self.completed_tool_calls = []
        self.should_stop = False
        self.is_paused = False
        
        logger.info("Loop reset")
    
    def pause(self) -> None:
        """Pause the loop"""
        self.is_paused = True
        self.state = LoopState.PAUSED
        logger.info("Loop paused")
    
    def resume(self) -> None:
        """Resume the loop"""
        self.is_paused = False
        self.state = LoopState.THINKING
        logger.info("Loop resumed")
    
    def stop(self) -> None:
        """Stop the loop"""
        self.should_stop = True
        self.state = LoopState.STOPPED
        logger.info("Loop stopped")
    
    def get_loop_status(self) -> Dict:
        """Get the current loop status"""
        return {
            'state': self.state.value,
            'iteration': self.context.iteration,
            'max_iterations': self.max_iterations,
            'has_plan': self.current_plan is not None,
            'plan_completion': self.current_plan.get_completion_percentage() if self.current_plan else 0,
            'tool_calls_completed': len(self.completed_tool_calls),
            'tool_calls_pending': len(self.pending_tool_calls),
            'actions_taken': len(self.context.actions_taken),
            'observations': len(self.context.observations),
            'errors': len(self.context.errors),
            'duration': time.time() - self.context.start_time
        }
    
    def add_event_handler(self, event: str, handler: Callable[[Dict], Awaitable[None]]) -> None:
        """Add an event handler"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)
    
    async def _trigger_event(self, event: str, data: Dict) -> None:
        """Trigger an event"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")