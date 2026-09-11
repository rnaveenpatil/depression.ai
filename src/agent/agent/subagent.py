"""
SubAgent Module - Hierarchical Agent System
Manages sub-agents for parallel task execution and delegation
"""

import asyncio
import uuid
import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import queue

from agent.utils.logging import get_logger
from agent.utils.errors import SubAgentError, TimeoutError
from agent.agent.agent import Agent
from agent.llm.provider import LLMProvider, Message
from agent.tools.registry import ToolRegistry

logger = get_logger(__name__)

class SubAgentStatus(Enum):
    """Status of a sub-agent"""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    TERMINATED = "terminated"

class SubAgentType(Enum):
    """Types of sub-agents"""
    COORDINATOR = "coordinator"  # Manages other sub-agents
    EXECUTOR = "executor"       # Executes specific tasks
    MONITOR = "monitor"         # Monitors and reports
    VALIDATOR = "validator"     # Validates results
    SPECIALIST = "specialist"   # Domain specialist

@dataclass
class SubAgentTask:
    """Task assigned to a sub-agent"""
    id: str
    description: str
    assigned_at: float = field(default_factory=time.time)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubAgent:
    """Representation of a sub-agent"""
    id: str
    name: str
    type: SubAgentType
    parent_id: Optional[str]
    status: SubAgentStatus = SubAgentStatus.CREATED
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[SubAgentTask] = None
    completed_tasks: List[SubAgentTask] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    communication_channel: Optional[asyncio.Queue] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.communication_channel:
            self.communication_channel = asyncio.Queue()

class SubAgentManager:
    """
    Advanced SubAgent Management System with:
    - Hierarchical agent delegation
    - Parallel task execution
    - Inter-agent communication
    - Resource management
    - Fault tolerance
    """
    
    def __init__(
        self,
        agent: Agent,
        config: Dict[str, Any]
    ):
        self.agent = agent
        self.config = config
        
        # Sub-agent management
        self.subagents: Dict[str, SubAgent] = {}
        self.active_subagents: Dict[str, asyncio.Task] = {}
        self.agent_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
        # Configuration
        self.max_subagents = config.get('max_subagents', 10)
        self.max_depth = config.get('max_depth', 3)
        self.default_timeout = config.get('default_timeout', 60)
        self.enable_parallel = config.get('enable_parallel', True)
        self.enable_communication = config.get('enable_communication', True)
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.broadcast_channels: Dict[str, asyncio.Queue] = {}
        
        # Monitoring
        self.metrics = {
            'subagents_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_messages': 0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Resource pools
        self.resource_pools = {}
        
        logger.info("SubAgentManager initialized")
    
    async def create_subagent(
        self,
        name: str,
        task: str,
        capabilities: List[str],
        agent_type: SubAgentType = SubAgentType.EXECUTOR,
        parent_id: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> SubAgent:
        """
        Create and initialize a new sub-agent
        
        Args:
            name: Name of the sub-agent
            task: Description of the task
            capabilities: List of capabilities
            agent_type: Type of sub-agent
            parent_id: Parent agent ID if hierarchical
            config: Additional configuration
        
        Returns:
            The created SubAgent
        """
        if len(self.subagents) >= self.max_subagents:
            raise SubAgentError(f"Maximum sub-agents reached ({self.max_subagents})")
        
        # Check depth limit
        if parent_id:
            depth = self._get_depth(parent_id)
            if depth >= self.max_depth:
                raise SubAgentError(f"Maximum depth reached ({self.max_depth})")
        
        # Create sub-agent
        subagent = SubAgent(
            name=name,
            type=agent_type,
            parent_id=parent_id,
            capabilities=capabilities,
            context=config or {}
        )
        
        # Store sub-agent
        self.subagents[subagent.id] = subagent
        
        # Update hierarchy
        if parent_id:
            if parent_id not in self.agent_hierarchy:
                self.agent_hierarchy[parent_id] = []
            self.agent_hierarchy[parent_id].append(subagent.id)
        
        # Create broadcast channel
        self.broadcast_channels[subagent.id] = asyncio.Queue()
        
        # Initialize the sub-agent
        await self._initialize_subagent(subagent)
        
        # Assign initial task
        await self.assign_task(subagent.id, SubAgentTask(
            id=str(uuid.uuid4()),
            description=task
        ))
        
        self.metrics['subagents_created'] += 1
        
        logger.info(f"Created sub-agent {name} ({subagent.id}) of type {agent_type.value}")
        return subagent
    
    async def _initialize_subagent(self, subagent: SubAgent) -> None:
        """Initialize a sub-agent's internal state"""
        subagent.status = SubAgentStatus.INITIALIZING
        
        try:
            # Create an isolated context for the sub-agent
            subagent.context['created_at'] = time.time()
            subagent.context['capabilities'] = subagent.capabilities
            subagent.context['llm_config'] = self._get_subagent_llm_config(subagent)
            
            # Start the sub-agent's processing loop
            task = asyncio.create_task(self._subagent_loop(subagent))
            self.active_subagents[subagent.id] = task
            
            subagent.status = SubAgentStatus.READY
            await self._trigger_event('on_subagent_ready', {'subagent': subagent})
            
        except Exception as e:
            subagent.status = SubAgentStatus.FAILED
            logger.error(f"Failed to initialize sub-agent {subagent.id}: {e}")
            raise SubAgentError(f"Initialization failed: {e}")
    
    async def _subagent_loop(self, subagent: SubAgent) -> None:
        """
        Main processing loop for a sub-agent
        Handles task execution, communication, and monitoring
        """
        while subagent.status not in [
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.TERMINATED
        ]:
            try:
                # Check for new tasks
                if subagent.current_task:
                    await self._process_task(subagent)
                else:
                    # Wait for tasks or messages
                    try:
                        message = await asyncio.wait_for(
                            subagent.communication_channel.get(),
                            timeout=1.0
                        )
                        await self._handle_message(subagent, message)
                    except asyncio.TimeoutError:
                        # Periodic heartbeat
                        await self._heartbeat_check(subagent)
                        
            except asyncio.CancelledError:
                logger.warning(f"Sub-agent {subagent.id} loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in sub-agent loop {subagent.id}: {e}")
                await self._handle_loop_error(subagent, e)
    
    async def _process_task(self, subagent: SubAgent) -> None:
        """Process the current task of a sub-agent"""
        if not subagent.current_task:
            return
        
        subagent.status = SubAgentStatus.WORKING
        task = subagent.current_task
        
        try:
            # Process based on sub-agent type
            if subagent.type == SubAgentType.COORDINATOR:
                result = await self._process_coordinator_task(subagent, task)
            elif subagent.type == SubAgentType.EXECUTOR:
                result = await self._process_executor_task(subagent, task)
            elif subagent.type == SubAgentType.MONITOR:
                result = await self._process_monitor_task(subagent, task)
            elif subagent.type == SubAgentType.VALIDATOR:
                result = await self._process_validator_task(subagent, task)
            elif subagent.type == SubAgentType.SPECIALIST:
                result = await self._process_specialist_task(subagent, task)
            else:
                result = await self._process_generic_task(subagent, task)
            
            # Task completed
            task.status = "completed"
            task.result = result
            subagent.completed_tasks.append(task)
            subagent.current_task = None
            
            self.metrics['tasks_completed'] += 1
            
            # Update metrics
            subagent.metrics['last_success'] = time.time()
            subagent.metrics['tasks_completed'] = subagent.metrics.get('tasks_completed', 0) + 1
            
            await self._trigger_event('on_task_completed', {
                'subagent': subagent,
                'task': task,
                'result': result
            })
            
            logger.debug(f"Sub-agent {subagent.id} completed task: {task.description[:50]}...")
            
        except Exception as e:
            # Task failed
            task.status = "failed"
            task.error = str(e)
            subagent.current_task = None
            
            self.metrics['tasks_failed'] += 1
            subagent.metrics['last_failure'] = time.time()
            subagent.metrics['tasks_failed'] = subagent.metrics.get('tasks_failed', 0) + 1
            
            logger.error(f"Sub-agent {subagent.id} task failed: {e}")
            
            # Try to recover
            await self._handle_task_failure(subagent, task, e)
    
    async def _process_coordinator_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process a coordinator task - manages other sub-agents"""
        # Decompose task into sub-tasks
        subtasks = await self._decompose_task(task.description, subagent.context)
        
        # Create child sub-agents for each sub-task
        child_results = []
        for i, subtask_desc in enumerate(subtasks):
            child = await self.create_subagent(
                name=f"{subagent.name}_child_{i}",
                task=subtask_desc,
                capabilities=subagent.capabilities,
                agent_type=SubAgentType.EXECUTOR,
                parent_id=subagent.id
            )
            
            # Wait for child to complete
            result = await self.wait_for_subagent(child.id)
            child_results.append(result)
        
        return {
            'subtasks_completed': len(child_results),
            'results': child_results
        }
    
    async def _process_executor_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process an executor task - executes using tools"""
        # Use the agent's tools to execute
        tool_name = task.metadata.get('tool', 'execute')
        params = task.metadata.get('params', {})
        
        # Use LLM to determine the best approach
        llm_response = await self._get_executor_plan(task, subagent.context)
        
        # Execute the plan
        result = await self._execute_plan_with_tools(llm_response, subagent)
        
        return result
    
    async def _process_monitor_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process a monitor task - watches and reports"""
        # Monitor a specific resource or process
        target = task.metadata.get('target')
        duration = task.metadata.get('duration', 30)
        
        observations = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Observe the target
            observation = await self._observe_target(target, subagent)
            observations.append(observation)
            
            # Check for anomalies
            if self._detect_anomaly(observation):
                await self._trigger_event('on_anomaly_detected', {
                    'subagent': subagent,
                    'observation': observation
                })
            
            await asyncio.sleep(task.metadata.get('interval', 2))
        
        return {
            'duration': duration,
            'observations': observations,
            'anomalies_detected': len([o for o in observations if o.get('anomaly', False)])
        }
    
    async def _process_validator_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process a validator task - validates results"""
        target_result = task.metadata.get('target_result')
        validation_criteria = task.metadata.get('criteria', {})
        
        # Validate against criteria
        validations = []
        
        for criteria_key, expected in validation_criteria.items():
            actual = target_result.get(criteria_key)
            is_valid = await self._validate_value(
                criteria_key, actual, expected, subagent.context
            )
            validations.append({
                'criteria': criteria_key,
                'expected': expected,
                'actual': actual,
                'valid': is_valid
            })
        
        is_valid = all(v['valid'] for v in validations)
        
        return {
            'valid': is_valid,
            'validations': validations,
            'score': sum(1 for v in validations if v['valid']) / len(validations) if validations else 0
        }
    
    async def _process_specialist_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process a specialist task - domain-specific expertise"""
        domain = task.metadata.get('domain', 'general')
        
        # Get domain-specific context
        domain_context = await self._get_domain_context(domain, subagent)
        
        # Process with specialized knowledge
        result = await self._process_with_specialized_knowledge(
            task.description,
            domain_context,
            subagent
        )
        
        return result
    
    async def _process_generic_task(self, subagent: SubAgent, task: SubAgentTask) -> Any:
        """Process a generic task"""
        # Use the agent's LLM to handle the task
        messages = [
            Message(
                role="system",
                content=f"You are sub-agent {subagent.name} with capabilities: {', '.join(subagent.capabilities)}"
            ),
            Message(
                role="user",
                content=f"Complete this task: {task.description}\nContext: {json.dumps(subagent.context, default=str)}"
            )
        ]
        
        response = await self.agent.llm.complete(messages)
        return response.content
    
    async def assign_task(
        self,
        subagent_id: str,
        task: SubAgentTask
    ) -> None:
        """Assign a task to a sub-agent"""
        if subagent_id not in self.subagents:
            raise SubAgentError(f"Sub-agent {subagent_id} not found")
        
        subagent = self.subagents[subagent_id]
        
        if subagent.current_task:
            raise SubAgentError(f"Sub-agent {subagent_id} already has a task")
        
        subagent.current_task = task
        await self._trigger_event('on_task_assigned', {
            'subagent': subagent,
            'task': task
        })
    
    async def send_message(
        self,
        from_id: str,
        to_id: str,
        message: Dict[str, Any]
    ) -> None:
        """Send a message between sub-agents"""
        if to_id not in self.subagents:
            raise SubAgentError(f"Sub-agent {to_id} not found")
        
        # Add message metadata
        message['from'] = from_id
        message['to'] = to_id
        message['timestamp'] = time.time()
        
        # Queue message
        await self.subagents[to_id].communication_channel.put(message)
        self.metrics['total_messages'] += 1
        
        logger.debug(f"Message sent from {from_id} to {to_id}")
    
    async def broadcast_message(
        self,
        from_id: str,
        message: Dict[str, Any],
        recipients: Optional[List[str]] = None
    ) -> None:
        """Broadcast a message to multiple sub-agents"""
        if recipients is None:
            recipients = list(self.subagents.keys())
            recipients.remove(from_id)  # Don't send to self
        
        for recipient_id in recipients:
            await self.send_message(from_id, recipient_id, message)
    
    async def _handle_message(self, subagent: SubAgent, message: Dict) -> None:
        """Handle an incoming message"""
        msg_type = message.get('type', 'unknown')
        
        if msg_type == 'task':
            # New task assigned
            task = SubAgentTask(
                id=message['task_id'],
                description=message['description'],
                metadata=message.get('metadata', {})
            )
            await self.assign_task(subagent.id, task)
            
        elif msg_type == 'query':
            # Query for information
            query = message.get('query')
            context = message.get('context', {})
            result = await self._handle_query(subagent, query, context)
            
            # Send response
            await self.send_message(
                subagent.id,
                message['from'],
                {'type': 'response', 'result': result}
            )
            
        elif msg_type == 'command':
            # Execute a command
            command = message.get('command')
            params = message.get('params', {})
            result = await self._execute_command(subagent, command, params)
            
            # Send response
            await self.send_message(
                subagent.id,
                message['from'],
                {'type': 'command_response', 'result': result}
            )
        
        elif msg_type == 'heartbeat':
            # Heartbeat check
            subagent.last_active = time.time()
            await self.send_message(
                subagent.id,
                message['from'],
                {'type': 'heartbeat_response', 'status': subagent.status.value}
            )
        
        elif msg_type == 'error':
            # Error message
            logger.error(f"Sub-agent {subagent.id} received error from {message['from']}: {message.get('error')}")
    
    async def wait_for_subagent(
        self,
        subagent_id: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Wait for a sub-agent to complete its task"""
        if subagent_id not in self.subagents:
            raise SubAgentError(f"Sub-agent {subagent_id} not found")
        
        subagent = self.subagents[subagent_id]
        timeout = timeout or self.default_timeout
        
        start_time = time.time()
        
        while subagent.status not in [
            SubAgentStatus.COMPLETED,
            SubAgentStatus.FAILED,
            SubAgentStatus.TERMINATED
        ]:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Sub-agent {subagent_id} timed out")
            await asyncio.sleep(0.1)
        
        return {
            'id': subagent.id,
            'name': subagent.name,
            'status': subagent.status.value,
            'tasks_completed': len(subagent.completed_tasks),
            'metrics': subagent.metrics,
            'last_result': subagent.completed_tasks[-1].result if subagent.completed_tasks else None
        }
    
    async def terminate_subagent(
        self,
        subagent_id: str,
        force: bool = False
    ) -> None:
        """Terminate a sub-agent"""
        if subagent_id not in self.subagents:
            raise SubAgentError(f"Sub-agent {subagent_id} not found")
        
        subagent = self.subagents[subagent_id]
        
        # Cancel the sub-agent's task
        if subagent_id in self.active_subagents:
            self.active_subagents[subagent_id].cancel()
            try:
                await self.active_subagents[subagent_id]
            except asyncio.CancelledError:
                pass
        
        subagent.status = SubAgentStatus.TERMINATED
        
        # Terminate children recursively
        if subagent_id in self.agent_hierarchy:
            for child_id in self.agent_hierarchy[subagent_id]:
                if child_id in self.subagents:
                    await self.terminate_subagent(child_id, force)
        
        # Clean up
        del self.subagents[subagent_id]
        if subagent_id in self.active_subagents:
            del self.active_subagents[subagent_id]
        if subagent_id in self.broadcast_channels:
            del self.broadcast_channels[subagent_id]
        
        logger.info(f"Sub-agent {subagent_id} terminated")
    
    def _get_depth(self, agent_id: str) -> int:
        """Get the depth of an agent in the hierarchy"""
        depth = 0
        current = agent_id
        
        while current in self.subagents and self.subagents[current].parent_id:
            depth += 1
            current = self.subagents[current].parent_id
        
        return depth
    
    def _get_subagent_llm_config(self, subagent: SubAgent) -> Dict:
        """Get LLM configuration for a sub-agent"""
        # Use smaller/faster models for sub-agents by default
        return {
            'model': self.config.get('subagent_model', 'gpt-3.5-turbo'),
            'temperature': 0.5,
            'max_tokens': 1000
        }
    
    async def _decompose_task(self, task_desc: str, context: Dict) -> List[str]:
        """Decompose a task into sub-tasks"""
        messages = [
            Message(
                role="system",
                content="Break down this task into smaller, executable sub-tasks."
            ),
            Message(
                role="user",
                content=f"Task: {task_desc}\nContext: {json.dumps(context, default=str)}"
            )
        ]
        
        response = await self.agent.llm.complete(messages)
        
        # Parse sub-tasks from response
        import re
        subtasks = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', response.content, re.DOTALL)
        return [s.strip() for s in subtasks if s.strip()]
    
    async def _get_executor_plan(self, task: SubAgentTask, context: Dict) -> Dict:
        """Get an execution plan for an executor"""
        messages = [
            Message(
                role="system",
                content="Create a step-by-step plan to execute this task."
            ),
            Message(
                role="user",
                content=f"Task: {task.description}\nContext: {json.dumps(context, default=str)}"
            )
        ]
        
        response = await self.agent.llm.complete(messages)
        return {'plan': response.content}
    
    async def _execute_plan_with_tools(self, plan: Dict, subagent: SubAgent) -> Any:
        """Execute a plan using available tools"""
        # This is a simplified implementation
        # In production, this would use the tool registry
        return {'status': 'executed', 'plan': plan}
    
    async def _observe_target(self, target: str, subagent: SubAgent) -> Dict:
        """Observe a target for monitoring"""
        # Simplified monitoring
        return {
            'target': target,
            'timestamp': time.time(),
            'status': 'operational',
            'metrics': {'cpu': 50, 'memory': 60}
        }
    
    def _detect_anomaly(self, observation: Dict) -> bool:
        """Detect anomalies in observations"""
        # Simplified anomaly detection
        return observation.get('metrics', {}).get('cpu', 0) > 90
    
    async def _validate_value(self, key: str, actual: Any, expected: Any, context: Dict) -> bool:
        """Validate a value against expected criteria"""
        if isinstance(expected, (int, float)):
            if isinstance(actual, (int, float)):
                return abs(actual - expected) < 0.01
        elif isinstance(expected, str):
            return str(actual) == expected
        elif isinstance(expected, list):
            return isinstance(actual, list) and all(a in expected for a in actual)
        elif isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                self._validate_value(k, actual.get(k), v, context)
                for k, v in expected.items()
            )
        return actual == expected
    
    async def _handle_query(self, subagent: SubAgent, query: str, context: Dict) -> Any:
        """Handle a query from another agent"""
        messages = [
            Message(
                role="system",
                content=f"You are sub-agent {subagent.name}. Respond to queries based on your capabilities."
            ),
            Message(
                role="user",
                content=f"Query: {query}\nContext: {json.dumps(context, default=str)}"
            )
        ]
        
        response = await self.agent.llm.complete(messages)
        return response.content
    
    async def _execute_command(self, subagent: SubAgent, command: str, params: Dict) -> Any:
        """Execute a command"""
        # Simplified command execution
        return {'status': 'executed', 'command': command, 'params': params}
    
    async def _heartbeat_check(self, subagent: SubAgent) -> None:
        """Perform periodic heartbeat check"""
        subagent.last_active = time.time()
        
        # Check if sub-agent is stuck
        if subagent.status == SubAgentStatus.WORKING:
            task_start = subagent.current_task.assigned_at if subagent.current_task else None
            if task_start and time.time() - task_start > self.default_timeout:
                logger.warning(f"Sub-agent {subagent.id} may be stuck")
                await self._trigger_event('on_subagent_stuck', {'subagent': subagent})
    
    async def _handle_loop_error(self, subagent: SubAgent, error: Exception) -> None:
        """Handle an error in the sub-agent loop"""
        if subagent.status != SubAgentStatus.FAILED:
            subagent.status = SubAgentStatus.FAILED
            await self._trigger_event('on_subagent_failed', {
                'subagent': subagent,
                'error': str(error)
            })
    
    async def _handle_task_failure(
        self,
        subagent: SubAgent,
        task: SubAgentTask,
        error: Exception
    ) -> None:
        """Handle a task failure"""
        # Try to recover
        if subagent.current_task and len(subagent.completed_tasks) > 0:
            # Attempt to continue with next task
            subagent.current_task = None
            
            # Check if we should retry
            if task.metadata.get('retry_count', 0) < 3:
                task.metadata['retry_count'] = task.metadata.get('retry_count', 0) + 1
                await self.assign_task(subagent.id, task)
                logger.info(f"Retrying task {task.id} for sub-agent {subagent.id}")
            else:
                logger.error(f"Task {task.id} failed permanently for sub-agent {subagent.id}")
                subagent.status = SubAgentStatus.FAILED
                await self._trigger_event('on_task_failed_permanently', {
                    'subagent': subagent,
                    'task': task,
                    'error': str(error)
                })
    
    async def _get_domain_context(self, domain: str, subagent: SubAgent) -> Dict:
        """Get domain-specific context"""
        # Simplified - in production, this would use a knowledge base
        domain_contexts = {
            'python': {'libraries': ['numpy', 'pandas', 'requests']},
            'javascript': {'frameworks': ['react', 'node', 'express']},
            'devops': {'tools': ['docker', 'kubernetes', 'terraform']},
            'data_science': {'tools': ['jupyter', 'tensorflow', 'pytorch']}
        }
        return domain_contexts.get(domain, {})
    
    async def _process_with_specialized_knowledge(
        self,
        task: str,
        domain_context: Dict,
        subagent: SubAgent
    ) -> Any:
        """Process using specialized knowledge"""
        messages = [
            Message(
                role="system",
                content=f"You are a specialist in {domain_context.get('domain', 'general')}. "
                       f"Knowledge: {json.dumps(domain_context, default=str)}"
            ),
            Message(
                role="user",
                content=f"Process this task: {task}"
            )
        ]
        
        response = await self.agent.llm.complete(messages)
        return {'processed': True, 'result': response.content}
    
    async def _trigger_event(self, event: str, data: Dict) -> None:
        """Trigger an event"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add an event handler"""
        self.event_handlers[event].append(handler)
    
    async def shutdown(self) -> None:
        """Shutdown all sub-agents"""
        logger.info("Shutting down sub-agent manager...")
        
        # Terminate all active sub-agents
        for subagent_id in list(self.subagents.keys()):
            await self.terminate_subagent(subagent_id, force=True)
        
        # Clean up
        self.active_subagents.clear()
        self.subagents.clear()
        self.agent_hierarchy.clear()
        self.broadcast_channels.clear()
        
        logger.info("Sub-agent manager shutdown complete")