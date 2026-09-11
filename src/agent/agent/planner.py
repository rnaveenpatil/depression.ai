"""
Planner Module - Task Decomposition and Planning
Advanced planning system with hierarchical task decomposition,
dependency management, and adaptive re-planning
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from collections import defaultdict

from agent.utils.logging import get_logger
from agent.utils.errors import PlanningError, ExecutionError
from agent.llm.provider import LLMProvider, Message
from agent.tools.registry import ToolRegistry

logger = get_logger(__name__)

class TaskStatus(Enum):
    """Status of a task in the plan"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    RETRY = "retry"

class Priority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    OPTIONAL = 4

@dataclass
class Task:
    """Represents a single task in a plan"""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
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
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.value,
            'dependencies': self.dependencies,
            'subtasks': [t.to_dict() for t in self.subtasks],
            'tool_calls': self.tool_calls,
            'estimated_time': self.estimated_time,
            'actual_time': self.actual_time,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        task = cls(
            id=data['id'],
            description=data['description'],
            status=TaskStatus(data['status']),
            priority=Priority(data['priority']),
            dependencies=data['dependencies'],
            tool_calls=data['tool_calls'],
            estimated_time=data.get('estimated_time'),
            actual_time=data.get('actual_time'),
            result=data.get('result'),
            error=data.get('error'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            metadata=data.get('metadata', {})
        )
        task.subtasks = [cls.from_dict(st) for st in data.get('subtasks', [])]
        return task

@dataclass
class Plan:
    """Represents a complete plan with multiple tasks"""
    id: str
    goal: str
    tasks: List[Task]
    status: str = "active"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks that are ready to execute"""
        completed_ids = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        ready = []
        
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                deps_completed = all(dep in completed_ids for dep in task.dependencies)
                if deps_completed:
                    ready.append(task)
        
        return sorted(ready, key=lambda t: t.priority.value)
    
    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find a task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
            # Check subtasks
            found = self._find_task_in_subtasks(task, task_id)
            if found:
                return found
        return None
    
    def _find_task_in_subtasks(self, task: Task, task_id: str) -> Optional[Task]:
        for subtask in task.subtasks:
            if subtask.id == task_id:
                return subtask
            found = self._find_task_in_subtasks(subtask, task_id)
            if found:
                return found
        return None
    
    def get_completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if not self.tasks:
            return 0.0
        
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed"""
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'goal': self.goal,
            'tasks': [t.to_dict() for t in self.tasks],
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'completed_at': self.completed_at,
            'metrics': self.metrics,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Plan':
        plan = cls(
            id=data['id'],
            goal=data['goal'],
            tasks=[Task.from_dict(t) for t in data['tasks']],
            status=data['status'],
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            completed_at=data.get('completed_at'),
            metrics=data.get('metrics', {}),
            context=data.get('context', {})
        )
        return plan

class Planner:
    """
    Advanced Task Planner with:
    - Hierarchical task decomposition
    - Dependency resolution
    - Dynamic re-planning
    - Resource estimation
    - Parallel task execution optimization
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        config: Dict[str, Any]
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        
        # Planning parameters
        self.max_tasks_per_plan = config.get('max_tasks_per_plan', 20)
        self.max_subtasks_per_task = config.get('max_subtasks_per_task', 10)
        self.enable_parallel = config.get('enable_parallel', True)
        self.max_parallel_tasks = config.get('max_parallel_tasks', 5)
        self.adaptive_planning = config.get('adaptive_planning', True)
        
        # State
        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []
        self.execution_context: Dict[str, Any] = {}
        self.cached_plans: Dict[str, Plan] = {}
        
        # Metrics
        self.metrics = {
            'plans_created': 0,
            'tasks_decomposed': 0,
            'replans': 0,
            'avg_planning_time': 0.0
        }
        
        logger.info("Planner initialized")
    
    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Plan:
        """
        Create a detailed plan for achieving a goal
        
        Args:
            goal: The overall goal to achieve
            context: Additional context for planning
            constraints: Constraints like time, resources, etc.
        
        Returns:
            A complete Plan object
        """
        start_time = time.time()
        
        try:
            # Prepare planning context
            planning_context = self._prepare_planning_context(goal, context, constraints)
            
            # Generate plan using LLM
            plan_data = await self._generate_plan(goal, planning_context)
            
            # Validate and structure the plan
            plan = await self._structure_plan(goal, plan_data, constraints)
            
            # Validate dependencies
            self._validate_dependencies(plan)
            
            # Estimate resources
            await self._estimate_resources(plan)
            
            # Store the plan
            self.current_plan = plan
            self.plan_history.append(plan)
            self.metrics['plans_created'] += 1
            
            # Update metrics
            planning_time = time.time() - start_time
            self.metrics['avg_planning_time'] = (
                (self.metrics['avg_planning_time'] * (self.metrics['plans_created'] - 1) +
                 planning_time) / self.metrics['plans_created']
            )
            
            logger.info(f"Created plan with {len(plan.tasks)} tasks in {planning_time:.2f}s")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}", exc_info=True)
            raise PlanningError(f"Failed to create plan: {e}")
    
    async def _generate_plan(self, goal: str, context: Dict) -> Dict:
        """
        Use LLM to generate a plan
        """
        system_prompt = self._get_planning_system_prompt()
        user_prompt = self._get_planning_user_prompt(goal, context)
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        response = await self.llm.complete(
            messages=messages,
            temperature=0.3,  # Lower temperature for more structured output
            max_tokens=2000
        )
        
        # Parse the response
        try:
            plan_data = self._parse_plan_response(response.content)
            return plan_data
        except Exception as e:
            logger.error(f"Failed to parse plan: {e}")
            # Fallback to a simpler plan
            return self._create_fallback_plan(goal)
    
    def _get_planning_system_prompt(self) -> str:
        """Get the system prompt for planning"""
        return """
        You are an expert task planner. Your job is to decompose complex goals into 
        executable tasks with clear dependencies and priorities.
        
        Follow these principles:
        1. Break down the goal into discrete, actionable tasks
        2. Identify dependencies between tasks
        3. Assign appropriate priorities (critical, high, medium, low, optional)
        4. Suggest appropriate tools for each task
        5. Consider parallel execution opportunities
        6. Include verification/validation steps
        
        Output format:
        {
            "tasks": [
                {
                    "id": "task_1",
                    "description": "Task description",
                    "priority": "high",
                    "dependencies": [],
                    "subtasks": [...],
                    "tools": ["tool_name", ...],
                    "estimated_time": 30,  # in seconds
                    "validation": "How to validate this task"
                }
            ],
            "parallel_groups": [["task_1", "task_2"], ...],
            "estimated_total_time": 300
        }
        """
    
    def _get_planning_user_prompt(self, goal: str, context: Dict) -> str:
        """Get the user prompt for planning"""
        available_tools = [t.name for t in self.tool_registry.tools.values()]
        
        prompt = f"""
        Goal: {goal}
        
        Available Tools: {', '.join(available_tools)}
        
        Additional Context:
        {json.dumps(context, indent=2)}
        
        Create a detailed plan to achieve this goal. Consider:
        - What are the key steps?
        - What are the dependencies?
        - What tools should be used?
        - How can tasks be parallelized?
        - What are the validation points?
        """
        return prompt
    
    def _parse_plan_response(self, response: str) -> Dict:
        """Parse the LLM response into a structured plan"""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try to parse as JSON directly
                return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON, using fallback")
            return self._create_fallback_plan_data(response)
    
    def _create_fallback_plan_data(self, response: str) -> Dict:
        """Create a plan from unstructured response"""
        # Split by sentences or bullet points
        import re
        tasks = re.split(r'\d+\.\s*|•\s*|\n- ', response)
        
        plan_data = {
            'tasks': [],
            'parallel_groups': [],
            'estimated_total_time': 300
        }
        
        for i, task_desc in enumerate(tasks):
            if task_desc.strip():
                plan_data['tasks'].append({
                    'id': f"task_{i+1}",
                    'description': task_desc.strip(),
                    'priority': 'medium',
                    'dependencies': [],
                    'subtasks': [],
                    'tools': [],
                    'estimated_time': 30,
                    'validation': 'Verify task completion'
                })
        
        return plan_data
    
    def _create_fallback_plan(self, goal: str) -> Dict:
        """Create a minimal fallback plan"""
        return {
            'tasks': [
                {
                    'id': 'task_1',
                    'description': f'Analyze goal: {goal}',
                    'priority': 'critical',
                    'dependencies': [],
                    'subtasks': [],
                    'tools': ['analyze'],
                    'estimated_time': 30,
                    'validation': 'Goal understood'
                },
                {
                    'id': 'task_2',
                    'description': 'Execute primary tasks',
                    'priority': 'high',
                    'dependencies': ['task_1'],
                    'subtasks': [],
                    'tools': ['execute'],
                    'estimated_time': 60,
                    'validation': 'Tasks completed'
                },
                {
                    'id': 'task_3',
                    'description': 'Verify results',
                    'priority': 'medium',
                    'dependencies': ['task_2'],
                    'subtasks': [],
                    'tools': ['verify'],
                    'estimated_time': 30,
                    'validation': 'Results verified'
                }
            ],
            'parallel_groups': [],
            'estimated_total_time': 120
        }
    
    async def _structure_plan(
        self,
        goal: str,
        plan_data: Dict,
        constraints: Optional[Dict]
    ) -> Plan:
        """Structure the plan data into a Plan object"""
        tasks = []
        
        for task_data in plan_data.get('tasks', []):
            # Handle subtasks
            subtasks = []
            for st_data in task_data.get('subtasks', []):
                subtask = Task(
                    id=st_data.get('id', str(uuid.uuid4())[:8]),
                    description=st_data['description'],
                    priority=Priority[st_data.get('priority', 'medium').upper()],
                    tool_calls=[{'tool': t} for t in st_data.get('tools', [])],
                    estimated_time=st_data.get('estimated_time', 30),
                    metadata={'validation': st_data.get('validation', '')}
                )
                subtasks.append(subtask)
            
            task = Task(
                id=task_data.get('id', str(uuid.uuid4())[:8]),
                description=task_data['description'],
                priority=Priority[task_data.get('priority', 'medium').upper()],
                dependencies=task_data.get('dependencies', []),
                subtasks=subtasks,
                tool_calls=[{'tool': t} for t in task_data.get('tools', [])],
                estimated_time=task_data.get('estimated_time', 30),
                metadata={'validation': task_data.get('validation', '')}
            )
            tasks.append(task)
        
        # Create plan
        plan = Plan(
            id=str(uuid.uuid4()),
            goal=goal,
            tasks=tasks,
            context=plan_data.get('context', {}),
            metrics={
                'parallel_groups': plan_data.get('parallel_groups', []),
                'estimated_total_time': plan_data.get('estimated_total_time', 0),
                'constraints': constraints or {}
            }
        )
        
        return plan
    
    def _prepare_planning_context(
        self,
        goal: str,
        context: Optional[Dict],
        constraints: Optional[Dict]
    ) -> Dict:
        """Prepare the context for planning"""
        planning_context = {
            'goal': goal,
            'available_tools': list(self.tool_registry.tools.keys()),
            'max_tasks': self.max_tasks_per_plan,
            'enable_parallel': self.enable_parallel,
            'context': context or {},
            'constraints': constraints or {}
        }
        
        # Add previous plan info if available
        if self.current_plan:
            planning_context['previous_plan'] = {
                'tasks_completed': len([t for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]),
                'success_rate': self.current_plan.metrics.get('success_rate', 0),
                'previous_goal': self.current_plan.goal
            }
        
        return planning_context
    
    def _validate_dependencies(self, plan: Plan) -> None:
        """Validate all dependencies are valid"""
        task_ids = {t.id for t in plan.tasks}
        
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    logger.warning(f"Task {task.id} has invalid dependency: {dep}")
                    # Remove invalid dependency
                    task.dependencies.remove(dep)
    
    async def _estimate_resources(self, plan: Plan) -> None:
        """Estimate resources for each task"""
        for task in plan.tasks:
            if not task.estimated_time:
                # Estimate based on complexity
                task.estimated_time = 30 + len(task.subtasks) * 15 + len(task.dependencies) * 10
    
    async def execute_plan(
        self,
        plan: Plan,
        executor: Callable,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan with dynamic task scheduling
        
        Args:
            plan: The plan to execute
            executor: Function to execute tasks
            progress_callback: Optional progress callback
        
        Returns:
            Execution results and metrics
        """
        if not plan:
            raise PlanningError("No plan to execute")
        
        start_time = time.time()
        self.current_plan = plan
        
        # Track execution state
        executed_tasks = set()
        failed_tasks = []
        results = {}
        
        try:
            while not plan.is_complete():
                # Get ready tasks
                ready_tasks = plan.get_pending_tasks()
                
                if not ready_tasks:
                    # Check for deadlock
                    remaining = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
                    if remaining:
                        logger.warning(f"Deadlock detected - {len(remaining)} tasks pending with unmet dependencies")
                        # Try to recover by skipping blocked tasks
                        self._resolve_deadlock(plan)
                        continue
                    break
                
                # Execute ready tasks (potentially in parallel)
                if self.enable_parallel and len(ready_tasks) > 1:
                    # Limit parallel tasks
                    parallel_tasks = ready_tasks[:self.max_parallel_tasks]
                    await self._execute_parallel_tasks(
                        parallel_tasks, executor, results, progress_callback
                    )
                else:
                    # Execute sequentially
                    for task in ready_tasks[:1]:  # Execute one at a time
                        await self._execute_single_task(
                            task, executor, results, progress_callback
                        )
                
                # Update progress
                if progress_callback:
                    completion = plan.get_completion_percentage()
                    await progress_callback(completion, plan)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            success = all(t.status == TaskStatus.COMPLETED for t in plan.tasks)
            
            plan.metrics.update({
                'execution_time': execution_time,
                'success': success,
                'tasks_completed': len([t for t in plan.tasks if t.status == TaskStatus.COMPLETED]),
                'tasks_failed': len([t for t in plan.tasks if t.status == TaskStatus.FAILED]),
                'success_rate': sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED) / len(plan.tasks)
            })
            
            if success:
                plan.status = "completed"
                plan.completed_at = time.time()
            
            logger.info(f"Plan execution {'succeeded' if success else 'failed'} in {execution_time:.2f}s")
            
            return {
                'success': success,
                'plan_id': plan.id,
                'metrics': plan.metrics,
                'results': results,
                'failed_tasks': failed_tasks
            }
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}", exc_info=True)
            raise ExecutionError(f"Failed to execute plan: {e}")
    
    async def _execute_single_task(
        self,
        task: Task,
        executor: Callable,
        results: Dict,
        progress_callback: Optional[Callable]
    ) -> None:
        """Execute a single task"""
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()
        
        try:
            # Execute subtasks first if they exist
            if task.subtasks:
                subtask_results = []
                for subtask in task.subtasks:
                    await self._execute_single_task(
                        subtask, executor, results, progress_callback
                    )
                    subtask_results.append(subtask.result)
                task.result = subtask_results
            else:
                # Execute the main task
                result = await executor(task)
                task.result = result
            
            task.status = TaskStatus.COMPLETED
            task.actual_time = time.time() - task.created_at
            
            results[task.id] = {
                'success': True,
                'result': task.result,
                'time': task.actual_time
            }
            
            logger.debug(f"Task {task.id} completed")
            
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRY
                logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries})")
                # Retry after a delay
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_single_task(task, executor, results, progress_callback)
            else:
                task.status = TaskStatus.FAILED
                results[task.id] = {
                    'success': False,
                    'error': task.error,
                    'retries': task.retry_count
                }
                logger.error(f"Task {task.id} failed after {task.retry_count} retries: {task.error}")
    
    async def _execute_parallel_tasks(
        self,
        tasks: List[Task],
        executor: Callable,
        results: Dict,
        progress_callback: Optional[Callable]
    ) -> None:
        """Execute multiple tasks in parallel"""
        # Create tasks
        async_tasks = []
        for task in tasks:
            async_tasks.append(
                self._execute_single_task(task, executor, results, progress_callback)
            )
        
        # Execute in parallel
        await asyncio.gather(*async_tasks, return_exceptions=True)
    
    def _resolve_deadlock(self, plan: Plan) -> None:
        """Resolve deadlock by adjusting dependencies"""
        pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
        
        for task in pending:
            # Check if all dependencies are completed or failed
            deps_status = []
            for dep_id in task.dependencies:
                dep = plan.get_task_by_id(dep_id)
                if dep:
                    deps_status.append(dep.status)
            
            if all(s in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED] for s in deps_status):
                # All dependencies are resolved, so this task can proceed
                # Clean up failed dependencies
                for dep_id in task.dependencies:
                    dep = plan.get_task_by_id(dep_id)
                    if dep and dep.status == TaskStatus.FAILED:
                        # Mark as skipped to unblock
                        dep.status = TaskStatus.SKIPPED
                        logger.warning(f"Skipping failed dependency {dep_id} to unblock {task.id}")
    
    async def re_plan(
        self,
        plan: Plan,
        failed_tasks: List[Task],
        new_context: Optional[Dict] = None
    ) -> Plan:
        """
        Re-plan based on failures and new context
        
        Args:
            plan: The original plan
            failed_tasks: Tasks that failed
            new_context: Updated context
        
        Returns:
            A new plan
        """
        self.metrics['replans'] += 1
        
        # Analyze failures
        failure_analysis = self._analyze_failures(failed_tasks)
        
        # Create new goal based on remaining work
        remaining_goal = f"{plan.goal} - Adjust based on failures: {failure_analysis}"
        
        # Create new context
        context = {
            'original_plan': plan.to_dict(),
            'failed_tasks': [t.to_dict() for t in failed_tasks],
            'failure_analysis': failure_analysis,
            'completed_tasks': [t.to_dict() for t in plan.tasks if t.status == TaskStatus.COMPLETED],
            'new_context': new_context or {}
        }
        
        # Generate new plan
        new_plan = await self.create_plan(remaining_goal, context)
        
        # Add original plan ID to metadata
        new_plan.metadata['original_plan_id'] = plan.id
        new_plan.metadata['replan_reason'] = failure_analysis
        
        # Mark original plan as superseded
        plan.status = "superseded"
        
        logger.info(f"Replanned with {len(new_plan.tasks)} tasks")
        return new_plan
    
    def _analyze_failures(self, failed_tasks: List[Task]) -> str:
        """Analyze why tasks failed"""
        if not failed_tasks:
            return "No failures to analyze"
        
        failure_types = defaultdict(int)
        failure_details = []
        
        for task in failed_tasks:
            if "timeout" in str(task.error).lower():
                failure_types['timeout'] += 1
            elif "permission" in str(task.error).lower():
                failure_types['permission'] += 1
            elif "tool" in str(task.error).lower():
                failure_types['tool_error'] += 1
            else:
                failure_types['unknown'] += 1
            
            failure_details.append(f"Task {task.id}: {task.error}")
        
        analysis = f"Failed {len(failed_tasks)} tasks: "
        analysis += ", ".join([f"{k}: {v}" for k, v in failure_types.items()])
        analysis += ". Details: " + "; ".join(failure_details[:3])
        
        return analysis
    
    async def optimize_plan(self, plan: Plan) -> Plan:
        """
        Optimize a plan for better execution efficiency
        """
        # Identify parallelization opportunities
        parallel_groups = self._find_parallel_groups(plan)
        
        # Merge or split tasks for optimization
        optimized_tasks = self._optimize_task_structure(plan.tasks)
        
        # Create optimized plan
        optimized_plan = Plan(
            goal=plan.goal,
            tasks=optimized_tasks,
            context=plan.context,
            metrics={
                'original_tasks': len(plan.tasks),
                'optimized_tasks': len(optimized_tasks),
                'parallel_groups': len(parallel_groups)
            }
        )
        
        return optimized_plan
    
    def _find_parallel_groups(self, plan: Plan) -> List[List[str]]:
        """Find tasks that can be executed in parallel"""
        graph = defaultdict(set)
        
        # Build dependency graph
        for task in plan.tasks:
            for dep in task.dependencies:
                graph[dep].add(task.id)
                graph[task.id].add(dep)
        
        # Find tasks with no dependencies on each other
        groups = []
        remaining = set(plan.tasks)
        
        while remaining:
            group = []
            # Pick a task and find all that can run with it
            first = next(iter(remaining))
            group.append(first.id)
            remaining.remove(first)
            
            # Find other tasks with no dependencies to tasks in group
            for task in list(remaining):
                has_dep = any(dep in task.dependencies for dep in group)
                is_dep = any(task.id in graph[dep] for dep in group)
                
                if not has_dep and not is_dep:
                    group.append(task.id)
                    remaining.remove(task)
            
            groups.append(group)
        
        return groups
    
    def _optimize_task_structure(self, tasks: List[Task]) -> List[Task]:
        """Optimize the task structure"""
        # Merge tiny tasks
        merged_tasks = []
        i = 0
        while i < len(tasks):
            if i + 1 < len(tasks) and tasks[i].estimated_time < 10:
                # Merge with next task
                merged = Task(
                    description=f"{tasks[i].description} then {tasks[i+1].description}",
                    dependencies=tasks[i].dependencies + tasks[i+1].dependencies,
                    priority=min(tasks[i].priority.value, tasks[i+1].priority.value),
                    estimated_time=tasks[i].estimated_time + tasks[i+1].estimated_time
                )
                merged_tasks.append(merged)
                i += 2
            else:
                merged_tasks.append(tasks[i])
                i += 1
        
        return merged_tasks
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict]:
        """Get the status of a plan"""
        # Check current plan
        if self.current_plan and self.current_plan.id == plan_id:
            return self._get_plan_status_dict(self.current_plan)
        
        # Check history
        for plan in self.plan_history:
            if plan.id == plan_id:
                return self._get_plan_status_dict(plan)
        
        return None
    
    def _get_plan_status_dict(self, plan: Plan) -> Dict:
        return {
            'id': plan.id,
            'goal': plan.goal,
            'status': plan.status,
            'completion': plan.get_completion_percentage(),
            'tasks': [
                {
                    'id': t.id,
                    'description': t.description,
                    'status': t.status.value,
                    'priority': t.priority.name,
                    'time': t.actual_time or t.estimated_time
                }
                for t in plan.tasks
            ],
            'metrics': plan.metrics
        }
    
    async def save_plan(self, plan: Plan, storage) -> None:
        """Save a plan to storage"""
        plan_data = plan.to_dict()
        await storage.save_plan(plan.id, plan_data)
    
    async def load_plan(self, plan_id: str, storage) -> Optional[Plan]:
        """Load a plan from storage"""
        plan_data = await storage.load_plan(plan_id)
        if plan_data:
            return Plan.from_dict(plan_data)
        return None