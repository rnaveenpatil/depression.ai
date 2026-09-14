"""Canonical OpenCode-style agent execution loop.

Flow: session context -> LLM -> tool calls -> permission/execution -> tool
results -> canonical context -> LLM, repeated until the model returns a final
answer. AgentLoop owns execution telemetry only; ContextManager owns history.
"""
from __future__ import annotations
import asyncio, json, time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from agent.agent.planner import Plan, Planner, TaskStatus
from agent.llm.provider import LLMProvider, Message, ToolCall
from agent.tools.registry import ToolRegistry
from agent.utils.errors import TimeoutError
from agent.utils.logging import get_logger
from agent.context.runtime import add_tool_call, add_tool_result, get_model_messages
logger = get_logger(__name__)

class LoopState(Enum):
    IDLE="idle"; INITIALIZING="initializing"; THINKING="thinking"; PLANNING="planning"; ACTING="acting"; OBSERVING="observing"; EVALUATING="evaluating"; PAUSED="paused"; STOPPED="stopped"; ERROR="error"

@dataclass
class LoopContext:
    iteration: int=0
    tool_calls: List[ToolCall]=field(default_factory=list)
    observations: List[Dict[str,Any]]=field(default_factory=list)
    actions_taken: List[Dict[str,Any]]=field(default_factory=list)
    start_time: float=field(default_factory=time.time)
    last_action_time: float=field(default_factory=time.time)
    tokens_used: int=0
    cost: float=0.0
    errors: List[str]=field(default_factory=list)
    metadata: Dict[str,Any]=field(default_factory=dict)
    def add_observation(self, observation): self.observations.append(observation); self.metadata["last_observation_time"]=time.time()
    def add_action(self, action): self.actions_taken.append(action); self.last_action_time=time.time(); self.metadata["last_action_time"]=self.last_action_time
    def get_summary(self): return {"iteration":self.iteration,"total_actions":len(self.actions_taken),"total_observations":len(self.observations),"tokens_used":self.tokens_used,"cost":self.cost,"errors":len(self.errors),"duration":time.time()-self.start_time}

class AgentLoop:
    def __init__(self, agent: Any, llm: LLMProvider, tool_registry: ToolRegistry, planner: Planner, config: Dict[str,Any]):
        self.agent,self.llm,self.tool_registry,self.planner,self.config=agent,llm,tool_registry,planner,config
        self.max_iterations=config.get("max_iterations",50); self.max_tool_calls_per_iteration=config.get("max_tool_calls",5); self.max_history_length=config.get("max_history_length",40); self.enable_planning=config.get("enable_planning",True); self.enable_caching=config.get("enable_caching",True); self.default_timeout=config.get("timeout",60)
        self.state=LoopState.IDLE; self.context=LoopContext(); self.current_plan=None; self.pending_tool_calls=[]; self.completed_tool_calls=[]
        self.performance_history=deque(maxlen=100); self.tool_execution_times={}; self.tool_result_cache={}; self.event_handlers={}; self.should_stop=False; self.is_paused=False

    @property
    def model_context(self):
        return self.agent.context_manager

    async def run(self, query: str, context=None, tool_outputs=None, max_turns=None):
        self.context=LoopContext(); self.current_plan=None; self.pending_tool_calls=[]; self.completed_tool_calls=[]; self.should_stop=False; self.is_paused=False
        if max_turns is not None: self.max_iterations=max_turns
        try:
            self.state=LoopState.INITIALIZING
            if self.enable_planning and await self._should_plan(query): await self._create_plan(query,context)
            while not self.should_stop and self.context.iteration < self.max_iterations:
                while self.is_paused: await asyncio.sleep(.1)
                self.context.iteration+=1; self.state=LoopState.THINKING
                thought=await self._think()
                if thought.get("tool_calls"):
                    self.state=LoopState.ACTING; results=await self._act(thought["tool_calls"]); self.state=LoopState.OBSERVING; await self._observe(results); self.state=LoopState.EVALUATING
                    if not await self._evaluate(results): break
                    continue
                self.state=LoopState.STOPPED; return self._result(True,thought.get("response", ""))
            self.state=LoopState.STOPPED
            return self._result(True,await self._generate_final_response())
        except TimeoutError as exc:
            self.context.errors.append(str(exc)); self.state=LoopState.ERROR; return self._result(False,error=str(exc))
        except Exception as exc:
            logger.error("Loop failed: %s",exc,exc_info=True); self.context.errors.append(str(exc)); self.state=LoopState.ERROR; return self._result(False,error=str(exc))

    def _result(self, success, response="", error=None):
        out={"success":success,"response":response,"iteration":self.context.iteration,"tool_calls":len(self.completed_tool_calls),"context":self.context.get_summary(),"plan":self.current_plan.to_dict() if self.current_plan else None}
        if error: out["error"]=error
        return out

    async def _get_system_prompt(self):
        project=await self._get_project_context(); perms=self._get_permissions_context()
        return ("You are an advanced AI CLI agent.\n\nAvailable tools:\n"+self._get_tools_description()+"\n\nProject Context:\n"+json.dumps(project,indent=2,default=str)+"\n\nPermissions:\n"+json.dumps(perms,indent=2,default=str)+"\n\nRules:\n1. Use tools when necessary.\n2. Inspect every tool result before the next action.\n3. Never claim success without evidence.\n4. Respect permissions and security constraints.\n5. Continue until the task is complete or clarification is required.").strip()

    def _get_tools_description(self):
        return "\n".join(f"- {n}: {t.description}\n  Parameters: {json.dumps(t.parameters,default=str)}" for n,t in self.tool_registry.tools.items())
    async def _get_project_context(self):
        if self.agent and hasattr(self.agent,"workspace"):
            return {"project_path":str(self.agent.workspace.project_dir),"files":await self.agent.workspace.list_files(max_files=20),"git_info":await self.agent.workspace.get_git_info() if hasattr(self.agent.workspace,"get_git_info") else None}
        return {}
    def _get_permissions_context(self):
        p=getattr(self.agent,"permission_manager",None); return {"enabled":p.enabled,"auto_approve":p.auto_approve} if p else {"enabled":True,"auto_approve":False}
    async def _should_plan(self,q):
        return len(q.split())>20 or any(x in q.lower() for x in ("plan","steps","multiple","several"))

    async def _create_plan(self, query, context):
        try:
            self.state=LoopState.PLANNING; self.current_plan=await self.planner.create_plan(goal=query,context=context or {},constraints=self._get_plan_constraints())
            self.context.metadata["plan"]=self.current_plan.to_dict()
            await self.model_context.add_system_message("Execution plan guidance:\n"+json.dumps(self.current_plan.to_dict(),indent=2,default=str))
        except Exception as exc: logger.warning("Planning failed: %s",exc); self.current_plan=None
    def _get_plan_constraints(self): return {"max_tasks":20,"timeout":self.default_timeout,"available_tools":list(self.tool_registry.tools.keys())}

    async def _think(self):
        # The ContextManager owns the history. Do not append system/user here:
        # Agent.process_query already recorded the user turn.
        if not self.model_context.messages or self.model_context.messages[-1].role != "system":
            # Only add the system prompt once for a session.
            if not any(m.role=="system" and m.pinned for m in self.model_context.messages):
                await self.model_context.add_system_message(await self._get_system_prompt(),pinned=True)
        messages=get_model_messages(self.model_context,self.max_history_length)
        messages.append(Message(role="system",content="Current execution state: "+json.dumps(await self._get_state_context(),default=str)))
        if self.current_plan: messages.append(Message(role="system",content="Plan state: "+json.dumps(self._get_plan_context(),default=str)))
        response=await self.llm.complete_with_tools(messages=messages,tools=self._get_available_tools(),temperature=0.7,max_tokens=2000)
        usage=getattr(response,"usage",{}) or {}
        self.context.tokens_used += getattr(usage,"total_tokens",0) if hasattr(usage,"total_tokens") else usage.get("total_tokens",0)
        calls=list(getattr(response,"tool_calls",[]) or []); content=getattr(response,"content",None)
        if calls: await add_tool_call(self.model_context,content,calls)
        else: await self.model_context.add_assistant_message(content or "")
        self.context.tool_calls.extend(calls)
        return {"response":content,"tool_calls":calls}

    def _get_available_tools(self): return [{"type":"function","function":{"name":n,"description":t.description,"parameters":t.parameters}} for n,t in self.tool_registry.tools.items()]
    async def _get_state_context(self): return {"iteration":self.context.iteration,"max_iterations":self.max_iterations,"actions_taken":len(self.context.actions_taken),"observations":len(self.context.observations),"completed_tool_calls":len(self.completed_tool_calls),"pending_tool_calls":len(self.pending_tool_calls)}
    def _get_plan_context(self): return {"plan_id":self.current_plan.id,"goal":self.current_plan.goal,"completion":self.current_plan.get_completion_percentage(),"pending_tasks":[t.to_dict() for t in self.current_plan.get_pending_tasks()],"completed_tasks":[t.to_dict() for t in self.current_plan.tasks if t.status==TaskStatus.COMPLETED]} if self.current_plan else {}

    async def _act(self,calls):
        results=[]
        for call in list(calls)[:self.max_tool_calls_per_iteration]:
            try:
                key=f"{call.name}:{json.dumps(call.arguments,sort_keys=True,default=str)}"
                if self.enable_caching and key in self.tool_result_cache and time.time()-self.tool_result_cache[key]["timestamp"]<60:
                    result=self.tool_result_cache[key]["result"]; await add_tool_result(self.model_context,call,result); results.append({"tool":call.name,"tool_call_id":call.id,"result":result,"cached":True}); self.completed_tool_calls.append(call); continue
                start=time.time(); result=await self.agent.execute_tool(call.name,call.arguments); elapsed=time.time()-start
                if not isinstance(result,dict): result={"success":True,"result":result}
                self.tool_execution_times.setdefault(call.name,[]).append(elapsed); self.context.add_action({"tool":call.name,"tool_call_id":call.id,"params":call.arguments,"result":result,"time":elapsed})
                await add_tool_result(self.model_context,call,result)
                if self.enable_caching and result.get("success",False): self.tool_result_cache[key]={"result":result,"timestamp":time.time()}
                self.completed_tool_calls.append(call); results.append({"tool":call.name,"tool_call_id":call.id,"result":result,"execution_time":elapsed})
            except Exception as exc:
                result={"success":False,"error":str(exc)}; await add_tool_result(self.model_context,call,result); self.pending_tool_calls.append(call); results.append({"tool":call.name,"tool_call_id":call.id,"result":result,"error":str(exc),"success":False})
        return results
    async def _observe(self,results):
        for x in results:
            r=x.get("result") or {}; self.context.add_observation({"timestamp":time.time(),"tool":x.get("tool"),"tool_call_id":x.get("tool_call_id"),"success":bool(r.get("success",False)),"error":x.get("error") or r.get("error"),"execution_time":x.get("execution_time",0),"result":r})
    async def _evaluate(self,results):
        if self.context.iteration>=self.max_iterations:return False
        return True
    async def _generate_final_response(self):
        # Ask the model for a final response from the canonical context instead of reusing an old answer.
        messages=get_model_messages(self.model_context,self.max_history_length)
        messages.append(Message(role="user",content="Provide the final response to the user's original request. State what was actually completed and mention any remaining issue; do not claim unverified success."))
        response=await self.llm.complete(messages=messages)
        content=getattr(response,"content","")
        await self.model_context.add_assistant_message(content)
        return content
    async def _execute_plan_task(self,task):
        task.status=TaskStatus.IN_PROGRESS; task.updated_at=time.time(); self.context.metadata["active_plan_task"]=task.id
    def _update_metrics(self): self.performance_history.append({"timestamp":time.time(),"duration":time.time()-self.context.start_time,"iterations":self.context.iteration,"tool_calls":len(self.completed_tool_calls),"tokens":self.context.tokens_used,"errors":len(self.context.errors)})
    async def reset(self): self.context=LoopContext(); self.current_plan=None; self.pending_tool_calls=[]; self.completed_tool_calls=[]; self.should_stop=False; self.is_paused=False; self.state=LoopState.IDLE
    def pause(self): self.is_paused=True; self.state=LoopState.PAUSED
    def resume(self): self.is_paused=False; self.state=LoopState.THINKING
    def stop(self): self.should_stop=True; self.state=LoopState.STOPPED
    def get_loop_status(self): return {"state":self.state.value,"iteration":self.context.iteration,"max_iterations":self.max_iterations,"has_plan":self.current_plan is not None,"plan_completion":self.current_plan.get_completion_percentage() if self.current_plan else 0,"tool_calls_completed":len(self.completed_tool_calls),"tool_calls_pending":len(self.pending_tool_calls),"actions_taken":len(self.context.actions_taken),"observations":len(self.context.observations),"errors":len(self.context.errors),"duration":time.time()-self.context.start_time}
    def add_event_handler(self,event,handler): self.event_handlers.setdefault(event,[]).append(handler)
    async def _trigger_event(self,event,data):
        for handler in self.event_handlers.get(event,[]):
            try: await handler(data)
            except Exception as exc: logger.error("Event handler failed: %s",exc)
