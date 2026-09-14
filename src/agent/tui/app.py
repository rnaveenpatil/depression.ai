"""Depression.AI polished terminal workspace."""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static, Input, ListView, ListItem, Label, Button
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual import on, work
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget
from agent.tui.theme import OPENCODE_CSS, apply_opencode_theme
from agent.tui.views.header import HeaderBar
from agent.tui.views.status_bar import StatusBar
from agent.tui.views.sidebar import Sidebar
from agent.tui.views.chat import ChatView
from agent.tui.llm_providers import LLMModel, get_llm_config, PROVIDER_MODELS

class CommandPalette(ModalScreen[str]):
    DEFAULT_CSS = """
    CommandPalette { align: center middle; background: #000000 55%; }
    #palette-container { width: 64; max-width: 85%; height: auto; max-height: 24; background: $bg-panel; border: round $border-focused; }
    #palette-input { height: 3; background: $input-bg; border: none; border-bottom: solid $border; padding: 0 1; }
    #palette-list { height: auto; max-height: 18; }
    .palette-item { height: 2; padding: 0 2; color: $text; }
    .palette-item:hover { background: $bg-hover; color: $secondary; }
    .shortcut { color: $text-dim; }
    """
    COMMANDS = [("/help","Show help","?"),("/model","Switch model","m"),("/llm","LLM provider panel","l"),("/plan","Switch to Plan mode","1"),("/build","Switch to Build mode","2"),("/auto","Switch to Auto mode","3"),("/session","Manage sessions","s"),("/sessions","List all sessions",""),("/clear","Clear chat","Ctrl+L"),("/tools","List tools","t"),("/config","Show config",""),("/status","Show status",""),("/context","Show context",""),("/compact","Compact context",""),("/export","Export session",""),("/quit","Exit","Ctrl+D")]
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.filtered_commands=list(self.COMMANDS)
    def compose(self):
        with Widget(id="palette-container"):
            yield Input(placeholder="Command...", id="palette-input")
            with ListView(id="palette-list"):
                for c,d,s in self.filtered_commands: yield ListItem(Label(f"{c}  {d}"), Label(s, classes="shortcut"))
    @on(Input.Changed, "#palette-input")
    def _changed(self,event):
        q=event.value.lower(); self.filtered_commands=[x for x in self.COMMANDS if q in x[0].lower() or q in x[1].lower()]
        v=self.query_one("#palette-list",ListView); v.clear()
        for c,d,s in self.filtered_commands: v.append(ListItem(Label(f"{c}  {d}"),Label(s,classes="shortcut")))
    @on(ListView.Selected,"#palette-list")
    def _selected(self,event):
        if 0<=event.index<len(self.filtered_commands): self.dismiss(self.filtered_commands[event.index][0])
    @on(Input.Submitted,"#palette-input")
    def _submitted(self,event):
        if self.filtered_commands: self.dismiss(self.filtered_commands[0][0])
    def on_key(self,event):
        if event.key=="escape": self.dismiss(None)

class HelpScreen(ModalScreen[str]):
    DEFAULT_CSS="""
    HelpScreen { align:center middle; background:#000000 55%; }
    #help-container { width:70; height:auto; max-height:30; background:$bg-panel; border:round $border-focused; padding:1 2; }
    .help-title { text-style:bold; color:$primary; height:1; }
    .help-section { color:$secondary; text-style:bold; height:1; margin:1 0 0 0; }
    .help-row { height:1; color:$text; }
    """
    def compose(self):
        with Widget(id="help-container"):
            yield Static("DEPRESSION.AI  ·  KEYBOARD",classes="help-title")
            yield Static("Navigation",classes="help-section")
            yield Static("[help-key]Tab[/] switch mode    [help-key]Ctrl+P[/] command palette    [help-key]F2[/] sidebar",classes="help-row")
            yield Static("Input",classes="help-section")
            yield Static("[help-key]Enter[/] send    [help-key]Shift+Enter[/] newline    [help-key]Ctrl+C[/] cancel    [help-key]Ctrl+D[/] exit",classes="help-row")
            yield Static("Commands",classes="help-section")
            yield Static("/llm  /model  /plan  /build  /auto  /session  /tools  /context  /compact  /clear",classes="help-row")
    def on_key(self,event): self.dismiss(None)

class DepressionTUI(App):
    TITLE="DEPRESSION.AI"; SUB_TITLE="Agentic Workspace"; CSS=OPENCODE_CSS
    BINDINGS=[Binding("ctrl+p","command_palette","Command Palette",show=True),Binding("ctrl+l","clear_screen","Clear",show=True),Binding("ctrl+d","quit","Exit",show=True),Binding("ctrl+c","cancel","Cancel",show=True),Binding("tab","switch_mode","Switch Mode",show=True),Binding("f1","show_help","Help",show=True),Binding("f2","toggle_sidebar","Sidebar",show=True),Binding("escape","escape","Back",show=False)]
    current_mode=reactive[str]("build"); agent_status=reactive[str]("idle"); model_name=reactive[str]("—"); tokens_used=reactive[int](0); cost=reactive[float](0.0); session_id=reactive[str]("—")
    def __init__(self,agent_coordinator=None,config:dict=None,project_dir:str=None,model_override:str=None,provider_override:str=None,yolo:bool=False,no_sidebar:bool=False,session_id:str=None,**kwargs):
        super().__init__(**kwargs); self.coordinator=agent_coordinator; self.config=config or {}; self.project_dir=project_dir; self.model_override=model_override; self.provider_override=provider_override; self.yolo=yolo; self.no_sidebar=no_sidebar; self.session_id_override=session_id; self.sidebar_visible=not no_sidebar; self._init_task=None
    def compose(self):
        yield HeaderBar(id="header")
        with Widget(id="main-container"):
            yield ChatView(id="chat-panel")
            yield Sidebar(id="sidebar",on_llm_connect=self._on_llm_connect)
        yield StatusBar(id="status-bar")
    def on_mount(self):
        apply_opencode_theme(self); self.title="DEPRESSION.AI"; self.sub_title=f"{self.current_mode.upper()}  /  WORKSPACE"; self._load_sidebar_data(); self.query_one("#sidebar",Sidebar).display=self.sidebar_visible
        chat=self.query_one("#chat-panel",ChatView); chat.add_system("**DEPRESSION.AI**  ·  local + cloud agent\n\nType a task to begin.  **Tab** changes mode · **Ctrl+P** opens commands · **F2** toggles the workspace sidebar."); chat.add_divider()
        self.query_one("#status-bar",StatusBar).update_all(status="idle",model=self.model_name,tokens=self.tokens_used,cost=self.cost)
        self._init_task=self.run_worker(self._init_agent_async(),exclusive=True,group="agent-init",thread=True)
        try:self.query_one("#chat-input",Input).focus()
        except Exception:pass
    def _load_sidebar_data(self):
        sidebar=self.query_one("#sidebar",Sidebar); sidebar.set_sessions([{"id":"current","name":"Current Workspace","time":"now","active":True}]); sidebar.set_tools([{"name":n,"enabled":True} for n in ["terminal","filesystem","git","search","web","patch","browser","task","diagnostics"]]); selected=get_llm_config().get_selected_model_info()
        if selected: self.model_name=f"{selected.provider}/{selected.short_name}"; self.query_one("#header",HeaderBar).set_model(self.model_name); self.query_one("#status-bar",StatusBar).set_model(self.model_name)
    def _on_llm_connect(self,model:LLMModel):
        cfg=get_llm_config(); self.model_name=f"{model.provider}/{model.short_name}"; self.query_one("#header",HeaderBar).set_model(self.model_name); self.query_one("#status-bar",StatusBar).set_model(self.model_name); cfg.selected_model=model.name
        chat=self.query_one("#chat-panel",ChatView); chat.add_system(f"Configured **{model.display_name}** · `{cfg.get_base_url(model.provider)}` · key saved")
        if self.coordinator:self._apply_llm_to_coordinator(model)
    def _apply_llm_to_coordinator(self,model:LLMModel):
        try:
            cfg=get_llm_config(); key=cfg.get_api_key(model.provider); url=cfg.get_base_url(model.provider)
            for agent in [self.coordinator.plan_agent,self.coordinator.build_agent]:
                llm=getattr(agent,"llm",None)
                if llm:
                    if hasattr(llm,"api_key"):llm.api_key=key
                    if hasattr(llm,"base_url"):llm.base_url=url
                    if hasattr(llm,"model"):llm.model=model.name
            try:
                from agent.llm.provider import get_llm_registry
                registry=get_llm_registry(); registry.set_api_key(model.provider,key)
                for provider in getattr(registry,"providers",{}).values():
                    if getattr(provider,"name","")==model.provider or getattr(provider,"name","")=="base":
                        provider.base_url=url
                registry.set_model(model.name)
            except Exception: pass
            self.query_one("#chat-panel",ChatView).add_system(f"Runtime model set to **{model.display_name}**")
        except Exception as e:self.query_one("#chat-panel",ChatView).add_error(f"Failed to update agent: {e}")
    async def action_command_palette(self):
        result=await self.push_screen_wait(CommandPalette());
        if result:await self._execute_command(result)
    async def action_show_help(self):await self.push_screen_wait(HelpScreen())
    def action_clear_screen(self):self.query_one("#chat-panel",ChatView).clear_messages()
    def action_cancel(self):
        if self.agent_status in ("thinking","acting"):self.agent_status="idle"; self.query_one("#status-bar",StatusBar).set_status("idle")
    def action_switch_mode(self):
        modes=["plan","build","auto"]; self.current_mode=modes[(modes.index(self.current_mode)+1)%3] if self.current_mode in modes else "build"; 
        if self.coordinator:self.coordinator.set_mode(self.current_mode)
        self.query_one("#header",HeaderBar).set_mode(self.current_mode); self.sub_title=f"{self.current_mode.upper()}  /  WORKSPACE"; self.query_one("#chat-panel",ChatView).add_system(f"Switched to {self.current_mode.upper()} mode")
    def action_toggle_sidebar(self):self.sidebar_visible=not self.sidebar_visible; self.query_one("#sidebar",Sidebar).display=self.sidebar_visible
    def action_escape(self):self.query_one("#chat-panel",ChatView).clear_input()
    async def _execute_command(self,command:str):
        chat=self.query_one("#chat-panel",ChatView)
        if command=="/quit":self.exit();return
        if command=="/clear":chat.clear_messages();return
        if command=="/help":await self.action_show_help();return
        if command in ("/plan","/build","/auto"):self.current_mode=command[1:];self.query_one("#header",HeaderBar).set_mode(self.current_mode);chat.add_system(f"Switched to {self.current_mode.upper()} mode");return
        if command in ("/model","/llm"):self.query_one("#sidebar",Sidebar).switch_to_llm();chat.add_system("LLM panel opened in the right sidebar.");return
        if command=="/status":chat.add_system(f"Mode: {self.current_mode}\nModel: {self.model_name}\nTokens: {self.tokens_used:,}\nCost: ${self.cost:.4f}\nSession: {self.session_id[:8]}");return
        if command=="/tools":chat.add_system("Tools shown in the right sidebar.");self.query_one("#sidebar",Sidebar).active_tab="tools";return
        if command.startswith("/session"):chat.add_system("Session commands: /session new, /session list, /session load <id>, /session save");return
        chat.add_error(f"Unknown command: {command}")
    @on(Input.Submitted,"#chat-input")
    def _on_input_submitted(self,event):self._send_message()
    def _send_message(self):
        chat=self.query_one("#chat-panel",ChatView); text=chat.get_input_text().strip()
        if text:chat.add_message("user",text);chat.clear_input();self._process_message(text)
    @work(exclusive=True,group="message-processor")
    async def _process_message(self,text:str):
        status=self.query_one("#status-bar",StatusBar);chat=self.query_one("#chat-panel",ChatView);self.agent_status="thinking";status.set_status("thinking")
        try:
            if self.coordinator:
                chat.add_thinking("Analyzing..."); result=await asyncio.wait_for(self.coordinator.process_query(text,mode=self.current_mode,auto_execute=True),timeout=120);chat.remove_thinking()
                if result.get("success"):
                    chat.add_message("agent",result.get("execution") or result.get("response", ""));
                    if "tokens" in result:self.tokens_used=result["tokens"];status.set_tokens(result["tokens"])
                    if "cost" in result:self.cost=result["cost"];status.set_cost(result["cost"])
                else:chat.add_error(result.get("error","Unknown error"))
            else:chat.add_message("agent",f"Received: {text}\n\nConnect an agent coordinator for full functionality.")
        except asyncio.TimeoutError:chat.add_error("Request timed out after 120 seconds.")
        except Exception as e:chat.add_error(f"Error: {e}")
        finally:self.agent_status="idle";status.set_status("idle");status.set_tool("")
    def set_coordinator(self,coordinator):self.coordinator=coordinator
    def set_model(self,model:str):self.model_name=model;self.query_one("#header",HeaderBar).set_model(model);self.query_one("#status-bar",StatusBar).set_model(model)
    def set_session(self,session_id:str):self.session_id=session_id;self.query_one("#header",HeaderBar).set_session(session_id)
    def update_metrics(self,tokens:int=0,cost:float=0.0):self.tokens_used=tokens;self.cost=cost;self.query_one("#status-bar",StatusBar).update_all(tokens=tokens,cost=cost)
    def show_tool_call(self,tool_name:str,params:dict=None,result:Any=None,success:bool=True,duration:float=None):self.query_one("#chat-panel",ChatView).add_tool_call(tool_name,params,result,success,duration)
    def stream_response(self,text:str):self.query_one("#chat-panel",ChatView).append_stream(text)
    async def action_quit(self):
        if self.coordinator:
            try:await self.coordinator.shutdown()
            except Exception:pass
        if hasattr(self,"session_manager") and self.session_manager:
            try:await self.session_manager.save_current_session()
            except Exception:pass
        self.exit()
    async def _init_agent_async(self):
        try:
            from agent.config.loader import load_config
            from agent.utils.platform import get_data_dir,get_cache_dir
            from agent.storage.database import Database
            from agent.storage.cache import Cache
            from agent.project.workspace import WorkspaceManager
            from agent.session.session import SessionManager
            from agent.context.manager import ContextManager
            from agent.llm.provider import get_llm_registry
            from agent.permissions.manager import PermissionManager
            from agent.agent.dual_agent import create_dual_agent_system
            cfg=load_config(cache=True)
            if self.model_override:cfg.setdefault("llm",{})["model"]=self.model_override
            if self.provider_override:cfg.setdefault("llm",{})["provider"]=self.provider_override
            if self.yolo:cfg.setdefault("permissions",{})["auto_approve"]=True
            database=Database(str(get_data_dir("depression")/"agent.db"));await database.initialize();cache=Cache(cache_dir=str(get_cache_dir("depression")));project_dir=self.project_dir or cfg.get("workspace",{}).get("path",".");workspace=WorkspaceManager(workspace_dir=str(get_data_dir("depression")),project_dir=project_dir);await workspace.initialize();session_manager=SessionManager(database=database);session=await session_manager.load_session(self.session_id_override) if self.session_id_override else await session_manager.get_or_create_session();registry=get_llm_registry();context=ContextManager(workspace=workspace,session=session,config=cfg.get("context",{}),llm=registry);await context.initialize();permissions=PermissionManager(config=cfg.get("permissions",{}),ui=None);coordinator=await create_dual_agent_system(config=cfg,session=session,context_manager=context,permission_manager=permissions,workspace=workspace,database=database,ui=None,cache=cache);self.call_from_thread(self._on_agent_ready,coordinator,session,workspace,session_manager)
        except Exception as e:self.call_from_thread(self._on_agent_error,str(e))
    def _on_agent_ready(self,coordinator,session,workspace,session_manager):
        self.coordinator=coordinator;self.session_manager=session_manager;self.workspace=workspace;status=coordinator.get_status();self.set_model(f"Plan: {status.get('plan_agent',{}).get('model','—')} | Build: {status.get('build_agent',{}).get('model','—')}");self.set_session(session.id);sidebar=self.query_one("#sidebar",Sidebar)
        try:sidebar.set_tools([{"name":n,"enabled":True} for n in coordinator.plan_agent.tool_registry.list_tools()])
        except Exception:pass
        try:sidebar.set_files([{ "name":str(f.relative_to(workspace.get_project_dir())),"is_dir":f.is_dir(),"size":self._human_size(f.stat().st_size) if f.is_file() else ""} for f in workspace.list_files()][:100])
        except Exception:pass
        self._load_sessions_list();self.query_one("#chat-panel",ChatView).add_system(f"Agent connected! Session: {session.id[:8]}");self.query_one("#chat-panel",ChatView).add_divider()
    def _on_agent_error(self,error:str):self.query_one("#chat-panel",ChatView).add_error(f"Agent initialization failed: {error}\nRunning in demo mode.");self.query_one("#chat-panel",ChatView).add_divider()
    def _load_sessions_list(self):
        try:
            if hasattr(self,"session_manager") and self.session_manager:self.query_one("#sidebar",Sidebar).set_sessions([{"id":"current","name":"Current Session","time":"now","active":True}])
        except Exception:pass
    @staticmethod
    def _human_size(size:int)->str:
        for unit in ["B","KB","MB","GB"]:
            if size<1024:return f"{size:.0f}{unit}"
            size/=1024
        return f"{size:.1f}TB"
    def action_toggle_llm_panel(self):self.query_one("#sidebar",Sidebar).switch_to_llm()
