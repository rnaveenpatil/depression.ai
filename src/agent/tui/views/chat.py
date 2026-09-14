"""Terminal-first chat view for Depression.AI."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Input, Markdown
from textual.containers import Container
from agent.tui.widgets.tool_card import ToolCard

BG = "#000000"; PANEL = "#0b0b0b"; BORDER = "#242424"; TEXT = "#e5e5e5"; MUTED = "#777777"
GREEN = "#75f0a8"; CYAN = "#6bdcff"; VIOLET = "#c29cff"; RED = "#ff7777"


class ThinkingIndicator(Widget):
    DEFAULT_CSS = f"ThinkingIndicator {{ height: 1; padding: 0 1; color: {VIOLET}; }}"
    def compose(self) -> ComposeResult:
        yield Static("· thinking…")


class MessageBubble(Widget):
    DEFAULT_CSS = f"""
    MessageBubble {{ height: auto; min-height: 1; margin: 0 0 1 0; padding: 0 1; }}
    .msg-header {{ height: 1; text-style: bold; }}
    .msg-body {{ padding: 0 2; height: auto; }}
    .role-user .msg-header {{ color: {CYAN}; }}
    .role-agent .msg-header {{ color: {GREEN}; }}
    .role-system .msg-header {{ color: {MUTED}; }}
    .role-error .msg-header {{ color: {RED}; }}
    """
    NAMES = {"user": "You", "agent": "Depression", "system": "System", "error": "Error"}
    ICONS = {"user": "❯", "agent": "◆", "system": "·", "error": "×"}
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None, **kwargs):
        super().__init__(**kwargs); self.role = role; self.content = content; self.timestamp = timestamp or datetime.now(); self.add_class(f"role-{role}")
    def compose(self) -> ComposeResult:
        yield Static(f"{self.ICONS.get(self.role, '·')} {self.NAMES.get(self.role, self.role.title())}", classes="msg-header")
        if self.role in ("agent", "system"):
            yield Markdown(self.content, classes="msg-body")
        else:
            yield Static(self.content, classes="msg-body")


class ChatView(Widget):
    """Main terminal conversation: transcript above, single prompt below."""
    DEFAULT_CSS = f"""
    ChatView {{ height: 1fr; background: {BG}; layout: vertical; }}
    #chat-scroll {{ height: 1fr; overflow-y: auto; padding: 1 2; }}
    #chat-input-area {{ height: 3; background: {PANEL}; border-top: solid {BORDER}; layout: horizontal; padding: 0 2; }}
    #input-prompt-label {{ width: 2; color: {GREEN}; text-style: bold; content-align: left middle; margin: 1 0; }}
    #chat-input {{ width: 1fr; height: 1; background: transparent; border: none; color: {TEXT}; margin: 1 0; padding: 0; }}
    #chat-input:focus {{ border: none; }}
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.messages: List[Dict[str, Any]] = []; self._streaming = False; self._stream_buffer = ""; self._stream_bubble = None; self._thinking_indicator = None
    def compose(self) -> ComposeResult:
        with Container(id="chat-scroll"): pass
        with Widget(id="chat-input-area"):
            yield Static("❯", id="input-prompt-label")
            yield Input(placeholder="Ask Depression.AI anything…", id="chat-input")
    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> None:
        self.messages.append({"role": role, "content": content, "timestamp": timestamp or datetime.now()})
        try:
            scroll = self.query_one("#chat-scroll"); scroll.mount(MessageBubble(role, content, timestamp)); scroll.scroll_end(animate=False)
        except Exception: pass
    def add_tool_call(self, tool_name: str, params: Optional[Dict[str, Any]] = None, result: Optional[Any] = None, success: bool = True, duration: Optional[float] = None) -> None:
        try:
            scroll = self.query_one("#chat-scroll"); scroll.mount(ToolCard(tool_name=tool_name, params=params, result=result, success=success, duration=duration)); scroll.scroll_end(animate=False)
        except Exception: pass
    def add_system(self, content: str) -> None: self.add_message("system", content)
    def add_error(self, content: str) -> None: self.add_message("error", content)
    def add_thinking(self, content: str = "") -> None:
        try:
            self._thinking_indicator = ThinkingIndicator(); scroll = self.query_one("#chat-scroll"); scroll.mount(self._thinking_indicator); scroll.scroll_end(animate=False)
        except Exception: pass
    def remove_thinking(self) -> None:
        if self._thinking_indicator:
            try: self._thinking_indicator.remove()
            except Exception: pass
            self._thinking_indicator = None
    def add_divider(self) -> None:
        try: self.query_one("#chat-scroll").mount(Static("─" * 48))
        except Exception: pass
    def start_streaming(self) -> None:
        self._streaming = True; self._stream_buffer = ""; self._stream_bubble = None; self.remove_thinking(); self.add_message("agent", "")
        try:
            scroll = self.query_one("#chat-scroll")
            for child in reversed(list(scroll.children)):
                if isinstance(child, MessageBubble) and child.role == "agent": self._stream_bubble = child; break
        except Exception: pass
    def append_stream(self, text: str) -> None:
        if not self._streaming or self._stream_bubble is None: return
        self._stream_buffer += text
        try:
            self._stream_bubble.remove_children(); self._stream_bubble.mount(Static("◆ Depression", classes="msg-header")); self._stream_bubble.mount(Markdown(self._stream_buffer, classes="msg-body")); self.query_one("#chat-scroll").scroll_end(animate=False)
        except Exception: pass
    def end_streaming(self) -> str:
        self._streaming = False; final = self._stream_buffer; self._stream_buffer = ""; self._stream_bubble = None; return final
    def clear_messages(self) -> None:
        self.messages.clear()
        try: self.query_one("#chat-scroll").remove_children()
        except Exception: pass
    def get_input_text(self) -> str:
        try: return self.query_one("#chat-input", Input).value
        except Exception: return ""
    def clear_input(self) -> None:
        try: self.query_one("#chat-input", Input).value = ""
        except Exception: pass
    def set_input(self, text: str) -> None:
        try: self.query_one("#chat-input", Input).value = text
        except Exception: pass
