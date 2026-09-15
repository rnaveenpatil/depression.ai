"""Terminal-native chat view — transcript above, single-line prompt below."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Input, Markdown
from textual.containers import Container, Horizontal

from agent.tui.widgets.tool_card import ToolCard


class ThinkingIndicator(Widget):
    DEFAULT_CSS = "ThinkingIndicator { height: 1; padding: 0 2; color: #b794f6; }"

    def compose(self) -> ComposeResult:
        yield Static("[#b794f6]◆[/] [#8a849a]thinking…[/]")


class MessageBubble(Widget):
    DEFAULT_CSS = """
    MessageBubble { height: auto; min-height: 1; margin: 0 0 1 0; padding: 0 2; }
    MessageBubble Markdown { padding: 0 0 0 2; }
    """

    def __init__(self, role: str, content: str,
                 timestamp: Optional[datetime] = None, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()

    def compose(self) -> ComposeResult:
        if self.role == "user":
            t = Text()
            t.append("❯ ", style="bold #ff9bc7")
            t.append(self.content, style="bold #e8e3f0")
            yield Static(t)
        elif self.role == "agent":
            yield Static("[bold #7ef7c0]◆[/] [bold #ff9bc7]Depression[/]")
            yield Markdown(self.content)
        elif self.role == "error":
            yield Static(f"[bold #ff6b8a]× {self.content}[/]")
        else:
            yield Static(f"[#8a849a]· {self.content}[/]")


class ChatView(Widget):
    DEFAULT_CSS = """
    ChatView { height: 1fr; background: #08070c; layout: vertical; }

    #chat-scroll { height: 1fr; overflow-y: auto; padding: 1 0 0 0; }

    #chat-input-area {
        height: 3;
        background: #08070c;
        border-top: solid #2a2735;
        layout: horizontal;
        padding: 0 2;
    }
    #input-prompt-label {
        width: 3;
        color: #ff9bc7;
        text-style: bold;
        content-align: left middle;
    }
    #chat-input {
        width: 1fr;
        height: 3;
        background: transparent;
        border: none;
        color: #e8e3f0;
        padding: 0;
    }
    #chat-input:focus { border: none; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[Dict[str, Any]] = []
        self._streaming = False
        self._stream_buffer = ""
        self._stream_bubble = None
        self._thinking_indicator = None

    def compose(self) -> ComposeResult:
        with Container(id="chat-scroll"):
            pass
        with Horizontal(id="chat-input-area"):
            yield Static("❯", id="input-prompt-label")
            yield Input(placeholder="ask Depression.AI anything…", id="chat-input")

    def add_message(self, role: str, content: str,
                    timestamp: Optional[datetime] = None) -> None:
        self.messages.append({"role": role, "content": content,
                              "timestamp": timestamp or datetime.now()})
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(MessageBubble(role, content, timestamp))
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def add_tool_call(self, tool_name: str,
                      params: Optional[Dict[str, Any]] = None,
                      result: Optional[Any] = None,
                      success: bool = True,
                      duration: Optional[float] = None) -> None:
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(ToolCard(tool_name=tool_name, params=params,
                                  result=result, success=success,
                                  duration=duration))
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def add_system(self, content: str) -> None:
        self.add_message("system", content)

    def add_error(self, content: str) -> None:
        self.add_message("error", content)

    def add_divider(self) -> None:
        try:
            self.query_one("#chat-scroll").mount(
                Static("[#2a2735]" + "─" * 60 + "[/]"))
        except Exception:
            pass

    def add_thinking(self, content: str = "") -> None:
        try:
            self._thinking_indicator = ThinkingIndicator()
            scroll = self.query_one("#chat-scroll")
            scroll.mount(self._thinking_indicator)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def remove_thinking(self) -> None:
        if self._thinking_indicator:
            try:
                self._thinking_indicator.remove()
            except Exception:
                pass
            self._thinking_indicator = None

    def start_streaming(self) -> None:
        self._streaming = True
        self._stream_buffer = ""
        self.remove_thinking()
        self.add_message("agent", "")
        try:
            scroll = self.query_one("#chat-scroll")
            for child in reversed(list(scroll.children)):
                if isinstance(child, MessageBubble) and child.role == "agent":
                    self._stream_bubble = child
                    break
        except Exception:
            pass

    def append_stream(self, text: str) -> None:
        if not self._streaming or self._stream_bubble is None:
            return
        self._stream_buffer += text
        try:
            self._stream_bubble.remove_children()
            self._stream_bubble.mount(
                Static("[bold #7ef7c0]◆[/] [bold #ff9bc7]Depression[/]"))
            self._stream_bubble.mount(Markdown(self._stream_buffer))
            self.query_one("#chat-scroll").scroll_end(animate=False)
        except Exception:
            pass

    def end_streaming(self) -> str:
        self._streaming = False
        final = self._stream_buffer
        self._stream_buffer = ""
        self._stream_bubble = None
        return final

    def clear_messages(self) -> None:
        self.messages.clear()
        try:
            self.query_one("#chat-scroll").remove_children()
        except Exception:
            pass

    def get_input_text(self) -> str:
        try:
            return self.query_one("#chat-input", Input).value
        except Exception:
            return ""

    def clear_input(self) -> None:
        try:
            self.query_one("#chat-input", Input).value = ""
        except Exception:
            pass

    def set_input(self, text: str) -> None:
        try:
            self.query_one("#chat-input", Input).value = text
        except Exception:
            pass