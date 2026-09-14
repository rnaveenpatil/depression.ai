"""
Chat View

The main chat interface with:
- Message history with role-colored headers
- Streaming text display
- Tool call visualization
- Thinking/reasoning display with claude-code style animation
- Auto-scroll to bottom
- Message selection for copy
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static, Input, Label, Markdown
from textual.reactive import reactive
from textual import on, work
from textual.containers import Container

from agent.tui.widgets.tool_card import ToolCard


# Inline colors (opencode style)
BG = "#000000"
BG_PANEL = "#1a1a1a"
BORDER = "#333333"
PRIMARY = "#00ff80"
SECONDARY = "#00ccff"
ACCENT = "#ffcc00"
TEXT = "#e0e0e0"
TEXT_MUTED = "#888888"
TEXT_DIM = "#555555"
TEXT_BRIGHT = "#ffffff"
ROLE_USER = "#00ccff"
ROLE_AGENT = "#00ff80"
ROLE_TOOL = "#ffcc00"
ROLE_SYSTEM = "#ff8800"
ROLE_ERROR = "#ff3333"
ROLE_THINKING = "#aa88ff"


class ThinkingIndicator(Widget):
    """Claude-code style thinking indicator with animated dots."""
    
    DEFAULT_CSS = f"""
    ThinkingIndicator {{
        height: 1;
        padding: 0 0 0 2;
        color: {ROLE_THINKING};
    }}
    
    .thinking-dots {{
        height: 1;
        layout: horizontal;
    }}
    
    .dot {{
        width: 6;
        height: 1;
        background: {ROLE_THINKING};
        margin: 0 1;
    }}
    
    .dot-0 {{ animation: pulse 1.4s ease-in-out infinite; }}
    .dot-1 {{ animation: pulse 1.4s ease-in-out infinite 0.2s; }}
    .dot-2 {{ animation: pulse 1.4s ease-in-out infinite 0.4s; }}
    
    @keyframes pulse {{
        0%, 80%, 100% {{ opacity: 0.3; background: {ROLE_THINKING}; }}
        40% {{ opacity: 1; background: {TEXT_BRIGHT}; }}
    }}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._animation_task = None

    def compose(self) -> ComposeResult:
        yield Static("●", classes="dot dot-0")
        yield Static("●", classes="dot dot-1")
        yield Static("●", classes="dot dot-2")

    def on_mount(self) -> None:
        self._animation_task = self.set_interval(0.15, self._animate)

    def on_unmount(self) -> None:
        if self._animation_task:
            self._animation_task.stop()

    def _animate(self) -> None:
        """Trigger CSS animation by toggling a class."""
        pass  # CSS handles the animation


class MessageBubble(Widget):
    """A single message bubble."""

    DEFAULT_CSS = f"""
    MessageBubble {{
        height: auto;
        min-height: 1;
        margin: 0 0 1 0;
        padding: 0 1;
    }}

    .msg-header {{
        height: 1;
        padding: 0 0 0 1;
        text-style: bold;
    }}

    .msg-body {{
        padding: 0 0 0 2;
        height: auto;
    }}

    .msg-timestamp {{
        color: {TEXT_DIM};
        text-style: italic;
    }}

    /* Role colors */
    .role-user .msg-header {{
        color: {ROLE_USER};
    }}
    .role-agent .msg-header {{
        color: {ROLE_AGENT};
    }}
    .role-tool .msg-header {{
        color: {ROLE_TOOL};
    }}
    .role-system .msg-header {{
        color: {ROLE_SYSTEM};
    }}
    .role-error .msg-header {{
        color: {ROLE_ERROR};
    }}
    .role-thinking .msg-header {{
        color: {ROLE_THINKING};
    }}
    """

    ROLE_ICONS = {
        "user": "❯",
        "agent": "◆",
        "tool": "⚙",
        "system": "ℹ",
        "error": "✗",
        "thinking": "∴",
    }

    ROLE_NAMES = {
        "user": "You",
        "agent": "Agent",
        "tool": "Tool",
        "system": "System",
        "error": "Error",
        "thinking": "Thinking",
    }

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.add_class(f"role-{role}")

    def compose(self) -> ComposeResult:
        icon = self.ROLE_ICONS.get(self.role, "·")
        name = self.ROLE_NAMES.get(self.role, self.role.title())
        time_str = self.timestamp.strftime("%H:%M")

        yield Static(
            f" {icon} {name}  [msg-timestamp]{time_str}[/]",
            classes="msg-header",
        )

        # Use Markdown widget for agent/system messages, plain text for others
        if self.role in ("agent", "system"):
            yield Markdown(self.content, classes="msg-body")
        else:
            for line in self.content.split("\n") or [""]:
                yield Static(f" {line}", classes="msg-body")


class ChatView(Widget):
    """The main chat view - opencode style."""

    DEFAULT_CSS = f"""
    ChatView {{
        height: 1fr;
        background: {BG};
        layout: vertical;
    }}

    #chat-scroll {{
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }}

    #chat-input-area {{
        height: 3;
        background: {BG_PANEL};
        border-top: solid {BORDER};
        layout: horizontal;
        padding: 0 1;
    }}

    #input-prompt-label {{
        width: 2;
        height: 1;
        content-align: center middle;
        color: {PRIMARY};
        text-style: bold;
        margin: 1 0 1 0;
    }}

    #chat-input {{
        width: 1fr;
        height: 1;
        background: transparent;
        border: none;
        color: {TEXT};
        padding: 0;
        margin: 1 0 1 0;
    }}

    #chat-input:focus {{
        border: none;
    }}
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

        with Widget(id="chat-input-area"):
            yield Static(">", id="input-prompt-label")
            yield Input(
                placeholder="Type a message... (Enter to send, / for commands)",
                id="chat-input",
            )

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None) -> None:
        """Add a message to the chat."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": timestamp or datetime.now(),
        }
        self.messages.append(msg)

        bubble = MessageBubble(role, content, timestamp)
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(bubble)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def add_tool_call(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        duration: Optional[float] = None,
    ) -> None:
        """Add a tool call card to the chat."""
        card = ToolCard(
            tool_name=tool_name,
            params=params,
            result=result,
            success=success,
            duration=duration,
        )
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(card)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.add_message("system", content)

    def add_error(self, content: str) -> None:
        """Add an error message."""
        self.add_message("error", content)

    def add_thinking(self, content: str = "") -> None:
        """Add a claude-code style thinking indicator with animated dots."""
        try:
            scroll = self.query_one("#chat-scroll")
            self._thinking_indicator = ThinkingIndicator()
            scroll.mount(self._thinking_indicator)
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def remove_thinking(self) -> None:
        """Remove the thinking indicator."""
        if self._thinking_indicator:
            try:
                self._thinking_indicator.remove()
            except Exception:
                pass
            self._thinking_indicator = None

    def add_divider(self) -> None:
        """Add a visual divider."""
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.mount(Static("─" * 60, classes="msg-divider"))
        except Exception:
            pass

    def start_streaming(self) -> None:
        """Start streaming a response."""
        self._streaming = True
        self._stream_buffer = ""
        self._stream_bubble = None
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
        """Append text to the streaming response."""
        if not self._streaming or self._stream_bubble is None:
            return
        self._stream_buffer += text
        try:
            self._stream_bubble.remove_children()
            self._stream_bubble.mount(
                Static(f" ◆ Agent  {datetime.now().strftime('%H:%M')}", classes="msg-header"),
            )
            self._stream_bubble.mount(Markdown(self._stream_buffer, classes="msg-body"))
            scroll = self.query_one("#chat-scroll")
            scroll.scroll_end(animate=False)
        except Exception:
            pass

    def end_streaming(self) -> str:
        """End the streaming response."""
        self._streaming = False
        final_content = self._stream_buffer
        self._stream_buffer = ""
        self._stream_bubble = None
        return final_content

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        try:
            scroll = self.query_one("#chat-scroll")
            scroll.remove_children()
        except Exception:
            pass

    def get_input_text(self) -> str:
        """Get the current input text."""
        try:
            return self.query_one("#chat-input", Input).value
        except Exception:
            return ""

    def clear_input(self) -> None:
        """Clear the input area."""
        try:
            self.query_one("#chat-input", Input).value = ""
        except Exception:
            pass

    def set_input(self, text: str) -> None:
        """Set the input text."""
        try:
            self.query_one("#chat-input", Input).value = text
        except Exception:
            pass