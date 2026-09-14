"""Minimal terminal theme for Depression.AI."""
from __future__ import annotations
from textual.app import App
from textual.theme import Theme as TextualTheme

DEPRESSION_THEME = TextualTheme(
    name="depression",
    primary="#75f0a8", secondary="#6bdcff", accent="#c29cff",
    warning="#ffd166", error="#ff7777", success="#75f0a8",
    surface="#0b0b0b", panel="#101010", boost="#181818",
    foreground="#e5e5e5", background="#000000", dark=True,
    variables={
        "bg":"#000000", "bg-panel":"#0b0b0b", "bg-hover":"#171717", "bg-selected":"#1d1d1d",
        "border":"#242424", "border-focused":"#75f0a8", "primary":"#75f0a8", "secondary":"#6bdcff",
        "accent":"#c29cff", "warning":"#ffd166", "error":"#ff7777", "success":"#75f0a8",
        "text":"#e5e5e5", "text-muted":"#777777", "text-dim":"#4d4d4d", "text-bright":"#ffffff",
        "input-bg":"#0b0b0b", "input-border":"#242424", "input-focus":"#75f0a8",
    },
)
OPENCODE_THEME = DEPRESSION_THEME

OPENCODE_CSS = """
Screen { background: #000000; color: #e5e5e5; }
* { scrollbar-background: #000000; scrollbar-color: #303030; scrollbar-corner-color: #000000; }
#main-container { height: 1fr; layout: horizontal; }
#chat-panel { width: 1fr; background: #000000; }
#sidebar { width: 34; min-width: 30; background: #0b0b0b; border-left: solid #242424; }
#status-bar { height: 1; background: #0b0b0b; border-top: solid #242424; color: #777777; }
Button { background: transparent; border: none; }
ListView { background: transparent; border: none; }
ListView > ListItem { background: transparent; border: none; }
ListView > ListItem:hover { background: #171717; }
Markdown { background: transparent; }
"""


def apply_opencode_theme(app: App) -> None:
    """Install the terminal theme without changing application behavior."""
    try:
        app.register_theme(DEPRESSION_THEME)
        app.theme = "depression"
    except Exception:
        pass
