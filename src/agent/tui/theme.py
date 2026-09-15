"""Terminal theme for Depression.AI — princess CLI aesthetic."""
from __future__ import annotations

from textual.app import App
from textual.theme import Theme as TextualTheme


DEPRESSION_THEME = TextualTheme(
    name="depression",
    primary="#ff9bc7",
    secondary="#b794f6",
    accent="#7ee7ff",
    warning="#ffd28a",
    error="#ff6b8a",
    success="#7ef7c0",
    surface="#0e0d12",
    panel="#131118",
    boost="#1a1820",
    foreground="#e8e3f0",
    background="#08070c",
    dark=True,
    variables={},
)

OPENCODE_THEME = DEPRESSION_THEME

OPENCODE_CSS = """
Screen { background: #08070c; color: #e8e3f0; }
* {
    scrollbar-background: #08070c;
    scrollbar-color: #2a2735;
    scrollbar-corner-color: #08070c;
}
#main-container { height: 1fr; layout: horizontal; }
#chat-panel    { width: 1fr; background: #08070c; }
#sidebar       { width: 38; min-width: 32; background: #0e0d12; border-left: solid #2a2735; }
#status-bar    { height: 1; background: #0e0d12; border-top: solid #2a2735; color: #8a849a; }
Button         { background: transparent; border: none; }
ListView       { background: transparent; border: none; }
ListView > ListItem        { background: transparent; border: none; }
ListView > ListItem:hover  { background: #1a1820; }
Markdown       { background: transparent; }
"""


def apply_opencode_theme(app: App) -> None:
    try:
        app.register_theme(DEPRESSION_THEME)
        app.theme = "depression"
    except Exception:
        pass