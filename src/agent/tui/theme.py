"""
Depression.AI TUI - Opencode-style Theme

Single cyberpunk theme matching opencode aesthetic with depression.ai branding.
"""

from __future__ import annotations

from textual.theme import Theme as TextualTheme
from textual.app import App


# ======================================================================
# OPENCODE-STYLE CYBERPUNK THEME
# ======================================================================

OPENCODE_THEME = TextualTheme(
    name="opencode",
    primary="#00ff80",      # Matrix green - primary accent
    secondary="#00ccff",    # Cyan - secondary
    accent="#ffcc00",       # Gold - accent
    warning="#ff8800",      # Orange
    error="#ff3333",        # Red
    success="#00ff80",      # Green
    surface="#0d0d0d",      # Near black
    panel="#1a1a1a",        # Dark panel
    boost="#2a2a2a",        # Hover state
    foreground="#e0e0e0",   # Main text
    background="#000000",   # Pure black
    dark=True,
    variables={
        # Base colors
        "bg": "#000000",
        "bg-panel": "#1a1a1a",
        "bg-hover": "#2a2a2a",
        "bg-selected": "#1a3a2a",
        "border": "#333333",
        "border-focused": "#00ff80",
        
        # Accent colors (opencode style)
        "primary": "#00ff80",
        "secondary": "#00ccff",
        "accent": "#ffcc00",
        "warning": "#ff8800",
        "error": "#ff3333",
        "success": "#00ff80",
        
        # Text colors
        "text": "#e0e0e0",
        "text-muted": "#888888",
        "text-dim": "#555555",
        "text-bright": "#ffffff",
        
        # Role colors
        "role-user": "#00ccff",
        "role-agent": "#00ff80",
        "role-tool": "#ffcc00",
        "role-system": "#ff8800",
        "role-error": "#ff3333",
        "role-thinking": "#aa88ff",
        
        # Status colors
        "status-idle": "#888888",
        "status-thinking": "#aa88ff",
        "status-working": "#00ff80",
        "status-success": "#00ff80",
        "status-error": "#ff3333",
        
        # Sidebar
        "sidebar-bg": "#0a0a0a",
        "sidebar-border": "#222222",
        
        # Input
        "input-bg": "#1a1a1a",
        "input-border": "#333333",
        "input-focus": "#00ff80",
    }
)


# ======================================================================
# CSS - OPENCODE STYLE (with inline colors)
# ======================================================================

OPENCODE_CSS = """
/* ============================================
   DEPRESSION.AI - OPENCODE STYLE TUI
   ============================================ */

/* ---- GLOBAL ---- */
Screen {
    background: #000000;
    color: #e0e0e0;
    layers: base overlay;
}

* {
    scrollbar-background: #000000;
    scrollbar-color: #333333;
    scrollbar-corner-color: #000000;
}

* :focus {
    border: none;
}

/* ---- HEADER BAR ---- */
#header {
    height: 1;
    background: #1a1a1a;
    border-bottom: solid #333333;
    dock: top;
    padding: 0 1;
    layout: horizontal;
}

#header-left {
    width: auto;
    height: 1;
    content-align: left middle;
    layout: horizontal;
}

#header-title {
    color: #00ff80;
    text-style: bold;
    padding: 0 1;
}

#header-mode {
    width: auto;
    height: 1;
    content-align: center middle;
    text-style: bold;
    padding: 0 1;
    color: #00ff80;
    background: #1a3a2a;
}

#header-model {
    width: 1fr;
    height: 1;
    content-align: center middle;
    color: #888888;
    padding: 0 1;
}

#header-right {
    width: auto;
    height: 1;
    content-align: right middle;
    layout: horizontal;
}

#header-session {
    color: #555555;
    padding: 0 1;
}

#header-time {
    color: #555555;
    padding: 0 1;
}

/* ---- MAIN LAYOUT ---- */
#main-container {
    height: 1fr;
    layout: horizontal;
}

/* ---- SIDEBAR ---- */
#sidebar {
    width: 30;
    background: #0a0a0a;
    border-right: solid #222222;
    layout: vertical;
}

#sidebar-tabs {
    height: 1;
    layout: horizontal;
    background: #1a1a1a;
    border-bottom: solid #333333;
}

.sidebar-tab-btn {
    width: 1fr;
    height: 1;
    background: transparent;
    border: none;
    color: #888888;
    content-align: center middle;
    text-style: bold;
}

.sidebar-tab-btn:hover {
    background: #2a2a2a;
    color: #e0e0e0;
}

.sidebar-tab-btn.active {
    background: #1a3a2a;
    color: #00ff80;
    border-bottom: solid #00ff80;
}

#sidebar-content {
    height: 1fr;
    overflow-y: auto;
    padding: 0 1;
}

#sidebar-sessions { height: auto; }
#sidebar-tools { height: auto; display: none; }
#sidebar-files { height: auto; display: none; }
#sidebar-llm { height: auto; display: none; }

/* Session items */
.session-item {
    height: 2;
    padding: 0 1;
    background: transparent;
    color: #e0e0e0;
    border: none;
    margin: 0;
}
.session-item:hover { background: #2a2a2a; }
.session-item.active { background: #1a3a2a; color: #00ff80; }
.session-name { width: 1fr; overflow: hidden; text-style: bold; }
.session-meta { width: auto; color: #555555; }

/* Tool items */
.tool-item {
    height: 1;
    padding: 0 1;
    color: #888888;
    layout: horizontal;
}
.tool-item:hover { background: #2a2a2a; color: #00ff80; }
.tool-icon { width: 2; color: #ffcc00; }
.tool-name { width: 1fr; color: #e0e0e0; }
.tool-status { width: auto; color: #00ff80; }

/* File items */
.file-item {
    height: 1;
    padding: 0 1;
    layout: horizontal;
}
.file-item:hover { background: #2a2a2a; }
.file-icon { width: 2; color: #00ccff; }
.file-name { width: 1fr; color: #e0e0e0; }
.file-size { width: auto; color: #555555; }

/* ---- CHAT PANEL ---- */
#chat-panel {
    width: 1fr;
    background: #000000;
    layout: vertical;
}

#chat-messages {
    height: 1fr;
    overflow-y: auto;
    padding: 1;
}

/* Message bubbles - opencode style */
.message {
    height: auto;
    min-height: 1;
    margin: 0 0 1 0;
    padding: 0;
}

.msg-header {
    height: 1;
    padding: 0 0 0 0;
}

.msg-body {
    padding: 0 0 0 2;
    height: auto;
    color: #e0e0e0;
}

.msg-timestamp { color: #555555; text-style: italic; }

/* Role-based header colors */
.role-user .msg-header { color: #00ccff; }
.role-agent .msg-header { color: #00ff80; }
.role-tool .msg-header { color: #ffcc00; }
.role-system .msg-header { color: #ff8800; }
.role-error .msg-header { color: #ff3333; }
.role-thinking .msg-header { color: #aa88ff; }

/* Markdown styling - opencode style */
.message Markdown {
    color: #e0e0e0;
}

.message .md-h1 { color: #00ff80; text-style: bold; margin: 1 0 0 0; }
.message .md-h2 { color: #00ccff; text-style: bold; margin: 1 0 0 0; }
.message .md-h3 { color: #ffcc00; text-style: bold; }
.message .md-paragraph { color: #e0e0e0; }
.message .md-code-block { 
    background: #1a1a1a; 
    border: solid #333333; 
    padding: 1; 
    color: #ffcc00; 
    margin: 0 0 1 0; 
}
.message .md-code-inline { background: #1a1a1a; color: #ffcc00; padding: 0 1; }
.message .md-blockquote { border-left: solid #00ff80; color: #888888; padding: 0 0 0 1; }
.message .md-list-item { color: #e0e0e0; }
.message .md-list-bullet { color: #00ff80; }
.message .md-divider { color: #333333; }
.message .md-link { color: #00ccff; text-style: underline; }
.message .md-strong { text-style: bold; color: #ffffff; }
.message .md-emphasis { text-style: italic; }

/* ---- INPUT AREA ---- */
#chat-input-area {
    height: 3;
    background: #1a1a1a;
    border-top: solid #333333;
    layout: horizontal;
    padding: 0 1;
}

#input-prompt {
    width: 2;
    height: 1;
    content-align: center middle;
    color: #00ff80;
    text-style: bold;
    margin: 1 0 1 0;
}

#chat-input {
    width: 1fr;
    height: 1;
    background: transparent;
    border: none;
    color: #e0e0e0;
    padding: 0;
    margin: 1 0 1 0;
}

#chat-input:focus { border: none; }

/* ---- STATUS BAR ---- */
#status-bar {
    height: 1;
    background: #000000;
    dock: bottom;
    padding: 0 1;
    layout: horizontal;
    border-top: solid #333333;
}

#status-left {
    width: auto;
    height: 1;
    content-align: left middle;
    layout: horizontal;
}

#status-indicator {
    height: 1;
    content-align: center middle;
    padding: 0 1;
}

#status-model {
    width: 1fr;
    height: 1;
    content-align: center middle;
    color: #888888;
}

#status-right {
    width: auto;
    height: 1;
    content-align: right middle;
    layout: horizontal;
}

#status-tokens {
    height: 1;
    content-align: center middle;
    color: #555555;
    padding: 0 1;
}

#status-cost {
    height: 1;
    content-align: center middle;
    color: #555555;
    padding: 0 1;
}

#status-shortcuts {
    height: 1;
    content-align: right middle;
    color: #555555;
    padding: 0 1;
}

/* ---- COMMAND PALETTE ---- */
#command-palette {
    width: 70;
    max-width: 80%;
    height: auto;
    max-height: 24;
    background: #1a1a1a;
    border: solid #00ff80;
    layer: overlay;
    align: center middle;
    padding: 0;
}

#palette-input {
    height: 3;
    background: #1a1a1a;
    border-bottom: solid #00ff80;
    padding: 0 1;
    color: #e0e0e0;
}

#palette-list { height: auto; max-height: 18; overflow-y: auto; }
.palette-item { height: 1; padding: 0 1; color: #e0e0e0; }
.palette-item:hover { background: #2a2a2a; color: #00ff80; }
.palette-item .shortcut { color: #555555; width: auto; text-align: right; }

/* ---- HELP SCREEN ---- */
#help-container {
    width: 70;
    height: auto;
    max-height: 30;
    background: #1a1a1a;
    border: solid #00ff80;
    padding: 1 2;
    layer: overlay;
    align: center middle;
}

.help-title { text-style: bold; color: #00ff80; margin: 0 0 1 0; }
.help-section { color: #00ccff; text-style: bold; margin: 1 0 0 0; }
.help-row { height: 1; color: #e0e0e0; }
.help-key { color: #ffcc00; text-style: bold; }
.help-desc { color: #888888; }

/* ---- TOOL EXECUTION ---- */
.tool-call { color: #ffcc00; text-style: bold; }
.tool-success { color: #00ff80; }
.tool-error { color: #ff3333; }
.tool-duration { color: #555555; }

/* ---- THINKING ANIMATION (claude-code style) ---- */
.thinking-dots {
    height: 1;
    color: #aa88ff;
    content-align: left middle;
    padding: 0 0 0 2;
    layout: horizontal;
}

.thinking-dots .dot {
    width: 6;
    height: 1;
    background: #aa88ff;
    margin: 0 1;
}

/* Animation is handled by ThinkingIndicator widget via set_interval */

/* ---- DIFF VIEWER ---- */
.diff-file-header { background: #1a1a1a; color: #00ff80; text-style: bold; height: 1; padding: 0 1; }
.diff-hunk-header { color: #00ccff; height: 1; padding: 0 1; }
.diff-added { background: #0a2a0a; color: #00ff80; height: 1; padding: 0 1; }
.diff-removed { background: #2a0a0a; color: #ff3333; height: 1; padding: 0 1; }
.diff-context { color: #888888; height: 1; padding: 0 1; }

/* ---- BUTTONS ---- */
.neon-btn {
    background: #1a1a1a;
    border: solid #333333;
    color: #e0e0e0;
    padding: 0 2;
    height: 3;
    min-width: 10;
    content-align: center middle;
    text-style: bold;
}
.neon-btn:hover { background: #2a2a2a; border: solid #00ff80; color: #00ff80; }
.neon-btn:focus { border: solid #00ff80; color: #00ff80; }
.neon-btn.primary { border: solid #00ff80; color: #00ff80; }
.neon-btn.success { border: solid #00ff80; color: #00ff80; }
.neon-btn.danger { border: solid #ff3333; color: #ff3333; }
.neon-btn.warning { border: solid #ff8800; color: #ff8800; }

/* ---- SPINNER ---- */
.spinner { height: 1; width: auto; color: #00ff80; }

/* ---- LLM PANEL ---- */
#llm-header { height: 1; background: #1a1a1a; color: #00ff80; text-style: bold; padding: 0 1; border-bottom: solid #333333; }
#llm-model-list { height: 1fr; overflow-y: auto; }
#llm-config-area { height: auto; padding: 0 1; border-top: solid #333333; }
.llm-label { height: 1; color: #888888; text-style: bold; }
#llm-base-url { height: 1; background: #1a1a1a; color: #00ccff; padding: 0 1; border: solid #333333; }
#llm-api-key-input { height: 1; background: #1a1a1a; color: #e0e0e0; padding: 0 1; border: solid #333333; }
#llm-connect-btn { height: 3; background: #1a1a1a; border: solid #00ff80; color: #00ff80; text-style: bold; content-align: center middle; margin: 0 1; }
#llm-connect-btn:hover { background: #2a2a2a; border: solid #00ff80; color: #00ff80; }
#llm-status { height: 1; color: #555555; padding: 0 1; }
.status-connected { color: #00ff80; }
.status-disconnected { color: #555555; }
.status-error { color: #ff3333; }

/* Model list items */
.model-item { height: 2; padding: 0 1; border-bottom: solid #333333; }
.model-item:hover { background: #2a2a2a; }
.model-item.selected { background: #1a3a2a; border-left: solid #00ff80; }
.model-name { color: #e0e0e0; text-style: bold; height: 1; }
.model-meta { height: 1; color: #555555; }
.provider-header { height: 1; padding: 0 1; background: #1a1a1a; text-style: bold; color: #00ccff; }
"""


# ======================================================================
# THEME APPLICATION
# ======================================================================

def apply_opencode_theme(app: App) -> None:
    """Apply the opencode-style theme to the app."""
    app.register_theme(OPENCODE_THEME)
    app.theme = "opencode"