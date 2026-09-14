"""Depression.AI futuristic glass/neon Textual theme.

The visual direction is intentionally closer to a polished web application
than a traditional terminal: deep space surfaces, restrained neon accents,
clear hierarchy, generous spacing, and focused interaction states.
"""

from __future__ import annotations

from textual.app import App
from textual.theme import Theme as TextualTheme


DEPRESSION_THEME = TextualTheme(
    name="depression",
    primary="#8f7cff",
    secondary="#54d9ff",
    accent="#ff8bdc",
    warning="#ffc857",
    error="#ff6b8a",
    success="#62f5b0",
    surface="#11111d",
    panel="#171729",
    boost="#23233d",
    foreground="#f3f1ff",
    background="#070711",
    dark=True,
    variables={
        "bg": "#070711",
        "bg-panel": "#11111d",
        "bg-hover": "#20203a",
        "bg-selected": "#252047",
        "border": "#2c2b4a",
        "border-focused": "#8f7cff",
        "primary": "#8f7cff",
        "secondary": "#54d9ff",
        "accent": "#ff8bdc",
        "warning": "#ffc857",
        "error": "#ff6b8a",
        "success": "#62f5b0",
        "text": "#f3f1ff",
        "text-muted": "#9996b3",
        "text-dim": "#5d5a77",
        "text-bright": "#ffffff",
        "role-user": "#54d9ff",
        "role-agent": "#8f7cff",
        "role-tool": "#ffc857",
        "role-system": "#ff8bdc",
        "role-error": "#ff6b8a",
        "role-thinking": "#b78cff",
        "status-idle": "#77738e",
        "status-thinking": "#b78cff",
        "status-working": "#62f5b0",
        "status-success": "#62f5b0",
        "status-error": "#ff6b8a",
        "sidebar-bg": "#0b0b16",
        "sidebar-border": "#272640",
        "input-bg": "#141422",
        "input-border": "#2c2b4a",
        "input-focus": "#8f7cff",
    },
)

# Keep the legacy symbol so integrations importing OPENCODE_THEME continue
# to work, while making the new visual system the default.
OPENCODE_THEME = DEPRESSION_THEME


OPENCODE_CSS = """
/* ================================================================
   DEPRESSION.AI — FUTURISTIC GLASS UI
   ================================================================ */

Screen {
    background: #070711;
    color: #f3f1ff;
    layers: base overlay;
}

* {
    scrollbar-background: #070711;
    scrollbar-color: #302e50;
    scrollbar-corner-color: #070711;
}

/* Thin application chrome. The hero header itself owns the large branding. */
#header {
    background: #070711;
    border-bottom: solid #2c2b4a;
}

#main-container {
    height: 1fr;
    layout: horizontal;
}

/* Web-app style sidebar: stable navigation, clear selected state. */
#sidebar {
    width: 36;
    min-width: 28;
    background: #0b0b16;
    border-right: solid #272640;
}

#sidebar-tabs {
    height: 3;
    background: #11111d;
    border-bottom: solid #272640;
}

.sidebar-tab-btn {
    background: transparent;
    border: none;
    color: #77738e;
    text-style: bold;
}

.sidebar-tab-btn:hover {
    background: #20203a;
    color: #f3f1ff;
}

.sidebar-tab-btn.active {
    background: #252047;
    color: #8f7cff;
    border-bottom: tall #8f7cff;
}

#sidebar-content {
    padding: 1;
    background: #0b0b16;
}

ListView {
    background: transparent;
    border: none;
}

ListView > ListItem {
    background: transparent;
    border: none;
}

ListView > ListItem:hover {
    background: #20203a;
}

ListView > ListItem.active {
    background: #252047;
    border-left: tall #54d9ff;
}

/* Session cards */
.session-item {
    height: 3;
    margin: 0 0 1 0;
    padding: 0 1;
    background: #11111d;
}

.session-item:hover {
    background: #20203a;
}

.session-item.active {
    background: #252047;
    border-left: tall #54d9ff;
}

.session-name { color: #f3f1ff; text-style: bold; }
.session-meta { color: #5d5a77; }

/* Tool and file rows */
.tool-item, .file-item {
    height: 2;
    padding: 0 1;
    background: transparent;
}

.tool-item:hover, .file-item:hover { background: #20203a; }
.tool-icon { color: #ffc857; width: 3; }
.tool-name { color: #c9c6df; width: 1fr; }
.tool-status { color: #62f5b0; width: auto; }
.file-icon { width: 3; color: #54d9ff; }
.file-name { color: #c9c6df; width: 1fr; }
.file-size { color: #5d5a77; width: auto; }

/* Chat: breathing room and clean message hierarchy. */
#chat-panel {
    width: 1fr;
    background: #070711;
}

#chat-messages {
    padding: 2 3;
    background: #070711;
}

.message {
    margin: 0 0 2 0;
    padding: 0;
}

.msg-header { height: 1; text-style: bold; }
.msg-body { padding: 0 0 0 2; color: #ddd9ef; }
.msg-timestamp { color: #5d5a77; }
.role-user .msg-header { color: #54d9ff; }
.role-agent .msg-header { color: #8f7cff; }
.role-tool .msg-header { color: #ffc857; }
.role-system .msg-header { color: #ff8bdc; }
.role-error .msg-header { color: #ff6b8a; }
.role-thinking .msg-header { color: #b78cff; }

.message .md-h1 { color: #ffffff; text-style: bold; margin: 1 0 1 0; }
.message .md-h2 { color: #54d9ff; text-style: bold; margin: 1 0 1 0; }
.message .md-h3 { color: #8f7cff; text-style: bold; }
.message .md-code-block {
    background: #11111d;
    border: solid #2c2b4a;
    padding: 1 2;
    color: #ffc857;
}
.message .md-code-inline { background: #171729; color: #ffc857; padding: 0 1; }
.message .md-blockquote { border-left: solid #8f7cff; color: #9996b3; padding: 0 0 0 1; }
.message .md-list-bullet { color: #8f7cff; }
.message .md-link { color: #54d9ff; text-style: underline; }
.message .md-strong { color: #ffffff; text-style: bold; }

/* Floating composer, intentionally unlike the single-line classic CLI. */
#chat-input-area {
    height: 5;
    margin: 0 2 1 2;
    padding: 1 2;
    background: #141422;
    border: round #2c2b4a;
    layout: horizontal;
}

#input-prompt {
    width: 3;
    color: #8f7cff;
    text-style: bold;
    content-align: center middle;
}

#chat-input {
    width: 1fr;
    height: 3;
    background: transparent;
    border: none;
    color: #f3f1ff;
    padding: 0 1;
}

#chat-input:focus { border: none; }

/* Bottom status strip */
#status-bar {
    height: 2;
    background: #0b0b16;
    border-top: solid #272640;
    padding: 0 2;
}

#status-model { color: #9996b3; }
#status-tokens, #status-cost, #status-shortcuts { color: #5d5a77; }

/* Modal / command palette */
#command-palette, #help-container {
    background: #11111d;
    border: round #8f7cff;
}

#palette-input {
    background: #141422;
    border-bottom: solid #2c2b4a;
    color: #f3f1ff;
}

.palette-item:hover { background: #20203a; color: #54d9ff; }

/* Tool execution */
.tool-call { color: #ffc857; text-style: bold; }
.tool-success { color: #62f5b0; }
.tool-error { color: #ff6b8a; }
.tool-duration { color: #5d5a77; }

/* Thinking indicator */
.thinking-dots { color: #b78cff; }

/* Diffs */
.diff-file-header { background: #11111d; color: #8f7cff; text-style: bold; }
.diff-hunk-header { color: #54d9ff; }
.diff-added { background: #10251f; color: #62f5b0; }
.diff-removed { background: #2a121b; color: #ff6b8a; }
.diff-context { color: #77738e; }

/* Buttons */
.neon-btn {
    background: #141422;
    border: solid #2c2b4a;
    color: #ddd9ef;
}
.neon-btn:hover, .neon-btn:focus {
    background: #20203a;
    border: solid #8f7cff;
    color: #ffffff;
}
.neon-btn.primary { border: solid #8f7cff; color: #8f7cff; }
.neon-btn.success { border: solid #62f5b0; color: #62f5b0; }
.neon-btn.danger { border: solid #ff6b8a; color: #ff6b8a; }
.neon-btn.warning { border: solid #ffc857; color: #ffc857; }

/* Provider panel */
#llm-header { background: #11111d; color: #8f7cff; border-bottom: solid #2c2b4a; }
#llm-model-list { background: transparent; }
#llm-config-area { border-top: solid #2c2b4a; }
#llm-base-url, #llm-api-key-input { background: #141422; border: solid #2c2b4a; color: #ddd9ef; }
#llm-connect-btn { background: #141422; border: solid #62f5b0; color: #62f5b0; }
#llm-connect-btn:hover { background: #20203a; }
.model-item { background: transparent; border-bottom: solid #23233d; }
.model-item:hover { background: #20203a; }
.model-item.selected { background: #252047; border-left: tall #8f7cff; }
.model-name { color: #f3f1ff; text-style: bold; }
.model-meta { color: #5d5a77; }
.provider-header { background: #11111d; color: #54d9ff; text-style: bold; }
"""


def apply_opencode_theme(app: App) -> None:
    """Apply the futuristic Depression.AI theme to a Textual app."""
    app.register_theme(DEPRESSION_THEME)
    app.theme = "depression"
