"""Big ASCII banner header — CLI-native (opencode / claude-code style)."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static
from textual.reactive import reactive
from textual.containers import Horizontal


BANNER = (
    "[bold #ff9bc7]██████╗ ███████╗██████╗ ██████╗ ███████╗███████╗███████╗██╗ ██████╗ ███╗   ██╗[/]\n"
    "[bold #ff9bc7]██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║██╔═══██╗████╗  ██║[/]\n"
    "[bold #b794f6]██║  ██║█████╗  ██████╔╝██████╔╝█████╗  ███████╗███████╗██║██║   ██║██╔██╗ ██║[/]\n"
    "[bold #b794f6]██║  ██║██╔══╝  ██╔═══╝ ██╔═══╝ ██╔══╝  ╚════██║╚════██║██║██║   ██║██║╚██╗██║[/]\n"
    "[bold #7ee7ff]██████╔╝███████╗██║     ██║     ███████╗███████║███████║██║╚██████╔╝██║ ╚████║[/]\n"
    "[bold #7ee7ff]╚═════╝ ╚══════╝╚═╝     ╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝[/]"
)

ANGEL = (
    "[#ff9bc7]        ✧ ♡ ✧[/]\n"
    "[#ff9bc7]      ╭─────────╮[/]\n"
    "[#b794f6]      │  [bold #ffffff]◕[/]   [bold #ffffff]◕[/]  │[/]\n"
    "[#b794f6]      │    [#ff9bc7]♡[/]    │[/]\n"
    "[#b794f6]      ╰────┬────╯[/]\n"
    "[#7ee7ff]           ╲╱[/]\n"
)


class HeaderBar(Widget):
    """Oversized brand banner + angel mascot + one-line tagline."""

    DEFAULT_CSS = """
    HeaderBar {
        height: 9;
        background: #08070c;
        layout: horizontal;
        padding: 1 2 0 2;
    }
    #banner-col { width: 1fr; height: 8; }
    #banner     { height: 6; }
    #tagline    { height: 1; margin-top: 1; }
    #angel-col  { width: auto; height: 8; padding: 0 0 0 3; }
    #angel      { height: 6; }
    """

    mode       = reactive("build")
    model      = reactive("no model")
    session_id = reactive("—")

    MODE_ICONS = {"plan": "◇", "build": "◆", "auto": "⟡"}

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Widget(id="banner-col"):
                yield Static(BANNER, id="banner")
                yield Static(self._tagline_text(), id="tagline")
            with Widget(id="angel-col"):
                yield Static(ANGEL, id="angel")

    def _tagline_text(self) -> str:
        icon = self.MODE_ICONS.get(self.mode, "◆")
        ses = (self.session_id or "—")[:8]
        return (
            f"  {icon} [bold #7ef7c0]{self.mode.upper()}[/]  "
            f"[#524d60]·[/]  [#7ee7ff]✦ angelic coding companion[/]  "
            f"[#524d60]·[/]  [#8a849a]{self.model}[/]  "
            f"[#524d60]·[/]  [#8a849a]ses:{ses}[/]"
        )

    def _refresh_tagline(self) -> None:
        try:
            self.query_one("#tagline", Static).update(self._tagline_text())
        except Exception:
            pass

    def set_mode(self, mode: str) -> None:
        self.mode = mode or "build"
        self._refresh_tagline()

    def set_model(self, model: str) -> None:
        self.model = model or "no model"
        self._refresh_tagline()

    def set_session(self, session_id: str) -> None:
        self.session_id = session_id or "—"
        self._refresh_tagline()