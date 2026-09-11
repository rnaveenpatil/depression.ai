"""
UI Module - Terminal renderer.

Features:
    - Rich color palette with truecolor/256/16 fallback
    - Animated spinners and progress bars
    - Boxed panels, dividers, and header banners
    - Message role rendering (user / agent / tool / system)
    - Markdown-ish rendering for assistant output
    - Diff viewer
    - Tables and key/value blocks
    - Status bar with model, tokens, cost, session
    - Streaming text
    - Graceful degradation when color is unavailable
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# COLORS
# ======================================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    REVERSE = "\033[7m"

    BLACK = "\033[30m"; RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"; BRIGHT_RED = "\033[91m"; BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"; BRIGHT_BLUE = "\033[94m"; BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"; BRIGHT_WHITE = "\033[97m"

    BG_BLACK = "\033[40m"; BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_WHITE = "\033[107m"

    @staticmethod
    def rgb(r: int, g: int, b: int, bg: bool = False) -> str:
        return f"\033[48;2;{r};{g};{b}m" if bg else f"\033[38;2;{r};{g};{b}m"


class Palette:
    """Semantic color palette."""
    PRIMARY = Colors.rgb(140, 170, 255)
    PRIMARY_BOLD = Colors.BOLD + Colors.rgb(160, 190, 255)
    ACCENT = Colors.rgb(255, 180, 120)
    SUCCESS = Colors.rgb(120, 220, 140)
    WARNING = Colors.rgb(255, 200, 90)
    ERROR = Colors.rgb(255, 110, 110)
    INFO = Colors.rgb(140, 210, 255)
    MUTED = Colors.rgb(140, 140, 155)
    DIM_TEXT = Colors.rgb(90, 90, 105)
    AGENT = Colors.rgb(180, 150, 255)
    USER = Colors.rgb(120, 200, 255)
    TOOL = Colors.rgb(255, 170, 120)
    THINKING = Colors.rgb(200, 160, 255)
    HIGHLIGHT = Colors.BG_BRIGHT_BLACK + Colors.BRIGHT_WHITE


class Term:
    @staticmethod
    def supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False

    @staticmethod
    def supports_truecolor() -> bool:
        if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
            return True
        if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm", "vscode"):
            return True
        return False

    @staticmethod
    def width(default: int = 80) -> int:
        try:
            return shutil.get_terminal_size((default, 24)).columns
        except Exception:
            return default


# ======================================================================
# ICONS
# ======================================================================

class Icons:
    USER = "❯"
    AGENT = "◆"
    TOOL = "⚙"
    THINKING = "∴"
    SUCCESS = "✓"
    ERROR = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    BULLET = "•"
    ARROW = "→"
    CHEVRON = "›"
    SPARKLE = "✨"
    BRAIN = "🧠"
    FIRE = "🔥"
    LIGHTNING = "⚡"
    DIAMOND = "💎"
    STAR = "★"
    GEAR = "⚙"
    FOLDER = "📁"
    FILE = "📄"
    CODE = "⌘"
    TERMINAL = "⌨"
    CLOCK = "⏱"
    HOURGLASS = "⏳"


# ======================================================================
# SPINNER
# ======================================================================

class Spinner:
    STYLES = {
        "braille": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"],
        "dots":    ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
        "line":    ["-", "\\", "|", "/"],
        "arc":     ["◜", "◠", "◝", "◞", "◡", "◟"],
    }

    def __init__(self, message="Working", style="braille", color=None, interval=0.08):
        self.message = message
        self.frames = self.STYLES.get(style, self.STYLES["braille"])
        self.color = color or Palette.PRIMARY
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._active = False
        self._start = 0.0
        self._i = 0

    def start(self):
        if self._active:
            return self
        self._active = True
        self._stop.clear()
        self._start = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            frame = self.frames[self._i % len(self.frames)]
            elapsed = time.time() - self._start
            try:
                sys.stdout.write(
                    f"\r{self.color}{frame}{Colors.RESET} "
                    f"{Palette.MUTED}{self.message}{Palette.DIM_TEXT} ({elapsed:.1f}s){Colors.RESET} "
                )
                sys.stdout.flush()
            except Exception:
                break
            self._i += 1
            self._stop.wait(self.interval)

    def stop(self, clear=True):
        if not self._active:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._active = False
        if clear:
            try:
                sys.stdout.write("\r" + " " * (Term.width() - 1) + "\r")
                sys.stdout.flush()
            except Exception:
                pass

    def __enter__(self): return self.start()
    def __exit__(self, *args): self.stop()

    def update(self, message):
        self.message = message


# ======================================================================
# HELPERS
# ======================================================================

def strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def visible_len(s: str) -> int:
    return len(strip_ansi(s))


def pad_right(s: str, width: int, fill=" ") -> str:
    diff = width - visible_len(s)
    return s + fill * diff if diff > 0 else s


def pad_left(s: str, width: int, fill=" ") -> str:
    diff = width - visible_len(s)
    return fill * diff + s if diff > 0 else s


def truncate(s: str, width: int, ellipsis="…") -> str:
    if visible_len(s) <= width:
        return s
    plain = strip_ansi(s)
    return plain[: width - len(ellipsis)] + ellipsis


def gradient_text(text: str, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> str:
    if not text or not Term.supports_truecolor():
        return f"{Palette.PRIMARY}{text}{Colors.RESET}"
    n = max(1, len(text) - 1)
    out = []
    for i, ch in enumerate(text):
        t = i / n
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        out.append(f"{Colors.rgb(r, g, b)}{ch}")
    return "".join(out) + Colors.RESET


# ======================================================================
# UI
# ======================================================================

class UI:
    def __init__(self, color: Optional[bool] = None, width: Optional[int] = None, compact: bool = False):
        self.color = Term.supports_color() if color is None else color
        self.truecolor = Term.supports_truecolor() and self.color
        self.width = width or min(Term.width(), 120)
        self.compact = compact
        self._spinner: Optional[Spinner] = None
        self._theme = "dark"
        self._metrics = {"model": "—", "tokens": 0, "cost": 0.0, "session": "—"}

    # ------------------------------------------------------------------

    def _c(self, code: str, text: str) -> str:
        return f"{code}{text}{Colors.RESET}" if self.color else text

    def _w(self, text: str = "", end: str = "\n") -> None:
        try:
            sys.stdout.write(text + end)
            sys.stdout.flush()
        except Exception:
            pass

    def _clear_line(self) -> None:
        try:
            sys.stdout.write("\r" + " " * (Term.width() - 1) + "\r")
            sys.stdout.flush()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # HEADER / BANNER
    # ------------------------------------------------------------------

    def print_banner(self, title="CLI AGENT", subtitle="", version="v1.0.0") -> None:
        self._w()
        logo = [
            "  ██████╗██╗     ██╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗",
            " ██╔════╝██║     ██║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝",
            " ██║     ██║     ██║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ",
            " ██║     ██║     ██║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ",
            " ╚██████╗███████╗██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ",
            "  ╚═════╝╚══════╝╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ",
        ]
        for line in logo:
            self._w("  " + gradient_text(line, (140, 170, 255), (200, 140, 255)))
        if subtitle:
            self._w("  " + self._c(Palette.MUTED, subtitle))
        self._w("  " + self._c(Palette.DIM_TEXT, version))
        self._w()

    def print_header(self, title: str, subtitle: str = "") -> None:
        self._w()
        self._w(self._c(Palette.PRIMARY_BOLD, f"{Icons.SPARKLE} {title}"))
        if subtitle:
            self._w(self._c(Palette.MUTED, f"  {subtitle}"))
        self.print_divider()

    def print_divider(self, char: str = "─", color: Optional[str] = None) -> None:
        self._w(self._c(color or Palette.DIM_TEXT, char * self.width))

    def print_separator(self) -> None:
        self.print_divider("─", Palette.DIM_TEXT)

    # ------------------------------------------------------------------
    # MESSAGES
    # ------------------------------------------------------------------

    def print_success(self, msg): self._w(f"{self._c(Palette.SUCCESS, Icons.SUCCESS)} {msg}")
    def print_error(self, msg):   self._w(f"{self._c(Palette.ERROR, Icons.ERROR)} {self._c(Palette.ERROR, msg)}")
    def print_warning(self, msg): self._w(f"{self._c(Palette.WARNING, Icons.WARNING)} {self._c(Palette.WARNING, msg)}")
    def print_info(self, msg):    self._w(f"{self._c(Palette.INFO, Icons.INFO)} {msg}")
    def print_status(self, msg):  self._w(f"{self._c(Palette.MUTED, Icons.BULLET)} {self._c(Palette.MUTED, msg)}")
    def print_debug(self, msg):   self._w(self._c(Palette.DIM_TEXT, f"[{Icons.GEAR}] {msg}"))
    def print_plain(self, msg=""): self._w(msg)

    # ------------------------------------------------------------------
    # ROLE MESSAGES
    # ------------------------------------------------------------------

    def print_user_message(self, message: str) -> None:
        self._w()
        self._w(f"{self._c(Palette.USER, Icons.USER)} {self._c(Palette.USER + Colors.BOLD, 'You')}")
        for line in message.splitlines() or [""]:
            self._w(f"  {line}")
        self._w()

    def print_agent_message(self, message: str, markdown: bool = True, animate: bool = False) -> None:
        self._w()
        self._w(f"{self._c(Palette.AGENT, Icons.AGENT)} {self._c(Palette.AGENT + Colors.BOLD, 'Agent')}")
        if animate:
            self.stream_text(message)
        elif markdown:
            self._render_markdown(message)
        else:
            for line in message.splitlines() or [""]:
                self._w(f"  {line}")
        self._w()

    def print_system_message(self, message: str) -> None:
        self._w()
        self._w(f"{self._c(Palette.INFO, Icons.INFO)} {self._c(Palette.INFO + Colors.BOLD, 'System')}")
        for line in message.splitlines():
            self._w(f"  {self._c(Palette.MUTED, line)}")
        self._w()

    def print_tool_call(
        self,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        success: bool = True,
        duration: Optional[float] = None,
    ) -> None:
        color = Palette.SUCCESS if success else Palette.ERROR
        icon = Icons.SUCCESS if success else Icons.ERROR
        line = f"  {self._c(Palette.TOOL, Icons.TOOL)} {self._c(Palette.TOOL + Colors.BOLD, tool_name)}"
        if duration is not None:
            line += f" {self._c(Palette.DIM_TEXT, f'({duration:.2f}s)')}"
        line += f" {self._c(color, icon)}"
        self._w(line)

        if params:
            for k, v in list(params.items())[:6]:
                vs = str(v)
                if len(vs) > 80:
                    vs = vs[:77] + "…"
                self._w(f"    {self._c(Palette.MUTED, k)} = {self._c(Palette.DIM_TEXT, vs)}")

        if result is not None and not success:
            for line in str(result).splitlines()[:5]:
                self._w(f"    {self._c(Palette.ERROR, line)}")

    # ------------------------------------------------------------------
    # STREAM
    # ------------------------------------------------------------------

    def stream_text(self, text: str, delay: float = 0.005) -> None:
        try:
            for ch in text:
                sys.stdout.write(ch)
                sys.stdout.flush()
                if delay > 0:
                    time.sleep(delay)
            sys.stdout.write("\n")
        except Exception:
            self._w(text)

    # ------------------------------------------------------------------
    # MARKDOWN
    # ------------------------------------------------------------------

    def _render_markdown(self, text: str) -> None:
        in_code = False
        for line in text.splitlines():
            stripped = line.strip()

            if stripped.startswith("```"):
                if in_code:
                    self._w(f"  {self._c(Palette.DIM_TEXT, '└' + '─' * (self.width - 4))}")
                    in_code = False
                else:
                    label = stripped[3:].strip()
                    header = f"┌ {label} " if label else "┌"
                    header += "─" * max(0, self.width - 4 - len(header))
                    self._w(f"  {self._c(Palette.DIM_TEXT, header)}")
                    in_code = True
                continue

            if in_code:
                self._w(f"  {self._c(Palette.DIM_TEXT, '│')} {self._c(Palette.ACCENT, line)}")
                continue

            if stripped.startswith("### "):
                self._w(f"  {self._c(Palette.PRIMARY + Colors.BOLD, stripped[4:])}")
            elif stripped.startswith("## "):
                self._w(f"  {self._c(Palette.PRIMARY_BOLD, stripped[3:])}")
            elif stripped.startswith("# "):
                self._w(f"  {self._c(Palette.PRIMARY_BOLD + Colors.UNDERLINE, stripped[2:])}")
            elif stripped.startswith(("- ", "* ", "+ ")):
                self._w(f"  {self._c(Palette.ACCENT, Icons.BULLET)} {stripped[2:]}")
            elif stripped.startswith("> "):
                self._w(f"  {self._c(Palette.DIM_TEXT, '│')} {self._c(Palette.MUTED + Colors.ITALIC, stripped[2:])}")
            elif stripped in ("---", "***", "___"):
                self.print_divider()
            elif not stripped:
                self._w()
            else:
                self._w(f"  {line}")

    # ------------------------------------------------------------------
    # BOXES
    # ------------------------------------------------------------------

    def print_box(
        self,
        content: str,
        title: str = "",
        title_color: Optional[str] = None,
        border_color: Optional[str] = None,
        width: Optional[int] = None,
    ) -> None:
        title_color = title_color or Palette.PRIMARY
        border_color = border_color or Palette.DIM_TEXT
        width = width or self.width
        inner = width - 4

        if title:
            title_text = f" {title} "
            remaining = width - 2 - visible_len(title_text)
            if remaining < 0:
                title_text = truncate(title_text, width - 2)
                remaining = 0
            top = (
                f"{self._c(border_color, '╭─')}"
                f"{self._c(title_color + Colors.BOLD, title_text)}"
                f"{self._c(border_color, '─' * remaining + '╮')}"
            )
        else:
            top = self._c(border_color, "╭" + "─" * (width - 2) + "╮")
        self._w(top)

        for line in content.splitlines() or [""]:
            line = truncate(line, inner)
            pad = inner - visible_len(line)
            self._w(f"{self._c(border_color, '│')} {line}{' ' * pad} {self._c(border_color, '│')}")
        self._w(self._c(border_color, "╰" + "─" * (width - 2) + "╯"))

    # ------------------------------------------------------------------
    # DIFF
    # ------------------------------------------------------------------

    def print_diff(self, old: str, new: str, filename: str = "") -> None:
        import difflib
        self._w()
        if filename:
            self._w(self._c(Palette.PRIMARY_BOLD, f"{Icons.FILE} {filename}"))
        for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=3):
            if line.startswith("+++") or line.startswith("---"):
                self._w(self._c(Palette.DIM_TEXT, line))
            elif line.startswith("@@"):
                self._w(self._c(Palette.INFO, line))
            elif line.startswith("+"):
                self._w(self._c(Palette.SUCCESS, line))
            elif line.startswith("-"):
                self._w(self._c(Palette.ERROR, line))
            else:
                self._w(line)
        self._w()

    # ------------------------------------------------------------------
    # TABLE
    # ------------------------------------------------------------------

    def print_table(self, headers: List[str], rows: List[List[str]], title: str = "") -> None:
        if not headers and not rows:
            return
        cols = len(headers) if headers else len(rows[0])
        widths = [0] * cols
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], visible_len(h))
        for row in rows:
            for i, cell in enumerate(row[:cols]):
                widths[i] = max(widths[i], visible_len(str(cell)))

        self._w()
        if title:
            self._w(self._c(Palette.PRIMARY_BOLD, title))

        def fmt(cells, bold=False):
            parts = []
            for i, cell in enumerate(cells[:cols]):
                s = str(cell)
                pad = widths[i] - visible_len(s)
                styled = self._c(Palette.PRIMARY + Colors.BOLD, s) if bold else s
                parts.append(styled + " " * pad)
            return "  " + " │ ".join(parts)

        if headers:
            self._w(fmt(headers, bold=True))
            self._w(self._c(Palette.DIM_TEXT, "  " + "─┼─".join("─" * w for w in widths)))
        for row in rows:
            self._w(fmt(row))
        self._w()

    # ------------------------------------------------------------------
    # SPINNER / PROGRESS
    # ------------------------------------------------------------------

    def start_spinner(self, message="Working", style="braille", color=None) -> Spinner:
        self.stop_spinner()
        self._spinner = Spinner(message, style, color or Palette.PRIMARY).start()
        return self._spinner

    def stop_spinner(self) -> None:
        if self._spinner:
            self._spinner.stop()
            self._spinner = None

    def update_spinner(self, message: str) -> None:
        if self._spinner:
            self._spinner.update(message)

    # ------------------------------------------------------------------
    # STATUS BAR
    # ------------------------------------------------------------------

    def set_status(self, **kwargs) -> None:
        self._metrics.update(kwargs)

    def print_status_bar(self) -> None:
        left = f"{self._c(Palette.AGENT, Icons.AGENT)} {self._c(Palette.MUTED, self._metrics.get('model', '—'))}"
        tokens = self._metrics.get('tokens', 0)
        cost = self._metrics.get('cost', 0.0)
        session = self._metrics.get('session', '—')
        center = (
            f"{self._c(Palette.DIM_TEXT, Icons.LIGHTNING + ' ' + f'{tokens:,}' + ' tok')} "
            f"{self._c(Palette.DIM_TEXT, '· $' + f'{cost:.4f}')}"
        )
        right = f"{self._c(Palette.DIM_TEXT, Icons.CLOCK + ' ' + session)}"
        width = self.width
        line = truncate(pad_right(left, width // 3) + center + pad_left(right, width // 3), width)
        bg = Palette.HIGHLIGHT if self.color else ""
        self._w(f"{bg}{line}{Colors.RESET}")

    # ------------------------------------------------------------------
    # SPECIAL
    # ------------------------------------------------------------------

    def print_welcome(self, version="v1.0.0", model="—") -> None:
        self._w()
        self.print_box(
            content=f"Welcome to CLI Agent {version}\nModel: {model}\nType /help for commands, or just type a query.",
            title="Welcome",
        )
        self._w()

    def print_shortcuts(self) -> None:
        self.print_table(
            ["Key", "Action"],
            [
                ["Enter", "Send message"],
                ["Alt+Enter", "Newline"],
                ["Ctrl+C", "Cancel / clear"],
                ["Ctrl+D", "Exit"],
                ["Ctrl+R", "History search"],
                ["Ctrl+L", "Clear screen"],
                ["Tab", "Autocomplete"],
                ["/help", "Show commands"],
            ],
            title="Shortcuts",
        )

    def print_model_info(self, model: Dict[str, Any]) -> None:
        content = "\n".join(f"{k}: {v}" for k, v in model.items())
        self.print_box(content=content, title=f"{Icons.BRAIN} Model", title_color=Palette.AGENT)

    def print_tool_result(self, tool_name: str, result: Any, duration: Optional[float] = None) -> None:
        title = f"{Icons.TOOL} {tool_name}"
        if duration is not None:
            title += f" ({duration:.2f}s)"
        content = str(result)
        if len(content) > 2000:
            content = content[:2000] + "\n… (truncated)"
        self.print_box(content=content, title=title, title_color=Palette.TOOL)

    # ------------------------------------------------------------------
    # THEME / CLEAR
    # ------------------------------------------------------------------

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        if theme == "light":
            Palette.PRIMARY = Colors.rgb(60, 90, 200)
            Palette.MUTED = Colors.rgb(100, 100, 110)
            Palette.DIM_TEXT = Colors.rgb(140, 140, 150)
        elif theme == "dark":
            Palette.PRIMARY = Colors.rgb(140, 170, 255)
            Palette.MUTED = Colors.rgb(140, 140, 155)
            Palette.DIM_TEXT = Colors.rgb(90, 90, 105)

    def clear_screen(self) -> None:
        os.system("cls" if os.name == "nt" else "clear")

    def newline(self, count: int = 1) -> None:
        for _ in range(count):
            self._w()


# ======================================================================
# MODULE-LEVEL
# ======================================================================

_default_ui: Optional[UI] = None


def get_ui(**kwargs) -> UI:
    global _default_ui
    if _default_ui is None:
        _default_ui = UI(**kwargs)
    return _default_ui


def reset_ui() -> None:
    global _default_ui
    _default_ui = None


__all__ = [
    "UI", "Colors", "Palette", "Term", "Spinner", "Icons",
    "gradient_text", "strip_ansi", "visible_len",
    "pad_right", "pad_left", "truncate",
    "get_ui", "reset_ui",
]