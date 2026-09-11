"""
CLI Input Module - Advanced Terminal Input Handler
Handles:
- Multi-line input with proper continuation
- Command and file path autocompletion
- Persistent command history with search
- Keyboard shortcuts (Ctrl+C, Ctrl+D, Ctrl+L, etc.)
- Bracket paste handling
- Vi/Emacs editing modes
- Input validation and sanitization
- Async input with cancellation support
"""

import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple, Iterator
from dataclasses import dataclass, field
from enum import Enum

# prompt_toolkit provides advanced terminal input capabilities
try:
    from prompt_toolkit import PromptSession, prompt as ptk_prompt
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import (
        Completer,
        Completion,
        WordCompleter,
        PathCompleter,
        FuzzyCompleter,
        merge_completers,
    )
    from prompt_toolkit.document import Document
    from prompt_toolkit.formatted_text import HTML, ANSI
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.widgets import TextArea, Frame, Label
    from prompt_toolkit.enums import EditingMode
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    PromptSession = None  # type: ignore

from agent.utils.logging import get_logger
from agent.utils.errors import InputError

logger = get_logger(__name__)


class InputMode(Enum):
    """Input modes for the editor"""
    EMACS = "emacs"
    VI = "vi"


class ContinuationType(Enum):
    """Types of line continuations"""
    NONE = "none"
    EXPLICIT = "explicit"       # User typed \ at end
    IMPLICIT = "implicit"       # Unclosed brackets/quotes
    MANUAL = "manual"           # User pressed Shift+Enter / Alt+Enter


@dataclass
class InputResult:
    """Result of an input operation"""
    text: str
    cancelled: bool = False
    eof: bool = False
    multiline: bool = False
    attachments: List[str] = field(default_factory=list)


class SlashCommandCompleter(Completer):
    """
    Completer for slash commands.
    Suggests commands when input starts with '/'.
    """

    def __init__(self, command_processor: Any):
        self.command_processor = command_processor

    def get_completions(self, document: Document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor

        # Only complete slash commands
        if not text.startswith("/"):
            return

        # Extract the command prefix (everything after / up to first space)
        after_slash = text[1:]
        if " " in after_slash:
            # Already has arguments - complete subcommands or exit
            return

        prefix = after_slash.lower()

        # Get commands from the processor
        try:
            commands = self.command_processor.commands
        except AttributeError:
            return

        for name, cmd in commands.items():
            if cmd.hidden and not prefix:
                continue

            # Match by name or alias
            candidates = [name] + list(cmd.aliases)
            for candidate in candidates:
                if candidate.lower().startswith(prefix):
                    yield Completion(
                        candidate,
                        start_position=-len(prefix),
                        display=f"/{candidate}",
                        display_meta=cmd.description[:60],
                    )
                    break


class FilePathCompleter(Completer):
    """
    Smart file-path completer that activates when input contains @path
    or when the cursor is in a path-like context.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or os.getcwd())

    def get_completions(self, document: Document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor

        # Look for @-prefixed paths
        match = re.search(r'@([\w./\-]*)$', text)
        if not match:
            return

        partial = match.group(1)

        # Determine base directory
        if partial.startswith("/"):
            base = Path("/")
            partial_rel = partial[1:]
        elif partial.startswith("~"):
            base = Path.home()
            partial_rel = partial[2:]
        else:
            base = self.base_dir
            partial_rel = partial

        # Split into dir and filename parts
        if "/" in partial_rel:
            dir_part, file_part = partial_rel.rsplit("/", 1)
            search_dir = base / dir_part
        else:
            dir_part = ""
            file_part = partial_rel
            search_dir = base

        if not search_dir.is_dir():
            return

        try:
            entries = sorted(
                search_dir.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except (PermissionError, OSError):
            return

        for entry in entries:
            if entry.name.startswith(".") and not file_part.startswith("."):
                continue

            if not entry.name.lower().startswith(file_part.lower()):
                continue

            # Build the completion text
            if dir_part:
                completion_text = f"{dir_part}/{entry.name}"
            else:
                completion_text = entry.name

            if entry.is_dir():
                completion_text += "/"
                display_meta = "directory"
            else:
                display_meta = self._human_size(entry.stat().st_size)

            yield Completion(
                completion_text,
                start_position=-len(partial),
                display=entry.name + ("/" if entry.is_dir() else ""),
                display_meta=display_meta,
            )

    def _human_size(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class ShellHistoryCompleter(Completer):
    """Complete from previous shell history"""

    def __init__(self, history_file: Optional[Path] = None):
        self.history_file = history_file or Path.home() / ".bash_history"
        self._history: List[str] = []
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            if self.history_file.exists():
                with open(self.history_file, "r", errors="ignore") as f:
                    self._history = [
                        line.strip() for line in f
                        if line.strip() and not line.startswith("#")
                    ][-2000:]  # last 2000 lines
        except Exception as e:
            logger.debug(f"Could not load shell history: {e}")
        self._loaded = True

    def get_completions(self, document: Document, complete_event) -> Iterator[Completion]:
        self._load()
        text = document.text_before_cursor

        # Only complete after ! prefix (like Claude Code)
        if not text.startswith("!"):
            return

        prefix = text[1:].lower()
        if not prefix:
            return

        seen = set()
        for cmd in reversed(self._history):
            if prefix in cmd.lower() and cmd not in seen:
                seen.add(cmd)
                yield Completion(
                    cmd,
                    start_position=-len(prefix),
                    display=cmd[:100],
                )
                if len(seen) >= 10:
                    break


class CommandCompleter(Completer):
    """
    Merged completer that dispatches to the right sub-completer
    based on the current input context.
    """

    def __init__(self, command_processor: Any, base_dir: Optional[str] = None):
        self.slash_completer = FuzzyCompleter(SlashCommandCompleter(command_processor))
        self.path_completer = FilePathCompleter(base_dir)
        self.shell_completer = ShellHistoryCompleter()

    def get_completions(self, document: Document, complete_event) -> Iterator[Completion]:
        text = document.text_before_cursor

        if text.startswith("/"):
            yield from self.slash_completer.get_completions(document, complete_event)
        elif "@" in text.split()[-1] if text.split() else False:
            yield from self.path_completer.get_completions(document, complete_event)
        elif text.startswith("!"):
            yield from self.shell_completer.get_completions(document, complete_event)


class InputLexer(Lexer):
    """
    Lexer that highlights input:
    - Slash commands in bold cyan
    - @paths in yellow
    - !shell commands in green
    - Quoted strings in magenta
    """

    def lex_document(self, document: Document):
        def get_line(lineno: int):
            line = document.lines[lineno]
            return list(self._lex_line(line))
        return get_line

    def _lex_line(self, line: str):
        # Simple token-based highlighting
        i = 0
        n = len(line)
        while i < n:
            # Slash command at start
            if i == 0 and line.startswith("/"):
                end = line.find(" ")
                if end == -1:
                    end = n
                yield ("class:command", line[i:end])
                i = end
                continue

            # @path
            if line[i] == "@":
                j = i + 1
                while j < n and line[j] not in " \t\n":
                    j += 1
                yield ("class:path", line[i:j])
                i = j
                continue

            # !shell
            if i == 0 and line.startswith("!"):
                yield ("class:shell", line[i:])
                return

            # Quoted string
            if line[i] in "\"'":
                quote = line[i]
                j = i + 1
                while j < n and line[j] != quote:
                    if line[j] == "\\":
                        j += 2
                    else:
                        j += 1
                if j < n:
                    j += 1
                yield ("class:string", line[i:j])
                i = j
                continue

            # Default
            j = i
            while j < n and line[j] not in "@/\"'":
                j += 1
            if j == i:
                j = i + 1
            yield ("", line[i:j])
            i = j


class InputHandler:
    """
    Advanced terminal input handler.

    Features:
    - Async prompt with cancellation
    - Slash command autocompletion
    - File path completion via @ prefix
    - Shell history completion via ! prefix
    - Persistent history with search
    - Multi-line input via Shift+Enter, Alt+Enter, or explicit continuation
    - Bracket paste handling
    - Vi and Emacs editing modes
    - Custom keybindings (Ctrl+L, Ctrl+R, Ctrl+O, etc.)
    - Syntax highlighting for commands, paths, and strings
    """

    # Prompt symbols
    PROMPT_MAIN = "🧠 "
    PROMPT_CONTINUATION = "   ... "
    PROMPT_COMMAND = "⌘ "

    def __init__(
        self,
        command_processor: Optional[Any] = None,
        history_file: Optional[Path] = None,
        base_dir: Optional[str] = None,
        editing_mode: InputMode = InputMode.EMACS,
    ):
        self.command_processor = command_processor
        self.base_dir = base_dir or os.getcwd()

        # History file for persistence
        if history_file is None:
            history_file = Path.home() / ".agent" / "input_history"
            history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file = history_file

        # Input mode
        self.editing_mode = editing_mode

        # Multi-line state
        self._multiline_buffer: List[str] = []
        self._in_multiline = False

        # Session (created lazily)
        self._session: Optional[PromptSession] = None

        # Fallback state (when prompt_toolkit is unavailable)
        self._fallback_history: List[str] = []
        self._fallback_history_index = 0

        # Event callbacks
        self.on_paste: Optional[Callable[[str], None]] = None
        self.on_interrupt: Optional[Callable[[], None]] = None

        # Check prompt_toolkit availability
        if not PROMPT_TOOLKIT_AVAILABLE:
            logger.warning(
                "prompt_toolkit not available - falling back to basic input. "
                "Install with: pip install prompt_toolkit"
            )

        logger.info(
            f"InputHandler initialized "
            f"(mode={editing_mode.value}, toolkit={PROMPT_TOOLKIT_AVAILABLE})"
        )

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    def _create_session(self) -> "PromptSession":
        """Create and configure the prompt_toolkit session"""
        # History
        try:
            history = FileHistory(str(self.history_file))
        except Exception:
            history = InMemoryHistory()

        # Completer
        completer = None
        if self.command_processor is not None:
            completer = CommandCompleter(
                command_processor=self.command_processor,
                base_dir=self.base_dir,
            )

        # Key bindings
        kb = self._build_key_bindings()

        # Style
        style = self._build_style()

        # Lexer
        lexer = InputLexer()

        session = PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            complete_while_typing=True,
            lexer=lexer,
            key_bindings=kb,
            style=style,
            editing_mode=(
                EditingMode.VI if self.editing_mode == InputMode.VI
                else EditingMode.EMACS
            ),
            multiline=False,  # We handle multiline ourselves
            enable_history_search=True,
            mouse_support=False,
            enable_open_in_editor=True,
            enable_system_prompt=True,
            enable_suspend=True,
            reserve_space_for_menu=8,
            bottom_toolbar=self._bottom_toolbar,
        )

        return session

    def _build_key_bindings(self) -> "KeyBindings":
        """Build custom key bindings"""
        kb = KeyBindings()

        # ------------------------------------------------------------------
        # Multiline / continuation
        # ------------------------------------------------------------------

        @kb.add("escape", "enter")  # Alt+Enter
        def _(event):
            """Insert a newline instead of submitting (multiline input)"""
            event.current_buffer.insert_text("\n")
            event.current_buffer.validate_and_handle()

        @kb.add("c-j")  # Ctrl+J
        def _(event):
            """Ctrl+J: insert newline (multiline)"""
            event.current_buffer.insert_text("\n")

        @kb.add("c-m")  # Enter
        def _(event):
            """Enter: submit or continue multiline"""
            buf = event.current_buffer
            text = buf.text

            # Check for explicit continuation
            if text.endswith("\\"):
                buf.text = text[:-1] + "\n"
                return

            # Check for implicit continuation (unclosed brackets)
            if self._needs_continuation(text):
                buf.insert_text("\n")
                return

            # Normal submit
            buf.validate_and_handle()

        # ------------------------------------------------------------------
        # Cancel / EOF
        # ------------------------------------------------------------------

        @kb.add("c-c")
        def _(event):
            """Ctrl+C: clear current input or exit"""
            buf = event.current_buffer
            if buf.text:
                # Clear current input
                buf.reset()
                self._in_multiline = False
                self._multiline_buffer = []
            else:
                # Signal interrupt
                if self.on_interrupt:
                    self.on_interrupt()
                event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def _(event):
            """Ctrl+D: EOF on empty buffer"""
            buf = event.current_buffer
            if not buf.text:
                event.app.exit(exception=EOFError())
            else:
                # Delete character at cursor (default behavior)
                buf.delete()

        # ------------------------------------------------------------------
        # Screen / history
        # ------------------------------------------------------------------

        @kb.add("c-l")
        def _(event):
            """Ctrl+L: clear screen"""
            event.app.renderer.clear()
            event.app.invalidate()

        @kb.add("c-r")
        def _(event):
            """Ctrl+R: reverse search (built-in)"""
            event.app.current_buffer.start_history_lines_completion()

        @kb.add("c-o")
        def _(event):
            """Ctrl+O: toggle multiline view"""
            event.app.current_buffer.open_in_editor()

        @kb.add("c-x", "c-e")
        def _(event):
            """Ctrl+X Ctrl+E: open in external editor"""
            event.app.current_buffer.open_in_editor()

        # ------------------------------------------------------------------
        # Paste handling
        # ------------------------------------------------------------------

        @kb.add(Keys.BracketedPaste)
        def _(event):
            """Handle bracket paste - insert as-is"""
            data = event.data
            data = data.replace("\r\n", "\n").replace("\r", "\n")
            event.current_buffer.insert_text(data)
            if self.on_paste:
                try:
                    self.on_paste(data)
                except Exception as e:
                    logger.debug(f"on_paste callback error: {e}")

        # ------------------------------------------------------------------
        # Command palette (Ctrl+P)
        # ------------------------------------------------------------------

        @kb.add("c-p")
        def _(event):
            """Ctrl+P: show command palette (prefill with /)"""
            buf = event.current_buffer
            if not buf.text.startswith("/"):
                buf.insert_text("/")

        # ------------------------------------------------------------------
        # Tab completion (default behavior is fine, but we augment)
        # ------------------------------------------------------------------

        @kb.add("c-space")
        def _(event):
            """Ctrl+Space: force completion menu"""
            buf = event.current_buffer
            buf.start_completion(select_first=False)

        # ------------------------------------------------------------------
        # Cancel completion
        # ------------------------------------------------------------------

        @kb.add("escape")
        def _(event):
            """Esc: cancel completion menu"""
            buf = event.current_buffer
            if buf.complete_state:
                buf.cancel_completion()
            else:
                # Vi normal mode toggle if in vi mode
                if self.editing_mode == InputMode.VI:
                    pass

        return kb

    def _build_style(self) -> "Style":
        """Build the input style"""
        return Style.from_dict({
            "prompt": "ansicyan bold",
            "continuation": "ansibrightblack",
            "command": "ansicyan bold",
            "path": "ansiyellow",
            "shell": "ansigreen",
            "string": "ansimagenta",
            "bottom-toolbar": "bg:#222222 #888888",
            "completion-menu.completion": "bg:#222222 #ffffff",
            "completion-menu.completion.current": "bg:#444444 #ffffff bold",
            "completion-menu.meta.completion": "bg:#222222 #888888",
            "completion-menu.meta.completion.current": "bg:#444444 #aaaaaa",
            "scrollbar.background": "bg:#222222",
            "scrollbar.button": "bg:#444444",
            "auto-suggestion": "ansibrightblack italic",
        })

    def _bottom_toolbar(self) -> "ANSI":
        """Bottom toolbar text"""
        hints = [
            ("Enter", "send"),
            ("Alt+Enter", "newline"),
            ("Ctrl+C", "cancel"),
            ("Ctrl+D", "exit"),
            ("Ctrl+R", "search"),
            ("Tab", "complete"),
        ]
        parts = []
        for key, desc in hints:
            parts.append(f"<b>{key}</b> {desc}")
        text = "  ".join(parts)
        return ANSI(f"\x1b[90m{text}\x1b[0m")

    # ------------------------------------------------------------------
    # Continuation detection
    # ------------------------------------------------------------------

    def _needs_continuation(self, text: str) -> bool:
        """
        Check if the input requires a continuation line.
        Returns True if:
        - Bracket depth is unbalanced
        - Quoted string is unterminated
        """
        if not text.strip():
            return False

        # Scan for balanced brackets and quotes
        depth = 0
        i = 0
        n = len(text)
        in_quote: Optional[str] = None

        while i < n:
            ch = text[i]

            if in_quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in "\"'`":
                    in_quote = ch
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1

            i += 1

        return depth > 0 or in_quote is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_input(
        self,
        prompt: Optional[str] = None,
        history: Optional[List[Any]] = None,
        multiline: bool = False,
        default: str = "",
        validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
    ) -> Optional[str]:
        """
        Get input from the user asynchronously.

        Args:
            prompt: Custom prompt (default: 🧠)
            history: Optional list of prior history items (informational)
            multiline: Force multiline mode
            default: Pre-fill the buffer with this text
            validator: Optional (is_valid, error_msg) callable

        Returns:
            The user's input string, or None on EOF
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return await self._fallback_get_input(prompt, default)

        # Create session lazily
        if self._session is None:
            self._session = self._create_session()

        prompt_text = prompt if prompt is not None else self.PROMPT_MAIN

        # Determine if this is a command input
        if default:
            # Pre-fill
            pass

        try:
            with patch_stdout():
                # Run the prompt in an executor so we don't block the loop
                loop = asyncio.get_event_loop()

                def _prompt():
                    try:
                        return self._session.prompt(
                            message=prompt_text,
                            default=default,
                            multiline=False,  # we handle manually
                        )
                    except KeyboardInterrupt:
                        return None
                    except EOFError:
                        return None

                result = await loop.run_in_executor(None, _prompt)

                # Handle None (EOF or Ctrl+C on empty)
                if result is None:
                    raise EOFError()

                # Trim trailing whitespace
                text = result.rstrip()

                # Apply validator if provided
                if validator and text:
                    ok, err = validator(text)
                    if not ok:
                        logger.debug(f"Validation failed: {err}")

                return text

        except KeyboardInterrupt:
            raise
        except EOFError:
            return None
        except Exception as e:
            logger.error(f"Input error: {e}", exc_info=True)
            return None

    async def _fallback_get_input(
        self,
        prompt: Optional[str],
        default: str,
    ) -> Optional[str]:
        """
        Fallback input method when prompt_toolkit is not available.
        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        prompt_text = prompt if prompt is not None else self.PROMPT_MAIN

        def _read() -> Optional[str]:
            try:
                line = input(f"{prompt_text}{default}")
                return line
            except (EOFError, KeyboardInterrupt):
                return None

        return await asyncio.to_thread(_read)

    async def get_multiline_input(
        self,
        prompt: Optional[str] = None,
        end_marker: str = "\\end",
        max_lines: int = 100,
    ) -> str:
        """
        Explicit multi-line input mode.
        User types lines until they enter the end marker.
        """
        lines: List[str] = []
        prompt_text = prompt or self.PROMPT_MAIN

        self._print_multiline_header()

        while len(lines) < max_lines:
            try:
                line = await self.get_input(
                    prompt=self.PROMPT_CONTINUATION,
                    multiline=True,
                )
            except (EOFError, KeyboardInterrupt):
                break

            if line is None:
                break

            if line.strip() == end_marker:
                break

            lines.append(line)

        return "\n".join(lines)

    def _print_multiline_header(self) -> None:
        """Print a header for multiline mode"""
        try:
            from agent.cli.ui import Colors
            print(f"\n{Colors.CYAN}Entering multiline mode. Type '\\end' on its own line to finish.{Colors.RESET}\n")
        except ImportError:
            print("\nEntering multiline mode. Type '\\end' to finish.\n")

    async def get_confirmation(
        self,
        message: str,
        default: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Ask the user for a yes/no confirmation.

        Args:
            message: The question to ask
            default: Default answer if user presses Enter
            timeout: Optional timeout in seconds

        Returns:
            True if confirmed, False otherwise
        """
        suffix = " [Y/n]" if default else " [y/N]"
        prompt = f"{message}{suffix} "

        async def _ask() -> bool:
            try:
                answer = await self.get_input(prompt=prompt)
            except (EOFError, KeyboardInterrupt):
                return False

            if answer is None or not answer.strip():
                return default

            answer = answer.strip().lower()
            return answer in ("y", "yes", "true", "1")

        if timeout:
            try:
                return await asyncio.wait_for(_ask(), timeout=timeout)
            except asyncio.TimeoutError:
                return default

        return await _ask()

    async def get_choice(
        self,
        message: str,
        choices: List[Tuple[str, str]],
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Present a list of options and get the user's choice.

        Args:
            message: Prompt message
            choices: List of (value, label) tuples
            default: Default value if user presses Enter

        Returns:
            Selected value, or None on cancel
        """
        print(f"\n{message}")
        for i, (value, label) in enumerate(choices, 1):
            marker = "*" if value == default else " "
            print(f"  {marker} {i}. {label}")

        prompt = f"Select [1-{len(choices)}]"
        if default is not None:
            prompt += f" (default: {default})"
        prompt += ": "

        try:
            answer = await self.get_input(prompt=prompt)
        except (EOFError, KeyboardInterrupt):
            return None

        if answer is None or not answer.strip():
            return default

        answer = answer.strip()

        # Numeric choice
        if answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]

        # Match by value or label
        answer_lower = answer.lower()
        for value, label in choices:
            if answer_lower in (value.lower(), label.lower()):
                return value

        return None

    async def get_password(self, prompt: str = "Password: ") -> Optional[str]:
        """
        Get password input (masked).
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            import getpass
            try:
                return await asyncio.to_thread(getpass.getpass, prompt)
            except (EOFError, KeyboardInterrupt):
                return None

        if self._session is None:
            self._session = self._create_session()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._session.prompt(
                    message=prompt,
                    is_password=True,
                ),
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception as e:
            logger.error(f"Password input error: {e}")
            return None

    def get_history(self) -> List[str]:
        """
        Get the input history.
        """
        if PROMPT_TOOLKIT_AVAILABLE and self._session is not None:
            try:
                return [
                    entry
                    for entry in self._session.history.get_strings()
                ]
            except Exception:
                pass
        return list(self._fallback_history)

    def clear_history(self) -> None:
        """Clear the input history"""
        if PROMPT_TOOLKIT_AVAILABLE and self._session is not None:
            try:
                self._session.history = InMemoryHistory()
            except Exception:
                pass

        self._fallback_history.clear()

        # Also clear the history file
        try:
            if self.history_file.exists():
                self.history_file.unlink()
        except Exception as e:
            logger.debug(f"Could not clear history file: {e}")

    def append_to_history(self, text: str) -> None:
        """Programmatically add an entry to history"""
        if not text.strip():
            return
        try:
            if self._session is not None:
                self._session.history.append_string(text)
            else:
                self._fallback_history.append(text)
        except Exception as e:
            logger.debug(f"Could not append to history: {e}")

    def set_editing_mode(self, mode: InputMode) -> None:
        """Switch between Emacs and Vi editing modes"""
        self.editing_mode = mode
        if self._session is not None:
            self._session.editing_mode = (
                EditingMode.VI if mode == InputMode.VI else EditingMode.EMACS
            )
        logger.info(f"Editing mode switched to: {mode.value}")

    def set_base_dir(self, path: str) -> None:
        """Update the base directory for path completion"""
        self.base_dir = path
        if self._session is not None and self._session.completer is not None:
            # Rebuild the session with the new base dir
            self._session = self._create_session()

    async def prompt_with_preview(
        self,
        prompt: str,
        preview_text: str,
        preview_title: str = "Preview",
    ) -> Optional[str]:
        """
        Display a preview box above the prompt.
        Useful for diff confirmation, file content, etc.
        """
        # Print the preview box
        width = min(100, _terminal_width() - 4)
        print()
        print("┌" + "─" * (width - 2) + "┐")
        title = f" {preview_title} "
        print("│" + title.center(width - 2) + "│")
        print("├" + "─" * (width - 2) + "┤")

        for line in preview_text.splitlines()[:30]:
            display = line[: width - 4]
            print("│ " + display.ljust(width - 4) + " │")

        if len(preview_text.splitlines()) > 30:
            print("│ " + "... (truncated)".ljust(width - 4) + " │")

        print("└" + "─" * (width - 2) + "┘")
        print()

        return await self.get_input(prompt=prompt)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _terminal_width(default: int = 80) -> int:
    """Get terminal width safely"""
    try:
        return os.get_terminal_size().columns
    except OSError:
        return default


# ----------------------------------------------------------------------
# Module-level convenience functions
# ----------------------------------------------------------------------

_default_handler: Optional[InputHandler] = None


def get_input_handler(
    command_processor: Optional[Any] = None,
    **kwargs,
) -> InputHandler:
    """Get or create the global input handler"""
    global _default_handler
    if _default_handler is None or command_processor is not None:
        _default_handler = InputHandler(
            command_processor=command_processor,
            **kwargs,
        )
    return _default_handler


async def prompt(
    message: str = "🧠 ",
    default: str = "",
    validator: Optional[Callable[[str], Tuple[bool, str]]] = None,
) -> Optional[str]:
    """Convenience function: prompt for input"""
    handler = get_input_handler()
    return await handler.get_input(
        prompt=message,
        default=default,
        validator=validator,
    )


async def confirm(
    message: str,
    default: bool = False,
    timeout: Optional[float] = None,
) -> bool:
    """Convenience function: ask yes/no question"""
    handler = get_input_handler()
    return await handler.get_confirmation(message, default=default, timeout=timeout)


async def choose(
    message: str,
    choices: List[Tuple[str, str]],
    default: Optional[str] = None,
) -> Optional[str]:
    """Convenience function: choose from a list"""
    handler = get_input_handler()
    return await handler.get_choice(message, choices, default=default)


# ----------------------------------------------------------------------
# Exports
# ----------------------------------------------------------------------

__all__ = [
    "InputHandler",
    "InputMode",
    "InputResult",
    "ContinuationType",
    "CommandCompleter",
    "SlashCommandCompleter",
    "FilePathCompleter",
    "ShellHistoryCompleter",
    "InputLexer",
    "get_input_handler",
    "prompt",
    "confirm",
    "choose",
]