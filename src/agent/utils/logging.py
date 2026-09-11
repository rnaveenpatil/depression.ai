"""
Logging Module - Centralized logging with rotation, colored console output,
and an in-memory ring buffer for `/logs` command support.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# ======================================================================
# CONFIG
# ======================================================================

DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_BUFFER_SIZE = 2000
_log_buffer: deque = deque(maxlen=_LOG_BUFFER_SIZE)
_buffer_lock = threading.Lock()

_configured = False
_console_handler: Optional[logging.Handler] = None
_file_handler: Optional[logging.Handler] = None


# ======================================================================
# COLOR SUPPORT
# ======================================================================

_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[1;41m" # white on red
}
_RESET = "\033[0m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False


class _ColorFormatter(logging.Formatter):
    """Formatter that colorizes the level name."""

    def __init__(self, fmt: str, datefmt: str, color: bool = True):
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        if self.color:
            color = _COLORS.get(record.levelname, "")
            original = record.levelname
            record.levelname = f"{color}{original}{_RESET}"
            try:
                return super().format(record)
            finally:
                record.levelname = original
        return super().format(record)


class _RingBufferHandler(logging.Handler):
    """Keeps the last N records in memory."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "timestamp": record.created,
                "pathname": record.pathname,
                "lineno": record.lineno,
            }
            with _buffer_lock:
                _log_buffer.append(entry)
        except Exception:
            pass


# ======================================================================
# SETUP
# ======================================================================

def setup_logging(
    level: int | str = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console: bool = True,
    color: Optional[bool] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    quiet_libraries: Optional[List[str]] = None,
) -> None:
    """
    Configure the root logger. Idempotent.
    """
    global _configured, _console_handler, _file_handler

    root = logging.getLogger()
    if _configured:
        root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
        return

    root.handlers.clear()
    root.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))

    if color is None:
        color = _supports_color()

    # Console handler
    if console:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setFormatter(_ColorFormatter(fmt, date_format, color=color))
        ch.setLevel(logging.NOTSET)
        root.addHandler(ch)
        _console_handler = ch

    # File handler
    if log_file:
        path = Path(os.path.expanduser(log_file))
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(fmt, date_format))
        fh.setLevel(logging.NOTSET)
        root.addHandler(fh)
        _file_handler = fh

    # Ring buffer for /logs
    rb = _RingBufferHandler()
    rb.setLevel(logging.NOTSET)
    root.addHandler(rb)

    # Quiet noisy libraries
    for lib in (quiet_libraries or [
        "httpx", "httpcore", "urllib3", "asyncio", "botocore", "boto3",
        "s3transfer", "urllib3.connectionpool", "websockets",
    ]):
        logging.getLogger(lib).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("Logging configured")


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger."""
    return logging.getLogger(name)


def set_log_level(level: int | str) -> None:
    """Change the root logger level at runtime."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)


def get_recent_logs(
    level: Optional[str] = None,
    limit: int = 100,
    name_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return the most recent log records from the in-memory ring buffer.
    """
    min_level = 0
    if level:
        min_level = getattr(logging, level.upper(), 0)

    out: List[Dict[str, Any]] = []
    with _buffer_lock:
        for entry in reversed(_log_buffer):
            if level and getattr(logging, entry["level"], 0) < min_level:
                continue
            if name_filter and name_filter not in entry["name"]:
                continue
            out.append(entry)
            if len(out) >= limit:
                break
    return list(reversed(out))


def clear_log_buffer() -> None:
    with _buffer_lock:
        _log_buffer.clear()


__all__ = [
    "get_logger",
    "setup_logging",
    "get_recent_logs",
    "set_log_level",
    "clear_log_buffer",
]