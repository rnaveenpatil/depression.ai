"""
CLI Module - Command-line interface components

Exports:
    CommandProcessor  — Slash command processor
    Command           — Command definition
    CommandCategory   — Command category enum
    UI                — Terminal UI
    InputHandler      — Input handling
"""

from agent.cli.commands import CommandProcessor, Command, CommandCategory
from agent.cli.ui import UI, Icons, Palette, Colors
from agent.cli.input import InputHandler

__all__ = [
    "CommandProcessor",
    "Command",
    "CommandCategory",
    "UI",
    "Icons",
    "Palette",
    "Colors",
    "InputHandler",
]