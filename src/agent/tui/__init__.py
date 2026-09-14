"""
TUI Module - Futuristic Terminal User Interface for depression.ai

A next-generation TUI built with Textual, featuring:
- Cyberpunk-inspired neon theme with truecolor gradients
- Split-panel layout with resizable sidebar
- Real-time streaming chat with markdown rendering
- Live tool execution visualization
- Session management with persistence
- Interactive command palette
- Animated status indicators
- File tree browser
- Diff viewer
"""

from agent.tui.app import DepressionTUI

__all__ = ["DepressionTUI"]
