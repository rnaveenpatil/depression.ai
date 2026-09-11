"""
Session Module - Conversation and task state management.

Exports:
    SessionManager  — orchestrates session lifecycle (create/load/save/delete)
    Session         — single conversation session with messages and metadata
    SessionHistory  — append-only message log with search
    SessionState    — runtime state (tokens, cost, plan, current task)
"""

from agent.session.session import Session, SessionManager
from agent.session.history import SessionHistory, HistoryEntry
from agent.session.state import SessionState, SessionStatus

__all__ = [
    "Session",
    "SessionManager",
    "SessionHistory",
    "HistoryEntry",
    "SessionState",
    "SessionStatus",
]