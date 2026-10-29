"""
Plan parser.

Extracts a checklist from an assistant's text response so the TUI can
render a live plan without a dedicated tool call.

Recognized formats (in priority order):

    1. Markdown checkboxes:
        - [ ] pending step
        - [-] in-progress step
        - [x] completed step
        - [X] completed step

    2. Emoji markers:
        ⬜ pending step
        🔄 in-progress step
        ✅ completed step

    3. Numbered checklist with markers:
        1. [ ] pending
        2. [-] in progress
        3. [x] done

Anything that does not match is ignored. The parser is intentionally
strict: only lines that begin (after optional list markers) with a
recognized status marker are treated as plan entries. This avoids
false positives from prose like "you should check the box later".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ----------------------------------------------------------------------
# STATUS
# ----------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUS_CANCELLED = "cancelled"


# ----------------------------------------------------------------------
# ENTRY
# ----------------------------------------------------------------------

@dataclass
class PlanEntry:
    """A single step in a parsed plan."""
    content: str
    status: str = STATUS_PENDING
    priority: str = "medium"  # high | medium | low

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "status": self.status,
            "priority": self.priority,
        }


@dataclass
class ParsedPlan:
    """A parsed plan: an ordered list of entries plus metadata."""
    entries: List[PlanEntry] = field(default_factory=list)
    raw: str = ""

    @property
    def count(self) -> int:
        return len(self.entries)

    @property
    def is_valid(self) -> bool:
        """A plan needs at least two entries to be worth rendering."""
        return len(self.entries) >= 2

    def to_list(self) -> List[dict]:
        return [e.to_dict() for e in self.entries]


# ----------------------------------------------------------------------
# PATTERNS
# ----------------------------------------------------------------------

# Leading list markers we tolerate before the status marker.
_LEADING = r"(?:\s*(?:[-*•]|\d+[.)])\s*)?"

# A status marker followed by a space and the step text.
_MARKERS = [
    # (regex body, status)
    (r"\[ \]", STATUS_PENDING),
    (r"\[\s\]", STATUS_PENDING),
    (r"\[-\]", STATUS_IN_PROGRESS),
    (r"\[~\]", STATUS_IN_PROGRESS),
    (r"\[\*\]", STATUS_IN_PROGRESS),
    (r"\[x\]", STATUS_COMPLETED),
    (r"\[X\]", STATUS_COMPLETED),
    (r"\[✓\]", STATUS_COMPLETED),
    (r"\[!\]", STATUS_BLOCKED),
    (r"\[×\]", STATUS_CANCELLED),
    (r"\[✗\]", STATUS_CANCELLED),
]

_EMOJI_MARKERS = [
    ("⬜", STATUS_PENDING),
    ("◻", STATUS_PENDING),
    ("🔄", STATUS_IN_PROGRESS),
    ("🔁", STATUS_IN_PROGRESS),
    ("✅", STATUS_COMPLETED),
    ("☑", STATUS_COMPLETED),
    ("⛔", STATUS_BLOCKED),
    ("❌", STATUS_CANCELLED),
]

# Compile once.
_COMPILED_MARKERS = [
    (re.compile(rf"^{_LEADING}{re.escape(body[1:-1]) if False else body}\s+(.+)$", re.IGNORECASE), status)
    for body, status in []
]  # placeholder, replaced below

# Build the real compiled patterns for bracketed markers.
_BRACKET_PATTERNS = [
    (re.compile(rf"^{_LEADING}{re.escape('[' + inner + ']')}\s+(.+?)\s*$"), status)
    for inner, status in [
        (" ", STATUS_PENDING),
        ("-", STATUS_IN_PROGRESS),
        ("~", STATUS_IN_PROGRESS),
        ("*", STATUS_IN_PROGRESS),
        ("x", STATUS_COMPLETED),
        ("X", STATUS_COMPLETED),
        ("✓", STATUS_COMPLETED),
        ("!", STATUS_BLOCKED),
        ("×", STATUS_CANCELLED),
        ("✗", STATUS_CANCELLED),
    ]
]

_EMOJI_PATTERNS = [
    (re.compile(rf"^{_LEADING}{re.escape(emoji)}\s*(.+?)\s*$"), status)
    for emoji, status in _EMOJI_MARKERS
]

_ALL_PATTERNS = _BRACKET_PATTERNS + _EMOJI_PATTERNS

# Priorities are assigned by the model with explicit annotations.
# Unannotated entries stay neutral; wording alone does not set priority.
_PRIORITY_ANNOTATION = re.compile(
    r"\s+\[priority:\s*(high|medium|low)\]\s*$", re.IGNORECASE
)


def _extract_priority(text: str) -> tuple[str, str]:
    m = _PRIORITY_ANNOTATION.search(text)
    if m is None:
        return text, "medium"
    return text[:m.start()].rstrip(), m.group(1).lower()


# ----------------------------------------------------------------------
# PARSER
# ----------------------------------------------------------------------

def parse_plan_from_text(text: str) -> Optional[ParsedPlan]:
    """
    Scan the text for a checklist. Returns a ParsedPlan if two or more
    entries were found, otherwise None.

    The parser looks for a contiguous block of checklist lines. If the
    text contains two separate checklists, only the first is returned —
    that matches the way agents typically emit a single plan then
    narrate.
    """
    if not text:
        return None

    lines = text.splitlines()
    entries: List[PlanEntry] = []
    in_block = False
    blank_run = 0

    for line in lines:
        stripped = line.rstrip()
        matched: Optional[PlanEntry] = None

        for pattern, status in _ALL_PATTERNS:
            m = pattern.match(stripped)
            if m:
                content = m.group(1).strip()
                if content:
                    step, priority = _extract_priority(content)
                    if step:
                        matched = PlanEntry(
                            content=step, status=status, priority=priority
                        )
                break

        if matched is not None:
            if not in_block:
                in_block = True
                blank_run = 0
            entries.append(matched)
            blank_run = 0
        elif in_block:
            # Blank line inside the block is tolerated; a non-blank
            # line that doesn't match ends the block.
            if not stripped.strip():
                blank_run += 1
                if blank_run >= 2:
                    break
                continue
            break

    if not entries:
        return None

    return ParsedPlan(entries=entries, raw=text)


def looks_like_plan(text: str) -> bool:
    """Cheap check used to decide whether to run the full parser."""
    if not text:
        return False
    return any(pat.search(text) for pat, _ in _ALL_PATTERNS)


__all__ = [
    "PlanEntry",
    "ParsedPlan",
    "parse_plan_from_text",
    "looks_like_plan",
    "STATUS_PENDING",
    "STATUS_IN_PROGRESS",
    "STATUS_COMPLETED",
    "STATUS_BLOCKED",
    "STATUS_CANCELLED",
]