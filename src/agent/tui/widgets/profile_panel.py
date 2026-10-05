"""
Profile panel — the signed-in user, as read from the local profile.

Fields shown:
    photo    the Google avatar URL (rendered as a short link; the TUI
             cannot render images, so we show the URL and the initials)
    name     display name
    email    gmail address
    uid      Firebase localId
    provider google.com
    verified email_verified flag
    last     last login time
    counts   login count, created at

Reads straight from the ProfileStore so it is always current, and
refreshes when the app tells it the identity changed.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from agent.tui.onboarding.profile import ProfileStore, UserProfile


GREEN = "#00ff66"
GREEN_DIM = "#00aa44"
GREEN_GLOW = "#88ffbb"
AMBER = "#ffcc44"
ERROR = "#ff4466"
TEXT = "#aaffcc"
MUTED = "#3d8c5c"
DIM = "#1a5c33"
BORDER = "#0a3d20"


def _esc(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("[", r"\[")


def _fmt_ts(ts: Any) -> str:
    try:
        if not ts:
            return "—"
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "—"


class ProfilePanel(Vertical):
    DEFAULT_CSS = f"""
    ProfilePanel {{
        height: auto;
        width: 100%;
        padding: 0 1;
    }}
    ProfilePanel > Static {{
        height: auto;
        width: 100%;
    }}
    ProfilePanel .plabel {{ color: {MUTED}; }}
    ProfilePanel .pvalue {{ color: {TEXT}; }}
    ProfilePanel .pname  {{ color: {GREEN_GLOW}; text-style: bold; }}
    ProfilePanel .pemail {{ color: {GREEN}; }}
    ProfilePanel .pdivider {{ color: {BORDER}; }}
    ProfilePanel .pmuted {{ color: {DIM}; }}
    """

    def __init__(self, store: Optional[ProfileStore] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._store = store if store is not None else ProfileStore()
        self._body: Static | None = None

    def compose(self) -> ComposeResult:
        self._body = Static(self._build_markup(), markup=True)
        yield self._body

    def on_mount(self) -> None:
        # Refresh once a second in case onboarding completes after mount.
        self.set_interval(1.0, self.refresh_profile)

    def refresh_profile(self) -> None:
        if self._body is None:
            return
        try:
            self._body.update(self._build_markup())
        except Exception:
            pass

    def _build_markup(self) -> str:
        profile = self._store.load()
        if profile is None:
            return (
                f"[bold {GREEN}]▌ PROFILE[/]\n\n"
                f"[{MUTED}]not signed in[/]\n"
                f"[{DIM}]sign in with Google from the welcome page[/]"
            )

        name = profile.display_name or "—"
        initials = profile.initials or "?"
        email = profile.email or "—"
        uid = profile.uid or "—"
        provider = profile.provider or "—"
        verified = "yes" if profile.email_verified else "no"
        verified_color = GREEN if profile.email_verified else AMBER
        photo = profile.photo_url or "—"
        created = _fmt_ts(profile.created_at)
        last = _fmt_ts(profile.last_login_at)
        count = profile.login_count or 1
        gmail = "yes" if profile.is_gmail else "no"

        lines = [
            f"[bold {GREEN}]▌ PROFILE[/]",
            "",
            f"[{GREEN_GLOW}][bold]{_esc(initials):>4}[/bold][/]  "
            f"[bold {GREEN_GLOW}]{_esc(name)}[/]",
            f"     [{GREEN}]{_esc(email)}[/]",
            "",
            f"[{MUTED}]provider[/]    [{TEXT}]{_esc(provider)}[/]",
            f"[{MUTED}]gmail[/]       [{TEXT}]{gmail}[/]",
            f"[{MUTED}]verified[/]    [{verified_color}]{verified}[/]",
            f"[{MUTED}]uid[/]         [{DIM}]{_esc(uid[:20] + ('…' if len(uid) > 20 else ''))}[/]",
            f"[{MUTED}]logins[/]      [{TEXT}]{count}[/]",
            f"[{MUTED}]created[/]     [{DIM}]{_esc(created)}[/]",
            f"[{MUTED}]last login[/]  [{DIM}]{_esc(last)}[/]",
            "",
            f"[{MUTED}]photo[/]",
        ]
        if photo and photo != "—":
            short = photo if len(photo) <= 42 else photo[:39] + "…"
            lines.append(f"[{DIM}]{_esc(short)}[/]")
        else:
            lines.append(f"[{DIM}]—[/]")
        return "\n".join(lines)


__all__ = ["ProfilePanel"]