"""
Realtime Browser Tool - Real-time browser automation via Playwright.

Full-featured browser control, distinct from the lightweight 'browser' tool:
    - Persistent profile (cookies/logins survive across agent restarts)
    - ARIA snapshot with stable element refs (e1, e2, ...) for reliable targeting
    - Console and network capture for debugging
    - Multi-tab management (open, switch, close)
    - Idle auto-close to prevent leaked Chromium processes
    - Session modes: ephemeral | persistent | cdp (attach to running Chrome)
    - Human handoff (pause for manual login)

Falls back gracefully when Playwright is not installed.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# SESSION STATE
# ======================================================================

@dataclass
class TabState:
    id: str
    page: Any
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


@dataclass
class ConsoleEntry:
    timestamp: float
    level: str
    text: str
    location: str = ""


@dataclass
class NetworkEntry:
    timestamp: float
    method: str
    url: str
    status: Optional[int] = None
    resource_type: str = ""
    failed: Optional[str] = None


# ======================================================================
# TOOL
# ======================================================================

class RealtimeBrowserTool(BaseTool):
    name = "realtime_browser"
    description = (
        "Control a real browser in real time: navigate, click, type, extract "
        "content, take screenshots, run JS, manage tabs, read console/network "
        "logs. Use 'snapshot' to get an ARIA tree with stable element refs "
        "(e1, e2, ...) that you can pass to 'click'/'type' via the 'ref' parameter."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate", "click", "type", "press", "hover", "select",
                    "snapshot", "text", "html", "screenshot", "eval", "wait",
                    "tab_open", "tab_list", "tab_switch", "tab_close",
                    "console", "network", "back", "forward", "reload",
                    "pause_for_user", "close",
                ],
            },
            "url": {"type": "string", "description": "For navigate / tab_open"},
            "selector": {"type": "string", "description": "CSS selector"},
            "ref": {"type": "string", "description": "Element ref from 'snapshot' (e.g. 'e3')"},
            "value": {"type": "string", "description": "For type / select"},
            "key": {"type": "string", "description": "For press, e.g. 'Enter'"},
            "script": {"type": "string", "description": "For eval"},
            "path": {"type": "string", "description": "Screenshot output path"},
            "tab_id": {"type": "string", "description": "Tab id for tab_switch/tab_close"},
            "timeout_ms": {"type": "integer", "default": 15000},
            "clear": {"type": "boolean", "default": False, "description": "Clear console/network buffer before reading"},
            "wait_until": {
                "type": "string",
                "enum": ["load", "domcontentloaded", "networkidle", "commit"],
                "default": "domcontentloaded",
            },
            "prompt": {"type": "string", "description": "Message to show user during pause_for_user"},
        },
        "required": ["action"],
    }
    timeout = 120.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.headless: bool = cfg.get("headless", True)
        self.allow_network: bool = cfg.get("allow_network", True)
        self.session_mode: str = cfg.get("session_mode", "ephemeral")
        self.cdp_url: str = cfg.get("cdp_url", "http://localhost:9222")
        self.profile_dir: str = cfg.get(
            "profile_dir",
            str(Path.home() / ".cache" / "agent-browser" / "profile"),
        )
        self.idle_timeout: float = float(cfg.get("idle_timeout", 300.0))
        self.max_tabs: int = int(cfg.get("max_tabs", 10))
        self.console_buffer: int = int(cfg.get("console_buffer", 500))
        self.network_buffer: int = int(cfg.get("network_buffer", 500))
        self.user_agent: Optional[str] = cfg.get("user_agent")
        self.viewport: Dict[str, int] = cfg.get(
            "viewport", {"width": 1280, "height": 800}
        )

        self._pw = None
        self._context = None
        self._browser = None

        self._tabs: Dict[str, TabState] = {}
        self._active_tab: Optional[str] = None
        self._tab_counter = 0

        self._console: List[ConsoleEntry] = []
        self._network: List[NetworkEntry] = []

        self._idle_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def _ensure_browser(self) -> Optional[str]:
        if self._context is not None:
            return None
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return (
                "playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )

        try:
            self._pw = await async_playwright().start()

            if self.session_mode == "cdp":
                self._browser = await self._pw.chromium.connect_over_cdp(self.cdp_url)
                contexts = self._browser.contexts
                self._context = contexts[0] if contexts else await self._browser.new_context()
            elif self.session_mode == "persistent":
                Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
                self._context = await self._pw.chromium.launch_persistent_context(
                    self.profile_dir,
                    headless=self.headless,
                    viewport=self.viewport,
                    user_agent=self.user_agent,
                )
            else:
                self._browser = await self._pw.chromium.launch(headless=self.headless)
                self._context = await self._browser.new_context(
                    viewport=self.viewport,
                    user_agent=self.user_agent,
                )

            self._context.on("page", lambda p: asyncio.create_task(self._wire_page(p)))
            for p in self._context.pages:
                await self._wire_page(p)

            self._schedule_idle_close()
            return None
        except Exception as e:
            logger.error("Failed to launch browser: %s", e, exc_info=True)
            return f"Failed to launch browser: {e}"

    async def _wire_page(self, page: Any) -> None:
        page.on("console", lambda msg: self._on_console(msg))
        page.on("pageerror", lambda exc: self._on_console_error(exc))
        page.on("requestfailed", lambda req: self._on_request_failed(req))
        page.on("response", lambda resp: self._on_response(resp))

    def _on_console(self, msg: Any) -> None:
        try:
            self._console.append(
                ConsoleEntry(
                    timestamp=time.time(),
                    level=getattr(msg, "type", "log"),
                    text=getattr(msg, "text", ""),
                    location=str(getattr(msg, "location", "")),
                )
            )
            if len(self._console) > self.console_buffer:
                del self._console[: len(self._console) - self.console_buffer]
        except Exception:
            pass

    def _on_console_error(self, exc: Any) -> None:
        self._console.append(
            ConsoleEntry(timestamp=time.time(), level="pageerror", text=str(exc))
        )
        if len(self._console) > self.console_buffer:
            del self._console[: len(self._console) - self.console_buffer]

    def _on_request_failed(self, req: Any) -> None:
        try:
            self._network.append(
                NetworkEntry(
                    timestamp=time.time(),
                    method=getattr(req, "method", ""),
                    url=getattr(req, "url", ""),
                    resource_type=getattr(req, "resource_type", ""),
                    failed=str(getattr(req, "failure", "") or "failed"),
                )
            )
            if len(self._network) > self.network_buffer:
                del self._network[: len(self._network) - self.network_buffer]
        except Exception:
            pass

    def _on_response(self, resp: Any) -> None:
        try:
            self._network.append(
                NetworkEntry(
                    timestamp=time.time(),
                    method=getattr(resp.request, "method", ""),
                    url=getattr(resp, "url", ""),
                    status=getattr(resp, "status", None),
                    resource_type=getattr(resp.request, "resource_type", ""),
                )
            )
            if len(self._network) > self.network_buffer:
                del self._network[: len(self._network) - self.network_buffer]
        except Exception:
            pass

    def _schedule_idle_close(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_watchdog())

    async def _idle_watchdog(self) -> None:
        try:
            while self._context is not None:
                await asyncio.sleep(30)
                if not self._tabs:
                    continue
                newest = max(t.last_used for t in self._tabs.values())
                if time.time() - newest > self.idle_timeout:
                    logger.info("Idle browser closed after %.0fs", self.idle_timeout)
                    await self._close_context()
                    return
        except asyncio.CancelledError:
            pass

    async def _close_context(self) -> None:
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._context = None
        self._browser = None
        self._pw = None
        self._tabs.clear()
        self._active_tab = None

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------

    async def _new_tab(self, url: str = "about:blank") -> TabState:
        if len(self._tabs) >= self.max_tabs:
            lru_id = min(self._tabs, key=lambda k: self._tabs[k].last_used)
            await self._close_tab(lru_id)

        self._tab_counter += 1
        tab_id = f"t{self._tab_counter}"
        page = await self._context.new_page()
        await self._wire_page(page)
        if url and url != "about:blank":
            await page.goto(url, wait_until="domcontentloaded")
        tab = TabState(id=tab_id, page=page)
        self._tabs[tab_id] = tab
        self._active_tab = tab_id
        return tab

    async def _close_tab(self, tab_id: str) -> bool:
        tab = self._tabs.pop(tab_id, None)
        if not tab:
            return False
        try:
            await tab.page.close()
        except Exception:
            pass
        if self._active_tab == tab_id:
            self._active_tab = next(iter(self._tabs), None)
        return True

    async def _get_active_tab(
        self, create_if_missing: bool = True
    ) -> Optional[TabState]:
        if self._active_tab and self._active_tab in self._tabs:
            tab = self._tabs[self._active_tab]
            tab.last_used = time.time()
            return tab
        if self._tabs:
            self._active_tab = next(iter(self._tabs))
            return self._tabs[self._active_tab]
        if create_if_missing:
            return await self._new_tab()
        return None

    # ------------------------------------------------------------------
    # SNAPSHOT (ARIA tree with stable refs)
    # ------------------------------------------------------------------

    async def _snapshot(self, page: Any) -> Dict[str, Any]:
        js = r"""
        () => {
          const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return false;
            const s = window.getComputedStyle(el);
            if (s.visibility === 'hidden' || s.display === 'none' || s.opacity === '0') return false;
            return true;
          };
          const isInteractive = (el) => {
            const tag = el.tagName.toLowerCase();
            if (['a','button','input','select','textarea','summary','details'].includes(tag)) return true;
            const role = el.getAttribute('role');
            if (role && ['button','link','checkbox','radio','tab','menuitem','option','switch','textbox','combobox'].includes(role)) return true;
            if (el.hasAttribute('onclick') || el.hasAttribute('contenteditable')) return true;
            return false;
          };
          const label = (el) => {
            const aria = el.getAttribute('aria-label');
            if (aria) return aria.trim();
            const text = (el.innerText || el.textContent || '').trim();
            if (text) return text.slice(0, 120);
            const ph = el.getAttribute('placeholder');
            if (ph) return ph;
            const alt = el.getAttribute('alt');
            if (alt) return alt;
            const title = el.getAttribute('title');
            if (title) return title;
            const name = el.getAttribute('name');
            if (name) return name;
            return el.tagName.toLowerCase();
          };

          document.querySelectorAll('[data-agent-ref]').forEach(n => n.removeAttribute('data-agent-ref'));

          let counter = 0;
          const lines = [];
          const walk = (el, depth) => {
            if (depth > 30 || !el) return;
            const tag = el.tagName ? el.tagName.toLowerCase() : '';
            if (['script','style','noscript','template'].includes(tag)) return;
            let ref = null;
            if (isInteractive(el) && isVisible(el)) {
              counter += 1;
              ref = 'e' + counter;
              el.setAttribute('data-agent-ref', ref);
            }
            const visible = el === document.body || isVisible(el);
            if (visible && (ref || (el.innerText || '').trim().length > 0)) {
              const ind = '  '.repeat(depth);
              const role = el.getAttribute('role') || tag;
              const text = ref ? label(el) : (el.innerText || '').trim().split('\n')[0].slice(0, 100);
              lines.push(`${ind}${ref ? '['+ref+'] ' : ''}${role}${text ? ': ' + text : ''}`);
            }
            for (const child of el.children || []) walk(child, depth + 1);
          };
          walk(document.body, 0);
          return { lines: lines.slice(0, 400), refs: counter, url: location.href, title: document.title };
        }
        """
        try:
            data = await page.evaluate(js)
        except Exception as e:
            return {"success": False, "error": f"snapshot failed: {e}"}
        return {
            "success": True,
            "url": data.get("url"),
            "title": data.get("title"),
            "refs": data.get("refs"),
            "snapshot": "\n".join(data.get("lines") or []),
        }

    async def _resolve_locator(self, page: Any, params: Dict[str, Any]) -> Any:
        ref = params.get("ref")
        selector = params.get("selector")
        if ref:
            return page.locator(f'[data-agent-ref="{ref}"]')
        if selector:
            return page.locator(selector)
        raise ValueError("Provide 'ref' (from snapshot) or 'selector'")

    # ------------------------------------------------------------------
    # DISPATCH
    # ------------------------------------------------------------------

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_network and params.get("action") in ("navigate", "tab_open"):
            return {"success": False, "error": "Network access disabled"}

        action = params.get("action", "")
        if action == "close":
            await self._close_context()
            return {"success": True, "closed": True}

        async with self._lock:
            err = await self._ensure_browser()
            if err:
                return {"success": False, "error": err}

            try:
                return await self._dispatch(action, params)
            except Exception as e:
                logger.error("Browser action '%s' failed: %s", action, e, exc_info=True)
                return {"success": False, "error": str(e)}

    async def _dispatch(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        timeout = int(params.get("timeout_ms", 15000))

        if action == "tab_open":
            url = params.get("url", "about:blank")
            tab = await self._new_tab(url)
            return {
                "success": True,
                "tab_id": tab.id,
                "url": tab.page.url,
                "title": await tab.page.title(),
            }
        if action == "tab_list":
            return {
                "success": True,
                "active": self._active_tab,
                "tabs": [
                    {"id": t.id, "url": t.page.url, "title": await t.page.title()}
                    for t in self._tabs.values()
                ],
            }
        if action == "tab_switch":
            tid = params.get("tab_id")
            if tid not in self._tabs:
                return {"success": False, "error": f"Tab {tid} not found"}
            self._active_tab = tid
            self._tabs[tid].last_used = time.time()
            return {"success": True, "active": tid, "url": self._tabs[tid].page.url}
        if action == "tab_close":
            tid = params.get("tab_id") or self._active_tab
            ok = await self._close_tab(tid)
            return {"success": ok, "closed": tid if ok else None}

        tab = await self._get_active_tab()
        if tab is None:
            return {"success": False, "error": "No active tab and none could be created"}
        page = tab.page
        tab.last_used = time.time()

        if action == "navigate":
            url = params.get("url", "")
            if not url:
                return {"success": False, "error": "navigate requires 'url'"}
            if not url.startswith(("http://", "https://", "file://", "about:")):
                url = "https://" + url
            await page.goto(
                url, timeout=timeout, wait_until=params.get("wait_until", "domcontentloaded")
            )
            return {
                "success": True,
                "tab_id": tab.id,
                "url": page.url,
                "title": await page.title(),
            }

        if action == "back":
            await page.go_back(timeout=timeout)
            return {"success": True, "url": page.url}
        if action == "forward":
            await page.go_forward(timeout=timeout)
            return {"success": True, "url": page.url}
        if action == "reload":
            await page.reload(timeout=timeout)
            return {"success": True, "url": page.url}

        if action == "click":
            loc = await self._resolve_locator(page, params)
            await loc.click(timeout=timeout)
            return {"success": True, "url": page.url}

        if action == "type":
            loc = await self._resolve_locator(page, params)
            await loc.fill(params.get("value", ""), timeout=timeout)
            return {"success": True}

        if action == "press":
            if params.get("ref") or params.get("selector"):
                loc = await self._resolve_locator(page, params)
            else:
                loc = page
            await loc.press(params.get("key", "Enter"), timeout=timeout)
            return {"success": True}

        if action == "hover":
            loc = await self._resolve_locator(page, params)
            await loc.hover(timeout=timeout)
            return {"success": True}

        if action == "select":
            loc = await self._resolve_locator(page, params)
            await loc.select_option(params.get("value", ""), timeout=timeout)
            return {"success": True}

        if action == "snapshot":
            return await self._snapshot(page)

        if action == "text":
            if params.get("ref") or params.get("selector"):
                loc = await self._resolve_locator(page, params)
                text = await loc.inner_text(timeout=timeout)
            else:
                text = await page.inner_text("body", timeout=timeout)
            return {
                "success": True,
                "text": text[:10000],
                "truncated": len(text) > 10000,
            }

        if action == "html":
            html = await page.content()
            return {
                "success": True,
                "html": html[:20000],
                "truncated": len(html) > 20000,
            }

        if action == "screenshot":
            path = params.get("path") or f"screenshot_{int(time.time())}.png"
            await page.screenshot(path=path, full_page=True)
            return {"success": True, "path": os.path.abspath(path)}

        if action == "eval":
            script = params.get("script", "")
            if not script:
                return {"success": False, "error": "eval requires 'script'"}
            result = await page.evaluate(script)
            return {"success": True, "result": result}

        if action == "wait":
            ms = int(params.get("timeout_ms", 1000))
            await page.wait_for_timeout(ms)
            return {"success": True, "waited_ms": ms}

        if action == "console":
            if params.get("clear"):
                self._console.clear()
            return {
                "success": True,
                "entries": [e.__dict__ for e in self._console[-100:]],
                "count": len(self._console),
            }

        if action == "network":
            if params.get("clear"):
                self._network.clear()
            return {
                "success": True,
                "entries": [e.__dict__ for e in self._network[-100:]],
                "count": len(self._network),
            }

        if action == "pause_for_user":
            prompt = params.get(
                "prompt", "Browser paused for manual interaction. Resume when ready."
            )
            return {
                "success": True,
                "needs_user_input": True,
                "prompt": prompt,
                "url": page.url,
                "message": (
                    "Take over the browser window, then call "
                    "realtime_browser with action='snapshot' to resume."
                ),
            }

        return {"success": False, "error": f"Unknown action: {action}"}

    # ------------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
        await self._close_context()