"""
Browser Tool - Headless browser automation via Playwright (optional).

Falls back gracefully when Playwright isn't installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.tools.path_guard import OutputPathError, resolve_output_path
from agent.utils.logging import get_logger

from agent.tools.web import BlockedRequestError, _assert_public_url

logger = get_logger(__name__)


class BrowserTool(BaseTool):
    name = "browser"
    description = "Interact with web pages: navigate, click, type, extract text, take screenshots."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["navigate", "click", "type", "text", "html", "screenshot", "eval", "wait"],
            },
            "url": {"type": "string"},
            "selector": {"type": "string"},
            "value": {"type": "string"},
            "script": {"type": "string"},
            "path": {"type": "string", "description": "Screenshot output path"},
            "timeout_ms": {"type": "integer", "default": 15000},
        },
        "required": ["action"],
    }
    timeout = 120.0

    read_only = False
    mutating = True
    category = "net"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.headless: bool = cfg.get("headless", True)
        self.allow_network: bool = cfg.get("allow_network", True)
        self.allow_private: bool = bool(cfg.get("allow_private_network", False))
        self.allow_eval: bool = bool(cfg.get("allow_eval", False))
        self.allowlist: List[str] = list(cfg.get("allowlist") or [])
        self.sandbox_root: Optional[str] = cfg.get("sandbox_root")
        self._browser = None
        self._page = None
        self._pw = None

    def _checked_url(self, raw: str) -> str:
        url = (raw or "").strip()
        if not url:
            raise ValueError("URL is required")
        if not url.startswith(("http://", "https://")):
            if url.startswith(("file:", "javascript:", "data:", "about:")):
                raise ValueError("URL scheme not allowed")
            url = "https://" + url
        if self.allow_private:
            import urllib.parse
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("only HTTP(S) URLs with a hostname are allowed")
            if self.allowlist and parsed.hostname.lower() not in {
                h.strip().lower() for h in self.allowlist
            }:
                raise ValueError(f"host not in allowlist: {parsed.hostname}")
        else:
            _assert_public_url(url, self.allowlist or None)
        return url

    async def _guard_route(self, route: Any) -> None:
        try:
            request_url = route.request.url
            if request_url.startswith(("data:", "blob:", "about:")):
                await route.continue_()
                return
            if self.allow_private:
                await route.continue_()
                return
            self._checked_url(request_url)
            await route.continue_()
        except Exception:
            await route.abort()

    async def _ensure_browser(self) -> Optional[str]:
        if self._browser is not None:
            return None
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return "playwright is not installed (pip install playwright && playwright install chromium)"
        try:
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=self.headless)
            context = await self._browser.new_context()
            await context.route("**/*", self._guard_route)
            self._page = await context.new_page()
            return None
        except Exception as e:
            await self.shutdown()
            return f"Failed to launch browser: {e}"

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        action = params.get("action", "")
        network_actions = {"navigate", "eval"}
        if not self.allow_network and action in network_actions:
            return {"success": False, "error": "Network access disabled"}

        err = await self._ensure_browser()
        if err:
            return {"success": False, "error": err}

        try:
            if action == "navigate":
                try:
                    url = self._checked_url(params.get("url", ""))
                except (ValueError, BlockedRequestError) as e:
                    return {"success": False, "error": str(e)}
                timeout = max(1000, min(int(params.get("timeout_ms", 15000)), 120000))
                await self._page.goto(url, timeout=timeout)
                return {"success": True, "url": self._page.url, "title": await self._page.title()}
            if action == "click":
                await self._page.click(params["selector"])
                return {"success": True}
            if action == "type":
                await self._page.fill(params["selector"], params.get("value", ""))
                return {"success": True}
            if action == "text":
                if params.get("selector"):
                    text = await self._page.inner_text(params["selector"])
                else:
                    text = await self._page.inner_text("body")
                return {"success": True, "text": text[:5000]}
            if action == "html":
                html = await self._page.content()
                return {"success": True, "html": html[:10000]}
            if action == "screenshot":
                raw = params.get("path") or "screenshot.png"
                try:
                    path = resolve_output_path(raw, self.sandbox_root)
                except OutputPathError as e:
                    return {"success": False, "error": str(e)}
                await self._page.screenshot(path=str(path))
                return {"success": True, "path": str(path)}
            if action == "eval":
                if not self.allow_eval:
                    return {"success": False, "error": "eval is disabled by browser config"}
                result = await self._page.evaluate(params.get("script", ""))
                return {"success": True, "result": result}
            if action == "wait":
                ms = max(0, min(int(params.get("timeout_ms", 1000)), 120000))
                await self._page.wait_for_timeout(ms)
                return {"success": True}
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def shutdown(self) -> None:
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
        self._browser = None
        self._page = None
        self._pw = None
