"""
Browser Tool - Headless browser automation via Playwright (optional).

Falls back gracefully when Playwright isn't installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.headless: bool = cfg.get("headless", True)
        self.allow_network: bool = cfg.get("allow_network", True)
        self._browser = None
        self._page = None

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
            self._page = await self._browser.new_page()
            return None
        except Exception as e:
            return f"Failed to launch browser: {e}"

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_network:
            return {"success": False, "error": "Network access disabled"}

        err = await self._ensure_browser()
        if err:
            return {"success": False, "error": err}

        action = params.get("action", "")
        try:
            if action == "navigate":
                url = params.get("url", "")
                if not url.startswith("http"):
                    url = "https://" + url
                await self._page.goto(url, timeout=int(params.get("timeout_ms", 15000)))
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
                path = params.get("path", "screenshot.png")
                await self._page.screenshot(path=path)
                return {"success": True, "path": path}
            if action == "eval":
                result = await self._page.evaluate(params.get("script", ""))
                return {"success": True, "result": result}
            if action == "wait":
                ms = int(params.get("timeout_ms", 1000))
                await self._page.wait_for_timeout(ms)
                return {"success": True}
            return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def shutdown(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
            if getattr(self, "_pw", None):
                await self._pw.stop()
        except Exception:
            pass