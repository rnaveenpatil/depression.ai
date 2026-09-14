"""
Web Tool - HTTP fetch and basic web search.

Features:
    - HTTP GET/POST with headers, timeout, redirects
    - Strip HTML to plain text
    - DuckDuckGo HTML search (no API key)
    - Response size limits
"""

from __future__ import annotations

import asyncio
import html
import re
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class WebTool(BaseTool):
    name = "web"
    description = "Fetch URLs, extract page text, or search the web (DuckDuckGo)."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["fetch", "search"]},
            "url": {"type": "string"},
            "query": {"type": "string"},
            "method": {"type": "string", "default": "GET"},
            "headers": {"type": "object"},
            "body": {"type": "string"},
            "max_bytes": {"type": "integer", "default": 500000},
            "text_only": {"type": "boolean", "default": True},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["action"],
    }
    timeout = 60.0

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.user_agent = cfg.get(
            "user_agent",
            "Mozilla/5.0 (compatible; CLIAgent/1.0; +https://example.com/bot)",
        )
        self.allow_network: bool = cfg.get("allow_network", True)

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_network:
            return {"success": False, "error": "Network access disabled"}

        action = params.get("action", "fetch")
        if action == "fetch":
            return await self._fetch(params)
        if action == "search":
            return await self._search(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    async def _fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url")
        if not url:
            return {"success": False, "error": "fetch requires 'url'"}
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {"User-Agent": self.user_agent}
        headers.update(params.get("headers") or {})

        method = params.get("method", "GET").upper()
        body = params.get("body")
        max_bytes = int(params.get("max_bytes", 500_000))

        # Retry with exponential backoff (opencode-style resilient fetch)
        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    r = await client.request(
                        method, url,
                        headers=headers,
                        content=body.encode() if body else None,
                    )
                break
            except Exception as e:
                last_err = e
                if attempt == 2:
                    return {"success": False, "error": str(last_err)}
                await asyncio.sleep(0.5 * (attempt + 1))

        raw = r.content[:max_bytes]
        text = raw.decode(r.encoding or "utf-8", errors="replace")

        out: Dict[str, Any] = {
            "success": r.status_code < 400,
            "url": str(r.url),
            "status": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "size": len(r.content),
            "truncated": len(r.content) > max_bytes,
        }
        if r.status_code >= 400:
            out["error"] = f"HTTP {r.status_code}"
            out["content"] = text[:2000]
            return out

        if params.get("text_only", True) and "html" in out["content_type"].lower():
            out["content"] = self._html_to_text(text)
        else:
            out["content"] = text

        return out

    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        if not query:
            return {"success": False, "error": "search requires 'query'"}

        limit = int(params.get("limit", 10))
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})

        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                r = await client.get(url, headers={"User-Agent": self.user_agent})
            except Exception as e:
                return {"success": False, "error": str(e)}

        results: List[Dict[str, str]] = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            r.text,
            re.DOTALL,
        ):
            link = html.unescape(m.group(1))
            title = self._strip_tags(m.group(2))
            results.append({"title": title, "url": link})
            if len(results) >= limit:
                break

        return {"success": True, "query": query, "results": results, "count": len(results)}

    @staticmethod
    def _html_to_text(html_text: str) -> str:
        text = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _strip_tags(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s)).strip()