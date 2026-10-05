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
import ipaddress
import re
import socket
import urllib.parse
from typing import Any, Dict, List, Optional

import httpx

from agent.tools.registry import BaseTool
from agent.utils.logging import get_logger

logger = get_logger(__name__)


class BlockedRequestError(Exception):
    """Raised when a URL resolves to a network range we refuse to reach."""


def _is_blocked_ip(ip: ipaddress._BaseAddress) -> bool:
    """True for addresses that must never be reachable from a model-driven fetch.

    Covers loopback, RFC1918/ULA private space, link-local (which is where
    the cloud metadata service lives), CGNAT, multicast, reserved, and the
    IPv4-mapped/compatibility blocks that would otherwise sneak a v4 target
    past a v6-looking check.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _is_blocked_ip(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return _is_blocked_ip(ip.sixtofour)
        if ip.teredo is not None:
            return _is_blocked_ip(ip.teredo[1])
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _assert_public_url(url: str, allowlist: Optional[List[str]] = None) -> str:
    """Validate scheme + resolved IP for every host in a URL.

    Guarding only the literal hostname is not enough: an attacker-controlled
    DNS name can resolve to 127.0.0.1 or 169.254.169.254. Every hostname in
    the URL is therefore resolved and each address is checked.
    """
    parsed = urllib.parse.urlsplit(url)

    if parsed.scheme not in ("http", "https"):
        raise BlockedRequestError(f"scheme not allowed: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise BlockedRequestError("URL has no host")

    if allowlist:
        allowed = {h.strip().lower() for h in allowlist}
        if host.lower() not in allowed:
            raise BlockedRequestError(f"host not in allowlist: {host}")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise BlockedRequestError(f"cannot resolve host {host}: {e}") from e

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise BlockedRequestError(
                f"{host} resolves to blocked address {addr}"
            )
    return url


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
        # Opt-in escape hatch: when set, only these hosts may be fetched.
        self.allowlist: List[str] = list(cfg.get("allowlist") or [])
        self.allow_private: bool = bool(cfg.get("allow_private_network", False))
        self.max_redirects: int = int(cfg.get("max_redirects", 5))

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.allow_network:
            return {"success": False, "error": "Network access disabled"}

        action = params.get("action", "fetch")
        if action == "fetch":
            return await self._fetch(params)
        if action == "search":
            return await self._search(params)
        return {"success": False, "error": f"Unknown action: {action}"}

    def _check_url(self, url: str) -> None:
        """Raise :class:`BlockedRequestError` for disallowed destinations."""
        if self.allow_private:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in ("http", "https"):
                raise BlockedRequestError(f"scheme not allowed: {parsed.scheme!r}")
            if not parsed.hostname:
                raise BlockedRequestError("URL has no host")
            if self.allowlist:
                allowed = {h.strip().lower() for h in self.allowlist}
                if parsed.hostname.lower() not in allowed:
                    raise BlockedRequestError(
                        f"host not in allowlist: {parsed.hostname}"
                    )
            return
        _assert_public_url(url, self.allowlist or None)

    async def _fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url")
        if not url:
            return {"success": False, "error": "fetch requires 'url'"}
        if not url.startswith(("http://", "https://")):
            # A bare host gets https://, but an explicit non-HTTP scheme
            # (file://, gopher://, ftp://) is a probe attempt, not a typo.
            scheme = urllib.parse.urlsplit(url).scheme
            if scheme and len(scheme) > 1:
                return {
                    "success": False,
                    "error": f"blocked: scheme not allowed: {scheme!r}",
                }
            url = "https://" + url

        headers = {"User-Agent": self.user_agent}
        headers.update(params.get("headers") or {})

        method = params.get("method", "GET").upper()
        body = params.get("body")
        max_bytes = int(params.get("max_bytes", 500_000))

        # Redirects are followed manually so each hop is re-validated: an
        # allowed host must not be able to bounce us to a link-local address.
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=30, follow_redirects=False
            ) as client:
                r = None
                last_err = None
                for attempt in range(3):
                    try:
                        self._check_url(current)
                        r = await client.request(
                            method, current,
                            headers=headers,
                            content=body.encode() if body else None,
                        )
                        break
                    except BlockedRequestError as e:
                        return {"success": False, "error": f"blocked: {e}"}
                    except Exception as e:
                        last_err = e
                        if attempt == 2:
                            return {"success": False, "error": str(last_err)}
                        await asyncio.sleep(0.5 * (attempt + 1))

                hops = 0
                while r is not None and r.is_redirect and hops < self.max_redirects:
                    nxt = r.headers.get("location")
                    if not nxt:
                        break
                    current = urllib.parse.urljoin(str(r.url), nxt)
                    try:
                        self._check_url(current)
                    except BlockedRequestError as e:
                        return {
                            "success": False,
                            "error": f"blocked redirect: {e}",
                            "url": current,
                        }
                    hops += 1
                    r = await client.request(
                        method, current,
                        headers=headers,
                        content=body.encode() if body else None,
                    )
        except BlockedRequestError as e:
            return {"success": False, "error": f"blocked: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

        if r is None:
            return {"success": False, "error": "request produced no response"}

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