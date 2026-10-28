"""
Web tool for depression.ai.

Features:
- Direct HTTP fetching first (fast path)
- Optional Chromium/Playwright fallback for JavaScript-rendered pages
- DuckDuckGo search with Bing fallback (both retained)
- Public-network URL checks to reduce SSRF risk
- Main-content extraction, language metadata, and bounded output

Optional browser support:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import asyncio
import base64
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
    """Return True for private, local, reserved, or otherwise unsafe addresses."""
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
    """Validate URL scheme, hostname, allowlist, and resolved IP addresses."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedRequestError(f"scheme not allowed: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise BlockedRequestError("URL has no host")
    if parsed.username or parsed.password:
        raise BlockedRequestError("URLs containing embedded credentials are blocked")
    if allowlist and host.lower() not in {h.strip().lower() for h in allowlist}:
        raise BlockedRequestError(f"host not in allowlist: {host}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise BlockedRequestError(f"cannot resolve host {host}: {exc}") from exc
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _is_blocked_ip(ip):
            raise BlockedRequestError(f"{host} resolves to blocked address {ip}")
    return url


class WebTool(BaseTool):
    name = "web"
    description = (
        "Search the web using DuckDuckGo then Bing, fetch URLs directly, and "
        "optionally use Chromium as a fallback for JavaScript-rendered pages."
    )
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
            "browser_fallback": {
                "type": "boolean",
                "default": True,
                "description": "Use Chromium if HTTP fetch fails or returns likely JavaScript-shell content.",
            },
            "wait_until": {
                "type": "string",
                "enum": ["domcontentloaded", "load", "networkidle"],
                "default": "domcontentloaded",
            },
        },
        "required": ["action"],
    }

    timeout = 60.0
    read_only = True
    mutating = False
    category = "net"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.user_agent = cfg.get(
            "user_agent",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        )
        self.allow_network: bool = cfg.get("allow_network", True)
        self.allowlist: List[str] = list(cfg.get("allowlist") or [])
        self.allow_private: bool = bool(cfg.get("allow_private_network", False))
        self.max_redirects: int = int(cfg.get("max_redirects", 5))
        self.browser_enabled: bool = bool(cfg.get("browser_fallback", True))
        self.browser_timeout_ms: int = int(cfg.get("browser_timeout_ms", 25000))

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
        if self.allow_private:
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise BlockedRequestError("only HTTP(S) URLs with a hostname are allowed")
            if parsed.username or parsed.password:
                raise BlockedRequestError("URLs containing embedded credentials are blocked")
            if self.allowlist and parsed.hostname.lower() not in {
                h.strip().lower() for h in self.allowlist
            }:
                raise BlockedRequestError(f"host not in allowlist: {parsed.hostname}")
            return
        _assert_public_url(url, self.allowlist or None)

    async def _fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url")
        if not url:
            return {"success": False, "error": "fetch requires 'url'"}
        if not url.startswith(("http://", "https://")):
            scheme = urllib.parse.urlsplit(url).scheme
            if scheme and len(scheme) > 1:
                return {"success": False, "error": f"blocked: scheme not allowed: {scheme!r}"}
            url = "https://" + url

        headers = {"User-Agent": self.user_agent}
        headers.update(params.get("headers") or {})
        method = str(params.get("method", "GET")).upper()
        body = params.get("body")
        max_bytes = max(1024, min(int(params.get("max_bytes", 500_000)), 2_000_000))
        text_only = params.get("text_only", True)
        browser_fallback = bool(params.get("browser_fallback", self.browser_enabled))
        wait_until = params.get("wait_until", "domcontentloaded")
        if wait_until not in {"domcontentloaded", "load", "networkidle"}:
            wait_until = "domcontentloaded"

        result = await self._fetch_http(
            url=url,
            method=method,
            headers=headers,
            body=body,
            max_bytes=max_bytes,
            text_only=text_only,
        )

        # Chromium is a fallback, not the default: HTTP is cheaper and faster.
        should_try_browser = (
            browser_fallback
            and method == "GET"
            and (
                not result.get("success")
                or self._looks_like_js_shell(result.get("content", ""))
                or result.get("status") in {401, 403, 406, 429}
            )
        )
        if should_try_browser:
            browser_result = await self._fetch_browser(
                url, max_bytes=max_bytes, wait_until=wait_until
            )
            if browser_result.get("success") and len(browser_result.get("content", "")) > len(
                result.get("content", "")
            ):
                browser_result["fallback"] = "chromium"
                browser_result["http_attempt"] = {
                    "success": result.get("success"),
                    "status": result.get("status"),
                    "error": result.get("error"),
                }
                return browser_result
            if not result.get("success") and not browser_result.get("success"):
                result["browser_fallback_error"] = browser_result.get("error", "Chromium failed")
        return result

    async def _fetch_http(
        self,
        url: str,
        method: str,
        headers: Dict[str, str],
        body: Optional[str],
        max_bytes: int,
        text_only: bool,
    ) -> Dict[str, Any]:
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(25.0), follow_redirects=False
            ) as client:
                response = None
                for _ in range(3):
                    self._check_url(current)
                    try:
                        response = await client.request(
                            method,
                            current,
                            headers=headers,
                            content=body.encode("utf-8") if body else None,
                        )
                        break
                    except BlockedRequestError:
                        raise
                    except Exception:
                        await asyncio.sleep(0.35)
                if response is None:
                    return {"success": False, "url": current, "error": "HTTP request failed"}

                hops = 0
                while response.is_redirect and hops < self.max_redirects:
                    location = response.headers.get("location")
                    if not location:
                        break
                    current = urllib.parse.urljoin(str(response.url), location)
                    self._check_url(current)
                    response = await client.request(method, current, headers=headers)
                    hops += 1

            raw = response.content[:max_bytes]
            charset = response.encoding or "utf-8"
            decoded = raw.decode(charset, errors="replace")
            content_type = response.headers.get("content-type", "")
            out: Dict[str, Any] = {
                "success": response.status_code < 400,
                "url": str(response.url),
                "status": response.status_code,
                "content_type": content_type,
                "size": len(response.content),
                "truncated": len(response.content) > max_bytes,
                "fetch_method": "http",
            }
            if response.status_code >= 400:
                out["error"] = f"HTTP {response.status_code}"
                out["content"] = decoded[:2000]
                return out

            if text_only and "html" in content_type.lower():
                out["title"] = self._extract_title(decoded)
                out["language"] = self._extract_language(decoded)
                out["content"] = self._html_to_text(decoded)
            else:
                out["content"] = decoded
            return out
        except BlockedRequestError as exc:
            return {"success": False, "url": current, "error": f"blocked: {exc}"}
        except Exception as exc:
            logger.debug("HTTP web fetch failed for %s: %s", current, exc)
            return {"success": False, "url": current, "error": str(exc)}

    async def _fetch_browser(
        self, url: str, max_bytes: int, wait_until: str
    ) -> Dict[str, Any]:
        """Fetch a page with Chromium when raw HTTP is insufficient."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {
                "success": False,
                "url": url,
                "error": "Chromium fallback unavailable. Install with: pip install playwright && playwright install chromium",
            }

        try:
            self._check_url(url)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=self.user_agent,
                        java_script_enabled=True,
                        ignore_https_errors=False,
                    )

                    # Validate every browser request, including subresources, to
                    # reduce SSRF risk. Block non-web schemes and unsafe hosts.
                    async def guard_route(route: Any) -> None:
                        request_url = route.request.url
                        try:
                            if request_url.startswith(("data:", "blob:", "about:")):
                                await route.continue_()
                                return
                            self._check_url(request_url)
                            await route.continue_()
                        except Exception:
                            await route.abort()

                    await context.route("**/*", guard_route)
                    page = await context.new_page()
                    response = await page.goto(
                        url,
                        wait_until=wait_until,
                        timeout=self.browser_timeout_ms,
                    )
                    # Some pages never reach networkidle; use a short, bounded
                    # wait after DOM readiness instead of waiting indefinitely.
                    try:
                        await page.wait_for_timeout(500)
                    except Exception:
                        pass

                    final_url = page.url
                    self._check_url(final_url)
                    title = await page.title()
                    page_html = await page.content()
                    content = self._extract_page_text(page_html)
                    if len(content.strip()) < 200:
                        # Body text can rescue pages where article extraction is too strict.
                        content = await page.locator("body").inner_text(timeout=5000)
                    content = self._clean_text(content)[:max_bytes]
                    status = response.status if response else 200
                    language = ""
                    try:
                        language = await page.locator("html").get_attribute("lang") or ""
                    except Exception:
                        pass
                    return {
                        "success": status < 400 and bool(content),
                        "url": final_url,
                        "status": status,
                        "content_type": "text/html",
                        "title": title,
                        "language": language,
                        "content": content,
                        "size": len(content.encode("utf-8")),
                        "truncated": len(content) >= max_bytes,
                        "fetch_method": "chromium",
                    }
                finally:
                    await browser.close()
        except BlockedRequestError as exc:
            return {"success": False, "url": url, "error": f"blocked: {exc}"}
        except Exception as exc:
            logger.debug("Chromium web fetch failed for %s: %s", url, exc)
            return {"success": False, "url": url, "error": f"Chromium fetch failed: {exc}"}

    async def _search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query")
        if not query:
            return {"success": False, "error": "search requires 'query'"}
        limit = max(1, min(int(params.get("limit", 10)), 20))
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "en,en-US;q=0.9,*;q=0.5",
        }
        errors: List[str] = []

        # Keep DuckDuckGo as the primary search backend.
        ddg_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(ddg_url, headers=headers)
            if response.status_code == 200:
                results = self._parse_ddg(response.text, limit)
                if results:
                    return self._search_result(query, results, "duckduckgo")
            errors.append(f"duckduckgo HTTP {response.status_code} or no parseable results")
        except Exception as exc:
            errors.append(f"duckduckgo: {exc}")

        # Preserve Bing as the second search backend.
        bing_url = "https://www.bing.com/search?" + urllib.parse.urlencode({"q": query})
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                response = await client.get(bing_url, headers=headers)
            if response.status_code == 200:
                results = self._parse_bing(response.text, limit)
                if results:
                    return self._search_result(query, results, "bing")
            errors.append(f"bing HTTP {response.status_code} or no parseable results")
        except Exception as exc:
            errors.append(f"bing: {exc}")

        # Last-resort Chromium search: only attempted if installed. Search
        # engines and their HTML parsers remain the primary/fallback path above.
        if self.browser_enabled:
            browser_result = await self._search_browser(query, limit)
            if browser_result.get("success"):
                return browser_result
            errors.append(f"chromium: {browser_result.get('error', 'no results')}")

        return {
            "success": False,
            "query": query,
            "error": "web search returned no results (" + "; ".join(errors) + ")",
            "results": [],
            "count": 0,
        }

    @staticmethod
    def _search_result(query: str, results: List[Dict[str, str]], backend: str) -> Dict[str, Any]:
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "backend": backend,
        }

    async def _search_browser(self, query: str, limit: int) -> Dict[str, Any]:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {"success": False, "error": "Playwright is not installed"}
        search_url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
        try:
            self._check_url(search_url)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(user_agent=self.user_agent)
                    async def guard_route(route: Any) -> None:
                        try:
                            request_url = route.request.url
                            if request_url.startswith(("data:", "blob:", "about:")):
                                await route.continue_()
                            else:
                                self._check_url(request_url)
                                await route.continue_()
                        except Exception:
                            await route.abort()
                    await context.route("**/*", guard_route)
                    page = await context.new_page()
                    await page.goto(search_url, wait_until="domcontentloaded", timeout=self.browser_timeout_ms)
                    await page.wait_for_timeout(500)
                    links = await page.locator("a").evaluate_all(
                        """els => els.map(a => ({
                            title: (a.innerText || '').trim(),
                            url: a.href || ''
                        })).filter(x => x.title && /^https?:/.test(x.url)
                          && !/google\\.(com|[a-z.]+)\\/search/.test(x.url))"""
                    )
                    results: List[Dict[str, str]] = []
                    seen = set()
                    for item in links:
                        link = item.get("url", "")
                        title = self._clean_text(item.get("title", ""))
                        if not link or not title or link in seen:
                            continue
                        host = (urllib.parse.urlsplit(link).hostname or "").lower()
                        if host.endswith("google.com") or host == "webcache.googleusercontent.com":
                            continue
                        seen.add(link)
                        results.append({"title": title[:300], "url": link})
                        if len(results) >= limit:
                            break
                    if results:
                        return self._search_result(query, results, "chromium-google")
                    return {"success": False, "error": "Chromium search returned no parseable results"}
                finally:
                    await browser.close()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _parse_ddg(self, text: str, limit: int) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for match in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            text, re.DOTALL | re.IGNORECASE,
        ):
            link = html.unescape(match.group(1))
            title = self._strip_tags(match.group(2))
            if title and link:
                results.append({"title": title, "url": link})
            if len(results) >= limit:
                break
        return results

    def _parse_bing(self, text: str, limit: int) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for block in re.findall(r'<li class="b_algo".*?</li>', text, re.DOTALL | re.IGNORECASE):
            match = re.search(
                r'<h2[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block, re.DOTALL | re.IGNORECASE,
            )
            if not match:
                continue
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.DOTALL | re.IGNORECASE)
            results.append({
                "title": self._strip_tags(match.group(2)),
                "url": self._unwrap_bing_url(html.unescape(match.group(1))),
                "snippet": self._strip_tags(snippet_match.group(1)) if snippet_match else "",
            })
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _unwrap_bing_url(url: str) -> str:
        if "bing.com/ck/a" not in url:
            return url
        try:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            token = (query.get("u") or [""])[0]
            if not token.startswith("a1"):
                return url
            payload = token[2:]
            payload += "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload).decode("utf-8", errors="replace")
            return decoded if decoded.startswith(("http://", "https://")) else url
        except Exception:
            return url

    @staticmethod
    def _looks_like_js_shell(content: str) -> bool:
        text = content.lower()
        if len(text.strip()) < 500:
            return True
        markers = (
            "enable javascript to run this app",
            "you need to enable javascript",
            "javascript is required",
            "please enable javascript",
            "id=\"root\"></div>",
            "id=\"app\"></div>",
            "loading...",
            "checking your browser",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _extract_title(source: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", source, re.I | re.S)
        return WebTool._clean_text(html.unescape(match.group(1))) if match else ""

    @staticmethod
    def _extract_language(source: str) -> str:
        match = re.search(r"<html[^>]*\blang\s*=\s*[\"']([^\"']+)", source, re.I)
        return html.unescape(match.group(1)) if match else ""

    @staticmethod
    def _clean_text(text: str) -> str:
        text = html.unescape(text or "")
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _extract_page_text(source: str) -> str:
        """Extract readable main content when optional trafilatura is installed."""
        try:
            import trafilatura
            extracted = trafilatura.extract(
                source,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=False,
            )
            if extracted and len(extracted.strip()) >= 100:
                return WebTool._clean_text(extracted)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Main-content extraction failed: %s", exc)
        return WebTool._html_to_text(source)

    @staticmethod
    def _html_to_text(html_text: str) -> str:
        # Prefer a content extractor when available; retain a dependency-free fallback.
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=True,
                include_links=False,
                favor_precision=False,
            )
            if extracted and len(extracted.strip()) >= 100:
                return WebTool._clean_text(extracted)
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Main-content extraction failed: %s", exc)
        # Remove non-content blocks before removing tags.
        text = re.sub(r"<!--.*?-->", " ", html_text, flags=re.S)
        text = re.sub(
            r"<(script|style|noscript|svg|canvas|iframe|nav|footer|header|aside)\b[^>]*>.*?</\1\s*>",
            " ", text, flags=re.I | re.S,
        )
        text = re.sub(r"<(script|style|noscript)\b[^>]*/?>", " ", text, flags=re.I)
        # Preserve paragraph and heading boundaries for easier model reading.
        text = re.sub(r"</(p|div|article|section|h[1-6]|li|tr|br)\s*>", "\n", text, flags=re.I)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return WebTool._clean_text(text)

    @staticmethod
    def _strip_tags(value: str) -> str:
        return WebTool._clean_text(re.sub(r"<[^>]+>", "", value))
