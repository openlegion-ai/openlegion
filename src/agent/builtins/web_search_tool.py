"""Web search tool using DuckDuckGo HTML.

No API key required. Extracts titles, URLs, and snippets from search results.
Available to every agent as a built-in skill.
"""

from __future__ import annotations

import logging
import os
import re
from html import unescape

import httpx

from src.agent.skills import skill
from src.shared.utils import sanitize_for_prompt

logger = logging.getLogger("agent.web_search")

_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS = 10
_PROVIDER = "duckduckgo_html"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

_CAPTCHA_SIGNALS = ("anomaly-modal", "bots use DuckDuckGo", "confirm this search was made by a human")


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", unescape(html)).strip()


def _is_captcha_response(html: str) -> bool:
    """Detect DuckDuckGo CAPTCHA / bot-challenge pages."""
    return any(signal in html for signal in _CAPTCHA_SIGNALS)


def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """Extract search results from DuckDuckGo HTML response."""
    results = []

    titles = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )
    snippets = re.findall(
        r'class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL,
    )

    for i, (url, raw_title) in enumerate(titles):
        if len(results) >= max_results:
            break
        title = _strip_tags(raw_title)
        snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""
        if title and url.startswith("http"):
            results.append({
                "title": sanitize_for_prompt(title),
                "url": sanitize_for_prompt(url),
                "snippet": sanitize_for_prompt(snippet),
            })

    return results


def _normalize_max_results(value: int) -> int:
    try:
        return max(1, min(int(value), _MAX_RESULTS))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RESULTS


@skill(
    name="web_search",
    description=(
        "Search current public web pages via DuckDuckGo HTML search. Returns "
        "structured results with title, URL, and snippet. Best for fast research, "
        "fact-checking, and recent news discovery."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Natural-language search query (for example: 'openclaw criticism 2026')",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results to return (1-10, default 5)",
            "default": _DEFAULT_MAX_RESULTS,
        },
    },
)
async def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo (no API key needed)."""
    requested_max = max_results
    max_results = _normalize_max_results(max_results)
    proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")

    try:
        client_kwargs: dict = {"follow_redirects": True, "timeout": 15}
        if proxy_url:
            client_kwargs["proxy"] = proxy_url
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        if _is_captcha_response(resp.text):
            logger.warning("DuckDuckGo returned CAPTCHA for query=%r", query)
            return {
                "query": query,
                "results": [],
                "count": 0,
                "provider": _PROVIDER,
                "max_results_requested": requested_max,
                "max_results_used": max_results,
                "error": (
                    "DuckDuckGo returned a CAPTCHA challenge. "
                    "Use browser_navigate to search Google directly, e.g. "
                    "https://www.google.com/search?q=your+query"
                ),
                "error_type": "captcha",
            }

        results = _parse_ddg_html(resp.text, max_results)

        if not results:
            return {
                "query": query,
                "results": [],
                "count": 0,
                "provider": _PROVIDER,
                "max_results_requested": requested_max,
                "max_results_used": max_results,
                "note": "No results found. Try rephrasing or use browser_navigate to search directly.",
            }

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "provider": _PROVIDER,
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
    except httpx.TimeoutException:
        return {
            "error": "Search timed out while waiting for DuckDuckGo. Try again in a moment.",
            "error_type": "timeout",
            "query": query,
            "provider": _PROVIDER,
            "results": [],
            "count": 0,
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
    except httpx.HTTPStatusError as e:
        return {
            "error": (
                f"Search provider returned HTTP {e.response.status_code}. "
                "This is usually temporary; retry with a narrower query."
            ),
            "error_type": "http_status",
            "status_code": e.response.status_code,
            "query": query,
            "provider": _PROVIDER,
            "results": [],
            "count": 0,
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
    except httpx.HTTPError as e:
        return {
            "error": f"Search request failed due to a network/protocol issue: {type(e).__name__}",
            "error_type": "network",
            "query": query,
            "provider": _PROVIDER,
            "results": [],
            "count": 0,
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": "unexpected",
            "query": query,
            "provider": _PROVIDER,
            "results": [],
            "count": 0,
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
