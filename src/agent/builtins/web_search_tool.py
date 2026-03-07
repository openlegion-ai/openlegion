"""Web search tool using DuckDuckGo HTML.

No API key required. Extracts titles, URLs, and snippets from search results.
Available to every agent as a built-in skill.
"""

from __future__ import annotations

import re
from html import unescape

import httpx

from src.agent.skills import skill

_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS = 10
_PROVIDER = "duckduckgo_html"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", unescape(html)).strip()


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
            results.append({"title": title, "url": url, "snippet": snippet})

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
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=_HEADERS,
            )
            resp.raise_for_status()

        results = _parse_ddg_html(resp.text, max_results)

        if not results:
            return {
                "query": query,
                "results": [],
                "count": 0,
                "provider": _PROVIDER,
                "max_results_requested": requested_max,
                "max_results_used": max_results,
                "note": "No results found. Try rephrasing or use http_request to visit a URL directly.",
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
            "max_results_requested": requested_max,
            "max_results_used": max_results,
        }
