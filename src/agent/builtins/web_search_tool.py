"""Web search tool using DuckDuckGo HTML.

No API key required. Extracts titles, URLs, and snippets from search results.
Available to every agent as a built-in skill.
"""

from __future__ import annotations

import re
from html import unescape

import httpx

from src.agent.skills import skill

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


@skill(
    name="web_search",
    description=(
        "Search the web for current information. Returns titles, URLs, and "
        "snippets. Use this for research, fact-checking, or finding recent news."
    ),
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 5)",
            "default": 5,
        },
    },
)
async def web_search(query: str, max_results: int = 5, **_kwargs) -> dict:
    """Search the web using DuckDuckGo (no API key needed)."""
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
                "note": "No results found. Try rephrasing or use http_request to visit a URL directly.",
            }

        return {"query": query, "results": results, "count": len(results)}
    except httpx.TimeoutException:
        return {"error": "Search timed out. Try again.", "query": query, "results": []}
    except Exception as e:
        return {"error": str(e), "query": query, "results": []}
