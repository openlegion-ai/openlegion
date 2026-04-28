"""Web search tool using Exa AI-powered search.

Requires an Exa API key (``OPENLEGION_CRED_EXA_API_KEY`` preferred, with
``EXA_API_KEY`` accepted as a fallback). Returns semantically ranked results
with optional highlights, full text, or LLM-generated summary per result.
Supports category filtering, domain/text filters, and date ranges.

Docs: https://exa.ai/docs/reference/search
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.agent.skills import skill
from src.shared.utils import sanitize_for_prompt

logger = logging.getLogger("agent.exa_search")

_PROVIDER = "exa"
_API_URL = "https://api.exa.ai/search"
_INTEGRATION_NAME = "openlegion"

_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS = 25
_DEFAULT_TEXT_CAP = 1000
_SNIPPET_CAP = 500

_DEFAULT_SEARCH_TYPE = "auto"
_VALID_SEARCH_TYPES = frozenset({
    "auto", "neural", "fast", "deep-lite", "deep", "deep-reasoning", "instant",
})

_DEFAULT_CONTENT_MODE = "highlights"
_VALID_CONTENT_MODES = frozenset({"highlights", "text", "summary", "all"})

_VALID_CATEGORIES = frozenset({
    "company", "research paper", "news", "personal site",
    "financial report", "people",
})


@dataclass(slots=True)
class ExaResult:
    """Typed view of a single Exa search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None
    author: str | None = None
    score: float | None = None
    highlights: list[str] = field(default_factory=list)
    summary: str | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }
        if self.published_date:
            out["published_date"] = self.published_date
        if self.author:
            out["author"] = self.author
        if self.score is not None:
            out["score"] = self.score
        if self.highlights:
            out["highlights"] = self.highlights
        if self.summary:
            out["summary"] = self.summary
        return out


def _get_api_key() -> str:
    """Resolve the Exa API key from agent-tier or standard env var."""
    return (
        os.environ.get("OPENLEGION_CRED_EXA_API_KEY")
        or os.environ.get("EXA_API_KEY")
        or ""
    )


def _normalize_max_results(value: Any) -> int:
    try:
        return max(1, min(int(value), _MAX_RESULTS))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_RESULTS


def _build_contents(content_mode: str, text_max_chars: int) -> dict:
    """Build the ``contents`` payload. Content types are NOT mutually exclusive."""
    text_cap = text_max_chars if text_max_chars and text_max_chars > 0 else _DEFAULT_TEXT_CAP
    if content_mode == "text":
        return {"text": {"maxCharacters": text_cap}}
    if content_mode == "summary":
        return {"summary": True}
    if content_mode == "all":
        return {
            "highlights": True,
            "text": {"maxCharacters": text_cap},
            "summary": True,
        }
    return {"highlights": True}


def _extract_snippet(item: dict) -> str:
    """Cascade through available content fields for a readable snippet."""
    highlights = item.get("highlights") or []
    if highlights:
        joined = " ".join(str(h) for h in highlights if h)
        if joined:
            return joined[:_SNIPPET_CAP]
    summary = item.get("summary")
    if summary:
        return str(summary)[:_SNIPPET_CAP]
    text = item.get("text")
    if text:
        return str(text)[:_SNIPPET_CAP]
    return ""


def _parse_result(item: dict) -> ExaResult:
    highlights_raw = item.get("highlights") or []
    highlights = [sanitize_for_prompt(str(h)) for h in highlights_raw if h]
    summary_raw = item.get("summary")
    summary = sanitize_for_prompt(str(summary_raw)) if summary_raw else None
    score_raw = item.get("score")
    try:
        score = float(score_raw) if score_raw is not None else None
    except (TypeError, ValueError):
        score = None
    return ExaResult(
        title=sanitize_for_prompt(str(item.get("title") or "")),
        url=sanitize_for_prompt(str(item.get("url") or "")),
        snippet=sanitize_for_prompt(_extract_snippet(item)),
        published_date=item.get("publishedDate") or None,
        author=sanitize_for_prompt(str(item["author"])) if item.get("author") else None,
        score=score,
        highlights=highlights,
        summary=summary,
    )


def _build_payload(
    query: str,
    max_results: int,
    search_type: str,
    contents: dict,
    category: str,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
    include_text: list[str] | None,
    exclude_text: list[str] | None,
    start_published_date: str,
    end_published_date: str,
) -> dict:
    payload: dict[str, Any] = {
        "query": query,
        "numResults": max_results,
        "type": search_type,
        "contents": contents,
    }
    if category:
        payload["category"] = category
    if include_domains:
        payload["includeDomains"] = list(include_domains)
    if exclude_domains:
        payload["excludeDomains"] = list(exclude_domains)
    if include_text:
        payload["includeText"] = list(include_text)
    if exclude_text:
        payload["excludeText"] = list(exclude_text)
    if start_published_date:
        payload["startPublishedDate"] = start_published_date
    if end_published_date:
        payload["endPublishedDate"] = end_published_date
    return payload


def _base_response(
    query: str,
    requested_max: int,
    max_results: int,
    search_type: str,
    content_mode: str,
) -> dict:
    return {
        "query": query,
        "results": [],
        "count": 0,
        "provider": _PROVIDER,
        "search_type": search_type,
        "content_mode": content_mode,
        "max_results_requested": requested_max,
        "max_results_used": max_results,
    }


@skill(
    name="exa_search",
    description=(
        "Search the web using Exa AI-powered search. Returns semantically ranked "
        "results with optional highlights, full text, or LLM-generated summary per "
        "result. Supports category filtering (company, research paper, news, "
        "personal site, financial report, people), domain/text filters, and "
        "published-date ranges. Requires OPENLEGION_CRED_EXA_API_KEY or EXA_API_KEY."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Natural-language search query",
        },
        "max_results": {
            "type": "integer",
            "description": "Number of results to return (1-25, default 5)",
            "default": _DEFAULT_MAX_RESULTS,
        },
        "search_type": {
            "type": "string",
            "description": (
                "Search mode. 'auto' (default) lets Exa pick; 'neural' for semantic, "
                "'fast' for low-latency, 'deep'/'deep-lite'/'deep-reasoning' for "
                "thorough multi-step research, 'instant' for cached results."
            ),
            "enum": sorted(_VALID_SEARCH_TYPES),
            "default": _DEFAULT_SEARCH_TYPE,
        },
        "content_mode": {
            "type": "string",
            "description": (
                "What content to retrieve per result. 'highlights' (default) returns "
                "LLM-selected snippets; 'text' returns page text; 'summary' returns "
                "an LLM summary; 'all' returns all three."
            ),
            "enum": sorted(_VALID_CONTENT_MODES),
            "default": _DEFAULT_CONTENT_MODE,
        },
        "text_max_chars": {
            "type": "integer",
            "description": "Max characters of page text to retrieve per result (default 1000).",
            "default": _DEFAULT_TEXT_CAP,
        },
        "category": {
            "type": "string",
            "description": (
                "Restrict to a category: company, research paper, news, personal site, "
                "financial report, people. Empty string for no filter."
            ),
            "default": "",
        },
        "include_domains": {
            "type": "array",
            "description": "Only return results from these domains (e.g. ['arxiv.org']).",
            "items": {"type": "string"},
            "default": [],
        },
        "exclude_domains": {
            "type": "array",
            "description": "Exclude results from these domains.",
            "items": {"type": "string"},
            "default": [],
        },
        "include_text": {
            "type": "array",
            "description": "Only return results containing these phrases.",
            "items": {"type": "string"},
            "default": [],
        },
        "exclude_text": {
            "type": "array",
            "description": "Exclude results containing these phrases.",
            "items": {"type": "string"},
            "default": [],
        },
        "start_published_date": {
            "type": "string",
            "description": "Only results published on/after this date (ISO 8601, e.g. 2026-01-01).",
            "default": "",
        },
        "end_published_date": {
            "type": "string",
            "description": "Only results published on/before this date (ISO 8601).",
            "default": "",
        },
    },
)
async def exa_search(
    query: str,
    max_results: int = _DEFAULT_MAX_RESULTS,
    search_type: str = _DEFAULT_SEARCH_TYPE,
    content_mode: str = _DEFAULT_CONTENT_MODE,
    text_max_chars: int = _DEFAULT_TEXT_CAP,
    category: str = "",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    include_text: list[str] | None = None,
    exclude_text: list[str] | None = None,
    start_published_date: str = "",
    end_published_date: str = "",
) -> dict:
    """Search the web using Exa and return structured, sanitized results."""
    requested_max = max_results
    max_results = _normalize_max_results(max_results)

    if search_type not in _VALID_SEARCH_TYPES:
        search_type = _DEFAULT_SEARCH_TYPE
    if content_mode not in _VALID_CONTENT_MODES:
        content_mode = _DEFAULT_CONTENT_MODE
    if category and category not in _VALID_CATEGORIES:
        category = ""

    base = _base_response(query, requested_max, max_results, search_type, content_mode)

    api_key = _get_api_key()
    if not api_key:
        return {
            **base,
            "error": (
                "Exa API key not configured. Set OPENLEGION_CRED_EXA_API_KEY "
                "(agent tier) or EXA_API_KEY."
            ),
            "error_type": "missing_api_key",
        }

    contents = _build_contents(content_mode, text_max_chars)
    payload = _build_payload(
        query=query,
        max_results=max_results,
        search_type=search_type,
        contents=contents,
        category=category,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        include_text=include_text,
        exclude_text=exclude_text,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
    )

    headers = {
        "x-api-key": api_key,
        "x-exa-integration": _INTEGRATION_NAME,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    client_kwargs: dict = {"follow_redirects": False, "timeout": 30}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(_API_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        return {
            **base,
            "error": "Exa search timed out. Try again with a narrower query.",
            "error_type": "timeout",
        }
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        error_type = "auth_error" if status in (401, 403) else "http_status"
        hint = ""
        if status in (401, 403):
            hint = " Check that OPENLEGION_CRED_EXA_API_KEY is valid."
        elif status == 429:
            error_type = "rate_limit"
            hint = " Exa rate limit reached."
        return {
            **base,
            "error": f"Exa returned HTTP {status}.{hint}",
            "error_type": error_type,
            "status_code": status,
        }
    except httpx.HTTPError as e:
        return {
            **base,
            "error": f"Exa request failed: {type(e).__name__}",
            "error_type": "network",
        }
    except ValueError:
        return {
            **base,
            "error": "Exa returned a non-JSON response.",
            "error_type": "parse_error",
        }
    except Exception as e:
        logger.exception("Unexpected error during Exa search")
        return {
            **base,
            "error": str(e),
            "error_type": "unexpected",
        }

    raw_results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(raw_results, list):
        raw_results = []

    parsed = [_parse_result(item) for item in raw_results if isinstance(item, dict)]
    dict_results = [r.to_dict() for r in parsed]

    return {
        **base,
        "results": dict_results,
        "count": len(dict_results),
    }
