"""Tests for web_search built-in tool behavior."""

from __future__ import annotations

import httpx
import pytest


class _DummyResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://html.duckduckgo.com/html/")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("search failed", request=request, response=response)


class _DummyClient:
    def __init__(self, response: _DummyResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *_args, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.asyncio
async def test_web_search_includes_provider_and_clamps_max_results(monkeypatch):
    from src.agent.builtins import web_search_tool

    html = """
    <a class="result__a" href="https://example.com/story">Example Story</a>
    <a class="result__snippet">Snippet one</a>
    """
    monkeypatch.setattr(
        web_search_tool.httpx,
        "AsyncClient",
        lambda **_kwargs: _DummyClient(response=_DummyResponse(html)),
    )

    result = await web_search_tool.web_search("openclaw sentiment", max_results=99)

    assert result["provider"] == "duckduckgo_html"
    assert result["max_results_requested"] == 99
    assert result["max_results_used"] == 10
    assert result["count"] == 1
    assert result["results"][0]["url"] == "https://example.com/story"


@pytest.mark.asyncio
async def test_web_search_timeout_returns_descriptive_error(monkeypatch):
    from src.agent.builtins import web_search_tool

    monkeypatch.setattr(
        web_search_tool.httpx,
        "AsyncClient",
        lambda **_kwargs: _DummyClient(error=httpx.TimeoutException("timeout")),
    )

    result = await web_search_tool.web_search("openclaw", max_results=3)

    assert result["error_type"] == "timeout"
    assert "DuckDuckGo" in result["error"]
    assert result["provider"] == "duckduckgo_html"
    assert result["max_results_used"] == 3
    assert result["results"] == []
