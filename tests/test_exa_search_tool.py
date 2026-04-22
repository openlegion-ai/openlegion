"""Tests for exa_search built-in tool behavior."""

from __future__ import annotations

import json

import httpx
import pytest


class _DummyResponse:
    def __init__(self, payload: dict | str | None = None, status_code: int = 200,
                 raise_on_json: bool = False):
        if isinstance(payload, str):
            self.text = payload
            self._payload: dict | None = None
        else:
            self._payload = payload or {}
            self.text = json.dumps(self._payload)
        self.status_code = status_code
        self._raise_on_json = raise_on_json

    def json(self) -> dict:
        if self._raise_on_json:
            raise ValueError("not json")
        return self._payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.exa.ai/search")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


class _DummyClient:
    def __init__(self, response: _DummyResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error
        self.last_url: str | None = None
        self.last_json: dict | None = None
        self.last_headers: dict | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, *, json=None, headers=None, **_kwargs):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        if self._error is not None:
            raise self._error
        return self._response


def _patch_httpx(monkeypatch, module, response=None, error=None) -> _DummyClient:
    """Install a _DummyClient in place of httpx.AsyncClient; return the instance."""
    client = _DummyClient(response=response, error=error)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **_kwargs: client)
    return client


@pytest.fixture(autouse=True)
def _clear_exa_env(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_CRED_EXA_API_KEY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    yield


# ── Registration & disabled state ────────────────────────────


@pytest.mark.asyncio
async def test_exa_search_missing_api_key_returns_descriptive_error(monkeypatch):
    from src.agent.builtins import exa_search_tool

    # No client should be created — call shouldn't hit the network.
    sentinel = {"called": False}

    def _boom(**_kwargs):
        sentinel["called"] = True
        raise AssertionError("httpx client should not be constructed without an API key")

    monkeypatch.setattr(exa_search_tool.httpx, "AsyncClient", _boom)

    result = await exa_search_tool.exa_search("test query")

    assert result["error_type"] == "missing_api_key"
    assert "EXA_API_KEY" in result["error"]
    assert result["provider"] == "exa"
    assert result["count"] == 0
    assert result["results"] == []
    assert sentinel["called"] is False


@pytest.mark.asyncio
async def test_exa_search_prefers_openlegion_cred_env(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("OPENLEGION_CRED_EXA_API_KEY", "cred-key")
    monkeypatch.setenv("EXA_API_KEY", "fallback-key")
    client = _patch_httpx(monkeypatch, exa_search_tool,
                          response=_DummyResponse({"results": []}))

    await exa_search_tool.exa_search("test")

    assert client.last_headers["x-api-key"] == "cred-key"
    assert client.last_headers["x-exa-integration"] == "openlegion"


@pytest.mark.asyncio
async def test_exa_search_falls_back_to_standard_env(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "fallback-key")
    client = _patch_httpx(monkeypatch, exa_search_tool,
                          response=_DummyResponse({"results": []}))

    await exa_search_tool.exa_search("test")

    assert client.last_headers["x-api-key"] == "fallback-key"


# ── Response parsing & snippet fallbacks ─────────────────────


@pytest.mark.asyncio
async def test_exa_search_parses_highlights(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    payload = {
        "results": [
            {
                "id": "abc",
                "url": "https://example.com/a",
                "title": "Example A",
                "publishedDate": "2026-03-01",
                "author": "Alice",
                "score": 0.87,
                "highlights": ["first snippet", "second snippet"],
                "highlightScores": [0.9, 0.8],
            },
        ],
    }
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse(payload))

    result = await exa_search_tool.exa_search("hello")

    assert result["provider"] == "exa"
    assert result["count"] == 1
    r = result["results"][0]
    assert r["url"] == "https://example.com/a"
    assert r["title"] == "Example A"
    assert r["published_date"] == "2026-03-01"
    assert r["author"] == "Alice"
    assert r["score"] == pytest.approx(0.87)
    assert r["highlights"] == ["first snippet", "second snippet"]
    assert "first snippet" in r["snippet"]
    assert "second snippet" in r["snippet"]


@pytest.mark.asyncio
async def test_exa_search_snippet_falls_back_to_summary(monkeypatch):
    """When highlights are missing, snippet should fall back to summary."""
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    payload = {
        "results": [
            {
                "url": "https://example.com/b",
                "title": "Example B",
                "summary": "This is an AI-generated summary of the page.",
            },
        ],
    }
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse(payload))

    result = await exa_search_tool.exa_search("hello", content_mode="summary")

    r = result["results"][0]
    assert "AI-generated summary" in r["snippet"]
    assert r["summary"] == "This is an AI-generated summary of the page."
    assert "highlights" not in r  # never set when empty


@pytest.mark.asyncio
async def test_exa_search_snippet_falls_back_to_text(monkeypatch):
    """When highlights and summary are missing, snippet should fall back to text."""
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    long_text = "Full page body. " * 100  # > 500 chars
    payload = {
        "results": [
            {
                "url": "https://example.com/c",
                "title": "Example C",
                "text": long_text,
            },
        ],
    }
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse(payload))

    result = await exa_search_tool.exa_search("hello", content_mode="text")

    r = result["results"][0]
    # Snippet is capped at 500 chars
    assert r["snippet"].startswith("Full page body.")
    assert len(r["snippet"]) <= 500
    # Full text is NOT returned at top level (we only expose snippet + highlights/summary)


@pytest.mark.asyncio
async def test_exa_search_empty_results(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse({"results": []}))

    result = await exa_search_tool.exa_search("obscure query")

    assert result["count"] == 0
    assert result["results"] == []
    assert "error" not in result


@pytest.mark.asyncio
async def test_exa_search_handles_missing_results_key(monkeypatch):
    """API response without 'results' shouldn't crash — yields empty list."""
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse({}))

    result = await exa_search_tool.exa_search("q")

    assert result["count"] == 0
    assert result["results"] == []


# ── Payload construction ─────────────────────────────────────


@pytest.mark.asyncio
async def test_exa_search_payload_includes_filters(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    client = _patch_httpx(monkeypatch, exa_search_tool,
                          response=_DummyResponse({"results": []}))

    await exa_search_tool.exa_search(
        "llm research",
        max_results=3,
        search_type="neural",
        content_mode="all",
        text_max_chars=500,
        category="research paper",
        include_domains=["arxiv.org"],
        exclude_domains=["example.com"],
        include_text=["transformer"],
        exclude_text=["cat"],
        start_published_date="2026-01-01",
        end_published_date="2026-04-01",
    )

    body = client.last_json
    assert body["query"] == "llm research"
    assert body["numResults"] == 3
    assert body["type"] == "neural"
    assert body["category"] == "research paper"
    assert body["includeDomains"] == ["arxiv.org"]
    assert body["excludeDomains"] == ["example.com"]
    assert body["includeText"] == ["transformer"]
    assert body["excludeText"] == ["cat"]
    assert body["startPublishedDate"] == "2026-01-01"
    assert body["endPublishedDate"] == "2026-04-01"
    # content_mode="all" enables highlights, text, and summary together
    contents = body["contents"]
    assert contents["highlights"] is True
    assert contents["text"] == {"maxCharacters": 500}
    assert contents["summary"] is True


@pytest.mark.asyncio
async def test_exa_search_clamps_max_results(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    client = _patch_httpx(monkeypatch, exa_search_tool,
                          response=_DummyResponse({"results": []}))

    result = await exa_search_tool.exa_search("q", max_results=999)

    assert result["max_results_requested"] == 999
    assert result["max_results_used"] == 25
    assert client.last_json["numResults"] == 25


@pytest.mark.asyncio
async def test_exa_search_rejects_invalid_enums(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    client = _patch_httpx(monkeypatch, exa_search_tool,
                          response=_DummyResponse({"results": []}))

    result = await exa_search_tool.exa_search(
        "q",
        search_type="keyword",  # removed from API
        content_mode="bogus",
        category="invalid",
    )

    # search_type and content_mode should fall back to defaults
    assert client.last_json["type"] == "auto"
    assert client.last_json["contents"] == {"highlights": True}
    assert "category" not in client.last_json
    assert result["search_type"] == "auto"
    assert result["content_mode"] == "highlights"


# ── Error paths ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exa_search_timeout(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    _patch_httpx(monkeypatch, exa_search_tool,
                 error=httpx.TimeoutException("timeout"))

    result = await exa_search_tool.exa_search("q")

    assert result["error_type"] == "timeout"
    assert result["count"] == 0


@pytest.mark.asyncio
async def test_exa_search_auth_error(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "bad-key")
    _patch_httpx(monkeypatch, exa_search_tool,
                 response=_DummyResponse({}, status_code=401))

    result = await exa_search_tool.exa_search("q")

    assert result["error_type"] == "auth_error"
    assert result["status_code"] == 401
    assert "OPENLEGION_CRED_EXA_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_exa_search_rate_limit(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    _patch_httpx(monkeypatch, exa_search_tool,
                 response=_DummyResponse({}, status_code=429))

    result = await exa_search_tool.exa_search("q")

    assert result["error_type"] == "rate_limit"
    assert result["status_code"] == 429


@pytest.mark.asyncio
async def test_exa_search_non_json_response(monkeypatch):
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    _patch_httpx(monkeypatch, exa_search_tool,
                 response=_DummyResponse("<html>oops</html>", raise_on_json=True))

    result = await exa_search_tool.exa_search("q")

    assert result["error_type"] == "parse_error"
    assert result["count"] == 0


# ── Sanitization ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_exa_search_strips_invisible_chars(monkeypatch):
    """Results should be passed through sanitize_for_prompt."""
    from src.agent.builtins import exa_search_tool

    monkeypatch.setenv("EXA_API_KEY", "k")
    # Zero-width joiner + bidi override in the title should be stripped.
    dirty_title = "Normal​title‮ injected"
    payload = {
        "results": [
            {"url": "https://example.com", "title": dirty_title,
             "highlights": ["clean snippet"]},
        ],
    }
    _patch_httpx(monkeypatch, exa_search_tool, response=_DummyResponse(payload))

    result = await exa_search_tool.exa_search("q")

    title = result["results"][0]["title"]
    assert "​" not in title
    assert "‮" not in title
    assert "Normal" in title and "title" in title
