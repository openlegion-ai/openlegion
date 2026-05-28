"""Unit tests for LLMClient — focused on the retryable-error classifier.

The classifier at src/agent/llm.py:~192 decides whether a mesh-returned LLM
failure is transient (raise LLMRetryableError → backoff retry in
_llm_call_with_retry) or permanent (raise RuntimeError → fail the task).

Operator reported tasks dying on "Anthropic OAuth error: Model X returned
empty response" — the Claude subscription throttle signal. Pre-fix, that
phrase matched no keyword in the _retryable tuple and fell through to a
permanent RuntimeError, killing the task on the first throttle hit. This
test pins the post-fix behavior: empty-response is now retryable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.llm import LLMClient, LLMRetryableError


def _make_mesh_response(
    error_msg: str,
    *,
    error_type: str | None = None,
    error_meta: dict | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    body: dict = {"success": False, "error": error_msg}
    if error_type is not None:
        body["error_type"] = error_type
    if error_meta is not None:
        body["error_meta"] = error_meta
    resp.json = MagicMock(return_value=body)
    return resp


def _make_llm_client() -> LLMClient:
    client = LLMClient(
        mesh_url="http://mesh.test",
        agent_id="test-agent",
        default_model="anthropic/claude-sonnet-4-6",
    )
    mock_http = AsyncMock()
    client._get_client = AsyncMock(return_value=mock_http)
    return client


@pytest.mark.asyncio
async def test_empty_response_classified_as_retryable():
    """Claude subscription throttle ('empty response') must surface as
    LLMRetryableError so _llm_call_with_retry backs off rather than
    failing the task on the first occurrence."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response(
            "Anthropic OAuth error: Model anthropic/claude-sonnet-4-6 returned empty response",
        ),
    )
    with pytest.raises(LLMRetryableError):
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_non_stream_empty_response_no_choices_retryable():
    """The non-stream OAuth path (credentials.py:~2687) emits
    'LLM returned empty response (no choices) for model X' — same root
    cause, must classify identically."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response(
            "LLM returned empty response (no choices) for model anthropic/claude-sonnet-4-6",
        ),
    )
    with pytest.raises(LLMRetryableError):
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_rate_limit_still_retryable():
    """Pre-existing classifications must still fire — regression guard
    for the keyword tuple extension."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response("openai rate limit exceeded"),
    )
    with pytest.raises(LLMRetryableError):
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_transient_error_type_classified_as_retryable():
    """Mesh-tagged ``error_type="transient"`` (the typed channel emitted by
    ``execute_api_call`` when ``LLMTransientError`` propagates from the
    OAuth wrap at credentials.py:~1727 or the LiteLLM empty-choices path
    at credentials.py:~2685) must surface as ``LLMRetryableError`` so
    ``_llm_call_with_retry`` backs off. Mirrors the existing
    ``auth_failure`` / ``config_error`` typed branches."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response(
            "Anthropic OAuth error: Connection to the AI provider was "
            "interrupted. Retrying may help.",
            error_type="transient",
            error_meta={
                "provider": "anthropic",
                "model": "anthropic/claude-sonnet-4-6",
                "retry_after": None,
            },
        ),
    )
    with pytest.raises(LLMRetryableError):
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_retrying_may_help_substring_retryable_backstop():
    """Backstop path — when the mesh did NOT tag ``error_type="transient"``
    (un-tagged third-party wrapper or a path the outer handler didn't
    catch), the substring tuple must still classify ``"Retrying may help"``
    as retryable. The phrase is the deliberate suffix from
    ``friendly_streaming_error`` (src/shared/utils.py:78); matching it
    keeps the helper's contract usable as a fallback signal even without
    the typed channel."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response(
            "Anthropic OAuth error: Connection to the AI provider was "
            "interrupted. Retrying may help.",
            # No error_type tag — exercises the substring path.
        ),
    )
    with pytest.raises(LLMRetryableError):
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_unrelated_runtime_error_still_permanent():
    """Permanent failures must NOT be reclassified as retryable.
    Budget-exhausted / config-error / invalid-prompt should still fail
    the task immediately."""
    client = _make_llm_client()
    mock_http = await client._get_client()
    mock_http.post = AsyncMock(
        return_value=_make_mesh_response("Budget exceeded for agent test-agent"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        await client.chat(system="s", messages=[{"role": "user", "content": "x"}])
    # Must NOT be the retryable subclass.
    assert not isinstance(exc_info.value, LLMRetryableError)
