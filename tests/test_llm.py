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

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.llm import LLMClient, LLMRetryableError
from src.shared.errors import LLMAuthError


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


# --------------------------------------------------------------------------
# _parse_tool_calls — truncated-args retryable carve-out
#
# A streamed tool call whose JSON ``arguments`` got truncated (provider hit
# the output token cap mid-tool-call, or the stream dropped before the JSON
# closed) used to degrade to ``{"raw": <truncated>}`` and dispatch a stub that
# failed downstream with a confusing missing-required-param error (the
# ``edit_agent`` incident). It is now raised as ``LLMRetryableError`` so the
# call is retried (and the streaming caller falls back to non-streaming).
# Empty / whitespace-only args remain a legitimate no-arg call.
# --------------------------------------------------------------------------


def test_parse_tool_calls_truncated_args_raises_retryable():
    """Non-empty unparseable args = truncated mid-stream → retryable."""
    raw = [{
        "name": "edit_agent",
        "arguments": '{"agent_id": "page-validator", "field": "instructions", "value": "You are page-validator',
    }]
    with pytest.raises(LLMRetryableError) as exc_info:
        LLMClient._parse_tool_calls(raw)
    # Message surfaces the tool name + a truncated preview for the operator.
    assert "edit_agent" in str(exc_info.value)


def test_parse_tool_calls_empty_string_args_is_no_arg_call():
    """Empty-string args = legitimate no-arg tool call → {} (no raise)."""
    raw = [{"name": "check_inbox", "arguments": ""}]
    parsed = LLMClient._parse_tool_calls(raw)
    assert parsed is not None
    assert len(parsed) == 1
    assert parsed[0].name == "check_inbox"
    assert parsed[0].arguments == {}


def test_parse_tool_calls_whitespace_only_args_is_no_arg_call():
    """Whitespace-only args = legitimate no-arg tool call → {} (no raise)."""
    raw = [{"name": "check_inbox", "arguments": "   \n\t  "}]
    parsed = LLMClient._parse_tool_calls(raw)
    assert parsed is not None
    assert parsed[0].arguments == {}


def test_parse_tool_calls_valid_object_args_parsed():
    """Valid JSON object args parse into the dict unchanged."""
    raw = [{"name": "edit_agent", "arguments": '{"agent_id": "x", "field": "model"}'}]
    parsed = LLMClient._parse_tool_calls(raw)
    assert parsed is not None
    assert parsed[0].arguments == {"agent_id": "x", "field": "model"}


def test_parse_tool_calls_valid_non_dict_json_normalised_to_empty():
    """Valid-but-non-dict JSON (e.g. ``"42"`` → int) normalises to {} and
    MUST NOT raise — distinct from the truncated-args case."""
    raw = [{"name": "noop", "arguments": "42"}]
    parsed = LLMClient._parse_tool_calls(raw)
    assert parsed is not None
    assert parsed[0].arguments == {}


# --------------------------------------------------------------------------
# chat_stream — error classification + truncated done-frame
# --------------------------------------------------------------------------


def _make_stream_client(frames: list[dict]) -> LLMClient:
    """Build an LLMClient whose ``client.stream(...)`` yields the given SSE
    ``data:`` frames via an async context manager."""
    client = LLMClient(
        mesh_url="http://mesh.test",
        agent_id="test-agent",
        default_model="anthropic/claude-sonnet-4-6",
    )

    class _FakeResponse:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for frame in frames:
                yield f"data: {json.dumps(frame)}"

    class _FakeStreamCtx:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, *exc):
            return False

    mock_http = MagicMock()
    mock_http.stream = MagicMock(return_value=_FakeStreamCtx())
    client._get_client = AsyncMock(return_value=mock_http)
    return client


@pytest.mark.asyncio
async def test_chat_stream_truncated_done_frame_raises_retryable():
    """A ``done`` frame carrying a truncated tool-call ``arguments`` must
    raise ``LLMRetryableError`` out of the generator so the loop falls back
    to non-streaming."""
    client = _make_stream_client([
        {
            "type": "done",
            "content": "",
            "tool_calls": [{
                "name": "edit_agent",
                "arguments": '{"agent_id": "page-validator", "field": "instructions", "value": "You are',
            }],
            "tokens_used": 10,
            "model": "anthropic/claude-sonnet-4-6",
        },
    ])
    with pytest.raises(LLMRetryableError):
        async for _ in client.chat_stream(
            system="s", messages=[{"role": "user", "content": "x"}],
        ):
            pass


@pytest.mark.asyncio
async def test_chat_stream_transient_error_type_retryable():
    """Mesh-tagged ``error_type="transient"`` mid-stream must now retry
    (was a bare RuntimeError before the helper extraction)."""
    client = _make_stream_client([
        {
            "error": "Connection to the AI provider was interrupted.",
            "error_type": "transient",
            "error_meta": {"provider": "anthropic"},
        },
    ])
    with pytest.raises(LLMRetryableError):
        async for _ in client.chat_stream(
            system="s", messages=[{"role": "user", "content": "x"}],
        ):
            pass


@pytest.mark.asyncio
async def test_chat_stream_untagged_substring_backstop_retryable():
    """Substring backstop now works for streaming — an untagged error whose
    message contains ``retrying may help`` is retryable."""
    client = _make_stream_client([
        {"error": "Provider hiccup. Retrying may help."},
    ])
    with pytest.raises(LLMRetryableError):
        async for _ in client.chat_stream(
            system="s", messages=[{"role": "user", "content": "x"}],
        ):
            pass


@pytest.mark.asyncio
async def test_chat_stream_auth_failure_still_raises_auth_error():
    """Sanity — typed ``auth_failure`` still raises ``LLMAuthError`` (not
    the retryable subclass) through the shared helper."""
    client = _make_stream_client([
        {
            "error": "Invalid API key",
            "error_type": "auth_failure",
            "error_meta": {"provider": "anthropic", "http_status": 401},
        },
    ])
    with pytest.raises(LLMAuthError):
        async for _ in client.chat_stream(
            system="s", messages=[{"role": "user", "content": "x"}],
        ):
            pass


def test_default_max_output_tokens_is_8192():
    """The hardcoded 4096 cap was too small for large tool-call payloads."""
    llm = LLMClient(mesh_url="http://mesh:8420", agent_id="a1")
    assert llm.max_output_tokens == 8192


def test_max_output_tokens_override():
    llm = LLMClient(
        mesh_url="http://mesh:8420", agent_id="a1", max_output_tokens=32000,
    )
    assert llm.max_output_tokens == 32000


@pytest.mark.asyncio
async def test_chat_uses_instance_max_tokens_default():
    """chat() must send the instance max_output_tokens when caller omits it."""
    llm = LLMClient(
        mesh_url="http://mesh:8420", agent_id="a1", max_output_tokens=16384,
    )
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "success": True,
                "data": {"content": "ok", "tool_calls": [], "tokens_used": 1,
                         "model": "m"},
            }

    class _Client:
        async def post(self, url, json=None, params=None, headers=None):
            captured["max_tokens"] = json["params"]["max_tokens"]
            return _Resp()

    with patch.object(llm, "_get_client", AsyncMock(return_value=_Client())):
        await llm.chat(system="s", messages=[{"role": "user", "content": "hi"}])
    assert captured["max_tokens"] == 16384
