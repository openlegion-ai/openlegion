"""Unit tests for credential vault."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.credentials import CredentialVault, _extract_content
from src.shared.types import APIProxyRequest


@pytest.fixture
def vault():
    v = CredentialVault()
    return v


def test_load_credentials_from_env(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENLEGION_CRED_BRAVE_SEARCH_API_KEY", "bsk-456")
    v = CredentialVault()
    assert v.credentials.get("anthropic_api_key") == "sk-test-123"
    assert v.credentials.get("brave_search_api_key") == "bsk-456"


async def test_unknown_service_returns_error(vault):
    req = APIProxyRequest(service="nonexistent", action="do_thing")
    result = await vault.execute_api_call(req)
    assert not result.success
    assert "Unknown service" in result.error


async def test_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    v = CredentialVault()
    req = APIProxyRequest(service="anthropic", action="chat")
    result = await v.execute_api_call(req)
    assert not result.success
    assert "not configured" in result.error


def test_handler_dispatch():
    v = CredentialVault()
    assert "llm" in v.service_handlers
    assert "anthropic" in v.service_handlers
    assert "apollo" in v.service_handlers
    assert "hunter" in v.service_handlers
    assert "brave_search" in v.service_handlers
    assert "openai" in v.service_handlers


async def test_llm_missing_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_CRED_OPENAI_API_KEY", raising=False)
    v = CredentialVault()
    req = APIProxyRequest(
        service="llm",
        action="chat",
        params={"model": "openai/gpt-4o-mini", "messages": []},
    )
    result = await v.execute_api_call(req)
    assert not result.success
    assert "No API key" in result.error


def test_get_api_key_for_model(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-anthropic")
    v = CredentialVault()
    assert v._get_api_key_for_model("openai/gpt-4o-mini") == "sk-openai"
    assert v._get_api_key_for_model("gpt-4o") == "sk-openai"
    assert v._get_api_key_for_model("anthropic/claude-sonnet-4-5-20250929") == "sk-anthropic"
    assert v._get_api_key_for_model("unknown/model") is None


async def test_unknown_action_returns_error(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-test")
    v = CredentialVault()
    req = APIProxyRequest(service="anthropic", action="nonexistent_action")
    result = await v.execute_api_call(req)
    assert not result.success
    assert "Unknown action" in result.error


# ── Failover integration tests ────────────────────────────────


def test_vault_with_failover_config(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )
    models = v._failover_chain.get_models_to_try("anthropic/claude-haiku-4-5-20251001")
    assert models == ["anthropic/claude-haiku-4-5-20251001", "openai/gpt-4o-mini"]


async def test_handle_llm_failover_on_rate_limit(monkeypatch):
    """First model rate-limited, second succeeds."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )

    import litellm

    call_count = 0

    async def mock_acompletion(model, messages, api_key, **kwargs):
        nonlocal call_count
        call_count += 1
        if model.startswith("anthropic/"):
            raise litellm.RateLimitError(
                message="rate limited", model=model, llm_provider="anthropic",
            )
        # OpenAI succeeds
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "hello from fallback"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 42
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert result.data["content"] == "hello from fallback"
    assert result.data["model"] == "openai/gpt-4o-mini"
    assert call_count == 2


async def test_handle_llm_no_failover_on_bad_request(monkeypatch):
    """400 BadRequestError doesn't cascade — it's a permanent error."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )

    import litellm

    async def mock_acompletion(model, messages, api_key, **kwargs):
        raise litellm.BadRequestError(
            message="invalid request", model=model, llm_provider="anthropic",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-haiku-4-5-20251001", "messages": []},
        )
        result = await v.execute_api_call(req)

    assert not result.success
    assert "invalid request" in result.error


async def test_handle_llm_all_models_exhausted(monkeypatch):
    """All models fail → error returned."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )

    import litellm

    async def mock_acompletion(model, messages, api_key, **kwargs):
        raise litellm.ServiceUnavailableError(
            message=f"{model} down", model=model, llm_provider="test",
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-haiku-4-5-20251001", "messages": []},
        )
        result = await v.execute_api_call(req)

    assert not result.success
    assert "down" in result.error


async def test_stream_failover(monkeypatch):
    """Streaming: first model fails on connection, second streams OK."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )

    import litellm

    async def mock_chunk_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "streamed"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    call_count = 0

    async def mock_acompletion(model, messages, api_key, stream=False, **kwargs):
        nonlocal call_count
        call_count += 1
        if model.startswith("anthropic/"):
            raise litellm.ServiceUnavailableError(
                message="down", model=model, llm_provider="anthropic",
            )
        # Return an async generator for streaming
        return mock_chunk_generator()

    import json
    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]},
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

    assert call_count == 2
    # Should have text_delta and done events
    assert any("streamed" in e for e in events)
    assert any("done" in e for e in events)


# ── Hot-reload credential management ──────────────────────────


def test_add_credential_stores_in_memory(monkeypatch):
    monkeypatch.delenv("OPENLEGION_CRED_MY_SVC_KEY", raising=False)
    v = CredentialVault()
    # Patch _persist_to_env to avoid writing to real .env
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        handle = v.add_credential("my_svc_key", "secret123")
        assert v.credentials["my_svc_key"] == "secret123"
        assert handle == "$CRED{my_svc_key}"
    finally:
        cred_mod._persist_to_env = original


def test_add_credential_routes_api_base_to_api_bases(monkeypatch):
    """add_credential() with _api_base suffix stores in api_bases, not credentials."""
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        v.add_credential("openai_api_base", "https://gateway.example.com/v1")
        assert "openai_api_base" not in v.credentials
        assert v.api_bases["openai_api_base"] == "https://gateway.example.com/v1"
    finally:
        cred_mod._persist_to_env = original


def test_add_credential_regular_key_not_in_api_bases(monkeypatch):
    """add_credential() with a normal key stores in credentials, not api_bases."""
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        v.add_credential("openai_api_key", "sk-test")
        assert v.credentials["openai_api_key"] == "sk-test"
        assert "openai_api_key" not in v.api_bases
    finally:
        cred_mod._persist_to_env = original


def test_add_credential_persists_to_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=keep\n")

    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    from src.host.credentials import _persist_to_env

    v = CredentialVault()
    _persist_to_env("OPENLEGION_CRED_TEST_KEY", "val123", env_file=str(env_file))

    content = env_file.read_text()
    assert "OPENLEGION_CRED_TEST_KEY=val123" in content
    assert "EXISTING_VAR=keep" in content


def test_resolve_credential():
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        v.add_credential("found_key", "the_value")
        assert v.resolve_credential("found_key") == "the_value"
        assert v.resolve_credential("not_found") is None
    finally:
        cred_mod._persist_to_env = original


def test_list_credential_names(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_ALPHA_KEY", "a")
    monkeypatch.setenv("OPENLEGION_CRED_BETA_KEY", "b")
    v = CredentialVault()
    names = v.list_credential_names()
    assert "alpha_key" in names
    assert "beta_key" in names


def test_has_credential(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_EXISTS_KEY", "yes")
    v = CredentialVault()
    assert v.has_credential("exists_key") is True
    assert v.has_credential("nope_key") is False


# ── Custom API base URL tests ─────────────────────────────────


def test_load_api_base_from_env(monkeypatch):
    """OPENLEGION_CRED_*_API_BASE env vars are loaded into api_bases dict."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()
    assert v.api_bases.get("openai_api_base") == "https://gateway.example.com/v1"
    # API base should NOT appear in regular credentials
    assert "openai_api_base" not in v.credentials
    # Regular key should still be loaded
    assert v.credentials.get("openai_api_key") == "sk-test"


def test_get_api_base_for_model(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://custom.example.com/v1")
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_BASE", "https://anthropic.proxy.com")
    v = CredentialVault()
    assert v._get_api_base_for_model("openai/gpt-4o-mini") == "https://custom.example.com/v1"
    assert v._get_api_base_for_model("gpt-4o") == "https://custom.example.com/v1"
    assert v._get_api_base_for_model("anthropic/claude-haiku-4-5-20251001") == "https://anthropic.proxy.com"
    assert v._get_api_base_for_model("deepseek/deepseek-chat") is None
    assert v._get_api_base_for_model("unknown/model") is None


def test_no_api_base_configured(monkeypatch):
    """When no API base is set, _get_api_base_for_model returns None."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()
    assert v._get_api_base_for_model("openai/gpt-4o") is None


async def test_llm_chat_passes_api_base(monkeypatch):
    """api_base is forwarded to litellm.acompletion when configured."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://gateway.example.com/v1")
    v = CredentialVault()

    captured_kwargs = {}

    async def mock_acompletion(model, messages, api_key, **kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 10
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 5
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert captured_kwargs.get("api_base") == "https://gateway.example.com/v1"


async def test_llm_chat_no_api_base_when_not_configured(monkeypatch):
    """api_base is NOT passed to litellm when not configured."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()

    captured_kwargs = {}

    async def mock_acompletion(model, messages, api_key, **kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 10
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 5
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert "api_base" not in captured_kwargs


async def test_stream_passes_api_base(monkeypatch):
    """api_base is forwarded during streaming LLM calls."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://gateway.example.com/v1")
    v = CredentialVault()

    captured_kwargs = {}

    async def mock_chunk_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hi"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    async def mock_acompletion(model, messages, api_key, stream=False, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_chunk_generator()

    import json
    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

    assert captured_kwargs.get("api_base") == "https://gateway.example.com/v1"


async def test_cost_tracking_uses_actual_model(monkeypatch):
    """Cost is attributed to the model that actually responded."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")

    cost_tracker = MagicMock()
    cost_tracker.check_budget.return_value = {"allowed": True}
    v = CredentialVault(
        cost_tracker=cost_tracker,
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )

    import litellm

    async def mock_acompletion(model, messages, api_key, **kwargs):
        if model.startswith("anthropic/"):
            raise litellm.RateLimitError(
                message="rate limited", model=model, llm_provider="anthropic",
            )
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 100
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-haiku-4-5-20251001", "messages": []},
        )
        result = await v.execute_api_call(req, agent_id="test-agent")

    assert result.success
    # Cost should be tracked against actual model (openai/gpt-4o-mini)
    cost_tracker.track.assert_called_once()
    call_args = cost_tracker.track.call_args
    assert call_args[0][0] == "test-agent"
    assert call_args[0][1] == "openai/gpt-4o-mini"


# ── Extended thinking / _extract_content tests ──────────────────


def test_extract_content_string():
    """String content passes through unchanged."""
    text, thinking = _extract_content("hello world")
    assert text == "hello world"
    assert thinking is None


def test_extract_content_list_with_thinking():
    """List with thinking and text blocks splits correctly."""
    blocks = [
        {"type": "thinking", "thinking": "Let me reason about this..."},
        {"type": "text", "text": "The answer is 42."},
    ]
    text, thinking = _extract_content(blocks)
    assert text == "The answer is 42."
    assert thinking == "Let me reason about this..."


def test_extract_content_list_no_thinking():
    """List without thinking blocks returns None for thinking."""
    blocks = [
        {"type": "text", "text": "Part 1. "},
        {"type": "text", "text": "Part 2."},
    ]
    text, thinking = _extract_content(blocks)
    assert text == "Part 1. Part 2."
    assert thinking is None


def test_extract_content_none():
    """None/empty content returns empty string."""
    text, thinking = _extract_content(None)
    assert text == ""
    assert thinking is None

    text2, thinking2 = _extract_content("")
    assert text2 == ""
    assert thinking2 is None


def test_extract_content_multiple_thinking_blocks():
    """Multiple thinking blocks are concatenated."""
    blocks = [
        {"type": "thinking", "thinking": "Step 1. "},
        {"type": "thinking", "thinking": "Step 2."},
        {"type": "text", "text": "Done."},
    ]
    text, thinking = _extract_content(blocks)
    assert text == "Done."
    assert thinking == "Step 1. Step 2."


async def test_llm_chat_handles_list_content(monkeypatch):
    """Full path: list content from LiteLLM → extracted text + thinking_content."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    v = CredentialVault()

    async def mock_acompletion(model, messages, api_key, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        # Simulate extended thinking response — content is a list of blocks
        resp.choices[0].message.content = [
            {"type": "thinking", "thinking": "I need to think deeply."},
            {"type": "text", "text": "The final answer."},
        ]
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 100
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 50
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "think hard"}],
            },
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert result.data["content"] == "The final answer."
    assert result.data["thinking_content"] == "I need to think deeply."


def test_thinking_params_anthropic():
    """LLMClient generates correct Anthropic thinking params."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="high")
    params = client._get_thinking_params("anthropic/claude-sonnet-4-5-20250929")
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 25_000}
    assert params["temperature"] == 1.0
    # max_tokens must exceed budget_tokens (Anthropic requirement)
    assert params["max_tokens"] > 25_000


def test_thinking_params_anthropic_max_tokens_covers_budget():
    """max_tokens is auto-set to budget + 4096 for Anthropic thinking."""
    from src.agent.llm import LLMClient

    for level, expected_budget in [("low", 5_000), ("medium", 10_000), ("high", 25_000)]:
        client = LLMClient(mesh_url="http://test", thinking=level)
        params = client._get_thinking_params("anthropic/claude-sonnet-4-5-20250929")
        assert params["max_tokens"] == expected_budget + 4096


def test_thinking_params_openai_o_series():
    """LLMClient generates correct OpenAI reasoning params."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="medium")
    params = client._get_thinking_params("openai/o3")
    assert params == {"reasoning_effort": "medium"}


def test_thinking_params_openai_o_series_variants():
    """All o-series model name formats are detected."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="low")
    for model in ["openai/o1", "openai/o3", "openai/o4-mini", "o3", "o4-mini"]:
        params = client._get_thinking_params(model)
        assert params == {"reasoning_effort": "low"}, f"Failed for model: {model}"


def test_thinking_params_no_false_positive_on_slash_o():
    """Models with '/o' in the name but not o-series are not matched."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="high")
    # These models contain '/o' but are NOT OpenAI o-series
    for model in ["together/opt-350m", "huggingface/opt-1.3b"]:
        params = client._get_thinking_params(model)
        assert params == {}, f"False positive match for model: {model}"


def test_thinking_params_off():
    """thinking='off' returns empty params."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="off")
    params = client._get_thinking_params("anthropic/claude-sonnet-4-5-20250929")
    assert params == {}


def test_thinking_params_unsupported_model():
    """Unsupported model returns empty params even with thinking enabled."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="high")
    params = client._get_thinking_params("groq/llama-3.1-70b")
    assert params == {}


def test_thinking_invalid_level_falls_back_to_off():
    """Invalid thinking level falls back to 'off' with a warning."""
    from src.agent.llm import LLMClient

    client = LLMClient(mesh_url="http://test", thinking="extreme")
    assert client.thinking == "off"
    params = client._get_thinking_params("anthropic/claude-sonnet-4-5-20250929")
    assert params == {}


async def test_stream_collects_thinking_content(monkeypatch):
    """Streaming: reasoning_content tokens are collected and included in done event."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    v = CredentialVault()

    async def mock_chunk_generator():
        # First chunk: thinking token
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = None
        chunk1.choices[0].delta.reasoning_content = "Let me think..."
        chunk1.choices[0].delta.tool_calls = None
        yield chunk1

        # Second chunk: more thinking
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.reasoning_content = " Step 2."
        chunk2.choices[0].delta.tool_calls = None
        yield chunk2

        # Third chunk: actual text content
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = "The answer is 42."
        chunk3.choices[0].delta.reasoning_content = None
        chunk3.choices[0].delta.tool_calls = None
        yield chunk3

    async def mock_acompletion(model, messages, api_key, stream=False, **kwargs):
        return mock_chunk_generator()

    import json as _json
    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "think"}],
            },
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

    # Parse the done event
    done_events = [e for e in events if "done" in e]
    assert done_events, "No done event emitted"
    done_data = _json.loads(done_events[0].split("data: ")[1].strip())
    assert done_data["content"] == "The answer is 42."
    assert done_data["thinking_content"] == "Let me think... Step 2."

    # Text deltas should NOT include thinking tokens
    text_events = [e for e in events if "text_delta" in e]
    for te in text_events:
        data = _json.loads(te.split("data: ")[1].strip())
        assert "think" not in data.get("content", "").lower() or "42" in data.get("content", "")


async def test_stream_no_thinking_content_when_absent(monkeypatch):
    """Streaming: done event omits thinking_content when no reasoning tokens."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault()

    async def mock_chunk_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "hello"
        chunk.choices[0].delta.tool_calls = None
        # No reasoning_content attribute at all
        del chunk.choices[0].delta.reasoning_content
        yield chunk

    async def mock_acompletion(model, messages, api_key, stream=False, **kwargs):
        return mock_chunk_generator()

    import json as _json
    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

    done_events = [e for e in events if "done" in e]
    assert done_events
    done_data = _json.loads(done_events[0].split("data: ")[1].strip())
    assert "thinking_content" not in done_data


async def test_reasoning_content_attribute_fallback(monkeypatch):
    """Non-streaming: thinking extracted from msg.reasoning_content when content is a string."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    v = CredentialVault()

    async def mock_acompletion(model, messages, api_key, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        # litellm normalizes content to string, puts thinking in reasoning_content
        resp.choices[0].message.content = "The final answer."
        resp.choices[0].message.reasoning_content = "I thought about this carefully."
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 100
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 50
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "think hard"}],
            },
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert result.data["content"] == "The final answer."
    assert result.data["thinking_content"] == "I thought about this carefully."


async def test_reasoning_content_not_duplicated(monkeypatch):
    """When both content blocks AND reasoning_content exist, don't duplicate."""
    monkeypatch.setenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", "sk-ant")
    v = CredentialVault()

    async def mock_acompletion(model, messages, api_key, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        # Content is a list with thinking blocks — _extract_content handles it
        resp.choices[0].message.content = [
            {"type": "thinking", "thinking": "Deep thought."},
            {"type": "text", "text": "42."},
        ]
        # Also has reasoning_content (should be ignored since _extract_content found it)
        resp.choices[0].message.reasoning_content = "Deep thought."
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 50
        resp.usage.prompt_tokens = 25
        resp.usage.completion_tokens = 25
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "anthropic/claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "think"}],
            },
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert result.data["content"] == "42."
    # Should use the one from _extract_content, not duplicate
    assert result.data["thinking_content"] == "Deep thought."
