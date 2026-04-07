"""Unit tests for credential vault."""

from unittest.mock import MagicMock, patch

import pytest

from src.host.credentials import (
    AGENT_PREFIX,
    SYSTEM_CREDENTIAL_PROVIDERS,
    SYSTEM_PREFIX,
    CredentialVault,
    _extract_content,
    is_system_credential,
)
from src.shared.types import APIProxyRequest, APIProxyResponse


@pytest.fixture
def vault():
    v = CredentialVault()
    return v


def test_load_credentials_from_env(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENLEGION_CRED_BRAVE_SEARCH_API_KEY", "bsk-456")
    v = CredentialVault()
    # System-prefix keys go to system tier
    assert v.system_credentials.get("anthropic_api_key") == "sk-test-123"
    assert "anthropic_api_key" not in v.credentials
    # Agent-prefix keys stay in agent tier
    assert v.credentials.get("brave_search_api_key") == "bsk-456"


async def test_unknown_service_returns_error(vault):
    req = APIProxyRequest(service="nonexistent", action="do_thing")
    result = await vault.execute_api_call(req)
    assert not result.success
    assert "Unknown service" in result.error


async def test_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_CRED_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    v = CredentialVault()
    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "anthropic/claude-sonnet-4-5-20250929", "messages": []},
    )
    result = await v.execute_api_call(req)
    assert not result.success
    assert "No API key" in result.error


def test_handler_dispatch():
    v = CredentialVault()
    assert "llm" in v.service_handlers
    assert "apollo" in v.service_handlers
    assert "hunter" in v.service_handlers
    assert "brave_search" in v.service_handlers


async def test_llm_missing_key_returns_error(monkeypatch):
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_CRED_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-anthropic")
    v = CredentialVault()
    assert v._get_api_key_for_model("openai/gpt-4o-mini") == "sk-openai"
    assert v._get_api_key_for_model("gpt-4o") == "sk-openai"
    assert v._get_api_key_for_model("anthropic/claude-sonnet-4-5-20250929") == "sk-anthropic"
    assert v._get_api_key_for_model("unknown/model") is None


async def test_unknown_action_returns_error(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-test")
    v = CredentialVault()
    req = APIProxyRequest(
        service="llm", action="nonexistent_action",
        params={"model": "anthropic/claude-sonnet-4-5-20250929"},
    )
    result = await v.execute_api_call(req)
    assert not result.success
    assert "Unknown action" in result.error


# ── Remove credential tests ───────────────────────────────────


def test_remove_credential():
    vault = CredentialVault()
    vault.add_credential("test_api_key", "sk-secret")
    assert vault.has_credential("test_api_key")

    existed = vault.remove_credential("test_api_key")
    assert existed is True
    assert not vault.has_credential("test_api_key")
    assert vault.resolve_credential("test_api_key") is None


def test_remove_credential_not_found():
    vault = CredentialVault()
    existed = vault.remove_credential("nonexistent_key")
    assert existed is False


def test_remove_api_base():
    vault = CredentialVault()
    vault.api_bases["openai_api_base"] = "https://example.com"
    existed = vault.remove_credential("openai_api_base")
    assert existed is True
    assert "openai_api_base" not in vault.api_bases


def test_remove_from_env_file(tmp_path):
    from src.host.credentials import _remove_from_env

    env_file = tmp_path / ".env"
    env_file.write_text("FOO=bar\nOPENLEGION_CRED_TEST=secret\nBAZ=qux\n")
    _remove_from_env("OPENLEGION_CRED_TEST", env_file=str(env_file))
    content = env_file.read_text()
    assert "OPENLEGION_CRED_TEST" not in content
    assert "FOO=bar" in content
    assert "BAZ=qux" in content


def test_remove_from_env_file_quoted(tmp_path):
    """_remove_from_env handles the new quoted format."""
    from src.host.credentials import _remove_from_env

    env_file = tmp_path / ".env"
    env_file.write_text("FOO='bar'\nOPENLEGION_CRED_KEY='secret$val'\nBAZ='qux'\n")
    _remove_from_env("OPENLEGION_CRED_KEY", env_file=str(env_file))
    content = env_file.read_text()
    assert "OPENLEGION_CRED_KEY" not in content
    assert "FOO='bar'" in content
    assert "BAZ='qux'" in content


# ── Failover integration tests ────────────────────────────────


def test_vault_with_failover_config(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
    v = CredentialVault(
        failover_config={"anthropic/claude-haiku-4-5-20251001": ["openai/gpt-4o-mini"]},
    )
    models = v._failover_chain.get_models_to_try("anthropic/claude-haiku-4-5-20251001")
    assert models == ["anthropic/claude-haiku-4-5-20251001", "openai/gpt-4o-mini"]


def test_auto_failover_chains(monkeypatch):
    """Without explicit failover config, auto-chains are built from featured models."""
    for p in ("anthropic", "openai", "deepseek", "moonshot", "minimax", "xai", "groq", "zai"):
        monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p.upper()}_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "gem-key")
    v = CredentialVault()
    # gemini/gemini-2.5-pro should failover to gemini/gemini-2.5-flash
    models = v._failover_chain.get_models_to_try("gemini/gemini-2.5-pro")
    assert "gemini/gemini-2.5-pro" in models
    assert "gemini/gemini-2.5-flash" in models
    # The flash model should be after pro
    assert models.index("gemini/gemini-2.5-pro") < models.index("gemini/gemini-2.5-flash")


async def test_handle_llm_failover_on_rate_limit(monkeypatch):
    """First model rate-limited, second succeeds."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
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


def test_add_credential_strips_whitespace(monkeypatch):
    """add_credential() strips leading/trailing whitespace from values.

    Trailing spaces from terminal paste corrupt tokens and cause auth
    failures (e.g. Notion 401 when the stored value has a trailing space).
    """
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    persisted = {}
    cred_mod._persist_to_env = lambda k, val, **kw: persisted.update({k: val})
    try:
        v.add_credential("notion", "  ntn_secret_abc123  ")
        assert v.credentials["notion"] == "ntn_secret_abc123"
        # Persisted value should also be stripped
        assert persisted[f"{AGENT_PREFIX}NOTION"] == "ntn_secret_abc123"
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
    monkeypatch.delenv("OPENLEGION_CRED_TEST_KEY", raising=False)

    from src.host.credentials import _persist_to_env

    _persist_to_env("OPENLEGION_CRED_TEST_KEY", "val123", env_file=str(env_file))

    content = env_file.read_text()
    assert "OPENLEGION_CRED_TEST_KEY='val123'" in content
    assert "EXISTING_VAR=keep" in content


def test_persist_to_env_quotes_survive_dotenv_reload(tmp_path, monkeypatch):
    """Values with $ must survive a python-dotenv round-trip (no interpolation)."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)

    # Token with $ that would be corrupted without quoting + interpolate=False
    token = "sk-ant-oat01-abc$xyz$123"
    _persist_to_env("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", token, env_file=str(env_file))

    # Simulate restart: read back the same way production does
    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_SYSTEM_ANTHROPIC_API_KEY"] == token


def test_persist_to_env_quotes_with_hash(tmp_path, monkeypatch):
    """Values with # must not be truncated as inline comments."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_TEST", raising=False)
    value = "key-with-hash#inside"
    _persist_to_env("OPENLEGION_CRED_TEST", value, env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_TEST"] == value


def test_persist_to_env_single_quote_in_value(tmp_path, monkeypatch):
    """Values containing single quotes use double-quote fallback."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_QUOTE", raising=False)
    value = "it's-a-test"
    _persist_to_env("OPENLEGION_CRED_QUOTE", value, env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_QUOTE"] == value


def test_persist_to_env_single_quote_and_dollar(tmp_path, monkeypatch):
    """Values with both ' and $ use double-quote fallback."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_COMBO", raising=False)
    value = "it's$money"
    _persist_to_env("OPENLEGION_CRED_COMBO", value, env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_COMBO"] == value


def test_persist_to_env_all_special_chars(tmp_path, monkeypatch):
    """Round-trip a value containing every special char: ' " $ \\ #."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_SPECIAL", raising=False)
    value = "a'b\"c$d\\e#f"
    _persist_to_env("OPENLEGION_CRED_SPECIAL", value, env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_SPECIAL"] == value


def test_persist_to_env_empty_value(tmp_path, monkeypatch):
    """Empty string round-trips correctly."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_EMPTY", raising=False)
    _persist_to_env("OPENLEGION_CRED_EMPTY", "", env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_EMPTY"] == ""


def test_persist_to_env_overwrites_quoted_value(tmp_path, monkeypatch):
    """Updating an already-quoted .env entry works correctly."""
    from dotenv import dotenv_values

    from src.host.credentials import _persist_to_env

    env_file = tmp_path / ".env"
    monkeypatch.delenv("OPENLEGION_CRED_UPD", raising=False)
    _persist_to_env("OPENLEGION_CRED_UPD", "old$val", env_file=str(env_file))
    _persist_to_env("OPENLEGION_CRED_UPD", "new$val", env_file=str(env_file))

    loaded = dotenv_values(str(env_file), interpolate=False)
    assert loaded["OPENLEGION_CRED_UPD"] == "new$val"
    # Must not duplicate lines
    content = env_file.read_text()
    assert content.count("OPENLEGION_CRED_UPD=") == 1


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
    monkeypatch.setenv("OPENLEGION_SYSTEM_BETA_KEY", "b")
    v = CredentialVault()
    names = v.list_credential_names()
    # list_credential_names returns both tiers combined
    assert "alpha_key" in names
    assert "beta_key" in names


def test_has_credential(monkeypatch):
    monkeypatch.setenv("OPENLEGION_CRED_EXISTS_KEY", "yes")
    monkeypatch.setenv("OPENLEGION_SYSTEM_SYS_KEY", "sys-yes")
    v = CredentialVault()
    assert v.has_credential("exists_key") is True
    assert v.has_credential("sys_key") is True  # system tier
    assert v.has_credential("nope_key") is False


# ── Custom API base URL tests ─────────────────────────────────


def test_load_api_base_from_env(monkeypatch):
    """OPENLEGION_CRED_*_API_BASE env vars are loaded into api_bases dict."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://gateway.example.com/v1")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()
    assert v.api_bases.get("openai_api_base") == "https://gateway.example.com/v1"
    # API base should NOT appear in regular credentials
    assert "openai_api_base" not in v.credentials
    # Regular key should be in system tier
    assert v.system_credentials.get("openai_api_key") == "sk-test"


def test_get_api_base_for_model(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_BASE", "https://custom.example.com/v1")
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_BASE", "https://anthropic.proxy.com")
    v = CredentialVault()
    assert v._get_api_base_for_model("openai/gpt-4o-mini") == "https://custom.example.com/v1"
    assert v._get_api_base_for_model("gpt-4o") == "https://custom.example.com/v1"
    assert v._get_api_base_for_model("anthropic/claude-haiku-4-5-20251001") == "https://anthropic.proxy.com"
    assert v._get_api_base_for_model("deepseek/deepseek-chat") is None
    assert v._get_api_base_for_model("unknown/model") is None


def test_no_api_base_configured(monkeypatch):
    """When no API base is set, _get_api_base_for_model returns None."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()
    assert v._get_api_base_for_model("openai/gpt-4o") is None


async def test_llm_chat_passes_api_base(monkeypatch):
    """api_base is forwarded to litellm.acompletion when configured."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_BASE", "https://gateway.example.com/v1")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_BASE", "https://gateway.example.com/v1")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")

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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-oai")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
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
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
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


# ── System credential detection tests ────────────────────────────


def test_is_system_credential_api_keys():
    """Provider API keys are system credentials."""
    assert is_system_credential("anthropic_api_key") is True
    assert is_system_credential("openai_api_key") is True
    assert is_system_credential("gemini_api_key") is True
    assert is_system_credential("deepseek_api_key") is True
    assert is_system_credential("moonshot_api_key") is True
    assert is_system_credential("minimax_api_key") is True
    assert is_system_credential("xai_api_key") is True
    assert is_system_credential("groq_api_key") is True
    assert is_system_credential("zai_api_key") is True


def test_is_system_credential_api_bases():
    """Provider API base URLs are system credentials."""
    assert is_system_credential("anthropic_api_base") is True
    assert is_system_credential("openai_api_base") is True


def test_is_system_credential_case_insensitive():
    """Detection is case-insensitive."""
    assert is_system_credential("ANTHROPIC_API_KEY") is True
    assert is_system_credential("OpenAI_API_Base") is True


def test_is_system_credential_non_system():
    """Non-provider credentials are not system credentials."""
    assert is_system_credential("brightdata_cdp_url") is False
    assert is_system_credential("myapp_password") is False
    assert is_system_credential("custom_api_key") is False
    assert is_system_credential("brave_search_api_key") is False
    assert is_system_credential("apollo_api_key") is False
    assert is_system_credential("hunter_api_key") is False


def test_list_agent_credential_names(monkeypatch):
    """list_agent_credential_names excludes system credentials."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("OPENLEGION_CRED_BRIGHTDATA_CDP_URL", "wss://...")
    monkeypatch.setenv("OPENLEGION_CRED_MYAPP_PASSWORD", "secret")
    v = CredentialVault()
    agent_creds = v.list_agent_credential_names()
    assert "brightdata_cdp_url" in agent_creds
    assert "myapp_password" in agent_creds
    # System-prefix key stays in system tier
    assert "anthropic_api_key" not in agent_creds


def test_list_agent_credential_names_empty():
    """list_agent_credential_names returns empty when only system creds exist."""
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        v.add_credential("anthropic_api_key", "sk-test", system=True)
        agent_creds = v.list_agent_credential_names()
        assert "anthropic_api_key" not in agent_creds
        # But it should be in system credentials
        assert "anthropic_api_key" in v.list_system_credential_names()
    finally:
        cred_mod._persist_to_env = original


def test_system_credential_providers_includes_override_providers():
    """SYSTEM_CREDENTIAL_PROVIDERS must include all providers from prefix overrides.

    Any provider in _PROVIDER_PREFIX_OVERRIDES must also be in
    SYSTEM_CREDENTIAL_PROVIDERS so that its API key is correctly
    classified as system-tier (not agent-accessible).
    """
    override_providers = frozenset(CredentialVault._PROVIDER_PREFIX_OVERRIDES.values())
    missing = override_providers - SYSTEM_CREDENTIAL_PROVIDERS
    assert not missing, (
        f"Providers from _PROVIDER_PREFIX_OVERRIDES missing from "
        f"SYSTEM_CREDENTIAL_PROVIDERS: {missing}"
    )


def test_system_credential_providers_includes_curated():
    """SYSTEM_CREDENTIAL_PROVIDERS must include all curated providers."""
    from src.shared.models import _PROVIDER_LABELS
    curated = frozenset(_PROVIDER_LABELS.keys())
    missing = curated - SYSTEM_CREDENTIAL_PROVIDERS
    assert not missing, (
        f"Curated providers missing from SYSTEM_CREDENTIAL_PROVIDERS: {missing}"
    )


# ── Two-tier credential system tests ──────────────────────────


def test_system_prefix_loads_to_system_credentials(monkeypatch):
    """OPENLEGION_SYSTEM_* env vars load into system_credentials dict."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-sys")
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "gk-sys")
    v = CredentialVault()
    assert v.system_credentials["openai_api_key"] == "sk-sys"
    assert v.system_credentials["gemini_api_key"] == "gk-sys"
    # Should NOT appear in agent-tier credentials
    assert "openai_api_key" not in v.credentials
    assert "gemini_api_key" not in v.credentials


def test_cred_prefix_provider_keys_stay_in_agent_tier(monkeypatch):
    """OPENLEGION_CRED_ provider keys are NOT auto-promoted — they stay agent-tier."""
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-legacy")
    v = CredentialVault()
    # Key stays in agent tier, not promoted
    assert "openai_api_key" in v.credentials
    assert "openai_api_key" not in v.system_credentials


def test_both_prefixes_land_in_respective_tiers(monkeypatch):
    """OPENLEGION_SYSTEM_ and OPENLEGION_CRED_ for same name go to their own tiers."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-system")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_KEY", "sk-agent")
    v = CredentialVault()
    assert v.system_credentials["openai_api_key"] == "sk-system"
    assert v.credentials["openai_api_key"] == "sk-agent"
    # Proxy only uses system tier
    assert v._get_api_key_for_model("openai/gpt-4o-mini") == "sk-system"


def test_get_api_key_only_uses_system_tier(monkeypatch):
    """_get_api_key_for_model only checks system_credentials."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-system")
    v = CredentialVault()
    assert v._get_api_key_for_model("openai/gpt-4o-mini") == "sk-system"


def test_get_api_key_ignores_agent_tier(monkeypatch):
    """_get_api_key_for_model does NOT fall back to agent-tier credentials."""
    v = CredentialVault()
    v.credentials["openai_api_key"] = "sk-agent-only"
    assert v._get_api_key_for_model("openai/gpt-4o-mini") is None


def test_add_credential_system_true():
    """add_credential(system=True) stores in system_credentials."""
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    persisted = {}
    cred_mod._persist_to_env = lambda k, val, **kw: persisted.update({k: val})
    try:
        v.add_credential("openai_api_key", "sk-new", system=True)
        assert v.system_credentials["openai_api_key"] == "sk-new"
        assert "openai_api_key" not in v.credentials
        # Env key should use SYSTEM prefix
        assert f"{SYSTEM_PREFIX}OPENAI_API_KEY" in persisted
    finally:
        cred_mod._persist_to_env = original


def test_add_credential_system_false():
    """add_credential(system=False) stores in credentials (agent tier)."""
    v = CredentialVault()
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    persisted = {}
    cred_mod._persist_to_env = lambda k, val, **kw: persisted.update({k: val})
    try:
        v.add_credential("myapp_token", "tok-123", system=False)
        assert v.credentials["myapp_token"] == "tok-123"
        assert "myapp_token" not in v.system_credentials
        # Env key should use CRED prefix
        assert f"{AGENT_PREFIX}MYAPP_TOKEN" in persisted
    finally:
        cred_mod._persist_to_env = original


def test_resolve_credential_never_returns_system_tier(monkeypatch):
    """resolve_credential only returns agent-tier credentials."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-system")
    monkeypatch.setenv("OPENLEGION_CRED_MYAPP_KEY", "app-key")
    v = CredentialVault()
    # System tier: not resolvable
    assert v.resolve_credential("openai_api_key") is None
    # Agent tier: resolvable
    assert v.resolve_credential("myapp_key") == "app-key"


def test_list_system_credential_names(monkeypatch):
    """list_system_credential_names returns only system tier."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-sys")
    monkeypatch.setenv("OPENLEGION_CRED_MYAPP_KEY", "app-key")
    v = CredentialVault()
    sys_names = v.list_system_credential_names()
    assert "openai_api_key" in sys_names
    assert "myapp_key" not in sys_names


def test_remove_credential_both_tiers():
    """remove_credential searches both tiers and cleans both env prefixes."""
    v = CredentialVault()
    v.system_credentials["test_key"] = "sys-val"
    v.credentials["test_key"] = "agent-val"

    import src.host.credentials as cred_mod
    original = cred_mod._remove_from_env
    removed_keys = []
    cred_mod._remove_from_env = lambda k, **kw: removed_keys.append(k)
    try:
        existed = v.remove_credential("test_key")
        assert existed is True
        assert "test_key" not in v.system_credentials
        assert "test_key" not in v.credentials
        # Both prefixes should be cleaned
        assert f"{SYSTEM_PREFIX}TEST_KEY" in removed_keys
        assert f"{AGENT_PREFIX}TEST_KEY" in removed_keys
    finally:
        cred_mod._remove_from_env = original


def test_system_api_base_takes_precedence(monkeypatch):
    """OPENLEGION_SYSTEM_*_API_BASE takes precedence over OPENLEGION_CRED_*_API_BASE."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_BASE", "https://system-proxy.com")
    monkeypatch.setenv("OPENLEGION_CRED_OPENAI_API_BASE", "https://legacy-proxy.com")
    v = CredentialVault()
    assert v.api_bases["openai_api_base"] == "https://system-proxy.com"


def test_agent_tier_cred_not_auto_promoted(monkeypatch):
    """Non-provider OPENLEGION_CRED_ keys stay in agent tier."""
    monkeypatch.setenv("OPENLEGION_CRED_BRIGHTDATA_CDP_URL", "wss://test")
    monkeypatch.setenv("OPENLEGION_CRED_BRAVE_SEARCH_API_KEY", "bsk-123")
    v = CredentialVault()
    assert "brightdata_cdp_url" in v.credentials
    assert "brave_search_api_key" in v.credentials
    assert "brightdata_cdp_url" not in v.system_credentials
    assert "brave_search_api_key" not in v.system_credentials


class TestGetAuthForModel:
    """Tests for _get_auth_for_model key resolution."""

    def test_get_auth_for_model_standard_key(self, monkeypatch):
        """Standard API key returns empty auth headers."""
        monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-regular")
        v = CredentialVault()
        api_key, headers = v._get_auth_for_model("anthropic/claude-sonnet-4-6")
        assert api_key == "sk-ant-regular"
        assert headers == {}

    def test_get_auth_for_model_no_key(self, monkeypatch):
        """Missing key returns (None, {})."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
        v = CredentialVault()
        api_key, headers = v._get_auth_for_model("anthropic/claude-sonnet-4-6")
        assert api_key is None
        assert headers == {}

    def test_providers_with_credentials(self, monkeypatch):
        monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-openai")
        # Clear other provider keys so the test is deterministic
        for p in ("gemini", "deepseek", "moonshot", "minimax", "xai", "groq", "zai"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p.upper()}_API_KEY", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
        v = CredentialVault()
        providers = v.get_providers_with_credentials()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "gemini" not in providers

    def test_providers_with_credentials_empty(self, monkeypatch):
        for provider in ("anthropic", "openai", "gemini", "deepseek",
                         "moonshot", "minimax", "xai", "groq", "zai"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{provider.upper()}_API_KEY", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
        v = CredentialVault()
        providers = v.get_providers_with_credentials()
        assert providers == set()

    def test_providers_with_credentials_includes_ollama_with_base(self, monkeypatch):
        """Ollama is included when its API base is configured (keyless provider)."""
        for provider in ("anthropic", "openai", "gemini", "deepseek",
                         "moonshot", "minimax", "xai", "groq", "zai"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{provider.upper()}_API_KEY", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
        monkeypatch.setenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", "http://localhost:11434")
        v = CredentialVault()
        providers = v.get_providers_with_credentials()
        assert providers == {"ollama"}

    def test_providers_with_credentials_includes_openai_oauth(self, monkeypatch):
        """OpenAI OAuth counts as having OpenAI credentials."""
        for p in ("anthropic", "openai", "gemini", "deepseek",
                   "moonshot", "minimax", "xai", "groq", "zai"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p.upper()}_API_KEY", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_OPENAI_OAUTH",
            '{"access_token":"tok","refresh_token":"ref"}',
        )
        v = CredentialVault()
        providers = v.get_providers_with_credentials()
        assert "openai" in providers

    def test_openai_oauth_not_in_system_credentials(self, monkeypatch):
        """OPENLEGION_SYSTEM_OPENAI_OAUTH should not appear in system_credentials."""
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_OPENAI_OAUTH",
            '{"access_token":"tok","refresh_token":"ref"}',
        )
        v = CredentialVault()
        assert "openai_oauth" not in v.system_credentials
        assert v._has_openai_oauth()

    def test_get_auth_for_model_oauth_token(self, monkeypatch):
        """OAuth setup-token returns key with empty headers (OAuth bypasses LiteLLM)."""
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_API_KEY",
            "sk-ant-oat01-" + "x" * 80,
        )
        v = CredentialVault()
        api_key, headers = v._get_auth_for_model("anthropic/claude-sonnet-4-6")
        assert api_key.startswith("sk-ant-oat01-")
        # OAuth tokens bypass LiteLLM — SDK handles auth headers natively
        assert headers == {}


# ── Keyless providers (Ollama) ──────────────────────────────────


class TestKeylessProvider:
    """Tests for keyless provider handling (e.g. Ollama)."""

    def test_is_keyless_provider_ollama(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()
        assert v._is_keyless_provider("ollama/llama3") is True

    def test_is_keyless_provider_ollama_chat(self, monkeypatch):
        """ollama_chat/ prefix also maps to the ollama provider."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()
        assert v._is_keyless_provider("ollama_chat/llama3") is True

    def test_is_keyless_provider_cloud(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()
        assert v._is_keyless_provider("openai/gpt-4o") is False
        assert v._is_keyless_provider("anthropic/claude-sonnet-4-6") is False

    def test_is_keyless_provider_unknown(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()
        assert v._is_keyless_provider("unknown-model") is False

    @pytest.mark.asyncio
    async def test_discover_ollama_models_success(self, monkeypatch):
        """discover_ollama_models parses Ollama API response correctly."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()

        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3.3:latest"},
                    {"name": "mistral:7b"},
                    {"name": "codellama:latest"},
                ]
            },
        )

        async def mock_get(*args, **kwargs):
            return mock_response

        client = await v._get_http_client()
        monkeypatch.setattr(client, "get", mock_get)

        models = await v.discover_ollama_models()
        # :latest should be stripped, :7b kept
        assert "ollama/llama3.3" in models
        assert "ollama/mistral:7b" in models
        assert "ollama/codellama" in models
        # Should be sorted
        assert models == sorted(models)

    @pytest.mark.asyncio
    async def test_discover_ollama_models_unreachable(self, monkeypatch):
        """discover_ollama_models returns empty list on connection error."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()

        import httpx

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        client = await v._get_http_client()
        monkeypatch.setattr(client, "get", mock_get)

        models = await v.discover_ollama_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_discover_ollama_models_custom_base(self, monkeypatch):
        """discover_ollama_models uses custom API base if configured."""
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_OLLAMA_API_BASE", "http://192.168.1.100:11434",
        )
        v = CredentialVault()

        captured_url = []

        async def mock_get(url, **kwargs):
            captured_url.append(url)
            import httpx
            return httpx.Response(200, json={"models": []})

        client = await v._get_http_client()
        monkeypatch.setattr(client, "get", mock_get)

        await v.discover_ollama_models()
        assert captured_url[0] == "http://192.168.1.100:11434/api/tags"

    @pytest.mark.asyncio
    async def test_discover_ollama_models_deduplicates(self, monkeypatch):
        """discover_ollama_models deduplicates models."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        v = CredentialVault()

        import httpx

        mock_response = httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "llama3:latest"},
                ]
            },
        )

        async def mock_get(*args, **kwargs):
            return mock_response

        client = await v._get_http_client()
        monkeypatch.setattr(client, "get", mock_get)

        models = await v.discover_ollama_models()
        assert models == ["ollama/llama3"]


# ── OAuth token detection ──────────────────────────────────────


class TestOAuthTokenHandling:
    """Tests for OAuth setup-token detection and API body conversion."""

    def test_is_oauth_token(self):
        from src.host.credentials import is_oauth_token
        assert is_oauth_token("sk-ant-oat01-" + "x" * 80)
        assert not is_oauth_token("sk-ant-api03-regular-key")
        assert not is_oauth_token("")
        assert not is_oauth_token("some-random-token")

    def test_build_anthropic_body_basic(self):
        """Converts LiteLLM-style params to Anthropic format."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
            "temperature": 0.5,
        }
        body = CredentialVault._build_anthropic_body(params)
        assert body["model"] == "claude-sonnet-4-6"
        assert body["system"] == "You are helpful."
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.5

    def test_build_anthropic_body_with_tools(self):
        """Converts OpenAI-style tools to Anthropic format."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }],
        }
        body = CredentialVault._build_anthropic_body(params)
        assert len(body["tools"]) == 1
        tool = body["tools"][0]
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web"
        assert "input_schema" in tool

    def test_parse_anthropic_response_text(self):
        """Parses a simple text response."""
        data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-4-6",
        }
        result = CredentialVault._parse_anthropic_response(data, "anthropic/claude-sonnet-4-6")
        assert result["content"] == "Hello!"
        assert result["tokens_used"] == 15
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert result["tool_calls"] == []

    def test_parse_anthropic_response_tool_use(self):
        """Parses a tool-use response."""
        import json
        data = {
            "content": [
                {"type": "text", "text": "Let me search."},
                {"type": "tool_use", "name": "search", "input": {"q": "test"}},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 30},
        }
        result = CredentialVault._parse_anthropic_response(data, "anthropic/claude-sonnet-4-6")
        assert result["content"] == "Let me search."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"
        assert json.loads(result["tool_calls"][0]["arguments"]) == {"q": "test"}

    def test_parse_anthropic_response_thinking(self):
        """Parses response with thinking blocks."""
        data = {
            "content": [
                {"type": "thinking", "thinking": "reasoning here"},
                {"type": "text", "text": "Answer"},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 100},
        }
        result = CredentialVault._parse_anthropic_response(data, "anthropic/test")
        assert result["content"] == "Answer"
        assert result["thinking_content"] == "reasoning here"

    def test_build_anthropic_body_tool_choice_auto(self):
        """tool_choice='auto' converts to Anthropic format."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"function": {"name": "t", "parameters": {"type": "object"}}}],
            "tool_choice": "auto",
        }
        body = CredentialVault._build_anthropic_body(params)
        assert body["tool_choice"] == {"type": "auto"}

    def test_build_anthropic_body_tool_choice_required(self):
        """tool_choice='required' maps to Anthropic 'any'."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"function": {"name": "t", "parameters": {"type": "object"}}}],
            "tool_choice": "required",
        }
        body = CredentialVault._build_anthropic_body(params)
        assert body["tool_choice"] == {"type": "any"}

    def test_build_anthropic_body_tool_choice_specific(self):
        """tool_choice with specific function maps to Anthropic 'tool' type."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"function": {"name": "search", "parameters": {"type": "object"}}}],
            "tool_choice": {"function": {"name": "search"}},
        }
        body = CredentialVault._build_anthropic_body(params)
        assert body["tool_choice"] == {"type": "tool", "name": "search"}

    def test_build_anthropic_body_top_p(self):
        """top_p is forwarded to body."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "top_p": 0.9,
        }
        body = CredentialVault._build_anthropic_body(params)
        assert body["top_p"] == 0.9

    def test_build_anthropic_body_tool_choice_none_removes_tools(self):
        """tool_choice='none' removes tools from body entirely."""
        params = {
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [{"function": {"name": "t", "parameters": {"type": "object"}}}],
            "tool_choice": "none",
        }
        body = CredentialVault._build_anthropic_body(params)
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_build_anthropic_body_converts_tool_messages(self):
        """OpenAI-format tool_calls and role:'tool' are converted to Anthropic format."""
        params = {
            "model": "anthropic/claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "search for cats"},
                {
                    "role": "assistant",
                    "content": "Let me search.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "cats"}'},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": "Found 3 results",
                    "tool_call_id": "call_abc",
                },
                {"role": "assistant", "content": "I found 3 results about cats."},
            ],
        }
        body = CredentialVault._build_anthropic_body(params)
        msgs = body["messages"]
        assert len(msgs) == 4

        # Assistant with tool_use blocks
        assert msgs[1]["role"] == "assistant"
        assert isinstance(msgs[1]["content"], list)
        assert msgs[1]["content"][0] == {"type": "text", "text": "Let me search."}
        assert msgs[1]["content"][1]["type"] == "tool_use"
        assert msgs[1]["content"][1]["id"] == "call_abc"
        assert msgs[1]["content"][1]["name"] == "search"
        assert msgs[1]["content"][1]["input"] == {"q": "cats"}

        # Tool result → user with tool_result block
        assert msgs[2]["role"] == "user"
        assert isinstance(msgs[2]["content"], list)
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert msgs[2]["content"][0]["tool_use_id"] == "call_abc"
        assert msgs[2]["content"][0]["content"] == "Found 3 results"

        # Final assistant unchanged
        assert msgs[3] == {"role": "assistant", "content": "I found 3 results about cats."}

    def test_build_anthropic_body_merges_consecutive_tool_results(self):
        """Multiple consecutive tool results merge into one user message."""
        params = {
            "model": "anthropic/claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "do two things"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "content": "result_a", "tool_call_id": "c1"},
                {"role": "tool", "content": "result_b", "tool_call_id": "c2"},
            ],
        }
        body = CredentialVault._build_anthropic_body(params)
        msgs = body["messages"]
        assert len(msgs) == 3  # user, assistant, merged-user

        # Assistant: empty content should not produce a text block
        assistant_blocks = msgs[1]["content"]
        assert all(b["type"] == "tool_use" for b in assistant_blocks)
        assert len(assistant_blocks) == 2

        # Merged tool results
        assert msgs[2]["role"] == "user"
        assert len(msgs[2]["content"]) == 2
        assert msgs[2]["content"][0]["tool_use_id"] == "c1"
        assert msgs[2]["content"][1]["tool_use_id"] == "c2"

    def test_build_anthropic_body_converts_multimodal_tool_results(self):
        """Tool results with image_url blocks are converted to Anthropic image format."""
        params = {
            "model": "anthropic/claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "take a screenshot"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ss",
                            "type": "function",
                            "function": {"name": "browser_screenshot", "arguments": "{}"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_ss",
                    "content": [
                        {"type": "text", "text": '{"status": "screenshot captured"}'},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                },
            ],
        }
        body = CredentialVault._build_anthropic_body(params)
        msgs = body["messages"]

        # Tool result is in a user message with tool_result block
        tool_user_msg = msgs[2]
        assert tool_user_msg["role"] == "user"
        tool_result = tool_user_msg["content"][0]
        assert tool_result["type"] == "tool_result"

        # The content inside tool_result should have Anthropic image format
        inner = tool_result["content"]
        assert isinstance(inner, list)
        assert inner[0] == {"type": "text", "text": '{"status": "screenshot captured"}'}
        assert inner[1]["type"] == "image"
        assert inner[1]["source"]["type"] == "base64"
        assert inner[1]["source"]["media_type"] == "image/png"
        assert inner[1]["source"]["data"] == "iVBORw0KGgo="


class TestVisionStripping:
    """Non-vision models get image blocks stripped from tool messages."""

    def test_prepare_llm_params_strips_images_for_non_vision_model(self, vault):
        """Tool messages with image_url blocks are flattened to text for non-vision models."""
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "groq/llama-3.3-70b-versatile",
                "messages": [
                    {"role": "user", "content": "take a screenshot"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "browser_screenshot", "arguments": "{}"}},
                        ],
                    },
                    {
                        "role": "tool", "tool_call_id": "c1",
                        "content": [
                            {"type": "text", "text": '{"status": "screenshot captured"}'},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ],
                    },
                ],
            },
        )
        with patch("src.host.credentials._model_supports_vision", return_value=False):
            sanitized, _ = vault._prepare_llm_params(
                request, "groq/llama-3.3-70b-versatile",
            )
        tool_msg = [m for m in sanitized if m.get("role") == "tool"][0]
        # Content must be flattened to text-only string
        assert isinstance(tool_msg["content"], str)
        assert "screenshot captured" in tool_msg["content"]
        assert "image_url" not in tool_msg["content"]

    def test_prepare_llm_params_keeps_images_for_vision_model(self, vault):
        """Tool messages with image_url blocks are preserved for vision models."""
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "openai/gpt-4o",
                "messages": [
                    {"role": "user", "content": "take a screenshot"},
                    {
                        "role": "assistant", "content": "",
                        "tool_calls": [
                            {"id": "c1", "type": "function",
                             "function": {"name": "browser_screenshot", "arguments": "{}"}},
                        ],
                    },
                    {
                        "role": "tool", "tool_call_id": "c1",
                        "content": [
                            {"type": "text", "text": '{"status": "screenshot captured"}'},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ],
                    },
                ],
            },
        )
        with patch("src.host.credentials._model_supports_vision", return_value=True):
            sanitized, _ = vault._prepare_llm_params(
                request, "openai/gpt-4o",
            )
        tool_msg = [m for m in sanitized if m.get("role") == "tool"][0]
        # Content must remain as multimodal list
        assert isinstance(tool_msg["content"], list)
        assert len(tool_msg["content"]) == 2


# ── Ollama thinking disable tests ─────────────────────────────


class TestOllamaThinkingDisable:
    """Verify reasoning_effort='none' is injected for Ollama + tools."""

    def test_ollama_with_tools_gets_reasoning_effort_none(self, vault):
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "ollama/qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "exec"}}],
            },
        )
        _, extra = vault._prepare_llm_params(request, "ollama/qwen3")
        assert extra["reasoning_effort"] == "none"

    def test_ollama_without_tools_no_reasoning_effort(self, vault):
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "ollama/qwen3",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        _, extra = vault._prepare_llm_params(request, "ollama/qwen3")
        assert "reasoning_effort" not in extra

    def test_non_ollama_with_tools_no_reasoning_effort(self, vault):
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "openai/gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "exec"}}],
            },
        )
        _, extra = vault._prepare_llm_params(request, "openai/gpt-4o")
        assert "reasoning_effort" not in extra

    def test_ollama_explicit_reasoning_effort_not_overridden(self, vault):
        request = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "ollama/qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "exec"}}],
                "reasoning_effort": "high",
            },
        )
        _, extra = vault._prepare_llm_params(request, "ollama/qwen3")
        assert extra["reasoning_effort"] == "high"


# ── OAuth async integration tests ─────────────────────────────


@pytest.mark.asyncio
async def test_oauth_chat_success(monkeypatch):
    """_oauth_chat collects streamed SSE into a final response."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    # Mock the streaming path (OAuth always streams, like pi-ai)
    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'type': 'text_delta', 'content': 'Hello!'})}\n\n"
        done = {
            'type': 'done', 'content': 'Hello!', 'tokens_used': 15,
            'model': model, 'tool_calls': [],
        }
        yield f"data: {_json.dumps(done)}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )
    result = await v._oauth_chat(req, v.system_credentials["anthropic_api_key"], "anthropic/claude-sonnet-4-6")
    assert result.success
    assert result.data["content"] == "Hello!"
    assert result.data["oauth"] is True


@pytest.mark.asyncio
async def test_oauth_chat_error(monkeypatch):
    """_oauth_chat propagates errors from the streaming path."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'error': 'Anthropic API connection error: Connection refused'})}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    with pytest.raises(RuntimeError, match="connection error"):
        await v._oauth_chat(req, "sk-ant-oat01-" + "x" * 80, "anthropic/claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_oauth_chat_401(monkeypatch):
    """_oauth_chat propagates 401 errors from streaming path."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'error': 'Anthropic OAuth failed: Unauthorized (token may be expired)'})}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    with pytest.raises(RuntimeError, match="Anthropic OAuth"):
        await v._oauth_chat(req, "sk-ant-oat01-" + "x" * 80, "anthropic/claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_oauth_chat_401_expired_token(monkeypatch):
    """_oauth_chat raises token-may-have-expired message on 401."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'error': 'OAuth auth failed (token may have expired): disabled'})}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    with pytest.raises(RuntimeError, match="token may have expired"):
        await v._oauth_chat(req, "sk-ant-oat01-" + "x" * 80, "anthropic/claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_oauth_chat_500_propagates(monkeypatch):
    """_oauth_chat propagates 500 errors from streaming path."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'error': 'Anthropic API error (HTTP 500): Internal Server Error'})}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await v._oauth_chat(req, "sk-ant-oat01-" + "x" * 80, "anthropic/claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_handle_llm_routes_oauth_to_direct_path(monkeypatch):
    """_handle_llm routes OAuth tokens to _oauth_chat, not LiteLLM."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "test"}
    v._oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    result = await v._handle_llm(req)
    v._oauth_chat.assert_called_once()
    assert result.success


# ── OAuth cost-tracking bypass tests ───────────────────────────


@pytest.mark.asyncio
async def test_oauth_skips_cost_tracking(monkeypatch):
    """OAuth calls via execute_api_call must NOT record costs."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    cost_tracker = MagicMock()
    cost_tracker.preflight_check.return_value = {"allowed": True}
    v = CredentialVault(cost_tracker=cost_tracker)

    # Mock _oauth_chat directly (it delegates to streaming internally)
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "Hello!", "tokens_used": 75, "oauth": True}
    v._oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    result = await v.execute_api_call(req, agent_id="test-agent")

    assert result.success
    cost_tracker.track.assert_not_called()
    cost_tracker.preflight_check.assert_not_called()


@pytest.mark.asyncio
async def test_oauth_skips_budget_lock(monkeypatch):
    """OAuth calls must not acquire the per-agent budget lock."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    cost_tracker = MagicMock()
    v = CredentialVault(cost_tracker=cost_tracker)

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "ok", "tokens_used": 8, "oauth": True}
    v._oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    result = await v.execute_api_call(req, agent_id="test-agent")
    assert result.success
    assert "test-agent" not in v._budget_locks


@pytest.mark.asyncio
async def test_oauth_response_has_oauth_flag(monkeypatch):
    """_oauth_chat response data includes oauth=True flag."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    v = CredentialVault()

    async def mock_stream(request, api_key, model):
        done = {'type': 'done', 'content': 'test', 'tokens_used': 8, 'model': model, 'tool_calls': []}
        yield f"data: {_json.dumps(done)}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    result = await v._oauth_chat(req, v.system_credentials["anthropic_api_key"], "anthropic/claude-sonnet-4-6")
    assert result.data["oauth"] is True


@pytest.mark.asyncio
async def test_oauth_stream_skips_cost_tracking(monkeypatch):
    """Streaming OAuth calls via stream_llm must NOT record costs."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    cost_tracker = MagicMock()
    v = CredentialVault(cost_tracker=cost_tracker)

    # Mock _oauth_chat_stream to yield a simple done event
    async def mock_stream(request, api_key, model):
        yield f"data: {_json.dumps({'type': 'text_delta', 'content': 'hi'})}\n\n"
        done = {'type': 'done', 'content': 'hi', 'tokens_used': 100, 'model': model, 'tool_calls': []}
        yield f"data: {_json.dumps(done)}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    events = []
    async for event in v.stream_llm(req, agent_id="test-agent"):
        events.append(event)

    assert any("done" in e for e in events)
    cost_tracker.track.assert_not_called()
    cost_tracker.preflight_check.assert_not_called()


@pytest.mark.asyncio
async def test_oauth_stream_skips_preflight_even_when_over_budget(monkeypatch):
    """OAuth streaming bypasses budget enforcement even when budget is exceeded."""
    import json as _json

    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-ant-oat01-" + "x" * 80)
    cost_tracker = MagicMock()
    cost_tracker.preflight_check.return_value = {"allowed": False}
    v = CredentialVault(cost_tracker=cost_tracker)

    async def mock_stream(request, api_key, model):
        done = {'type': 'done', 'content': 'ok', 'tokens_used': 10, 'model': model, 'tool_calls': []}
        yield f"data: {_json.dumps(done)}\n\n"

    v._oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    events = []
    async for event in v.stream_llm(req, agent_id="test-agent"):
        events.append(event)

    assert any("done" in e for e in events)
    assert not any("Budget exceeded" in e for e in events)
    cost_tracker.preflight_check.assert_not_called()


@pytest.mark.asyncio
async def test_regular_key_still_tracks_costs(monkeypatch):
    """Non-OAuth calls must still track costs normally (regression guard)."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-regular-key")
    cost_tracker = MagicMock()
    cost_tracker.preflight_check.return_value = {
        "allowed": True, "estimated_cost": 0.01,
        "daily_used": 0, "daily_limit": 10,
        "monthly_used": 0, "monthly_limit": 200,
    }
    v = CredentialVault(cost_tracker=cost_tracker)

    async def mock_acompletion(model, messages, api_key, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 100
        resp.usage.prompt_tokens = 60
        resp.usage.completion_tokens = 40
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        result = await v.execute_api_call(req, agent_id="test-agent")

    assert result.success
    cost_tracker.preflight_check.assert_called_once()
    cost_tracker.track.assert_called_once()
    call_args = cost_tracker.track.call_args[0]
    assert call_args[0] == "test-agent"
    assert call_args[1] == "openai/gpt-4o-mini"


# ── Budget lock timeout returns error ──────────────────────────


@pytest.mark.asyncio
async def test_budget_lock_timeout_returns_error(monkeypatch):
    """Budget lock timeout returns an error instead of silently retrying."""
    import asyncio

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    v = CredentialVault()
    cost_tracker = MagicMock()
    cost_tracker.check_budget.return_value = {
        "allowed": True, "daily_used": 0, "daily_limit": 10,
        "monthly_used": 0, "monthly_limit": 200, "estimated_cost": 0.01,
    }
    v.cost_tracker = cost_tracker

    # Simulate the lock timeout by patching asyncio.wait_for to raise TimeoutError
    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )

    lock = asyncio.Lock()
    v._budget_locks = {"agent1": lock}

    # Patch wait_for to immediately raise TimeoutError
    with patch("src.host.credentials.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        result = await v.execute_api_call(req, agent_id="agent1")

    assert result.success is False
    assert "Budget lock contention" in result.error


# ── OpenAI Codex Responses API tests ─────────────────────────


class TestOpenAICodexHelpers:
    """Tests for OpenAI Codex Responses API helper methods."""

    def test_is_oauth_token_anthropic_only(self):
        from src.host.credentials import is_oauth_token
        assert is_oauth_token("sk-ant-oat01-" + "x" * 80)
        assert not is_oauth_token("sk-oai-oat-" + "x" * 80)
        assert not is_oauth_token("sk-regular-key")

    def test_openai_oauth_headers_structure(self):
        headers = CredentialVault._openai_oauth_headers("tok-abc", "acct-123")
        assert headers["Authorization"] == "Bearer tok-abc"
        assert headers["chatgpt-account-id"] == "acct-123"
        assert headers["Content-Type"] == "application/json"
        assert headers["OpenAI-Beta"] == "responses=experimental"

    def test_openai_oauth_headers_no_account_id(self):
        headers = CredentialVault._openai_oauth_headers("tok-abc", "")
        assert "chatgpt-account-id" not in headers

    def test_has_openai_oauth_false(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        v = CredentialVault()
        assert v._has_openai_oauth() is False

    def test_has_openai_oauth_true(self, monkeypatch):
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_OPENAI_OAUTH",
            '{"access_token":"tok","refresh_token":"ref"}',
        )
        v = CredentialVault()
        assert v._has_openai_oauth() is True

    def test_jwt_expiry_extraction(self):
        import base64
        import json
        # Build a fake JWT with exp claim
        header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
        payload_data = {"exp": 1700000000, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-abc"}}
        payload = base64.urlsafe_b64encode(json.dumps(payload_data).encode()).rstrip(b"=").decode()
        token = f"{header}.{payload}.sig"
        assert CredentialVault._extract_jwt_expiry(token) == 1700000000
        assert CredentialVault._extract_account_id_from_jwt(token) == "acct-abc"

    def test_jwt_expiry_invalid_token(self):
        assert CredentialVault._extract_jwt_expiry("not-a-jwt") == 0
        assert CredentialVault._extract_account_id_from_jwt("not-a-jwt") == ""

    def test_load_codex_auth_flat(self, tmp_path, monkeypatch):
        import json
        auth_dir = tmp_path / ".codex"
        auth_dir.mkdir()
        auth_file = auth_dir / "auth.json"
        auth_file.write_text(json.dumps({
            "access_token": "tok-flat",
            "refresh_token": "ref-flat",
        }))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = CredentialVault.load_codex_auth()
        assert result["access_token"] == "tok-flat"

    def test_load_codex_auth_nested(self, tmp_path, monkeypatch):
        import json
        auth_dir = tmp_path / ".codex"
        auth_dir.mkdir()
        auth_file = auth_dir / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {
                "access_token": "tok-nested",
                "refresh_token": "ref-nested",
            },
        }))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = CredentialVault.load_codex_auth()
        assert result["access_token"] == "tok-nested"

    def test_load_codex_auth_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = CredentialVault.load_codex_auth()
        assert result is None


class TestBuildOpenAIResponsesBody:
    """Tests for _build_openai_responses_body message conversion."""

    def test_basic_user_message(self):
        params = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = CredentialVault._build_openai_responses_body(params)
        assert body["model"] == "gpt-4o"
        assert body["store"] is False
        assert body["stream"] is True
        assert body["text"] == {"verbosity": "medium"}
        assert body["include"] == ["reasoning.encrypted_content"]
        # tool_choice/parallel_tool_calls only present when tools exist
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body
        assert "max_output_tokens" not in body
        assert len(body["input"]) == 1
        assert body["input"][0]["content"][0]["type"] == "input_text"
        assert body["input"][0]["content"][0]["text"] == "Hello"

    def test_system_to_instructions(self):
        params = {
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        body = CredentialVault._build_openai_responses_body(params)
        assert body["instructions"] == "You are helpful."
        assert len(body["input"]) == 1

    def test_assistant_with_tool_calls(self):
        params = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "search"},
                {
                    "role": "assistant",
                    "content": "Searching...",
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "search", "arguments": '{"q":"test"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "results"},
            ],
        }
        body = CredentialVault._build_openai_responses_body(params)
        # user (no type), assistant text (type=message), function_call, function_call_output
        assert any(i.get("type") == "function_call" for i in body["input"])
        assert any(i.get("type") == "function_call_output" for i in body["input"])
        # User message has no "type" key (pi-ai format)
        user_item = body["input"][0]
        assert "type" not in user_item
        assert user_item["role"] == "user"

    def test_tools_unwrapped(self):
        params = {
            "model": "gpt-4o",
            "messages": [],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }],
        }
        body = CredentialVault._build_openai_responses_body(params)
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "search"
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["strict"] is None
        assert body["tool_choice"] == "auto"
        assert body["parallel_tool_calls"] is True

    def test_strips_prefix(self):
        params = {"model": "openai/gpt-4o-mini", "messages": []}
        body = CredentialVault._build_openai_responses_body(params)
        assert body["model"] == "gpt-4o-mini"

    def test_bare_model(self):
        params = {"model": "gpt-4o", "messages": []}
        body = CredentialVault._build_openai_responses_body(params)
        assert body["model"] == "gpt-4o"

    def test_unsupported_params_excluded(self):
        """max_tokens, max_completion_tokens, reasoning_effort are not sent to Codex."""
        params = {
            "model": "o3", "messages": [],
            "max_completion_tokens": 2048, "max_tokens": 1024,
            "reasoning_effort": "high",
        }
        body = CredentialVault._build_openai_responses_body(params)
        assert "max_output_tokens" not in body
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        assert "reasoning_effort" not in body

    def test_multimodal_user_content(self):
        params = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }],
        }
        body = CredentialVault._build_openai_responses_body(params)
        assert len(body["input"]) == 1
        content = body["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "What is this?"}
        assert content[1] == {"type": "input_image", "detail": "auto", "image_url": "https://example.com/img.png"}


class TestParseOpenAIResponsesResponse:
    """Tests for _parse_openai_responses_response."""

    def test_text_response(self):
        data = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = CredentialVault._parse_openai_responses_response(data, "openai/gpt-4o")
        assert result["content"] == "Hello!"
        assert result["tokens_used"] == 15
        assert result["tool_calls"] == []

    def test_function_call_response(self):
        data = {
            "output": [
                {"type": "function_call", "name": "search", "arguments": '{"q":"test"}'},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        result = CredentialVault._parse_openai_responses_response(data, "openai/gpt-4o")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search"

    def test_reasoning_response(self):
        data = {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I think..."}]},
                {"type": "message", "content": [{"type": "output_text", "text": "Answer"}]},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 100},
        }
        result = CredentialVault._parse_openai_responses_response(data, "o3")
        assert result["content"] == "Answer"
        assert result["thinking_content"] == "I think..."

    def test_empty_output(self):
        data = {"output": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        result = CredentialVault._parse_openai_responses_response(data, "openai/gpt-4o")
        assert result["content"] == ""
        assert result["tool_calls"] == []

    def test_no_usage(self):
        data = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
        result = CredentialVault._parse_openai_responses_response(data, "openai/gpt-4o")
        assert result["content"] == "ok"
        assert result["tokens_used"] == 0


@pytest.mark.asyncio
async def test_codex_chat_success(monkeypatch):
    """_openai_oauth_chat collects streamed SSE into a final response."""
    import json as _json

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()

    # Mock the streaming path (Codex always streams)
    async def mock_stream(request, model):
        yield f"data: {_json.dumps({'type': 'text_delta', 'content': 'Hello!'})}\n\n"
        done = {
            'type': 'done', 'content': 'Hello!', 'tokens_used': 15,
            'model': model, 'tool_calls': [],
            'input_tokens': 10, 'output_tokens': 5,
        }
        yield f"data: {_json.dumps(done)}\n\n"

    v._openai_oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        },
    )
    result = await v._openai_oauth_chat(req, "openai/gpt-4o")
    assert result.success
    assert result.data["content"] == "Hello!"
    assert result.data["oauth"] is True


@pytest.mark.asyncio
async def test_codex_chat_error(monkeypatch):
    """_openai_oauth_chat propagates errors from the streaming path."""
    import json as _json

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()

    async def mock_stream(request, model):
        yield f"data: {_json.dumps({'error': 'OpenAI Codex API error (HTTP 500)'})}\n\n"

    v._openai_oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await v._openai_oauth_chat(req, "openai/gpt-4o")


@pytest.mark.asyncio
async def test_codex_chat_connect_error(monkeypatch):
    """_openai_oauth_chat raises RuntimeError on ConnectError."""
    import json as _json

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()

    async def mock_stream(request, model):
        yield f"data: {_json.dumps({'error': 'connection error: Connection refused'})}\n\n"

    v._openai_oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    with pytest.raises(RuntimeError, match="connection error"):
        await v._openai_oauth_chat(req, "openai/gpt-4o")


@pytest.mark.asyncio
async def test_codex_routing_bare_model(monkeypatch):
    """Bare model name 'gpt-4o' routes through Codex path when no API key."""
    from unittest.mock import AsyncMock

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "test"}
    v._openai_oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    result = await v._handle_llm(req)
    v._openai_oauth_chat.assert_called_once()
    assert result.success


@pytest.mark.asyncio
async def test_codex_routing_prefixed_model(monkeypatch):
    """Prefixed model 'openai/gpt-4o' routes through Codex path when no API key."""
    from unittest.mock import AsyncMock

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "test"}
    v._openai_oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    result = await v._handle_llm(req)
    v._openai_oauth_chat.assert_called_once()
    assert result.success


@pytest.mark.asyncio
async def test_codex_skips_cost_tracking(monkeypatch):
    """Codex calls via execute_api_call must NOT record costs."""
    from unittest.mock import AsyncMock

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    cost_tracker = MagicMock()
    cost_tracker.preflight_check.return_value = {"allowed": True}
    v = CredentialVault(cost_tracker=cost_tracker)

    # Mock _openai_oauth_chat directly (it delegates to streaming internally)
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.data = {"content": "Hi", "tokens_used": 75, "oauth": True}
    v._openai_oauth_chat = AsyncMock(return_value=mock_result)

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    result = await v.execute_api_call(req, agent_id="test-agent")
    assert result.success
    cost_tracker.track.assert_not_called()
    cost_tracker.preflight_check.assert_not_called()


@pytest.mark.asyncio
async def test_codex_stream_skips_cost_tracking(monkeypatch):
    """Streaming Codex calls via stream_llm must NOT record costs."""
    import json as _json

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    cost_tracker = MagicMock()
    v = CredentialVault(cost_tracker=cost_tracker)

    async def mock_stream(request, model):
        yield f"data: {_json.dumps({'type': 'text_delta', 'content': 'hi'})}\n\n"
        done = {'type': 'done', 'content': 'hi', 'tokens_used': 100, 'model': model, 'tool_calls': []}
        yield f"data: {_json.dumps(done)}\n\n"

    v._openai_oauth_chat_stream = mock_stream

    req = APIProxyRequest(
        service="llm", action="chat",
        params={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    events = []
    async for event in v.stream_llm(req, agent_id="test-agent"):
        events.append(event)

    assert any("done" in e for e in events)
    cost_tracker.track.assert_not_called()
    cost_tracker.preflight_check.assert_not_called()


@pytest.mark.asyncio
async def test_token_refresh_fresh(monkeypatch):
    """_ensure_openai_oauth_token returns cached token when not expired."""
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"tok","refresh_token":"ref","account_id":"acct","expires_at":9999999999}',
    )
    v = CredentialVault()
    token, acct = await v._ensure_openai_oauth_token()
    assert token == "tok"
    assert acct == "acct"


@pytest.mark.asyncio
async def test_token_refresh_expired(monkeypatch):
    """_ensure_openai_oauth_token refreshes when expired."""
    import base64

    import httpx

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"old","refresh_token":"ref","account_id":"acct","expires_at":0}',
    )
    v = CredentialVault()

    # Build fake JWT for new token
    import json as _json
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_data = {"exp": 9999999999, "https://api.openai.com/auth": {"chatgpt_account_id": "new-acct"}}
    payload = base64.urlsafe_b64encode(_json.dumps(payload_data).encode()).rstrip(b"=").decode()
    new_jwt = f"{header}.{payload}.sig"

    mock_response = httpx.Response(
        200,
        json={"access_token": new_jwt, "refresh_token": "new-ref"},
        request=httpx.Request("POST", "https://auth.openai.com/oauth/token"),
    )

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.is_closed = False
    v._http_client = mock_client

    # Patch _persist_to_env to avoid writing to real .env
    import src.host.credentials as cred_mod
    original = cred_mod._persist_to_env
    cred_mod._persist_to_env = lambda *a, **kw: None
    try:
        token, acct = await v._ensure_openai_oauth_token()
        assert token == new_jwt
        assert acct == "new-acct"
    finally:
        cred_mod._persist_to_env = original


@pytest.mark.asyncio
async def test_token_refresh_failure(monkeypatch):
    """_ensure_openai_oauth_token raises on refresh failure."""
    import httpx

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv(
        "OPENLEGION_SYSTEM_OPENAI_OAUTH",
        '{"access_token":"old","refresh_token":"ref","account_id":"acct","expires_at":0}',
    )
    v = CredentialVault()

    async def mock_post(*args, **kwargs):
        return httpx.Response(
            400,
            text="invalid_grant",
            request=httpx.Request("POST", "https://auth.openai.com/oauth/token"),
        )

    mock_client = MagicMock()
    mock_client.post = mock_post
    mock_client.is_closed = False
    v._http_client = mock_client

    with pytest.raises(RuntimeError, match="token refresh failed"):
        await v._ensure_openai_oauth_token()


@pytest.mark.asyncio
async def test_token_refresh_no_creds(monkeypatch):
    """_ensure_openai_oauth_token raises when no creds configured."""
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    v = CredentialVault()
    with pytest.raises(RuntimeError, match="No OpenAI OAuth credentials"):
        await v._ensure_openai_oauth_token()


# ── normalize_openai_oauth tests ─────────────────────────────────


def test_normalize_openai_oauth_flat():
    """Flat format with access_token at top level passes through."""
    data = {"access_token": "tok", "refresh_token": "ref", "account_id": "acct"}
    result = CredentialVault.normalize_openai_oauth(data)
    assert result is data  # same object — no transformation needed


def test_normalize_openai_oauth_nested():
    """Nested Codex CLI format is flattened correctly."""
    data = {
        "tokens": {
            "access_token": "tok",
            "refresh_token": "ref",
            "account_id": "acct-123",
        },
        "last_refresh": "2025-01-01T00:00:00Z",
    }
    result = CredentialVault.normalize_openai_oauth(data)
    assert result == {
        "access_token": "tok",
        "refresh_token": "ref",
        "account_id": "acct-123",
    }


def test_normalize_openai_oauth_nested_no_account_id():
    """Nested format without account_id omits it."""
    data = {
        "tokens": {"access_token": "tok", "refresh_token": "ref"},
    }
    result = CredentialVault.normalize_openai_oauth(data)
    assert result == {"access_token": "tok", "refresh_token": "ref"}
    assert "account_id" not in result


def test_normalize_openai_oauth_nested_no_refresh():
    """Nested format without refresh_token defaults to empty string."""
    data = {"tokens": {"access_token": "tok"}}
    result = CredentialVault.normalize_openai_oauth(data)
    assert result["access_token"] == "tok"
    assert result["refresh_token"] == ""


def test_normalize_openai_oauth_missing_access_token():
    """Dict without access_token returns None."""
    assert CredentialVault.normalize_openai_oauth({"refresh_token": "ref"}) is None
    assert CredentialVault.normalize_openai_oauth({}) is None


def test_normalize_openai_oauth_nested_empty_tokens():
    """Nested format with empty tokens dict returns None."""
    assert CredentialVault.normalize_openai_oauth({"tokens": {}}) is None
    assert CredentialVault.normalize_openai_oauth({"tokens": "not-a-dict"}) is None


# ── Anthropic structured OAuth tests ──────────────────────────


class TestAnthropicOAuth:
    """Tests for structured Anthropic OAuth credential support."""

    def test_has_anthropic_oauth_false(self):
        """_has_anthropic_oauth returns False when no credentials loaded."""
        v = CredentialVault()
        assert v._has_anthropic_oauth() is False

    def test_has_anthropic_oauth_true(self, monkeypatch):
        """_has_anthropic_oauth returns True when credentials are loaded."""
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-test","refresh_token":"ref","expires_at":9999999999}',
        )
        v = CredentialVault()
        assert v._has_anthropic_oauth() is True

    def test_store_anthropic_oauth(self):
        """store_anthropic_oauth persists credentials in memory and .env."""
        v = CredentialVault()
        import src.host.credentials as cred_mod
        original = cred_mod._persist_to_env
        persisted = {}
        cred_mod._persist_to_env = lambda k, val, **kw: persisted.update({k: val})
        try:
            creds = {
                "access_token": "sk-ant-oat01-abc",
                "refresh_token": "ref-tok",
                "expires_at": 1234567890,
            }
            v.store_anthropic_oauth(creds)
            assert v._has_anthropic_oauth()
            assert v._anthropic_oauth["access_token"] == "sk-ant-oat01-abc"
            assert "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH" in persisted
        finally:
            cred_mod._persist_to_env = original

    def test_load_claude_cli_auth_valid(self, tmp_path, monkeypatch):
        """load_claude_cli_auth reads and normalizes Claude CLI credentials."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        import json
        creds_file.write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-test123",
                "refreshToken": "ref-tok-456",
                "expiresAt": 1234567890,
            }
        }))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        result = CredentialVault.load_claude_cli_auth()
        assert result is not None
        assert result["access_token"] == "sk-ant-oat01-test123"
        assert result["refresh_token"] == "ref-tok-456"
        assert result["expires_at"] == 1234567890

    def test_load_claude_cli_auth_missing_file(self, tmp_path, monkeypatch):
        """load_claude_cli_auth returns None when file doesn't exist."""
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert CredentialVault.load_claude_cli_auth() is None

    def test_load_claude_cli_auth_missing_field(self, tmp_path, monkeypatch):
        """load_claude_cli_auth returns None when claudeAiOauth is absent."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        import json
        creds_file.write_text(json.dumps({"someOtherField": "value"}))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert CredentialVault.load_claude_cli_auth() is None

    def test_load_claude_cli_auth_missing_access_token(self, tmp_path, monkeypatch):
        """load_claude_cli_auth returns None when accessToken is missing."""
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir()
        creds_file = claude_dir / ".credentials.json"
        import json
        creds_file.write_text(json.dumps({
            "claudeAiOauth": {"refreshToken": "ref", "expiresAt": 123}
        }))
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert CredentialVault.load_claude_cli_auth() is None

    @pytest.mark.asyncio
    async def test_ensure_anthropic_oauth_token_fresh(self):
        """_ensure_anthropic_oauth_token returns token when not expired."""
        v = CredentialVault()
        import time
        v._anthropic_oauth = {
            "access_token": "sk-ant-oat01-fresh",
            "expires_at": int(time.time()) + 3600,
        }
        token = await v._ensure_anthropic_oauth_token()
        assert token == "sk-ant-oat01-fresh"

    @pytest.mark.asyncio
    async def test_ensure_anthropic_oauth_token_no_expiry(self):
        """_ensure_anthropic_oauth_token returns token when no expires_at set."""
        v = CredentialVault()
        v._anthropic_oauth = {
            "access_token": "sk-ant-oat01-noexp",
        }
        token = await v._ensure_anthropic_oauth_token()
        assert token == "sk-ant-oat01-noexp"

    @pytest.mark.asyncio
    async def test_ensure_anthropic_oauth_token_expired(self):
        """_ensure_anthropic_oauth_token raises on expired token."""
        v = CredentialVault()
        v._anthropic_oauth = {
            "access_token": "sk-ant-oat01-expired",
            "expires_at": 1000,  # long past
        }
        with pytest.raises(RuntimeError, match="expired"):
            await v._ensure_anthropic_oauth_token()

    @pytest.mark.asyncio
    async def test_ensure_anthropic_oauth_token_none(self):
        """_ensure_anthropic_oauth_token raises when no credentials."""
        v = CredentialVault()
        with pytest.raises(RuntimeError, match="No Anthropic OAuth"):
            await v._ensure_anthropic_oauth_token()

    def test_providers_with_credentials_includes_anthropic_oauth(self, monkeypatch):
        """Anthropic OAuth counts as having Anthropic credentials."""
        for p in ("anthropic", "openai", "gemini", "deepseek",
                   "moonshot", "minimax", "xai", "groq", "zai"):
            monkeypatch.delenv(f"OPENLEGION_SYSTEM_{p.upper()}_API_KEY", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OLLAMA_API_BASE", raising=False)
        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-test","expires_at":9999999999}',
        )
        v = CredentialVault()
        providers = v.get_providers_with_credentials()
        assert "anthropic" in providers

    def test_anthropic_oauth_not_in_system_credentials(self, monkeypatch):
        """OPENLEGION_SYSTEM_ANTHROPIC_OAUTH should not appear in system_credentials."""
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-test"}',
        )
        v = CredentialVault()
        assert "anthropic_oauth" not in v.system_credentials
        assert v._has_anthropic_oauth()

    @pytest.mark.asyncio
    async def test_handle_llm_routes_anthropic_oauth(self, monkeypatch):
        """_handle_llm routes through structured OAuth when no regular API key."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-testabc"}',
        )
        v = CredentialVault()

        # Mock _oauth_chat to verify it's called with the right token
        oauth_called = {}

        async def mock_oauth_chat(request, api_key, model):
            oauth_called["api_key"] = api_key
            oauth_called["model"] = model
            return APIProxyResponse(
                success=True,
                data={"content": "oauth reply", "tokens_used": 10, "model": model, "tool_calls": []},
            )

        monkeypatch.setattr(v, "_oauth_chat", mock_oauth_chat)

        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = await v.execute_api_call(req)

        assert result.success
        assert oauth_called["api_key"] == "sk-ant-oat01-testabc"
        assert oauth_called["model"] == "anthropic/claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_stream_llm_routes_anthropic_oauth(self, monkeypatch):
        """stream_llm routes through OAuth streaming for structured Anthropic OAuth."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-streamtest"}',
        )
        v = CredentialVault()

        stream_called = {}

        async def mock_oauth_chat_stream(request, api_key, model):
            stream_called["api_key"] = api_key
            import json as _json
            done = {"type": "done", "content": "streamed", "tool_calls": [],
                    "tokens_used": 5, "model": model}
            yield f"data: {_json.dumps(done)}\n\n"

        monkeypatch.setattr(v, "_oauth_chat_stream", mock_oauth_chat_stream)

        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

        assert stream_called["api_key"] == "sk-ant-oat01-streamtest"
        assert any("done" in e for e in events)

    @pytest.mark.asyncio
    async def test_oauth_budget_skip_anthropic_oauth(self, monkeypatch):
        """Budget enforcement is skipped for Anthropic structured OAuth."""
        monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv(
            "OPENLEGION_SYSTEM_ANTHROPIC_OAUTH",
            '{"access_token":"sk-ant-oat01-budget"}',
        )
        cost_tracker = MagicMock()
        v = CredentialVault(cost_tracker=cost_tracker)

        async def mock_oauth_chat(request, api_key, model):
            return APIProxyResponse(
                success=True,
                data={"content": "ok", "tokens_used": 10, "model": model, "tool_calls": [], "oauth": True},
            )

        monkeypatch.setattr(v, "_oauth_chat", mock_oauth_chat)

        req = APIProxyRequest(
            service="llm", action="chat",
            params={"model": "anthropic/claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
        )
        result = await v.execute_api_call(req, agent_id="test-agent")

        assert result.success
        # Budget should NOT have been checked (OAuth = subscription-based)
        cost_tracker.preflight_check.assert_not_called()
        cost_tracker.check_budget.assert_not_called()


# ── Custom provider model rewrite tests ────────────────────────


def test_rewrite_known_provider_with_api_base(monkeypatch):
    """Known provider with api_base → no rewrite."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    v = CredentialVault()
    result = v._rewrite_model_for_litellm("openai/gpt-4o", "https://gateway.example.com/v1")
    assert result == "openai/gpt-4o"


def test_rewrite_unknown_provider_with_api_base():
    """Unknown provider with api_base → rewrite to openai/ prefix."""
    v = CredentialVault()
    result = v._rewrite_model_for_litellm(
        "chatrhino/Chatrhino-750B", "http://api.customprovider.com/api/v1",
    )
    assert result == "openai/Chatrhino-750B"


def test_rewrite_unknown_provider_without_api_base():
    """Unknown provider without api_base → no rewrite."""
    v = CredentialVault()
    result = v._rewrite_model_for_litellm("chatrhino/Chatrhino-750B", None)
    assert result == "chatrhino/Chatrhino-750B"


def test_rewrite_no_slash_with_api_base():
    """Bare model name with api_base → openai/{model}."""
    v = CredentialVault()
    result = v._rewrite_model_for_litellm("mymodel", "http://localhost:8000/v1")
    assert result == "openai/mymodel"


def test_rewrite_no_api_base():
    """No api_base → always returns model unchanged."""
    v = CredentialVault()
    assert v._rewrite_model_for_litellm("chatrhino/model", None) == "chatrhino/model"
    assert v._rewrite_model_for_litellm("openai/gpt-4o", None) == "openai/gpt-4o"
    assert v._rewrite_model_for_litellm("mymodel", None) == "mymodel"


def test_rewrite_empty_api_base():
    """Empty string api_base → treated as no api_base."""
    v = CredentialVault()
    assert v._rewrite_model_for_litellm("chatrhino/model", "") == "chatrhino/model"


async def test_chat_uses_rewritten_model_for_custom_provider(monkeypatch):
    """Non-streaming chat rewrites custom provider model for litellm."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_KEY", "sk-rhino")
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_BASE", "http://api.chatrhino.com/v1")
    v = CredentialVault()

    captured = {}

    async def mock_acompletion(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "hello"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = 10
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 5
        return resp

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "chatrhino/Chatrhino-750B",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        result = await v.execute_api_call(req)

    assert result.success
    # litellm should receive the rewritten model
    assert captured["model"] == "openai/Chatrhino-750B"
    # Response data should use original model for cost tracking
    assert result.data["model"] == "chatrhino/Chatrhino-750B"


async def test_stream_uses_rewritten_model_for_custom_provider(monkeypatch):
    """Streaming chat rewrites custom provider model for litellm."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_KEY", "sk-rhino")
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_BASE", "http://api.chatrhino.com/v1")
    v = CredentialVault()

    captured = {}

    async def mock_chunk_generator():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "streamed"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    async def mock_acompletion(**kwargs):
        captured.update(kwargs)
        return mock_chunk_generator()

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        req = APIProxyRequest(
            service="llm", action="chat",
            params={
                "model": "chatrhino/Chatrhino-750B",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        events = []
        async for event in v.stream_llm(req):
            events.append(event)

    # litellm should receive the rewritten model
    assert captured["model"] == "openai/Chatrhino-750B"
    assert any("streamed" in e for e in events)


async def test_embed_uses_rewritten_model_for_custom_provider(monkeypatch):
    """Embedding rewrites custom provider model for litellm."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_KEY", "sk-rhino")
    monkeypatch.setenv("OPENLEGION_SYSTEM_CHATRHINO_API_BASE", "http://api.chatrhino.com/v1")
    v = CredentialVault()

    captured = {}

    async def mock_aembedding(**kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.data = [{"embedding": [0.1, 0.2, 0.3]}]
        return resp

    with patch("litellm.aembedding", side_effect=mock_aembedding):
        req = APIProxyRequest(
            service="llm", action="embed",
            params={
                "model": "chatrhino/Chatrhino-750B",
                "text": "hello world",
            },
        )
        result = await v.execute_api_call(req)

    assert result.success
    assert captured["model"] == "openai/Chatrhino-750B"


class TestDiscoverOpenlegionModels:
    """Tests for CredentialVault.discover_openlegion_models()."""

    @pytest.fixture
    def vault_with_openlegion(self):
        """Vault with openlegion api_base and api_key configured."""
        vault = CredentialVault()
        vault.system_credentials["openlegion_api_key"] = "test-key"
        vault.api_bases["openlegion_api_base"] = "https://gw.example.com/v1"
        return vault

    @pytest.mark.asyncio
    async def test_returns_prefixed_model_ids(self, vault_with_openlegion, monkeypatch):
        from unittest.mock import AsyncMock

        import httpx
        gateway_response = {
            "data": [
                {"id": "openai/gpt-5.4", "pricing": {"input": "0.0000025", "output": "0.000015"}},
                {"id": "anthropic/claude-sonnet-4-6", "pricing": {"input": "0.003", "output": "0.015"}},
            ]
        }
        mock_resp = httpx.Response(200, json=gateway_response)
        client = await vault_with_openlegion._get_http_client()
        monkeypatch.setattr(client, "get", AsyncMock(return_value=mock_resp))

        models, pricing = await vault_with_openlegion.discover_openlegion_models()
        assert "openlegion/openai/gpt-5.4" in models
        assert "openlegion/anthropic/claude-sonnet-4-6" in models

    @pytest.mark.asyncio
    async def test_returns_empty_without_api_base(self):
        vault = CredentialVault()
        models, pricing = await vault.discover_openlegion_models()
        assert models == []
        assert pricing == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_http_error(self, vault_with_openlegion, monkeypatch):
        from unittest.mock import AsyncMock

        import httpx
        mock_resp = httpx.Response(500)
        client = await vault_with_openlegion._get_http_client()
        monkeypatch.setattr(client, "get", AsyncMock(return_value=mock_resp))

        models, pricing = await vault_with_openlegion.discover_openlegion_models()
        assert models == []
        assert pricing == {}

    @pytest.mark.asyncio
    async def test_filters_non_chat_models(self, vault_with_openlegion, monkeypatch):
        from unittest.mock import AsyncMock

        import httpx
        gateway_response = {
            "data": [
                {"id": "openai/gpt-5.4", "pricing": {"input": "0.0000025", "output": "0.000015"}},
                {"id": "openai/text-embedding-3-small"},
                {"id": "openai/dall-e-3"},
            ]
        }
        mock_resp = httpx.Response(200, json=gateway_response)
        client = await vault_with_openlegion._get_http_client()
        monkeypatch.setattr(client, "get", AsyncMock(return_value=mock_resp))

        models, pricing = await vault_with_openlegion.discover_openlegion_models()
        assert "openlegion/openai/gpt-5.4" in models
        assert not any("embedding" in m for m in models)
        assert not any("dall-e" in m for m in models)

    @pytest.mark.asyncio
    async def test_returns_pricing(self, vault_with_openlegion, monkeypatch):
        from unittest.mock import AsyncMock

        import httpx
        gateway_response = {
            "data": [
                {"id": "openai/gpt-5.4", "pricing": {"input": "0.0000025", "output": "0.000015"}},
            ]
        }
        mock_resp = httpx.Response(200, json=gateway_response)
        client = await vault_with_openlegion._get_http_client()
        monkeypatch.setattr(client, "get", AsyncMock(return_value=mock_resp))

        models, pricing = await vault_with_openlegion.discover_openlegion_models()
        assert "openlegion/openai/gpt-5.4" in models
        assert "openai/gpt-5.4" in pricing
        inp, out = pricing["openai/gpt-5.4"]
        assert abs(inp - 0.0025) < 1e-10
        assert abs(out - 0.015) < 1e-10


class TestRewriteModelForLitellm:
    """Tests for _rewrite_model_for_litellm explicit openlegion handling."""

    @pytest.fixture
    def vault(self):
        return CredentialVault()

    def test_openlegion_openai_model(self, vault):
        result = vault._rewrite_model_for_litellm(
            "openlegion/openai/gpt-5.4", "https://gw.example.com/v1",
        )
        assert result == "openai/openai/gpt-5.4"

    def test_openlegion_anthropic_model(self, vault):
        result = vault._rewrite_model_for_litellm(
            "openlegion/anthropic/claude-sonnet-4-6", "https://gw.example.com/v1",
        )
        assert result == "openai/anthropic/claude-sonnet-4-6"

    def test_openlegion_deepseek_model(self, vault):
        result = vault._rewrite_model_for_litellm(
            "openlegion/deepseek/deepseek-chat", "https://gw.example.com/v1",
        )
        assert result == "openai/deepseek/deepseek-chat"

    def test_no_rewrite_without_api_base(self, vault):
        result = vault._rewrite_model_for_litellm(
            "openlegion/openai/gpt-5.4", None,
        )
        assert result == "openlegion/openai/gpt-5.4"

    def test_native_provider_unchanged(self, vault):
        result = vault._rewrite_model_for_litellm(
            "anthropic/claude-sonnet-4-6", "https://custom.example.com",
        )
        assert result == "anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_default_model_fallback_when_no_api_key(monkeypatch):
    """When the requested model has no API key, fall back to default_model."""
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "gem-key")
    v = CredentialVault(default_model="gemini/gemini-2.0-flash")

    call_log = []

    async def fake_call_fn(model, api_key, api_base, auth_headers):
        call_log.append(model)
        return {"content": "hello", "tool_calls": [], "tokens_used": 10}, model

    result, used_model = await v._call_llm_with_failover(
        "anthropic/claude-haiku-4-5", fake_call_fn,
    )
    assert used_model == "gemini/gemini-2.0-flash"
    assert call_log == ["gemini/gemini-2.0-flash"]


@pytest.mark.asyncio
async def test_default_model_fallback_not_used_when_primary_works(monkeypatch):
    """When the primary model has an API key, don't fall back."""
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "gem-key")
    v = CredentialVault(default_model="gemini/gemini-2.0-flash")

    call_log = []

    async def fake_call_fn(model, api_key, api_base, auth_headers):
        call_log.append(model)
        return {"content": "hello", "tool_calls": [], "tokens_used": 10}, model

    result, used_model = await v._call_llm_with_failover(
        "anthropic/claude-haiku-4-5", fake_call_fn,
    )
    assert used_model == "anthropic/claude-haiku-4-5"
    assert call_log == ["anthropic/claude-haiku-4-5"]


@pytest.mark.asyncio
async def test_auto_fallback_when_default_model_also_missing(monkeypatch):
    """When both requested and default_model lack keys, fall back to any available provider."""
    # Only gemini has a key
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_GEMINI_API_KEY", "gem-key")
    v = CredentialVault(default_model="openai/gpt-4o-mini")  # no openai key!

    call_log = []

    async def fake_call_fn(model, api_key, api_base, auth_headers):
        call_log.append(model)
        return {"content": "hello", "tool_calls": [], "tokens_used": 10}, model

    result, used_model = await v._call_llm_with_failover(
        "anthropic/claude-haiku-4-5", fake_call_fn,
    )
    # Should have fallen back to a gemini model
    assert "gemini" in used_model
    assert len(call_log) == 1
    assert "gemini" in call_log[0]
