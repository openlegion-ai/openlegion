"""Tests for per-agent proxy resolution chain."""

import os
from unittest.mock import patch

from src.cli.proxy import (
    _assemble_proxy_url,
    build_proxy_env_vars,
    parse_proxy_url,
    resolve_agent_proxy,
    sanitize_agent_id_for_env,
    validate_proxy_url,
)


class TestValidateProxyUrl:
    def test_valid_http(self):
        assert validate_proxy_url("http://proxy.example.com:8080") is True

    def test_socks5_rejected(self):
        """SOCKS5 is no longer a supported proxy scheme — httpx[socks] removed."""
        assert validate_proxy_url("socks5://user:pass@proxy.example.com:1080") is False

    def test_valid_https(self):
        assert validate_proxy_url("https://proxy.example.com:443") is True

    def test_invalid_scheme(self):
        assert validate_proxy_url("ftp://proxy.example.com:21") is False

    def test_missing_port(self):
        assert validate_proxy_url("http://proxy.example.com") is False

    def test_empty_string(self):
        assert validate_proxy_url("") is False

    def test_garbage(self):
        assert validate_proxy_url("not-a-url") is False

    def test_port_zero_rejected(self):
        """Port 0 is not a usable proxy endpoint — reject as misconfiguration."""
        assert validate_proxy_url("http://proxy.example.com:0") is False

    def test_http_and_https_are_only_valid_schemes(self):
        """Only HTTP and HTTPS are accepted proxy schemes."""
        assert validate_proxy_url("http://proxy.example.com:3128") is True
        assert validate_proxy_url("https://proxy.example.com:443") is True
        assert validate_proxy_url("socks5://proxy.example.com:1080") is False
        assert validate_proxy_url("socks4://proxy.example.com:1080") is False
        assert validate_proxy_url("socks5h://proxy.example.com:1080") is False


class TestParseProxyUrl:
    def test_full_url_with_auth(self):
        result = parse_proxy_url("http://user:p%40ss@host:8080")
        assert result == {
            "url": "http://host:8080",
            "username": "user",
            "password": "p@ss",
            "full_url": "http://user:p%40ss@host:8080",
        }

    def test_socks5_returns_none(self):
        """SOCKS5 URLs are rejected — parse returns None."""
        assert parse_proxy_url("socks5://user:p%40ss@host:1080") is None

    def test_url_without_auth(self):
        result = parse_proxy_url("http://host:8080")
        assert result == {
            "url": "http://host:8080",
            "username": "",
            "password": "",
            "full_url": "http://host:8080",
        }

    def test_invalid_url_returns_none(self):
        assert parse_proxy_url("garbage") is None


class TestResolveAgentProxy:
    """Test the proxy resolution chain: custom -> inherit -> system -> None."""

    def test_mode_custom_resolves_credential(self):
        agents_cfg = {
            "test-agent": {
                "proxy": {"mode": "custom", "credential": "agent_test-agent_proxy"}
            }
        }
        env = {"OPENLEGION_CRED_agent_test-agent_proxy": "http://u:p@host:8080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result == "http://u:p@host:8080"

    def test_mode_custom_missing_credential_returns_none(self):
        """Custom mode with missing credential must NOT fall through to system proxy."""
        agents_cfg = {
            "test-agent": {
                "proxy": {"mode": "custom", "credential": "agent_test-agent_proxy"}
            }
        }
        env = {"BROWSER_PROXY_URL": "http://system:8080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result is None

    def test_mode_direct_returns_none(self):
        agents_cfg = {
            "test-agent": {"proxy": {"mode": "direct"}}
        }
        result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result is None

    def test_mode_inherit_uses_browser_proxy_env(self):
        agents_cfg = {"test-agent": {}}
        env = {
            "BROWSER_PROXY_URL": "http://managed:8080",
            "BROWSER_PROXY_USER": "muser",
            "BROWSER_PROXY_PASS": "mpass",
        }
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result == "http://muser:mpass@managed:8080"

    def test_mode_inherit_uses_system_proxy_env(self):
        agents_cfg = {"test-agent": {}}
        env = {"OPENLEGION_SYSTEM_PROXY": "http://sys:p@host:8080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result == "http://sys:p@host:8080"

    def test_system_proxy_takes_precedence_over_browser_proxy(self):
        """User override (OPENLEGION_SYSTEM_PROXY) wins over managed (BROWSER_PROXY_*)."""
        agents_cfg = {"test-agent": {}}
        env = {
            "BROWSER_PROXY_URL": "http://managed:8080",
            "OPENLEGION_SYSTEM_PROXY": "https://self-hosted:1080",
        }
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert "self-hosted:1080" in result

    def test_no_proxy_anywhere_returns_none(self):
        agents_cfg = {"test-agent": {}}
        # Clear ALL proxy-related env vars to avoid test pollution
        clear_vars = {
            "BROWSER_PROXY_URL": "", "BROWSER_PROXY_USER": "", "BROWSER_PROXY_PASS": "",
            "OPENLEGION_SYSTEM_PROXY": "", "HTTP_PROXY": "", "HTTPS_PROXY": "",
        }
        with patch.dict(os.environ, clear_vars, clear=False):
            # Also need to ensure they're truly absent, not empty
            for k in clear_vars:
                os.environ.pop(k, None)
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result is None

    def test_malformed_custom_url_falls_through(self):
        agents_cfg = {
            "test-agent": {
                "proxy": {"mode": "custom", "credential": "agent_test-agent_proxy"}
            }
        }
        env = {"OPENLEGION_CRED_agent_test-agent_proxy": "not-a-valid-url"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("BROWSER_PROXY_URL", None)
            os.environ.pop("OPENLEGION_SYSTEM_PROXY", None)
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result is None

    def test_agent_not_in_config_uses_system(self):
        agents_cfg = {}
        env = {"BROWSER_PROXY_URL": "http://system:8080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("unknown-agent", agents_cfg, {})
        assert "system:8080" in result


class TestSanitizeAgentIdForEnv:
    def test_hyphens_replaced(self):
        assert sanitize_agent_id_for_env("sales-bot") == "sales_bot"

    def test_dots_replaced(self):
        assert sanitize_agent_id_for_env("agent.v2") == "agent_v2"

    def test_clean_id_unchanged(self):
        assert sanitize_agent_id_for_env("researcher") == "researcher"

    def test_multiple_special_chars(self):
        assert sanitize_agent_id_for_env("my-agent.v2-beta") == "my_agent_v2_beta"


class TestBuildProxyEnvVars:
    def test_with_proxy_returns_env_dict(self):
        result = build_proxy_env_vars("http://proxy:8080")
        assert result == {
            "HTTP_PROXY": "http://proxy:8080",
            "HTTPS_PROXY": "http://proxy:8080",
            "NO_PROXY": "host.docker.internal,127.0.0.1,localhost",
        }

    def test_none_proxy_returns_empty_dict(self):
        assert build_proxy_env_vars(None) == {}

    def test_empty_string_proxy_returns_empty_dict(self):
        assert build_proxy_env_vars("") == {}

    def test_custom_no_proxy_appended(self):
        result = build_proxy_env_vars("http://proxy:8080", no_proxy_user="10.0.0.0/8,myhost")
        assert result["NO_PROXY"] == "host.docker.internal,127.0.0.1,localhost,10.0.0.0/8,myhost"

    def test_mandatory_no_proxy_always_present(self):
        result = build_proxy_env_vars("https://proxy:1080")
        assert "host.docker.internal" in result["NO_PROXY"]
        assert "127.0.0.1" in result["NO_PROXY"]
        assert "localhost" in result["NO_PROXY"]

    def test_https_proxy_url_preserved(self):
        result = build_proxy_env_vars("https://user:pass@proxy:1080")
        assert result["HTTP_PROXY"] == "https://user:pass@proxy:1080"
        assert result["HTTPS_PROXY"] == "https://user:pass@proxy:1080"


class TestAssembleProxyUrl:
    def test_no_credentials(self):
        assert _assemble_proxy_url("http://proxy:8080") == "http://proxy:8080"

    def test_username_only(self):
        result = _assemble_proxy_url("http://proxy:8080", username="user")
        assert result == "http://user@proxy:8080"

    def test_username_and_password(self):
        result = _assemble_proxy_url("http://proxy:8080", username="user", password="pass")
        assert result == "http://user:pass@proxy:8080"

    def test_special_chars_encoded(self):
        result = _assemble_proxy_url("http://proxy:8080", username="user@domain", password="p@ss:word")
        assert "user%40domain" in result
        assert "p%40ss%3Aword" in result
        # Verify it roundtrips through validation
        assert validate_proxy_url(result) is True

    def test_https_with_credentials(self):
        result = _assemble_proxy_url("https://proxy:1080", username="u", password="p")
        assert result == "https://u:p@proxy:1080"
