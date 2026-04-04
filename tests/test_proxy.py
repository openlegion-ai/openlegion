"""Tests for per-agent proxy resolution chain."""

import os
from unittest.mock import patch

import pytest

from src.cli.proxy import resolve_agent_proxy, parse_proxy_url, validate_proxy_url, sanitize_agent_id_for_env


class TestValidateProxyUrl:
    def test_valid_http(self):
        assert validate_proxy_url("http://proxy.example.com:8080") is True

    def test_valid_socks5(self):
        assert validate_proxy_url("socks5://user:pass@proxy.example.com:1080") is True

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


class TestParseProxyUrl:
    def test_full_url_with_auth(self):
        result = parse_proxy_url("socks5://user:p%40ss@host:1080")
        assert result == {
            "url": "socks5://host:1080",
            "username": "user",
            "password": "p@ss",
            "full_url": "socks5://user:p%40ss@host:1080",
        }

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
        env = {"OPENLEGION_CRED_agent_test-agent_proxy": "socks5://u:p@host:1080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result == "socks5://u:p@host:1080"

    def test_mode_custom_missing_credential_falls_through_to_system(self):
        agents_cfg = {
            "test-agent": {
                "proxy": {"mode": "custom", "credential": "agent_test-agent_proxy"}
            }
        }
        env = {"BROWSER_PROXY_URL": "http://system:8080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result is not None
        assert "system:8080" in result

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
        env = {"OPENLEGION_SYSTEM_PROXY": "socks5://sys:p@host:1080"}
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert result == "socks5://sys:p@host:1080"

    def test_browser_proxy_takes_precedence_over_system_proxy(self):
        agents_cfg = {"test-agent": {}}
        env = {
            "BROWSER_PROXY_URL": "http://managed:8080",
            "OPENLEGION_SYSTEM_PROXY": "socks5://self-hosted:1080",
        }
        with patch.dict(os.environ, env, clear=False):
            result = resolve_agent_proxy("test-agent", agents_cfg, {})
        assert "managed:8080" in result

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
