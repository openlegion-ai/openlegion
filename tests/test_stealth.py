"""Tests for browser stealth/fingerprint configuration."""

from unittest.mock import patch

from src.browser.stealth import build_launch_options, get_proxy_config, _PROXY_NOT_SET


class TestBuildLaunchOptionsProxy:
    def test_no_proxy_param_no_env_gives_no_proxy(self):
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-1", "/tmp/profile")
        assert "proxy" not in opts
        assert opts.get("geoip") is not True

    def test_proxy_param_dict_is_used(self):
        proxy = {"server": "socks5://host:1080", "username": "u", "password": "p"}
        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-1", "/tmp/profile", proxy=proxy)
        assert opts["proxy"] == proxy
        assert opts["geoip"] is True

    def test_proxy_param_overrides_env(self):
        proxy = {"server": "socks5://host:1080", "username": "u", "password": "p"}
        with patch.dict("os.environ", {"BROWSER_PROXY_URL": "http://env:8080"}, clear=False):
            opts = build_launch_options("agent-1", "/tmp/profile", proxy=proxy)
        assert opts["proxy"] == proxy
        assert opts["geoip"] is True

    def test_proxy_none_means_no_proxy(self):
        """proxy=None explicitly means no proxy (direct mode)."""
        with patch.dict("os.environ", {"BROWSER_PROXY_URL": "http://env:8080"}, clear=False):
            opts = build_launch_options("agent-1", "/tmp/profile", proxy=None)
        assert "proxy" not in opts
        assert opts.get("geoip") is not True

    def test_proxy_not_passed_falls_back_to_env(self):
        """When proxy kwarg is not passed at all, legacy env var behavior."""
        with patch.dict("os.environ", {"BROWSER_PROXY_URL": "http://env:8080"}, clear=False):
            opts = build_launch_options("agent-1", "/tmp/profile")
        assert opts.get("proxy") is not None
        assert opts["proxy"]["server"] == "http://env:8080"
        assert opts["geoip"] is True

    def test_sentinel_is_not_none(self):
        """_PROXY_NOT_SET sentinel is distinct from None."""
        assert _PROXY_NOT_SET is not None
