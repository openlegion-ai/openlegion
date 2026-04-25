"""Phase 3 §6.4 / §6.6 — UA guard + NetworkInformation per-agent fingerprint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ── §6.4 Firefox UA tripwire ──────────────────────────────────────────────────


class TestFirefoxUAGuard:
    def test_firefox_ua_passes(self):
        from src.browser.stealth import _assert_firefox_ua

        _assert_firefox_ua("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) "
                           "Gecko/20100101 Firefox/138.0")  # no raise

    def test_chromium_ua_raises(self):
        """Sec-CH-UA-* would leak inconsistency; refuse the UA early."""
        from src.browser.stealth import _assert_firefox_ua

        with pytest.raises(ValueError, match="Sec-CH-UA"):
            _assert_firefox_ua(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )

    def test_empty_ua_is_no_op(self):
        """We never want to fail-fast on a missing UA — it's a normal
        path (no override requested)."""
        from src.browser.stealth import _assert_firefox_ua

        _assert_firefox_ua("")  # no raise
        _assert_firefox_ua(None)  # type: ignore — also no raise

    def test_build_launch_options_with_chromium_uaversion_would_not_raise(
        self, monkeypatch,
    ):
        """``BROWSER_UA_VERSION`` only takes a version string; the OS
        template wraps it as Firefox. There's no in-band way for an
        operator to inject a Chromium UA via the env var. This test
        documents that fact — the guard fires only if a future code
        change introduces a non-Firefox template path."""
        from src.browser.stealth import build_launch_options

        monkeypatch.setenv("BROWSER_UA_VERSION", "138.0")
        opts = build_launch_options("agent-1", "/tmp/profile")
        ua = opts["config"].get("navigator.userAgent")
        assert ua and "Firefox/138.0" in ua


# ── §6.6 NetworkInformation per-agent stable values ──────────────────────────


class TestNetworkInformation:
    def test_pick_network_info_is_deterministic(self):
        from src.browser.stealth import pick_network_info

        a = pick_network_info("agent-stable")
        b = pick_network_info("agent-stable")
        assert a == b

    def test_pick_network_info_different_agents_differ(self):
        from src.browser.stealth import pick_network_info

        picks = {
            tuple(sorted(pick_network_info(f"a{i}").items()))
            for i in range(50)
        }
        # Across 50 agents we expect a healthy spread of (downlink, rtt)
        # combinations — not a single universal value.
        assert len(picks) >= 30

    def test_pick_network_info_in_band(self):
        """Stay within plausible desktop broadband ranges."""
        from src.browser.stealth import pick_network_info

        for i in range(200):
            ni = pick_network_info(f"agent-{i}")
            assert ni["effectiveType"] == "4g"
            assert 5.0 <= ni["downlink"] <= 20.0
            assert 20 <= ni["rtt"] <= 120
            assert ni["saveData"] is False

    def test_build_launch_options_writes_navigator_connection(self, monkeypatch):
        from src.browser.stealth import build_launch_options, pick_network_info

        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-x", "/tmp/profile")
        cfg = opts.get("config") or {}
        expected = pick_network_info("agent-x")
        assert cfg["navigator.connection.effectiveType"] == expected["effectiveType"]
        assert cfg["navigator.connection.downlink"] == expected["downlink"]
        assert cfg["navigator.connection.rtt"] == expected["rtt"]
        assert cfg["navigator.connection.saveData"] is False
        assert opts["i_know_what_im_doing"] is True

    def test_ua_override_does_not_clobber_netinfo(self, monkeypatch):
        """Setting BROWSER_UA_VERSION used to overwrite the whole config
        dict — must not regress now that we share the dict with §6.6
        keys."""
        from src.browser.stealth import build_launch_options, pick_network_info

        monkeypatch.setenv("BROWSER_UA_VERSION", "138.0")
        opts = build_launch_options("agent-y", "/tmp/profile")
        cfg = opts["config"]
        expected = pick_network_info("agent-y")
        # Both kinds of override coexist
        assert cfg["navigator.userAgent"].endswith("Firefox/138.0")
        assert cfg["navigator.connection.downlink"] == expected["downlink"]
