"""Phase 3 §6.4 / §6.6 — UA guard + Firefox NetworkInformation contract."""

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


# ── §6.6 NetworkInformation helper + Firefox absence contract ────────────────


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
        """Stay within plausible desktop broadband ranges. ``effectiveType``
        and ``saveData`` carry light entropy now (matches real-world fleet
        diversity) — accept the documented value sets rather than the
        prior fixed pair."""
        from src.browser.stealth import pick_network_info

        et_counts = {"4g": 0, "3g": 0, "2g": 0}
        save_data_seen = {True: 0, False: 0}
        for i in range(200):
            ni = pick_network_info(f"agent-{i}")
            assert ni["effectiveType"] in ("4g", "3g", "2g")
            assert isinstance(ni["saveData"], bool)
            assert 5.0 <= ni["downlink"] <= 20.0
            assert 20 <= ni["rtt"] <= 120
            et_counts[ni["effectiveType"]] += 1
            save_data_seen[ni["saveData"]] += 1
        # 4g should be the dominant value (≥80% per the weighting).
        assert et_counts["4g"] / 200 >= 0.80
        # saveData=False is the overwhelming majority (≥90%).
        assert save_data_seen[False] / 200 >= 0.90

    def test_firefox_launch_options_do_not_write_navigator_connection(
        self, monkeypatch,
    ):
        from src.browser.stealth import build_launch_options

        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-x", "/tmp/profile")
        cfg = opts.get("config") or {}
        assert not any(k.startswith("navigator.connection.") for k in cfg)
        assert "config" not in opts
        assert "i_know_what_im_doing" not in opts

    def test_ua_override_config_stays_firefox_only(self, monkeypatch):
        """UA override still uses Camoufox config, but must not bring back
        ``navigator.connection`` on the Firefox-shaped path."""
        from src.browser.stealth import build_launch_options

        monkeypatch.setenv("BROWSER_UA_VERSION", "138.0")
        opts = build_launch_options("agent-y", "/tmp/profile")
        cfg = opts["config"]
        assert cfg["navigator.userAgent"].endswith("Firefox/138.0")
        assert not any(k.startswith("navigator.connection.") for k in cfg)
        assert opts["i_know_what_im_doing"] is True
