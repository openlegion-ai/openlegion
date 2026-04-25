"""Tests for browser stealth/fingerprint configuration."""

from unittest.mock import patch

from src.browser.stealth import build_launch_options


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

    def test_default_proxy_is_none(self):
        """When proxy kwarg is not passed, default is None (no proxy)."""
        with patch.dict("os.environ", {"BROWSER_PROXY_URL": "http://env:8080"}, clear=False):
            opts = build_launch_options("agent-1", "/tmp/profile")
        assert "proxy" not in opts
        assert opts.get("geoip") is not True


class TestResolutionPool:
    """§6.1: per-agent resolution pool, deterministic from agent_id."""

    def test_pick_resolution_is_deterministic(self):
        from src.browser.stealth import pick_resolution

        # Same input → same output, every time. Survives restart /
        # profile wipe by design.
        assert pick_resolution("alpha") == pick_resolution("alpha")
        assert pick_resolution("beta-42") == pick_resolution("beta-42")

    def test_pick_resolution_returns_pool_entry(self):
        from src.browser.stealth import _RESOLUTION_POOL, pick_resolution

        valid = {res for res, _ in _RESOLUTION_POOL}
        for agent_id in ("a", "b", "x-x-x", "canary-probe", "agent-1234"):
            assert pick_resolution(agent_id) in valid

    def test_pick_resolution_distribution_matches_weights(self):
        """Across a large sample, empirical picks should approximate the
        weights. We don't assert perfect alignment — just that no bucket
        is drastically under/over-represented (would catch e.g. a typo in
        the cumulative walk)."""
        from src.browser.stealth import _RESOLUTION_POOL, pick_resolution

        weights = {res: w for res, w in _RESOLUTION_POOL}
        counts: dict = {res: 0 for res in weights}
        N = 5000
        for i in range(N):
            counts[pick_resolution(f"agent-{i}")] += 1

        for res, expected_weight in weights.items():
            observed = counts[res] / N
            # Allow generous drift (±5pp) — this is a sanity check on
            # the cumulative-bucket loop, not a statistical claim.
            assert abs(observed - expected_weight) < 0.05, (
                f"{res}: observed {observed:.3f} vs expected {expected_weight}"
            )

    def test_build_launch_options_applies_resolution(self):
        from src.browser.stealth import build_launch_options, pick_resolution

        with patch.dict("os.environ", {}, clear=True):
            opts = build_launch_options("agent-xyz", "/tmp/profile")

        expected = pick_resolution("agent-xyz")
        assert opts["window"] == expected
        # Screen fingerprint (when browserforge is available) matches.
        screen = opts.get("screen")
        if screen is not None:
            assert screen.max_width == expected[0]
            assert screen.max_height == expected[1]

    def test_different_agents_can_get_different_resolutions(self):
        """Sanity: across a small sample we see at least 2 distinct
        resolutions. If the pick function collapsed to one bucket this
        test would catch it quickly."""
        from src.browser.stealth import pick_resolution

        picked = {pick_resolution(f"a{i}") for i in range(50)}
        assert len(picked) >= 2

    def test_pick_resolution_handles_empty_agent_id(self):
        """Defensive: AGENT_ID_RE_PATTERN forbids it upstream, but the
        function must not crash if called with an empty string."""
        from src.browser.stealth import _RESOLUTION_POOL, pick_resolution

        result = pick_resolution("")
        valid = {res for res, _ in _RESOLUTION_POOL}
        assert result in valid


class TestQuietStartupPrefs:
    """Phase 3 §6.2 follow-on: quiet-startup prefs prevent Firefox's
    first-run UI (about:welcome, default-browser nag, profile-reset
    prompt) from blocking automation. Even though v2 migration
    deliberately preserves ``compatibility.ini`` to avoid triggering
    these, a fresh profile or a Firefox version bump can also cross
    those code paths — these prefs are belt-and-suspenders."""

    def test_first_run_prompts_disabled(self):
        from src.browser.stealth import _stealth_prefs

        prefs = _stealth_prefs()
        assert prefs["browser.shell.checkDefaultBrowser"] is False
        assert prefs["browser.aboutwelcome.enabled"] is False
        assert prefs["browser.startup.homepage_override.mstone"] == "ignore"
        assert prefs["startup.homepage_welcome_url"] == ""
        assert prefs["browser.disableResetPrompt"] is True
