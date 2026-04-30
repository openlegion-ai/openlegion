"""Per-platform pre-nav timing posture (LinkedIn / X / Twitter / Meta).

Tests the host-matching logic, Gaussian sampling determinism, and the
``BrowserManager._apply_platform_pre_nav_delay`` integration including
the operator override flag.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from src.browser import flags as _flags_module
from src.browser.stealth import (
    _PLATFORM_TIMING_PROFILES,
    get_platform_timing_profile,
    pick_platform_pre_nav_delay,
)


@pytest.fixture(autouse=True)
def _reset_flag_caches():
    """Operator-settings + per-agent override state is module-global —
    reset between tests so a leaked override from one case doesn't
    silently change the next case's gate decision."""
    _flags_module._operator_settings = None
    _flags_module._agent_overrides.clear()
    yield
    _flags_module._operator_settings = None
    _flags_module._agent_overrides.clear()


# ── 1: host matching ──────────────────────────────────────────────────


class TestHostMatching:
    def test_apex_matches(self):
        assert get_platform_timing_profile("https://linkedin.com/") is not None
        assert get_platform_timing_profile("https://x.com/foo") is not None
        assert get_platform_timing_profile("https://facebook.com/") is not None

    def test_subdomain_matches(self):
        # Bare-domain semantics: any subdomain of a registered host
        # should resolve to the parent's profile.
        for sub in (
            "https://www.linkedin.com/",
            "https://mobile.linkedin.com/",
            "https://api.linkedin.com/v2/foo",
            "https://m.facebook.com/",
            "https://touch.facebook.com/",
        ):
            assert get_platform_timing_profile(sub) is not None, sub

    def test_unknown_host_returns_none(self):
        assert get_platform_timing_profile("https://example.com/") is None
        assert get_platform_timing_profile("https://github.com/") is None
        # Lookalike — must NOT match.
        assert get_platform_timing_profile("https://notlinkedin.com/") is None
        assert get_platform_timing_profile("https://linkedin.com.evil.com/") is None

    def test_malformed_url_returns_none(self):
        assert get_platform_timing_profile("") is None
        assert get_platform_timing_profile("not-a-url") is None
        assert get_platform_timing_profile("javascript:alert(1)") is None

    def test_label_present_for_each_profile(self):
        # Every profile must carry a non-empty short-label string for
        # log-line use. A missing label would produce ``None`` in the
        # logger format string.
        for host, profile in _PLATFORM_TIMING_PROFILES.items():
            label = profile.get("label")
            assert isinstance(label, str) and label, f"{host} missing label"


# ── 2: Gaussian sampling ──────────────────────────────────────────────


class TestSampling:
    def test_unknown_host_returns_zero(self):
        delay, label = pick_platform_pre_nav_delay("https://example.com/")
        assert delay == 0.0
        assert label is None

    def test_known_host_returns_label_and_positive_delay(self):
        rng = random.Random(42)
        delay, label = pick_platform_pre_nav_delay(
            "https://linkedin.com/", rng=rng,
        )
        assert label == "linkedin"
        assert delay > 0.0

    def test_clamped_to_min_max(self):
        # Force the Gaussian to a value far outside the clamp window
        # via a stub rng; both directions.
        class StubRng:
            def __init__(self, value):
                self.value = value
            def gauss(self, mu, sigma):
                return self.value

        # Below min → clamped up.
        d_lo, _ = pick_platform_pre_nav_delay(
            "https://x.com/", rng=StubRng(-100.0),
        )
        x_profile = _PLATFORM_TIMING_PROFILES["x.com"]
        assert d_lo == x_profile["min_s"]

        # Above max → clamped down.
        d_hi, _ = pick_platform_pre_nav_delay(
            "https://x.com/", rng=StubRng(999.0),
        )
        assert d_hi == x_profile["max_s"]

    def test_deterministic_with_seeded_rng(self):
        # Same seed → same sample. Useful for fleet-scale audit (an
        # operator can replay a navigate's exact dwell from logs).
        rng_a = random.Random(1234)
        rng_b = random.Random(1234)
        d_a, _ = pick_platform_pre_nav_delay("https://linkedin.com/", rng=rng_a)
        d_b, _ = pick_platform_pre_nav_delay("https://linkedin.com/", rng=rng_b)
        assert d_a == d_b

    def test_distribution_within_clamp_window(self):
        # Sanity-check: 200 samples all fall in [min_s, max_s] for a
        # given platform. If the clamp logic regressed, this would
        # surface as out-of-range samples.
        rng = random.Random(777)
        profile = _PLATFORM_TIMING_PROFILES["facebook.com"]
        lo, hi = profile["min_s"], profile["max_s"]
        for _ in range(200):
            d, _label = pick_platform_pre_nav_delay(
                "https://facebook.com/", rng=rng,
            )
            assert lo <= d <= hi


# ── 3: BrowserManager integration ─────────────────────────────────────


class TestNavigateIntegration:
    """Verifies the manager-side wiring: when navigate is called against
    a protected platform, the dwell is applied; when called against an
    unrelated host, it is not; and the operator flag disables it."""

    @pytest.mark.asyncio
    async def test_dwell_applied_on_protected_platform(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)

        sleeps: list[float] = []

        async def _fake_sleep(s):
            sleeps.append(s)

        with patch("src.browser.service.asyncio.sleep", _fake_sleep):
            await mgr._apply_platform_pre_nav_delay(
                "agent-1", "https://linkedin.com/",
            )

        # Exactly one sleep with a positive duration.
        assert len(sleeps) == 1
        assert sleeps[0] > 0.0

    @pytest.mark.asyncio
    async def test_dwell_skipped_for_unknown_host(self):
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        sleeps: list[float] = []

        async def _fake_sleep(s):
            sleeps.append(s)

        with patch("src.browser.service.asyncio.sleep", _fake_sleep):
            await mgr._apply_platform_pre_nav_delay(
                "agent-1", "https://example.com/",
            )

        assert sleeps == []

    @pytest.mark.asyncio
    async def test_dwell_skipped_when_flag_disabled(self):
        """Operator can disable globally via the flag layer."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        sleeps: list[float] = []

        async def _fake_sleep(s):
            sleeps.append(s)

        # Disable via the operator-settings layer (mirrors what the
        # dashboard would write into ``config/settings.json``).
        _flags_module._operator_settings = {
            "BROWSER_PLATFORM_TIMING_ENABLED": "false",
        }

        with patch("src.browser.service.asyncio.sleep", _fake_sleep):
            await mgr._apply_platform_pre_nav_delay(
                "agent-1", "https://linkedin.com/",
            )

        assert sleeps == [], (
            "flag disabled but dwell was applied — operator override broken"
        )

    @pytest.mark.asyncio
    async def test_dwell_skipped_per_agent_override(self):
        """Per-agent override wins over operator default."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager.__new__(BrowserManager)
        sleeps: list[float] = []

        async def _fake_sleep(s):
            sleeps.append(s)

        _flags_module.set_agent_override(
            "agent-1", "BROWSER_PLATFORM_TIMING_ENABLED", "false",
        )

        with patch("src.browser.service.asyncio.sleep", _fake_sleep):
            await mgr._apply_platform_pre_nav_delay(
                "agent-1", "https://linkedin.com/",
            )

        assert sleeps == []

        # Other agents still get the dwell.
        with patch("src.browser.service.asyncio.sleep", _fake_sleep):
            await mgr._apply_platform_pre_nav_delay(
                "agent-2", "https://linkedin.com/",
            )
        assert len(sleeps) == 1 and sleeps[0] > 0.0
