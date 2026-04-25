"""Phase 3 §6.5 — referrer realism on navigate."""

from __future__ import annotations

import random


class TestSameOrigin:
    def test_previous_same_host_yields_same_origin_referer(self):
        from src.browser.stealth import pick_referer

        ref = pick_referer(
            "https://www.example.com/products",
            previous_url="https://www.example.com/category",
        )
        assert ref == "https://www.example.com/"

    def test_previous_different_host_does_not_match(self):
        from src.browser.stealth import pick_referer

        ref = pick_referer(
            "https://www.example.com/products",
            previous_url="https://other.com/page",
            rng=random.Random(0),
        )
        # Falls through to search default (deterministic via seeded rng).
        assert ref in (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )

    def test_previous_url_with_no_scheme_is_ignored(self):
        from src.browser.stealth import pick_referer

        # Malformed previous URL must not crash and must fall through
        # to a normal pick instead of yielding "://example.com/".
        ref = pick_referer(
            "https://www.example.com/page",
            previous_url="example.com",
            rng=random.Random(0),
        )
        assert ref.startswith("https://")
        assert "example.com" not in ref


class TestDirectNavHosts:
    def test_gmail_returns_empty_referer(self):
        """Gmail / GitHub etc. — search referers would themselves be
        suspicious (nobody Googles 'gmail.com' to check email)."""
        from src.browser.stealth import pick_referer

        for host in ("https://mail.google.com/inbox",
                     "https://github.com/myorg/myrepo",
                     "https://app.slack.com/client/T123"):
            assert pick_referer(host) == ""

    def test_direct_nav_overrides_social(self):
        """Even if a host is in BOTH _DIRECT_NAV_HOSTS and _SOCIAL_REFERERS
        (it isn't today, but defensive): direct should win."""
        from src.browser.stealth import _DIRECT_NAV_HOSTS, pick_referer

        assert "github.com" in _DIRECT_NAV_HOSTS
        assert pick_referer("https://github.com/whatever") == ""


class TestSocialPool:
    def test_twitter_picks_tco_some_of_the_time(self):
        """t.co is plausible but not universal; ~30% of nav should
        produce it. We don't assert a specific frequency, just that
        across 100 picks both shapes appear."""
        from src.browser.stealth import pick_referer

        rng = random.Random(42)
        picks = [
            pick_referer("https://twitter.com/foo", rng=rng)
            for _ in range(100)
        ]
        assert "https://t.co/" in picks
        # Search referers fill the gap when the social coin-flip says no
        assert any(p.startswith("https://www.google.com") for p in picks)

    def test_x_com_uses_twitter_pool(self):
        from src.browser.stealth import pick_referer

        rng = random.Random(0)
        picks = [
            pick_referer("https://x.com/alice", rng=rng) for _ in range(50)
        ]
        # Both shapes present
        assert "https://t.co/" in picks


class TestSearchDefault:
    def test_unknown_host_yields_search_referer(self):
        from src.browser.stealth import pick_referer

        rng = random.Random(7)
        ref = pick_referer("https://www.someshop.com/", rng=rng)
        assert ref in (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )


class TestRollingHistoryAvoidance:
    def test_avoids_recently_used_referer(self):
        """The picker should not return a referer already in the rolling
        history when alternatives are available."""
        from src.browser.stealth import pick_referer

        # All three search referers in history → pool exhausted → fall
        # back rather than return empty (which would itself be a tell).
        recent = (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )
        ref = pick_referer(
            "https://example.com/", recent_referers=recent,
            rng=random.Random(0),
        )
        assert ref in recent  # exhaustion fallback

    def test_partial_history_yields_unseen_referer(self):
        from src.browser.stealth import pick_referer

        recent = ("https://www.google.com/",)
        rng = random.Random(0)
        # Across many picks with google in recent, all picks should be
        # something else (Bing or DDG).
        picks = {
            pick_referer("https://shop.example/", recent_referers=recent,
                         rng=rng)
            for _ in range(20)
        }
        assert "https://www.google.com/" not in picks
        assert picks <= {
            "https://www.bing.com/", "https://duckduckgo.com/",
        }


class TestEdgeCases:
    def test_invalid_url_returns_empty(self):
        from src.browser.stealth import pick_referer

        assert pick_referer("") == ""
        assert pick_referer("not-a-url") == ""

    def test_url_without_host_returns_empty(self):
        from src.browser.stealth import pick_referer

        assert pick_referer("file:///etc/passwd") == ""
