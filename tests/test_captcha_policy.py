"""Tests for :mod:`src.browser.captcha_policy` (Phase 8 §11.18).

Covers:
* Hardcoded UNSOLVABLE / LOW_SUCCESS classification.
* Operator env-var overrides (force-solve, skip-solve) and their precedence.
* Domain-match semantics: bare-domain (eTLD+1 + subdomains) vs leading-dot
  (subdomains only).
* Hostname canonicalization edge cases: ``www.`` strip, port strip, IPv6,
  malformed URLs, paths / queries, empty input.

Env-var caches are parsed at module import time, so tests that exercise
the override path use :func:`importlib.reload` inside a patched
``os.environ`` context — same pattern used elsewhere in the suite for
import-time-bound config.
"""

from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest

from src.browser import captcha_policy


def _reload_with_env(env: dict[str, str]):
    """Reload ``captcha_policy`` with the given env applied.

    Returns the freshly-imported module so tests can call its functions
    against the new env-var-derived caches.
    """
    with mock.patch.dict(os.environ, env, clear=False):
        return importlib.reload(captcha_policy)


@pytest.fixture(autouse=True)
def _restore_module():
    """Ensure each test sees the default (no-override) module state.

    Some tests reload with custom env; this fixture reloads back to the
    pristine ambient env after every test so subsequent tests don't see
    leaked override caches.
    """
    yield
    # Drop both override env vars before reloading so we don't pick up
    # whatever ambient value was in the dev shell.
    scrubbed = {
        k: v for k, v in os.environ.items()
        if k not in {
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS",
        }
    }
    with mock.patch.dict(os.environ, scrubbed, clear=True):
        importlib.reload(captcha_policy)


# ── Hardcoded classification ───────────────────────────────────────────────


class TestHardcodedClassification:
    def test_unknown_domain_returns_default(self):
        assert captcha_policy.get_site_policy("https://example.com/") == "default"

    def test_accounts_google_is_low_success(self):
        assert (
            captcha_policy.get_site_policy("https://accounts.google.com/signup")
            == "low_success"
        )

    def test_mail_google_is_default(self):
        # `accounts.google.com` is the only Google entry; siblings under the
        # same registrable domain stay on the normal solver path.
        assert (
            captcha_policy.get_site_policy("https://mail.google.com/")
            == "default"
        )

    def test_twitter_is_low_success(self):
        assert captcha_policy.get_site_policy("https://twitter.com/i/flow/signup") == "low_success"

    def test_x_com_is_low_success(self):
        assert captcha_policy.get_site_policy("https://x.com/i/flow/signup") == "low_success"

    def test_linkedin_subdomain_is_low_success(self):
        # `linkedin.com` is a bare entry → matches any subdomain.
        assert (
            captcha_policy.get_site_policy("https://www.linkedin.com/login")
            == "low_success"
        )

    def test_instagram_login_is_low_success(self):
        assert (
            captcha_policy.get_site_policy("https://www.instagram.com/accounts/login/")
            == "low_success"
        )

    def test_signup_amazon_subdomain_is_low_success(self):
        assert (
            captcha_policy.get_site_policy("https://us.signup.amazon.com/")
            == "low_success"
        )

    def test_cloudflare_challenge_is_unsolvable(self):
        assert (
            captcha_policy.get_site_policy("https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g")
            == "unsolvable"
        )

    def test_humansecurity_is_unsolvable(self):
        assert (
            captcha_policy.get_site_policy("https://www.humansecurity.com/")
            == "unsolvable"
        )

    def test_datadome_iframe_host_is_unsolvable(self):
        assert (
            captcha_policy.get_site_policy("https://geo.captcha-delivery.com/captcha/")
            == "unsolvable"
        )


# ── Domain-match semantics ─────────────────────────────────────────────────


class TestDomainMatching:
    def test_bare_entry_matches_eTLD1_and_subdomains(self):
        # `linkedin.com` (bare) matches both the apex and any subdomain.
        assert captcha_policy._matches("linkedin.com", "linkedin.com")
        assert captcha_policy._matches("foo.linkedin.com", "linkedin.com")
        assert captcha_policy._matches("a.b.linkedin.com", "linkedin.com")

    def test_bare_entry_does_not_match_unrelated_suffix(self):
        # Naive `.endswith("linkedin.com")` would false-match
        # `evil-linkedin.com`. Our matcher requires a `.` boundary.
        assert not captcha_policy._matches("evil-linkedin.com", "linkedin.com")
        assert not captcha_policy._matches("linkedin.com.attacker.io", "linkedin.com")

    def test_leading_dot_entry_matches_subdomains_only(self):
        assert captcha_policy._matches("foo.example.com", ".example.com")
        # Apex itself is NOT a match for leading-dot entries.
        assert not captcha_policy._matches("example.com", ".example.com")


# ── Operator overrides ─────────────────────────────────────────────────────


class TestOperatorOverrides:
    def test_force_solve_overrides_low_success(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "accounts.google.com",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "",
        })
        assert m.get_site_policy("https://accounts.google.com/signup") == "default"
        assert m.is_force_solve("https://accounts.google.com/") is True

    def test_force_solve_overrides_unsolvable(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "challenges.cloudflare.com",
        })
        assert m.get_site_policy("https://challenges.cloudflare.com/x") == "default"

    def test_skip_solve_overrides_default(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "example.com",
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "",
        })
        assert m.get_site_policy("https://example.com/") == "unsolvable"
        assert m.is_skip_solve("https://example.com/") is True

    def test_force_solve_wins_over_skip_solve(self):
        # If the operator lists the same host in both, force-solve takes
        # precedence (matches the docstring's documented order).
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "example.com",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "example.com",
        })
        assert m.get_site_policy("https://example.com/") == "default"

    def test_bare_env_entry_matches_subdomains(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "evilcorp.com",
        })
        assert m.get_site_policy("https://www.evilcorp.com/") == "unsolvable"
        assert m.get_site_policy("https://api.evilcorp.com/") == "unsolvable"
        assert m.get_site_policy("https://evilcorp.com/") == "unsolvable"

    def test_leading_dot_env_entry_matches_subdomains_only(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": ".tenant.example.com",
        })
        assert m.get_site_policy("https://a.tenant.example.com/") == "unsolvable"
        # Apex `tenant.example.com` is not in the leading-dot scope.
        assert m.get_site_policy("https://tenant.example.com/") == "default"

    def test_empty_env_var_handled(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "   ",
        })
        # No entries parsed; hardcoded list still drives decisions.
        assert m.get_site_policy("https://example.com/") == "default"
        assert m.get_site_policy("https://accounts.google.com/") == "low_success"

    def test_whitespace_and_blanks_stripped(self):
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "  one.com ,, two.com  ,",
        })
        assert m.get_site_policy("https://one.com/") == "unsolvable"
        assert m.get_site_policy("https://two.com/") == "unsolvable"

    def test_malformed_env_entries_skipped(self):
        # URL-shaped entries are dropped with a warning; well-formed
        # entries on the same line still take effect.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS":
                "https://bad.com/path,good.com,worse.com/x?q=1",
        })
        assert m.get_site_policy("https://good.com/") == "unsolvable"
        assert m.get_site_policy("https://bad.com/") == "default"
        assert m.get_site_policy("https://worse.com/") == "default"

    def test_unset_env_means_no_overrides(self):
        # Sanity: with no env, the override predicates are False for all hosts.
        assert captcha_policy.is_force_solve("https://accounts.google.com/") is False
        assert captcha_policy.is_skip_solve("https://accounts.google.com/") is False


# ── URL canonicalization edge cases ────────────────────────────────────────


class TestURLCanonicalization:
    def test_path_and_query_ignored(self):
        # Only host matters — query strings and paths must not trigger
        # spurious matches via redaction or substring search.
        assert (
            captcha_policy.get_site_policy(
                "https://accounts.google.com/signin/v2/identifier?service=mail"
            )
            == "low_success"
        )

    def test_port_stripped(self):
        # urlsplit().hostname drops the port for us; verify the policy
        # still classifies correctly when one is present.
        assert (
            captcha_policy.get_site_policy("https://accounts.google.com:8443/")
            == "low_success"
        )

    def test_uppercase_host_normalized(self):
        assert (
            captcha_policy.get_site_policy("https://ACCOUNTS.GOOGLE.COM/")
            == "low_success"
        )

    def test_www_prefix_stripped(self):
        # `www.linkedin.com` matches the bare `linkedin.com` entry both via
        # the www-strip AND the suffix match — either path produces
        # low_success. Asserts no double-strip / mis-strip surprises.
        assert (
            captcha_policy.get_site_policy("https://www.linkedin.com/")
            == "low_success"
        )

    def test_ipv6_host_handled_gracefully(self):
        # IPv6 hostnames don't appear in our hardcoded list, and bracket
        # parsing must not raise. Result: default.
        assert captcha_policy.get_site_policy("https://[::1]/") == "default"
        assert (
            captcha_policy.get_site_policy("https://[2001:db8::1]:8443/path")
            == "default"
        )

    def test_malformed_url_returns_default(self):
        # Truly broken inputs → default (fail-open). The solver pipeline
        # downstream has its own gates.
        assert captcha_policy.get_site_policy("not a url at all") == "default"
        assert captcha_policy.get_site_policy("") == "default"

    def test_non_string_input_returns_default(self):
        # Defensive: caller bug shouldn't crash a hot-path lookup.
        assert captcha_policy.get_site_policy(None) == "default"  # type: ignore[arg-type]

    def test_url_with_userinfo_uses_host(self):
        # ``user:pass@host`` → host is what matters.
        assert (
            captcha_policy.get_site_policy("https://user:pass@accounts.google.com/")
            == "low_success"
        )

    def test_no_scheme_url_returns_default(self):
        # urlsplit on a bare hostname yields no `hostname` attribute →
        # we treat it as unparseable and fall through to default.
        assert captcha_policy.get_site_policy("accounts.google.com/x") == "default"
