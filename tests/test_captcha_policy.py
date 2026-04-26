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

    def test_amazon_apex_is_low_success(self):
        # Amazon's auth portal lives at amazon.com/ap/{register,signin}.
        # There is no signup.amazon.com (NXDOMAIN); apex-scoping is the
        # right granularity given Amazon's site-wide bot-detection posture.
        assert (
            captcha_policy.get_site_policy("https://www.amazon.com/ap/register")
            == "low_success"
        )
        assert (
            captcha_policy.get_site_policy("https://amazon.com/")
            == "low_success"
        )

    def test_accounts_google_subdomain_match(self):
        # Hardcoded entries use the SAME bare-domain matching rule as
        # operator overrides — they match the listed host AND any deeper
        # subdomain.  Documenting this explicitly so future maintainers
        # don't assume "host-exact" semantics for the hardcoded list.
        assert (
            captcha_policy.get_site_policy("https://foo.accounts.google.com/")
            == "low_success"
        )

    def test_x_com_subdomain_match(self):
        # ``x.com`` is a one-letter eTLD+1; verify single-character labels
        # don't confuse the matcher.  ``mobile.x.com`` is intentionally
        # classified low_success along with the apex (Twitter/X serve the
        # same risky auth flow at any sub-host).
        assert captcha_policy.get_site_policy("https://x.com/") == "low_success"
        assert (
            captcha_policy.get_site_policy("https://mobile.x.com/")
            == "low_success"
        )

    def test_x_com_does_not_falsely_match_suffix_collisions(self):
        # The dot-boundary check has to hold even for one-letter eTLD+1
        # entries: ``evilx.com`` ends with ``x.com`` as a substring but
        # not as a labeled suffix, so it must NOT match.  Same for the
        # ``x.com.attacker.io`` shape (host that contains the entry but
        # extends past it).
        assert captcha_policy.get_site_policy("https://evilx.com/") == "default"
        assert (
            captcha_policy.get_site_policy("https://x.com.attacker.io/")
            == "default"
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

    def test_wildcard_prefix_normalized_to_bare_domain(self):
        # ``*.example.com`` is the canonical DNS / firewall syntax for
        # "every subdomain of example.com".  We strip the prefix and treat
        # it as a bare-domain entry (which already covers apex + every
        # subdomain via suffix match).  Critically, the entry must NOT be
        # kept as the literal string ``*.example.com`` — that would never
        # match any real page and would silently swallow operator intent.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "*.example.com",
        })
        assert "*.example.com" not in m._SKIP_SOLVE_DOMAINS
        assert "example.com" in m._SKIP_SOLVE_DOMAINS
        assert m.get_site_policy("https://a.example.com/") == "unsolvable"
        assert m.get_site_policy("https://example.com/") == "unsolvable"

    def test_bare_star_entry_dropped(self):
        # A token that's just ``*`` (or ``*.``) leaves nothing after
        # stripping — drop with a warning, don't crash, don't silently
        # accept a frozenset entry of ``""`` that would match every host.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "*,*.,good.com",
        })
        assert "" not in m._SKIP_SOLVE_DOMAINS
        assert "." not in m._SKIP_SOLVE_DOMAINS
        assert "good.com" in m._SKIP_SOLVE_DOMAINS
        # Sanity: an unrelated host should NOT match — proves ``""`` did
        # not sneak into the frozenset (``host.endswith("")`` is True).
        assert m.get_site_policy("https://unrelated.io/") == "default"

    def test_whitespace_only_env_yields_no_entries(self):
        # Pathological values like " , , " must parse to empty without
        # warnings — operators commonly leave commented-out entries around.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": " , , ",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": ",,,",
        })
        assert m._FORCE_SOLVE_DOMAINS == frozenset()
        assert m._SKIP_SOLVE_DOMAINS == frozenset()

    def test_force_and_skip_helpers_are_independent(self):
        # is_force_solve / is_skip_solve return their raw membership;
        # precedence is applied only inside get_site_policy.  When the
        # operator lists the same host in both, both helpers say True
        # and the policy resolves to "default" (force wins).
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "dual.com",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "dual.com",
        })
        assert m.is_force_solve("https://dual.com/") is True
        assert m.is_skip_solve("https://dual.com/") is True
        assert m.get_site_policy("https://dual.com/") == "default"

    def test_env_override_matches_subdomains_like_hardcoded(self):
        # Confirms the docstring's "no asymmetry" claim: an env entry
        # ``google.com`` matches every google sub-host, exactly the same
        # as if ``google.com`` were in the hardcoded list.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "google.com",
        })
        # accounts.google.com is a hardcoded LOW_SUCCESS entry; force-solve
        # on the registrable domain neutralizes it.
        assert m.get_site_policy("https://accounts.google.com/") == "default"
        assert m.get_site_policy("https://mail.google.com/") == "default"
        assert m.get_site_policy("https://google.com/") == "default"


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

    def test_hostless_schemes_return_default(self):
        # ``data:``, ``file:``, and ``about:`` have no host → default
        # (fail-open).  Verifies _hostname() doesn't raise on these.
        assert captcha_policy.get_site_policy("data:text/html,<p>hi</p>") == "default"
        assert captcha_policy.get_site_policy("file:///etc/hosts") == "default"
        assert captcha_policy.get_site_policy("about:blank") == "default"
        # ``https://`` with no authority component also has no host.
        assert captcha_policy.get_site_policy("https://") == "default"

    def test_idn_punycode_match_literal_form(self):
        # IDN matching is literal-form: the IDN unicode form and its
        # punycode (``xn--``) equivalent are NOT auto-normalized to each
        # other.  Operators who care about both forms must list both.
        # This test pins the documented behavior so a future "helpfully"
        # auto-IDNA-encoding change doesn't slip through unnoticed.
        m = _reload_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "xn--wgv71a.jp",
        })
        # Punycode form in URL → matches the punycode entry.
        assert m.get_site_policy("https://xn--wgv71a.jp/") == "unsolvable"
        # Unicode form in URL → does NOT match the punycode entry.
        assert m.get_site_policy("https://日本.jp/") == "default"
