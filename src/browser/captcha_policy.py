"""Site classification policy for captcha handling (Phase 8 §11.18).

A pure-stdlib lookup module that maps a page URL to one of three captcha
policy buckets. Consumed by the captcha detection / solver pipeline (§11.16,
the §11.13-aware detect path) so a single source of truth governs whether a
solver call is even attempted.

Buckets
-------

* ``unsolvable`` — captcha is behavioral-only or fingerprint-locked. Never
  charge the solver. Always escalate via ``request_captcha_help`` (§11.14).
  Examples: HUMAN Security "Press & Hold", DataDome behavioral blocker,
  Cloudflare Under-Attack-Mode after a wait.
* ``low_success`` — captcha is technically solvable but token-IP /
  fingerprint sensitive (Google signup, Twitter signup, LinkedIn auth).
  Solver attempted ONCE at ``solver_confidence: "low"``; on first failure,
  ``next_action`` upgrades to ``request_captcha_help`` rather than retry.
* ``default`` — normal solver flow.

Precedence (highest → lowest)
-----------------------------

1. ``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS`` — operator forces normal flow
   even on a hardcoded-unsolvable host.  Returns ``"default"``.
2. ``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS`` — operator forces escalation
   even on a host we'd normally solve.  Returns ``"unsolvable"``.
3. Hardcoded ``_UNSOLVABLE_DOMAINS`` set.  Returns ``"unsolvable"``.
4. Hardcoded ``_LOW_SUCCESS_DOMAINS`` set.  Returns ``"low_success"``.
5. Otherwise → ``"default"``.

Domain-match semantics
----------------------

Hostname canonicalization: ``urllib.parse.urlsplit(url).hostname`` →
lower-cased → leading ``www.`` stripped.

* **Bare domain** entry (``example.com``) — matches ``example.com`` AND any
  subdomain (``foo.example.com``, ``a.b.example.com``).  Most permissive;
  matches the operator intent of "block this org".
* **Leading-dot** entry (``.example.com``) — matches subdomains ONLY
  (``foo.example.com``), NOT ``example.com`` itself.  Useful when an
  operator wants to block a tenant's subdomains but allow the apex.

The hardcoded site list uses the bare-domain convention.

Operator override env vars are comma-separated.  Whitespace around entries
is stripped.  Empty / unset env = no overrides.  Wildcard prefixes
(``*.example.com``) are NOT supported by design — domain-suffix matching
already covers the same intent without parser complexity.

Caveat
------

The hardcoded list is intentionally short.  Real-world classification of
behavioral-only captchas is an operations problem with quarterly drift
(provider rebrands, selector changes — see §11.18).  Operators expand
coverage via the env-var allow/block lists; reload requires service
restart (out of scope here).

URL redaction
-------------

Any log line emitted by this module that contains a URL flows through
:func:`src.shared.redaction.redact_url` first.  In practice the module
emits no per-call logs (this is a hot-path lookup); the logger only
warns at module load if env-var parsing finds malformed entries.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlsplit

from src.shared.utils import setup_logging

logger = setup_logging("browser.captcha_policy")


SitePolicy = Literal["unsolvable", "low_success", "default"]


# ── Hardcoded site lists ───────────────────────────────────────────────────


# Behavioral-only / fingerprint-locked. Solver call is wasted spend on these
# hosts; route straight to operator escalation. Kept SHORT on purpose — the
# real-world long tail lives in operator overrides (see module docstring).
#
# Why these specific entries:
# - ``challenges.cloudflare.com`` — CF challenge platform host. Pages served
#   here are interstitials, not target content; reaching one means the parent
#   site is in Under-Attack mode and human action is required.
# - ``humansecurity.com`` — HUMAN Security's own marketing / demo domain.
#   Real protected sites embed HUMAN selectors (legacy ``data-v="px-button"``
#   from the 2022 PerimeterX rebrand, modern ``[data-human-security]``);
#   those selector hits live in §11.3 detection. The DOMAIN listed here only
#   blocks demos / docs hosted by HUMAN themselves.
# - ``captcha-delivery.com`` — DataDome's CDN. The ``/blocker`` iframe path
#   indicates behavioral; the ``/solver`` path is solvable. Domain-level
#   policy is conservative: any DataDome iframe is treated as unsolvable
#   here, with the per-iframe path check living in §11.3.
_UNSOLVABLE_DOMAINS: frozenset[str] = frozenset({
    "challenges.cloudflare.com",
    "humansecurity.com",
    "captcha-delivery.com",
})


# Token-IP + fingerprint sensitive: solvable in principle, but solver tokens
# minted from the provider's IP have ~50% success against the target's
# server-side risk score. Policy: try once at low confidence, escalate on
# failure (§11.18 + §11.13). Entries are full hostnames so adjacent
# subdomains under the same registrable domain (``mail.google.com``,
# ``drive.google.com``) keep the default policy — only signup / auth flows
# are flagged.
_LOW_SUCCESS_DOMAINS: frozenset[str] = frozenset({
    "accounts.google.com",   # Google signup / login
    "twitter.com",            # Twitter signup
    "x.com",                  # X (Twitter rebrand) signup
    "linkedin.com",           # LinkedIn auth
    "signup.amazon.com",      # Amazon retail signup
    "instagram.com",          # Instagram login / signup
})


# ── Env-var override parsing (module-load time) ────────────────────────────


_FORCE_SOLVE_ENV = "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS"
_SKIP_SOLVE_ENV = "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS"


def _parse_domain_list(raw: str, env_name: str) -> frozenset[str]:
    """Parse a comma-separated domain list from an env-var value.

    Whitespace around entries is stripped; empty entries dropped. Each
    surviving entry is lower-cased.  Leading-dot entries are preserved
    verbatim (their subdomain-only semantics are honored at match time).

    Malformed entries (containing whitespace inside the token, ``://``,
    ``/``, or ``?``) are skipped with a startup warning.  We never crash
    on bad operator config — that's a denial-of-service vector if the
    module sits on a hot path.
    """
    out: set[str] = set()
    for token in raw.split(","):
        entry = token.strip().lower()
        if not entry:
            continue
        # Reject things that are URLs or contain whitespace / path chars.
        # Leading-dot is allowed (``.example.com``) — that's our own
        # subdomain-only sigil, not a path char.
        if (
            "://" in entry
            or "/" in entry
            or "?" in entry
            or " " in entry
            or "\t" in entry
        ):
            logger.warning(
                "Ignoring malformed %s entry %r (looks like a URL or path)",
                env_name, token,
            )
            continue
        out.add(entry)
    return frozenset(out)


_FORCE_SOLVE_DOMAINS: frozenset[str] = _parse_domain_list(
    os.environ.get(_FORCE_SOLVE_ENV, ""), _FORCE_SOLVE_ENV,
)

_SKIP_SOLVE_DOMAINS: frozenset[str] = _parse_domain_list(
    os.environ.get(_SKIP_SOLVE_ENV, ""), _SKIP_SOLVE_ENV,
)


# ── Hostname extraction + matching ─────────────────────────────────────────


def _hostname(url: str) -> str | None:
    """Return the canonicalized hostname of ``url``, or None on failure.

    Canonicalization:
    * lower-case
    * strip leading ``www.``
    * port is dropped (urlsplit.hostname does this for us)
    * IPv6 literals returned without surrounding brackets

    Returns None for empty input, malformed URLs, or URLs with no host.
    Callers treat None as "no policy match" → falls through to default.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    host = parts.hostname
    if not host:
        return None
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _matches(host: str, entry: str) -> bool:
    """Return True iff ``host`` matches the policy ``entry``.

    Semantics (see module docstring):
    * Bare ``example.com`` → matches ``example.com`` AND any subdomain.
    * Leading-dot ``.example.com`` → subdomains only.
    """
    if entry.startswith("."):
        # Subdomain-only: must end with the entry, but the entry itself
        # (without leading dot) is NOT a match.
        return host.endswith(entry)
    # Bare entry: exact host or any subdomain.
    if host == entry:
        return True
    return host.endswith("." + entry)


def _matches_any(host: str, entries: frozenset[str]) -> bool:
    """Return True iff ``host`` matches any entry in ``entries``."""
    for entry in entries:
        if _matches(host, entry):
            return True
    return False


# ── Public API ─────────────────────────────────────────────────────────────


def is_force_solve(url: str) -> bool:
    """True iff the operator has forced solver attempts for this URL's host.

    Exposed separately from :func:`get_site_policy` so callers that want
    to log "operator override applied" (with the URL run through
    ``redact_url`` first) can do so without re-running the precedence
    chain.
    """
    host = _hostname(url)
    if host is None:
        return False
    return _matches_any(host, _FORCE_SOLVE_DOMAINS)


def is_skip_solve(url: str) -> bool:
    """True iff the operator has forced solver-skip for this URL's host."""
    host = _hostname(url)
    if host is None:
        return False
    return _matches_any(host, _SKIP_SOLVE_DOMAINS)


def get_site_policy(url: str) -> SitePolicy:
    """Classify ``url`` into a captcha-policy bucket.

    Order of precedence:

    1. ``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS`` → ``"default"``
    2. ``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS`` → ``"unsolvable"``
    3. Hardcoded :data:`_UNSOLVABLE_DOMAINS` → ``"unsolvable"``
    4. Hardcoded :data:`_LOW_SUCCESS_DOMAINS` → ``"low_success"``
    5. Otherwise → ``"default"``

    Malformed / hostless URLs return ``"default"`` (fail-open).  The
    solver pipeline downstream still has its own confidence gates and
    cost caps — this module is a routing hint, not a security boundary.
    """
    host = _hostname(url)
    if host is None:
        return "default"

    # Operator overrides win over hardcoded entries (in either direction).
    if _matches_any(host, _FORCE_SOLVE_DOMAINS):
        return "default"
    if _matches_any(host, _SKIP_SOLVE_DOMAINS):
        return "unsolvable"

    if _matches_any(host, _UNSOLVABLE_DOMAINS):
        return "unsolvable"
    if _matches_any(host, _LOW_SUCCESS_DOMAINS):
        return "low_success"

    return "default"
