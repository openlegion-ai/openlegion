"""Core browser manager — per-agent Camoufox instance lifecycle.

Manages lazy-started Camoufox browser instances, one per agent.
Each agent gets its own persistent profile, BrowserForge fingerprint,
and browser context on a shared Xvnc display.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import mimetypes
import os
import random
import re
import subprocess
import time
import uuid
import weakref
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from src.browser import captcha_policy
from src.browser.captcha import (
    _ANTIBOT_KINDS,
    SolveResult,
    _classify_behavioral,
    _classify_cf_state,
    _classify_recaptcha,
    _redact_clientkey_text,
    get_solver,
)
from src.browser.js_challenge import classify_js_challenge
from src.browser.profile_schema import migrate_profile
from src.browser.redaction import CredentialRedactor
from src.browser.ref_handle import RefHandle, RefStale, ShadowHop
from src.browser.stealth import (
    DEFAULT_DEVICE_PROFILE,
    _canonical_host,
    build_launch_options,
    build_mobile_init_script,
    get_device_profile,
    pick_platform_pre_nav_delay,
    pick_referer,
    validate_referer,
)
from src.browser.timing import (
    action_delay,
    click_dwell,
    keystroke_delay,
    navigation_jitter,
    pre_click_settle,
    scroll_increment,
    scroll_pause,
    scroll_ramp,
    think_pause,
    x11_settle_delay,
    x11_step_delay,
)
from src.shared.redaction import redact_url
from src.shared.types import AGENT_ID_RE_PATTERN
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("browser.service")


# ── §11.13 structured CAPTCHA detection envelope ──────────────────────────
# Both helpers below produce literal-string enums (see plan §11.13). We do
# NOT use Python ``enum.Enum``: the wire format is JSON strings, and a real
# enum would either need a json encoder shim or repr-leak risk. Strings
# also keep the §11.1 / §11.3 / §11.16 / §11.18 follow-ups as pure data
# changes — no type plumbing.
#
# Enum reference (kept here so future PRs editing the helpers see the full
# vocabulary at a glance):
#
#   kind:
#     "recaptcha-v2-checkbox" | "recaptcha-v2-invisible" | "recaptcha-v3"
#     | "recaptcha-enterprise-v2" | "recaptcha-enterprise-v3"
#     | "hcaptcha" | "turnstile"
#     | "cf-interstitial-auto" | "cf-interstitial-behavioral"
#     | "cf-interstitial-turnstile" | "px-press-hold"
#     | "datadome-behavioral" | "unknown"
#     §11.1 lands the four reCAPTCHA variants; §11.3 lands the CF tri-state
#     plus the behavioral-only kinds (px-press-hold / datadome-behavioral).
#
#   solver_outcome:
#     "solved" | "timeout" | "rejected" | "injection_failed"
#     | "no_solver" | "unsupported" | "skipped_behavioral"
#     | "rate_limited" | "cost_cap" | "captcha_during_solve"   (§11.14)
#     - solved: token retrieved AND injected (or no injection needed).
#     - timeout: solver did not return a verdict in time.
#     - rejected: solver concluded the captcha cannot be solved (provider
#       errorId, sitekey not extractable, polling exhausted, OR — until the
#       solver API is enriched per Concern #10 below — a successful token
#       fetch followed by a failed injection that we cannot distinguish at
#       this layer).
#     - injection_failed: token fetched but injection rejected.
#       Surfaced when ``SolveResult.token`` is non-None but
#       ``injection_succeeded`` is False — the provider was paid (cost
#       counted) but our DOM injection failed. ``injection_failure_reason``
#       is set to ``"injection_failed_unspecified"`` for now; finer
#       disambiguation (CSP block, textarea_not_found, etc.) is a
#       §11.6/§11.20 deferred item.
#     - no_solver: no provider configured (``CAPTCHA_SOLVER_PROVIDER`` empty).
#     - unsupported: detected kind has no solver path. *RESERVED* until the
#       kind→provider matrix lands.
#     - skipped_behavioral: detected kind is behavioral-only (CF Under
#       Attack, PerimeterX Press & Hold, DataDome behavioral blocker) —
#       solver can't help; the agent should call ``request_captcha_help``.
#       Emitted by §11.3 detection BEFORE solver health/breaker gates so
#       behavioral-only flows don't consume health-check or breaker quota.
#
#   solver_confidence:
#     "high" | "medium" | "low" | "behavioral-only"
#     Reflects confidence in the *report* (kind classification + outcome
#     joint). Solved → "high"; timeout / exception / placeholder kinds →
#     "low" or "medium". "behavioral-only" is *RESERVED* for §11.18 — used
#     when no real captcha widget is shown and only the behavioral
#     fingerprint is the signal.
#
#   next_action:
#     "solved" | "wait" | "notify_user" | "request_captcha_help" | "ignored"
#     - "ignored" is *RESERVED* — emitted in the future for low-importance
#       captchas (analytics consent, ad-iframe captchas, etc.) that the
#       agent can safely skip. No code path emits it today.

# Selector-classification confidence: the kinds below are firmly
# disambiguated either by an unambiguous selector (``hcaptcha`` /
# ``turnstile``) or by the §11.1 reCAPTCHA variant classifier — when the
# classifier returns one of the four precise variants (v2-invisible / v3 /
# Enterprise-v2 / Enterprise-v3) we trust it as "high". The coarse
# ``recaptcha-v2-checkbox`` placeholder stays "low" because it's the
# fallback the classifier emits when it can't disambiguate. CF placeholders
# remain "low" until §11.3 lands variant detection.
_FIRM_KINDS = frozenset({
    "hcaptcha", "turnstile",
    "recaptcha-v2-invisible", "recaptcha-v3",
    "recaptcha-enterprise-v2", "recaptcha-enterprise-v3",
})


# ── §11.13 valid kind enum (for hint validation in solve_captcha) ─────────
# Kept in sync with the docstring at the top of this module. New kinds added
# here as §11.1 / §11.3 land richer classification.
#
# §11.1 split the prior coarse ``recaptcha-enterprise`` placeholder into
# the v2/v3 enterprise variants ``recaptcha-enterprise-v{2,3}``. The legacy
# coarse alias was REMOVED from this enum (review polish, finding F16) — it
# had no operational effect (no task-table entry, no pricing entry), so
# accepting it here just papered over miswired agent hints. Agents using
# the alias now get a clean ``invalid_input`` error pointing at the precise
# variants.
_VALID_CAPTCHA_KINDS: frozenset[str] = frozenset({
    "recaptcha-v2-checkbox", "recaptcha-v2-invisible", "recaptcha-v3",
    "recaptcha-enterprise-v2", "recaptcha-enterprise-v3",
    "hcaptcha", "turnstile",
    # §11.3 — CF interstitial tri-state + behavioral-only kinds.
    "cf-interstitial-auto", "cf-interstitial-behavioral",
    "cf-interstitial-turnstile",
    "px-press-hold", "datadome-behavioral",
    # §19 — tier-1 anti-bot JS-challenge frameworks. Detection-only;
    # never solver-attempted. Caller routes to ``request_captcha_help``
    # for operator intervention via the VNC handoff.
    "js-challenge-akamai", "js-challenge-kasada",
    "js-challenge-fingerprintjs", "js-challenge-imperva",
    "js-challenge-f5",
    "unknown",
})


# ── Behavioral-only hint rejection (§11.14 polish) ────────────────────────
# These kinds have no task-table entry — passing them as ``hint=`` to
# ``solve_captcha`` would cause a silent no-op (provider call skipped, the
# envelope's auto-classified kind is overridden but the solver was never
# going to engage anyway). Reject at the validator with a message that
# points at the correct path (``request_captcha_help``).
_BEHAVIORAL_KINDS: frozenset[str] = frozenset({
    "px-press-hold",
    "datadome-behavioral",
    "cf-interstitial-auto",
    "cf-interstitial-behavioral",
    # §19 — tier-1 anti-bot JS-challenge kinds. No solver task entry;
    # operator must intervene via VNC. Reject as ``hint`` value with a
    # message pointing at ``request_captcha_help``.
    "js-challenge-akamai",
    "js-challenge-kasada",
    "js-challenge-fingerprintjs",
    "js-challenge-imperva",
    "js-challenge-f5",
})


# Patience window for ``solve_captcha(retry_previous=True)`` (§11.14 polish).
# When the agent says "the captcha may not have rendered yet" we wait up
# to this many milliseconds AND re-check ONCE before returning the
# ``no captcha on current page`` envelope. Bound at the call site so the
# overall solve_captcha latency stays predictable.
_RETRY_PREVIOUS_RECHECK_MS = 500


# §11.3 — wait duration for CF auto-resolving JS challenge ("Just a moment").
# Spec calls for 5–10s; we pick 8s (midpoint) as a reasonable default:
# legitimate fingerprints typically clear in <5s, while 10s+ would stall
# agent loops on dead pages. One wait+recheck cycle per ``_check_captcha``
# call (no retry loop) — if the page hasn't navigated by then we fall
# through to behavioral classification.
_CF_AUTO_WAIT_SECONDS = 8.0


# ── Filename sanitizer for downloads ──────────────────────────────────────
# Browser-supplied ``Content-Disposition: filename="..."`` is attacker-
# controlled. A response carrying ``filename="../../etc/foo"`` would
# escape the configured download dir if joined naively. Strip path
# components, collapse unsafe chars, and bound length. Mirrors the
# mesh-side ``_sanitize_artifact_name`` so both sides of the transfer
# apply the same rule (defense in depth).
_DOWNLOAD_NAME_SAFE_RE = re.compile(r"[^\w.\-]+")


def _sanitize_download_filename(suggested: str) -> str:
    """Reduce a browser-supplied filename to a safe basename.

    Returns ``"download.bin"`` when the result would be empty so the
    caller never sees a path-traversal-shaped string land on disk.
    """
    name = (suggested or "").strip()
    if "/" in name or "\\" in name:
        name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = _DOWNLOAD_NAME_SAFE_RE.sub("_", name)
    name = name.strip("._-")
    if len(name) > 180:
        name = name[:180].strip("._-")
    if not name:
        name = "download.bin"
    if len(name) < 2:
        name = name + "_"
    return name


# ── §22 anti-bot solver opt-in helper ─────────────────────────────────────
def _solver_supports_kind(solver: object, kind: str) -> bool:
    """Return ``True`` iff ``solver.supports_kind(kind)`` is genuinely True.

    Three-state defensive read: the bundled ``CaptchaSolver`` and the
    ``MultiProviderSolver`` wrapper both expose
    :meth:`supports_kind`; legacy / mocked solvers may not. We require
    a strict ``is True`` comparison rather than truthiness so that
    ``MagicMock`` auto-attribute children (which test suites attach to
    ``BrowserManager._captcha_solver``) don't accidentally enable the
    anti-bot solver path. Tests opt-in by configuring
    ``solver.supports_kind = MagicMock(return_value=True)``.

    Returns ``False`` on any exception so a buggy third-party subclass
    can't crash the upstream :meth:`_check_captcha` flow.
    """
    method = getattr(solver, "supports_kind", None)
    if not callable(method):
        return False
    try:
        return method(kind) is True
    except Exception:
        logger.debug("supports_kind raised", exc_info=True)
        return False


# ── §11.14 per-agent solve rate limiter ───────────────────────────────────
# Module-level dict guarded by an asyncio.Lock. Each agent gets a deque of
# unix timestamps for solve attempts in the last hour; entries older than
# 1h are pruned on each access. The limit is read from the
# CAPTCHA_RATE_LIMIT_PER_HOUR flag (default 20), matching the trim spec.
#
# Note: shared module-level state means multi-tenant deployments where
# different processes serve different agents won't see a unified rate
# limit — by design (matches how the cost counter is also process-local in
# this trim). Per-process is sufficient because the browser service is a
# single container, and the cap is a soft anti-abuse signal not a billing
# control (cost counter handles billing).
_solve_rate_window: dict[str, deque[float]] = {}
_solve_rate_lock: asyncio.Lock | None = None
_solve_rate_lock_loop: asyncio.AbstractEventLoop | None = None
_SOLVE_RATE_WINDOW_SECONDS = 3600.0


def _get_solve_rate_lock() -> asyncio.Lock:
    """Return a solve-rate lock bound to the active event loop."""
    global _solve_rate_lock, _solve_rate_lock_loop
    loop = asyncio.get_running_loop()
    if _solve_rate_lock is None or _solve_rate_lock_loop is not loop:
        _solve_rate_lock = asyncio.Lock()
        _solve_rate_lock_loop = loop
    return _solve_rate_lock


async def _check_solve_rate(agent_id: str, limit_per_hour: int) -> bool:
    """Return ``True`` if the agent is OVER the per-hour solve limit.

    Side effect on miss: records the current timestamp so the NEXT call's
    pruning sees this attempt. Caller is expected to invoke once per
    intended solve; double-counting would distort the rate.

    ``limit_per_hour <= 0`` disables the limiter (returns False always) so
    operators can opt out via the env var.
    """
    if limit_per_hour <= 0:
        return False
    async with _get_solve_rate_lock():
        now = time.time()
        cutoff = now - _SOLVE_RATE_WINDOW_SECONDS
        # ``deque(maxlen=N)`` is fixed at construction time. If the
        # operator raises ``CAPTCHA_RATE_LIMIT_PER_HOUR`` mid-run, the
        # existing bucket would silently drop new timestamps once it
        # reached its old maxlen — under-reporting the rate and letting
        # a solve through that should have been blocked. Recreate the
        # bucket whenever the configured limit grows.
        target_maxlen = limit_per_hour * 4
        bucket = _solve_rate_window.get(agent_id)
        if bucket is None or (bucket.maxlen or 0) < target_maxlen:
            bucket = deque(bucket or (), maxlen=target_maxlen)
            _solve_rate_window[agent_id] = bucket
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit_per_hour:
            return True
        bucket.append(now)
        return False


def _resolve_rate_limit(agent_id: str) -> int:
    """Read the per-hour solve-rate limit for an agent, with fallback default."""
    from src.browser.flags import get_int
    return get_int(
        "CAPTCHA_RATE_LIMIT_PER_HOUR", 20,
        agent_id=agent_id, min_value=0, max_value=10000,
    )


def _resolve_cost_cap(agent_id: str) -> int:
    """Read the monthly per-agent cost cap (USD) and convert to MILLICENTS.

    Returns 0 when unset or invalid (caller treats 0 as "no cap"). Reads
    via ``flags.get_str`` so per-agent overrides take precedence over the
    raw env var (matches the pattern used elsewhere for captcha flags).

    Unit math: ``captcha_cost_counter`` stores spend in millicents
    (1/1000 of a cent = 1/100_000 of a dollar). Convert
    ``$X → X * 100_000``. Example: ``$0.50 → 50_000 millicents``;
    a 2captcha v2-checkbox solve at 100 millicents lets ~500 solves
    accumulate before the cap fires (the operator's intent), versus the
    pre-fix ``* 100`` that tripped after the FIRST solve.
    """
    from src.browser.flags import get_str
    cap_usd_str = get_str(
        "CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH", "", agent_id=agent_id,
    )
    if not cap_usd_str:
        return 0
    try:
        return int(round(float(cap_usd_str) * 100_000))
    except ValueError:
        logger.warning(
            "Invalid CAPTCHA_COST_LIMIT_USD_PER_AGENT_MONTH=%r — treating "
            "as unset", cap_usd_str,
        )
        return 0


def _max_published_solve_cost_millicents(provider: str, kind: str) -> int | None:
    """Return the max known price for ``(provider, kind)``.

    A solve can end up proxy-aware or proxyless depending on solver-proxy
    compatibility. For cost-cap reservations we reserve the maximum
    published tier up front, then refund any difference after the provider
    result reports the actual path used.
    """
    from src.browser import captcha_cost_counter as _cost

    candidates = [
        _cost.estimate_millicents(provider, kind, proxy_aware=False),
        _cost.estimate_millicents(provider, kind, proxy_aware=True),
    ]
    priced = [c for c in candidates if c is not None]
    return max(priced) if priced else None


# ── §11.14 / §2.7 audit-log aggregator ────────────────────────────────────
#
# Operators need to see when the cost-cap / rate-limit / behavioral-skip
# gates fire — otherwise a misconfigured cap silently blocks every solve
# without any signal. The §2.7 cadence forbids per-call dashboard events
# (would flood the WebSocket on a hot loop); aggregate per minute keyed
# by ``(agent_id, outcome, kind)``.
#
# State is module-global because the BrowserManager is a singleton in the
# browser service container. The flush is driven from
# :meth:`BrowserManager._emit_metrics` (already on a 60s tick); see
# ``_drain_captcha_audit`` below.
_captcha_audit_lock: asyncio.Lock | None = None
_captcha_audit_lock_loop: asyncio.AbstractEventLoop | None = None
# {(agent_id, outcome, kind): {"count": int, "first_ts": float, "last_url": str}}
_captcha_audit_buckets: dict[tuple[str, str, str], dict] = {}


def _get_captcha_audit_lock() -> asyncio.Lock:
    """Return a CAPTCHA audit lock bound to the active event loop."""
    global _captcha_audit_lock, _captcha_audit_lock_loop
    loop = asyncio.get_running_loop()
    if _captcha_audit_lock is None or _captcha_audit_lock_loop is not loop:
        _captcha_audit_lock = asyncio.Lock()
        _captcha_audit_lock_loop = loop
    return _captcha_audit_lock


async def _record_captcha_audit_event(
    agent_id: str, outcome: str, kind: str, page_url: str,
    *, policy: str | None = None,
) -> None:
    """Aggregate a captcha gate event into the per-minute bucket.

    Outcomes recorded: ``cost_cap``, ``rate_limited``, ``skipped_behavioral``,
    ``low_success_failed`` (§11.18).
    Aggregation by ``(agent_id, outcome, kind)`` so a noisy retry loop on
    one agent doesn't drown out the others. ``page_url`` is redacted via
    :func:`redact_url` before storage; only the most recent redacted URL
    in the bucket is retained (the bucket is dashboard signal, not an
    audit trail).

    ``policy`` (optional) records the §11.18 :func:`captcha_policy.get_site_policy`
    classification (``"unsolvable"`` / ``"low_success"`` / ``"default"``) that
    drove the gate decision, so operators can see WHY a solve was skipped or
    escalated. Stored on the bucket and surfaced on the drained
    ``captcha_gate`` payload as ``policy``. Last-write-wins within an
    aggregation window — same convention as ``last_url``.
    """
    async with _get_captcha_audit_lock():
        key = (agent_id, outcome, kind)
        bucket = _captcha_audit_buckets.get(key)
        now = time.time()
        safe_url = redact_url(page_url) if page_url else ""
        if bucket is None:
            _captcha_audit_buckets[key] = {
                "count": 1,
                "first_ts": now,
                "last_url": safe_url,
                "policy": policy,
            }
        else:
            bucket["count"] += 1
            bucket["last_url"] = safe_url
            if policy is not None:
                bucket["policy"] = policy


async def _drain_captcha_audit() -> list[dict]:
    """Atomically swap the audit-bucket dict and return the drained payloads.

    Called once per metrics-emit tick. Each returned dict is one
    EventBus payload; the caller is responsible for actually emitting
    via the configured ``metrics_sink``.
    """
    async with _get_captcha_audit_lock():
        if not _captcha_audit_buckets:
            return []
        # Snapshot copy under lock so the subsequent ``clear()`` doesn't
        # wipe out the in-flight dict (dict assignment is by reference,
        # not value).
        buckets = dict(_captcha_audit_buckets)
        _captcha_audit_buckets.clear()
    drained = []
    for (agent_id, outcome, kind), info in buckets.items():
        # NOTE: the dashboard metrics poller (host/server.py) routes
        # browser-service events into the EventBus keyed on the payload's
        # ``agent_id`` field; emitting ``agent`` instead used to silently
        # drop these events from the per-agent captcha history. The key
        # name MUST match the poller's expectation — a regression test
        # in test_check_captcha_metered.py asserts the payload exposes
        # ``agent_id``, not ``agent``.
        payload = {
            "type": "captcha_gate",
            "agent_id": agent_id,
            "outcome": outcome,
            "kind": kind,
            "count": info["count"],
            "first_ts": info["first_ts"],
            "url": info["last_url"],
        }
        # §11.18 — surface the site-policy classification so operators see
        # WHY a solve was skipped/escalated.  Older buckets created before
        # the field was added may be missing it (back-compat with the
        # ``policy=None`` default in :func:`_record_captcha_audit_event`).
        if info.get("policy") is not None:
            payload["policy"] = info["policy"]
        drained.append(payload)
    return drained


# ── §20 — session-persistence audit aggregator ────────────────────────────
#
# Snapshot/restore events surface to the operator dashboard via the same
# per-minute aggregation pattern as the captcha audit log (§11.14 / §2.7
# forbid per-call events). Aggregated by ``(agent_id, action, success)``
# so a stuck periodic-snapshot loop on one agent doesn't drown out the
# rest. Cookie values and origin domains are NEVER recorded — privacy
# posture (§20). The bucket holds counts + first/last timestamps only.
#
# State is module-global because the BrowserManager is a singleton in
# the browser-service container. Flush is driven from
# :meth:`BrowserManager._emit_metrics` (already on the 60s tick); see
# ``_drain_session_audit`` below.
_session_audit_lock: asyncio.Lock | None = None
_session_audit_lock_loop: asyncio.AbstractEventLoop | None = None
# {(agent_id, action, success_bool): {"count": int, "first_ts": float}}
_session_audit_buckets: dict[tuple[str, str, bool], dict] = {}


def _get_session_audit_lock() -> asyncio.Lock:
    """Return a session audit lock bound to the active event loop."""
    global _session_audit_lock, _session_audit_lock_loop
    loop = asyncio.get_running_loop()
    if _session_audit_lock is None or _session_audit_lock_loop is not loop:
        _session_audit_lock = asyncio.Lock()
        _session_audit_lock_loop = loop
    return _session_audit_lock


async def _record_session_audit_event(
    agent_id: str, action: str, success: bool,
) -> None:
    """Aggregate a session snapshot/restore event into the per-minute bucket.

    ``action`` is ``"session_snapshot"`` or ``"session_restore"``. The
    bucket aggregates by ``(agent_id, action, success)`` so a misconfigured
    flag or broken disk surfaces clearly. URL and origin domains are
    NEVER captured here — that would leak which sites the agent is
    logged into through the dashboard event stream.
    """
    async with _get_session_audit_lock():
        key = (agent_id, action, bool(success))
        bucket = _session_audit_buckets.get(key)
        now = time.time()
        if bucket is None:
            _session_audit_buckets[key] = {
                "count": 1,
                "first_ts": now,
            }
        else:
            bucket["count"] += 1


async def _drain_session_audit() -> list[dict]:
    """Atomically swap the session audit buckets and return drained payloads.

    Called once per metrics-emit tick. Each returned dict is one
    EventBus payload. Mirrors :func:`_drain_captcha_audit` but with the
    ``session_event`` type so the dashboard can route events into a
    dedicated panel.
    """
    async with _get_session_audit_lock():
        if not _session_audit_buckets:
            return []
        buckets = dict(_session_audit_buckets)
        _session_audit_buckets.clear()
    drained = []
    for (agent_id, action, success), info in buckets.items():
        # NOTE: ``agent_id`` (not ``agent``) — the dashboard metrics
        # poller routes browser-service events into the per-agent
        # EventBus by this exact field name, same as the captcha audit
        # path.  No ``url`` or ``origin`` field — privacy posture (§20).
        drained.append({
            "type": "session_event",
            "agent_id": agent_id,
            "action": action,
            "success": success,
            "count": info["count"],
            "first_ts": info["first_ts"],
        })
    return drained


# ── platform pre-nav delay audit aggregator ──────────────────────────────
#
# Companion to the per-platform pre-nav dwell (LinkedIn / X / Meta).  Per
# §2.7 we cannot emit one event per applied dwell — that would flood the
# WebSocket on a hot navigate loop.  Aggregate by ``(agent_id, host)`` so
# the dashboard's per-platform success panel can compute count + average
# dwell duration without scraping the INFO log.
#
# State is module-global because the BrowserManager is a singleton in
# the browser service container.  Drain is driven from
# :meth:`BrowserManager._emit_metrics` (already on the 60s tick); see
# ``_drain_platform_timing_audit`` below.
_platform_timing_audit_lock: asyncio.Lock | None = None
_platform_timing_audit_lock_loop: asyncio.AbstractEventLoop | None = None
# {(agent_id, host): {"count": int, "total_delay_s": float, "first_ts": float}}
_platform_timing_audit_buckets: dict[tuple[str, str], dict] = {}


def _get_platform_timing_audit_lock() -> asyncio.Lock:
    """Return a platform-timing audit lock bound to the active event loop."""
    global _platform_timing_audit_lock, _platform_timing_audit_lock_loop
    loop = asyncio.get_running_loop()
    if (
        _platform_timing_audit_lock is None
        or _platform_timing_audit_lock_loop is not loop
    ):
        _platform_timing_audit_lock = asyncio.Lock()
        _platform_timing_audit_lock_loop = loop
    return _platform_timing_audit_lock


async def _record_platform_timing_audit_event(
    agent_id: str, host: str, delay_s: float,
) -> None:
    """Aggregate one applied pre-nav dwell into the per-minute bucket.

    ``host`` is the canonical host (lower-cased, ``www.`` stripped) —
    callers route through :func:`src.browser.stealth._canonical_host`.
    Bucket aggregates by ``(agent_id, host)`` so a single agent looping
    on one platform doesn't drown out the others.  ``delay_s`` is
    summed so the drain payload can carry a true running average.
    """
    if not host or delay_s < 0:
        return
    async with _get_platform_timing_audit_lock():
        key = (agent_id, host)
        bucket = _platform_timing_audit_buckets.get(key)
        now = time.time()
        if bucket is None:
            _platform_timing_audit_buckets[key] = {
                "count": 1,
                "total_delay_s": float(delay_s),
                "first_ts": now,
            }
        else:
            bucket["count"] += 1
            bucket["total_delay_s"] += float(delay_s)


async def _drain_platform_timing_audit() -> list[dict]:
    """Atomically swap the platform-timing audit buckets and return payloads.

    Called once per metrics-emit tick. Mirrors
    :func:`_drain_captcha_audit` — each returned dict is one EventBus
    payload of type ``platform_pre_nav_delay`` so the dashboard's
    per-platform success aggregator can route it into the right host
    bucket.
    """
    async with _get_platform_timing_audit_lock():
        if not _platform_timing_audit_buckets:
            return []
        buckets = dict(_platform_timing_audit_buckets)
        _platform_timing_audit_buckets.clear()
    drained = []
    for (agent_id, host), info in buckets.items():
        drained.append({
            "type": "platform_pre_nav_delay",
            # ``agent_id`` (not ``agent``) — same convention as the
            # captcha / session / fingerprint audit paths.
            "agent_id": agent_id,
            "host": host,
            "count": info["count"],
            "total_delay_s": round(info["total_delay_s"], 4),
            "first_ts": info["first_ts"],
        })
    return drained


# ── §22 / §11.17 — fingerprint health monitoring ─────────────────────────
#
# Anti-bot frameworks sometimes accept a solver-injected token mechanically
# (the solve "succeeds" from our perspective) but reject the SESSION the
# token was injected into — typically when the agent's BrowserForge
# fingerprint has been observed solving captchas across many sites and is
# now flagged. The solver is healthy; the FINGERPRINT is burned.
#
# Per-agent rolling window of post-solve outcomes:
#   - True  → page rejected the solve (selector still present, "verification
#             failed" / "try again" / "could not be verified" / "robot detected"
#             text on page).
#   - False → page accepted the solve (navigated away, no captcha selector).
#
# Burn detection: window full (>=10 entries) AND >=50% True. While burned,
# subsequent ``_check_captcha`` envelopes carry ``fingerprint_burn: True``
# and ``next_action="retry_with_fresh_profile"``, signalling the operator
# to rotate the profile manually (auto-rotation is too destructive — it
# wipes session cookies / login state and can cascade other agents' work).
#
# Operator clears the burn manually via ``POST .../fingerprint-health/reset``
# after rotating the profile.  The window also clears naturally once 10
# consecutive accepts roll the True entries out.
#
# Exclusions per §11.17 spec:
#   - ``skipped_behavioral`` and ``request_captcha_help`` outcomes do NOT
#     count toward the rejection rate.  They're correct escalations, not
#     rejections — the solver never attempted-and-injected a token, so
#     post-solve page state has nothing to tell us about the fingerprint.
#   - Only outcomes where the solver actually attempted-and-injected a
#     token (``solver_outcome="solved"``, page-state monitor verifies)
#     contribute to the window.
#
# State is module-global because the BrowserManager is a singleton in the
# browser-service container.  Per-agent ``deque(maxlen=10)`` is sufficient
# (no new sliding-window primitive needed); reuses the same shape as
# ``CaptchaSolver._solver_failure_timestamps``.
_FINGERPRINT_WINDOW_SIZE = 10
_FINGERPRINT_BURN_THRESHOLD = 0.5  # >= 50% rejected → burned
# Page-state monitor: how long we wait after a ``solved`` envelope to
# observe whether the page accepted or rejected the token.
_FINGERPRINT_MONITOR_TIMEOUT_S = 10.0
# Text fragments that indicate the post-solve page is still rejecting the
# session.  Matched case-insensitively against the page's visible body
# text inside the monitor's poll loop.  Multi-vendor coverage: the agent
# may be solving captchas on sites protected by Cloudflare, DataDome,
# PerimeterX/HUMAN, Akamai BMP, Imperva, F5 Distributed Cloud, Kasada, or
# native vendor implementations.  False-positive cost is low (one entry
# in a 10-deep window), and missing a rejection costs us much more (the
# fingerprint stays in service after it has been flagged).
_FINGERPRINT_REJECTION_TEXT: tuple[str, ...] = (
    # Generic — vendor-agnostic phrasing reused across stacks.
    "verification failed",
    "try again",
    "could not be verified",
    "robot detected",
    "are you a robot",
    "please complete the verification",
    "security check",
    "checking if the site connection is secure",
    "additional verification required",
    "human verification",
    # Cloudflare — interstitial / 1xxx error codes.  The interstitial
    # body shows "Just a moment..." while running the JS challenge; if
    # we still see it 10s after token injection the challenge re-armed.
    "just a moment",
    "ray id",
    "cloudflare ray id",
    "error 1020",  # Access denied — site rule blocked the IP.
    "error 1015",  # Rate-limited.
    "error 1010",  # Browser signature blocked.
    # DataDome — branded block page.
    "blocked by datadome",
    "request unsuccessful",
    "incident id",  # Imperva + DataDome both surface this.
    # PerimeterX / HUMAN — branded block page text.
    "please verify you are a human",
    "press and hold",  # PerimeterX press-and-hold variant.
    "this site uses an additional layer of security",
    # Imperva — generic block.
    "access denied",
    "request blocked",
    "you don't have permission to access",
    # Akamai BMP — branded reference text.
    "reference #",  # "Reference #18.deadbeef.1234567890" pattern.
    # Generic anti-bot rejection language.
    "you have been blocked",
    "unusual activity",
    "suspicious activity",
    "session expired",
    "challenge failed",
    "captcha failed",
    "captcha invalid",
)
# Vendor-specific element selectors — far more deterministic than text
# scanning (language-independent, layout-stable).  Each selector here
# matches a known anti-bot vendor's interstitial / block page.  If any
# selector resolves with count>0 on the post-solve page, the session
# was rejected.
_FINGERPRINT_REJECTION_SELECTORS: tuple[str, ...] = (
    # Cloudflare — challenge / error pages (#cf-error-details and the
    # 1xxx error containers; #challenge-error-text on the JS challenge
    # interstitial).
    "[id^=cf-error]",
    "#challenge-error-text",
    ".cf-error-details",
    ".cf-browser-verification",
    # DataDome — geo / device block page.
    "#ddm-blocked",
    ".dd-r-blocked",
    "[data-dd-block]",
    # PerimeterX / HUMAN — block / press-and-hold page (#px-captcha
    # appears on the press-and-hold variant; .px-block-spam on the
    # outright block).
    "#px-captcha",
    ".px-block-spam",
    # Imperva — incident page (the iframe wraps the actual block content).
    "#main-iframe[src*=incident]",
    # Akamai BMP — bm-error reference container.
    "[id^=bm-error]",
    # F5 Distributed Cloud / Shape Security — generic block iframe.
    "iframe[src*=shape]",
)
_fingerprint_lock: asyncio.Lock | None = None
_fingerprint_lock_loop: asyncio.AbstractEventLoop | None = None
# {agent_id: deque[bool]} — entries True = rejected, False = accepted.
_fingerprint_window: dict[str, deque[bool]] = {}
# {agent_id: float} — unix timestamp of the most recent recorded signal.
# Surfaced by the dashboard endpoint as ``last_signal_ts``.
_fingerprint_last_signal: dict[str, float] = {}


def _get_fingerprint_lock() -> asyncio.Lock:
    """Return a fingerprint-state lock bound to the active event loop."""
    global _fingerprint_lock, _fingerprint_lock_loop
    loop = asyncio.get_running_loop()
    if _fingerprint_lock is None or _fingerprint_lock_loop is not loop:
        _fingerprint_lock = asyncio.Lock()
        _fingerprint_lock_loop = loop
    return _fingerprint_lock


async def _record_fingerprint_outcome(
    agent_id: str, rejected: bool,
) -> bool:
    """Append a post-solve outcome to the agent's rolling fingerprint window.

    Returns the burn-state AFTER the append.  Caller can use the return
    value to decide whether to emit a ``fingerprint_burn`` audit event.
    The deque is bounded at :data:`_FINGERPRINT_WINDOW_SIZE` so the
    natural rollover happens once 10 consecutive accepts arrive.
    """
    async with _get_fingerprint_lock():
        bucket = _fingerprint_window.get(agent_id)
        if bucket is None:
            bucket = deque(maxlen=_FINGERPRINT_WINDOW_SIZE)
            _fingerprint_window[agent_id] = bucket
        bucket.append(bool(rejected))
        _fingerprint_last_signal[agent_id] = time.time()
        return _is_burned_locked(bucket)


def _is_burned_locked(bucket: deque[bool]) -> bool:
    """Compute burn state from a window snapshot.

    Caller must hold :func:`_get_fingerprint_lock`.  Burn fires only
    when the window is FULL — partial windows are not enough signal.
    """
    if len(bucket) < _FINGERPRINT_WINDOW_SIZE:
        return False
    rejected = sum(1 for v in bucket if v)
    return (rejected / len(bucket)) >= _FINGERPRINT_BURN_THRESHOLD


async def _is_fingerprint_burned(agent_id: str) -> bool:
    """Read-only burn check for ``_check_captcha`` to decorate envelopes."""
    async with _get_fingerprint_lock():
        bucket = _fingerprint_window.get(agent_id)
        if bucket is None:
            return False
        return _is_burned_locked(bucket)


async def _get_fingerprint_health(agent_id: str) -> dict:
    """Return the dashboard-shaped health payload for one agent.

    Shape is contract-stable — see plan §22 dashboard panel and
    ``test_fingerprint_health::test_health_endpoint_shape``::

        {"window_size": int, "rejection_rate": float, "burned": bool,
         "last_signal_ts": str | None}

    ``rejection_rate`` is 0.0 when the window is empty (avoid divide-by-zero
    surprises in the dashboard).  ``last_signal_ts`` is ISO-8601 UTC, or
    ``None`` when no signal has ever been recorded for this agent.
    """
    async with _get_fingerprint_lock():
        bucket = _fingerprint_window.get(agent_id)
        last_ts = _fingerprint_last_signal.get(agent_id)
        if bucket is None or len(bucket) == 0:
            window_size = 0
            rejection_rate = 0.0
            burned = False
        else:
            window_size = len(bucket)
            rejected = sum(1 for v in bucket if v)
            rejection_rate = rejected / window_size
            burned = _is_burned_locked(bucket)
    return {
        "window_size": window_size,
        "rejection_rate": rejection_rate,
        "burned": burned,
        "last_signal_ts": _iso8601_utc(last_ts) if last_ts else None,
    }


async def _reset_fingerprint_window(agent_id: str) -> bool:
    """Operator-triggered reset.  Returns True if state was cleared.

    Called after the operator manually rotates the BrowserForge fingerprint
    (no auto-rotation per §22 — see header note).  Drops both the rolling
    window and the last-signal timestamp; the next post-solve outcome
    starts a fresh window.
    """
    async with _get_fingerprint_lock():
        had_state = (
            agent_id in _fingerprint_window
            or agent_id in _fingerprint_last_signal
        )
        _fingerprint_window.pop(agent_id, None)
        _fingerprint_last_signal.pop(agent_id, None)
    return had_state


# ── §22 — fingerprint audit aggregator (per-minute, no URL leak) ──────────
#
# Mirrors ``_drain_captcha_audit`` / ``_drain_session_audit`` — aggregated
# per minute keyed by ``(agent_id, signal, page_origin)`` so a stuck retry
# loop on one site doesn't drown out the rest of the fleet.  ``page_origin``
# is the redacted netloc only (no path, no query, no fragment) so the
# dashboard event stream cannot be used to infer which sites the agent is
# logged into.  Cookie values, full URLs, and query parameters are NEVER
# included — same privacy posture as the §20 session audit.
#
# Signals: ``rejected`` / ``accepted`` (page-state monitor outcome) and
# ``fingerprint_burn`` (one-shot when a fresh outcome trips the threshold).
_fingerprint_audit_lock: asyncio.Lock | None = None
_fingerprint_audit_lock_loop: asyncio.AbstractEventLoop | None = None
# {(agent_id, signal, page_origin): {"count": int, "first_ts": float}}
_fingerprint_audit_buckets: dict[tuple[str, str, str], dict] = {}


def _get_fingerprint_audit_lock() -> asyncio.Lock:
    """Return a fingerprint audit lock bound to the active event loop."""
    global _fingerprint_audit_lock, _fingerprint_audit_lock_loop
    loop = asyncio.get_running_loop()
    if (
        _fingerprint_audit_lock is None
        or _fingerprint_audit_lock_loop is not loop
    ):
        _fingerprint_audit_lock = asyncio.Lock()
        _fingerprint_audit_lock_loop = loop
    return _fingerprint_audit_lock


def _page_origin_for_audit(page_url: str) -> str:
    """Reduce a URL to ``netloc`` only (no path / query / fragment).

    Returns ``""`` on any parse failure so the audit event still records
    the signal even when the source URL is malformed.  Mirrors the
    "counts only, no values" privacy posture used by the §20 session
    audit aggregator.  Userinfo (``user:pass@host``) is stripped.
    """
    if not page_url:
        return ""
    try:
        parsed = urlparse(page_url)
    except Exception:
        return ""
    host = parsed.hostname or ""
    if not host:
        return ""
    if parsed.port:
        return f"{host}:{parsed.port}"
    return host


async def _record_fingerprint_audit_event(
    agent_id: str, signal: str, page_origin: str,
) -> None:
    """Aggregate a fingerprint signal into the per-minute bucket.

    ``signal`` is one of ``"rejected"`` / ``"accepted"`` / ``"fingerprint_burn"``.
    ``page_origin`` MUST already be reduced to a redacted netloc — callers
    use :func:`_page_origin_for_audit` on the live page URL before this
    call.  Specific URLs are NEVER stored.
    """
    async with _get_fingerprint_audit_lock():
        key = (agent_id, signal, page_origin)
        bucket = _fingerprint_audit_buckets.get(key)
        now = time.time()
        if bucket is None:
            _fingerprint_audit_buckets[key] = {
                "count": 1,
                "first_ts": now,
            }
        else:
            bucket["count"] += 1


async def _drain_fingerprint_audit() -> list[dict]:
    """Atomically swap fingerprint audit buckets and return drained payloads.

    Called from :meth:`BrowserManager._emit_metrics` once per minute. Each
    returned dict is one EventBus payload with ``type="fingerprint_event"``
    so the dashboard can route it to a dedicated panel without clashing
    with the captcha / session audit streams.
    """
    async with _get_fingerprint_audit_lock():
        if not _fingerprint_audit_buckets:
            return []
        buckets = dict(_fingerprint_audit_buckets)
        _fingerprint_audit_buckets.clear()
    drained = []
    for (agent_id, signal, page_origin), info in buckets.items():
        # ``agent_id`` (not ``agent``) — the dashboard metrics poller
        # routes browser-service events into the per-agent EventBus by
        # this exact field name, same convention as the captcha and
        # session audit paths.
        drained.append({
            "type": "fingerprint_event",
            "agent_id": agent_id,
            "signal": signal,
            "page_origin": page_origin,
            "count": info["count"],
            "first_ts": info["first_ts"],
        })
    return drained


def _kind_confidence(kind: str) -> str:
    """Default ``solver_confidence`` for a no-solver path, derived from how
    confidently we classified the *kind*. Placeholder kinds (§11.1 / §11.3)
    map to "low"; firmly-disambiguated kinds map to "high"; "unknown" → "low".
    """
    if kind in _FIRM_KINDS:
        return "high"
    return "low"


def _is_httpx_timeout(exc: BaseException) -> bool:
    """Detect httpx.TimeoutException without making httpx a hard import in
    this module. ``httpx`` is already a transitive dep (used by the bundled
    ``CaptchaSolver`` in src/browser/captcha.py); third-party solver
    subclasses may also raise its exceptions. The local import keeps service
    startup cheap when httpx hasn't been pulled in yet.
    """
    try:
        import httpx  # noqa: PLC0415 — local-import is intentional
    except Exception:
        return False
    return isinstance(exc, httpx.TimeoutException)


def _captcha_envelope(
    *,
    kind: str,
    solver_attempted: bool,
    solver_outcome: str,
    solver_confidence: str,
    next_action: str,
    injection_failure_reason: str | None = None,
) -> dict:
    """Build the §11.13 ``data`` block for a found-captcha case.

    ``injection_failure_reason`` must be ``None`` unless
    ``solver_outcome == "injection_failed"``. The field is always present
    (set to ``None`` rather than absent) so downstream consumers can rely
    on a stable shape.
    """
    return {
        "captcha_found": True,
        "kind": kind,
        "solver_attempted": solver_attempted,
        "solver_outcome": solver_outcome,
        "injection_failure_reason": injection_failure_reason,
        "solver_confidence": solver_confidence,
        "next_action": next_action,
    }


def _with_legacy_fields(envelope: dict) -> dict:
    """Soft-deprecated shim — populate the old ``type`` / ``message`` fields
    so agents whose rules still match against the freeform string keep
    working. New agents should read the structured fields directly.

    Applied uniformly to BOTH the captcha-found and no-captcha cases — the
    old shape was ``{captcha_found: false, message: "No CAPTCHA detected"}``
    (no ``type`` field) so we only re-add ``message`` for that branch.
    """
    out = dict(envelope)
    if not out.get("captcha_found"):
        out.setdefault("message", "No CAPTCHA detected")
        return out
    kind = out.get("kind", "unknown")
    next_action = out.get("next_action", "notify_user")
    out["type"] = kind  # kept for back-compat; was a CSS selector before.
    out["message"] = (
        f"CAPTCHA detected of kind {kind}; next_action: {next_action}"
    )
    return out


_ACTIONABLE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "searchbox", "slider", "spinbutton", "switch", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "treeitem",
})

_CONTEXT_ROLES = frozenset({
    "heading", "img", "dialog", "alertdialog", "alert",
    "listbox", "tree", "grid", "toolbar", "menu", "status",
})

# §7.7 semantic filters. Each value maps to a frozenset of roles that
# admit nodes into the snapshot output. ``None`` (no filter passed) is
# handled separately and falls back to the historical default of
# ``_ACTIONABLE_ROLES ∪ _CONTEXT_ROLES``.
_FILTER_INPUTS = frozenset({
    "textbox", "searchbox", "checkbox", "radio", "combobox",
    "slider", "spinbutton", "switch",
})
_FILTER_HEADINGS = frozenset({"heading"})
_FILTER_LANDMARKS = frozenset({
    "navigation", "main", "complementary", "banner", "contentinfo",
    "form", "region", "dialog", "alertdialog",
})
_FILTER_PRESETS: dict[str, frozenset[str]] = {
    "actionable": _ACTIONABLE_ROLES,
    "inputs": _FILTER_INPUTS,
    "headings": _FILTER_HEADINGS,
    "landmarks": _FILTER_LANDMARKS,
}


def _resolve_filter_roles(filter_name: str | None) -> frozenset[str] | None:
    """Map a ``filter`` parameter to the role frozenset to admit.

    ``None`` ⇒ ``None`` (caller falls back to the historical default).
    Unknown name ⇒ raises ``ValueError`` with a helpful list of valid
    options. Empty string is treated like ``None`` so callers passing
    JSON ``""`` from a UI default don't accidentally narrow the result.

    Case-insensitive — LLMs frequently capitalize parameter values. A
    naive case-sensitive lookup would have rejected ``"Actionable"`` /
    ``"INPUTS"`` with ``invalid_input``; the agent then has to retry.
    """
    if filter_name is None:
        return None
    if not isinstance(filter_name, str):
        raise ValueError(
            f"filter must be a string, got {type(filter_name).__name__}"
        )
    key = filter_name.strip().lower()
    if key == "":
        return None
    preset = _FILTER_PRESETS.get(key)
    if preset is None:
        valid = ", ".join(sorted(_FILTER_PRESETS))
        raise ValueError(
            f"Unknown filter {filter_name!r}; valid options: {valid}"
        )
    return preset


def _iso8601_utc(ts_unix: float) -> str:
    """Format a unix timestamp as ISO-8601 UTC with seconds precision.

    Returns ``""`` when ``ts_unix`` is falsy (0 / None) so callers don't
    have to special-case the absence of a timestamp.
    """
    if not ts_unix:
        return ""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(ts_unix), tz=timezone.utc).isoformat(
        timespec="seconds",
    ).replace("+00:00", "Z")


def _err(
    code: str, message: str, retry_after_ms: int | None = None,
) -> dict:
    """Build a Phase 5 §2.3 structured error envelope.

    Per §2.3 the ``retry_after_ms`` field is always present
    (``null`` when not applicable), so callers can rely on the shape.
    """
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "retry_after_ms": retry_after_ms,
        },
    }


_MAX_SNAPSHOT_ELEMENTS = 200
_MAX_WALK_DEPTH = 50

_MAX_FRAME_NESTING = 3

# Token shape for frame_id values produced by `_register_frame`. Used
# in `_resolve_frame_arg` to distinguish a detached/unknown frame_id
# token (raise ref_stale) from a URL-substring miss (return None).
_FRAME_ID_RE = re.compile(r"^f-[0-9a-f]{8}$")


# §7.2 v2 format — depth indent is capped so a 50-deep DOM doesn't
# explode into 100-character indents. Anything past this depth shares
# the cap-line indent (still distinguishable as "deep" but not bytewise
# punishing).
_V2_MAX_INDENT_DEPTH = 4
_V2_NO_LANDMARK_KEY = ""


def _format_snapshot_v2(
    lines: list[str],
    entries: list[tuple[str, str, str, str, str, int]],
) -> str:
    """Render the snapshot in §7.2 ``v2`` format.

    Group entries by landmark and emit each group under a section
    header (``# nav: Top``) instead of suffixing every element with
    ``(navigation: Top)``. Indent depth is capped at
    :data:`_V2_MAX_INDENT_DEPTH`.

    Modal-mode preamble lines (those starting with ``**`` in the v1
    output) are passed through verbatim ahead of the section blocks
    so the agent still sees the modal context.

    Args:
        lines: the v1 line list — only used for ``**`` preamble lines.
        entries: per-element tuples
            ``(ref_id, role, name, attr_str, landmark, depth)``.

    Returns the rendered string. Always begins with the
    ``# snapshot-v2`` version marker so parsers can detect the format
    without out-of-band signaling.
    """
    if not entries:
        # Empty result. Still emit the marker so a parser using the
        # first line for routing decisions doesn't trip. Also pass
        # through any modal-banner preamble that v1 produced.
        preamble = [ln for ln in lines if ln.startswith("**")]
        if preamble:
            return "# snapshot-v2\n" + "\n".join(preamble)
        return "# snapshot-v2\n(no interactive elements)"

    # Modal-banner preamble (lines starting ``**``) precedes the
    # element output. Keeps the modal-scoped warning visible.
    preamble = [ln for ln in lines if ln.startswith("**")]

    # Preserve insertion order — the dict-by-design key order matches
    # the order entries were emitted, which is doc order.
    groups: dict[str, list[tuple[str, str, str, str, int]]] = {}
    for ref_id, role, name, attr_str, landmark, depth in entries:
        key = landmark or _V2_NO_LANDMARK_KEY
        groups.setdefault(key, []).append(
            (ref_id, role, name, attr_str, depth),
        )

    out: list[str] = ["# snapshot-v2"]
    out.extend(preamble)
    for landmark_key, group_entries in groups.items():
        if landmark_key == _V2_NO_LANDMARK_KEY:
            out.append("# (no landmark)")
        else:
            # Sanitize newlines in landmark keys: a malicious DOM
            # node with ``aria-label="x\n# fake-section: pwn"`` would
            # otherwise inject a phantom section header into the
            # parsed output (operator scripts reading the v2 format
            # split on '#' prefixes). Replace with single spaces so
            # the structural marker stays one-line.
            out.append(f"# {_v2_strip_newlines(landmark_key)}")
        for ref_id, role, name, attr_str, depth in group_entries:
            indent_depth = min(depth, _V2_MAX_INDENT_DEPTH)
            indent = "  " * indent_depth
            # Same sanitization on per-element name + attr_str so an
            # accessible-name with embedded ``\n# fake`` can't escape
            # to a fake section header.
            safe_name = _v2_strip_newlines(name)
            safe_attr = _v2_strip_newlines(attr_str)
            # Iframe stubs (ref_id == "") are content nodes, not
            # clickable handles — render without the ``[ref_id]``
            # prefix so the v2 surface matches v1 (``- iframe "name"``).
            if ref_id == "":
                out.append(f"{indent}- {role} \"{safe_name}\"{safe_attr}")
            else:
                out.append(f"{indent}- [{ref_id}] {role} \"{safe_name}\"{safe_attr}")

    return "\n".join(out)


def _v2_strip_newlines(s: str) -> str:
    """Collapse ``\\n``/``\\r`` to single spaces.

    v2 promotes ``# ``-prefixed lines to structural meaning. A DOM
    node with newlines in its accessible name or landmark would
    otherwise inject phantom section headers into the parsed
    snapshot. Cheap belt-and-braces over the JS-walker side which
    only ``.trim()``s whitespace endpoints.
    """
    if not s:
        return s
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

# Block schemes that could expose local files or browser internals.
# about: covers about:logins (saved passwords), about:config, etc.
# moz-extension: / chrome-extension: cover installed extensions.
_BLOCKED_URL_SCHEMES = frozenset({
    "file", "javascript", "data", "blob",
    "about", "moz-extension", "chrome-extension", "chrome",
})
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})



_MAX_WAIT_MS = 10000  # 10 seconds max wait after navigation
_MAX_SCROLL_PX = 10000  # 10000 pixels max per scroll call
_CLICK_TIMEOUT_MS = 10000  # 10 seconds — SPAs like X need time for animations/overlays
_WAIT_FOR_TIMEOUT_MS = 30000  # 30 seconds max for wait_for_element
_AGENT_ID_RE = re.compile(AGENT_ID_RE_PATTERN)
_VALID_WAIT_UNTIL = frozenset({"domcontentloaded", "load", "networkidle", "commit"})
# Characters that mark a natural word/clause boundary in typed text.
# After one of these, the next character gets a higher think-pause probability
# to model the hesitation a human feels when starting the next word or sentence.
_WORD_BOUNDARY_CHARS = frozenset(" ,.:;!?\n\t")
# Adjacent keys on QWERTY layout for natural typo injection.
# Includes same-row neighbors and diagonal keys above/below.
_TYPO_NEIGHBORS: dict[str, str] = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'weadxz', 'd': 'ersfxc', 'f': 'rtdgcv',
    'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm',
    'l': 'opk',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}
# Roles where aria-disabled="true" should NOT block click attempts.
# SPA frameworks (X/Twitter, Gmail) keep aria-disabled on buttons/links while
# handling clicks via JS — the visual state and handler are the source of truth,
# not the ARIA attribute.  Intentionally narrow: menuitem, switch, option are
# excluded because force-clicking genuinely disabled items in those roles causes
# unwanted side-effects (selecting unavailable options, toggling locked switches).
_ARIA_FORCE_ROLES = frozenset({"button", "link"})
# Button names (lowercased) that indicate a modal close/dismiss action.
# When clicking these inside a modal doesn't dismiss it, we fall back to
# pressing Escape — Camoufox's patched Firefox has known issues where
# pointer events on modal close buttons silently fail in some SPAs
# (X/Twitter compose modal, etc.).
_MODAL_CLOSE_NAMES = frozenset({"close", "×", "✕", "✖"})
# Playwright key names → xdotool key names. Playwright follows the KeyboardEvent.key
# spec; xdotool uses X11 keysym names. Only keys that differ need mapping.
_PLAYWRIGHT_TO_XDOTOOL = {
    "Enter": "Return", "Backspace": "BackSpace", "Delete": "Delete",
    "Space": "space", "ArrowUp": "Up", "ArrowDown": "Down",
    "ArrowLeft": "Left", "ArrowRight": "Right", "PageUp": "Prior",
    "PageDown": "Next", "Control": "ctrl", "Shift": "shift",
    "Alt": "alt", "Meta": "super",
}
# CSS selector for modal dialog detection via DOM queries.
# Used in both snapshot() (to scope the a11y tree) and _locator_from_ref()
# (to scope click/type locators). Must stay in sync — hence a single constant.
_MODAL_SELECTOR = (
    '[role="dialog"]:not([aria-hidden="true"]), '
    '[aria-modal="true"]:not([aria-hidden="true"]), '
    'dialog[open]'
)


# Implicit role map — single source of truth shared between the JS a11y
# walker and the Python-side shadow-path resolver. Tag names are uppercase
# (matches DOM ``Element.tagName``). Injected into the JS body via
# ``json.dumps()`` so the JS literally sees this Python dict's contents.
_IMPLICIT_ROLE_MAP: dict[str, str] = {
    "BUTTON": "button", "TEXTAREA": "textbox", "SELECT": "combobox",
    "OPTION": "option",
    "IMG": "img", "H1": "heading", "H2": "heading", "H3": "heading",
    "H4": "heading", "H5": "heading", "H6": "heading", "DIALOG": "dialog",
    "NAV": "navigation", "MAIN": "main", "HEADER": "banner",
    "FOOTER": "contentinfo",
    "ASIDE": "complementary", "FORM": "form",
}


# Shared JS source for accessible-name extraction. Injected into BOTH the
# walker (which produces ``name`` for emitted nodes) and the Stage-2
# resolver (which matches candidates by name inside a shadowRoot). Single
# source of truth — if these diverged, an element snapshotted with name
# from e.g. ``placeholder`` would never re-resolve via Stage 2 and the
# agent would see spurious ``ref_stale``. Defines two functions:
#
#   implicitRoleFor(el)              — implicit ARIA role mapping
#   accessibleName(el, role)         — same priority chain as W3C ACCNAME:
#       aria-label → aria-labelledby → label[for=] / wrapping <label>
#       → alt (img) → placeholder/title (input/textarea/select)
#       → textContent for name-from-content roles → title fallback
#
# IMPLICIT and INPUT_ROLES must already be in scope at the injection site.
_JS_NAME_HELPERS = r"""
    function implicitRoleFor(el) {
        const r = el.getAttribute('role');
        if (r) return r.split(/\s+/)[0].toLowerCase();
        if (el.tagName === 'A') return el.hasAttribute('href') ? 'link' : null;
        if (el.tagName === 'INPUT') return INPUT_ROLES[(el.type||'text').toLowerCase()] || null;
        if (el.getAttribute('contenteditable') === 'true') return 'textbox';
        return IMPLICIT[el.tagName] || null;
    }
    function accessibleName(el, role) {
        let n = el.getAttribute('aria-label');
        if (n) return n.trim();
        const by = el.getAttribute('aria-labelledby');
        if (by) {
            const root = el.getRootNode();
            const lookup = (root && root.getElementById) ? root : document;
            const t = by.split(/\s+/).map(id => {
                const ref = lookup.getElementById ? lookup.getElementById(id) : document.getElementById(id);
                return ref ? ref.textContent.trim() : '';
            }).filter(Boolean).join(' ');
            if (t) return t;
        }
        if (el.tagName === 'IMG') return (el.alt || '').trim();
        if (['INPUT','TEXTAREA','SELECT'].includes(el.tagName)) {
            if (el.id) {
                const root = el.getRootNode();
                const scope = (root && root.querySelector) ? root : document;
                const lbl = scope.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (lbl) return lbl.textContent.trim().slice(0, 200);
            }
            const wrap = el.closest('label');
            if (wrap) {
                const c = wrap.cloneNode(true);
                c.querySelectorAll('input,textarea,select').forEach(i => i.remove());
                const t = c.textContent.trim();
                if (t) return t.slice(0, 200);
            }
            return (el.placeholder || el.title || '').trim();
        }
        if (['button','link','tab','menuitem','menuitemcheckbox','menuitemradio',
            'switch','option','treeitem','heading','alert','alertdialog','dialog',
            'listbox','toolbar','menu','status'
        ].includes(role)) {
            const t = el.textContent;
            if (t) { const s = t.trim(); if (s) return s.slice(0, 200); }
        }
        return (el.title || '').trim();
    }
"""


def _build_js_a11y_tree() -> str:
    """Build the JS a11y walker source with the implicit role map injected.

    The walker descends ``el.shadowRoot`` when ``shadowRoot.mode === 'open'``
    (closed roots are unreachable per the spec). Each emitted node carries
    a ``shadow_path`` array; on the page side this is empty for light-DOM
    nodes and accumulates ``{selector, occurrence, discriminator}`` triples
    as the walker crosses shadow boundaries.

    Discriminator priority: ``data-testid`` > stable ``id`` (UUID-shaped
    rejected) > structural fingerprint of host (tagName + className +
    childElementCount). Always a string — ``ShadowHop.discriminator`` is
    guaranteed non-empty by the JS side.
    """
    implicit_json = json.dumps(_IMPLICIT_ROLE_MAP)
    return r"""((rootEl) => {
    const MAX_WALK_DEPTH = __MAX_WALK_DEPTH__;
    const ACTIONABLE = new Set([
        'button','link','textbox','checkbox','radio','combobox','searchbox',
        'slider','spinbutton','switch','tab','menuitem','menuitemcheckbox',
        'menuitemradio','option','treeitem'
    ]);
    const CONTEXT = new Set([
        'heading','img','dialog','alertdialog','alert',
        'listbox','tree','grid','toolbar','menu','status'
    ]);
    const LANDMARK = new Set([
        'navigation','main','complementary','banner','contentinfo',
        'form','region','dialog','alertdialog'
    ]);
    const ROLES = new Set([...ACTIONABLE, ...CONTEXT, ...LANDMARK]);
    const IMPLICIT = __IMPLICIT_ROLE_MAP__;
    const INPUT_ROLES = {
        text:'textbox',email:'textbox',url:'textbox',tel:'textbox',
        password:'textbox',search:'searchbox',
        checkbox:'checkbox',radio:'radio',
        range:'slider',number:'spinbutton',
        submit:'button',reset:'button',button:'button'
    };
    const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const REACT_ID_RE = /^(:r|r:|:R)[0-9a-zA-Z]+:?$/;
    function isStableId(id) {
        if (!id) return false;
        if (UUID_RE.test(id)) return false;
        if (REACT_ID_RE.test(id)) return false;
        if (id.length >= 16 && /^[0-9a-f]+$/i.test(id)) return false;
        return true;
    }
    function bestStableId(el) {
        const t = el.getAttribute('data-testid')
            || el.getAttribute('data-test')
            || el.getAttribute('data-qa');
        if (t) return 'testid:' + t;
        const id = el.getAttribute('id');
        if (id && isStableId(id)) return 'id:' + id;
        // Structural fingerprint excludes class/style/data-* —
        // those flip on hover/focus/aria-state and would break the
        // discriminator across snapshots. tagName + childElementCount
        // + sorted attribute names (state-free subset) is stable.
        const attrNames = [];
        for (const a of el.attributes) {
            const n = a.name;
            if (n === 'class' || n === 'style') continue;
            if (n.startsWith('data-')) continue;
            attrNames.push(n);
        }
        attrNames.sort();
        const fp = el.tagName + '|' + (el.childElementCount || 0)
            + '|' + attrNames.join(',');
        return 'fp:' + fp;
    }
    function cssPath(host) {
        const tag = host.tagName.toLowerCase();
        const id = host.getAttribute('id');
        if (id && isStableId(id)) return tag + '#' + CSS.escape(id);
        const t = host.getAttribute('data-testid');
        if (t) return tag + '[data-testid="' + t.replace(/"/g, '\\"') + '"]';
        return tag;
    }
    // Walk-scoped cache for ``querySelectorAll`` results so a page with
    // N shadow hosts at the same root no longer triggers O(N²) work
    // (each emit ran a fresh ``scopeRoot.querySelectorAll(selector)``).
    // Keyed by (scopeRoot, selector); both keys are alive only for the
    // duration of one walker invocation. ``Map`` keyed on the live
    // root reference is safe because the walker runs synchronously
    // and the scopeRoot is held by caller frames during the walk.
    const occurrenceCache = new Map();
    function siblingOccurrence(host, selector, scopeRoot) {
        // Per-hop scope mirrors the stage-1 resolver: first hop uses
        // ``document``, subsequent hops use the parent ``shadowRoot``
        // (since stage-1 walks ``root = host.shadowRoot`` after each
        // hop and queries ``root.querySelectorAll(hop.selector)``).
        // Walker passes ``document`` for the top-level call and the
        // parent shadowRoot when descending into a nested host.
        const root = scopeRoot || document;
        let scopeMap = occurrenceCache.get(root);
        if (!scopeMap) {
            scopeMap = new Map();
            occurrenceCache.set(root, scopeMap);
        }
        let candidates = scopeMap.get(selector);
        if (!candidates) {
            candidates = Array.from(root.querySelectorAll(selector));
            scopeMap.set(selector, candidates);
        }
        const idx = candidates.indexOf(host);
        return idx === -1 ? 0 : idx;
    }
__NAME_HELPERS__
    // Walker name/role wrappers — single source of truth shared with
    // the stage-2 resolver.
    const getRole = implicitRoleFor;
    const getName = accessibleName;
    function isVisible(el) {
        if (el.getAttribute && el.getAttribute('aria-hidden') === 'true') return false;
        const s = getComputedStyle(el);
        if (s.visibility === 'hidden' || s.visibility === 'collapse') return false;
        if (parseFloat(s.opacity) === 0) return false;
        if (!el.offsetParent && el !== document.body && el !== document.documentElement) {
            if (s.display === 'none') return false;
            if (s.position !== 'fixed' && s.position !== 'sticky') return false;
        }
        return true;
    }
    function isCrossOriginFrame(el) {
        try {
            if (el.contentDocument === null) return true;
        } catch (e) {
            return true;
        }
        try {
            const src = el.getAttribute('src') || '';
            if (!src || src === 'about:blank') return false;
            const u = new URL(src, window.location.href);
            return u.origin !== window.location.origin;
        } catch (e) {
            return true;
        }
    }
    function walk(el, d, parentLandmark, shadowPath, currentRoot) {
        if (d > MAX_WALK_DEPTH || !el || el.nodeType !== 1) return null;
        const tag = el.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE')
            return null;
        if (!isVisible(el)) return null;
        if (tag === 'IFRAME' || tag === 'FRAME') {
            const src = el.getAttribute('src') || '';
            const title = (el.getAttribute('title') || '').trim();
            const opaque = isCrossOriginFrame(el);
            // srcdoc / anonymous iframes (no src, or src=about:blank)
            // get an empty ``frame_url``; the Python descent uses
            // ``iframe_index`` (sibling position among same-document
            // iframes) as the descent key for those, and as a tie-
            // breaker when two iframes legitimately share a URL.
            let iframeIndex = -1;
            try {
                const allFrames = el.ownerDocument.querySelectorAll(
                    'iframe, frame'
                );
                for (let i = 0; i < allFrames.length; i++) {
                    if (allFrames[i] === el) { iframeIndex = i; break; }
                }
            } catch (e) {
                // ownerDocument can be null in pathological cases — fall
                // through with -1 so the Python side treats this as
                // "no index hint available".
            }
            const stub = {
                role: 'iframe',
                name: opaque ? 'cross-origin' : (title || src).slice(0, 200),
                frame_url: src,
                opaque: opaque,
                iframe_index: iframeIndex,
            };
            if (parentLandmark) stub.landmark = parentLandmark;
            return stub;
        }
        const role = getRole(el);
        let childLandmark = parentLandmark;
        if (role && LANDMARK.has(role)) {
            const lname = getName(el, role);
            childLandmark = lname ? role + ': ' + lname.slice(0, 50) : role;
        }
        const children = [];
        for (const child of el.children) {
            const r = walk(child, d + 1, childLandmark, shadowPath, currentRoot);
            if (r) children.push(r);
        }
        if (el.shadowRoot && el.shadowRoot.mode === 'open') {
            const sel = cssPath(el);
            // Scope occurrence to ``currentRoot`` (document for the
            // first hop, parent shadowRoot for subsequent hops). Mirrors
            // stage-1 resolver scope so nested hosts index identically
            // on both sides.
            const occ = siblingOccurrence(el, sel, currentRoot);
            const disc = bestStableId(el);
            const nextPath = shadowPath.concat([{
                selector: sel, occurrence: occ, discriminator: disc,
            }]);
            for (const child of el.shadowRoot.children) {
                const r = walk(child, d + 1, childLandmark, nextPath, el.shadowRoot);
                if (r) children.push(r);
            }
        }
        if (!role || !ROLES.has(role)) {
            if (!children.length) return null;
            if (children.length === 1) return children[0];
            return { role: 'none', name: '', children };
        }
        const nd = { role, name: getName(el, role) };
        if (shadowPath.length) nd.shadow_path = shadowPath;
        if (parentLandmark) nd.landmark = parentLandmark;
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') nd.disabled = true;
        const chkRoles = ['checkbox','radio','switch','menuitemcheckbox','menuitemradio'];
        if (chkRoles.includes(role)) {
            nd.checked = !!(el.checked) || el.getAttribute('aria-checked') === 'true';
        }
        if (el.getAttribute('aria-selected') === 'true') nd.selected = true;
        if ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && el.value !== '') {
            nd.value = String(el.value).slice(0, 500);
        }
        if (el.getAttribute('contenteditable') === 'true' && el.textContent) {
            const cv = el.textContent.trim();
            if (cv && !nd.value) nd.value = cv.slice(0, 500);
        }
        if (children.length) nd.children = children;
        return nd;
    }
    const start = rootEl || document.body || document.documentElement;
    const tree = walk(start, 0, null, [], document);
    if (rootEl) return tree || { role: 'none', name: '', children: [] };
    if (!tree) return { role: 'WebArea', name: document.title || '', children: [] };
    if (tree.role === 'none')
        return { role: 'WebArea', name: document.title || '', children: tree.children || [] };
    return { role: 'WebArea', name: document.title || '', children: [tree] };
})""".replace(
        "__NAME_HELPERS__", _JS_NAME_HELPERS,
    ).replace(
        "__IMPLICIT_ROLE_MAP__", implicit_json,
    ).replace(
        "__MAX_WALK_DEPTH__", str(_MAX_WALK_DEPTH),
    )


# ── JS-based accessibility tree builder ──────────────────────────────────
# Fallback when page.accessibility.snapshot() is unavailable (Camoufox
# bundles a Playwright version that removed or never exposed the API).
# Walks the DOM using standard APIs (getAttribute, getComputedStyle) and
# returns the same {role, name, children, disabled, ...} tree structure
# that the Python _walk() function expects.
#
# Descends ``el.shadowRoot`` when ``shadowRoot.mode === 'open'``; closed
# shadow roots are unreachable by web spec and remain invisible. Emitted
# nodes inside shadow DOM carry a ``shadow_path`` array of
# ``{selector, occurrence, discriminator}`` hops that the Python side
# folds into ``RefHandle.shadow_path`` for resolution.
#
# Called as:
#   page.evaluate(_JS_A11Y_TREE)          — full page tree
#   element_handle.evaluate(_JS_A11Y_TREE) — scoped to element
_JS_A11Y_TREE = _build_js_a11y_tree()


# Phase 6 §9.3: viewport-relative click pre-check.
# Returns null when document.elementFromPoint(x, y) yields no element
# (off-page or transparent root). Otherwise returns a JSON-friendly dict
# with the hit element's tag/role/accessible-name plus a mask trace.
#
# Why we walk inside-out and stop at the first ``pointer-events: auto``
# ancestor: per the CSS spec, ``pointer-events`` does NOT inherit; an
# inner ``pointer-events: auto`` re-enables hit-testing for that subtree
# even when an outer ancestor is ``pointer-events: none``. Naïve
# "scan all ancestors for pointer-events: none" would falsely flag the
# common pattern of a ``pointer-events: none`` overlay container with
# ``pointer-events: auto`` on its interactive children. visibility/
# display/opacity hide the subtree outright, so any ancestor matching
# those is reported as the masking element regardless of pointer-events.
_JS_ELEMENT_FROM_POINT = r"""([x, y]) => {
    const el = document.elementFromPoint(x, y);
    if (!el) return null;
    const elementSelector = (node) => {
        if (!node || node.nodeType !== 1) return null;
        const tag = node.tagName ? node.tagName.toLowerCase() : "";
        if (node.id) return tag + "#" + node.id;
        const cls = (node.className && typeof node.className === "string")
            ? node.className.trim().split(/\s+/).filter(Boolean).slice(0, 2).join(".")
            : "";
        return cls ? tag + "." + cls : tag;
    };
    const accName = (node) => {
        try {
            const aria = node.getAttribute && node.getAttribute("aria-label");
            if (aria) return aria;
            const title = node.getAttribute && node.getAttribute("title");
            if (title) return title;
            const txt = node.innerText || node.textContent || "";
            return txt.slice(0, 80);
        } catch (e) { return ""; }
    };
    let masked_by = null;
    let mask_reason = "";
    let pointer_events_decided = false;
    let cur = el;
    while (cur && cur !== document.body && cur !== document.documentElement) {
        let cs;
        try { cs = window.getComputedStyle(cur); } catch (e) { cs = null; }
        if (cs) {
            // display:none and visibility:hidden mask the subtree
            // regardless of pointer-events. Check these unconditionally.
            if (cs.display === "none") {
                masked_by = elementSelector(cur);
                mask_reason = "display";
                break;
            }
            if (cs.visibility === "hidden") {
                masked_by = elementSelector(cur);
                mask_reason = "visibility";
                break;
            }
            // opacity:0 — fully transparent, treat as masked.
            if (cs.opacity === "0") {
                masked_by = elementSelector(cur);
                mask_reason = "opacity";
                break;
            }
            // pointer-events: walk inside-out. The closest ancestor to
            // the hit element with ``pointer-events: auto`` re-enables
            // hit-testing for everything below it; stop checking
            // pointer-events at that boundary. A ``none`` ancestor seen
            // before any ``auto`` boundary masks the click.
            if (!pointer_events_decided && cs.pointerEvents) {
                if (cs.pointerEvents === "auto") {
                    pointer_events_decided = true;
                } else if (cs.pointerEvents === "none") {
                    masked_by = elementSelector(cur);
                    mask_reason = "pointer-events";
                    break;
                }
            }
        }
        cur = cur.parentElement;
    }
    return {
        tag: el.tagName ? el.tagName.toLowerCase() : "",
        role: (el.getAttribute && el.getAttribute("role")) || null,
        name: accName(el),
        masked_by: masked_by,
        mask_reason: mask_reason,
    };
}"""


# Stage-1 (walk shadow_path → return inner shadowRoot) and Stage-2
# (role+name match inside that shadowRoot → ElementHandle) for the
# Playwright-correct two-stage shadow resolver. ``get_by_role`` does NOT
# pierce shadow boundaries, so non-empty ``shadow_path`` falls through
# this evaluate_handle pair instead of the locator API.
# Sentinel key for stage-1 errors. Uses a unique prefixed name rather than
# bare ``error`` so a page that happens to set ``shadowRoot.error = 'foo'``
# (or any expando the page might attach) cannot masquerade as a resolver
# error. Stage-2 also rejects any rooted shadow root carrying this key as
# an expando — defence-in-depth, since the property is on a ShadowRoot
# object that the page does not normally write to.
_JS_RESOLVER_ERROR_KEY = "__OL_RESOLVER_ERROR__"

_JS_SHADOW_RESOLVE_STAGE1 = r"""(args) => {
    const ERR_KEY = "__OL_RESOLVER_ERROR__";
    const path = JSON.parse(args.path);
    // ``scope_root`` (optional): when the ref was captured during a
    // modal-scoped snapshot, ``_locator_from_ref`` patches a modal
    // selector onto the handle. Stage-1 must start its walk at the
    // modal element instead of ``document`` so a same-selector shadow
    // host living *outside* the dialog cannot match. Without this,
    // shadow refs effectively bypass modal scoping, and clicks could
    // land on a duplicate-named element behind the overlay.
    let root;
    if (args.scope_root) {
        root = document.querySelector(args.scope_root);
        if (!root) {
            const e = {}; e[ERR_KEY] = "scope_root_missing"; return e;
        }
    } else {
        root = document;
    }
    for (const hop of path) {
        const candidates = root.querySelectorAll(hop.selector);
        const host = candidates[hop.occurrence];
        if (!host || !host.shadowRoot) {
            const e = {}; e[ERR_KEY] = "stale_host_missing"; return e;
        }
        function isStableId(id) {
            if (!id) return false;
            if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)) return false;
            if (/^(:r|r:|:R)[0-9a-zA-Z]+:?$/.test(id)) return false;
            if (id.length >= 16 && /^[0-9a-f]+$/i.test(id)) return false;
            return true;
        }
        function bestStableId(el) {
            const t = el.getAttribute('data-testid')
                || el.getAttribute('data-test')
                || el.getAttribute('data-qa');
            if (t) return 'testid:' + t;
            const id = el.getAttribute('id');
            if (id && isStableId(id)) return 'id:' + id;
            const attrNames = [];
            for (const a of el.attributes) {
                const n = a.name;
                if (n === 'class' || n === 'style') continue;
                if (n.startsWith('data-')) continue;
                attrNames.push(n);
            }
            attrNames.sort();
            const fp = el.tagName + '|' + (el.childElementCount || 0)
                + '|' + attrNames.join(',');
            return 'fp:' + fp;
        }
        const got = bestStableId(host);
        if (got !== hop.discriminator) {
            const e = {}; e[ERR_KEY] = "stale_discriminator_mismatch"; return e;
        }
        root = host.shadowRoot;
    }
    return root;
}"""


def _build_js_shadow_resolve_stage2() -> str:
    """Stage-2 resolver: pick the role+name match at ``occurrence`` inside
    the ShadowRoot stage-1 returned.

    Uses the SAME ``accessibleName(el, role)`` helper as the walker
    (injected via ``__NAME_HELPERS__``). If the two diverged, an element
    snapshotted with name from e.g. ``placeholder=Search`` would never
    match Stage 2 here and the agent would see spurious ``ref_stale``.
    Defence-in-depth against page-controlled expandos: rejects any
    ``root`` carrying the resolver-error sentinel even if the call site
    forgot to pre-check it.
    """
    implicit_json = json.dumps(_IMPLICIT_ROLE_MAP)
    return r"""((root, args) => {
    const ERR_KEY = "__OL_RESOLVER_ERROR__";
    // TOCTOU sentinel distinct from null/undefined: caller maps this to
    // RefStale (transient — shadow root detached between stage 1 and
    // stage 2 evaluate_handle calls). A bare null result, by contrast,
    // means stage 2 ran successfully but the element isn't at the
    // requested occurrence inside an otherwise-live shadow root —
    // that's a real "element vanished" condition.
    if (!root) {
        const e = {}; e[ERR_KEY] = "shadow_root_detached"; return e;
    }
    if (typeof root === 'object' && ERR_KEY in root) {
        // Stage-1 already errored out; surface the same sentinel so the
        // caller can distinguish from an element-not-found null.
        return root;
    }
    const role = args.role;
    const name = args.name;
    const occurrence = args.occurrence;
    const IMPLICIT = __IMPLICIT_ROLE_MAP__;
    const INPUT_ROLES = {
        text:'textbox',email:'textbox',url:'textbox',tel:'textbox',
        password:'textbox',search:'searchbox',
        checkbox:'checkbox',radio:'radio',
        range:'slider',number:'spinbutton',
        submit:'button',reset:'button',button:'button'
    };
__NAME_HELPERS__
    // Normalize the snapshot-time name so the empty case is unambiguous:
    // the walker buckets unnamed siblings as ``(role, "", path)``. Stage 2
    // must match elements whose live accessible name is *also* empty when
    // the ref carries an empty name — otherwise unnamed elements collapse
    // with same-role *named* siblings (the previous ``if (!name) return
    // true;`` admitted all same-role candidates, returning candidates[0]
    // and landing the click on the wrong element).
    const expected = (name || '').trim();
    let candidates;
    try {
        candidates = Array.from(root.querySelectorAll('*')).filter(el => {
            const r = implicitRoleFor(el);
            if (r !== role) return false;
            // Match using the SAME accessible-name extraction the walker
            // ran when emitting the snapshot. Anything else (e.g. only
            // ``aria-label || textContent``) misses elements named via
            // ``placeholder``, ``alt``, ``aria-labelledby``, ``label[for=]``,
            // or ``title`` — producing spurious RefStale.
            const n = (accessibleName(el, role) || '').trim();
            return n === expected;
        });
    } catch (_e) {
        // ``querySelectorAll`` on a detached ShadowRoot can throw. Treat
        // as a transient detach rather than an element-not-found.
        const e = {}; e[ERR_KEY] = "shadow_root_detached"; return e;
    }
    return candidates[occurrence] || null;
})""".replace(
        "__NAME_HELPERS__", _JS_NAME_HELPERS,
    ).replace(
        "__IMPLICIT_ROLE_MAP__", implicit_json,
    )


_JS_SHADOW_RESOLVE_STAGE2 = _build_js_shadow_resolve_stage2()


def _short_ua(ua: str) -> str:
    """Compact a UA string for log output — keep the tail Firefox-version
    bit, drop the OS/locale boilerplate readers don't need at INFO level."""
    if not ua:
        return ""
    # Most useful bit is "Firefox/138.0" at the end; everything before is
    # noise for debugging fingerprint regressions.
    if "Firefox/" in ua:
        return "Firefox/" + ua.split("Firefox/", 1)[1]
    return ua[:80]


def _js_string(value: str) -> str:
    """Escape a Python string for safe interpolation into a JS literal.

    Used by the ``navigator.connection`` init-script to inject the
    per-agent ``effectiveType`` value. The values are drawn from a
    fixed pool (``"4g"`` etc.) so injection is bounded today, but
    a defensive escape costs nothing and prevents future agent-id-
    derived values from breaking out.
    """
    return ("'" + value
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n") + "'")


# §6.6 navigator.connection fallback. Defines a getter on
# ``Navigator.prototype`` that returns a frozen object matching the
# NetworkInformation API surface real Chromium-shaped browsers expose.
# ``configurable: true`` so a future Camoufox upgrade that adds native
# support can override this. Runs before any page script via
# ``BrowserContext.add_init_script``.
_NAV_CONNECTION_INIT_SCRIPT = """
(() => {{
  try {{
    if (typeof navigator !== 'undefined' && navigator.connection !== undefined) {{
      return;  // Camoufox / Firefox already exposes the API
    }}
    const fake = Object.freeze({{
      effectiveType: {effective},
      downlink: {downlink},
      rtt: {rtt},
      saveData: {save_data},
      type: 'wifi',
      addEventListener: () => {{}},
      removeEventListener: () => {{}},
      dispatchEvent: () => false,
      onchange: null,
    }});
    Object.defineProperty(Navigator.prototype, 'connection', {{
      get: () => fake,
      configurable: true,
      enumerable: true,
    }});
  }} catch (_e) {{
    // Defensive: any failure here is operator-debuggable via the
    // §6.3 navigator self-test, which will flag the missing API.
  }}
}})();
"""


def _is_empty_payload(payload: dict) -> bool:
    """True when a drain produced no activity *in this interval*.

    Used by :meth:`BrowserManager._emit_metrics` to filter out idle-
    agent payloads so the history buffer doesn't flood with no-ops.
    Only per-minute counters count here — the rolling click window
    persists across drains and would permanently bypass the filter if
    included (any agent that ever clicked would be "non-idle" forever).

    Payloads with an explicit ``kind`` (e.g. §6.3 ``nav_probe``) are
    one-shot events, not drain samples — they are never "empty" even
    when the per-minute counter fields are absent.
    """
    if payload.get("kind"):
        return False
    return not any((
        payload.get("click_success"),
        payload.get("click_fail"),
        payload.get("nav_timeout"),
        payload.get("snapshot_count"),
    ))


def _encode_screenshot(
    png_bytes: bytes,
    fmt: str,
    quality: int,
    scale: float,
    *,
    agent_id: str = "",
) -> tuple[bytes, str]:
    """Encode a Playwright PNG to WebP / PNG with optional downscale.

    Returns ``(encoded_bytes, actual_format)``. ``actual_format`` may be
    ``"png"`` even when ``fmt="webp"`` was requested — Pillow may be
    absent in the dev path or fail on a corrupt frame; PNG fallback
    keeps the agent unblocked rather than returning an error.

    The function is intentionally synchronous and pure — easy to unit
    test and reason about. Pillow does its own threading internally;
    callers should either be on a worker thread or accept that an
    ~1080p WebP encode runs in ~10–20 ms.
    """
    # Fast path: caller asked for PNG and no scale change → pass through.
    if fmt == "png" and abs(scale - 1.0) < 1e-3:
        return png_bytes, "png"

    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        # Pillow missing — log once per encode attempt at debug only;
        # this is expected on the agent-side dev path where Pillow isn't
        # bundled. Caller still gets a usable PNG. Narrowed to
        # ImportError specifically so non-import failures (e.g. partially
        # broken install raising OSError at module init) surface as bugs
        # rather than silently downgrading.
        logger.debug(
            "Pillow not installed; falling back to PNG (agent=%s)", agent_id,
        )
        return png_bytes, "png"

    # Decompression-bomb protection. Pillow's ``MAX_IMAGE_PIXELS`` is a
    # MODULE-LEVEL global; rebinding it on every screenshot encode (the
    # earlier placement) was both wasteful and racy with any other
    # Pillow consumer in the process. Set once, here, immediately after
    # the import succeeds — first encode in the process pins the cap
    # for everyone, subsequent encodes are no-ops since the value is
    # idempotent. 50 MP covers any realistic browser viewport (8K
    # ≈ 33 MP) but blocks adversarial blowups; Pillow raises
    # ``DecompressionBombError`` once exceeded and the catch-all below
    # falls back to the raw PNG bytes.
    if Image.MAX_IMAGE_PIXELS != 50_000_000:
        Image.MAX_IMAGE_PIXELS = 50_000_000

    try:
        img = Image.open(BytesIO(png_bytes))
        # Downscale via Lanczos when requested. Avoids the no-op resize
        # cost when scale is effectively 1.0.
        if abs(scale - 1.0) >= 1e-3 and scale > 0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        out = BytesIO()
        if fmt == "webp":
            # Convert to RGB first — WebP doesn't accept palette or
            # certain RGBA modes from Pillow versions <10.4 cleanly.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(out, format="WEBP", quality=quality, method=4)
            return out.getvalue(), "webp"
        # PNG re-encode (only reached when scale != 1.0 above).
        img.save(out, format="PNG", optimize=True)
        return out.getvalue(), "png"
    except Exception as e:
        logger.warning(
            "Screenshot %s encode failed (%s); falling back to original PNG",
            fmt, e,
        )
        return png_bytes, "png"


# ── §11.4 / §18.2 captcha re-detection after non-navigate actions ──────────
#
# Two-call install/read-back pattern: install a MutationObserver before the
# action runs, then read back any added DOM nodes after the action returns
# and intersect them with the existing ``_check_captcha`` selector list. If
# any added node matches a captcha selector we route through the full
# ``_check_captcha`` → ``_metered_solve`` chain so the agent gets the §11.13
# envelope AND every gate (rate-limit, cost-cap, kill-switch, breaker,
# behavioral classification) fires uniformly. See plan docs §11.4 + §18.2.
#
# Probe attaches to a per-instance randomised window property
# (``inst._captcha_probe_var``, e.g. ``__telemetry_a1b2c3d4``) so
# concurrent observers from prior calls are overwritten cleanly. The
# property is defined ``enumerable: false`` so it does NOT show up
# in ``Object.keys(window)`` / ``for..in window`` walks — anti-bot
# scripts looking for unfamiliar ``__ol_*`` (or any other prefix)
# globals cannot fingerprint our automation. The probe is torn down
# in the read-back (``obs.disconnect()`` + ``delete window[probeVar]``)
# so a navigation between install and read-back leaves no residue.
_JS_CAPTCHA_REDETECT_INSTALL = """
(probeVar) => {
  try {
    const existing = window[probeVar];
    if (existing && existing.obs) {
      try { existing.obs.disconnect(); } catch (e) {}
    }
    const state = { adds: [], obs: null };
    Object.defineProperty(window, probeVar, {
      value: state,
      enumerable: false,
      configurable: true,
      writable: true,
    });
    if (!document.body) return;
    const obs = new MutationObserver(records => {
      for (const r of records) {
        for (const n of r.addedNodes) {
          if (n && n.nodeType === 1) {
            state.adds.push(n);
          }
        }
      }
    });
    obs.observe(document.body, { childList: true, subtree: true });
    state.obs = obs;
  } catch (e) {
    // Swallow — install failure must never break the action.
  }
}
"""

_JS_CAPTCHA_REDETECT_READBACK = """
(args) => {
  const probeVar = args[0];
  const selectors = args[1] || [];
  const p = window[probeVar];
  if (!p) return [];
  try { if (p.obs) p.obs.disconnect(); } catch (e) {}
  const hits = new Set();
  try {
    for (const n of (p.adds || [])) {
      for (const sel of selectors) {
        try {
          if (n.matches && (n.matches(sel) || (n.querySelector && n.querySelector(sel)))) {
            hits.add(sel);
          }
        } catch (e) {}
      }
    }
  } catch (e) {}
  try { delete window[probeVar]; } catch (e) {}
  return [...hits];
}
"""

# Reused by ``_with_captcha_redetect`` so the JS read-back filter and the
# Python ``_check_captcha`` selector list cannot drift apart. Mirrors the
# inline list inside ``_check_captcha``.
_CAPTCHA_REDETECT_SELECTORS: tuple[str, ...] = (
    'iframe[src*="recaptcha"]',
    'iframe[src*="hcaptcha"]',
    '[class*="cf-turnstile"]',
    'iframe[src*="challenges.cloudflare.com"]',
    'iframe[src*="captcha"]',
    '[class*="captcha"]',
    '#captcha',
)

# Rate-limit window for the post-action re-detection trigger. A single
# action that mutates the DOM heavily (SPA route change, list re-render)
# can fire dozens of mutation records; without a per-instance throttle a
# rapid sequence of clicks would each invoke ``_check_captcha`` in full
# (selector probe + classifiers + locator counts). 2s matches the typical
# captcha-render delay and keeps the steady-state cost at ≤1 probe / 2s
# regardless of action cadence. The metered_solve gates downstream are
# ALSO rate-limited (per-hour) so this is purely a CPU-cost throttle.
_REDETECT_MIN_INTERVAL_S: float = 2.0


def _classify_diff_scope(
    inst,
    *,
    snapshot_page_id: str,
    previous: dict | None,
    current_url: str,
    current_dialog_active: bool,
) -> str:
    """Decide which scope label applies to the current diff request.

    Order matters — earlier checks shadow later ones (e.g. a tab change
    that also crossed a navigation reports as ``tab_changed`` because
    the agent's mental model is "I switched tabs"; the URL change is a
    consequence). The five values map to §7.3 contract:

    * ``tab_changed`` — active page differs from the page that owned the
      previous baseline (``inst.last_active_page_id``).
    * ``navigation`` — same tab, but the URL changed since baseline.
    * ``modal_opened`` / ``modal_closed`` — the ``dialog_active`` state
      flipped between baseline and now.
    * ``frame_changed`` — at least one ref's frame_id from the previous
      baseline is no longer attached. Today the iframe walker is
      Phase 8.4; this scope is scaffolded for that future wiring.
    * ``same`` — none of the above. Diff content is meaningful.

    ``previous=None`` (no baseline yet) is treated as ``navigation``
    by the caller so the agent still gets useful output on the very
    first ``diff_from_last`` call.
    """
    # Tab-change detection runs FIRST — it's based on ``last_active_page_id``,
    # which the caller maintains independently of per-page baselines. A
    # switch to a never-baselined tab still reports tab_changed (the
    # agent's mental model is "I switched tabs", not "I navigated"); the
    # response is a full snapshot either way, but the scope label drives
    # operator analytics and matches §7.3 spec wording.
    if (
        inst.last_active_page_id is not None
        and inst.last_active_page_id != snapshot_page_id
    ):
        return "tab_changed"
    # No baseline for the active page yet — first call on this tab.
    if previous is None:
        return "navigation"
    if previous.get("url") != current_url:
        return "navigation"
    prev_dialog = bool(previous.get("dialog_active", False))
    if prev_dialog != current_dialog_active:
        return "modal_opened" if current_dialog_active else "modal_closed"
    # ``frame_changed`` detection is scaffolded for §8.4. We hold the
    # default to ``same`` until that phase wires per-frame UUIDs.
    return "same"


def _compute_snapshot_diff(
    previous: dict[str, dict],
    current: dict[str, dict],
) -> dict:
    """Diff two ``element_key → summary`` maps.

    Returns ``{added, removed, changed, unchanged_count}`` where each
    entry in the lists is a small descriptor dict the agent can read
    directly. The descriptors are intentionally compact: role, name,
    landmark, current ref id (where applicable), and any state delta.
    """
    prev_keys = set(previous.keys())
    curr_keys = set(current.keys())

    added_keys = curr_keys - prev_keys
    removed_keys = prev_keys - curr_keys
    common_keys = curr_keys & prev_keys

    # Sort by walk-emission order using the numeric suffix of ``ref_id``.
    # Lexicographic sort would order "e10" < "e2" — diff lists then look
    # randomized as soon as a page emits ≥10 refs (most non-trivial
    # pages). ``ref_id`` is always shaped ``e<int>``; fall back to lex
    # sort defensively if the prefix ever shifts.
    def _ref_sort_key(key: str) -> tuple:
        rid = current[key].get("ref_id", "")
        if isinstance(rid, str) and rid.startswith("e") and rid[1:].isdigit():
            return (0, int(rid[1:]))
        return (1, rid)

    added: list[dict] = []
    for k in sorted(added_keys, key=_ref_sort_key):
        s = current[k]
        added.append({
            "ref": s["ref_id"],
            "role": s["role"],
            "name": s["name"],
            "landmark": s["landmark"],
        })

    removed: list[dict] = []
    for k in sorted(removed_keys):
        s = previous[k]
        removed.append({
            "role": s["role"],
            "name": s["name"],
            "landmark": s.get("landmark", ""),
        })

    changed: list[dict] = []
    unchanged = 0
    for k in common_keys:
        prev_s = previous[k]
        curr_s = current[k]
        delta: dict = {}
        for field in ("disabled", "value", "checked"):
            if prev_s.get(field) != curr_s.get(field):
                delta[field] = {
                    "from": prev_s.get(field),
                    "to": curr_s.get(field),
                }
        if delta:
            changed.append({
                "ref": curr_s["ref_id"],
                "role": curr_s["role"],
                "name": curr_s["name"],
                "landmark": curr_s.get("landmark", ""),
                **delta,
            })
        else:
            unchanged += 1
    # Stable order for the changed list — by current ref id so the
    # output is deterministic across snapshots that touch the same
    # elements in different orders.
    changed.sort(key=lambda d: d["ref"])

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": unchanged,
    }


def _extract_text_from_a11y(tree: dict | None, max_chars: int = 5000) -> str:
    """Extract readable text from an accessibility snapshot tree.

    Walks leaf nodes to avoid duplicating text that parent containers
    aggregate from their children.  Used by ``navigate()`` as a
    stealth-safe alternative to ``page.evaluate("document.body.innerText")``
    — the a11y API reads from Firefox's internal accessibility service
    with zero JavaScript execution in the page context.
    """
    if not tree:
        return ""
    parts: list[str] = []
    total = 0

    def _collect(node: dict) -> bool:
        nonlocal total
        if not isinstance(node, dict) or total >= max_chars:
            return total < max_chars
        children = node.get("children")
        if children:
            for child in children:
                if not _collect(child):
                    return False
        else:
            name = (node.get("name") or "").strip()
            if name:
                parts.append(name)
                total += len(name) + 1
        return total < max_chars

    _collect(tree)
    return " ".join(parts)[:max_chars]


class CamoufoxInstance:
    """Wrapper around a single Camoufox browser for one agent."""

    def __init__(self, agent_id: str, browser, context, page):
        self.agent_id = agent_id
        self.browser = browser
        self.context = context
        self.page = page
        self.last_activity = time.time()
        # Rich ref identity (§4.2): ref_id → RefHandle carrying page_id,
        # frame_id, shadow_path, scope_root, role/name/occurrence, and
        # (populated later by diff-mode) element_key.
        self.refs: dict[str, RefHandle] = {}
        self.dialog_active: bool = False  # True when snapshot scoped to a modal dialog
        self.dialog_detected: bool = False  # True when a modal was found (even if scoping failed)
        self._lock: asyncio.Lock | None = None
        self.x11_wid: int | None = None  # X11 window ID for targeted focus
        # P0.3: vestigial — the snapshot tree builder always uses the JS
        # walker now. Still consulted by ``navigate()`` for body-text
        # extraction, where the native ``page.accessibility.snapshot()``
        # is acceptable (no shadow descent needed for a text summary).
        self._js_snapshot_mode: bool = False
        self._user_control: bool = False  # True when user has VNC control
        # Per-Page stable UUID maps. Page objects survive navigation within a
        # tab; UUIDs are stable for the life of the Page. Refs carry a
        # ``page_id`` so resolution can detect a closed tab as stale (§4.2).
        #
        # ``page_ids`` is a WeakKeyDictionary so closed Pages drop out of
        # the forward map automatically. The earlier ``dict[int, str]``
        # keyed by ``id(page)`` had a silent re-bind bug: once a Page is
        # GC'd Python recycles its ``id()``, so a freshly-opened tab
        # could land on the SAME ``id`` value as a closed one. The
        # reverse map (already weak) had dropped the entry, but the
        # forward lookup found the stale UUID and re-bound it to the
        # new Page — so an old ref with the recycled UUID resolved to
        # a different tab. WeakKey closes that hole.
        self._page_id_counter: int = 0
        self.page_ids: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )
        # WeakValueDictionary so closed Pages drop out of the reverse
        # lookup automatically. Plain dict would pin every Page ever
        # opened for the lifetime of the CamoufoxInstance.
        self.page_ids_inv: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        # Per-Frame stable UUID maps. Frames navigate / detach as the page
        # changes; WeakKey here so detached Frames drop out of the forward
        # map automatically, and WeakValue on the reverse map so closed
        # frames drop from resolution.
        self.frame_ids: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )
        self.frame_ids_inv: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        # Register the initial page so refs captured on it resolve correctly.
        self._register_page(page)

        # Per-agent metrics (§4.6). Per-minute counters reset at each emit
        # cycle so dashboards see rate-of-change, not monotonic totals.
        # Snapshot byte sizes accumulate as a list (p50/p95 at emit time);
        # size samples are small (~200/min/agent at most) and simpler wins.
        self.m_click_success: int = 0
        self.m_click_fail: int = 0
        self.m_nav_timeout: int = 0
        self.m_snapshot_bytes: list[int] = []
        # §5.2 rolling click-success-rate: deque of booleans for the last
        # 100 click outcomes. Unlike the per-minute counters above, this
        # window is NOT reset on drain — it's a user-facing live gauge
        # exposed via /browser/{agent}/status and in the per-minute metric
        # payload, giving operators an "is the browser currently healthy"
        # signal that doesn't flap on low-traffic minutes.
        self.click_window: deque[bool] = deque(maxlen=100)
        # §7.3 diff-mode baseline: per-page snapshot of {element_key →
        # RefHandle} from the most recent ``browser_get_elements`` call,
        # plus the URL and modal state at that time. ``diff_from_last``
        # snapshots compare against this baseline.
        # Keyed by ``page_id`` so multi-tab agents preserve baselines
        # across ``browser_switch_tab`` (§8.6). Each tab resumes diffing
        # from the cached state when re-entered.
        self.last_snapshot: dict[str, dict] = {}
        # Track the last-active page_id so tab_changed can be detected
        # by comparing to the page_id captured at the previous snapshot.
        self.last_active_page_id: str | None = None
        # §5.3 behavioral entropy recorder (dev-only). Always constructed;
        # every record_* call short-circuits when the feature flag is
        # off so production pays no cost.
        from src.browser.recorder import BehaviorRecorder
        self.recorder = BehaviorRecorder(agent_id)
        # §6.5 rolling-5 history of recently-used referers. The picker
        # uses this to avoid immediate repeats so a fleet at scale
        # doesn't all show the same Google referer back-to-back. Resets
        # on browser restart, matching a real user-session boundary.
        self.recent_referers: deque[str] = deque(maxlen=5)
        # §6.5 first-real-navigate gate. With ``persistent_context=True``
        # the browser resumes whatever page was open last session — the
        # picker would otherwise treat that stale URL as a "previous page"
        # and fabricate a same-origin referer for the next nav, even though
        # there's been no recent navigation in this session. The flag flips
        # to True after the first navigate completes; subsequent navs may
        # use ``inst.page.url`` as the previous-URL hint.
        self.had_real_navigate: bool = False
        # §6.3 navigator self-test result. ``None`` until the post-launch
        # probe runs. Populated dict (see ``BrowserManager._run_navigator_probe``)
        # exposes ``ok`` + ``mismatches`` + raw signal values for dashboard /
        # status endpoint consumers.
        self.probe_result: dict | None = None
        # §9.1 network-inspection log. Bounded deque of recent
        # ``request`` / ``requestfailed`` events, populated by listeners
        # attached at the BrowserContext level so new tabs (in-page
        # ``window.open()`` or ``browser_open_tab``) are covered too. URLs
        # are redacted via ``shared.redaction.redact_url`` at store-time;
        # the deque NEVER holds raw URLs. ``_network_attached`` is the
        # idempotency flag for ``BrowserManager._attach_network_listeners``.
        self.network_log: deque[dict] = deque(maxlen=200)
        self._network_attached: bool = False
        # §11.14 explicit-solve guard.
        # ``_captcha_solving`` is set for the duration of a manager-level
        # ``solve_captcha`` invocation; if a NEW captcha is detected by
        # ``_check_captcha`` while this flag is set we surface
        # ``solver_outcome="captcha_during_solve"`` instead of recursing
        # into another solve attempt (could otherwise pile up provider
        # cost and deadlock against the per-instance lock).
        self._captcha_solving: bool = False
        # §11.4 / §18.2 post-action captcha re-detection state.
        # ``_last_redetect_ts`` carries a ``time.monotonic()`` reading
        # of the most recent successful read-back; the wrapper skips the
        # full ``_check_captcha`` invocation when within
        # :data:`_REDETECT_MIN_INTERVAL_S` of the previous trigger so a
        # heavily-mutating action doesn't hammer the selector probe.
        # ``_pending_captcha_envelope`` carries the §11.13 envelope from
        # the most recent post-action detection; the next ``snapshot()``
        # call surfaces and clears it so a polling agent sees the
        # envelope even when it didn't read the action's response.
        self._last_redetect_ts: float = 0.0
        self._pending_captcha_envelope: dict | None = None
        # §22 — fingerprint health monitor tasks.  ``_check_captcha``
        # spawns a fire-and-forget post-solve monitor (10s window
        # observing whether the page accepted or rejected the injected
        # token).  Tracked here so ``BrowserManager._stop_instance`` can
        # cancel any in-flight monitors at agent stop — otherwise a
        # task left behind would race against a deleted Page and emit
        # spurious rejection signals.  Discarded on completion via the
        # task's ``add_done_callback`` (see ``_spawn_fingerprint_monitor``).
        self._fingerprint_monitor_tasks: set[asyncio.Task] = set()
        # Per-instance random property name for the captcha re-detection
        # MutationObserver state. Anti-bot scripts that walk
        # ``Object.keys(window)`` looking for unfamiliar ``__ol_*``
        # patterns can fingerprint our automation; randomising the name
        # AND defining the property as non-enumerable (see
        # ``_JS_CAPTCHA_REDETECT_INSTALL``) keeps the probe out of
        # generic enumeration. The shape ``__telemetry_<8-hex>`` mimics
        # ad-tech / RUM globals that real sites carry, so a passive
        # observer cannot trivially distinguish ours from a third-party
        # SDK. ``secrets.token_hex`` is stdlib, no extra dependency.
        import secrets
        self._captcha_probe_var: str = f"__telemetry_{secrets.token_hex(4)}"

    @property
    def lock(self) -> asyncio.Lock:
        """Per-instance lock, lazily initialized on first async access.

        Loop binding is fixed once initialized; cross-loop access raises
        ``RuntimeError`` (from the underlying ``asyncio.Lock``) loudly
        rather than silently re-creating the lock — silently swapping
        the lock under a coroutine that already holds the previous
        instance breaks mutual exclusion. Production runs single-loop
        per :class:`BrowserManager`; cross-loop is a test-isolation
        concern that should fail loud.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @lock.setter
    def lock(self, value: asyncio.Lock) -> None:
        """Test-only override. Honors the swap; does NOT track loop."""
        self._lock = value

    def _register_page(self, page) -> str:
        """Assign a stable UUID to a Page if not already registered.

        Idempotent — re-registering the same Page returns its existing UUID.
        Called on CamoufoxInstance creation (for the initial page) and will
        be called again by ``browser_open_tab`` (§8.6) for new tabs.
        """
        existing = self.page_ids.get(page)
        if existing is not None:
            return existing
        self._page_id_counter += 1
        new_id = f"p{self._page_id_counter}-{uuid.uuid4().hex[:8]}"
        self.page_ids[page] = new_id
        self.page_ids_inv[new_id] = page
        return new_id

    def _page_id_for(self, page) -> str:
        """Return the stable UUID for ``page`` (registering if new)."""
        return self._register_page(page)

    def _resolve_page_id(self, page_id: str):
        """Return the Page for ``page_id`` or raise :class:`RefStale`.

        A ref whose ``page_id`` is unknown to this instance points to a
        closed tab (or never existed). Distinct from "element not found"
        — the caller should prompt the agent to re-snapshot.
        """
        page = self.page_ids_inv.get(page_id)
        if page is None:
            raise RefStale("tab closed or unknown page_id", ref=None)
        return page

    def _register_frame(self, frame) -> str:
        existing = self.frame_ids.get(frame)
        if existing is not None:
            return existing
        new_id = f"f-{uuid.uuid4().hex[:8]}"
        self.frame_ids[frame] = new_id
        self.frame_ids_inv[new_id] = frame
        return new_id

    def _resolve_frame_id(self, frame_id: str):
        frame = self.frame_ids_inv.get(frame_id)
        if frame is None:
            raise RefStale("frame_detached", ref=None)
        return frame

    def seed_refs_legacy(self, legacy: "dict[str, dict]") -> None:
        """Test helper: build ``RefHandle`` entries from v1-shape dicts.

        Uses the instance's current page as the target ``page_id`` so
        ``_locator_from_ref`` resolves correctly without the test having to
        know the generated UUID. If ``self.dialog_active`` is True, seeds
        refs with ``scope_root`` pointing at the modal selector — matching
        what a live snapshot emits during modal scoping. Not for production
        use — agent skills don't construct RefHandles, snapshots do.
        """
        page_id = self._page_id_for(self.page)
        scope = _MODAL_SELECTOR if self.dialog_active else None
        self.refs = {
            rid: RefHandle.light_dom(
                page_id=page_id,
                scope_root=scope,
                role=entry.get("role", ""),
                name=entry.get("name", ""),
                occurrence=entry.get("index", 0),
                disabled=bool(entry.get("disabled", False)),
            )
            for rid, entry in legacy.items()
        }

    def touch(self):
        self.last_activity = time.time()

    def rolling_click_success_rate(self) -> float | None:
        """Fraction of the last 100 clicks that succeeded, or ``None``.

        Returns ``None`` when no clicks have been recorded yet — callers
        should render this as "—" rather than "0%", which would misleadingly
        imply catastrophic failure on a freshly-booted agent.
        """
        if not self.click_window:
            return None
        successes = sum(1 for ok in self.click_window if ok)
        return successes / len(self.click_window)

    def drain_metrics(self) -> dict:
        """Snapshot counters and reset the per-minute ones to zero.

        Called by :meth:`BrowserManager._emit_metrics` every minute. The
        rolling 100-click window is NOT reset — it continues to track the
        most recent 100 clicks across emit cycles. Emits the rolling rate
        alongside per-minute counters so the dashboard can show both
        "activity in the last minute" and "health over recent work."
        """
        snaps = self.m_snapshot_bytes
        snap_count = len(snaps)
        if snap_count:
            sorted_snaps = sorted(snaps)
            p50 = sorted_snaps[snap_count // 2]
            p95_idx = max(0, min(snap_count - 1, int(snap_count * 0.95)))
            p95 = sorted_snaps[p95_idx]
        else:
            p50 = 0
            p95 = 0
        out = {
            "agent_id": self.agent_id,
            "click_success": self.m_click_success,
            "click_fail": self.m_click_fail,
            "nav_timeout": self.m_nav_timeout,
            "snapshot_count": snap_count,
            "snapshot_bytes_p50": p50,
            "snapshot_bytes_p95": p95,
            "click_window_size": len(self.click_window),
            "click_success_rate_100": self.rolling_click_success_rate(),
        }
        # Reset the per-minute counters; the rolling window persists.
        self.m_click_success = 0
        self.m_click_fail = 0
        self.m_nav_timeout = 0
        self.m_snapshot_bytes = []
        return out


class BrowserManager:
    """Manages per-agent Camoufox browser instances.

    Browsers are lazy-started on first use and auto-stopped after
    idle timeout. Max concurrent browsers is configurable.
    """

    def __init__(
        self,
        profiles_dir: str = "/data/profiles",
        max_concurrent: int = 5,
        idle_timeout_minutes: int = 30,
        *,
        metrics_sink=None,
    ):
        """Per-agent Camoufox lifecycle manager.

        Args:
            metrics_sink: optional callable ``(payload: dict) -> None`` that
                receives per-agent aggregate metrics once per minute. When
                ``None``, metrics counters still increment but nothing is
                emitted — tests can pass a list's ``append`` method to
                capture payloads; production wires this to the dashboard
                :class:`EventBus`.
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.idle_timeout = idle_timeout_minutes * 60
        self._instances: dict[str, CamoufoxInstance] = {}
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._playwright = None
        self._user_focused_agent: str | None = None  # set by explicit focus() call
        self.redactor = CredentialRedactor()
        self._proxy_configs: dict[str, dict | None] = {}
        self.boot_id: str = str(uuid.uuid4())
        self._captcha_solver = get_solver()
        self._metrics_sink = metrics_sink
        # Per-agent rolling buffer of recent emit payloads (§5.1/§5.2) used by
        # the mesh's periodic poll to forward metrics to the dashboard
        # EventBus. Kept as a monotonic sequence so repeated polls can
        # request only what they haven't seen. Bounded so a long-lived
        # service with many agents doesn't grow without bound.
        self._metrics_history: deque[dict] = deque(maxlen=1024)
        self._metrics_seq: int = 0
        self._upload_recv_gc_task: asyncio.Task | None = None
        self._download_gc_task: asyncio.Task | None = None
        # Nonces of downloads currently being streamed to the mesh.
        # GC skips files whose nonce is in this set so a slow or near-TTL
        # transfer isn't reaped mid-stream.
        self._active_download_nonces: set[str] = set()
        # Detect Playwright's private artifact-stream channel availability
        # once at init. When unavailable we refuse downloads with a
        # service_unavailable envelope rather than silently degrading to a
        # racy drain-then-check fallback.
        self._download_streaming_available: bool = (
            self._detect_download_streaming()
        )
        if not self._download_streaming_available:
            logger.critical(
                "Playwright private artifact-stream API unavailable; "
                "browser_download will return service_unavailable.",
            )
        # §20 — per-agent elapsed-seconds counter for periodic session
        # snapshots. Incremented by 60 on each metrics tick; resets to 0
        # after a snapshot fires. Keyed by agent_id so an agent that just
        # started doesn't immediately snapshot (would write an empty
        # storage_state to disk for nothing). See ``_periodic_session_snapshots``.
        self._session_snapshot_elapsed_s: dict[str, int] = {}

    def _manager_lock(self) -> asyncio.Lock:
        """Return the manager lock for the currently running event loop.

        The browser manager is often constructed from synchronous FastAPI
        setup/test code. Python 3.9 binds ``asyncio.Lock`` to the current
        loop during construction, so creating it eagerly is fragile after
        pytest or startup code has closed a previous default loop.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    @staticmethod
    def _detect_download_streaming() -> bool:
        try:
            from playwright._impl._connection import from_channel  # noqa: F401
        except Exception:
            return False
        return True

    async def start_cleanup_loop(self):
        """Start background task that cleans up idle browsers."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._upload_recv_gc_task = asyncio.create_task(self._upload_recv_gc_loop())
        self._download_gc_task = asyncio.create_task(self._download_gc_loop())

    async def _upload_recv_gc_once(self) -> int:
        """Reap orphan upload-recv files older than the stage TTL.

        Mirrors the mesh-side reaper: under normal flow ``upload_file`` cleans
        its own files via the try/finally, but a hard crash between
        ``_stage_upload`` and ``upload_file`` (chooser timeout, ref-not-found,
        process kill) would leak bytes to the recv dir.
        """
        from src.browser.flags import get_int as _flag_int
        from src.browser.flags import get_str as _flag_str

        recv_dir = Path(_flag_str("OPENLEGION_UPLOAD_RECV_DIR", "/tmp/upload-recv"))
        if not recv_dir.exists():
            return 0
        ttl_s = _flag_int(
            "OPENLEGION_UPLOAD_STAGE_TTL_S", 60, min_value=5, max_value=3600,
        )
        now = time.time()
        reaped = 0
        try:
            entries = list(recv_dir.iterdir())
        except OSError:
            return 0
        for child in entries:
            try:
                age = now - child.stat().st_mtime
            except OSError:
                continue
            if age > ttl_s:
                try:
                    child.unlink()
                    reaped += 1
                except OSError:
                    continue
        return reaped

    async def _upload_recv_gc_loop(self):
        while True:
            try:
                await self._upload_recv_gc_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("upload_recv gc tick failed: %s", e)
            await asyncio.sleep(30)

    async def _download_gc_loop(self):
        """Periodically delete stale download staging files.

        The download() flow normally cleans up via the mesh-side
        ``_download_cleanup`` call, but a mesh crash mid-stream would
        otherwise leak files into ``BROWSER_DOWNLOAD_DIR`` until the
        container restarts. Janitor mirrors the upload-stage GC.
        """
        from src.browser.flags import get_int as _flag_int
        from src.browser.flags import get_str as _flag_str
        ttl = _flag_int(
            "BROWSER_DOWNLOAD_TTL_S", 60, min_value=1, max_value=86400,
        )
        while True:
            await asyncio.sleep(30)
            try:
                dl_dir = Path(_flag_str(
                    "BROWSER_DOWNLOAD_DIR", "/tmp/downloads",
                ))
                if not dl_dir.is_dir():
                    continue
                now = time.time()
                for entry in list(dl_dir.iterdir()):
                    try:
                        if not entry.is_file():
                            continue
                        nonce = entry.name.split("-", 1)[0]
                        if nonce in self._active_download_nonces:
                            continue
                        if (now - entry.stat().st_mtime) > ttl:
                            entry.unlink(missing_ok=True)
                    except OSError:
                        continue
            except Exception:
                logger.debug("Download GC pass failed", exc_info=True)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                await self._cleanup_idle()
            except Exception as e:
                logger.warning("Cleanup loop error: %s", e)
            # Emit per-minute metrics AFTER idle-cleanup — instances that
            # just got stopped had their counters drained in ``_stop_instance``.
            try:
                await self._emit_metrics()
            except Exception as e:
                logger.warning("Metrics emit error: %s", e)

    async def _emit_metrics(self):
        """Drain per-agent counters and fan them out.

        Runs on the same 60s tick as idle cleanup (per §2.7: per-call
        events are forbidden; aggregates only). Writes each payload to the
        in-memory history buffer so the mesh can poll it, then forwards to
        the optional ``metrics_sink`` callback (for tests / in-process
        wiring). Counters always reset, whether or not a sink is attached,
        so a long-idle service doesn't grow memory.

        Per-instance drain failures are caught — a single agent with a
        corrupt counter must not abort the emit loop and starve the other
        agents' data.
        """
        now = time.time()
        # Take a consistent view of the instance list; ``drain_metrics()``
        # is a fully synchronous read-then-reset so no ``await`` boundary
        # opens between the counter read and its zeroing. Under asyncio's
        # single-threaded event loop, an in-flight hot-path task that
        # holds ``inst.lock`` cannot run between those two statements —
        # its coroutine is suspended elsewhere. This is WHY we don't
        # need ``inst.lock`` here. If ``drain_metrics`` ever grows an
        # ``await``, that invariant breaks and this must take the lock
        # or swap counter objects atomically.
        for inst in list(self._instances.values()):
            try:
                payload = inst.drain_metrics()
            except Exception as e:
                logger.warning(
                    "drain_metrics failed for '%s': %s", inst.agent_id, e,
                )
                continue
            # Skip payloads with zero activity AND an empty rolling window —
            # idle agents should not flood the history buffer (meshes that
            # were briefly offline will otherwise replay dozens of no-op
            # entries on reconnect, evicting live signal from the dashboard
            # ring buffer).
            if _is_empty_payload(payload):
                continue
            self._metrics_seq += 1
            payload["seq"] = self._metrics_seq
            payload["ts"] = now
            self._metrics_history.append(payload)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(payload)
            except Exception as e:
                logger.warning(
                    "Metrics sink raised for '%s': %s", inst.agent_id, e,
                )

        # §11.14 / §2.7 — drain the captcha audit-log buckets so the
        # cost-cap / rate-limit / behavioral-skip events surface to the
        # dashboard once per minute (NOT per call). Reuses the same
        # ``metrics_sink`` plumbing that delivers per-agent metrics.
        try:
            audit_events = await _drain_captcha_audit()
        except Exception as e:
            logger.warning("captcha audit drain failed: %s", e)
            audit_events = []
        for ev in audit_events:
            ev = dict(ev)
            self._metrics_seq += 1
            ev["seq"] = self._metrics_seq
            ev["ts"] = now
            self._metrics_history.append(ev)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(ev)
            except Exception as e:
                logger.warning(
                    "captcha audit sink raised for '%s': %s",
                    ev.get("agent_id", ""), e,
                )

        # §20 — periodic session snapshots. Hooked into the same 60s tick
        # so we don't add a second background task; per-agent elapsed
        # counters fire when they cross the configured interval. Runs
        # AFTER metrics drain so a long snapshot can't starve the metrics
        # path; runs BEFORE the session-audit drain so this tick's
        # snapshot events surface immediately.
        try:
            await self._periodic_session_snapshots()
        except Exception as e:
            logger.warning("periodic session snapshot pass failed: %s", e)

        # §20 — drain the session audit-log buckets (snapshot/restore
        # events). Same per-minute aggregation pattern as captcha — no
        # cookie values, no origin domains.
        try:
            session_events = await _drain_session_audit()
        except Exception as e:
            logger.warning("session audit drain failed: %s", e)
            session_events = []
        for ev in session_events:
            ev = dict(ev)
            self._metrics_seq += 1
            ev["seq"] = self._metrics_seq
            ev["ts"] = now
            self._metrics_history.append(ev)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(ev)
            except Exception as e:
                logger.warning(
                    "session audit sink raised for '%s': %s",
                    ev.get("agent_id", ""), e,
                )

        # §22 — drain the fingerprint audit-log buckets (post-solve
        # accept/reject signals + burn-state crossings).  Same per-minute
        # aggregation pattern; ``page_origin`` is netloc-only, no full
        # URLs / paths / queries.  Routed to the EventBus as
        # ``type="fingerprint_event"`` so the dashboard can render a
        # dedicated panel without colliding with captcha or session
        # streams.
        try:
            fp_events = await _drain_fingerprint_audit()
        except Exception as e:
            logger.warning("fingerprint audit drain failed: %s", e)
            fp_events = []
        for ev in fp_events:
            ev = dict(ev)
            self._metrics_seq += 1
            ev["seq"] = self._metrics_seq
            ev["ts"] = now
            self._metrics_history.append(ev)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(ev)
            except Exception as e:
                logger.warning(
                    "fingerprint audit sink raised for '%s': %s",
                    ev.get("agent_id", ""), e,
                )

        # Drain the platform pre-nav-delay audit buckets — feeds the
        # dashboard's per-platform success panel with count + average
        # dwell per (agent, host).  Same per-minute aggregation pattern.
        try:
            pt_events = await _drain_platform_timing_audit()
        except Exception as e:
            logger.warning("platform-timing audit drain failed: %s", e)
            pt_events = []
        for ev in pt_events:
            ev = dict(ev)
            self._metrics_seq += 1
            ev["seq"] = self._metrics_seq
            ev["ts"] = now
            self._metrics_history.append(ev)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(ev)
            except Exception as e:
                logger.warning(
                    "platform-timing audit sink raised for '%s': %s",
                    ev.get("agent_id", ""), e,
                )

        # Phase 10 §24 — per-tenant spend-threshold alerts. Walks every
        # tenant currently active in the cost counter, asks the threshold
        # tracker which percentages crossed THIS tick, and ships a
        # ``tenant_spend_threshold`` payload through the same metrics_sink.
        # Keeping it on the metrics tick (not on every ``add_cost``) avoids
        # firing a flag-load + project-config read on the hot solve path.
        try:
            await self._emit_tenant_threshold_alerts(now)
        except Exception as e:
            logger.warning("tenant threshold emit error: %s", e)

    async def _emit_tenant_threshold_alerts(self, now: float) -> None:
        """Emit ``tenant_spend_threshold`` payloads for newly-crossed caps.

        For each tenant currently visible in the per-agent cost counter,
        reads ``CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH`` (treating the
        tenant ID as the ``agent_id`` arg to ``flags.get_int`` so per-
        tenant overrides land on the same precedence chain operators
        already use), converts to millicents, and asks the cost counter
        which threshold percentages crossed since the last tick. Each
        crossing produces ONE ``tenant_spend_threshold`` payload routed
        through the metrics_sink + history buffer.
        """
        from src.browser import captcha_cost_counter as ccc
        from src.browser import flags as _flags

        # Snapshot the current set of agents → tenants from the cost
        # counter's state. Inlining the walk avoids exposing ``_state``
        # publicly; the lock is acquired by ``get_tenant_total`` below.
        async with ccc._get_lock():
            agent_ids = list(ccc._state.keys())

        tenants: set[str] = set()
        for agent_id in agent_ids:
            tid = ccc._tenant_for(agent_id)
            if tid:
                tenants.add(tid)

        if not tenants:
            return

        for tenant_id in tenants:
            # Per-tenant cap, in USD. Treat the tenant ID like an agent
            # ID for flag-precedence purposes — operators set per-tenant
            # caps via the same ``set_agent_override`` machinery they
            # already use for per-agent flags. Default 0 means "no cap"
            # which short-circuits inside ``record_tenant_threshold_alerts``.
            cap_usd = _flags.get_int(
                "CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH",
                0,
                agent_id=tenant_id,
                min_value=0,
            )
            cap_millicents = cap_usd * 100_000
            if cap_millicents <= 0:
                continue

            captured: list[dict] = []

            def _capture(payload: dict, _bucket: list = captured) -> None:
                _bucket.append(payload)

            try:
                fired = await ccc.record_tenant_threshold_alerts(
                    tenant_id, cap_millicents, _capture,
                )
            except Exception as e:
                logger.warning(
                    "tenant threshold check failed for %s: %s", tenant_id, e,
                )
                continue
            if not fired:
                continue

            for payload in captured:
                event = {
                    "type": "tenant_spend_threshold",
                    # ``agent_id`` carries the tenant — the dashboard's
                    # event router keys EventBus payloads on
                    # ``payload['agent_id']`` so dropping this off in the
                    # tenant slot lets the existing relay surface the
                    # event without a special case.
                    "agent_id": tenant_id,
                    **payload,
                }
                self._metrics_seq += 1
                event["seq"] = self._metrics_seq
                event["ts"] = now
                self._metrics_history.append(event)
                if self._metrics_sink is None:
                    continue
                try:
                    self._metrics_sink(event)
                except Exception as e:
                    logger.warning(
                        "tenant threshold sink raised for '%s': %s",
                        tenant_id, e,
                    )

    async def _periodic_session_snapshots(self) -> None:
        """§20 — opportunistically snapshot live session state mid-flight.

        Process kills (OOM, kernel panic, ``docker kill``) bypass the
        graceful-shutdown path entirely; without a periodic snapshot the
        agent's freshest cookies + ``localStorage`` since the last
        clean-stop would be lost. Instead of running a separate background
        task, we hook into the existing 60s metrics tick: per-agent
        elapsed-second counters accumulate, and a snapshot fires when a
        counter crosses the configured interval (default 300s, range
        [60, 3600]). RPO at the default = at most ~5 minutes of state.

        Failures are logged + swallowed — same posture as the on-shutdown
        snapshot. A bad disk on one agent must not abort the metrics
        loop and starve the rest of the fleet.

        No-op when ``BROWSER_SESSION_PERSISTENCE_ENABLED=false`` (default)
        — the elapsed counters still tick for housekeeping but no I/O
        happens. The flag is checked per-agent so an operator override
        for one agent works without affecting the others.
        """
        from src.browser.flags import get_bool, get_int
        from src.browser.session_persistence import snapshot_session

        # Single global default-bound; per-agent flag overrides apply
        # below as we iterate.
        global_default = get_bool("BROWSER_SESSION_PERSISTENCE_ENABLED", False)

        # Stale agent_ids in the elapsed-counter dict (instance closed
        # between ticks) — drop them to keep the dict size bounded over
        # long deployments with rotating agent ids.
        live_ids = set(self._instances.keys())
        stale = [
            aid for aid in self._session_snapshot_elapsed_s
            if aid not in live_ids
        ]
        for aid in stale:
            self._session_snapshot_elapsed_s.pop(aid, None)

        for agent_id, inst in list(self._instances.items()):
            enabled = get_bool(
                "BROWSER_SESSION_PERSISTENCE_ENABLED",
                global_default,
                agent_id=agent_id,
            )
            if not enabled:
                # Reset the counter so a flag flip mid-deployment doesn't
                # immediately fire a snapshot from accumulated stale time.
                self._session_snapshot_elapsed_s.pop(agent_id, None)
                continue
            interval_s = get_int(
                "BROWSER_SESSION_PERIODIC_SNAPSHOT_S",
                300,
                agent_id=agent_id,
                min_value=60,
                max_value=3600,
            )
            elapsed = self._session_snapshot_elapsed_s.get(agent_id, 0) + 60
            if elapsed < interval_s:
                self._session_snapshot_elapsed_s[agent_id] = elapsed
                continue
            # Cross the threshold — fire the snapshot and reset the
            # counter. Reset BEFORE the await so a slow snapshot can't
            # double-fire on the next tick.
            self._session_snapshot_elapsed_s[agent_id] = 0
            try:
                ok = await snapshot_session(agent_id, inst.context)
            except Exception as e:
                logger.warning(
                    "periodic session snapshot for '%s' raised: %s",
                    agent_id, e,
                )
                ok = False
            try:
                await _record_session_audit_event(
                    agent_id, "session_snapshot", ok,
                )
            except Exception as e:
                logger.debug(
                    "session audit record failed for '%s': %s", agent_id, e,
                )

    def get_recent_metrics(self, since_seq: int = 0) -> dict:
        """Return buffered metric payloads with ``seq > since_seq``.

        Shape: ``{"current_seq": N, "metrics": [...]}``. The poller passes
        back ``current_seq`` as ``since_seq`` on the next call to get only
        new payloads. On service restart the seq counter resets to 0 — the
        poller detects this via the ``boot_id`` on ``/browser/status`` and
        resets its high-water mark.
        """
        metrics = [p for p in self._metrics_history if p.get("seq", 0) > since_seq]
        return {
            "current_seq": self._metrics_seq,
            "boot_id": self.boot_id,
            "metrics": metrics,
        }

    async def _cleanup_idle(self):
        now = time.time()
        async with self._manager_lock():
            to_stop = [
                agent_id for agent_id, inst in self._instances.items()
                if now - inst.last_activity > self.idle_timeout
            ]
            for agent_id in to_stop:
                logger.info("Stopping idle browser for '%s'", agent_id)
                await self._stop_instance(agent_id)

    async def touch_all(self) -> int:
        """Reset the idle timer for every running browser instance.

        Called by the VNC keepalive while a user is actively viewing the
        display, so a watched browser is never killed by the idle cleanup.
        Returns the number of instances touched.
        """
        async with self._manager_lock():
            for inst in self._instances.values():
                inst.touch()
            return len(self._instances)

    async def refocus_active(self) -> None:
        """Re-assert X11 focus on the user's viewed browser window.

        Called periodically by the VNC keepalive.  When a modal, popup, or
        internal Firefox dialog steals X11 focus, subsequent VNC mouse clicks
        go to the wrong window and appear to do nothing.

        Prefers the agent the user explicitly focused (via the dashboard
        Browser button) over the most recently active instance.  This
        prevents background agent browser operations from stealing the
        VNC display away from what the user is watching.
        """
        async with self._manager_lock():
            if not self._instances:
                return
            # Prefer user's explicit focus over MRU
            if (
                self._user_focused_agent
                and self._user_focused_agent in self._instances
            ):
                target = self._instances[self._user_focused_agent]
            else:
                target = max(self._instances.values(), key=lambda i: i.last_activity)
            wid = target.x11_wid
        if not wid:
            # No WID known — skip xdotool entirely.  The fallback
            # `search --class firefox` matches ALL Firefox windows and
            # raises whichever it finds first, breaking multi-agent
            # browser switching.
            return
        try:
            wid_s = str(wid)
            cmd = ["xdotool", "windowmap", "--sync", wid_s,
                   "windowraise", wid_s, "windowfocus", wid_s]
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, timeout=3),
            )
        except Exception:
            pass

    async def get_or_start(self, agent_id: str) -> CamoufoxInstance:
        """Get existing browser or start a new one for the agent."""
        if not _AGENT_ID_RE.match(agent_id):
            raise ValueError(f"Invalid agent_id: {agent_id!r}")
        async with self._manager_lock():
            if agent_id in self._instances:
                inst = self._instances[agent_id]
                inst.touch()
                return inst

            # Enforce max concurrent
            if len(self._instances) >= self.max_concurrent:
                # Stop least recently used
                oldest_id = min(self._instances, key=lambda a: self._instances[a].last_activity)
                logger.info("Max browsers reached, stopping LRU '%s'", oldest_id)
                await self._stop_instance(oldest_id)

            # Start while holding lock to prevent duplicate instances for same agent
            instance = await self._start_browser(agent_id)
            self._instances[agent_id] = instance
            return instance

    async def _ensure_playwright(self):
        """Start the shared Playwright instance if not running."""
        if self._playwright is None:
            from playwright.async_api import async_playwright
            ctx = async_playwright()
            pw = await ctx.start()
            self._pw_context = ctx
            self._playwright = pw
        return self._playwright

    async def _get_firefox_wids(self) -> set[int]:
        """Return the set of current X11 window IDs for Firefox windows."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "search", "--class", "firefox"],
                    capture_output=True, text=True, timeout=2,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                return {int(w) for w in result.stdout.strip().split("\n") if w.strip()}
        except Exception:
            pass
        return set()

    async def _discover_new_wid(self, before: set[int]) -> int | None:
        """Poll for a new Firefox X11 window that wasn't in *before*.

        Takes the highest WID when multiple new windows appear, since X11
        assigns incrementing IDs — the highest is the most recently created
        (the main browser window, not a transient popup from startup).
        """
        for _ in range(30):  # up to ~6s
            current = await self._get_firefox_wids()
            new = current - before
            if new:
                return max(new)
            await asyncio.sleep(0.2)
        return None

    async def _start_browser(self, agent_id: str) -> CamoufoxInstance:
        """Launch a Camoufox browser for an agent."""
        from camoufox.async_api import AsyncNewBrowser

        pw = await self._ensure_playwright()

        profile_dir = str(self.profiles_dir / agent_id)
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        # Bring the profile up to the current schema version BEFORE Camoufox
        # opens it (§4.4). Post-launch migration would race Camoufox's own
        # writes into the directory. Idempotent on already-current profiles;
        # on failure restores the pre-migration backup and re-raises so we
        # never launch against a half-migrated profile.
        #
        # Two error shapes propagate:
        #   * ``ProfileMigrationBusy`` — a peer process holds the lock and
        #     the on-disk version is below target. Retryable; agent can
        #     call again in a few seconds.
        #   * Any other exception — migration hit a real failure and the
        #     backup has already been restored. Not safely retryable until
        #     a human investigates.
        from src.browser.profile_schema import (
            ProfileMigrationBusy,
            sync_adblock_extension,
        )
        try:
            migrate_profile(Path(profile_dir))
        except ProfileMigrationBusy:
            logger.warning(
                "Profile migration for '%s' busy (peer holds lock); "
                "refusing to launch until they finish", agent_id,
            )
            raise
        except Exception:
            logger.exception(
                "Profile migration failed for '%s' — aborting browser start",
                agent_id,
            )
            raise

        # Phase 4 §7.1 — make sure the ad-blocker XPI matches the operator's
        # current ``BROWSER_ENABLE_ADBLOCK`` setting. This is intentionally
        # separate from the schema migration: the migration runs once per
        # version bump, but flag toggles + image rebuilds with newer XPIs
        # need to take effect on every launch. Best-effort — never blocks
        # the browser from starting.
        sync_adblock_extension(Path(profile_dir))

        # §19.3 / Phase 10 §21: device profile selection. Read with
        # ``agent_id`` so per-agent overrides (operator-settings or
        # dashboard flag panel) take precedence over operator-wide env.
        from src.browser.flags import get_str as _flag_get_str_dp
        device_profile = _flag_get_str_dp(
            "BROWSER_DEVICE_PROFILE",
            DEFAULT_DEVICE_PROFILE,
            agent_id=agent_id,
        ) or DEFAULT_DEVICE_PROFILE
        # ``get_device_profile`` logs a warning + falls back when the value
        # is unknown; we only need to capture the name we'll forward to
        # ``build_launch_options`` (which performs the same fallback).
        logger.info(
            "Agent '%s' device profile: %s", agent_id, device_profile,
        )

        proxy_config = self.get_proxy_config(agent_id)
        if proxy_config is not None:
            if proxy_config.get("url"):
                # Per-agent proxy configured — use it
                proxy_arg: dict = {"server": proxy_config["url"]}
                if proxy_config.get("username"):
                    proxy_arg["username"] = proxy_config["username"]
                if proxy_config.get("password"):
                    proxy_arg["password"] = proxy_config["password"]
                options = build_launch_options(
                    agent_id, profile_dir, proxy=proxy_arg,
                    device_profile=device_profile,
                )
            else:
                # Explicitly no proxy (direct mode or inherit with no system proxy)
                options = build_launch_options(
                    agent_id, profile_dir, proxy=None,
                    device_profile=device_profile,
                )
        else:
            # No per-agent config pushed yet — start without proxy.
            # The mesh will push the correct config shortly after startup
            # which triggers a reset, relaunching with the right proxy.
            logger.warning("No proxy config pushed for '%s' yet, starting without proxy", agent_id)
            options = build_launch_options(
                agent_id, profile_dir, proxy=None,
                device_profile=device_profile,
            )

        # Log which proxy is being used for debuggability
        _proxy_opt = options.get("proxy")
        if _proxy_opt:
            _p_server = _proxy_opt.get("server", "?")
            logger.info("Starting Camoufox for '%s' (profile=%s, proxy=%s)", agent_id, profile_dir, _p_server)
        else:
            logger.info("Starting Camoufox for '%s' (profile=%s, no proxy)", agent_id, profile_dir)

        # Snapshot existing Firefox windows so we can identify the new one
        wids_before = await self._get_firefox_wids()

        # persistent_context=True → returns a BrowserContext directly.
        # geoip=True makes Camoufox connect through the proxy to resolve
        # the egress IP for fingerprint-consistent timezone/locale.  If the
        # proxy is slow to handshake, this can fail.  Retry once with geoip
        # (proxy may just need time), then fall back without it as last resort.
        try:
            browser = await AsyncNewBrowser(pw, **options)
        except Exception as e:
            if not options.get("geoip"):
                raise
            logger.warning(
                "Camoufox launch failed for '%s' with geoip (%s), retrying with geoip after brief wait",
                agent_id, e,
            )
            await asyncio.sleep(2)
            try:
                browser = await AsyncNewBrowser(pw, **options)
            except Exception as e2:
                logger.warning(
                    "Camoufox geoip retry failed for '%s' (%s), "
                    "falling back without geoip — fingerprint won't match proxy location",
                    agent_id, e2,
                )
                options.pop("geoip", None)
                browser = await AsyncNewBrowser(pw, **options)
        context = browser

        # §19.3 / Phase 10 §21: inject the navigator-override init script
        # for mobile profiles BEFORE any page is created. ``add_init_script``
        # is registered on the BrowserContext so every page (current + new
        # tabs) receives it at ``document_start``, ahead of any site script.
        # Returns ``None`` for desktop profiles — no-op in that case.
        mobile_init = build_mobile_init_script(get_device_profile(device_profile))
        if mobile_init is not None:
            try:
                await context.add_init_script(script=mobile_init)
                logger.debug(
                    "Agent '%s' mobile navigator init script injected (profile=%s)",
                    agent_id, device_profile,
                )
            except Exception as e:
                # Non-fatal: a missing init script weakens the spoof but
                # doesn't break the browser. Log and continue.
                logger.warning(
                    "Agent '%s' add_init_script failed for profile %s: %s",
                    agent_id, device_profile, e,
                )

        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        # §6.6 ``navigator.connection`` REMOVED on the Firefox UA path.
        #
        # Real Firefox does NOT expose ``navigator.connection`` (the
        # NetworkInformation API is unimplemented as of FF 138; Chromium
        # is the only major engine shipping it on by default). Injecting
        # a synthetic ``navigator.connection`` on a UA stamped
        # ``Firefox/`` was itself a strong cluster signal: zero real
        # Firefox users in the population have the API, so any Firefox-
        # shaped client that DOES expose it is by definition spoofed.
        # Fingerprint.com / Creep.js / DataDome key directly on this
        # inconsistency.
        #
        # ``_assert_firefox_ua`` enforces a Firefox UA at startup, so
        # the right behavior is to MATCH real-Firefox population —
        # leave the API absent. The §6.6 plan predates this analysis;
        # the navigator probe below was updated to stop treating a
        # missing ``navigator.connection`` as a mismatch.
        #
        # If a future build moves to a Chromium-shaped UA, the spoof
        # path can be restored (gated on UA family); the
        # ``_NAV_CONNECTION_INIT_SCRIPT`` template is retained for that
        # purpose.

        inst = CamoufoxInstance(agent_id, browser, context, page)

        # §20 — restore the previously-snapshotted session state.
        #
        # Camoufox's ``persistent_context=True`` already retains cookies +
        # localStorage in the on-disk profile dir, so this is genuinely a
        # SECOND-CHANNEL restore: the JSON sidecar is the operator-visible,
        # operator-clearable copy of session state that survives even when
        # the profile dir is wiped (operator support, profile rotation,
        # template-based agent re-spawn). On a normal restart the sidecar
        # and profile dir agree; if they disagree (sidecar fresher than the
        # on-disk profile, e.g. after a profile-dir reset), the sidecar
        # wins for cookies / localStorage so the agent stays logged in.
        #
        # We use Playwright's ``add_cookies`` + ``add_init_script`` here
        # rather than launching with ``storage_state`` because Camoufox's
        # ``persistent_context=True`` path does not pass ``storage_state``
        # through to Firefox (it owns the profile dir directly). The end
        # state is the same: cookies merged into the cookie jar; localStorage
        # seeded on each origin's first navigation via the init script.
        await self._maybe_restore_session(inst)

        # §9.1 wire request listeners at the BrowserContext level so new
        # tabs (in-page ``window.open()`` or ``browser_open_tab``) inherit
        # them automatically. Idempotent — also re-runs after browser RESET
        # because RESET drops the whole instance and the next get_or_start
        # creates a fresh one with ``_network_attached=False``.
        self._attach_network_listeners(inst)

        # Discover the new X11 window for targeted focus
        wid = await self._discover_new_wid(wids_before)
        if wid:
            inst.x11_wid = wid
            logger.debug("Agent '%s' browser window: X11 WID %d", agent_id, wid)
            # Start idle mouse jitter for human-like fidgeting
            inst._jitter_task = asyncio.create_task(self._idle_mouse_jitter(inst))
        else:
            logger.warning(
                "Could not discover X11 WID for '%s' — interactions on "
                "high-sensitivity sites will use CDP (isTrusted=false)",
                agent_id,
            )
            inst._jitter_task = None

        # §6.3 run the navigator self-test once. Best-effort — a probe
        # failure must not block browser start (the inconsistency is
        # itself the operator's signal to investigate).
        try:
            await self._run_navigator_probe(inst)
        except Exception as e:
            logger.warning(
                "Navigator self-test probe failed for '%s': %s", agent_id, e,
            )

        return inst

    async def _maybe_restore_session(self, inst: CamoufoxInstance) -> None:
        """§20 — restore cookies + localStorage from the per-agent sidecar.

        No-op when:
          * ``BROWSER_SESSION_PERSISTENCE_ENABLED=false`` (the default;
            operators must opt in deliberately because storage_state
            files contain live session tokens).
          * No sidecar exists for this agent.
          * The sidecar fails to parse (logged; sidecar is preserved).

        Best-effort: a restore failure is logged and we record the audit
        event with ``success=False``, but the browser starts anyway with
        whatever the on-disk profile already had. Blocking startup on a
        bad sidecar would be a worse failure mode than running with a
        slightly-stale session.
        """
        from src.browser.flags import get_bool
        from src.browser.session_persistence import (
            restore_session,
            session_path,
        )

        agent_id = inst.agent_id
        enabled = get_bool(
            "BROWSER_SESSION_PERSISTENCE_ENABLED", False, agent_id=agent_id,
        )
        if not enabled:
            return
        if not session_path(agent_id).exists():
            # No sidecar — common case for the first launch on an agent
            # that never had its session captured. Don't emit an audit
            # event for this (would be noise; the dashboard should not
            # show "restore failed" when there was nothing to restore).
            return

        # Build a context_factory that applies the loaded ``storage_state``
        # to ``inst.context`` and returns it. Camoufox's persistent_context
        # path doesn't accept ``storage_state`` at launch, so we apply
        # cookies + localStorage post-launch here. If a future migration
        # off persistent_context happens, swap this for a ``new_context``
        # call that takes ``storage_state`` directly.
        context = inst.context

        async def _apply_state(*, storage_state: dict) -> object:
            cookies = storage_state.get("cookies") or []
            origins = storage_state.get("origins") or []
            if cookies:
                try:
                    await context.add_cookies(cookies)
                except Exception as e:
                    logger.warning(
                        "session restore for '%s': add_cookies failed: %s",
                        agent_id, e,
                    )
                    raise
            # Seed localStorage on first navigation per origin via an
            # init script. Real Playwright ``new_context(storage_state=)``
            # does the same dance under the hood (mass-injects per-origin
            # storage on first navigation). We build one script that
            # checks ``window.location.origin`` and applies the matching
            # entries. Idempotent — re-running on a page already seeded
            # just overwrites the same keys with the same values.
            if origins:
                import json as _json
                origin_map = {}
                for entry in origins:
                    if not isinstance(entry, dict):
                        continue
                    origin = entry.get("origin")
                    items = entry.get("localStorage") or []
                    if not isinstance(origin, str) or not isinstance(items, list):
                        continue
                    origin_map[origin] = items
                if origin_map:
                    script = (
                        "(() => { try {"
                        f"  const map = {_json.dumps(origin_map)};"
                        "  const entries = map[window.location.origin];"
                        "  if (!entries) return;"
                        "  for (const e of entries) {"
                        "    if (e && typeof e.name === 'string') {"
                        "      try { window.localStorage.setItem(e.name, e.value); } "
                        "      catch (_) {}"
                        "    }"
                        "  }"
                        "} catch (_) {} })();"
                    )
                    try:
                        await context.add_init_script(script)
                    except Exception as e:
                        logger.warning(
                            "session restore for '%s': add_init_script "
                            "failed: %s", agent_id, e,
                        )
                        raise
            return context

        restored = None
        try:
            restored = await restore_session(agent_id, _apply_state)
        except Exception as e:
            # restore_session itself catches its inner errors; this
            # branch only fires if something exotic blew up.
            logger.warning(
                "session restore for '%s' raised unexpectedly: %s",
                agent_id, e,
            )
        success = restored is not None
        try:
            await _record_session_audit_event(
                agent_id, "session_restore", success,
            )
        except Exception as e:
            logger.debug(
                "session audit record failed for '%s': %s", agent_id, e,
            )

    async def _run_navigator_probe(self, inst: CamoufoxInstance) -> None:
        """Read key navigator/Intl signals from the live page and validate
        them against the configured fingerprint.

        Stores the result on ``inst.probe_result`` and emits a one-shot
        ``nav_probe`` payload via ``self._metrics_sink`` (when wired). At
        the dashboard layer this surfaces as a ``browser_nav_probe``
        event, distinct from per-minute drain payloads.

        **Probes on a temporary ``about:blank`` page** so we read the
        platform / browser signals as the engine sees them, not as some
        loaded site has possibly shadowed them via an injected content
        script. With ``persistent_context=True`` the page resumes whatever
        the agent had open last session; the probe must not clobber that
        active tab just to get a clean diagnostic read.

        Mismatches the probe flags:
          * ``navigator.webdriver !== false`` — the canonical bot tell
          * ``navigator.platform`` doesn't match our configured ``os`` hint
          * ``navigator.userAgent`` lacks ``Firefox/`` (would mean §6.4
            tripwire was bypassed somehow at runtime)

        Per the plan: "WARNING if webdriver !== false or mismatch."

        Note: ``navigator.connection`` is intentionally NOT checked.
        Real Firefox doesn't expose NetworkInformation; we read the
        signal field for diagnostics but do not flag its absence.
        """
        os_hint = os.environ.get("BROWSER_OS", "windows").lower()
        expected_platform = {
            "windows": "Win32",
            "macos": "MacIntel",
            "linux": "Linux x86_64",
        }.get(os_hint)

        probe_page = inst.page
        temp_page = None
        # Best-effort isolate the probe context. ``about:blank`` is a
        # special URL that Firefox loads instantly with a fresh,
        # script-free document — perfect for reading raw navigator
        # signals. Use a temporary page so a persistent-profile restart
        # keeps the agent's active/resumed tab intact. If a temp page
        # cannot be opened, fall back to probing the current page without
        # navigating it.
        try:
            new_page = getattr(inst.context, "new_page", None)
            if callable(new_page):
                temp_page = await new_page()
                probe_page = temp_page
                await temp_page.goto("about:blank", timeout=5000)
        except Exception as e:
            logger.debug(
                "Probe temporary about:blank page failed for '%s' "
                "(continuing on current page without navigation): %s",
                inst.agent_id, e,
            )
            if temp_page is not None:
                try:
                    await temp_page.close()
                except Exception:
                    pass
                temp_page = None
            probe_page = inst.page

        # Cap the probe so a hung renderer (rare but observed under
        # heavy CPU pressure or stuck Camoufox stacks) cannot block
        # ``get_or_start`` indefinitely. The probe holds the manager
        # lock through this call site, so a stall here freezes every
        # other agent's start/stop/touch.
        try:
            signals = await asyncio.wait_for(
                probe_page.evaluate(
                    "() => ({"
                    "  webdriver: navigator.webdriver,"
                    "  plugins_len: navigator.plugins ? navigator.plugins.length : -1,"
                    "  mimeTypes_len: navigator.mimeTypes ? navigator.mimeTypes.length : -1,"
                    "  hardwareConcurrency: navigator.hardwareConcurrency,"
                    "  deviceMemory: navigator.deviceMemory,"
                    "  userAgent: navigator.userAgent,"
                    "  platform: navigator.platform,"
                    "  language: navigator.language,"
                    "  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,"
                    "  conn_effective: navigator.connection ? navigator.connection.effectiveType : null,"
                    "  conn_downlink: navigator.connection ? navigator.connection.downlink : null,"
                    "  conn_rtt: navigator.connection ? navigator.connection.rtt : null"
                    "})",
                ),
                timeout=10.0,
            )
        except Exception as e:
            inst.probe_result = {
                "ok": False, "mismatches": [f"evaluate failed: {e}"],
                "signals": {},
            }
            logger.warning(
                "Navigator probe evaluate failed for '%s': %s",
                inst.agent_id, e,
            )
            return
        finally:
            if temp_page is not None:
                try:
                    await temp_page.close()
                except Exception as e:
                    logger.debug(
                        "Probe temporary page close failed for '%s': %s",
                        inst.agent_id, e,
                    )

        mismatches: list[str] = []
        if signals.get("webdriver") is not False:
            mismatches.append(
                f"webdriver={signals.get('webdriver')!r} (expected False)",
            )
        if expected_platform and signals.get("platform") != expected_platform:
            mismatches.append(
                f"platform={signals.get('platform')!r} "
                f"(expected {expected_platform!r} for os={os_hint!r})",
            )
        ua = signals.get("userAgent", "")
        if ua and "Firefox/" not in ua:
            mismatches.append(f"userAgent lacks 'Firefox/': {ua!r}")
        # ``navigator.connection`` is intentionally absent on the Firefox
        # UA path (matches real Firefox population — see comment near
        # ``_NAV_CONNECTION_INIT_SCRIPT``). Read for diagnostics but do
        # not flag its absence as a mismatch.

        ok = not mismatches
        inst.probe_result = {
            "ok": ok,
            "mismatches": mismatches,
            "signals": signals,
        }

        if ok:
            logger.info(
                "Navigator probe OK for '%s': platform=%s, ua=%s, tz=%s",
                inst.agent_id, signals.get("platform"),
                _short_ua(ua), signals.get("timezone"),
            )
        else:
            logger.warning(
                "Navigator probe MISMATCH for '%s': %s",
                inst.agent_id, "; ".join(mismatches),
            )

        # One-shot emit so operators see the result on the dashboard
        # without waiting for the next per-minute drain. Distinguished
        # from drain payloads by the ``kind`` field; Phase 2.1's history
        # buffer + mesh poll forward both shapes. We write to the history
        # buffer FIRST (so a missing sink doesn't drop the event) and
        # call the optional sink for in-process consumers (tests).
        probe_payload = {
            "kind": "nav_probe",
            "agent_id": inst.agent_id,
            "ok": ok,
            "mismatches": mismatches,
            "signals": signals,
        }
        self._metrics_seq += 1
        probe_payload["seq"] = self._metrics_seq
        probe_payload["ts"] = time.time()
        self._metrics_history.append(probe_payload)
        if self._metrics_sink is not None:
            try:
                self._metrics_sink(probe_payload)
            except Exception as e:
                logger.debug(
                    "metrics_sink raised on nav_probe for '%s': %s",
                    inst.agent_id, e,
                )

    async def stop(self, agent_id: str) -> None:
        """Stop and clean up a specific agent's browser."""
        async with self._manager_lock():
            await self._stop_instance(agent_id)

    async def _stop_instance(self, agent_id: str) -> None:
        """Internal stop — caller must hold ``self._manager_lock()``."""
        inst = self._instances.pop(agent_id, None)
        if inst is None:
            return
        # Drain counters BEFORE the instance disappears from the fleet.
        # Otherwise any clicks / snapshots / nav attempts since the last
        # minute-tick are silently lost when idle cleanup or explicit
        # stop fires. The periodic _emit_metrics hook only sees
        # still-live instances; post-pop is the final accounting chance.
        # Always write to the history buffer (even without a sink) so the
        # mesh poller sees the last minute of activity for a freshly-stopped
        # agent on its next tick. Empty payloads are skipped — no point
        # flooding the history with no-ops for agents that never did
        # anything.
        try:
            payload = inst.drain_metrics()
            if not _is_empty_payload(payload):
                self._metrics_seq += 1
                payload["seq"] = self._metrics_seq
                payload["ts"] = time.time()
                self._metrics_history.append(payload)
                if self._metrics_sink is not None:
                    self._metrics_sink(payload)
        except Exception as e:
            logger.warning(
                "Final metrics drain failed for '%s': %s", agent_id, e,
            )
        if self._user_focused_agent == agent_id:
            self._user_focused_agent = None
        # Drop the agent's solve-rate bucket so a long-running deployment
        # with rotating agent ids doesn't accumulate dead entries forever.
        # Acquired separately because the rate-limit lock and manager
        # lock are independent; reordering would risk a future deadlock.
        try:
            async with _get_solve_rate_lock():
                _solve_rate_window.pop(agent_id, None)
        except RuntimeError:
            # Outside a running loop (shouldn't happen — _stop_instance
            # is only called from async paths), best-effort.
            _solve_rate_window.pop(agent_id, None)
        # §22 — cancel any in-flight fingerprint health monitors so they
        # cannot record spurious rejection signals against a torn-down
        # Page.  We do NOT drop the rolling window itself: the operator
        # may want to inspect the burn state via the dashboard endpoint
        # AFTER stopping a flagged agent, and a future re-launch with the
        # same agent id should surface the existing window so the burn
        # alert isn't lost on a routine restart.  The window is cleared
        # explicitly by the ``fingerprint-health/reset`` endpoint.
        monitor_tasks = list(
            getattr(inst, "_fingerprint_monitor_tasks", set()) or [],
        )
        for task in monitor_tasks:
            task.cancel()
        for task in monitor_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        jitter = getattr(inst, '_jitter_task', None)
        if jitter:
            jitter.cancel()
        # §5.3 dump the behavior recorder buffer (no-op when disabled or
        # empty). Runs before ``context.close()`` so a hung browser close
        # doesn't eat the diagnostic data. Acquire ``inst.lock`` first so
        # any in-flight click/type/scroll/navigate on this instance
        # finishes its ``record_*`` append BEFORE we flush — otherwise
        # the last 1-2 events land in the deque after the dump has
        # already cleared it and are silently lost.
        recorder = getattr(inst, "recorder", None)
        if recorder is not None:
            try:
                async with inst.lock:
                    # Recorder dump writes JSONL synchronously and the
                    # buffer can hold ~10K events (multi-MB). Off-load
                    # to a worker thread so the manager lock isn't held
                    # across blocking file I/O while every other agent
                    # waits on _emit_metrics or lifecycle ops.
                    await asyncio.to_thread(recorder.dump, reason="stop")
            except Exception as e:
                logger.debug(
                    "Recorder dump failed for '%s': %s", agent_id, e,
                )
        # §20 — snapshot session state BEFORE ``context.close()``. Order
        # matters: ``storage_state()`` queries the live BrowserContext, so
        # closing first would give us a closed-context error. Best-effort:
        # never block shutdown on a failed snapshot. Drops the per-agent
        # elapsed counter so a re-launch starts a fresh interval.
        try:
            from src.browser.flags import get_bool
            from src.browser.session_persistence import snapshot_session
            if get_bool(
                "BROWSER_SESSION_PERSISTENCE_ENABLED", False, agent_id=agent_id,
            ):
                ok = await snapshot_session(agent_id, inst.context)
                try:
                    await _record_session_audit_event(
                        agent_id, "session_snapshot", ok,
                    )
                except Exception as e:
                    logger.debug(
                        "session audit record failed for '%s': %s",
                        agent_id, e,
                    )
        except Exception as e:
            logger.warning(
                "Stop-time session snapshot for '%s' failed: %s",
                agent_id, e,
            )
        self._session_snapshot_elapsed_s.pop(agent_id, None)
        try:
            await inst.context.close()
        except Exception as e:
            logger.debug("Error closing browser for '%s': %s", agent_id, e)
        logger.info("Stopped browser for '%s'", agent_id)

    async def stop_all(self) -> None:
        """Stop all browser instances and clean up Playwright."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if getattr(self, "_upload_recv_gc_task", None):
            self._upload_recv_gc_task.cancel()
            try:
                await self._upload_recv_gc_task
            except (asyncio.CancelledError, Exception):
                pass
            self._upload_recv_gc_task = None
        if getattr(self, "_download_gc_task", None):
            self._download_gc_task.cancel()
            try:
                await self._download_gc_task
            except (asyncio.CancelledError, Exception):
                pass
            self._download_gc_task = None
        async with self._manager_lock():
            for agent_id in list(self._instances.keys()):
                await self._stop_instance(agent_id)
        if self._captcha_solver:
            await self._captcha_solver.close()
        if self._playwright:
            with contextlib.suppress(Exception):
                await self._pw_context.__aexit__(None, None, None)
            self._playwright = None

    async def reset(self, agent_id: str) -> None:
        """Reset browser session — close and reopen (preserves profile)."""
        await self.stop(agent_id)
        # Next get_or_start will create a fresh instance with same profile

    async def _apply_platform_pre_nav_delay(
        self, agent_id: str, url: str,
    ) -> None:
        """Sleep a platform-specific Gaussian dwell before navigating.

        No-op when ``url`` is not on the protected-platform list (LinkedIn
        / X / Twitter / Meta) or when the operator has disabled the
        feature via ``BROWSER_PLATFORM_TIMING_ENABLED=false``. The flag
        check lives here (not in ``stealth.py``) so the operator-tuning
        and per-agent-override paths flow through the standard
        :mod:`src.browser.flags` precedence chain. See
        :func:`src.browser.stealth.pick_platform_pre_nav_delay` for the
        sampling logic and per-platform tuning rationale.
        """
        from src.browser.flags import get_bool
        if not get_bool("BROWSER_PLATFORM_TIMING_ENABLED", True, agent_id=agent_id):
            return
        delay_s, label = pick_platform_pre_nav_delay(url)
        if delay_s <= 0 or label is None:
            return
        # Single INFO log per applied delay so operators can correlate
        # latency reports with the platform-timing posture without
        # having to chase agent-side traces. URL is intentionally NOT
        # logged at INFO — the host label is enough; full URLs flow
        # through the recorder's redaction pipeline elsewhere.
        logger.info(
            "platform_pre_nav_delay agent=%s platform=%s delay=%.2fs",
            agent_id, label, delay_s,
        )
        # Record into the per-minute audit aggregator so the dashboard's
        # per-platform success panel can show count + average dwell.  Use
        # the canonical host (lower-cased, ``www.`` stripped) so binning
        # matches the dashboard side and the captcha audit path.  The
        # call is best-effort — a recording failure must not block the
        # navigation it's instrumenting.
        try:
            host = _canonical_host(url) or label
            await _record_platform_timing_audit_event(agent_id, host, delay_s)
        except Exception as e:
            logger.debug("platform-timing audit record failed: %s", e)
        await asyncio.sleep(delay_s)

    def set_proxy_config(self, agent_id: str, config: dict | None) -> None:
        """Store proxy config for an agent. Pass None to clear."""
        if config is None:
            self._proxy_configs.pop(agent_id, None)
        else:
            self._proxy_configs[agent_id] = config

    def get_proxy_config(self, agent_id: str) -> dict | None:
        """Get stored proxy config for an agent, or None."""
        return self._proxy_configs.get(agent_id)

    async def get_status(self, agent_id: str) -> dict:
        """Get status for a specific agent's browser.

        Includes the rolling 100-click success rate (§5.2) as a live gauge —
        distinct from the per-minute counters, which only flow via EventBus.
        Operators polling /status see the current health signal without
        waiting for the next emit tick.
        """
        async with self._manager_lock():
            inst = self._instances.get(agent_id)
            if not inst:
                return {"running": False}
            status = {
                "running": True,
                "idle_seconds": int(time.time() - inst.last_activity),
                "url": inst.page.url if inst.page else "",
                "click_window_size": len(inst.click_window),
                "click_success_rate_100": inst.rolling_click_success_rate(),
            }
            # §6.3 navigator probe summary (boot-once). When ``probe_result``
            # is None the probe hasn't run yet (instance just started).
            # Operators polling /status get the same signal as the dashboard
            # nav-probe event; we only surface the high-level shape, not the
            # raw signals payload (those are in the EventBus event).
            if inst.probe_result is not None:
                status["probe_ok"] = inst.probe_result["ok"]
                status["probe_mismatches"] = list(
                    inst.probe_result.get("mismatches") or [],
                )
            return status

    async def get_service_status(self) -> dict:
        """Get overall service health."""
        async with self._manager_lock():
            return {
                "healthy": True,
                "active_browsers": len(self._instances),
                "max_concurrent": self.max_concurrent,
                "agents": list(self._instances.keys()),
                "boot_id": self.boot_id,
            }

    async def get_fingerprint_health(self, agent_id: str) -> dict:
        """§22 — return the per-agent fingerprint health summary.

        Returns the contract-stable shape from :func:`_get_fingerprint_health`
        without requiring a live BrowserContext — the metric persists on
        module-level state, so the operator can inspect a flagged agent
        even after it has been stopped (useful when deciding whether to
        rotate the profile before restart).

        Defense-in-depth: ``agent_id`` is regex-validated even though
        the dashboard-side proxy already enforces this — a future caller
        that bypasses the proxy must not be able to harvest module-level
        state for arbitrary keys.  An invalid id returns the empty shape
        rather than raising, matching the read-mostly contract.
        """
        if not _AGENT_ID_RE.fullmatch(agent_id or ""):
            return {
                "window_size": 0,
                "rejection_rate": 0.0,
                "burned": False,
                "last_signal_ts": None,
            }
        return await _get_fingerprint_health(agent_id)

    async def reset_fingerprint_health(self, agent_id: str) -> dict:
        """§22 — clear the rolling rejection window for ``agent_id``.

        Returns ``{"reset": <bool>}`` indicating whether any state was
        cleared.  Operator-only — gated upstream by the dashboard CSRF
        check / browser-service bearer auth.  The expectation is that
        the operator has rotated the profile manually first; the
        endpoint does NOT touch the BrowserContext or storage state.

        Defense-in-depth: ``agent_id`` is regex-validated.  An invalid
        id returns ``{"reset": False}`` without touching state.
        """
        if not _AGENT_ID_RE.fullmatch(agent_id or ""):
            return {"reset": False}
        cleared = await _reset_fingerprint_window(agent_id)
        return {"reset": cleared}

    async def focus(self, agent_id: str) -> bool:
        """Bring an agent's browser window to VNC foreground.

        Auto-starts the browser if it isn't running yet, so the user
        always sees a window when they click "Browser" in the dashboard.

        Also records this as the user's explicitly focused agent so
        ``refocus_active()`` keeps this window visible even when other
        agents are using their browsers in the background.

        Two-layer raise:
        1. bring_to_front() — browser-protocol level (activates the tab)
        2. xdotool windowmap + windowraise — X11 level (unmaps if iconic,
           then raises in the stacking order so VNC actually sees it)
        """
        self._user_focused_agent = agent_id
        try:
            inst = await self.get_or_start(agent_id)
        except Exception as e:
            logger.warning("Focus: browser failed to start for '%s': %s", agent_id, e)
            self._user_focused_agent = None
            return False
        async with inst.lock:
            try:
                await inst.page.bring_to_front()
                inst.touch()
            except Exception as e:
                logger.debug("Focus failed for '%s': %s", agent_id, e)
                return False
            # Best-effort X11 raise so VNC sees the window. bring_to_front()
            # only works at the browser-protocol layer; on X11 with Openbox
            # the OS window can still be below the root window (e.g. after a
            # popup closes without returning focus). windowmap handles the
            # minimised/iconic case; windowraise moves it to the top of the
            # stacking order. Failures here are non-fatal — the tab is already
            # focused at the protocol level.
            #
            # When a specific X11 window ID is known, target it directly.
            # Without this, `search --class firefox` matches ALL Firefox
            # windows and raises whichever it finds first — breaking
            # per-agent browser switching on the shared VNC display.
            try:
                wid = inst.x11_wid
                if wid:
                    wid_s = str(wid)
                    cmd = ["xdotool", "windowmap", "--sync", wid_s,
                           "windowraise", wid_s, "windowfocus", wid_s]
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(cmd, capture_output=True, timeout=3),
                    )
                else:
                    # No WID known — skip xdotool entirely.  The fallback
                    # `search --class firefox` matches ALL Firefox windows
                    # and raises whichever it finds first, which is wrong
                    # in multi-agent scenarios.  bring_to_front() above
                    # already activated the correct tab at the browser-
                    # protocol level.
                    logger.debug(
                        "No X11 WID for '%s'; skipping xdotool raise", agent_id,
                    )
            except Exception as e:
                logger.debug("xdotool raise skipped for '%s': %s", agent_id, e)
            return True

    async def set_user_control(self, agent_id: str, enabled: bool) -> dict:
        """Toggle user browser control.

        When enabled, pauses agent X11 input (mouse jitter, click, type,
        scroll) so the user can interact via VNC without cursor fighting.
        Browser read operations (snapshot, screenshot) remain available.
        """
        inst = self._instances.get(agent_id)
        if not inst:
            return {"success": False, "error": "No browser instance"}
        inst._user_control = enabled
        logger.info(
            "User %s browser control for %s",
            "took" if enabled else "released", agent_id,
        )
        return {"success": True, "user_control": enabled}

    # ── Browser operations ──────────────────────────────────

    async def navigate(
        self, agent_id: str, url: str, wait_ms: int = 1000,
        wait_until: str = "domcontentloaded",
        snapshot_after: bool = False,
        referer: str | None = None,
    ) -> dict:
        """Navigate to URL and return page text.

        wait_until controls when Playwright considers navigation complete:
          - "domcontentloaded" (default): HTML parsed; fast but JS may not have run
          - "load": all resources loaded; good for most sites
          - "networkidle": no network requests for 500ms; best for heavy SPAs (X, etc.)
          - "commit": first byte received; fastest, rarely useful

        referer (Phase 3 §6.5): override the Referer header / document.referrer
        for this nav. ``None`` (default) lets the service pick a plausible
        value from :func:`src.browser.stealth.pick_referer` based on the
        target host and the agent's recent nav history. Pass an empty
        string ``""`` to explicitly send NO referer (equivalent to a
        bookmarked / typed-URL arrival). Pass a specific URL to override
        the picker entirely — useful when the agent is following a known
        link from a specific page.
        """
        # Validate URL scheme
        try:
            parsed = urlparse(url)
        except Exception:
            return {"success": False, "error": "Invalid URL"}
        if parsed.scheme.lower() in _BLOCKED_URL_SCHEMES:
            return {"success": False, "error": f"URL scheme '{parsed.scheme}' is not allowed"}
        if wait_until not in _VALID_WAIT_UNTIL:
            valid = sorted(_VALID_WAIT_UNTIL)
            return {"success": False, "error": f"Invalid wait_until: {wait_until!r}. Use one of: {valid}"}
        # Cap wait_ms
        wait_ms = max(0, min(wait_ms, _MAX_WAIT_MS))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            if inst._user_control:
                return {
                    "success": False,
                    "error": "User has browser control — action paused until control is released.",
                }

            # §6.5 referer realism. ``referer is None`` ⇒ picker decides;
            # explicit ``""`` ⇒ direct navigation (no referer); any other
            # string ⇒ caller override, validated before reaching
            # Playwright (the agent skill is LLM-callable so a malformed
            # value can land here from untrusted-by-default input).
            if referer is None:
                # Only honour ``inst.page.url`` as the previous-URL hint
                # AFTER the first navigate this session — otherwise a
                # persistent profile resume would falsely indicate
                # internal-link arrival on the very first nav.
                previous_url = (
                    inst.page.url if inst.page and inst.had_real_navigate
                    else ""
                )
                resolved_referer = pick_referer(
                    url,
                    previous_url=previous_url,
                    recent_referers=tuple(inst.recent_referers),
                )
            else:
                # Caller override — must validate. ValueError surfaces
                # to the agent as a navigate error; better than silently
                # forwarding ``javascript:alert(1)`` to Playwright.
                try:
                    resolved_referer = validate_referer(referer)
                except ValueError as e:
                    return {
                        "success": False,
                        "error": f"invalid referer: {e}",
                    }

            # Maintain the rolling-5 history on the instance. Both
            # picker output and validated overrides are tracked so the
            # picker can see "we just used a direct/social/search
            # pattern" and rotate accordingly.
            inst.recent_referers.append(resolved_referer)

            # Per-platform pre-nav dwell. Fires only on known high-
            # protection platforms (LinkedIn / X / Meta) where in-house
            # behavioral models score sub-second arrivals as bot signal.
            # No-op for any other host.
            await self._apply_platform_pre_nav_delay(agent_id, url)

            # Playwright accepts ``referer`` for goto and sets both the
            # network header and document.referrer consistently. Empty
            # string ⇒ omit the kwarg ⇒ Playwright sends no Referer.
            goto_kwargs: dict = {"wait_until": wait_until, "timeout": 30000}
            if resolved_referer:
                goto_kwargs["referer"] = resolved_referer

            # Single retry on timeout — transient network issues get a second chance.
            for attempt in range(2):
                try:
                    await inst.page.goto(url, **goto_kwargs)
                    break
                except Exception as e:
                    if attempt == 0 and "timeout" in str(e).lower():
                        logger.debug("Navigation timeout, retrying: %s", url)
                        await asyncio.sleep(2)
                        continue
                    # Give up — if this was a timeout (including after retry),
                    # log it for §4.6 metrics. Non-timeout failures go in a
                    # generic bucket (just counted as click_fail… actually,
                    # navigation is distinct; only timeouts go here).
                    if "timeout" in str(e).lower():
                        inst.m_nav_timeout += 1
                    return {"success": False, "error": str(e)}

            # §5.3 recorder: log host only, never the full URL — query
            # strings and fragments routinely carry secrets.
            inst.recorder.record_navigate(
                host=parsed.hostname or "", wait_until=wait_until,
            )

            # §6.5: future navs may now use ``inst.page.url`` as a
            # previous-URL hint for the picker. The flag stays True for
            # the lifetime of this CamoufoxInstance; a browser restart
            # creates a new instance and resets it.
            inst.had_real_navigate = True

            inst.dialog_active = False
            inst.dialog_detected = False
            # Same-tab navigation invalidates every ref captured against
            # the prior document — the underlying elements are gone.
            # Without clearing, a later ``click(ref="e3")`` would call
            # ``get_by_role(role, name=name).nth(occ)`` against the NEW
            # page; any element with matching role+name+occurrence on
            # the new page gets silently clicked, RefStale never fires.
            # ``open_tab`` and ``switch_tab`` already do this; navigate
            # missed it. Also drop the diff baseline so the next snapshot
            # is reported as ``scope="navigation"`` rather than diffed
            # against the stale prior page.
            inst.refs = {}
            inst.last_snapshot.clear()
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000 + navigation_jitter())
            try:
                title = await inst.page.title()
                current_url = inst.page.url
                body_text = ""
                # Always extract body at the historical 5000-char cap so
                # we have a usable fallback if the snapshot path fails
                # below. We trim to a 1000-char preview only AFTER the
                # snapshot succeeds — that's when the agent has the full
                # element tree and doesn't need a long body. If the
                # snapshot fails, we ship the full body so the agent
                # isn't stranded with truncated text + empty snapshot.
                if not inst._js_snapshot_mode:
                    try:
                        _a11y = await inst.page.accessibility.snapshot()
                        body_text = _extract_text_from_a11y(
                            _a11y, max_chars=5000,
                        )
                    except AttributeError:
                        inst._js_snapshot_mode = True
                    except Exception:
                        pass
                result = {
                    "success": True,
                    "data": {
                        "url": self.redactor.redact(agent_id, current_url),
                        "title": self.redactor.redact(agent_id, title),
                        # Body filled in below once we know whether the
                        # optional snapshot succeeded — see body cap
                        # comment.
                        "body": "",
                    },
                }
                # Auto-detect CAPTCHAs so the agent knows immediately.
                # _check_captcha now always returns the §11.13 envelope:
                # only surface to the agent when something actually needs
                # their attention (found AND not auto-solved).
                envelope = await self._check_captcha(inst)
                if (
                    envelope.get("captcha_found")
                    and envelope.get("solver_outcome") != "solved"
                ):
                    result["captcha"] = _with_legacy_fields(envelope)
                snapshot_succeeded = False
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    snap_data = snap.get("data") or {}
                    result["snapshot"] = snap_data
                    snapshot_succeeded = bool(snap.get("success") and snap_data)
                # §7.6: shrink body to 1000-char preview ONLY when the
                # snapshot actually carried back element refs. A failed
                # snapshot would otherwise leave the agent with both a
                # truncated body AND an empty/{} snapshot — strictly
                # worse than the snapshot_after=False path. Restore the
                # full body in that failure case.
                final_body = (
                    body_text[:1000] if snapshot_succeeded else body_text
                )
                result["data"]["body"] = self.redactor.redact(
                    agent_id, final_body,
                )
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _build_a11y_tree(self, inst: CamoufoxInstance, root=None):
        """Get accessibility tree via the JS DOM walker.

        Phase 5 §8.3 collapsed the prior "native first / JS fallback"
        dispatch into the JS walker as the sole path. Firefox's native
        ``nsIAccessibilityService`` (exposed through Playwright's
        ``page.accessibility.snapshot()``) is faster and more invisible
        to anti-bot DOM proxies, but it does NOT pierce open shadow
        boundaries — so a default Camoufox path silently lost shadow
        content. Phase 5 (shadow + iframe support) made the JS walker
        the canonical implementation, and the native path was retained
        only as a perf optimisation; with shadow descent baked into the
        JS walker, dual paths produced inconsistent snapshots depending
        on which path won the race.

        Always runs the JS walker now, regardless of the (now-vestigial)
        ``inst._js_snapshot_mode`` flag. The flag remains in
        :class:`CamoufoxInstance` for the navigate-mode body extraction
        path that still consults ``page.accessibility.snapshot()`` for a
        cheap text-only summary; that path is independent of the
        snapshot tree builder.
        """
        try:
            if root:
                return await root.evaluate(_JS_A11Y_TREE)
            return await inst.page.evaluate(_JS_A11Y_TREE)
        except Exception as e:
            logger.debug("JS a11y tree builder failed: %s", e)
            return None

    async def snapshot(
        self,
        agent_id: str,
        filter: str | None = None,
        from_ref: str | None = None,
        diff_from_last: bool = False,
        frame: str | None = None,
        include_frames: bool = True,
    ) -> dict:
        """Get accessibility tree with element refs.

        ``filter`` constrains which elements appear in the result (§7.7):

        - ``None`` (default) — actionable + context roles (the historic
          behavior, balanced for general-purpose agents).
        - ``"actionable"`` — only roles the agent can act on
          (button, link, textbox, ...). Skips heading/img/landmark.
        - ``"inputs"`` — form-input roles only. Useful when the agent is
          mid-form and doesn't need the navigation skeleton.
        - ``"headings"`` — heading nodes only. ``browser_find_text`` for
          structural orientation when the page is huge.
        - ``"landmarks"`` — top-level region nodes only (navigation, main,
          complementary, ...). Cheap orientation pass before drilling in.

        ``from_ref`` (§7.4) restricts the snapshot to the subtree rooted
        at an element captured in a previous call. Pass an ``e<N>`` ref
        from ``inst.refs``; the returned tree shows only that element
        and its descendants. Combine with ``filter`` to focus a busy
        list on its inputs, etc.

        ``diff_from_last`` (§7.3) — when ``True``, compare against the
        cached baseline for the active tab and return a structural diff
        instead of the full tree:

        * scope ``"same"`` / ``"modal_opened"`` / ``"modal_closed"`` /
          ``"frame_changed"`` — payload is
          ``{added: [...], removed: [...], changed: [...],
             unchanged_count: N, scope: "..."}``.
        * scope ``"navigation"`` — main-frame URL changed since the
          previous snapshot. Diffing across navigation produces all-
          removed + all-added noise, so we return the full snapshot
          alongside ``scope: "navigation"``.
        * scope ``"tab_changed"`` — active tab differs from the one
          baselined. Same full-snapshot return path. The previous tab's
          baseline is retained under its ``page_id`` so returning to
          that tab via ``browser_switch_tab`` resumes diffing where it
          left off.

        Cross-PR (§7.3 ↔ §7.4/§7.7): when ``filter`` or ``from_ref`` is
        set, the resulting refs are a subset of the page. Persisting
        that subset as the diff baseline would cause the next
        unfiltered ``diff_from_last`` to report the omitted elements
        as ``removed``. ``_snapshot_impl`` skips the baseline update
        for scoped/filtered calls — they're informational, not anchors.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            result = await self._snapshot_impl(
                inst, agent_id,
                filter=filter,
                from_ref=from_ref,
                diff_from_last=diff_from_last,
                frame=frame,
                include_frames=include_frames,
            )
            # §11.4 / §18.2 — surface and CLEAR any pending captcha
            # envelope captured by ``_with_captcha_redetect`` after a
            # prior click/type/press_key/fill_form. This double-surfaces
            # the §11.13 envelope: agents that read the action response
            # see it inline; agents that poll via ``snapshot()`` after
            # acting see it here. Cleared on this read so a third
            # poll doesn't repeat the same envelope (the snapshot itself
            # would have called ``_check_captcha`` if the captcha is
            # still on the page — but we deliberately don't run a probe
            # here, snapshot is supposed to be cheap).
            pending = inst._pending_captcha_envelope
            if pending is not None and isinstance(result, dict) and result.get("success"):
                data = result.setdefault("data", {})
                data.setdefault("captcha", pending)
                inst._pending_captcha_envelope = None
            return result

    async def _snapshot_impl(
        self,
        inst: CamoufoxInstance,
        agent_id: str,
        filter: str | None = None,
        from_ref: str | None = None,
        diff_from_last: bool = False,
        _skip_baseline: bool = False,
        frame: str | None = None,
        include_frames: bool = True,
    ) -> dict:
        """Snapshot implementation.  Caller must hold ``inst.lock``."""
        # Read BROWSER_SNAPSHOT_FORMAT exactly once at entry so a flag
        # flip mid-call (operator settings hot-reload, agent override
        # change racing the snapshot) cannot produce mixed v1/v2 output
        # between the from_ref early-return and the main-path return.
        from src.browser.flags import get_str as _flag_get_str
        snapshot_fmt = (
            _flag_get_str(
                "BROWSER_SNAPSHOT_FORMAT", "v1", agent_id=agent_id,
            )
            .strip()
            .lower()
        )
        # Resolve the optional filter once so the inner _walk sees a
        # frozenset rather than re-deriving on every node.
        try:
            allowed_roles = _resolve_filter_roles(filter)
        except ValueError as e:
            return {
                "success": False,
                "error": {
                    "code": "invalid_input",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }
        try:
            if frame is not None and from_ref is not None:
                return {
                    "success": False,
                    "error": {
                        "code": "invalid_input",
                        "message": (
                            "frame and from_ref are mutually exclusive"
                        ),
                        "retry_after_ms": None,
                    },
                }
            target_frame = None
            if frame is not None:
                try:
                    target_frame = self._resolve_frame_arg(inst, frame)
                except RefStale as rs:
                    return {
                        "success": False,
                        "error": {
                            "code": "ref_stale",
                            "message": str(rs),
                            "retry_after_ms": None,
                        },
                    }
                if target_frame is None:
                    return {
                        "success": False,
                        "error": {
                            "code": "not_found",
                            "message": (
                                f"frame {frame!r} did not match any frame "
                                "url-substring or frame_id"
                            ),
                            "retry_after_ms": None,
                        },
                    }
            # ── Optional scoped root via ``from_ref`` (§7.4) ────────────────
            scoped_root_handle = None
            if from_ref is not None:
                if not from_ref:
                    return {
                        "success": False,
                        "error": {
                            "code": "invalid_input",
                            "message": "from_ref must be a non-empty ref id",
                            "retry_after_ms": None,
                        },
                    }
                if from_ref not in inst.refs:
                    return {
                        "success": False,
                        "error": {
                            "code": "not_found",
                            "message": f"ref {from_ref!r} not in current snapshot",
                            "retry_after_ms": None,
                        },
                    }
                # Resolve to a Playwright ElementHandle. ``_locator_from_ref``
                # may raise RefStale on closed-tab refs — that's the same
                # contract every other ref-using path follows; surface as
                # ref_stale so the agent re-snapshots.
                try:
                    handle_or_loc = await self._locator_from_ref(inst, from_ref)
                    if handle_or_loc is None:
                        return {
                            "success": False,
                            "error": {
                                "code": "not_found",
                                "message": f"ref {from_ref!r} could not be resolved",
                                "retry_after_ms": None,
                            },
                        }
                    if hasattr(handle_or_loc, "element_handle"):
                        scoped_root_handle = await handle_or_loc.element_handle(
                            timeout=2000,
                        )
                    else:
                        scoped_root_handle = handle_or_loc
                except RefStale as e:
                    return {
                        "success": False,
                        "error": {
                            "code": "ref_stale",
                            "message": str(e),
                            "retry_after_ms": None,
                        },
                    }
                if scoped_root_handle is None:
                    # Locator resolved structurally but the element is no
                    # longer in the DOM — common for dynamic SPAs that
                    # tear down and re-render between snapshots.
                    return {
                        "success": False,
                        "error": {
                            "code": "ref_stale",
                            "message": (
                                f"ref {from_ref!r} no longer attached to the page; "
                                "re-snapshot to get fresh refs"
                            ),
                            "retry_after_ms": None,
                        },
                    }

            if target_frame is not None and scoped_root_handle is None:
                try:
                    tree = await target_frame.evaluate(_JS_A11Y_TREE)
                except Exception as exc:
                    logger.debug(
                        "Frame snapshot evaluate failed for %s: %s",
                        agent_id, exc,
                    )
                    tree = None
            else:
                tree = await self._build_a11y_tree(
                    inst, root=scoped_root_handle,
                )
            if not tree:
                # Honor ``BROWSER_SNAPSHOT_FORMAT`` for the empty-tree
                # short-circuit too. Pre-fix this returned a bare string
                # regardless of format, so agents parsing on the
                # ``# snapshot-v2`` first-line marker classified empty
                # pages as v1 and could attempt v1-shaped re-parsing.
                empty_text = (
                    "# snapshot-v2\n(empty page)"
                    if snapshot_fmt == "v2"
                    else "(empty page)"
                )
                return {
                    "success": True,
                    "data": {"snapshot": empty_text, "refs": {}},
                }

            lines: list[str] = []
            refs: dict[str, RefHandle] = {}
            # Snapshot page_id up front — resolves to Page that was active
            # when the snapshot was taken. If the agent later switches tabs,
            # refs still carry their original page_id so resolution targets
            # the right tab (or raises RefStale if the tab is closed).
            snapshot_page_id = inst._page_id_for(inst.page)
            target_frame_id = (
                inst._register_frame(target_frame)
                if target_frame is not None
                else None
            )
            ref_counter = [0]
            # Counts occurrences of each (role, name) pair so we can
            # disambiguate duplicate elements (e.g. X's two composer nodes).
            occurrence_counts: dict[tuple, int] = {}

            # Collect entries for §7.2 v2 rendering AND build v1 lines in
            # parallel. The entry list is a structured intermediate so we
            # can pivot between formats post-walk without a second tree
            # traversal. Each entry: (ref_id, role, name, attr_str,
            # landmark, depth).
            entries: list[tuple[str, str, str, str, str, int]] = []
            # §7.3: diff-mode also needs the per-ref attr summary keyed by
            # element_key so changed-state detection can compare values.
            ref_summary: dict[str, dict] = {}

            pending_frames: list[tuple] = []

            def _walk(node, depth=0, current_frame_id=None, frame_nesting=0):
                if depth > _MAX_WALK_DEPTH:
                    return
                role = node.get("role", "")
                name = node.get("name", "")
                if role == "iframe":
                    # Stub name is title || src — long URLs in srcs are
                    # common; cap so a single iframe doesn't blow up the
                    # snapshot byte count.
                    if name:
                        name = name[:200]
                    frame_url = node.get("frame_url", "")
                    is_opaque = bool(node.get("opaque"))
                    iframe_index = node.get("iframe_index", -1)
                    landmark = node.get("landmark", "")
                    ctx_str = f" ({landmark})" if landmark else ""
                    suffix = " [cross-origin]" if is_opaque else ""
                    # ``frame_nesting`` here is the iframe-level of the
                    # CURRENT walking frame (0 = main). The stub being
                    # emitted is one level deeper, so descent would only
                    # succeed if ``frame_nesting + 1 <= _MAX_FRAME_NESTING``.
                    # When that fails we still emit the stub (so the agent
                    # sees the iframe exists) but tag it ``[depth-capped]``
                    # so the agent doesn't waste a turn fishing for refs
                    # that will never appear.
                    is_depth_capped = (
                        not is_opaque
                        and include_frames
                        and frame_nesting + 1 > _MAX_FRAME_NESTING
                    )
                    cap_suffix = " [depth-capped]" if is_depth_capped else ""
                    line = (
                        f"{'  ' * depth}- iframe \"{name}\""
                        f"{suffix}{cap_suffix}{ctx_str}"
                    )
                    lines.append(line)
                    # §7.2 v2 also needs to see iframe stubs — they were
                    # previously lines-only, which made them invisible
                    # under the v2 renderer. Emit a synthetic entry with
                    # an empty ref_id; ``_format_snapshot_v2`` renders
                    # iframe entries without the ``[ref_id]`` prefix to
                    # match the v1 wire shape (iframe stubs are not
                    # clickable handles).
                    # Preserve leading-space convention for v2 attr_str:
                    # ``suffix``/``cap_suffix`` already start with a space
                    # (e.g. ``" [cross-origin]"``), so concatenation gives
                    # ``" [cross-origin] [depth-capped]"`` when both apply.
                    iframe_attr = f"{suffix}{cap_suffix}"
                    entries.append(
                        ("", "iframe", name, iframe_attr, landmark, depth),
                    )
                    # Descend into same-origin iframes regardless of
                    # ``frame_url`` truthiness — srcdoc / about:blank
                    # iframes legitimately emit ``frame_url=""`` and
                    # must still be traversed via ``iframe_index``.
                    if not is_opaque and include_frames:
                        idx = iframe_index if isinstance(iframe_index, int) else -1
                        pending_frames.append(
                            (frame_url, depth, current_frame_id, idx),
                        )
                    return
                # ``allowed_roles=None`` means "use the historical default"
                # (actionable ∪ context). Any explicit filter shrinks that.
                if allowed_roles is None:
                    is_admitted = role in _ACTIONABLE_ROLES or role in _CONTEXT_ROLES
                else:
                    is_admitted = role in allowed_roles
                if is_admitted:
                    if ref_counter[0] < _MAX_SNAPSHOT_ELEMENTS:
                        ref_id = f"e{ref_counter[0]}"
                        ref_counter[0] += 1

                        # Materialize shadow_path NOW so the occurrence
                        # key folds it: stage-2 picks candidates inside
                        # one shadow root, so two same-named elements in
                        # different roots must not share an occurrence
                        # counter (would index past stage-2's candidates
                        # array → spurious RefStale).
                        raw_shadow = node.get("shadow_path") or ()
                        shadow_hops: tuple[ShadowHop, ...] = tuple(
                            ShadowHop(
                                selector=str(hop.get("selector", "")),
                                occurrence=int(hop.get("occurrence", 0)),
                                discriminator=str(hop.get("discriminator", "")),
                            )
                            for hop in raw_shadow
                        )
                        key = (role, name, shadow_hops)
                        occ = occurrence_counts.get(key, 0)
                        occurrence_counts[key] = occ + 1

                        attrs = []
                        if node.get("checked") is not None:
                            attrs.append(f"checked={node['checked']}")
                        if node.get("selected"):
                            attrs.append("selected")
                        if node.get("disabled"):
                            attrs.append("disabled")
                        if node.get("value"):
                            val = node["value"]
                            attrs.append(f"value={val}")
                        if occ > 0:
                            attrs.append(f"dup:{occ + 1}")
                        attr_str = f" [{', '.join(attrs)}]" if attrs else ""

                        # Structural context from nearest landmark ancestor
                        landmark = node.get("landmark", "")
                        ctx_str = f" ({landmark})" if landmark else ""

                        line = f"{'  ' * depth}- [{ref_id}] {role} \"{name}\"{attr_str}{ctx_str}"
                        lines.append(line)
                        entries.append(
                            (ref_id, role, name, attr_str, landmark, depth),
                        )
                        # §7.3: stable element-key for cross-snapshot
                        # diffing. Today only ``compute_element_key``'s
                        # role+name+landmark path (priorities 3/4) is
                        # wired; data-testid / dom-id extraction (priorities
                        # 1/2) is a follow-up that will plug into this
                        # same hash via ``test_id=`` / ``dom_id=`` kwargs
                        # without breaking already-seeded handles.
                        #
                        # Known keying limitations until that lands:
                        #
                        # - Two elements with the SAME role+name+landmark
                        #   collide on the same key. ``ref_summary`` keeps
                        #   the latest one (logged at debug); the previous
                        #   one falls out of the diff baseline. Diff misses
                        #   add/remove of duplicates within a single
                        #   landmark (e.g. multiple "Like" buttons in a
                        #   feed). Acceptable for v1: most agent flows
                        #   target uniquely-named elements.
                        # - Unnamed siblings (priority-4 fallback) include
                        #   ``sibling_index`` in their key. Removing one
                        #   shifts the surviving siblings' indices →
                        #   reported as a remove+add pair instead of
                        #   "unchanged". Same fix: data-testid extraction.
                        from src.browser.ref_handle import compute_element_key
                        # ``shadow_hops`` already materialized above so
                        # the occurrence key could fold it. frame_id
                        # stays constant today (no iframe walker until
                        # §8.4). ``compute_element_key`` folds both so
                        # identical role+name+landmark in distinct
                        # shadow roots produce different element_keys.
                        elem_key = compute_element_key(
                            role=role, name=name, landmark=landmark,
                            sibling_index=occ,
                            frame_id=current_frame_id,
                            shadow_path=shadow_hops,
                        )
                        # scope_root is finalized after the modal-scoping
                        # branch below. For now record the unscoped handle;
                        # we overwrite scope_root once we know the final
                        # dialog_active state (see scope-root patching below).
                        if shadow_hops:
                            refs[ref_id] = RefHandle.shadow(
                                page_id=snapshot_page_id,
                                scope_root=None,
                                shadow_path=shadow_hops,
                                role=role,
                                name=name,
                                occurrence=occ,
                                disabled=bool(node.get("disabled")),
                                element_key=elem_key,
                                frame_id=current_frame_id,
                            )
                        elif current_frame_id is not None:
                            refs[ref_id] = RefHandle(
                                page_id=snapshot_page_id,
                                frame_id=current_frame_id,
                                shadow_path=(),
                                scope_root=None,
                                role=role,
                                name=name,
                                occurrence=occ,
                                disabled=bool(node.get("disabled")),
                                element_key=elem_key,
                            )
                        else:
                            refs[ref_id] = RefHandle.light_dom(
                                page_id=snapshot_page_id,
                                scope_root=None,
                                role=role,
                                name=name,
                                occurrence=occ,
                                disabled=bool(node.get("disabled")),
                                element_key=elem_key,
                            )
                        # Diff-mode summary — keyed by element_key so the
                        # next snapshot can match across re-renders that
                        # change ref ids. A duplicate key inside one
                        # snapshot drops the earlier entry from the diff
                        # baseline and reports it as ``removed`` next time
                        # — log so the operator can investigate what's
                        # producing identical-looking siblings.
                        if elem_key in ref_summary:
                            logger.debug(
                                "element_key collision in snapshot for %s: "
                                "key=%s overwritten by ref %s (was %s); "
                                "diff baseline will lose the earlier one",
                                agent_id, elem_key, ref_id,
                                ref_summary[elem_key].get("ref_id"),
                            )
                        ref_summary[elem_key] = {
                            "ref_id": ref_id,
                            "role": role,
                            "name": name,
                            "landmark": landmark,
                            "disabled": bool(node.get("disabled")),
                            "value": node.get("value", ""),
                            "checked": node.get("checked"),
                        }
                for child in node.get("children", []):
                    _walk(child, depth + 1, current_frame_id, frame_nesting)

            async def _descend_frames(parent_frame, parent_frame_id,
                                      frame_nesting):
                # ``_MAX_FRAME_NESTING`` counts iframe levels beyond the
                # main frame. With cap=3: main(0) + L1 + L2 + L3 are
                # admitted; an L4 stub triggers this guard. The plan
                # §8.4 phrasing "Nesting cap: 3 levels" refers to these
                # iframe-only levels — main is treated as the host
                # surface and not counted.
                if frame_nesting >= _MAX_FRAME_NESTING:
                    capped = [
                        e for e in pending_frames if e[2] == parent_frame_id
                    ]
                    if capped:
                        # The corresponding stub lines were emitted by
                        # ``_walk`` already with a ``[depth-capped]`` tag
                        # so the agent can see them. Log the count so an
                        # operator chasing "where did the deep iframe go?"
                        # has a single grep target.
                        try:
                            parent_url = parent_frame.url
                        except Exception:
                            parent_url = "<unknown>"
                        logger.debug(
                            "iframe depth cap reached at %s; %d nested "
                            "frame(s) not descended",
                            parent_url, len(capped),
                        )
                    pending_frames[:] = [
                        e for e in pending_frames
                        if e[2] != parent_frame_id
                    ]
                    return
                drained = [
                    e for e in pending_frames if e[2] == parent_frame_id
                ]
                pending_frames[:] = [
                    e for e in pending_frames if e[2] != parent_frame_id
                ]
                # Track per-parent-frame consumed children so two stubs
                # that share a URL (e.g. duplicate ad iframes from one
                # provider, or two srcdoc iframes that emit empty
                # ``frame_url``) descend into DISTINCT child frames
                # instead of both matching the first hit. Keyed by
                # ``id(frame)`` because Playwright Frame objects aren't
                # hashable in all binding versions.
                consumed_child_ids: set[int] = set()
                try:
                    children_frames = list(parent_frame.child_frames)
                except Exception:
                    children_frames = []
                # Filter to frames whose underlying iframe element is still
                # attached. Playwright may briefly retain ``Frame`` objects
                # whose ``<iframe>`` was just removed (GC lag); the JS
                # walker counts only LIVE iframes via
                # ``ownerDocument.querySelectorAll``, so its
                # ``iframe_index`` is positional over live frames only. If
                # we let detached frames stay in ``children_frames``, the
                # index-based fallback below would target the wrong slot.
                live_children: list = []
                for cf in children_frames:
                    try:
                        if hasattr(cf, "is_detached") and cf.is_detached():
                            continue
                    except Exception:
                        # ``is_detached()`` itself may raise on torn-down
                        # bindings — fall through and trust the URL match
                        # tiers below to handle it.
                        pass
                    live_children.append(cf)
                for stub_url, stub_depth, _stub_parent_id, stub_index in drained:
                    target_child = None
                    # Phase 1: prefer exact URL match against an unconsumed
                    # child. Falls through to substring + iframe_index
                    # tiers so stubs whose ``frame_url`` was a partial
                    # match (or empty for srcdoc) still resolve.
                    if stub_url:
                        for cf in live_children:
                            if id(cf) in consumed_child_ids:
                                continue
                            try:
                                cf_url = cf.url or ""
                            except Exception:
                                cf_url = ""
                            if cf_url == stub_url:
                                target_child = cf
                                break
                        if target_child is None:
                            for cf in live_children:
                                if id(cf) in consumed_child_ids:
                                    continue
                                try:
                                    cf_url = cf.url or ""
                                except Exception:
                                    cf_url = ""
                                if stub_url in cf_url:
                                    target_child = cf
                                    break
                        if target_child is None:
                            # Walker emitted ``frame_url=src`` from the
                            # iframe ATTRIBUTE; by the time descent runs,
                            # an in-page navigation may have changed
                            # ``frame.url``. Falls through to the index
                            # tier below — log so debugging walks aren't
                            # blind to this race.
                            try:
                                live_urls = [c.url for c in live_children]
                            except Exception:
                                live_urls = []
                            logger.debug(
                                "Frame URL changed during snapshot for "
                                "%s: stub_url=%r, live child urls=%r",
                                agent_id, stub_url, live_urls,
                            )
                    if target_child is None:
                        # Empty frame_url (srcdoc / about:blank-style
                        # anonymous iframe) or no URL match: fall back
                        # to the JS-emitted sibling index. The walker
                        # tags every iframe stub with its position
                        # among the parent document's iframes so two
                        # same-URL or empty-URL siblings are still
                        # individually addressable. Index is positional
                        # over LIVE iframes (matching the walker's
                        # ``ownerDocument.querySelectorAll`` count), so
                        # we index into ``live_children`` not
                        # ``children_frames``.
                        if (
                            stub_index is not None
                            and 0 <= stub_index < len(live_children)
                            and id(live_children[stub_index])
                            not in consumed_child_ids
                        ):
                            target_child = live_children[stub_index]
                    if target_child is None and len(live_children) == 1 and not consumed_child_ids:
                        # Last-resort single-child fallback preserves
                        # behavior on pages where the JS walker emitted
                        # neither URL nor index (e.g. legacy stubs from
                        # tests written before iframe_index existed).
                        target_child = live_children[0]
                    if target_child is None:
                        continue
                    consumed_child_ids.add(id(target_child))
                    child_frame_id = inst._register_frame(target_child)
                    try:
                        # Cross-origin classification (``isCrossOriginFrame``
                        # in ``_JS_A11Y_TREE``) runs within each frame's
                        # OWN context, so a grandchild iframe's ``opaque``
                        # bit reflects ITS origin vs its parent's, not vs
                        # the main frame's. Spec-correct (Firefox SOP),
                        # but means a grandchild whose origin matches main
                        # may still be classified opaque if its parent is
                        # on a different origin.
                        sub_tree = await target_child.evaluate(_JS_A11Y_TREE)
                    except Exception as exc:
                        logger.debug(
                            "Frame walk failed for %s: %s", agent_id, exc,
                        )
                        continue
                    if not sub_tree:
                        continue
                    _walk(
                        sub_tree, stub_depth + 1, child_frame_id,
                        frame_nesting + 1,
                    )
                    await _descend_frames(
                        target_child, child_frame_id, frame_nesting + 1,
                    )

            # When ``from_ref`` is set the caller is asking for a deep
            # scope into a specific element; modal-detection would
            # second-guess that intent, so skip straight to a single
            # ``_walk`` of the scoped tree.
            if scoped_root_handle is not None:
                # Snapshot the live modal state BEFORE walking — clearing
                # ``dialog_active`` here would let subsequent
                # ``_locator_from_ref`` resolutions of these scoped refs
                # bypass modal scoping, allowing duplicate role+name
                # elements behind the overlay to silently match. We keep
                # the live state intact so the next non-scoped snapshot
                # re-detects modals naturally; in the meantime, scoped
                # refs taken inside a modal still resolve through the
                # modal selector.
                was_modal = bool(inst.dialog_active)
                _walk(tree, 0, target_frame_id)
                pending_frames.clear()
                # If the agent took this scoped snapshot while a modal
                # was open, patch ``scope_root`` on every emitted ref so
                # ``_locator_from_ref`` keeps clicks bounded to the dialog
                # subtree. Without this the scoped refs could resolve to
                # identical-named elements behind the overlay.
                if was_modal:
                    for rid, handle in refs.items():
                        if handle.scope_root is None:
                            refs[rid] = RefHandle(
                                page_id=handle.page_id,
                                frame_id=handle.frame_id,
                                shadow_path=handle.shadow_path,
                                scope_root=_MODAL_SELECTOR,
                                role=handle.role,
                                name=handle.name,
                                occurrence=handle.occurrence,
                                disabled=handle.disabled,
                                element_key=handle.element_key,
                            )
                # NOTE: scoped snapshots replace ``inst.refs`` because
                # the walker assigns fresh ``e0..eN`` ids on every walk
                # and there is no separate namespace for scoped refs.
                # That means a prior full snapshot's refs become
                # unresolvable after a ``snapshot(from_ref=...)``;
                # callers should re-snapshot the full tree before
                # operating outside the scope, or queue all scope-local
                # actions before the next full snapshot. Documenting
                # the contract loudly here — the docstring's
                # "informational, not anchors" framing is aspirational;
                # making it true requires a per-call ref namespace,
                # which is a wire-format change.
                inst.refs = refs
                # Cross-PR fix (#749 ↔ #750): the from_ref early-return
                # must honor the same v2 dispatch as the main return
                # path. Without this, scoped snapshots stayed v1 even
                # when ``BROWSER_SNAPSHOT_FORMAT=v2`` was set — agents
                # parsing on the ``# snapshot-v2`` first-line marker
                # would see mixed formats. ``snapshot_fmt`` was captured
                # at the top of ``_snapshot_impl`` so a flag flip mid-
                # call can't desynchronize the two return paths.
                if snapshot_fmt == "v2":
                    snapshot_text = _format_snapshot_v2(lines, entries)
                else:
                    snapshot_text = (
                        "\n".join(lines) if lines else "(no interactive elements)"
                    )
                snapshot_text = self.redactor.redact(agent_id, snapshot_text)
                inst.m_snapshot_bytes.append(len(snapshot_text))
                response_refs = {
                    rid: h.to_agent_dict() for rid, h in refs.items()
                }
                return {
                    "success": True,
                    "data": {"snapshot": snapshot_text, "refs": response_refs},
                }

            # When a modal dialog is open, scope to only dialog elements
            # so agents don't see/click elements behind the overlay
            # (e.g. X's sidebar "Post" button behind the compose modal).
            #
            # Skip the modal probe when ``target_frame is not None`` —
            # the iframe-scoped snapshot reports refs from the iframe's
            # own document, but the main-page modal probe runs against
            # ``inst.page``. Pre-fix, a main-page-modal-open page that
            # also took a ``frame=`` snapshot would tag every iframe-
            # scoped ref with ``scope_root=_MODAL_SELECTOR``; the
            # resolver would then run ``frame.locator(scope_root)``
            # against the iframe's document and silently miss the
            # modal that lives in the parent.
            if target_frame is not None:
                modal_els = []
            else:
                modal_els = await inst.page.query_selector_all(_MODAL_SELECTOR)
            vp = inst.page.viewport_size
            visible_modals = []
            for el in modal_els:
                if await self._is_visible_modal(el, vp):
                    visible_modals.append(el)

            # Deduplicate nested modals: if modal A contains modal B,
            # snapshot(root=A) already includes B's elements.
            if len(visible_modals) > 1:
                deduped = []
                for i, el in enumerate(visible_modals):
                    is_nested = False
                    for j, other in enumerate(visible_modals):
                        if i != j:
                            try:
                                if await other.evaluate(
                                    "(parent, child) => parent.contains(child)", el
                                ):
                                    is_nested = True
                                    break
                            except Exception:
                                pass
                    if not is_nested:
                        deduped.append(el)
                visible_modals = deduped if deduped else visible_modals

            if visible_modals:
                inst.dialog_detected = True
                inst.dialog_active = True
                lines.append("** Modal dialog is open — only dialog elements are shown **")
                for el in visible_modals:
                    subtree = await self._build_a11y_tree(inst, root=el)
                    if subtree:
                        _walk(subtree, 0, target_frame_id)
                # Modals are typically light-DOM in the main frame; iframe
                # descent skipped for scoping fidelity.
                pending_frames.clear()
                actionable_refs = [
                    r for r in refs.values() if r.role in _ACTIONABLE_ROLES
                ]
                # Progressive retry: 300 ms then 500 ms — gives SPAs like X
                # enough time for modal animations and Lexical editor init.
                retry_waits = [0.3, 0.5]
                while not actionable_refs and retry_waits:
                    wait = retry_waits.pop(0)
                    logger.debug(
                        "Modal scoping produced 0 actionable refs — "
                        "retrying after %.0f ms", wait * 1000,
                    )
                    await asyncio.sleep(wait)
                    refs.clear()
                    lines.clear()
                    # §7.2: ``entries`` is the parallel structure v2 renders
                    # from. Forgetting to reset it here would leak entries
                    # from the discarded scoping pass into the v2 output —
                    # invisible in v1 (which renders ``lines`` only) but
                    # produces phantom refs that don't match ``inst.refs``
                    # under v2.
                    entries.clear()
                    # §7.3 baseline integrity: ``ref_summary`` is the
                    # element_key→summary map persisted to last_snapshot.
                    # Without this clear, the discarded scoping pass leaks
                    # entries into the diff baseline and the next
                    # ``diff_from_last`` call reports phantom removals.
                    ref_summary.clear()
                    ref_counter[0] = 0
                    occurrence_counts.clear()
                    lines.append("** Modal dialog is open — only dialog elements are shown **")
                    # Re-query modal elements — handles go stale when SPAs
                    # like X/Twitter re-render the modal during the wait.
                    fresh_modals = []
                    for el in (await inst.page.query_selector_all(_MODAL_SELECTOR)):
                        if await self._is_visible_modal(el, vp):
                            fresh_modals.append(el)
                    if fresh_modals:
                        visible_modals = fresh_modals
                    for el in visible_modals:
                        try:
                            subtree = await self._build_a11y_tree(inst, root=el)
                            if subtree:
                                _walk(subtree, 0, target_frame_id)
                        except Exception:
                            pass
                    # Modals are typically light-DOM in the main frame;
                    # iframe descent skipped for scoping fidelity.
                    pending_frames.clear()
                    actionable_refs = [
                        r for r in refs.values() if r.role in _ACTIONABLE_ROLES
                    ]
                if not actionable_refs:
                    logger.warning(
                        "Modal detected but scoping produced 0 actionable "
                        "refs after retries — falling back to full tree "
                        "for %s", agent_id,
                    )
                    # Keep dialog_active=True so _locator_from_ref stays
                    # scoped to modal elements.  This prevents clicks from
                    # targeting elements behind the overlay (e.g. X's feed
                    # "Post" button behind the compose modal).  A modal-
                    # scoped click that can't find the element will timeout
                    # rather than hit the wrong target.
                    lines.clear()
                    # Same reset rationale as the retry branch above —
                    # discard fallback's parallel structures (entries
                    # for v2 rendering, ref_summary for §7.3 diff
                    # baseline) so the post-fallback _walk(tree) is
                    # the only contributor.
                    entries.clear()
                    ref_summary.clear()
                    lines.append(
                        "** A modal dialog is open but its elements could "
                        "not be isolated — elements with a (dialog: ...) "
                        "or similar landmark annotation are in the modal; "
                        "others are behind the overlay **"
                    )
                    _walk(tree, 0, target_frame_id)
                    if include_frames:
                        if target_frame is None:
                            await _descend_frames(
                                inst.page.main_frame, None, 0,
                            )
                        else:
                            # frame=X with nested same-origin frames inside
                            # X: descend so refs from inner frames carry
                            # their own frame_id. Nesting cap is relative
                            # to the target — start at depth 0.
                            await _descend_frames(
                                target_frame, target_frame_id, 0,
                            )
                    else:
                        pending_frames.clear()
            else:
                inst.dialog_active = False
                inst.dialog_detected = False
                _walk(tree, 0, target_frame_id)
                if include_frames:
                    if target_frame is None:
                        await _descend_frames(
                            inst.page.main_frame, None, 0,
                        )
                    else:
                        # frame=X with nested same-origin frames inside X:
                        # descend so refs from inner frames carry their
                        # own frame_id. Nesting cap is relative to the
                        # target — start at depth 0.
                        await _descend_frames(
                            target_frame, target_frame_id, 0,
                        )
                else:
                    pending_frames.clear()

            # Patch scope_root on refs captured during modal scoping so
            # `_locator_from_ref` queries are bounded to the dialog subtree.
            # (Refs emitted before the modal branch don't have scope_root;
            # set it now that we know the final dialog_active state.)
            #
            # Skip iframe-scoped refs: the modal lives in the main frame
            # (see _descend_frames; iframes aren't scoped by `dialog_active`),
            # so applying the modal selector to a Frame in `_locator_from_ref`
            # would query the iframe's document for a selector that lives in
            # the parent — silent click failure.
            if inst.dialog_active:
                for rid, handle in refs.items():
                    if handle.scope_root is None and handle.frame_id is None:
                        refs[rid] = RefHandle(
                            page_id=handle.page_id,
                            frame_id=handle.frame_id,
                            shadow_path=handle.shadow_path,
                            scope_root=_MODAL_SELECTOR,
                            role=handle.role,
                            name=handle.name,
                            occurrence=handle.occurrence,
                            disabled=handle.disabled,
                            element_key=handle.element_key,
                        )

            inst.refs = refs
            # §7.2 — choose between v1 (per-element landmark suffix) and
            # v2 (landmark headers + capped indent). Use the snapshot_fmt
            # captured at the top of this method so a mid-call flag flip
            # can't desynchronize this return path with the from_ref
            # early-return above.
            if snapshot_fmt == "v2":
                snapshot_text = _format_snapshot_v2(lines, entries)
            else:
                snapshot_text = (
                    "\n".join(lines) if lines else "(no interactive elements)"
                )
            snapshot_text = self.redactor.redact(agent_id, snapshot_text)
            # Record snapshot byte size for §4.6 metrics. Collected per call;
            # drained as p50/p95 on the next minute tick.
            inst.m_snapshot_bytes.append(len(snapshot_text))
            # Agent-visible `refs` uses the minimal dict shape (backward
            # compatible); RefHandle is strictly an internal detail.
            response_refs = {rid: h.to_agent_dict() for rid, h in refs.items()}

            # ── §7.3 diff-mode handling ─────────────────────────────────────
            current_url = inst.page.url
            current_dialog_active = bool(inst.dialog_active)
            previous_baseline = inst.last_snapshot.get(snapshot_page_id)
            scope = _classify_diff_scope(
                inst,
                snapshot_page_id=snapshot_page_id,
                previous=previous_baseline,
                current_url=current_url,
                current_dialog_active=current_dialog_active,
            )

            # Persist new baseline regardless of diff mode so the *next*
            # diff_from_last call has a fresh anchor. Per-page so multi-tab
            # state is preserved.
            #
            # Cross-PR (§7.3 ↔ §7.4/§7.7): when ``filter`` or ``from_ref``
            # is set, the resulting refs are a SUBSET of the page. Storing
            # that subset as the diff baseline would cause the next
            # unfiltered ``diff_from_last`` call to report all the
            # filtered-out elements as ``removed``. Skip the baseline
            # update for scoped/filtered calls — they're informational
            # snapshots, not anchors. ``last_active_page_id`` still
            # advances so tab-change detection stays accurate.
            is_subset_snapshot = (
                allowed_roles is not None
                or from_ref is not None
                or target_frame is not None
                or include_frames is False
            )
            if not is_subset_snapshot and not _skip_baseline:
                inst.last_snapshot[snapshot_page_id] = {
                    "refs_by_key": dict(ref_summary),
                    "url": current_url,
                    "dialog_active": current_dialog_active,
                }
            inst.last_active_page_id = snapshot_page_id

            if diff_from_last and scope in ("same", "modal_opened",
                                             "modal_closed", "frame_changed"):
                # Subset snapshot (filter / from_ref / target_frame /
                # include_frames=False) producing a diff against a full
                # baseline reports every filtered-out element as
                # ``removed`` — phantom signal that misleads the agent.
                # PR 781 already excludes subset snapshots from updating
                # the baseline; also short-circuit the diff computation
                # to a full-snapshot return so the agent isn't given
                # a confidently-wrong delta.
                if is_subset_snapshot:
                    return {
                        "success": True,
                        "data": {
                            "snapshot": snapshot_text,
                            "refs": response_refs,
                            "scope": "navigation",
                        },
                    }
                if previous_baseline is None:
                    # No prior baseline — fall through to the full-snapshot
                    # path so the agent still gets useful output. Effective
                    # scope becomes "navigation" since "added/removed against
                    # nothing" is meaningless.
                    return {
                        "success": True,
                        "data": {
                            "snapshot": snapshot_text,
                            "refs": response_refs,
                            "scope": "navigation",
                        },
                    }
                diff = _compute_snapshot_diff(
                    previous_baseline["refs_by_key"], ref_summary,
                )
                return {
                    "success": True,
                    "data": {**diff, "scope": scope},
                }

            # Either diff_from_last=False, or scope is navigation/tab_changed
            # — return the full snapshot. ``scope`` is included only when
            # diff_from_last=True so non-diff callers see the historical
            # response shape.
            data: dict = {"snapshot": snapshot_text, "refs": response_refs}
            if diff_from_last:
                data["scope"] = scope
            return {"success": True, "data": data}
        except Exception as e:
            # Match the §2.3 error envelope used by the new from_ref /
            # filter paths above so agents see a uniform shape regardless
            # of which branch raised. Preserves the message under
            # ``error.message`` so existing log scrapers still see the
            # underlying string.
            logger.exception("Snapshot failed for %s", agent_id)
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": str(e),
                    "retry_after_ms": None,
                },
            }

    @staticmethod
    async def _is_visible_modal(el, vp_size: dict | None) -> bool:
        """Check if a modal element is genuinely visible with real area.

        Filters zero-area or off-screen modals (e.g. LinkedIn's background
        messaging panels) that pass Playwright's ``is_visible()`` but are
        not true dialog overlays.
        """
        try:
            if not await el.is_visible():
                return False
            bb = await el.bounding_box()
            if not bb or bb["width"] < 10 or bb["height"] < 10:
                return False
            if vp_size:
                if (bb["x"] + bb["width"] <= 0 or bb["x"] >= vp_size["width"]
                        or bb["y"] + bb["height"] <= 0 or bb["y"] >= vp_size["height"]):
                    return False
            return True
        except Exception:
            return False

    async def _locator_from_ref(self, inst: CamoufoxInstance, ref: str):
        """Build a Playwright locator (or ElementHandle) from a stored RefHandle.

        Resolution order (§4.2):
            1. ``page_id`` → Page object (raises :class:`RefStale` if the
               tab has closed).
            2. ``frame_id`` → Frame (None = main frame). Set when the
               ref was captured inside an iframe; the resolver then runs
               every subsequent step against that Frame instead of the
               top-level Page, so shadow-piercing and ``get_by_role``
               operate inside the iframe's own document.
            3. ``shadow_path`` — walk open shadow roots via the
               §8.3 two-stage ``evaluate_handle`` pattern (against the
               Frame when ``frame_id`` is set, else the Page). Returns an
               :class:`ElementHandle` rather than a ``Locator`` because
               ``get_by_role`` does not pierce shadow boundaries.
            4. ``scope_root`` — modal selector bound during snapshot, so
               occurrence indices match.
            5. ``get_by_role(role, name=name, exact=True).nth(occurrence)``.

        Returns ``None`` when ``ref`` isn't in ``inst.refs`` (classic
        not-found).  Raises :class:`RefStale` when the ref points to a
        closed tab, a missing shadow host, or a discriminator mismatch —
        callers should report ``ref_stale`` so the agent re-snapshots.
        """
        handle = inst.refs.get(ref)
        if handle is None:
            return None
        page = inst._resolve_page_id(handle.page_id)
        if handle.frame_id is not None:
            frame = inst._resolve_frame_id(handle.frame_id)
            base = frame
        else:
            base = page
        if handle.shadow_path:
            return await self._resolve_shadow_element(base, handle, ref)
        if handle.scope_root:
            base = base.locator(handle.scope_root)
        if handle.name:
            locator = base.get_by_role(handle.role, name=handle.name, exact=True)
        else:
            locator = base.get_by_role(handle.role)
        return locator.nth(handle.occurrence)

    async def _resolve_shadow_element(self, base, handle: RefHandle, ref: str):
        """Two-stage resolver for refs whose ``shadow_path`` is non-empty.

        ``base`` is the JS-evaluation root for the resolver — a Page for
        main-frame refs, or a Frame for refs captured inside an iframe.
        Both share ``evaluate_handle``; the JS resolver runs in whichever
        document ``base`` represents, so an iframe-scoped ref walks the
        shadow path inside the iframe's document, not the top page.

        Stage 1 walks the path to the inner ``shadowRoot``, verifying
        each host's discriminator. Stage 2 picks the role+name match at
        the requested occurrence inside that root. Either stage can
        raise :class:`RefStale` when the DOM has shifted since snapshot.

        Error taxonomy:

        * Stage-1 ``stale_host_missing``      → RefStale (host removed).
        * Stage-1 ``stale_discriminator_mismatch`` → RefStale (host swapped).
        * Stage-2 ``shadow_root_detached``    → RefStale (TOCTOU between
          stage 1 succeeding and stage 2 running).
        * Stage-2 returns null                → RefStale (element no
          longer present at the requested occurrence inside an otherwise
          live shadow root).

        Errors are signaled via a unique ``__OL_RESOLVER_ERROR__`` key
        rather than a plain ``error`` property so a page that happens to
        attach an ``error`` expando to a ShadowRoot cannot masquerade as
        a resolver failure.
        """
        path_payload = [
            {
                "selector": hop.selector,
                "occurrence": hop.occurrence,
                "discriminator": hop.discriminator,
            }
            for hop in handle.shadow_path
        ]
        # Pass scope_root through so Stage 1 starts its walk inside the
        # modal subtree when a ref was captured under modal scoping.
        # Without this, a same-selector shadow host living outside the
        # dialog could resolve and the click would land on the wrong
        # element entirely.
        stage1 = await base.evaluate_handle(
            _JS_SHADOW_RESOLVE_STAGE1,
            {
                "path": json.dumps(path_payload),
                "scope_root": handle.scope_root,
            },
        )
        # Stage 1 is a JSHandle that we OWN — Playwright does not GC
        # JSHandles automatically (they pin the JS object on the page
        # side). Wrap the entire stage-2 invocation in a try/finally so
        # discriminator-mismatch / detach / RefStale paths don't leak
        # the handle. Stage 2 is returned to the caller as an
        # ElementHandle on success; ownership transfers there. On any
        # failure path we dispose stage2 ourselves before raising so
        # neither handle outlives this method.
        stage2 = None
        try:
            # Read the stage-1 error sentinel by exact key. Page-
            # controlled ``error`` expandos cannot match because we look
            # up ``__OL_RESOLVER_ERROR__`` specifically.
            try:
                err = await stage1.evaluate(
                    "(v) => (v && typeof v === 'object' && '__OL_RESOLVER_ERROR__' in v)"
                    " ? v.__OL_RESOLVER_ERROR__ : null",
                )
            except Exception:
                err = None
            if err == "stale_host_missing":
                raise RefStale("shadow host missing", ref=ref)
            if err == "stale_discriminator_mismatch":
                raise RefStale("shadow host discriminator changed", ref=ref)
            if err == "scope_root_missing":
                # Modal closed between snapshot and resolve — same RefStale
                # contract as a missing host. Agent re-snapshots.
                raise RefStale("modal scope_root missing", ref=ref)
            stage2 = await stage1.evaluate_handle(
                _JS_SHADOW_RESOLVE_STAGE2,
                {
                    "role": handle.role,
                    "name": handle.name,
                    "occurrence": handle.occurrence,
                },
            )
            # Distinguish a detached shadow root (transient TOCTOU between
            # stage 1 and stage 2) from a genuine element-not-found. The
            # detached case surfaces as the ``shadow_root_detached``
            # sentinel so the caller emits a transient ``ref_stale``
            # rather than a potentially-misleading ``not_found``.
            try:
                stage2_err = await stage2.evaluate(
                    "(v) => (v && typeof v === 'object' && '__OL_RESOLVER_ERROR__' in v)"
                    " ? v.__OL_RESOLVER_ERROR__ : null",
                )
            except Exception:
                stage2_err = None
            if stage2_err == "shadow_root_detached":
                raise RefStale("shadow root detached during resolve", ref=ref)
            element = stage2.as_element()
            if element is None:
                raise RefStale(
                    "shadow element not found at occurrence", ref=ref,
                )
            # Success — ownership of the underlying handle transfers to
            # the caller via ``element``. Detach our local ``stage2``
            # reference so the finally block below does not dispose the
            # handle out from under the returned ElementHandle.
            # (``as_element`` returns the same underlying handle, so
            # disposing ``stage2`` would invalidate ``element`` too.)
            stage2 = None
            return element
        except Exception:
            # Any error path — dispose stage2 if we got that far. The
            # outer finally takes care of stage1.
            if stage2 is not None:
                try:
                    await stage2.dispose()
                except Exception:
                    pass
                stage2 = None
            raise
        finally:
            try:
                await stage1.dispose()
            except Exception:
                pass

    def _resolve_frame_arg(self, inst: CamoufoxInstance, frame_arg: str):
        if not isinstance(frame_arg, str) or not frame_arg.strip():
            return None
        token = frame_arg.strip()
        direct = inst.frame_ids_inv.get(token)
        if direct is not None:
            return direct
        # Frame-id-shaped tokens that miss the inverse map signal a
        # detached frame, not a URL substring miss. Surfacing as
        # RefStale lets callers re-snapshot rather than chase a
        # non-existent URL substring.
        if _FRAME_ID_RE.match(token):
            raise RefStale("frame_detached", ref=None)
        page = inst.page
        # Two-pass lookup. Pass 1 prefers DIRECT children of the main
        # frame to localize substring matches; a deeply-nested malicious
        # iframe sharing a URL substring with a legitimate top-level
        # frame should not be picked over the legitimate one. Pass 2
        # falls back to the full frame tree only when no top-level
        # match was found.
        #
        # When two frames share the same URL exactly, the FIRST matching
        # one wins (iteration order). Callers needing to disambiguate
        # duplicate-URL siblings should use the frame_id token from
        # ``RefHandle.to_agent_dict()`` instead of a URL substring.
        def _scan(frames):
            ex = None
            sub = None
            for f in frames:
                if f is page.main_frame:
                    continue
                try:
                    url = f.url or ""
                except Exception:
                    continue
                if url == token:
                    ex = f
                    break
                if sub is None and token in url:
                    sub = f
            return ex, sub

        try:
            top_children = list(page.main_frame.child_frames)
        except Exception:
            top_children = []
        exact, substring = _scan(top_children)
        if exact is None and substring is None:
            exact, substring = _scan(page.frames)
        return exact if exact is not None else substring

    async def _human_click(self, page, locator, *, force: bool = False,
                           timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Click with a preceding hover so the mouse visibly moves to the target.

        Playwright's ``locator.click()`` dispatches a click at the element's
        center coordinates but does NOT generate the ``mousemove`` events a
        real user produces while moving the cursor to the target.  Anti-bot
        systems (X/Twitter, Cloudflare) track mouse-movement patterns and
        flag clicks that appear without any prior movement.

        The hover-then-click pattern:
        1. ``locator.hover()`` — Playwright scrolls the element into view and
           moves the mouse along a path to the element center.  With Camoufox's
           ``humanize=True``, this path includes natural-looking Bézier curves.
        2. Brief settle (20–60 ms) — models the human reaction gap between
           arriving at the target and pressing the button.
        3. ``page.mouse.click(x, y)`` — fires the mousedown/mouseup at the
           current mouse position (already on the element from the hover).

        When ``force=True``, falls back to ``locator.click(force=True)`` since
        hover may fail on elements obscured by overlays. We still call
        ``scroll_into_view_if_needed`` first — Playwright's ``force=True``
        skips ALL actionability checks INCLUDING the implicit scroll-
        into-view, so a forced click on an off-fold element fires the
        click event at the element's geometric position which can land
        OUTSIDE the visible viewport. From the operator's VNC view, the
        click visually "lands nowhere." This bites in particular on
        SPA frameworks (X / Twitter / Gmail) that use ``aria-disabled``
        on visually-active buttons — the click loop auto-applies
        ``force=True`` for those (see ``_ARIA_FORCE_ROLES`` gate in
        :meth:`click`), so the force path is hit MORE often than
        operators expect. Scrolling explicitly is cheap and additive.
        """
        if force:
            try:
                await locator.scroll_into_view_if_needed(timeout=timeout)
            except Exception:
                # Element may be detached / cross-origin / have no box;
                # the forced click below will still attempt at the last-
                # known position. Better to log via the click error path
                # than to fail-fast here.
                pass
            await locator.click(timeout=timeout, force=True)
            return
        try:
            await locator.hover(timeout=timeout)
            await asyncio.sleep(random.uniform(0.02, 0.06))
        except Exception:
            pass  # Hover failed — click below will still attempt
        await locator.click(timeout=timeout, force=False)

    async def _human_click_selector(self, page, selector: str, *,
                                    force: bool = False,
                                    timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Like _human_click but takes a CSS selector instead of a locator.

        Hovers first to generate natural mouse movement, then clicks.
        On the ``force=True`` path, scrolls the locator into view BEFORE
        the forced click — Playwright's ``force=True`` skips
        scroll-into-view; without this an off-fold force-click fires
        outside the viewport and visibly lands nowhere on VNC. Mirror
        of :meth:`_human_click`'s force path.
        """
        if force:
            try:
                await page.locator(selector).scroll_into_view_if_needed(timeout=timeout)
            except Exception:
                pass
            await page.click(selector, timeout=timeout, force=True)
            return
        try:
            await page.hover(selector, timeout=timeout)
            await asyncio.sleep(random.uniform(0.02, 0.06))
        except Exception:
            pass  # Hover failed — click will still work
        await page.click(selector, timeout=timeout, force=False)

    async def _x11_move_to(
        self, inst: CamoufoxInstance, target_x: int, target_y: int,
    ) -> None:
        """Move mouse to (target_x, target_y) via xdotool with a Bezier trajectory.

        Generates a natural-looking curved mouse path using cubic Bezier
        interpolation with randomized control points. Real human wrist
        movement produces slight S-curves, not straight lines.

        Velocity easing (cubic ease-in-out) models Fitts' Law: slow
        departure, fast cruise, slow precision-landing. Step count
        scales with distance so short movements stay snappy and long
        movements stay smooth.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool move")

        wid_s = str(wid)
        loop = asyncio.get_running_loop()

        # Get current mouse position
        start_x, start_y = 0, 0
        loc_result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True, text=True, timeout=3,
            ),
        )
        if loc_result.returncode == 0:
            for part in loc_result.stdout.split():
                if part.startswith("x:"):
                    start_x = int(part[2:])
                elif part.startswith("y:"):
                    start_y = int(part[2:])

        # Bezier control points — offset perpendicular to the line
        dx = target_x - start_x
        dy = target_y - start_y
        dist = max(1, (dx * dx + dy * dy) ** 0.5)
        # Perpendicular unit vector
        perp_x, perp_y = -dy / dist, dx / dist
        # Control points with randomized perpendicular offset (scaled by distance)
        spread = min(dist * 0.3, 60)
        off1 = random.uniform(-spread, spread)
        off2 = random.uniform(-spread, spread)
        cp1_x = start_x + dx * 0.25 + perp_x * off1
        cp1_y = start_y + dy * 0.25 + perp_y * off1
        cp2_x = start_x + dx * 0.75 + perp_x * off2
        cp2_y = start_y + dy * 0.75 + perp_y * off2

        # Scale step count with distance — short moves stay snappy,
        # long moves stay smooth.  Range: 3 steps (tiny) to 14 (across screen).
        steps = max(3, min(14, int(dist / 80) + random.randint(2, 4)))

        for i in range(1, steps + 1):
            # Raw parameter
            raw_t = i / steps
            # Cubic ease-in-out: slow start, fast middle, slow landing
            # Models Fitts' Law deceleration as cursor approaches target
            if raw_t < 0.5:
                t = 4 * raw_t * raw_t * raw_t
            else:
                t = 1 - ((-2 * raw_t + 2) ** 3) / 2

            u = 1 - t
            wp_x = int(
                u**3 * start_x + 3 * u**2 * t * cp1_x
                + 3 * u * t**2 * cp2_x + t**3 * target_x
            )
            wp_y = int(
                u**3 * start_y + 3 * u**2 * t * cp1_y
                + 3 * u * t**2 * cp2_y + t**3 * target_y
            )
            wp_x = max(0, wp_x)
            wp_y = max(0, wp_y)
            mv_result = await loop.run_in_executor(
                None,
                lambda x=wp_x, y=wp_y: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(x), str(y)],
                    capture_output=True, timeout=3,
                ),
            )
            if mv_result.returncode != 0:
                raise RuntimeError(
                    f"xdotool mousemove failed (rc={mv_result.returncode})"
                )
            await asyncio.sleep(x11_step_delay())

        # Overshoot + correction for long movements — models the human
        # tendency to slightly overshoot the target and make a tiny
        # corrective flick back. Only on ~30% of long movements.
        if dist > 300 and random.random() < 0.3:
            # Direction from last control point toward target
            end_dx = target_x - cp2_x
            end_dy = target_y - cp2_y
            end_dist = max(1, (end_dx**2 + end_dy**2) ** 0.5)
            overshoot_px = random.uniform(3, 8)
            ov_x = max(0, int(target_x + end_dx / end_dist * overshoot_px))
            ov_y = max(0, int(target_y + end_dy / end_dist * overshoot_px))
            # Overshoot
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(ov_x), str(ov_y)],
                    capture_output=True, timeout=3,
                ),
            )
            await asyncio.sleep(x11_step_delay())
            # Correct back to exact target
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(target_x), str(target_y)],
                    capture_output=True, timeout=3,
                ),
            )
            await asyncio.sleep(x11_step_delay())

    async def _x11_ensure_in_viewport(
        self, inst: CamoufoxInstance, locator, *,
        timeout: int = _CLICK_TIMEOUT_MS,
    ) -> None:
        """Scroll element into viewport using X11 wheel events.

        Replaces ``locator.scroll_into_view_if_needed()`` for the X11
        input path.  Protocol-level ``scrollIntoView`` produces scroll
        events WITHOUT ``WheelEvent`` — a detectable automation signal.
        X11 button 4/5 produces real ``WheelEvent`` with
        ``deltaMode=DOM_DELTA_LINE``, matching physical hardware.

        Scrolls in small increments (2–3 notches per batch), re-measures
        the element position after each batch to prevent overshoot.
        Falls back to protocol scroll for edge cases (elements inside
        scrollable inner containers, elements not yet in the DOM).
        """
        if not inst.x11_wid:
            await locator.scroll_into_view_if_needed(timeout=timeout)
            return

        vp = inst.page.viewport_size
        if not vp:
            await locator.scroll_into_view_if_needed(timeout=timeout)
            return

        vp_h = vp["height"]
        margin = 60

        for _ in range(10):
            box = await locator.bounding_box()
            if box is None:
                break  # Not in DOM — protocol scroll only option

            center_y = box["y"] + box["height"] / 2
            if margin <= center_y <= vp_h - margin:
                return  # Element is visible

            button = "4" if center_y < margin else "5"
            prev_center = center_y

            batch = random.randint(2, 3)
            for _ in range(batch):
                try:
                    await self._x11_scroll_notch(inst, button)
                except Exception:
                    break
                await asyncio.sleep(scroll_pause() * 0.4)

            # Wait for smooth scrolling to settle
            await asyncio.sleep(0.10)

            # Check if element position actually changed
            new_box = await locator.bounding_box()
            if new_box is None:
                break
            new_center = new_box["y"] + new_box["height"] / 2
            if abs(new_center - prev_center) < 2:
                break  # Scroll didn't move element — inner container

        # Fallback for edge cases
        await locator.scroll_into_view_if_needed(timeout=timeout)

    async def _x11_click(self, inst: CamoufoxInstance, locator, *,
                         timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Click via xdotool for isTrusted=true events.

        Bot-detection systems (ArkoseLabs on X/Twitter) hook addEventListener
        and reject clicks where event.isTrusted is false.  CDP-dispatched
        clicks always have isTrusted=false.  xdotool injects real X11
        ButtonPress/ButtonRelease events through the kernel input stack,
        which the browser marks isTrusted=true.

        Steps:
        1. _x11_ensure_in_viewport — scrolls element into viewport using
           X11 wheel events (real WheelEvent with DOM_DELTA_LINE) instead
           of protocol-level scrollIntoView (no WheelEvents, detectable)
        2. Get element bounding box (viewport coords)
        3. Bezier mouse trajectory via _x11_move_to
        4. mousedown + human dwell + mouseup (not instant click)
        """
        # 1. Scroll into view — prefer X11 wheel events over protocol scroll
        await self._x11_ensure_in_viewport(inst, locator, timeout=timeout)
        await asyncio.sleep(x11_settle_delay())

        # 2. Get element position — jitter within inner area, not dead center
        box = await locator.bounding_box()
        if not box:
            raise RuntimeError("Element has no bounding box — not visible")
        # Real humans don't click dead center — offset within inner 60%
        jitter_x = random.uniform(-0.15, 0.15) * box["width"]
        jitter_y = random.uniform(-0.10, 0.10) * box["height"]
        target_x = int(box["x"] + box["width"] / 2 + jitter_x)
        target_y = int(box["y"] + box["height"] / 2 + jitter_y)

        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool click")

        # 3. Move mouse with natural Bezier trajectory
        await self._x11_move_to(inst, target_x, target_y)

        # 4. Click with human-like dwell time (mousedown -> hold -> mouseup)
        wid_s = str(wid)
        loop = asyncio.get_running_loop()
        await asyncio.sleep(pre_click_settle())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mousedown", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )
        await asyncio.sleep(click_dwell())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mouseup", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )

    async def _x11_click_xy(
        self, inst: CamoufoxInstance, x: float, y: float,
    ) -> None:
        """Click at viewport coords (x, y) via xdotool — trusted-event path.

        ``click_xy`` previously dispatched ``page.mouse.click(x, y)``
        directly, which generates a CDP-injected click with
        ``isTrusted=false``. Bot-detection listeners check the trust
        flag and an agent that primarily uses ref-clicks (X11) but
        falls back to xy-clicks (CDP) emits a single
        ``isTrusted=false`` event in an otherwise all-trusted session
        — a sharp anti-bot cluster signal. Route through the same
        Bezier + dwell shape ``_x11_click`` uses so the trusted-event
        property holds across click-by-coords too.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool click")
        await self._x11_move_to(inst, int(x), int(y))
        wid_s = str(wid)
        loop = asyncio.get_running_loop()
        await asyncio.sleep(pre_click_settle())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mousedown", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )
        await asyncio.sleep(click_dwell())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mouseup", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )

    async def _x11_hover(self, inst: CamoufoxInstance, locator) -> None:
        """Move mouse to element via xdotool for isTrusted=true mousemove events."""
        await self._x11_ensure_in_viewport(inst, locator)
        await asyncio.sleep(x11_settle_delay())

        box = await locator.bounding_box()
        if not box:
            raise RuntimeError("Element has no bounding box — not visible")
        # Jitter within inner area — same as _x11_click for consistency
        jitter_x = random.uniform(-0.15, 0.15) * box["width"]
        jitter_y = random.uniform(-0.10, 0.10) * box["height"]
        target_x = int(box["x"] + box["width"] / 2 + jitter_x)
        target_y = int(box["y"] + box["height"] / 2 + jitter_y)

        await self._x11_move_to(inst, target_x, target_y)

    async def _idle_mouse_jitter(self, inst: CamoufoxInstance) -> None:
        """Periodic mouse micro-movement to simulate human fidgeting.

        Real users constantly micro-move the mouse while reading — small
        drifts, twitches, and repositioning. A mouse that is perfectly
        still for seconds between actions is a textbook bot pattern
        detected by ArkoseLabs and DataDome.
        """
        while True:
            await asyncio.sleep(random.uniform(2.0, 7.0))
            if not inst.x11_wid or inst.lock.locked() or inst._user_control:
                continue
            try:
                dx = random.randint(-3, 3)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda _dx=dx, _dy=dy: subprocess.run(
                        ["xdotool", "mousemove_relative", "--sync",
                         "--", str(_dx), str(_dy)],
                        capture_output=True, timeout=2,
                    ),
                )
            except asyncio.CancelledError:
                return
            except Exception:
                pass

    async def _x11_type(self, inst: CamoufoxInstance, text: str,
                        *, typos: bool = True) -> None:
        """Type text via xdotool for isTrusted=true key events.

        Same rationale as _x11_click — bot-detection checks isTrusted on
        keydown/keyup in tweet composer textareas.  xdotool key/type
        generates real X11 KeyPress/KeyRelease events.

        When *typos* is True (default), injects occasional typo +
        backspace corrections to simulate natural human error patterns.
        Zero-typo typing at consistent speed is one of the strongest
        bot signals.  Typos are placed mid-word only (avoiding handles,
        hashtags, URLs) and capped at a per-text budget.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool type")

        loop = asyncio.get_running_loop()
        wid_s = str(wid)

        # ── Typo budget ──────────────────────────────────────────
        # Pre-select positions for typo injection.  Only mid-word
        # alphabetic characters qualify (skips handles, hashtags,
        # URLs).  Budget scales with text length.
        typo_positions: set[int] = set()
        if typos:
            candidates = [
                i for i, c in enumerate(text)
                if c.isalpha() and c.lower() in _TYPO_NEIGHBORS
                and i > 0 and text[i - 1].isalpha()
            ]
            alpha_count = len(candidates)
            if alpha_count >= 15:
                expected = max(1.0, alpha_count / 120)
                budget = max(0, int(random.gauss(expected, expected * 0.5)))
                budget = min(budget, 4)
                if budget > 0 and len(candidates) >= budget:
                    typo_positions = set(random.sample(candidates, budget))

        # ── Character loop ───────────────────────────────────────
        prev_char = ""
        for i, char in enumerate(text):
            # Word-boundary think pauses
            pause_prob = 0.08 if prev_char in _WORD_BOUNDARY_CHARS else 0.015
            if random.random() < pause_prob:
                await asyncio.sleep(think_pause())

            # Typo injection — wrong adjacent key → pause → backspace → correct
            if i in typo_positions:
                wrong = random.choice(_TYPO_NEIGHBORS[char.lower()])
                if char.isupper():
                    wrong = wrong.upper()
                # Type wrong character
                await loop.run_in_executor(
                    None,
                    lambda c=wrong: subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--window", wid_s,
                         "--delay", "0", "--", c],
                        capture_output=True, timeout=3,
                    ),
                )
                await asyncio.sleep(keystroke_delay(wrong))
                # Pause — noticing the error
                await asyncio.sleep(random.uniform(0.15, 0.4))
                # Backspace to correct
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s,
                         "BackSpace"],
                        capture_output=True, timeout=3,
                    ),
                )
                await asyncio.sleep(random.uniform(0.03, 0.08))

            # Type the correct character
            if char == "\n":
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s, "Return"],
                        capture_output=True, timeout=3,
                    ),
                )
            elif char == "\t":
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s, "Tab"],
                        capture_output=True, timeout=3,
                    ),
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda c=char: subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--window", wid_s,
                         "--delay", "0", "--", c],
                        capture_output=True, timeout=3,
                    ),
                )
            await asyncio.sleep(keystroke_delay(char))
            prev_char = char

    async def _x11_key(self, inst: CamoufoxInstance, key: str) -> None:
        """Send a key combination via xdotool for isTrusted=true key events."""
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool key")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "--window", str(wid), key],
                capture_output=True, timeout=3,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(f"xdotool key {key!r} failed (rc={result.returncode})")

    async def _x11_scroll_notch(self, inst: CamoufoxInstance, button: str) -> None:
        """Send a single scroll notch via xdotool button 4 (up) or 5 (down).

        X11 button 4/5 events are processed by Firefox identically to
        physical mouse wheel input, producing ``WheelEvent`` with
        ``deltaMode=DOM_DELTA_LINE`` — matching real hardware.  Playwright's
        ``page.mouse.wheel()`` instead uses ``nsIDOMWindowUtils.sendWheelEvent``
        with ``deltaMode=DOM_DELTA_PIXEL``, which is a detectable fingerprint.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool scroll")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "click", "--clearmodifiers", "--window", str(wid), button],
                capture_output=True, timeout=3,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(f"xdotool scroll button {button} failed (rc={result.returncode})")

    @staticmethod
    def _playwright_key_to_xdotool(key: str) -> str:
        """Convert a Playwright key name to xdotool key name."""
        parts = key.split("+")
        mapped = [_PLAYWRIGHT_TO_XDOTOOL.get(p, p) for p in parts]
        return "+".join(mapped)

    def _is_x11_site(self, inst: CamoufoxInstance) -> bool:
        """Whether to use X11 input injection for this page.

        X11/xdotool injects real kernel-level InputEvents that the browser
        marks ``isTrusted=true``.  CDP-dispatched events always carry
        ``isTrusted=false``, which bot-detection systems (DataDome,
        Cloudflare, PerimeterX, ArkoseLabs) broadly check — not just on
        Twitter.  Using X11 input everywhere eliminates this signal.

        Falls back to CDP automatically on failure (see call sites).
        """
        return True

    async def click(
        self, agent_id: str, ref: str | None = None,
        selector: str | None = None, force: bool = False,
        snapshot_after: bool = False,
        timeout_ms: int | None = None,
        frame: str | None = None,
    ) -> dict:
        """Click element by ref or CSS selector.

        force=True bypasses Playwright's actionability checks (visibility,
        stability, not-covered, enabled). Use when the element is visually
        present in VNC but Playwright reports it as covered by an overlay.

        For button/link roles that were disabled in the last snapshot,
        force is applied automatically — SPA frameworks (X/Twitter, Gmail)
        commonly set aria-disabled="true" on buttons that are still clickable
        via JS handlers. Playwright blocks clicks on aria-disabled elements
        unless force=True, so we bypass the check for these roles.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            async def _click_body() -> dict:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                raw_timeout = _CLICK_TIMEOUT_MS if timeout_ms is None else timeout_ms
                _timeout = max(1000, min(raw_timeout, 30000))
                use_force = force
                resolved_frame = None
                if frame is not None:
                    try:
                        resolved_frame = self._resolve_frame_arg(inst, frame)
                    except RefStale as rs:
                        return {
                            "success": False,
                            "error": {
                                "code": "ref_stale",
                                "message": str(rs),
                            },
                        }
                    if resolved_frame is None:
                        return {
                            "success": False,
                            "error": (
                                f"frame {frame!r} did not match any frame "
                                "url-substring or frame_id"
                            ),
                        }
                if ref and ref in inst.refs:
                    ref_info = inst.refs[ref]
                    if frame is not None:
                        resolved_frame_id = (
                            inst._register_frame(resolved_frame)
                            if resolved_frame is not None else None
                        )
                        # Caller asserted a specific frame; ref must agree.
                        # Fires on main-frame refs (frame_id=None) too — a
                        # frame= arg with a main-frame ref is a bug, not a
                        # silently-ignored hint.
                        if ref_info.frame_id != resolved_frame_id:
                            return {
                                "success": False,
                                "error": {
                                    "code": "invalid_input",
                                    "message": (
                                        "frame argument conflicts with "
                                        "ref's frame"
                                    ),
                                },
                            }
                    # Auto-force for disabled button/link roles — aria-disabled
                    # on SPA buttons doesn't mean the JS click handler won't fire.
                    # BUT: when a modal was detected and scoping failed
                    # (dialog_detected=True, dialog_active=False), disabled
                    # buttons are likely behind the overlay — don't force them.
                    modal_unscoped = (
                        inst.dialog_detected and not inst.dialog_active
                    )
                    if (not use_force
                            and ref_info.disabled
                            and ref_info.role in _ARIA_FORCE_ROLES
                            and not modal_unscoped):
                        use_force = True
                        logger.debug(
                            "Auto-force click on disabled %s ref=%s for '%s'",
                            ref_info.role, ref, agent_id,
                        )
                    try:
                        locator = await self._locator_from_ref(inst, ref)
                    except RefStale as rs:
                        return {
                            "success": False,
                            "error": {
                                "code": "ref_stale",
                                "message": str(rs),
                            },
                        }
                    if locator:
                        if inst.x11_wid and self._is_x11_site(inst):
                            try:
                                await self._x11_click(inst, locator, timeout=_timeout)
                            except Exception as e:
                                logger.warning(
                                    "X11 click failed for '%s', falling back to CDP: %s",
                                    agent_id, e,
                                )
                                await self._human_click(inst.page, locator, force=use_force, timeout=_timeout)
                        else:
                            await self._human_click(inst.page, locator, force=use_force, timeout=_timeout)
                    else:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                elif selector:
                    if resolved_frame is not None:
                        # X11 click path requires page-level coordinates;
                        # iframe-scoped selectors resolve through
                        # Playwright's frame locator only. Anti-bot benefit
                        # of the X11 path is lost for iframe clicks —
                        # accepted tradeoff.
                        loc = resolved_frame.locator(selector).first
                        try:
                            await self._human_click(
                                inst.page, loc, force=force, timeout=_timeout,
                            )
                        except Exception as e:
                            return {"success": False, "error": str(e)}
                    elif inst.x11_wid and self._is_x11_site(inst):
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_click(inst, loc, timeout=_timeout)
                        except Exception as e:
                            logger.warning(
                                "X11 click failed for '%s' (selector), falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click_selector(inst.page, selector, force=force, timeout=_timeout)
                    else:
                        await self._human_click_selector(inst.page, selector, force=force, timeout=_timeout)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())

                # Fallback: if a close-type button was clicked inside a
                # modal but the modal persists, press Escape.  Camoufox's
                # patched Firefox silently drops pointer events on some
                # SPA modal close buttons (X/Twitter compose modal, etc.).
                if inst.dialog_active and ref and ref in inst.refs:
                    ri = inst.refs[ref]
                    nm = (ri.name or "").lower().strip()
                    is_close = ri.role == "button" and (
                        nm in _MODAL_CLOSE_NAMES
                        or nm.startswith("close")
                    )
                    if is_close:
                        await asyncio.sleep(0.3)
                        still_open = False
                        vp = inst.page.viewport_size
                        try:
                            modal_els = await inst.page.query_selector_all(
                                _MODAL_SELECTOR,
                            )
                            for el in modal_els:
                                if await self._is_visible_modal(el, vp):
                                    still_open = True
                                    break
                        except Exception:
                            pass
                        if still_open:
                            logger.info(
                                "Close-button click did not dismiss "
                                "modal for %s — sending Escape",
                                agent_id,
                            )
                            if inst.x11_wid and self._is_x11_site(inst):
                                try:
                                    await self._x11_key(inst, "Escape")
                                except Exception:
                                    await inst.page.keyboard.press("Escape")
                            else:
                                await inst.page.keyboard.press("Escape")
                            await asyncio.sleep(0.5)
                            # Escape may surface a confirmation dialog
                            # (e.g. "Discard draft?" on X/Twitter).
                            # Click through it to finish dismissing.
                            try:
                                confirm = inst.page.locator(
                                    _MODAL_SELECTOR,
                                ).get_by_role(
                                    "button", name="Discard",
                                )
                                if await confirm.count() > 0:
                                    if inst.x11_wid and self._is_x11_site(inst):
                                        try:
                                            await self._x11_click(
                                                inst, confirm.first,
                                            )
                                        except Exception:
                                            await self._human_click(
                                                inst.page, confirm.first,
                                                force=True,
                                            )
                                    else:
                                        await self._human_click(
                                            inst.page, confirm.first,
                                            force=True,
                                        )
                                    await asyncio.sleep(action_delay())
                                    logger.info(
                                        "Clicked Discard on confirmation"
                                        " dialog for %s", agent_id,
                                    )
                            except Exception:
                                pass

                inst.m_click_success += 1
                inst.click_window.append(True)
                # Recorder doesn't need the x11/cdp routing detail —
                # the click dispatch chooses internally and the timing
                # distribution is what §5.3/§9.5 consumes.
                inst.recorder.record_click(method="auto", success=True)
                result = {"success": True, "data": {"clicked": ref or selector}}
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result

            try:
                action_result, captcha_envelope = await self._with_captcha_redetect(
                    inst, _click_body(),
                )
            except Exception as e:
                inst.m_click_fail += 1
                inst.click_window.append(False)
                inst.recorder.record_click(method="auto", success=False)
                return {"success": False, "error": str(e)}

            # §11.4 / §18.2 — re-detection envelope is ADDITIVE: existing
            # callers continue to see the original action_result shape;
            # new agents that read ``"captcha"`` get the §11.13 envelope
            # when a captcha appeared as a side effect of the click.
            if (
                captcha_envelope is not None
                and isinstance(action_result, dict)
                and action_result.get("success")
            ):
                action_result.setdefault("captcha", captcha_envelope)
            return action_result

    async def click_xy(
        self, agent_id: str, x: float, y: float,
    ) -> dict:
        """Phase 6 §9.3: click at viewport-relative pixel coordinates.

        ``(x, y)`` are viewport-relative pixels — origin is the top-left
        of the rendered page area (NOT the screen, NOT including window
        chrome). The method first calls ``document.elementFromPoint(x, y)``
        and walks up the ancestor chain to detect overlay/visibility/
        pointer-events masking; on a clean hit the click is dispatched via
        ``page.mouse.click(x, y)``.

        Returns a §2.3 success or error envelope. The mask walk respects
        the CSS cascade for ``pointer-events``: an inner ancestor with
        ``pointer-events: auto`` overrides an outer ``pointer-events: none``,
        so we cannot naively flag any ancestor with ``none`` as masked —
        we walk inside-out and stop the pointer-events check at the first
        ``auto`` boundary.
        """
        # Reject bool first — Python's ``True == 1`` would otherwise pass
        # the (int, float) check; a Boolean coordinate is a caller bug.
        if isinstance(x, bool) or isinstance(y, bool):
            return _err(
                "invalid_input",
                "x and y must be numbers, not booleans",
            )
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return _err(
                "invalid_input",
                "x and y must be numbers",
            )
        # Coerce ints to floats so downstream math is uniform.
        x = float(x)
        y = float(y)
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return _err(
                "invalid_input",
                "x and y must be finite numbers",
            )

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return _err(
                        "conflict",
                        "User has browser control — action paused until "
                        "control is released.",
                    )
                # ``viewport_size`` is normally populated for Camoufox
                # (fingerprint-driven, see ``stealth.pick_resolution``).
                # Defensive: in some Playwright configurations (no explicit
                # context viewport, e.g. tests with a default-page context)
                # ``viewport_size`` returns ``None``. Negative coords are
                # always invalid; for the upper bound, when viewport size
                # is unknown skip the bounds check and let Playwright's
                # ``mouse.click`` reject out-of-window coordinates with
                # its own error (caught below as ``service_unavailable``).
                if x < 0 or y < 0:
                    return _err(
                        "invalid_input",
                        f"coordinate ({x}, {y}) out of viewport bounds "
                        f"(coordinates must be non-negative)",
                    )
                vp = inst.page.viewport_size or {}
                vw = vp.get("width") if isinstance(vp, dict) else None
                vh = vp.get("height") if isinstance(vp, dict) else None
                if (isinstance(vw, (int, float))
                        and isinstance(vh, (int, float))
                        and (x >= vw or y >= vh)):
                    return _err(
                        "invalid_input",
                        f"coordinate ({x}, {y}) out of viewport bounds "
                        f"({int(vw)} x {int(vh)})",
                    )

                hit = await inst.page.evaluate(_JS_ELEMENT_FROM_POINT, [x, y])
                if hit is None:
                    return _err(
                        "no_element_at_point",
                        f"no element at ({x}, {y})",
                    )
                masked_by = hit.get("masked_by") if isinstance(hit, dict) else None
                if masked_by:
                    # Redact name + role: ``elementFromPoint`` returns
                    # raw DOM text (innerText slice). Snapshot output is
                    # redacted; click_xy must honor the same boundary so
                    # an agent doesn't get a privileged channel to read
                    # credentials that leaked into element text. Role is
                    # an attribute the page can set freely; bound length
                    # so a malicious page can't bloat responses.
                    actual = {
                        "tag": hit.get("tag", "")[:64],
                        "role": (hit.get("role") or "")[:64] or None,
                        "name": self.redactor.redact(
                            agent_id, hit.get("name", "") or "",
                        )[:200],
                    }
                    return {
                        "success": False,
                        "error": {
                            "code": "invalid_input",
                            "message": (
                                f"click at ({x}, {y}) would be intercepted "
                                f"by an ancestor element"
                            ),
                            "retry_after_ms": None,
                            "data": {
                                "actual_element": actual,
                                # Back-compat for the first branch revision;
                                # callers should prefer ``actual_element``.
                                "actual": actual,
                                "masked_by": masked_by,
                                "mask_reason": hit.get("mask_reason", ""),
                            },
                        },
                    }

                # Clean hit — dispatch the click. Prefer X11/xdotool
                # (real kernel InputEvents → ``isTrusted=true``) when
                # the agent has an X11 window registered; fall back to
                # CDP only when X11 isn't usable. Pre-fix this used
                # ``page.mouse.click`` unconditionally — a single such
                # CDP click in an otherwise-X11 session emits an
                # ``isTrusted=false`` event that anti-bot listeners
                # cluster on.
                if isinstance(inst.x11_wid, int) and self._is_x11_site(inst):
                    try:
                        await self._x11_click_xy(inst, x, y)
                    except Exception as e:
                        logger.warning(
                            "X11 xy-click failed for '%s' at (%s, %s); "
                            "falling back to CDP: %s",
                            agent_id, x, y, e,
                        )
                        await inst.page.mouse.click(x, y)
                else:
                    await inst.page.mouse.click(x, y)
                await asyncio.sleep(action_delay())

                # Don't drop the diff baseline here. Ref-click / type /
                # scroll / fill_form mutate the DOM identically and
                # don't drop their baselines either; dropping only on
                # xy-click made diff behavior depend on which click
                # entry-point the agent used. Keeping the baseline
                # means the next ``snapshot(diff_from_last=True)`` will
                # surface what THIS click changed — which is what the
                # diff feature is for. The next snapshot rebuilds the
                # ref-summary from scratch regardless, so there's no
                # stale-data hazard.

                # Post-click CAPTCHA re-detection — coordinate clicks
                # frequently land on interstitial challenge widgets.
                # _check_captcha now always returns the §11.13 envelope
                # (truthy even with no captcha), so we must explicitly
                # check ``captcha_found`` AND skip auto-solved cases —
                # only surface to the agent when something actually
                # needs their attention.
                envelope = await self._check_captcha(inst)

                inst.m_click_success += 1
                inst.click_window.append(True)
                inst.recorder.record_click(method="xy", success=True)
                result = {
                    "success": True,
                    "data": {
                        "clicked_at": {"x": x, "y": y},
                        "actual_element": {
                            "tag": hit.get("tag", "")[:64],
                            "role": (hit.get("role") or "")[:64] or None,
                            "name": self.redactor.redact(
                                agent_id, hit.get("name", "") or "",
                            )[:200],
                        },
                    },
                }
                if (
                    envelope.get("captcha_found")
                    and envelope.get("solver_outcome") != "solved"
                ):
                    result["data"]["captcha"] = _with_legacy_fields(envelope)
                return result
            except Exception as e:
                inst.m_click_fail += 1
                inst.click_window.append(False)
                inst.recorder.record_click(method="xy", success=False)
                return _err("service_unavailable", str(e))

    async def hover(
        self, agent_id: str, ref: str | None = None,
        selector: str | None = None,
    ) -> dict:
        """Move the mouse over an element without clicking.

        Useful for hover-triggered dropdowns, tooltip visibility, and navigation
        menus that only reveal sub-items on mouseover.  After hovering, call
        snapshot() to see the newly visible elements.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if ref and ref in inst.refs:
                    locator = await self._locator_from_ref(inst, ref)
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    if inst.x11_wid and self._is_x11_site(inst):
                        try:
                            await self._x11_hover(inst, locator)
                        except Exception as e:
                            logger.warning(
                                "X11 hover failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await locator.hover(timeout=_CLICK_TIMEOUT_MS)
                    else:
                        await locator.hover(timeout=_CLICK_TIMEOUT_MS)
                elif selector:
                    if inst.x11_wid and self._is_x11_site(inst):
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_hover(inst, loc)
                        except Exception as e:
                            logger.warning(
                                "X11 hover failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await inst.page.hover(selector, timeout=_CLICK_TIMEOUT_MS)
                    else:
                        await inst.page.hover(selector, timeout=_CLICK_TIMEOUT_MS)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"hovered": ref or selector}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def type_text(self, agent_id: str, ref: str | None = None, selector: str | None = None,
                        text: str = "", clear: bool = True,
                        fast: bool = False, snapshot_after: bool = False,
                        frame: str | None = None) -> dict:
        """Type text into element.

        fast=True uses minimal inter-key delays (8 ms) — still fires real
        keyDown/keyUp events for framework compatibility, but skips
        human-variance timing and think pauses.  Suitable for search
        queries, URLs, and non-sensitive form fields.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            async def _type_body() -> dict:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                # Click to focus, then optionally select-all to clear.
                # Never use fill() — it atomically sets the DOM value and bypasses
                # the keyboard event chain, so React/Vue apps (e.g. X's tweet
                # composer) don't see individual keystrokes and won't activate
                # submit buttons or update their controlled-component state.
                #
                # The entire interaction chain (focus click, select-all, typing)
                # uses X11 input so all events carry isTrusted=true. Mixed
                # CDP+X11 sequences create a detectable signal.
                _use_x11 = bool(inst.x11_wid) and self._is_x11_site(inst)

                resolved_frame = None
                if frame is not None:
                    try:
                        resolved_frame = self._resolve_frame_arg(inst, frame)
                    except RefStale as rs:
                        return {
                            "success": False,
                            "error": {
                                "code": "ref_stale",
                                "message": str(rs),
                            },
                        }
                    if resolved_frame is None:
                        return {
                            "success": False,
                            "error": (
                                f"frame {frame!r} did not match any frame "
                                "url-substring or frame_id"
                            ),
                        }

                if ref and ref in inst.refs:
                    if frame is not None:
                        ref_info = inst.refs[ref]
                        resolved_frame_id = (
                            inst._register_frame(resolved_frame)
                            if resolved_frame is not None else None
                        )
                        # Caller asserted a specific frame; ref must agree.
                        # Fires on main-frame refs (frame_id=None) too — a
                        # frame= arg with a main-frame ref is a bug, not a
                        # silently-ignored hint.
                        if ref_info.frame_id != resolved_frame_id:
                            return {
                                "success": False,
                                "error": {
                                    "code": "invalid_input",
                                    "message": (
                                        "frame argument conflicts with "
                                        "ref's frame"
                                    ),
                                },
                            }
                    try:
                        locator = await self._locator_from_ref(inst, ref)
                    except RefStale as rs:
                        return {
                            "success": False,
                            "error": {
                                "code": "ref_stale",
                                "message": str(rs),
                            },
                        }
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    if _use_x11:
                        try:
                            await self._x11_click(inst, locator)
                        except Exception as e:
                            logger.warning(
                                "X11 focus click failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click(inst.page, locator)
                    else:
                        await self._human_click(inst.page, locator)
                elif selector:
                    if resolved_frame is not None:
                        # X11 click path requires page-level coordinates;
                        # iframe-scoped selectors resolve through
                        # Playwright's frame locator only. Anti-bot benefit
                        # of the X11 path is lost for iframe focus-clicks —
                        # accepted tradeoff.
                        loc = resolved_frame.locator(selector).first
                        try:
                            await self._human_click(inst.page, loc)
                        except Exception as e:
                            return {"success": False, "error": str(e)}
                    elif _use_x11:
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_click(inst, loc)
                        except Exception as e:
                            logger.warning(
                                "X11 focus click failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click_selector(inst.page, selector)
                    else:
                        await self._human_click_selector(inst.page, selector)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}

                # Settle after focus — SPA editors (Lexical, ProseMirror, Draft.js)
                # may expand or initialise event listeners on focus click.
                await asyncio.sleep(0.10 if fast else action_delay())

                if clear:
                    if _use_x11:
                        try:
                            await self._x11_key(inst, "ctrl+a")
                        except Exception as e:
                            logger.warning(
                                "X11 ctrl+a failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await inst.page.keyboard.press("Control+a")
                    else:
                        await inst.page.keyboard.press("Control+a")
                    await asyncio.sleep(0.05)

                if _use_x11:
                    await self._x11_type(inst, text, typos=not fast)
                elif fast:
                    await self._type_fast(inst.page, text)
                else:
                    await self._type_with_variance(inst.page, text)

                # Settle after typing — framework state (React, Lexical, Vue)
                # batches DOM reconciliation asynchronously.
                await asyncio.sleep(0.10 if fast else action_delay())

                inst.recorder.record_keystrokes(
                    char_count=len(text),
                    fast=fast,
                    method="x11" if _use_x11 else ("cdp-fast" if fast else "cdp"),
                )
                result = {"success": True, "data": {"typed_into": ref or selector, "length": len(text)}}
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result

            try:
                action_result, captcha_envelope = await self._with_captcha_redetect(
                    inst, _type_body(),
                )
            except Exception as e:
                return {"success": False, "error": str(e)}

            if (
                captcha_envelope is not None
                and isinstance(action_result, dict)
                and action_result.get("success")
            ):
                action_result.setdefault("captcha", captcha_envelope)
            return action_result

    async def evaluate(self, agent_id: str, expression: str) -> dict:
        """Execute JavaScript and return result.

        Intentionally NOT exposed via the HTTP API (server.py) — arbitrary
        JS execution is a sandbox-escape vector.  Used only internally
        (e.g. navigate body text extraction, scrolling).
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                result = await inst.page.evaluate(expression)
                result = self.redactor.deep_redact(agent_id, result)
                return {"success": True, "data": {"result": result}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def screenshot(
        self,
        agent_id: str,
        full_page: bool = False,
        format: str = "webp",
        quality: int = 75,
        scale: float = 1.0,
    ) -> dict:
        """Take a screenshot and return it as base64.

        ``format`` controls the encoding:
        - ``"webp"`` (default) — lossy WebP at ``quality`` (1–100). Roughly
          5–10× smaller than PNG for the same visual content; the
          difference compounds heavily across multi-step browsing tasks
          where the agent may pull dozens of screenshots per task.
        - ``"png"`` — original lossless PNG path (Playwright native).
          Selected automatically if WebP encoding fails (e.g. Pillow
          missing in dev env, corrupt frame buffer) so callers always
          get a usable image.

        ``scale`` (0.5–1.0) rescales the captured image post-encode-prep,
        for further token savings when full-fidelity isn't needed. The
        Playwright native scale option is intentionally NOT used —
        Playwright applies it via ``deviceScaleFactor`` which mutates the
        viewport's pixel ratio and can leak fingerprint signal. Pillow
        downscale here is a pure post-process.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        # Validate inputs early — reject unknown formats with a clear
        # error rather than silently falling through to PNG. Operator
        # default comes from ``BROWSER_SCREENSHOT_FORMAT`` (per §2.1) so
        # an operator can globally force PNG without changing the caller.
        # ``.strip()`` guards against trailing whitespace from JSON UI
        # defaults; ``.lower()`` normalizes case.
        from src.browser.flags import get_str
        if not format:
            format = get_str(
                "BROWSER_SCREENSHOT_FORMAT", "webp", agent_id=agent_id,
            )
        fmt = format.strip().lower()
        if fmt not in ("webp", "png"):
            return {"success": False, "error": f"Unsupported screenshot format: {format!r}"}
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            quality = 75
        quality = max(1, min(100, quality))
        try:
            scale_f = float(scale)
        except (TypeError, ValueError):
            scale_f = 1.0
        scale_f = max(0.5, min(1.0, scale_f))

        async with inst.lock:
            try:
                # Ask Playwright for PNG either way — WebP encoding happens
                # post-capture so we can downscale and quality-tune in a
                # single Pillow pass without touching the page renderer.
                png_bytes = await inst.page.screenshot(full_page=full_page)
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Pillow encode is synchronous and ~10–20 ms on a 1080p frame —
        # offload to a thread so we don't block the event loop. Pillow
        # releases the GIL during its C-level encode steps, so this
        # actually parallelizes across concurrent agent screenshots.
        encoded, used_format = await asyncio.to_thread(
            _encode_screenshot,
            png_bytes, fmt, quality, scale_f, agent_id=agent_id,
        )
        b64 = base64.b64encode(encoded).decode()
        return {
            "success": True,
            "data": {
                "image_base64": b64,
                "format": used_format,
                "bytes": len(encoded),
            },
        }

    async def _type_with_variance(self, page, text: str) -> None:
        """Type text character-by-character with human-like inter-key delays.

        Uses keyboard.press(char) for every printable character so Playwright
        sends real CDP keyDown/keyUp events.  The browser — not injected JS —
        generates the resulting beforeinput and input events, which carry
        isTrusted=true.  React/Lexical controlled contenteditable elements
        (e.g. X's tweet composer) only update their state — and enable submit
        buttons — when beforeinput.isTrusted is true.

        Why not execCommand('insertText')?  In Firefox (Camoufox's base) the
        event fired by execCommand has isTrusted=false, so Lexical ignores it:
        text appears in the DOM visually but the Post button stays disabled.

        Why not keyboard.type(char)?  Playwright's type() uses CDP
        Input.insertText which injects text without any key events — no keydown,
        no beforeinput.  Same problem.

        keyboard.press(char) → CDP Input.dispatchKeyEvent(keyDown + keyUp) →
        browser generates trusted beforeinput → Lexical/React state updates →
        Post button becomes enabled.

        Fallback: if keyboard.press() raises (character outside Playwright's key
        map, e.g. accented letters, emoji), use keyboard.type() so the character
        at least appears.

        Think-pauses are weighted to word/clause boundaries: 8 % probability
        before the first character of each new word, 1.5 % mid-word.
        """
        prev_char = ""
        for char in text:
            # Word-boundary characters signal a clause break — higher pause
            # probability for the character starting the next word/clause.
            pause_prob = 0.08 if prev_char in _WORD_BOUNDARY_CHARS else 0.015
            if random.random() < pause_prob:
                await asyncio.sleep(think_pause())

            if char == "\n":
                await page.keyboard.press("Enter")
            elif char == "\t":
                await page.keyboard.press("Tab")
            else:
                # Real key events → isTrusted=true beforeinput → framework
                # state updates (Post button lights up on X, etc.)
                try:
                    await page.keyboard.press(char)
                except Exception:
                    # Outside Playwright's key map — fall back, text appears
                    # but framework state may not update.
                    await page.keyboard.type(char)
            await asyncio.sleep(keystroke_delay(char))
            prev_char = char

    async def _type_fast(self, page, text: str) -> None:
        """Type text with minimal delay — still fires real key events.

        Uses keyboard.press(char) for isTrusted=true events so React/Lexical
        state updates work. Inter-key delay is a small Gaussian (μ≈8 ms,
        σ≈3 ms, clamped non-negative) rather than a constant — perfectly
        periodic 8 ms keystrokes are FFT-detectable and zero real human
        types with that exact cadence. Suitable for search queries, URLs,
        and non-sensitive form fields where full human-realistic timing
        is unnecessary but a flat fingerprint is unwanted.
        """
        for char in text:
            if char == "\n":
                await page.keyboard.press("Enter")
            elif char == "\t":
                await page.keyboard.press("Tab")
            else:
                try:
                    await page.keyboard.press(char)
                except Exception:
                    await page.keyboard.type(char)
            # Gaussian jitter ~ N(8 ms, 3 ms), floor at 1 ms so we
            # never schedule a zero-or-negative sleep.
            await asyncio.sleep(max(0.001, random.gauss(0.008, 0.003)))

    async def scroll(self, agent_id: str, direction: str = "down",
                     amount: int = 0, ref: str | None = None) -> dict:
        """Smooth-scroll the page in randomized increments.

        Args:
            direction: "up" or "down"
            amount: total pixels to scroll (0 = one viewport height)
            ref: element ref to scroll into view instead of pixel scrolling
        """
        if direction not in ("up", "down"):
            return {"success": False, "error": f"Invalid direction: {direction!r} (use 'up' or 'down')"}

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                # Scroll element into view
                if ref:
                    if ref not in inst.refs:
                        return {"success": False, "error": f"Ref '{ref}' not found in snapshot"}
                    locator = await self._locator_from_ref(inst, ref)
                    if locator:
                        await locator.scroll_into_view_if_needed(timeout=5000)
                        return {"success": True, "data": {"scrolled_to_ref": ref}}
                    return {"success": False, "error": f"Ref '{ref}' not found"}

                # Pixel-based scrolling
                if amount <= 0:
                    vp = inst.page.viewport_size
                    amount = vp["height"] if vp else 800
                amount = min(amount, _MAX_SCROLL_PX)

                sign = -1 if direction == "up" else 1
                _use_x11 = bool(inst.x11_wid) and self._is_x11_site(inst)

                if _use_x11:
                    # X11 scroll: each button 4/5 click ≈ 3 lines ≈ 53 px.
                    # Produces real WheelEvent with deltaMode=DOM_DELTA_LINE,
                    # matching physical mouse hardware.
                    _PX_PER_NOTCH = 53
                    button = "4" if direction == "up" else "5"
                    total_notches = max(1, round(amount / _PX_PER_NOTCH))
                    scrolled = 0
                    for i in range(total_notches):
                        progress = i / max(1, total_notches)
                        ramp = scroll_ramp(progress)
                        try:
                            await self._x11_scroll_notch(inst, button)
                        except Exception as e:
                            logger.warning(
                                "X11 scroll failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            # Fall back to CDP for actual remaining distance
                            remaining_px = max(0, amount - scrolled)
                            await inst.page.mouse.wheel(0, remaining_px * sign)
                            scrolled += remaining_px
                            break
                        scrolled += _PX_PER_NOTCH
                        if i < total_notches - 1:
                            await asyncio.sleep(scroll_pause() / max(0.5, ramp))
                    scrolled = min(scrolled, amount)
                else:
                    # CDP fallback when X11 unavailable
                    scrolled = 0
                    while scrolled < amount:
                        remaining = amount - scrolled
                        progress = scrolled / amount if amount > 0 else 1.0
                        ramp = scroll_ramp(progress)
                        step = max(40, int(scroll_increment() * ramp))
                        step = min(step, remaining)
                        delta = step * sign
                        await inst.page.mouse.wheel(0, delta)
                        scrolled += step
                        if scrolled < amount:
                            await asyncio.sleep(scroll_pause() / max(0.5, ramp))

                inst.recorder.record_scroll(
                    direction=direction,
                    delta=scrolled,
                    method="x11" if _use_x11 else "cdp",
                )
                return {
                    "success": True,
                    "data": {"direction": direction, "pixels": scrolled},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def wait_for_element(
        self, agent_id: str, selector: str, state: str = "visible",
        timeout_ms: int = 10000,
    ) -> dict:
        """Wait for a CSS selector to reach the given state.

        state: "visible" (default), "attached", "hidden", or "detached".
        Useful before clicking elements on SPAs that animate in after load.
        """
        _valid_states = frozenset({"visible", "attached", "hidden", "detached"})
        if state not in _valid_states:
            return {"success": False, "error": f"Invalid state: {state!r}. Use one of: {sorted(_valid_states)}"}
        timeout_ms = max(0, min(timeout_ms, _WAIT_FOR_TIMEOUT_MS))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                await inst.page.wait_for_selector(selector, state=state, timeout=timeout_ms)
                return {"success": True, "data": {"selector": selector, "state": state}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _with_captcha_redetect(
        self,
        inst: CamoufoxInstance,
        action_coro,
    ):
        """§11.4 / §18.2 — wrap an action with MutationObserver-based
        post-action captcha re-detection.

        Pattern:

          1. Install a ``MutationObserver`` on ``document.body`` capturing
             every added Element node.
          2. Run ``action_coro`` (the original action body).
          3. Read back the captured nodes against the
             :data:`_CAPTCHA_REDETECT_SELECTORS` list. If any node matches
             AND we're outside the per-instance rate-limit window, route
             through the full :meth:`_check_captcha` chain (which goes
             through :meth:`_metered_solve` and so respects every gate:
             rate-limit, cost-cap, kill-switch, breaker, behavioral).
          4. Stash the resulting §11.13 envelope on
             ``inst._pending_captcha_envelope`` AND return it alongside
             the action result so the agent sees it inline.

        **Failure modes — all swallowed.** Install failure (page closed,
        CSP block) means we run the action with no probe. Read-back
        failure (navigation between install and read-back, page closed)
        treats the probe as never having fired — empty hits, no
        ``_check_captcha`` call.

        **Navigation during action.** When the page navigates away,
        ``window.__ol_captcha_probe`` lives on the old document and is
        torn down. The read-back from the new page returns ``[]`` and we
        skip the auto-trigger; the navigate path's own ``_check_captcha``
        call (in :meth:`navigate`) is the right defense for the new page.

        **Rate-limit.** The wrapper checks
        ``inst._last_redetect_ts`` against
        :data:`_REDETECT_MIN_INTERVAL_S` BEFORE invoking
        ``_check_captcha`` — a probe-storm during a single mutation-heavy
        action triggers the full chain at most once per 2s. The
        ``_metered_solve`` gates remain authoritative for solver cost /
        rate; this throttle is a CPU-cost guard for the auto-detect path.

        **Empty-probe defense.** Spec calls for triggering only on a
        match. We respect that — pages that swap captchas via
        ``innerHTML`` replace (rare) will miss the MutationObserver and
        agents must fall back to the explicit ``solve_captcha`` skill.
        Always firing ``_check_captcha`` would defeat the cost-saving
        purpose of this wrapper (each call walks 7 selectors via
        ``locator(...).count()``).

        **Flag gate.** ``BROWSER_CAPTCHA_REDETECT_ENABLED=false`` makes
        the wrapper a passthrough — action runs, no install/read-back,
        no auto-trigger.

        Returns ``(result, envelope_or_none)`` where ``result`` is the
        action's own return value (passed through unchanged) and
        ``envelope_or_none`` is the §11.13 envelope when re-detection
        fired and matched, or ``None`` otherwise. Callers add the
        envelope to their action response under a ``"captcha"`` key
        when present.
        """
        from src.browser.flags import get_bool

        agent_id = inst.agent_id

        if not get_bool(
            "BROWSER_CAPTCHA_REDETECT_ENABLED", True, agent_id=agent_id,
        ):
            # Passthrough — operator disabled re-detection. No probe, no
            # read-back, no auto-trigger.
            return await action_coro, None

        # Snapshot the page object + URL up front so the read-back can
        # detect navigation (page swap or URL change) and skip cleanly.
        pre_page = inst.page
        try:
            pre_url = inst.page.url or ""
        except Exception:
            pre_url = ""

        # 1. Install. Failures are non-fatal — we still run the action.
        # Do not run read-back unless install succeeded; a failed install
        # could otherwise read stale probe state left by a prior action.
        probe_var = inst._captcha_probe_var
        probe_installed = False
        try:
            await inst.page.evaluate(_JS_CAPTCHA_REDETECT_INSTALL, probe_var)
            probe_installed = True
        except Exception as e:
            logger.debug(
                "captcha redetect: install failed for %s: %s",
                agent_id, e,
            )

        # 2. Run the action. Always propagate exceptions — the wrapper
        # exists to add detection, not to swallow action failures. When
        # the action fails after a successful install, run the read-back
        # with the REAL selector list so we can both tear down the
        # observer AND surface "the action failed because a captcha
        # appeared mid-flight" as actionable diagnostic context. The
        # captured hits ride on the exception via ``captcha_redetect_hits``
        # so callers / observability can distinguish a generic action
        # failure from a captcha-induced one.
        try:
            result = await action_coro
        except Exception as exc:
            if probe_installed:
                try:
                    captcha_hits = await inst.page.evaluate(
                        _JS_CAPTCHA_REDETECT_READBACK,
                        [probe_var, list(_CAPTCHA_REDETECT_SELECTORS)],
                    )
                    if not isinstance(captcha_hits, list):
                        captcha_hits = []
                    if captcha_hits:
                        logger.info(
                            "captcha redetect: action failed AFTER captcha "
                            "appeared for %s — selectors hit: %s. Likely "
                            "cause of failure.",
                            agent_id, captcha_hits,
                        )
                        try:
                            exc.captcha_redetect_hits = captcha_hits  # type: ignore[attr-defined]
                        except Exception:
                            # Some exception types (raised by C extensions)
                            # reject attribute assignment. Don't let that
                            # mask the original failure.
                            pass
                except Exception as readback_err:
                    logger.debug(
                        "captcha redetect: read-back failed during "
                        "action-failure tear-down for %s: %s",
                        agent_id, readback_err,
                    )
            raise

        # 3. Read back the captured added nodes.
        navigated = False
        try:
            current_url = inst.page.url or ""
        except Exception:
            current_url = ""
        if inst.page is not pre_page or current_url != pre_url:
            navigated = True

        hits: list[str] = []
        if probe_installed and not navigated:
            try:
                hits = await inst.page.evaluate(
                    _JS_CAPTCHA_REDETECT_READBACK,
                    [probe_var, list(_CAPTCHA_REDETECT_SELECTORS)],
                )
                if not isinstance(hits, list):
                    hits = []
            except Exception as e:
                logger.debug(
                    "captcha redetect: read-back failed for %s: %s",
                    agent_id, e,
                )
                hits = []

        if not hits:
            # No captcha-shaped DOM additions detected — fastest path,
            # no further work.
            return result, None

        # 4. Per-instance rate-limit. Probe storms (heavily-mutating
        # actions, agent loops issuing back-to-back clicks) must not
        # invoke the full 7-selector locator probe more than
        # ``_REDETECT_MIN_INTERVAL_S`` per instance.
        now = time.monotonic()
        if (
            inst._last_redetect_ts > 0
            and (now - inst._last_redetect_ts) < _REDETECT_MIN_INTERVAL_S
        ):
            logger.debug(
                "captcha redetect: rate-limited for %s (%.2fs since last)",
                agent_id, now - inst._last_redetect_ts,
            )
            return result, None
        inst._last_redetect_ts = now

        # 5. Full ``_check_captcha`` chain — gates fire inside.
        try:
            envelope = await self._check_captcha(inst)
        except Exception as e:
            logger.debug(
                "captcha redetect: _check_captcha raised for %s: %s",
                agent_id, e,
            )
            return result, None

        if not envelope.get("captcha_found"):
            return result, None
        if envelope.get("solver_outcome") == "solved":
            # Solver auto-cleared it — nothing for the agent to do.
            return result, None

        # Stash for snapshot integration AND return inline so the agent
        # sees it on whichever code path it consults next.
        legacy = _with_legacy_fields(envelope)
        inst._pending_captcha_envelope = legacy
        return result, legacy

    async def _metered_solve(
        self,
        inst: CamoufoxInstance,
        sel: str,
        kind: str,
    ) -> SolveResult:
        """Run rate + cost gates, invoke the solver, account on token retrieval.

        This is the SINGLE entry point to ``self._captcha_solver.solve()``.
        Centralizing the gates here closes the long-standing bug where
        ``solve_captcha`` checked the rate-limit + cost-cap AFTER
        ``_check_captcha`` had already invoked the solver, AND where the
        navigate / click auto-detect paths bypassed the gates entirely.
        Now every path that detects a captcha and runs a solver flows
        through ``_check_captcha`` → ``_metered_solve`` → gates → solver.

        Order of operations:

          1. Fleet-wide kill switch.
          2. Cost-cap gate (reserve maximum published price under lock).
          3. Rate-limit gate (records a slot only on a real attempt — gates
             that short-circuit don't burn the per-hour budget).
          4. Provider/pricing sanity when a cost cap requires accounting.
          5. Solver HTTP call. ``solve()`` returns :class:`SolveResult`.
          6. Cost accounting. The reservation is corrected/refunded when
             ``result.token`` is known. Actual increments fire when a token
             was retrieved regardless of ``injection_succeeded`` — the
             provider was paid the moment the token came back; injection
             failure is our problem, not theirs.

        Returns the :class:`SolveResult` so the caller (``_check_captcha``)
        can build the §11.13 envelope from the per-call data — no shared
        instance state to race on.
        """
        agent_id = inst.agent_id
        try:
            page_url = inst.page.url or ""
        except Exception:
            page_url = ""

        # Gate 0: fleet-wide kill switch. ``CAPTCHA_DISABLED`` short-circuits
        # BEFORE any other gate so the auto-detect entry points (navigate /
        # click) never reach the provider. The duplicate early-return inside
        # ``solve_captcha`` was removed once this gate landed — single
        # checkpoint inside ``_metered_solve`` covers all callers uniformly.
        from src.browser.flags import get_bool
        if get_bool("CAPTCHA_DISABLED", False, agent_id=agent_id):
            await _record_captcha_audit_event(
                agent_id, "kill_switch_active", kind, page_url,
            )
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
                skipped="disabled",
            )

        provider = getattr(self._captcha_solver, "provider", "")
        reserved_millicents = 0

        # Gate 1: cost-cap. Reserve the maximum published price under the
        # counter lock before the provider HTTP call. That makes concurrent
        # solves for the same agent see in-flight spend and prevents each
        # request from independently passing a read-only ``over_cap`` check.
        # We refund or correct the reservation after the provider result is
        # known. Order stays cost → rate so a cost-blocked solve doesn't
        # burn rate slots that should still be available when the cap resets.
        cap_millicents = _resolve_cost_cap(agent_id)
        if cap_millicents > 0:
            from src.browser import captcha_cost_counter as _cost
            if not isinstance(provider, str) or not provider:
                logger.warning(
                    "captcha solve: solver has no string ``provider`` "
                    "attribute; cost cap is configured (cap=%s "
                    "millicents) — failing closed so an untracked "
                    "charge cannot bypass the cap. Configure the "
                    "solver with a known provider name (2captcha / "
                    "capsolver) to proceed.",
                    cap_millicents,
                )
                await _record_captcha_audit_event(
                    agent_id, "provider_missing", kind, page_url,
                )
                return SolveResult(
                    token=None, injection_succeeded=False,
                    used_proxy_aware=False, compat_rejected=False,
                    skipped="provider_missing",
                )
            reservation = _max_published_solve_cost_millicents(provider, kind)
            if reservation is None:
                logger.warning(
                    "captcha solve: no published rate for provider=%s "
                    "kind=%s while cost cap is configured (cap=%s "
                    "millicents) — failing closed so an untracked charge "
                    "cannot bypass the cap.",
                    provider, kind, cap_millicents,
                )
                await _record_captcha_audit_event(
                    agent_id, "price_missing", kind, page_url,
                )
                return SolveResult(
                    token=None, injection_succeeded=False,
                    used_proxy_aware=False, compat_rejected=False,
                    skipped="price_missing",
                )
            allowed, _total = await _cost.check_and_charge(
                agent_id, cap_millicents, reservation,
            )
            if not allowed:
                await _record_captcha_audit_event(
                    agent_id, "cost_cap", kind, page_url,
                )
                return SolveResult(
                    token=None, injection_succeeded=False,
                    used_proxy_aware=False, compat_rejected=False,
                    skipped="cost_cap",
                )
            reserved_millicents = reservation

        # All paths past the cost-cap gate must reconcile the
        # reservation. Wrap the rest of the function in try/finally with
        # a ``reservation_settled`` flag so that ANY exit (return,
        # raise, ``CancelledError``) cleans up the reserved spend rather
        # than leaving it stuck on the agent's bucket. Pre-fix, an
        # ``asyncio.Task.cancel`` mid-solve (caller timed out, browser
        # restart, container shutdown) would leave the reservation
        # permanently committed even though no token was ever received.
        from src.browser import captcha_cost_counter as _cost
        reservation_settled = False
        try:
            # Gate 2: rate-limit. ``_check_solve_rate`` consumes a slot
            # only on the actual-attempt path (returning False); a True
            # return means we are over the limit, no slot consumed.
            rate_limit = _resolve_rate_limit(agent_id)
            if await _check_solve_rate(agent_id, rate_limit):
                await _record_captcha_audit_event(
                    agent_id, "rate_limited", kind, page_url,
                )
                return SolveResult(
                    token=None, injection_succeeded=False,
                    used_proxy_aware=False, compat_rejected=False,
                    skipped="rate_limited",
                )

            # Gate 3: provider sanity. When a cost cap is configured, we
            # MUST be able to attribute every solve to a published rate
            # so the cap math stays honest. A solver without a string
            # ``provider`` cannot be priced; in that case we fail closed
            # (skip the solve) rather than let an untrackable charge
            # slip past the cap. When no cap is configured the warning
            # still fires so operators see the misconfiguration, but we
            # proceed with the solve — keeps tests and custom solver
            # integrations from breaking outright.
            if not isinstance(provider, str) or not provider:
                logger.warning(
                    "captcha solve: solver has no string ``provider`` "
                    "attribute; cost accounting will be SKIPPED for "
                    "this solve (cap not configured so the warning is "
                    "informational). Set solver.provider to a known "
                    "name to enable cost tracking.",
                )

            # Run the actual solver. Any exception bubbles up so the
            # caller can build the appropriate ``timeout``/``rejected``
            # envelope; we don't swallow them here because the
            # envelope-mapping is caller-specific. Audit the
            # exception-during-solve case so operator dashboards see
            # the storm before the breaker trips.
            try:
                result = await self._captcha_solver.solve(
                    inst.page, sel, page_url,
                    agent_id=agent_id, kind=kind,
                )
            except Exception:
                await _record_captcha_audit_event(
                    agent_id, "solver_exception", kind, page_url,
                )
                raise

            # §11.8 multi-provider: re-read the wrapper's ``provider``
            # AFTER ``solve()`` so the cost-accounting tier matches the
            # provider that actually served the solve, not the one we
            # captured pre-call. On a mid-call failover (primary fatal
            # at ``createTask`` → wrapper retries on secondary inside
            # the same ``solve()`` call) the active solver flips, and
            # using the stale pre-call snapshot would charge the agent
            # at primary's pricing for a secondary-served solve. The
            # post-solve read is single-coroutine-safe — there is no
            # await between this read and the cost-counter update.
            actual_provider = getattr(self._captcha_solver, "provider", "")
            if not (isinstance(actual_provider, str) and actual_provider):
                actual_provider = provider

            # Cost accounting fires on token retrieval, NOT on
            # injection success — the provider already charged for the
            # token. Skip accounting when no token came back (gates /
            # sitekey-fail / provider-reject / breaker / unreachable).
            if result.token is not None:
                # Defensive: solver mocks in tests sometimes leave
                # ``provider`` as a MagicMock auto-spec child; only
                # proceed when we have an actual string we can pass to
                # ``estimate_millicents``. The provider-missing case
                # was warned above; here we just skip the increment
                # silently and let the finally-refund clean up the
                # reservation.
                if isinstance(actual_provider, str) and actual_provider:
                    millicents = _cost.estimate_millicents(
                        actual_provider, kind,
                        proxy_aware=result.used_proxy_aware,
                    )
                    if millicents is None:
                        logger.warning(
                            "captcha solve: no published rate for "
                            "provider=%s kind=%s proxy_aware=%s — "
                            "keeping any reserved cost-cap charge; "
                            "otherwise skipping cost increment "
                            "(under-count > over-count)",
                            actual_provider, kind, result.used_proxy_aware,
                        )
                        # With a configured cap, we reserved the max
                        # published tier before the provider call. If
                        # the solver reports an unpriced actual tier
                        # after returning a token, keep that reservation
                        # instead of refunding it; refunding would let an
                        # untracked provider charge bypass the cap.
                        if reserved_millicents:
                            reservation_settled = True
                    else:
                        if reserved_millicents:
                            await _cost.adjust_cost(
                                agent_id,
                                millicents - reserved_millicents,
                            )
                        else:
                            await _cost.add_cost(agent_id, millicents)
                        reservation_settled = True
            return result
        finally:
            # If we got here without committing the reservation (token-
            # less return, gate-trip return, raise, cancellation),
            # refund. ``adjust_cost`` is clamped at zero so a refund
            # that races with another adjustment can't drive negative.
            #
            # ``asyncio.CancelledError`` is a ``BaseException`` (Py 3.8+),
            # so a plain ``contextlib.suppress(Exception)`` does NOT
            # catch it — and ``await`` on the refund coro is itself a
            # cancellation point that re-raises CancelledError, leaving
            # the reservation permanently committed (silent over-charge).
            # ``asyncio.shield`` wraps the coro in a Task that the outer
            # cancellation cannot propagate INTO; the local ``await``
            # still raises CancelledError but the shielded Task runs
            # to completion in the background. We re-raise to honour
            # cancellation semantics.
            if reserved_millicents and not reservation_settled:
                refund_coro = _cost.adjust_cost(
                    agent_id, -reserved_millicents,
                )
                try:
                    await asyncio.shield(refund_coro)
                except asyncio.CancelledError:
                    # Outer task was cancelled; the shielded refund
                    # Task continues independently. Re-raise to honour
                    # cancellation.
                    raise
                except Exception:
                    logger.warning(
                        "captcha cost refund failed", exc_info=True,
                    )

    async def _monitor_post_solve_state(
        self,
        inst: CamoufoxInstance,
        captcha_selectors: list[str],
        kind: str,
        page_origin: str,
    ) -> None:
        """§22 — observe post-solve page state and record outcome.

        Runs as a fire-and-forget task spawned from :meth:`_check_captcha`
        AFTER a ``solver_outcome="solved"`` envelope has been built.  We
        do NOT block the action response on this — the operator-facing
        burn metric is best-effort (an outcome we couldn't classify is
        better dropped than blocking the agent loop on a 10s wait).

        Signals:
          * Same captcha selector still present after the timeout →
            rejected. (Anti-bot framework re-rendered the challenge.)
          * Page text contains one of :data:`_FINGERPRINT_REJECTION_TEXT`
            within the timeout → rejected.
          * Navigation away from the captcha page AND no captcha selector
            on the new page → accepted.
          * Stale signal (no clear outcome after the timeout) → DROP.
            Polluting the metric with ambiguity would inflate the
            rejection rate on slow sites and drive false-positive burns.

        The monitor never raises — exceptions are caught and logged at
        DEBUG so a torn-down Page during cancellation doesn't surface as
        a noisy ERROR in the agent log.
        """
        agent_id = inst.agent_id
        deadline = time.monotonic() + _FINGERPRINT_MONITOR_TIMEOUT_S
        # Capture the URL at spawn-time so we can detect a navigation away.
        try:
            initial_url = inst.page.url or ""
        except Exception:
            initial_url = ""
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Stale — drop without recording.  See docstring.
                    return
                # Sleep in small slices so a redirect chain that finishes
                # fast can be detected before the full 10s elapses; cheap
                # because the rejection-text check is one ``inner_text()``
                # call per slice.
                await asyncio.sleep(min(0.5, remaining))

                # Has the page navigated to a different URL?
                try:
                    current_url = inst.page.url or ""
                except Exception:
                    # Page closed mid-monitor (agent stopped, tab gone).
                    # Treat as a stale signal — no record.
                    return

                # Probe captcha selectors on the live page.  Wrap each in
                # its own try so a single bad selector doesn't kill the
                # whole monitor pass.
                still_present = False
                for sel in captcha_selectors:
                    try:
                        if await inst.page.locator(sel).count() > 0:
                            still_present = True
                            break
                    except Exception:
                        # Locator raises during nav / detached frame.
                        # Don't conclude rejection from a probe failure.
                        continue

                # Vendor-specific element probes — these are the most
                # reliable signal because they're language-independent
                # and survive layout changes.  Cloudflare 1xxx error
                # containers, DataDome / PerimeterX / Imperva / Akamai /
                # F5 block-page selectors all surface here.  A single
                # match is enough — the monitor's job is to tell us
                # the *fingerprint* is flagged, not which vendor flagged it.
                vendor_block_found = False
                for sel in _FINGERPRINT_REJECTION_SELECTORS:
                    try:
                        if await inst.page.locator(sel).count() > 0:
                            vendor_block_found = True
                            break
                    except Exception:
                        # Locator raises during nav / detached frame.
                        # Try the next selector — a probe failure for
                        # one vendor doesn't tell us anything about
                        # whether another vendor flagged us.
                        continue

                if vendor_block_found:
                    await self._handle_post_solve_outcome(
                        agent_id, rejected=True, page_origin=page_origin,
                    )
                    return

                # Look for explicit rejection text.  Bound the read so a
                # huge body doesn't stall the monitor; ``inner_text`` on
                # ``body`` is fine for any normal anti-bot interstitial.
                rejection_text_found = False
                try:
                    body_text = await inst.page.locator("body").inner_text(
                        timeout=2000,
                    )
                    if body_text:
                        haystack = body_text.lower()
                        for needle in _FINGERPRINT_REJECTION_TEXT:
                            if needle in haystack:
                                rejection_text_found = True
                                break
                except Exception:
                    # Best-effort — body read failed (cross-origin frame
                    # / page closed / read timeout).  Don't infer
                    # rejection from a read failure.
                    pass

                if rejection_text_found:
                    await self._handle_post_solve_outcome(
                        agent_id, rejected=True, page_origin=page_origin,
                    )
                    return

                # Navigation-away check: only counts as accepted when
                # the new page also has no captcha selector. A redirect
                # to a new captcha page (multi-step gate) keeps us in
                # the loop until the deadline rather than recording a
                # premature accept.
                if current_url != initial_url and not still_present:
                    await self._handle_post_solve_outcome(
                        agent_id, rejected=False, page_origin=page_origin,
                    )
                    return

                # Same URL + still present after we've had at least a
                # small chunk of the budget — keep waiting until deadline
                # to see if the page clears.  Only the FINAL still-present
                # check (after the full timeout) records a rejection,
                # which we do at the top-of-loop fall-through.
                if remaining <= 0.5 and still_present:
                    await self._handle_post_solve_outcome(
                        agent_id, rejected=True, page_origin=page_origin,
                    )
                    return
        except asyncio.CancelledError:
            # Agent stopped — don't record anything.  Re-raise to honour
            # cancellation.
            raise
        except Exception:
            # Anything else — log + drop.  Monitors must NEVER crash an
            # agent's hot path.
            logger.debug(
                "fingerprint monitor for %s raised unexpectedly",
                agent_id, exc_info=True,
            )
            return
        # Suppress unused-variable lint for ``kind`` — kept on the
        # signature for future extension (per-kind monitor tuning).
        _ = kind

    async def _handle_post_solve_outcome(
        self, agent_id: str, *, rejected: bool, page_origin: str,
    ) -> None:
        """Record an outcome to the rolling window + audit aggregator.

        Centralised so the monitor task and any future explicit-record
        callers (e.g. an action that observes a rejection synchronously)
        emit consistent signals to the dashboard.  Burn-state crossing
        emits the ``fingerprint_burn`` audit event exactly once per
        crossing — repeated calls within the burn window do not re-emit.
        """
        try:
            was_burned_before = await _is_fingerprint_burned(agent_id)
            now_burned = await _record_fingerprint_outcome(
                agent_id, rejected,
            )
            signal = "rejected" if rejected else "accepted"
            await _record_fingerprint_audit_event(
                agent_id, signal, page_origin,
            )
            if now_burned and not was_burned_before:
                # Fresh crossing of the burn threshold — surface as a
                # distinct signal so the dashboard panel can highlight
                # the transition (rather than relying on aggregating
                # rejection counts).  Per-burn emit, NOT per-rejection.
                await _record_fingerprint_audit_event(
                    agent_id, "fingerprint_burn", page_origin,
                )
                logger.warning(
                    "fingerprint burn detected for %s (rejection rate "
                    ">=%.0f%%); next captcha solves will surface "
                    "next_action=retry_with_fresh_profile",
                    agent_id, _FINGERPRINT_BURN_THRESHOLD * 100,
                )
        except Exception:
            logger.debug(
                "fingerprint outcome record failed for %s",
                agent_id, exc_info=True,
            )

    def _spawn_fingerprint_monitor(
        self,
        inst: CamoufoxInstance,
        captcha_selectors: list[str],
        kind: str,
    ) -> None:
        """Schedule :meth:`_monitor_post_solve_state` as a tracked task.

        Registers the task on ``inst._fingerprint_monitor_tasks`` so
        :meth:`_stop_instance` can cancel it at agent stop.  Tasks
        self-deregister on completion via ``add_done_callback`` so the
        set doesn't grow unbounded for a long-running agent.
        """
        try:
            page_url = inst.page.url or ""
        except Exception:
            page_url = ""
        page_origin = _page_origin_for_audit(page_url)
        try:
            task = asyncio.create_task(
                self._monitor_post_solve_state(
                    inst, captcha_selectors, kind, page_origin,
                ),
                name=f"fp-monitor-{inst.agent_id}",
            )
        except RuntimeError:
            # No running loop — extremely unusual (we got here via an
            # ``async def``); play safe and skip the spawn rather than
            # raise into the caller.
            return
        inst._fingerprint_monitor_tasks.add(task)
        task.add_done_callback(
            lambda t, _inst=inst: _inst._fingerprint_monitor_tasks.discard(t),
        )

    async def _check_captcha(self, inst: CamoufoxInstance) -> dict:
        """Check for CAPTCHA elements and attempt auto-solve if configured.

        **Lock hold time.** When the §11.3 CF auto-resolving challenge
        path fires, this method holds ``inst.lock`` for up to
        :data:`_CF_AUTO_WAIT_SECONDS` (8s) while awaiting the recheck.
        Other in-flight requests to the same agent block during this
        window. This is intentional — the agent should not interact with
        the page while CF resolves, and the alternative (releasing and
        re-acquiring) opens a window where ``click()`` could fire on a
        challenge widget mid-resolution.

        Returns the §11.13 structured envelope ``data`` block in all cases:
          - ``{"captcha_found": false}`` when no captcha selector matched.
          - ``{"captcha_found": true, "kind": ..., "solver_attempted": ...,
                "solver_outcome": ..., "injection_failure_reason": ...,
                "solver_confidence": ..., "next_action": ...}`` otherwise.

        Callers should treat ``solver_outcome == "solved"`` as "no agent
        action needed" — the captcha was detected but already cleared.

        Classification dispatches the matched selector through three
        live-page classifiers in order:

        1. §11.1 reCAPTCHA variant classifier (``_classify_recaptcha``) —
           upgrades the coarse ``recaptcha-v2-checkbox`` placeholder to
           ``recaptcha-v{2-invisible,3,enterprise-v2,enterprise-v3}``.
        2. §11.3 behavioral-only classifier (``_classify_behavioral``) —
           detects PerimeterX Press & Hold and DataDome ``/blocker``
           pages. Runs **before** the §11.16 solver health/breaker gates
           so behavioral-only flows don't consume health-check or
           breaker quota; the solver genuinely cannot help with these.
        3. §11.3 Cloudflare tri-state classifier (``_classify_cf_state``):
           - ``auto``: wait :data:`_CF_AUTO_WAIT_SECONDS` once and
             re-check the selector list. If page navigated away, return
             ``solver_outcome="solved"``; otherwise treat as behavioral.
           - ``behavioral``: emit ``cf-interstitial-behavioral`` with
             ``skipped_behavioral`` outcome.
           - ``turnstile``: route through the existing Turnstile solver
             path with kind upgraded to ``cf-interstitial-turnstile``
             and ``solver_confidence`` forced to ``"low"`` regardless
             of the solver verdict (CF binds Turnstile tokens to session
             cookies; the solve may not unblock the session).
           - ``none``: fall through to the existing flow unchanged.

        §11.16 layers two solver-health side-channels on top of the §11.13
        envelope, checked BEFORE attempting ``solver.solve()``:

        * ``solver.is_solver_unreachable()`` — health probe marked the
          provider unreachable for this instance-session. Short-circuits
          to ``solver_outcome="no_solver"`` /
          ``next_action="request_captcha_help"``; the API is never
          contacted. The check is async + lazy: when the per-process
          probe hasn't fired, the read awaits the probe rather than
          falling through to ``solver.solve()`` and finding out there.
          This closes the original first-captcha-of-session bug where
          the gate read fired before the probe ran.
        * ``solver.is_breaker_open()`` — sliding-window circuit breaker
          is tripped. Short-circuits to ``solver_outcome="timeout"`` /
          ``next_action="request_captcha_help"`` plus a top-level
          additive ``"breaker_open": True`` flag (deliberately *not*
          encoded as a new enum value, to avoid colliding with §11.13's
          ``solver_outcome`` set; a richer ``service_unavailable``
          outcome may land in a follow-up).

        Solver invocation is delegated to :meth:`_metered_solve` which
        runs the rate-limit + cost-cap gates BEFORE the HTTP call and
        accounts cost on token retrieval (not injection success). All
        callers — navigate auto-detect, click auto-detect, ``solve_captcha``,
        ``detect_captcha`` — flow through this single method, so the
        gates can no longer be bypassed by reaching the solver from a
        non-explicit path.
        """
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            '[class*="cf-turnstile"]',
            'iframe[src*="challenges.cloudflare.com"]',
            'iframe[src*="captcha"]',
            '[class*="captcha"]',
            '#captcha',
        ]
        try:
            for sel in captcha_selectors:
                if await inst.page.locator(sel).count() > 0:
                    kind = self._classify_kind(sel)
                    # §11.1 — when the matched selector is a reCAPTCHA, run
                    # the precise variant classifier on the live page to
                    # disambiguate v2-checkbox / v2-invisible / v3 /
                    # Enterprise-v2 / Enterprise-v3. The coarse selector
                    # classifier returns ``recaptcha-v2-checkbox`` as the
                    # safe default; the variant classifier upgrades the
                    # ``kind`` whenever it returns anything other than
                    # ``unknown``. Falls back silently to the coarse value
                    # if ``page.evaluate`` fails (test mocks, frames,
                    # registry obfuscation, etc.).
                    if "recaptcha" in sel:
                        try:
                            classified = await _classify_recaptcha(inst.page)
                            variant = classified.get("variant") or "unknown"
                            if variant != "unknown":
                                kind = variant
                        except Exception:
                            logger.debug(
                                "_classify_recaptcha raised; falling back to "
                                "coarse kind=%s", kind, exc_info=True,
                            )

                    # §22 — anti-bot kind solver path. When the active
                    # solver provider declares a task type for the
                    # detected anti-bot kind (CapSolver publishes
                    # AntiAkamaiBMPTask / AntiImpervaTask / AntiKasadaTask /
                    # DataDomeSliderTask; 2Captcha publishes none) we
                    # OPT-IN to the solver path with forced low
                    # confidence. Otherwise we keep the legacy
                    # operator-escalation path. The two-step gate
                    # (``supports_kind`` + low-confidence forcing) is
                    # what keeps a 2Captcha-only deployment from spilling
                    # ``ERROR_INVALID_TASK_TYPE`` errors that would trip
                    # the §11.16 breaker.
                    antibot_force_low_confidence = False

                    # §19 — tier-1 anti-bot JS-challenge classifier runs
                    # BEFORE the §11.3 behavioral classifier. JS challenges
                    # (Akamai Bot Manager, Kasada, FingerprintJS Pro,
                    # Imperva ABP, F5 Bot Defense) typically nest ABOVE
                    # CAPTCHA in the typical site stack — they fire first
                    # and gate access entirely, so detecting them takes
                    # precedence over the §11.3 behavioral pass.
                    #
                    # §22 — kinds with a CapSolver task type (Akamai,
                    # Imperva, Kasada) fall through to the solver path
                    # IFF the active solver supports them. FingerprintJS
                    # and F5 have no CapSolver task type and remain
                    # operator-escalation-only.
                    js_vendor = await classify_js_challenge(inst.page)
                    if js_vendor:
                        js_kind = f"js-challenge-{js_vendor}"
                        try:
                            page_url = inst.page.url or ""
                        except Exception:
                            page_url = ""
                        # Anti-bot solver opt-in. Three-step gate:
                        #   1. Kind must be in :data:`_ANTIBOT_KINDS`
                        #      (excludes ``js-challenge-fingerprintjs``
                        #      and ``js-challenge-f5`` — no published
                        #      CapSolver task type).
                        #   2. Solver must be configured.
                        #   3. Active solver must declare a task type
                        #      for the kind via ``supports_kind`` —
                        #      keeps 2Captcha from spilling
                        #      ``ERROR_INVALID_TASK_TYPE``.
                        if (
                            js_kind in _ANTIBOT_KINDS
                            and self._captcha_solver is not None
                            and _solver_supports_kind(
                                self._captcha_solver, js_kind,
                            )
                        ):
                            logger.info(
                                "JS-challenge detected (%s); active solver "
                                "supports the kind, attempting solver path "
                                "with forced low confidence",
                                js_kind,
                            )
                            kind = js_kind
                            antibot_force_low_confidence = True
                            # Fall through to the standard solver path
                            # (cf_force_low_confidence is set below).
                        else:
                            logger.info(
                                "JS-challenge detected (%s); skipping solver, "
                                "escalating to request_captcha_help",
                                js_kind,
                            )
                            await _record_captcha_audit_event(
                                inst.agent_id, "skipped_behavioral",
                                js_kind, page_url,
                            )
                            return _captcha_envelope(
                                kind=js_kind,
                                solver_attempted=False,
                                solver_outcome="skipped_behavioral",
                                solver_confidence="behavioral-only",
                                next_action="request_captcha_help",
                            )

                    # §11.3 — behavioral-only classifier runs BEFORE the
                    # solver health/breaker gates so behavioral-only flows
                    # (PerimeterX Press & Hold, DataDome blocker) don't
                    # consume health-check or breaker quota. PerimeterX
                    # remains operator-escalation-only — no published
                    # CapSolver task type as of April 2026.
                    #
                    # §22 — DataDome behavioral routes through the solver
                    # path IFF the active solver supports
                    # ``datadome-behavioral`` (CapSolver does;
                    # ``DataDomeSliderTask``).
                    if not antibot_force_low_confidence:
                        behavioral_kind = await _classify_behavioral(inst.page)
                        if behavioral_kind:
                            try:
                                page_url = inst.page.url or ""
                            except Exception:
                                page_url = ""
                            # Same three-step opt-in gate as the JS-challenge
                            # branch. ``px-press-hold`` is excluded by the
                            # ``_ANTIBOT_KINDS`` check (no published CapSolver
                            # task type) and stays on the operator-escalation
                            # path; ``datadome-behavioral`` opts in when
                            # CapSolver is the active provider.
                            if (
                                behavioral_kind in _ANTIBOT_KINDS
                                and self._captcha_solver is not None
                                and _solver_supports_kind(
                                    self._captcha_solver, behavioral_kind,
                                )
                            ):
                                logger.info(
                                    "Behavioral-only challenge detected (%s); "
                                    "active solver supports the kind, "
                                    "attempting solver path with forced "
                                    "low confidence",
                                    behavioral_kind,
                                )
                                kind = behavioral_kind
                                antibot_force_low_confidence = True
                                # Fall through to standard solver path.
                            else:
                                logger.info(
                                    "Behavioral-only challenge detected (%s); "
                                    "skipping solver, escalating to "
                                    "request_captcha_help",
                                    behavioral_kind,
                                )
                                # Audit-log so operators see the activity
                                # in the dashboard (per §2.7 cadence —
                                # aggregated, not per-call). URL flows
                                # through ``redact_url`` inside the
                                # recorder.
                                await _record_captcha_audit_event(
                                    inst.agent_id, "skipped_behavioral",
                                    behavioral_kind, page_url,
                                )
                                return _captcha_envelope(
                                    kind=behavioral_kind,
                                    solver_attempted=False,
                                    solver_outcome="skipped_behavioral",
                                    solver_confidence="behavioral-only",
                                    next_action="request_captcha_help",
                                )

                    # §11.3 — Cloudflare interstitial tri-state classifier.
                    # ``auto`` waits once and re-checks; ``behavioral``
                    # short-circuits to the behavioral envelope; ``turnstile``
                    # falls through to the existing solver path with a
                    # forced-``low`` confidence override; ``none`` keeps
                    # the existing legacy fallback (for CF iframes that
                    # don't expose any of the discriminating anchors).
                    cf_force_low_confidence = False
                    cf_state = await _classify_cf_state(inst.page)
                    if cf_state == "behavioral":
                        logger.info(
                            "CF interstitial classified as behavioral "
                            "(under-attack / persistent challenge); "
                            "skipping solver",
                        )
                        try:
                            page_url = inst.page.url or ""
                        except Exception:
                            page_url = ""
                        await _record_captcha_audit_event(
                            inst.agent_id, "skipped_behavioral",
                            "cf-interstitial-behavioral", page_url,
                        )
                        return _captcha_envelope(
                            kind="cf-interstitial-behavioral",
                            solver_attempted=False,
                            solver_outcome="skipped_behavioral",
                            solver_confidence="behavioral-only",
                            next_action="request_captcha_help",
                        )
                    if cf_state == "auto":
                        logger.info(
                            "CF auto-resolving challenge detected; "
                            "waiting %.1fs for it to clear",
                            _CF_AUTO_WAIT_SECONDS,
                        )
                        await asyncio.sleep(_CF_AUTO_WAIT_SECONDS)
                        # Re-run the selector match — if the page has
                        # navigated away from the challenge, no captcha
                        # selector matches. If still on the challenge,
                        # treat as behavioral (one wait+recheck cycle
                        # only — never loop).
                        still_present = False
                        for recheck_sel in captcha_selectors:
                            try:
                                if (
                                    await inst.page.locator(recheck_sel).count()
                                    > 0
                                ):
                                    still_present = True
                                    break
                            except Exception:
                                # Page closed / navigated mid-recheck —
                                # treat as cleared.
                                logger.debug(
                                    "CF auto recheck locator raised; "
                                    "treating as cleared",
                                    exc_info=True,
                                )
                                break
                        if not still_present:
                            return _captcha_envelope(
                                kind="cf-interstitial-auto",
                                solver_attempted=False,
                                solver_outcome="solved",
                                solver_confidence="medium",
                                next_action="solved",
                            )
                        logger.info(
                            "CF auto-resolving challenge still present "
                            "after wait; treating as behavioral",
                        )
                        try:
                            page_url = inst.page.url or ""
                        except Exception:
                            page_url = ""
                        await _record_captcha_audit_event(
                            inst.agent_id, "skipped_behavioral",
                            "cf-interstitial-behavioral", page_url,
                        )
                        return _captcha_envelope(
                            kind="cf-interstitial-behavioral",
                            solver_attempted=False,
                            solver_outcome="skipped_behavioral",
                            solver_confidence="behavioral-only",
                            next_action="request_captcha_help",
                        )
                    if cf_state == "turnstile":
                        # CF-bound Turnstile: existing Turnstile solver
                        # path runs, but the resulting confidence is
                        # forced to "low" because CF binds the token to
                        # session cookies and the solve may not unblock
                        # the session even when the verdict is good.
                        kind = "cf-interstitial-turnstile"
                        cf_force_low_confidence = True
                    # cf_state == "none" — fall through to existing
                    # standalone-Turnstile / generic flow unchanged.

                    # ── §11.18 — site-policy classification ──────────────
                    # Resolve the captcha policy for the live page URL once
                    # so we can route ``unsolvable`` hosts past the solver
                    # entirely (saves spend on sites the solver can't
                    # actually crack — CF Under-Attack, HUMAN Security,
                    # DataDome) and downgrade ``low_success`` hosts to a
                    # "try once at low confidence, escalate on failure"
                    # flow (Google / Twitter / LinkedIn auth where token-IP
                    # binding makes the solve unreliable).
                    #
                    # Hook position: AFTER §11.1 reCAPTCHA + §11.3 CF /
                    # behavioral classifiers (so ``kind`` is precise and
                    # ``cf_force_low_confidence`` is already known) but
                    # BEFORE the §11.16 health/breaker gates and the
                    # ``_metered_solve`` call — i.e. before any solver
                    # spend.  Operator overrides
                    # (``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS`` /
                    # ``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS``) are
                    # applied inside ``get_site_policy``.
                    try:
                        page_url_for_policy = inst.page.url or ""
                    except Exception:
                        page_url_for_policy = ""
                    site_policy = captcha_policy.get_site_policy(
                        page_url_for_policy,
                    )
                    # ``unsolvable`` short-circuits exactly like §11.3
                    # behavioral — same envelope shape, same audit
                    # outcome — but driven by host classification rather
                    # than live-page selectors.  The two paths are
                    # complementary: §11.3 catches behavioral-only
                    # challenges by their DOM signature; §11.18 catches
                    # them by the host even if our DOM heuristics miss
                    # (e.g. a CF Under-Attack page rendered before our
                    # JS probe gets a chance to run).
                    if site_policy == "unsolvable":
                        logger.info(
                            "Captcha host classified as unsolvable "
                            "(%s, kind=%s); skipping solver, escalating "
                            "to request_captcha_help",
                            redact_url(page_url_for_policy), kind,
                        )
                        await _record_captcha_audit_event(
                            inst.agent_id, "skipped_behavioral",
                            kind, page_url_for_policy,
                            policy="unsolvable",
                        )
                        return _captcha_envelope(
                            kind=kind,
                            solver_attempted=False,
                            solver_outcome="skipped_behavioral",
                            solver_confidence="behavioral-only",
                            next_action="request_captcha_help",
                        )

                    # ``low_success`` is NOT a short-circuit.  We continue
                    # to the solver but force ``solver_confidence="low"``
                    # on the result envelope and, on a non-``solved``
                    # outcome, upgrade ``next_action`` from ``notify_user``
                    # to ``request_captcha_help`` so the agent stops
                    # retrying on sites where retries don't help.
                    low_success = (site_policy == "low_success")

                    # Local helper applies the §11.3 CF-Turnstile
                    # confidence override (see ``cf_force_low_confidence``)
                    # AND the §11.18 ``low_success`` overrides to every
                    # return path through the solver block. Token
                    # validity is unrelated to the per-call solve verdict
                    # for CF-bound Turnstile, so the CF override fires
                    # regardless of ``solver_outcome``.  The ``low_success``
                    # override always forces confidence to "low"; on a
                    # FAILED solve outcome it ALSO upgrades
                    # ``next_action`` to ``request_captcha_help`` (the
                    # operator told us this host is unreliable; escalate
                    # rather than letting the agent retry).
                    async def _finalize(envelope: dict) -> dict:
                        # Async helper — used to be sync + fire-and-forget
                        # via ``loop.create_task`` for the low_success audit
                        # event, which silently dropped the event on
                        # shutdown (the task was never tracked or awaited).
                        # Now we ``await`` the audit recorder directly so
                        # the event lands in the per-minute aggregation
                        # bucket before the helper returns. The bucket
                        # itself is drained on the existing 60s metrics
                        # tick — no per-call EventBus emit, so this is
                        # cheap (a single dict update under an asyncio
                        # lock).
                        if cf_force_low_confidence:
                            envelope["solver_confidence"] = "low"
                        # §22 — anti-bot kinds (Akamai BMP / Imperva /
                        # Kasada / DataDome behavioral) are inherently
                        # low-confidence: the solver token can be
                        # rejected at the application layer for IP /
                        # fingerprint mismatches the solver has no
                        # visibility into. Force ``low`` confidence and,
                        # on a non-``solved`` outcome, upgrade
                        # ``next_action`` to ``request_captcha_help`` so
                        # the agent doesn't retry — anti-bot challenges
                        # have low success rates by design and retries
                        # rarely improve the result.
                        if antibot_force_low_confidence:
                            envelope["solver_confidence"] = "low"
                            outcome = envelope.get("solver_outcome")
                            if outcome and outcome != "solved":
                                envelope["next_action"] = "request_captcha_help"
                        if low_success:
                            envelope["solver_confidence"] = "low"
                            outcome = envelope.get("solver_outcome")
                            if outcome and outcome != "solved":
                                envelope["next_action"] = "request_captcha_help"
                                envelope["low_success_failed"] = True
                                # Audit: surface the upgrade so operators see
                                # "low-success-attempted-and-failed" distinctly
                                # from a vanilla provider rejection.
                                await _record_captcha_audit_event(
                                    inst.agent_id,
                                    "low_success_failed",
                                    envelope.get("kind", "unknown"),
                                    page_url_for_policy,
                                    policy="low_success",
                                )
                        # §22 — fingerprint burn override.  Once the
                        # rolling rejection rate crosses the burn
                        # threshold every subsequent ``_check_captcha``
                        # envelope carries ``fingerprint_burn=True`` and
                        # routes the agent to ``retry_with_fresh_profile``.
                        # Auto-rotation is intentionally NOT done here —
                        # rotating a profile mid-flight would wipe
                        # session cookies / login state and cascade into
                        # other agents' work.  The operator clears the
                        # burn manually via the dashboard reset endpoint
                        # after rotating the profile.  Skips
                        # ``solver_outcome=="solved"`` envelopes so a
                        # genuine solve still surfaces ``next_action="solved"``
                        # — the override applies to the FOLLOWING
                        # captcha encounter, not the one we're currently
                        # finalising (which we still let through to
                        # produce a real signal for the monitor).  When
                        # ``solver_attempted=False`` (no_solver / breaker /
                        # cost_cap escalations) we still mark the
                        # envelope so the agent loop sees the burn state
                        # before it requests operator help.
                        if await _is_fingerprint_burned(inst.agent_id):
                            envelope["fingerprint_burn"] = True
                            outcome = envelope.get("solver_outcome")
                            if outcome != "solved":
                                envelope["next_action"] = (
                                    "retry_with_fresh_profile"
                                )
                        # §22 — schedule the post-solve monitor when the
                        # solver actually injected a token.  Fire-and-forget
                        # so the agent's action response is not blocked on
                        # the 10s observation window.  Skipped escalations
                        # (``skipped_behavioral`` / ``request_captcha_help``)
                        # are NOT monitored — no token was injected, so
                        # the page state has nothing to tell us about the
                        # fingerprint.  Failed-solve outcomes (rejected /
                        # injection_failed / timeout) are also not
                        # monitored — the page never received a valid
                        # token, so a still-present captcha is expected
                        # rather than a fingerprint signal.
                        if (
                            envelope.get("solver_outcome") == "solved"
                            and envelope.get("solver_attempted")
                        ):
                            self._spawn_fingerprint_monitor(
                                inst, captcha_selectors,
                                envelope.get("kind", "unknown"),
                            )
                        return envelope

                    if self._captcha_solver:
                        # §11.16 short-circuits — check solver health
                        # BEFORE attempting a solve. ``is_solver_unreachable``
                        # is async + lazy: when the per-process health
                        # probe hasn't fired yet, the read drives it
                        # under the solver's state-lock so the FIRST
                        # captcha of a session sees the gate fire (the
                        # prior synchronous read happened before the
                        # probe could run inside ``solve()``).
                        if await self._captcha_solver.is_solver_unreachable():
                            logger.warning(
                                "CAPTCHA detected (%s), solver marked unreachable; skipping",
                                sel,
                            )
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="no_solver",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))
                        if self._captcha_solver.is_breaker_open():
                            logger.warning(
                                "CAPTCHA detected (%s), solver breaker open; skipping",
                                sel,
                            )
                            envelope = _captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="timeout",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            )
                            # Additive top-level flag — deliberately not a
                            # new ``solver_outcome`` enum value; preserves
                            # §11.13 contract while letting callers detect
                            # the breaker-open case distinctly.
                            envelope["breaker_open"] = True
                            return await _finalize(envelope)
                        logger.info("CAPTCHA detected (%s), attempting auto-solve", sel)
                        # Single entry point for solver invocation. The
                        # rate-limit + cost-cap gates fire inside
                        # ``_metered_solve`` BEFORE the solver HTTP call,
                        # and accounting fires on token retrieval (NOT on
                        # injection success — provider was paid already).
                        try:
                            result = await self._metered_solve(inst, sel, kind)
                        except asyncio.TimeoutError:
                            # Solver took too long — true "timeout" semantic.
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=True,
                                solver_outcome="timeout",
                                solver_confidence="low",
                                next_action="notify_user",
                            ))
                        except Exception as exc:
                            # Network / JSON-decode / programmer errors all
                            # land here. A third-party CaptchaSolver subclass
                            # could let httpx errors bubble up; the bundled
                            # ``CaptchaSolver`` (src/browser/captcha.py)
                            # converts them into a no-token :class:`SolveResult`
                            # internally. We treat httpx timeouts as
                            # "timeout" and other exceptions as "rejected" —
                            # closest fit in the §11.13 enum.
                            if _is_httpx_timeout(exc):
                                # Redact ``repr(exc)`` — httpx error strings
                                # can include the request URL which carries
                                # the ``clientKey``. Pair with
                                # :func:`redact_url` like the bundled solver
                                # already does.
                                logger.warning(
                                    "Auto-solve timed out (httpx): %s",
                                    _redact_clientkey_text(redact_url(repr(exc))),
                                )
                                return await _finalize(_captcha_envelope(
                                    kind=kind, solver_attempted=True,
                                    solver_outcome="timeout",
                                    solver_confidence="low",
                                    next_action="notify_user",
                                ))
                            # ``logger.exception`` includes the traceback
                            # which can carry URL-shaped or clientKey-laden
                            # strings. Use ``logger.warning`` with a
                            # redacted ``repr(exc)`` instead.
                            logger.warning(
                                "Auto-solve raised, falling back: %s",
                                _redact_clientkey_text(redact_url(repr(exc))),
                                exc_info=False,
                            )
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=True,
                                solver_outcome="rejected",
                                solver_confidence="low",
                                next_action="notify_user",
                            ))

                        # ── Map :class:`SolveResult` to §11.13 envelope ──
                        # Gate-skipped paths return without ``solver_attempted``
                        # because no provider HTTP call ran.
                        if result.skipped == "disabled":
                            # ``CAPTCHA_DISABLED`` kill switch — auto-detect
                            # and explicit ``solve_captcha`` both surface this
                            # as the existing ``no_solver`` envelope so callers
                            # don't have to learn a new outcome value.
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="no_solver",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))
                        if result.skipped == "provider_missing":
                            # Provider sanity check (cap-on case) — same shape
                            # as ``no_solver`` but a distinct audit event was
                            # already emitted inside ``_metered_solve`` so
                            # operators can see the misconfiguration.
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="no_solver",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))
                        if result.skipped == "price_missing":
                            # Published-rate sanity check (cap-on case).
                            # Same agent-facing shape as no_solver; the
                            # audit outcome distinguishes it for operators.
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="no_solver",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))
                        if result.skipped == "rate_limited":
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="rate_limited",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))
                        if result.skipped == "cost_cap":
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=False,
                                solver_outcome="cost_cap",
                                solver_confidence=_kind_confidence(kind),
                                next_action="request_captcha_help",
                            ))

                        if result.token is None:
                            # No token retrieved — provider was not
                            # charged. Conflates (a) provider verdict
                            # reject / errorId>0, (b) sitekey not
                            # extractable, (c) outer-deadline timeout
                            # captured inside ``solve()``. Map all to
                            # ``rejected`` (closest §11.13 fit).
                            logger.warning("Auto-solve failed, falling back to manual")
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=True,
                                solver_outcome="rejected",
                                solver_confidence="low",
                                next_action="notify_user",
                            ))

                        # Token retrieved. Provider was charged. Cost was
                        # already incremented inside ``_metered_solve``.
                        if not result.injection_succeeded:
                            # §11.13 ``injection_failed`` outcome — the
                            # provider returned a valid token but our
                            # ``_inject_token`` rejected it. The bool
                            # surface from ``_inject_token`` doesn't tell
                            # us which case it was (CSP block, textarea
                            # not found, callback not in config, widget
                            # not found) — that's a §11.6/§11.20 deferred
                            # item; surface ``injection_failed_unspecified``
                            # for now so operators see the case distinctly.
                            return await _finalize(_captcha_envelope(
                                kind=kind, solver_attempted=True,
                                solver_outcome="injection_failed",
                                solver_confidence="low",
                                next_action="notify_user",
                                injection_failure_reason=(
                                    "injection_failed_unspecified"
                                ),
                            ))

                        # §11.2 — compat-rejected proxy fallback downgrades
                        # confidence to "low" even on a successful solve;
                        # the operator's chosen IP profile didn't reach
                        # the provider so the token came from the provider's
                        # pool, not the operator's. ``_finalize`` further
                        # applies the §11.3 CF-Turnstile force-low.
                        confidence = "low" if result.compat_rejected else "high"
                        return await _finalize(_captcha_envelope(
                            kind=kind, solver_attempted=True,
                            solver_outcome="solved",
                            solver_confidence=confidence,
                            next_action="solved",
                        ))
                    # No solver configured — surface to agent for manual VNC.
                    # Confidence reflects how firmly we classified the kind:
                    # firmly-disambiguated kinds (hcaptcha, turnstile) →
                    # "high"; placeholders (recaptcha-v2-checkbox,
                    # cf-interstitial-auto) and "unknown" → "low" until
                    # §11.1 / §11.3 land variant detection.
                    return await _finalize(_captcha_envelope(
                        kind=kind, solver_attempted=False,
                        solver_outcome="no_solver",
                        solver_confidence=_kind_confidence(kind),
                        next_action="notify_user",
                    ))
        except Exception:
            logger.debug("captcha detection raised", exc_info=True)
        return {"captcha_found": False}

    def _classify_kind(self, selector: str) -> str:
        """Map the matched selector onto the coarse §11.13 ``kind`` value.

        First-pass classification only — the live-page classifiers in
        ``_check_captcha`` (§11.1 ``_classify_recaptcha``, §11.3
        ``_classify_cf_state`` / ``_classify_behavioral``) refine these
        placeholders against the actual page state.
        """
        if "recaptcha" in selector:
            # v2-checkbox is the safe default; the §11.1 classifier
            # upgrades to v2-invisible / v3 / Enterprise variants when
            # the live page exposes the disambiguating signals.
            return "recaptcha-v2-checkbox"
        if "hcaptcha" in selector:
            return "hcaptcha"
        if "cf-turnstile" in selector:
            return "turnstile"
        if "challenges.cloudflare.com" in selector:
            # Coarse CF default. The §11.3 classifier in ``_check_captcha``
            # routes to ``cf-interstitial-{auto,behavioral,turnstile}``
            # based on live-page anchors (challenge frame, error 1020,
            # embedded Turnstile widget). When the JS probe finds none
            # of those, this placeholder stays.
            return "cf-interstitial-auto"
        # Generic 'captcha' selectors and #captcha fall through to unknown.
        return "unknown"

    async def detect_captcha(self, agent_id: str) -> dict:
        """Detect CAPTCHAs on the current page.

        Returns the §11.13 envelope. The legacy ``type`` and ``message``
        fields are populated for backward compatibility with rules that
        condition on the old shape; new agents should read the structured
        fields directly.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            envelope = await self._check_captcha(inst)
            return {
                "success": True,
                "data": _with_legacy_fields(envelope),
            }

    # ── §11.14 explicit-trigger captcha skills ────────────────────────────

    async def solve_captcha(
        self,
        agent_id: str,
        *,
        hint: str | None = None,
        retry_previous: bool = False,
        target_ref: str | None = None,
    ) -> dict:
        """Agent-triggered CAPTCHA solve.

        Layered around :meth:`_check_captcha` so we get §11.13 envelope
        semantics, §11.16 health/breaker short-circuits, AND the
        :meth:`_metered_solve` rate-limit + cost-cap gates for free —
        the pre-refactor implementation duplicated these gates here, but
        they fired AFTER ``_check_captcha`` had already invoked the
        solver, defeating the purpose. All gates now sit inside
        ``_metered_solve`` and apply uniformly across navigate auto-detect,
        click auto-detect, and explicit ``solve_captcha`` calls.

        Local pre-checks (cheap, no provider involvement):
          1. ``hint`` validation — bad hint → ``invalid_input`` error.
             Behavioral-only kinds (PerimeterX, DataDome, CF interstitial)
             are also rejected here because the solver has no task entry
             for them; the correct path is ``request_captcha_help``.
          2. No-captcha early return — saves a solver invocation + cost.
             When ``retry_previous=True`` and the initial detection finds
             nothing, we wait :data:`_RETRY_PREVIOUS_RECHECK_MS` and
             re-check ONCE; this covers the "page just navigated, captcha
             is rendering" race.
          3. Recursive-solve guard — re-entrant ``_check_captcha`` while a
             solve is in flight surfaces ``captcha_during_solve``.

        ``CAPTCHA_DISABLED`` kill switch + rate-limit + cost-cap all fire
        INSIDE ``_check_captcha`` → ``_metered_solve`` BEFORE any solver
        HTTP call. The duplicate kill-switch check that used to live at
        the top of this method was removed once the metered-solve path
        landed it — keeping a second copy invited drift.

        ``target_ref`` accepts a snapshot ref for selecting one captcha among
        many on the same page; multi-captcha enumeration lands in §11.6.
        Until then, we LOG and IGNORE — the top-ranked visible widget gets
        solved. Documented in the skill description so agents don't expect
        targeting precision yet.

        ``retry_previous`` semantics (review polish): the original
        "retry the most recent failed solve attempt against (sitekey, url)"
        spec required tracking solver internals across calls and racing
        with the auto-classifier on whether the displayed captcha is the
        same one. The new, simpler semantic is: **"be patient — the
        captcha may not have rendered yet"**. If the initial detection
        returns ``captcha_found=False`` we sleep briefly and re-detect
        once. Bounded by :data:`_RETRY_PREVIOUS_RECHECK_MS` so total
        latency stays predictable.
        """
        # Gate 1: hint validation (cheap, before lock). The fleet-wide
        # ``CAPTCHA_DISABLED`` kill switch fires inside ``_metered_solve``
        # so navigate / click auto-detect paths are covered too — keeping
        # a duplicate here invited drift between entry points.
        if hint is not None:
            if not isinstance(hint, str):
                return _err(
                    "invalid_input",
                    f"hint must be one of: {sorted(_VALID_CAPTCHA_KINDS)}",
                )
            hint = hint.strip()
            # Behavioral-only kinds are valid classifier outputs but
            # have no solver task entry — treating them as a hint would
            # be a silent no-op on the solver path. Reject loudly so the
            # agent sees the correct routing immediately.
            if hint in _BEHAVIORAL_KINDS:
                return _err(
                    "invalid_input",
                    f"hint={hint!r} is behavioral-only — use request_captcha_help",
                )
            if hint not in _VALID_CAPTCHA_KINDS:
                return _err(
                    "invalid_input",
                    f"hint must be one of: {sorted(_VALID_CAPTCHA_KINDS)}",
                )

        if target_ref:
            logger.warning(
                "solve_captcha: target_ref=%r ignored — multi-captcha "
                "enumeration is §11.6 (deferred). Top-ranked visible widget "
                "will be solved.", target_ref,
            )

        inst = await self.get_or_start(agent_id)
        inst.touch()

        async with inst.lock:
            # Gate 3: no-captcha early return. Set the recursive-solve
            # guard BEFORE the call so a captcha that appears during
            # detection isn't mistaken for one we should now solve.
            #
            # NOTE on ``_captcha_solving``: under the current architecture
            # ``inst.lock`` already serializes all ``solve_captcha`` calls
            # per agent — concurrent reentry is impossible. The guard is
            # kept as defensive code: never observed True in practice;
            # future-proofing against design changes that release the
            # lock around the solver call. Removing it is safe today.
            if inst._captcha_solving:
                return {
                    "success": True,
                    "data": _with_legacy_fields(_captcha_envelope(
                        kind="unknown", solver_attempted=False,
                        solver_outcome="captcha_during_solve",
                        solver_confidence="low",
                        next_action="request_captcha_help",
                    )),
                }

            inst._captcha_solving = True
            try:
                # ``_check_captcha`` runs the full pipeline:
                # detection → classifiers → §11.16 health/breaker gates
                # → ``_metered_solve`` (rate-limit + cost-cap +
                # provider call + cost increment on token retrieval).
                # Cost / rate-limit short-circuits surface as
                # ``solver_outcome="cost_cap"`` / ``"rate_limited"``
                # in the returned envelope — no extra gate logic needed
                # here. Cost accounting is handled inside
                # ``_metered_solve`` so this method no longer touches
                # the cost counter directly.
                envelope = await self._check_captcha(inst)

                # ``retry_previous`` patience window: when the agent
                # signals "the captcha may not have rendered yet", give
                # the page a brief moment and re-check ONCE before
                # giving up. Bounded by ``_RETRY_PREVIOUS_RECHECK_MS``.
                if (
                    retry_previous
                    and not envelope.get("captcha_found")
                ):
                    await asyncio.sleep(_RETRY_PREVIOUS_RECHECK_MS / 1000.0)
                    envelope = await self._check_captcha(inst)

                if not envelope.get("captcha_found"):
                    # No-captcha early return — distinct from the §11.13
                    # ``captcha_found: false`` shape; surface a friendly
                    # top-level message for the agent.
                    return {
                        "success": True,
                        "data": {
                            "captcha_found": False,
                            "message": "No captcha on current page",
                        },
                    }

                kind = envelope.get("kind", "unknown")
                if hint is not None and hint != kind:
                    # Hint overrides the auto-classified kind in the
                    # envelope so downstream consumers see what the
                    # agent declared. We don't re-run the solver with
                    # a different task type here — that's §11.1's
                    # variant-aware solver work; the hint at this layer
                    # is purely classification metadata.
                    envelope = dict(envelope)
                    envelope["kind"] = hint

                return {
                    "success": True,
                    "data": _with_legacy_fields(envelope),
                }
            finally:
                inst._captcha_solving = False

    async def request_captcha_help(
        self, agent_id: str, *, service: str, description: str,
    ) -> dict:
        """Agent-triggered request for human help on a CAPTCHA.

        Mirrors the ``request_browser_login`` semantics: the BrowserManager
        emits the dashboard handoff event and returns immediately. The
        operator-completion path (``/api/browser-captcha-help/complete``
        in the dashboard) enqueues a steer message to the agent — there
        is no background blocking task in the BrowserManager itself.

        Used for behavioral-only challenges (CF Under Attack, Press &
        Hold), persistent rejections, or when ``solve_captcha`` returned
        ``cost_cap`` / ``rate_limited``.
        """
        if not service:
            return _err("invalid_input", "service is required")
        if not description:
            return _err("invalid_input", "description is required")

        inst = await self.get_or_start(agent_id)
        inst.touch()
        # Acquire the lock so we present a stable page URL to the dashboard
        # event and so a concurrent solve doesn't redirect the page mid-handoff.
        async with inst.lock:
            try:
                page_url = inst.page.url or ""
            except Exception:
                page_url = ""
            return {
                "success": True,
                "data": {
                    "requested": True,
                    "service": service[:128],
                    "description": description[:500],
                    "url": page_url[:2048],
                },
            }

    # ── File transfer (Phase 1.5 infrastructure) ─────────────────────────

    async def upload_file(
        self, agent_id: str, ref: str, local_paths: list[str],
        *, timeout_ms: int = 10000,
    ) -> dict:
        """Drive a native file-chooser via Playwright on behalf of the agent.

        The ``local_paths`` list points at files inside the browser container
        that the mesh staged for us. Caller is responsible for writing those
        bytes to disk BEFORE invoking this method; we just pass them to
        ``page.expect_file_chooser`` → ``chooser.set_files``.

        Playwright's ``expect_file_chooser`` is a context manager that
        resolves when the page triggers a chooser. The chooser fires in
        response to a click on an ``<input type="file">`` (or equivalent
        aria-labelled element); we handle that click here as part of the
        contract so agents don't need to coordinate the race.

        Returns ``{success, data: {uploaded: [path, …]}}`` or an error envelope.
        """
        # Defense in depth: confine ``local_paths`` to the configured
        # upload-receive directory BEFORE entering the body of this
        # method. The mesh-side ``upload_apply`` only forwards paths
        # returned by ``_stage_upload`` (which writes under
        # ``OPENLEGION_UPLOAD_RECV_DIR``), but a buggy or compromised
        # mesh caller carrying the bearer token could otherwise smuggle
        # arbitrary on-disk paths here — both as a read-via-chooser
        # exfil vector AND as an unconditional unlink of any reachable
        # file via the cleanup ``finally`` at the end of this method.
        # Validating BEFORE the try/finally ensures a rejected path is
        # never reached by the cleanup unlink.
        from src.browser.flags import get_str as _flag_str
        recv_dir = Path(
            _flag_str("OPENLEGION_UPLOAD_RECV_DIR", "/tmp/upload-recv"),
        ).resolve()
        try:
            recv_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        # Resolve every input path canonically and remember the result.
        # We pass the RESOLVED paths to ``chooser.set_files`` later —
        # not the originals — so that an attacker controlling the
        # filesystem under ``recv_dir`` cannot swap a symlink target
        # between validation and upload (TOCTOU). When a path resolves
        # to a target outside ``recv_dir`` we reject up front.
        resolved_paths: list[str] = []
        for p in local_paths:
            try:
                resolved = Path(p).resolve()
            except (OSError, RuntimeError) as e:
                return {
                    "success": False,
                    "error": f"Upload path not resolvable: {e}",
                }
            try:
                resolved.relative_to(recv_dir)
            except ValueError:
                return {
                    "success": False,
                    "error": "Upload path outside receive dir",
                }
            resolved_paths.append(str(resolved))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused",
                    }
                for p in resolved_paths:
                    if not Path(p).is_file():
                        return {
                            "success": False,
                            "error": f"Upload path not found: {p}",
                        }
                locator = await self._locator_from_ref(inst, ref)
                if not locator:
                    return {"success": False, "error": f"Ref '{ref}' not found"}
                async with inst.page.expect_file_chooser(timeout=timeout_ms) as info:
                    if isinstance(inst.x11_wid, int) and self._is_x11_site(inst):
                        try:
                            await self._x11_click(
                                inst, locator, timeout=timeout_ms,
                            )
                        except Exception as e:
                            logger.warning(
                                "X11 upload click failed for '%s', "
                                "falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click(
                                inst.page, locator, timeout=timeout_ms,
                            )
                    else:
                        await self._human_click(
                            inst.page, locator, timeout=timeout_ms,
                        )
                chooser = await info.value
                await chooser.set_files(resolved_paths)
                await asyncio.sleep(action_delay())
                return {
                    "success": True,
                    "data": {"uploaded": list(resolved_paths)},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
            finally:
                for p in resolved_paths:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except Exception:
                        logger.debug("Stage cleanup failed for %s", p)

    async def download(
        self, agent_id: str, ref: str,
        *,
        download_dir: str | None = None,
        timeout_ms: int = 30000,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> dict:
        """Click ``ref`` and capture the resulting download to disk.

        Reads the download chunk-by-chunk from Playwright's underlying
        artifact stream. A running byte counter aborts the transfer if
        ``max_bytes`` is exceeded — bytes never accumulate past the cap
        on disk. Refuses with ``service_unavailable`` when the private
        artifact-stream API is missing rather than silently degrading
        to a racy drain-then-check fallback.
        """
        if download_dir is None:
            from src.browser.flags import get_str as _flag_str
            download_dir = _flag_str("BROWSER_DOWNLOAD_DIR", "/tmp/downloads")
        if not self._download_streaming_available:
            return {
                "success": False,
                "error": {
                    "code": "service_unavailable",
                    "message": (
                        "Download streaming unavailable: bounded size cap "
                        "requires Playwright's private artifact stream API"
                    ),
                },
            }
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused",
                    }
                locator = await self._locator_from_ref(inst, ref)
                if not locator:
                    return {"success": False, "error": f"Ref '{ref}' not found"}

                Path(download_dir).mkdir(parents=True, exist_ok=True)
                async with inst.page.expect_download(timeout=timeout_ms) as info:
                    if isinstance(inst.x11_wid, int) and self._is_x11_site(inst):
                        try:
                            await self._x11_click(
                                inst, locator, timeout=timeout_ms,
                            )
                        except Exception as e:
                            logger.warning(
                                "X11 download click failed for '%s', "
                                "falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click(
                                inst.page, locator, timeout=timeout_ms,
                            )
                    else:
                        await self._human_click(
                            inst.page, locator, timeout=timeout_ms,
                        )
                download = await info.value
                # Sanitize before joining — Content-Disposition is
                # attacker-controlled and a ``../../`` escape would
                # write outside ``download_dir`` (which is writable in
                # the browser container). The leading nonce neutralizes
                # leading-dot and absolute-path tricks but path
                # separators inside ``suggested`` would still escape.
                suggested = _sanitize_download_filename(
                    download.suggested_filename or "download.bin",
                )
                nonce = uuid.uuid4().hex[:12]
                dest = Path(download_dir) / f"{nonce}-{suggested}"

                size = await self._stream_download_to_disk(
                    download, dest, max_bytes,
                )
                if size is None:
                    return {
                        "success": False,
                        "error": f"Download exceeds {max_bytes} bytes",
                    }

                mime = mimetypes.guess_type(suggested)[0] or "application/octet-stream"
                return {
                    "success": True,
                    "data": {
                        "path": str(dest),
                        "nonce": nonce,
                        "size_bytes": size,
                        "suggested_filename": suggested,
                        "mime_type": mime,
                    },
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _stream_download_to_disk(
        self, download, dest: Path, max_bytes: int,
    ) -> int | None:
        """Drain a Playwright Download into ``dest`` in bounded chunks.

        Returns the byte size on success, ``None`` if the cap was exceeded
        (the partial file is unlinked and the artifact is cancelled before
        Playwright fully buffers it). Raises on transport errors so the
        caller's outer try/except returns the original message.

        Uses Playwright's private ``_artifact`` channel because the public
        Download API only exposes ``save_as()`` / ``path()``, both of which
        wait for the full transfer to finish before returning. Without
        chunked enforcement an attacker-controlled download could fill the
        container's ``/tmp`` before the post-transfer size check fires.

        Caller must have already verified ``_download_streaming_available``;
        this method has no fallback. Missing private channel raises so the
        download is aborted rather than silently degrading.
        """
        import base64

        artifact = getattr(download, "_artifact", None)
        channel = getattr(artifact, "_channel", None) if artifact else None
        if channel is None:
            with contextlib.suppress(Exception):
                await download.cancel()
            raise RuntimeError(
                "Playwright artifact channel unavailable",
            )
        stream_channel = await channel.send("saveAsStream", None)
        from playwright._impl._connection import from_channel
        stream = from_channel(stream_channel)

        total = 0
        chunk_size = 64 * 1024
        try:
            with dest.open("wb") as out:
                while True:
                    binary = await stream._channel.send(
                        "read", None, {"size": chunk_size},
                    )
                    if not binary:
                        break
                    chunk = base64.b64decode(binary)
                    if total + len(chunk) > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        with contextlib.suppress(Exception):
                            await download.cancel()
                        return None
                    out.write(chunk)
                    total += len(chunk)
        except Exception:
            with contextlib.suppress(Exception):
                dest.unlink(missing_ok=True)
            raise
        return total

    async def find_text(
        self, agent_id: str, query: str, scroll: bool = True,
    ) -> dict:
        """Find elements whose accessible name contains ``query`` (case-folded).

        Performs a fresh snapshot to populate ``inst.refs``, then scans every
        ref's accessible name with :meth:`str.casefold` for Unicode-aware
        case-insensitive substring matching. Returns up to 50 matches in
        snapshot order. When ``scroll`` is True and matches exist, the first
        match is scrolled into view (best-effort, non-fatal).
        """
        if not isinstance(query, str) or not (1 <= len(query) <= 500):
            return _err(
                "invalid_input",
                "query must be a non-empty string up to 500 chars",
            )
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            if inst._user_control:
                return _err(
                    "conflict",
                    "User has browser control — action paused until control is released.",
                )
            return await self._find_text_impl(
                inst, agent_id, query, scroll=scroll,
            )

    async def _find_text_impl(
        self, inst: "CamoufoxInstance", agent_id: str, query: str,
        *, scroll: bool = True,
    ) -> dict:
        """Lock-free body of :meth:`find_text`.

        Caller must hold ``inst.lock`` and have already cleared the
        ``inst._user_control`` gate. Extracted so compound primitives
        (e.g. §9.4 ``fill_form``) can reuse the matching pipeline
        without dropping and re-acquiring the per-instance lock —
        ``asyncio.Lock`` is not reentrant, so calling :meth:`find_text`
        directly from inside a held-lock section would deadlock.
        """
        try:
            snap = await self._snapshot_impl(
                inst, agent_id, _skip_baseline=True,
            )
            if not snap.get("success"):
                return snap

            needle = query.casefold()
            viewport = inst.page.viewport_size or {}
            vw = int(viewport.get("width") or 0)
            vh = int(viewport.get("height") or 0)

            matches: list[dict] = []
            truncated = False
            first_locator = None
            for ref_id, handle in inst.refs.items():
                name = handle.name or ""
                # Match against the REDACTED accessible name so an
                # agent-supplied query can't probe credentials that
                # leaked into ``aria-label`` / tooltip text. Snapshots
                # already redact what they emit; ``find_text`` must
                # honor the same boundary or it becomes a confirm/deny
                # oracle for known secret prefixes.
                redacted_name = self.redactor.redact(agent_id, name)
                if not name or needle not in redacted_name.casefold():
                    continue
                if len(matches) >= 50:
                    truncated = True
                    break
                locator = await self._locator_from_ref(inst, ref_id)
                in_viewport = False
                if locator is not None:
                    try:
                        visible = await locator.is_visible()
                    except Exception:
                        visible = False
                    if visible and vw > 0 and vh > 0:
                        try:
                            box = await locator.bounding_box()
                        except Exception:
                            box = None
                        if box:
                            bx = float(box.get("x", 0))
                            by = float(box.get("y", 0))
                            bw = float(box.get("width", 0))
                            bh = float(box.get("height", 0))
                            in_viewport = (
                                bx + bw > 0 and by + bh > 0
                                and bx < vw and by < vh
                            )
                matches.append({
                    "ref": ref_id,
                    "text": sanitize_for_prompt(redacted_name)[:200],
                    "in_viewport": bool(in_viewport),
                })
                if first_locator is None and locator is not None:
                    first_locator = locator

            if scroll and first_locator is not None:
                try:
                    await first_locator.scroll_into_view_if_needed(timeout=3000)
                except Exception as e:
                    logger.debug(
                        "scroll_into_view_if_needed failed for %s: %s",
                        agent_id, e,
                    )

            return {
                "success": True,
                "data": {
                    "matches": matches,
                    "total": len(matches),
                    "truncated": truncated,
                },
            }
        except Exception as e:
            logger.debug("find_text failed for %s: %s", agent_id, e)
            return _err("service_unavailable", "find_text failed")

    # ── §9.4 fill_form ───────────────────────────────────────────────────

    # Per-field max value length. Long enough that legitimate form values
    # (textarea bodies, etc.) fit; short enough that an unbounded
    # prompt-injected value is rejected with a clean ``invalid_input``
    # before reaching Playwright's ``fill`` (which has no documented cap).
    _FILL_FORM_MAX_VALUE_CHARS = 10000
    _FILL_FORM_MAX_LABEL_CHARS = 500
    _FILL_FORM_MAX_FIELDS = 50
    _FILL_FORM_PREFERRED_ROLES = frozenset({"textbox", "searchbox", "spinbutton"})

    @staticmethod
    def _classify_fill_error(exc: Exception) -> str:
        """Map a Playwright ``locator.fill`` exception → structured reason.

        Third-pass review §9.4 concern 1: ``find_text`` can return refs
        for non-input elements (a ``<label>``, ``<span>`` containing the
        label text, etc.). Calling ``fill()`` on those raises a Playwright
        error like ``"Element is not an <input>, <textarea> or
        [contenteditable]"``. Without classification the agent sees a
        generic ``type_failed`` and can't plan a recovery — with a
        ``not_fillable`` code it knows to fall back to
        ``browser_get_elements`` to find the underlying input near the
        label.

        Returns one of: ``not_fillable``, ``timeout``, ``detached``,
        ``hidden``, ``disabled``, ``other``.
        """
        msg = str(exc).lower()
        # Order matters: TimeoutError is a subclass check; the string
        # heuristics below cover both Playwright's ``Error`` class and
        # plain Python exceptions.
        if isinstance(exc, TimeoutError) or "timeout" in msg:
            return "timeout"
        if (
            "is not an <input>" in msg
            or "is not an <textarea>" in msg
            or "not an editable" in msg
            or "not editable" in msg
            or "not fillable" in msg
            or "[contenteditable]" in msg
        ):
            return "not_fillable"
        if "detached" in msg or "not attached" in msg:
            return "detached"
        if "not visible" in msg or "hidden" in msg:
            return "hidden"
        if "disabled" in msg or "readonly" in msg or "read-only" in msg:
            return "disabled"
        return "other"

    async def fill_form(
        self, agent_id: str, fields: list[dict],
        *, submit_after: bool = False,
    ) -> dict:
        """Sequence of find-text + fill across multiple form fields (§9.4).

        For each field, locates the input by visible label (reusing
        :meth:`_find_text_impl`), resolves the ref to a locator
        (:meth:`_locator_from_ref`), and calls Playwright's ``fill``.

        We use ``fill`` rather than the per-keystroke ``type_text``
        because (a) ``fill`` clears existing content first, making
        idempotent retries safe (no double-prefix on resume after
        CAPTCHA); (b) per-key humanization across many fields is
        excessive when the agent has explicitly chosen the compound
        path. Agents needing per-key entropy on a sensitive field
        (rare; mostly password fields with bot detection) should fall
        back to ``browser_type`` field-by-field.

        On CAPTCHA detection mid-flow (after any successful fill) the
        loop stops, the remaining un-attempted fields are echoed back
        verbatim in ``remaining`` so the agent can resume after
        solving, and we do NOT submit even if ``submit_after=True``.
        See §9.4 for the partial-success protocol.
        """
        if not isinstance(fields, list) or not fields:
            return _err(
                "invalid_input",
                "fields must be a non-empty list of {label, value} entries",
            )
        if len(fields) > self._FILL_FORM_MAX_FIELDS:
            return _err(
                "invalid_input",
                f"fields list exceeds max length {self._FILL_FORM_MAX_FIELDS}",
            )

        # Validate each field upfront so we don't half-fill the form before
        # discovering an invalid entry deep in the list.
        normalized: list[dict] = []
        for idx, raw in enumerate(fields):
            if not isinstance(raw, dict):
                return _err(
                    "invalid_input",
                    f"fields[{idx}] must be an object",
                )
            label = raw.get("label")
            value = raw.get("value")
            if not isinstance(label, str) or not (
                1 <= len(label) <= self._FILL_FORM_MAX_LABEL_CHARS
            ):
                return _err(
                    "invalid_input",
                    f"fields[{idx}].label must be a string 1–"
                    f"{self._FILL_FORM_MAX_LABEL_CHARS} chars",
                )
            if (
                not isinstance(value, str)
                or len(value) > self._FILL_FORM_MAX_VALUE_CHARS
            ):
                return _err(
                    "invalid_input",
                    f"fields[{idx}].value must be a string up to "
                    f"{self._FILL_FORM_MAX_VALUE_CHARS} chars",
                )
            # NUL bytes confuse Playwright's keyboard pipeline; reject early
            # rather than letting fill() raise a less-clear error mid-flow.
            if "\x00" in value:
                return _err(
                    "invalid_input",
                    f"fields[{idx}].value contains null byte",
                )
            field_submit = raw.get("submit_after", False)
            normalized.append({
                "label": label,
                "value": value,
                "submit_after": bool(field_submit),
            })

        inst = await self.get_or_start(agent_id)
        inst.touch()

        # One lock for the entire form fill — per §2.4, ``inst.lock`` already
        # serializes per-instance page ops; acquiring it once gives us a
        # consistent view of the page across the find→fill→captcha-check
        # cycle for every field. Per-field lock acquisition would let
        # another action interleave between find_text and fill, breaking
        # the ref freshness guarantee.
        async with inst.lock:
            async def _fill_body() -> dict:
                if inst._user_control:
                    return _err(
                        "conflict",
                        "User has browser control — action paused until control is released.",
                    )

                filled: list[dict] = []
                last_locator = None
                # Track whether ANY Enter press has succeeded — per-field
                # submit_after immediately before the captcha check is the
                # most common trigger for a mid-flow captcha (submission is
                # often what gates the challenge). When that happens, the
                # form may have ALREADY been submitted with partial data, so
                # the captcha-envelope must report ``submitted=True`` rather
                # than the agent thinking it can safely resume by re-typing.
                submitted = False
                for i, field in enumerate(normalized):
                    label = field["label"]
                    value = field["value"]
                    field_submit = field["submit_after"]

                    # 1) Locate the field via find_text. scroll=True so the
                    #    matched input is brought into view before the fill;
                    #    avoids "element not in viewport" failures on long
                    #    forms.
                    find_res = await self._find_text_impl(
                        inst, agent_id, label, scroll=True,
                    )
                    # Output-side label hygiene: defensively redact the label
                    # echoed back in ``filled[]``. Mirrors :meth:`_find_text_impl`
                    # line ~5261, which redacts the matched ``text`` it returns.
                    # If an agent passes a label like ``"Token: abc123"`` the
                    # response should not echo the secret in plaintext into the
                    # transcript — same defense as the URL redaction in §9.1.
                    # (``remaining[]`` deliberately keeps labels & values
                    # verbatim — see :meth:`_fill_form_captcha_envelope` — so
                    # the agent can resume after solving without losing data.)
                    safe_label = sanitize_for_prompt(
                        self.redactor.redact(agent_id, label)
                    )

                    if not find_res.get("success"):
                        # Snapshot/find_text failure is service-side; surface
                        # as type_failed so the loop continues — the agent
                        # can re-snapshot and retry.
                        err = find_res.get("error") or {}
                        raw_reason = (
                            str(err.get("message")) if isinstance(err, dict)
                            else str(err)
                        ) or "find_text failed"
                        filled.append({
                            "label": safe_label,
                            "status": "type_failed",
                            "reason": self.redactor.redact(agent_id, raw_reason),
                        })
                        continue

                    matches = (find_res.get("data") or {}).get("matches") or []
                    if not matches:
                        filled.append({"label": safe_label, "status": "not_found"})
                        # CAPTCHA may have appeared as a result of the snapshot
                        # itself (rare — e.g. a JS challenge that injects on
                        # every navigation). Check before continuing so we
                        # don't keep pummeling a blocked page with snapshots.
                        # _check_captcha now always returns the §11.13 envelope
                        # (truthy even with no captcha); only stop the loop
                        # when a captcha was actually found AND not auto-solved.
                        envelope = await self._check_captcha(inst)
                        if (
                            envelope.get("captcha_found")
                            and envelope.get("solver_outcome") != "solved"
                        ):
                            return self._fill_form_captcha_envelope(
                                filled, normalized[i + 1:],
                                _with_legacy_fields(envelope),
                                submitted=submitted,
                            )
                        continue

                    # Prefer fillable controls when the visible label text and
                    # the input's accessible name both match. Snapshot order can
                    # put a <label> before its <input>; picking that label first
                    # would yield a needless not_fillable failure even though the
                    # correct textbox is present in the same match set.
                    def _preferred_match(m: dict) -> bool:
                        ref_id = m.get("ref")
                        handle = inst.refs.get(ref_id)
                        role = (getattr(handle, "role", "") or "").lower()
                        disabled = bool(getattr(handle, "disabled", False))
                        return (
                            role in self._FILL_FORM_PREFERRED_ROLES
                            and not disabled
                        )

                    pick = (
                        next(
                            (
                                m for m in matches
                                if m.get("in_viewport") and _preferred_match(m)
                            ),
                            None,
                        )
                        or next((m for m in matches if _preferred_match(m)), None)
                        or next((m for m in matches if m.get("in_viewport")), None)
                        or matches[0]
                    )
                    ref = pick.get("ref")

                    try:
                        locator = await self._locator_from_ref(inst, ref)
                    except RefStale as rs:
                        filled.append({
                            "label": safe_label,
                            "ref": ref,
                            "status": "type_failed",
                            "reason": self.redactor.redact(agent_id, str(rs)),
                        })
                        continue
                    if locator is None:
                        filled.append({
                            "label": safe_label,
                            "ref": ref,
                            "status": "type_failed",
                            "reason": "ref not found",
                        })
                        continue

                    try:
                        await locator.fill(value)
                    except Exception as e:
                        # Element no longer attached / disabled / hidden /
                        # not-fillable (label-element returned by find_text).
                        # Don't bail the whole form: subsequent fields may still
                        # be reachable. ``reason`` is a structured code (see
                        # :meth:`_classify_fill_error`) so the agent can plan a
                        # specific recovery — e.g. re-snapshot on ``detached``,
                        # fall back to ``browser_get_elements`` on
                        # ``not_fillable``.
                        reason_code = self._classify_fill_error(e)
                        filled.append({
                            "label": safe_label,
                            "ref": ref,
                            "status": "type_failed",
                            "reason": reason_code,
                            "detail": self.redactor.redact(agent_id, str(e))[:200],
                        })
                        continue

                    filled.append({
                        "label": safe_label,
                        "ref": ref,
                        "status": "filled",
                    })
                    last_locator = locator

                    # Per-field submit_after fires after a successful fill but
                    # BEFORE the captcha check, because submitting can be what
                    # triggers the captcha. If the press succeeds we mark
                    # ``submitted=True`` so a follow-up captcha envelope tells
                    # the agent the form was already submitted (possibly with
                    # partial data) — the agent must NOT just resume typing
                    # the remaining fields without first re-checking page
                    # state.
                    if field_submit:
                        try:
                            await locator.press("Enter")
                            submitted = True
                        except Exception as e:
                            logger.debug(
                                "fill_form per-field submit failed for %s "
                                "label=%r: %s", agent_id, label, e,
                            )

                    # 2) After each successful fill (or per-field submit), check
                    #    for a captcha. If found, stop the loop and return
                    #    partial_success — the agent must solve before resuming.
                    #    Captcha mid-flow takes priority over top-level
                    #    submit_after: we never auto-submit a half-completed
                    #    form behind a captcha.
                    #    _check_captcha now always returns the §11.13 envelope
                    #    (truthy even with no captcha); only break out when a
                    #    captcha was actually found AND not auto-solved.
                    envelope = await self._check_captcha(inst)
                    if (
                        envelope.get("captcha_found")
                        and envelope.get("solver_outcome") != "solved"
                    ):
                        return self._fill_form_captcha_envelope(
                            filled, normalized[i + 1:],
                            _with_legacy_fields(envelope),
                            submitted=submitted,
                        )

                all_filled = all(f.get("status") == "filled" for f in filled)
                # 3) Final submit — only if no captcha interrupted us AND every
                #    requested field was filled. Top-level submit_after is the
                #    "submit the completed form" affordance; submitting after a
                #    not_found/type_failed field would send partial data without
                #    the caller explicitly opting into that via per-field
                #    submit_after.
                if submit_after and last_locator is not None and all_filled:
                    try:
                        await last_locator.press("Enter")
                        submitted = True
                    except Exception as e:
                        logger.debug(
                            "fill_form final submit failed for %s: %s",
                            agent_id, e,
                        )

                partial = not all_filled
                return {
                    "success": True,
                    "data": {
                        "partial_success": partial,
                        "captcha_required": False,
                        "filled": filled,
                        "remaining": [],
                        "submitted": submitted,
                    },
                }

            # §11.4 / §18.2 — outer wrapper. The body itself fires
            # ``_check_captcha`` after each field; the wrapper adds a
            # final post-action probe so a captcha that appears AFTER
            # the final submit (e.g. submit-triggered challenge that
            # renders too late for the inner check) is still surfaced.
            try:
                action_result, captcha_envelope = await self._with_captcha_redetect(
                    inst, _fill_body(),
                )
            except Exception as e:
                return {"success": False, "error": str(e)}

            # If the body already returned a captcha envelope under the
            # §9.4 partial-success path (``data.captcha_required=True``)
            # we leave it alone. The outer wrapper only adds the
            # ``"captcha"`` field on the SUCCESS / no-captcha-yet path
            # to avoid duplicating the same envelope.
            if (
                captcha_envelope is not None
                and isinstance(action_result, dict)
                and action_result.get("success")
                and not (action_result.get("data") or {}).get("captcha_required")
            ):
                data = action_result.setdefault("data", {})
                data.setdefault("captcha", captcha_envelope)
            return action_result

    @staticmethod
    def _fill_form_captcha_envelope(
        filled: list[dict], remaining_fields: list[dict], captcha: dict,
        *, submitted: bool = False,
    ) -> dict:
        """Compose the §9.4 partial-success envelope on captcha mid-flow.

        ``remaining`` echoes ``label`` + ``value`` + per-field
        ``submit_after`` verbatim — the agent supplied these and needs
        them unchanged to resume after solving. The redactor runs on
        ``error.message`` strings but NOT on ``remaining[].value``: the
        agent already has the value (it sent it to us); echoing it back
        is not a leak, and stripping it would break resume.

        ``submitted`` is True when a per-field ``submit_after=True``
        Enter-press succeeded earlier in this call (and was likely the
        trigger for the captcha). The agent uses this to decide whether
        to plain-resume (re-type ``remaining``) or re-snapshot first to
        check whether the partial-data submit went through.
        """
        return {
            "success": True,
            "data": {
                "partial_success": True,
                "captcha_required": True,
                "filled": filled,
                "remaining": [
                    {
                        "label": f["label"],
                        "value": f["value"],
                        "submit_after": f.get("submit_after", False),
                    }
                    for f in remaining_fields
                ],
                "captcha": captcha,
                "submitted": submitted,
            },
        }

    async def open_tab(
        self, agent_id: str, url: str, snapshot_after: bool = False,
    ) -> dict:
        """Open ``url`` in a new tab and make it the active page.

        Cookies and storage are shared with existing tabs (same browser
        context). The new page is registered in ``inst.page_ids`` so refs
        captured against it resolve correctly. On goto failure the new
        page is closed and the previous active tab is restored.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return _err("invalid_input", "Invalid URL")
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_URL_SCHEMES:
            return _err(
                "invalid_input",
                f"URL scheme '{parsed.scheme}' is not allowed",
            )

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            if inst._user_control:
                return _err(
                    "conflict",
                    "User has browser control — action paused until control is released.",
                )
            previous_page = inst.page
            try:
                new_page = await inst.context.new_page()
            except Exception as e:
                logger.debug("open_tab new_page failed for %s: %s", agent_id, e)
                return _err("service_unavailable", "Failed to open new tab")

            try:
                resolved_referer = ""
                try:
                    previous_url = (
                        previous_page.url
                        if previous_page is not None and inst.had_real_navigate
                        else ""
                    )
                    resolved_referer = pick_referer(
                        url,
                        previous_url=previous_url,
                        recent_referers=tuple(inst.recent_referers),
                    )
                except Exception as e:
                    logger.debug("open_tab referer pick failed: %s", e)
                    resolved_referer = ""

                goto_kwargs: dict = {
                    "wait_until": "domcontentloaded", "timeout": 30000,
                }
                if resolved_referer:
                    goto_kwargs["referer"] = resolved_referer
                # Per-platform pre-nav dwell — same posture as the main
                # ``navigate`` path. New-tab arrivals on these platforms
                # are also profiled by the in-house behavioral models, so
                # the delay applies here too.
                await self._apply_platform_pre_nav_delay(agent_id, url)
                try:
                    await new_page.goto(url, **goto_kwargs)
                except Exception as e:
                    logger.debug(
                        "open_tab goto failed for %s url=%s: %s",
                        agent_id, url, e,
                    )
                    try:
                        await new_page.close()
                    except Exception:
                        pass
                    inst.page = previous_page
                    return _err("service_unavailable", "Failed to navigate to URL")

                # Register page only after goto succeeds — a failed goto
                # closes the page, so registering before would leak an
                # entry in inst.page_ids per failed open_tab call.
                page_id = inst._register_page(new_page)

                inst.recent_referers.append(resolved_referer)
                inst.had_real_navigate = True

                inst.page = new_page
                inst.refs = {}  # Stale refs from previous tab's snapshot
                inst.dialog_active = False
                inst.dialog_detected = False
                try:
                    await new_page.bring_to_front()
                except Exception:
                    pass

                title = ""
                try:
                    title = await new_page.title()
                except Exception:
                    pass
                tab_index = len(inst.context.pages) - 1
                current_url = new_page.url

                data = {
                    "page_id": page_id,
                    "tab_index": tab_index,
                    "url": self.redactor.redact(agent_id, current_url),
                    "title": self.redactor.redact(agent_id, title),
                }
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    if snap.get("success"):
                        data["snapshot"] = snap.get("data") or {}
                    else:
                        data["snapshot_error"] = snap.get("error")
                return {"success": True, "data": data}
            except Exception as e:
                logger.debug("open_tab failed for %s: %s", agent_id, e)
                try:
                    await new_page.close()
                except Exception:
                    pass
                inst.page = previous_page
                return _err("service_unavailable", "open_tab failed")

    # ── §9.1 Network inspection ─────────────────────────────────

    def _attach_network_listeners(self, inst: CamoufoxInstance) -> None:
        """Wire BrowserContext-level ``request`` / ``requestfailed`` listeners.

        Idempotent — second and subsequent calls return immediately. Listeners
        are attached on the *context*. Per Playwright docs the context-level
        ``request`` event is "emitted when a request is issued from any pages
        created through this context", so this single hook covers:

        * the initial page (``inst.page``) without a separate per-page hookup,
        * any pre-existing pages already in ``inst.context.pages`` (e.g. a
          Camoufox profile that restored prior tabs at launch — rare, but
          per-page wiring used to silently miss these),
        * tabs opened later via in-page ``window.open()`` or
          :meth:`open_tab`.

        ``requestfailed`` is similarly context-scoped.
        """
        if inst._network_attached:
            return
        try:
            inst.context.on(
                "request",
                lambda req: self._record_request(inst, req),
            )
            inst.context.on(
                "requestfailed",
                lambda req: self._record_request_failed(inst, req),
            )
        except Exception as e:
            # Listener wiring is best-effort: a Camoufox build that doesn't
            # surface these events shouldn't take the browser down. The
            # `inspect_requests` reader returns an empty list cleanly.
            logger.debug(
                "Network listener wiring failed for '%s': %s",
                inst.agent_id, e,
            )
            return
        inst._network_attached = True

    def _record_request(self, inst: CamoufoxInstance, req) -> None:
        """Record an outbound request. Listener; never raises."""
        try:
            from src.shared.redaction import redact_url
            url = redact_url(getattr(req, "url", "") or "")
            method = getattr(req, "method", "") or ""
            resource_type = getattr(req, "resource_type", "") or ""
            inst.network_log.append({
                "url": url,
                "method": method,
                "resource_type": resource_type,
                "ts": time.time(),
                "status": None,
                "blocked_by_adblock": False,
                "user_cancelled": False,
                "failed_network": False,
                # Internal pairing sentinel — set True the first time a
                # ``requestfailed`` matches this entry so concurrent
                # identical-URL failures pair to DIFFERENT log entries
                # rather than overwriting the same one. Stripped from the
                # response payload by ``inspect_requests``.
                "_failure_tagged": False,
                # Pair ``requestfailed`` back to the exact Playwright
                # Request object when possible. We hold a STRONG reference
                # to the request object (not ``id(req)``) because Python
                # explicitly recycles ``id()`` values for non-overlapping
                # object lifetimes — on Python 3.12 the allocator is
                # aggressive enough that a freshly-created mock in a unit
                # test routinely lands at the same address as a previously
                # GC'd request, causing ``id()``-based pairing to match the
                # wrong entry. URL+method is retained as a fallback for
                # bindings that synthesize a distinct object for the
                # failure callback.
                "_request_key": req,
            })
        except Exception as e:
            logger.debug("network listener record failed: %s", e)

    def _record_request_failed(self, inst: CamoufoxInstance, req) -> None:
        """Mark the matching log entry as blocked / cancelled / failed.

        Updates by URL+method match if found, else discards the failure
        update. ``requestfailed`` fires asynchronously and may overlap with
        the request having already scrolled out of the maxlen=200 window.
        Creating a phantom entry would mislead operators about traffic that
        actually happened.

        When two identical-URL requests fire in parallel and both fail
        (e.g. two ``<img>`` loads of the same blocked tracker), each
        ``requestfailed`` event tags a *different* log entry — we walk
        newest-first and skip entries whose ``_failure_tagged`` is already
        True. Without this, the second ``requestfailed`` would overwrite
        the same entry and the first request would silently appear as
        "successful" in the log.
        """
        try:
            from src.shared.redaction import redact_url
            url = redact_url(getattr(req, "url", "") or "")
            method = getattr(req, "method", "") or ""
            failure = getattr(req, "failure", None)
            # Playwright surfaces ``failure`` as a property; some bindings
            # may expose it as a callable. Handle both.
            if callable(failure):
                try:
                    failure = failure()
                except Exception:
                    failure = None
            err = getattr(failure, "errorText", "") if failure else ""
            if not isinstance(err, str):
                err = str(err) if err else ""
            blocked = False
            cancelled = False
            for marker in ("BLOCKED_BY_CLIENT", "CONTENT_BLOCKED", "BLOCKED_BY_POLICY"):
                if marker in err:
                    blocked = True
                    break
            # Why classify BINDING_ABORTED separately from blocked: user-
            # cancelled is a real outcome (page nav interrupted, user hit
            # stop), not adblock — agents debugging form submits should see
            # these distinctly.
            if not blocked and "BINDING_ABORTED" in err:
                cancelled = True
            failed_net = not (blocked or cancelled)

            # Prefer the exact Request-object match. Playwright sends the
            # same Request object to ``request`` and ``requestfailed``; using
            # that identity prevents a failed older request from tagging a
            # newer identical URL+method that is still in flight. We compare
            # by ``is`` (object identity) on the strong reference stashed at
            # record time — NOT by ``id()``, which CPython recycles across
            # non-overlapping lifetimes (the original bug).
            for entry in reversed(inst.network_log):
                if (
                    entry.get("_request_key") is req
                    and not entry.get("_failure_tagged")
                ):
                    entry["blocked_by_adblock"] = blocked
                    entry["user_cancelled"] = cancelled
                    entry["failed_network"] = failed_net
                    entry["_failure_tagged"] = True
                    return

            # Fallback for tests / alternate bindings that surface distinct
            # request objects for failure events: update the newest matching
            # entry that hasn't already been tagged. Reverse iteration keeps
            # rapid-fire identical failures paired 1:1 via ``_failure_tagged``.
            for entry in reversed(inst.network_log):
                if (
                    entry["url"] == url
                    and entry["method"] == method
                    and not entry.get("_failure_tagged")
                ):
                    entry["blocked_by_adblock"] = blocked
                    entry["user_cancelled"] = cancelled
                    entry["failed_network"] = failed_net
                    entry["_failure_tagged"] = True
                    return
            # No untagged match → discard. Do NOT append; would create a
            # phantom entry.
        except Exception as e:
            logger.debug("network listener fail-record failed: %s", e)

    async def inspect_requests(
        self,
        agent_id: str,
        *,
        include_blocked: bool = False,
        limit: int = 50,
    ) -> dict:
        """Return a snapshot of recent network requests for ``agent_id``.

        URLs in the deque are already redacted at store-time. The returned
        ``requests`` list is sorted newest-first and capped at ``limit``
        (which is itself capped at the deque maxlen of 200). When
        ``include_blocked`` is False, entries flagged ``blocked_by_adblock``
        are filtered out so agents aren't confused by adblock-suppressed
        third-party trackers; the count of dropped entries is returned as
        ``dropped_blocked``.
        """
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            if inst._user_control:
                return _err(
                    "conflict",
                    "User has browser control — action paused until control is released.",
                )
            # Snapshot under the lock so we don't race a concurrent listener
            # append. The deque itself is thread-safe for single op-pends but
            # multi-step iteration + filter benefits from the lock.
            entries = list(inst.network_log)
            total = len(entries)
            dropped = 0
            visible: list[dict] = []
            # Iterate newest-first.
            for entry in reversed(entries):
                if entry.get("blocked_by_adblock") and not include_blocked:
                    dropped += 1
                    continue
                if len(visible) >= limit:
                    continue
                ts_unix = entry.get("ts") or 0
                visible.append({
                    "url": entry.get("url", ""),
                    "method": entry.get("method", ""),
                    "resource_type": entry.get("resource_type", ""),
                    "ts": _iso8601_utc(ts_unix),
                    "status": entry.get("status"),
                    "blocked_by_adblock": bool(entry.get("blocked_by_adblock")),
                    "user_cancelled": bool(entry.get("user_cancelled")),
                    "failed_network": bool(entry.get("failed_network")),
                })
            return {
                "success": True,
                "data": {
                    "requests": visible,
                    "total": total,
                    "dropped_blocked": dropped,
                },
            }

    async def press_key(self, agent_id: str, key: str) -> dict:
        """Press a keyboard key or combination (e.g. 'Enter', 'Escape', 'Control+a').

        Dispatches a real keyDown/keyUp event pair via Playwright, producing
        trusted keyboard events.  Useful for dismissing modals (Escape),
        submitting forms (Enter), tabbing between fields (Tab), or keyboard
        navigation (ArrowUp/ArrowDown).
        """
        if not key or len(key) > 50:
            return {"success": False, "error": "Invalid key"}
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            async def _press_body() -> dict:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                if inst.x11_wid and self._is_x11_site(inst):
                    xkey = self._playwright_key_to_xdotool(key)
                    try:
                        await self._x11_key(inst, xkey)
                    except Exception as e:
                        logger.warning(
                            "X11 press_key failed for '%s', falling back to CDP: %s",
                            agent_id, e,
                        )
                        await inst.page.keyboard.press(key)
                else:
                    await inst.page.keyboard.press(key)
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"pressed": key}}

            try:
                action_result, captcha_envelope = await self._with_captcha_redetect(
                    inst, _press_body(),
                )
            except Exception as e:
                return {"success": False, "error": str(e)}

            if (
                captcha_envelope is not None
                and isinstance(action_result, dict)
                and action_result.get("success")
            ):
                action_result.setdefault("captcha", captcha_envelope)
            return action_result

    async def go_back(self, agent_id: str) -> dict:
        """Navigate back in browser history."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                response = await inst.page.go_back(timeout=10000)
                inst.dialog_active = False  # New page — stale modal state
                inst.dialog_detected = False
                await asyncio.sleep(action_delay())
                title = await inst.page.title()
                url = self.redactor.redact(agent_id, inst.page.url)
                title = self.redactor.redact(agent_id, title)
                navigated = response is not None
                return {"success": True, "data": {"url": url, "title": title, "navigated": navigated}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def go_forward(self, agent_id: str) -> dict:
        """Navigate forward in browser history."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                response = await inst.page.go_forward(timeout=10000)
                inst.dialog_active = False  # New page — stale modal state
                inst.dialog_detected = False
                await asyncio.sleep(action_delay())
                title = await inst.page.title()
                url = self.redactor.redact(agent_id, inst.page.url)
                title = self.redactor.redact(agent_id, title)
                navigated = response is not None
                return {"success": True, "data": {"url": url, "title": title, "navigated": navigated}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def switch_tab(self, agent_id: str, tab_index: int = -1) -> dict:
        """List open tabs and optionally switch to one.

        tab_index=-1 (default): list all tabs without switching.
        tab_index>=0: switch to that tab index and clear stale refs.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                pages = inst.context.pages
                if not pages:
                    return {"success": False, "error": "No tabs open"}

                # Build tab list
                tabs = []
                active_index = 0
                for i, page in enumerate(pages):
                    is_active = page == inst.page
                    if is_active:
                        active_index = i
                    try:
                        title = await page.title()
                    except Exception:
                        title = "(loading)"
                    tabs.append({
                        "index": i,
                        "url": self.redactor.redact(agent_id, page.url),
                        "title": self.redactor.redact(agent_id, title),
                        "active": is_active,
                    })

                # Switch if requested
                if tab_index >= 0:
                    if tab_index >= len(pages):
                        return {
                            "success": False,
                            "error": f"Tab {tab_index} out of range (0-{len(pages) - 1})",
                        }
                    inst.page = pages[tab_index]
                    await inst.page.bring_to_front()
                    inst.refs = {}  # Stale refs from previous tab's snapshot
                    inst.dialog_active = False  # New tab may not have a dialog
                    inst.dialog_detected = False
                    active_index = tab_index
                    for t in tabs:
                        t["active"] = t["index"] == tab_index

                return {
                    "success": True,
                    "data": {"tabs": tabs, "active_tab": active_index},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def import_cookies(
        self, agent_id: str, cookies: list[dict],
    ) -> dict:
        """Phase 6 §9.2: import operator-supplied cookies into the agent's
        browser context.

        Invoked by the operator-only dashboard endpoint — agents NEVER
        call this. Validation, format detection (Playwright vs Netscape),
        and shape coercion happen upstream in the dashboard helper
        :func:`_validate_cookies` so this method receives a pre-validated
        list of Playwright-shaped cookie dicts.

        Merge semantics: this calls Playwright's ``add_cookies`` which
        MERGES with the existing context — a (name, domain, path) tuple
        collision overwrites the prior cookie; non-colliding entries are
        appended. There is no "wipe and replace" — operators wanting a
        clean slate should reset the agent profile first.

        At-rest leak warning: Firefox stores cookies plaintext in
        ``cookies.sqlite`` inside the agent profile dir — operators
        handling high-value sessions must use encrypted volumes or
        ephemeral profiles (see plan §13 risk register).
        """
        if not isinstance(cookies, list):
            return _err("invalid_input", "cookies must be a list")
        try:
            inst = await self.get_or_start(agent_id)
        except Exception as e:
            logger.warning(
                "import_cookies: failed to start browser for '%s': %s",
                agent_id, e,
            )
            return _err("service_unavailable", "Browser unavailable")
        inst.touch()
        async with inst.lock:
            try:
                # Empty list is a valid no-op (count=0).
                if cookies:
                    await inst.context.add_cookies(cookies)
                return {
                    "success": True,
                    "data": {"imported": len(cookies)},
                }
            except Exception as e:
                # Defense-in-depth: never echo a Playwright error back to
                # the operator — earlier review noted that a substring
                # heuristic ("value" in msg) was fragile, e.g. a future
                # Playwright change that renamed the offending field
                # (or used the cookie ``name``) could leak the secret.
                # The upstream ``_validate_cookies`` already rejects
                # malformed entries, so any error here is unexpected.
                # Log raw error server-side ONLY; return a generic shape.
                logger.warning(
                    "import_cookies: Playwright add_cookies failed for "
                    "'%s': %s", agent_id, e,
                )
                return _err(
                    "invalid_input",
                    "cookie import failed (see service logs)",
                )
