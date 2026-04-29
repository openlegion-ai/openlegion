"""CAPTCHA-solving service integration.

Supports 2Captcha and CapSolver as solving providers.  Both are called via
their public HTTP APIs using httpx — no additional dependencies required.

When configured (via browser flags ``CAPTCHA_SOLVER_PROVIDER`` and
``CAPTCHA_SOLVER_KEY``), the browser service will automatically attempt to
solve CAPTCHAs detected after navigation. If solving fails or no solver is
configured, the existing fallback (ask user via VNC) is preserved.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import httpx

from src.browser import flags, timing
from src.shared.redaction import redact_url
from src.shared.utils import setup_logging

logger = setup_logging("browser.captcha")


# ── §11.2/§11.13 structured solver result ─────────────────────────────────
#
# Replaces the old ``solve() -> bool`` + per-instance scratch-pad attrs
# (``last_used_proxy_aware`` / ``last_compat_rejected``) which raced across
# concurrent agents sharing a single CaptchaSolver instance.
#
# The dataclass is frozen so callers cannot accidentally mutate it after
# they've stamped it onto an envelope; concurrent solves on different
# agents now each own their own SolveResult and the cross-agent clobber
# is structurally impossible.
@dataclass(frozen=True)
class SolveResult:
    """Per-call solver result.

    Attributes:
        token: Provider-issued solution token, or ``None`` when the solver
            never retrieved one (sitekey extraction failed, provider
            returned ``errorId>0``, the breaker / health gate fired, etc.).
        injection_succeeded: Only meaningful when ``token`` is not ``None``.
            ``False`` indicates the provider was paid but the token failed
            to land in the page DOM — accounting MUST still increment
            because the provider already charged. Caller surfaces
            ``solver_outcome="injection_failed"`` in this case.
        used_proxy_aware: ``True`` iff the body sent to the provider used
            the proxy-aware task family + carried the dedicated solver-proxy
            credential fields. Drives cost-counter pricing tier (~3× the
            proxyless rate) and replaces the old per-instance
            ``last_used_proxy_aware`` scratch attr.
        compat_rejected: ``True`` iff a solver proxy was configured but the
            (provider, variant, type) tuple was rejected by the compat
            table — the body fell back to proxyless. Drives the
            ``solver_confidence="low"`` envelope downgrade. Replaces the
            old per-instance ``last_compat_rejected`` scratch attr.
        skipped: When non-``None``, the metering layer short-circuited
            BEFORE the solver HTTP call. One of:
              * ``"disabled"`` — ``CAPTCHA_DISABLED`` kill switch active.
                Replaces the per-call early-return that used to live in
                ``solve_captcha``; the auto-detect entry points
                (navigate / click) now flow through the same gate.
              * ``"rate_limited"`` — per-agent solve-rate gate fired.
              * ``"cost_cap"`` — per-agent monthly cost-cap reached.
              * ``"provider_missing"`` — solver lacks a string ``provider``
                attribute AND a cost cap is configured; we fail closed
                rather than let an untrackable charge slip past the cap.
              * ``"price_missing"`` — solver/provider exists but the
                ``(provider, kind)`` tuple has no published cost while a
                cost cap is configured; we fail closed for the same
                reason.
            ``None`` for a real solver attempt (success or failure).

    Field-population guarantee for ``_metered_solve`` consumers:
      * Any ``skipped`` value → ``token=None, injection_succeeded=False,
        used_proxy_aware=False, compat_rejected=False``.
      * ``skipped=None``: every other field reflects the actual solver
        attempt; ``token is None`` indicates a failed solve.
    """

    token: str | None
    injection_succeeded: bool
    used_proxy_aware: bool
    compat_rejected: bool
    skipped: str | None = None

    def __bool__(self) -> bool:
        """Truthiness == "agent-visible success" (token retrieved AND injected).

        Mirrors the old ``solve() -> bool`` semantics so existing call sites
        that still write ``if result: ...`` keep working without code
        changes. New code should branch on ``result.token is not None`` for
        cost-accounting decisions and ``result.injection_succeeded`` for
        agent-visible outcome — these two are no longer the same thing.
        """
        return self.token is not None and self.injection_succeeded

# §11.9 — per-type solver timeout. Each kind enum string maps to the
# overall solve deadline in milliseconds (sitekey extract → submit →
# poll-until-ready). The 30s ``httpx`` timeout for individual provider
# requests stays intact (see ``_get_client``); this controls the OUTER
# ``asyncio.wait_for(...)`` wrapping the whole solve pipeline.
#
# Defaults are tuned per provider documentation:
#   * v3 / Enterprise-v3: 60s — provider-side score lookup, no human
#     interaction; faster than v2.
#   * v2 / hCaptcha: 120s — requires worker pool to view & click images.
#   * Turnstile / cf-interstitial-turnstile: 180s — CF challenges
#     occasionally chain a re-issue, longest in the wild.
#
# FunCaptcha / GeeTest / AWS WAF entries are reserved for §11.5 deferred
# work and intentionally not listed; unknown kinds use ``_FALLBACK``.
_SOLVE_TIMEOUT_DEFAULTS_MS: dict[str, int] = {
    "recaptcha-v2-checkbox":     120_000,
    "recaptcha-v2-invisible":    120_000,
    "recaptcha-v3":               60_000,
    "recaptcha-enterprise-v2":   120_000,
    "recaptcha-enterprise-v3":    60_000,
    "hcaptcha":                  120_000,
    "turnstile":                 180_000,
    "cf-interstitial-turnstile": 180_000,
    # FunCaptcha / GeeTest / AWS WAF entries reserved for §11.5 deferred work
    # Anti-bot platform task types (CapSolver-only, see ``_ANTIBOT_KINDS``).
    # Provider-side solve takes 60-180s; we pick the high end as default
    # because all four platforms (Akamai BMP, Imperva, Kasada, DataDome)
    # are known-slow and a tight default would surface as a flood of
    # ``solver_outcome="timeout"`` envelopes on legitimate solves.
    # Operators can override per-kind via
    # ``CAPTCHA_TIMEOUT_<KIND_UPPER_UNDERSCORE>_MS`` (see
    # :meth:`_timeout_seconds_for_kind`).
    "js-challenge-akamai":       180_000,
    "js-challenge-imperva":      180_000,
    "js-challenge-kasada":       180_000,
    "datadome-behavioral":       180_000,
}
_SOLVE_TIMEOUT_FALLBACK_MS = 120_000  # for unknown / behavioral kinds
_POLL_INTERVAL = 5    # seconds between result polls
_SUPPORTED_PROVIDERS = ("2captcha", "capsolver")

# Health check (§11.16): 5s budget per provider. Latency above the warn
# threshold marks the solver "degraded"; the call still counts as healthy
# but operators should route new solves to a configured secondary.
_HEALTH_CHECK_TIMEOUT = 5.0
_HEALTH_DEGRADED_LATENCY = 3.0

# Circuit breaker (§11.16): 3 failures inside a 5-min sliding window trip
# the breaker for 10 min. We track timestamps in a bounded deque so the
# math is "count of entries newer than NOW-300s" — no per-failure
# counter+timestamp pair to keep coherent.
_BREAKER_FAILURE_WINDOW = 300.0    # 5 min
_BREAKER_FAILURE_THRESHOLD = 3
_BREAKER_OPEN_DURATION = 600.0      # 10 min


# Fatal provider error markers — substring (case-insensitive) matched
# against the ``errorDescription`` returned by a solver provider. When a
# response carries one of these, the issue is operator-actionable
# (revoked / mistyped key, drained balance, banned account) and will not
# recover until the operator intervenes. We mark the process-wide solver
# UNREACHABLE so subsequent solves short-circuit cleanly via the existing
# health-check gate, instead of letting the per-process circuit breaker
# trip on three of these in a row. The breaker is meant to ride out
# transient provider outages — three "zero balance" errors from one
# agent should not lock the whole fleet's solver out for 10 minutes;
# the operator needs to fix billing, not wait for a backoff window.
#
# Both 2Captcha and CapSolver use these or near-identical strings as of
# 2026-04. Substring + case-insensitive match keeps us robust against
# minor wording drift ("Invalid API key" vs "ERROR_INVALID_API_KEY").
_FATAL_PROVIDER_ERROR_MARKERS: frozenset[str] = frozenset({
    "ERROR_KEY_DOES_NOT_EXIST",
    "ERROR_WRONG_USER_KEY",
    "ERROR_ZERO_BALANCE",
    "ERROR_KEY_DENIED_ACCESS",
    "ERROR_INVALID_API_KEY",
    "ERROR_INSUFFICIENT_BALANCE",
    "ERROR_USER_NOT_FOUND",
    "ERROR_KEY_BANNED",
    "ERROR_ACCOUNT_SUSPENDED",
})


def _is_fatal_provider_error(error_description: object) -> bool:
    """True iff ``error_description`` names an operator-actionable error.

    See :data:`_FATAL_PROVIDER_ERROR_MARKERS` for the rationale and the
    canonical marker list. ``error_description`` is whatever
    ``data.get("errorDescription")`` returned — usually a string but
    occasionally ``None`` or numeric on malformed responses.
    """
    if error_description is None:
        return False
    upper = str(error_description).upper()
    return any(marker in upper for marker in _FATAL_PROVIDER_ERROR_MARKERS)

# /getBalance endpoints — both providers expose these and accept the same
# JSON body shape (``{"clientKey": "..."}``). Both return
# ``{"errorId": 0, "balance": <float>}`` on success.
_HEALTH_URLS: dict[str, str] = {
    "2captcha": "https://api.2captcha.com/getBalance",
    "capsolver": "https://api.capsolver.com/getBalance",
}


def _redact_clientkey(body: dict) -> dict:
    """Return a shallow copy of ``body`` with ``clientKey`` masked.

    Solver providers occasionally echo the ``clientKey`` field back inside
    error responses (real, observed behavior). Anything that flows to the
    logger — request bodies, response bodies, error tracebacks — must scrub
    that field first. Pair with :func:`redact_url` for the request URL.
    """
    if not isinstance(body, dict) or "clientKey" not in body:
        return body
    out = dict(body)
    out["clientKey"] = "[REDACTED]"
    return out


# Provider error strings sometimes embed the raw key as
# ``clientKey=VALUE`` or ``"clientKey":"VALUE"``. Catch both spellings
# so an exception ``str()`` doesn't leak it through the logger.
_CLIENTKEY_IN_TEXT = re.compile(
    r'(clientKey)\s*["\']?\s*[:=]\s*["\']?([A-Za-z0-9_\-]+)["\']?',
    re.IGNORECASE,
)

# Solver task IDs (UUIDs and integer strings both used by 2captcha /
# CapSolver). Echoed in error responses; redact on logging so a hostile
# provider error containing a stitched-together credential string can't
# leak via the task identifier path.
_TASKID_IN_TEXT = re.compile(
    r'(taskId)\s*["\']?\s*[:=]\s*["\']?'
    r'([A-Za-z0-9_\-]{6,})["\']?',
    re.IGNORECASE,
)


def _redact_clientkey_text(text: str) -> str:
    """Strip ``clientKey=VALUE`` / ``"clientKey":"VALUE"`` and ``taskId=…``.

    Pair with :func:`redact_url` (URL-shaped secrets) and
    :func:`_redact_clientkey` (dict bodies) before logging anything that
    might have come from a solver provider's error response.
    """
    if not text:
        return text
    out = _CLIENTKEY_IN_TEXT.sub(r"\1=[REDACTED]", text)
    out = _TASKID_IN_TEXT.sub(r"\1=[REDACTED]", out)
    return out


# §22 — keys carrying a solution payload for CapSolver anti-bot tasks.
# CapSolver's anti-bot product surface (AntiAkamaiBMPTask /
# AntiImpervaTask / AntiKasadaTask / DataDomeSliderTask) returns
# ``cookies`` / ``userAgent`` / ``sensorData`` instead of the
# CAPTCHA-style ``gRecaptchaResponse`` / ``token`` field. Surface the
# solution as a JSON-serialised string so the caller's accounting path
# fires (provider was paid) while making clear via the token shape that
# it isn't a CAPTCHA-injectable token. The ``_inject_token`` path doesn't
# match the anti-bot families and surfaces ``injection_failed``, which
# is the honest signal until a future PR threads the cookies / userAgent
# back into ``page.context`` directly.
_ANTIBOT_SOLUTION_KEYS: frozenset[str] = frozenset({
    "cookies", "userAgent", "sensorData", "headers",
})


def _extract_solution_token(solution: object, captcha_type: str) -> str | None:
    """Extract a token-shaped string from a provider ``solution`` payload.

    The default behavior (CAPTCHA-family tasks) reads the standard
    ``gRecaptchaResponse`` / ``token`` fields. For anti-bot kinds the
    solution carries cookies / userAgent / sensorData instead; we
    JSON-serialise the dict so the caller's "token retrieved" path fires
    even though there is no CAPTCHA-style token. Token injection still
    fails downstream for these kinds (no anti-bot inject path yet) but
    the cost-accounting + breaker semantics need a non-``None`` value to
    distinguish "provider charged us" from "provider rejected".
    """
    if not isinstance(solution, dict):
        return None
    # Standard CAPTCHA fields take precedence regardless of kind — covers
    # tests that mock anti-bot responses with ``gRecaptchaResponse`` and
    # the rare case where CapSolver routes an anti-bot task back through
    # the standard path.
    token = solution.get("gRecaptchaResponse") or solution.get("token")
    if token:
        return token
    if captcha_type in _ANTIBOT_KINDS:
        # Any presence of an anti-bot solution key signals a real
        # provider-side success. Serialise the whole dict so the agent /
        # operator can see what was returned without us trying to
        # inject a cookie set we don't know how to apply yet.
        if any(k in solution for k in _ANTIBOT_SOLUTION_KEYS):
            try:
                import json
                return json.dumps(solution, sort_keys=True, default=str)
            except (TypeError, ValueError):
                # Defensive — solution payloads are JSON over the wire
                # so this shouldn't happen, but fall through cleanly
                # rather than raising to the caller.
                return None
    return None


# Map from detected selector pattern to a canonical CAPTCHA type.
_CAPTCHA_TYPE_MAP: dict[str, str] = {
    "recaptcha": "recaptcha",
    "hcaptcha": "hcaptcha",
    "challenges.cloudflare.com": "turnstile",
    "cf-turnstile": "turnstile",
    "captcha": "recaptcha",  # generic fallback — most common type
}


# §11.1 — provider task-type tables, structured as
# ``dict[variant, dict[provider_field, value]]`` so each variant can
# declare task-specific extras (``isInvisible``, ``isEnterprise``) without
# growing the call-site logic. The submit-task code paths merge these
# extras into the task body alongside ``websiteURL`` / ``websiteKey``.
#
# Task-name verification (April 2026, against current public docs):
#   * 2captcha — https://2captcha.com/api-docs/recaptcha-v{2,3}{,-enterprise}
#     - v2 / v2-invisible: same task name ``RecaptchaV2TaskProxyless`` with
#       ``isInvisible`` flag distinguishing invisible.
#     - v3: ``RecaptchaV3TaskProxyless`` requires ``minScore`` + accepts
#       ``pageAction``. Enterprise v3 is the same task name with
#       ``isEnterprise: true`` — 2captcha has NO standalone
#       ``RecaptchaV3EnterpriseTaskProxyless`` type as of April 2026
#       (drift from the early spec; verified against the v3 doc page).
#     - Enterprise v2: ``RecaptchaV2EnterpriseTaskProxyless`` (distinct).
#   * CapSolver — https://docs.capsolver.com/en/guide/captcha/ReCaptchaV{2,3}/
#     - Distinct task names for each variant. ``isInvisible`` is the v2
#       knob; ``minScore`` is documented but score filtering may be
#       performed downstream at validation time rather than affecting
#       the solve itself (we still pass it; CapSolver tolerates extra
#       fields). ``pageAction`` is documented as optional.
#
# Drift here is silent (``ERROR_INVALID_TASK_TYPE``); re-verify against
# provider docs when bumping variants.
#
# §11.2 — each variant entry is now ``{"proxyless": <task name>,
# "proxy_aware": <task name> | None, "extra": {<flags>}}``. The task-body
# builder picks ``proxy_aware`` when a dedicated solver proxy is configured
# AND the compat table allows it; otherwise it falls back to ``proxyless``
# and (when the fallback is *due to* a compat-table rejection) sets the
# envelope's ``solver_confidence`` to ``"low"``. ``proxy_aware`` may be
# ``None`` for variants where the provider does not document a
# proxy-aware task type (e.g. 2captcha v3 has no documented proxy variant
# as of April 2026 — only ``RecaptchaV3TaskProxyless``).
#
# 2Captcha
_2CAPTCHA_TASK_TYPES: dict[str, dict[str, object]] = {
    # Legacy ``"recaptcha"`` key — kept so the classifier-unknown fallback
    # in :func:`_classify_captcha` still has somewhere to land. Previously
    # mapped to ``NormalRecaptchaTaskProxyless`` which 2captcha retired in
    # April 2026 (submitting it returns ``ERROR_INVALID_TASK_TYPE``).
    # Aliased to ``RecaptchaV2TaskProxyless`` — the safe v2-checkbox default
    # — so the legacy task name is no longer sent over the wire.
    "recaptcha": {
        "proxyless": "RecaptchaV2TaskProxyless",
        "proxy_aware": "RecaptchaV2Task",
        "extra": {},
    },
    "hcaptcha": {
        "proxyless": "HCaptchaTaskProxyless",
        "proxy_aware": "HCaptchaTask",
        "extra": {},
    },
    "turnstile": {
        "proxyless": "TurnstileTaskProxyless",
        "proxy_aware": "TurnstileTask",
        "extra": {},
    },
    # §11.1 reCAPTCHA variant matrix
    "recaptcha-v2-checkbox": {
        "proxyless": "RecaptchaV2TaskProxyless",
        "proxy_aware": "RecaptchaV2Task",
        "extra": {},
    },
    "recaptcha-v2-invisible": {
        "proxyless": "RecaptchaV2TaskProxyless",
        "proxy_aware": "RecaptchaV2Task",
        "extra": {"isInvisible": True},
    },
    # 2captcha v3 has only ``RecaptchaV3TaskProxyless`` documented as of
    # April 2026 — no proxy-aware variant is published. Solvers fall back
    # to proxyless and surface ``solver_confidence="low"``.
    "recaptcha-v3": {
        "proxyless": "RecaptchaV3TaskProxyless",
        "proxy_aware": None,
        "extra": {},
    },
    "recaptcha-enterprise-v2": {
        "proxyless": "RecaptchaV2EnterpriseTaskProxyless",
        "proxy_aware": "RecaptchaV2EnterpriseTask",
        "extra": {},
    },
    # 2captcha uses the same v3 task with isEnterprise=true (no separate
    # ``RecaptchaV3EnterpriseTaskProxyless`` type as of April 2026); also
    # no documented proxy-aware v3 variant.
    "recaptcha-enterprise-v3": {
        "proxyless": "RecaptchaV3TaskProxyless",
        "proxy_aware": None,
        "extra": {"isEnterprise": True},
    },
}

# §22 — anti-bot platform task types (CapSolver only).
#
# 2Captcha does NOT publish equivalents for these platforms — their
# product surface is reCAPTCHA / hCaptcha / Turnstile / FunCAPTCHA /
# GeeTest / image, NOT platform-sensor anti-bot. Adding these kinds to
# :data:`_2CAPTCHA_TASK_TYPES` would emit ``ERROR_INVALID_TASK_TYPE`` on
# every solve and trip the §11.16 breaker. The dispatch in
# :class:`CaptchaSolver.supports_kind` short-circuits anti-bot kinds when
# the active provider is 2Captcha so the §11.13 envelope routes to
# operator escalation (matches today's behavior for these kinds).
#
# CapSolver task types verified against
# https://docs.capsolver.com/en/guide/captcha/ as of April 2026:
#   * AntiAkamaiBMPTask        — Akamai Bot Manager
#   * AntiImpervaTask          — Imperva Advanced Bot Protection
#   * AntiKasadaTask           — Kasada
#   * DataDomeSliderTask       — DataDome behavioral / slider
#
# Anti-bot tasks REQUIRE a proxy — CapSolver does not publish any
# ``Proxyless`` variant for these. ``proxyless`` is intentionally ``None``
# so :meth:`_build_task_body` returns ``(None, ...)`` (i.e. unsupported)
# when the operator hasn't configured ``CAPTCHA_SOLVER_PROXY_*`` env
# vars; the agent then surfaces an operator-escalation envelope rather
# than silently routing to the always-fail proxyless path.
#
# Drift here is silent (``ERROR_INVALID_TASK_TYPE``); re-verify against
# CapSolver docs when bumping variants. The anti-bot product surface
# rotates faster than the standard CAPTCHA task types — provider docs
# drift quarterly. The §11.16 breaker + fatal-error gate keeps the
# fleet's solver path safe even when our table goes stale (an unknown
# task name surfaces as a per-call error and the operator sees the
# audit-log signal long before three failures trip the breaker for
# kinds the agent loop should never have routed to the solver).
#
# Anti-bot kinds are LOW-CONFIDENCE by design: failure rates can exceed
# 50% even with the right task type because tokens are rejected at the
# application layer for IP / fingerprint mismatches the solver has no
# visibility into. The dispatch in :meth:`solve` skips the breaker tick
# on anti-bot failures (see ``_record_solver_outcome`` call sites) so a
# burst of legitimate-but-rejected anti-bot solves doesn't lock the
# whole fleet's solver out of the standard CAPTCHA paths for 10 minutes.
_ANTIBOT_KINDS: frozenset[str] = frozenset({
    "js-challenge-akamai",
    "js-challenge-imperva",
    "js-challenge-kasada",
    "datadome-behavioral",
})


# CapSolver
_CAPSOLVER_TASK_TYPES: dict[str, dict[str, object]] = {
    "recaptcha": {
        "proxyless": "ReCaptchaV2TaskProxyLess",  # legacy alias
        "proxy_aware": "ReCaptchaV2Task",
        "extra": {},
    },
    "hcaptcha": {
        "proxyless": "HCaptchaTaskProxyLess",
        "proxy_aware": "HCaptchaTask",
        "extra": {},
    },
    "turnstile": {
        # CapSolver uses ``AntiTurnstileTask`` family for both proxyless
        # and proxy-aware (drop the "ProxyLess" suffix for proxy-aware).
        "proxyless": "AntiTurnstileTaskProxyLess",
        "proxy_aware": "AntiTurnstileTask",
        "extra": {},
    },
    # §11.1 reCAPTCHA variant matrix
    "recaptcha-v2-checkbox": {
        "proxyless": "ReCaptchaV2TaskProxyLess",
        "proxy_aware": "ReCaptchaV2Task",
        "extra": {},
    },
    "recaptcha-v2-invisible": {
        "proxyless": "ReCaptchaV2TaskProxyLess",
        "proxy_aware": "ReCaptchaV2Task",
        "extra": {"isInvisible": True},
    },
    "recaptcha-v3": {
        "proxyless": "ReCaptchaV3TaskProxyLess",
        "proxy_aware": "ReCaptchaV3Task",
        "extra": {},
    },
    "recaptcha-enterprise-v2": {
        "proxyless": "ReCaptchaV2EnterpriseTaskProxyLess",
        "proxy_aware": "ReCaptchaV2EnterpriseTask",
        "extra": {},
    },
    "recaptcha-enterprise-v3": {
        "proxyless": "ReCaptchaV3EnterpriseTaskProxyLess",
        "proxy_aware": "ReCaptchaV3EnterpriseTask",
        "extra": {},
    },
    # §22 — anti-bot platform task types. Proxy-required (no proxyless
    # variant published). Best-effort: CapSolver's anti-bot product
    # surface rotates faster than the standard CAPTCHA tasks — operators
    # who hit ``ERROR_INVALID_TASK_TYPE`` should re-verify the task name
    # against the live docs (see the comment block on ``_ANTIBOT_KINDS``
    # for the rationale and the breaker-skip semantics).
    "js-challenge-akamai": {
        "proxyless": None,
        "proxy_aware": "AntiAkamaiBMPTask",
        "extra": {},
    },
    "js-challenge-imperva": {
        "proxyless": None,
        "proxy_aware": "AntiImpervaTask",
        "extra": {},
    },
    "js-challenge-kasada": {
        "proxyless": None,
        "proxy_aware": "AntiKasadaTask",
        "extra": {},
    },
    "datadome-behavioral": {
        "proxyless": None,
        "proxy_aware": "DataDomeSliderTask",
        "extra": {},
    },
}


# §11.2 — solver-proxy compatibility table.
#
# Hardcoded set of accepted ``proxyType`` values per (provider, variant).
# Verified against 2captcha + CapSolver public docs in April 2026:
#
# * 2captcha — RecaptchaV2Task, HCaptchaTask, TurnstileTask all document
#   ``http | socks4 | socks5`` (no ``https``). Source:
#   https://2captcha.com/api-docs/recaptcha-v2 and the equivalent docs
#   for hcaptcha + cloudflare-turnstile.
# * CapSolver — all proxy-aware tasks document ``http | https | socks4 |
#   socks5``. Source: https://docs.capsolver.com/en/guide/api-how-to-use-proxy/
#   ("proxyType: socks5 | http | https | socks4").
#
# 2captcha FunCaptcha is documented to **reject SOCKS5** (FunCaptcha is
# deferred to §11.5; this is a placeholder for when that lands):
#   ``("2captcha", "funcaptcha"): {"http", "https", "socks4"}``
# Add when the FunCaptcha variant is wired in §11.5.
#
# A variant whose ``proxy_aware`` is ``None`` (e.g. 2captcha v3) still
# lives in the compat table for clarity, but the task-body builder
# short-circuits to proxyless before consulting compat.
_SOLVER_PROXY_COMPAT: dict[tuple[str, str], set[str]] = {
    # 2captcha — uniformly {http, socks4, socks5} (NO https) on all
    # documented proxy-aware task types.
    ("2captcha", "recaptcha-v2-checkbox"):    {"http", "socks4", "socks5"},
    ("2captcha", "recaptcha-v2-invisible"):   {"http", "socks4", "socks5"},
    ("2captcha", "recaptcha-enterprise-v2"):  {"http", "socks4", "socks5"},
    ("2captcha", "hcaptcha"):                 {"http", "socks4", "socks5"},
    ("2captcha", "turnstile"):                {"http", "socks4", "socks5"},
    # CapSolver — full set on all documented proxy-aware task types.
    ("capsolver", "recaptcha-v2-checkbox"):   {"http", "https", "socks4", "socks5"},
    ("capsolver", "recaptcha-v2-invisible"):  {"http", "https", "socks4", "socks5"},
    ("capsolver", "recaptcha-v3"):            {"http", "https", "socks4", "socks5"},
    ("capsolver", "recaptcha-enterprise-v2"): {"http", "https", "socks4", "socks5"},
    ("capsolver", "recaptcha-enterprise-v3"): {"http", "https", "socks4", "socks5"},
    ("capsolver", "hcaptcha"):                {"http", "https", "socks4", "socks5"},
    ("capsolver", "turnstile"):               {"http", "https", "socks4", "socks5"},
    # §22 — anti-bot tasks accept the same proxy-type set as the
    # standard CapSolver tasks. CapSolver's anti-bot docs do not
    # individually enumerate accepted proxyType values; we mirror the
    # standard set as the safe default.
    ("capsolver", "js-challenge-akamai"):     {"http", "https", "socks4", "socks5"},
    ("capsolver", "js-challenge-imperva"):    {"http", "https", "socks4", "socks5"},
    ("capsolver", "js-challenge-kasada"):     {"http", "https", "socks4", "socks5"},
    ("capsolver", "datadome-behavioral"):     {"http", "https", "socks4", "socks5"},
}


_VALID_PROXY_TYPES = frozenset({"http", "https", "socks4", "socks5"})


@dataclass(frozen=True)
class SolverProxyConfig:
    """Dedicated solver-side proxy.

    **NOT** the agent's primary egress proxy. The agent's primary proxy
    creds are NEVER forwarded to a solver provider — the threat model is
    that handing your scraping proxy creds to a third-party solver is a
    direct credential-leak vector. This struct is loaded from a
    *separate* env-var family (``CAPTCHA_SOLVER_PROXY_*``) which an
    operator opts into by setting all five fields.
    """

    proxy_type: str       # one of {"http", "https", "socks4", "socks5"}
    address: str
    port: int
    login: str
    password: str

    def to_request_fields(self) -> dict[str, object]:
        """Return the ``proxyType/Address/Port/Login/Password`` body fields.

        Both 2captcha and CapSolver use this exact field naming in the
        ``task`` block of ``createTask`` requests.
        """
        return {
            "proxyType": self.proxy_type,
            "proxyAddress": self.address,
            "proxyPort": self.port,
            "proxyLogin": self.login,
            "proxyPassword": self.password,
        }


def _normalize_proxy_type(raw: str) -> str | None:
    """Coerce ``socks5h://``/``socks5h``/``HTTP``/etc. to the canonical name.

    Returns the canonical type (``"socks5"`` etc.) or ``None`` for an
    unrecognized scheme. ``socks5h`` (DNS-via-SOCKS) collapses to
    ``socks5`` because both providers list only ``socks5`` — the ``h``
    variant is a client-side tunneling preference that doesn't change
    what the provider expects.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    # Strip ``://`` if present (URL-style).
    if "://" in s:
        s = s.split("://", 1)[0]
    # Collapse the ``socks5h`` (and theoretical ``socks4a``) variants to
    # the base name documented by the providers.
    if s == "socks5h":
        s = "socks5"
    if s == "socks4a":
        s = "socks4"
    return s if s in _VALID_PROXY_TYPES else None


# Once-per-session warning gate (§11.16 cadence): the loader logs at most
# one warning across the process lifetime when partial config is detected,
# rather than spamming on every solver call.
_proxy_config_warned: bool = False


def _reset_proxy_config_warning() -> None:
    """Test helper — clear the once-per-session warning flag."""
    global _proxy_config_warned
    _proxy_config_warned = False


def get_solver_proxy_config(
    *, agent_id: str | None = None,
) -> SolverProxyConfig | None:
    """Load the dedicated solver proxy from env via ``flags`` helpers.

    Behavior:

    * If ``CAPTCHA_SOLVER_PROXY_TYPE`` is unset → returns ``None``
      (proxyless task types will be used; this is the default).
    * If ``CAPTCHA_SOLVER_PROXY_TYPE`` is set, all five fields
      (``TYPE``, ``ADDRESS``, ``PORT``, ``LOGIN``, ``PASSWORD``) are
      required. Partial config logs a single warning at solver init and
      returns ``None`` (falls back to proxyless).
    * Bad scheme (anything other than http/https/socks4/socks5 after
      normalization) logs a warning and returns ``None``.

    Reads via ``flags.get_str``/``get_int`` so per-agent overrides and
    operator settings.json take precedence over env vars (matches the
    rest of the browser flag surface).
    """
    global _proxy_config_warned
    # Local import — avoid circular: flags imports nothing from captcha.
    from src.browser import flags as _flags

    raw_type = _flags.get_str(
        "CAPTCHA_SOLVER_PROXY_TYPE", "", agent_id=agent_id,
    ).strip()
    if not raw_type:
        # Default off — proxyless task types are used.
        return None

    proxy_type = _normalize_proxy_type(raw_type)
    if proxy_type is None:
        if not _proxy_config_warned:
            logger.warning(
                "CAPTCHA_SOLVER_PROXY_TYPE=%r is not in {http, https, socks4, "
                "socks5} (after socks5h→socks5 normalization); falling back "
                "to proxyless solver tasks.",
                raw_type,
            )
            _proxy_config_warned = True
        return None

    address = _flags.get_str(
        "CAPTCHA_SOLVER_PROXY_ADDRESS", "", agent_id=agent_id,
    ).strip()
    # ``get_int`` returns the default when unset; use a sentinel default
    # so we can distinguish "unset" from "explicitly 0".
    raw_port_str = _flags.get_str(
        "CAPTCHA_SOLVER_PROXY_PORT", "", agent_id=agent_id,
    ).strip()
    login = _flags.get_str(
        "CAPTCHA_SOLVER_PROXY_LOGIN", "", agent_id=agent_id,
    )
    password = _flags.get_str(
        "CAPTCHA_SOLVER_PROXY_PASSWORD", "", agent_id=agent_id,
    )

    # All 5 fields required when type is set.
    missing: list[str] = []
    if not address:
        missing.append("CAPTCHA_SOLVER_PROXY_ADDRESS")
    if not raw_port_str:
        missing.append("CAPTCHA_SOLVER_PROXY_PORT")
    if not login:
        missing.append("CAPTCHA_SOLVER_PROXY_LOGIN")
    if not password:
        missing.append("CAPTCHA_SOLVER_PROXY_PASSWORD")

    port = 0
    if raw_port_str:
        try:
            port = int(raw_port_str)
            if port <= 0 or port > 65535:
                missing.append("CAPTCHA_SOLVER_PROXY_PORT(out-of-range)")
        except ValueError:
            missing.append("CAPTCHA_SOLVER_PROXY_PORT(non-integer)")

    if missing:
        if not _proxy_config_warned:
            logger.warning(
                "CAPTCHA_SOLVER_PROXY_TYPE is set but required fields are "
                "missing/invalid: %s — falling back to proxyless solver "
                "tasks. Set all 5 CAPTCHA_SOLVER_PROXY_* env vars to enable "
                "the dedicated solver proxy.",
                ", ".join(missing),
            )
            _proxy_config_warned = True
        return None

    return SolverProxyConfig(
        proxy_type=proxy_type,
        address=address,
        port=port,
        login=login,
        password=password,
    )


def _solver_proxy_compatible(provider: str, captcha_type: str, proxy_type: str) -> bool:
    """Return True iff ``(provider, captcha_type)`` documents ``proxy_type``.

    Variants absent from the compat table are treated as not-compatible —
    we'd rather fall back to proxyless than send a request shape the
    provider hasn't documented.
    """
    allowed = _SOLVER_PROXY_COMPAT.get((provider.lower(), captcha_type))
    if allowed is None:
        return False
    return proxy_type in allowed


# Default ``pageAction`` when the classifier couldn't extract one. Most
# v3 implementations accept a generic ``"verify"`` action — solving with
# the wrong action returns a token bound to the wrong action and the
# verifying server may still reject the score, but it's the best
# permissive fallback we have.
_DEFAULT_V3_ACTION = "verify"
_DEFAULT_V3_MIN_SCORE = 0.7


def _build_single_solver(
    provider_flag: str, key_flag: str, *, label: str,
) -> CaptchaSolver | None:
    """Resolve one ``(provider, key)`` flag pair into a :class:`CaptchaSolver`.

    Shared loader for the primary and secondary solver slots used by
    :func:`get_solver`. Returns ``None`` when either flag is unset or the
    provider name isn't in :data:`_SUPPORTED_PROVIDERS` — the latter logs
    a warning so an operator typo (``CAPTCHA_SOLVER_PROVIDER_SECONDARY=foo``)
    surfaces cleanly instead of silently degrading to single-provider.
    """
    provider = flags.get_str(provider_flag, "").strip().lower()
    api_key = flags.get_str(key_flag, "").strip()
    if not provider or not api_key:
        return None
    if provider not in _SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown %s %r (expected one of %s), %s solver disabled",
            provider_flag, provider, ", ".join(_SUPPORTED_PROVIDERS), label,
        )
        return None
    logger.info("CAPTCHA %s solver configured: provider=%s", label, provider)
    return CaptchaSolver(provider, api_key)


def get_solver() -> "MultiProviderSolver | None":
    """Return the configured solver wrapper, or ``None`` if none configured.

    Reads ``CAPTCHA_SOLVER_PROVIDER`` / ``CAPTCHA_SOLVER_KEY`` for the
    primary slot and ``CAPTCHA_SOLVER_PROVIDER_SECONDARY`` /
    ``CAPTCHA_SOLVER_KEY_SECONDARY`` for the §11.8 failover slot at
    process startup. **Per-agent overrides are NOT supported for
    provider/key** — the solver is constructed once in
    ``BrowserManager.__init__`` and shared process-wide. The returned
    wrapper carries stateful breaker / health-check / cost-counter
    coupling on each underlying :class:`CaptchaSolver` that would not
    be safe to swap per-call.

    Returns:
      * ``None`` — neither slot configured (no solver).
      * :class:`MultiProviderSolver` with ``secondary=None`` —
        single-provider deployment; failover paths are no-ops.
      * :class:`MultiProviderSolver` with both slots populated —
        full §11.8 failover behavior.

    The wrapper exposes the same public surface as :class:`CaptchaSolver`
    (``solve``, ``is_solver_unreachable``, ``is_breaker_open``,
    ``health_check``, ``close``, ``provider``) so the rest of
    :mod:`src.browser.service` stays unchanged.

    Use ``CAPTCHA_DISABLED`` per-agent to disable solving for a specific
    agent. Solver-proxy creds DO support per-agent override (see
    :func:`get_solver_proxy_config`).

    Live-reload requires browser-service restart; flag changes via
    ``config/settings.json`` take effect on the next process start.
    """
    primary = _build_single_solver(
        "CAPTCHA_SOLVER_PROVIDER", "CAPTCHA_SOLVER_KEY", label="primary",
    )
    if primary is None:
        return None
    secondary = _build_single_solver(
        "CAPTCHA_SOLVER_PROVIDER_SECONDARY",
        "CAPTCHA_SOLVER_KEY_SECONDARY",
        label="secondary",
    )
    if secondary is not None:
        logger.info(
            "CAPTCHA failover armed: primary=%s secondary=%s",
            primary.provider, secondary.provider,
        )
    return MultiProviderSolver(primary, secondary)


def _classify_captcha(selector: str) -> str:
    """Map a CSS selector string to a canonical CAPTCHA type."""
    sel_lower = selector.lower()
    for pattern, captcha_type in _CAPTCHA_TYPE_MAP.items():
        if pattern in sel_lower:
            return captcha_type
    return "recaptcha"  # safe default


# ── §11.1 reCAPTCHA variant classifier ───────────────────────────────────


# Single JS probe walking ``window.___grecaptcha_cfg`` plus the script /
# DOM signals. Returning a structured dict from one ``page.evaluate`` is
# cheaper than five round trips and keeps the page-side logic auditable
# in one place. Designed to never throw — every branch wraps in
# try/catch and falls back to ``null``.
_CLASSIFY_RECAPTCHA_JS = r"""
() => {
  const out = {
    enterprise: false,
    v3: false,
    sitekeys: [],         // collected from registry walk
    actions_by_key: {},   // sitekey -> action (v3 only)
    invisible_by_key: {}, // sitekey -> bool (explicit-render size)
    enterprise_script: false,
    v3_render_param: null,    // sitekey from api.js?render=...
  };
  try {
    // ── 1. Script-tag scan: enterprise namespace + v3 render param ──
    const scripts = document.querySelectorAll('script[src]');
    for (const s of scripts) {
      const src = s.getAttribute('src') || '';
      if (
        src.indexOf('enterprise.recaptcha.net') !== -1 ||
        src.indexOf('recaptcha/enterprise/') !== -1 ||
        src.indexOf('/enterprise.js') !== -1
      ) {
        out.enterprise = true;
        out.enterprise_script = true;
      }
      // v3-only render parameter on api.js: ?render=<sitekey>
      const m = src.match(/[?&]render=([^&]+)/);
      if (m && m[1] && m[1] !== 'explicit') {
        out.v3 = true;
        out.v3_render_param = decodeURIComponent(m[1]);
      }
    }
    // ── 2. Global ``grecaptcha.enterprise`` presence ───────────────
    try {
      if (
        typeof window.grecaptcha !== 'undefined' &&
        window.grecaptcha &&
        typeof window.grecaptcha.enterprise === 'object'
      ) {
        out.enterprise = true;
      }
    } catch (e) { /* defensive */ }
    // ── 3. ``___grecaptcha_cfg`` registry walk ─────────────────────
    let cfg = null;
    try { cfg = window.___grecaptcha_cfg; } catch (e) { cfg = null; }
    if (cfg && cfg.clients) {
      for (const cid in cfg.clients) {
        const client = cfg.clients[cid];
        // The widget config is nested arbitrarily deep inside ``client``
        // (Google obfuscates the tree). Walk shallowly looking for the
        // canonical fields. Depth 6 is enough in practice.
        const seen = new Set();
        const stack = [[client, 0]];
        let foundSitekey = null;
        let foundAction = null;
        let foundSize = null;
        while (stack.length) {
          const [node, d] = stack.pop();
          if (!node || typeof node !== 'object' || d > 6) continue;
          if (seen.has(node)) continue;
          seen.add(node);
          for (const k in node) {
            const v = node[k];
            if (k === 'sitekey' && typeof v === 'string' && v.length > 10) {
              foundSitekey = foundSitekey || v;
            }
            if (k === 'action' && typeof v === 'string' && v.length > 0) {
              foundAction = foundAction || v;
            }
            if (k === 'size' && typeof v === 'string') {
              foundSize = foundSize || v;
            }
            if (v && typeof v === 'object') {
              stack.push([v, d + 1]);
            }
          }
        }
        if (foundSitekey) {
          out.sitekeys.push(foundSitekey);
          if (foundAction) {
            out.v3 = true;
            out.actions_by_key[foundSitekey] = foundAction;
          }
          if (foundSize) {
            out.invisible_by_key[foundSitekey] = (foundSize === 'invisible');
          }
        }
      }
    }
    // ── 4. DOM ``data-sitekey`` fallback ───────────────────────────
    if (out.sitekeys.length === 0) {
      const el = document.querySelector('[data-sitekey]');
      if (el) {
        const sk = el.getAttribute('data-sitekey');
        if (sk) out.sitekeys.push(sk);
        // ``data-size="invisible"`` on the widget div is the v2-invisible
        // marker for explicit-render pages.
        const ds = el.getAttribute('data-size');
        if (ds && sk) out.invisible_by_key[sk] = (ds === 'invisible');
      }
    }
  } catch (e) { /* defensive — never throw to the page */ }
  return out;
}
"""


async def _classify_recaptcha(page) -> dict:
    """Classify a reCAPTCHA widget into one of five variants.

    Returns a dict with::

        {
          "variant": "recaptcha-v2-checkbox" | "recaptcha-v2-invisible"
                   | "recaptcha-v3" | "recaptcha-enterprise-v2"
                   | "recaptcha-enterprise-v3" | "unknown",
          "sitekey": str | None,
          "action": str | None,      # v3 only — extracted from registry
          "min_score": float | None, # always None — operator config
        }

    Falls back gracefully:

    * ``variant: "unknown"`` is returned if the page doesn't expose enough
      signal (e.g. JS not yet loaded, registry obfuscated, no
      ``data-sitekey``). Callers should fall through to the existing
      v2-checkbox heuristic when they want the safe default.
    * ``min_score`` is **never** filled here — it's an operator-tuned
      knob (``CAPTCHA_RECAPTCHA_V3_MIN_SCORE``) read at solve time, not
      a property of the page.
    * ``action`` extraction is **best-effort** for v3. If the widget
      renders explicitly via ``grecaptcha.execute(sitekey, {action: X})``
      where ``X`` is a string literal in a separate script we don't see,
      we return ``None`` and let ``_check_captcha`` log a warning. The
      solver will still produce a token, but it'll be bound to a
      different action and the target server may reject — known
      limitation, documented at the call site.

    Logging uses the existing ``browser.captcha`` logger; no new logger.
    """
    result: dict = {
        "variant": "unknown",
        "sitekey": None,
        "action": None,
        "min_score": None,
    }
    try:
        probe = await page.evaluate(_CLASSIFY_RECAPTCHA_JS)
    except Exception:
        logger.debug("_classify_recaptcha: page.evaluate failed", exc_info=True)
        return result
    if not isinstance(probe, dict):
        return result

    enterprise = bool(probe.get("enterprise"))
    is_v3 = bool(probe.get("v3"))
    sitekeys = probe.get("sitekeys") or []
    actions_by_key = probe.get("actions_by_key") or {}
    invisible_by_key = probe.get("invisible_by_key") or {}
    v3_render_param = probe.get("v3_render_param") or None

    # Sitekey: prefer registry walk; fall back to v3 render param; then None.
    sitekey: str | None = None
    if sitekeys:
        sitekey = sitekeys[0]
    elif v3_render_param:
        sitekey = v3_render_param
    if isinstance(sitekey, str):
        sitekey = sitekey.strip() or None

    # Action: only meaningful for v3 widgets. Use the registry mapping
    # for the chosen sitekey.
    action: str | None = None
    if is_v3 and sitekey and sitekey in actions_by_key:
        candidate = actions_by_key.get(sitekey)
        if isinstance(candidate, str) and candidate.strip():
            action = candidate.strip()

    # Variant decision tree.
    if is_v3 and enterprise:
        variant = "recaptcha-enterprise-v3"
    elif is_v3:
        variant = "recaptcha-v3"
    elif enterprise:
        variant = "recaptcha-enterprise-v2"
    else:
        # v2 — distinguish checkbox vs invisible.
        invisible = False
        if sitekey and sitekey in invisible_by_key:
            invisible = bool(invisible_by_key.get(sitekey))
        elif invisible_by_key:
            # No sitekey match but some widget on the page is invisible.
            invisible = any(bool(v) for v in invisible_by_key.values())
        if invisible:
            variant = "recaptcha-v2-invisible"
        elif sitekey or v3_render_param:
            variant = "recaptcha-v2-checkbox"
        else:
            variant = "unknown"

    result["variant"] = variant
    result["sitekey"] = sitekey
    result["action"] = action
    return result


# ── §11.3 Cloudflare interstitial tri-state classifier ────────────────────
#
# CF interstitial is not one thing. We distinguish three states so that
# ``_check_captcha`` can route each appropriately:
#
#   "auto"       — CF auto-resolving JS challenge ("Checking your
#                  browser...") with NO Turnstile widget. Wait+recheck.
#   "behavioral" — Under Attack Mode (error 1020) or persistent challenge
#                  with no solvable widget. Skip solver, escalate.
#   "turnstile"  — CF interstitial with embedded Turnstile widget. Solve
#                  via existing Turnstile path but with low confidence
#                  (token may not unblock the session).
#   "none"       — Doesn't match any CF interstitial pattern (caller falls
#                  through to standalone Turnstile / other handling).
#
# The classifier reads ``page.title`` and runs a single lightweight JS
# probe; both calls are wrapped in try/except so a closed page or evaluate
# failure simply collapses the result to "none" rather than raising.

_CF_STATE_PROBE_JS = r"""
() => {
  const out = {
    has_challenge_running: false,
    has_turnstile: false,
    has_cf_error_1020: false,
    has_challenge_error_text: false,
  };
  try {
    if (document.querySelector('#challenge-running')) {
      out.has_challenge_running = true;
    }
    // Accept both ``.cf-turnstile`` (new) and ``[class*="cf-turnstile"]``
    // partial-match for safety against version drift. ``data-sitekey`` is
    // not strictly required here — its absence is interpreted by the
    // Python side which prefers the auto-resolving path when no widget
    // is renderable.
    const ts = document.querySelector('.cf-turnstile, [class*="cf-turnstile"]');
    if (ts) out.has_turnstile = true;
    const errEl = document.querySelector('#cf-error-details');
    if (errEl) {
      const txt = (errEl.textContent || '').toLowerCase();
      // CF error code 1020 = "Access denied" / Under Attack Mode block.
      if (txt.indexOf('1020') !== -1) out.has_cf_error_1020 = true;
    }
    if (document.querySelector('#challenge-error-text')) {
      out.has_challenge_error_text = true;
    }
  } catch (e) { /* defensive — never throw to the page */ }
  return out;
}
"""


async def _classify_cf_state(page) -> str:
    """Classify a CF interstitial into one of four states.

    Returns one of ``"auto"`` / ``"behavioral"`` / ``"turnstile"`` /
    ``"none"``. Never raises — any page-side failure (closed page, JS
    sandbox error, page navigated mid-call) collapses to ``"none"``.

    Detection precedence (highest to lowest):

    1. ``has_cf_error_1020`` OR ``has_challenge_error_text`` →
       ``"behavioral"``. These are the Under Attack / persistent-challenge
       markers; the agent is being soft-blocked and a solver call won't
       help.
    2. ``has_turnstile`` AND title starts with "Just a moment" →
       ``"turnstile"``. Solve via existing flow but caller marks
       confidence low (~50% success; tokens are session-cookie bound).
    3. ``has_challenge_running`` AND title starts with "Just a moment"
       AND NOT ``has_turnstile`` → ``"auto"``. The classic "Checking your
       browser..." auto-resolving JS challenge — wait then re-check.
    4. Anything else → ``"none"``.
    """
    try:
        title = await page.title()
    except Exception:
        title = ""
    try:
        probe = await page.evaluate(_CF_STATE_PROBE_JS)
    except Exception:
        logger.debug("_classify_cf_state: page.evaluate failed", exc_info=True)
        return "none"
    if not isinstance(probe, dict):
        return "none"

    title_str = title if isinstance(title, str) else ""
    title_match = title_str.startswith("Just a moment")

    if probe.get("has_cf_error_1020") or probe.get("has_challenge_error_text"):
        return "behavioral"
    if probe.get("has_turnstile") and title_match:
        return "turnstile"
    if probe.get("has_challenge_running") and title_match:
        return "auto"
    return "none"


# ── §11.3 Behavioral-only challenge classifier ────────────────────────────
#
# Non-CF challenges that present no solvable widget — the only correct
# response is to escalate to ``request_captcha_help``. We detect via DOM
# query rather than touching ``page.title`` so the classifier works on
# challenge overlays that don't change the document title.

_BEHAVIORAL_PROBE_JS = r"""
() => {
  const out = { px: false, datadome: false };
  try {
    // PerimeterX / HUMAN Security "Press & Hold". Both legacy
    // (``#px-captcha``, ``button[data-v="px-button"]``) and the modern
    // HUMAN-rebrand selectors per §11.18 (``[data-human-security]``,
    // ``[class*="human-challenge"]``).
    if (
      document.querySelector('#px-captcha') ||
      document.querySelector('button[data-v="px-button"]') ||
      document.querySelector('[data-human-security]') ||
      document.querySelector('[class*="human-challenge"]')
    ) {
      out.px = true;
    }
    // DataDome behavioral blocker — only the ``/blocker`` path. The
    // generic ``captcha-delivery.com`` host is also used by the solvable
    // slider (deferred §11.5); we MUST NOT flag that as behavioral.
    const ddIframes = document.querySelectorAll(
      'iframe[src*="captcha-delivery.com/blocker"]'
    );
    if (ddIframes.length > 0) out.datadome = true;
  } catch (e) { /* defensive */ }
  return out;
}
"""


async def _classify_behavioral(page) -> str | None:
    """Detect non-CF behavioral-only challenges that should bypass the solver.

    Returns the §11.13 ``kind`` enum string for the matched challenge, or
    ``None`` if no behavioral challenge is present. Never raises.

    Detection order (highest precedence first):
      1. PerimeterX / HUMAN Security "Press & Hold" → ``"px-press-hold"``.
      2. DataDome behavioral blocker (``/blocker`` path only) →
         ``"datadome-behavioral"``.

    The DataDome solvable-slider path (``captcha-delivery.com`` without
    ``/blocker``) is intentionally NOT detected here — it routes through
    §11.5 (deferred) once that solver lands.
    """
    try:
        probe = await page.evaluate(_BEHAVIORAL_PROBE_JS)
    except Exception:
        logger.debug("_classify_behavioral: page.evaluate failed", exc_info=True)
        return None
    if not isinstance(probe, dict):
        return None
    if probe.get("px"):
        return "px-press-hold"
    if probe.get("datadome"):
        return "datadome-behavioral"
    return None


class CaptchaSolver:
    """Async CAPTCHA solver using 2Captcha or CapSolver HTTP APIs."""

    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        # ── §11.16 state ────────────────────────────────────────────────
        # Per-process gate: the health check fires the first time ANY
        # agent calls solve() in this BrowserManager session. The solver
        # client is shared across agents in the same process, so checking
        # once per process (not per agent) matches what we're actually
        # verifying — a single underlying httpx client to a single
        # provider endpoint.
        self._solver_health_checked: bool = False
        self._solver_unreachable: bool = False
        self._solver_health_degraded: bool = False
        # Sliding-window failure tracking. maxlen=10 caps memory if we
        # ever wedge into a long failure storm; only the entries within
        # the 5-min window matter for the breaker decision.
        self._solver_failure_timestamps: deque[float] = deque(maxlen=10)
        self._solver_breaker_until: float = 0.0
        # Coordinates breaker reads/writes with health-check init across
        # concurrent agents that share this solver instance.
        self._state_lock: asyncio.Lock | None = None
        self._state_lock_loop: asyncio.AbstractEventLoop | None = None
        # NOTE: Prior versions of this class exposed ``last_used_proxy_aware``
        # and ``last_compat_rejected`` instance attributes that ``solve()``
        # stamped after every call so the BrowserManager could pick up the
        # proxy-aware billing tier. Those attrs RACED ACROSS CONCURRENT
        # AGENTS sharing a single CaptchaSolver instance — agent A's solve
        # could overwrite agent B's flags between B's stamp and B's read.
        # Replaced by the per-call :class:`SolveResult` dataclass returned
        # from :meth:`solve`. Subclasses that referenced the old attrs need
        # to migrate to the new ``solve() -> SolveResult`` shape.
        # ── §11.9 per-type timeout table ────────────────────────────────
        # Resolved once at solver init from the static defaults plus any
        # ``CAPTCHA_TIMEOUT_<KIND_UPPER_UNDERSCORE>_MS`` env overrides.
        # Operator-tunable but not per-call — restart the browser service
        # to pick up env var changes.
        self._solve_timeouts_ms: dict[str, int] = self._resolve_solve_timeouts()

    def _get_state_lock(self) -> asyncio.Lock:
        """Return a lock bound to the currently running event loop.

        Python 3.9 binds ``asyncio.Lock`` at construction time. Creating the
        solver from synchronous startup/test code after a previous loop closed
        raises ``RuntimeError``. Lazily creating the lock inside async paths
        keeps the solver safe for sync construction and normal browser-service
        use.
        """
        loop = asyncio.get_running_loop()
        if self._state_lock is None or self._state_lock_loop is not loop:
            self._state_lock = asyncio.Lock()
            self._state_lock_loop = loop
        return self._state_lock

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── §11.9 per-type timeout resolution ──────────────────────────────

    @staticmethod
    def _resolve_solve_timeouts() -> dict[str, int]:
        """Build the kind → timeout-ms table from defaults + env overrides.

        Env override pattern: ``CAPTCHA_TIMEOUT_<KIND_UPPER_UNDERSCORE>_MS``.
        For example ``CAPTCHA_TIMEOUT_RECAPTCHA_V3_MS=45000`` overrides the
        v3 entry to 45 seconds. Hyphens in kind names are converted to
        underscores when building the env-var name. Read once at solver
        init — reconfig requires a service restart (matches the §11.9 spec
        for solver-level constants).
        """
        resolved: dict[str, int] = {}
        for kind, default_ms in _SOLVE_TIMEOUT_DEFAULTS_MS.items():
            env_name = f"CAPTCHA_TIMEOUT_{kind.replace('-', '_').upper()}_MS"
            resolved[kind] = flags.get_int(env_name, default_ms, min_value=1)
        return resolved

    def _timeout_seconds_for_kind(self, kind: str | None) -> float:
        """Return the overall solve timeout (seconds) for ``kind``.

        Unknown / behavioral / ``None`` kinds map to
        :data:`_SOLVE_TIMEOUT_FALLBACK_MS`. The httpx per-request timeout
        configured on the shared client (30s) is unrelated — this controls
        the OUTER ``asyncio.wait_for`` deadline on the whole solve
        pipeline (sitekey extract + submit + poll loop).
        """
        ms = self._solve_timeouts_ms.get(kind or "", _SOLVE_TIMEOUT_FALLBACK_MS)
        return ms / 1000.0

    # ── §11.16 health check + circuit breaker ──────────────────────────

    async def is_solver_unreachable(self) -> bool:
        """Return ``True`` iff the solver has been marked unreachable.

        Lazy: when the per-process health probe hasn't fired yet, this
        method drives it before reading the cached state. Prior versions
        of this method were synchronous and read a pre-set flag —
        :meth:`solve` was the *only* path that triggered the probe, so
        the FIRST captcha of a session ALWAYS slipped past the
        :meth:`_check_captcha` ``is_solver_unreachable()`` gate (which
        runs BEFORE :meth:`solve`). Lazy + async closes that hole: the
        probe fires on the gate's read, not on the eventual solve.

        After the probe completes the result is sticky for the rest of
        the instance-session — subsequent calls return the cached flag
        without re-probing.
        """
        if not self._solver_health_checked:
            await self._ensure_health_checked()
        return self._solver_unreachable

    def is_breaker_open(self) -> bool:
        """True iff a tripped breaker is still within its 10-min window.

        On the first read after expiry, we proactively reset the breaker
        timestamp AND the failure-window deque. Functionally the deque
        prune in :meth:`_record_solver_outcome` already drops stale
        entries before the next decision, but resetting here ensures
        that if the next failure arrives less than 5 min after the
        breaker auto-clears (e.g. failures at t=100/200/300 trip the
        breaker at t=300, expires at t=900; a single new failure at
        t=901 prunes correctly because t=300 is past 901-300=601, so
        the deque is empty before append). The defense-in-depth is
        cheap and prevents a future change from accidentally re-tripping
        the breaker on stale entries.
        """
        if self._solver_breaker_until == 0.0:
            return False
        if self._solver_breaker_until > time.time():
            return True
        # Breaker auto-clears: drop the timestamp and the stale failure
        # window so the solver gets a clean restart.
        self._solver_breaker_until = 0.0
        self._solver_failure_timestamps.clear()
        return False

    def supports_kind(self, kind: str | None) -> bool:
        """True iff this solver's provider has a published task type for ``kind``.

        Drives the §22 routing decision in :meth:`_check_captcha`: anti-bot
        kinds (Akamai BMP / Imperva / Kasada / DataDome) are dispatched to
        the solver path only when the active provider declares a task
        type for them. 2Captcha does not publish anti-bot task types
        (their product is reCAPTCHA / hCaptcha / Turnstile / FunCAPTCHA /
        GeeTest / image), so :data:`_2CAPTCHA_TASK_TYPES` returns
        ``False`` for every anti-bot kind and the caller routes those
        cleanly to operator escalation instead of contacting an unsupported
        provider.

        Returns ``False`` for ``None`` and for kinds the provider table
        doesn't list. Lookup is case-sensitive — the §11.13 envelope
        kinds are canonical lowercase-hyphen.
        """
        if not isinstance(kind, str) or not kind:
            return False
        prov = (self.provider or "").lower()
        if prov == "2captcha":
            return kind in _2CAPTCHA_TASK_TYPES
        if prov == "capsolver":
            return kind in _CAPSOLVER_TASK_TYPES
        # Unknown provider — fall through to "no" rather than guessing.
        return False

    async def health_check(
        self, provider: str | None = None,
    ) -> Literal["healthy", "degraded", "unreachable"]:
        """Probe the solver's ``/getBalance`` endpoint with a 5s budget.

        ``provider`` defaults to ``self.provider``. Returns one of:

        * ``healthy`` — HTTP 200, ``balance`` is non-negative numeric, and
          latency is under :data:`_HEALTH_DEGRADED_LATENCY`.
        * ``degraded`` — HTTP 200 but latency exceeded the warn threshold.
          Not fatal; logged so operators see slow upstream and can route
          new solves to a configured secondary.
        * ``unreachable`` — timeout, connection error, 5xx, non-200 status,
          or a non-zero ``errorId`` in the JSON body. Caller marks the
          solver unreachable for the rest of this instance-session.

        Logging never includes the raw ``clientKey``: the URL flows
        through :func:`redact_url`, the request body through
        :func:`_redact_clientkey`, and exception strings through
        :func:`_redact_clientkey_text`.
        """
        prov = (provider or self.provider).lower()
        url = _HEALTH_URLS.get(prov)
        if url is None:
            logger.warning("health_check: unknown provider %r", prov)
            return "unreachable"

        client = self._get_client()
        body = {"clientKey": self.api_key}
        safe_url = redact_url(url)
        safe_body = _redact_clientkey(body)
        start = time.monotonic()
        try:
            resp = await client.post(url, json=body, timeout=_HEALTH_CHECK_TIMEOUT)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            logger.warning(
                "Solver health check timed out (provider=%s url=%s body=%s)",
                prov, safe_url, safe_body,
            )
            return "unreachable"
        except httpx.HTTPError as e:
            logger.warning(
                "Solver health check connection error (provider=%s url=%s err=%s)",
                prov, safe_url,
                _redact_clientkey_text(redact_url(str(e))),
            )
            return "unreachable"
        except Exception as e:  # noqa: BLE001 — defensive log + return
            logger.warning(
                "Solver health check unexpected error (provider=%s url=%s err=%s)",
                prov, safe_url,
                _redact_clientkey_text(redact_url(str(e))),
            )
            return "unreachable"

        latency = time.monotonic() - start

        if resp.status_code != 200:
            logger.warning(
                "Solver health check non-200 (provider=%s url=%s status=%d)",
                prov, safe_url, resp.status_code,
            )
            return "unreachable"

        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Solver health check returned non-JSON (provider=%s url=%s)",
                prov, safe_url,
            )
            return "unreachable"

        if data.get("errorId", 0) != 0:
            logger.warning(
                "Solver health check error (provider=%s url=%s errorId=%s)",
                prov, safe_url, data.get("errorId"),
            )
            return "unreachable"

        # Both providers always return a numeric ``balance`` on success.
        # A missing/non-numeric field means we hit the wrong endpoint, the
        # provider returned an unexpected shape, or a proxy is interposing —
        # all of which are "don't trust this solver". Treat as unreachable
        # rather than silently passing through.
        if "balance" not in data:
            logger.warning(
                "Solver health check missing balance field (provider=%s)", prov,
            )
            return "unreachable"
        try:
            balance_f = float(data["balance"])
        except (TypeError, ValueError):
            logger.warning(
                "Solver health check non-numeric balance (provider=%s)", prov,
            )
            return "unreachable"
        if balance_f < 0:
            logger.warning(
                "Solver health check returned negative balance (provider=%s)", prov,
            )
            return "unreachable"

        if latency > _HEALTH_DEGRADED_LATENCY:
            logger.warning(
                "Solver health check degraded (provider=%s latency=%.2fs)",
                prov, latency,
            )
            return "degraded"
        logger.info(
            "Solver health check ok (provider=%s latency=%.2fs)", prov, latency,
        )
        return "healthy"

    async def _ensure_health_checked(self) -> None:
        """Run the per-process health check exactly once.

        Sets ``_solver_unreachable`` on a sticky basis so subsequent
        solves skip the provider entirely without re-probing.
        """
        if self._solver_health_checked:
            return
        async with self._get_state_lock():
            if self._solver_health_checked:
                return
            outcome = await self.health_check()
            self._solver_health_checked = True
            if outcome == "unreachable":
                self._solver_unreachable = True
            elif outcome == "degraded":
                self._solver_health_degraded = True

    def _handle_provider_error_response(self, data: dict) -> None:
        """Log + classify an ``errorId>0`` response from a solver provider.

        Always logs the (redacted) ``errorDescription`` so operators can
        diagnose. When the description matches a known fatal-config
        marker, additionally flips the per-process unreachable flag so
        subsequent solves short-circuit through the existing health
        gate. Without this, three "zero balance" errors from one agent
        in 5 minutes would trip the §11.16 breaker for the whole
        BrowserManager — the breaker tracks transient PROVIDER outages,
        not operator-actionable config faults.
        """
        description = data.get("errorDescription")
        # Render a clean placeholder when the provider returned no
        # description — ``str(None)`` would log the literal "None"
        # which is misleading in operator dashboards.
        if description is None or description == "":
            safe_desc = f"errorId={data.get('errorId')}"
        else:
            safe_desc = _redact_clientkey_text(str(description))
        if _is_fatal_provider_error(description):
            # Sticky for the rest of this process — the docstring on
            # ``_solver_unreachable`` (and the §11.16 breaker test
            # suite) treat the flag as one-way; an operator clears it
            # by restarting the browser service after fixing the key /
            # balance. This keeps semantics aligned with the existing
            # health-check unreachable path.
            self._solver_unreachable = True
            logger.warning(
                "%s reported a fatal config error (%s) — disabling solver "
                "for this session. Operator must fix the API key / balance "
                "and restart the browser service.",
                self.provider, safe_desc,
            )
            return
        logger.warning(
            "%s solver error: %s", self.provider, safe_desc,
        )

    async def _record_solver_outcome(self, success: bool) -> None:
        """Update the breaker state after a solve attempt.

        On success: reset both the failure window and any tripped breaker.
        On failure: append a timestamp, prune entries older than the 5-min
        window, then trip the breaker if 3+ entries remain.
        """
        async with self._get_state_lock():
            now = time.time()
            if success:
                self._solver_failure_timestamps.clear()
                self._solver_breaker_until = 0.0
                return
            self._solver_failure_timestamps.append(now)
            cutoff = now - _BREAKER_FAILURE_WINDOW
            while (
                self._solver_failure_timestamps
                and self._solver_failure_timestamps[0] < cutoff
            ):
                self._solver_failure_timestamps.popleft()
            if len(self._solver_failure_timestamps) >= _BREAKER_FAILURE_THRESHOLD:
                self._solver_breaker_until = now + _BREAKER_OPEN_DURATION
                logger.warning(
                    "Solver circuit breaker TRIPPED until %.0f (failures=%d)",
                    self._solver_breaker_until,
                    len(self._solver_failure_timestamps),
                )

    async def solve(
        self,
        page,
        selector: str,
        page_url: str,
        *,
        agent_id: str | None = None,
        kind: str | None = None,
    ) -> SolveResult:
        """Attempt to solve a CAPTCHA on the page.

        Args:
            page: Playwright page object.
            selector: The CSS selector that matched the CAPTCHA element.
            page_url: The current page URL.
            agent_id: Optional — for per-agent override of solver-proxy
                config via :func:`flags.set_agent_override`.
            kind: §11.13 envelope kind enum string (e.g. ``"recaptcha-v3"``,
                ``"hcaptcha"``, ``"cf-interstitial-turnstile"``). When
                supplied, the §11.9 per-type timeout table is keyed on this
                value; otherwise the timeout falls back to the variant
                classifier's output (which loses the CF-bound override
                signal). Always pass when the caller already has it.

        Returns:
            :class:`SolveResult` describing the outcome. Fields:

              * ``token`` — non-``None`` iff a token was retrieved from
                the provider (regardless of whether injection succeeded).
                Cost accounting MUST fire on a non-``None`` token because
                the provider already charged.
              * ``injection_succeeded`` — ``True`` iff the token landed
                in the page DOM. Only meaningful when ``token`` is set.
              * ``used_proxy_aware`` / ``compat_rejected`` — replaces the
                old per-instance scratch attrs. Read directly off the
                returned :class:`SolveResult`; concurrent solves on
                different agents now own their own SolveResult objects
                instead of racing on shared instance state.

            Short-circuit returns (no provider HTTP call):
              * solver unreachable → ``token=None, injection_succeeded=False``.
              * breaker open → same.
        """
        # Per-process gate — runs at most once even under concurrent solves.
        # NOTE: ``is_solver_unreachable`` is the authoritative gate (now
        # async + lazy in §11.16); awaiting the helper here is kept as a
        # belt-and-suspenders cache primer for direct callers that don't
        # go through the gate.
        await self._ensure_health_checked()

        if self._solver_unreachable:
            logger.info(
                "Skipping solve: solver marked unreachable for this session "
                "(provider=%s)", self.provider,
            )
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            )

        if self.is_breaker_open():
            logger.warning(
                "Skipping solve: solver circuit breaker open until %.0f (provider=%s)",
                self._solver_breaker_until, self.provider,
            )
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            )

        # §22 — anti-bot platform kinds (Akamai BMP, Imperva, Kasada,
        # DataDome) take precedence over selector-based classification.
        # The caller already classified via ``js_challenge.classify_*`` /
        # ``_classify_behavioral`` and passed the kind; we use it
        # directly so the §11.13 envelope kind matches what the
        # ``_check_captcha`` audit trail recorded. These kinds DO NOT use
        # ``websiteKey``-style sitekey markers and skipping the DOM
        # extraction below saves a no-op ``page.evaluate`` round-trip.
        antibot_path = kind is not None and kind in _ANTIBOT_KINDS

        page_action: str | None = None
        sitekey: str | None = None
        if antibot_path:
            captcha_type = kind  # type: ignore[assignment]
        else:
            captcha_type = _classify_captcha(selector)
            # §11.1 — when the coarse selector classifier says "recaptcha",
            # run the precise variant classifier so v3 / v2-invisible /
            # Enterprise widgets get the right provider task type. Falls
            # back to ``captcha_type`` when the classifier returns
            # ``unknown`` (e.g. the registry isn't accessible from this
            # frame); legacy selector-based behavior is preserved.
            if captcha_type == "recaptcha":
                classified = await _classify_recaptcha(page)
                variant = classified.get("variant") or "unknown"
                if variant != "unknown":
                    captcha_type = variant
                sitekey = classified.get("sitekey")
                page_action = classified.get("action")
        logger.info(
            "Attempting to solve %s CAPTCHA on %s",
            captcha_type,
            redact_url(page_url),
        )

        if antibot_path:
            # Anti-bot kinds don't use a sitekey — set to empty string and
            # skip the DOM extractor. ``_build_task_body`` omits
            # ``websiteKey`` for these kinds (see the body-builder branch
            # gated on ``_ANTIBOT_KINDS``).
            sitekey = ""
        else:
            if not sitekey:
                sitekey = await self._extract_sitekey(page, captcha_type)
            if not sitekey:
                logger.warning(
                    "Could not extract sitekey for %s CAPTCHA", captcha_type,
                )
                # LOCAL failure — sitekey couldn't be extracted from the
                # page DOM. The provider was never contacted, so the
                # §11.16 breaker MUST NOT count this. Pre-fix, three
                # sitekey-extraction failures from a single agent (e.g.
                # an unsupported widget variant the DOM-extractor doesn't
                # recognize) tripped the breaker for the entire
                # BrowserManager and blocked real solves for every other
                # agent. The breaker tracks PROVIDER reliability — local
                # classifier gaps belong elsewhere.
                return SolveResult(
                    token=None, injection_succeeded=False,
                    used_proxy_aware=False, compat_rejected=False,
                )

        # §11.2 — load the dedicated solver-proxy config (NOT the agent's
        # primary egress proxy; see ``get_solver_proxy_config`` docstring).
        # Loader returns ``None`` when unset → proxyless task types.
        proxy_config = get_solver_proxy_config(agent_id=agent_id)

        # §11.9 — pick the overall solve deadline based on the kind the
        # caller already classified (CF-bound Turnstile, etc.) when
        # available, falling back to the local ``captcha_type`` for direct
        # callers that don't have envelope context. The httpx per-request
        # 30s timeout (set on the shared client) stays — this controls only
        # the outer ``asyncio.wait_for`` deadline on the whole pipeline.
        timeout_kind = kind if kind is not None else captcha_type
        timeout_s = self._timeout_seconds_for_kind(timeout_kind)

        try:
            (
                token,
                used_proxy_aware,
                compat_rejected,
                provider_contacted,
            ) = await asyncio.wait_for(
                self._submit_and_poll(
                    captcha_type, sitekey, page_url,
                    page_action=page_action, proxy_config=proxy_config,
                    kind=timeout_kind,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "CAPTCHA solve timed out after %.1fs (kind=%s)",
                timeout_s, timeout_kind,
            )
            # ``asyncio.wait_for`` cancelled ``_submit_and_poll`` mid-flight
            # — by the time the outer deadline fires, ``createTask`` has
            # already been issued (the task-body builder + provider HTTP
            # call complete in well under a second; the deadline only
            # bites during the poll loop). Treat as provider-contacted.
            #
            # §22 — anti-bot kinds skip the breaker tick. Failure rates
            # for Akamai BMP / Imperva / Kasada / DataDome are inherently
            # high (token-IP / fingerprint binding rejections at the
            # application layer); a burst of legitimate-but-rejected
            # anti-bot solves shouldn't lock the whole fleet's solver
            # out of the standard CAPTCHA path for 10 minutes.
            if not antibot_path:
                await self._record_solver_outcome(success=False)
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            )
        except Exception as exc:
            # Redact the exception text — provider error responses
            # occasionally echo ``clientKey`` back. Pair with
            # :func:`redact_url` for any URL-shaped data in ``repr(exc)``.
            logger.warning(
                "CAPTCHA solve failed: %s",
                _redact_clientkey_text(redact_url(repr(exc))),
            )
            # An exception bubbling out of ``_submit_and_poll`` is rare —
            # both per-provider helpers catch and convert into ``None``
            # tokens. When it does happen, conservatively treat it as
            # provider-contacted so a programmer-error flood doesn't
            # silently mask a real provider outage. Anti-bot kinds still
            # skip the tick (§22 rationale above).
            if not antibot_path:
                await self._record_solver_outcome(success=False)
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            )

        if not token:
            # Token retrieval failed. Only count the breaker when a real
            # provider request was issued. ``provider_contacted=False``
            # means ``_build_task_body`` rejected the variant locally
            # (no row in the per-provider task table) — the captcha is
            # unsupported, NOT a provider outage signal.
            #
            # ``_solver_unreachable`` flipping mid-call means the provider
            # returned a fatal-config error (revoked key, drained balance,
            # banned account) — see :func:`_is_fatal_provider_error`.
            # Those need operator intervention, not breaker backoff;
            # skipping the tick keeps a misconfigured key from locking
            # the whole BrowserManager out of solver duty for 10 minutes.
            #
            # §22 — anti-bot kinds skip the tick by design (see the
            # exception-path comment block above for the rationale).
            if (
                provider_contacted
                and not self._solver_unreachable
                and not antibot_path
            ):
                await self._record_solver_outcome(success=False)
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=used_proxy_aware,
                compat_rejected=compat_rejected,
            )

        # §11.11 — solve-pacing Gaussian delay between solver token
        # retrieval and DOM injection. Real users take 5-15s between a
        # captcha appearing and a form submit; instant token injection is
        # a low-but-real anti-bot signal. Only fires on the success path —
        # failed solves (no token, timeout, exception) skip the pacing
        # because there's nothing to inject.
        await timing.captcha_solve_delay()

        injected = await self._inject_token(page, captcha_type, token)
        if injected:
            logger.info(
                "CAPTCHA solved and token injected successfully "
                "(proxy_aware=%s compat_rejected=%s)",
                used_proxy_aware, compat_rejected,
            )
        else:
            logger.warning("CAPTCHA solved but token injection failed")
        # Breaker tracks SOLVER reliability (provider returned a token), not
        # injection success. Provider was paid the moment the token came
        # back; injection failure is our problem, not theirs.
        await self._record_solver_outcome(success=True)
        return SolveResult(
            token=token,
            injection_succeeded=bool(injected),
            used_proxy_aware=used_proxy_aware,
            compat_rejected=compat_rejected,
        )

    async def _extract_sitekey(self, page, captcha_type: str) -> str | None:
        """Extract the sitekey from the page DOM."""
        # All reCAPTCHA variants share the same DOM-level sitekey markers
        # (``[data-sitekey]`` and the ``iframe[src*="recaptcha"]?k=…``
        # parameter); collapse the family to a single iframe-fallback path
        # so the v2-checkbox / v2-invisible / v3 / Enterprise variants
        # don't each need their own branch here.
        family = captcha_type
        if captcha_type.startswith("recaptcha"):
            family = "recaptcha"
        try:
            # Try data-sitekey attribute first (works for reCAPTCHA, hCaptcha, Turnstile)
            sitekey = await page.evaluate(
                "() => document.querySelector('[data-sitekey]')?.getAttribute('data-sitekey')"
            )
            if sitekey:
                return sitekey.strip()

            # Fall back to parsing iframe src for sitekey parameter
            if family == "recaptcha":
                src = await page.evaluate(
                    "() => document.querySelector('iframe[src*=\"recaptcha\"]')?.src"
                )
                if src:
                    match = re.search(r'[?&]k=([^&]+)', src)
                    if match:
                        return match.group(1)

            if family == "hcaptcha":
                src = await page.evaluate(
                    "() => document.querySelector('iframe[src*=\"hcaptcha\"]')?.src"
                )
                if src:
                    match = re.search(r'[?&]sitekey=([^&]+)', src)
                    if match:
                        return match.group(1)

            if family == "turnstile":
                # Turnstile sometimes stores config in a script or div attribute
                sitekey = await page.evaluate("""() => {
                    const el = document.querySelector('[class*="cf-turnstile"]');
                    return el?.getAttribute('data-sitekey') || null;
                }""")
                if sitekey:
                    return sitekey.strip()

        except Exception:
            logger.debug("Error extracting sitekey", exc_info=True)
        return None

    async def _submit_and_poll(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
        proxy_config: SolverProxyConfig | None = None,
        kind: str | None = None,
    ) -> tuple[str | None, bool, bool, bool]:
        """Submit CAPTCHA to solving service and poll for result.

        ``page_action`` is the v3 action string from the classifier; it's
        merged into the task body for v3 / Enterprise-v3 variants. v2
        variants ignore it. ``min_score`` is read inside the per-provider
        helpers from :data:`CAPTCHA_RECAPTCHA_V3_MIN_SCORE` (operator
        config, not a page property).

        ``kind`` is the §11.13 envelope kind (or the local ``captcha_type``
        when the caller doesn't have envelope context). Used to size the
        polling-loop iteration count via the §11.9 per-type timeout
        table — the outer ``asyncio.wait_for`` in ``solve()`` is the
        authoritative deadline; the iteration cap here just prevents an
        infinite loop when a provider stays in ``processing`` past the
        outer deadline.

        Returns ``(token, used_proxy_aware, compat_rejected, provider_contacted)``:

          * ``token`` — provider-issued solution, or ``None`` on any
            failure (local task-body rejection, createTask error, poll
            error, errorId>0, never-ready).
          * ``used_proxy_aware`` / ``compat_rejected`` — see
            ``_build_task_body``.
          * ``provider_contacted`` — ``True`` iff a ``createTask`` HTTP
            request was actually attempted. ``False`` for purely-local
            failures (unknown ``captcha_type`` not in the provider table,
            ``_build_task_body`` returning ``None``). The §11.16 circuit
            breaker MUST only count provider-contacted failures —
            otherwise three unsupported-variant requests from one agent
            trip the breaker for the whole BrowserManager and block real
            solves for every other agent.
        """
        if self.provider == "2captcha":
            return await self._solve_2captcha(
                captcha_type, sitekey, page_url,
                page_action=page_action, proxy_config=proxy_config,
                kind=kind,
            )
        return await self._solve_capsolver(
            captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
            kind=kind,
        )

    # ── Task-body builder ─────────────────────────────────────────────────────

    def _build_task_body(
        self,
        provider_table: dict[str, dict[str, object]],
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None,
        proxy_config: SolverProxyConfig | None = None,
        provider_name: str | None = None,
    ) -> tuple[dict | None, bool, bool]:
        """Merge provider task-type fields with v3 + proxy extras.

        Returns ``(body, used_proxy_aware, compat_rejected)``:

        * ``body`` is the JSON ``task`` dict, or ``None`` if
          ``captcha_type`` isn't in the provider table (caller treats as
          "unsupported").
        * ``used_proxy_aware`` is ``True`` iff the body uses the
          ``proxy_aware`` task name and includes the ``proxyType/...``
          credential fields. ``False`` for the proxyless fallback.
        * ``compat_rejected`` is ``True`` iff a proxy was configured but
          the (provider, variant, type) tuple was rejected by
          :data:`_SOLVER_PROXY_COMPAT`. Callers that get back
          ``(body, False, True)`` should still send the proxyless body
          AND set ``solver_confidence="low"`` on the envelope so the
          operator sees that the dedicated proxy isn't being honored
          for this variant.

        For v3 and v3-enterprise the function reads the operator-configured
        ``CAPTCHA_RECAPTCHA_V3_MIN_SCORE`` flag (default 0.7) and applies
        the permissive ``"verify"`` fallback when ``page_action`` is
        missing — see :data:`_DEFAULT_V3_ACTION` for the rationale.

        SECURITY: ``proxy_config`` MUST be a :class:`SolverProxyConfig`
        loaded via :func:`get_solver_proxy_config` from the dedicated
        ``CAPTCHA_SOLVER_PROXY_*`` env-var family. The agent's primary
        egress proxy (per-agent state in
        ``BrowserManager._proxy_configs``) is NEVER threaded into this
        path — see ``test_captcha_solver_proxy.py::test_primary_proxy_creds_never_leak``.
        """
        entry = provider_table.get(captcha_type)
        if not entry:
            return None, False, False
        # Tolerate the legacy flat shape (``{"type": ..., "isInvisible": ...}``)
        # in case third-party subclasses haven't migrated. The §11.2 shape
        # is ``{"proxyless": ..., "proxy_aware": ..., "extra": {...}}``.
        if "proxyless" in entry:
            proxyless_name = entry.get("proxyless")
            proxy_aware_name = entry.get("proxy_aware")
            static_extras = dict(entry.get("extra") or {})
        else:
            proxyless_name = entry.get("type")
            proxy_aware_name = None
            static_extras = {k: v for k, v in entry.items() if k != "type"}

        # §22 — anti-bot platform task types don't use ``websiteKey``
        # (no DOM sitekey marker on Akamai BMP / Imperva / Kasada /
        # DataDome challenges; those tasks identify the site by URL +
        # the proxy-bound IP fingerprint). Including an empty
        # ``websiteKey`` field would cause CapSolver to reject the task
        # with ``ERROR_KEY_MUST_NOT_BE_EMPTY``; omit the field entirely
        # for anti-bot kinds. ``sitekey`` is always ``""`` for these
        # variants because :meth:`solve` skips DOM extraction and passes
        # an empty string in.
        body: dict[str, object] = {
            "websiteURL": page_url,
        }
        if captcha_type not in _ANTIBOT_KINDS:
            body["websiteKey"] = sitekey
        # Merge static extras (``isInvisible``, ``isEnterprise``).
        for k, v in static_extras.items():
            body[k] = v

        # v3 extras — applied to both standard and enterprise v3 task entries.
        is_v3 = captcha_type in ("recaptcha-v3", "recaptcha-enterprise-v3")
        if is_v3:
            # Operator-configured min score. ``flags.get_float`` is
            # contracted not to raise — bad values fall back to the
            # default at the flag-helper level. The previous defensive
            # try/except was swallowing programmer errors in the flag
            # module silently; let any future bug there surface
            # immediately.
            from src.browser import flags as _flags
            min_score = _flags.get_float(
                "CAPTCHA_RECAPTCHA_V3_MIN_SCORE",
                _DEFAULT_V3_MIN_SCORE,
                min_value=0.1,
                max_value=0.9,
            )
            body["minScore"] = min_score
            action = (page_action or "").strip()
            if not action:
                logger.warning(
                    "v3 reCAPTCHA solve: pageAction missing; falling back to "
                    "%r — token may be bound to the wrong action and rejected "
                    "by the target server (kind=%s sitekey=%s)",
                    _DEFAULT_V3_ACTION, captcha_type, sitekey[:8] + "…",
                )
                action = _DEFAULT_V3_ACTION
            body["pageAction"] = action

        # ── §11.2 proxy-aware selection ────────────────────────────────
        used_proxy_aware = False
        compat_rejected = False
        provider = (provider_name or self.provider).lower()
        if proxy_config is not None:
            if proxy_aware_name is None:
                # No documented proxy-aware task for this variant
                # (e.g. 2captcha v3). Fall back to proxyless and treat
                # as a soft compat rejection so the envelope reflects
                # reduced confidence.
                compat_rejected = True
                logger.warning(
                    "Solver proxy configured but provider=%s has no "
                    "documented proxy-aware task for variant=%s — "
                    "falling back to proxyless task type. "
                    "solver_confidence will be downgraded.",
                    provider, captcha_type,
                )
            elif _solver_proxy_compatible(
                provider, captcha_type, proxy_config.proxy_type,
            ):
                used_proxy_aware = True
                body["type"] = proxy_aware_name
                # Inject credential fields into the task body.
                body.update(proxy_config.to_request_fields())
            else:
                compat_rejected = True
                logger.warning(
                    "Solver proxy type=%s not in compat-table for "
                    "provider=%s variant=%s — falling back to proxyless. "
                    "solver_confidence will be downgraded. Configure a "
                    "different proxy scheme (compat: %s) or accept "
                    "proxyless for this variant.",
                    proxy_config.proxy_type, provider, captcha_type,
                    sorted(_SOLVER_PROXY_COMPAT.get(
                        (provider, captcha_type), set(),
                    )),
                )

        if not used_proxy_aware:
            if not proxyless_name:
                # Defensive — every entry should declare a proxyless name.
                return None, False, compat_rejected
            body["type"] = proxyless_name

        return body, used_proxy_aware, compat_rejected

    # ── 2Captcha ──────────────────────────────────────────────────────────────

    async def _solve_2captcha(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
        proxy_config: SolverProxyConfig | None = None,
        kind: str | None = None,
    ) -> tuple[str | None, bool, bool, bool]:
        client = self._get_client()
        task, used_proxy_aware, compat_rejected = self._build_task_body(
            _2CAPTCHA_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
            provider_name="2captcha",
        )
        if not task:
            # Local failure — the variant isn't in the provider table.
            # ``provider_contacted=False`` so the breaker is NOT polluted
            # by unsupported-variant requests (see ``_submit_and_poll``
            # docstring for the full rationale).
            return None, used_proxy_aware, compat_rejected, False

        # Submit task
        payload = {
            "clientKey": self.api_key,
            "task": task,
        }
        # Mark provider_contacted=True the moment we attempt the HTTP
        # call. From here every failure path is a real provider
        # interaction (network error, errorId>0, never-ready) and the
        # breaker SHOULD count it.
        try:
            resp = await client.post("https://api.2captcha.com/createTask", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Provider error responses sometimes echo ``clientKey`` back
            # in the body / exception text. Strip before logging — the
            # bundled ``_redact_clientkey_text`` is the single redactor
            # for these strings (do NOT introduce a parallel one).
            logger.warning(
                "2Captcha createTask failed: %s",
                _redact_clientkey_text(redact_url(str(e))),
            )
            return None, used_proxy_aware, compat_rejected, True
        if data.get("errorId", 0) != 0:
            self._handle_provider_error_response(data)
            return None, used_proxy_aware, compat_rejected, True
        task_id = data.get("taskId")
        if not task_id:
            return None, used_proxy_aware, compat_rejected, True

        # Poll for result. The iteration cap is the §11.9 per-type
        # timeout in seconds divided by the poll interval; the outer
        # ``asyncio.wait_for`` in :meth:`solve` is the authoritative
        # deadline (it cancels the loop mid-iteration when the budget
        # expires) — this loop just bounds the case where a provider
        # never returns "ready" and the task is invoked outside ``solve``
        # (test harnesses, future direct callers).
        max_iterations = max(1, int(self._timeout_seconds_for_kind(kind) / _POLL_INTERVAL))
        for _ in range(max_iterations):
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                resp = await client.post(
                    "https://api.2captcha.com/getTaskResult",
                    json={"clientKey": self.api_key, "taskId": task_id},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(
                    "2Captcha getTaskResult failed: %s",
                    _redact_clientkey_text(redact_url(str(e))),
                )
                return None, used_proxy_aware, compat_rejected, True
            if data.get("errorId", 0) != 0:
                self._handle_provider_error_response(data)
                return None, used_proxy_aware, compat_rejected, True
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                token = _extract_solution_token(solution, captcha_type)
                return token, used_proxy_aware, compat_rejected, True
            # status == "processing" — keep polling
        return None, used_proxy_aware, compat_rejected, True

    # ── CapSolver ─────────────────────────────────────────────────────────────

    async def _solve_capsolver(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
        proxy_config: SolverProxyConfig | None = None,
        kind: str | None = None,
    ) -> tuple[str | None, bool, bool, bool]:
        client = self._get_client()
        task, used_proxy_aware, compat_rejected = self._build_task_body(
            _CAPSOLVER_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
            provider_name="capsolver",
        )
        if not task:
            # Local failure — see ``_solve_2captcha`` for breaker rationale.
            return None, used_proxy_aware, compat_rejected, False

        # Submit task
        payload = {
            "clientKey": self.api_key,
            "task": task,
        }
        try:
            resp = await client.post("https://api.capsolver.com/createTask", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(
                "CapSolver createTask failed: %s",
                _redact_clientkey_text(redact_url(str(e))),
            )
            return None, used_proxy_aware, compat_rejected, True
        if data.get("errorId", 0) != 0:
            self._handle_provider_error_response(data)
            return None, used_proxy_aware, compat_rejected, True
        task_id = data.get("taskId")
        if not task_id:
            return None, used_proxy_aware, compat_rejected, True

        # Poll for result. See ``_solve_2captcha`` for why the loop bound
        # is sized off the §11.9 per-type table.
        max_iterations = max(1, int(self._timeout_seconds_for_kind(kind) / _POLL_INTERVAL))
        for _ in range(max_iterations):
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                resp = await client.post(
                    "https://api.capsolver.com/getTaskResult",
                    json={"clientKey": self.api_key, "taskId": task_id},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(
                    "CapSolver getTaskResult failed: %s",
                    _redact_clientkey_text(redact_url(str(e))),
                )
                return None, used_proxy_aware, compat_rejected, True
            if data.get("errorId", 0) != 0:
                self._handle_provider_error_response(data)
                return None, used_proxy_aware, compat_rejected, True
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                token = _extract_solution_token(solution, captcha_type)
                return token, used_proxy_aware, compat_rejected, True
        return None, used_proxy_aware, compat_rejected, True

    # ── Token injection ───────────────────────────────────────────────────────

    async def _inject_token(self, page, captcha_type: str, token: str) -> bool:
        """Inject the solved CAPTCHA token into the page.

        ``captcha_type`` is matched against the variant *family* (``recaptcha``,
        ``hcaptcha``, ``turnstile``); the §11.1 reCAPTCHA variants
        (``recaptcha-v2-checkbox`` / ``...-v3`` / ``...-enterprise-v2`` / etc.)
        all share the same ``g-recaptcha-response`` injection path.

        Iframe handling: hCaptcha and Turnstile render their
        ``<input name="*-response">`` element INSIDE the widget iframe.
        Running injection JS only in the top-level document would miss
        the response input on widgets that don't auto-mirror to a
        parent textarea, returning ``injection_failed`` even though
        the provider returned a valid token. Walk every frame and
        bubble up "any frame succeeded" as the success bit.
        """
        family = captcha_type
        if captcha_type.startswith("recaptcha"):
            family = "recaptcha"

        async def _eval_in_all_frames(js: str) -> list:
            """Run ``js`` in every frame; return the list of per-frame
            return values. Failures (cross-origin frame, frame detached,
            ...) are swallowed per-frame so one bad frame doesn't
            abort the rest of the walk.

            Always evaluates against ``page`` first (the top-level
            document) so existing single-frame tests/mocks still see
            their evaluate call. Then walks ``page.frames`` if
            available; cross-origin / mock objects that don't yield a
            real iterable degrade cleanly to the page-only path.
            """
            results: list = []
            # Always try the top-level page first — covers single-frame
            # mocks in tests and the common case where the response
            # input is mirrored at the top level.
            try:
                results.append(await page.evaluate(js, token))
            except Exception:
                pass
            # Walk the frame tree if Playwright exposes one. ``frames``
            # may be missing (test mocks), non-iterable (AsyncMock
            # auto-attr), or include the main frame already covered
            # above; iterating defensively skips all of those. Playwright
            # includes ``page.main_frame`` in ``page.frames``; skip it so
            # callbacks do not fire twice in the top-level document.
            frames = getattr(page, "frames", None)
            try:
                frame_iter = list(frames) if frames is not None else []
            except TypeError:
                frame_iter = []
            main_frame = getattr(page, "main_frame", None)
            for frame in frame_iter:
                if frame is page or frame is main_frame:
                    continue
                evaluator = getattr(frame, "evaluate", None)
                if not callable(evaluator):
                    continue
                try:
                    results.append(await evaluator(js, token))
                except Exception:
                    # Cross-origin / detached / SecurityError — common
                    # for the captcha provider's iframe origin. Don't
                    # propagate; one bad frame shouldn't kill the walk.
                    continue
            return results

        try:
            if family == "recaptcha":
                results = await _eval_in_all_frames("""(token) => {
                    let updated = false;
                    const fire = (el) => {
                        try {
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        } catch (e) {}
                    };
                    const textarea = document.getElementById('g-recaptcha-response');
                    if (textarea) {
                        textarea.style.display = '';
                        textarea.value = token;
                        fire(textarea);
                        updated = true;
                    }
                    // Also try hidden textareas in iframes
                    document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {
                        el.value = token;
                        fire(el);
                        updated = true;
                    });
                    // Trigger callback if available
                    if (typeof ___grecaptcha_cfg !== 'undefined') {
                        const clients = ___grecaptcha_cfg.clients;
                        if (clients) {
                            for (const cid in clients) {
                                const client = clients[cid];
                                // Walk the client object to find the callback
                                const walk = (obj, depth) => {
                                    if (depth > 5 || !obj) return;
                                    for (const key in obj) {
                                        if (typeof obj[key] === 'function' && key === 'callback') {
                                            try {
                                                obj[key](token);
                                                updated = true;
                                            } catch (e) {}
                                            return;
                                        }
                                        if (typeof obj[key] === 'object') walk(obj[key], depth + 1);
                                    }
                                };
                                walk(client, 0);
                            }
                        }
                    }
                    return updated;
                }""")
                return any(bool(r) for r in results)

            if family == "hcaptcha":
                # hCaptcha widget renders the response input inside its
                # iframe. Set the textarea BEFORE calling
                # ``hcaptcha.execute()`` would risk the SDK re-running
                # the challenge and clobbering our token; we just set
                # the textarea + dispatch input/change events and let
                # the embedding form's submit handler pick it up.
                # Real-world flow: provider verifies the token
                # server-side from the form post; the SDK callback is
                # not strictly required.
                # hCaptcha exposes the response token under
                # ``[name="h-captcha-response"]`` only. The previous
                # implementation also wrote into
                # ``[name="g-recaptcha-response"]`` — that's a pure
                # cross-family mistake (g-recaptcha-response is a
                # reCAPTCHA field; touching it from the hCaptcha branch
                # leaks tokens across families on pages embedding both
                # widgets and silently flips ``updated=true`` whenever
                # an unrelated reCAPTCHA field exists).
                results = await _eval_in_all_frames("""(token) => {
                    let updated = false;
                    const fire = (el) => {
                        try {
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        } catch (e) {}
                    };
                    document.querySelectorAll('[name="h-captcha-response"]').forEach(el => {
                        el.value = token;
                        fire(el);
                        updated = true;
                    });
                    return updated;
                }""")
                return any(bool(r) for r in results)

            if family == "turnstile":
                # Turnstile renders the response input inside the
                # widget iframe; the embedding form often holds a
                # mirror ``[name="cf-turnstile-response"]`` at the top
                # level too. Walk both so we hit whichever variant the
                # site uses.
                # Turnstile selector tightening: only fire on the
                # canonical ``[name="cf-turnstile-response"]`` field
                # AND only when the page actually carries Turnstile
                # widget context (a ``.cf-turnstile``/``[class*=cf-turnstile]``
                # ancestor or a Cloudflare-challenge iframe). The prior
                # ``input[name*="turnstile"]`` substring fallback was a
                # false-positive vector — A/B test flag inputs and
                # marketing pixels containing ``turnstile`` substrings
                # in unrelated pages would return ``updated=true`` and
                # let us bill the user for a "successful" injection
                # that landed nowhere.
                per_frame = await _eval_in_all_frames("""(token) => {
                    let updated = false;
                    let widget_count = 0;
                    const fire = (el) => {
                        try {
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        } catch (e) {}
                    };
                    const turnstile_iframe = document.querySelector(
                        'iframe[src*="challenges.cloudflare.com"]'
                    );
                    const inputs = document.querySelectorAll(
                        '[name="cf-turnstile-response"]'
                    );
                    for (const input of inputs) {
                        // Confirm Turnstile widget context: the input
                        // is inside (or adjacent to) a ``.cf-turnstile``
                        // wrapper, OR a Cloudflare challenge iframe is
                        // on the page. Without context the field is
                        // probably a coincidentally-named hidden input.
                        const widget = input.closest(
                            '.cf-turnstile, [class*="cf-turnstile"]'
                        );
                        if (widget || turnstile_iframe) {
                            input.value = token;
                            fire(input);
                            updated = true;
                        }
                    }
                    // Trigger callback if available. On pages with
                    // multiple Turnstile widgets we fire the callback
                    // for ALL of them — picking just the first widget
                    // (the prior behavior) silently routed the solved
                    // token to the wrong form on multi-form pages.
                    if (typeof turnstile !== 'undefined') {
                        try {
                            const widgets = turnstile._widgets || {};
                            const ids = Object.keys(widgets);
                            widget_count = ids.length;
                            for (const id of ids) {
                                const cb = widgets[id]?.callback;
                                if (typeof cb === 'function') {
                                    try { cb(token); updated = true; }
                                    catch (e) {}
                                }
                            }
                        } catch(e) {}
                    }
                    return { updated: updated, widget_count: widget_count };
                }""")
                total_widgets = sum(
                    int(r.get("widget_count", 0))
                    for r in per_frame
                    if isinstance(r, dict)
                )
                if total_widgets > 1:
                    logger.info(
                        "Turnstile injection saw %d widgets across all "
                        "frames; fired callback on all (prior behavior "
                        "fired only the first widget)",
                        total_widgets,
                    )
                return any(
                    bool(r.get("updated") if isinstance(r, dict) else r)
                    for r in per_frame
                )

        except Exception:
            logger.debug("Token injection error", exc_info=True)
        return False


# ── §11.8 multi-provider failover wrapper ──────────────────────────────────


class MultiProviderSolver:
    """Failover wrapper around two :class:`CaptchaSolver` instances.

    Public surface intentionally mirrors :class:`CaptchaSolver` so the
    BrowserManager's :meth:`_metered_solve` and the up-stream gates in
    ``service.py`` keep working without changes:

      * :attr:`provider` — *active* solver's provider name (the one used
        for the most recent solve, or the one that WILL be used when read
        before a solve). Drives cost-counter pricing tier lookups.
      * :meth:`solve` — async; routes to primary unless primary is
        unreachable / breaker-open, then secondary. Mid-call fatal
        primary failures (``token=None`` + ``_solver_unreachable`` newly
        flipped) trigger one transparent retry on the secondary.
      * :meth:`is_solver_unreachable` — async; True iff BOTH solvers are
        unreachable (single-provider config: True iff primary is).
      * :meth:`is_breaker_open` — sync; True iff BOTH breakers are open.
      * :meth:`health_check` — runs both, returns the worst outcome
        (``unreachable`` > ``degraded`` > ``healthy``).
      * :meth:`close` — closes both underlying clients.

    State isolation: each underlying :class:`CaptchaSolver` keeps its own
    breaker, health-check flag, and failure window. A fatal-config error
    on the primary does not affect the secondary, and vice versa — that
    independence is the entire point of the wrapper.

    Single-provider deployments (``secondary=None``) get a transparent
    pass-through: every method behaves exactly as if the primary were
    being used directly. Failover paths short-circuit cheaply.
    """

    def __init__(
        self,
        primary: CaptchaSolver,
        secondary: CaptchaSolver | None,
    ) -> None:
        self.primary: CaptchaSolver = primary
        self.secondary: CaptchaSolver | None = secondary
        # The provider name that will be used for the NEXT solve under
        # current state. Mutated by :meth:`_pick_solver` before a solve
        # and again on mid-call failover so ``self.provider`` returns
        # the right tier for cost accounting reads in
        # :meth:`BrowserManager._metered_solve`.
        self._active_solver: CaptchaSolver = primary

    # ── pricing-tier surface for ``_metered_solve`` ─────────────────────

    @property
    def provider(self) -> str:
        """Active solver's provider name.

        Read by ``_metered_solve`` BEFORE :meth:`solve` to pick the
        cost-cap reservation tier and AFTER :meth:`solve` to pick the
        accounting tier. The wrapper updates ``_active_solver`` based on
        current health state in :meth:`_pick_solver` so a pre-solve read
        sees the solver that will actually be used.
        """
        return self._active_solver.provider

    # ── routing ─────────────────────────────────────────────────────────

    async def _pick_solver(
        self, kind: str | None = None,
    ) -> CaptchaSolver | None:
        """Choose the solver for the next solve attempt.

        Returns the secondary when the primary is unreachable OR has an
        open breaker; otherwise returns the primary. Returns ``None``
        when both are unavailable — caller surfaces a no-token result
        without contacting either provider.

        §22 — when ``kind`` names an anti-bot platform task type that the
        primary doesn't support (e.g. primary is 2Captcha, kind is
        ``js-challenge-akamai``) the wrapper PREFERS the secondary even
        though the primary is healthy. This is the kind-aware routing
        the BrowserManager relies on so a 2Captcha-primary deployment
        with a CapSolver secondary still surfaces anti-bot solves
        through the secondary instead of routing to operator escalation.
        Without this branch, ``_pick_solver`` would always return the
        healthy primary and the per-provider task-table lookup inside
        ``_solve_2captcha`` would emit a no-token local-failure result.
        For non-anti-bot kinds and for kinds the primary supports, the
        existing primary-first routing is preserved.

        Side effect: updates ``self._active_solver`` so the
        :attr:`provider` property reads the right tier for cost
        accounting on the upcoming solve.
        """
        primary_unreachable = await self.primary.is_solver_unreachable()
        primary_breaker_open = self.primary.is_breaker_open()
        primary_skip = primary_unreachable or primary_breaker_open

        # §22 — kind-aware routing for anti-bot kinds. We only consult
        # the support table when ``kind`` is provided AND the primary
        # doesn't declare support; that keeps the existing primary-first
        # bias for every other code path (the prior failover branch's
        # tests assume an unconditional primary preference).
        kind_aware_route = (
            kind in _ANTIBOT_KINDS
            and not self.primary.supports_kind(kind)
            and self.secondary is not None
            and self.secondary.supports_kind(kind)
        )
        if kind_aware_route:
            secondary_unreachable = await self.secondary.is_solver_unreachable()
            secondary_breaker_open = self.secondary.is_breaker_open()
            if not (secondary_unreachable or secondary_breaker_open):
                self._active_solver = self.secondary
                return self.secondary
            # Secondary is the only provider that supports the kind, but
            # it's down. Surface the no-solver shape rather than letting
            # a doomed primary call generate an ``ERROR_INVALID_TASK_TYPE``
            # response that would tick its breaker.
            self._active_solver = self.primary
            return None

        if not primary_skip:
            self._active_solver = self.primary
            return self.primary

        if self.secondary is None:
            # Single-provider config — no failover available. Keep
            # ``_active_solver`` pointing at primary so the provider
            # property remains consistent with what the caller would
            # have seen pre-failover-feature.
            self._active_solver = self.primary
            return None

        secondary_unreachable = await self.secondary.is_solver_unreachable()
        secondary_breaker_open = self.secondary.is_breaker_open()
        if secondary_unreachable or secondary_breaker_open:
            # Both out — surface the same "no solver path" semantics as
            # today's unreachable / breaker-open envelopes.
            self._active_solver = self.primary
            return None

        self._active_solver = self.secondary
        return self.secondary

    async def solve(
        self,
        page,
        selector: str,
        page_url: str,
        *,
        agent_id: str | None = None,
        kind: str | None = None,
    ) -> SolveResult:
        """Solve a CAPTCHA using primary, falling over to secondary on failure.

        Routing decisions:

          1. Primary healthy → call ``primary.solve()``. On a clean
             ``token is None`` AND ``_solver_unreachable`` newly flipped
             (commit 2e889bd's fatal-config gate), retry once on the
             secondary in the same call. The flag is sticky so all
             subsequent solves bypass primary.
          2. Primary unreachable / breaker-open → skip primary entirely,
             call ``secondary.solve()``.
          3. Both unreachable / breaker-open → return a no-token
             :class:`SolveResult` without contacting either provider,
             matching today's "no solver" envelope path.
        """
        target = await self._pick_solver(kind=kind)
        if target is None:
            # Both primary and secondary are unavailable. Mirror the
            # short-circuit shape used by the underlying solver's own
            # unreachable / breaker-open paths.
            logger.info(
                "Skipping solve: both solvers unavailable "
                "(primary=%s secondary=%s)",
                self.primary.provider,
                self.secondary.provider if self.secondary else "<unset>",
            )
            return SolveResult(
                token=None, injection_succeeded=False,
                used_proxy_aware=False, compat_rejected=False,
            )

        result = await target.solve(
            page, selector, page_url,
            agent_id=agent_id, kind=kind,
        )

        # §11.8 mid-call failover: primary's fatal-config gate
        # (``_handle_provider_error_response`` from commit 2e889bd) flips
        # ``_solver_unreachable`` AFTER the provider HTTP call has already
        # returned a no-token outcome. The pre-solve gate in
        # :meth:`_pick_solver` couldn't see that yet. When it happens on
        # the primary AND a secondary is configured, retry once.
        if (
            target is self.primary
            and result.token is None
            and self.primary._solver_unreachable
            and self.secondary is not None
        ):
            secondary_unreachable = await self.secondary.is_solver_unreachable()
            secondary_breaker_open = self.secondary.is_breaker_open()
            if not (secondary_unreachable or secondary_breaker_open):
                logger.warning(
                    "Primary solver flipped unreachable mid-call; "
                    "retrying on secondary (primary=%s secondary=%s)",
                    self.primary.provider, self.secondary.provider,
                )
                self._active_solver = self.secondary
                return await self.secondary.solve(
                    page, selector, page_url,
                    agent_id=agent_id, kind=kind,
                )

        return result

    # ── health surface ──────────────────────────────────────────────────

    async def is_solver_unreachable(self) -> bool:
        """True iff BOTH solvers are unreachable.

        Drives the pre-solve gate in :meth:`_check_captcha`. Returning
        True yields the existing ``no_solver`` envelope path;
        single-provider deployments collapse to "primary unreachable".
        """
        if not await self.primary.is_solver_unreachable():
            return False
        if self.secondary is None:
            return True
        return await self.secondary.is_solver_unreachable()

    def is_breaker_open(self) -> bool:
        """True iff BOTH breakers are open.

        Drives the pre-solve breaker gate in :meth:`_check_captcha`.
        Single-provider deployments collapse to "primary breaker open".
        """
        if not self.primary.is_breaker_open():
            return False
        if self.secondary is None:
            return True
        return self.secondary.is_breaker_open()

    def supports_kind(self, kind: str | None) -> bool:
        """True iff at least one underlying solver supports ``kind``.

        Mirrors :meth:`CaptchaSolver.supports_kind` but folds across both
        primary + secondary so an operator with CapSolver as the secondary
        can still route §22 anti-bot kinds (Akamai BMP / Imperva /
        Kasada / DataDome) to the solver even when 2Captcha is primary.
        Pre-solve gate semantics: when ``False`` the caller skips the
        solver path entirely and emits the existing operator-escalation
        envelope. Health gating (``is_solver_unreachable`` /
        ``is_breaker_open``) is checked separately in
        :meth:`_check_captcha`; this method only answers "does any
        configured provider declare a task type for this kind".
        """
        if self.primary.supports_kind(kind):
            return True
        if self.secondary is not None and self.secondary.supports_kind(kind):
            return True
        return False

    async def health_check(
        self, provider: str | None = None,
    ) -> Literal["healthy", "degraded", "unreachable"]:
        """Probe both solvers and return the worst outcome.

        Worst-case ranking: ``unreachable`` > ``degraded`` > ``healthy``.
        Operators reading this signal want to know whether at least one
        provider is reachable; ``healthy`` here means "both healthy",
        ``degraded`` means "at least one degraded but neither
        unreachable", ``unreachable`` means "both unreachable" — the
        only state where solving is truly unavailable.

        ``provider`` argument is accepted for parity with
        :meth:`CaptchaSolver.health_check` but ignored (the wrapper
        always probes both underlying solvers).
        """
        primary_outcome = await self.primary.health_check()
        if self.secondary is None:
            return primary_outcome
        secondary_outcome = await self.secondary.health_check()
        # Worst-case fold: any "unreachable" wins only when BOTH agree;
        # otherwise downgrade to the worse of the two non-unreachable
        # outcomes.
        if primary_outcome == "unreachable" and secondary_outcome == "unreachable":
            return "unreachable"
        if primary_outcome == "degraded" or secondary_outcome == "degraded":
            return "degraded"
        if primary_outcome == "unreachable" or secondary_outcome == "unreachable":
            # One side unreachable, the other healthy/degraded — surface
            # ``degraded`` so operators see the partial fault without
            # the wrapper claiming "totally unreachable".
            return "degraded"
        return "healthy"

    async def close(self) -> None:
        """Close both underlying solver clients."""
        await self.primary.close()
        if self.secondary is not None:
            await self.secondary.close()
