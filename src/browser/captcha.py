"""CAPTCHA-solving service integration.

Supports 2Captcha and CapSolver as solving providers.  Both are called via
their public HTTP APIs using httpx — no additional dependencies required.

When configured (via CAPTCHA_SOLVER_PROVIDER and CAPTCHA_SOLVER_KEY env vars),
the browser service will automatically attempt to solve CAPTCHAs detected
after navigation.  If solving fails or no solver is configured, the existing
fallback (ask user via VNC) is preserved.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections import deque
from typing import Literal

import httpx

from src.shared.redaction import redact_url
from src.shared.utils import setup_logging

logger = setup_logging("browser.captcha")

_SOLVE_TIMEOUT = 120  # max seconds to wait for a solution
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
# 2Captcha
_2CAPTCHA_TASK_TYPES: dict[str, dict[str, object]] = {
    "recaptcha": {"type": "NormalRecaptchaTaskProxyless"},  # legacy alias
    "hcaptcha": {"type": "HCaptchaTaskProxyless"},
    "turnstile": {"type": "TurnstileTaskProxyless"},
    # §11.1 reCAPTCHA variant matrix
    "recaptcha-v2-checkbox": {"type": "RecaptchaV2TaskProxyless"},
    "recaptcha-v2-invisible": {
        "type": "RecaptchaV2TaskProxyless",
        "isInvisible": True,
    },
    "recaptcha-v3": {"type": "RecaptchaV3TaskProxyless"},
    "recaptcha-enterprise-v2": {"type": "RecaptchaV2EnterpriseTaskProxyless"},
    # 2captcha uses the same v3 task with isEnterprise=true (no separate
    # ``RecaptchaV3EnterpriseTaskProxyless`` type as of April 2026).
    "recaptcha-enterprise-v3": {
        "type": "RecaptchaV3TaskProxyless",
        "isEnterprise": True,
    },
}

# CapSolver
_CAPSOLVER_TASK_TYPES: dict[str, dict[str, object]] = {
    "recaptcha": {"type": "ReCaptchaV2TaskProxyLess"},  # legacy alias
    "hcaptcha": {"type": "HCaptchaTaskProxyLess"},
    "turnstile": {"type": "AntiTurnstileTaskProxyLess"},
    # §11.1 reCAPTCHA variant matrix
    "recaptcha-v2-checkbox": {"type": "ReCaptchaV2TaskProxyLess"},
    "recaptcha-v2-invisible": {
        "type": "ReCaptchaV2TaskProxyLess",
        "isInvisible": True,
    },
    "recaptcha-v3": {"type": "ReCaptchaV3TaskProxyLess"},
    "recaptcha-enterprise-v2": {"type": "ReCaptchaV2EnterpriseTaskProxyLess"},
    "recaptcha-enterprise-v3": {"type": "ReCaptchaV3EnterpriseTaskProxyLess"},
}


# Default ``pageAction`` when the classifier couldn't extract one. Most
# v3 implementations accept a generic ``"verify"`` action — solving with
# the wrong action returns a token bound to the wrong action and the
# verifying server may still reject the score, but it's the best
# permissive fallback we have.
_DEFAULT_V3_ACTION = "verify"
_DEFAULT_V3_MIN_SCORE = 0.7


def get_solver() -> CaptchaSolver | None:
    """Create a CaptchaSolver from environment variables, or None if not configured."""
    provider = os.environ.get("CAPTCHA_SOLVER_PROVIDER", "").strip().lower()
    api_key = os.environ.get("CAPTCHA_SOLVER_KEY", "").strip()
    if not provider or not api_key:
        return None
    if provider not in _SUPPORTED_PROVIDERS:
        logger.warning(
            "Unknown CAPTCHA_SOLVER_PROVIDER %r (expected one of %s), solver disabled",
            provider,
            ", ".join(_SUPPORTED_PROVIDERS),
        )
        return None
    logger.info("CAPTCHA solver configured: provider=%s", provider)
    return CaptchaSolver(provider, api_key)


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
        self._state_lock: asyncio.Lock = asyncio.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ── §11.16 health check + circuit breaker ──────────────────────────

    def is_solver_unreachable(self) -> bool:
        """Sticky for the rest of the instance-session once health-check fails."""
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
        async with self._state_lock:
            if self._solver_health_checked:
                return
            outcome = await self.health_check()
            self._solver_health_checked = True
            if outcome == "unreachable":
                self._solver_unreachable = True
            elif outcome == "degraded":
                self._solver_health_degraded = True

    async def _record_solver_outcome(self, success: bool) -> None:
        """Update the breaker state after a solve attempt.

        On success: reset both the failure window and any tripped breaker.
        On failure: append a timestamp, prune entries older than the 5-min
        window, then trip the breaker if 3+ entries remain.
        """
        async with self._state_lock:
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

    async def solve(self, page, selector: str, page_url: str) -> bool:
        """Attempt to solve a CAPTCHA on the page.

        Args:
            page: Playwright page object.
            selector: The CSS selector that matched the CAPTCHA element.
            page_url: The current page URL.

        Returns:
            True if the CAPTCHA was solved and token injected, False
            otherwise. On unreachable solver / open breaker, returns False
            without issuing a provider HTTP call. Callers should consult
            :meth:`is_solver_unreachable` and :meth:`is_breaker_open` to
            distinguish those cases from a genuine solve failure.
        """
        # Per-process gate — runs at most once even under concurrent solves.
        await self._ensure_health_checked()

        if self._solver_unreachable:
            logger.info(
                "Skipping solve: solver marked unreachable for this session "
                "(provider=%s)", self.provider,
            )
            return False

        if self.is_breaker_open():
            logger.warning(
                "Skipping solve: solver circuit breaker open until %.0f (provider=%s)",
                self._solver_breaker_until, self.provider,
            )
            return False

        captcha_type = _classify_captcha(selector)
        # §11.1 — when the coarse selector classifier says "recaptcha", run
        # the precise variant classifier so v3 / v2-invisible / Enterprise
        # widgets get the right provider task type. Falls back to
        # ``captcha_type`` when the classifier returns ``unknown`` (e.g. the
        # registry isn't accessible from this frame); legacy selector-based
        # behavior is preserved.
        page_action: str | None = None
        sitekey: str | None = None
        if captcha_type == "recaptcha":
            classified = await _classify_recaptcha(page)
            variant = classified.get("variant") or "unknown"
            if variant != "unknown":
                captcha_type = variant
            sitekey = classified.get("sitekey")
            page_action = classified.get("action")
        logger.info("Attempting to solve %s CAPTCHA on %s", captcha_type, page_url)

        if not sitekey:
            sitekey = await self._extract_sitekey(page, captcha_type)
        if not sitekey:
            logger.warning("Could not extract sitekey for %s CAPTCHA", captcha_type)
            await self._record_solver_outcome(success=False)
            return False

        try:
            token = await asyncio.wait_for(
                self._submit_and_poll(
                    captcha_type, sitekey, page_url, page_action=page_action,
                ),
                timeout=_SOLVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("CAPTCHA solve timed out after %ds", _SOLVE_TIMEOUT)
            await self._record_solver_outcome(success=False)
            return False
        except Exception:
            logger.exception("CAPTCHA solve failed")
            await self._record_solver_outcome(success=False)
            return False

        if not token:
            await self._record_solver_outcome(success=False)
            return False

        injected = await self._inject_token(page, captcha_type, token)
        if injected:
            logger.info("CAPTCHA solved and token injected successfully")
        else:
            logger.warning("CAPTCHA solved but token injection failed")
        await self._record_solver_outcome(success=injected)
        return injected

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
    ) -> str | None:
        """Submit CAPTCHA to solving service and poll for result.

        ``page_action`` is the v3 action string from the classifier; it's
        merged into the task body for v3 / Enterprise-v3 variants. v2
        variants ignore it. ``min_score`` is read inside the per-provider
        helpers from :data:`CAPTCHA_RECAPTCHA_V3_MIN_SCORE` (operator
        config, not a page property).
        """
        if self.provider == "2captcha":
            return await self._solve_2captcha(
                captcha_type, sitekey, page_url, page_action=page_action,
            )
        return await self._solve_capsolver(
            captcha_type, sitekey, page_url, page_action=page_action,
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
    ) -> dict | None:
        """Merge provider task-type fields with v3-specific extras.

        Returns ``None`` if ``captcha_type`` isn't in the provider table
        (caller treats as "unsupported"). For v3 and v3-enterprise the
        function reads the operator-configured ``CAPTCHA_RECAPTCHA_V3_MIN_SCORE``
        flag (default 0.7) and applies the permissive ``"verify"`` fallback
        when ``page_action`` is missing — see :data:`_DEFAULT_V3_ACTION`
        for the rationale.
        """
        entry = provider_table.get(captcha_type)
        if not entry:
            return None
        body: dict[str, object] = {
            "websiteURL": page_url,
            "websiteKey": sitekey,
        }
        # Merge static extras (``isInvisible``, ``isEnterprise``, ``type``).
        for k, v in entry.items():
            body[k] = v
        # v3 extras — applied to both standard and enterprise v3 task entries.
        is_v3 = captcha_type in ("recaptcha-v3", "recaptcha-enterprise-v3")
        if is_v3:
            # Operator-configured min score. ``flags.get_float`` clamps;
            # we additionally clamp 0.1-0.9 to match the env-var doc.
            try:
                from src.browser import flags as _flags
                min_score = _flags.get_float(
                    "CAPTCHA_RECAPTCHA_V3_MIN_SCORE",
                    _DEFAULT_V3_MIN_SCORE,
                    min_value=0.1,
                    max_value=0.9,
                )
            except Exception:
                # Defensive: if flags import fails (shouldn't, but the
                # solver module is sometimes imported in test contexts
                # without ``src.browser`` initialized) fall back to the
                # default rather than aborting the solve.
                min_score = _DEFAULT_V3_MIN_SCORE
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
        return body

    # ── 2Captcha ──────────────────────────────────────────────────────────────

    async def _solve_2captcha(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
    ) -> str | None:
        client = self._get_client()
        task = self._build_task_body(
            _2CAPTCHA_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action,
        )
        if not task:
            return None

        # Submit task
        payload = {
            "clientKey": self.api_key,
            "task": task,
        }
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
            return None
        if data.get("errorId", 0) != 0:
            logger.warning(
                "2Captcha submit error: %s",
                _redact_clientkey_text(str(data.get("errorDescription"))),
            )
            return None
        task_id = data.get("taskId")
        if not task_id:
            return None

        # Poll for result
        for _ in range(int(_SOLVE_TIMEOUT / _POLL_INTERVAL)):
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
                return None
            if data.get("errorId", 0) != 0:
                logger.warning(
                    "2Captcha poll error: %s",
                    _redact_clientkey_text(str(data.get("errorDescription"))),
                )
                return None
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                return solution.get("gRecaptchaResponse") or solution.get("token")
            # status == "processing" — keep polling
        return None

    # ── CapSolver ─────────────────────────────────────────────────────────────

    async def _solve_capsolver(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
    ) -> str | None:
        client = self._get_client()
        task = self._build_task_body(
            _CAPSOLVER_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action,
        )
        if not task:
            return None

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
            return None
        if data.get("errorId", 0) != 0:
            logger.warning(
                "CapSolver submit error: %s",
                _redact_clientkey_text(str(data.get("errorDescription"))),
            )
            return None
        task_id = data.get("taskId")
        if not task_id:
            return None

        # Poll for result
        for _ in range(int(_SOLVE_TIMEOUT / _POLL_INTERVAL)):
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
                return None
            if data.get("errorId", 0) != 0:
                logger.warning(
                    "CapSolver poll error: %s",
                    _redact_clientkey_text(str(data.get("errorDescription"))),
                )
                return None
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                return solution.get("gRecaptchaResponse") or solution.get("token")
        return None

    # ── Token injection ───────────────────────────────────────────────────────

    async def _inject_token(self, page, captcha_type: str, token: str) -> bool:
        """Inject the solved CAPTCHA token into the page.

        ``captcha_type`` is matched against the variant *family* (``recaptcha``,
        ``hcaptcha``, ``turnstile``); the §11.1 reCAPTCHA variants
        (``recaptcha-v2-checkbox`` / ``...-v3`` / ``...-enterprise-v2`` / etc.)
        all share the same ``g-recaptcha-response`` injection path.
        """
        family = captcha_type
        if captcha_type.startswith("recaptcha"):
            family = "recaptcha"
        try:
            if family == "recaptcha":
                await page.evaluate("""(token) => {
                    const textarea = document.getElementById('g-recaptcha-response');
                    if (textarea) {
                        textarea.style.display = '';
                        textarea.value = token;
                    }
                    // Also try hidden textareas in iframes
                    document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {
                        el.value = token;
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
                                            obj[key](token);
                                            return;
                                        }
                                        if (typeof obj[key] === 'object') walk(obj[key], depth + 1);
                                    }
                                };
                                walk(client, 0);
                            }
                        }
                    }
                }""", token)
                return True

            if family == "hcaptcha":
                await page.evaluate("""(token) => {
                    const textarea = document.querySelector('[name="h-captcha-response"]');
                    if (textarea) textarea.value = token;
                    document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {
                        el.value = token;
                    });
                    // Trigger hcaptcha callback
                    if (typeof hcaptcha !== 'undefined' && hcaptcha.getRespKey) {
                        try { hcaptcha.execute(); } catch(e) {}
                    }
                }""", token)
                return True

            if family == "turnstile":
                await page.evaluate("""(token) => {
                    // Find the Turnstile response input
                    const input = document.querySelector('[name="cf-turnstile-response"]')
                        || document.querySelector('input[name*="turnstile"]');
                    if (input) input.value = token;
                    // Trigger callback if available
                    if (typeof turnstile !== 'undefined') {
                        try {
                            const widgetId = turnstile.getResponse ? null : Object.keys(turnstile._widgets || {})[0];
                            if (widgetId && turnstile._widgets[widgetId]?.callback) {
                                turnstile._widgets[widgetId].callback(token);
                            }
                        } catch(e) {}
                    }
                }""", token)
                return True

        except Exception:
            logger.debug("Token injection error", exc_info=True)
        return False
