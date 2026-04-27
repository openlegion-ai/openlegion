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
from dataclasses import dataclass
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
    "recaptcha": {
        "proxyless": "NormalRecaptchaTaskProxyless",  # legacy alias
        "proxy_aware": None,
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
        self._state_lock: asyncio.Lock = asyncio.Lock()
        # ── §11.2 last-solve proxy metadata ─────────────────────────────
        # ``solve()`` stamps these so the BrowserManager caller can pick
        # them up without changing the public ``solve() -> bool`` return
        # contract. ``last_used_proxy_aware`` reflects what was actually
        # sent (proxy-aware task name + creds vs proxyless); the cost
        # counter uses this to apply proxy-aware pricing. ``last_compat_rejected``
        # is True when a proxy was configured but the (provider, variant,
        # type) tuple rejected — caller downgrades ``solver_confidence``
        # to "low".
        self.last_used_proxy_aware: bool = False
        self.last_compat_rejected: bool = False

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

    async def solve(
        self,
        page,
        selector: str,
        page_url: str,
        *,
        agent_id: str | None = None,
    ) -> bool:
        """Attempt to solve a CAPTCHA on the page.

        Args:
            page: Playwright page object.
            selector: The CSS selector that matched the CAPTCHA element.
            page_url: The current page URL.
            agent_id: Optional — for per-agent override of solver-proxy
                config via :func:`flags.set_agent_override`.

        Returns:
            True if the CAPTCHA was solved and token injected, False
            otherwise. On unreachable solver / open breaker, returns False
            without issuing a provider HTTP call. Callers should consult
            :meth:`is_solver_unreachable` and :meth:`is_breaker_open` to
            distinguish those cases from a genuine solve failure.

        After every call (success or failure) the caller may inspect
        :attr:`last_used_proxy_aware` and :attr:`last_compat_rejected`
        to learn whether the dedicated solver proxy was applied (drives
        cost-counter pricing tier and ``solver_confidence`` downgrades).
        """
        # Reset the per-call metadata up front so a short-circuit return
        # (unreachable / breaker) doesn't carry stale flags into the
        # caller's envelope.
        self.last_used_proxy_aware = False
        self.last_compat_rejected = False

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

        # §11.2 — load the dedicated solver-proxy config (NOT the agent's
        # primary egress proxy; see ``get_solver_proxy_config`` docstring).
        # Loader returns ``None`` when unset → proxyless task types.
        proxy_config = get_solver_proxy_config(agent_id=agent_id)

        try:
            token, used_proxy_aware, compat_rejected = await asyncio.wait_for(
                self._submit_and_poll(
                    captcha_type, sitekey, page_url,
                    page_action=page_action, proxy_config=proxy_config,
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

        # Stamp metadata before returning so the caller can read the
        # actual task type that was sent.
        self.last_used_proxy_aware = used_proxy_aware
        self.last_compat_rejected = compat_rejected

        if not token:
            await self._record_solver_outcome(success=False)
            return False

        injected = await self._inject_token(page, captcha_type, token)
        if injected:
            logger.info(
                "CAPTCHA solved and token injected successfully "
                "(proxy_aware=%s compat_rejected=%s)",
                used_proxy_aware, compat_rejected,
            )
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
        proxy_config: SolverProxyConfig | None = None,
    ) -> tuple[str | None, bool, bool]:
        """Submit CAPTCHA to solving service and poll for result.

        ``page_action`` is the v3 action string from the classifier; it's
        merged into the task body for v3 / Enterprise-v3 variants. v2
        variants ignore it. ``min_score`` is read inside the per-provider
        helpers from :data:`CAPTCHA_RECAPTCHA_V3_MIN_SCORE` (operator
        config, not a page property).

        Returns ``(token, used_proxy_aware, compat_rejected)`` so the
        caller (``solve``) can mark the envelope's confidence and inform
        the cost counter whether proxy-aware pricing applies.
        """
        if self.provider == "2captcha":
            return await self._solve_2captcha(
                captcha_type, sitekey, page_url,
                page_action=page_action, proxy_config=proxy_config,
            )
        return await self._solve_capsolver(
            captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
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

        body: dict[str, object] = {
            "websiteURL": page_url,
            "websiteKey": sitekey,
        }
        # Merge static extras (``isInvisible``, ``isEnterprise``).
        for k, v in static_extras.items():
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
    ) -> tuple[str | None, bool, bool]:
        client = self._get_client()
        task, used_proxy_aware, compat_rejected = self._build_task_body(
            _2CAPTCHA_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
            provider_name="2captcha",
        )
        if not task:
            return None, used_proxy_aware, compat_rejected

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
            return None, used_proxy_aware, compat_rejected
        if data.get("errorId", 0) != 0:
            logger.warning(
                "2Captcha submit error: %s",
                _redact_clientkey_text(str(data.get("errorDescription"))),
            )
            return None, used_proxy_aware, compat_rejected
        task_id = data.get("taskId")
        if not task_id:
            return None, used_proxy_aware, compat_rejected

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
                return None, used_proxy_aware, compat_rejected
            if data.get("errorId", 0) != 0:
                logger.warning(
                    "2Captcha poll error: %s",
                    _redact_clientkey_text(str(data.get("errorDescription"))),
                )
                return None, used_proxy_aware, compat_rejected
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                token = solution.get("gRecaptchaResponse") or solution.get("token")
                return token, used_proxy_aware, compat_rejected
            # status == "processing" — keep polling
        return None, used_proxy_aware, compat_rejected

    # ── CapSolver ─────────────────────────────────────────────────────────────

    async def _solve_capsolver(
        self,
        captcha_type: str,
        sitekey: str,
        page_url: str,
        *,
        page_action: str | None = None,
        proxy_config: SolverProxyConfig | None = None,
    ) -> tuple[str | None, bool, bool]:
        client = self._get_client()
        task, used_proxy_aware, compat_rejected = self._build_task_body(
            _CAPSOLVER_TASK_TYPES, captcha_type, sitekey, page_url,
            page_action=page_action, proxy_config=proxy_config,
            provider_name="capsolver",
        )
        if not task:
            return None, used_proxy_aware, compat_rejected

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
            return None, used_proxy_aware, compat_rejected
        if data.get("errorId", 0) != 0:
            logger.warning(
                "CapSolver submit error: %s",
                _redact_clientkey_text(str(data.get("errorDescription"))),
            )
            return None, used_proxy_aware, compat_rejected
        task_id = data.get("taskId")
        if not task_id:
            return None, used_proxy_aware, compat_rejected

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
                return None, used_proxy_aware, compat_rejected
            if data.get("errorId", 0) != 0:
                logger.warning(
                    "CapSolver poll error: %s",
                    _redact_clientkey_text(str(data.get("errorDescription"))),
                )
                return None, used_proxy_aware, compat_rejected
            if data.get("status") == "ready":
                solution = data.get("solution", {})
                token = solution.get("gRecaptchaResponse") or solution.get("token")
                return token, used_proxy_aware, compat_rejected
        return None, used_proxy_aware, compat_rejected

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
