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

# 2Captcha task type mapping
_2CAPTCHA_TASK_TYPES: dict[str, str] = {
    "recaptcha": "NormalRecaptchaTaskProxyless",
    "hcaptcha": "HCaptchaTaskProxyless",
    "turnstile": "TurnstileTaskProxyless",
}

# CapSolver task type mapping
_CAPSOLVER_TASK_TYPES: dict[str, str] = {
    "recaptcha": "ReCaptchaV2TaskProxyLess",
    "hcaptcha": "HCaptchaTaskProxyLess",
    "turnstile": "AntiTurnstileTaskProxyLess",
}


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
        logger.info("Attempting to solve %s CAPTCHA on %s", captcha_type, page_url)

        sitekey = await self._extract_sitekey(page, captcha_type)
        if not sitekey:
            logger.warning("Could not extract sitekey for %s CAPTCHA", captcha_type)
            await self._record_solver_outcome(success=False)
            return False

        try:
            token = await asyncio.wait_for(
                self._submit_and_poll(captcha_type, sitekey, page_url),
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
        try:
            # Try data-sitekey attribute first (works for reCAPTCHA, hCaptcha, Turnstile)
            sitekey = await page.evaluate(
                "() => document.querySelector('[data-sitekey]')?.getAttribute('data-sitekey')"
            )
            if sitekey:
                return sitekey.strip()

            # Fall back to parsing iframe src for sitekey parameter
            if captcha_type == "recaptcha":
                src = await page.evaluate(
                    "() => document.querySelector('iframe[src*=\"recaptcha\"]')?.src"
                )
                if src:
                    match = re.search(r'[?&]k=([^&]+)', src)
                    if match:
                        return match.group(1)

            if captcha_type == "hcaptcha":
                src = await page.evaluate(
                    "() => document.querySelector('iframe[src*=\"hcaptcha\"]')?.src"
                )
                if src:
                    match = re.search(r'[?&]sitekey=([^&]+)', src)
                    if match:
                        return match.group(1)

            if captcha_type == "turnstile":
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

    async def _submit_and_poll(self, captcha_type: str, sitekey: str, page_url: str) -> str | None:
        """Submit CAPTCHA to solving service and poll for result."""
        if self.provider == "2captcha":
            return await self._solve_2captcha(captcha_type, sitekey, page_url)
        return await self._solve_capsolver(captcha_type, sitekey, page_url)

    # ── 2Captcha ──────────────────────────────────────────────────────────────

    async def _solve_2captcha(self, captcha_type: str, sitekey: str, page_url: str) -> str | None:
        client = self._get_client()
        task_type = _2CAPTCHA_TASK_TYPES.get(captcha_type)
        if not task_type:
            return None

        # Submit task
        payload = {
            "clientKey": self.api_key,
            "task": {
                "type": task_type,
                "websiteURL": page_url,
                "websiteKey": sitekey,
            },
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

    async def _solve_capsolver(self, captcha_type: str, sitekey: str, page_url: str) -> str | None:
        client = self._get_client()
        task_type = _CAPSOLVER_TASK_TYPES.get(captcha_type)
        if not task_type:
            return None

        # Submit task
        payload = {
            "clientKey": self.api_key,
            "task": {
                "type": task_type,
                "websiteURL": page_url,
                "websiteKey": sitekey,
            },
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
        """Inject the solved CAPTCHA token into the page."""
        try:
            if captcha_type == "recaptcha":
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

            if captcha_type == "hcaptcha":
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

            if captcha_type == "turnstile":
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
