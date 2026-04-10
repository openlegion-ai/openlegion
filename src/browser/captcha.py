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

import httpx

from src.shared.utils import setup_logging

logger = setup_logging("browser.captcha")

_SOLVE_TIMEOUT = 120  # max seconds to wait for a solution
_POLL_INTERVAL = 5    # seconds between result polls
_SUPPORTED_PROVIDERS = ("2captcha", "capsolver")

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

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def solve(self, page, selector: str, page_url: str) -> bool:
        """Attempt to solve a CAPTCHA on the page.

        Args:
            page: Playwright page object.
            selector: The CSS selector that matched the CAPTCHA element.
            page_url: The current page URL.

        Returns:
            True if the CAPTCHA was solved and token injected, False otherwise.
        """
        captcha_type = _classify_captcha(selector)
        logger.info("Attempting to solve %s CAPTCHA on %s", captcha_type, page_url)

        sitekey = await self._extract_sitekey(page, captcha_type)
        if not sitekey:
            logger.warning("Could not extract sitekey for %s CAPTCHA", captcha_type)
            return False

        try:
            token = await asyncio.wait_for(
                self._submit_and_poll(captcha_type, sitekey, page_url),
                timeout=_SOLVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("CAPTCHA solve timed out after %ds", _SOLVE_TIMEOUT)
            return False
        except Exception:
            logger.exception("CAPTCHA solve failed")
            return False

        if not token:
            return False

        injected = await self._inject_token(page, captcha_type, token)
        if injected:
            logger.info("CAPTCHA solved and token injected successfully")
        else:
            logger.warning("CAPTCHA solved but token injection failed")
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
        resp = await client.post("https://api.2captcha.com/createTask", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorId", 0) != 0:
            logger.warning("2Captcha submit error: %s", data.get("errorDescription"))
            return None
        task_id = data.get("taskId")
        if not task_id:
            return None

        # Poll for result
        for _ in range(int(_SOLVE_TIMEOUT / _POLL_INTERVAL)):
            await asyncio.sleep(_POLL_INTERVAL)
            resp = await client.post(
                "https://api.2captcha.com/getTaskResult",
                json={"clientKey": self.api_key, "taskId": task_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errorId", 0) != 0:
                logger.warning("2Captcha poll error: %s", data.get("errorDescription"))
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
        resp = await client.post("https://api.capsolver.com/createTask", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorId", 0) != 0:
            logger.warning("CapSolver submit error: %s", data.get("errorDescription"))
            return None
        task_id = data.get("taskId")
        if not task_id:
            return None

        # Poll for result
        for _ in range(int(_SOLVE_TIMEOUT / _POLL_INTERVAL)):
            await asyncio.sleep(_POLL_INTERVAL)
            resp = await client.post(
                "https://api.capsolver.com/getTaskResult",
                json={"clientKey": self.api_key, "taskId": task_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("errorId", 0) != 0:
                logger.warning("CapSolver poll error: %s", data.get("errorDescription"))
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
