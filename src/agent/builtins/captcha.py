"""CAPTCHA detection and solving via 2Captcha / CapSolver APIs.

Detects reCAPTCHA v2/v3/Enterprise, hCaptcha, and Cloudflare Turnstile on a
Playwright page, solves them using a paid API (user-provided key via vault),
and injects the token back into the page.

No extra pip dependencies — both APIs are plain REST over httpx.
"""

from __future__ import annotations

import asyncio

import httpx

from src.shared.utils import setup_logging

logger = setup_logging("agent.captcha")

# JavaScript snippet that detects common CAPTCHA widgets on the page.
_DETECT_JS = """
(() => {
    const result = {};
    const scriptSrcs = [...document.querySelectorAll('script[src]')]
        .map(s => s.src);

    // --- reCAPTCHA ---
    const recapEl = document.querySelector('.g-recaptcha');
    const hasEnterprise = scriptSrcs.some(s => s.includes('recaptcha/enterprise'));
    const hasRecapApi = scriptSrcs.some(s => s.includes('recaptcha/api'));

    if (recapEl || hasEnterprise || hasRecapApi) {
        const sitekey = (recapEl && recapEl.getAttribute('data-sitekey')) || '';
        const size = (recapEl && recapEl.getAttribute('data-size')) || '';
        if (hasEnterprise) {
            result.type = 'recaptcha_enterprise';
        } else if (size === 'invisible' || (!recapEl && hasRecapApi)) {
            result.type = 'recaptcha_v3';
        } else {
            result.type = 'recaptcha_v2';
        }
        result.sitekey = sitekey;
        return result;
    }

    // --- hCaptcha ---
    const hcapEl = document.querySelector('.h-captcha');
    if (hcapEl || scriptSrcs.some(s => s.includes('hcaptcha.com'))) {
        const sitekey = (hcapEl && hcapEl.getAttribute('data-sitekey')) || '';
        result.type = 'hcaptcha';
        result.sitekey = sitekey;
        return result;
    }

    // --- Cloudflare Turnstile ---
    const cfEl = document.querySelector('.cf-turnstile');
    if (cfEl || scriptSrcs.some(s => s.includes('challenges.cloudflare.com'))) {
        const sitekey = (cfEl && cfEl.getAttribute('data-sitekey')) || '';
        result.type = 'turnstile';
        result.sitekey = sitekey;
        return result;
    }

    return null;
})()
"""

# Task type mapping per provider.
_TASK_TYPES = {
    "2captcha": {
        "recaptcha_v2": "RecaptchaV2TaskProxyless",
        "recaptcha_v3": "RecaptchaV3TaskProxyless",
        "recaptcha_enterprise": "ReCaptchaV2EnterpriseTaskProxyless",
        "hcaptcha": "HCaptchaTaskProxyless",
        "turnstile": "TurnstileTaskProxyless",
    },
    "capsolver": {
        "recaptcha_v2": "RecaptchaV2TaskProxyless",
        "recaptcha_v3": "RecaptchaV3TaskProxyless",
        "recaptcha_enterprise": "ReCaptchaV2EnterpriseTaskProxyless",
        "hcaptcha": "HCaptchaTaskProxyless",
        "turnstile": "AntiTurnstileTaskProxyless",
    },
}

_PROVIDER_URLS = {
    "capsolver": "https://api.capsolver.com",
    "2captcha": "https://api.2captcha.com",
}

_POLL_INTERVAL = 3  # seconds
_POLL_TIMEOUT = 90  # seconds


async def detect_captcha(page: object) -> dict | None:
    """Detect a CAPTCHA widget on the current page.

    Returns a dict with ``type`` and ``sitekey`` keys, or ``None`` if no
    CAPTCHA is found.
    """
    try:
        result = await page.evaluate(_DETECT_JS)  # type: ignore[union-attr]
    except Exception as exc:
        logger.debug("CAPTCHA detection JS failed: %s", exc)
        return None

    if not result or not isinstance(result, dict) or "type" not in result:
        return None

    logger.info(
        "Detected CAPTCHA: %s (sitekey=%s)",
        result["type"],
        result.get("sitekey", "")[:16] + "...",
    )
    return result


async def solve_captcha(
    captcha_info: dict,
    page_url: str,
    mesh_client: object,
) -> str | None:
    """Solve a CAPTCHA using 2Captcha or CapSolver.

    Resolves API keys from the vault (``capsolver_key`` preferred, then
    ``2captcha_key``).  Returns the solution token string, or ``None`` if
    no key is configured or solving fails.
    """
    # Resolve provider + key — prefer CapSolver (faster)
    provider: str | None = None
    api_key: str | None = None
    for prov, cred_name in [("capsolver", "capsolver_key"), ("2captcha", "2captcha_key")]:
        key = await mesh_client.vault_resolve(cred_name)  # type: ignore[union-attr]
        if key:
            provider = prov
            api_key = key
            break

    if not provider or not api_key:
        logger.info("No CAPTCHA API key in vault (capsolver_key / 2captcha_key)")
        return None

    captcha_type = captcha_info["type"]
    sitekey = captcha_info.get("sitekey", "")
    task_type = _TASK_TYPES.get(provider, {}).get(captcha_type)
    if not task_type:
        logger.warning("Unsupported CAPTCHA type %s for provider %s", captcha_type, provider)
        return None

    base_url = _PROVIDER_URLS[provider]

    # Build the task payload
    task: dict = {
        "type": task_type,
        "websiteURL": page_url,
        "websiteKey": sitekey,
    }
    if captcha_type == "recaptcha_v3":
        task["pageAction"] = "verify"
        task["minScore"] = 0.5

    create_payload = {"clientKey": api_key, "task": task}

    async with httpx.AsyncClient(timeout=30) as client:
        # Create task
        try:
            resp = await client.post(f"{base_url}/createTask", json=create_payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("CAPTCHA createTask failed: %s", exc)
            return None

        task_id = data.get("taskId")
        if not task_id:
            error_desc = data.get("errorDescription", data.get("errorCode", "unknown"))
            logger.error("CAPTCHA createTask error: %s", error_desc)
            return None

        logger.info("CAPTCHA task created: provider=%s task_id=%s", provider, task_id)

        # Poll for result
        poll_payload = {"clientKey": api_key, "taskId": task_id}
        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT:
            await asyncio.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL
            try:
                resp = await client.post(f"{base_url}/getTaskResult", json=poll_payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.debug("CAPTCHA poll error: %s", exc)
                continue

            status = data.get("status", "")
            if status == "ready":
                solution = data.get("solution", {})
                token = (
                    solution.get("gRecaptchaResponse")
                    or solution.get("token")
                    or solution.get("text")
                    or ""
                )
                if token:
                    logger.info("CAPTCHA solved (%s, %.0fs)", captcha_type, elapsed)
                    return token
                logger.warning("CAPTCHA solution has no token: %s", solution)
                return None
            if status == "failed":
                logger.error("CAPTCHA solving failed: %s", data.get("errorDescription", ""))
                return None
            # status == "processing" — keep polling

        logger.error("CAPTCHA solving timed out after %ds", _POLL_TIMEOUT)
        return None


# JavaScript templates for injecting solved tokens.
_INJECT_RECAPTCHA_JS = """
(token) => {
    const textarea = document.getElementById('g-recaptcha-response');
    if (textarea) {
        textarea.style.display = '';
        textarea.value = token;
    }
    // Also set any hidden textareas (multiple reCAPTCHA widgets)
    document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {
        el.value = token;
    });
    // Try to invoke the callback
    try {
        const cfg = window.___grecaptcha_cfg;
        if (cfg && cfg.clients) {
            for (const cid of Object.keys(cfg.clients)) {
                const c = cfg.clients[cid];
                // Walk the client tree looking for a callback function
                for (const k of Object.keys(c)) {
                    const v = c[k];
                    if (v && typeof v === 'object') {
                        for (const k2 of Object.keys(v)) {
                            if (typeof v[k2] === 'function') {
                                try { v[k2](token); } catch(e) {}
                            }
                        }
                    }
                }
            }
        }
    } catch(e) {}
    return true;
}
"""

_INJECT_HCAPTCHA_JS = """
(token) => {
    const textarea = document.querySelector('textarea[name="h-captcha-response"]');
    if (textarea) textarea.value = token;
    // Invoke hcaptcha callback if available
    try {
        if (window.hcaptcha) {
            // hcaptcha stores callbacks internally; trigger via DOM event
            const iframe = document.querySelector('iframe[src*="hcaptcha"]');
            if (iframe) {
                iframe.dispatchEvent(new Event('hcaptcha-success'));
            }
        }
    } catch(e) {}
    return true;
}
"""

_INJECT_TURNSTILE_JS = """
(token) => {
    const input = document.querySelector('input[name="cf-turnstile-response"]');
    if (input) input.value = token;
    // Try turnstile callback
    try {
        if (window.turnstile && window.turnstile._callbacks) {
            for (const cb of Object.values(window.turnstile._callbacks)) {
                if (typeof cb === 'function') {
                    try { cb(token); } catch(e) {}
                }
            }
        }
    } catch(e) {}
    return true;
}
"""

_INJECT_JS = {
    "recaptcha_v2": _INJECT_RECAPTCHA_JS,
    "recaptcha_v3": _INJECT_RECAPTCHA_JS,
    "recaptcha_enterprise": _INJECT_RECAPTCHA_JS,
    "hcaptcha": _INJECT_HCAPTCHA_JS,
    "turnstile": _INJECT_TURNSTILE_JS,
}


async def inject_captcha_token(
    page: object,
    captcha_info: dict,
    token: str,
) -> bool:
    """Inject a solved CAPTCHA token into the page.

    Returns ``True`` if injection succeeded (JS ran without error).
    """
    captcha_type = captcha_info["type"]
    js_template = _INJECT_JS.get(captcha_type)
    if not js_template:
        logger.warning("No injection template for CAPTCHA type: %s", captcha_type)
        return False

    try:
        result = await page.evaluate(js_template, token)  # type: ignore[union-attr]
        logger.info("Injected CAPTCHA token for %s", captcha_type)
        return bool(result)
    except Exception as exc:
        logger.error("CAPTCHA token injection failed: %s", exc)
        return False
