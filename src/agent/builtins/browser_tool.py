"""Browser automation via shared Camoufox browser service.

All browser operations are proxied through the mesh to the shared browser
service container. Agent containers no longer bundle Chrome or VNC.
Credential resolution ($CRED{} handles) happens agent-side before sending
text to the browser service, so secrets never transit as plaintext names.
"""

from __future__ import annotations

import re

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.browser")

_CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")

# Pattern-based credential redaction (defense in depth on agent side)
_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-ant-api[A-Za-z0-9\-]{20,}"),
    re.compile(r"gho_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),
    re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9/+=])"),
]

# Per-agent resolved credential values for exact-match redaction
_resolved_credential_values: set[str] = set()


def _redact_credentials(text: str) -> str:
    """Replace common secret patterns with [REDACTED]."""
    if not text:
        return text
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def _redact_resolved_credentials(text: str) -> str:
    """Replace any resolved $CRED{} values with [REDACTED]."""
    if not text or not _resolved_credential_values:
        return text
    for value in _resolved_credential_values:
        text = text.replace(value, "[REDACTED]")
    return text


def _deep_redact(obj):
    """Recursively redact credential values from any JSON-serializable structure."""
    if isinstance(obj, str):
        return _redact_resolved_credentials(_redact_credentials(obj))
    if isinstance(obj, dict):
        return {k: _deep_redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_redact(item) for item in obj]
    return obj


async def _browser_command(mesh_client, action: str, params: dict | None = None) -> dict:
    """Send a browser command through mesh. Returns the result dict."""
    if not mesh_client:
        return {"error": "Browser requires mesh connectivity"}
    try:
        result = await mesh_client.browser_command(action, params or {})
        return _deep_redact(result)
    except Exception as e:
        return {"error": _deep_redact(str(e))}


@skill(
    name="browser_navigate",
    description=(
        "Navigate your browser to a URL and return the page text. "
        "Use this to visit any website: sign-up pages, dashboards, search engines, "
        "web apps. You can then use browser_get_elements to get element refs, "
        "then browser_click and browser_type with refs to interact."
    ),
    parameters={
        "url": {"type": "string", "description": "URL to navigate to"},
        "wait_ms": {
            "type": "integer",
            "description": "Milliseconds to wait after load (default 1000)",
            "default": 1000,
        },
    },
)
async def browser_navigate(url: str, wait_ms: int = 1000, *, mesh_client=None) -> dict:
    """Navigate to a URL via the browser service."""
    return await _browser_command(mesh_client, "navigate", {"url": url, "wait_ms": wait_ms})


@skill(
    name="browser_get_elements",
    description=(
        "Get all interactive elements on the current page with ref IDs you "
        "can use to click and type. Returns buttons, links, inputs, selects, "
        "etc. as a structured list with ref IDs (e1, e2, ...). Pass these "
        "refs to browser_click(ref='e3') and browser_type(ref='e5'). Call "
        "this after browser_navigate and after any action that changes the "
        "page. This returns structured text, NOT a visual image — use "
        "browser_screenshot for that."
    ),
    parameters={},
)
async def browser_get_elements(*, mesh_client=None) -> dict:
    """Return an accessibility tree snapshot with element refs."""
    return await _browser_command(mesh_client, "snapshot")


@skill(
    name="browser_screenshot",
    description=(
        "Take a screenshot of the current page. "
        "Returns base64-encoded PNG image data."
    ),
    parameters={
        "full_page": {
            "type": "boolean",
            "description": "Capture full scrollable page (default false)",
            "default": False,
        },
    },
)
async def browser_screenshot(full_page: bool = False, *, mesh_client=None) -> dict:
    """Take a screenshot via the browser service."""
    return await _browser_command(mesh_client, "screenshot", {"full_page": full_page})


@skill(
    name="browser_click",
    description=(
        "Click an element on the current page. Preferred: use ref from "
        "browser_get_elements (e.g. ref='e3'). Fallback: use a CSS selector."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector to click. Not needed if ref is provided.",
            "default": "",
        },
        "ref": {
            "type": "string",
            "description": (
                "Preferred: element ref from browser_get_elements "
                "(e.g. 'e3'). Use this instead of selector when available."
            ),
            "default": "",
        },
    },
)
async def browser_click(selector: str = "", ref: str = "", *, mesh_client=None) -> dict:
    """Click an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}
    return await _browser_command(mesh_client, "click", {"ref": ref, "selector": selector})


@skill(
    name="browser_type",
    description=(
        "Type text into a form field on the current page. Clears the field first, "
        "then enters the new text. Preferred: use ref from browser_get_elements "
        "(e.g. ref='e5'). Fallback: use a CSS selector. "
        "Use $CRED{name} handles to type secrets (e.g. text='$CRED{twitter_password}') "
        "— the value is resolved from the vault and never exposed to you."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the input field. Not needed if ref is provided.",
            "default": "",
        },
        "text": {
            "type": "string",
            "description": (
                "Text to type. Use $CRED{name} for secrets, "
                "e.g. '$CRED{twitter_password}'"
            ),
        },
        "ref": {
            "type": "string",
            "description": (
                "Preferred: element ref from browser_get_elements "
                "(e.g. 'e5'). Use this instead of selector when available."
            ),
            "default": "",
        },
    },
)
async def browser_type(
    text: str, selector: str = "", ref: str = "", *, mesh_client=None,
) -> dict:
    """Type text into an element. Resolves $CRED{} handles agent-side."""
    if not text:
        return {"error": "The 'text' parameter is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}

    # Resolve $CRED{name} handles agent-side (secrets never transit as names)
    is_credential = False
    actual_text = text
    cred_matches = _CRED_HANDLE_RE.findall(text)
    if cred_matches:
        if not mesh_client:
            return {"error": "$CRED{} handles require mesh connectivity for resolution"}
        for cred_name in cred_matches:
            resolved = await mesh_client.vault_resolve(cred_name)
            if resolved is None:
                return {"error": f"Credential not found: {cred_name}"}
            actual_text = actual_text.replace(f"$CRED{{{cred_name}}}", resolved)
            if len(resolved) >= 4:
                _resolved_credential_values.add(resolved)
        is_credential = True

    result = await _browser_command(
        mesh_client, "type",
        {"ref": ref, "selector": selector, "text": actual_text, "clear": True,
         "is_credential": is_credential},
    )

    # Never return credential values to the LLM
    if is_credential and result.get("success"):
        result["data"] = {"typed": "[credential]", "ref": ref or selector}
    return result


@skill(
    name="browser_scroll",
    description=(
        "Scroll the browser page. Supports scrolling by direction (up/down) "
        "with an optional pixel amount, or scrolling a specific element into view "
        "using a ref from browser_get_elements. Default scrolls down by one viewport height. "
        "Call browser_get_elements after scrolling to see the updated page content."
    ),
    parameters={
        "direction": {
            "type": "string",
            "description": "Scroll direction: 'up' or 'down' (default 'down')",
            "default": "down",
        },
        "amount": {
            "type": "integer",
            "description": "Pixels to scroll (0 = one viewport height)",
            "default": 0,
        },
        "ref": {
            "type": "string",
            "description": "Element ref from browser_get_elements to scroll into view (e.g. 'e5')",
            "default": "",
        },
    },
)
async def browser_scroll(
    direction: str = "down", amount: int = 0, ref: str = "", *, mesh_client=None,
) -> dict:
    """Scroll the page or scroll an element into view."""
    params = {"direction": direction, "amount": amount}
    if ref:
        params["ref"] = ref
    return await _browser_command(mesh_client, "scroll", params)



@skill(
    name="browser_reset",
    description=(
        "Reset the browser by closing the current session and starting fresh. "
        "Use this when the browser is stuck or showing errors. "
        "Profile is preserved — cookies and sessions survive."
    ),
    parameters={},
)
async def browser_reset(*, mesh_client=None) -> dict:
    """Force-close the browser session so the next call gets a fresh one."""
    _resolved_credential_values.clear()
    return await _browser_command(mesh_client, "reset")


@skill(
    name="browser_solve_captcha",
    description=(
        "Manually trigger CAPTCHA detection and solving on the current page. "
        "Usually NOT needed — browser_navigate auto-detects CAPTCHAs. "
        "Use this only when a CAPTCHA appears AFTER navigation."
    ),
    parameters={},
)
async def browser_solve_captcha(*, mesh_client=None) -> dict:
    """Detect and solve a CAPTCHA on the current page."""
    return await _browser_command(mesh_client, "solve_captcha")
