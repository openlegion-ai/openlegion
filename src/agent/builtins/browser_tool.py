"""Browser automation via shared Camoufox browser service.

All browser operations are proxied through the mesh to the shared browser
service container. Agent containers no longer bundle Chrome or VNC.
"""

from __future__ import annotations

from src.agent.skills import skill

# ``_redact_credentials`` is re-exported here (via the aliased import) so
# existing callers that imported it from this module — e.g. ``tests/test_builtins.py``
# — keep working after the Phase 1.3 consolidation. The canonical
# implementation lives in :mod:`src.shared.redaction`; do not edit patterns
# here.
from src.shared.redaction import (
    deep_redact as _deep_redact,
)
from src.shared.redaction import (
    redact_string as _redact_credentials,  # noqa: F401
)
from src.shared.utils import setup_logging

logger = setup_logging("agent.browser")


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
        "then browser_click and browser_type with refs to interact. "
        "For heavy SPAs (X/Twitter, Gmail, etc.) use wait_until='networkidle' or "
        "wait_until='load' so the page fully renders before you read elements. "
        "Set snapshot_after=true to include element refs in the response (saves "
        "a separate browser_get_elements call)."
    ),
    parameters={
        "url": {"type": "string", "description": "URL to navigate to"},
        "wait_ms": {
            "type": "integer",
            "description": "Extra milliseconds to wait after load signal (default 1000)",
            "default": 1000,
        },
        "wait_until": {
            "type": "string",
            "description": (
                "When to consider navigation complete. "
                "'domcontentloaded' (default, fast) — HTML parsed but JS may not have run. "
                "'load' — all resources loaded, good for most sites. "
                "'networkidle' — no network activity for 500ms, best for SPAs like X/Twitter. "
                "'commit' — first byte received, fastest."
            ),
            "default": "domcontentloaded",
        },
        "snapshot_after": {
            "type": "boolean",
            "description": "Include element snapshot in the response (default false)",
            "default": False,
        },
    },
    parallel_safe=False,
)
async def browser_navigate(
    url: str, wait_ms: int = 1000, wait_until: str = "domcontentloaded",
    snapshot_after: bool = False,
    *, mesh_client=None,
) -> dict:
    """Navigate to a URL via the browser service."""
    return await _browser_command(
        mesh_client, "navigate",
        {"url": url, "wait_ms": wait_ms, "wait_until": wait_until,
         "snapshot_after": snapshot_after},
    )


@skill(
    name="browser_get_elements",
    description=(
        "Get all interactive elements on the current page with ref IDs you "
        "can use to click and type. Returns buttons, links, inputs, selects, "
        "etc. as a structured list with ref IDs (e1, e2, ...). Pass these "
        "refs to browser_click(ref='e3') and browser_type(ref='e5'). Call "
        "this after browser_navigate and after any action that changes the "
        "page. Elements include structural context like (navigation), "
        "(dialog: Name) — when duplicates exist, prefer the one inside the "
        "relevant container. This returns structured text, NOT a visual "
        "image — use browser_screenshot for that."
    ),
    parameters={},
    parallel_safe=False,
)
async def browser_get_elements(*, mesh_client=None) -> dict:
    """Return an accessibility tree snapshot with element refs."""
    return await _browser_command(mesh_client, "snapshot")


@skill(
    name="browser_wait_for",
    description=(
        "Wait for a CSS selector to appear (or disappear) on the current page. "
        "Use this before browser_click or browser_type on elements that animate "
        "in after page load — e.g. on X/Twitter, wait for the tweet composer "
        "to finish rendering before trying to type into it. "
        "Returns success once the element reaches the desired state, or error on timeout."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector to wait for (e.g. '[data-testid=\"tweetTextarea_0\"]')",
        },
        "state": {
            "type": "string",
            "description": (
                "Element state to wait for: "
                "'visible' (default) — present in DOM and visible. "
                "'attached' — present in DOM (may be hidden). "
                "'hidden' — not visible or not in DOM. "
                "'detached' — removed from DOM."
            ),
            "default": "visible",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Max milliseconds to wait (default 10000, max 30000)",
            "default": 10000,
        },
    },
    parallel_safe=False,
)
async def browser_wait_for(
    selector: str, state: str = "visible", timeout_ms: int = 10000,
    *, mesh_client=None,
) -> dict:
    """Wait for a CSS selector to reach the given state."""
    if not selector:
        return {"error": "The 'selector' parameter is required"}
    return await _browser_command(
        mesh_client, "wait_for",
        {"selector": selector, "state": state, "timeout_ms": timeout_ms},
    )


@skill(
    name="browser_screenshot",
    description=(
        "Take a screenshot of the current page. "
        "Returns a visual PNG image you can see directly."
    ),
    parameters={
        "full_page": {
            "type": "boolean",
            "description": "Capture full scrollable page (default false)",
            "default": False,
        },
    },
    parallel_safe=False,
    loop_exempt=True,
)
async def browser_screenshot(full_page: bool = False, *, mesh_client=None) -> dict:
    """Take a screenshot via the browser service.

    Extracts image_base64 from the raw result *before* ``_deep_redact`` runs,
    because the broad credential-redaction patterns (40+ hex/base64 chars)
    would corrupt any PNG payload.  The base64 data is returned under the
    ``_image`` key so ``_run_tool`` can build a multimodal content block.
    """
    if not mesh_client:
        return {"error": "Browser requires mesh connectivity"}
    try:
        raw = await mesh_client.browser_command("screenshot", {"full_page": full_page})
    except Exception as e:
        return {"error": _deep_redact(str(e))}

    # Pull out image data before redaction can corrupt it.
    # Browser service returns {"success": ..., "data": {"image_base64": ..., ...}}
    image_data = None
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict) and data.get("image_base64"):
            image_data = data.pop("image_base64")

    result = _deep_redact(raw)

    if image_data:
        result["_image"] = {"data": image_data, "media_type": "image/png"}
        # Give the LLM a short text summary instead of the raw base64 blob
        result.setdefault("status", "screenshot captured")

    return result


@skill(
    name="browser_click",
    description=(
        "Click an element on the current page. Preferred: use ref from "
        "browser_get_elements (e.g. ref='e3'). Fallback: use a CSS selector. "
        "Buttons and links marked [disabled] in browser_get_elements should still "
        "be clicked — SPAs like X/Twitter use aria-disabled on buttons that are "
        "visually active, and the system handles this automatically. "
        "If the click times out because of an overlay or animation, try "
        "browser_wait_for first, or set force=true to bypass actionability checks. "
        "Set snapshot_after=true to include updated element refs in the response."
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
        "force": {
            "type": "boolean",
            "description": (
                "Bypass Playwright actionability checks (visibility, stability, "
                "not-covered-by-overlay). Use when the element is visually present "
                "in the browser but click keeps timing out due to an overlay. Default false."
            ),
            "default": False,
        },
        "snapshot_after": {
            "type": "boolean",
            "description": "Include element snapshot in the response (default false)",
            "default": False,
        },
        "timeout_ms": {
            "type": "integer",
            "description": (
                "Click timeout in milliseconds. Increase for heavy SPAs with "
                "animations or link card previews (e.g. 20000 for X thread "
                "'Post all'). Default 10000, max 30000."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_click(
    selector: str = "", ref: str = "", force: bool = False,
    snapshot_after: bool = False, timeout_ms: int | None = None,
    *, mesh_client=None,
) -> dict:
    """Click an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}
    cmd = {"ref": ref, "selector": selector, "force": force,
           "snapshot_after": snapshot_after}
    if timeout_ms is not None:
        cmd["timeout_ms"] = timeout_ms
    return await _browser_command(mesh_client, "click", cmd)


@skill(
    name="browser_type",
    description=(
        "Type text into a form field on the current page. Optionally clears the "
        "field first (default); set clear=false to append. "
        "Preferred: use ref from browser_get_elements "
        "(e.g. ref='e5'). Fallback: use a CSS selector. "
        "Set fast=true for search queries, URLs, and non-sensitive fields to "
        "type quickly. Set snapshot_after=true to include updated element refs."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the input field. Not needed if ref is provided.",
            "default": "",
        },
        "text": {
            "type": "string",
            "description": "Text to type into the field",
        },
        "ref": {
            "type": "string",
            "description": (
                "Preferred: element ref from browser_get_elements "
                "(e.g. 'e5'). Use this instead of selector when available."
            ),
            "default": "",
        },
        "clear": {
            "type": "boolean",
            "description": (
                "Clear the field before typing (default true). Set false to "
                "append at the cursor position. Note: in rich-text editors "
                "the cursor lands where the focus click hits, which may not "
                "be at the end of existing text."
            ),
            "default": True,
        },
        "fast": {
            "type": "boolean",
            "description": (
                "Type quickly with minimal delay. Use for search queries, "
                "URLs, and non-sensitive fields. Don't use for login forms "
                "or social media post composition. Default false."
            ),
            "default": False,
        },
        "snapshot_after": {
            "type": "boolean",
            "description": "Include element snapshot in the response (default false)",
            "default": False,
        },
    },
    parallel_safe=False,
)
async def browser_type(
    text: str, selector: str = "", ref: str = "", clear: bool = True,
    fast: bool = False, snapshot_after: bool = False, *, mesh_client=None,
) -> dict:
    """Type text into an element."""
    if not text:
        return {"error": "The 'text' parameter is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}

    return await _browser_command(
        mesh_client, "type",
        {"ref": ref, "selector": selector, "text": text, "clear": clear,
         "fast": fast, "snapshot_after": snapshot_after},
    )


@skill(
    name="browser_hover",
    description=(
        "Move the mouse over an element without clicking it. "
        "Use this to trigger hover-activated dropdowns, navigation sub-menus, "
        "or tooltips that only appear on mouseover. "
        "After hovering, call browser_get_elements to see the newly visible items. "
        "Preferred: use ref from browser_get_elements (e.g. ref='e3'). "
        "Fallback: use a CSS selector."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the element to hover. Not needed if ref is provided.",
            "default": "",
        },
        "ref": {
            "type": "string",
            "description": "Element ref from browser_get_elements (e.g. 'e3'). Preferred over selector.",
            "default": "",
        },
    },
    parallel_safe=False,
)
async def browser_hover(
    selector: str = "", ref: str = "", *, mesh_client=None,
) -> dict:
    """Hover over an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}
    return await _browser_command(
        mesh_client, "hover",
        {"ref": ref, "selector": selector},
    )


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
    parallel_safe=False,
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
    parallel_safe=False,
)
async def browser_reset(*, mesh_client=None) -> dict:
    """Force-close the browser session so the next call gets a fresh one."""
    return await _browser_command(mesh_client, "reset")


@skill(
    name="browser_press_key",
    description=(
        "Press a keyboard key or shortcut. Use this to dismiss modals (Escape), "
        "submit forms (Enter), navigate between fields (Tab), use arrow keys "
        "(ArrowUp, ArrowDown), or fire shortcuts (Control+a, Control+c). "
        "Key names follow Playwright conventions: Enter, Escape, Tab, Backspace, "
        "Delete, Space, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Home, End, "
        "PageUp, PageDown, F1-F12. Modifiers: Control+key, Shift+key, Alt+key."
    ),
    parameters={
        "key": {
            "type": "string",
            "description": (
                "Key to press — e.g. 'Escape', 'Enter', 'Tab', 'ArrowDown', "
                "'Control+a', 'Shift+Tab'"
            ),
        },
    },
    parallel_safe=False,
)
async def browser_press_key(key: str, *, mesh_client=None) -> dict:
    """Press a keyboard key or combination."""
    if not key:
        return {"error": "The 'key' parameter is required"}
    return await _browser_command(mesh_client, "press_key", {"key": key})


@skill(
    name="browser_go_back",
    description=(
        "Navigate back in browser history (like clicking the Back button). "
        "Returns the URL and title of the page navigated to."
    ),
    parameters={},
    parallel_safe=False,
)
async def browser_go_back(*, mesh_client=None) -> dict:
    """Go back one page in browser history."""
    return await _browser_command(mesh_client, "go_back")


@skill(
    name="browser_go_forward",
    description=(
        "Navigate forward in browser history (like clicking the Forward button). "
        "Returns the URL and title of the page navigated to."
    ),
    parameters={},
    parallel_safe=False,
)
async def browser_go_forward(*, mesh_client=None) -> dict:
    """Go forward one page in browser history."""
    return await _browser_command(mesh_client, "go_forward")


@skill(
    name="browser_switch_tab",
    description=(
        "List all open browser tabs, or switch to a specific tab. "
        "Call without tab_index to see all open tabs — useful when a popup or "
        "OAuth window opens. Call with tab_index to switch to that tab. "
        "After switching, call browser_get_elements to see the new tab's content. "
        "Refs from the previous tab are cleared on switch."
    ),
    parameters={
        "tab_index": {
            "type": "integer",
            "description": (
                "Tab index to switch to (from the tab list). "
                "Omit or set to -1 to just list all open tabs without switching."
            ),
            "default": -1,
        },
    },
    parallel_safe=False,
)
async def browser_switch_tab(tab_index: int = -1, *, mesh_client=None) -> dict:
    """List open tabs and optionally switch to one."""
    return await _browser_command(
        mesh_client, "switch_tab", {"tab_index": tab_index},
    )


@skill(
    name="browser_detect_captcha",
    description=(
        "Detect CAPTCHAs (reCAPTCHA, hCaptcha, Cloudflare Turnstile) on the "
        "current page. When a CAPTCHA solver is configured, CAPTCHAs are "
        "solved automatically after navigation. If auto-solving fails or no "
        "solver is configured, the CAPTCHA must be solved manually via VNC."
    ),
    parameters={},
    parallel_safe=False,
)
async def browser_detect_captcha(*, mesh_client=None) -> dict:
    """Detect CAPTCHAs on the current page."""
    return await _browser_command(mesh_client, "detect_captcha")


@skill(
    name="request_browser_login",
    description=(
        "Ask the user to log in to a website through a live browser view "
        "in their chat. Use this when an automation needs a cookie-based "
        "login that can't be done via API keys (e.g. Twitter, LinkedIn, "
        "TikTok web). The browser navigates to the login URL and an "
        "interactive VNC viewer appears in the user's chat. After the user "
        "finishes, you receive a notification.\n\n"
        "IMPORTANT: session cookies persist in the TARGET agent's browser "
        "profile. If you're orchestrating a login for another agent (e.g. "
        "operator setting up a login for social-manager), pass ``agent_id`` "
        "with the worker's ID so cookies land in their profile. If you're "
        "the agent that will use the login yourself, omit ``agent_id`` "
        "(defaults to self)."
    ),
    parameters={
        "url": {
            "type": "string",
            "description": "Login page URL (e.g. 'https://tiktok.com/login')",
        },
        "service": {
            "type": "string",
            "description": "Service name (e.g. 'TikTok', 'LinkedIn', 'Instagram')",
        },
        "description": {
            "type": "string",
            "description": "Tell the user what to do (e.g. 'Please log in to your TikTok account')",
        },
        "agent_id": {
            "type": "string",
            "description": (
                "Optional: target agent ID whose browser profile should "
                "receive the login. Defaults to the calling agent. Use "
                "this when orchestrating logins on behalf of another "
                "agent — cookies persist in the target's profile."
            ),
            "default": "",
        },
    },
    parallel_safe=False,
)
async def request_browser_login(
    url: str, service: str, description: str, agent_id: str = "",
    *, mesh_client=None, **_kw,
) -> dict:
    """Request user login via live browser view in chat."""
    if not mesh_client:
        return {"error": "Browser login requires mesh connectivity"}
    if not url:
        return {"error": "url is required"}
    if not service:
        return {"error": "service is required"}

    # Defensive: LLMs sometimes pass null/non-string despite the schema.
    # Coerce to str so .strip() never raises and an empty/None claim
    # falls through to the self-browser path.
    target = (str(agent_id).strip() if agent_id else "") or None

    # Navigate the target agent's browser to the login page first
    try:
        await mesh_client.browser_command(
            "navigate", {"url": url}, target_agent_id=target,
        )
    except Exception as e:
        return {"error": f"Failed to navigate browser to {url}: {e}"}

    # Emit browser login request event to the dashboard
    try:
        await mesh_client.request_browser_login(
            url=url, service=service, description=description,
            target_agent_id=target,
        )
    except Exception as e:
        logger.warning("Failed to emit browser login request for %s: %s", service, e)
        return {
            "requested": False,
            "service": service,
            "url": url,
            "target_agent": target,
            "message": (
                f"Browser navigated to {url} but failed to show the login card "
                f"in the user's chat. Try calling notify_user to ask them to "
                f"log in via the browser on your settings page instead."
            ),
        }

    return {
        "requested": True,
        "service": service,
        "url": url,
        "target_agent": target,
        "message": (
            f"Browser login request sent to user. The browser is showing {url}. "
            f"Wait for the user to complete the login — you will be notified. "
            f"Do NOT use browser tools until the user confirms."
        ),
    }
