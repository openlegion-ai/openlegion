"""Browser automation via shared Camoufox browser service.

All browser operations are proxied through the mesh to the shared browser
service container. Agent containers no longer bundle Chrome or VNC.
"""

from __future__ import annotations

from urllib.parse import urlparse

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
        "a separate browser_get_elements call). "
        "Leave 'referer' unset for normal browsing — the service picks a "
        "plausible referer (search engine, same-origin, etc.) based on the "
        "destination so the page sees a realistic arrival pattern. Override "
        "only when you're explicitly modeling a specific click-through."
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
        "referer": {
            "type": "string",
            "description": (
                "Optional referer URL to send with the navigation. "
                "Default: leave unset and let the service pick. "
                "Pass an empty string to explicitly send no referer "
                "(equivalent to a typed URL or bookmark)."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_navigate(
    url: str, wait_ms: int = 1000, wait_until: str = "domcontentloaded",
    snapshot_after: bool = False,
    referer: str | None = None,
    *, mesh_client=None,
) -> dict:
    """Navigate to a URL via the browser service."""
    params: dict = {
        "url": url, "wait_ms": wait_ms, "wait_until": wait_until,
        "snapshot_after": snapshot_after,
    }
    # Forward referer ONLY when the caller specified it — None means
    # "let the service pick", which is signalled by the param's absence
    # so the browser-service navigate picker fires.
    if referer is not None:
        params["referer"] = referer
    return await _browser_command(mesh_client, "navigate", params)


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
        "image — use browser_screenshot for that.\n\n"
        "Optional: pass filter='actionable'|'inputs'|'headings'|'landmarks' "
        "to narrow the result (cheaper tokens on huge pages). Pass "
        "from_ref='e3' to scope the snapshot to a specific element's "
        "subtree — useful for drilling into a list, table, or form section. "
        "Pass diff_from_last=True after an action to get only what changed "
        "since the last snapshot — added/removed/changed elements with a "
        "'scope' label (same | modal_opened | modal_closed | frame_changed "
        "| navigation | tab_changed). Returns a full snapshot when scope is "
        "navigation/tab_changed."
    ),
    parameters={
        "filter": {
            "type": "string",
            "description": (
                "Optional role filter. 'actionable' = clickable / typeable "
                "elements only; 'inputs' = form inputs only; 'headings' = "
                "section headings for orientation; 'landmarks' = top-level "
                "regions (navigation/main/...). Omit for the default mix."
            ),
        },
        "from_ref": {
            "type": "string",
            "description": (
                "Optional ref id (e.g. 'e7') from a previous "
                "browser_get_elements call. When set, the snapshot is "
                "rooted at that element's subtree. Note: refs returned "
                "from a scoped snapshot resolve against the whole page, "
                "not the subtree — if a modal or other overlay is open, "
                "duplicate role+name elements behind it could match. "
                "When in doubt, take a non-scoped browser_get_elements "
                "before clicking refs from a scoped result."
            ),
        },
        "diff_from_last": {
            "type": "boolean",
            "description": (
                "When True, return only what changed since the previous "
                "snapshot in this tab. Useful after a click/type to see "
                "what your action did without re-paying the full snapshot "
                "token cost."
            ),
            "default": False,
        },
        "frame": {
            "type": "string",
            "description": (
                "Optional frame selector. Matched as a URL substring "
                "against frame URLs, or as a frame_id token returned by "
                "a prior snapshot. When set, the snapshot walks only "
                "that frame instead of main-frame plus iframes."
            ),
        },
        "include_frames": {
            "type": "boolean",
            "description": (
                "When True (default) the snapshot descends into "
                "same-origin iframes, emitting refs from inner frames "
                "with their own frame_id. Set False to walk only the "
                "main frame (or, with frame= set, only that frame) "
                "without recursing — useful when iframe contents are "
                "noise for the current task."
            ),
            "default": True,
        },
    },
    parallel_safe=False,
)
async def browser_get_elements(
    filter: str | None = None,
    from_ref: str | None = None,
    diff_from_last: bool = False,
    frame: str | None = None,
    include_frames: bool = True,
    *,
    mesh_client=None,
) -> dict:
    """Return an accessibility tree snapshot with element refs."""
    payload: dict = {}
    if filter is not None:
        payload["filter"] = filter
    if from_ref is not None:
        payload["from_ref"] = from_ref
    if diff_from_last:
        payload["diff_from_last"] = True
    if frame is not None:
        payload["frame"] = frame
    if not include_frames:
        payload["include_frames"] = False
    return await _browser_command(mesh_client, "snapshot", payload)


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
        "Take a screenshot of the current page. Returns a visual image you "
        "can see directly. Defaults: WebP at quality=75, full resolution. "
        "Use format='png' for lossless capture (e.g. when comparing pixels "
        "or feeding into an OCR pipeline). Use scale=0.75 to shrink the "
        "image further for token-cheap reconnaissance shots."
    ),
    parameters={
        "full_page": {
            "type": "boolean",
            "description": "Capture full scrollable page (default false)",
            "default": False,
        },
        "format": {
            "type": "string",
            "description": (
                "Image format: 'webp' (default, smaller, lossy) or 'png' "
                "(lossless, larger). WebP is ~5–10× cheaper in tokens."
            ),
            "default": "webp",
        },
        "quality": {
            "type": "integer",
            "description": (
                "WebP quality 1–100 (default 75). Ignored for PNG. Lower "
                "values trade visual fidelity for smaller payloads."
            ),
            "default": 75,
        },
        "scale": {
            "type": "number",
            "description": (
                "Resize factor 0.5–1.0 (default 1.0). 0.75 keeps the page "
                "readable while cutting bytes ~45%."
            ),
            "default": 1.0,
        },
    },
    parallel_safe=False,
    loop_exempt=True,
)
async def browser_screenshot(
    full_page: bool = False,
    format: str = "webp",
    quality: int = 75,
    scale: float = 1.0,
    *,
    mesh_client=None,
) -> dict:
    """Take a screenshot via the browser service.

    Extracts image_base64 from the raw result *before* ``_deep_redact`` runs,
    because the broad credential-redaction patterns (40+ hex/base64 chars)
    would corrupt any image payload. The base64 data is returned under the
    ``_image`` key so ``_run_tool`` can build a multimodal content block.

    The browser service is the source of truth for the actual encoding
    used — it may fall back to PNG when WebP encoding fails (e.g. Pillow
    missing or a corrupt frame buffer). The returned ``_image.media_type``
    reflects what was actually emitted, not what was requested.
    """
    if not mesh_client:
        return {"error": "Browser requires mesh connectivity"}
    try:
        raw = await mesh_client.browser_command(
            "screenshot",
            {
                "full_page": full_page,
                "format": format,
                "quality": quality,
                "scale": scale,
            },
        )
    except Exception as e:
        return {"error": _deep_redact(str(e))}

    # Pull out image data before redaction can corrupt it.
    # Browser service returns {"success": ..., "data":
    #   {"image_base64": ..., "format": "webp"|"png", "bytes": int}}
    image_data = None
    actual_format = "png"
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, dict) and data.get("image_base64"):
            image_data = data.pop("image_base64")
            actual_format = (data.get("format") or "png").lower()

    result = _deep_redact(raw)

    if image_data:
        media_type = "image/webp" if actual_format == "webp" else "image/png"
        result["_image"] = {"data": image_data, "media_type": media_type}
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
        "frame": {
            "type": "string",
            "description": (
                "Optional frame selector for selector-based clicks. "
                "Matched as a URL substring against frame URLs, or as a "
                "frame_id token from a prior snapshot. Refs from a "
                "snapshot already carry their frame, so this argument is "
                "redundant when ref is set and conflicts return an error."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_click(
    selector: str = "", ref: str = "", force: bool = False,
    snapshot_after: bool = False, timeout_ms: int | None = None,
    frame: str | None = None,
    *, mesh_client=None,
) -> dict:
    """Click an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}
    cmd = {"ref": ref, "selector": selector, "force": force,
           "snapshot_after": snapshot_after}
    if timeout_ms is not None:
        cmd["timeout_ms"] = timeout_ms
    if frame is not None:
        cmd["frame"] = frame
    return await _browser_command(mesh_client, "click", cmd)


@skill(
    name="browser_click_xy",
    description=(
        "Click at viewport-relative CSS pixel coordinates (x, y). Useful "
        "for canvas, custom-rendered widgets, or non-accessible "
        "interactives where browser_get_elements does not surface a ref. "
        "Performs a document.elementFromPoint pre-check — if an overlay "
        "or pointer-events:none ancestor would intercept the click, the "
        "call returns invalid_input with the actual hit element's "
        "tag/role/name so you can re-target. "
        "Coordinates are CSS pixels (NOT device pixels — devicePixelRatio "
        "is ignored), viewport-relative, and bounded by the current "
        "viewport (call browser_status for exact width/height — fleet "
        "resolutions vary, e.g. 1920x1080, 1366x768, 1536x864). "
        "Coordinates are NOT document-absolute: after a scroll, resize, "
        "or any DOM-relayout interaction, the same (x, y) hits a "
        "different element — recompute coords from a fresh snapshot. "
        "Limitations: (a) cross-origin iframes — elementFromPoint "
        "returns the <iframe> element, not its inner document, so "
        "actual_element will be uninformative; (b) shadow DOM — "
        "elementFromPoint returns the shadow host, not the inner shadow "
        "element; (c) the post-click CAPTCHA check is best-effort and "
        "may race the challenge render — call browser_detect_captcha "
        "after a short delay if the click likely triggered a challenge. "
        "Prefer browser_click(ref) when possible — coordinate clicks "
        "are brittle across resolution changes."
    ),
    parameters={
        "x": {
            "type": "number",
            "description": (
                "Viewport-relative x coordinate in CSS pixels. Origin is "
                "the top-left of the rendered page area; must satisfy "
                "0 <= x < viewport.width."
            ),
        },
        "y": {
            "type": "number",
            "description": (
                "Viewport-relative y coordinate in CSS pixels. Origin is "
                "the top-left of the rendered page area; must satisfy "
                "0 <= y < viewport.height."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_click_xy(
    x: float, y: float, *, mesh_client=None,
) -> dict:
    """Click at viewport-relative CSS pixel coordinates."""
    return await _browser_command(
        mesh_client, "click_xy", {"x": x, "y": y},
    )


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
        "frame": {
            "type": "string",
            "description": (
                "Optional frame selector for selector-based typing. "
                "Matched as a URL substring against frame URLs, or as a "
                "frame_id token from a prior snapshot. Refs from a "
                "snapshot already carry their frame, so this argument is "
                "redundant when ref is set and conflicts return an error."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_type(
    text: str, selector: str = "", ref: str = "", clear: bool = True,
    fast: bool = False, snapshot_after: bool = False,
    frame: str | None = None, *, mesh_client=None,
) -> dict:
    """Type text into an element."""
    if not text:
        return {"error": "The 'text' parameter is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_get_elements) or 'selector' (CSS)"}

    cmd = {"ref": ref, "selector": selector, "text": text, "clear": clear,
           "fast": fast, "snapshot_after": snapshot_after}
    if frame is not None:
        cmd["frame"] = frame
    return await _browser_command(mesh_client, "type", cmd)


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
    name="browser_find_text",
    description=(
        "Find elements on the current page whose accessible name contains "
        "the query string. Matching is Unicode-aware case-insensitive via "
        "str.casefold() — 'EMAIL' matches 'email', 'STRASSE' matches "
        "'straße'. Returns up to 50 matches in snapshot order, each with "
        "a ref usable by browser_click / browser_type plus an in_viewport "
        "flag. When scroll=true (default) and at least one match exists, "
        "the first match is scrolled into view. Use this to locate a "
        "specific button or link by its visible label without scanning "
        "the whole element list yourself."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": (
                "Substring to search for (1–500 chars). Case-insensitive."
            ),
        },
        "scroll": {
            "type": "boolean",
            "description": (
                "If true (default), scroll the first match into view. "
                "Set false to inspect matches without affecting scroll "
                "position."
            ),
            "default": True,
        },
    },
    parallel_safe=False,
)
async def browser_find_text(
    query: str, scroll: bool = True, *, mesh_client=None,
) -> dict:
    """Find elements whose accessible name contains the query."""
    if not query:
        return {"error": "The 'query' parameter is required"}
    return await _browser_command(
        mesh_client, "find_text", {"query": query, "scroll": scroll},
    )


@skill(
    name="browser_fill_form",
    description=(
        "Fill multiple form fields by their visible labels in one call. "
        "For each field, locates the input by label text (find_text under "
        "the hood), then fills the value (clears existing content first). "
        "On CAPTCHA detection mid-flow, returns partial_success with the "
        "fields that were filled and the fields that remain — solve the "
        "CAPTCHA, then call browser_fill_form again with the remaining "
        "fields. Set submit_after=true (top-level OR per-field) to press "
        "Enter after typing. Each field requires a 'label' (visible text "
        "near the input) and 'value' (string to fill). The response "
        "always includes 'submitted': true if Enter was pressed (final "
        "or per-field), false otherwise. If submitted=true alongside "
        "captcha_required=true, the form may have ALREADY been submitted "
        "with partial data — re-snapshot before blindly resuming. Use "
        "this when you have a clear list of name/email/password fields; "
        "for adaptive flows where labels reveal themselves only after "
        "each click, use browser_get_elements + browser_type "
        "field-by-field instead."
    ),
    parameters={
        "fields": {
            "type": "array",
            "description": (
                "Ordered list of fields to fill. Each entry: "
                "{label: <visible label text>, value: <string to fill>, "
                "submit_after?: <press Enter after this field, default "
                "false>}. 1–50 entries; label 1–500 chars; value up to "
                "10000 chars."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": (
                            "Visible label text for the input "
                            "(case-insensitive substring match)."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to fill into the input.",
                    },
                    "submit_after": {
                        "type": "boolean",
                        "description": (
                            "Press Enter on this field after filling. "
                            "Default false."
                        ),
                        "default": False,
                    },
                },
                "required": ["label", "value"],
            },
        },
        "submit_after": {
            "type": "boolean",
            "description": (
                "Press Enter on the LAST filled field after the loop "
                "completes (skipped if a CAPTCHA stopped the flow). "
                "Default false."
            ),
            "default": False,
        },
    },
    parallel_safe=False,
)
async def browser_fill_form(
    fields: list, submit_after: bool = False, *, mesh_client=None,
) -> dict:
    """Fill a sequence of form fields by visible label."""
    if not isinstance(fields, list):
        return {"error": "The 'fields' parameter must be an array"}
    return await _browser_command(
        mesh_client, "fill_form",
        {"fields": fields, "submit_after": submit_after},
    )


@skill(
    name="browser_open_tab",
    description=(
        "Open a URL in a new browser tab. The new tab becomes the active "
        "page — subsequent browser_get_elements / browser_click / etc. "
        "operate on it. Cookies and storage are shared with existing "
        "tabs. Use browser_switch_tab to return to the previous tab. "
        "Set snapshot_after=true to include the element snapshot in the "
        "response (saves a separate browser_get_elements call)."
    ),
    parameters={
        "url": {
            "type": "string",
            "description": (
                "URL to open in a new tab. Must be http:// or https://."
            ),
        },
        "snapshot_after": {
            "type": "boolean",
            "description": (
                "Include element snapshot in the response (default false)."
            ),
            "default": False,
        },
    },
    parallel_safe=False,
)
async def browser_open_tab(
    url: str, snapshot_after: bool = False, *, mesh_client=None,
) -> dict:
    """Open a URL in a new tab and make it the active page."""
    if not url:
        return {"error": "The 'url' parameter is required"}
    try:
        scheme = urlparse(url).scheme.lower()
    except Exception:
        scheme = ""
    if scheme not in ("http", "https"):
        return {"error": "URL must start with http:// or https://"}
    return await _browser_command(
        mesh_client, "open_tab",
        {"url": url, "snapshot_after": snapshot_after},
    )


@skill(
    name="browser_inspect_requests",
    description=(
        "List recent network requests from the current browser context "
        "(URLs only, redacted; no bodies/headers). Useful for verifying "
        "which third-party endpoints a page hit, debugging when a page "
        "seems broken because adblock dropped requests, or confirming a "
        "form POST went through. Returns up to 50 most-recent entries by "
        "default. Set include_blocked=true to include adblock-suppressed "
        "entries (verbose; usually you don't want this).\n\n"
        "Buffer holds the most recent 200 requests per agent; older "
        "entries are evicted automatically. URLs are redacted at "
        "store-time (userinfo stripped, JWT-shaped path segments masked, "
        "sensitive query values like token/api_key replaced with "
        "[REDACTED]).\n\n"
        "Each entry has: url (str, redacted), method (str), "
        "resource_type (str, one of: document, stylesheet, image, media, "
        "font, script, texttrack, xhr, fetch, eventsource, websocket, "
        "manifest, other), ts (str, ISO-8601 UTC like "
        "'2026-04-26T12:34:56Z'), status (int|null — currently always "
        "null; reserved for future response-status capture), "
        "blocked_by_adblock (bool), user_cancelled (bool), "
        "failed_network (bool)."
    ),
    parameters={
        "include_blocked": {
            "type": "boolean",
            "description": (
                "Include requests blocked by the in-browser ad-blocker. "
                "Default false — adblock-suppressed trackers are usually "
                "noise. Set true when debugging missing analytics or "
                "third-party widgets."
            ),
            "default": False,
        },
        "limit": {
            "type": "integer",
            "description": (
                "Max number of newest-first entries to return. Default 50, "
                "max 200 (the underlying buffer size). Values above 200 "
                "are coerced to 200."
            ),
            "default": 50,
        },
    },
    parallel_safe=False,
)
async def browser_inspect_requests(
    include_blocked: bool = False, limit: int = 50, *, mesh_client=None,
) -> dict:
    """List recent network requests for the current browser context."""
    return await _browser_command(
        mesh_client, "inspect_requests",
        {"include_blocked": bool(include_blocked), "limit": int(limit)},
    )


@skill(
    name="browser_detect_captcha",
    description=(
        "Detect CAPTCHAs (reCAPTCHA, hCaptcha, Cloudflare Turnstile, etc.) "
        "on the current page. When a CAPTCHA solver is configured, "
        "CAPTCHAs are solved automatically after navigation. Returns the "
        "structured envelope: data.captcha_found is the primary signal; "
        "when true, also inspect data.kind (e.g. 'recaptcha-v2-checkbox', "
        "'turnstile', 'cf-interstitial-auto', 'unknown'), "
        "data.solver_outcome ('solved', 'no_solver', 'timeout', "
        "'rejected', ...), and data.next_action ('solved', 'wait', "
        "'notify_user', 'request_captcha_help', ...). The legacy "
        "data.type and data.message fields remain for back-compat but are "
        "deprecated — prefer the structured fields. solver_outcome=='solved' "
        "means no agent action is required; the captcha was already cleared."
    ),
    parameters={},
    parallel_safe=False,
)
async def browser_detect_captcha(*, mesh_client=None) -> dict:
    """Detect CAPTCHAs on the current page."""
    return await _browser_command(mesh_client, "detect_captcha")


def _upload_max_bytes() -> int:
    """Per-file upload cap. Read from the same env var as the mesh +
    browser layers so an operator override stays consistent across all
    three. Defaults to 50 MB."""
    import os as _os
    raw = _os.environ.get("OPENLEGION_UPLOAD_STAGE_MAX_MB", "50")
    try:
        mb = max(1, int(raw))
    except ValueError:
        mb = 50
    return mb * 1024 * 1024


_UPLOAD_MAX_FILES = 5


@skill(
    name="browser_upload_file",
    description=(
        "Upload one or more workspace files to a file-input element. "
        "Provide a `ref` from a prior browser_get_elements snapshot pointing "
        "at an <input type=\"file\"> (or aria-equivalent). `paths` is a list "
        "of workspace files (1..5) to upload — these are read from /data and "
        "forwarded to the browser. Each file ≤50MB. "
        "Optional `idempotency_key` allows the caller to dedupe retries "
        "explicitly across calls; when omitted a fresh key is generated. "
        "Returns {success, data: {uploaded: [path, ...]}}."
    ),
    parameters={
        "ref": {
            "type": "string",
            "description": "Element ref (e.g. 'e7') for the file-input element.",
        },
        "paths": {
            "type": "array",
            "description": (
                "Workspace paths under /data to upload (1..5 files). "
                "Paths may be passed without the /data/ prefix — e.g. "
                "'uploads/resume.pdf'. Each file must be <=50MB."
            ),
            "items": {"type": "string"},
        },
        "idempotency_key": {
            "type": "string",
            "description": (
                "Optional caller-supplied key for cross-call dedupe. "
                "Same key + same caller + same content within the stage "
                "TTL returns the existing staged handle."
            ),
        },
    },
    parallel_safe=False,
)
async def browser_upload_file(
    ref: str,
    paths: list[str],
    idempotency_key: str | None = None,
    *,
    mesh_client=None,
) -> dict:
    """Stage workspace files via mesh and drive the browser file-chooser."""
    if not mesh_client:
        return {"error": "Browser requires mesh connectivity"}
    if not ref or not isinstance(ref, str):
        return {"error": "ref is required"}
    if not isinstance(paths, list) or not paths:
        return {"error": "paths must be a non-empty list"}
    if len(paths) > _UPLOAD_MAX_FILES:
        return {"error": f"at most {_UPLOAD_MAX_FILES} files per upload"}
    if not all(isinstance(p, str) and p for p in paths):
        return {"error": "paths must be a list of non-empty strings"}

    from src.agent.builtins.file_tool import _safe_path

    resolved_paths: list = []
    suggested_filenames: list[str] = []
    for path in paths:
        try:
            safe = _safe_path(path)
        except (ValueError, OSError) as e:
            return {"error": f"Invalid workspace path '{path}': {e}"}
        if not safe.is_file():
            return {"error": f"Upload path not found: {path}"}
        try:
            size = safe.stat().st_size
        except OSError as e:
            return {"error": f"Cannot stat '{path}': {e}"}
        cap = _upload_max_bytes()
        if size > cap:
            return {
                "error": (
                    f"File '{path}' is {size} bytes; per-file cap is "
                    f"{cap} bytes ({cap // (1024 * 1024)}MB)"
                ),
            }
        resolved_paths.append(safe)
        suggested_filenames.append(safe.name)

    import uuid as _uuid
    idem_key = idempotency_key if isinstance(idempotency_key, str) and idempotency_key else _uuid.uuid4().hex
    staged_handles: list[str] = []
    try:
        for i, safe_path in enumerate(resolved_paths):
            stage_key = f"{idem_key}-{i}"
            with open(safe_path, "rb") as fh:
                stage_resp = await mesh_client.browser_upload_stage(
                    fh, idempotency_key=stage_key,
                )
            handle = stage_resp.get("staged_handle")
            if not handle:
                return {"error": "Mesh did not return a staged_handle"}
            staged_handles.append(handle)
    except Exception as e:
        return {"error": _deep_redact(str(e))}

    try:
        result = await mesh_client.browser_upload_apply({
            "ref": ref,
            "staged_handles": staged_handles,
            "suggested_filenames": suggested_filenames,
            "idempotency_key": idem_key,
        })
        return _deep_redact(result)
    except Exception as e:
        return {"error": _deep_redact(str(e))}


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


@skill(
    name="browser_download",
    description=(
        "Trigger a download by clicking the given ref and save the result "
        "to your /artifacts directory. The browser captures the download "
        "event (≤50MB) and the result becomes a normal artifact you can "
        "read with the file/artifact tools. "
        "Returns {success, data: {artifact_name, size_bytes, mime_type}}. "
        "Disabled fleet-wide if BROWSER_DOWNLOADS_DISABLED is set."
    ),
    parameters={
        "ref": {
            "type": "string",
            "description": "Element ref returned by browser_get_elements that triggers the download.",
        },
        "timeout_ms": {
            "type": "integer",
            "description": "How long to wait for the download to start (default 30000).",
            "default": 30000,
        },
    },
    parallel_safe=False,
)
async def browser_download(
    ref: str, timeout_ms: int = 30000,
    *, mesh_client=None,
) -> dict:
    """Trigger a download and save it as an artifact via mesh→agent ingest."""
    if not mesh_client:
        return {"error": "Browser download requires mesh connectivity"}
    if not ref:
        return {"error": "ref is required"}
    try:
        result = await mesh_client.browser_download(
            ref=ref, timeout_ms=timeout_ms,
        )
        return _deep_redact(result)
    except Exception as e:
        return {"error": _deep_redact(str(e))}
