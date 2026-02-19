"""Browser automation via Playwright.

Provides headless Chromium access for web scraping, testing, and interaction.
A single browser instance is lazily initialized per agent process and reused.
"""

from __future__ import annotations

from pathlib import Path

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.browser")

_browser = None
_context = None
_page = None
_page_refs: dict[str, object] = {}

_ACTIONABLE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "searchbox", "slider", "spinbutton", "switch", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "treeitem",
})

_CONTEXT_ROLES = frozenset({
    "heading", "img", "dialog", "alertdialog", "alert",
})

_MAX_SNAPSHOT_ELEMENTS = 200


async def _get_page():
    """Lazily initialize a persistent browser context and page."""
    global _browser, _context, _page
    if _page and not _page.is_closed():
        return _page

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is not installed. The agent container must include "
            "playwright and chromium. See Dockerfile.agent."
        )

    pw = await async_playwright().start()
    _browser = await pw.chromium.launch(headless=True)
    _context = await _browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    _page = await _context.new_page()
    return _page


def _flatten_tree(node: dict) -> list[dict]:
    """Recursively flatten an accessibility tree into actionable/context elements."""
    results: list[dict] = []
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    if role in _ACTIONABLE_ROLES | _CONTEXT_ROLES and (name or value):
        entry: dict = {"role": role, "name": name}
        if value:
            entry["value"] = value
        if "checked" in node:
            entry["checked"] = node["checked"]
        if "level" in node:
            entry["level"] = node["level"]
        if "disabled" in node:
            entry["disabled"] = node["disabled"]
        if "required" in node:
            entry["required"] = node["required"]
        if "selected" in node:
            entry["selected"] = node["selected"]
        if "expanded" in node:
            entry["expanded"] = node["expanded"]
        results.append(entry)

    for child in node.get("children", []):
        results.extend(_flatten_tree(child))

    return results


@skill(
    name="browser_navigate",
    description=(
        "Navigate your Chromium browser to a URL and return the page text. "
        "Use this to visit any website: sign-up pages, dashboards, search engines, "
        "web apps. You can then use browser_snapshot to get element refs, "
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
async def browser_navigate(url: str, wait_ms: int = 1000) -> dict:
    """Navigate to a URL and extract text content."""
    try:
        _page_refs.clear()
        page = await _get_page()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        if wait_ms:
            await page.wait_for_timeout(wait_ms)
        text = await page.inner_text("body")
        return {
            "url": page.url,
            "title": await page.title(),
            "status": response.status if response else 0,
            "content": text[:50000],
        }
    except Exception as e:
        return {"error": str(e), "url": url}


@skill(
    name="browser_snapshot",
    description=(
        "Get a structured accessibility snapshot of the current page. "
        "Returns interactive elements (buttons, links, inputs, etc.) with "
        "ref IDs (e1, e2, ...) that you can pass to browser_click(ref=) "
        "and browser_type(ref=). Call this after browser_navigate and after "
        "any action that changes the page."
    ),
    parameters={},
)
async def browser_snapshot() -> dict:
    """Return an accessibility tree snapshot with element refs."""
    try:
        page = await _get_page()
        tree = await page.accessibility.snapshot()
        if not tree:
            return {"url": page.url, "title": await page.title(), "element_count": 0, "elements": []}

        flat = _flatten_tree(tree)
        truncated = len(flat) > _MAX_SNAPSHOT_ELEMENTS
        flat = flat[:_MAX_SNAPSHOT_ELEMENTS]

        _page_refs.clear()

        # Track role+name occurrences for disambiguation via .nth()
        seen: dict[tuple[str, str], int] = {}
        elements = []
        for i, el in enumerate(flat):
            ref = f"e{i + 1}"
            role = el["role"]
            name = el["name"]
            key = (role, name)

            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1

            locator = page.get_by_role(role, name=name, exact=True)
            if occurrence > 0:
                locator = locator.nth(occurrence)

            _page_refs[ref] = locator

            entry = {"ref": ref, **el}
            elements.append(entry)

        result: dict = {
            "url": page.url,
            "title": await page.title(),
            "element_count": len(elements),
            "elements": elements,
        }
        if truncated:
            result["truncated"] = True
        return result
    except Exception as e:
        return {"error": str(e)}


@skill(
    name="browser_screenshot",
    description="Take a screenshot of the current page. Saves to /data.",
    parameters={
        "filename": {
            "type": "string",
            "description": "Filename to save as (default 'screenshot.png')",
            "default": "screenshot.png",
        },
        "full_page": {
            "type": "boolean",
            "description": "Capture full scrollable page (default false)",
            "default": False,
        },
    },
)
async def browser_screenshot(
    filename: str = "screenshot.png", full_page: bool = False
) -> dict:
    """Take a screenshot and save it to the data volume."""
    try:
        page = await _get_page()
        save_path = Path("/data") / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(save_path), full_page=full_page)
        return {"path": str(save_path), "size": save_path.stat().st_size}
    except Exception as e:
        return {"error": str(e)}


@skill(
    name="browser_click",
    description=(
        "Click an element on the current page. Preferred: use ref from browser_snapshot "
        "(e.g. ref='e3'). Fallback: use a CSS selector."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the element to click (optional if ref is given)",
            "default": "",
        },
        "ref": {
            "type": "string",
            "description": "Element ref from browser_snapshot (e.g. 'e1')",
            "default": "",
        },
    },
)
async def browser_click(selector: str = "", ref: str = "") -> dict:
    """Click an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_snapshot) or 'selector' (CSS)"}
    try:
        page = await _get_page()
        if ref:
            locator = _page_refs.get(ref)
            if not locator:
                return {"error": f"Unknown ref '{ref}'. Call browser_snapshot first to get current refs."}
            await locator.click(timeout=10000)
            await page.wait_for_timeout(500)
            return {"clicked": ref, "url": page.url}
        else:
            await page.click(selector, timeout=10000)
            await page.wait_for_timeout(500)
            return {"clicked": selector, "url": page.url}
    except Exception as e:
        return {"error": str(e), "selector": selector, "ref": ref}


@skill(
    name="browser_type",
    description=(
        "Type text into a form field on the current page. Clears the field first, "
        "then enters the new text. Preferred: use ref from browser_snapshot "
        "(e.g. ref='e5'). Fallback: use a CSS selector."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the input element (optional if ref is given)",
            "default": "",
        },
        "text": {"type": "string", "description": "Text to type"},
        "ref": {
            "type": "string",
            "description": "Element ref from browser_snapshot (e.g. 'e1')",
            "default": "",
        },
    },
)
async def browser_type(text: str, selector: str = "", ref: str = "") -> dict:
    """Type text into an element by ref or CSS selector."""
    if not text:
        return {"error": "The 'text' parameter is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_snapshot) or 'selector' (CSS)"}
    try:
        page = await _get_page()
        if ref:
            locator = _page_refs.get(ref)
            if not locator:
                return {"error": f"Unknown ref '{ref}'. Call browser_snapshot first to get current refs."}
            await locator.fill(text, timeout=10000)
            return {"typed": text, "ref": ref}
        else:
            await page.fill(selector, text, timeout=10000)
            return {"typed": text, "selector": selector}
    except Exception as e:
        return {"error": str(e), "selector": selector, "ref": ref}


@skill(
    name="browser_evaluate",
    description=(
        "Execute JavaScript in the browser page and return the result. "
        "Use for DOM inspection, extracting data, or triggering page actions "
        "that CSS selectors can't reach."
    ),
    parameters={
        "script": {"type": "string", "description": "JavaScript code to evaluate"},
    },
)
async def browser_evaluate(script: str) -> dict:
    """Evaluate JavaScript in the page context."""
    try:
        page = await _get_page()
        result = await page.evaluate(script)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
