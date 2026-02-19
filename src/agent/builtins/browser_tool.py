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


@skill(
    name="browser_navigate",
    description=(
        "Navigate your Chromium browser to a URL and return the page text. "
        "Use this to visit any website: sign-up pages, dashboards, search engines, "
        "web apps. You can then use browser_click and browser_type to interact."
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
        "Click an element on the current page. Use CSS selectors like "
        "'button[type=submit]', 'a.signup-link', '#next-btn', or 'input[name=email]'."
    ),
    parameters={
        "selector": {"type": "string", "description": "CSS selector of the element to click"},
    },
)
async def browser_click(selector: str) -> dict:
    """Click an element matching the CSS selector."""
    try:
        page = await _get_page()
        await page.click(selector, timeout=10000)
        await page.wait_for_timeout(500)
        return {"clicked": selector, "url": page.url}
    except Exception as e:
        return {"error": str(e), "selector": selector}


@skill(
    name="browser_type",
    description=(
        "Type text into a form field on the current page. Clears the field first, "
        "then enters the new text. Use with browser_click to fill out forms."
    ),
    parameters={
        "selector": {"type": "string", "description": "CSS selector of the input element"},
        "text": {"type": "string", "description": "Text to type"},
    },
)
async def browser_type(selector: str, text: str) -> dict:
    """Type text into an element matching the CSS selector."""
    try:
        page = await _get_page()
        await page.fill(selector, text, timeout=10000)
        return {"typed": text, "selector": selector}
    except Exception as e:
        return {"error": str(e), "selector": selector}


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
