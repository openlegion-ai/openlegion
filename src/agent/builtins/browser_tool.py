"""Browser automation via Playwright.

Provides headless Chromium access for web scraping, testing, and interaction.
A single browser instance is lazily initialized per agent process and reused.
Supports three backends via BROWSER_BACKEND env var: basic, stealth, advanced.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.browser")

_pw = None  # Playwright instance (needs explicit stop on cleanup)
_camoufox_cm = None  # Camoufox context manager (needs __aexit__ on cleanup)
_browser = None
_context = None
_page = None
_launch_lock = asyncio.Lock()
_page_refs: dict[str, object] = {}
_credential_filled_refs: set[str] = set()  # refs that had $CRED{} typed into them
_page_op_lock = asyncio.Lock()  # serializes all page operations (Playwright pages aren't concurrent-safe)

_ACTIONABLE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "searchbox", "slider", "spinbutton", "switch", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "treeitem",
})

_CONTEXT_ROLES = frozenset({
    "heading", "img", "dialog", "alertdialog", "alert",
})

_MAX_SNAPSHOT_ELEMENTS = 200

# Regex to parse aria_snapshot() YAML lines: `- role "name" [attr=val]`
_ARIA_LINE_RE = re.compile(
    r"^\s*-\s+"            # indent + bullet
    r"(\w+)"               # role
    r'(?:\s+"([^"]*)")?'   # optional "name"
    r"(?:\s+\[([^\]]*)\])?"  # optional [attributes]
)

_CRED_HANDLE_RE = re.compile(r"\$CRED\{([^}]+)\}")

# Patterns for redacting common secret formats in browser output
_REDACT_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),          # OpenAI / Anthropic keys
    re.compile(r"ghp_[A-Za-z0-9]{36,}"),            # GitHub PATs
    re.compile(r"gho_[A-Za-z0-9]{36,}"),            # GitHub OAuth tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),    # GitHub fine-grained PATs
    re.compile(r"xoxb-[A-Za-z0-9\-]{20,}"),         # Slack bot tokens
    re.compile(r"xoxp-[A-Za-z0-9\-]{20,}"),         # Slack user tokens
    re.compile(r"AKIA[A-Z0-9]{16}"),                 # AWS access key IDs
    re.compile(r"(?<![A-Za-z0-9])[A-Fa-f0-9]{40,}(?![A-Za-z0-9])"),  # 40+ hex chars
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9+/]{40,}={0,2}(?![A-Za-z0-9/+=])"),  # base64 40+ chars
]


def _redact_credentials(text: str) -> str:
    """Replace common secret patterns with [REDACTED]."""
    if not text:
        return text
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


async def _get_page(*, mesh_client=None):
    """Lazily initialize a persistent browser context and page."""
    global _browser, _context, _page
    async with _launch_lock:
        if _page and not _page.is_closed():
            return _page

        backend = os.environ.get("BROWSER_BACKEND", "basic")

        if backend == "stealth":
            _browser, _context, _page = await _launch_stealth()
        elif backend == "advanced":
            _browser, _context, _page = await _launch_advanced(mesh_client)
        else:
            _browser, _context, _page = await _launch_basic()

        return _page


async def _launch_basic():
    """Launch standard Playwright Chromium."""
    global _pw
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is not installed. The agent container must include "
            "playwright and chromium. See Dockerfile.agent."
        )
    _pw = await async_playwright().start()
    browser = await _pw.chromium.launch(headless=True)
    context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = await context.new_page()
    logger.info("Browser backend: basic (Playwright Chromium)")
    return browser, context, page


async def _launch_stealth():
    """Launch Camoufox anti-detect browser.

    Camoufox handles its own user-agent and fingerprint rotation,
    so we intentionally skip the custom user_agent set in _launch_basic().
    """
    global _camoufox_cm
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        raise RuntimeError(
            "camoufox is not installed. The agent container must include "
            "camoufox. See Dockerfile.agent."
        )
    _camoufox_cm = AsyncCamoufox(headless=True)
    browser = await _camoufox_cm.__aenter__()
    context = await browser.new_context(viewport={"width": 1280, "height": 720})
    page = await context.new_page()
    logger.info("Browser backend: stealth (Camoufox)")
    return browser, context, page


async def _launch_advanced(mesh_client):
    """Connect to Bright Data Scraping Browser via CDP."""
    global _pw
    if not mesh_client:
        raise RuntimeError(
            "Advanced browser backend requires mesh connectivity for credential resolution"
        )
    cdp_url = await mesh_client.vault_resolve("brightdata_cdp_url")
    if not cdp_url:
        raise RuntimeError(
            "Credential 'brightdata_cdp_url' not found in vault. "
            "Set OPENLEGION_CRED_BRIGHTDATA_CDP_URL on the host."
        )
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is not installed. The agent container must include "
            "playwright. See Dockerfile.agent."
        )
    _pw = await async_playwright().start()
    browser = await _pw.chromium.connect_over_cdp(cdp_url, timeout=120000)
    # Always create a fresh context for a clean state with known viewport
    context = await browser.new_context(viewport={"width": 1280, "height": 720})
    page = await context.new_page()
    logger.info("Browser backend: advanced (Bright Data CDP)")
    return browser, context, page


async def browser_cleanup():
    """Release browser resources. Called on agent shutdown."""
    global _pw, _camoufox_cm, _browser, _context, _page
    try:
        if _page and not _page.is_closed():
            await _page.close()
    except Exception as e:
        logger.debug("Error closing browser page: %s", e)
    try:
        if _context:
            await _context.close()
    except Exception as e:
        logger.debug("Error closing browser context: %s", e)
    try:
        if _browser:
            await _browser.close()
    except Exception as e:
        logger.debug("Error closing browser: %s", e)
    try:
        if _camoufox_cm:
            await _camoufox_cm.__aexit__(None, None, None)
    except Exception as e:
        logger.debug("Error closing camoufox: %s", e)
    try:
        if _pw:
            await _pw.stop()
    except Exception as e:
        logger.debug("Error stopping playwright: %s", e)
    _pw = _camoufox_cm = _browser = _context = _page = None


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
async def browser_navigate(url: str, wait_ms: int = 1000, *, mesh_client=None) -> dict:
    """Navigate to a URL and extract text content."""
    async with _page_op_lock:
        try:
            _page_refs.clear()
            _credential_filled_refs.clear()
            page = await _get_page(mesh_client=mesh_client)
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            if wait_ms:
                await page.wait_for_timeout(wait_ms)
            text = await page.inner_text("body")
            return {
                "url": page.url,
                "title": await page.title(),
                "status": response.status if response else 0,
                "content": _redact_credentials(text[:50000]),
            }
        except Exception as e:
            return {"error": str(e), "url": url}


def _parse_aria_snapshot(yaml_text: str) -> list[dict]:
    """Parse Playwright aria_snapshot() YAML into a flat list of elements."""
    elements: list[dict] = []
    for line in yaml_text.strip().splitlines():
        m = _ARIA_LINE_RE.match(line)
        if not m:
            continue
        role = m.group(1)
        name = m.group(2) or ""
        attrs_str = m.group(3) or ""

        if role not in _ACTIONABLE_ROLES | _CONTEXT_ROLES:
            continue
        if not name:
            continue

        entry: dict = {"role": role, "name": name}
        if attrs_str:
            for attr in attrs_str.split(","):
                attr = attr.strip()
                if "=" in attr:
                    k, v = attr.split("=", 1)
                    entry[k.strip()] = v.strip()
        elements.append(entry)
    return elements


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
async def browser_snapshot(*, mesh_client=None) -> dict:
    """Return an accessibility tree snapshot with element refs."""
    async with _page_op_lock:
        return await _browser_snapshot_inner(mesh_client=mesh_client)


async def _browser_snapshot_inner(*, mesh_client=None) -> dict:
    """Inner snapshot logic. Caller must hold _page_op_lock."""
    try:
        page = await _get_page(mesh_client=mesh_client)

        # Modern Playwright (1.49+): aria_snapshot() returns YAML text
        # Legacy Playwright: page.accessibility.snapshot() returns dict tree
        flat: list[dict] = []
        try:
            yaml_text = await page.locator("body").aria_snapshot()
            if not isinstance(yaml_text, str):
                raise TypeError("aria_snapshot did not return a string")
            flat = _parse_aria_snapshot(yaml_text)
        except (AttributeError, TypeError):
            # Fallback for older Playwright versions
            tree = await page.accessibility.snapshot()
            if tree:
                flat = _flatten_tree(tree)

        if not flat:
            return {"url": page.url, "title": await page.title(), "element_count": 0, "elements": []}

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

            # Redact common secret patterns from element text
            redacted_el = dict(el)
            if "name" in redacted_el:
                redacted_el["name"] = _redact_credentials(redacted_el["name"])
            if "value" in redacted_el:
                redacted_el["value"] = _redact_credentials(redacted_el["value"])

            entry = {"ref": ref, **redacted_el}
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


async def _draw_labels(image_path: str) -> dict[str, str]:
    """Overlay numbered labels on interactive elements in a screenshot.

    Iterates ``_page_refs`` (populated by ``browser_snapshot``), gets each
    element's bounding box, draws a red rectangle + white number label, and
    returns a mapping of label numbers to descriptions.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed — labeled screenshot unavailable")
        return {}

    if not _page_refs:
        return {}

    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
    except Exception as e:
        logger.warning("Failed to open screenshot for labeling: %s", e)
        return {}

    # Try DejaVu font, fall back to PIL default
    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            pass

    labels: dict[str, str] = {}
    for ref, locator in _page_refs.items():
        num = ref.lstrip("e")
        try:
            bbox = await locator.bounding_box(timeout=1000)
        except Exception:
            continue
        if bbox is None:
            continue

        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        # Draw red rectangle around element
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        # Draw label background + number
        label_text = num
        text_bbox = draw.textbbox((0, 0), label_text, font=font) if font else (0, 0, 10, 12)
        tw = text_bbox[2] - text_bbox[0] + 6
        th = text_bbox[3] - text_bbox[1] + 4
        label_y = max(0, y - th)
        draw.rectangle([x, label_y, x + tw, label_y + th], fill="red")
        draw.text((x + 3, label_y + 2), label_text, fill="white", font=font)

        labels[num] = f"{ref} at ({int(x)},{int(y)}) {int(w)}x{int(h)}"

    img.save(image_path)
    return labels


@skill(
    name="browser_screenshot",
    description=(
        "Take a screenshot of the current page. Saves to /data. "
        "Use labeled=true to overlay numbered labels on interactive elements "
        "(requires browser_snapshot to have been called first, or auto-calls it)."
    ),
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
        "labeled": {
            "type": "boolean",
            "description": "Overlay numbered labels on interactive elements (default false)",
            "default": False,
        },
    },
)
async def browser_screenshot(
    filename: str = "screenshot.png",
    full_page: bool = False,
    labeled: bool = False,
    *,
    mesh_client=None,
) -> dict:
    """Take a screenshot and save it to the data volume."""
    async with _page_op_lock:
        try:
            page = await _get_page(mesh_client=mesh_client)

            # Auto-snapshot if labeled requested but no refs available
            if labeled and not _page_refs:
                await _browser_snapshot_inner(mesh_client=mesh_client)

            save_path = Path("/data") / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(save_path), full_page=full_page)

            result: dict = {"path": str(save_path), "size": save_path.stat().st_size}

            if labeled:
                labels = await _draw_labels(str(save_path))
                result["labeled"] = True
                result["label_count"] = len(labels)
                result["labels"] = labels
                # Update file size after drawing labels
                result["size"] = save_path.stat().st_size

            return result
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
async def browser_click(selector: str = "", ref: str = "", *, mesh_client=None) -> dict:
    """Click an element by ref or CSS selector."""
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_snapshot) or 'selector' (CSS)"}
    async with _page_op_lock:
        try:
            page = await _get_page(mesh_client=mesh_client)
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
async def browser_type(
    text: str, selector: str = "", ref: str = "", *, mesh_client=None,
) -> dict:
    """Type text into an element by ref or CSS selector."""
    if not text:
        return {"error": "The 'text' parameter is required"}
    if not selector and not ref:
        return {"error": "Provide either 'ref' (from browser_snapshot) or 'selector' (CSS)"}

    # Resolve $CRED{name} handles to actual values (credential-blind typing)
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
        is_credential = True

    async with _page_op_lock:
        try:
            page = await _get_page(mesh_client=mesh_client)
            if ref:
                locator = _page_refs.get(ref)
                if not locator:
                    return {"error": f"Unknown ref '{ref}'. Call browser_snapshot first to get current refs."}
                await locator.fill(actual_text, timeout=10000)
            else:
                await page.fill(selector, actual_text, timeout=10000)

            # Track which refs have been filled with credentials
            if is_credential and ref:
                _credential_filled_refs.add(ref)

            # Never return credential values to the LLM
            display_text = "[credential]" if is_credential else text
            result: dict = {"typed": display_text}
            if ref:
                result["ref"] = ref
            else:
                result["selector"] = selector
            return result
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
async def browser_evaluate(script: str, *, mesh_client=None) -> dict:
    """Evaluate JavaScript in the page context."""
    async with _page_op_lock:
        try:
            page = await _get_page(mesh_client=mesh_client)
            result = await page.evaluate(script)
            # Redact any credential values that might appear in evaluate results
            if isinstance(result, str):
                result = _redact_credentials(result)
            elif isinstance(result, dict):
                result = {
                    k: _redact_credentials(v) if isinstance(v, str) else v
                    for k, v in result.items()
                }
            elif isinstance(result, list):
                result = [
                    _redact_credentials(item) if isinstance(item, str) else item
                    for item in result
                ]
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}
