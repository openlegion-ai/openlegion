"""Core browser manager — per-agent Camoufox instance lifecycle.

Manages lazy-started Camoufox browser instances, one per agent.
Each agent gets its own persistent profile, fingerprint, and
browser context on a shared Xvnc display.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from src.browser.redaction import CredentialRedactor
from src.browser.stealth import build_launch_options
from src.browser.timing import (
    action_delay,
    keystroke_delay,
    navigation_jitter,
    scroll_increment,
    scroll_pause,
)
from src.shared.utils import setup_logging

logger = setup_logging("browser.service")

_ACTIONABLE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "searchbox", "slider", "spinbutton", "switch", "tab", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "treeitem",
})

_CONTEXT_ROLES = frozenset({
    "heading", "img", "dialog", "alertdialog", "alert",
})

_MAX_SNAPSHOT_ELEMENTS = 200

_BLOCKED_URL_SCHEMES = frozenset({"file", "javascript", "data", "blob"})
_MAX_WAIT_MS = 10000  # 10 seconds max wait after navigation
_MAX_SCROLL_PX = 10000  # 10000 pixels max per scroll call
_AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


class CamoufoxInstance:
    """Wrapper around a single Camoufox browser for one agent."""

    def __init__(self, agent_id: str, browser, context, page):
        self.agent_id = agent_id
        self.browser = browser
        self.context = context
        self.page = page
        self.last_activity = time.time()
        self.refs: dict[str, dict] = {}  # ref_id -> {"role": ..., "name": ...}
        self.credential_filled_refs: set[str] = set()
        self.lock = asyncio.Lock()  # serialize page operations per instance

    def touch(self):
        self.last_activity = time.time()


class BrowserManager:
    """Manages per-agent Camoufox browser instances.

    Browsers are lazy-started on first use and auto-stopped after
    idle timeout. Max concurrent browsers is configurable.
    """

    def __init__(
        self,
        profiles_dir: str = "/data/profiles",
        max_concurrent: int = 5,
        idle_timeout_minutes: int = 30,
    ):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.idle_timeout = idle_timeout_minutes * 60
        self._instances: dict[str, CamoufoxInstance] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._playwright = None
        self.redactor = CredentialRedactor()

    async def start_cleanup_loop(self):
        """Start background task that cleans up idle browsers."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(60)
            try:
                await self._cleanup_idle()
            except Exception as e:
                logger.debug("Cleanup loop error: %s", e)

    async def _cleanup_idle(self):
        now = time.time()
        async with self._lock:
            to_stop = [
                agent_id for agent_id, inst in self._instances.items()
                if now - inst.last_activity > self.idle_timeout
            ]
            for agent_id in to_stop:
                logger.info("Stopping idle browser for '%s'", agent_id)
                await self._stop_instance(agent_id)

    async def get_or_start(self, agent_id: str) -> CamoufoxInstance:
        """Get existing browser or start a new one for the agent."""
        if not _AGENT_ID_RE.match(agent_id):
            raise ValueError(f"Invalid agent_id: {agent_id!r}")
        async with self._lock:
            if agent_id in self._instances:
                inst = self._instances[agent_id]
                inst.touch()
                return inst

            # Enforce max concurrent
            if len(self._instances) >= self.max_concurrent:
                # Stop least recently used
                oldest_id = min(self._instances, key=lambda a: self._instances[a].last_activity)
                logger.info("Max browsers reached, stopping LRU '%s'", oldest_id)
                await self._stop_instance(oldest_id)

            # Start while holding lock to prevent duplicate instances for same agent
            instance = await self._start_browser(agent_id)
            self._instances[agent_id] = instance
            return instance

    async def _ensure_playwright(self):
        """Start the shared Playwright instance if not running."""
        if self._playwright is None:
            from playwright.async_api import async_playwright
            ctx = async_playwright()
            pw = await ctx.start()
            self._pw_context = ctx
            self._playwright = pw
        return self._playwright

    async def _start_browser(self, agent_id: str) -> CamoufoxInstance:
        """Launch a Camoufox browser for an agent."""
        from camoufox.async_api import AsyncNewBrowser

        pw = await self._ensure_playwright()

        profile_dir = str(self.profiles_dir / agent_id)
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        options = build_launch_options(agent_id, profile_dir)
        logger.info("Starting Camoufox for '%s' (profile=%s)", agent_id, profile_dir)

        # persistent_context=True → returns a BrowserContext directly
        browser = await AsyncNewBrowser(pw, **options)
        context = browser
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        return CamoufoxInstance(agent_id, browser, context, page)

    async def stop(self, agent_id: str) -> None:
        """Stop and clean up a specific agent's browser."""
        async with self._lock:
            await self._stop_instance(agent_id)

    async def _stop_instance(self, agent_id: str) -> None:
        """Internal stop — caller must hold self._lock."""
        inst = self._instances.pop(agent_id, None)
        if inst is None:
            return
        try:
            await inst.context.close()
        except Exception as e:
            logger.debug("Error closing browser for '%s': %s", agent_id, e)
        self.redactor.clear_agent(agent_id)
        logger.info("Stopped browser for '%s'", agent_id)

    async def stop_all(self) -> None:
        """Stop all browser instances and clean up Playwright."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        async with self._lock:
            for agent_id in list(self._instances.keys()):
                await self._stop_instance(agent_id)
        if self._playwright:
            try:
                await self._pw_context.__aexit__(None, None, None)
            except Exception:
                pass
            self._playwright = None

    async def reset(self, agent_id: str) -> None:
        """Reset browser session — close and reopen (preserves profile)."""
        await self.stop(agent_id)
        # Next get_or_start will create a fresh instance with same profile

    async def get_status(self, agent_id: str) -> dict:
        """Get status for a specific agent's browser."""
        async with self._lock:
            inst = self._instances.get(agent_id)
            if not inst:
                return {"running": False}
            return {
                "running": True,
                "idle_seconds": int(time.time() - inst.last_activity),
                "url": inst.page.url if inst.page else "",
            }

    async def get_service_status(self) -> dict:
        """Get overall service health."""
        async with self._lock:
            return {
                "healthy": True,
                "active_browsers": len(self._instances),
                "max_concurrent": self.max_concurrent,
                "agents": list(self._instances.keys()),
            }

    async def focus(self, agent_id: str) -> bool:
        """Bring an agent's browser window to VNC foreground.

        Auto-starts the browser if it isn't running yet, so the user
        always sees a window when they click "Browser" in the dashboard.
        """
        inst = await self.get_or_start(agent_id)
        async with inst.lock:
            try:
                await inst.page.bring_to_front()
                inst.touch()
                return True
            except Exception as e:
                logger.debug("Focus failed for '%s': %s", agent_id, e)
                return False

    # ── Browser operations ──────────────────────────────────

    async def navigate(self, agent_id: str, url: str, wait_ms: int = 1000) -> dict:
        """Navigate to URL and return page text."""
        # Validate URL scheme
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
        except Exception:
            return {"success": False, "error": "Invalid URL"}
        if parsed.scheme.lower() in _BLOCKED_URL_SCHEMES:
            return {"success": False, "error": f"URL scheme '{parsed.scheme}' is not allowed"}
        # Cap wait_ms
        wait_ms = max(0, min(wait_ms, _MAX_WAIT_MS))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                await inst.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                if wait_ms > 0:
                    await asyncio.sleep(wait_ms / 1000 + navigation_jitter())
                title = await inst.page.title()
                current_url = inst.page.url
                # Extract body text for the agent (truncated to prevent huge payloads)
                body_text = await inst.page.evaluate("() => document.body?.innerText?.slice(0, 20000) || ''")
                return {
                    "success": True,
                    "data": {
                        "url": self.redactor.redact(agent_id, current_url),
                        "title": self.redactor.redact(agent_id, title),
                        "body": self.redactor.redact(agent_id, body_text),
                    },
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def snapshot(self, agent_id: str) -> dict:
        """Get accessibility tree with element refs."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                tree = await inst.page.accessibility.snapshot()
                if not tree:
                    return {"success": True, "data": {"snapshot": "(empty page)", "refs": {}}}

                lines = []
                refs: dict[str, dict] = {}
                ref_counter = [0]

                def _walk(node, depth=0):
                    role = node.get("role", "")
                    name = node.get("name", "")
                    if role in _ACTIONABLE_ROLES or role in _CONTEXT_ROLES:
                        if ref_counter[0] < _MAX_SNAPSHOT_ELEMENTS:
                            ref_id = f"e{ref_counter[0]}"
                            ref_counter[0] += 1
                            attrs = []
                            if node.get("checked") is not None:
                                attrs.append(f"checked={node['checked']}")
                            if node.get("selected"):
                                attrs.append("selected")
                            if node.get("disabled"):
                                attrs.append("disabled")
                            if node.get("value"):
                                val = node["value"]
                                if ref_id in inst.credential_filled_refs:
                                    val = "****"
                                attrs.append(f"value={val}")
                            attr_str = f" [{', '.join(attrs)}]" if attrs else ""
                            line = f"{'  ' * depth}- [{ref_id}] {role} \"{name}\"{attr_str}"
                            lines.append(line)
                            refs[ref_id] = {"role": role, "name": name}
                    for child in node.get("children", []):
                        _walk(child, depth + 1)

                _walk(tree)
                inst.refs = refs  # Store refs for click/type by ref ID
                snapshot_text = "\n".join(lines) if lines else "(no interactive elements)"
                snapshot_text = self.redactor.redact(agent_id, snapshot_text)
                return {"success": True, "data": {"snapshot": snapshot_text, "refs": refs}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    def _locator_from_ref(self, inst: CamoufoxInstance, ref: str):
        """Build a Playwright locator from a stored ref's role+name."""
        info = inst.refs.get(ref)
        if not info:
            return None
        role = info["role"]
        name = info.get("name", "")
        if name:
            return inst.page.get_by_role(role, name=name)
        return inst.page.get_by_role(role)

    async def click(self, agent_id: str, ref: str | None = None, selector: str | None = None) -> dict:
        """Click element by ref or CSS selector."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if ref and ref in inst.refs:
                    locator = self._locator_from_ref(inst, ref)
                    if locator:
                        await locator.click(timeout=5000)
                    else:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                elif selector:
                    await inst.page.click(selector, timeout=5000)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"clicked": ref or selector}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def type_text(self, agent_id: str, ref: str | None = None, selector: str | None = None,
                        text: str = "", clear: bool = True, is_credential: bool = False) -> dict:
        """Type text into element. Credential values should be pre-resolved by agent."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if ref and ref in inst.refs:
                    locator = self._locator_from_ref(inst, ref)
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    if clear:
                        await locator.fill(text)
                    else:
                        await locator.click(timeout=5000)
                        await self._type_with_variance(inst.page, text)
                elif selector:
                    if clear:
                        await inst.page.fill(selector, text)
                    else:
                        await inst.page.click(selector, timeout=5000)
                        await self._type_with_variance(inst.page, text)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}

                # Only track credential values for redaction, not all typed text
                if is_credential:
                    self.redactor.track_resolved_value(agent_id, text)
                    if ref:
                        inst.credential_filled_refs.add(ref)

                return {"success": True, "data": {"typed_into": ref or selector, "length": len(text)}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def evaluate(self, agent_id: str, expression: str) -> dict:
        """Execute JavaScript and return result.

        Intentionally NOT exposed via the HTTP API (server.py) — arbitrary
        JS execution is a sandbox-escape vector.  Used only internally
        (e.g. navigate body text extraction, scrolling).
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                result = await inst.page.evaluate(expression)
                result = self.redactor.deep_redact(agent_id, result)
                return {"success": True, "data": {"result": result}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def screenshot(self, agent_id: str, full_page: bool = False) -> dict:
        """Take screenshot, return base64 PNG."""
        import base64

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                png_bytes = await inst.page.screenshot(full_page=full_page)
                b64 = base64.b64encode(png_bytes).decode()
                return {"success": True, "data": {"image_base64": b64, "format": "png"}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _type_with_variance(self, page, text: str) -> None:
        """Type text character-by-character with human-like inter-key delays."""
        for char in text:
            await page.keyboard.press(char)
            await asyncio.sleep(keystroke_delay(char))

    async def scroll(self, agent_id: str, direction: str = "down",
                     amount: int = 0, ref: str | None = None) -> dict:
        """Smooth-scroll the page in randomized increments.

        Args:
            direction: "up" or "down"
            amount: total pixels to scroll (0 = one viewport height)
            ref: element ref to scroll into view instead of pixel scrolling
        """
        if direction not in ("up", "down"):
            return {"success": False, "error": f"Invalid direction: {direction!r} (use 'up' or 'down')"}

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                # Scroll element into view
                if ref:
                    if ref not in inst.refs:
                        return {"success": False, "error": f"Ref '{ref}' not found in snapshot"}
                    locator = self._locator_from_ref(inst, ref)
                    if locator:
                        await locator.scroll_into_view_if_needed(timeout=5000)
                        return {"success": True, "data": {"scrolled_to_ref": ref}}
                    return {"success": False, "error": f"Ref '{ref}' not found"}

                # Pixel-based scrolling
                if amount <= 0:
                    vp = inst.page.viewport_size
                    amount = vp["height"] if vp else 800
                amount = min(amount, _MAX_SCROLL_PX)

                sign = -1 if direction == "up" else 1
                scrolled = 0
                while scrolled < amount:
                    step = min(scroll_increment(), amount - scrolled)
                    delta = step * sign
                    await inst.page.evaluate(
                        f"window.scrollBy({{top: {delta}, behavior: 'smooth'}})"
                    )
                    scrolled += step
                    if scrolled < amount:
                        await asyncio.sleep(scroll_pause())

                return {
                    "success": True,
                    "data": {"direction": direction, "pixels": scrolled},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def solve_captcha(self, agent_id: str) -> dict:
        """Attempt CAPTCHA detection and solving."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                captcha_selectors = [
                    'iframe[src*="recaptcha"]',
                    'iframe[src*="hcaptcha"]',
                    'iframe[src*="captcha"]',
                    '[class*="captcha"]',
                    '#captcha',
                ]
                found = None
                for sel in captcha_selectors:
                    count = await inst.page.locator(sel).count()
                    if count > 0:
                        found = sel
                        break

                if not found:
                    return {"success": True, "data": {"captcha_found": False, "message": "No CAPTCHA detected"}}

                return {
                    "success": True,
                    "data": {
                        "captcha_found": True,
                        "captcha_type": found,
                        "message": "CAPTCHA detected. Manual solving may be required via VNC.",
                    },
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
