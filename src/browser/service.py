"""Core browser manager — per-agent Camoufox instance lifecycle.

Manages lazy-started Camoufox browser instances, one per agent.
Each agent gets its own persistent profile, fingerprint, and
browser context on a shared Xvnc display.
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from src.browser.redaction import CredentialRedactor
from src.browser.stealth import build_launch_options
from src.browser.timing import (
    action_delay,
    keystroke_delay,
    navigation_jitter,
    scroll_increment,
    scroll_pause,
    think_pause,
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
    "listbox", "tree", "grid", "toolbar", "menu", "status",
})

_MAX_SNAPSHOT_ELEMENTS = 200

# Block schemes that could expose local files or browser internals.
# about: covers about:logins (saved passwords), about:config, etc.
# moz-extension: / chrome-extension: cover installed extensions.
_BLOCKED_URL_SCHEMES = frozenset({
    "file", "javascript", "data", "blob",
    "about", "moz-extension", "chrome-extension", "chrome",
})
_MAX_WAIT_MS = 10000  # 10 seconds max wait after navigation
_MAX_SCROLL_PX = 10000  # 10000 pixels max per scroll call
_CLICK_TIMEOUT_MS = 10000  # 10 seconds — SPAs like X need time for animations/overlays
_WAIT_FOR_TIMEOUT_MS = 30000  # 30 seconds max for wait_for_element
_AGENT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_VALID_WAIT_UNTIL = frozenset({"domcontentloaded", "load", "networkidle", "commit"})
# Characters that mark a natural word/clause boundary in typed text.
# After one of these, the next character gets a higher think-pause probability
# to model the hesitation a human feels when starting the next word or sentence.
_WORD_BOUNDARY_CHARS = frozenset(" ,.:;!?\n\t")
# Roles where aria-disabled="true" should NOT block click attempts.
# SPA frameworks (X/Twitter, Gmail) keep aria-disabled on buttons/links while
# handling clicks via JS — the visual state and handler are the source of truth,
# not the ARIA attribute.  Intentionally narrow: menuitem, switch, option are
# excluded because force-clicking genuinely disabled items in those roles causes
# unwanted side-effects (selecting unavailable options, toggling locked switches).
_ARIA_FORCE_ROLES = frozenset({"button", "link"})
# CSS selector for modal dialog detection via DOM queries.
# Used in both snapshot() (to scope the a11y tree) and _locator_from_ref()
# (to scope click/type locators). Must stay in sync — hence a single constant.
_MODAL_SELECTOR = (
    '[role="dialog"]:not([aria-hidden="true"]), '
    '[aria-modal="true"]:not([aria-hidden="true"]), '
    'dialog[open]'
)

# ── JS-based accessibility tree builder ──────────────────────────────────
# Fallback when page.accessibility.snapshot() is unavailable (Camoufox
# bundles a Playwright version that removed or never exposed the API).
# Walks the DOM using standard APIs (getAttribute, getComputedStyle) and
# returns the same {role, name, children, disabled, ...} tree structure
# that the Python _walk() function expects.
#
# Called as:
#   page.evaluate(_JS_A11Y_TREE)          — full page tree
#   element_handle.evaluate(_JS_A11Y_TREE) — scoped to element
_JS_A11Y_TREE = r"""(rootEl) => {
    const ACTIONABLE = new Set([
        'button','link','textbox','checkbox','radio','combobox','searchbox',
        'slider','spinbutton','switch','tab','menuitem','menuitemcheckbox',
        'menuitemradio','option','treeitem'
    ]);
    const CONTEXT = new Set([
        'heading','img','dialog','alertdialog','alert',
        'listbox','tree','grid','toolbar','menu','status'
    ]);
    const ROLES = new Set([...ACTIONABLE, ...CONTEXT]);
    const LANDMARK = new Set([
        'navigation','main','complementary','banner','contentinfo',
        'form','region','dialog','alertdialog'
    ]);
    const IMPLICIT = {
        BUTTON:'button',TEXTAREA:'textbox',SELECT:'combobox',OPTION:'option',
        IMG:'img',H1:'heading',H2:'heading',H3:'heading',
        H4:'heading',H5:'heading',H6:'heading',DIALOG:'dialog',
        NAV:'navigation',MAIN:'main',HEADER:'banner',FOOTER:'contentinfo',
        ASIDE:'complementary',FORM:'form'
    };
    const INPUT_ROLES = {
        text:'textbox',email:'textbox',url:'textbox',tel:'textbox',
        password:'textbox',search:'searchbox',
        checkbox:'checkbox',radio:'radio',
        range:'slider',number:'spinbutton',
        submit:'button',reset:'button',button:'button'
    };
    let refCounter = 0;
    function getRole(el) {
        const r = el.getAttribute('role');
        if (r) return r.split(/\s+/)[0].toLowerCase();
        if (el.tagName === 'A') return el.hasAttribute('href') ? 'link' : null;
        if (el.tagName === 'INPUT') return INPUT_ROLES[(el.type||'text').toLowerCase()] || null;
        if (el.getAttribute('contenteditable') === 'true') return 'textbox';
        return IMPLICIT[el.tagName] || null;
    }
    function getName(el, role) {
        let n = el.getAttribute('aria-label');
        if (n) return n.trim();
        const by = el.getAttribute('aria-labelledby');
        if (by) {
            const t = by.split(/\s+/).map(id => {
                const ref = document.getElementById(id);
                return ref ? ref.textContent.trim() : '';
            }).filter(Boolean).join(' ');
            if (t) return t;
        }
        if (el.tagName === 'IMG') return (el.alt || '').trim();
        if (['INPUT','TEXTAREA','SELECT'].includes(el.tagName)) {
            if (el.id) {
                const lbl = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
                if (lbl) return lbl.textContent.trim().slice(0, 200);
            }
            const wrap = el.closest('label');
            if (wrap) {
                const c = wrap.cloneNode(true);
                c.querySelectorAll('input,textarea,select').forEach(i => i.remove());
                const t = c.textContent.trim();
                if (t) return t.slice(0, 200);
            }
            return (el.placeholder || el.title || '').trim();
        }
        if (['button','link','tab','menuitem','menuitemcheckbox','menuitemradio',
            'switch','option','treeitem','heading','alert','alertdialog','dialog',
            'listbox','toolbar','menu','status'
        ].includes(role)) {
            const t = el.textContent;
            if (t) { const s = t.trim(); if (s) return s.slice(0, 200); }
        }
        return (el.title || '').trim();
    }
    function isVisible(el) {
        if (el.getAttribute('aria-hidden') === 'true') return false;
        const s = getComputedStyle(el);
        if (s.visibility === 'hidden' || s.visibility === 'collapse') return false;
        if (parseFloat(s.opacity) === 0) return false;
        if (!el.offsetParent && el !== document.body && el !== document.documentElement) {
            if (s.display === 'none') return false;
            if (s.position !== 'fixed' && s.position !== 'sticky') return false;
        }
        return true;
    }
    function walk(el, d, parentLandmark) {
        if (d > 50 || !el || el.nodeType !== 1) return null;
        const tag = el.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'NOSCRIPT' || tag === 'TEMPLATE')
            return null;
        if (!isVisible(el)) return null;
        const role = getRole(el);
        let childLandmark = parentLandmark;
        if (role && LANDMARK.has(role)) {
            const lname = getName(el, role);
            childLandmark = lname ? role + ': ' + lname.slice(0, 50) : role;
        }
        const children = [];
        for (const child of el.children) {
            const r = walk(child, d + 1, childLandmark);
            if (r) children.push(r);
        }
        if (!role || !ROLES.has(role)) {
            if (!children.length) return null;
            if (children.length === 1) return children[0];
            return { role: 'none', name: '', children };
        }
        const nd = { role, name: getName(el, role) };
        if (refCounter < 200) {
            const refId = 'e' + refCounter;
            el.setAttribute('data-olref', refId);
            nd.refId = refId;
            refCounter++;
        }
        if (parentLandmark) nd.landmark = parentLandmark;
        if (el.disabled || el.getAttribute('aria-disabled') === 'true') nd.disabled = true;
        const chkRoles = ['checkbox','radio','switch','menuitemcheckbox','menuitemradio'];
        if (chkRoles.includes(role)) {
            nd.checked = !!(el.checked) || el.getAttribute('aria-checked') === 'true';
        }
        if (el.getAttribute('aria-selected') === 'true') nd.selected = true;
        if ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && el.value !== '') {
            nd.value = String(el.value).slice(0, 500);
        }
        if (el.getAttribute('contenteditable') === 'true' && el.textContent) {
            const cv = el.textContent.trim();
            if (cv && !nd.value) nd.value = cv.slice(0, 500);
        }
        if (children.length) nd.children = children;
        return nd;
    }
    document.querySelectorAll('[data-olref]').forEach(el => el.removeAttribute('data-olref'));
    const start = rootEl || document.body || document.documentElement;
    const tree = walk(start, 0, null);
    if (rootEl) return tree || { role: 'none', name: '', children: [] };
    if (!tree) return { role: 'WebArea', name: document.title || '', children: [] };
    if (tree.role === 'none')
        return { role: 'WebArea', name: document.title || '', children: tree.children || [] };
    return { role: 'WebArea', name: document.title || '', children: [tree] };
}"""


class CamoufoxInstance:
    """Wrapper around a single Camoufox browser for one agent."""

    def __init__(self, agent_id: str, browser, context, page):
        self.agent_id = agent_id
        self.browser = browser
        self.context = context
        self.page = page
        self.last_activity = time.time()
        self.refs: dict[str, dict] = {}  # ref_id -> {"role", "name", "index", "disabled"}
        self.credential_filled_refs: set[str] = set()
        self.dialog_active: bool = False  # True when snapshot scoped to a modal dialog
        self.dialog_detected: bool = False  # True when a modal was found (even if scoping failed)
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
        self._js_snapshot_mode: bool = False  # True after page.accessibility fails
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
                logger.warning("Cleanup loop error: %s", e)

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

    async def touch_all(self) -> int:
        """Reset the idle timer for every running browser instance.

        Called by the VNC keepalive while a user is actively viewing the
        display, so a watched browser is never killed by the idle cleanup.
        Returns the number of instances touched.
        """
        async with self._lock:
            for inst in self._instances.values():
                inst.touch()
            return len(self._instances)

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

        Two-layer raise:
        1. bring_to_front() — browser-protocol level (activates the tab)
        2. xdotool windowmap + windowraise — X11 level (unmaps if iconic,
           then raises in the stacking order so VNC actually sees it)
        """
        try:
            inst = await self.get_or_start(agent_id)
        except Exception as e:
            logger.debug("Focus get_or_start failed for '%s': %s", agent_id, e)
            return False
        async with inst.lock:
            try:
                await inst.page.bring_to_front()
                inst.touch()
            except Exception as e:
                logger.debug("Focus failed for '%s': %s", agent_id, e)
                return False
            # Best-effort X11 raise so VNC sees the window. bring_to_front()
            # only works at the browser-protocol layer; on X11 with Openbox
            # the OS window can still be below the root window (e.g. after a
            # popup closes without returning focus). windowmap handles the
            # minimised/iconic case; windowraise moves it to the top of the
            # stacking order. Failures here are non-fatal — the tab is already
            # focused at the protocol level.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "search", "--class", "firefox",
                         "windowmap", "--sync", "windowraise", "windowfocus"],
                        capture_output=True, timeout=3,
                    ),
                )
            except Exception as e:
                logger.debug("xdotool raise skipped for '%s': %s", agent_id, e)
            return True

    # ── Browser operations ──────────────────────────────────

    async def navigate(
        self, agent_id: str, url: str, wait_ms: int = 1000,
        wait_until: str = "domcontentloaded",
        snapshot_after: bool = False,
    ) -> dict:
        """Navigate to URL and return page text.

        wait_until controls when Playwright considers navigation complete:
          - "domcontentloaded" (default): HTML parsed; fast but JS may not have run
          - "load": all resources loaded; good for most sites
          - "networkidle": no network requests for 500ms; best for heavy SPAs (X, etc.)
          - "commit": first byte received; fastest, rarely useful
        """
        # Validate URL scheme
        try:
            parsed = urlparse(url)
        except Exception:
            return {"success": False, "error": "Invalid URL"}
        if parsed.scheme.lower() in _BLOCKED_URL_SCHEMES:
            return {"success": False, "error": f"URL scheme '{parsed.scheme}' is not allowed"}
        if wait_until not in _VALID_WAIT_UNTIL:
            valid = sorted(_VALID_WAIT_UNTIL)
            return {"success": False, "error": f"Invalid wait_until: {wait_until!r}. Use one of: {valid}"}
        # Cap wait_ms
        wait_ms = max(0, min(wait_ms, _MAX_WAIT_MS))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            # Single retry on timeout — transient network issues get a second chance.
            for attempt in range(2):
                try:
                    await inst.page.goto(url, wait_until=wait_until, timeout=30000)
                    break
                except Exception as e:
                    if attempt == 0 and "timeout" in str(e).lower():
                        logger.debug("Navigation timeout, retrying: %s", url)
                        await asyncio.sleep(2)
                        continue
                    return {"success": False, "error": str(e)}

            inst.dialog_active = False
            inst.dialog_detected = False
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000 + navigation_jitter())
            try:
                title = await inst.page.title()
                current_url = inst.page.url
                body_text = await inst.page.evaluate("() => document.body?.innerText?.slice(0, 5000) || ''")
                result = {
                    "success": True,
                    "data": {
                        "url": self.redactor.redact(agent_id, current_url),
                        "title": self.redactor.redact(agent_id, title),
                        "body": self.redactor.redact(agent_id, body_text),
                    },
                }
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _build_a11y_tree(self, inst: CamoufoxInstance, root=None):
        """Get accessibility tree, falling back to JS-based DOM walk.

        Playwright's ``page.accessibility.snapshot()`` may not exist in
        Camoufox (modified Firefox bundles its own Playwright build).
        On first ``AttributeError``, switches permanently to a JS-based
        tree builder that uses standard DOM APIs (``getAttribute``,
        ``getComputedStyle``) to produce the same ``{role, name, children}``
        tree structure the rest of the snapshot pipeline expects.
        """
        if not getattr(self, "_js_snapshot_mode", False):
            try:
                if root:
                    return await inst.page.accessibility.snapshot(root=root)
                return await inst.page.accessibility.snapshot()
            except AttributeError:
                logger.warning(
                    "page.accessibility not available — "
                    "switching to JS-based accessibility tree"
                )
                self._js_snapshot_mode = True
            except Exception:
                pass  # Other snapshot failures — fall through to JS

        # JS fallback: walk DOM to build equivalent tree
        try:
            if root:
                return await root.evaluate(_JS_A11Y_TREE)
            return await inst.page.evaluate(_JS_A11Y_TREE)
        except Exception as e:
            logger.debug("JS a11y tree fallback failed: %s", e)
            return None

    async def snapshot(self, agent_id: str) -> dict:
        """Get accessibility tree with element refs."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            return await self._snapshot_impl(inst, agent_id)

    async def _snapshot_impl(self, inst: CamoufoxInstance, agent_id: str) -> dict:
        """Snapshot implementation.  Caller must hold ``inst.lock``."""
        try:
            tree = await self._build_a11y_tree(inst)
            if not tree:
                return {"success": True, "data": {"snapshot": "(empty page)", "refs": {}}}

            lines: list[str] = []
            refs: dict[str, dict] = {}
            ref_counter = [0]
            # Counts occurrences of each (role, name) pair so we can
            # disambiguate duplicate elements (e.g. X's two composer nodes).
            occurrence_counts: dict[tuple, int] = {}

            _MAX_WALK_DEPTH = 50

            def _walk(node, depth=0):
                if depth > _MAX_WALK_DEPTH:
                    return
                role = node.get("role", "")
                name = node.get("name", "")
                if role in _ACTIONABLE_ROLES or role in _CONTEXT_ROLES:
                    if ref_counter[0] < _MAX_SNAPSHOT_ELEMENTS:
                        # Use ref ID stamped by JS walker when available
                        ref_id = node.get("refId") or f"e{ref_counter[0]}"
                        ref_counter[0] += 1

                        key = (role, name)
                        occ = occurrence_counts.get(key, 0)
                        occurrence_counts[key] = occ + 1

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
                        if occ > 0:
                            attrs.append(f"dup:{occ + 1}")
                        attr_str = f" [{', '.join(attrs)}]" if attrs else ""

                        # Structural context from nearest landmark ancestor
                        landmark = node.get("landmark", "")
                        ctx_str = f" ({landmark})" if landmark else ""

                        line = f"{'  ' * depth}- [{ref_id}] {role} \"{name}\"{attr_str}{ctx_str}"
                        lines.append(line)
                        refs[ref_id] = {
                            "role": role, "name": name, "index": occ,
                            "disabled": bool(node.get("disabled")),
                            "has_olref": bool(node.get("refId")),
                        }
                for child in node.get("children", []):
                    _walk(child, depth + 1)

            # When a modal dialog is open, scope to only dialog elements
            # so agents don't see/click elements behind the overlay
            # (e.g. X's sidebar "Post" button behind the compose modal).
            modal_els = await inst.page.query_selector_all(_MODAL_SELECTOR)
            visible_modals = []
            for el in modal_els:
                try:
                    if await el.is_visible():
                        visible_modals.append(el)
                except Exception:
                    pass

            # Deduplicate nested modals: if modal A contains modal B,
            # snapshot(root=A) already includes B's elements.
            if len(visible_modals) > 1:
                deduped = []
                for i, el in enumerate(visible_modals):
                    is_nested = False
                    for j, other in enumerate(visible_modals):
                        if i != j:
                            try:
                                if await other.evaluate(
                                    "(parent, child) => parent.contains(child)", el
                                ):
                                    is_nested = True
                                    break
                            except Exception:
                                pass
                    if not is_nested:
                        deduped.append(el)
                visible_modals = deduped if deduped else visible_modals

            if visible_modals:
                inst.dialog_detected = True
                inst.dialog_active = True
                lines.append("** Modal dialog is open — only dialog elements are shown **")
                for el in visible_modals:
                    subtree = await self._build_a11y_tree(inst, root=el)
                    if subtree:
                        _walk(subtree)
                actionable_refs = [
                    r for r in refs.values() if r["role"] in _ACTIONABLE_ROLES
                ]
                # Progressive retry: 300 ms then 500 ms — gives SPAs like X
                # enough time for modal animations and Lexical editor init.
                retry_waits = [0.3, 0.5]
                while not actionable_refs and retry_waits:
                    wait = retry_waits.pop(0)
                    logger.debug(
                        "Modal scoping produced 0 actionable refs — "
                        "retrying after %.0f ms", wait * 1000,
                    )
                    await asyncio.sleep(wait)
                    refs.clear()
                    lines.clear()
                    ref_counter[0] = 0
                    occurrence_counts.clear()
                    lines.append("** Modal dialog is open — only dialog elements are shown **")
                    for el in visible_modals:
                        try:
                            subtree = await self._build_a11y_tree(inst, root=el)
                            if subtree:
                                _walk(subtree)
                        except Exception:
                            pass
                    actionable_refs = [
                        r for r in refs.values() if r["role"] in _ACTIONABLE_ROLES
                    ]
                if not actionable_refs:
                    logger.warning(
                        "Modal detected but scoping produced 0 actionable "
                        "refs after retries — falling back to full tree "
                        "for %s", agent_id,
                    )
                    inst.dialog_active = False
                    lines.clear()
                    lines.append(
                        "** A modal dialog is open but its elements could "
                        "not be isolated — elements with a (dialog: ...) "
                        "or similar landmark annotation are in the modal; "
                        "others are behind the overlay **"
                    )
                    _walk(tree)
            else:
                inst.dialog_active = False
                inst.dialog_detected = False
                _walk(tree)

            inst.refs = refs
            snapshot_text = "\n".join(lines) if lines else "(no interactive elements)"
            snapshot_text = self.redactor.redact(agent_id, snapshot_text)
            return {"success": True, "data": {"snapshot": snapshot_text, "refs": refs}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _locator_from_ref(self, inst: CamoufoxInstance, ref: str):
        """Build a Playwright locator from a stored ref.

        Primary strategy: use the ``data-olref`` attribute stamped on the DOM
        element during the JS accessibility tree walk.  This is an exact,
        unambiguous pointer to the element the snapshot saw — immune to
        scope mismatches and substring-matching issues.

        Fallback (when the native Playwright accessibility API was used
        instead of the JS walker, or when the data attribute was removed
        by a framework re-render): semantic locator via role + exact name
        + nth(index), scoped to the modal when one is active.
        """
        info = inst.refs.get(ref)
        if not info:
            return None

        # Primary: direct locator via data-olref stamped during snapshot
        if info.get("has_olref"):
            return inst.page.locator(f'[data-olref="{ref}"]')

        # Fallback: semantic locator
        role = info["role"]
        name = info.get("name", "")
        idx = info.get("index", 0)
        if inst.dialog_active:
            base = inst.page.locator(_MODAL_SELECTOR)
        else:
            base = inst.page
        locator = base.get_by_role(role, name=name, exact=True) if name else base.get_by_role(role)
        return locator.nth(idx)

    async def click(
        self, agent_id: str, ref: str | None = None,
        selector: str | None = None, force: bool = False,
        snapshot_after: bool = False,
    ) -> dict:
        """Click element by ref or CSS selector.

        force=True bypasses Playwright's actionability checks (visibility,
        stability, not-covered, enabled). Use when the element is visually
        present in VNC but Playwright reports it as covered by an overlay.

        For button/link roles that were disabled in the last snapshot,
        force is applied automatically — SPA frameworks (X/Twitter, Gmail)
        commonly set aria-disabled="true" on buttons that are still clickable
        via JS handlers. Playwright blocks clicks on aria-disabled elements
        unless force=True, so we bypass the check for these roles.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                use_force = force
                if ref and ref in inst.refs:
                    ref_info = inst.refs[ref]
                    # Auto-force for disabled button/link roles — aria-disabled
                    # on SPA buttons doesn't mean the JS click handler won't fire.
                    # BUT: when a modal was detected and scoping failed
                    # (dialog_detected=True, dialog_active=False), disabled
                    # buttons are likely behind the overlay — don't force them.
                    modal_unscoped = (
                        inst.dialog_detected and not inst.dialog_active
                    )
                    if (not use_force
                            and ref_info.get("disabled")
                            and ref_info.get("role") in _ARIA_FORCE_ROLES
                            and not modal_unscoped):
                        use_force = True
                        logger.debug(
                            "Auto-force click on disabled %s ref=%s for '%s'",
                            ref_info["role"], ref, agent_id,
                        )
                    locator = self._locator_from_ref(inst, ref)
                    if locator:
                        await locator.click(timeout=_CLICK_TIMEOUT_MS, force=use_force)
                    else:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                elif selector:
                    await inst.page.click(selector, timeout=_CLICK_TIMEOUT_MS, force=force)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())
                result = {"success": True, "data": {"clicked": ref or selector}}
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def hover(
        self, agent_id: str, ref: str | None = None,
        selector: str | None = None,
    ) -> dict:
        """Move the mouse over an element without clicking.

        Useful for hover-triggered dropdowns, tooltip visibility, and navigation
        menus that only reveal sub-items on mouseover.  After hovering, call
        snapshot() to see the newly visible elements.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if ref and ref in inst.refs:
                    locator = self._locator_from_ref(inst, ref)
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    await locator.hover(timeout=_CLICK_TIMEOUT_MS)
                elif selector:
                    await inst.page.hover(selector, timeout=_CLICK_TIMEOUT_MS)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"hovered": ref or selector}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def type_text(self, agent_id: str, ref: str | None = None, selector: str | None = None,
                        text: str = "", clear: bool = True, is_credential: bool = False,
                        fast: bool = False, snapshot_after: bool = False) -> dict:
        """Type text into element. Credential values should be pre-resolved by agent.

        fast=True uses minimal inter-key delays (8 ms) — still fires real
        keyDown/keyUp events for framework compatibility, but skips
        human-variance timing and think pauses.  Suitable for search
        queries, URLs, and non-sensitive form fields.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                # Always click to focus, then optionally select-all to clear.
                # Never use fill() — it atomically sets the DOM value and bypasses
                # the keyboard event chain, so React/Vue apps (e.g. X's tweet
                # composer) don't see individual keystrokes and won't activate
                # submit buttons or update their controlled-component state.
                if ref and ref in inst.refs:
                    locator = self._locator_from_ref(inst, ref)
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    await locator.click(timeout=_CLICK_TIMEOUT_MS)
                elif selector:
                    await inst.page.click(selector, timeout=_CLICK_TIMEOUT_MS)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}

                # Settle after focus — SPA editors (Lexical, ProseMirror, Draft.js)
                # may expand or initialise event listeners on focus click.
                await asyncio.sleep(0.10 if fast else action_delay())

                if clear:
                    await inst.page.keyboard.press("Control+a")
                    await asyncio.sleep(0.05)

                if fast:
                    await self._type_fast(inst.page, text)
                else:
                    await self._type_with_variance(inst.page, text)

                # Settle after typing — framework state (React, Lexical, Vue)
                # batches DOM reconciliation asynchronously.
                await asyncio.sleep(0.10 if fast else action_delay())

                if is_credential:
                    self.redactor.track_resolved_value(agent_id, text)
                    if ref:
                        inst.credential_filled_refs.add(ref)

                result = {"success": True, "data": {"typed_into": ref or selector, "length": len(text)}}
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result
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
        """Type text character-by-character with human-like inter-key delays.

        Uses keyboard.press(char) for every printable character so Playwright
        sends real CDP keyDown/keyUp events.  The browser — not injected JS —
        generates the resulting beforeinput and input events, which carry
        isTrusted=true.  React/Lexical controlled contenteditable elements
        (e.g. X's tweet composer) only update their state — and enable submit
        buttons — when beforeinput.isTrusted is true.

        Why not execCommand('insertText')?  In Firefox (Camoufox's base) the
        event fired by execCommand has isTrusted=false, so Lexical ignores it:
        text appears in the DOM visually but the Post button stays disabled.

        Why not keyboard.type(char)?  Playwright's type() uses CDP
        Input.insertText which injects text without any key events — no keydown,
        no beforeinput.  Same problem.

        keyboard.press(char) → CDP Input.dispatchKeyEvent(keyDown + keyUp) →
        browser generates trusted beforeinput → Lexical/React state updates →
        Post button becomes enabled.

        Fallback: if keyboard.press() raises (character outside Playwright's key
        map, e.g. accented letters, emoji), use keyboard.type() so the character
        at least appears.

        Think-pauses are weighted to word/clause boundaries: 8 % probability
        before the first character of each new word, 1.5 % mid-word.
        """
        prev_char = ""
        for char in text:
            # Word-boundary characters signal a clause break — higher pause
            # probability for the character starting the next word/clause.
            pause_prob = 0.08 if prev_char in _WORD_BOUNDARY_CHARS else 0.015
            if random.random() < pause_prob:
                await asyncio.sleep(think_pause())

            if char == "\n":
                await page.keyboard.press("Enter")
            elif char == "\t":
                await page.keyboard.press("Tab")
            else:
                # Real key events → isTrusted=true beforeinput → framework
                # state updates (Post button lights up on X, etc.)
                try:
                    await page.keyboard.press(char)
                except Exception:
                    # Outside Playwright's key map — fall back, text appears
                    # but framework state may not update.
                    await page.keyboard.type(char)
            await asyncio.sleep(keystroke_delay(char))
            prev_char = char

    async def _type_fast(self, page, text: str) -> None:
        """Type text with minimal delay — still fires real key events.

        Uses keyboard.press(char) for isTrusted=true events so React/Lexical
        state updates work, but with a fixed 8 ms inter-key delay (no
        variance, no think pauses).  Suitable for search queries, URLs,
        and non-sensitive form fields where human-realistic timing is
        unnecessary.
        """
        for char in text:
            if char == "\n":
                await page.keyboard.press("Enter")
            elif char == "\t":
                await page.keyboard.press("Tab")
            else:
                try:
                    await page.keyboard.press(char)
                except Exception:
                    await page.keyboard.type(char)
            await asyncio.sleep(0.008)

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
                        "(d) => window.scrollBy({top: d, behavior: 'smooth'})", delta
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

    async def wait_for_element(
        self, agent_id: str, selector: str, state: str = "visible",
        timeout_ms: int = 10000,
    ) -> dict:
        """Wait for a CSS selector to reach the given state.

        state: "visible" (default), "attached", "hidden", or "detached".
        Useful before clicking elements on SPAs that animate in after load.
        """
        _valid_states = frozenset({"visible", "attached", "hidden", "detached"})
        if state not in _valid_states:
            return {"success": False, "error": f"Invalid state: {state!r}. Use one of: {sorted(_valid_states)}"}
        timeout_ms = max(0, min(timeout_ms, _WAIT_FOR_TIMEOUT_MS))

        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                await inst.page.wait_for_selector(selector, state=state, timeout=timeout_ms)
                return {"success": True, "data": {"selector": selector, "state": state}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def solve_captcha(self, agent_id: str) -> dict:
        """Detect CAPTCHAs on the current page."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                captcha_selectors = [
                    'iframe[src*="recaptcha"]',
                    'iframe[src*="hcaptcha"]',
                    'iframe[src*="challenges.cloudflare.com"]',
                    'iframe[src*="captcha"]',
                    '[class*="cf-turnstile"]',
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

    async def press_key(self, agent_id: str, key: str) -> dict:
        """Press a keyboard key or combination (e.g. 'Enter', 'Escape', 'Control+a').

        Dispatches a real keyDown/keyUp event pair via Playwright, producing
        trusted keyboard events.  Useful for dismissing modals (Escape),
        submitting forms (Enter), tabbing between fields (Tab), or keyboard
        navigation (ArrowUp/ArrowDown).
        """
        if not key or len(key) > 50:
            return {"success": False, "error": "Invalid key"}
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                await inst.page.keyboard.press(key)
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"pressed": key}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def go_back(self, agent_id: str) -> dict:
        """Navigate back in browser history."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                response = await inst.page.go_back(timeout=10000)
                inst.dialog_active = False  # New page — stale modal state
                inst.dialog_detected = False
                await asyncio.sleep(action_delay())
                title = await inst.page.title()
                url = self.redactor.redact(agent_id, inst.page.url)
                title = self.redactor.redact(agent_id, title)
                navigated = response is not None
                return {"success": True, "data": {"url": url, "title": title, "navigated": navigated}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def go_forward(self, agent_id: str) -> dict:
        """Navigate forward in browser history."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                response = await inst.page.go_forward(timeout=10000)
                inst.dialog_active = False  # New page — stale modal state
                inst.dialog_detected = False
                await asyncio.sleep(action_delay())
                title = await inst.page.title()
                url = self.redactor.redact(agent_id, inst.page.url)
                title = self.redactor.redact(agent_id, title)
                navigated = response is not None
                return {"success": True, "data": {"url": url, "title": title, "navigated": navigated}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def switch_tab(self, agent_id: str, tab_index: int = -1) -> dict:
        """List open tabs and optionally switch to one.

        tab_index=-1 (default): list all tabs without switching.
        tab_index>=0: switch to that tab index and clear stale refs.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                pages = inst.context.pages
                if not pages:
                    return {"success": False, "error": "No tabs open"}

                # Build tab list
                tabs = []
                active_index = 0
                for i, page in enumerate(pages):
                    is_active = page == inst.page
                    if is_active:
                        active_index = i
                    try:
                        title = await page.title()
                    except Exception:
                        title = "(loading)"
                    tabs.append({
                        "index": i,
                        "url": self.redactor.redact(agent_id, page.url),
                        "title": self.redactor.redact(agent_id, title),
                        "active": is_active,
                    })

                # Switch if requested
                if tab_index >= 0:
                    if tab_index >= len(pages):
                        return {
                            "success": False,
                            "error": f"Tab {tab_index} out of range (0-{len(pages) - 1})",
                        }
                    inst.page = pages[tab_index]
                    await inst.page.bring_to_front()
                    inst.refs = {}  # Stale refs from previous tab's snapshot
                    inst.dialog_active = False  # New tab may not have a dialog
                    inst.dialog_detected = False
                    active_index = tab_index
                    for t in tabs:
                        t["active"] = t["index"] == tab_index

                return {
                    "success": True,
                    "data": {"tabs": tabs, "active_tab": active_index},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
