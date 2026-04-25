"""Core browser manager — per-agent Camoufox instance lifecycle.

Manages lazy-started Camoufox browser instances, one per agent.
Each agent gets its own persistent profile, BrowserForge fingerprint,
and browser context on a shared Xvnc display.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import mimetypes
import os
import random
import re
import subprocess
import time
import uuid
import weakref
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

from src.browser.captcha import get_solver
from src.browser.profile_schema import migrate_profile
from src.browser.redaction import CredentialRedactor
from src.browser.ref_handle import RefHandle, RefStale
from src.browser.stealth import build_launch_options, pick_referer, validate_referer
from src.browser.timing import (
    action_delay,
    click_dwell,
    keystroke_delay,
    navigation_jitter,
    pre_click_settle,
    scroll_increment,
    scroll_pause,
    scroll_ramp,
    think_pause,
    x11_settle_delay,
    x11_step_delay,
)
from src.shared.types import AGENT_ID_RE_PATTERN
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


# §7.2 v2 format — depth indent is capped so a 50-deep DOM doesn't
# explode into 100-character indents. Anything past this depth shares
# the cap-line indent (still distinguishable as "deep" but not bytewise
# punishing).
_V2_MAX_INDENT_DEPTH = 4
_V2_NO_LANDMARK_KEY = ""


def _format_snapshot_v2(
    lines: list[str],
    entries: list[tuple[str, str, str, str, str, int]],
) -> str:
    """Render the snapshot in §7.2 ``v2`` format.

    Group entries by landmark and emit each group under a section
    header (``# nav: Top``) instead of suffixing every element with
    ``(navigation: Top)``. Indent depth is capped at
    :data:`_V2_MAX_INDENT_DEPTH`.

    Modal-mode preamble lines (those starting with ``**`` in the v1
    output) are passed through verbatim ahead of the section blocks
    so the agent still sees the modal context.

    Args:
        lines: the v1 line list — only used for ``**`` preamble lines.
        entries: per-element tuples
            ``(ref_id, role, name, attr_str, landmark, depth)``.

    Returns the rendered string. Always begins with the
    ``# snapshot-v2`` version marker so parsers can detect the format
    without out-of-band signaling.
    """
    if not entries:
        # Empty result. Still emit the marker so a parser using the
        # first line for routing decisions doesn't trip. Also pass
        # through any modal-banner preamble that v1 produced.
        preamble = [ln for ln in lines if ln.startswith("**")]
        if preamble:
            return "# snapshot-v2\n" + "\n".join(preamble)
        return "# snapshot-v2\n(no interactive elements)"

    # Modal-banner preamble (lines starting ``**``) precedes the
    # element output. Keeps the modal-scoped warning visible.
    preamble = [ln for ln in lines if ln.startswith("**")]

    # Preserve insertion order — the dict-by-design key order matches
    # the order entries were emitted, which is doc order.
    groups: dict[str, list[tuple[str, str, str, str, int]]] = {}
    for ref_id, role, name, attr_str, landmark, depth in entries:
        key = landmark or _V2_NO_LANDMARK_KEY
        groups.setdefault(key, []).append(
            (ref_id, role, name, attr_str, depth),
        )

    out: list[str] = ["# snapshot-v2"]
    out.extend(preamble)
    for landmark_key, group_entries in groups.items():
        if landmark_key == _V2_NO_LANDMARK_KEY:
            out.append("# (no landmark)")
        else:
            # Sanitize newlines in landmark keys: a malicious DOM
            # node with ``aria-label="x\n# fake-section: pwn"`` would
            # otherwise inject a phantom section header into the
            # parsed output (operator scripts reading the v2 format
            # split on '#' prefixes). Replace with single spaces so
            # the structural marker stays one-line.
            out.append(f"# {_v2_strip_newlines(landmark_key)}")
        for ref_id, role, name, attr_str, depth in group_entries:
            indent_depth = min(depth, _V2_MAX_INDENT_DEPTH)
            indent = "  " * indent_depth
            # Same sanitization on per-element name + attr_str so an
            # accessible-name with embedded ``\n# fake`` can't escape
            # to a fake section header.
            safe_name = _v2_strip_newlines(name)
            safe_attr = _v2_strip_newlines(attr_str)
            out.append(f"{indent}- [{ref_id}] {role} \"{safe_name}\"{safe_attr}")

    return "\n".join(out)


def _v2_strip_newlines(s: str) -> str:
    """Collapse ``\\n``/``\\r`` to single spaces.

    v2 promotes ``# ``-prefixed lines to structural meaning. A DOM
    node with newlines in its accessible name or landmark would
    otherwise inject phantom section headers into the parsed
    snapshot. Cheap belt-and-braces over the JS-walker side which
    only ``.trim()``s whitespace endpoints.
    """
    if not s:
        return s
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

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
_AGENT_ID_RE = re.compile(AGENT_ID_RE_PATTERN)
_VALID_WAIT_UNTIL = frozenset({"domcontentloaded", "load", "networkidle", "commit"})
# Characters that mark a natural word/clause boundary in typed text.
# After one of these, the next character gets a higher think-pause probability
# to model the hesitation a human feels when starting the next word or sentence.
_WORD_BOUNDARY_CHARS = frozenset(" ,.:;!?\n\t")
# Adjacent keys on QWERTY layout for natural typo injection.
# Includes same-row neighbors and diagonal keys above/below.
_TYPO_NEIGHBORS: dict[str, str] = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etdf', 't': 'ryfg',
    'y': 'tugh', 'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol',
    'a': 'qwsz', 's': 'weadxz', 'd': 'ersfxc', 'f': 'rtdgcv',
    'g': 'tyfhvb', 'h': 'yugjbn', 'j': 'uihknm', 'k': 'iojlm',
    'l': 'opk',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}
# Roles where aria-disabled="true" should NOT block click attempts.
# SPA frameworks (X/Twitter, Gmail) keep aria-disabled on buttons/links while
# handling clicks via JS — the visual state and handler are the source of truth,
# not the ARIA attribute.  Intentionally narrow: menuitem, switch, option are
# excluded because force-clicking genuinely disabled items in those roles causes
# unwanted side-effects (selecting unavailable options, toggling locked switches).
_ARIA_FORCE_ROLES = frozenset({"button", "link"})
# Button names (lowercased) that indicate a modal close/dismiss action.
# When clicking these inside a modal doesn't dismiss it, we fall back to
# pressing Escape — Camoufox's patched Firefox has known issues where
# pointer events on modal close buttons silently fail in some SPAs
# (X/Twitter compose modal, etc.).
_MODAL_CLOSE_NAMES = frozenset({"close", "×", "✕", "✖"})
# Playwright key names → xdotool key names. Playwright follows the KeyboardEvent.key
# spec; xdotool uses X11 keysym names. Only keys that differ need mapping.
_PLAYWRIGHT_TO_XDOTOOL = {
    "Enter": "Return", "Backspace": "BackSpace", "Delete": "Delete",
    "Space": "space", "ArrowUp": "Up", "ArrowDown": "Down",
    "ArrowLeft": "Left", "ArrowRight": "Right", "PageUp": "Prior",
    "PageDown": "Next", "Control": "ctrl", "Shift": "shift",
    "Alt": "alt", "Meta": "super",
}
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
    const start = rootEl || document.body || document.documentElement;
    const tree = walk(start, 0, null);
    if (rootEl) return tree || { role: 'none', name: '', children: [] };
    if (!tree) return { role: 'WebArea', name: document.title || '', children: [] };
    if (tree.role === 'none')
        return { role: 'WebArea', name: document.title || '', children: tree.children || [] };
    return { role: 'WebArea', name: document.title || '', children: [tree] };
}"""


def _short_ua(ua: str) -> str:
    """Compact a UA string for log output — keep the tail Firefox-version
    bit, drop the OS/locale boilerplate readers don't need at INFO level."""
    if not ua:
        return ""
    # Most useful bit is "Firefox/138.0" at the end; everything before is
    # noise for debugging fingerprint regressions.
    if "Firefox/" in ua:
        return "Firefox/" + ua.split("Firefox/", 1)[1]
    return ua[:80]


def _js_string(value: str) -> str:
    """Escape a Python string for safe interpolation into a JS literal.

    Used by the ``navigator.connection`` init-script to inject the
    per-agent ``effectiveType`` value. The values are drawn from a
    fixed pool (``"4g"`` etc.) so injection is bounded today, but
    a defensive escape costs nothing and prevents future agent-id-
    derived values from breaking out.
    """
    return ("'" + value
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n") + "'")


# §6.6 navigator.connection fallback. Defines a getter on
# ``Navigator.prototype`` that returns a frozen object matching the
# NetworkInformation API surface real Chromium-shaped browsers expose.
# ``configurable: true`` so a future Camoufox upgrade that adds native
# support can override this. Runs before any page script via
# ``BrowserContext.add_init_script``.
_NAV_CONNECTION_INIT_SCRIPT = """
(() => {{
  try {{
    if (typeof navigator !== 'undefined' && navigator.connection !== undefined) {{
      return;  // Camoufox / Firefox already exposes the API
    }}
    const fake = Object.freeze({{
      effectiveType: {effective},
      downlink: {downlink},
      rtt: {rtt},
      saveData: {save_data},
      type: 'wifi',
      addEventListener: () => {{}},
      removeEventListener: () => {{}},
      dispatchEvent: () => false,
      onchange: null,
    }});
    Object.defineProperty(Navigator.prototype, 'connection', {{
      get: () => fake,
      configurable: true,
      enumerable: true,
    }});
  }} catch (_e) {{
    // Defensive: any failure here is operator-debuggable via the
    // §6.3 navigator self-test, which will flag the missing API.
  }}
}})();
"""


def _is_empty_payload(payload: dict) -> bool:
    """True when a drain produced no activity *in this interval*.

    Used by :meth:`BrowserManager._emit_metrics` to filter out idle-
    agent payloads so the history buffer doesn't flood with no-ops.
    Only per-minute counters count here — the rolling click window
    persists across drains and would permanently bypass the filter if
    included (any agent that ever clicked would be "non-idle" forever).

    Payloads with an explicit ``kind`` (e.g. §6.3 ``nav_probe``) are
    one-shot events, not drain samples — they are never "empty" even
    when the per-minute counter fields are absent.
    """
    if payload.get("kind"):
        return False
    return not any((
        payload.get("click_success"),
        payload.get("click_fail"),
        payload.get("nav_timeout"),
        payload.get("snapshot_count"),
    ))


def _encode_screenshot(
    png_bytes: bytes,
    fmt: str,
    quality: int,
    scale: float,
    *,
    agent_id: str = "",
) -> tuple[bytes, str]:
    """Encode a Playwright PNG to WebP / PNG with optional downscale.

    Returns ``(encoded_bytes, actual_format)``. ``actual_format`` may be
    ``"png"`` even when ``fmt="webp"`` was requested — Pillow may be
    absent in the dev path or fail on a corrupt frame; PNG fallback
    keeps the agent unblocked rather than returning an error.

    The function is intentionally synchronous and pure — easy to unit
    test and reason about. Pillow does its own threading internally;
    callers should either be on a worker thread or accept that an
    ~1080p WebP encode runs in ~10–20 ms.
    """
    # Fast path: caller asked for PNG and no scale change → pass through.
    if fmt == "png" and abs(scale - 1.0) < 1e-3:
        return png_bytes, "png"

    try:
        from io import BytesIO

        from PIL import Image
    except ImportError:
        # Pillow missing — log once per encode attempt at debug only;
        # this is expected on the agent-side dev path where Pillow isn't
        # bundled. Caller still gets a usable PNG. Narrowed to
        # ImportError specifically so non-import failures (e.g. partially
        # broken install raising OSError at module init) surface as bugs
        # rather than silently downgrading.
        logger.debug(
            "Pillow not installed; falling back to PNG (agent=%s)", agent_id,
        )
        return png_bytes, "png"

    try:
        img = Image.open(BytesIO(png_bytes))
        # Downscale via Lanczos when requested. Avoids the no-op resize
        # cost when scale is effectively 1.0.
        if abs(scale - 1.0) >= 1e-3 and scale > 0:
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)

        out = BytesIO()
        if fmt == "webp":
            # Convert to RGB first — WebP doesn't accept palette or
            # certain RGBA modes from Pillow versions <10.4 cleanly.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(out, format="WEBP", quality=quality, method=4)
            return out.getvalue(), "webp"
        # PNG re-encode (only reached when scale != 1.0 above).
        img.save(out, format="PNG", optimize=True)
        return out.getvalue(), "png"
    except Exception as e:
        logger.warning(
            "Screenshot %s encode failed (%s); falling back to original PNG",
            fmt, e,
        )
        return png_bytes, "png"


def _extract_text_from_a11y(tree: dict | None, max_chars: int = 5000) -> str:
    """Extract readable text from an accessibility snapshot tree.

    Walks leaf nodes to avoid duplicating text that parent containers
    aggregate from their children.  Used by ``navigate()`` as a
    stealth-safe alternative to ``page.evaluate("document.body.innerText")``
    — the a11y API reads from Firefox's internal accessibility service
    with zero JavaScript execution in the page context.
    """
    if not tree:
        return ""
    parts: list[str] = []
    total = 0

    def _collect(node: dict) -> bool:
        nonlocal total
        if not isinstance(node, dict) or total >= max_chars:
            return total < max_chars
        children = node.get("children")
        if children:
            for child in children:
                if not _collect(child):
                    return False
        else:
            name = (node.get("name") or "").strip()
            if name:
                parts.append(name)
                total += len(name) + 1
        return total < max_chars

    _collect(tree)
    return " ".join(parts)[:max_chars]


class CamoufoxInstance:
    """Wrapper around a single Camoufox browser for one agent."""

    def __init__(self, agent_id: str, browser, context, page):
        self.agent_id = agent_id
        self.browser = browser
        self.context = context
        self.page = page
        self.last_activity = time.time()
        # Rich ref identity (§4.2): ref_id → RefHandle carrying page_id,
        # frame_id, shadow_path, scope_root, role/name/occurrence, and
        # (populated later by diff-mode) element_key.
        self.refs: dict[str, RefHandle] = {}
        self.dialog_active: bool = False  # True when snapshot scoped to a modal dialog
        self.dialog_detected: bool = False  # True when a modal was found (even if scoping failed)
        self.lock = asyncio.Lock()  # serialize page operations per instance
        self.x11_wid: int | None = None  # X11 window ID for targeted focus
        self._js_snapshot_mode: bool = False  # True after page.accessibility permanently fails
        self._user_control: bool = False  # True when user has VNC control
        # Per-Page stable UUID maps. Page objects survive navigation within a
        # tab; UUIDs are stable for the life of the Page. Refs carry a
        # ``page_id`` so resolution can detect a closed tab as stale (§4.2).
        self._page_id_counter: int = 0
        self.page_ids: dict = {}              # id(Page) -> str
        # WeakValueDictionary so closed Pages (GC'd by Playwright on tab
        # close) drop out of the reverse lookup automatically. Plain dict
        # here would pin every Page ever opened for the lifetime of the
        # CamoufoxInstance — a slow leak on agents that churn tabs.
        self.page_ids_inv: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        # Register the initial page so refs captured on it resolve correctly.
        self._register_page(page)

        # Per-agent metrics (§4.6). Per-minute counters reset at each emit
        # cycle so dashboards see rate-of-change, not monotonic totals.
        # Snapshot byte sizes accumulate as a list (p50/p95 at emit time);
        # size samples are small (~200/min/agent at most) and simpler wins.
        self.m_click_success: int = 0
        self.m_click_fail: int = 0
        self.m_nav_timeout: int = 0
        self.m_snapshot_bytes: list[int] = []
        # §5.2 rolling click-success-rate: deque of booleans for the last
        # 100 click outcomes. Unlike the per-minute counters above, this
        # window is NOT reset on drain — it's a user-facing live gauge
        # exposed via /browser/{agent}/status and in the per-minute metric
        # payload, giving operators an "is the browser currently healthy"
        # signal that doesn't flap on low-traffic minutes.
        self.click_window: deque[bool] = deque(maxlen=100)
        # §5.3 behavioral entropy recorder (dev-only). Always constructed;
        # every record_* call short-circuits when the feature flag is
        # off so production pays no cost.
        from src.browser.recorder import BehaviorRecorder
        self.recorder = BehaviorRecorder(agent_id)
        # §6.5 rolling-5 history of recently-used referers. The picker
        # uses this to avoid immediate repeats so a fleet at scale
        # doesn't all show the same Google referer back-to-back. Resets
        # on browser restart, matching a real user-session boundary.
        self.recent_referers: deque[str] = deque(maxlen=5)
        # §6.5 first-real-navigate gate. With ``persistent_context=True``
        # the browser resumes whatever page was open last session — the
        # picker would otherwise treat that stale URL as a "previous page"
        # and fabricate a same-origin referer for the next nav, even though
        # there's been no recent navigation in this session. The flag flips
        # to True after the first navigate completes; subsequent navs may
        # use ``inst.page.url`` as the previous-URL hint.
        self.had_real_navigate: bool = False
        # §6.3 navigator self-test result. ``None`` until the post-launch
        # probe runs. Populated dict (see ``BrowserManager._run_navigator_probe``)
        # exposes ``ok`` + ``mismatches`` + raw signal values for dashboard /
        # status endpoint consumers.
        self.probe_result: dict | None = None

    def _register_page(self, page) -> str:
        """Assign a stable UUID to a Page if not already registered.

        Idempotent — re-registering the same Page returns its existing UUID.
        Called on CamoufoxInstance creation (for the initial page) and will
        be called again by ``browser_open_tab`` (§8.6) for new tabs.
        """
        existing = self.page_ids.get(id(page))
        if existing is not None:
            return existing
        self._page_id_counter += 1
        new_id = f"p{self._page_id_counter}-{uuid.uuid4().hex[:8]}"
        self.page_ids[id(page)] = new_id
        self.page_ids_inv[new_id] = page
        return new_id

    def _page_id_for(self, page) -> str:
        """Return the stable UUID for ``page`` (registering if new)."""
        return self._register_page(page)

    def _resolve_page_id(self, page_id: str):
        """Return the Page for ``page_id`` or raise :class:`RefStale`.

        A ref whose ``page_id`` is unknown to this instance points to a
        closed tab (or never existed). Distinct from "element not found"
        — the caller should prompt the agent to re-snapshot.
        """
        page = self.page_ids_inv.get(page_id)
        if page is None:
            raise RefStale("tab closed or unknown page_id", ref=None)
        return page

    def seed_refs_legacy(self, legacy: "dict[str, dict]") -> None:
        """Test helper: build ``RefHandle`` entries from v1-shape dicts.

        Uses the instance's current page as the target ``page_id`` so
        ``_locator_from_ref`` resolves correctly without the test having to
        know the generated UUID. If ``self.dialog_active`` is True, seeds
        refs with ``scope_root`` pointing at the modal selector — matching
        what a live snapshot emits during modal scoping. Not for production
        use — agent skills don't construct RefHandles, snapshots do.
        """
        page_id = self._page_id_for(self.page)
        scope = _MODAL_SELECTOR if self.dialog_active else None
        self.refs = {
            rid: RefHandle.light_dom(
                page_id=page_id,
                scope_root=scope,
                role=entry.get("role", ""),
                name=entry.get("name", ""),
                occurrence=entry.get("index", 0),
                disabled=bool(entry.get("disabled", False)),
            )
            for rid, entry in legacy.items()
        }

    def touch(self):
        self.last_activity = time.time()

    def rolling_click_success_rate(self) -> float | None:
        """Fraction of the last 100 clicks that succeeded, or ``None``.

        Returns ``None`` when no clicks have been recorded yet — callers
        should render this as "—" rather than "0%", which would misleadingly
        imply catastrophic failure on a freshly-booted agent.
        """
        if not self.click_window:
            return None
        successes = sum(1 for ok in self.click_window if ok)
        return successes / len(self.click_window)

    def drain_metrics(self) -> dict:
        """Snapshot counters and reset the per-minute ones to zero.

        Called by :meth:`BrowserManager._emit_metrics` every minute. The
        rolling 100-click window is NOT reset — it continues to track the
        most recent 100 clicks across emit cycles. Emits the rolling rate
        alongside per-minute counters so the dashboard can show both
        "activity in the last minute" and "health over recent work."
        """
        snaps = self.m_snapshot_bytes
        snap_count = len(snaps)
        if snap_count:
            sorted_snaps = sorted(snaps)
            p50 = sorted_snaps[snap_count // 2]
            p95_idx = max(0, min(snap_count - 1, int(snap_count * 0.95)))
            p95 = sorted_snaps[p95_idx]
        else:
            p50 = 0
            p95 = 0
        out = {
            "agent_id": self.agent_id,
            "click_success": self.m_click_success,
            "click_fail": self.m_click_fail,
            "nav_timeout": self.m_nav_timeout,
            "snapshot_count": snap_count,
            "snapshot_bytes_p50": p50,
            "snapshot_bytes_p95": p95,
            "click_window_size": len(self.click_window),
            "click_success_rate_100": self.rolling_click_success_rate(),
        }
        # Reset the per-minute counters; the rolling window persists.
        self.m_click_success = 0
        self.m_click_fail = 0
        self.m_nav_timeout = 0
        self.m_snapshot_bytes = []
        return out


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
        *,
        metrics_sink=None,
    ):
        """Per-agent Camoufox lifecycle manager.

        Args:
            metrics_sink: optional callable ``(payload: dict) -> None`` that
                receives per-agent aggregate metrics once per minute. When
                ``None``, metrics counters still increment but nothing is
                emitted — tests can pass a list's ``append`` method to
                capture payloads; production wires this to the dashboard
                :class:`EventBus`.
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.idle_timeout = idle_timeout_minutes * 60
        self._instances: dict[str, CamoufoxInstance] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._playwright = None
        self._user_focused_agent: str | None = None  # set by explicit focus() call
        self.redactor = CredentialRedactor()
        self._proxy_configs: dict[str, dict | None] = {}
        self.boot_id: str = str(uuid.uuid4())
        self._captcha_solver = get_solver()
        self._metrics_sink = metrics_sink
        # Per-agent rolling buffer of recent emit payloads (§5.1/§5.2) used by
        # the mesh's periodic poll to forward metrics to the dashboard
        # EventBus. Kept as a monotonic sequence so repeated polls can
        # request only what they haven't seen. Bounded so a long-lived
        # service with many agents doesn't grow without bound.
        self._metrics_history: deque[dict] = deque(maxlen=1024)
        self._metrics_seq: int = 0

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
            # Emit per-minute metrics AFTER idle-cleanup — instances that
            # just got stopped had their counters drained in ``_stop_instance``.
            try:
                await self._emit_metrics()
            except Exception as e:
                logger.warning("Metrics emit error: %s", e)

    async def _emit_metrics(self):
        """Drain per-agent counters and fan them out.

        Runs on the same 60s tick as idle cleanup (per §2.7: per-call
        events are forbidden; aggregates only). Writes each payload to the
        in-memory history buffer so the mesh can poll it, then forwards to
        the optional ``metrics_sink`` callback (for tests / in-process
        wiring). Counters always reset, whether or not a sink is attached,
        so a long-idle service doesn't grow memory.

        Per-instance drain failures are caught — a single agent with a
        corrupt counter must not abort the emit loop and starve the other
        agents' data.
        """
        if not self._instances:
            return
        now = time.time()
        # Take a consistent view of the instance list; ``drain_metrics()``
        # is a fully synchronous read-then-reset so no ``await`` boundary
        # opens between the counter read and its zeroing. Under asyncio's
        # single-threaded event loop, an in-flight hot-path task that
        # holds ``inst.lock`` cannot run between those two statements —
        # its coroutine is suspended elsewhere. This is WHY we don't
        # need ``inst.lock`` here. If ``drain_metrics`` ever grows an
        # ``await``, that invariant breaks and this must take the lock
        # or swap counter objects atomically.
        for inst in list(self._instances.values()):
            try:
                payload = inst.drain_metrics()
            except Exception as e:
                logger.warning(
                    "drain_metrics failed for '%s': %s", inst.agent_id, e,
                )
                continue
            # Skip payloads with zero activity AND an empty rolling window —
            # idle agents should not flood the history buffer (meshes that
            # were briefly offline will otherwise replay dozens of no-op
            # entries on reconnect, evicting live signal from the dashboard
            # ring buffer).
            if _is_empty_payload(payload):
                continue
            self._metrics_seq += 1
            payload["seq"] = self._metrics_seq
            payload["ts"] = now
            self._metrics_history.append(payload)
            if self._metrics_sink is None:
                continue
            try:
                self._metrics_sink(payload)
            except Exception as e:
                logger.warning(
                    "Metrics sink raised for '%s': %s", inst.agent_id, e,
                )

    def get_recent_metrics(self, since_seq: int = 0) -> dict:
        """Return buffered metric payloads with ``seq > since_seq``.

        Shape: ``{"current_seq": N, "metrics": [...]}``. The poller passes
        back ``current_seq`` as ``since_seq`` on the next call to get only
        new payloads. On service restart the seq counter resets to 0 — the
        poller detects this via the ``boot_id`` on ``/browser/status`` and
        resets its high-water mark.
        """
        metrics = [p for p in self._metrics_history if p.get("seq", 0) > since_seq]
        return {
            "current_seq": self._metrics_seq,
            "boot_id": self.boot_id,
            "metrics": metrics,
        }

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

    async def refocus_active(self) -> None:
        """Re-assert X11 focus on the user's viewed browser window.

        Called periodically by the VNC keepalive.  When a modal, popup, or
        internal Firefox dialog steals X11 focus, subsequent VNC mouse clicks
        go to the wrong window and appear to do nothing.

        Prefers the agent the user explicitly focused (via the dashboard
        Browser button) over the most recently active instance.  This
        prevents background agent browser operations from stealing the
        VNC display away from what the user is watching.
        """
        async with self._lock:
            if not self._instances:
                return
            # Prefer user's explicit focus over MRU
            if (
                self._user_focused_agent
                and self._user_focused_agent in self._instances
            ):
                target = self._instances[self._user_focused_agent]
            else:
                target = max(self._instances.values(), key=lambda i: i.last_activity)
            wid = target.x11_wid
        if not wid:
            # No WID known — skip xdotool entirely.  The fallback
            # `search --class firefox` matches ALL Firefox windows and
            # raises whichever it finds first, breaking multi-agent
            # browser switching.
            return
        try:
            wid_s = str(wid)
            cmd = ["xdotool", "windowmap", "--sync", wid_s,
                   "windowraise", wid_s, "windowfocus", wid_s]
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, timeout=3),
            )
        except Exception:
            pass

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

    async def _get_firefox_wids(self) -> set[int]:
        """Return the set of current X11 window IDs for Firefox windows."""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "search", "--class", "firefox"],
                    capture_output=True, text=True, timeout=2,
                ),
            )
            if result.returncode == 0 and result.stdout.strip():
                return {int(w) for w in result.stdout.strip().split("\n") if w.strip()}
        except Exception:
            pass
        return set()

    async def _discover_new_wid(self, before: set[int]) -> int | None:
        """Poll for a new Firefox X11 window that wasn't in *before*.

        Takes the highest WID when multiple new windows appear, since X11
        assigns incrementing IDs — the highest is the most recently created
        (the main browser window, not a transient popup from startup).
        """
        for _ in range(30):  # up to ~6s
            current = await self._get_firefox_wids()
            new = current - before
            if new:
                return max(new)
            await asyncio.sleep(0.2)
        return None

    async def _start_browser(self, agent_id: str) -> CamoufoxInstance:
        """Launch a Camoufox browser for an agent."""
        from camoufox.async_api import AsyncNewBrowser

        pw = await self._ensure_playwright()

        profile_dir = str(self.profiles_dir / agent_id)
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        # Bring the profile up to the current schema version BEFORE Camoufox
        # opens it (§4.4). Post-launch migration would race Camoufox's own
        # writes into the directory. Idempotent on already-current profiles;
        # on failure restores the pre-migration backup and re-raises so we
        # never launch against a half-migrated profile.
        #
        # Two error shapes propagate:
        #   * ``ProfileMigrationBusy`` — a peer process holds the lock and
        #     the on-disk version is below target. Retryable; agent can
        #     call again in a few seconds.
        #   * Any other exception — migration hit a real failure and the
        #     backup has already been restored. Not safely retryable until
        #     a human investigates.
        from src.browser.profile_schema import (
            ProfileMigrationBusy,
            sync_adblock_extension,
        )
        try:
            migrate_profile(Path(profile_dir))
        except ProfileMigrationBusy:
            logger.warning(
                "Profile migration for '%s' busy (peer holds lock); "
                "refusing to launch until they finish", agent_id,
            )
            raise
        except Exception:
            logger.exception(
                "Profile migration failed for '%s' — aborting browser start",
                agent_id,
            )
            raise

        # Phase 4 §7.1 — make sure the ad-blocker XPI matches the operator's
        # current ``BROWSER_ENABLE_ADBLOCK`` setting. This is intentionally
        # separate from the schema migration: the migration runs once per
        # version bump, but flag toggles + image rebuilds with newer XPIs
        # need to take effect on every launch. Best-effort — never blocks
        # the browser from starting.
        sync_adblock_extension(Path(profile_dir))

        proxy_config = self.get_proxy_config(agent_id)
        if proxy_config is not None:
            if proxy_config.get("url"):
                # Per-agent proxy configured — use it
                proxy_arg: dict = {"server": proxy_config["url"]}
                if proxy_config.get("username"):
                    proxy_arg["username"] = proxy_config["username"]
                if proxy_config.get("password"):
                    proxy_arg["password"] = proxy_config["password"]
                options = build_launch_options(agent_id, profile_dir, proxy=proxy_arg)
            else:
                # Explicitly no proxy (direct mode or inherit with no system proxy)
                options = build_launch_options(agent_id, profile_dir, proxy=None)
        else:
            # No per-agent config pushed yet — start without proxy.
            # The mesh will push the correct config shortly after startup
            # which triggers a reset, relaunching with the right proxy.
            logger.warning("No proxy config pushed for '%s' yet, starting without proxy", agent_id)
            options = build_launch_options(agent_id, profile_dir, proxy=None)

        # Log which proxy is being used for debuggability
        _proxy_opt = options.get("proxy")
        if _proxy_opt:
            _p_server = _proxy_opt.get("server", "?")
            logger.info("Starting Camoufox for '%s' (profile=%s, proxy=%s)", agent_id, profile_dir, _p_server)
        else:
            logger.info("Starting Camoufox for '%s' (profile=%s, no proxy)", agent_id, profile_dir)

        # Snapshot existing Firefox windows so we can identify the new one
        wids_before = await self._get_firefox_wids()

        # persistent_context=True → returns a BrowserContext directly.
        # geoip=True makes Camoufox connect through the proxy to resolve
        # the egress IP for fingerprint-consistent timezone/locale.  If the
        # proxy is slow to handshake, this can fail.  Retry once with geoip
        # (proxy may just need time), then fall back without it as last resort.
        try:
            browser = await AsyncNewBrowser(pw, **options)
        except Exception as e:
            if not options.get("geoip"):
                raise
            logger.warning(
                "Camoufox launch failed for '%s' with geoip (%s), retrying with geoip after brief wait",
                agent_id, e,
            )
            await asyncio.sleep(2)
            try:
                browser = await AsyncNewBrowser(pw, **options)
            except Exception as e2:
                logger.warning(
                    "Camoufox geoip retry failed for '%s' (%s), "
                    "falling back without geoip — fingerprint won't match proxy location",
                    agent_id, e2,
                )
                options.pop("geoip", None)
                browser = await AsyncNewBrowser(pw, **options)
        context = browser
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        # §6.6 ``navigator.connection`` fallback. We pass the spoof
        # values via Camoufox's ``config`` dict above, but it's not
        # documented whether Camoufox honours those keys (Firefox itself
        # doesn't natively expose NetworkInformation). Install an
        # ``add_init_script`` that defines ``navigator.connection`` if
        # the property is missing — runs in every document context
        # before page scripts. ``Object.defineProperty`` on
        # ``Navigator.prototype`` with ``configurable: true`` matches
        # what real Chromium-shaped Firefox extension polyfills do.
        try:
            from src.browser.stealth import pick_network_info
            netinfo = pick_network_info(agent_id)
            await context.add_init_script(
                _NAV_CONNECTION_INIT_SCRIPT.format(
                    effective=_js_string(netinfo["effectiveType"]),
                    downlink=float(netinfo["downlink"]),
                    rtt=int(netinfo["rtt"]),
                    save_data="false",
                ),
            )
        except Exception as e:
            # Non-fatal — Camoufox may already have NetworkInformation
            # support, and the §6.3 probe will catch the gap if neither
            # path lands.
            logger.debug(
                "navigator.connection init-script failed for '%s': %s",
                agent_id, e,
            )

        inst = CamoufoxInstance(agent_id, browser, context, page)

        # Discover the new X11 window for targeted focus
        wid = await self._discover_new_wid(wids_before)
        if wid:
            inst.x11_wid = wid
            logger.debug("Agent '%s' browser window: X11 WID %d", agent_id, wid)
            # Start idle mouse jitter for human-like fidgeting
            inst._jitter_task = asyncio.create_task(self._idle_mouse_jitter(inst))
        else:
            logger.warning(
                "Could not discover X11 WID for '%s' — interactions on "
                "high-sensitivity sites will use CDP (isTrusted=false)",
                agent_id,
            )
            inst._jitter_task = None

        # §6.3 run the navigator self-test once. Best-effort — a probe
        # failure must not block browser start (the inconsistency is
        # itself the operator's signal to investigate).
        try:
            await self._run_navigator_probe(inst)
        except Exception as e:
            logger.warning(
                "Navigator self-test probe failed for '%s': %s", agent_id, e,
            )

        return inst

    async def _run_navigator_probe(self, inst: CamoufoxInstance) -> None:
        """Read key navigator/Intl signals from the live page and validate
        them against the configured fingerprint.

        Stores the result on ``inst.probe_result`` and emits a one-shot
        ``nav_probe`` payload via ``self._metrics_sink`` (when wired). At
        the dashboard layer this surfaces as a ``browser_nav_probe``
        event, distinct from per-minute drain payloads.

        **Probes on ``about:blank``** so we read the platform / browser
        signals as the engine sees them, not as some loaded site has
        possibly shadowed them via an injected content script. With
        ``persistent_context=True`` the page resumes whatever the agent
        had open last session — that page's globals could include
        custom getters on ``Navigator.prototype``. Forcing a navigation
        to ``about:blank`` first eliminates that path.

        Mismatches the probe flags:
          * ``navigator.webdriver !== false`` — the canonical bot tell
          * ``navigator.platform`` doesn't match our configured ``os`` hint
          * ``navigator.userAgent`` lacks ``Firefox/`` (would mean §6.4
            tripwire was bypassed somehow at runtime)
          * ``navigator.connection.*`` is undefined (would mean §6.6
            override silently failed inside Camoufox)

        Per the plan: "WARNING if webdriver !== false or mismatch."
        """
        os_hint = os.environ.get("BROWSER_OS", "windows").lower()
        expected_platform = {
            "windows": "Win32",
            "macos": "MacIntel",
            "linux": "Linux x86_64",
        }.get(os_hint)

        # Best-effort isolate the probe context. ``about:blank`` is a
        # special URL that Firefox loads instantly with a fresh,
        # script-free document — perfect for reading raw navigator
        # signals. If the goto fails (e.g. ``about:blank`` blocked by
        # some weird policy), we fall through and probe the current
        # page anyway; the result will still be populated and any
        # shadowing-induced mismatch is itself useful signal.
        #
        # Side-effect operators may notice: this clobbers the resumed
        # page on a persistent-profile restart. An agent that had
        # ``twitter.com`` open last session sees ``about:blank`` for a
        # split second after restart, then whatever its first action
        # navigates to. Cookies / localStorage / IndexedDB all survive
        # — only the loaded-page URL is lost.
        try:
            await inst.page.goto("about:blank", timeout=5000)
        except Exception as e:
            logger.debug(
                "Probe pre-nav to about:blank failed for '%s' "
                "(continuing on current page): %s", inst.agent_id, e,
            )

        try:
            signals = await inst.page.evaluate(
                "() => ({"
                "  webdriver: navigator.webdriver,"
                "  plugins_len: navigator.plugins ? navigator.plugins.length : -1,"
                "  mimeTypes_len: navigator.mimeTypes ? navigator.mimeTypes.length : -1,"
                "  hardwareConcurrency: navigator.hardwareConcurrency,"
                "  deviceMemory: navigator.deviceMemory,"
                "  userAgent: navigator.userAgent,"
                "  platform: navigator.platform,"
                "  language: navigator.language,"
                "  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,"
                "  conn_effective: navigator.connection ? navigator.connection.effectiveType : null,"
                "  conn_downlink: navigator.connection ? navigator.connection.downlink : null,"
                "  conn_rtt: navigator.connection ? navigator.connection.rtt : null"
                "})",
            )
        except Exception as e:
            inst.probe_result = {
                "ok": False, "mismatches": [f"evaluate failed: {e}"],
                "signals": {},
            }
            logger.warning(
                "Navigator probe evaluate failed for '%s': %s",
                inst.agent_id, e,
            )
            return

        mismatches: list[str] = []
        if signals.get("webdriver") is not False:
            mismatches.append(
                f"webdriver={signals.get('webdriver')!r} (expected False)",
            )
        if expected_platform and signals.get("platform") != expected_platform:
            mismatches.append(
                f"platform={signals.get('platform')!r} "
                f"(expected {expected_platform!r} for os={os_hint!r})",
            )
        ua = signals.get("userAgent", "")
        if ua and "Firefox/" not in ua:
            mismatches.append(f"userAgent lacks 'Firefox/': {ua!r}")
        # navigator.connection should be populated by §6.6 spoof. ``null``
        # means the override silently failed.
        if signals.get("conn_effective") is None:
            mismatches.append(
                "navigator.connection.effectiveType is null "
                "(§6.6 override may have failed)",
            )

        ok = not mismatches
        inst.probe_result = {
            "ok": ok,
            "mismatches": mismatches,
            "signals": signals,
        }

        if ok:
            logger.info(
                "Navigator probe OK for '%s': platform=%s, ua=%s, tz=%s",
                inst.agent_id, signals.get("platform"),
                _short_ua(ua), signals.get("timezone"),
            )
        else:
            logger.warning(
                "Navigator probe MISMATCH for '%s': %s",
                inst.agent_id, "; ".join(mismatches),
            )

        # One-shot emit so operators see the result on the dashboard
        # without waiting for the next per-minute drain. Distinguished
        # from drain payloads by the ``kind`` field; Phase 2.1's history
        # buffer + mesh poll forward both shapes. We write to the history
        # buffer FIRST (so a missing sink doesn't drop the event) and
        # call the optional sink for in-process consumers (tests).
        probe_payload = {
            "kind": "nav_probe",
            "agent_id": inst.agent_id,
            "ok": ok,
            "mismatches": mismatches,
            "signals": signals,
        }
        self._metrics_seq += 1
        probe_payload["seq"] = self._metrics_seq
        probe_payload["ts"] = time.time()
        self._metrics_history.append(probe_payload)
        if self._metrics_sink is not None:
            try:
                self._metrics_sink(probe_payload)
            except Exception as e:
                logger.debug(
                    "metrics_sink raised on nav_probe for '%s': %s",
                    inst.agent_id, e,
                )

    async def stop(self, agent_id: str) -> None:
        """Stop and clean up a specific agent's browser."""
        async with self._lock:
            await self._stop_instance(agent_id)

    async def _stop_instance(self, agent_id: str) -> None:
        """Internal stop — caller must hold self._lock."""
        inst = self._instances.pop(agent_id, None)
        if inst is None:
            return
        # Drain counters BEFORE the instance disappears from the fleet.
        # Otherwise any clicks / snapshots / nav attempts since the last
        # minute-tick are silently lost when idle cleanup or explicit
        # stop fires. The periodic _emit_metrics hook only sees
        # still-live instances; post-pop is the final accounting chance.
        # Always write to the history buffer (even without a sink) so the
        # mesh poller sees the last minute of activity for a freshly-stopped
        # agent on its next tick. Empty payloads are skipped — no point
        # flooding the history with no-ops for agents that never did
        # anything.
        try:
            payload = inst.drain_metrics()
            if not _is_empty_payload(payload):
                self._metrics_seq += 1
                payload["seq"] = self._metrics_seq
                payload["ts"] = time.time()
                self._metrics_history.append(payload)
                if self._metrics_sink is not None:
                    self._metrics_sink(payload)
        except Exception as e:
            logger.warning(
                "Final metrics drain failed for '%s': %s", agent_id, e,
            )
        if self._user_focused_agent == agent_id:
            self._user_focused_agent = None
        jitter = getattr(inst, '_jitter_task', None)
        if jitter:
            jitter.cancel()
        # §5.3 dump the behavior recorder buffer (no-op when disabled or
        # empty). Runs before ``context.close()`` so a hung browser close
        # doesn't eat the diagnostic data. Acquire ``inst.lock`` first so
        # any in-flight click/type/scroll/navigate on this instance
        # finishes its ``record_*`` append BEFORE we flush — otherwise
        # the last 1-2 events land in the deque after the dump has
        # already cleared it and are silently lost.
        recorder = getattr(inst, "recorder", None)
        if recorder is not None:
            try:
                async with inst.lock:
                    recorder.dump(reason="stop")
            except Exception as e:
                logger.debug(
                    "Recorder dump failed for '%s': %s", agent_id, e,
                )
        try:
            await inst.context.close()
        except Exception as e:
            logger.debug("Error closing browser for '%s': %s", agent_id, e)
        logger.info("Stopped browser for '%s'", agent_id)

    async def stop_all(self) -> None:
        """Stop all browser instances and clean up Playwright."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        async with self._lock:
            for agent_id in list(self._instances.keys()):
                await self._stop_instance(agent_id)
        if self._captcha_solver:
            await self._captcha_solver.close()
        if self._playwright:
            with contextlib.suppress(Exception):
                await self._pw_context.__aexit__(None, None, None)
            self._playwright = None

    async def reset(self, agent_id: str) -> None:
        """Reset browser session — close and reopen (preserves profile)."""
        await self.stop(agent_id)
        # Next get_or_start will create a fresh instance with same profile

    def set_proxy_config(self, agent_id: str, config: dict | None) -> None:
        """Store proxy config for an agent. Pass None to clear."""
        if config is None:
            self._proxy_configs.pop(agent_id, None)
        else:
            self._proxy_configs[agent_id] = config

    def get_proxy_config(self, agent_id: str) -> dict | None:
        """Get stored proxy config for an agent, or None."""
        return self._proxy_configs.get(agent_id)

    async def get_status(self, agent_id: str) -> dict:
        """Get status for a specific agent's browser.

        Includes the rolling 100-click success rate (§5.2) as a live gauge —
        distinct from the per-minute counters, which only flow via EventBus.
        Operators polling /status see the current health signal without
        waiting for the next emit tick.
        """
        async with self._lock:
            inst = self._instances.get(agent_id)
            if not inst:
                return {"running": False}
            status = {
                "running": True,
                "idle_seconds": int(time.time() - inst.last_activity),
                "url": inst.page.url if inst.page else "",
                "click_window_size": len(inst.click_window),
                "click_success_rate_100": inst.rolling_click_success_rate(),
            }
            # §6.3 navigator probe summary (boot-once). When ``probe_result``
            # is None the probe hasn't run yet (instance just started).
            # Operators polling /status get the same signal as the dashboard
            # nav-probe event; we only surface the high-level shape, not the
            # raw signals payload (those are in the EventBus event).
            if inst.probe_result is not None:
                status["probe_ok"] = inst.probe_result["ok"]
                status["probe_mismatches"] = list(
                    inst.probe_result.get("mismatches") or [],
                )
            return status

    async def get_service_status(self) -> dict:
        """Get overall service health."""
        async with self._lock:
            return {
                "healthy": True,
                "active_browsers": len(self._instances),
                "max_concurrent": self.max_concurrent,
                "agents": list(self._instances.keys()),
                "boot_id": self.boot_id,
            }

    async def focus(self, agent_id: str) -> bool:
        """Bring an agent's browser window to VNC foreground.

        Auto-starts the browser if it isn't running yet, so the user
        always sees a window when they click "Browser" in the dashboard.

        Also records this as the user's explicitly focused agent so
        ``refocus_active()`` keeps this window visible even when other
        agents are using their browsers in the background.

        Two-layer raise:
        1. bring_to_front() — browser-protocol level (activates the tab)
        2. xdotool windowmap + windowraise — X11 level (unmaps if iconic,
           then raises in the stacking order so VNC actually sees it)
        """
        self._user_focused_agent = agent_id
        try:
            inst = await self.get_or_start(agent_id)
        except Exception as e:
            logger.warning("Focus: browser failed to start for '%s': %s", agent_id, e)
            self._user_focused_agent = None
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
            #
            # When a specific X11 window ID is known, target it directly.
            # Without this, `search --class firefox` matches ALL Firefox
            # windows and raises whichever it finds first — breaking
            # per-agent browser switching on the shared VNC display.
            try:
                wid = inst.x11_wid
                if wid:
                    wid_s = str(wid)
                    cmd = ["xdotool", "windowmap", "--sync", wid_s,
                           "windowraise", wid_s, "windowfocus", wid_s]
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: subprocess.run(cmd, capture_output=True, timeout=3),
                    )
                else:
                    # No WID known — skip xdotool entirely.  The fallback
                    # `search --class firefox` matches ALL Firefox windows
                    # and raises whichever it finds first, which is wrong
                    # in multi-agent scenarios.  bring_to_front() above
                    # already activated the correct tab at the browser-
                    # protocol level.
                    logger.debug(
                        "No X11 WID for '%s'; skipping xdotool raise", agent_id,
                    )
            except Exception as e:
                logger.debug("xdotool raise skipped for '%s': %s", agent_id, e)
            return True

    async def set_user_control(self, agent_id: str, enabled: bool) -> dict:
        """Toggle user browser control.

        When enabled, pauses agent X11 input (mouse jitter, click, type,
        scroll) so the user can interact via VNC without cursor fighting.
        Browser read operations (snapshot, screenshot) remain available.
        """
        inst = self._instances.get(agent_id)
        if not inst:
            return {"success": False, "error": "No browser instance"}
        inst._user_control = enabled
        logger.info(
            "User %s browser control for %s",
            "took" if enabled else "released", agent_id,
        )
        return {"success": True, "user_control": enabled}

    # ── Browser operations ──────────────────────────────────

    async def navigate(
        self, agent_id: str, url: str, wait_ms: int = 1000,
        wait_until: str = "domcontentloaded",
        snapshot_after: bool = False,
        referer: str | None = None,
    ) -> dict:
        """Navigate to URL and return page text.

        wait_until controls when Playwright considers navigation complete:
          - "domcontentloaded" (default): HTML parsed; fast but JS may not have run
          - "load": all resources loaded; good for most sites
          - "networkidle": no network requests for 500ms; best for heavy SPAs (X, etc.)
          - "commit": first byte received; fastest, rarely useful

        referer (Phase 3 §6.5): override the Referer header / document.referrer
        for this nav. ``None`` (default) lets the service pick a plausible
        value from :func:`src.browser.stealth.pick_referer` based on the
        target host and the agent's recent nav history. Pass an empty
        string ``""`` to explicitly send NO referer (equivalent to a
        bookmarked / typed-URL arrival). Pass a specific URL to override
        the picker entirely — useful when the agent is following a known
        link from a specific page.
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
            if inst._user_control:
                return {
                    "success": False,
                    "error": "User has browser control — action paused until control is released.",
                }

            # §6.5 referer realism. ``referer is None`` ⇒ picker decides;
            # explicit ``""`` ⇒ direct navigation (no referer); any other
            # string ⇒ caller override, validated before reaching
            # Playwright (the agent skill is LLM-callable so a malformed
            # value can land here from untrusted-by-default input).
            if referer is None:
                # Only honour ``inst.page.url`` as the previous-URL hint
                # AFTER the first navigate this session — otherwise a
                # persistent profile resume would falsely indicate
                # internal-link arrival on the very first nav.
                previous_url = (
                    inst.page.url if inst.page and inst.had_real_navigate
                    else ""
                )
                resolved_referer = pick_referer(
                    url,
                    previous_url=previous_url,
                    recent_referers=tuple(inst.recent_referers),
                )
            else:
                # Caller override — must validate. ValueError surfaces
                # to the agent as a navigate error; better than silently
                # forwarding ``javascript:alert(1)`` to Playwright.
                try:
                    resolved_referer = validate_referer(referer)
                except ValueError as e:
                    return {
                        "success": False,
                        "error": f"invalid referer: {e}",
                    }

            # Maintain the rolling-5 history on the instance. Both
            # picker output and validated overrides are tracked so the
            # picker can see "we just used a direct/social/search
            # pattern" and rotate accordingly.
            inst.recent_referers.append(resolved_referer)

            # Playwright accepts ``referer`` for goto and sets both the
            # network header and document.referrer consistently. Empty
            # string ⇒ omit the kwarg ⇒ Playwright sends no Referer.
            goto_kwargs: dict = {"wait_until": wait_until, "timeout": 30000}
            if resolved_referer:
                goto_kwargs["referer"] = resolved_referer

            # Single retry on timeout — transient network issues get a second chance.
            for attempt in range(2):
                try:
                    await inst.page.goto(url, **goto_kwargs)
                    break
                except Exception as e:
                    if attempt == 0 and "timeout" in str(e).lower():
                        logger.debug("Navigation timeout, retrying: %s", url)
                        await asyncio.sleep(2)
                        continue
                    # Give up — if this was a timeout (including after retry),
                    # log it for §4.6 metrics. Non-timeout failures go in a
                    # generic bucket (just counted as click_fail… actually,
                    # navigation is distinct; only timeouts go here).
                    if "timeout" in str(e).lower():
                        inst.m_nav_timeout += 1
                    return {"success": False, "error": str(e)}

            # §5.3 recorder: log host only, never the full URL — query
            # strings and fragments routinely carry secrets.
            inst.recorder.record_navigate(
                host=parsed.hostname or "", wait_until=wait_until,
            )

            # §6.5: future navs may now use ``inst.page.url`` as a
            # previous-URL hint for the picker. The flag stays True for
            # the lifetime of this CamoufoxInstance; a browser restart
            # creates a new instance and resets it.
            inst.had_real_navigate = True

            inst.dialog_active = False
            inst.dialog_detected = False
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000 + navigation_jitter())
            try:
                title = await inst.page.title()
                current_url = inst.page.url
                body_text = ""
                # Always extract body at the historical 5000-char cap so
                # we have a usable fallback if the snapshot path fails
                # below. We trim to a 1000-char preview only AFTER the
                # snapshot succeeds — that's when the agent has the full
                # element tree and doesn't need a long body. If the
                # snapshot fails, we ship the full body so the agent
                # isn't stranded with truncated text + empty snapshot.
                if not inst._js_snapshot_mode:
                    try:
                        _a11y = await inst.page.accessibility.snapshot()
                        body_text = _extract_text_from_a11y(
                            _a11y, max_chars=5000,
                        )
                    except AttributeError:
                        inst._js_snapshot_mode = True
                    except Exception:
                        pass
                result = {
                    "success": True,
                    "data": {
                        "url": self.redactor.redact(agent_id, current_url),
                        "title": self.redactor.redact(agent_id, title),
                        # Body filled in below once we know whether the
                        # optional snapshot succeeded — see body cap
                        # comment.
                        "body": "",
                    },
                }
                # Auto-detect CAPTCHAs so the agent knows immediately
                captcha = await self._check_captcha(inst)
                if captcha:
                    result["captcha"] = captcha
                snapshot_succeeded = False
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    snap_data = snap.get("data") or {}
                    result["snapshot"] = snap_data
                    snapshot_succeeded = bool(snap.get("success") and snap_data)
                # §7.6: shrink body to 1000-char preview ONLY when the
                # snapshot actually carried back element refs. A failed
                # snapshot would otherwise leave the agent with both a
                # truncated body AND an empty/{} snapshot — strictly
                # worse than the snapshot_after=False path. Restore the
                # full body in that failure case.
                final_body = (
                    body_text[:1000] if snapshot_succeeded else body_text
                )
                result["data"]["body"] = self.redactor.redact(
                    agent_id, final_body,
                )
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def _build_a11y_tree(self, inst: CamoufoxInstance, root=None):
        """Get accessibility tree, falling back to JS-based DOM walk.

        Playwright's ``page.accessibility.snapshot()`` uses Firefox's native
        ``nsIAccessibilityService`` — a browser-internal API with zero
        JavaScript execution in the page context.  This makes it invisible
        to anti-bot systems that hook DOM APIs via Proxy.

        If the API is absent (``AttributeError``), permanently switches to
        a JS-based tree builder per-instance.  Transient failures get one
        retry before falling through to JS.
        """
        if not inst._js_snapshot_mode:
            for attempt in range(2):
                try:
                    if root:
                        return await inst.page.accessibility.snapshot(root=root)
                    return await inst.page.accessibility.snapshot()
                except AttributeError:
                    logger.warning(
                        "page.accessibility not available for %s — "
                        "switching to JS-based accessibility tree",
                        inst.agent_id,
                    )
                    inst._js_snapshot_mode = True
                    break
                except Exception:
                    if attempt == 0:
                        await asyncio.sleep(0.15)
                        continue
                    logger.warning(
                        "page.accessibility.snapshot() failed after retry "
                        "for %s — falling back to JS tree",
                        inst.agent_id,
                    )
                    break

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
            refs: dict[str, RefHandle] = {}
            # Snapshot page_id up front — resolves to Page that was active
            # when the snapshot was taken. If the agent later switches tabs,
            # refs still carry their original page_id so resolution targets
            # the right tab (or raises RefStale if the tab is closed).
            snapshot_page_id = inst._page_id_for(inst.page)
            ref_counter = [0]
            # Counts occurrences of each (role, name) pair so we can
            # disambiguate duplicate elements (e.g. X's two composer nodes).
            occurrence_counts: dict[tuple, int] = {}

            _MAX_WALK_DEPTH = 50

            # Collect entries for §7.2 v2 rendering AND build v1 lines in
            # parallel. The entry list is a structured intermediate so we
            # can pivot between formats post-walk without a second tree
            # traversal. Each entry: (ref_id, role, name, attr_str,
            # landmark, depth).
            entries: list[tuple[str, str, str, str, str, int]] = []

            def _walk(node, depth=0):
                if depth > _MAX_WALK_DEPTH:
                    return
                role = node.get("role", "")
                name = node.get("name", "")
                if role in _ACTIONABLE_ROLES or role in _CONTEXT_ROLES:
                    if ref_counter[0] < _MAX_SNAPSHOT_ELEMENTS:
                        ref_id = f"e{ref_counter[0]}"
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
                            attrs.append(f"value={val}")
                        if occ > 0:
                            attrs.append(f"dup:{occ + 1}")
                        attr_str = f" [{', '.join(attrs)}]" if attrs else ""

                        # Structural context from nearest landmark ancestor
                        landmark = node.get("landmark", "")
                        ctx_str = f" ({landmark})" if landmark else ""

                        line = f"{'  ' * depth}- [{ref_id}] {role} \"{name}\"{attr_str}{ctx_str}"
                        lines.append(line)
                        entries.append(
                            (ref_id, role, name, attr_str, landmark, depth),
                        )
                        # scope_root is finalized after the modal-scoping
                        # branch below. For now record the unscoped handle;
                        # we overwrite scope_root once we know the final
                        # dialog_active state (see scope-root patching below).
                        refs[ref_id] = RefHandle.light_dom(
                            page_id=snapshot_page_id,
                            scope_root=None,
                            role=role,
                            name=name,
                            occurrence=occ,
                            disabled=bool(node.get("disabled")),
                        )
                for child in node.get("children", []):
                    _walk(child, depth + 1)

            # When a modal dialog is open, scope to only dialog elements
            # so agents don't see/click elements behind the overlay
            # (e.g. X's sidebar "Post" button behind the compose modal).
            modal_els = await inst.page.query_selector_all(_MODAL_SELECTOR)
            vp = inst.page.viewport_size
            visible_modals = []
            for el in modal_els:
                if await self._is_visible_modal(el, vp):
                    visible_modals.append(el)

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
                    r for r in refs.values() if r.role in _ACTIONABLE_ROLES
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
                    # §7.2: ``entries`` is the parallel structure v2 renders
                    # from. Forgetting to reset it here would leak entries
                    # from the discarded scoping pass into the v2 output —
                    # invisible in v1 (which renders ``lines`` only) but
                    # produces phantom refs that don't match ``inst.refs``
                    # under v2.
                    entries.clear()
                    ref_counter[0] = 0
                    occurrence_counts.clear()
                    lines.append("** Modal dialog is open — only dialog elements are shown **")
                    # Re-query modal elements — handles go stale when SPAs
                    # like X/Twitter re-render the modal during the wait.
                    fresh_modals = []
                    for el in (await inst.page.query_selector_all(_MODAL_SELECTOR)):
                        if await self._is_visible_modal(el, vp):
                            fresh_modals.append(el)
                    if fresh_modals:
                        visible_modals = fresh_modals
                    for el in visible_modals:
                        try:
                            subtree = await self._build_a11y_tree(inst, root=el)
                            if subtree:
                                _walk(subtree)
                        except Exception:
                            pass
                    actionable_refs = [
                        r for r in refs.values() if r.role in _ACTIONABLE_ROLES
                    ]
                if not actionable_refs:
                    logger.warning(
                        "Modal detected but scoping produced 0 actionable "
                        "refs after retries — falling back to full tree "
                        "for %s", agent_id,
                    )
                    # Keep dialog_active=True so _locator_from_ref stays
                    # scoped to modal elements.  This prevents clicks from
                    # targeting elements behind the overlay (e.g. X's feed
                    # "Post" button behind the compose modal).  A modal-
                    # scoped click that can't find the element will timeout
                    # rather than hit the wrong target.
                    lines.clear()
                    # Same reset rationale as the retry branch above —
                    # discard fallback's ``entries`` so v2 rendering
                    # only sees the post-fallback _walk(tree) output.
                    entries.clear()
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

            # Patch scope_root on refs captured during modal scoping so
            # `_locator_from_ref` queries are bounded to the dialog subtree.
            # (Refs emitted before the modal branch don't have scope_root;
            # set it now that we know the final dialog_active state.)
            if inst.dialog_active:
                for rid, handle in refs.items():
                    if handle.scope_root is None:
                        refs[rid] = RefHandle(
                            page_id=handle.page_id,
                            frame_id=handle.frame_id,
                            shadow_path=handle.shadow_path,
                            scope_root=_MODAL_SELECTOR,
                            role=handle.role,
                            name=handle.name,
                            occurrence=handle.occurrence,
                            disabled=handle.disabled,
                            element_key=handle.element_key,
                        )

            inst.refs = refs
            # §7.2 — choose between v1 (per-element landmark suffix) and
            # v2 (landmark headers + capped indent). Flag default is v1
            # for the canary period, flips to v2 once parse rate ≥99%.
            from src.browser.flags import get_str
            fmt = (
                get_str("BROWSER_SNAPSHOT_FORMAT", "v1", agent_id=agent_id)
                .strip()
                .lower()
            )
            if fmt == "v2":
                snapshot_text = _format_snapshot_v2(lines, entries)
            else:
                snapshot_text = (
                    "\n".join(lines) if lines else "(no interactive elements)"
                )
            snapshot_text = self.redactor.redact(agent_id, snapshot_text)
            # Record snapshot byte size for §4.6 metrics. Collected per call;
            # drained as p50/p95 on the next minute tick.
            inst.m_snapshot_bytes.append(len(snapshot_text))
            # Agent-visible `refs` uses the minimal dict shape (backward
            # compatible); RefHandle is strictly an internal detail.
            response_refs = {rid: h.to_agent_dict() for rid, h in refs.items()}
            return {"success": True, "data": {"snapshot": snapshot_text, "refs": response_refs}}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def _is_visible_modal(el, vp_size: dict | None) -> bool:
        """Check if a modal element is genuinely visible with real area.

        Filters zero-area or off-screen modals (e.g. LinkedIn's background
        messaging panels) that pass Playwright's ``is_visible()`` but are
        not true dialog overlays.
        """
        try:
            if not await el.is_visible():
                return False
            bb = await el.bounding_box()
            if not bb or bb["width"] < 10 or bb["height"] < 10:
                return False
            if vp_size:
                if (bb["x"] + bb["width"] <= 0 or bb["x"] >= vp_size["width"]
                        or bb["y"] + bb["height"] <= 0 or bb["y"] >= vp_size["height"]):
                    return False
            return True
        except Exception:
            return False

    def _locator_from_ref(self, inst: CamoufoxInstance, ref: str):
        """Build a Playwright locator from a stored RefHandle.

        Resolution order (§4.2):
            1. ``page_id`` → Page object (raises :class:`RefStale` if the
               tab has closed).
            2. ``frame_id`` → Frame (None = main frame).  Always None in
               v1.2; populated by §8.4 iframe traversal.
            3. ``scope_root`` — modal selector bound during snapshot, so
               occurrence indices match.
            4. ``shadow_path`` — walk through open shadow roots.  Empty in
               v1.2; populated by §8.3 shadow DOM walker.
            5. ``get_by_role(role, name=name, exact=True).nth(occurrence)``.

        Returns ``None`` when ``ref`` isn't in ``inst.refs`` (classic
        not-found).  Raises :class:`RefStale` when the ref points to a
        closed tab — caller should report ``ref_stale`` to the agent so
        it knows to re-snapshot rather than retry.
        """
        handle = inst.refs.get(ref)
        if handle is None:
            return None
        # Resolve Page; may raise RefStale.
        page = inst._resolve_page_id(handle.page_id)
        # v1.2: frame/shadow always empty — light DOM, main frame. Those
        # branches activate in §8.3 / §8.4.
        base = page
        if handle.scope_root:
            base = page.locator(handle.scope_root)
        if handle.name:
            locator = base.get_by_role(handle.role, name=handle.name, exact=True)
        else:
            locator = base.get_by_role(handle.role)
        return locator.nth(handle.occurrence)

    async def _human_click(self, page, locator, *, force: bool = False,
                           timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Click with a preceding hover so the mouse visibly moves to the target.

        Playwright's ``locator.click()`` dispatches a click at the element's
        center coordinates but does NOT generate the ``mousemove`` events a
        real user produces while moving the cursor to the target.  Anti-bot
        systems (X/Twitter, Cloudflare) track mouse-movement patterns and
        flag clicks that appear without any prior movement.

        The hover-then-click pattern:
        1. ``locator.hover()`` — Playwright scrolls the element into view and
           moves the mouse along a path to the element center.  With Camoufox's
           ``humanize=True``, this path includes natural-looking Bézier curves.
        2. Brief settle (20–60 ms) — models the human reaction gap between
           arriving at the target and pressing the button.
        3. ``page.mouse.click(x, y)`` — fires the mousedown/mouseup at the
           current mouse position (already on the element from the hover).

        When ``force=True``, falls back to ``locator.click(force=True)`` since
        hover may fail on elements obscured by overlays.
        """
        if force:
            await locator.click(timeout=timeout, force=True)
            return
        try:
            await locator.hover(timeout=timeout)
            await asyncio.sleep(random.uniform(0.02, 0.06))
        except Exception:
            pass  # Hover failed — click below will still attempt
        await locator.click(timeout=timeout, force=False)

    async def _human_click_selector(self, page, selector: str, *,
                                    force: bool = False,
                                    timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Like _human_click but takes a CSS selector instead of a locator.

        Hovers first to generate natural mouse movement, then clicks.
        """
        if force:
            await page.click(selector, timeout=timeout, force=True)
            return
        try:
            await page.hover(selector, timeout=timeout)
            await asyncio.sleep(random.uniform(0.02, 0.06))
        except Exception:
            pass  # Hover failed — click will still work
        await page.click(selector, timeout=timeout, force=False)

    async def _x11_move_to(
        self, inst: CamoufoxInstance, target_x: int, target_y: int,
    ) -> None:
        """Move mouse to (target_x, target_y) via xdotool with a Bezier trajectory.

        Generates a natural-looking curved mouse path using cubic Bezier
        interpolation with randomized control points. Real human wrist
        movement produces slight S-curves, not straight lines.

        Velocity easing (cubic ease-in-out) models Fitts' Law: slow
        departure, fast cruise, slow precision-landing. Step count
        scales with distance so short movements stay snappy and long
        movements stay smooth.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool move")

        wid_s = str(wid)
        loop = asyncio.get_running_loop()

        # Get current mouse position
        start_x, start_y = 0, 0
        loc_result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "getmouselocation"],
                capture_output=True, text=True, timeout=3,
            ),
        )
        if loc_result.returncode == 0:
            for part in loc_result.stdout.split():
                if part.startswith("x:"):
                    start_x = int(part[2:])
                elif part.startswith("y:"):
                    start_y = int(part[2:])

        # Bezier control points — offset perpendicular to the line
        dx = target_x - start_x
        dy = target_y - start_y
        dist = max(1, (dx * dx + dy * dy) ** 0.5)
        # Perpendicular unit vector
        perp_x, perp_y = -dy / dist, dx / dist
        # Control points with randomized perpendicular offset (scaled by distance)
        spread = min(dist * 0.3, 60)
        off1 = random.uniform(-spread, spread)
        off2 = random.uniform(-spread, spread)
        cp1_x = start_x + dx * 0.25 + perp_x * off1
        cp1_y = start_y + dy * 0.25 + perp_y * off1
        cp2_x = start_x + dx * 0.75 + perp_x * off2
        cp2_y = start_y + dy * 0.75 + perp_y * off2

        # Scale step count with distance — short moves stay snappy,
        # long moves stay smooth.  Range: 3 steps (tiny) to 14 (across screen).
        steps = max(3, min(14, int(dist / 80) + random.randint(2, 4)))

        for i in range(1, steps + 1):
            # Raw parameter
            raw_t = i / steps
            # Cubic ease-in-out: slow start, fast middle, slow landing
            # Models Fitts' Law deceleration as cursor approaches target
            if raw_t < 0.5:
                t = 4 * raw_t * raw_t * raw_t
            else:
                t = 1 - ((-2 * raw_t + 2) ** 3) / 2

            u = 1 - t
            wp_x = int(
                u**3 * start_x + 3 * u**2 * t * cp1_x
                + 3 * u * t**2 * cp2_x + t**3 * target_x
            )
            wp_y = int(
                u**3 * start_y + 3 * u**2 * t * cp1_y
                + 3 * u * t**2 * cp2_y + t**3 * target_y
            )
            wp_x = max(0, wp_x)
            wp_y = max(0, wp_y)
            mv_result = await loop.run_in_executor(
                None,
                lambda x=wp_x, y=wp_y: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(x), str(y)],
                    capture_output=True, timeout=3,
                ),
            )
            if mv_result.returncode != 0:
                raise RuntimeError(
                    f"xdotool mousemove failed (rc={mv_result.returncode})"
                )
            await asyncio.sleep(x11_step_delay())

        # Overshoot + correction for long movements — models the human
        # tendency to slightly overshoot the target and make a tiny
        # corrective flick back. Only on ~30% of long movements.
        if dist > 300 and random.random() < 0.3:
            # Direction from last control point toward target
            end_dx = target_x - cp2_x
            end_dy = target_y - cp2_y
            end_dist = max(1, (end_dx**2 + end_dy**2) ** 0.5)
            overshoot_px = random.uniform(3, 8)
            ov_x = max(0, int(target_x + end_dx / end_dist * overshoot_px))
            ov_y = max(0, int(target_y + end_dy / end_dist * overshoot_px))
            # Overshoot
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(ov_x), str(ov_y)],
                    capture_output=True, timeout=3,
                ),
            )
            await asyncio.sleep(x11_step_delay())
            # Correct back to exact target
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["xdotool", "mousemove", "--sync", "--window", wid_s,
                     str(target_x), str(target_y)],
                    capture_output=True, timeout=3,
                ),
            )
            await asyncio.sleep(x11_step_delay())

    async def _x11_ensure_in_viewport(
        self, inst: CamoufoxInstance, locator, *,
        timeout: int = _CLICK_TIMEOUT_MS,
    ) -> None:
        """Scroll element into viewport using X11 wheel events.

        Replaces ``locator.scroll_into_view_if_needed()`` for the X11
        input path.  Protocol-level ``scrollIntoView`` produces scroll
        events WITHOUT ``WheelEvent`` — a detectable automation signal.
        X11 button 4/5 produces real ``WheelEvent`` with
        ``deltaMode=DOM_DELTA_LINE``, matching physical hardware.

        Scrolls in small increments (2–3 notches per batch), re-measures
        the element position after each batch to prevent overshoot.
        Falls back to protocol scroll for edge cases (elements inside
        scrollable inner containers, elements not yet in the DOM).
        """
        if not inst.x11_wid:
            await locator.scroll_into_view_if_needed(timeout=timeout)
            return

        vp = inst.page.viewport_size
        if not vp:
            await locator.scroll_into_view_if_needed(timeout=timeout)
            return

        vp_h = vp["height"]
        margin = 60

        for _ in range(10):
            box = await locator.bounding_box()
            if box is None:
                break  # Not in DOM — protocol scroll only option

            center_y = box["y"] + box["height"] / 2
            if margin <= center_y <= vp_h - margin:
                return  # Element is visible

            button = "4" if center_y < margin else "5"
            prev_center = center_y

            batch = random.randint(2, 3)
            for _ in range(batch):
                try:
                    await self._x11_scroll_notch(inst, button)
                except Exception:
                    break
                await asyncio.sleep(scroll_pause() * 0.4)

            # Wait for smooth scrolling to settle
            await asyncio.sleep(0.10)

            # Check if element position actually changed
            new_box = await locator.bounding_box()
            if new_box is None:
                break
            new_center = new_box["y"] + new_box["height"] / 2
            if abs(new_center - prev_center) < 2:
                break  # Scroll didn't move element — inner container

        # Fallback for edge cases
        await locator.scroll_into_view_if_needed(timeout=timeout)

    async def _x11_click(self, inst: CamoufoxInstance, locator, *,
                         timeout: int = _CLICK_TIMEOUT_MS) -> None:
        """Click via xdotool for isTrusted=true events.

        Bot-detection systems (ArkoseLabs on X/Twitter) hook addEventListener
        and reject clicks where event.isTrusted is false.  CDP-dispatched
        clicks always have isTrusted=false.  xdotool injects real X11
        ButtonPress/ButtonRelease events through the kernel input stack,
        which the browser marks isTrusted=true.

        Steps:
        1. _x11_ensure_in_viewport — scrolls element into viewport using
           X11 wheel events (real WheelEvent with DOM_DELTA_LINE) instead
           of protocol-level scrollIntoView (no WheelEvents, detectable)
        2. Get element bounding box (viewport coords)
        3. Bezier mouse trajectory via _x11_move_to
        4. mousedown + human dwell + mouseup (not instant click)
        """
        # 1. Scroll into view — prefer X11 wheel events over protocol scroll
        await self._x11_ensure_in_viewport(inst, locator, timeout=timeout)
        await asyncio.sleep(x11_settle_delay())

        # 2. Get element position — jitter within inner area, not dead center
        box = await locator.bounding_box()
        if not box:
            raise RuntimeError("Element has no bounding box — not visible")
        # Real humans don't click dead center — offset within inner 60%
        jitter_x = random.uniform(-0.15, 0.15) * box["width"]
        jitter_y = random.uniform(-0.10, 0.10) * box["height"]
        target_x = int(box["x"] + box["width"] / 2 + jitter_x)
        target_y = int(box["y"] + box["height"] / 2 + jitter_y)

        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool click")

        # 3. Move mouse with natural Bezier trajectory
        await self._x11_move_to(inst, target_x, target_y)

        # 4. Click with human-like dwell time (mousedown -> hold -> mouseup)
        wid_s = str(wid)
        loop = asyncio.get_running_loop()
        await asyncio.sleep(pre_click_settle())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mousedown", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )
        await asyncio.sleep(click_dwell())
        await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "mouseup", "--clearmodifiers", "--window", wid_s, "1"],
                capture_output=True, timeout=3,
            ),
        )

    async def _x11_hover(self, inst: CamoufoxInstance, locator) -> None:
        """Move mouse to element via xdotool for isTrusted=true mousemove events."""
        await self._x11_ensure_in_viewport(inst, locator)
        await asyncio.sleep(x11_settle_delay())

        box = await locator.bounding_box()
        if not box:
            raise RuntimeError("Element has no bounding box — not visible")
        # Jitter within inner area — same as _x11_click for consistency
        jitter_x = random.uniform(-0.15, 0.15) * box["width"]
        jitter_y = random.uniform(-0.10, 0.10) * box["height"]
        target_x = int(box["x"] + box["width"] / 2 + jitter_x)
        target_y = int(box["y"] + box["height"] / 2 + jitter_y)

        await self._x11_move_to(inst, target_x, target_y)

    async def _idle_mouse_jitter(self, inst: CamoufoxInstance) -> None:
        """Periodic mouse micro-movement to simulate human fidgeting.

        Real users constantly micro-move the mouse while reading — small
        drifts, twitches, and repositioning. A mouse that is perfectly
        still for seconds between actions is a textbook bot pattern
        detected by ArkoseLabs and DataDome.
        """
        while True:
            await asyncio.sleep(random.uniform(2.0, 7.0))
            if not inst.x11_wid or inst.lock.locked() or inst._user_control:
                continue
            try:
                dx = random.randint(-3, 3)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda _dx=dx, _dy=dy: subprocess.run(
                        ["xdotool", "mousemove_relative", "--sync",
                         "--", str(_dx), str(_dy)],
                        capture_output=True, timeout=2,
                    ),
                )
            except asyncio.CancelledError:
                return
            except Exception:
                pass

    async def _x11_type(self, inst: CamoufoxInstance, text: str,
                        *, typos: bool = True) -> None:
        """Type text via xdotool for isTrusted=true key events.

        Same rationale as _x11_click — bot-detection checks isTrusted on
        keydown/keyup in tweet composer textareas.  xdotool key/type
        generates real X11 KeyPress/KeyRelease events.

        When *typos* is True (default), injects occasional typo +
        backspace corrections to simulate natural human error patterns.
        Zero-typo typing at consistent speed is one of the strongest
        bot signals.  Typos are placed mid-word only (avoiding handles,
        hashtags, URLs) and capped at a per-text budget.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool type")

        loop = asyncio.get_running_loop()
        wid_s = str(wid)

        # ── Typo budget ──────────────────────────────────────────
        # Pre-select positions for typo injection.  Only mid-word
        # alphabetic characters qualify (skips handles, hashtags,
        # URLs).  Budget scales with text length.
        typo_positions: set[int] = set()
        if typos:
            candidates = [
                i for i, c in enumerate(text)
                if c.isalpha() and c.lower() in _TYPO_NEIGHBORS
                and i > 0 and text[i - 1].isalpha()
            ]
            alpha_count = len(candidates)
            if alpha_count >= 15:
                expected = max(1.0, alpha_count / 120)
                budget = max(0, int(random.gauss(expected, expected * 0.5)))
                budget = min(budget, 4)
                if budget > 0 and len(candidates) >= budget:
                    typo_positions = set(random.sample(candidates, budget))

        # ── Character loop ───────────────────────────────────────
        prev_char = ""
        for i, char in enumerate(text):
            # Word-boundary think pauses
            pause_prob = 0.08 if prev_char in _WORD_BOUNDARY_CHARS else 0.015
            if random.random() < pause_prob:
                await asyncio.sleep(think_pause())

            # Typo injection — wrong adjacent key → pause → backspace → correct
            if i in typo_positions:
                wrong = random.choice(_TYPO_NEIGHBORS[char.lower()])
                if char.isupper():
                    wrong = wrong.upper()
                # Type wrong character
                await loop.run_in_executor(
                    None,
                    lambda c=wrong: subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--window", wid_s,
                         "--delay", "0", "--", c],
                        capture_output=True, timeout=3,
                    ),
                )
                await asyncio.sleep(keystroke_delay(wrong))
                # Pause — noticing the error
                await asyncio.sleep(random.uniform(0.15, 0.4))
                # Backspace to correct
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s,
                         "BackSpace"],
                        capture_output=True, timeout=3,
                    ),
                )
                await asyncio.sleep(random.uniform(0.03, 0.08))

            # Type the correct character
            if char == "\n":
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s, "Return"],
                        capture_output=True, timeout=3,
                    ),
                )
            elif char == "\t":
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["xdotool", "key", "--clearmodifiers", "--window", wid_s, "Tab"],
                        capture_output=True, timeout=3,
                    ),
                )
            else:
                await loop.run_in_executor(
                    None,
                    lambda c=char: subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", "--window", wid_s,
                         "--delay", "0", "--", c],
                        capture_output=True, timeout=3,
                    ),
                )
            await asyncio.sleep(keystroke_delay(char))
            prev_char = char

    async def _x11_key(self, inst: CamoufoxInstance, key: str) -> None:
        """Send a key combination via xdotool for isTrusted=true key events."""
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool key")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "key", "--clearmodifiers", "--window", str(wid), key],
                capture_output=True, timeout=3,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(f"xdotool key {key!r} failed (rc={result.returncode})")

    async def _x11_scroll_notch(self, inst: CamoufoxInstance, button: str) -> None:
        """Send a single scroll notch via xdotool button 4 (up) or 5 (down).

        X11 button 4/5 events are processed by Firefox identically to
        physical mouse wheel input, producing ``WheelEvent`` with
        ``deltaMode=DOM_DELTA_LINE`` — matching real hardware.  Playwright's
        ``page.mouse.wheel()`` instead uses ``nsIDOMWindowUtils.sendWheelEvent``
        with ``deltaMode=DOM_DELTA_PIXEL``, which is a detectable fingerprint.
        """
        wid = inst.x11_wid
        if not wid:
            raise RuntimeError("No X11 window ID — cannot use xdotool scroll")
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["xdotool", "click", "--clearmodifiers", "--window", str(wid), button],
                capture_output=True, timeout=3,
            ),
        )
        if result.returncode != 0:
            raise RuntimeError(f"xdotool scroll button {button} failed (rc={result.returncode})")

    @staticmethod
    def _playwright_key_to_xdotool(key: str) -> str:
        """Convert a Playwright key name to xdotool key name."""
        parts = key.split("+")
        mapped = [_PLAYWRIGHT_TO_XDOTOOL.get(p, p) for p in parts]
        return "+".join(mapped)

    def _is_x11_site(self, inst: CamoufoxInstance) -> bool:
        """Whether to use X11 input injection for this page.

        X11/xdotool injects real kernel-level InputEvents that the browser
        marks ``isTrusted=true``.  CDP-dispatched events always carry
        ``isTrusted=false``, which bot-detection systems (DataDome,
        Cloudflare, PerimeterX, ArkoseLabs) broadly check — not just on
        Twitter.  Using X11 input everywhere eliminates this signal.

        Falls back to CDP automatically on failure (see call sites).
        """
        return True

    async def click(
        self, agent_id: str, ref: str | None = None,
        selector: str | None = None, force: bool = False,
        snapshot_after: bool = False,
        timeout_ms: int | None = None,
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
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                raw_timeout = _CLICK_TIMEOUT_MS if timeout_ms is None else timeout_ms
                _timeout = max(1000, min(raw_timeout, 30000))
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
                            and ref_info.disabled
                            and ref_info.role in _ARIA_FORCE_ROLES
                            and not modal_unscoped):
                        use_force = True
                        logger.debug(
                            "Auto-force click on disabled %s ref=%s for '%s'",
                            ref_info.role, ref, agent_id,
                        )
                    locator = self._locator_from_ref(inst, ref)
                    if locator:
                        if inst.x11_wid and self._is_x11_site(inst):
                            try:
                                await self._x11_click(inst, locator, timeout=_timeout)
                            except Exception as e:
                                logger.warning(
                                    "X11 click failed for '%s', falling back to CDP: %s",
                                    agent_id, e,
                                )
                                await self._human_click(inst.page, locator, force=use_force, timeout=_timeout)
                        else:
                            await self._human_click(inst.page, locator, force=use_force, timeout=_timeout)
                    else:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                elif selector:
                    if inst.x11_wid and self._is_x11_site(inst):
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_click(inst, loc, timeout=_timeout)
                        except Exception as e:
                            logger.warning(
                                "X11 click failed for '%s' (selector), falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click_selector(inst.page, selector, force=force, timeout=_timeout)
                    else:
                        await self._human_click_selector(inst.page, selector, force=force, timeout=_timeout)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())

                # Fallback: if a close-type button was clicked inside a
                # modal but the modal persists, press Escape.  Camoufox's
                # patched Firefox silently drops pointer events on some
                # SPA modal close buttons (X/Twitter compose modal, etc.).
                if inst.dialog_active and ref and ref in inst.refs:
                    ri = inst.refs[ref]
                    nm = (ri.name or "").lower().strip()
                    is_close = ri.role == "button" and (
                        nm in _MODAL_CLOSE_NAMES
                        or nm.startswith("close")
                    )
                    if is_close:
                        await asyncio.sleep(0.3)
                        still_open = False
                        vp = inst.page.viewport_size
                        try:
                            modal_els = await inst.page.query_selector_all(
                                _MODAL_SELECTOR,
                            )
                            for el in modal_els:
                                if await self._is_visible_modal(el, vp):
                                    still_open = True
                                    break
                        except Exception:
                            pass
                        if still_open:
                            logger.info(
                                "Close-button click did not dismiss "
                                "modal for %s — sending Escape",
                                agent_id,
                            )
                            if inst.x11_wid and self._is_x11_site(inst):
                                try:
                                    await self._x11_key(inst, "Escape")
                                except Exception:
                                    await inst.page.keyboard.press("Escape")
                            else:
                                await inst.page.keyboard.press("Escape")
                            await asyncio.sleep(0.5)
                            # Escape may surface a confirmation dialog
                            # (e.g. "Discard draft?" on X/Twitter).
                            # Click through it to finish dismissing.
                            try:
                                confirm = inst.page.locator(
                                    _MODAL_SELECTOR,
                                ).get_by_role(
                                    "button", name="Discard",
                                )
                                if await confirm.count() > 0:
                                    if inst.x11_wid and self._is_x11_site(inst):
                                        try:
                                            await self._x11_click(
                                                inst, confirm.first,
                                            )
                                        except Exception:
                                            await self._human_click(
                                                inst.page, confirm.first,
                                                force=True,
                                            )
                                    else:
                                        await self._human_click(
                                            inst.page, confirm.first,
                                            force=True,
                                        )
                                    await asyncio.sleep(action_delay())
                                    logger.info(
                                        "Clicked Discard on confirmation"
                                        " dialog for %s", agent_id,
                                    )
                            except Exception:
                                pass

                inst.m_click_success += 1
                inst.click_window.append(True)
                # Recorder doesn't need the x11/cdp routing detail —
                # the click dispatch chooses internally and the timing
                # distribution is what §5.3/§9.5 consumes.
                inst.recorder.record_click(method="auto", success=True)
                result = {"success": True, "data": {"clicked": ref or selector}}
                if snapshot_after:
                    snap = await self._snapshot_impl(inst, agent_id)
                    result["snapshot"] = snap.get("data", {})
                return result
            except Exception as e:
                inst.m_click_fail += 1
                inst.click_window.append(False)
                inst.recorder.record_click(method="auto", success=False)
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
                    if inst.x11_wid and self._is_x11_site(inst):
                        try:
                            await self._x11_hover(inst, locator)
                        except Exception as e:
                            logger.warning(
                                "X11 hover failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await locator.hover(timeout=_CLICK_TIMEOUT_MS)
                    else:
                        await locator.hover(timeout=_CLICK_TIMEOUT_MS)
                elif selector:
                    if inst.x11_wid and self._is_x11_site(inst):
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_hover(inst, loc)
                        except Exception as e:
                            logger.warning(
                                "X11 hover failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await inst.page.hover(selector, timeout=_CLICK_TIMEOUT_MS)
                    else:
                        await inst.page.hover(selector, timeout=_CLICK_TIMEOUT_MS)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}
                await asyncio.sleep(action_delay())
                return {"success": True, "data": {"hovered": ref or selector}}
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def type_text(self, agent_id: str, ref: str | None = None, selector: str | None = None,
                        text: str = "", clear: bool = True,
                        fast: bool = False, snapshot_after: bool = False) -> dict:
        """Type text into element.

        fast=True uses minimal inter-key delays (8 ms) — still fires real
        keyDown/keyUp events for framework compatibility, but skips
        human-variance timing and think pauses.  Suitable for search
        queries, URLs, and non-sensitive form fields.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                # Click to focus, then optionally select-all to clear.
                # Never use fill() — it atomically sets the DOM value and bypasses
                # the keyboard event chain, so React/Vue apps (e.g. X's tweet
                # composer) don't see individual keystrokes and won't activate
                # submit buttons or update their controlled-component state.
                #
                # The entire interaction chain (focus click, select-all, typing)
                # uses X11 input so all events carry isTrusted=true. Mixed
                # CDP+X11 sequences create a detectable signal.
                _use_x11 = bool(inst.x11_wid) and self._is_x11_site(inst)

                if ref and ref in inst.refs:
                    locator = self._locator_from_ref(inst, ref)
                    if not locator:
                        return {"success": False, "error": f"Ref '{ref}' not found"}
                    if _use_x11:
                        try:
                            await self._x11_click(inst, locator)
                        except Exception as e:
                            logger.warning(
                                "X11 focus click failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click(inst.page, locator)
                    else:
                        await self._human_click(inst.page, locator)
                elif selector:
                    if _use_x11:
                        loc = inst.page.locator(selector).first
                        try:
                            await self._x11_click(inst, loc)
                        except Exception as e:
                            logger.warning(
                                "X11 focus click failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await self._human_click_selector(inst.page, selector)
                    else:
                        await self._human_click_selector(inst.page, selector)
                else:
                    return {"success": False, "error": "Must provide ref or selector"}

                # Settle after focus — SPA editors (Lexical, ProseMirror, Draft.js)
                # may expand or initialise event listeners on focus click.
                await asyncio.sleep(0.10 if fast else action_delay())

                if clear:
                    if _use_x11:
                        try:
                            await self._x11_key(inst, "ctrl+a")
                        except Exception as e:
                            logger.warning(
                                "X11 ctrl+a failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            await inst.page.keyboard.press("Control+a")
                    else:
                        await inst.page.keyboard.press("Control+a")
                    await asyncio.sleep(0.05)

                if _use_x11:
                    await self._x11_type(inst, text, typos=not fast)
                elif fast:
                    await self._type_fast(inst.page, text)
                else:
                    await self._type_with_variance(inst.page, text)

                # Settle after typing — framework state (React, Lexical, Vue)
                # batches DOM reconciliation asynchronously.
                await asyncio.sleep(0.10 if fast else action_delay())

                inst.recorder.record_keystrokes(
                    char_count=len(text),
                    fast=fast,
                    method="x11" if _use_x11 else ("cdp-fast" if fast else "cdp"),
                )
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

    async def screenshot(
        self,
        agent_id: str,
        full_page: bool = False,
        format: str = "webp",
        quality: int = 75,
        scale: float = 1.0,
    ) -> dict:
        """Take a screenshot and return it as base64.

        ``format`` controls the encoding:
        - ``"webp"`` (default) — lossy WebP at ``quality`` (1–100). Roughly
          5–10× smaller than PNG for the same visual content; the
          difference compounds heavily across multi-step browsing tasks
          where the agent may pull dozens of screenshots per task.
        - ``"png"`` — original lossless PNG path (Playwright native).
          Selected automatically if WebP encoding fails (e.g. Pillow
          missing in dev env, corrupt frame buffer) so callers always
          get a usable image.

        ``scale`` (0.5–1.0) rescales the captured image post-encode-prep,
        for further token savings when full-fidelity isn't needed. The
        Playwright native scale option is intentionally NOT used —
        Playwright applies it via ``deviceScaleFactor`` which mutates the
        viewport's pixel ratio and can leak fingerprint signal. Pillow
        downscale here is a pure post-process.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        # Validate inputs early — reject unknown formats with a clear
        # error rather than silently falling through to PNG. Operator
        # default comes from ``BROWSER_SCREENSHOT_FORMAT`` (per §2.1) so
        # an operator can globally force PNG without changing the caller.
        # ``.strip()`` guards against trailing whitespace from JSON UI
        # defaults; ``.lower()`` normalizes case.
        from src.browser.flags import get_str
        if not format:
            format = get_str(
                "BROWSER_SCREENSHOT_FORMAT", "webp", agent_id=agent_id,
            )
        fmt = format.strip().lower()
        if fmt not in ("webp", "png"):
            return {"success": False, "error": f"Unsupported screenshot format: {format!r}"}
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            quality = 75
        quality = max(1, min(100, quality))
        try:
            scale_f = float(scale)
        except (TypeError, ValueError):
            scale_f = 1.0
        scale_f = max(0.5, min(1.0, scale_f))

        async with inst.lock:
            try:
                # Ask Playwright for PNG either way — WebP encoding happens
                # post-capture so we can downscale and quality-tune in a
                # single Pillow pass without touching the page renderer.
                png_bytes = await inst.page.screenshot(full_page=full_page)
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Pillow encode is synchronous and ~10–20 ms on a 1080p frame —
        # offload to a thread so we don't block the event loop. Pillow
        # releases the GIL during its C-level encode steps, so this
        # actually parallelizes across concurrent agent screenshots.
        encoded, used_format = await asyncio.to_thread(
            _encode_screenshot,
            png_bytes, fmt, quality, scale_f, agent_id=agent_id,
        )
        b64 = base64.b64encode(encoded).decode()
        return {
            "success": True,
            "data": {
                "image_base64": b64,
                "format": used_format,
                "bytes": len(encoded),
            },
        }

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
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
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
                _use_x11 = bool(inst.x11_wid) and self._is_x11_site(inst)

                if _use_x11:
                    # X11 scroll: each button 4/5 click ≈ 3 lines ≈ 53 px.
                    # Produces real WheelEvent with deltaMode=DOM_DELTA_LINE,
                    # matching physical mouse hardware.
                    _PX_PER_NOTCH = 53
                    button = "4" if direction == "up" else "5"
                    total_notches = max(1, round(amount / _PX_PER_NOTCH))
                    scrolled = 0
                    for i in range(total_notches):
                        progress = i / max(1, total_notches)
                        ramp = scroll_ramp(progress)
                        try:
                            await self._x11_scroll_notch(inst, button)
                        except Exception as e:
                            logger.warning(
                                "X11 scroll failed for '%s', falling back to CDP: %s",
                                agent_id, e,
                            )
                            # Fall back to CDP for actual remaining distance
                            remaining_px = max(0, amount - scrolled)
                            await inst.page.mouse.wheel(0, remaining_px * sign)
                            scrolled += remaining_px
                            break
                        scrolled += _PX_PER_NOTCH
                        if i < total_notches - 1:
                            await asyncio.sleep(scroll_pause() / max(0.5, ramp))
                    scrolled = min(scrolled, amount)
                else:
                    # CDP fallback when X11 unavailable
                    scrolled = 0
                    while scrolled < amount:
                        remaining = amount - scrolled
                        progress = scrolled / amount if amount > 0 else 1.0
                        ramp = scroll_ramp(progress)
                        step = max(40, int(scroll_increment() * ramp))
                        step = min(step, remaining)
                        delta = step * sign
                        await inst.page.mouse.wheel(0, delta)
                        scrolled += step
                        if scrolled < amount:
                            await asyncio.sleep(scroll_pause() / max(0.5, ramp))

                inst.recorder.record_scroll(
                    direction=direction,
                    delta=scrolled,
                    method="x11" if _use_x11 else "cdp",
                )
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

    async def _check_captcha(self, inst: CamoufoxInstance) -> dict | None:
        """Check for CAPTCHA elements and attempt auto-solve if configured.

        Returns a dict with captcha details if found and unsolved, None if
        no CAPTCHA or if it was solved automatically.
        """
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            'iframe[src*="challenges.cloudflare.com"]',
            'iframe[src*="captcha"]',
            '[class*="cf-turnstile"]',
            '[class*="captcha"]',
            '#captcha',
        ]
        try:
            for sel in captcha_selectors:
                if await inst.page.locator(sel).count() > 0:
                    # Attempt auto-solve if a solver is configured
                    if self._captcha_solver:
                        logger.info("CAPTCHA detected (%s), attempting auto-solve", sel)
                        solved = await self._captcha_solver.solve(
                            inst.page, sel, inst.page.url,
                        )
                        if solved:
                            return None  # solved — don't report to agent
                        logger.warning("Auto-solve failed, falling back to manual")

                    return {
                        "type": sel,
                        "message": (
                            "CAPTCHA detected — you cannot bypass this. "
                            "Use notify_user to ask the user for help, "
                            "then wait before retrying."
                        ),
                    }
        except Exception:
            pass
        return None

    async def detect_captcha(self, agent_id: str) -> dict:
        """Detect CAPTCHAs on the current page."""
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            captcha = await self._check_captcha(inst)
            if captcha:
                return {"success": True, "data": {"captcha_found": True, **captcha}}
            return {"success": True, "data": {"captcha_found": False, "message": "No CAPTCHA detected"}}

    # ── File transfer (Phase 1.5 infrastructure) ─────────────────────────

    async def upload_file(
        self, agent_id: str, ref: str, local_paths: list[str],
        *, timeout_ms: int = 10000,
    ) -> dict:
        """Drive a native file-chooser via Playwright on behalf of the agent.

        The ``local_paths`` list points at files inside the browser container
        that the mesh staged for us. Caller is responsible for writing those
        bytes to disk BEFORE invoking this method; we just pass them to
        ``page.expect_file_chooser`` → ``chooser.set_files``.

        Playwright's ``expect_file_chooser`` is a context manager that
        resolves when the page triggers a chooser. The chooser fires in
        response to a click on an ``<input type="file">`` (or equivalent
        aria-labelled element); we handle that click here as part of the
        contract so agents don't need to coordinate the race.

        Returns ``{success, data: {uploaded: [path, …]}}`` or an error envelope.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused",
                    }
                # Validate every local path before opening the chooser —
                # Playwright's ``set_files`` raises a cryptic error if a
                # path is missing, and the chooser-open side-effect has
                # already happened by then.  Fail fast, keep the error
                # message actionable.
                for p in local_paths:
                    if not Path(p).is_file():
                        return {
                            "success": False,
                            "error": f"Upload path not found: {p}",
                        }
                locator = self._locator_from_ref(inst, ref)
                if not locator:
                    return {"success": False, "error": f"Ref '{ref}' not found"}
                # Race: the click that triggers the chooser must happen
                # INSIDE the ``expect_file_chooser`` context, otherwise we
                # may miss the event. Playwright's pattern is exactly this.
                async with inst.page.expect_file_chooser(timeout=timeout_ms) as info:
                    await locator.click(timeout=timeout_ms)
                chooser = await info.value
                await chooser.set_files(local_paths)
                await asyncio.sleep(action_delay())
                return {
                    "success": True,
                    "data": {"uploaded": list(local_paths)},
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def download(
        self, agent_id: str, ref: str,
        *,
        download_dir: str = "/tmp/downloads",
        timeout_ms: int = 30000,
        max_bytes: int = 50 * 1024 * 1024,
    ) -> dict:
        """Click ``ref`` and capture the resulting download to disk.

        Uses Playwright's ``page.expect_download`` context. On download-start,
        the file streams to ``download_dir/{nonce}-{suggested_filename}``
        with a running byte counter that aborts if ``max_bytes`` is exceeded.

        Returns ``{success, data: {path, size_bytes, suggested_filename, mime_type}}``.
        The caller (mesh proxy) is responsible for streaming the file from
        ``path`` to the agent's ``/artifacts/ingest`` endpoint and deleting
        the local copy afterwards.
        """
        inst = await self.get_or_start(agent_id)
        inst.touch()
        async with inst.lock:
            try:
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused",
                    }
                locator = self._locator_from_ref(inst, ref)
                if not locator:
                    return {"success": False, "error": f"Ref '{ref}' not found"}

                Path(download_dir).mkdir(parents=True, exist_ok=True)
                async with inst.page.expect_download(timeout=timeout_ms) as info:
                    await locator.click(timeout=timeout_ms)
                download = await info.value
                suggested = download.suggested_filename or "download.bin"
                nonce = uuid.uuid4().hex[:12]
                dest = Path(download_dir) / f"{nonce}-{suggested}"
                await download.save_as(str(dest))

                # Post-transfer size enforcement. Content-Length is a hint;
                # streaming enforcement is the authoritative check.
                size = dest.stat().st_size
                if size > max_bytes:
                    dest.unlink(missing_ok=True)
                    return {
                        "success": False,
                        "error": f"Download exceeds {max_bytes} bytes ({size})",
                    }

                mime = mimetypes.guess_type(suggested)[0] or "application/octet-stream"
                return {
                    "success": True,
                    "data": {
                        "path": str(dest),
                        "size_bytes": size,
                        "suggested_filename": suggested,
                        "mime_type": mime,
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
                if inst._user_control:
                    return {
                        "success": False,
                        "error": "User has browser control — action paused until control is released.",
                    }
                if inst.x11_wid and self._is_x11_site(inst):
                    xkey = self._playwright_key_to_xdotool(key)
                    try:
                        await self._x11_key(inst, xkey)
                    except Exception as e:
                        logger.warning(
                            "X11 press_key failed for '%s', falling back to CDP: %s",
                            agent_id, e,
                        )
                        await inst.page.keyboard.press(key)
                else:
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
