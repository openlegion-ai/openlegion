"""Browser automation via Playwright CDP.

Every agent gets Chrome + KasmVNC by default.  Chrome runs as a subprocess
on the X display for VNC interaction; Playwright connects on-demand via CDP
for programmatic control, then disconnects after each operation so Chrome
runs clean for manual use.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.browser")

_pw = None  # Playwright instance (needs explicit stop on cleanup)
_browser = None
_context = None
_page = None
_launch_lock = asyncio.Lock()
_page_refs: dict[str, object] = {}
_credential_filled_refs: set[str] = set()  # refs that had $CRED{} typed into them
_page_op_lock = asyncio.Lock()  # serializes all page operations (Playwright pages aren't concurrent-safe)
_vnc_proc = None  # Xvnc (KasmVNC) subprocess
_chrome_proc = None  # Chrome subprocess (launched without Playwright)

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
    """Connect Playwright to the running Chrome via CDP on demand."""
    global _browser, _context, _page
    async with _launch_lock:
        if _page and not _page.is_closed():
            return _page

        _browser, _context, _page = await _launch_persistent()

        return _page


def _cleanup_stale_profile():
    """Kill orphaned browser processes and remove stale lock files.

    After a crash, Chrome leaves SingletonLock/SingletonSocket/SingletonCookie
    in the profile directory.  A new browser instance refuses to start if these
    exist.  This helper cleans up before each launch.
    """
    profile_dir = Path("/data/browser_profile")
    if not profile_dir.exists():
        return

    # Kill any orphaned browser processes using this profile
    try:
        subprocess.run(
            ["pkill", "-f", "browser_profile"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass  # pkill may not exist or no matching processes

    # Remove stale Chrome lock files
    lock_names = (
        "SingletonLock", "SingletonSocket", "SingletonCookie",
    )
    for name in lock_names:
        lock_path = profile_dir / name
        if lock_path.exists() or lock_path.is_symlink():
            try:
                lock_path.unlink()
                logger.info("Removed stale lock file: %s", lock_path)
            except OSError as e:
                logger.debug("Failed to remove %s: %s", lock_path, e)


def _find_chromium_binary() -> str:
    """Find Chrome/Chromium binary, preferring Google Chrome for TLS authenticity."""
    import glob as globmod

    # Prefer Google Chrome — authentic TLS fingerprint that matches UA string.
    # Chrome for Testing (Playwright's bundled Chromium) has a different JA3/JA4
    # fingerprint that anti-bot systems detect and block.
    for name in ["google-chrome-stable", "google-chrome"]:
        try:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    # Fallback to Playwright's bundled Chromium
    for pattern in [
        "/opt/pw-browsers/chromium-*/chrome-linux64/chrome",
        "/opt/pw-browsers/chromium-*/chrome-linux/chrome",
        "/opt/pw-browsers/chrome-*/chrome-linux64/chrome",
        "/opt/pw-browsers/chrome-*/chrome-linux/chrome",
    ]:
        matches = sorted(globmod.glob(pattern))
        if matches:
            return matches[-1]
    for name in ["chromium-browser", "chromium"]:
        try:
            result = subprocess.run(
                ["which", name], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
    raise RuntimeError("No Chrome/Chromium binary found in container")


def _launch_chrome_subprocess():
    """Launch Chrome directly as a subprocess — no Playwright, no CDP.

    The browser runs clean on the X display for VNC interaction.  Playwright
    connects later via ``--remote-debugging-port`` only when the agent needs
    programmatic access (screenshots, navigation, etc.).

    This separation is critical: Playwright's CDP instrumentation
    (Runtime.enable, Target.setAutoAttach with waitForDebuggerOnStart)
    causes popup freezes and anti-bot detection on sites like Reddit.
    """
    global _chrome_proc

    chrome_bin = _find_chromium_binary()
    profile_dir = "/data/browser_profile"
    Path(profile_dir).mkdir(parents=True, exist_ok=True)

    _chrome_proc = subprocess.Popen(
        [
            chrome_bin,
            f"--user-data-dir={profile_dir}",
            # Required for Docker — Chrome's sandbox needs kernel capabilities
            # that aren't available in containers with no-new-privileges.
            "--no-sandbox",
            # Use /tmp instead of /dev/shm for shared memory.  Docker's /dev/shm
            # can fill up under heavy JS; when it does, renderers crash silently
            # (buttons stop responding, "Aw Snap!" pages).
            "--disable-dev-shm-usage",
            # NOTE: do NOT add --disable-gpu — reCAPTCHA Enterprise uses WebGL
            # and Canvas fingerprinting.  Disabling GPU is a bot signal that
            # causes invisible CAPTCHA to silently block form submissions.
            # Disable Chrome's built-in async DNS client — it bypasses Docker
            # Desktop's DNS forwarder, causing 10-30s timeouts on every request.
            # Belt-and-suspenders with the enterprise policy in the Dockerfile.
            "--disable-features=AsyncDns",
            "--no-first-run",
            "--no-default-browser-check",
            # No extensions or sync needed in the container.
            "--disable-extensions",
            "--disable-sync",
            "--disable-translate",
            # NOTE: do NOT add --disable-background-networking — reCAPTCHA
            # Enterprise needs background network calls to Google for scoring.
            # Limit renderer processes — one active tab + one spare.
            "--renderer-process-limit=2",
            # --test-type suppresses the "unsupported command-line flag" infobar.
            # --disable-infobars was deprecated and ironically triggers its own
            # warning banner in newer Chrome, which shifts page content down.
            "--test-type",
            "--disable-blink-features=AutomationControlled",
            "--disable-popup-blocking",
            "--window-size=1280,720",
            "--window-position=0,0",
            "--remote-debugging-port=9222",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    logger.info("Launched Chrome (PID %d) for VNC — no CDP attached", _chrome_proc.pid)


async def _launch_persistent():
    """Connect Playwright to the already-running Chrome via CDP.

    Chrome was launched directly by ``_launch_chrome_subprocess()`` during
    ``start_browser()``.  This function connects Playwright on-demand
    when the agent needs programmatic browser control.
    """
    global _pw
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError(
            "playwright is not installed. The agent container must include "
            "playwright and chromium. See Dockerfile.agent."
        )
    _pw = await async_playwright().start()
    browser = await _pw.chromium.connect_over_cdp(
        "http://localhost:9222", timeout=10000,
    )
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else await context.new_page()
    logger.info("Playwright connected to Chrome via CDP (agent control)")
    return browser, context, page


async def _browser_cleanup_soft():
    """Close browser/context/page but leave Xvnc alive.

    Used by ``browser_reset`` in persistent mode to restart the browser
    while keeping the VNC session and profile directory intact.
    """
    global _pw, _browser, _context, _page, _chrome_proc
    # Disconnect Playwright CDP session
    try:
        if _page and not _page.is_closed():
            await _page.close()
    except Exception as e:
        logger.debug("Error closing browser page (soft): %s", e)
    try:
        if _context:
            await _context.close()
    except Exception as e:
        logger.debug("Error closing browser context (soft): %s", e)
    try:
        if _browser:
            await _browser.close()
    except Exception as e:
        logger.debug("Error closing browser (soft): %s", e)
    try:
        if _pw:
            await _pw.stop()
    except Exception as e:
        logger.debug("Error stopping playwright (soft): %s", e)
    # Kill Chrome subprocess
    if _chrome_proc:
        try:
            _chrome_proc.terminate()
            _chrome_proc.wait(timeout=5)
        except Exception as e:
            logger.debug("Error stopping Chrome subprocess: %s", e)
        _chrome_proc = None
    _pw = _browser = _context = _page = None
    _page_refs.clear()
    _credential_filled_refs.clear()
    _cleanup_stale_profile()
    # Relaunch Chrome clean (no Playwright) for VNC
    try:
        _launch_chrome_subprocess()
        await asyncio.sleep(1)
    except (RuntimeError, OSError):
        logger.debug("Could not relaunch Chrome subprocess (not in container?)")


async def _disconnect_cdp():
    """Disconnect Playwright CDP without killing Chrome.

    Removes all CDP instrumentation (Runtime.enable, Target.setAutoAttach,
    addScriptToEvaluateOnNewDocument) so Chrome runs clean on the X display
    for VNC interaction.  Called after each browser tool operation completes.

    On a ``connect_over_cdp`` connection, ``browser.close()`` just closes
    the WebSocket — it does NOT terminate the Chrome process.  The next
    ``_get_page()`` call will reconnect via ``connect_over_cdp`` on demand.

    This is critical for sites like Reddit whose login uses in-page modals
    that rely on JavaScript event handlers.  Playwright's CDP sets
    ``Target.setAutoAttach(waitForDebuggerOnStart=true)`` which can freeze
    new targets and interfere with normal JS execution.
    """
    global _pw, _browser, _context, _page
    try:
        if _browser:
            await _browser.close()
    except Exception as e:
        logger.debug("Error closing CDP connection: %s", e)
    try:
        if _pw:
            await _pw.stop()
    except Exception as e:
        logger.debug("Error stopping Playwright process: %s", e)
    _pw = _browser = _context = _page = None
    _page_refs.clear()
    _credential_filled_refs.clear()
    logger.debug("Disconnected Playwright CDP — Chrome running clean")


_VNC_INPUT_FIX_JS = r"""
(() => {
    if (window.__vnc_input_fix) return;
    window.__vnc_input_fix = true;
    const nativeSetter = Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype, 'value'
    ).set;
    const tracked = new Map();
    function scan() {
        document.querySelectorAll('*').forEach(el => {
            if (el.shadowRoot) {
                el.shadowRoot.querySelectorAll('input, textarea').forEach(inp => {
                    if (!tracked.has(inp)) tracked.set(inp, inp.value);
                });
            }
        });
    }
    setInterval(() => {
        scan();
        tracked.forEach((prev, inp) => {
            if (inp.value !== prev) {
                tracked.set(inp, inp.value);
                nativeSetter.call(inp, inp.value);
                inp.dispatchEvent(new InputEvent('input', {
                    bubbles: true, composed: true,
                    inputType: 'insertText', data: inp.value,
                }));
                inp.dispatchEvent(new Event('change', {bubbles: true, composed: true}));
            }
        });
    }, 300);
})();
"""


async def _inject_vnc_input_fix():
    """Inject a script into Chrome that fixes Shadow DOM input handling for VNC.

    Modern sites (Reddit, etc.) use web components with Shadow DOM inputs.
    VNC keyboard events update the inner <input> value visually but don't
    fire the framework events (InputEvent with composed:true) that the
    outer component needs to update its state.  This causes buttons to
    stay disabled even after the user types credentials.

    The fix: a small polling script that detects value changes in Shadow DOM
    inputs and dispatches proper InputEvent/change events.  Registered via
    CDP Page.addScriptToEvaluateOnNewDocument so it runs on every page load.
    The CDP connection is closed immediately after registration.
    """
    try:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await pw.chromium.connect_over_cdp("http://127.0.0.1:9222")
        cdp = await browser.contexts[0].new_cdp_session(browser.contexts[0].pages[0])
        # Register script for all future page loads
        await cdp.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": _VNC_INPUT_FIX_JS},
        )
        # Also run it on the current page
        await cdp.send("Runtime.evaluate", {"expression": _VNC_INPUT_FIX_JS})
        await browser.close()
        await pw.stop()
        logger.info("Injected VNC Shadow DOM input fix")
    except Exception as e:
        logger.debug("Could not inject VNC input fix (non-fatal): %s", e)


async def start_browser():
    """Start Chrome + KasmVNC at container boot.

    Called from ``__main__.py`` lifespan.
    KasmVNC's Xvnc is an X server, VNC server, and web server in a single
    process — no need for separate websockify or noVNC.  It serves its own
    modern web client with seamless clipboard, smooth scrolling, and webp
    compression.

    The web listen port defaults to 6080 but can be overridden via the
    ``VNC_PORT`` env var (used in host-network mode to avoid collisions
    when multiple persistent agents share the host network namespace).
    """
    global _vnc_proc

    listen_port = os.environ.get("VNC_PORT", "6080")

    # Start KasmVNC Xvnc — combined X server + VNC server + web server.
    # BasicAuth is disabled; access control is handled by Docker port mapping.
    _vnc_proc = subprocess.Popen(
        [
            "Xvnc", ":99",
            "-geometry", "1280x720",
            "-depth", "24",
            "-websocketPort", listen_port,
            "-httpd", "/usr/share/kasmvnc/www",
            "-sslOnly", "0",
            "-SecurityTypes", "None",
            "-disableBasicAuth",
            "-AlwaysShared",
            "-interface", "0.0.0.0",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    await asyncio.sleep(0.5)
    if _vnc_proc.poll() is not None:
        raise RuntimeError(
            f"Xvnc (KasmVNC) exited immediately (code {_vnc_proc.returncode})"
        )
    os.environ["DISPLAY"] = ":99"
    logger.info("Started KasmVNC Xvnc on :99, web on :%s", listen_port)

    # Start a lightweight window manager for proper window stacking in VNC.
    # Override the Dockerfile's openbox config to NOT maximize popup/dialog
    # windows — sites like Reddit open login modals that should stay at their
    # natural size, not be force-maximized over the main Chrome window.
    _openbox_config = Path("/home/agent/.config/openbox/rc.xml")
    try:
        _openbox_config.parent.mkdir(parents=True, exist_ok=True)
        _openbox_config.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<openbox_config xmlns="http://openbox.org/3.4/rc">\n'
            "  <desktops><number>1</number></desktops>\n"
            "  <applications>\n"
            '    <application class="*">\n'
            "      <decor>no</decor>\n"
            "      <maximized>true</maximized>\n"
            "    </application>\n"
            "    <!-- Don't maximize popup/dialog windows -->\n"
            '    <application role="pop-up">\n'
            "      <maximized>no</maximized>\n"
            "      <decor>yes</decor>\n"
            "    </application>\n"
            '    <application type="dialog">\n'
            "      <maximized>no</maximized>\n"
            "      <decor>yes</decor>\n"
            "    </application>\n"
            "  </applications>\n"
            "</openbox_config>\n"
        )
    except OSError as e:
        logger.debug("Could not write openbox config: %s", e)
    subprocess.Popen(
        ["openbox"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Launch Chrome directly — no Playwright, no CDP instrumentation.
    # The browser runs clean for VNC interaction.  Playwright connects
    # later (via _get_page → _launch_persistent → connect_over_cdp)
    # only when the agent needs programmatic control.
    _cleanup_stale_profile()
    _launch_chrome_subprocess()
    await asyncio.sleep(1)  # let Chrome initialize before agent actions
    await _inject_vnc_input_fix()


async def browser_cleanup():
    """Release browser resources. Called on agent shutdown."""
    global _pw, _browser, _context, _page, _vnc_proc, _chrome_proc
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
        if _pw:
            await _pw.stop()
    except Exception as e:
        logger.debug("Error stopping playwright: %s", e)
    if _chrome_proc:
        try:
            _chrome_proc.terminate()
            _chrome_proc.wait(timeout=5)
        except Exception as e:
            logger.debug("Error stopping Chrome: %s", e)
    if _vnc_proc:
        try:
            _vnc_proc.terminate()
            _vnc_proc.wait(timeout=5)
        except Exception as e:
            logger.debug("Error stopping Xvnc: %s", e)
    _pw = _browser = _context = _page = None
    _vnc_proc = _chrome_proc = None
    _page_refs.clear()
    _credential_filled_refs.clear()


# Error patterns that indicate a dead CDP session (navigation limit, tunnel drop).
# Keep these specific — broad patterns like "Protocol error" cause false positives
# on non-fatal errors (e.g. invalid URL) and mask the real issue from the agent.
_CDP_DEAD_SESSION_PATTERNS = (
    "Page.navigate limit reached",
    "ERR_TUNNEL_CONNECTION_FAILED",
    "ERR_NAME_NOT_RESOLVED",
    "Target closed",
    "Session closed",
    "Browser has been closed",
    "Connection refused",
    "Connection closed",
)


def _is_dead_session_error(error_msg: str) -> bool:
    """Check if an error indicates the CDP session is dead and needs reset."""
    return any(pattern.lower() in error_msg.lower() for pattern in _CDP_DEAD_SESSION_PATTERNS)


@skill(
    name="browser_reset",
    description=(
        "Reset the browser by closing the current session and starting fresh. "
        "Use this when the browser is stuck or showing errors. "
        "Profile and VNC are preserved — cookies and sessions survive."
    ),
    parameters={},
)
async def browser_reset(*, mesh_client=None) -> dict:
    """Force-close the browser session so the next call gets a fresh one."""
    async with _page_op_lock:
        await _browser_cleanup_soft()
        logger.info("Browser session reset (profile preserved)")
        return {
            "status": "reset",
            "message": (
                "Browser session closed. Profile and VNC are preserved. "
                "Next browser call will reopen with existing cookies/sessions."
            ),
        }


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
        "Navigate your browser to a URL and return the page text. "
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
    """Navigate to a URL and extract text content.

    If the CDP session is dead, automatically resets the browser and retries
    once with a fresh session.  Disconnects Playwright CDP after extracting
    content so Chrome runs clean for VNC interaction.
    """
    async with _page_op_lock:
        for attempt in range(2):
            try:
                _page_refs.clear()
                _credential_filled_refs.clear()
                page = await _get_page(mesh_client=mesh_client)
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                if wait_ms:
                    await page.wait_for_timeout(wait_ms)
                text = await page.inner_text("body")
                result = {
                    "url": page.url,
                    "title": await page.title(),
                    "status": response.status if response else 0,
                    "content": _redact_credentials(text[:50000]),
                }
                # Disconnect CDP so Chrome runs clean for VNC interaction.
                await _disconnect_cdp()
                return result
            except Exception as e:
                error_msg = str(e)
                if attempt == 0 and _is_dead_session_error(error_msg):
                    logger.warning(
                        "Dead CDP session detected (%s), resetting browser and retrying",
                        error_msg[:120],
                    )
                    await _browser_cleanup_soft()
                    continue
                await _disconnect_cdp()
                return {"error": error_msg, "url": url}


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

        # Playwright 1.49+: aria_snapshot() returns YAML text
        # Older Playwright: page.accessibility.snapshot() returns dict tree
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

            await _disconnect_cdp()
            return result
        except Exception as e:
            await _disconnect_cdp()
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
    """Click an element by ref or CSS selector.

    In persistent mode, disconnects CDP after clicking so any modals or
    popups triggered by the click can render without CDP interference.
    """
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
                result = {"clicked": ref, "url": page.url}
            else:
                await page.click(selector, timeout=10000)
                await page.wait_for_timeout(500)
                result = {"clicked": selector, "url": page.url}
            # Disconnect CDP so modals/popups triggered by the click
            # can render without Playwright's CDP interference.
            await _disconnect_cdp()
            return result
        except Exception as e:
            await _disconnect_cdp()
            return {"error": str(e), "selector": selector, "ref": ref}


@skill(
    name="browser_type",
    description=(
        "Type text into a form field on the current page. Clears the field first, "
        "then enters the new text. Preferred: use ref from browser_snapshot "
        "(e.g. ref='e5'). Fallback: use a CSS selector. "
        "Use $CRED{name} handles to type secrets (e.g. text='$CRED{twitter_password}') "
        "— the value is resolved from the vault and never exposed to you."
    ),
    parameters={
        "selector": {
            "type": "string",
            "description": "CSS selector of the input element (optional if ref is given)",
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
            await _disconnect_cdp()
            return result
        except Exception as e:
            await _disconnect_cdp()
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
            await _disconnect_cdp()
            return {"result": result}
        except Exception as e:
            await _disconnect_cdp()
            return {"error": str(e)}
