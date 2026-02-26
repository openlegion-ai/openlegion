"""Browser automation via Playwright / Camoufox.

Provides browser access for web scraping, testing, and interaction.
A single browser instance is lazily initialized per agent process and reused.
Supports four backends via BROWSER_BACKEND env var:
basic, stealth, advanced, persistent.

The persistent backend uses Playwright Chromium with stealth flags and a
JS init script for anti-detection.  The stealth backend uses Camoufox
(patched Firefox).
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

_pw = None  # Playwright/Patchright instance (needs explicit stop on cleanup)
_camoufox_cm = None  # Camoufox context manager (needs __aexit__ on cleanup)
_browser = None
_context = None
_page = None
_launch_lock = asyncio.Lock()
_page_refs: dict[str, object] = {}
_credential_filled_refs: set[str] = set()  # refs that had $CRED{} typed into them
_page_op_lock = asyncio.Lock()  # serializes all page operations (Playwright pages aren't concurrent-safe)
_vnc_proc = None  # Xvnc (KasmVNC) subprocess (persistent backend)
_chrome_proc = None  # Chrome subprocess (persistent backend, launched without Playwright)

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

        backend = os.environ.get("BROWSER_BACKEND", "persistent")

        if backend == "stealth":
            _browser, _context, _page = await _launch_stealth()
        elif backend == "advanced":
            _browser, _context, _page = await _launch_advanced(mesh_client)
        elif backend == "persistent":
            _browser, _context, _page = await _launch_persistent()
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


def _ensure_xvfb():
    """Start Xvfb virtual display if not already running.

    Camoufox's built-in ``headless="virtual"`` hangs in Docker containers.
    Starting Xvfb ourselves and setting DISPLAY is the proven approach.
    """
    if os.environ.get("DISPLAY"):
        return
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x720x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
        logger.info("Started Xvfb virtual display on :99")
    except FileNotFoundError:
        logger.warning("Xvfb not installed — falling back to headless mode")


async def _launch_stealth():
    """Launch Camoufox anti-detect browser.

    Camoufox handles its own user-agent and fingerprint rotation,
    so we intentionally skip the custom user_agent set in _launch_basic().
    Uses Xvfb virtual display for proper rendering and anti-detection.
    """
    global _camoufox_cm
    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        raise RuntimeError(
            "camoufox is not installed. The agent container must include "
            "camoufox. See Dockerfile.agent."
        )
    _ensure_xvfb()
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
    try:
        cdp_url = await mesh_client.vault_resolve("brightdata_cdp_url")
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve 'brightdata_cdp_url' from vault: {e}. "
            "Check agent permissions in config/permissions.json."
        ) from e
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
    # Use Bright Data's default context — it includes managed fingerprinting,
    # CAPTCHA solving, and anti-detection.  Creating a new context would strip
    # those protections away.
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else await context.new_page()
    logger.info("Browser backend: advanced (Bright Data CDP)")
    return browser, context, page


_STEALTH_INIT_SCRIPT = """
// Hide navigator.webdriver — backup for --disable-blink-features flag.
Object.defineProperty(navigator, 'webdriver', {
    get: () => undefined, configurable: true,
});

// Fake chrome.runtime (real Chrome has this, Playwright doesn't)
if (!window.chrome) window.chrome = {};
if (!window.chrome.runtime) {
    window.chrome.runtime = {
        connect: () => {},
        sendMessage: () => {},
        onMessage: {addListener: () => {}, removeListener: () => {}},
        onConnect: {addListener: () => {}, removeListener: () => {}},
    };
}

// Fake plugins array (automation Chromium reports empty)
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const p = [
            {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer',
             description: 'Portable Document Format', length: 1},
            {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
             description: '', length: 1},
            {name: 'Native Client', filename: 'internal-nacl-plugin',
             description: '', length: 2},
        ];
        p.refresh = () => {};
        return p;
    },
});

// Fix permissions.query
const origQuery = window.navigator.permissions.query.bind(
    window.navigator.permissions
);
window.navigator.permissions.query = (params) => {
    if (params.name === 'notifications')
        return Promise.resolve({state: Notification.permission});
    return origQuery(params);
};

// Clean up automation globals
delete window.__playwright;
delete window.__pw_manual;
delete window.__pwInitScripts;
for (const key of Object.keys(window)) {
    if (key.startsWith('cdc_') || key.startsWith('__pw')) delete window[key];
}

// Fix navigator.languages
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

// Fake connection.rtt (0 in automation, ~50 in real browsers)
if (navigator.connection) {
    Object.defineProperty(navigator.connection, 'rtt', {get: () => 50});
}

// Spoof navigator.userActivation (X checks hasBeenActive)
if (navigator.userActivation) {
    Object.defineProperty(navigator.userActivation, 'hasBeenActive', {
        get: () => true,
    });
    Object.defineProperty(navigator.userActivation, 'isActive', {
        get: () => false,
    });
}
"""


def _cleanup_stale_profile():
    """Kill orphaned browser processes and remove stale lock files.

    After a crash, Chrome leaves SingletonLock/SingletonSocket/SingletonCookie
    and Firefox leaves lock/.parentlock in the profile directory.  A new browser
    instance refuses to start if these exist.
    This helper cleans up before each launch.
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

    # Remove stale lock files (Chrome + Firefox)
    lock_names = (
        "SingletonLock", "SingletonSocket", "SingletonCookie",  # Chrome
        "lock", ".parentlock",  # Firefox
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
            # Disable Chrome's built-in async DNS client — it bypasses Docker
            # Desktop's DNS forwarder, causing 10-30s timeouts on every request.
            # Belt-and-suspenders with the enterprise policy in the Dockerfile.
            "--disable-features=AsyncDns",
            "--no-first-run",
            "--no-default-browser-check",
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
    ``start_persistent_browser()``.  This function connects Playwright
    on-demand when the agent needs programmatic browser control.
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

    # Override Playwright's Target.setAutoAttach — by default Playwright sets
    # waitForDebuggerOnStart:true which freezes every new target (including
    # popup windows from window.open()).  Reddit's login flow opens a popup
    # that hangs indefinitely because it's waiting for the debugger to resume.
    try:
        cdp = await context.new_cdp_session(page)
        await cdp.send("Target.setAutoAttach", {
            "autoAttach": True,
            "waitForDebuggerOnStart": False,
            "flatten": True,
        })
        await cdp.detach()
    except Exception as e:
        logger.debug("Could not override Target.setAutoAttach: %s", e)

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
    # Relaunch Chrome clean (no Playwright) for VNC — only in persistent mode
    if os.environ.get("BROWSER_BACKEND") == "persistent":
        try:
            _launch_chrome_subprocess()
            await asyncio.sleep(1)
        except RuntimeError:
            logger.debug("Could not relaunch Chrome subprocess (not in container?)")


async def start_persistent_browser():
    """Start persistent browser + KasmVNC at container boot.

    Called from ``__main__.py`` lifespan when ``BROWSER_BACKEND=persistent``.
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

    # Start a lightweight window manager so popup windows (e.g. Reddit login)
    # can be stacked, focused, and managed properly in the VNC session.
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


async def browser_cleanup():
    """Release browser resources. Called on agent shutdown."""
    global _pw, _camoufox_cm, _browser, _context, _page, _vnc_proc, _chrome_proc
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
    _pw = _camoufox_cm = _browser = _context = _page = None
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
        "Use this when the browser is stuck, showing tunnel errors, or has hit "
        "navigation limits. After reset, the next browser_navigate call will "
        "establish a new connection (and a new IP for Bright Data)."
    ),
    parameters={},
)
async def browser_reset(*, mesh_client=None) -> dict:
    """Force-close the browser session so the next call gets a fresh one."""
    async with _page_op_lock:
        backend = os.environ.get("BROWSER_BACKEND", "persistent")
        if backend == "persistent":
            await _browser_cleanup_soft()
            logger.info("Browser session reset (backend: persistent, profile preserved)")
            return {
                "status": "reset",
                "backend": backend,
                "message": (
                    "Browser session closed. Profile and VNC are preserved. "
                    "Next browser call will reopen with existing cookies/sessions."
                ),
            }
        await browser_cleanup()
        logger.info("Browser session reset (backend: %s)", backend)
        return {
            "status": "reset",
            "backend": backend,
            "message": (
                "Browser session closed. Next browser call will establish "
                "a fresh connection."
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

    If the CDP session is dead (navigation limit, tunnel failure), automatically
    resets the browser and retries once with a fresh session.
    """
    async with _page_op_lock:
        backend = os.environ.get("BROWSER_BACKEND", "persistent")
        for attempt in range(2):
            try:
                _page_refs.clear()
                _credential_filled_refs.clear()
                page = await _get_page(mesh_client=mesh_client)
                # Bright Data premium domains can take up to 2 min for
                # CAPTCHA solving and proxy rotation; basic/stealth are fast.
                nav_timeout = 120_000 if backend == "advanced" else 30_000
                response = await page.goto(url, wait_until="domcontentloaded", timeout=nav_timeout)
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
                error_msg = str(e)
                if attempt == 0 and _is_dead_session_error(error_msg):
                    logger.warning(
                        "Dead CDP session detected (%s), resetting browser and retrying",
                        error_msg[:120],
                    )
                    if backend == "persistent":
                        await _browser_cleanup_soft()
                    else:
                        await browser_cleanup()
                    continue
                return {"error": error_msg, "url": url}
        return {"error": "Navigation failed after session reset", "url": url}


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
