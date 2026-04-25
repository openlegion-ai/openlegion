"""Stealth canary runner (Phase 2 §5.4) — operator opt-in.

Drives a dedicated Camoufox profile against a small set of public
stealth-scanner sites so operators can detect fingerprint regressions
before customer traffic notices them.

**Scope (important):** canary scanners inspect *client-side JS*
fingerprints only. TLS, H2, and network-layer signals (JA3/JA4, TCP
fingerprinting, proxy IP reputation) are invisible to these sites; a
clean canary score does NOT guarantee clean traffic against a
behavioral-fraud provider. The canary is an early-warning layer, not a
certification.

**Why a dedicated profile.** The canary visits its targets through the
same `BrowserManager` as production agents, so it inherits every
launch-time stealth setting from `build_launch_options` (Camoufox
config, fonts, resolution, referrer pool), the current profile schema
version (§4.4 migrate_profile), and whatever the operator has active
in ``config/settings.json``. The isolation is at the profile
directory only: the canary must not share cookies / localStorage with
any production agent. That's achieved by giving it a dedicated
agent-id (:data:`CANARY_AGENT_ID`), so its profile lives under its own
``/data/profiles/<id>`` path.

**This is a floor estimate, not parity.** A fresh canary profile has
no browsing history, no storage estimate entropy, no IndexedDB
entries, and no service-worker registrations — so it will *score
slightly cleaner* than a 30-day-old production profile on scanners
that sample those signals. Treat a clean canary score as a necessary
condition for stealth, not a sufficient one. Real-world detection
also involves TLS/H2/proxy-IP signals that client-side scanners
cannot see (see §2.8).

**Default off.** ``BROWSER_CANARY_ENABLED=true`` is required in the
flags system; the HTTP endpoint returns 403 otherwise. We don't want
customer fleets hammering these scanners — some of them are run by
small teams on donated infrastructure, and a 1000-agent customer
generating one run per browser restart would be abuse.

**Rate limit.** One run per 23 hours by default, persisted to disk so
restarts don't reset the counter. Operators can force a run with
``force=True`` if they're actively debugging; the caller is expected
to have operator-level access at the mesh layer.

**Score.** Each scanner contributes a raw signals dict plus an optional
0-100 score when we can parse one. Sites we can't parse contribute
a ``"manual"`` marker — the operator visits the saved screenshot.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.browser.flags import get_bool
from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.browser.service import BrowserManager

logger = setup_logging("browser.canary")


# The canary uses a stable agent-id so its profile survives across
# runs (faster startup, better cookie / cache warmup realism). Chosen
# to match AGENT_ID_RE_PATTERN but be unlikely to collide with any
# real fleet template.
CANARY_AGENT_ID = "canary-probe"

# Serialize canary runs so two concurrent force=True callers can't
# corrupt the state file or race on the single canary profile.
# Lazy + loop-tracked so tests that construct fresh event loops
# (pytest-asyncio's function scope) don't hit "Future attached to a
# different loop". In production there's exactly one loop per process
# so the rebinding branch never fires.
_run_lock: asyncio.Lock | None = None
_run_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_run_lock() -> asyncio.Lock:
    global _run_lock, _run_lock_loop
    loop = asyncio.get_running_loop()
    if _run_lock is None or _run_lock_loop is not loop:
        _run_lock = asyncio.Lock()
        _run_lock_loop = loop
    return _run_lock

_SCANNERS: list[dict] = [
    {
        "name": "bot.sannysoft.com",
        "url": "https://bot.sannysoft.com/",
        "wait_until": "load",
        "wait_ms": 2000,
    },
    {
        "name": "creepjs",
        "url": "https://abrahamjuliot.github.io/creepjs/",
        "wait_until": "networkidle",
        "wait_ms": 8000,
    },
    {
        "name": "fingerprint.com",
        "url": "https://fingerprint.com/products/bot-detection/",
        "wait_until": "networkidle",
        "wait_ms": 4000,
    },
    {
        "name": "pixelscan.net",
        "url": "https://www.pixelscan.net/",
        "wait_until": "networkidle",
        "wait_ms": 6000,
    },
]

_DEFAULT_STATE_PATH = Path("/data/canary/state.json")
_DEFAULT_REPORT_DIR = Path("/data/canary/reports")
_MIN_RUN_INTERVAL_S = 23 * 3600  # one run per ~day


class CanaryDisabledError(RuntimeError):
    """Raised when the canary is called but BROWSER_CANARY_ENABLED is false."""


class CanaryRateLimitedError(RuntimeError):
    """Raised when a canary run is attempted too soon after the previous run."""

    def __init__(self, retry_after_s: int):
        super().__init__(f"canary rate-limited; retry in {retry_after_s}s")
        self.retry_after_s = retry_after_s


def canary_enabled() -> bool:
    """Single read point; see :mod:`src.browser.flags`."""
    return get_bool("BROWSER_CANARY_ENABLED", default=False)


def _read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)


async def run_canary(
    manager: BrowserManager,
    *,
    force: bool = False,
    state_path: Path | None = None,
    report_dir: Path | None = None,
) -> dict:
    """Run the canary scan and return a structured report.

    Raises :class:`CanaryDisabledError` when the feature flag is off,
    :class:`CanaryRateLimitedError` when called within
    ``_MIN_RUN_INTERVAL_S`` of the previous successful run (unless
    ``force=True``). Any scanner-specific failure is captured in the
    report rather than raised — a canary that crashes on one site
    should still surface signal from the others.

    Serialized via :data:`_RUN_LOCK` so two concurrent ``force=True``
    callers don't race on the canary profile, state file, or browser
    slot. The lock is acquired AFTER the flag+rate-limit checks so
    those still return quickly for callers that can't run right now.
    Canary teardown runs under ``try/finally`` so a cancelled HTTP
    request (client disconnect) still releases the profile lock.
    """
    if not canary_enabled():
        raise CanaryDisabledError(
            "BROWSER_CANARY_ENABLED is false — operator opt-in required",
        )

    state_path = state_path or _DEFAULT_STATE_PATH
    report_dir = report_dir or _DEFAULT_REPORT_DIR

    state = _read_state(state_path)
    last_run = float(state.get("last_run_ts", 0) or 0)
    now = time.time()
    if not force and now - last_run < _MIN_RUN_INTERVAL_S:
        raise CanaryRateLimitedError(
            retry_after_s=int(_MIN_RUN_INTERVAL_S - (now - last_run)),
        )

    async with _get_run_lock():
        # Re-check the rate-limit under the lock so a queue of waiters
        # doesn't all run sequentially after the holder finishes.
        state = _read_state(state_path)
        last_run = float(state.get("last_run_ts", 0) or 0)
        now = time.time()
        if not force and now - last_run < _MIN_RUN_INTERVAL_S:
            raise CanaryRateLimitedError(
                retry_after_s=int(_MIN_RUN_INTERVAL_S - (now - last_run)),
            )

        report: dict = {
            "ts": now,
            "boot_id": manager.boot_id,
            "agent_id": CANARY_AGENT_ID,
            "scanners": [],
        }

        # Stop any lingering canary instance so each run starts fresh on
        # stealth config (useful when operators iterate on §6.x knobs).
        with contextlib.suppress(Exception):
            await manager.stop(CANARY_AGENT_ID)

        try:
            for scanner in _SCANNERS:
                entry = await _run_single_scanner(
                    manager, scanner, report_dir, now,
                )
                report["scanners"].append(entry)
        finally:
            # Always attempt to stop the canary — a cancelled HTTP
            # request (client disconnect) mustn't leave the Camoufox
            # profile locked. ``asyncio.shield`` protects the stop
            # from our own CancelledError so the cleanup task runs
            # to completion even if the outer caller gives up; the
            # ``suppress(Exception)`` handles real errors without
            # touching CancelledError (which still propagates, as
            # it must, so the caller knows we were cancelled).
            with contextlib.suppress(Exception):
                await asyncio.shield(manager.stop(CANARY_AGENT_ID))

        # Heuristic overall score: average of scanner scores that produced
        # a numeric value. Callers should treat this as a smoke-test
        # signal, not a certification — operators read per-scanner detail.
        numeric_scores = [
            s["score"] for s in report["scanners"]
            if isinstance(s.get("score"), (int, float))
        ]
        report["overall_score"] = (
            sum(numeric_scores) / len(numeric_scores)
            if numeric_scores else None
        )

        state["last_run_ts"] = now
        state["last_overall_score"] = report["overall_score"]
        _write_state(state_path, state)
        return report


async def _run_single_scanner(
    manager: BrowserManager,
    scanner: dict,
    report_dir: Path,
    ts: float,
) -> dict:
    """Navigate + screenshot one scanner; never raises.

    Status progression: ``unknown`` → ``nav_failed`` / ``timeout`` /
    ``error`` (on navigate), or → ``ok`` (navigate succeeded). Parse
    and screenshot failures are tracked separately under
    ``parse_error`` / ``screenshot_error`` so a bad score-extractor
    doesn't mask a successful nav — operators debugging a specific
    scanner need to know nav DID succeed before they chase the
    wrong symptom.
    """
    name = scanner["name"]
    url = scanner["url"]
    entry: dict = {"name": name, "url": url, "status": "unknown"}

    # Navigate phase — the only phase that sets a non-ok status.
    try:
        nav = await asyncio.wait_for(
            manager.navigate(
                CANARY_AGENT_ID,
                url,
                wait_ms=scanner["wait_ms"],
                wait_until=scanner["wait_until"],
            ),
            timeout=45,
        )
    except asyncio.TimeoutError:
        entry["status"] = "timeout"
        return entry
    except Exception as e:  # noqa: BLE001 — best-effort per-scanner
        entry["status"] = "error"
        entry["error"] = str(e)[:200]
        return entry

    if not nav.get("success"):
        entry["status"] = "nav_failed"
        entry["error"] = (nav.get("error") or "")[:200]
        return entry
    entry["status"] = "ok"

    # Screenshot phase — failures here annotate the entry but don't
    # demote status. A page that navigated but can't screenshot is
    # still useful signal; operators read the status first.
    try:
        # Force PNG: the canary report is read by humans for pixel-level
        # comparisons across runs. WebP's lossy default would smear
        # detection artefacts (color rings, font renderings) operators
        # are watching for. Cost is fine — canary fires hourly.
        shot = await manager.screenshot(
            CANARY_AGENT_ID, full_page=True, format="png",
        )
        if shot.get("success"):
            report_dir.mkdir(parents=True, exist_ok=True)
            path = report_dir / f"{int(ts)}-{name}.png"
            path.write_bytes(base64.b64decode(shot["data"]["image_base64"]))
            entry["screenshot_path"] = str(path)
    except Exception as e:  # noqa: BLE001
        entry["screenshot_error"] = str(e)[:200]

    # Parse phase — score extraction is best-effort per-site. Parse
    # failures do NOT demote status from "ok" because the nav DID
    # succeed; mark them separately so a stale DOM selector doesn't
    # surface as a mystery "error" row.
    try:
        entry["score"] = await _parse_scanner_score(manager, name)
    except Exception as e:  # noqa: BLE001
        entry["score"] = None
        entry["parse_error"] = str(e)[:200]
    return entry


async def _parse_scanner_score(manager, name: str) -> float | None:
    """Try to pull a 0-100 score out of the page DOM.

    Scanner HTML is a moving target; this is best-effort. When a scanner
    redesigns, the operator sees a dropped-to-None score on the next run
    and we update the selector. Keeping the logic site-specific and
    defensive is the right trade-off — a single generic parser would be
    brittle in the other direction.
    """
    if name == "bot.sannysoft.com":
        # Sannysoft reports each test as .passed / .failed rows. Count
        # passes vs total as a rough score.
        snap = await manager.evaluate(
            CANARY_AGENT_ID,
            "() => ({"
            " pass: document.querySelectorAll('.passed').length,"
            " fail: document.querySelectorAll('.failed').length"
            "})",
        )
        if snap.get("success"):
            r = snap["data"].get("result") or {}
            p = int(r.get("pass", 0))
            f = int(r.get("fail", 0))
            if p + f > 0:
                return round(100.0 * p / (p + f), 1)
    # Other scanners: no reliable machine-readable score — defer to the
    # saved screenshot for manual review.
    return None
