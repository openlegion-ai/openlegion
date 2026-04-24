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
same `BrowserManager` as production agents, meaning it inherits every
stealth tweak in `build_launch_options`, the current profile schema
version (§4.4 migrate_profile), per-agent fonts, resolution, referrer
pool — everything. The only thing isolated is the profile directory:
the canary must not share cookies / localStorage with any production
agent. That's achieved by giving it a dedicated agent-id
(:data:`CANARY_AGENT_ID`), so its profile lives under its own
``/data/profiles/<id>`` path and its cookie jar is its own.

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

    for scanner in _SCANNERS:
        entry = await _run_single_scanner(manager, scanner, report_dir, now)
        report["scanners"].append(entry)

    # Stop canary after the sweep to release the profile lock and free
    # its browser slot. A future run will respawn cleanly.
    with contextlib.suppress(Exception):
        await manager.stop(CANARY_AGENT_ID)

    # Heuristic overall score: average of scanner scores that produced a
    # numeric value. Callers should treat this as a smoke-test signal,
    # not a certification — operators read per-scanner details.
    numeric_scores = [
        s["score"] for s in report["scanners"]
        if isinstance(s.get("score"), (int, float))
    ]
    report["overall_score"] = (
        sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
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
    """Navigate + screenshot one scanner; never raises."""
    name = scanner["name"]
    url = scanner["url"]
    entry: dict = {"name": name, "url": url, "status": "unknown"}
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
        if not nav.get("success"):
            entry["status"] = "nav_failed"
            entry["error"] = nav.get("error", "")[:200]
            return entry
        entry["status"] = "ok"

        shot = await manager.screenshot(CANARY_AGENT_ID, full_page=True)
        if shot.get("success"):
            try:
                report_dir.mkdir(parents=True, exist_ok=True)
                path = report_dir / f"{int(ts)}-{name}.png"
                # Decode base64 back to bytes for disk.
                import base64
                path.write_bytes(base64.b64decode(
                    shot["data"]["image_base64"],
                ))
                entry["screenshot_path"] = str(path)
            except OSError as e:
                logger.warning("Canary screenshot save failed: %s", e)

        # Per-scanner score parsing is best-effort; we don't throw on
        # parse failures. When we can't extract a numeric score, leave
        # it as None so the overall average skips this entry.
        entry["score"] = await _parse_scanner_score(manager, name)
    except asyncio.TimeoutError:
        entry["status"] = "timeout"
    except Exception as e:  # noqa: BLE001 — best-effort per-scanner
        entry["status"] = "error"
        entry["error"] = str(e)[:200]
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
