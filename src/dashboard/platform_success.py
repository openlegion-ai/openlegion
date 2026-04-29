"""Per-platform success aggregation for the dashboard stealth/browser panel.

Subscribes to ``EventBus`` emits and maintains an in-memory rolling
window of per-host counters covering captcha outcomes, fingerprint burns,
navigations, and applied platform pre-nav delays.  Operators see
"which sites is my fleet succeeding on, and which ones flag / burn
fingerprints" without having to scrape logs or replay events.

State is intentionally process-local — a mesh restart resets the panel.
The browser-service container's :mod:`src.browser.captcha_cost_counter`
already persists tenant-level rollups; this module is best-effort
observability, not an audit trail.

All ingestion is O(1) per event:
  * One dict lookup + integer/float increment.
  * Old samples are pruned lazily on read (snapshot()) so a bursty
    write phase cannot starve the event loop.

The aggregator deliberately does NOT introduce a new SQLite table —
keeping the lifecycle process-local lets us add the panel without a
migration story.  An operator who wants persistence can request a
follow-up.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from src.shared.utils import setup_logging

logger = setup_logging("dashboard.platform_success")

# ── Configuration ────────────────────────────────────────────────────────

# Rolling-window length.  24h matches the operator's expected debugging
# horizon: most stealth-regression investigations look at "today vs
# yesterday".  Longer windows would inflate memory without surfacing any
# signal that's not also visible in the event stream.
WINDOW_SECONDS = 24 * 3600

# Cap per-host sample lists at this size — protects against a runaway
# loop on a single host hammering the aggregator.  At ~1 event/sec
# sustained, a host fills 24h with 86400 samples; 10000 keeps memory
# bounded while still preserving enough resolution for averages.
MAX_SAMPLES_PER_HOST = 10000

# How many platforms to surface in the snapshot response.  Enough to
# show the long tail (operator can paginate via the EventBus log if they
# need the rest) without blowing up the JSON payload.
DEFAULT_TOP_N = 50


# ── Helpers ──────────────────────────────────────────────────────────────


def canonical_host(value: str) -> str | None:
    """Lower-cased hostname with leading ``www.`` stripped, or ``None``.

    Mirrors :func:`src.browser.stealth._canonical_host` and
    :func:`src.browser.captcha_policy._hostname` so the dashboard side
    bins traffic the same way the browser-service side does.  ``value``
    may be a full URL (``https://www.linkedin.com/foo``) or a bare
    hostname (``linkedin.com:443``); both reduce to ``linkedin.com``.
    """
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    # Treat anything without a scheme as a bare host.  ``urlparse`` will
    # otherwise stuff the whole string into ``path`` and yield no
    # hostname at all.
    if "://" in value:
        try:
            parsed = urlparse(value)
        except Exception:
            return None
        host = (parsed.hostname or "").lower()
    else:
        # Strip a trailing ``:port`` if present.
        host = value.split("/", 1)[0].split(":", 1)[0].lower()
    if not host:
        return None
    if host.startswith("www."):
        host = host[4:]
    return host or None


# Multi-segment public suffixes ("effective TLDs"). When a host's last
# two labels match one of these, the label is the THIRD-from-last
# component instead of the second. Without this, ``bbc.co.uk`` would
# label as ``co`` rather than ``bbc``. Kept short and operator-curated
# rather than pulling in the full publicsuffix list — operators who
# need exhaustive coverage can add to this set; the wrong label is a
# cosmetic dashboard issue, not a correctness problem.
_MULTI_SEGMENT_TLDS: frozenset[str] = frozenset({
    "co.uk", "co.jp", "co.kr", "co.nz", "co.za",
    "com.au", "com.br", "com.cn", "com.hk", "com.mx", "com.sg", "com.tw",
    "ac.uk", "gov.uk", "org.uk", "ne.jp", "or.jp",
})


def _platform_label(host: str) -> str:
    """Compress a host into a short label for display.

    ``linkedin.com`` → ``linkedin``; ``api.example.com`` →
    ``example``; ``bbc.co.uk`` → ``bbc`` (multi-segment TLD aware).
    Falls back to the full host for single-segment inputs
    (``localhost``).
    """
    parts = host.split(".")
    if len(parts) < 2:
        return host
    # Multi-segment TLD: ``bbc.co.uk`` → split is
    # ["bbc", "co", "uk"], last two joined = "co.uk" → label is parts[-3].
    if len(parts) >= 3:
        suffix = ".".join(parts[-2:])
        if suffix in _MULTI_SEGMENT_TLDS:
            return parts[-3]
    return parts[-2]


# ── State ────────────────────────────────────────────────────────────────


@dataclass
class _HostState:
    """Rolling per-host counters for a single platform.

    Each ``deque`` entry is ``(timestamp, count_or_value)``.  Counts
    use 1 per event; ``pre_nav_delays`` stores the dwell duration so
    we can compute a true average later.
    """

    navigations: deque = field(default_factory=deque)
    captcha_solved: deque = field(default_factory=deque)
    captcha_failed: deque = field(default_factory=deque)
    captcha_other: deque = field(default_factory=deque)
    fingerprint_burns: deque = field(default_factory=deque)
    pre_nav_delays: deque = field(default_factory=deque)


class PlatformSuccessAggregator:
    """In-memory rolling-window per-platform counter.

    Thread-safe via a single ``threading.Lock``.  All accessors are
    O(1) per event; ``snapshot`` is O(N) over the active host count.
    """

    def __init__(self, *, window_s: float = WINDOW_SECONDS,
                 time_fn: Any = None) -> None:
        self._window_s = window_s
        # Inject a clock so tests can stub time without monkey-patching
        # ``time.time`` globally (the dashboard server runs lots of other
        # time-dependent code that we don't want to perturb).
        self._time_fn = time_fn or time.time
        self._lock = threading.Lock()
        self._hosts: dict[str, _HostState] = {}

    # ── Mutation ──────────────────────────────────────────────────────

    def record_navigation(self, host: str | None, *, ts: float | None = None) -> None:
        """Record a nav request to ``host``.  Pre-nav delays count too."""
        if not host:
            return
        ts = ts if ts is not None else self._time_fn()
        with self._lock:
            state = self._hosts.setdefault(host, _HostState())
            _push(state.navigations, ts, 1)

    def record_captcha(
        self,
        host: str | None,
        outcome: str,
        *,
        ts: float | None = None,
    ) -> None:
        """Record a captcha-gate outcome for ``host``.

        ``outcome`` is one of the strings emitted by
        :func:`src.browser.service._record_captcha_audit_event`:
        ``success`` / ``failed`` / ``cost_cap`` / ``rate_limited`` /
        ``skipped_behavioral`` / ``low_success_failed`` etc.  Anything
        not explicitly matched is bucketed into ``captcha_other`` so
        the operator still sees activity volume.
        """
        if not host:
            return
        ts = ts if ts is not None else self._time_fn()
        outcome = (outcome or "").lower()
        with self._lock:
            state = self._hosts.setdefault(host, _HostState())
            if outcome == "success":
                _push(state.captcha_solved, ts, 1)
            elif outcome in ("failed", "low_success_failed"):
                _push(state.captcha_failed, ts, 1)
            else:
                _push(state.captcha_other, ts, 1)

    def record_fingerprint_burn(
        self, host: str | None, *, ts: float | None = None,
    ) -> None:
        if not host:
            return
        ts = ts if ts is not None else self._time_fn()
        with self._lock:
            state = self._hosts.setdefault(host, _HostState())
            _push(state.fingerprint_burns, ts, 1)

    def record_pre_nav_delay(
        self, host: str | None, delay_s: float,
        *, ts: float | None = None,
    ) -> None:
        """Record ONE applied pre-nav dwell of ``delay_s`` seconds.

        The aggregator stores the dwell duration so the snapshot can
        report a true mean.  ``record_pre_nav_delay`` also implies a
        navigation (the dwell only fires immediately before a nav), so
        callers SHOULD NOT separately call :meth:`record_navigation`
        for the same event — would double-count.
        """
        if not host:
            return
        try:
            d = float(delay_s)
        except (ValueError, TypeError):
            return
        if d < 0:
            return
        ts = ts if ts is not None else self._time_fn()
        with self._lock:
            state = self._hosts.setdefault(host, _HostState())
            _push(state.pre_nav_delays, ts, d)
            _push(state.navigations, ts, 1)

    # ── Reads ─────────────────────────────────────────────────────────

    def snapshot(self, *, top_n: int = DEFAULT_TOP_N) -> dict:
        """Build the JSON-serializable per-platform success snapshot.

        Returns ``{since: ISO8601, platforms: [...]}`` ordered by
        ``navigations`` desc.  Empty hosts (every counter pruned out
        of the window) are dropped.
        """
        now = self._time_fn()
        cutoff = now - self._window_s
        platforms: list[dict] = []
        with self._lock:
            for host, state in self._hosts.items():
                _prune(state.navigations, cutoff)
                _prune(state.captcha_solved, cutoff)
                _prune(state.captcha_failed, cutoff)
                _prune(state.captcha_other, cutoff)
                _prune(state.fingerprint_burns, cutoff)
                _prune(state.pre_nav_delays, cutoff)

                navigations = _sum_count(state.navigations)
                solved = _sum_count(state.captcha_solved)
                failed = _sum_count(state.captcha_failed)
                other = _sum_count(state.captcha_other)
                burns = _sum_count(state.fingerprint_burns)
                delay_count = len(state.pre_nav_delays)
                delay_sum = _sum_value(state.pre_nav_delays)
                avg_delay = (delay_sum / delay_count) if delay_count else 0.0

                # Drop hosts with absolutely no in-window activity —
                # they linger as dict keys after a long-idle period.
                if (
                    navigations == 0 and solved == 0 and failed == 0
                    and other == 0 and burns == 0 and delay_count == 0
                ):
                    continue

                events = solved + failed + other
                # Success rate must reflect ATTEMPTED solves only —
                # ``other`` rolls up gate-skipped outcomes (cost_cap,
                # rate_limited, skipped_behavioral, provider_missing,
                # price_missing) that never reached the solver. Including
                # them in the denominator made a fleet that hit cost cap
                # 100 times and successfully solved 5 captchas show
                # ``5/105 = 4.7%`` — operator-misleading. The honest
                # success rate is solver attempts that returned a token
                # ÷ total attempts (excluding gate skips). Hosts with
                # zero attempts are reported as ``None`` so the frontend
                # can render "—" instead of a misleading 100% / 0%.
                attempted = solved + failed
                if attempted > 0:
                    success_rate: float | None = solved / attempted
                else:
                    success_rate = None

                platforms.append({
                    "host": host,
                    "label": _platform_label(host),
                    "navigations": navigations,
                    "captcha_events": events,
                    "captcha_attempted": attempted,
                    "captcha_solved": solved,
                    "captcha_failed": failed,
                    "captcha_other": other,
                    "fingerprint_burns": burns,
                    "pre_nav_delays_applied": delay_count,
                    "avg_pre_nav_delay_s": round(avg_delay, 3),
                    "success_rate": (
                        round(success_rate, 3)
                        if success_rate is not None else None
                    ),
                })

        platforms.sort(key=lambda p: p["navigations"], reverse=True)
        if top_n and top_n > 0:
            platforms = platforms[:top_n]

        from datetime import datetime, timezone
        since_ts = now - self._window_s
        since_iso = datetime.fromtimestamp(
            since_ts, tz=timezone.utc,
        ).isoformat().replace("+00:00", "Z")

        return {
            "since": since_iso,
            "window_seconds": int(self._window_s),
            "platforms": platforms,
        }

    # ── EventBus wiring ───────────────────────────────────────────────

    def handle_event(self, evt: dict) -> None:
        """Single entry point for the EventBus listener.

        Dispatches based on the event ``type`` and the ``data.type``
        sub-tag used by the browser-service metrics drainer.  Designed
        to be cheap — pulls a few dict keys, no parsing of large
        payloads.
        """
        try:
            event_type = evt.get("type", "")
            data = evt.get("data") or {}
            if event_type != "browser_metrics":
                return
            sub_type = data.get("type", "")
            if sub_type == "captcha_gate":
                # ``url`` is a redacted page URL; if absent fall back to
                # nothing so we never silently bin under "" — that would
                # gather every host that lost its URL into one bucket.
                host = canonical_host(data.get("url") or "")
                outcome = data.get("outcome") or ""
                count = int(data.get("count") or 1)
                # Bucket carries an aggregated count — replay each event.
                for _ in range(max(count, 1)):
                    self.record_captcha(host, outcome)
            elif sub_type == "fingerprint_event":
                signal = data.get("signal") or ""
                if signal != "fingerprint_burn":
                    return  # only count burns; rejected/accepted are noisy
                host = canonical_host(data.get("page_origin") or "")
                count = int(data.get("count") or 1)
                for _ in range(max(count, 1)):
                    self.record_fingerprint_burn(host)
            elif sub_type == "platform_pre_nav_delay":
                host = canonical_host(data.get("host") or "")
                count = int(data.get("count") or 1)
                total = float(data.get("total_delay_s") or 0.0)
                # Replay the aggregated count so navigations++ each time;
                # split the dwell budget evenly.  A perfect-fidelity
                # reconstruction would need per-sample timestamps, which
                # would balloon the EventBus payload past §2.7's spirit.
                per = (total / count) if count > 0 else 0.0
                for _ in range(max(count, 1)):
                    self.record_pre_nav_delay(host, per)
        except Exception as e:
            # Belt-and-braces: a malformed payload must NOT poison the
            # EventBus broadcast loop.  Log at debug so an upstream bug
            # (e.g. a new event shape) is greppable but not noisy.
            logger.debug("platform_success.handle_event ignored: %s", e)


# ── Internals ────────────────────────────────────────────────────────────


def _push(d: deque, ts: float, value: float) -> None:
    """Append a sample, dropping the oldest if we hit the cap."""
    if len(d) >= MAX_SAMPLES_PER_HOST:
        d.popleft()
    d.append((ts, value))


def _prune(d: deque, cutoff: float) -> None:
    """Drop samples older than ``cutoff``.  Lazy — called on read."""
    while d and d[0][0] < cutoff:
        d.popleft()


def _sum_count(d: deque) -> int:
    """Sum the count column (each sample contributes ``value``)."""
    return int(sum(v for _, v in d))


def _sum_value(d: deque) -> float:
    """Sum the value column as float (for delay aggregates)."""
    return float(sum(v for _, v in d))
