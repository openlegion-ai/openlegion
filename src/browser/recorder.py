"""Behavioral entropy recorder (Phase 2 §5.3) — dev-only.

Captures input-event timestamps and basic shape metadata for later
offline analysis (§9.5 builds an entropy analyzer on top). Enabled via
``BROWSER_RECORD_BEHAVIOR=1``; otherwise every method is a cheap no-op
so the hot path (click/type/scroll/navigate) pays no cost in production.

**Privacy**: we never record the *content* of keystrokes or typed text.
For a keystroke or ``type_text`` call we store the length and inter-key
interval only. URL fragments and query strings are also dropped before
they reach disk — the goal is to analyze *timing distributions*, not
what the agent did.

The buffer is a bounded deque (default 10 000 events per instance). On
``dump()`` the buffer is flushed to ``/data/debug/{agent}-{ts}.jsonl``
(one event per line, ``jsonl`` so ``jq`` can stream it) and cleared.
``dump()`` runs from ``CamoufoxInstance`` teardown paths (explicit
reset, idle stop), so a running browser never blocks on disk I/O.

Directory creation is best-effort. If ``/data/debug`` can't be created
(read-only FS, unit-test environment), ``dump()`` logs and returns
silently — a missing dump is strictly better than crashing the reset.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from src.browser.flags import get_bool
from src.shared.utils import setup_logging

logger = setup_logging("browser.recorder")


_DEFAULT_BUFFER_SIZE = 10_000
_DEFAULT_DUMP_DIR = Path("/data/debug")


def recorder_enabled() -> bool:
    """Single read point for the feature flag.

    Goes through :mod:`src.browser.flags` so operator overrides and
    dashboard toggles work the same way as every other browser knob.
    """
    return get_bool("BROWSER_RECORD_BEHAVIOR", default=False)


class BehaviorRecorder:
    """Per-instance ring buffer of input-event shape metadata.

    When disabled, ``record_*`` methods short-circuit to a single
    boolean check — zero allocation. Safe to construct unconditionally.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        dump_dir: Path | None = None,
    ):
        self.agent_id = agent_id
        self._enabled = recorder_enabled()
        self._events: deque[dict] = deque(maxlen=buffer_size)
        self._dump_dir = dump_dir or _DEFAULT_DUMP_DIR
        # Track last event timestamp to compute inter-event intervals
        # cheaply without a full-buffer scan.
        self._last_ts: float | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def __len__(self) -> int:
        return len(self._events)

    def _append(self, kind: str, **fields: Any) -> None:
        """Shared append path — stamps timestamp + interval."""
        if not self._enabled:
            return
        now = time.time()
        interval = None if self._last_ts is None else round(now - self._last_ts, 6)
        self._last_ts = now
        event = {
            "ts": round(now, 6),
            "interval_s": interval,
            "type": kind,
            **fields,
        }
        self._events.append(event)

    def record_click(
        self,
        *,
        method: str,
        success: bool,
        dwell_ms: int | None = None,
    ) -> None:
        """``method`` = 'x11' | 'cdp' | 'playwright'."""
        self._append(
            "click",
            method=method,
            success=success,
            dwell_ms=dwell_ms,
        )

    def record_keystrokes(
        self,
        *,
        char_count: int,
        fast: bool,
        method: str,
    ) -> None:
        """Record a bulk ``type_text`` invocation.

        We intentionally do NOT store individual key intervals — that
        would bloat the buffer and the per-key distribution is better
        captured by instrumenting the humanize layer directly (which
        §9.5 handles offline from Camoufox logs).
        """
        self._append(
            "keystrokes",
            char_count=int(char_count),
            fast=bool(fast),
            method=method,
        )

    def record_scroll(
        self,
        *,
        direction: str,
        delta: int | None,
        method: str,
    ) -> None:
        self._append("scroll", direction=direction, delta=delta, method=method)

    def record_navigate(self, *, host: str, wait_until: str) -> None:
        """Record only the host, never the full URL.

        Query strings and fragments routinely carry secrets (OAuth
        codes, SigV4 signatures, magic-link tokens). The entropy
        analyzer cares about nav cadence, not destinations.
        """
        self._append("navigate", host=host, wait_until=wait_until)

    def dump(self, *, reason: str = "reset") -> Path | None:
        """Flush the buffer to ``/data/debug/{agent}-{ts}.jsonl``.

        Returns the written path on success, ``None`` on failure or
        when disabled / empty. Failures never propagate — recorder is
        strictly diagnostic.
        """
        if not self._enabled or not self._events:
            return None
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Recorder dump: cannot create %s: %s", self._dump_dir, e,
            )
            return None

        # Snapshot and clear the buffer atomically-ish — we're single-
        # threaded within an agent's lock, so this can't race against
        # record_* calls from this instance.
        events = list(self._events)
        self._events.clear()
        self._last_ts = None

        ts = int(time.time())
        safe_agent = _sanitize_filename(self.agent_id)
        target = self._dump_dir / f"{safe_agent}-{ts}.jsonl"
        try:
            # Write via a `.partial` file + rename so a crashed dump
            # doesn't leave half-written records next to good ones.
            partial = target.with_suffix(".jsonl.partial")
            with open(partial, "w", encoding="utf-8") as f:
                # Leading header so offline tools can sniff the format.
                f.write(json.dumps({
                    "schema": "openlegion.browser.recorder/v1",
                    "agent": self.agent_id,
                    "reason": reason,
                    "event_count": len(events),
                }) + "\n")
                for ev in events:
                    f.write(json.dumps(ev, separators=(",", ":")) + "\n")
            os.replace(partial, target)
        except OSError as e:
            logger.warning("Recorder dump write failed for %s: %s", target, e)
            return None
        logger.debug(
            "Recorder dumped %d events for '%s' → %s",
            len(events), self.agent_id, target,
        )
        return target


def _sanitize_filename(name: str) -> str:
    """Reduce an agent id to a filesystem-safe token."""
    # Agent ids already match AGENT_ID_RE_PATTERN (alnum + hyphen +
    # underscore) but be defensive against injected test values.
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return out[:64] or "agent"
