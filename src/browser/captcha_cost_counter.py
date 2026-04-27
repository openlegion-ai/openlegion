"""In-memory per-agent CAPTCHA cost counter (§11.10 simplified replacement).

The original §11.10 plan called for a SQLite-backed monthly counter so cost
caps survived process restarts with full transactional integrity. The Phase 8
trim deferred that complexity in favor of an in-memory dict snapshotted to a
JSON file on graceful shutdown. Restart loses at most one window's worth of
counted spend — acceptable because the cap is per-month and a process restart
inside a billing month is rare; the alternative (SQLite, WAL, schema-migration
plumbing) was disproportionate to the value.

Public surface:
  * :func:`add_cost(agent_id, cents)` — increment a per-agent monthly bucket.
    Resets the bucket when the calendar month rolls over.
  * :func:`over_cap(agent_id, cap_cents)` — read-only check.
  * :func:`snapshot()` / :func:`restore()` — JSON file persistence. Atomic
    write via ``os.replace`` after ``fsync``.
  * :func:`estimate_cents(provider, kind)` — fixed table of published rates
    so callers don't reimplement the lookup. Returns ``None`` when the
    variant isn't priced — caller should log a warning and skip counting
    (over-counting is worse than under-counting for trust).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from src.shared.utils import setup_logging

logger = setup_logging("browser.captcha_cost")


# Persistence path. Lives under ``data/`` (NOT a SQLite ``.db`` — see module
# docstring for the deferred-rationale; the trim spec is explicit on this).
_DEFAULT_PATH = "data/captcha_costs.json"


def _state_path() -> Path:
    return Path(os.environ.get("CAPTCHA_COST_COUNTER_PATH", _DEFAULT_PATH))


# ── Pricing table ──────────────────────────────────────────────────────────


# Published rates from 2captcha + CapSolver (April 2026), in US cents per
# successful solve. Keys follow the convention ``{provider}-{variant}`` so a
# future per-variant pricing tier (§11.1 reCAPTCHA v3 / Enterprise) can land
# without re-keying. When a kind isn't priced, callers SKIP counting rather
# than guessing — under-count > over-count for operator trust.
PRICING_CENTS: dict[str, int] = {
    # 2Captcha — published https://2captcha.com/2captcha-api#solving_recaptchav2_new
    "2captcha-recaptcha-v2-checkbox": 100,        # $1.00 / 1000 = 0.10c each → 0.1¢
    "2captcha-recaptcha-v2-invisible": 100,
    "2captcha-recaptcha-v3": 100,
    # §11.1 splits ``recaptcha-enterprise`` into v2/v3 enterprise variants;
    # both keep the same Enterprise rate. ``recaptcha-enterprise`` is
    # retained as a back-compat alias for any callers still emitting the
    # coarse kind from a hint override.
    "2captcha-recaptcha-enterprise": 200,
    "2captcha-recaptcha-enterprise-v2": 200,
    "2captcha-recaptcha-enterprise-v3": 200,
    "2captcha-hcaptcha": 100,
    "2captcha-turnstile": 200,
    # CapSolver — published https://docs.capsolver.com/guide/captcha/
    "capsolver-recaptcha-v2-checkbox": 80,
    "capsolver-recaptcha-v2-invisible": 80,
    "capsolver-recaptcha-v3": 80,
    "capsolver-recaptcha-enterprise": 200,
    "capsolver-recaptcha-enterprise-v2": 200,
    "capsolver-recaptcha-enterprise-v3": 200,
    "capsolver-hcaptcha": 100,
    "capsolver-turnstile": 60,
}


def estimate_cents(provider: str, kind: str) -> int | None:
    """Return published cost (cents) for one successful solve, or ``None``.

    ``None`` signals the variant isn't priced; callers should log a warning
    and SKIP the increment rather than attribute a guess to the agent's
    monthly spend (per the trim-spec note: over-counting is worse than
    under-counting for trust).

    Lookups are case-insensitive on both sides. The kind is normalized
    against the §11.13 enum so callers can pass a §11.13 ``kind`` directly.
    """
    if not provider or not kind:
        return None
    key = f"{provider.strip().lower()}-{kind.strip().lower()}"
    return PRICING_CENTS.get(key)


# ── State + lock ───────────────────────────────────────────────────────────


# {agent_id: {"month": "YYYY-MM", "cents": int}}
_state: dict[str, dict] = {}
_lock: asyncio.Lock = asyncio.Lock()


def _current_month() -> str:
    """Return the current UTC year-month as ``YYYY-MM``."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _bucket_for(agent_id: str, *, current_month: str | None = None) -> dict:
    """Return the per-agent bucket, resetting on month rollover.

    Caller must hold ``_lock``. Pass ``current_month`` if the caller is
    batch-processing across many agents and wants to compute it once.
    """
    cm = current_month or _current_month()
    bucket = _state.get(agent_id)
    if bucket is None or bucket.get("month") != cm:
        bucket = {"month": cm, "cents": 0}
        _state[agent_id] = bucket
    return bucket


# ── Public mutators / readers ──────────────────────────────────────────────


async def add_cost(agent_id: str, cents: int) -> int:
    """Add ``cents`` to ``agent_id``'s current-month bucket. Returns new total.

    Concurrent writers across asyncio tasks are serialized by ``_lock`` —
    matters because the browser service serves multiple agents off a single
    event loop. Negative or zero ``cents`` are silently dropped (defensive).
    """
    if cents <= 0:
        return await get_cents(agent_id)
    async with _lock:
        bucket = _bucket_for(agent_id)
        bucket["cents"] = int(bucket["cents"]) + int(cents)
        return bucket["cents"]


async def over_cap(agent_id: str, cap_cents: int) -> bool:
    """Return ``True`` iff this agent's current-month spend ≥ ``cap_cents``.

    ``cap_cents <= 0`` disables the cap (returns ``False`` regardless of
    spend) — operators set the env var to ``0`` to opt out.
    """
    if cap_cents <= 0:
        return False
    async with _lock:
        bucket = _bucket_for(agent_id)
        return int(bucket["cents"]) >= int(cap_cents)


async def get_cents(agent_id: str) -> int:
    """Return the current-month spend for ``agent_id`` in cents."""
    async with _lock:
        bucket = _bucket_for(agent_id)
        return int(bucket["cents"])


async def reset(agent_id: str | None = None) -> None:
    """Clear state. ``agent_id=None`` clears everything (test harness)."""
    async with _lock:
        if agent_id is None:
            _state.clear()
        else:
            _state.pop(agent_id, None)


# ── Persistence ────────────────────────────────────────────────────────────


async def snapshot(path: Path | str | None = None) -> bool:
    """Atomically write the in-memory state to ``path`` as JSON.

    Returns ``True`` on success. Failures log + return ``False`` rather than
    raise — the shutdown path must not abort because cost persistence failed.

    Atomic-write protocol: write to a sibling tmp file, ``fsync`` it,
    ``os.replace`` over the destination. Standard pattern used elsewhere in
    the codebase (e.g. config + cron persistence).
    """
    target = Path(path) if path else _state_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("captcha_cost snapshot: cannot create parent dir: %s", e)
        return False

    async with _lock:
        # JSON-serializable copy. Bucket dicts are already plain
        # ``{month, cents}`` so no custom encoder is needed.
        payload = {
            "version": 1,
            "saved_at": int(time.time()),
            "buckets": dict(_state),
        }

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except OSError as e:
        logger.warning("captcha_cost snapshot failed: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    logger.info(
        "captcha_cost snapshot wrote %d agent bucket(s) to %s",
        len(payload["buckets"]), target,
    )
    return True


async def restore(path: Path | str | None = None) -> int:
    """Load state from ``path`` if it exists. Returns count of buckets loaded.

    Missing / unreadable / malformed files are non-fatal — log + start
    fresh. Buckets whose ``month`` doesn't match the current month are
    dropped (they'd reset on next access anyway; no point keeping stale).
    """
    target = Path(path) if path else _state_path()
    if not target.exists():
        return 0
    try:
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("captcha_cost restore: %s — starting empty", e)
        return 0

    if not isinstance(payload, dict):
        logger.warning("captcha_cost restore: unexpected payload type")
        return 0

    buckets = payload.get("buckets", {})
    if not isinstance(buckets, dict):
        return 0

    cm = _current_month()
    loaded = 0
    async with _lock:
        _state.clear()
        for agent_id, bucket in buckets.items():
            if not isinstance(bucket, dict):
                continue
            month = bucket.get("month")
            cents = bucket.get("cents", 0)
            if not isinstance(month, str) or not isinstance(cents, int):
                continue
            if month != cm:
                # Stale month — would reset on first access; skip restoring.
                continue
            _state[str(agent_id)] = {"month": month, "cents": cents}
            loaded += 1
    logger.info(
        "captcha_cost restore loaded %d/%d agent bucket(s) from %s",
        loaded, len(buckets), target,
    )
    return loaded
