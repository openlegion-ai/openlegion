"""In-memory per-agent CAPTCHA cost counter (§11.10 simplified replacement).

The original §11.10 plan called for a SQLite-backed monthly counter so cost
caps survived process restarts with full transactional integrity. The Phase 8
trim deferred that complexity in favor of an in-memory dict snapshotted to a
JSON file on graceful shutdown. Restart loses at most one window's worth of
counted spend — acceptable because the cap is per-month and a process restart
inside a billing month is rare; the alternative (SQLite, WAL, schema-migration
plumbing) was disproportionate to the value.

UNITS — IMPORTANT.

All values are stored and compared in **MILLICENTS** (1/1000 of a US cent
= 1/100_000 of a US dollar). A 2captcha v2-checkbox solve is published at
$1.00 per 1000 solves = $0.001 per solve = 0.1 cents = **100 millicents**.

Conversion shortcuts:

  * dollars → millicents: ``int(round(usd * 100_000))``
  * millicents → cents (for human display): ``millicents / 1000.0``
  * millicents → dollars (for human display): ``millicents / 100_000.0``

Why millicents and not cents or ``Decimal``? Most published solve rates
are sub-cent (0.06¢ – 0.6¢). Storing as integer cents would round every
real per-solve charge to 0 or 1, blowing up the cap-tripping math. Storing
as ``Decimal`` USD is cleaner but a wider refactor; integer millicents
gives lossless arithmetic for the published rate table while keeping the
existing `int` storage / JSON shape unchanged.

Public surface:
  * :func:`add_cost(agent_id, millicents)` — increment a per-agent
    monthly bucket. Resets when the calendar month rolls over.
  * :func:`over_cap(agent_id, cap_millicents)` — read-only check.
  * :func:`snapshot()` / :func:`restore()` — JSON file persistence. Atomic
    write via ``os.replace`` after ``fsync``. The on-disk schema migrates
    legacy ``cents`` snapshots by multiplying ×1000 on load (logged once).
  * :func:`estimate_millicents(provider, kind)` — fixed table of published
    rates. Returns ``None`` when the variant isn't priced — caller should
    log a warning and skip counting (over-counting is worse than
    under-counting for trust).
  * :func:`get_millicents(agent_id)` — current-month spend.
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


# Published rates from 2captcha + CapSolver (April 2026), in **millicents**
# (1/1000 of a US cent) per successful solve. Keys follow the convention
# ``{provider}-{variant}`` for the proxyless tier; the proxy-aware tier
# uses a 2-tuple key ``(provider, variant)`` and lives in
# :data:`PRICING_MILLICENTS_PROXY_AWARE` below.
#
# Why two tables? Provider docs publish proxyless rates in a single line
# but proxy-aware rates as "approximately 3× the base" in their pricing
# pages. Keeping the dicts separate keeps the §11.1 string-key surface
# untouched while letting §11.2's tuple-key path coexist.
#
# When a kind isn't priced, callers SKIP counting rather than guessing —
# under-count > over-count for operator trust.
#
# Sanity: 100 millicents = 0.1 cents = $0.001/solve. 2Captcha publishes
# v2-checkbox at $1.00 / 1000 solves = $0.001/solve = 100 millicents.  ✓
PRICING_MILLICENTS: dict[str, int] = {
    # 2Captcha — published https://2captcha.com/2captcha-api#solving_recaptchav2_new
    "2captcha-recaptcha-v2-checkbox": 100,        # $1.00 / 1000 = 0.1¢ each → 100 millicents
    "2captcha-recaptcha-v2-invisible": 100,
    "2captcha-recaptcha-v3": 100,
    # §11.1 splits ``recaptcha-enterprise`` into v2/v3 enterprise variants;
    # both keep the same Enterprise rate. ``recaptcha-enterprise`` is
    # retained as a back-compat alias for any callers still emitting the
    # coarse kind from a hint override.
    "2captcha-recaptcha-enterprise": 200,         # $2.00 / 1000 = 0.2¢ each → 200 millicents
    "2captcha-recaptcha-enterprise-v2": 200,
    "2captcha-recaptcha-enterprise-v3": 200,
    "2captcha-hcaptcha": 100,
    "2captcha-turnstile": 200,
    # CF-bound Turnstile (§11.3 ``cf-interstitial-turnstile``) — solver
    # path is identical to standalone Turnstile; only the envelope ``kind``
    # differs. Aliased here so :func:`estimate_millicents` doesn't return
    # ``None`` for the CF variant and trip the spurious "no published rate"
    # warning.
    "2captcha-cf-interstitial-turnstile": 200,
    # CapSolver — published https://docs.capsolver.com/guide/captcha/
    "capsolver-recaptcha-v2-checkbox": 80,        # $0.80 / 1000 = 0.08¢ each → 80 millicents
    "capsolver-recaptcha-v2-invisible": 80,
    "capsolver-recaptcha-v3": 80,
    "capsolver-recaptcha-enterprise": 200,
    "capsolver-recaptcha-enterprise-v2": 200,
    "capsolver-recaptcha-enterprise-v3": 200,
    "capsolver-hcaptcha": 100,
    "capsolver-turnstile": 60,
    "capsolver-cf-interstitial-turnstile": 60,
}

# §11.2 — proxy-aware pricing tier (~3× the proxyless rate as published
# by both providers). Tuple key ``(provider, variant)``; ``proxy_aware``
# flag at lookup time picks this table over :data:`PRICING_MILLICENTS`.
#
# 2captcha v3 has no documented proxy-aware task type as of April 2026,
# so its proxy-aware entries are intentionally absent —
# :func:`estimate_millicents` falls through to the proxyless price for
# those (the body builder already falls back to proxyless when no
# proxy_aware task is documented, so paying proxyless rate matches what
# the provider sees).
PRICING_MILLICENTS_PROXY_AWARE: dict[tuple[str, str], int] = {
    # 2captcha — 3× the proxyless rate
    ("2captcha", "recaptcha-v2-checkbox"):    300,
    ("2captcha", "recaptcha-v2-invisible"):   300,
    ("2captcha", "recaptcha-enterprise-v2"):  600,
    ("2captcha", "hcaptcha"):                 300,
    ("2captcha", "turnstile"):                600,
    # CF-bound Turnstile takes the same proxy-aware tier as standalone
    # Turnstile — the solver task is identical (§11.3).
    ("2captcha", "cf-interstitial-turnstile"): 600,
    # CapSolver — 3× the proxyless rate
    ("capsolver", "recaptcha-v2-checkbox"):   240,
    ("capsolver", "recaptcha-v2-invisible"):  240,
    ("capsolver", "recaptcha-v3"):            240,
    ("capsolver", "recaptcha-enterprise-v2"): 600,
    ("capsolver", "recaptcha-enterprise-v3"): 600,
    ("capsolver", "hcaptcha"):                300,
    ("capsolver", "turnstile"):               180,
    ("capsolver", "cf-interstitial-turnstile"): 180,
}

# Back-compat aliases — third-party subclasses or future callers that
# import ``PRICING_CENTS`` will get the millicents table. Names retained
# so an out-of-tree subclass does not break at import time even if its
# arithmetic is now off-by-1000 (a logged warning is the worst that
# happens; the real fix landed inside this module).
PRICING_CENTS = PRICING_MILLICENTS
PRICING_CENTS_PROXY_AWARE = PRICING_MILLICENTS_PROXY_AWARE


def estimate_millicents(
    provider: str, kind: str, *, proxy_aware: bool = False,
) -> int | None:
    """Return published cost (millicents) for one successful solve, or ``None``.

    ``proxy_aware=True`` consults :data:`PRICING_MILLICENTS_PROXY_AWARE`
    first (~3× the proxyless rate). When the proxy-aware table doesn't
    have a row for the (provider, variant) tuple — e.g. 2captcha v3,
    where the provider has no documented proxy-aware task type — we fall
    through to the proxyless rate. That matches what the body builder did
    at the request layer (proxyless task body), so the price billed
    equals the request type sent.

    ``None`` signals the variant isn't priced; callers should log a warning
    and SKIP the increment rather than attribute a guess to the agent's
    monthly spend (per the trim-spec note: over-counting is worse than
    under-counting for trust).

    Lookups are case-insensitive on both sides. The kind is normalized
    against the §11.13 enum so callers can pass a §11.13 ``kind`` directly.
    """
    if not provider or not kind:
        return None
    p = provider.strip().lower()
    k = kind.strip().lower()
    if proxy_aware:
        mc = PRICING_MILLICENTS_PROXY_AWARE.get((p, k))
        if mc is not None:
            return mc
        # Fall through to the proxyless rate — the body builder degraded
        # to proxyless for this variant, so the price tier matches.
    key = f"{p}-{k}"
    return PRICING_MILLICENTS.get(key)


# Back-compat alias — third-party subclasses or older call sites using
# ``estimate_cents`` get the millicents value transparently. The name is
# off-by-1000 wrt its label but every internal caller has been migrated
# to ``estimate_millicents`` in the same change-set; this exists so an
# out-of-tree CaptchaSolver subclass keeps importing without a hard break.
estimate_cents = estimate_millicents


# ── State + lock ───────────────────────────────────────────────────────────


# {agent_id: {"month": "YYYY-MM", "millicents": int}}
_state: dict[str, dict] = {}
# Created lazily inside async paths to stay loop-safe — pytest-asyncio
# uses one event loop per test, so an import-time ``asyncio.Lock()``
# would bind to the first loop and fail on the second. Mirrors the
# pattern in ``service.py`` (``_get_solve_rate_lock`` etc.) and
# ``captcha.py`` (``CaptchaSolver._get_state_lock``).
_lock: asyncio.Lock | None = None
_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_lock() -> asyncio.Lock:
    """Return a cost-counter lock bound to the active event loop."""
    global _lock, _lock_loop
    loop = asyncio.get_running_loop()
    if _lock is None or _lock_loop is not loop:
        _lock = asyncio.Lock()
        _lock_loop = loop
    return _lock


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
        bucket = {"month": cm, "millicents": 0}
        _state[agent_id] = bucket
    return bucket


# ── Public mutators / readers ──────────────────────────────────────────────


async def add_cost(agent_id: str, millicents: int) -> int:
    """Add ``millicents`` to ``agent_id``'s current-month bucket.

    Returns the new total (millicents). Concurrent writers across asyncio
    tasks are serialized by ``_lock`` — matters because the browser
    service serves multiple agents off a single event loop. Non-positive
    inputs are silently dropped (defensive).
    """
    if millicents <= 0:
        return await get_millicents(agent_id)
    async with _get_lock():
        bucket = _bucket_for(agent_id)
        bucket["millicents"] = int(bucket["millicents"]) + int(millicents)
        return bucket["millicents"]


async def over_cap(agent_id: str, cap_millicents: int) -> bool:
    """Return ``True`` iff current-month spend ≥ ``cap_millicents``.

    ``cap_millicents <= 0`` disables the cap (returns ``False`` regardless
    of spend) — operators set the env var to ``0`` to opt out.

    Callers MUST pass the cap in millicents — the cap-USD env var is
    converted at the read site (see ``src/browser/service.py:
    _resolve_cost_cap``). Passing a cents value here would re-introduce
    the unit-mismatch bug that this module's docstring warns against.
    """
    if cap_millicents <= 0:
        return False
    async with _get_lock():
        bucket = _bucket_for(agent_id)
        return int(bucket["millicents"]) >= int(cap_millicents)


async def get_millicents(agent_id: str) -> int:
    """Return the current-month spend for ``agent_id`` in millicents."""
    async with _get_lock():
        bucket = _bucket_for(agent_id)
        return int(bucket["millicents"])


async def check_and_charge(
    agent_id: str, cap_millicents: int, millicents: int,
) -> tuple[bool, int]:
    """Atomically gate ``millicents`` against ``cap_millicents``.

    Returns ``(allowed, new_total)``:
      * ``allowed=True``  → bucket was UNDER cap; ``millicents`` was added.
      * ``allowed=False`` → bucket was AT/OVER cap; nothing was charged.

    Closes the race between separate ``over_cap`` / ``add_cost`` lock
    spans where two concurrent solves could each see ``bucket < cap``,
    each call ``add_cost``, and together push the bucket above cap. Hold
    the lock across both reads and writes here.

    ``cap_millicents <= 0`` means "no cap": always allowed; cost is still
    charged. ``millicents <= 0`` means "free attempt": always allowed;
    nothing is charged.
    """
    async with _get_lock():
        bucket = _bucket_for(agent_id)
        current = int(bucket["millicents"])
        if cap_millicents > 0 and current >= int(cap_millicents):
            return False, current
        if millicents > 0:
            bucket["millicents"] = current + int(millicents)
            return True, bucket["millicents"]
        return True, current


# Back-compat alias for an external caller that used to read cents. The
# name is preserved but the units are now MILLICENTS — the caller almost
# certainly wants ``get_millicents`` directly, but this avoids a hard
# break at import. Internal call sites have been migrated.
get_cents = get_millicents


async def reset(agent_id: str | None = None) -> None:
    """Clear state. ``agent_id=None`` clears everything (test harness)."""
    async with _get_lock():
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

    async with _get_lock():
        # JSON-serializable copy. Bucket dicts are already plain
        # ``{month, cents}`` so no custom encoder is needed.
        payload = {
            "version": 1,
            "saved_at": int(time.time()),
            "buckets": dict(_state),
        }

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        # Open with 0o600 from the start (umask-aware) so there is no
        # world-readable window before the explicit chmod below. The
        # cost ledger is operator-grade billing data — same posture as
        # ``.env`` files (CLAUDE.md §Security Boundaries).
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            os.close(fd)
            raise
        os.replace(tmp, target)
        # ``os.replace`` preserves the destination's mode on most
        # filesystems but the Python docs are not load-bearing on this;
        # explicit chmod after replace ensures 0o600 regardless of
        # whether the target existed (and what its prior mode was).
        os.chmod(target, 0o600)
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

    **Schema migration.** Older snapshots stored a ``cents`` field. This
    module now uses ``millicents`` (1/1000 of a cent) so the field was
    renamed; legacy ``cents`` values are migrated by multiplying ×1000.
    The migration logs once per restore so operators can correlate any
    apparent jump in monthly spend with the unit fix.
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
    migrated = 0
    async with _get_lock():
        _state.clear()
        for agent_id, bucket in buckets.items():
            if not isinstance(bucket, dict):
                continue
            month = bucket.get("month")
            if not isinstance(month, str):
                continue
            # Prefer the new-shape ``millicents`` field; fall back to the
            # legacy ``cents`` field with a ×1000 conversion. The
            # migration is idempotent (re-saving immediately produces a
            # millicents-only payload) so the ``cents`` branch only ever
            # fires once per snapshot file.
            mc = bucket.get("millicents")
            if not isinstance(mc, int):
                legacy_cents = bucket.get("cents")
                if isinstance(legacy_cents, int):
                    mc = legacy_cents * 1000
                    migrated += 1
                else:
                    continue
            if month != cm:
                # Stale month — would reset on first access; skip restoring.
                continue
            _state[str(agent_id)] = {"month": month, "millicents": int(mc)}
            loaded += 1
    if migrated:
        logger.info(
            "captcha_cost restore: migrated %d legacy cents bucket(s) "
            "to millicents (×1000) from %s",
            migrated, target,
        )
    logger.info(
        "captcha_cost restore loaded %d/%d agent bucket(s) from %s",
        loaded, len(buckets), target,
    )
    return loaded
