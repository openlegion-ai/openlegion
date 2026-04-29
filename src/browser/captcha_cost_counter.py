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
  * :func:`adjust_cost(agent_id, delta_millicents)` — apply a signed
    correction, clamped at zero. Used to refund pre-solve reservations.
  * :func:`over_cap(agent_id, cap_millicents)` — read-only check.
  * :func:`check_and_charge(agent_id, cap_millicents, millicents)` —
    atomic under-cap reservation/charge helper.
  * :func:`snapshot()` / :func:`restore()` — JSON file persistence. Atomic
    write via ``os.replace`` after ``fsync``. The on-disk schema migrates
    legacy ``cents`` snapshots by multiplying ×1000 on load (logged once).
  * :func:`estimate_millicents(provider, kind)` — fixed table of published
    rates. Returns ``None`` when the variant isn't priced — caller should
    log a warning and skip counting (over-counting is worse than
    under-counting for trust).
  * :func:`get_millicents(agent_id)` — current-month spend.

Tenant rollup (Phase 10 §24):
  * :func:`_tenant_for(agent_id)` — resolve an agent to its tenant via
    project membership in ``config/projects/`` (operators group agents
    into projects, and each project IS the tenant scope for billing
    rollups). Cached LRU(256). Returns ``None`` for unprojected agents.
  * :func:`get_tenant_total(tenant_id, since=None)` — sum every per-agent
    bucket whose owner resolves to ``tenant_id``. ``since`` is reserved
    for future snapshot-window filtering; today the in-memory state is
    always current-month so any ``since`` within the current month
    returns the live total.
  * :func:`get_tenant_breakdown(tenant_id)` — per-agent spend within a
    tenant for transparency on the operator billing-export endpoint.
  * :func:`record_tenant_threshold_alerts(tenant_id, cap_millicents,
    emit)` — fire 50/80/100% threshold alerts once per crossing per
    month, calling ``emit(payload)`` for each newly-crossed threshold.
    Crossing memory resets on month rollover.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

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


async def adjust_cost(agent_id: str, delta_millicents: int) -> int:
    """Apply a signed correction to the current-month bucket.

    Positive deltas behave like :func:`add_cost`; negative deltas refund a
    reservation or over-estimate. The bucket is clamped at zero so a double
    refund or process-restart mismatch cannot create negative spend.
    """
    if delta_millicents == 0:
        return await get_millicents(agent_id)
    async with _get_lock():
        bucket = _bucket_for(agent_id)
        current = int(bucket["millicents"])
        bucket["millicents"] = max(0, current + int(delta_millicents))
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
    the lock across both reads and writes here. BrowserManager uses this
    as a pre-solve reservation for the maximum published price, then calls
    :func:`adjust_cost` to refund or correct after the provider result is
    known.

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


# ── Phase 10 §24 — per-tenant rollup + threshold alerts ───────────────────
#
# Operators want a single billing-reconciliation number per tenant — the
# sum of every agent's CAPTCHA spend within that tenant's scope. The
# existing per-agent storage (above) stays unchanged; this block adds
# tenant-aware READ helpers that walk the same ``_state`` dict and group
# agents by their resolved tenant.
#
# **Tenant resolution.** The engine groups agents into projects via
# ``config/projects/<name>/metadata.yaml``; each project's ``members``
# list IS the tenant boundary for billing rollup. ``_tenant_for(agent_id)``
# resolves an agent → project name (or ``None`` for unprojected agents).
# Cached LRU(256) — projects rarely change at runtime, and the cache cuts
# ~256 YAML reads per CSV export down to one. Operators who reshape
# projects must call :func:`reset_tenant_cache` (or restart the service)
# to invalidate.
#
# **Threshold alerts.** Operators configure
# ``CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH`` per-tenant via the existing
# flags layer (treating the tenant ID as the ``agent_id`` arg to
# ``flags.get_int`` so per-tenant overrides land on the same precedence
# chain operators already use for per-agent flags). When a tenant crosses
# 50% / 80% / 100% of the configured cap, we emit one
# ``tenant_spend_threshold`` event per crossing per month. The
# ``_threshold_state`` dict tracks "fired this month" so a single crossing
# doesn't keep re-firing on every solve.


@functools.lru_cache(maxsize=256)
def _tenant_for(agent_id: str) -> str | None:
    """Resolve an agent ID to its tenant ID (project name).

    Reads ``config/projects/`` via :func:`src.cli.config._load_config`
    (which builds the reverse ``agent → project`` map at load time) and
    returns the project name for ``agent_id``, or ``None`` when the
    agent isn't in any project.

    LRU(256) cached because the CSV export and threshold-alert paths can
    each call it once per agent in the tenant's fleet — a 50-agent tenant
    on a 60s metrics tick would be 3000 YAML reads/min without the cache.
    Agents are added to / removed from projects rarely (operator action),
    and the cache invalidation hook is :func:`reset_tenant_cache`.

    The lookup is intentionally read-only and does NOT mutate config —
    misconfigured / partial state returns ``None`` and the caller treats
    the agent as untracked for tenant rollups (its per-agent bucket is
    still observable via :func:`get_millicents`).
    """
    if not agent_id:
        return None
    try:
        # Lazy import — ``cli.config`` imports YAML / file IO that we
        # don't want pulled in at the browser-service import time.
        from src.cli.config import _load_config
        cfg = _load_config()
    except Exception as e:
        logger.debug("tenant lookup: _load_config failed: %s", e)
        return None
    agent_projects = cfg.get("_agent_projects") or {}
    project = agent_projects.get(agent_id)
    if not isinstance(project, str) or not project:
        return None
    return project


def reset_tenant_cache() -> None:
    """Invalidate the :func:`_tenant_for` LRU cache.

    Call after mutating ``config/projects/`` so subsequent rollups see
    the new membership. Tests and the dashboard's project-edit endpoints
    use this hook.
    """
    _tenant_for.cache_clear()


async def get_tenant_total(
    tenant_id: str, *, since: datetime | None = None,
) -> int:
    """Sum current-month millicents across every agent in ``tenant_id``.

    ``since=None`` returns the live current-month total — the in-memory
    state already resets per calendar month, so summing the buckets IS
    the current-month total.

    ``since=<dt>`` is reserved for windowed reads ("last 7 days",
    "last 30 days") backed by saved snapshot timestamps. The current
    in-memory state has ONE timestamp granularity (the bucket's month),
    so any ``since`` falling within the current month returns the same
    live total — older windows are not retained in this trim. Operators
    who need finer windows should drive them off persisted snapshots
    (deferred per the same rationale as §11.10's SQLite trim).

    Returns 0 for unknown tenants (or a tenant whose agents have never
    been charged this month). Cross-tenant isolation is enforced by
    matching only buckets whose ``_tenant_for(agent_id)`` equals
    ``tenant_id``.
    """
    if not tenant_id:
        return 0
    cm = _current_month()
    async with _get_lock():
        # Snapshot the state under the lock so concurrent ``add_cost``
        # writes don't change the dict mid-walk. ``items()`` materializes
        # a list at call time so we don't hold the lock through the
        # tenant-resolution loop (which calls into ``cli.config``).
        items = list(_state.items())

    total = 0
    for agent_id, bucket in items:
        if not isinstance(bucket, dict):
            continue
        if bucket.get("month") != cm:
            continue
        if _tenant_for(agent_id) != tenant_id:
            continue
        # ``since`` filter is a no-op for in-memory state (see docstring),
        # but accept the kwarg so callers don't have to special-case it.
        if since is not None:
            # The bucket has month-granularity only; if ``since`` is in a
            # past month we have nothing to return (state was already
            # reset). If it's within the current month, the live total
            # IS what we have.
            if since.strftime("%Y-%m") != cm:
                continue
        total += int(bucket.get("millicents", 0))
    return total


async def get_tenant_breakdown(tenant_id: str) -> dict[str, int]:
    """Return ``{agent_id: millicents}`` for every agent in ``tenant_id``.

    Powers the operator CSV export so each agent's contribution to the
    tenant total is auditable. Agents with zero spend are still
    included if they resolve to ``tenant_id`` (the operator wants to
    SEE which agents in the tenant are active vs idle).
    """
    if not tenant_id:
        return {}
    cm = _current_month()
    async with _get_lock():
        items = list(_state.items())

    breakdown: dict[str, int] = {}
    for agent_id, bucket in items:
        if not isinstance(bucket, dict):
            continue
        if bucket.get("month") != cm:
            continue
        if _tenant_for(agent_id) != tenant_id:
            continue
        breakdown[agent_id] = int(bucket.get("millicents", 0))
    return breakdown


# ── Threshold alert state ──────────────────────────────────────────────────


# Default crossing percentages. Sized to give operators "early warning"
# (50%), "danger" (80%), and "exhausted" (100%) — three signals with
# escalating urgency. Keep ascending so the cross-detection loop walks
# them in order.
DEFAULT_THRESHOLD_PCTS: tuple[int, ...] = (50, 80, 100)

# {tenant_id: {"month": "YYYY-MM", "fired_pct": set[int]}}
_threshold_state: dict[str, dict[str, Any]] = {}
# Lock is created lazily, same loop-binding pattern as ``_lock`` above —
# the alert recorder runs inside the metrics-emit task, which lives on
# the browser service event loop.
_threshold_lock: asyncio.Lock | None = None
_threshold_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_threshold_lock() -> asyncio.Lock:
    """Return a threshold-tracking lock bound to the active event loop."""
    global _threshold_lock, _threshold_lock_loop
    loop = asyncio.get_running_loop()
    if _threshold_lock is None or _threshold_lock_loop is not loop:
        _threshold_lock = asyncio.Lock()
        _threshold_lock_loop = loop
    return _threshold_lock


def _threshold_bucket(tenant_id: str, current_month: str) -> dict[str, Any]:
    """Return the per-tenant threshold bucket, resetting on month rollover.

    Caller MUST hold ``_threshold_lock``. The reset behaviour mirrors
    :func:`_bucket_for` for the spend buckets: when the calendar month
    rolls over we drop ``fired_pct`` so the new month gets fresh alerts.
    """
    bucket = _threshold_state.get(tenant_id)
    if bucket is None or bucket.get("month") != current_month:
        bucket = {"month": current_month, "fired_pct": set()}
        _threshold_state[tenant_id] = bucket
    return bucket


async def record_tenant_threshold_alerts(
    tenant_id: str,
    cap_millicents: int,
    emit: Callable[[dict[str, Any]], Awaitable[None]] | Callable[[dict[str, Any]], None],
    *,
    pcts: tuple[int, ...] = DEFAULT_THRESHOLD_PCTS,
) -> list[int]:
    """Fire ``tenant_spend_threshold`` events for newly-crossed thresholds.

    ``cap_millicents`` is the configured monthly cap for ``tenant_id``,
    converted from the operator-set USD value (see flags.py
    ``CAPTCHA_COST_LIMIT_USD_PER_TENANT_MONTH``). When ``cap_millicents``
    is 0 or negative the cap is disabled and we return early without
    firing — operators set the flag to 0 to opt out.

    For each percentage in ``pcts``, we fire a single event the FIRST
    time the tenant's monthly spend crosses the corresponding millicents
    threshold. Subsequent calls within the same month do NOT re-fire the
    same percentage; the ``fired_pct`` set tracks what's already been
    surfaced. Month rollover resets the set.

    ``emit`` is the callback the dashboard wires up — typically
    ``EventBus.emit("tenant_spend_threshold", agent="", data=payload)``.
    Both sync and async callbacks are accepted (the metrics-emit path is
    sync, but the dashboard endpoint may want async).

    Returns the list of percentages newly fired this call (empty list if
    nothing crossed).
    """
    if not tenant_id or cap_millicents <= 0:
        return []
    spend = await get_tenant_total(tenant_id)
    cm = _current_month()
    fired_now: list[int] = []
    async with _get_threshold_lock():
        bucket = _threshold_bucket(tenant_id, cm)
        already = bucket["fired_pct"]
        for pct in sorted(pcts):
            if pct in already:
                continue
            crossing = (cap_millicents * pct) // 100
            if spend >= crossing:
                already.add(pct)
                fired_now.append(pct)
    # Emit OUTSIDE the lock — the callback may itself acquire other
    # locks (EventBus.emit takes its own threading.Lock) and we don't
    # want a cross-lock deadlock.
    for pct in fired_now:
        payload = {
            "tenant_id": tenant_id,
            "pct": pct,
            "spend_millicents": spend,
            "cap_millicents": cap_millicents,
            "spend_usd": round(spend / 100_000.0, 4),
            "cap_usd": round(cap_millicents / 100_000.0, 4),
            "month": cm,
        }
        try:
            result = emit(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:  # noqa: BLE001 — defensive; alert path
            logger.warning(
                "tenant_spend_threshold emit failed for tenant=%s pct=%d: %s",
                tenant_id, pct, e,
            )
    return fired_now


def reset_threshold_state(tenant_id: str | None = None) -> None:
    """Clear remembered threshold crossings. Test-harness + month-rollover hook.

    ``tenant_id=None`` clears every tenant's tracking — used by tests to
    reset state between cases. Production callers should rely on the
    automatic month-rollover reset baked into :func:`_threshold_bucket`.
    """
    if tenant_id is None:
        _threshold_state.clear()
    else:
        _threshold_state.pop(tenant_id, None)
