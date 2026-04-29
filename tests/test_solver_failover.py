"""Tests for §11.8: multi-provider CAPTCHA solver failover.

Covers the :class:`MultiProviderSolver` wrapper that fronts a primary +
optional secondary :class:`CaptchaSolver` so the BrowserManager can
transparently route around primary outages without hard-restarting.

Test surface (numbered to match the spec):

  1. Primary healthy + secondary configured → primary wins, secondary
     untouched.
  2. Primary ``_solver_unreachable=True`` (e.g. fatal-config error from
     a prior call) → wrapper routes to secondary on next solve.
  3. Primary breaker open → wrapper routes to secondary.
  4. Primary fatal-error mid-call (returns ``token=None`` AND flips
     ``_solver_unreachable``) → wrapper retries on secondary inside the
     SAME ``solve()`` call.
  5. Both primary and secondary unreachable → no-token result, no
     provider HTTP issued.
  6. Single-provider config (``secondary=None``) → wrapper transparent;
     behaves identically to the underlying primary.
  7. Cost accounting: when secondary wins, ``solver.provider`` returns
     the secondary's provider name so :meth:`_metered_solve` looks up
     the right pricing tier.
  8. ``health_check()`` runs both and returns the worst outcome.
  9. Typo'd ``CAPTCHA_SOLVER_PROVIDER_SECONDARY`` (unsupported value)
     → secondary falls back to ``None``; wrapper degrades gracefully to
     single-provider behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser.captcha import (
    CaptchaSolver,
    MultiProviderSolver,
    SolveResult,
    get_solver,
)

# ── helpers ─────────────────────────────────────────────────────────────


def _make_solver(provider: str = "2captcha", key: str = "SECRET-PRIMARY") -> CaptchaSolver:
    return CaptchaSolver(provider, key)


def _ok_balance_resp(balance: float = 12.34) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"errorId": 0, "balance": balance})
    return resp


def _stub_create_then_poll(token: str | None) -> list[MagicMock]:
    """Build a (createTask, getTaskResult) response pair for the solve flow."""
    create = MagicMock()
    create.json = MagicMock(return_value={"errorId": 0, "taskId": "T-1"})
    create.raise_for_status = MagicMock()
    poll = MagicMock()
    if token is None:
        poll.json = MagicMock(return_value={"errorId": 1, "errorDescription": "fail"})
    else:
        poll.json = MagicMock(return_value={
            "errorId": 0, "status": "ready",
            "solution": {"gRecaptchaResponse": token},
        })
    poll.raise_for_status = MagicMock()
    return [create, poll]


def _stub_fatal_create() -> MagicMock:
    """A createTask response carrying a fatal-config marker.

    Provider returns ``errorId>0`` with one of the operator-actionable
    descriptions (drained balance, revoked key). Triggers
    :meth:`CaptchaSolver._handle_provider_error_response` to flip
    ``_solver_unreachable`` to ``True``.
    """
    create = MagicMock()
    create.json = MagicMock(return_value={
        "errorId": 10,
        "errorDescription": "ERROR_ZERO_BALANCE",
    })
    create.raise_for_status = MagicMock()
    return create


def _solve_page() -> MagicMock:
    """Mock Playwright page that returns a sitekey + accepts injection."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="site-key-abc")
    page.url = "https://example.com"
    return page


def _attach_client(solver: CaptchaSolver, *, responses) -> AsyncMock:
    """Mount an AsyncMock httpx client on ``solver`` with scripted responses.

    ``responses`` may be a list (consumed in order) or any side_effect
    accepted by ``AsyncMock``.
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(side_effect=responses)
    solver._client = client
    return client


# ── 1: primary healthy → primary wins, secondary untouched ──────────────


@pytest.mark.asyncio
async def test_primary_healthy_secondary_untouched():
    """When primary is healthy, every solve hits the primary; the
    secondary's HTTP client is never invoked."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    primary_client = _attach_client(primary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-primary"),
    ])
    secondary_client = _attach_client(secondary, responses=[])

    page = _solve_page()
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    assert result.token == "tok-primary"
    # Primary saw the probe + createTask + poll; secondary saw nothing.
    assert primary_client.post.await_count == 3
    assert secondary_client.post.await_count == 0
    assert wrapper.provider == "2captcha"


# ── 2: primary unreachable → wrapper routes to secondary ────────────────


@pytest.mark.asyncio
async def test_primary_unreachable_routes_to_secondary():
    """Primary's sticky ``_solver_unreachable`` flag (e.g. set by a prior
    fatal-config response) must steer the wrapper to the secondary on
    subsequent solves WITHOUT contacting the primary at all."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    # Pre-flip the primary's sticky unreachable flag and the
    # health-checked latch so the wrapper sees "primary out" without
    # firing a new probe.
    primary._solver_unreachable = True
    primary._solver_health_checked = True

    primary_client = _attach_client(primary, responses=[])
    secondary_client = _attach_client(secondary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-secondary"),
    ])

    page = _solve_page()
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    assert result.token == "tok-secondary"
    assert primary_client.post.await_count == 0
    assert secondary_client.post.await_count == 3
    # Provider property reflects the active (winning) solver so cost
    # accounting picks up the secondary's pricing tier.
    assert wrapper.provider == "capsolver"


# ── 3: primary breaker open → wrapper routes to secondary ───────────────


@pytest.mark.asyncio
async def test_primary_breaker_open_routes_to_secondary():
    """Trip the primary's breaker (3 failures in 5 min) and verify the
    wrapper bypasses primary on the next solve, going straight to
    secondary."""
    import time

    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    primary._solver_health_checked = True
    base = 1_000_000.0
    with patch("src.browser.captcha.time.time", return_value=base):
        for _ in range(3):
            await primary._record_solver_outcome(success=False)
        assert primary.is_breaker_open() is True

    primary_client = _attach_client(primary, responses=[])
    secondary_client = _attach_client(secondary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-secondary-2"),
    ])

    page = _solve_page()
    with patch("src.browser.captcha.time.time", return_value=base), \
         patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    assert result.token == "tok-secondary-2"
    assert primary_client.post.await_count == 0
    assert secondary_client.post.await_count == 3
    assert wrapper.provider == "capsolver"
    # Primary breaker still open, primary's local state untouched.
    with patch("src.browser.captcha.time.time", return_value=base):
        assert primary.is_breaker_open() is True
    _ = time  # silence unused-import in narrow scope


# ── 4: primary fatal-error mid-call → automatic retry on secondary ──────


@pytest.mark.asyncio
async def test_primary_fatal_mid_call_retries_on_secondary():
    """The §11.8 mid-call failover path: primary appears healthy at the
    routing decision, gets called, and the provider returns a
    fatal-config error (``ERROR_ZERO_BALANCE``) which flips
    ``_solver_unreachable``. The wrapper must retry on secondary inside
    the SAME ``solve()`` call so the agent gets a token instead of an
    immediate ``no_solver`` envelope."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    # Primary: getBalance succeeds, then createTask returns a fatal
    # config error (drains balance scenario). After this the
    # ``_handle_provider_error_response`` path inside ``_solve_2captcha``
    # flips ``_solver_unreachable=True``.
    primary_client = _attach_client(primary, responses=[
        _ok_balance_resp(),
        _stub_fatal_create(),
    ])
    secondary_client = _attach_client(secondary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-failover"),
    ])

    page = _solve_page()
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    # Token came from the secondary (failover succeeded).
    assert result.token == "tok-failover"
    # Primary issued the probe + the fatal createTask.
    assert primary_client.post.await_count == 2
    # Secondary issued probe + create + poll.
    assert secondary_client.post.await_count == 3
    # Primary's flag flipped; subsequent solves go straight to secondary.
    assert primary._solver_unreachable is True
    # Provider property now reflects the secondary (cost tier).
    assert wrapper.provider == "capsolver"

    # Subsequent solve: primary is now sticky-unreachable so the wrapper
    # short-circuits to secondary directly with no further primary
    # traffic.
    secondary_client.post = AsyncMock(side_effect=_stub_create_then_poll("tok-2"))
    primary_post_count_before = primary_client.post.await_count
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result2 = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )
    assert result2.token == "tok-2"
    assert primary_client.post.await_count == primary_post_count_before


# ── 5: both unreachable → no-token result, no provider HTTP ────────────


@pytest.mark.asyncio
async def test_both_unreachable_returns_no_token_no_http():
    """If both solvers are marked unreachable, the wrapper must NOT
    contact either provider. The result mirrors today's no-solver
    envelope so the caller surfaces ``solver_outcome="no_solver"``."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    primary._solver_unreachable = True
    primary._solver_health_checked = True
    secondary._solver_unreachable = True
    secondary._solver_health_checked = True

    primary_client = _attach_client(primary, responses=[])
    secondary_client = _attach_client(secondary, responses=[])

    page = _solve_page()
    result = await wrapper.solve(
        page, 'iframe[src*="recaptcha"]', "https://example.com",
    )

    assert isinstance(result, SolveResult)
    assert result.token is None
    assert result.injection_succeeded is False
    assert primary_client.post.await_count == 0
    assert secondary_client.post.await_count == 0

    # Wrapper-level gate also reports both-unreachable to the caller.
    assert await wrapper.is_solver_unreachable() is True


# ── 6: single-provider config → transparent pass-through ────────────────


@pytest.mark.asyncio
async def test_single_provider_config_transparent_passthrough():
    """``secondary=None`` collapses every wrapper method to the
    underlying primary's behavior. Failover paths become no-ops."""
    primary = _make_solver("2captcha")
    wrapper = MultiProviderSolver(primary, None)

    primary_client = _attach_client(primary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-only"),
    ])

    page = _solve_page()
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    assert result.token == "tok-only"
    assert wrapper.provider == "2captcha"
    assert primary_client.post.await_count == 3

    # is_solver_unreachable / is_breaker_open delegate to primary alone.
    assert await wrapper.is_solver_unreachable() is False
    primary._solver_unreachable = True
    primary._solver_health_checked = True
    assert await wrapper.is_solver_unreachable() is True

    assert wrapper.is_breaker_open() is False


# ── 7: cost accounting picks up secondary's provider tier ──────────────


@pytest.mark.asyncio
async def test_cost_accounting_uses_secondary_provider_after_failover():
    """When the secondary wins, ``wrapper.provider`` must equal the
    secondary's provider name so ``_metered_solve``'s pricing lookup
    (via ``estimate_millicents(provider, kind, ...)``) picks the right
    tier. This is the property exercised by every reservation /
    accounting test in ``test_check_captcha_metered.py``."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    # Pre-solve provider is the primary's by default — the
    # ``_metered_solve`` reservation reads ``solver.provider`` *before*
    # calling solve, and at that point we haven't routed yet. The pick
    # is updated as soon as a solve attempt is committed.
    assert wrapper.provider == "2captcha"

    # Mark primary out; the next ``_pick_solver`` call (driven by
    # ``solve``) updates the active solver to the secondary.
    primary._solver_unreachable = True
    primary._solver_health_checked = True
    secondary_client = _attach_client(secondary, responses=[
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-from-secondary"),
    ])
    page = _solve_page()
    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        result = await wrapper.solve(
            page, 'iframe[src*="recaptcha"]', "https://example.com",
        )

    assert result.token == "tok-from-secondary"
    # After the routing decision the property reflects the active
    # solver — this is what ``_metered_solve`` reads to pick the
    # accounting tier.
    assert wrapper.provider == "capsolver"
    _ = secondary_client


# ── 8: health_check folds both solvers into a worst-case verdict ───────


@pytest.mark.asyncio
async def test_health_check_runs_both_and_reports_worst():
    """The wrapper's ``health_check`` calls each underlying probe and
    returns the worst outcome the operator should see:

      * both healthy → ``healthy``
      * one degraded, one healthy → ``degraded``
      * both unreachable → ``unreachable``
      * one unreachable, one healthy → ``degraded`` (partial fault, not
        a total outage; operator wants to know the wrapper is still
        usable through the survivor).
    """
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    # Both healthy
    primary.health_check = AsyncMock(return_value="healthy")
    secondary.health_check = AsyncMock(return_value="healthy")
    assert await wrapper.health_check() == "healthy"

    # One degraded, one healthy → degraded
    primary.health_check = AsyncMock(return_value="degraded")
    secondary.health_check = AsyncMock(return_value="healthy")
    assert await wrapper.health_check() == "degraded"

    # Both unreachable → unreachable
    primary.health_check = AsyncMock(return_value="unreachable")
    secondary.health_check = AsyncMock(return_value="unreachable")
    assert await wrapper.health_check() == "unreachable"

    # One unreachable, one healthy → degraded (partial fault)
    primary.health_check = AsyncMock(return_value="unreachable")
    secondary.health_check = AsyncMock(return_value="healthy")
    assert await wrapper.health_check() == "degraded"

    # One unreachable, one degraded → degraded
    primary.health_check = AsyncMock(return_value="unreachable")
    secondary.health_check = AsyncMock(return_value="degraded")
    assert await wrapper.health_check() == "degraded"

    # Single-provider config — health_check defers to primary alone.
    wrapper_solo = MultiProviderSolver(primary, None)
    primary.health_check = AsyncMock(return_value="degraded")
    assert await wrapper_solo.health_check() == "degraded"


# ── 9: typo'd secondary provider → graceful single-provider fallback ───


_SOLVER_FLAG_KEYS = (
    "CAPTCHA_SOLVER_PROVIDER",
    "CAPTCHA_SOLVER_KEY",
    "CAPTCHA_SOLVER_PROVIDER_SECONDARY",
    "CAPTCHA_SOLVER_KEY_SECONDARY",
)


def _patch_env(values: dict[str, str]):
    """Patch ``os.environ`` for the four solver flag keys.

    ``flags.get_str`` reads through ``os.environ.get`` after walking the
    per-agent + operator-settings layers, so patching the environment
    is the simplest way to drive :func:`get_solver` end-to-end without
    touching the test settings file.
    """
    import os

    cleaned: dict[str, str] = {k: "" for k in _SOLVER_FLAG_KEYS}
    cleaned.update(values)
    return patch.dict(os.environ, cleaned, clear=False)


def test_get_solver_unsupported_secondary_falls_back_to_single():
    """An unrecognised ``CAPTCHA_SOLVER_PROVIDER_SECONDARY`` value
    (``foo``) MUST NOT silently degrade the primary or raise. The
    wrapper builds with ``secondary=None`` and behaves exactly like a
    single-provider deployment."""
    with _patch_env({
        "CAPTCHA_SOLVER_PROVIDER": "2captcha",
        "CAPTCHA_SOLVER_KEY": "PRIMARY-KEY",
        "CAPTCHA_SOLVER_PROVIDER_SECONDARY": "foo",   # typo
        "CAPTCHA_SOLVER_KEY_SECONDARY": "SECONDARY-KEY",
    }):
        wrapper = get_solver()

    assert isinstance(wrapper, MultiProviderSolver)
    assert wrapper.primary.provider == "2captcha"
    assert wrapper.secondary is None
    # Provider exposed for cost lookups still resolves cleanly.
    assert wrapper.provider == "2captcha"


def test_get_solver_no_secondary_yields_wrapper_with_secondary_none():
    """The common single-provider deployment path: only the primary
    flags are set; the wrapper is built with ``secondary=None`` and
    every failover path becomes a no-op."""
    with _patch_env({
        "CAPTCHA_SOLVER_PROVIDER": "capsolver",
        "CAPTCHA_SOLVER_KEY": "ONLY-KEY",
    }):
        wrapper = get_solver()

    assert isinstance(wrapper, MultiProviderSolver)
    assert wrapper.primary.provider == "capsolver"
    assert wrapper.secondary is None


def test_get_solver_no_provider_yields_none():
    """Neither slot configured → ``get_solver`` returns ``None`` so the
    BrowserManager skips wrapper construction entirely (existing
    behavior)."""
    with _patch_env({}):
        assert get_solver() is None


def test_get_solver_both_configured_yields_armed_wrapper():
    """Happy path: both slots set, both supported → wrapper has both
    primary and secondary populated."""
    with _patch_env({
        "CAPTCHA_SOLVER_PROVIDER": "2captcha",
        "CAPTCHA_SOLVER_KEY": "PRIMARY-KEY",
        "CAPTCHA_SOLVER_PROVIDER_SECONDARY": "capsolver",
        "CAPTCHA_SOLVER_KEY_SECONDARY": "SECONDARY-KEY",
    }):
        wrapper = get_solver()

    assert isinstance(wrapper, MultiProviderSolver)
    assert wrapper.primary.provider == "2captcha"
    assert wrapper.secondary is not None
    assert wrapper.secondary.provider == "capsolver"


# ── close() closes both underlying clients ────────────────────────────


@pytest.mark.asyncio
async def test_close_closes_both_underlying_clients():
    """``close`` must propagate to BOTH solvers so neither leaks a
    connection pool on shutdown."""
    primary = _make_solver("2captcha")
    secondary = _make_solver("capsolver", "SECRET-SECONDARY")
    wrapper = MultiProviderSolver(primary, secondary)

    primary.close = AsyncMock()
    secondary.close = AsyncMock()
    await wrapper.close()
    primary.close.assert_awaited_once()
    secondary.close.assert_awaited_once()

    # Single-provider variant: only primary is closed; secondary is
    # untouched (it doesn't exist).
    primary2 = _make_solver("2captcha")
    wrapper2 = MultiProviderSolver(primary2, None)
    primary2.close = AsyncMock()
    await wrapper2.close()
    primary2.close.assert_awaited_once()
