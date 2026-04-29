"""Tests for §11.16: solver health check + circuit breaker.

Covers:

* Per-process health-check gate: first solve runs the check, subsequent
  solves do not.
* Health outcomes: healthy → flow continues, degraded → flow continues
  with warning, unreachable → solver skipped for the session.
* Sliding-window breaker: 3 failures in 5 min trips for 10 min; auto-
  clears after; single/double failures + later success keep it closed;
  failures spaced beyond the window do not trip.
* Concurrent solves while breaker is open all short-circuit.
* URL/body redaction: ``clientKey`` never appears in log lines.
* Cancellation: in-flight ``health_check`` cancels cleanly on close.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser.captcha import (
    _BREAKER_FAILURE_THRESHOLD,
    _BREAKER_FAILURE_WINDOW,
    _BREAKER_OPEN_DURATION,
    _HEALTH_DEGRADED_LATENCY,
    CaptchaSolver,
    _redact_clientkey,
    _redact_clientkey_text,
)

# ── helpers ───────────────────────────────────────────────────────────


def _make_solver(provider: str = "2captcha", key: str = "SECRET-KEY-DO-NOT-LEAK") -> CaptchaSolver:
    return CaptchaSolver(provider, key)


def _ok_balance_resp(balance: float = 12.34) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"errorId": 0, "balance": balance})
    return resp


def _err_balance_resp(error_id: int = 1, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(
        return_value={"errorId": error_id, "errorDescription": "ERROR_KEY_DOES_NOT_EXIST"},
    )
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


def _solve_page() -> MagicMock:
    """Mock Playwright page that returns a sitekey + accepts injection."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value="site-key-abc")
    page.url = "https://example.com"
    return page


# ── 1: first solve triggers health check; subsequent do not ───────────


@pytest.mark.asyncio
async def test_first_solve_runs_health_check_subsequent_do_not():
    solver = _make_solver()
    page = _solve_page()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False

    # Sequence: getBalance probe → create → poll → (no second probe) →
    # create → poll.
    responses = [
        _ok_balance_resp(),
        *_stub_create_then_poll("tok-1"),
        *_stub_create_then_poll("tok-2"),
    ]
    client.post = AsyncMock(side_effect=responses)
    solver._client = client

    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        ok1 = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
        ok2 = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")

    assert bool(ok1) is True
    assert bool(ok2) is True
    urls_called = [c.args[0] for c in client.post.call_args_list]
    assert urls_called[0].endswith("/getBalance")
    # Only ONE getBalance — the gate is sticky.
    assert urls_called.count("https://api.2captcha.com/getBalance") == 1


# ── 2/3: healthy + degraded both let the flow continue ────────────────


@pytest.mark.asyncio
async def test_health_check_healthy_continues():
    solver = _make_solver()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(return_value=_ok_balance_resp())
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "healthy"


@pytest.mark.asyncio
async def test_health_check_degraded_continues_with_warning(caplog):
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False

    # Force a synthetic latency above the warn threshold by stubbing
    # time.monotonic so end-time = start + (threshold + 1).
    counter = {"n": 0}
    def fake_monotonic():
        counter["n"] += 1
        if counter["n"] == 1:
            return 0.0
        return _HEALTH_DEGRADED_LATENCY + 1.0

    client.post = AsyncMock(return_value=_ok_balance_resp())
    solver._client = client

    with patch("src.browser.captcha.time.monotonic", side_effect=fake_monotonic):
        with caplog.at_level(logging.WARNING, logger="browser.captcha"):
            outcome = await solver.health_check()

    assert outcome == "degraded"
    # Warning emitted for operator visibility.
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("degraded" in m for m in msgs)


# ── 4: unreachable → solver skipped without calling provider ──────────


@pytest.mark.asyncio
async def test_health_check_unreachable_short_circuits_subsequent_solves():
    solver = _make_solver()
    page = _solve_page()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    # First post (getBalance) raises a timeout; no further posts should happen.
    client.post = AsyncMock(side_effect=httpx.TimeoutException("probe timed out"))
    solver._client = client

    ok = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
    assert bool(ok) is False
    assert await solver.is_solver_unreachable() is True
    assert client.post.await_count == 1  # only the probe

    # Subsequent solves don't even probe again — the gate is sticky.
    ok2 = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
    assert bool(ok2) is False
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_concurrent_first_solves_share_one_health_check():
    """Three agents call solve() simultaneously on a fresh solver. The
    health-check gate must be inside an asyncio.Lock so they all wait
    for the SAME probe instead of each firing their own."""
    solver = _make_solver()
    page = _solve_page()

    probe_started = asyncio.Event()
    probe_release = asyncio.Event()
    probe_calls = 0

    async def fake_post(url, *args, **kwargs):
        nonlocal probe_calls
        if url.endswith("/getBalance"):
            probe_calls += 1
            probe_started.set()
            await probe_release.wait()
            return _ok_balance_resp()
        # createTask / getTaskResult — return success immediately.
        if "createTask" in url:
            r = MagicMock()
            r.json = MagicMock(return_value={"errorId": 0, "taskId": "T"})
            r.raise_for_status = MagicMock()
            return r
        r = MagicMock()
        r.json = MagicMock(return_value={
            "errorId": 0, "status": "ready",
            "solution": {"gRecaptchaResponse": "tok"},
        })
        r.raise_for_status = MagicMock()
        return r

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(side_effect=fake_post)
    solver._client = client

    with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        # Kick off three concurrent solves before the probe completes.
        tasks = [
            asyncio.create_task(
                solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com"),
            )
            for _ in range(3)
        ]

        # Wait until ONE probe is in flight, then release it. If the gate
        # were unlocked, multiple probes would race past the check.
        await asyncio.wait_for(probe_started.wait(), timeout=2.0)
        # Give the event loop a tick so any racing concurrent probe call
        # would have entered fake_post too.
        for _ in range(5):
            await asyncio.sleep(0)
        probe_release.set()

        results = await asyncio.gather(*tasks)

    assert all(bool(r) is True for r in results)
    assert probe_calls == 1, (
        f"expected exactly one health probe, got {probe_calls} — gate is racy"
    )


@pytest.mark.asyncio
async def test_health_check_5xx_treated_as_unreachable():
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False

    bad = MagicMock()
    bad.status_code = 503
    bad.json = MagicMock(return_value={})
    client.post = AsyncMock(return_value=bad)
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "unreachable"


@pytest.mark.asyncio
async def test_health_check_errorid_treated_as_unreachable():
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(return_value=_err_balance_resp(error_id=1, status=200))
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "unreachable"


@pytest.mark.asyncio
async def test_health_check_missing_balance_treated_as_unreachable():
    """Provider returned 200 + valid JSON but the expected ``balance``
    field is absent. We don't know what that response actually means —
    wrong endpoint, response-shape change, transparent proxy — so the
    only safe answer is 'unreachable'."""
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    weird = MagicMock()
    weird.status_code = 200
    weird.json = MagicMock(return_value={"errorId": 0, "foo": "bar"})
    client.post = AsyncMock(return_value=weird)
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "unreachable"


@pytest.mark.asyncio
async def test_health_check_non_numeric_balance_treated_as_unreachable():
    """Provider returned a balance field but it's not numeric (e.g. a
    string error message in place of a number). Treat as unreachable."""
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    weird = MagicMock()
    weird.status_code = 200
    weird.json = MagicMock(return_value={"errorId": 0, "balance": "n/a"})
    client.post = AsyncMock(return_value=weird)
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "unreachable"


@pytest.mark.asyncio
async def test_health_check_non_json_treated_as_unreachable():
    """Provider returned 200 with HTML / non-JSON body (rare but
    happens behind misconfigured CDNs). Treat as unreachable."""
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    weird = MagicMock()
    weird.status_code = 200
    weird.json = MagicMock(side_effect=ValueError("not JSON"))
    client.post = AsyncMock(return_value=weird)
    solver._client = client

    outcome = await solver.health_check()
    assert outcome == "unreachable"


# ── 5: three consecutive failures trip the breaker for 10 min ─────────


@pytest.mark.asyncio
async def test_three_consecutive_failures_trips_breaker():
    solver = _make_solver()
    solver._solver_health_checked = True  # skip the probe in this slice

    base = 1_000_000.0
    with patch("src.browser.captcha.time.time", return_value=base):
        for _ in range(_BREAKER_FAILURE_THRESHOLD):
            await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is True
        assert solver._solver_breaker_until == pytest.approx(
            base + _BREAKER_OPEN_DURATION,
        )


# ── 6: after 10 min, breaker auto-clears ──────────────────────────────


@pytest.mark.asyncio
async def test_breaker_auto_clears_after_window():
    solver = _make_solver()
    solver._solver_health_checked = True
    base = 1_000_000.0

    with patch("src.browser.captcha.time.time", return_value=base):
        for _ in range(_BREAKER_FAILURE_THRESHOLD):
            await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is True

    # Jump past the 10-min open duration.
    with patch(
        "src.browser.captcha.time.time",
        return_value=base + _BREAKER_OPEN_DURATION + 1,
    ):
        assert solver.is_breaker_open() is False

    # New solve attempts proceed (no breaker short-circuit).
    page = _solve_page()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(side_effect=_stub_create_then_poll("tok-X"))
    solver._client = client

    with patch(
        "src.browser.captcha.time.time",
        return_value=base + _BREAKER_OPEN_DURATION + 1,
    ), patch("src.browser.captcha._POLL_INTERVAL", 0.001):
        ok = await solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")

    assert bool(ok) is True
    # createTask + getTaskResult, no probe (already checked).
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_breaker_auto_clear_resets_failure_window():
    """After auto-clear, the failure-window deque + breaker timestamp
    must both be reset so a single new failure doesn't immediately
    re-trip on stale entries."""
    solver = _make_solver()
    solver._solver_health_checked = True
    base = 1_000_000.0

    with patch("src.browser.captcha.time.time", return_value=base):
        for _ in range(_BREAKER_FAILURE_THRESHOLD):
            await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is True
        assert len(solver._solver_failure_timestamps) == _BREAKER_FAILURE_THRESHOLD

    # Reading is_breaker_open() AFTER the window expires must clear state.
    with patch(
        "src.browser.captcha.time.time",
        return_value=base + _BREAKER_OPEN_DURATION + 1,
    ):
        assert solver.is_breaker_open() is False
        assert solver._solver_breaker_until == 0.0
        assert len(solver._solver_failure_timestamps) == 0


# ── 7: a single failure does NOT trip the breaker ─────────────────────


@pytest.mark.asyncio
async def test_single_failure_does_not_trip():
    solver = _make_solver()
    solver._solver_health_checked = True
    await solver._record_solver_outcome(success=False)
    assert solver.is_breaker_open() is False


# ── 8: two failures + a success → counter resets ──────────────────────


@pytest.mark.asyncio
async def test_success_resets_breaker_counter():
    solver = _make_solver()
    solver._solver_health_checked = True
    base = 1_000_000.0

    with patch("src.browser.captcha.time.time", return_value=base):
        await solver._record_solver_outcome(success=False)
        await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is False

        # Success wipes the window.
        await solver._record_solver_outcome(success=True)
        assert len(solver._solver_failure_timestamps) == 0

        # A subsequent failure starts fresh — one isn't enough.
        await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is False
        assert len(solver._solver_failure_timestamps) == 1


# ── 9: failures spaced beyond the 5-min window do not trip ────────────


@pytest.mark.asyncio
async def test_failures_outside_window_do_not_trip():
    solver = _make_solver()
    solver._solver_health_checked = True
    base = 1_000_000.0
    far_future = base + _BREAKER_FAILURE_WINDOW + 60

    with patch("src.browser.captcha.time.time", return_value=base):
        await solver._record_solver_outcome(success=False)
        await solver._record_solver_outcome(success=False)
    # Third failure happens AFTER the 5-min window expires for the first
    # two. The deque should prune them and keep only the new entry.
    with patch("src.browser.captcha.time.time", return_value=far_future):
        await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is False
        assert len(solver._solver_failure_timestamps) == 1


@pytest.mark.asyncio
async def test_three_failures_staggered_inside_window_does_trip():
    """Failures at t=0, t=200, t=400 — all three are inside a 300-s
    sliding window viewed from t=400 (cutoff = 100, only t=0 falls
    out). Deque keeps [200, 400] after prune + new append → len 2 < 3
    → breaker NOT tripped. Then a fourth failure at t=450 prunes the
    same way (cutoff=150, drops 0 if still there but it's already
    gone): deque = [200, 400, 450], len = 3 → breaker DOES trip.

    This case lives just on the boundary of the sliding-window math —
    test it explicitly so a future deque-trim regression is caught."""
    solver = _make_solver()
    solver._solver_health_checked = True
    base = 1_000_000.0

    with patch("src.browser.captcha.time.time", return_value=base):
        await solver._record_solver_outcome(success=False)  # t=0
    with patch("src.browser.captcha.time.time", return_value=base + 200):
        await solver._record_solver_outcome(success=False)  # t=200
    with patch("src.browser.captcha.time.time", return_value=base + 400):
        # cutoff = 400 - 300 = 100; t=0 pruned, t=200/400 stay → len 2
        await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is False
        assert len(solver._solver_failure_timestamps) == 2

    with patch("src.browser.captcha.time.time", return_value=base + 450):
        # cutoff = 450 - 300 = 150; t=200/400/450 all stay → len 3 → trip
        await solver._record_solver_outcome(success=False)
        assert solver.is_breaker_open() is True


# ── 10: concurrent solves while breaker open all short-circuit ────────


@pytest.mark.asyncio
async def test_concurrent_solves_short_circuit_when_breaker_open():
    solver = _make_solver()
    solver._solver_health_checked = True

    # Trip the breaker.
    for _ in range(_BREAKER_FAILURE_THRESHOLD):
        await solver._record_solver_outcome(success=False)
    assert solver.is_breaker_open() is True

    page = _solve_page()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock()  # any call here is a bug
    solver._client = client

    results = await asyncio.gather(*(
        solver.solve(page, 'iframe[src*="recaptcha"]', "https://example.com")
        for _ in range(5)
    ))
    assert all(bool(r) is False for r in results)
    # No provider HTTP calls — breaker gate ran before any post().
    assert client.post.await_count == 0


# ── 11: URL/body redaction — clientKey never reaches the logger ───────


@pytest.mark.asyncio
async def test_solve_log_redacts_page_url_query(caplog):
    raw_url = "https://example.com/login?clientKey=SECRET-CLIENT-KEY&session=abc123"
    solver = _make_solver()
    solver._solver_health_checked = True
    page = AsyncMock()

    with (
        patch(
            "src.browser.captcha._classify_recaptcha",
            AsyncMock(return_value={
                "variant": "recaptcha-v2-checkbox",
                "sitekey": None,
                "action": None,
            }),
        ),
        patch.object(solver, "_extract_sitekey", AsyncMock(return_value=None)),
        caplog.at_level(logging.INFO, logger="browser.captcha"),
    ):
        await solver.solve(page, 'iframe[src*="recaptcha"]', raw_url)

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "SECRET-CLIENT-KEY" not in rendered
    assert "session=abc123" not in rendered
    assert "clientKey=" in rendered
    assert "session=" in rendered


@pytest.mark.asyncio
async def test_health_check_logs_redact_clientkey(caplog):
    raw_key = "SECRET-KEY-DO-NOT-LEAK-1234567890"
    solver = _make_solver(key=raw_key)

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    bad = MagicMock()
    bad.status_code = 503
    bad.json = MagicMock(return_value={})
    client.post = AsyncMock(return_value=bad)
    solver._client = client

    with caplog.at_level(logging.WARNING, logger="browser.captcha"):
        outcome = await solver.health_check()
    assert outcome == "unreachable"

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert raw_key not in joined


@pytest.mark.asyncio
async def test_health_check_timeout_log_redacts_clientkey(caplog):
    raw_key = "SECRET-KEY-DO-NOT-LEAK-1234567890"
    solver = _make_solver(key=raw_key)

    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    # Provider sometimes echoes clientKey into error strings — simulate.
    client.post = AsyncMock(side_effect=httpx.HTTPError(
        f"connect failed using clientKey={raw_key}",
    ))
    solver._client = client

    with caplog.at_level(logging.WARNING, logger="browser.captcha"):
        outcome = await solver.health_check()
    assert outcome == "unreachable"

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert raw_key not in joined


@pytest.mark.asyncio
async def test_health_check_request_body_logging_redacts_clientkey(caplog):
    """Verify the timeout-path log line scrubs the clientKey from the body."""
    raw_key = "SECRET-KEY-DO-NOT-LEAK-1234567890"
    solver = _make_solver(key=raw_key)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False
    client.post = AsyncMock(side_effect=httpx.TimeoutException("hung"))
    solver._client = client

    with caplog.at_level(logging.WARNING, logger="browser.captcha"):
        await solver.health_check()

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert raw_key not in joined
    assert "[REDACTED]" in joined


def test_redact_clientkey_dict_helper():
    raw = {"clientKey": "SECRET", "taskId": "T-1"}
    redacted = _redact_clientkey(raw)
    assert redacted["clientKey"] == "[REDACTED]"
    assert redacted["taskId"] == "T-1"
    # Original untouched (returned a copy).
    assert raw["clientKey"] == "SECRET"

    # Idempotent on dicts that don't have the key.
    no_key = {"foo": "bar"}
    assert _redact_clientkey(no_key) is no_key


def test_redact_clientkey_text_helper():
    secret = "ABC123-XYZ"

    # Every shape we've seen (or expect to see) in solver-provider error
    # output, request logs, and Python-side repr() / f-string traces.
    shapes = [
        f"connect failed using clientKey={secret}",                 # query/form
        f"clientKey: {secret}",                                     # header
        f'{{"clientKey": "{secret}", "errorId": 1}}',                # JSON
        f"{{'clientKey': '{secret}', 'errorId': 1}}",                 # Python repr
        f"clientKey={secret}&otherParam=foo",                       # mid-query
        f'CLIENTKEY="{secret}"',                                     # case-insensitive
    ]
    for shape in shapes:
        out = _redact_clientkey_text(shape)
        assert secret not in out, f"leaked in {shape!r}: {out!r}"
        assert "[REDACTED]" in out

    # No-op on safe strings.
    assert _redact_clientkey_text("plain text") == "plain text"
    assert _redact_clientkey_text("") == ""


def test_redact_clientkey_text_strips_taskid():
    """§11.14 / §11.15: solver task IDs (UUIDs and integers) appear in
    error responses. The shared redactor strips them so a hostile
    provider error containing a stitched-together secret-like string
    can't leak via the taskId path.

    2Captcha returns integer task IDs ("taskId": 9876543210); CapSolver
    returns UUID-shape strings. Both forms must be scrubbed.
    """
    shapes_with_uuid = [
        'taskId=8d2c1f3a-aaaa-4444-bbbb-1234567890ab',
        '"taskId": "8d2c1f3a-aaaa-4444-bbbb-1234567890ab"',
        "taskId: 9999999999",
        "TASKID='abcdef-0123456789'",
    ]
    for shape in shapes_with_uuid:
        out = _redact_clientkey_text(shape)
        assert "[REDACTED]" in out, f"taskId not redacted in {shape!r}: {out!r}"

    # Pure integer-form (2Captcha-style) — explicit coverage. Both quoted
    # and unquoted shapes must be scrubbed.
    integer_shapes = [
        'taskId=9876543210',
        'taskId="9876543210"',
        '{"errorId": 1, "taskId": 9876543210}',
        "got back taskId: 9876543210 from provider",
    ]
    for shape in integer_shapes:
        out = _redact_clientkey_text(shape)
        assert "9876543210" not in out, (
            f"integer taskId leaked in {shape!r}: {out!r}"
        )
        assert "[REDACTED]" in out


# ── 12: cancellation cleans up in-flight health_check ─────────────────


@pytest.mark.asyncio
async def test_health_check_cancellation_cleans_up():
    solver = _make_solver()
    client = AsyncMock(spec=httpx.AsyncClient)
    client.is_closed = False

    # Block forever inside the post() to simulate a hung probe.
    started = asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        await asyncio.sleep(60)

    client.post = AsyncMock(side_effect=hang)
    solver._client = client

    task = asyncio.create_task(solver.health_check())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    # Cancel mid-flight.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # close() should cleanly tear down even after cancellation.
    await solver.close()


# ── _check_captcha integration: surfaces breaker state ────────────────


@pytest.mark.asyncio
async def test_check_captcha_emits_no_solver_when_unreachable():
    """When health-check unreachable, _check_captcha decorates the result
    with solver_outcome='no_solver' + next_action='request_captcha_help'."""
    from src.browser.captcha import SolveResult as _SR
    from src.browser.service import BrowserManager

    mgr = BrowserManager.__new__(BrowserManager)
    fake_solver = AsyncMock()
    fake_solver.solve = AsyncMock(return_value=_SR(
        token=None, injection_succeeded=False,
        used_proxy_aware=False, compat_rejected=False,
    ))
    # ``is_solver_unreachable`` is now async (lazy probe).
    fake_solver.is_solver_unreachable = AsyncMock(return_value=True)
    fake_solver.is_breaker_open = MagicMock(return_value=False)
    mgr._captcha_solver = fake_solver

    inst = MagicMock()
    inst.page = AsyncMock()
    inst.page.url = "https://example.com"
    loc_match = MagicMock()
    loc_match.count = AsyncMock(return_value=1)
    loc_zero = MagicMock()
    loc_zero.count = AsyncMock(return_value=0)
    def by_sel(s):
        if s == 'iframe[src*="recaptcha"]':
            return loc_match
        return loc_zero
    inst.page.locator = MagicMock(side_effect=by_sel)

    out = await mgr._check_captcha(inst)
    assert out is not None
    # §11.13 envelope shape — captcha_found is the structural marker.
    assert out.get("captcha_found") is True
    assert out.get("solver_attempted") is False
    assert out.get("solver_outcome") == "no_solver"
    assert out.get("next_action") == "request_captcha_help"
    # ``kind`` and ``solver_confidence`` come from the envelope builder.
    assert "kind" in out
    assert "solver_confidence" in out


@pytest.mark.asyncio
async def test_check_captcha_emits_breaker_open_flag():
    """When breaker is open, _check_captcha sets breaker_open=True
    and solver_outcome='timeout' (the §11.13 follow-up will fold this
    into a structured ``service_unavailable`` value)."""
    from src.browser.captcha import SolveResult as _SR
    from src.browser.service import BrowserManager

    mgr = BrowserManager.__new__(BrowserManager)
    fake_solver = AsyncMock()
    fake_solver.solve = AsyncMock(return_value=_SR(
        token=None, injection_succeeded=False,
        used_proxy_aware=False, compat_rejected=False,
    ))
    fake_solver.is_solver_unreachable = AsyncMock(return_value=False)
    fake_solver.is_breaker_open = MagicMock(return_value=True)
    mgr._captcha_solver = fake_solver

    inst = MagicMock()
    inst.page = AsyncMock()
    inst.page.url = "https://example.com"
    loc_match = MagicMock()
    loc_match.count = AsyncMock(return_value=1)
    loc_zero = MagicMock()
    loc_zero.count = AsyncMock(return_value=0)
    def by_sel(s):
        if s == 'iframe[src*="recaptcha"]':
            return loc_match
        return loc_zero
    inst.page.locator = MagicMock(side_effect=by_sel)

    out = await mgr._check_captcha(inst)
    assert out is not None
    # §11.13 envelope + §11.16 additive breaker_open flag.
    assert out.get("captcha_found") is True
    assert out.get("solver_attempted") is False
    assert out.get("solver_outcome") == "timeout"
    assert out.get("breaker_open") is True
    assert out.get("next_action") == "request_captcha_help"


@pytest.mark.asyncio
async def test_check_captcha_no_decoration_when_solver_healthy():
    """Healthy solver, ordinary detect-and-solve returns the §11.13
    envelope WITHOUT a top-level ``breaker_open`` flag and with a
    ``solver_outcome`` other than ``no_solver`` (proves the §11.16
    short-circuits did not fire)."""
    from src.browser.captcha import SolveResult as _SR
    from src.browser.service import BrowserManager

    mgr = BrowserManager.__new__(BrowserManager)
    fake_solver = AsyncMock()
    # No-token result → "rejected" envelope on a healthy solver.
    fake_solver.solve = AsyncMock(return_value=_SR(
        token=None, injection_succeeded=False,
        used_proxy_aware=False, compat_rejected=False,
    ))
    fake_solver.is_solver_unreachable = AsyncMock(return_value=False)
    fake_solver.is_breaker_open = MagicMock(return_value=False)
    mgr._captcha_solver = fake_solver

    inst = MagicMock()
    inst.page = AsyncMock()
    inst.page.url = "https://example.com"
    loc_match = MagicMock()
    loc_match.count = AsyncMock(return_value=1)
    loc_zero = MagicMock()
    loc_zero.count = AsyncMock(return_value=0)
    def by_sel(s):
        if s == 'iframe[src*="recaptcha"]':
            return loc_match
        return loc_zero
    inst.page.locator = MagicMock(side_effect=by_sel)

    out = await mgr._check_captcha(inst)
    assert out is not None
    assert out.get("captcha_found") is True
    # No breaker decoration on a healthy solver.
    assert "breaker_open" not in out
    # Envelope still carries solver_outcome — but it must not be
    # "no_solver" (that's the unreachable short-circuit). With
    # solve()==False on a healthy solver the envelope reports "rejected".
    assert out.get("solver_outcome") != "no_solver"


# ── §11.16 Codex F3 — local failures must NOT pollute the breaker ──────


class TestBreakerLocalVsProviderFailures:
    """Codex F3 — pre-fix, ANY failure path inside ``solve()`` —
    including purely-local classification failures (sitekey extraction
    couldn't find the widget, ``_build_task_body`` rejected the variant)
    — recorded a breaker outcome. Three unsupported captchas from one
    agent tripped the breaker for the entire BrowserManager and blocked
    real solves for every other agent. The fix splits the failure paths:
    only provider-contacted failures count toward the breaker.
    """

    @pytest.mark.asyncio
    async def test_three_sitekey_extract_failures_do_not_trip_breaker(self):
        """Sitekey extraction failure is purely local — the page DOM
        lacked any matching marker. The provider was never contacted.
        The breaker must NOT count these.
        """
        solver = _make_solver()
        solver._solver_health_checked = True

        page = AsyncMock()
        # Sitekey extractor returns None — our DOM-walker found nothing.
        page.evaluate = AsyncMock(return_value=None)
        page.url = "https://example.com"

        # No HTTP client patching needed — the provider is never reached.
        for _ in range(_BREAKER_FAILURE_THRESHOLD):
            await solver.solve(
                page, 'iframe[src*="recaptcha"]', "https://example.com",
                kind="recaptcha-v2-checkbox",
            )

        assert len(solver._solver_failure_timestamps) == 0
        assert solver.is_breaker_open() is False

    @pytest.mark.asyncio
    async def test_three_unsupported_variant_failures_do_not_trip_breaker(self):
        """``_build_task_body`` returns ``None`` when the captcha kind
        isn't in the per-provider task table. That's a LOCAL failure —
        no createTask call ever fires. Breaker must not count these
        either, otherwise three unsupported challenges from one agent
        would block solves for the whole manager.
        """
        solver = _make_solver(provider="2captcha")
        solver._solver_health_checked = True

        page = _solve_page()  # sitekey extracted fine
        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False
        # If the test fails the wrong way, the provider would get hit.
        client.post = AsyncMock(side_effect=AssertionError(
            "provider must NOT be contacted for an unsupported variant",
        ))
        solver._client = client

        # Force ``_build_task_body`` to act as if the captcha kind is
        # missing from the provider table — return ``None``. This is
        # the same path an unrecognized captcha takes through
        # ``_solve_2captcha`` / ``_solve_capsolver``.
        with patch.object(
            solver, "_build_task_body",
            return_value=(None, False, False),
        ):
            for _ in range(_BREAKER_FAILURE_THRESHOLD):
                result = await solver.solve(
                    page, 'iframe[src*="recaptcha"]',
                    "https://example.com",
                    kind="recaptcha-v2-checkbox",
                )
                # Each call fails (no token) but provider was untouched.
                assert result.token is None

        assert len(solver._solver_failure_timestamps) == 0
        assert solver.is_breaker_open() is False
        # Provider really wasn't contacted.
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_three_provider_500_failures_DO_trip_breaker(self):
        """Provider-contacted failures (createTask 5xx, errorId>0,
        timeouts during polling) MUST still trip the breaker — that's
        the signal it was designed for."""
        solver = _make_solver(provider="2captcha")
        solver._solver_health_checked = True

        page = _solve_page()

        # createTask returns errorId>0 — provider WAS contacted.
        err_resp = MagicMock()
        err_resp.json = MagicMock(return_value={
            "errorId": 1, "errorDescription": "ERROR_KEY_DOES_NOT_EXIST",
        })
        err_resp.raise_for_status = MagicMock()

        client = AsyncMock(spec=httpx.AsyncClient)
        client.is_closed = False
        client.post = AsyncMock(return_value=err_resp)
        solver._client = client

        with patch("src.browser.captcha._POLL_INTERVAL", 0.001):
            for _ in range(_BREAKER_FAILURE_THRESHOLD):
                await solver.solve(
                    page, 'iframe[src*="recaptcha"]',
                    "https://example.com",
                    kind="recaptcha-v2-checkbox",
                )

        # Breaker SHOULD have tripped — these were real provider
        # failures, not local classification gaps.
        assert solver.is_breaker_open() is True
