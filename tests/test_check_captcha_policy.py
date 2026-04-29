"""Tests for §11.18 site-policy wiring inside ``BrowserManager._check_captcha``.

Phase 8 §11.18 ships :mod:`src.browser.captcha_policy` and these tests
verify it is actually consumed by the runtime — closing finding H5 from
the post-merge review (the module shipped DORMANT in PR #767, with no
caller, so operators setting ``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS`` /
``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS`` saw zero behavior change).

Coverage:

* ``unsolvable`` policy short-circuits before any solver call. Envelope
  matches the §11.3 behavioral path: ``solver_attempted=False``,
  ``solver_outcome="skipped_behavioral"``,
  ``solver_confidence="behavioral-only"``,
  ``next_action="request_captcha_help"``.  Solver mock is NOT called.
* ``low_success`` policy DOES call the solver, but on a SUCCESSFUL solve
  the envelope reports ``solver_confidence="low"`` (token-IP binding
  makes the solve unreliable even when the verdict was "good").
* ``low_success`` policy on a FAILED solve upgrades ``next_action`` to
  ``"request_captcha_help"`` (NOT ``"notify_user"``) AND tags the
  envelope with ``low_success_failed=True`` so operators see the case
  distinctly in the audit log.
* ``default`` policy is unchanged from current behavior.
* Operator override ``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS=accounts.google.com``
  neutralizes the hardcoded ``low_success`` and produces a normal solve.
* Operator override ``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS=example.com``
  forces a default site to ``unsolvable``.
* The policy classification is recorded on the audit-log event so the
  dashboard can show WHY a solve was skipped.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import captcha_cost_counter as cost
from src.browser import captcha_policy
from src.browser import service as svc
from src.browser.captcha import SolveResult
from src.browser.service import BrowserManager, CamoufoxInstance

# ── helpers ────────────────────────────────────────────────────────────────


def _solved() -> SolveResult:
    return SolveResult(
        token="tok",
        injection_succeeded=True,
        used_proxy_aware=False,
        compat_rejected=False,
    )


def _rejected() -> SolveResult:
    """Provider returned no token — solver effectively failed."""
    return SolveResult(
        token=None,
        injection_succeeded=False,
        used_proxy_aware=False,
        compat_rejected=False,
    )


def _mk_inst(*, page_url: str, agent_id: str = "agent-1") -> CamoufoxInstance:
    page = MagicMock()
    page.url = page_url
    locator = MagicMock()
    locator.count = AsyncMock(return_value=1)
    page.locator = MagicMock(return_value=locator)
    return CamoufoxInstance(agent_id, MagicMock(), MagicMock(), page)


def _mk_solver(*, return_value: SolveResult) -> MagicMock:
    s = MagicMock()
    s.provider = "2captcha"
    s.solve = AsyncMock(return_value=return_value)
    s.is_solver_unreachable = AsyncMock(return_value=False)
    s.is_breaker_open = MagicMock(return_value=False)
    return s


def _reload_policy_with_env(env: dict[str, str]):
    """Reload :mod:`src.browser.captcha_policy` with the given env applied
    AND re-bind ``svc.captcha_policy`` to the freshly-imported module.

    The service module imports captcha_policy by name (``from
    src.browser import captcha_policy``), so reloading the policy module
    in-place leaves ``svc.captcha_policy`` pointing at the OLD module
    object with stale env-var-derived caches. We refresh both.
    """
    with mock.patch.dict(os.environ, env, clear=False):
        fresh = importlib.reload(captcha_policy)
    # Rebind the name inside the service module so ``_check_captcha``'s
    # ``captcha_policy.get_site_policy(...)`` call resolves to the fresh
    # module.  Without this the test sees the unreloaded copy.
    svc.captcha_policy = fresh
    return fresh


@pytest.fixture(autouse=True)
async def _isolate_state(tmp_path, monkeypatch):
    """Reset cost / rate / audit state between tests."""
    monkeypatch.setenv(
        "CAPTCHA_COST_COUNTER_PATH", str(tmp_path / "captcha_costs.json"),
    )
    await cost.reset()
    svc._solve_rate_window.clear()
    async with svc._get_captcha_audit_lock():
        svc._captcha_audit_buckets.clear()
    yield
    await cost.reset()
    svc._solve_rate_window.clear()
    async with svc._get_captcha_audit_lock():
        svc._captcha_audit_buckets.clear()
    # Reset captcha_policy back to the ambient (no-override) state and
    # rebind on the service module so subsequent suites see pristine env.
    scrubbed = {
        k: v for k, v in os.environ.items()
        if k not in {
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS",
        }
    }
    with mock.patch.dict(os.environ, scrubbed, clear=True):
        fresh = importlib.reload(captcha_policy)
    svc.captcha_policy = fresh


@pytest.fixture()
def mgr(tmp_path):
    return BrowserManager(profiles_dir=str(tmp_path / "profiles"))


# ── 1. unsolvable policy → short-circuit, no solver call ─────────────────


class TestUnsolvablePolicy:
    @pytest.mark.asyncio
    async def test_hardcoded_unsolvable_short_circuits(self, mgr):
        """``challenges.cloudflare.com`` is hardcoded as unsolvable —
        solver must not be called even when configured & healthy."""
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        # CF host triggers the hardcoded UNSOLVABLE bucket.
        inst = _mk_inst(
            page_url="https://challenges.cloudflare.com/cdn-cgi/x",
        )

        envelope = await mgr._check_captcha(inst)
        assert envelope["captcha_found"] is True
        assert envelope["solver_attempted"] is False
        assert envelope["solver_outcome"] == "skipped_behavioral"
        assert envelope["solver_confidence"] == "behavioral-only"
        assert envelope["next_action"] == "request_captcha_help"
        # Solver mock NEVER awaited — gate fired before any HTTP call.
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skip_solve_env_overrides_default(self, mgr, monkeypatch):
        """``OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS=example.com`` — a domain
        that would otherwise be ``default`` is forced to ``unsolvable``."""
        _reload_policy_with_env({
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "example.com",
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "",
        })
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst(page_url="https://example.com/protected")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "skipped_behavioral"
        assert envelope["next_action"] == "request_captcha_help"
        solver.solve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_audit_event_records_policy_classification(
        self, tmp_path, monkeypatch,
    ):
        """The audit-log event MUST include ``policy="unsolvable"`` so the
        dashboard can show operators WHY a solve was skipped."""
        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_solved())
        inst = _mk_inst(page_url="https://www.humansecurity.com/")
        m._instances["agent-1"] = inst

        await m._check_captcha(inst)
        await m._emit_metrics()

        captcha_events = [
            e for e in events if e.get("type") == "captcha_gate"
        ]
        assert len(captcha_events) == 1, captcha_events
        ev = captcha_events[0]
        assert ev["outcome"] == "skipped_behavioral"
        assert ev["policy"] == "unsolvable"


# ── 2. low_success policy: solver IS called, confidence forced to low ────


class TestLowSuccessPolicy:
    @pytest.mark.asyncio
    async def test_solved_envelope_forced_low_confidence(self, mgr, monkeypatch):
        """``low_success`` host (accounts.google.com) — solver is called,
        but a SUCCESSFUL solve still surfaces ``solver_confidence="low"``
        because token-IP binding makes the solve unreliable even when
        the verdict is good."""
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        # accounts.google.com is hardcoded LOW_SUCCESS.
        inst = _mk_inst(page_url="https://accounts.google.com/signup")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_attempted"] is True
        assert envelope["solver_outcome"] == "solved"
        # Forced-low even though the underlying SolveResult would normally
        # give "high" (compat_rejected=False, used_proxy_aware=False).
        assert envelope["solver_confidence"] == "low"
        # Solver WAS called — low_success is NOT a short-circuit.
        solver.solve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_failed_solve_upgrades_next_action_to_request_captcha_help(
        self, mgr,
    ):
        """``low_success`` host + provider returned no token →
        ``solver_outcome="rejected"`` per the existing flow, but
        ``next_action`` is upgraded from ``notify_user`` to
        ``request_captcha_help`` (escalate after first failure)."""
        solver = _mk_solver(return_value=_rejected())
        mgr._captcha_solver = solver
        inst = _mk_inst(page_url="https://twitter.com/i/flow/signup")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "rejected"
        # The KEY assertion: NOT "notify_user".
        assert envelope["next_action"] == "request_captcha_help"
        # And the audit-distinctive flag.
        assert envelope.get("low_success_failed") is True

    @pytest.mark.asyncio
    async def test_low_success_failed_audit_event_emitted(
        self, tmp_path,
    ):
        """The low_success-failed path emits a dashboard audit event with
        outcome=``low_success_failed`` and ``policy="low_success"``."""
        events: list[dict] = []
        m = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=events.append,
        )
        m._captcha_solver = _mk_solver(return_value=_rejected())
        inst = _mk_inst(page_url="https://www.linkedin.com/login")
        m._instances["agent-1"] = inst

        await m._check_captcha(inst)
        # Drain — but the audit event is fired-and-forget via
        # asyncio.create_task inside ``_finalize``; let the task run.
        await asyncio.sleep(0)
        await m._emit_metrics()

        ls_events = [
            e for e in events
            if e.get("type") == "captcha_gate"
            and e.get("outcome") == "low_success_failed"
        ]
        assert len(ls_events) == 1, [
            e for e in events if e.get("type") == "captcha_gate"
        ]
        assert ls_events[0]["policy"] == "low_success"

    @pytest.mark.asyncio
    async def test_force_solve_env_overrides_low_success(self, mgr, monkeypatch):
        """``OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS=accounts.google.com`` —
        operator forces a normal solve flow on a hardcoded ``low_success``
        host. The envelope reports the solver's actual confidence (no
        forced-low override)."""
        _reload_policy_with_env({
            "OPENLEGION_CAPTCHA_FORCE_SOLVE_DOMAINS": "accounts.google.com",
            "OPENLEGION_CAPTCHA_SKIP_SOLVE_DOMAINS": "",
        })
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst(page_url="https://accounts.google.com/signup")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "solved"
        # The override neutralizes ``low_success`` — confidence reflects the
        # underlying solve verdict (compat_rejected=False → "high").
        assert envelope["solver_confidence"] == "high"
        assert envelope["next_action"] == "solved"
        assert "low_success_failed" not in envelope


# ── 3. default policy: unchanged from current behavior ────────────────────


class TestDefaultPolicy:
    @pytest.mark.asyncio
    async def test_default_policy_solved_high_confidence(self, mgr):
        """A vanilla site → policy=default → unchanged flow."""
        solver = _mk_solver(return_value=_solved())
        mgr._captcha_solver = solver
        inst = _mk_inst(page_url="https://example.com/protected")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "solved"
        assert envelope["solver_confidence"] == "high"
        assert envelope["next_action"] == "solved"
        assert "low_success_failed" not in envelope

    @pytest.mark.asyncio
    async def test_default_policy_rejected_keeps_notify_user(self, mgr):
        """A vanilla site + failed solve → ``next_action`` stays
        ``notify_user`` (default policy does NOT escalate to
        ``request_captcha_help``)."""
        solver = _mk_solver(return_value=_rejected())
        mgr._captcha_solver = solver
        inst = _mk_inst(page_url="https://example.com/protected")

        envelope = await mgr._check_captcha(inst)
        assert envelope["solver_outcome"] == "rejected"
        assert envelope["next_action"] == "notify_user"
        assert "low_success_failed" not in envelope
