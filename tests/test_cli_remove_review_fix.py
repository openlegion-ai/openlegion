"""Regression tests for M11 + M12 (Phase 0-5 integration review).

M12 — CLI ``/remove`` had no operator guard, so ``/remove operator`` + confirm
would offboard and destroy the human's front-door agent; the mesh (403/400)
and dashboard ("system agent") delete surfaces both refuse it.

M11 — CLI ``/remove`` skipped the per-agent data cleanup the mesh and dashboard
delete paths run (vault / private blackboard namespace / cost+trace rows /
wallet + ``permissions.reload()``), so a same-name recreate inherited the
deleted agent's wallet and private namespace.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class _FakeCtx:
    """Minimal stand-in for ``RuntimeContext`` — only the attributes
    ``_cmd_remove`` actually touches."""

    def __init__(self, agents):
        self.agents = dict(agents)
        self.runtime = MagicMock()
        self.router = MagicMock()
        self.transport = MagicMock()
        self.health_monitor = MagicMock()
        self.pubsub = MagicMock()
        self.cron_scheduler = MagicMock()
        self.cron_scheduler.remove_agent_jobs.return_value = 0
        self.lane_manager = MagicMock()
        self.connector_store = None
        self.event_bus = None
        self.offboard_agent = None
        self.cleanup_agent = MagicMock()
        self._dispatch_loop = None

    @property
    def dispatch_loop(self):
        return self._dispatch_loop


def _make_session(ctx):
    from src.cli.repl import REPLSession

    session = REPLSession.__new__(REPLSession)  # bypass __init__ (readline setup)
    session.ctx = ctx
    session.current = next(iter(ctx.agents), None)
    return session


def _patch_confirm_and_remove_agent(monkeypatch, confirm=True):
    monkeypatch.setattr("click.confirm", lambda *a, **k: confirm)
    monkeypatch.setattr("src.cli.config._remove_agent", lambda name, **kw: None)


class TestOperatorGuardM12:
    def test_remove_operator_by_name_is_refused(self, monkeypatch):
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx({"operator": {}, "scout": {}})
        session = _make_session(ctx)
        session._cmd_remove("operator")
        # Never stopped, never cleaned up — the guard returns before any teardown.
        ctx.runtime.stop_agent.assert_not_called()
        ctx.cleanup_agent.assert_not_called()

    def test_interactive_picker_excludes_operator(self, monkeypatch):
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx({"operator": {}, "scout": {}})
        session = _make_session(ctx)
        # Empty arg → picker. With operator excluded only ``scout`` remains, so
        # it auto-selects (len==1) and never prompts for the operator.
        session._cmd_remove("")
        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)

    def test_picker_with_only_operator_removes_nothing(self, monkeypatch):
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx({"operator": {}})
        session = _make_session(ctx)
        session._cmd_remove("")
        ctx.runtime.stop_agent.assert_not_called()
        ctx.cleanup_agent.assert_not_called()


class TestCleanupParityM11:
    def test_remove_runs_per_agent_cleanup(self, monkeypatch):
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx({"scout": {}})
        session = _make_session(ctx)
        session._cmd_remove("scout")
        # The mesh/dashboard-parity teardown must run for the removed agent.
        ctx.cleanup_agent.assert_called_once_with("scout")

    def test_cleanup_seam_unwired_is_tolerated(self, monkeypatch):
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx({"scout": {}})
        ctx.cleanup_agent = None  # mesh app not wired (degraded case)
        session = _make_session(ctx)
        # Must not raise even without the seam.
        session._cmd_remove("scout")
        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)
