"""CLI REPL ``/remove`` (plan §8 #15).

Bug-fix regression: ``/remove`` used to call ``stop_agent(name)`` WITHOUT
``remove_data=True`` — the agent's ``openlegion_data_*`` volume leaked
forever. Fixed here to ``remove_data=True`` (H12 parity with the mesh and
dashboard delete paths), with an offboarding-with-handover best-effort
attempt (``self.ctx.offboard_agent``, reached via the runtime context's
dispatch loop) ATTEMPTED first — the data-loss invariant every delete
surface must honor.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock

import pytest


class _FakeCtx:
    """Minimal stand-in for ``RuntimeContext`` — only the attributes
    ``_cmd_remove`` actually touches."""

    def __init__(self, dispatch_loop=None):
        self.agents = {"scout": {}}
        self.runtime = MagicMock()
        self.router = MagicMock()
        self.transport = MagicMock()
        self.health_monitor = MagicMock()
        self.pubsub = MagicMock()
        self.cron_scheduler = MagicMock()
        self.lane_manager = MagicMock()
        self.connector_store = None
        self.event_bus = None
        self.offboard_agent = None
        self._dispatch_loop = dispatch_loop

    @property
    def dispatch_loop(self):
        return self._dispatch_loop


def _make_session(ctx):
    from src.cli.repl import REPLSession

    session = REPLSession.__new__(REPLSession)  # bypass __init__ (readline setup)
    session.ctx = ctx
    session.current = next(iter(ctx.agents), None)
    return session


@pytest.fixture
def bg_loop():
    """A real background asyncio loop — mirrors ``RuntimeContext``'s own
    dispatch loop / thread, so ``asyncio.run_coroutine_threadsafe`` works
    exactly as it does in the live REPL."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)


def _patch_confirm_and_remove_agent(monkeypatch):
    monkeypatch.setattr("click.confirm", lambda *a, **k: True)
    monkeypatch.setattr("src.cli.config._remove_agent", lambda name, **kw: None)


class TestRemoveVolumeWipeBugFix:
    def test_remove_wipes_volume_remove_data_true(self, monkeypatch):
        """The bug: ``stop_agent`` used to be called WITHOUT
        ``remove_data`` — the volume leaked forever."""
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx()
        session = _make_session(ctx)
        session._cmd_remove("scout")
        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)

    def test_remove_without_offboard_wiring_prints_preferred_surface_note(self, monkeypatch, capsys):
        """Acceptable-minimum path: no ``offboard_agent``/dispatch_loop
        wired — still wipes the volume, but tells the operator dashboard/
        operator offboarding is preferred."""
        _patch_confirm_and_remove_agent(monkeypatch)
        ctx = _FakeCtx(dispatch_loop=None)
        session = _make_session(ctx)
        session._cmd_remove("scout")
        out = capsys.readouterr().out
        assert "dashboard" in out.lower() or "manage_agent" in out.lower()
        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)


class TestRemoveOffboardOrderProof:
    def test_offboard_attempted_before_volume_destruction(self, monkeypatch, bg_loop):
        """ORDER PROOF: when the offboard helper IS wired (the clean-
        wiring path attempted first), it must run strictly before
        ``stop_agent(..., remove_data=True)``."""
        _patch_confirm_and_remove_agent(monkeypatch)
        order: list[tuple] = []

        async def fake_offboard(agent_id, *, reason):
            order.append(("offboard", agent_id, reason))
            return {"handover_committed": True, "errors": []}

        ctx = _FakeCtx(dispatch_loop=bg_loop)
        ctx.offboard_agent = fake_offboard
        ctx.runtime.stop_agent = MagicMock(
            side_effect=lambda name, remove_data=False: order.append(("stop_agent", name, remove_data))
        )
        session = _make_session(ctx)
        session._cmd_remove("scout")

        assert order == [
            ("offboard", "scout", "delete"),
            ("stop_agent", "scout", True),
        ], f"offboard must precede volume destruction, got order={order}"

    def test_offboard_failure_is_best_effort_removal_still_proceeds(self, monkeypatch, bg_loop, capsys):
        _patch_confirm_and_remove_agent(monkeypatch)

        async def failing_offboard(agent_id, *, reason):
            raise RuntimeError("mesh unreachable")

        ctx = _FakeCtx(dispatch_loop=bg_loop)
        ctx.offboard_agent = failing_offboard
        session = _make_session(ctx)
        session._cmd_remove("scout")

        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)
        out = capsys.readouterr().out
        assert "warning" in out.lower()

    def test_offboard_manifest_errors_surfaced_but_removal_proceeds(self, monkeypatch, bg_loop, capsys):
        _patch_confirm_and_remove_agent(monkeypatch)

        async def offboard_with_errors(agent_id, *, reason):
            return {"handover_committed": False, "errors": ["team drive quota exceeded"]}

        ctx = _FakeCtx(dispatch_loop=bg_loop)
        ctx.offboard_agent = offboard_with_errors
        session = _make_session(ctx)
        session._cmd_remove("scout")

        ctx.runtime.stop_agent.assert_called_once_with("scout", remove_data=True)
        out = capsys.readouterr().out
        assert "quota exceeded" in out.lower()
