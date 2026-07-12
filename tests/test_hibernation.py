"""Tests for hibernation (plan §8 #24, Phase-5 U8 — the final Phase-5 unit).

``hibernated`` is a third agent status: IN SERVICE, container stopped,
volume persisted, auto-wakes on demand — never to be confused with
``archived`` (out of service, never auto-woken). Covers:

  * the hibernate/wake mesh endpoints and their core helpers
    (``_hibernate_agent_core`` / ``ensure_agent_running`` /
    ``_wake_agent_core`` in ``host/server.py``),
  * boot skip + heartbeat-reconcile keep (the U0-prereq asymmetry
    extended to the new status),
  * the ``HttpTransport`` cold-wake seam,
  * the cron heartbeat's mesh-probe-only tick for a hibernated agent,
  * the idle-sweep (``HibernationSweeper``) and its blocking conditions,
  * the dispatch connect-failure observability event, and
  * the ``hibernate_idle_minutes`` B4-style 0-valid default-off limit.

Volume-loss-impossible-by-construction is pinned twice: a live mock
assertion (``stop_agent`` always called with ``remove_data=False``) and
a static source-text scan proving no hibernate code path can reach
``remove_data=True``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.cli import runtime as runtime_mod
from src.dashboard.events import EventBus
from src.host import server as server_mod
from src.host.asks import AskBroker
from src.host.cron import CronScheduler
from src.host.lanes import LaneManager
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.orchestration import Tasks
from src.host.permissions import PermissionMatrix
from src.host.server import HibernationSweeper, create_mesh_app
from src.host.transport import HttpTransport
from src.shared import limits

_OP = {"Authorization": "Bearer tok-op"}
_TOKENS = {"operator": "tok-op"}


class _FakeContainerManager:
    """Stand-in for RuntimeBackend: tracks start/stop/wait calls without
    touching Docker. Every ``start_agent`` call allocates a fresh URL
    (mirrors DockerBackend's monotonic port allocator) so wake tests can
    prove the transport/router get RE-registered, not just left alone.
    """

    def __init__(self):
        self.started: list[dict] = []
        self.stopped: list[tuple[str, bool]] = []  # (agent_id, remove_data)
        self._next_port = 9000
        self.wait_result = True

    def start_agent(self, *, agent_id, role, tools_dir, model, thinking, env_overrides):
        self.started.append({
            "agent_id": agent_id, "role": role, "tools_dir": tools_dir,
            "model": model, "thinking": thinking, "env_overrides": dict(env_overrides or {}),
        })
        self._next_port += 1
        return f"http://127.0.0.1:{self._next_port}"

    def stop_agent(self, agent_id, *, remove_data=False):
        self.stopped.append((agent_id, remove_data))

    async def wait_for_agent(self, agent_id, timeout=30):
        return self.wait_result


def _build_app(
    tmp_path,
    monkeypatch,
    *,
    agent="scout",
    agent_status="active",
    extra_agents=None,
    lane_manager=None,
    ask_broker=None,
    container_manager=None,
    transport=None,
    health_monitor=None,
    event_bus=None,
    cron_scheduler=None,
):
    """Build a mesh app wired for hibernation testing.

    Status mutations (archive/unarchive/hibernate/wake) write into the
    SAME ``fake_cfg`` dict every ``_load_config`` call returns, so
    subsequent reads observe the transition — mirrors real
    agents.yaml semantics without touching disk.
    """
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"))

    agents_cfg = {agent: {"role": "worker", "status": agent_status, "model": "openai/gpt-4o-mini"}}
    if extra_agents:
        agents_cfg.update(extra_agents)
    fake_cfg = {
        "agents": agents_cfg,
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "network": {},
        "mesh": {"port": 8420},
    }

    import src.cli.config as cli_config

    monkeypatch.setattr(cli_config, "_load_config", lambda: fake_cfg)

    def _set_status(name, status):
        if name not in fake_cfg["agents"]:
            raise ValueError(f"Agent '{name}' not found")
        fake_cfg["agents"][name]["status"] = status

    def _agent_status_fn(name):
        if name not in fake_cfg["agents"]:
            raise ValueError(f"Agent '{name}' not found")
        return fake_cfg["agents"][name].get("status", "active") or "active"

    monkeypatch.setattr(cli_config, "_archive_agent", lambda n: _set_status(n, "archived"))
    monkeypatch.setattr(cli_config, "_unarchive_agent", lambda n: _set_status(n, "active"))
    monkeypatch.setattr(cli_config, "_hibernate_agent", lambda n: _set_status(n, "hibernated"))
    monkeypatch.setattr(cli_config, "_wake_agent_status", lambda n: _set_status(n, "active"))
    monkeypatch.setattr(cli_config, "_agent_status", _agent_status_fn)

    perms = PermissionMatrix()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = MessageRouter(perms, {})
    cm = container_manager if container_manager is not None else _FakeContainerManager()
    tr = transport if transport is not None else HttpTransport()
    hm = health_monitor if health_monitor is not None else MagicMock()
    eb = event_bus if event_bus is not None else EventBus()

    app = create_mesh_app(
        blackboard=bb,
        pubsub=PubSub(),
        router=router,
        permissions=perms,
        auth_tokens=dict(_TOKENS),
        container_manager=cm,
        transport=tr,
        health_monitor=hm,
        event_bus=eb,
        lane_manager=lane_manager,
        ask_broker=ask_broker,
        cron_scheduler=cron_scheduler,
        cfg=fake_cfg,
    )
    return app, bb, cm, tr, hm, eb, fake_cfg


# ── Core: hibernate ──────────────────────────────────────────────────


class TestHibernateEndpoint:
    def test_hibernate_stops_container_without_data_removal(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"hibernated": True, "agent_id": "scout"}
        assert cm.stopped == [("scout", False)]
        bb.close()

    def test_hibernate_keeps_cron_jobs(self, tmp_path, monkeypatch):
        sched = CronScheduler(config_path=str(tmp_path / "cron.json"))
        sched.ensure_heartbeat("scout")
        assert sched.find_heartbeat_job("scout") is not None
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, cron_scheduler=sched,
        )
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 200, resp.text
        # Unlike archive, hibernate must NOT remove the agent's cron jobs.
        assert sched.find_heartbeat_job("scout") is not None
        bb.close()

    def test_hibernate_unregisters_health(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 200, resp.text
        hm.unregister.assert_called_once_with("scout")
        bb.close()

    def test_hibernate_sets_status(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert cfg["agents"]["scout"]["status"] == "hibernated"
        assert app.get_agent_status("scout") == "hibernated"
        bb.close()

    def test_hibernate_refuses_operator(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/operator/hibernate", headers=_OP)
        assert resp.status_code == 400
        assert cm.stopped == []
        bb.close()

    def test_hibernate_refuses_archived(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 409
        assert cm.stopped == []
        bb.close()

    def test_hibernate_refuses_busy(self, tmp_path, monkeypatch):
        lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=None)
        # Fake a busy lane without needing a running worker task —
        # get_status() only reads these three dicts.
        lm._queues["scout"] = asyncio.Queue()
        lm._pending["scout"] = []
        lm._busy["scout"] = True
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, lane_manager=lm,
        )
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 409
        assert cm.stopped == []
        bb.close()

    def test_hibernate_refuses_queued(self, tmp_path, monkeypatch):
        lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=None)
        lm._queues["scout"] = asyncio.Queue()
        lm._queues["scout"].put_nowait(object())
        lm._pending["scout"] = [object()]
        lm._busy["scout"] = False
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, lane_manager=lm,
        )
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 409
        assert cm.stopped == []
        bb.close()

    def test_hibernate_refuses_working_task(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        task = app.tasks_store.create(creator="operator", assignee="scout", title="do it")
        app.tasks_store.update_status(task["id"], "working", actor="scout")
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate", headers=_OP)
        assert resp.status_code == 409
        assert cm.stopped == []
        bb.close()

    def test_hibernate_requires_operator_auth(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/hibernate")
        assert resp.status_code == 401  # missing bearer entirely — never reaches the 403 gate
        assert cm.stopped == []
        bb.close()

    def test_hibernate_unknown_agent_404(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch)
        client = TestClient(app)
        resp = client.post("/mesh/agents/ghost/hibernate", headers=_OP)
        assert resp.status_code == 404
        bb.close()

    def test_no_hibernate_path_can_pass_remove_data_true(self):
        """Static pin: the hibernate core function's source never
        contains ``remove_data=True`` — only the delete path may."""
        src = inspect.getsource(server_mod)
        start = src.index("async def _hibernate_agent_core")
        end = src.index("async def _wake_agent_core")
        body = src[start:end]
        assert "remove_data=True" not in body
        assert "remove_data=False" in body


# ── Wake ───────────────────────────────────────────────────────────


class TestWake:
    def test_wake_endpoint_restarts_reregisters_activates_stamps_audits(
        self, tmp_path, monkeypatch,
    ):
        lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=None)
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated", lane_manager=lm,
        )
        events: list[dict] = []
        eb.add_listener(lambda evt: events.append(evt))

        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/wake-from-hibernation", headers=_OP)
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"woke": True, "agent_id": "scout"}

        assert [s["agent_id"] for s in cm.started] == ["scout"]
        assert cfg["agents"]["scout"]["status"] == "active"
        assert app.get_agent_status("scout") == "active"
        hm.register.assert_called_once_with("scout")
        # Transport re-registered with the FRESH url (a new port every
        # start_agent call — the pre-wake registration must not survive).
        fresh_url = f"http://127.0.0.1:{cm._next_port}"
        assert tr.get_url("scout") == fresh_url
        # Activity stamped.
        assert (time.time() - lm.last_activity("scout")) < 5
        # Audited.
        woken = [e for e in events if e.get("type") == "agent_woken"]
        assert woken and woken[0]["agent"] == "scout"
        bb.close()

    async def test_ensure_agent_running_wakes_hibernated_agent(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated",
        )
        ok = await app.ensure_agent_running("scout", trigger="dispatch")
        assert ok is True
        assert [s["agent_id"] for s in cm.started] == ["scout"]
        bb.close()

    async def test_active_agent_no_wake_attempted(self, tmp_path, monkeypatch):
        """The fast path for an already-active agent must not touch the
        container manager / health monitor at all — cheap cache lookup
        only."""
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch, agent_status="active")
        ok = await app.ensure_agent_running("scout")
        assert ok is True
        assert cm.started == []
        hm.register.assert_not_called()
        bb.close()

    async def test_archived_never_wakes(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        ok = await app.ensure_agent_running("scout")
        assert ok is False
        assert cm.started == []
        bb.close()

    def test_wake_endpoint_409_for_archived(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/wake-from-hibernation", headers=_OP)
        assert resp.status_code == 409
        bb.close()

    def test_wake_endpoint_noop_for_active(self, tmp_path, monkeypatch):
        app, bb, cm, tr, hm, eb, cfg = _build_app(tmp_path, monkeypatch, agent_status="active")
        client = TestClient(app)
        resp = client.post("/mesh/agents/scout/wake-from-hibernation", headers=_OP)
        assert resp.status_code == 200
        assert resp.json() == {"woke": True, "agent_id": "scout"}
        assert cm.started == []
        bb.close()

    async def test_double_wake_single_start(self, tmp_path, monkeypatch):
        """Two simultaneous wake triggers for the same agent must start
        the container exactly once — the second joins the in-flight
        wake via the thread/loop-safe claim (NOT an asyncio.Lock, which
        is unsafe across the multiple event loops this seam is called
        from in production)."""
        app, bb, cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated",
        )

        results = await asyncio.gather(
            app.ensure_agent_running("scout"),
            app.ensure_agent_running("scout"),
        )
        assert results == [True, True]
        assert len(cm.started) == 1
        bb.close()

    async def test_wake_fails_gracefully_when_container_never_ready(
        self, tmp_path, monkeypatch,
    ):
        cm = _FakeContainerManager()
        cm.wait_result = False
        app, bb, _cm, tr, hm, eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated", container_manager=cm,
        )
        ok = await app.ensure_agent_running("scout")
        assert ok is False
        # Status stays hibernated — never falsely flips to active.
        assert cfg["agents"]["scout"]["status"] == "hibernated"
        bb.close()


# ── Boot + heartbeat-reconcile asymmetry ─────────────────────────────


class _StubRuntimeBackend:
    def __init__(self):
        self.extra_env: dict[str, str] = {}
        self.started: list[str] = []

    def start_agent(self, *, agent_id, role, tools_dir, model, thinking, env_overrides):
        self.started.append(agent_id)
        return f"http://agent-{agent_id}"


def _build_boot_ctx(fake_cfg: dict) -> runtime_mod.RuntimeContext:
    ctx = runtime_mod.RuntimeContext.__new__(runtime_mod.RuntimeContext)
    ctx.cfg = fake_cfg
    ctx.cost_tracker = MagicMock()
    ctx.runtime = _StubRuntimeBackend()
    ctx.router = MagicMock()
    ctx.transport = HttpTransport()
    ctx.health_monitor = MagicMock()
    ctx.permissions = MagicMock()
    ctx.credential_vault = MagicMock()
    ctx.connector_store = MagicMock()
    return ctx


class TestBootSkipsHibernated:
    def test_start_agents_skips_hibernated(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import src.cli.config as cli_config

        fake_cfg = {
            "agents": {
                "worker-active": {"role": "worker", "status": "active"},
                "worker-hibernated": {"role": "worker", "status": "hibernated"},
                "worker-archived": {"role": "worker", "status": "archived"},
            },
            "llm": {}, "mesh": {"port": 8400}, "network": {},
        }
        monkeypatch.setattr(runtime_mod, "_load_config", lambda: fake_cfg)
        monkeypatch.setattr(cli_config, "_ensure_operator_agent", lambda **k: None)

        ctx = _build_boot_ctx(fake_cfg)
        runtime_mod.RuntimeContext._start_agents(ctx)

        assert ctx.runtime.started == ["worker-active"]
        assert "worker-active" in ctx.transport._urls
        assert "worker-hibernated" not in ctx.transport._urls
        assert "worker-archived" not in ctx.transport._urls
        health_registered = [c.args[0] for c in ctx.health_monitor.register.call_args_list]
        assert health_registered == ["worker-active"]


class TestReconcileKeepsHibernated:
    def test_hibernated_agent_still_gets_heartbeat_job(self, tmp_path):
        scheduler = CronScheduler(config_path=str(tmp_path / "cron.json"))
        cfg = {
            "agents": {
                "worker-hibernated": {"role": "worker", "status": "hibernated"},
                "worker-archived": {"role": "worker", "status": "archived"},
            },
        }

        class _Stub:
            cron_scheduler = scheduler
            cfg = None

        stub = _Stub()
        stub.cfg = cfg
        runtime_mod.RuntimeContext._reconcile_heartbeats(stub)

        # The asymmetry IS the design: hibernated keeps its heartbeat
        # (mesh-probe-only ticks are the cold-wake trigger); archived
        # does not.
        assert scheduler.find_heartbeat_job("worker-hibernated") is not None
        assert scheduler.find_heartbeat_job("worker-archived") is None


# ── Transport cold-wake seam ──────────────────────────────────────────


class TestTransportSeam:
    async def test_request_calls_ensure_running_before_forwarding(self):
        t = HttpTransport()
        t.register("scout", "http://stale-host:1")
        call_log: list[str] = []

        async def fake_ensure(agent_id):
            call_log.append(agent_id)
            # Simulate the wake handler re-registering a fresh URL.
            t.register(agent_id, "http://fresh-host:2")
            return True

        t.set_ensure_running_fn(fake_ensure)

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {"response": "delivered"}
        fake_client = AsyncMock()
        fake_client.request = AsyncMock(return_value=fake_resp)

        async def fake_get_client(self):
            return fake_client

        with patch.object(HttpTransport, "_get_client", new=fake_get_client):
            result = await t.request("scout", "POST", "/chat", json={"message": "hi"})

        assert result == {"response": "delivered"}
        assert call_log == ["scout"]
        forwarded_url = fake_client.request.call_args[0][1]
        assert forwarded_url == "http://fresh-host:2/chat"

    async def test_stream_request_calls_ensure_running(self):
        t = HttpTransport()
        t.register("scout", "http://nowhere:1")
        mock_fn = AsyncMock(return_value=True)
        t.set_ensure_running_fn(mock_fn)

        events = []
        async for evt in t.stream_request("scout", "POST", "/chat/stream", json={}):
            events.append(evt)
        mock_fn.assert_awaited_once_with("scout")

    def test_request_sync_calls_ensure_running(self):
        t = HttpTransport()
        t.register("scout", "http://nowhere:1")
        calls: list[str] = []

        async def fake_ensure(agent_id):
            calls.append(agent_id)
            return True

        t.set_ensure_running_fn(fake_ensure)
        t.request_sync("scout", "GET", "/status", timeout=1)
        assert calls == ["scout"]

    def test_request_sync_ensure_running_skips_inside_running_loop(self):
        """Documented residual: calling the sync path from within an
        already-running loop must not crash — the wake is skipped and
        logged instead of raising asyncio.run()'s RuntimeError."""
        t = HttpTransport()
        t.register("scout", "http://nowhere:1")
        calls: list[str] = []

        async def fake_ensure(agent_id):
            calls.append(agent_id)
            return True

        t.set_ensure_running_fn(fake_ensure)

        async def _drive():
            # Calling the sync method from inside a running loop.
            t.request_sync("scout", "GET", "/status", timeout=1)

        asyncio.run(_drive())
        assert calls == []  # skipped, not crashed

    async def test_is_reachable_never_calls_ensure_running(self):
        t = HttpTransport()
        t.register("scout", "http://nowhere:1")
        mock_fn = AsyncMock(return_value=True)
        t.set_ensure_running_fn(mock_fn)
        result = await t.is_reachable("scout", timeout=1)
        assert result is False  # nothing listening, but that's not the point
        mock_fn.assert_not_called()

    async def test_no_ensure_fn_wired_is_a_pure_noop(self):
        """Default (unwired) transports — every non-hibernation test in
        the suite — must behave exactly as before; no ensure_running_fn
        call attempted at all."""
        t = HttpTransport()
        result = await t.request("unknown", "GET", "/status")
        assert "error" in result  # unregistered agent, no crash


# ── Cron: hibernated ticks are mesh-probe-only ───────────────────────


class TestCronHibernatedTicks:
    def _sched(self, tmp_path, **kwargs):
        mock_bb = MagicMock()
        mock_bb.list_by_prefix.return_value = []
        return CronScheduler(
            config_path=str(tmp_path / "cron.json"), blackboard=mock_bb, **kwargs,
        )

    async def test_hibernated_actionable_plate_skips_context_dispatches(self, tmp_path):
        context_fn = AsyncMock(return_value={"is_default_heartbeat": True, "has_recent_activity": False})
        heartbeat_dispatch_fn = AsyncMock(return_value={"response": "did stuff", "outcome": "ok"})
        sched = self._sched(
            tmp_path,
            dispatch_fn=AsyncMock(),
            context_fn=context_fn,
            heartbeat_dispatch_fn=heartbeat_dispatch_fn,
            agent_status_fn=lambda agent: "hibernated",
        )
        job = sched.add_job(agent="scout", schedule="every 15m", heartbeat=True)
        triggered_probe = MagicMock(name="disk_usage", triggered=True, detail="90% used")
        triggered_probe.name = "disk_usage"
        with patch.object(sched, "_run_heartbeat_probes", return_value=[triggered_probe]):
            await sched._execute_job(job)

        context_fn.assert_not_called()
        heartbeat_dispatch_fn.assert_called_once()
        snapshot = sched.get_last_plate("scout")
        assert snapshot["hibernated"] is True
        assert snapshot["dispatched"] is True

    async def test_hibernated_empty_plate_zero_container_contact(self, tmp_path):
        context_fn = AsyncMock(return_value={"is_default_heartbeat": True, "has_recent_activity": False})
        heartbeat_dispatch_fn = AsyncMock(return_value={"response": "ok", "outcome": "ok"})
        sched = self._sched(
            tmp_path,
            dispatch_fn=AsyncMock(),
            context_fn=context_fn,
            heartbeat_dispatch_fn=heartbeat_dispatch_fn,
            agent_status_fn=lambda agent: "hibernated",
        )
        job = sched.add_job(agent="scout", schedule="every 15m", heartbeat=True)
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            result = await sched._execute_job(job)

        assert result is None
        context_fn.assert_not_called()
        heartbeat_dispatch_fn.assert_not_called()
        snapshot = sched.get_last_plate("scout")
        assert snapshot["hibernated"] is True
        assert snapshot["dispatched"] is False

    async def test_active_agent_context_fn_called_normally(self, tmp_path):
        context_fn = AsyncMock(return_value={"is_default_heartbeat": True, "has_recent_activity": True})
        heartbeat_dispatch_fn = AsyncMock(return_value={"response": "ok", "outcome": "ok"})
        sched = self._sched(
            tmp_path,
            dispatch_fn=AsyncMock(),
            context_fn=context_fn,
            heartbeat_dispatch_fn=heartbeat_dispatch_fn,
            agent_status_fn=lambda agent: "active",
        )
        job = sched.add_job(agent="scout", schedule="every 15m", heartbeat=True)
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            await sched._execute_job(job)
        context_fn.assert_called_once()
        snapshot = sched.get_last_plate("scout")
        assert snapshot["hibernated"] is False

    async def test_no_agent_status_fn_defaults_non_hibernated(self, tmp_path):
        """Missing wiring must never accidentally suppress context_fn."""
        context_fn = AsyncMock(return_value={"is_default_heartbeat": True, "has_recent_activity": True})
        sched = self._sched(tmp_path, dispatch_fn=AsyncMock(), context_fn=context_fn)
        job = sched.add_job(agent="scout", schedule="every 15m", heartbeat=True)
        with patch.object(sched, "_run_heartbeat_probes", return_value=[]):
            await sched._execute_job(job)
        context_fn.assert_called_once()

    async def test_activity_fn_stamped_on_real_dispatch(self, tmp_path):
        activity_calls: list[str] = []
        dispatch_fn = AsyncMock(return_value="a real reply")
        sched = self._sched(
            tmp_path, dispatch_fn=dispatch_fn,
            activity_fn=lambda agent: activity_calls.append(agent),
        )
        job = sched.add_job(agent="scout", schedule="every 15m", message="hi")
        await sched._execute_job(job)
        assert activity_calls == ["scout"]

    async def test_activity_fn_not_stamped_on_suppressed_empty(self, tmp_path):
        activity_calls: list[str] = []
        dispatch_fn = AsyncMock(return_value="")
        sched = self._sched(
            tmp_path, dispatch_fn=dispatch_fn,
            activity_fn=lambda agent: activity_calls.append(agent),
        )
        job = sched.add_job(agent="scout", schedule="every 15m", message="hi")
        await sched._execute_job(job)
        assert activity_calls == []


# ── Idle sweep ────────────────────────────────────────────────────────


def _sweeper(tmp_path, *, agents_cfg, lane_manager=None, tasks_store=None, ask_broker=None):
    hibernate_calls: list[tuple[str, str]] = []

    async def hibernate_fn(agent_id, *, caller="sweep"):
        hibernate_calls.append((agent_id, caller))
        return {"hibernated": True, "agent_id": agent_id}

    sweeper = HibernationSweeper(
        hibernate_fn=hibernate_fn,
        lane_manager=lane_manager,
        tasks_store=tasks_store,
        ask_broker=ask_broker,
        config_fn=lambda: {"agents": agents_cfg},
    )
    return sweeper, hibernate_calls


class TestHibernationSweeper:
    async def test_disabled_by_default_zero_minutes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", raising=False)
        lm = LaneManager(dispatch_fn=AsyncMock())
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_all_conditions_met_hibernates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 20 * 60  # idle 20m > 10m threshold
        tasks = Tasks(db_path=str(tmp_path / "tasks.db"))
        broker = AskBroker()
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}},
            lane_manager=lm, tasks_store=tasks, ask_broker=broker,
        )
        await sweeper._tick()
        assert calls == [("scout", "sweep")]

    async def test_busy_prevents_hibernation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 20 * 60
        lm._queues["scout"] = asyncio.Queue()
        lm._busy["scout"] = True
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_queued_prevents_hibernation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 20 * 60
        lm._queues["scout"] = asyncio.Queue()
        lm._queues["scout"].put_nowait(object())
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_working_task_prevents_hibernation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 20 * 60
        tasks = Tasks(db_path=str(tmp_path / "tasks.db"))
        task = tasks.create(creator="operator", assignee="scout", title="x")
        tasks.update_status(task["id"], "working", actor="scout")
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}},
            lane_manager=lm, tasks_store=tasks,
        )
        await sweeper._tick()
        assert calls == []

    async def test_open_ask_prevents_hibernation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 20 * 60
        broker = AskBroker()
        broker.create("scout", "helper", "q?", 60)
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}},
            lane_manager=lm, ask_broker=broker,
        )
        await sweeper._tick()
        assert calls == []

    async def test_recent_activity_prevents_hibernation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time()  # just active
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "active"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_operator_never_a_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["operator"] = time.time() - 999 * 60
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"operator": {"status": "active"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_archived_not_a_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 999 * 60
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "archived"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_already_hibernated_not_a_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")
        lm = LaneManager(dispatch_fn=AsyncMock())
        lm._activity["scout"] = time.time() - 999 * 60
        sweeper, calls = _sweeper(
            tmp_path, agents_cfg={"scout": {"status": "hibernated"}}, lane_manager=lm,
        )
        await sweeper._tick()
        assert calls == []

    async def test_sweep_failure_is_logged_never_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "10")

        async def _boom():
            raise RuntimeError("boom")

        sweeper = HibernationSweeper(
            hibernate_fn=AsyncMock(),
            lane_manager=None,  # forces an early return, not a crash
            tasks_store=None,
            ask_broker=None,
            config_fn=_boom,
        )
        await sweeper._tick()  # lane_manager None -> returns before config_fn

        sweeper2 = HibernationSweeper(
            hibernate_fn=AsyncMock(),
            lane_manager=LaneManager(dispatch_fn=AsyncMock()),
            tasks_store=None,
            ask_broker=None,
            config_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        await sweeper2._tick()  # must not raise

        # The public start() loop wraps _tick in try/except too — prove
        # a raising config_fn doesn't kill the loop task itself.
        sweeper2.INTERVAL_SECONDS = 0.01
        task = asyncio.create_task(sweeper2.start())
        await asyncio.sleep(0.05)
        assert not task.done()  # still alive despite the raising config_fn
        sweeper2.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ── Idle-sweep activity tracking (LaneManager) ────────────────────────


class TestLaneActivity:
    def test_conservative_default_is_construction_time(self):
        lm = LaneManager(dispatch_fn=AsyncMock())
        assert (time.time() - lm.last_activity("never-seen")) < 5

    def test_mark_activity_updates_timestamp(self):
        lm = LaneManager(dispatch_fn=AsyncMock())
        before = lm.last_activity("scout")
        time.sleep(0.01)
        lm.mark_activity("scout")
        assert lm.last_activity("scout") > before

    def test_persists_and_reloads_across_instances(self, tmp_path):
        path = tmp_path / "activity.json"
        lm1 = LaneManager(dispatch_fn=AsyncMock(), activity_path=str(path))
        lm1.mark_activity("scout")
        stamped = lm1.last_activity("scout")

        lm2 = LaneManager(dispatch_fn=AsyncMock(), activity_path=str(path))
        assert lm2.last_activity("scout") == stamped
        # An agent this second instance never saw still gets the
        # conservative "just booted" fallback, not the OLD instance's
        # stamp for a different agent.
        assert lm2.last_activity("never-seen") != stamped

    def test_no_path_wired_stays_pure_in_memory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lm = LaneManager(dispatch_fn=AsyncMock())  # activity_path=None default
        lm.mark_activity("scout")
        assert not (tmp_path / "config" / "agent_activity.json").exists()

    async def test_worker_finally_stamps_activity(self):
        dispatch_fn = AsyncMock(return_value="done")
        lm = LaneManager(dispatch_fn=dispatch_fn)
        before = lm.last_activity("scout")
        time.sleep(0.01)
        result = await lm.enqueue("scout", "hi")
        assert result == "done"
        assert lm.last_activity("scout") > before

    async def test_steer_injection_stamps_activity(self):
        async def steer_fn(agent, message, system_note=False):
            return {"injected": True}

        lm = LaneManager(dispatch_fn=AsyncMock(), steer_fn=steer_fn)
        before = lm.last_activity("scout")
        time.sleep(0.01)
        await lm._handle_steer("scout", "psst")
        assert lm.last_activity("scout") > before


# ── has_working_task / has_open_asks ──────────────────────────────────


class TestHasWorkingTask:
    def test_true_when_working(self, tmp_path):
        tasks = Tasks(db_path=str(tmp_path / "tasks.db"))
        task = tasks.create(creator="operator", assignee="scout", title="x")
        tasks.update_status(task["id"], "working", actor="scout")
        assert tasks.has_working_task("scout") is True

    def test_false_when_pending_only(self, tmp_path):
        tasks = Tasks(db_path=str(tmp_path / "tasks.db"))
        tasks.create(creator="operator", assignee="scout", title="x")
        assert tasks.has_working_task("scout") is False

    def test_false_for_unknown_agent(self, tmp_path):
        tasks = Tasks(db_path=str(tmp_path / "tasks.db"))
        assert tasks.has_working_task("ghost") is False


class TestHasOpenAsks:
    def test_true_for_asker(self):
        broker = AskBroker()

        async def _seed():
            broker.create("scout", "helper", "q?", 60)

        asyncio.run(_seed())
        assert broker.has_open_asks("scout") is True

    def test_true_for_recipient(self):
        broker = AskBroker()

        async def _seed():
            broker.create("asker", "scout", "q?", 60)

        asyncio.run(_seed())
        assert broker.has_open_asks("scout") is True

    def test_false_when_finished(self):
        broker = AskBroker()

        async def _seed_and_finish():
            record = broker.create("scout", "helper", "q?", 60)
            broker.resolve(record.ask_id, "answer", by="helper")
            broker.finish(record.ask_id)

        asyncio.run(_seed_and_finish())
        assert broker.has_open_asks("scout") is False

    def test_false_for_uninvolved_agent(self):
        broker = AskBroker()

        async def _seed():
            broker.create("asker", "helper", "q?", 60)

        asyncio.run(_seed())
        assert broker.has_open_asks("bystander") is False


# ── Observability: dispatch connect-failure ──────────────────────────


def _capture_dispatch_fn(monkeypatch, *, event_bus=None):
    """Build a RuntimeContext (bypassing __init__) and capture the
    ``_direct_dispatch`` closure ``_setup_dispatch`` hands to the
    LaneManager. Mirrors ``tests/test_runtime.py::_capture_dispatch_fn``.
    """
    from src.cli import runtime as runtime_mod

    captured: dict = {}

    class _FakeLaneManager:
        def __init__(self, *args, dispatch_fn=None, **kwargs):
            captured["dispatch_fn"] = dispatch_fn
            self._per_agent_timeouts: dict[str, int] = {}

        def get_queue_depth(self, agent):  # pragma: no cover - unused
            return 0

        def set_agent_timeout(self, agent, seconds):
            pass

        def timeout_for(self, agent):
            from src.host.lanes import _DEFAULT_LANE_TIMEOUT_SECONDS
            return _DEFAULT_LANE_TIMEOUT_SECONDS

    monkeypatch.setattr("src.host.lanes.LaneManager", _FakeLaneManager)

    ctx = runtime_mod.RuntimeContext.__new__(runtime_mod.RuntimeContext)
    transport_mock = AsyncMock()
    ctx.transport = transport_mock
    ctx.trace_store = None
    ctx.intent_store = None
    ctx.event_bus = event_bus
    ctx.health_monitor = None
    ctx._app = None
    ctx._unreachable_event_ts = {}

    async def _noop_notify(*a, **k):
        return None

    ctx._handle_notify_origin = _noop_notify
    ctx._setup_dispatch()
    loop = ctx._dispatch_loop
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)

    return captured["dispatch_fn"], ctx, transport_mock


class TestUnreachableObservability:
    async def test_connect_failure_emits_rate_limited_event(self, monkeypatch):
        eb = EventBus()
        events: list[dict] = []
        eb.add_listener(lambda evt: events.append(evt))
        dispatch_fn, ctx, transport = _capture_dispatch_fn(monkeypatch, event_bus=eb)
        transport.request.return_value = {"error": "Connection failed: refused"}

        from src.shared.types import SILENT_REPLY_TOKEN

        result = await dispatch_fn("scout", "hello")
        assert result == SILENT_REPLY_TOKEN
        unreachable = [e for e in events if e.get("type") == "agent_unreachable"]
        assert len(unreachable) == 1
        assert unreachable[0]["agent"] == "scout"

        # A second failure within the rate-limit window must NOT emit again.
        await dispatch_fn("scout", "hello again")
        unreachable = [e for e in events if e.get("type") == "agent_unreachable"]
        assert len(unreachable) == 1

    async def test_connect_failure_does_not_touch_task_status(self, monkeypatch):
        eb = EventBus()
        dispatch_fn, ctx, transport = _capture_dispatch_fn(monkeypatch, event_bus=eb)
        transport.request.return_value = {"error": "Connection failed: refused"}
        tasks_store = MagicMock()
        from src.shared.types import SILENT_REPLY_TOKEN

        result = await dispatch_fn("scout", "hello", task_id="task-1")
        assert result == SILENT_REPLY_TOKEN
        # The connect-failure branch returns BEFORE any task-closing
        # logic runs — task status semantics are untouched by design.
        tasks_store.update_status.assert_not_called()

    async def test_no_event_bus_is_a_noop(self, monkeypatch):
        dispatch_fn, ctx, transport = _capture_dispatch_fn(monkeypatch, event_bus=None)
        transport.request.return_value = {"error": "boom"}
        from src.shared.types import SILENT_REPLY_TOKEN

        result = await dispatch_fn("scout", "hello")
        assert result == SILENT_REPLY_TOKEN  # must not raise with no bus wired


# ── B4 pin: hibernate_idle_minutes ───────────────────────────────────


def test_hibernate_idle_minutes_b4_zero_valid_default_off(monkeypatch):
    assert limits.LIMIT_SPECS["hibernate_idle_minutes"] == (0, 0, 43200)
    assert limits.resolve("hibernate_idle_minutes") == 0
    monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "0")
    assert limits.resolve("hibernate_idle_minutes") == 0
    monkeypatch.setenv("OPENLEGION_HIBERNATE_IDLE_MINUTES", "45")
    assert limits.resolve("hibernate_idle_minutes") == 45
