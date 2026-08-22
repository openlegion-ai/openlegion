"""Per-agent lifecycle serialisation (``src/host/agent_lifecycle.py``).

Two things are under test here. The primitive itself — mutual exclusion
across threads AND event loops, per-agent independence, refcount
reclamation, the busy timeout, cancellation safety, and the self-nest
guard. And the call sites: that a lifecycle operation genuinely cannot
interleave with another one for the same agent, proved by parking one
inside its container call and showing the other queues instead of running.

The interleaving tests are written to FAIL if the lock is removed: each
one asserts on the final state (an agent resurrected after a delete, a
config row removed out from under a create), not merely on ordering.
"""

from __future__ import annotations

import ast
import asyncio
import os
import pathlib
import threading
import time

import pytest

from src.host import agent_lifecycle as lifecycle
from src.host.agent_lifecycle import (
    AgentLifecycleBusy,
    agent_incarnation,
    agent_incarnation_token,
    agent_lifecycle_locked,
    agent_lifecycle_locked_async,
    incarnation_token_matches,
    lifecycle_refcount,
    retire_agent,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _no_leaked_lock_entries():
    """Every test must leave the lock table empty.

    The entries are refcounted precisely so an agent id that appears once
    (an ephemeral spawn, an agent created and deleted) doesn't leak a lock
    object each — this is where that gets checked.
    """
    yield
    leaked = dict(lifecycle._locks)
    with lifecycle._guard:
        lifecycle._locks.clear()
        lifecycle._holders.clear()
        lifecycle._incarnations.clear()
        lifecycle._incarnation_seq = 0
        lifecycle._incarnation_floor = 0
    assert not leaked, f"lock entries left behind: { {k: v[1] for k, v in leaked.items()} }"


def _spin_until(predicate, timeout=3.0):
    """Block until ``predicate()`` or the timeout. Returns the final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


async def _spin_until_async(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


# ─────────────────────────────────────────────────────────────────────
# The primitive
# ─────────────────────────────────────────────────────────────────────


class TestMutualExclusion:
    def test_two_threads_do_not_overlap(self):
        order: list[str] = []
        holder_in = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("worker"):
                order.append("holder-in")
                holder_in.set()
                release.wait(3)
                order.append("holder-out")

        def contender():
            holder_in.wait(3)
            with agent_lifecycle_locked("worker"):
                order.append("contender-in")

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=contender)
        t1.start()
        t2.start()
        try:
            assert holder_in.wait(3)
            # Refcount 2 proves the contender actually queued on the lock —
            # "it hasn't run yet" alone would also be satisfied by a thread
            # the scheduler simply hasn't got to.
            assert _spin_until(lambda: lifecycle_refcount("worker") == 2)
            assert order == ["holder-in"]
        finally:
            release.set()
            t1.join(3)
            t2.join(3)
        assert order == ["holder-in", "holder-out", "contender-in"]

    @pytest.mark.asyncio
    async def test_two_coroutines_do_not_overlap(self):
        order: list[str] = []
        holder_in = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with agent_lifecycle_locked_async("worker"):
                order.append("holder-in")
                holder_in.set()
                await release.wait()
                order.append("holder-out")

        async def contender():
            await holder_in.wait()
            async with agent_lifecycle_locked_async("worker"):
                order.append("contender-in")

        h = asyncio.create_task(holder())
        c = asyncio.create_task(contender())
        try:
            await asyncio.wait_for(holder_in.wait(), 3)
            assert await _spin_until_async(lambda: lifecycle_refcount("worker") == 2)
            assert order == ["holder-in"]
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(h, c), 3)
        assert order == ["holder-in", "holder-out", "contender-in"]

    @pytest.mark.asyncio
    async def test_a_thread_and_a_coroutine_contend_on_the_same_lock(self):
        """The sync and async entry points are the same lock, not two.

        Lifecycle operations genuinely run on both sides — the REPL and the
        boot paths are plain threads, the routes and sweeps are coroutines
        on two different loops — so a primitive that only excluded within
        one of those would exclude nothing.
        """
        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("worker"):
                held.set()
                release.wait(3)

        t = threading.Thread(target=holder)
        t.start()
        try:
            assert held.wait(3)
            with pytest.raises(AgentLifecycleBusy):
                async with agent_lifecycle_locked_async("worker", timeout=0.15):
                    pytest.fail("acquired a lock a thread was holding")
        finally:
            release.set()
            t.join(3)
        # Once the thread lets go, the coroutine gets in.
        async with agent_lifecycle_locked_async("worker", timeout=3):
            pass

    @pytest.mark.asyncio
    async def test_unrelated_agents_never_queue_on_each_other(self):
        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("alice"):
                held.set()
                release.wait(3)

        t = threading.Thread(target=holder)
        t.start()
        try:
            assert held.wait(3)
            # A slow container build for one agent must not stall every
            # other agent's lifecycle.
            async with agent_lifecycle_locked_async("bob", timeout=0.5):
                pass
        finally:
            release.set()
            t.join(3)

    @pytest.mark.asyncio
    async def test_the_wait_does_not_block_the_event_loop(self):
        """A queued acquire must yield, not spin the loop dead.

        The routes hold this lock across container calls, so a waiter that
        blocked its loop would take every unrelated HTTP request down with
        it.
        """
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.005)

        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("worker"):
                held.set()
                release.wait(3)

        t = threading.Thread(target=holder)
        t.start()
        tick_task = asyncio.create_task(ticker())
        try:
            assert held.wait(3)
            with pytest.raises(AgentLifecycleBusy):
                async with agent_lifecycle_locked_async("worker", timeout=0.2):
                    pass
            assert ticks > 3, "the loop made no progress while waiting on the lock"
        finally:
            tick_task.cancel()
            release.set()
            t.join(3)


class TestLockLifetime:
    @pytest.mark.asyncio
    async def test_entry_is_reclaimed_once_nobody_holds_or_waits(self):
        async with agent_lifecycle_locked_async("ephemeral-1"):
            assert lifecycle_refcount("ephemeral-1") == 1
        assert lifecycle_refcount("ephemeral-1") == 0
        assert "ephemeral-1" not in lifecycle._locks

    def test_lock_is_released_when_the_body_raises(self):
        with pytest.raises(ValueError):
            with agent_lifecycle_locked("worker"):
                raise ValueError("boom")
        # Still acquirable — a leaked lock would hang here forever.
        with agent_lifecycle_locked("worker"):
            pass

    @pytest.mark.asyncio
    async def test_lock_is_released_when_an_async_body_raises(self):
        with pytest.raises(ValueError):
            async with agent_lifecycle_locked_async("worker"):
                raise ValueError("boom")
        async with agent_lifecycle_locked_async("worker", timeout=1):
            pass

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_does_not_leak_the_lock(self):
        """Cancellation must land with the lock NOT held.

        This is why the async acquire polls instead of blocking in a worker
        thread: a cancelled coroutine would abandon that thread mid-acquire,
        and nothing would ever release what it went on to take.
        """
        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("worker"):
                held.set()
                release.wait(3)

        t = threading.Thread(target=holder)
        t.start()

        async def waiter():
            async with agent_lifecycle_locked_async("worker"):
                pytest.fail("waiter should have been cancelled before acquiring")

        w = asyncio.create_task(waiter())
        try:
            assert held.wait(3)
            assert await _spin_until_async(lambda: lifecycle_refcount("worker") == 2)
            w.cancel()
            with pytest.raises(asyncio.CancelledError):
                await w
            assert await _spin_until_async(lambda: lifecycle_refcount("worker") == 1)
        finally:
            release.set()
            t.join(3)
        # The cancelled waiter took nothing with it.
        async with agent_lifecycle_locked_async("worker", timeout=1):
            pass

    @pytest.mark.asyncio
    async def test_busy_timeout_names_the_agent(self):
        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("stuck-agent"):
                held.set()
                release.wait(3)

        t = threading.Thread(target=holder)
        t.start()
        try:
            assert held.wait(3)
            with pytest.raises(AgentLifecycleBusy) as exc:
                async with agent_lifecycle_locked_async("stuck-agent", timeout=0.1):
                    pass
            assert exc.value.agent_id == "stuck-agent"
            assert "stuck-agent" in str(exc.value)
        finally:
            release.set()
            t.join(3)


class TestSelfNestIsRejected:
    """Nesting is a bug, and it must be a LOUD one.

    A non-reentrant lock nested by accident deadlocks — in production, on
    the mesh loop. Raising instead turns that into a test failure.
    """

    @pytest.mark.asyncio
    async def test_async_nested_in_async_raises(self):
        with pytest.raises(RuntimeError, match="must not nest"):
            async with agent_lifecycle_locked_async("worker"):
                async with agent_lifecycle_locked_async("worker"):
                    pass

    @pytest.mark.asyncio
    async def test_sync_nested_in_async_raises(self):
        with pytest.raises(RuntimeError, match="must not nest"):
            async with agent_lifecycle_locked_async("worker"):
                with agent_lifecycle_locked("worker"):
                    pass

    def test_sync_nested_in_sync_raises(self):
        with pytest.raises(RuntimeError, match="must not nest"):
            with agent_lifecycle_locked("worker"):
                with agent_lifecycle_locked("worker"):
                    pass

    @pytest.mark.asyncio
    async def test_a_different_agent_may_be_locked_inside(self):
        # Only SELF-nesting is a bug; two different agents is just lock
        # ordering, which nothing in the codebase does but the guard must
        # not forbid.
        async with agent_lifecycle_locked_async("alice"):
            async with agent_lifecycle_locked_async("bob"):
                pass


# ─────────────────────────────────────────────────────────────────────
# The call sites
# ─────────────────────────────────────────────────────────────────────


class _FakeRuntime:
    """A runtime backend whose container calls can be parked mid-flight.

    Parking a stop is what creates the interleaving window the real Docker
    calls create: the routes run them off the loop, so everything else
    keeps running while one is in progress.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self._calls_lock = threading.Lock()
        self.extra_env: dict[str, str] = {}
        self.agents: dict[str, dict] = {}
        self.park_stop_for: tuple | None = None
        self.stop_parked = threading.Event()
        self.release_stop = threading.Event()
        self.start_env_overrides: dict[str, dict] = {}

    def _record(self, call):
        with self._calls_lock:
            self.calls.append(call)

    def stop_agent(self, agent_id, remove_data=False):
        self._record(("stop", agent_id, remove_data))
        if self.park_stop_for == (agent_id, remove_data):
            self.stop_parked.set()
            assert self.release_stop.wait(5), "parked stop was never released"

    def start_agent(self, *, agent_id, env_overrides=None, **_kw):
        self._record(("start", agent_id))
        self.start_env_overrides[agent_id] = dict(env_overrides or {})
        return f"http://{agent_id}:8400"

    async def wait_for_agent(self, agent_id, timeout=30):
        return True

    def kinds(self):
        with self._calls_lock:
            return [c[0] for c in self.calls]


def _endpoint(router, path, method):
    """The route handler itself, so two of them can be raced on one loop.

    The router is mounted under a ``/dashboard`` prefix, so match on the
    suffix rather than pinning the mount point here.
    """
    for route in router.routes:
        route_path = getattr(route, "path", "") or ""
        if route_path.endswith(path) and method in (getattr(route, "methods", ()) or ()):
            return route.endpoint
    raise AssertionError(f"no {method} *{path} on the dashboard router")


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    """A dashboard router wired to fakes, with agents.yaml faked in memory."""
    from src.cli import config as cli_config
    from src.dashboard.server import create_dashboard_router
    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard

    yaml_state = {"agents": {"worker": {"role": "assistant", "model": "openai/gpt-4o-mini"}}}
    perms_state: dict = {"permissions": {"worker": {}}}
    full_cfg = {
        "agents": yaml_state["agents"],
        "llm": {"default_model": "openai/gpt-4o-mini"},
        "network": {},
        "mesh": {},
    }

    import contextlib as _contextlib

    monkeypatch.setattr(cli_config, "_load_config", lambda: full_cfg)
    monkeypatch.setattr(cli_config, "_load_agents_yaml", lambda: yaml_state)
    monkeypatch.setattr(cli_config, "_save_agents_yaml", lambda data: yaml_state.update(data))
    monkeypatch.setattr(cli_config, "_load_permissions", lambda: perms_state)
    monkeypatch.setattr(cli_config, "_save_permissions", lambda data: perms_state.update(data))
    monkeypatch.setattr(cli_config, "_config_lock", _contextlib.nullcontext)
    monkeypatch.setattr(
        cli_config,
        "_remove_team_blackboard_permissions",
        lambda *a, **k: None,
    )

    def _create_agent(name, role, model):
        yaml_state["agents"][name] = {"role": role, "model": model}

    def _update_agent_field(name, field, value):
        if name in yaml_state["agents"]:
            yaml_state["agents"][name][field] = value

    monkeypatch.setattr(cli_config, "_create_agent", _create_agent)
    monkeypatch.setattr(cli_config, "_update_agent_field", _update_agent_field)

    registry = {"worker": "http://worker:8400"}
    runtime = _FakeRuntime()
    connectors = _ParkingConnectorStore()
    forgotten: list[str] = []
    blackboard = Blackboard(db_path=str(tmp_path / "bb.db"))
    costs = CostTracker(str(tmp_path / "costs.db"))

    router = create_dashboard_router(
        blackboard=blackboard,
        health_monitor=None,
        cost_tracker=costs,
        trace_store=None,
        event_bus=None,
        agent_registry=registry,
        runtime=runtime,
        router=None,
        transport=None,
        connector_store=connectors,
        forget_agent_status=forgotten.append,
    )
    yield router, registry, runtime, yaml_state, connectors, forgotten
    blackboard.close()
    costs.close()


class _ParkingConnectorStore:
    """Parks the delete inside the ONE window the race needs.

    The delete unregisters the agent, then awaits this call, and only then
    clears agents.yaml — so this is where a same-name create can slip past
    the duplicate-name check and get its config row eaten.
    """

    def __init__(self):
        self.park = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.removed: list[str] = []

    def remove_agent(self, agent_id):
        self.removed.append(agent_id)
        if self.park:
            self.entered.set()
            assert self.release.wait(5), "parked connector cleanup was never released"


class _FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


class TestDashboardRoutesSerialise:
    @pytest.mark.asyncio
    async def test_a_delete_cannot_land_between_a_restart_s_stop_and_start(self, dashboard):
        """The resurrection bug: the restart puts back what the delete removed.

        Without the lock the delete runs to completion inside the restart's
        stop→start window — volume destroyed, config dropped, agent
        unregistered — and then the restart's ``start_agent`` registers a
        fresh container for an agent that no longer exists anywhere else.
        """
        router, registry, runtime, yaml_state, connectors, forgotten = dashboard
        restart = _endpoint(router, "/api/agents/{agent_id}/restart", "POST")
        delete = _endpoint(router, "/api/agents/{agent_id}", "DELETE")

        runtime.park_stop_for = ("worker", False)
        restart_task = asyncio.create_task(restart("worker"))
        assert await asyncio.to_thread(runtime.stop_parked.wait, 5)

        delete_task = asyncio.create_task(delete("worker"))
        assert await _spin_until_async(lambda: lifecycle_refcount("worker") == 2), (
            "the delete never queued on the lifecycle lock — it ran straight "
            "into the restart's stop/start window"
        )
        assert ("stop", "worker", True) not in runtime.calls, (
            "the delete's volume-destroying stop ran while the restart was mid-flight"
        )

        runtime.release_stop.set()
        await asyncio.wait_for(restart_task, 5)
        await asyncio.wait_for(delete_task, 5)

        assert "worker" not in registry, "the restart resurrected a deleted agent"
        assert "worker" not in yaml_state["agents"]
        assert runtime.kinds() == ["stop", "start", "stop"]

    @pytest.mark.asyncio
    async def test_a_create_cannot_land_inside_a_delete(self, dashboard):
        """The other direction: the delete's config cleanup eats the create.

        The delete unregisters before it clears agents.yaml, so once it has
        popped the registry a same-name create gets past the duplicate check
        — and then the delete's ``_save_agents_yaml`` removes the row the
        create just wrote, leaving a live registered agent with no config.
        """
        router, registry, runtime, yaml_state, connectors, forgotten = dashboard
        create = _endpoint(router, "/api/agents", "POST")
        delete = _endpoint(router, "/api/agents/{agent_id}", "DELETE")

        connectors.park = True
        delete_task = asyncio.create_task(delete("worker"))
        assert await asyncio.to_thread(connectors.entered.wait, 5)
        assert "worker" not in registry, "the delete should have unregistered by now"

        create_task = asyncio.create_task(
            create(_FakeRequest({"name": "worker", "role": "assistant"})),
        )
        assert await _spin_until_async(lambda: lifecycle_refcount("worker") == 2), (
            "the create never queued on the lifecycle lock"
        )
        assert "start" not in runtime.kinds(), "the create started a container mid-delete"

        connectors.release.set()
        await asyncio.wait_for(delete_task, 5)
        result = await asyncio.wait_for(create_task, 5)

        assert result["created"] is True
        assert "worker" in registry
        assert "worker" in yaml_state["agents"], (
            "the delete removed the config row the create had just written"
        )

    @pytest.mark.asyncio
    async def test_a_restart_of_a_deleted_agent_404s_instead_of_resurrecting(self, dashboard):
        """The re-check under the lock, on its own.

        A restart that queued behind a delete must not run: the checks it
        passed were made against the pre-delete world.
        """
        from fastapi import HTTPException

        router, registry, runtime, _yaml, _connectors, _forgotten = dashboard
        restart = _endpoint(router, "/api/agents/{agent_id}/restart", "POST")

        async def archive_like():
            async with agent_lifecycle_locked_async("worker"):
                await asyncio.sleep(0.05)
                registry.pop("worker", None)

        holder = asyncio.create_task(archive_like())
        await asyncio.sleep(0)
        restart_task = asyncio.create_task(restart("worker"))
        await asyncio.wait_for(holder, 3)
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(restart_task, 3)
        assert exc.value.status_code == 404
        assert "start" not in runtime.kinds()

    @pytest.mark.asyncio
    async def test_a_wedged_lifecycle_operation_surfaces_as_409(self, dashboard, monkeypatch):
        from fastapi import HTTPException

        from src.dashboard import server as dash_server

        router, _registry, _runtime, _yaml, _connectors, _forgotten = dashboard
        restart = _endpoint(router, "/api/agents/{agent_id}/restart", "POST")

        monkeypatch.setattr(dash_server, "agent_lifecycle_locked_async", _always_busy)
        with pytest.raises(HTTPException) as exc:
            await restart("worker")
        assert exc.value.status_code == 409
        assert "worker" in str(exc.value.detail)


def _always_busy(agent_id, timeout=None):
    import contextlib

    @contextlib.asynccontextmanager
    async def _cm():
        raise AgentLifecycleBusy(agent_id, 300.0)
        yield  # pragma: no cover - unreachable, keeps this an async generator

    return _cm()


# ─────────────────────────────────────────────────────────────────────
# The health monitor's auto-restart
# ─────────────────────────────────────────────────────────────────────


def _make_monitor(monkeypatch, cfg=None):
    from unittest.mock import MagicMock

    from src.host import health as health_mod
    from src.host.health import AgentHealth, HealthMonitor

    monkeypatch.setattr(
        health_mod,
        "_load_config",
        lambda: cfg or {"agents": {"worker": {"role": "assistant"}}, "network": {}, "llm": {}},
    )
    runtime = _FakeRuntime()
    runtime.agents = {"worker": {"role": "assistant", "model": "m", "tools_dir": "", "thinking": ""}}
    monitor = HealthMonitor(
        runtime=runtime,
        transport=MagicMock(),
        router=MagicMock(),
        event_bus=None,
    )
    monitor.agents["worker"] = AgentHealth(agent_id="worker")
    return monitor, runtime


class TestHealthMonitorRestart:
    @pytest.mark.asyncio
    async def test_it_yields_to_a_concurrent_archive_without_starting_anything(
        self, monkeypatch,
    ):
        """An archive that holds the lock wins outright.

        The old code let the restart build the container first and only then
        noticed the agent had been deregistered, undoing its own work. Under
        the lock the restart never starts at all.
        """
        monitor, runtime = _make_monitor(monkeypatch)

        released = asyncio.Event()

        async def archive_like():
            async with agent_lifecycle_locked_async("worker"):
                await released.wait()
                monitor.agents.pop("worker", None)

        holder = asyncio.create_task(archive_like())
        await asyncio.sleep(0)
        restart = asyncio.create_task(monitor._try_restart("worker"))
        try:
            queued = await _spin_until_async(lambda: lifecycle_refcount("worker") == 2)
        finally:
            released.set()
            await asyncio.wait_for(holder, 3)
            await asyncio.wait_for(restart, 3)

        assert queued, "the restart never queued on the lifecycle lock"
        assert "start" not in runtime.kinds(), (
            "the health monitor rebuilt a container for an archived agent"
        )

    @pytest.mark.asyncio
    async def test_proxy_env_never_touches_the_shared_extra_env_dict(self, monkeypatch):
        """A global mutation here rides along on other agents' starts.

        ``extra_env`` is read by EVERY ``start_agent`` call, so writing the
        restarting agent's proxy into it hands that proxy to whichever
        unrelated agent happens to start next.
        """
        monkeypatch.setenv("OPENLEGION_CRED_TESTPROXY", "http://proxy.example:8080")
        monitor, runtime = _make_monitor(
            monkeypatch,
            cfg={
                "agents": {
                    "worker": {
                        "role": "assistant",
                        "proxy": {"mode": "custom", "credential": "TESTPROXY"},
                    },
                },
                "network": {},
                "llm": {},
            },
        )
        await asyncio.wait_for(monitor._try_restart("worker"), 5)

        assert "start" in runtime.kinds()
        assert "HTTP_PROXY" not in runtime.extra_env, (
            "proxy config leaked into the shared extra_env dict"
        )
        assert (
            runtime.start_env_overrides["worker"].get("HTTP_PROXY")
            == "http://proxy.example:8080"
        ), "the restarting agent never got its proxy at all"


# ─────────────────────────────────────────────────────────────────────
# Coverage of every lifecycle call site
# ─────────────────────────────────────────────────────────────────────


# Every ``start_agent`` / ``stop_agent`` / ``spawn_agent`` reference outside
# the runtime backend itself must sit inside a lifecycle lock — a lock only
# some mutators take excludes nothing. These are the deliberate exceptions,
# each with the reason it is safe.
_UNLOCKED_BY_DESIGN = {
    # A fresh ``generate_id("spawn")`` per call: no other lifecycle
    # operation can name this agent before it is registered, and the
    # backend's own lock already brackets the start with its TTL stamps.
    ("src/host/server.py", "spawn_agent"),
    # Boot. ``_start_agents`` runs at cli/runtime.py:385, before
    # ``_setup_dispatch`` and ``_start_mesh_server`` — there is no other
    # thread, loop, sweep or route in existence yet to race with.
    ("src/cli/runtime.py", "_start_agents"),
}

_LIFECYCLE_CALLERS = (
    "src/host/server.py",
    "src/host/health.py",
    "src/dashboard/server.py",
    "src/cli/repl.py",
    "src/cli/runtime.py",
)


def _unlocked_lifecycle_sites(path: pathlib.Path):
    tree = ast.parse(path.read_text())
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in {"start_agent", "stop_agent", "spawn_agent"}):
            continue
        cur, enclosing, locked = node, None, False
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, (ast.With, ast.AsyncWith)):
                for item in cur.items:
                    expr = item.context_expr
                    name = expr.func.id if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) else ""
                    if "lifecycle" in name:
                        locked = True
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)) and enclosing is None:
                enclosing = cur.name
        if not locked:
            found.append((enclosing, node.lineno))
    return found


class TestEveryLifecycleSiteIsCovered:
    def test_no_unlocked_start_or_stop_outside_the_documented_exceptions(self):
        unexpected = []
        for rel in _LIFECYCLE_CALLERS:
            for func, line in _unlocked_lifecycle_sites(REPO_ROOT / rel):
                if (rel, func) in _UNLOCKED_BY_DESIGN:
                    continue
                unexpected.append(f"{rel}:{line} in {func}()")
        assert not unexpected, (
            "lifecycle call sites outside a lifecycle lock:\n  "
            + "\n  ".join(unexpected)
            + "\nEither wrap them or add them to _UNLOCKED_BY_DESIGN with a reason."
        )

    def test_the_documented_exceptions_still_exist(self):
        """Keeps the allowlist honest — a stale entry would hide a new gap."""
        still_open = set()
        for rel in _LIFECYCLE_CALLERS:
            for func, _line in _unlocked_lifecycle_sites(REPO_ROOT / rel):
                still_open.add((rel, func))
        stale = _UNLOCKED_BY_DESIGN - still_open
        assert not stale, f"_UNLOCKED_BY_DESIGN entries no longer match any site: {stale}"


# ─────────────────────────────────────────────────────────────────────
# Revalidation under the lock (mesh)
# ─────────────────────────────────────────────────────────────────────


def _parked_archive(monkeypatch, cfg, park: threading.Event, release: threading.Event):
    """Make ``_archive_agent`` park BEFORE it writes the status.

    That is the only ordering that reproduces the races below: the archive
    is holding the lifecycle lock but has not yet flipped the status, so a
    wake / hibernate / unarchive still reads the pre-archive world, decides
    to act on it, and only then queues on the lock.
    """
    from src.cli import config as cli_config

    def _archive(name):
        park.set()
        assert release.wait(5), "parked archive was never released"
        cfg["agents"][name]["status"] = "archived"

    monkeypatch.setattr(cli_config, "_archive_agent", _archive)


def _run_in_thread(fn):
    box: dict = {}

    def _target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - surfaced by the caller
            box["error"] = e

    t = threading.Thread(target=_target)
    t.start()
    return t, box


class TestArchiveWinsAgainstQueuedOperations:
    """An archive that finishes first must not be undone by what queued.

    Every one of these decided to act while the agent was still in service
    — the status they read is stale by the time they hold the lock.
    """

    @pytest.mark.asyncio
    async def test_a_wake_queued_behind_an_archive_does_not_start_the_agent(
        self, tmp_path, monkeypatch,
    ):
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated",
        )
        try:
            assert app.get_agent_status("scout") == "hibernated"
            park, release = threading.Event(), threading.Event()
            _parked_archive(monkeypatch, cfg, park, release)

            client = TestClient(app)
            t, box = _run_in_thread(
                lambda: client.post("/mesh/agents/scout/archive", headers=_OP).status_code,
            )
            try:
                assert await asyncio.to_thread(park.wait, 5)
                # The archive holds the lock and has NOT flipped the status
                # yet, so the wake still sees a hibernated agent and claims.
                wake = asyncio.create_task(app.ensure_agent_running("scout", trigger="test"))
                assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2), (
                    "the wake never queued on the lifecycle lock"
                )
            finally:
                release.set()
                t.join(5)
            assert box.get("value") == 200, box

            assert await asyncio.wait_for(wake, 5) is False
            assert cm.started == [], "the wake started a container for an archived agent"
            assert cfg["agents"]["scout"]["status"] == "archived", (
                "the wake stamped an archived agent back to active"
            )
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_a_hibernate_queued_behind_an_archive_refuses(self, tmp_path, monkeypatch):
        """Constraint #14: ``archived`` and ``hibernated`` must never blur.

        ``_hibernate_agent`` sets the status unconditionally, so without the
        re-check the hibernate overwrites ``archived`` — and the agent
        silently becomes auto-wakeable again.
        """
        from fastapi import HTTPException
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(tmp_path, monkeypatch)
        try:
            park, release = threading.Event(), threading.Event()
            _parked_archive(monkeypatch, cfg, park, release)
            hibernate = app.hibernation_sweeper._hibernate_fn

            client = TestClient(app)
            t, box = _run_in_thread(
                lambda: client.post("/mesh/agents/scout/archive", headers=_OP).status_code,
            )
            try:
                assert await asyncio.to_thread(park.wait, 5)
                hib = asyncio.create_task(hibernate("scout", caller="sweep"))
                assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2), (
                    "the hibernate never queued on the lifecycle lock"
                )
            finally:
                release.set()
                t.join(5)
            assert box.get("value") == 200, box

            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(hib, 5)
            assert exc.value.status_code == 409
            assert cfg["agents"]["scout"]["status"] == "archived", (
                "the hibernate overwrote the archived status"
            )
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_an_unarchive_waits_for_the_archive_instead_of_being_overwritten(
        self, tmp_path, monkeypatch,
    ):
        """Unarchive writes no container, but it is still a status transition.

        Unlocked, it lands inside the archive and the archive's own write
        then clobbers it — the operator's unarchive silently does nothing.
        """
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, _cm, _tr, _hm, _eb, cfg = _build_app(tmp_path, monkeypatch)
        try:
            park, release = threading.Event(), threading.Event()
            _parked_archive(monkeypatch, cfg, park, release)

            client = TestClient(app)
            t_arch, box_arch = _run_in_thread(
                lambda: client.post("/mesh/agents/scout/archive", headers=_OP).status_code,
            )
            t_un = box_un = None
            try:
                assert await asyncio.to_thread(park.wait, 5)
                t_un, box_un = _run_in_thread(
                    lambda: client.post("/mesh/agents/scout/unarchive", headers=_OP).status_code,
                )
                assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2), (
                    "the unarchive never queued on the lifecycle lock"
                )
            finally:
                release.set()
                t_arch.join(5)
                if t_un is not None:
                    t_un.join(5)
            assert box_arch.get("value") == 200, box_arch
            assert box_un.get("value") == 200, box_un

            assert cfg["agents"]["scout"]["status"] == "active", (
                "the archive overwrote the unarchive that ran inside it"
            )
            assert app.get_agent_status("scout") == "active"
        finally:
            bb.close()

    def test_deleting_an_agent_clears_its_status_override(self, tmp_path, monkeypatch):
        """A recreated id must not inherit the previous agent's status.

        A stale ``hibernated`` / ``archived`` entry makes the fresh agent
        unreachable: never woken, skipped for lead selection, refused by the
        manual wake endpoint — while its config row says ``active``.
        """
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, _cm, _tr, _hm, _eb, _cfg = _build_app(tmp_path, monkeypatch)
        try:
            assert TestClient(app).post("/mesh/agents/scout/hibernate", headers=_OP).status_code == 200
            assert app.get_agent_status("scout") == "hibernated"
            app.cleanup_agent("scout")
            assert app.get_agent_status("scout") == "active", (
                "a deleted agent left its status behind for the next agent of that name"
            )
        finally:
            bb.close()


# ─────────────────────────────────────────────────────────────────────
# Revalidation under the lock (health monitor, REPL)
# ─────────────────────────────────────────────────────────────────────


class TestQueuedOperationsRevalidate:
    @pytest.mark.asyncio
    async def test_health_restart_bails_when_the_name_was_recreated(self, monkeypatch):
        """Name-equality is not enough — the object has to be the same one.

        A delete plus a same-name create repopulates ``self.agents[id]``, so a
        name check passes while ``health`` and the registry ``info`` this
        restart captured still describe the agent that was deleted.
        """
        from src.host.health import AgentHealth

        monitor, runtime = _make_monitor(monkeypatch)
        released = asyncio.Event()

        async def delete_and_recreate():
            async with agent_lifecycle_locked_async("worker"):
                await released.wait()
                monitor.agents["worker"] = AgentHealth(agent_id="worker")

        holder = asyncio.create_task(delete_and_recreate())
        await asyncio.sleep(0)
        restart = asyncio.create_task(monitor._try_restart("worker"))
        try:
            queued = await _spin_until_async(lambda: lifecycle_refcount("worker") == 2)
        finally:
            released.set()
            await asyncio.wait_for(holder, 3)
            await asyncio.wait_for(restart, 3)

        assert queued, "the restart never queued on the lifecycle lock"
        assert "start" not in runtime.kinds(), (
            "the restart rebuilt the previous incarnation's container"
        )

    @pytest.mark.asyncio
    async def test_ephemeral_cleanup_skips_an_agent_removed_while_it_queued(self, monkeypatch):
        """``del self.agents[id]`` raised here, aborting the whole sweep."""
        from src.host.health import AgentHealth

        monitor, runtime = _make_monitor(monkeypatch)
        runtime.agents["spawn-x"] = {
            "ephemeral": True, "ttl": 1, "spawned_at": time.time() - 60,
        }
        monitor.agents["spawn-x"] = AgentHealth(agent_id="spawn-x")
        released = asyncio.Event()

        async def delete_like():
            async with agent_lifecycle_locked_async("spawn-x"):
                await released.wait()
                runtime.agents.pop("spawn-x", None)
                monitor.agents.pop("spawn-x", None)

        holder = asyncio.create_task(delete_like())
        await asyncio.sleep(0)
        sweep = asyncio.create_task(monitor._cleanup_ephemeral_agents())
        try:
            queued = await _spin_until_async(lambda: lifecycle_refcount("spawn-x") == 2)
        finally:
            released.set()
            await asyncio.wait_for(holder, 3)
            # A KeyError here would abort every remaining agent in the pass.
            await asyncio.wait_for(sweep, 3)

        assert queued, "the cleanup never queued on the lifecycle lock"
        assert "stop" not in runtime.kinds(), "the cleanup tore down an agent already deleted"

    def test_repl_restart_bails_when_the_agent_was_deleted_while_it_waited(self):
        from unittest.mock import MagicMock

        from tests.test_repl_remove import _FakeCtx, _make_session

        ctx = _FakeCtx()
        ctx.runtime = MagicMock()
        session = _make_session(ctx)

        held = threading.Event()
        release = threading.Event()

        def holder():
            with agent_lifecycle_locked("scout"):
                held.set()
                assert release.wait(3)
                ctx.agents.pop("scout", None)

        t = threading.Thread(target=holder)
        t.start()
        restart_t = None
        try:
            assert held.wait(3)
            restart_t, box = _run_in_thread(lambda: session._restart_agent("scout"))
            assert _spin_until(lambda: lifecycle_refcount("scout") == 2), (
                "the REPL restart never queued on the lifecycle lock"
            )
        finally:
            release.set()
            t.join(3)
            if restart_t is not None:
                restart_t.join(3)
        assert not box.get("error"), box["error"]
        ctx.runtime.start_agent.assert_not_called()
        ctx.runtime.stop_agent.assert_not_called()


class TestTemplateSlotsRevalidate:
    """The template loop's config snapshot is stale by the time it starts.

    ``_apply_template`` writes every slot's config row under the fleet-wide
    ``_creation_lock`` and releases it long before the per-slot start loop
    reaches them, so an archive or delete can act on a name in between.
    """

    def _make_app(self, monkeypatch):
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock

        from src.cli import config as cli_config
        from src.host.mesh import Blackboard, MessageRouter, PubSub
        from src.host.permissions import PermissionMatrix
        from src.host.server import create_mesh_app

        rows: dict = {}

        def _load_config():
            # A fresh snapshot per call, like the real one re-reading
            # agents.yaml — the guard under test compares two of them.
            return {
                "agents": {k: dict(v) for k, v in rows.items()},
                "llm": {"default_model": "openai/gpt-4o-mini"},
                "network": {},
                "mesh": {},
            }

        monkeypatch.setattr(cli_config, "_load_config", _load_config)

        def _fake_apply_template(template_name, tpl, agent_overrides=None):
            rows["scout"] = {"role": "worker", "model": "openai/gpt-4o-mini"}
            return ["scout"]

        monkeypatch.setattr(cli_config, "_apply_template", _fake_apply_template)
        monkeypatch.setattr(cli_config, "_update_agent_field", lambda *a, **k: None)

        perms = PermissionMatrix.__new__(PermissionMatrix)
        perms.permissions = {}
        perms._config_path = "/tmp/__nonexistent_permissions_for_lifecycle_test.json"
        perms._reload_lock = threading.Lock()
        bb = Blackboard(db_path=":memory:")
        registry: dict[str, str] = {}
        cm = MagicMock()
        cm.project_root = Path("/tmp/test_project")
        cm.extra_env = {}
        cm.start_agent.return_value = "http://localhost:8401"
        cm.wait_for_agent = AsyncMock(return_value=True)
        app = create_mesh_app(
            blackboard=bb,
            pubsub=PubSub(),
            router=MessageRouter(permissions=perms, agent_registry=registry),
            permissions=perms,
            container_manager=cm,
        )
        return app, bb, cm, rows

    def test_a_slot_deleted_while_it_queued_is_not_started(self, monkeypatch):
        import contextlib

        from fastapi.testclient import TestClient

        from src.host import server as server_mod

        app, bb, cm, rows = self._make_app(monkeypatch)
        try:
            real_lock = server_mod.agent_lifecycle_locked_async

            @contextlib.asynccontextmanager
            async def _delete_once_acquired(agent_id, timeout=None):
                # Stands in for a delete that held the lock while this slot
                # queued behind it: by the time the slot runs, the row it was
                # about to start from is gone.
                async with real_lock(agent_id, timeout=timeout):
                    rows.pop(agent_id, None)
                    yield

            monkeypatch.setattr(
                server_mod, "agent_lifecycle_locked_async", _delete_once_acquired,
            )
            resp = TestClient(app).post("/mesh/fleet/apply", json={"template": "starter"})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["created"] == []
            assert [f["agent_id"] for f in body["failed"]] == ["scout"]
            assert "deleted" in body["failed"][0]["error"]
            cm.start_agent.assert_not_called()
        finally:
            bb.close()


# ─────────────────────────────────────────────────────────────────────
# Incarnation: the same name is not the same agent
# ─────────────────────────────────────────────────────────────────────


class TestIncarnation:
    def test_it_only_moves_when_an_id_is_retired(self):
        assert agent_incarnation("worker") == 0
        assert agent_incarnation("worker") == 0
        assert retire_agent("worker") == 1
        assert agent_incarnation("worker") == 1
        assert agent_incarnation("other") == 0

    @pytest.mark.asyncio
    async def test_a_delete_refuses_to_destroy_a_replacement(self, dashboard):
        """ABA. Every name-based check passes and the wrong agent dies.

        Delete A checks the name, then spends minutes offboarding. Delete B
        removes that agent and a create puts a NEW one under the same name.
        Delete A finally acquires the lock, sees the name present, and — on
        a name check alone — destroys the replacement's container, volume
        and config.
        """
        from fastapi import HTTPException

        router, registry, runtime, yaml_state, _connectors, _forgotten = dashboard
        delete = _endpoint(router, "/api/agents/{agent_id}", "DELETE")

        async def delete_then_recreate():
            """The other delete, plus the create that reuses the name."""
            async with agent_lifecycle_locked_async("worker"):
                await asyncio.sleep(0.05)
                registry.pop("worker", None)
                yaml_state["agents"].pop("worker", None)
                retire_agent("worker")
                registry["worker"] = "http://worker-2:8400"
                yaml_state["agents"]["worker"] = {"role": "assistant"}

        holder = asyncio.create_task(delete_then_recreate())
        await asyncio.sleep(0)
        queued = asyncio.create_task(delete("worker"))
        await asyncio.wait_for(holder, 3)

        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(queued, 5)
        assert exc.value.status_code == 409
        assert "recreated" in str(exc.value.detail)
        assert registry.get("worker") == "http://worker-2:8400", (
            "the queued delete destroyed the replacement"
        )
        assert "worker" in yaml_state["agents"]
        assert ("stop", "worker", True) not in runtime.calls

    @pytest.mark.asyncio
    async def test_a_restart_refuses_a_replacement(self, dashboard):
        from fastapi import HTTPException

        router, registry, runtime, _yaml, _connectors, _forgotten = dashboard
        restart = _endpoint(router, "/api/agents/{agent_id}/restart", "POST")

        async def delete_and_recreate():
            async with agent_lifecycle_locked_async("worker"):
                await asyncio.sleep(0.05)
                retire_agent("worker")
                registry["worker"] = "http://worker-2:8400"

        holder = asyncio.create_task(delete_and_recreate())
        await asyncio.sleep(0)
        restart_task = asyncio.create_task(restart("worker"))
        await asyncio.wait_for(holder, 3)
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(restart_task, 3)
        assert exc.value.status_code == 404
        assert "start" not in runtime.kinds(), "the restart bounced the replacement"

    @pytest.mark.asyncio
    async def test_restart_all_skips_a_replaced_agent(self, dashboard):
        """The fleet restart pins ONE config snapshot for every agent.

        Between that read and a given agent's turn at the lock sit the
        browser-service restart and every other agent's stop/start, so the
        row it would restart from can be several minutes stale.
        """
        router, registry, runtime, _yaml, _connectors, _forgotten = dashboard
        restart_all = _endpoint(router, "/api/restart-agents", "POST")

        async def replace_during_the_fan_out():
            async with agent_lifecycle_locked_async("worker"):
                await asyncio.sleep(0.05)
                retire_agent("worker")
                registry["worker"] = "http://worker-2:8400"

        holder = asyncio.create_task(replace_during_the_fan_out())
        await asyncio.sleep(0)
        result = await asyncio.wait_for(restart_all(), 5)
        await asyncio.wait_for(holder, 3)

        assert result["restarted"]["worker"] == "skipped: agent was replaced"
        assert "start" not in runtime.kinds()

    @pytest.mark.asyncio
    async def test_a_delete_drops_the_cached_lifecycle_status(self, dashboard):
        """The dashboard runs its own cleanup and never calls ``cleanup_agent``.

        Without the injected seam a recreated id inherits the previous
        agent's ``archived`` (every dispatch refuses it) or ``hibernated``
        (a pointless cold start over a running container).
        """
        router, _registry, _runtime, _yaml, _connectors, forgotten = dashboard
        delete = _endpoint(router, "/api/agents/{agent_id}", "DELETE")
        await asyncio.wait_for(delete("worker"), 5)
        assert forgotten == ["worker"]

    @pytest.mark.asyncio
    async def test_a_wake_reports_failure_when_it_did_not_wake_anything(
        self, tmp_path, monkeypatch,
    ):
        """"Status says active now" is not the same as "I woke it".

        Unarchive flips the status without starting a container, so a wake
        that queued behind one and then read ``active`` would report success
        for an agent that is still stopped — and the caller would dispatch
        into it. Deleting the agent (which drops the override entirely, so
        the default ``active`` applies) reads the same way.
        """
        from fastapi.testclient import TestClient

        from src.cli import config as cli_config
        from tests.test_hibernation import _OP, _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated",
        )
        try:
            park, release = threading.Event(), threading.Event()

            def _unarchive(name):
                park.set()
                assert release.wait(5)
                cfg["agents"][name]["status"] = "active"

            monkeypatch.setattr(cli_config, "_unarchive_agent", _unarchive)

            client = TestClient(app)
            t, box = _run_in_thread(
                lambda: client.post("/mesh/agents/scout/unarchive", headers=_OP).status_code,
            )
            try:
                assert await asyncio.to_thread(park.wait, 5)
                wake = asyncio.create_task(app.ensure_agent_running("scout", trigger="test"))
                assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2)
            finally:
                release.set()
                t.join(5)
            assert box.get("value") == 200, box

            assert await asyncio.wait_for(wake, 5) is False, (
                "the wake reported success without starting anything"
            )
            assert cm.started == []
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_a_mesh_delete_refuses_to_destroy_a_replacement(self, tmp_path, monkeypatch):
        """The mesh delete's window is the widest of all — a handover turn.

        By the time a confirmed ``agent_delete`` reaches the container, the
        name it was raised against can belong to a different agent.
        """
        from fastapi import HTTPException

        from tests.test_hibernation import _build_app

        # Archived: that is propose-delete's precondition, and the apply
        # side re-checks it — so this test exercises the incarnation guard
        # rather than tripping over the status one.
        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        try:
            delete = app.pending_executors["delete"]

            async def replace_while_the_delete_queues():
                async with agent_lifecycle_locked_async("scout"):
                    # Long enough for the delete below to capture the
                    # incarnation and reach the lock behind us.
                    await asyncio.sleep(0.05)
                    retire_agent("scout")

            holder = asyncio.create_task(replace_while_the_delete_queues())
            await asyncio.sleep(0)
            queued = asyncio.create_task(
                delete({
                    "target_kind": "agent",
                    "target_id": "scout",
                    "nonce": "n1",
                    # Stamped at propose time, when the name still belonged
                    # to the agent this confirmation was raised against.
                    "payload": {"incarnation": agent_incarnation_token("scout")},
                }),
            )
            await asyncio.wait_for(holder, 3)
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(queued, 5)
            assert exc.value.status_code == 409
            assert "recreated" in str(exc.value.detail)
            assert cm.stopped == [], "the queued delete destroyed the replacement"
        finally:
            bb.close()


class TestTemplateSlotIncarnation:
    def test_a_slot_deleted_and_recreated_is_not_started(self, monkeypatch):
        """The row is present in both snapshots, and is still a different agent.

        Presence alone cannot see this: the delete removed the row and the
        recreate put one back, so the guard has to compare incarnations.
        """
        import contextlib

        from fastapi.testclient import TestClient

        from src.host import server as server_mod

        app, bb, cm, rows = TestTemplateSlotsRevalidate()._make_app(monkeypatch)
        try:
            real_lock = server_mod.agent_lifecycle_locked_async

            @contextlib.asynccontextmanager
            async def _replace_once_acquired(agent_id, timeout=None):
                async with real_lock(agent_id, timeout=timeout):
                    # Deleted and recreated while this slot queued: the row
                    # is back, under the same name, for a different agent.
                    retire_agent(agent_id)
                    yield

            monkeypatch.setattr(
                server_mod, "agent_lifecycle_locked_async", _replace_once_acquired,
            )
            resp = TestClient(app).post("/mesh/fleet/apply", json={"template": "starter"})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["created"] == []
            assert [f["agent_id"] for f in body["failed"]] == ["scout"]
            assert "scout" in rows, "the row itself is still present"
            cm.start_agent.assert_not_called()
        finally:
            bb.close()


class TestRoundThreeGuards:
    @pytest.mark.asyncio
    async def test_restart_all_leaves_agents_created_after_its_snapshot_alone(self, dashboard):
        """The fan-out must restart exactly what it snapshotted.

        Config, incarnations and the target list are all read at the same
        moment; the browser-service restart then runs before any agent's
        turn. An agent created in that window has no row in the snapshot, so
        sweeping it in would stop a brand-new container and rebuild it from
        empty defaults — role "assistant", the default model, no tools_dir.
        """
        from src.host import runtime as runtime_mod

        router, registry, runtime, _yaml, _connectors, _forgotten = dashboard
        restart_all = _endpoint(router, "/api/restart-agents", "POST")

        # Take the browser-service branch so its window is real here too.
        monkeypatch_target = type(runtime)
        runtime.stop_browser_service = lambda: registry.__setitem__(
            "newcomer", "http://newcomer:8400",
        )
        runtime.start_browser_service = lambda: None
        original = runtime_mod.DockerBackend
        runtime_mod.DockerBackend = monkeypatch_target
        try:
            result = await asyncio.wait_for(restart_all(), 5)
        finally:
            runtime_mod.DockerBackend = original

        assert "newcomer" in registry, "the window never opened"
        assert "newcomer" not in result["restarted"], (
            "restart-all swept in an agent created after its config snapshot"
        )
        assert ("start", "newcomer") not in runtime.calls

    @pytest.mark.asyncio
    async def test_an_archive_refuses_a_replacement(self, tmp_path, monkeypatch):
        """The lock wait alone is enough of a window.

        Archive validated its target, then queued. By the time it holds the
        lock the name can belong to an agent that was never archived and
        should not be.
        """
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(tmp_path, monkeypatch)
        try:
            client = TestClient(app)
            async with agent_lifecycle_locked_async("scout"):
                t, box = _run_in_thread(
                    lambda: client.post("/mesh/agents/scout/archive", headers=_OP).status_code,
                )
                try:
                    assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2), (
                        "the archive never queued on the lifecycle lock"
                    )
                    retire_agent("scout")
                finally:
                    pass
            t.join(5)
            assert box.get("value") == 409, box
            assert cfg["agents"]["scout"]["status"] == "active", (
                "the archive took the replacement out of service"
            )
            assert cm.stopped == []
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_a_wake_refuses_a_replacement(self, tmp_path, monkeypatch):
        """The wake's claim named an agent that no longer exists."""
        from tests.test_hibernation import _build_app

        app, bb, cm, _tr, _hm, _eb, _cfg = _build_app(
            tmp_path, monkeypatch, agent_status="hibernated",
        )
        try:
            async def replace_while_the_wake_queues():
                async with agent_lifecycle_locked_async("scout"):
                    await asyncio.sleep(0.05)
                    retire_agent("scout")

            holder = asyncio.create_task(replace_while_the_wake_queues())
            await asyncio.sleep(0)
            wake = asyncio.create_task(app.ensure_agent_running("scout", trigger="test"))
            await asyncio.wait_for(holder, 3)
            assert await asyncio.wait_for(wake, 5) is False
            assert cm.started == [], "the wake started a container for the replacement"
        finally:
            bb.close()


class TestRoundFourGuards:
    def test_the_incarnation_table_is_bounded(self):
        from src.host import agent_lifecycle as al

        for i in range(al._MAX_INCARNATIONS + 200):
            retire_agent(f"spawn-{i}")
        assert len(al._incarnations) <= al._MAX_INCARNATIONS, (
            "ephemeral spawns mint a fresh id per call — the table has to shed"
        )
        # The most recent retirement always survives its own eviction pass,
        # and reads back higher than anything evicted before it.
        last = agent_incarnation(f"spawn-{al._MAX_INCARNATIONS + 199}")
        assert last == al._MAX_INCARNATIONS + 200
        assert last > al._incarnation_floor

    def test_eviction_can_never_re_issue_a_captured_value(self):
        """Eviction has to fail CLOSED.

        A per-id count restarts an evicted id at 1 and hands a stale holder
        of 1 a false match — it acts on the replacement believing it is the
        agent it captured. A global sequence plus a floor that rises past
        everything evicted makes an evicted entry read back HIGHER than any
        captured value, so the comparison can only refuse.
        """
        from src.host import agent_lifecycle as al

        captured = retire_agent("scout")
        for i in range(al._MAX_INCARNATIONS + 10):
            retire_agent(f"filler-{i}")
        assert "scout" not in al._incarnations, "the entry under test was not evicted"
        assert agent_incarnation("scout") != captured
        assert agent_incarnation("scout") > captured
        # And retiring the name again still cannot land back on it.
        assert retire_agent("scout") != captured

    @pytest.mark.asyncio
    async def test_offboard_refuses_a_replacement(self, tmp_path, monkeypatch):
        """Offboard's window is the handover turn plus the lock wait.

        Covers the outcome — nothing stopped, status untouched, 409. It does
        NOT discriminate where the check fires: the endpoint checks the
        incarnation again right after the handover so the lead teardown and
        status write BETWEEN the handover and the archive can't land on a
        replacement, and reaching only those needs a team with this agent as
        its lead. That earlier check is covered by inspection.
        """
        from fastapi.testclient import TestClient

        from tests.test_hibernation import _OP, _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(tmp_path, monkeypatch)
        try:
            client = TestClient(app)
            async with agent_lifecycle_locked_async("scout"):
                # The handover turn runs unlocked, so the request gets that
                # far and then queues; retire while it is parked there.
                t, box = _run_in_thread(
                    lambda: client.post("/mesh/agents/scout/offboard", headers=_OP).status_code,
                )
                assert await _spin_until_async(lambda: lifecycle_refcount("scout") == 2)
                retire_agent("scout")
            t.join(5)
            assert box.get("value") == 409, box
            assert cfg["agents"]["scout"]["status"] == "active"
            assert cm.stopped == []
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_restart_all_pins_targets_before_it_reads_config(self, dashboard, monkeypatch):
        """Capture order decides which agent the stale row is applied to.

        Captured BEFORE the config read, an agent replaced between the two
        has a stale incarnation and is skipped. Captured after, its NEW
        incarnation is pinned against the OLD agent's row — and it gets
        restarted with the previous agent's role, model and tools_dir.
        """
        from src.cli import config as cli_config

        router, registry, runtime, yaml_state, _connectors, _forgotten = dashboard
        restart_all = _endpoint(router, "/api/restart-agents", "POST")

        full_cfg = {
            "agents": yaml_state["agents"],
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "network": {},
            "mesh": {},
        }
        fired = {"done": False}

        def _load_config_that_replaces():
            # Stands in for a REPL/other-loop delete+create landing between
            # the target snapshot and the config read.
            if not fired["done"]:
                fired["done"] = True
                retire_agent("worker")
            return full_cfg

        monkeypatch.setattr(cli_config, "_load_config", _load_config_that_replaces)

        result = await asyncio.wait_for(restart_all(), 5)
        assert fired["done"], "the window never opened"
        assert result["restarted"]["worker"] == "skipped: agent was replaced"
        assert "start" not in runtime.kinds()


class TestRoundFiveGuards:
    @pytest.mark.asyncio
    async def test_a_confirmed_delete_checks_the_incarnation_it_was_proposed_against(
        self, tmp_path, monkeypatch,
    ):
        """The widest window in the system is a human confirmation.

        A delete proposal sits in the ledger for its whole TTL. Nothing else
        in the row can tell that the name changed hands in between — the
        target is stored as a name.
        """
        from fastapi import HTTPException

        from tests.test_hibernation import _build_app

        app, bb, cm, _tr, _hm, _eb, _cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        try:
            delete = app.pending_executors["delete"]
            proposed_at = agent_incarnation_token("scout")
            retire_agent("scout")  # deleted and recreated before confirmation

            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(
                    delete({
                        "target_kind": "agent",
                        "target_id": "scout",
                        "nonce": "n1",
                        "payload": {"incarnation": proposed_at},
                    }),
                    5,
                )
            assert exc.value.status_code == 409
            assert "cannot be matched" in str(exc.value.detail)
            assert cm.stopped == []
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_a_delete_row_with_no_stamp_is_refused(self, tmp_path, monkeypatch):
        """Rows proposed before this field exists cannot be checked.

        Running an irreversible action blind is the worse option; the
        operator re-proposes, and the row's TTL is minutes.
        """
        from fastapi import HTTPException

        from tests.test_hibernation import _build_app

        app, bb, cm, _tr, _hm, _eb, _cfg = _build_app(tmp_path, monkeypatch)
        try:
            delete = app.pending_executors["delete"]
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(
                    delete({"target_kind": "agent", "target_id": "scout", "nonce": "n1"}),
                    5,
                )
            assert exc.value.status_code == 409
            assert "cannot be matched" in str(exc.value.detail)
            assert cm.stopped == []
        finally:
            bb.close()

    @pytest.mark.asyncio
    async def test_restart_all_skips_an_agent_with_no_config_row(self, dashboard):
        """Restarting from a missing row rebuilds the container from defaults.

        That is how an ephemeral spawn — registry-only, never in agents.yaml
        — and an agent whose row vanished mid-snapshot both look here.
        """
        router, registry, runtime, _yaml, _connectors, _forgotten = dashboard
        restart_all = _endpoint(router, "/api/restart-agents", "POST")
        registry["spawn-abc"] = "http://spawn-abc:8400"

        result = await asyncio.wait_for(restart_all(), 5)

        assert result["restarted"]["spawn-abc"] == "skipped: no config for this agent"
        assert ("start", "spawn-abc") not in runtime.calls
        assert result["restarted"]["worker"] == "ready"


class TestDurableStamps:
    def test_a_token_from_another_process_never_matches(self):
        """The counter is in-memory; a pending-action row is not.

        A bare counter stamped ``0`` before a restart compares equal to a
        fresh process's ``0`` — for whichever agent holds the name
        afterwards. That is a false pass on an irreversible action.
        """
        from src.host import agent_lifecycle as al

        mine = agent_incarnation_token("scout")
        assert incarnation_token_matches("scout", mine)

        previous_process = f"{'0' * 32}:{os.getpid()}:scout:{agent_incarnation('scout')}"
        assert previous_process != mine
        assert not incarnation_token_matches("scout", previous_process), (
            "a stamp from a process whose counter is gone cannot be verified"
        )
        # And the bare counter it embeds WOULD have matched.
        assert previous_process.rsplit(":", 1)[1] == mine.rsplit(":", 1)[1]
        assert al._BOOT_ID in mine

    def test_a_token_is_agent_specific_and_moves_with_the_incarnation(self):
        scout = agent_incarnation_token("scout")
        assert not incarnation_token_matches("other", scout)
        assert not incarnation_token_matches("scout", None)
        assert not incarnation_token_matches("scout", "not-a-token")
        retire_agent("scout")
        assert not incarnation_token_matches("scout", scout)

    @pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork()")
    def test_a_forked_child_does_not_inherit_a_usable_token(self):
        """``fork()`` copies the boot id AND the counter.

        A pre-fork worker would otherwise accept a sibling's stamp: same
        uuid, same counter, different process — exactly the case the boot id
        exists to reject. The pid is read at call time for that; the uuid
        still covers pid reuse across ordinary restarts, which the pid alone
        would not.
        """
        token = agent_incarnation_token("scout")
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the forked child
            try:
                os.close(read_fd)
                matched = incarnation_token_matches("scout", token)
                os.write(write_fd, b"1" if matched else b"0")
                os.close(write_fd)
            finally:
                os._exit(0)
        os.close(write_fd)
        try:
            child_saw = os.read(read_fd, 1)
        finally:
            os.close(read_fd)
            os.waitpid(pid, 0)
        assert child_saw == b"0", (
            "a forked child accepted the parent's token — the boot id alone "
            "is copied by fork"
        )
        # The parent's own token still matches, so the pid did not break it.
        assert incarnation_token_matches("scout", token)


class TestUnarchiveDuringADelete:
    @pytest.mark.asyncio
    async def test_a_delete_refuses_an_agent_returned_to_service(self, tmp_path, monkeypatch):
        """Unarchive deliberately does NOT bump the incarnation.

        It is not a delete — the agent is the same one. So an unarchive
        completing while a delete's handover turn runs is invisible to the
        identity check, and only re-reading the status under the lock
        catches it. Propose-delete requires an archived agent precisely so
        the container is already stopped; destroying one that is back in
        service is the failure this prevents.
        """
        from fastapi import HTTPException

        from tests.test_hibernation import _build_app

        app, bb, cm, _tr, _hm, _eb, cfg = _build_app(
            tmp_path, monkeypatch, agent_status="archived",
        )
        try:
            delete = app.pending_executors["delete"]
            record = {
                "target_kind": "agent",
                "target_id": "scout",
                "nonce": "n1",
                "payload": {"incarnation": agent_incarnation_token("scout")},
            }

            async def unarchive_while_the_delete_queues():
                async with agent_lifecycle_locked_async("scout"):
                    await asyncio.sleep(0.05)
                    cfg["agents"]["scout"]["status"] = "active"

            holder = asyncio.create_task(unarchive_while_the_delete_queues())
            await asyncio.sleep(0)
            queued = asyncio.create_task(delete(record))
            await asyncio.wait_for(holder, 3)
            with pytest.raises(HTTPException) as exc:
                await asyncio.wait_for(queued, 5)
            assert exc.value.status_code == 409
            assert "returned to service" in str(exc.value.detail)
            assert cm.stopped == [], "the delete destroyed an agent back in service"
        finally:
            bb.close()
