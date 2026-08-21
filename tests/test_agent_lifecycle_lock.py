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
import pathlib
import threading
import time

import pytest

from src.host import agent_lifecycle as lifecycle
from src.host.agent_lifecycle import (
    AgentLifecycleBusy,
    agent_lifecycle_locked,
    agent_lifecycle_locked_async,
    lifecycle_refcount,
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
    )
    yield router, registry, runtime, yaml_state, connectors
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
        router, registry, runtime, yaml_state, connectors = dashboard
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
        router, registry, runtime, yaml_state, connectors = dashboard
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

        router, registry, runtime, _yaml, _connectors = dashboard
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

        router, _registry, _runtime, _yaml, _connectors = dashboard
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
