"""The LLM proxy must not do its blocking work on the mesh event loop.

Roadmap 1b. Every LLM call from every agent lands on ``/mesh/api`` or
``/mesh/api/stream``, and on the baseline each one spent ~19.6 ms of
synchronous work on the mesh loop before any of it reached a provider:

  * three uncached YAML parses (13.1 ms for a 30-agent fleet) — the H3
    model pin and the coordination-tier check each call ``_load_config``;
  * ~8 SQLite round-trips across two databases (~6 ms) for the budget
    preflight, the team envelope, and the post-call usage write;
  * a redact + INSERT + commit into the trace store.

Because that all ran on the loop, it was not just this request's latency —
it was a ceiling of ~51 LLM calls/sec for the whole mesh, and every
dashboard poll, health probe and routed message queued behind it.

Three changes, and one test class each:

  1. ``_load_config`` caches, keyed on the CONTENT of the three documents.
  2. ``CostTracker`` and ``TraceStore`` serialize their shared connection,
     which they must before anything may call them from a worker thread.
  3. The proxy's SQLite work runs through ``asyncio.to_thread``.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import os
import threading
from pathlib import Path

import pytest
import yaml

from src.host.costs import CostTracker
from src.host.traces import TraceStore
from src.shared.types import APIProxyRequest, APIProxyResponse

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── (1) the config cache ──────────────────────────────────────────────


@pytest.fixture
def cfg_files(tmp_path, monkeypatch):
    """Point ``cli.config`` at a private config dir and return the paths.

    Every test gets its own ``tmp_path``, and the cache key includes the
    paths, so tests can never see each other's cached parse.
    """
    import src.cli.config as cfgmod

    cdir = tmp_path / "config"
    cdir.mkdir()
    mesh, agents, network = cdir / "mesh.yaml", cdir / "agents.yaml", cdir / "network.yaml"
    monkeypatch.setattr(cfgmod, "CONFIG_FILE", mesh)
    monkeypatch.setattr(cfgmod, "AGENTS_FILE", agents)
    monkeypatch.setattr(cfgmod, "NETWORK_FILE", network)
    return {"mod": cfgmod, "mesh": mesh, "agents": agents, "network": network}


def _write_agents(path: Path, **rows) -> None:
    path.write_text(yaml.safe_dump({"agents": rows}))


class TestConfigCache:
    def test_unchanged_files_are_not_reparsed(self, cfg_files, monkeypatch):
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "openai/gpt-4o-mini"})

        parses = []
        real = cfgmod._yaml_loads
        monkeypatch.setattr(
            cfgmod, "_yaml_loads", lambda raw: (parses.append(1), real(raw))[1],
        )

        first = cfgmod._load_config()
        after_first = len(parses)
        for _ in range(5):
            assert cfgmod._load_config() == first
        assert after_first > 0, "the first call must actually parse"
        assert len(parses) == after_first, "repeat calls re-parsed the documents"

    def test_changed_content_is_picked_up(self, cfg_files):
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "a"})
        assert cfgmod._load_config()["agents"]["writer"]["model"] == "a"
        _write_agents(cfg_files["agents"], writer={"model": "bbbbbbbb"})
        assert cfgmod._load_config()["agents"]["writer"]["model"] == "bbbbbbbb"

    def test_same_size_rewrite_in_the_same_mtime_tick_is_seen(self, cfg_files):
        """The reason the key is content and not stat metadata.

        Linux stamps mtime from a coarse clock that only advances once per
        timer tick, so two writes inside one tick carry the SAME
        ``st_mtime_ns``. Pair that with an edit that happens to preserve
        the file size — swapping one model id for another of equal length
        does exactly that — and a (mtime, size) cache key is identical
        across a real change. Reproduced here deterministically by
        stamping the mtime back with ``os.utime``.
        """
        cfgmod = cfg_files["mod"]
        agents = cfg_files["agents"]

        _write_agents(agents, writer={"model": "openai/gpt-4o-mini"})
        before = agents.stat()
        assert cfgmod._load_config()["agents"]["writer"]["model"] == "openai/gpt-4o-mini"

        _write_agents(agents, writer={"model": "openai/gpt-4o-MAXI"})
        os.utime(agents, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = agents.stat()
        assert (after.st_mtime_ns, after.st_size, after.st_ino) == (
            before.st_mtime_ns, before.st_size, before.st_ino,
        ), "the fixture failed to reproduce an indistinguishable stat signature"

        assert cfgmod._load_config()["agents"]["writer"]["model"] == "openai/gpt-4o-MAXI"

    def test_caller_mutation_cannot_poison_the_cache(self, cfg_files):
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "openai/gpt-4o-mini"})

        mine = cfgmod._load_config()
        mine["agents"]["writer"]["model"] = "anthropic/claude-opus-5"
        mine["agents"]["intruder"] = {"model": "x"}
        mine["llm"]["default_model"] = "tampered"

        clean = cfgmod._load_config()
        assert clean["agents"]["writer"]["model"] == "openai/gpt-4o-mini"
        assert "intruder" not in clean["agents"]
        assert clean["llm"]["default_model"] != "tampered"

    def test_each_call_returns_an_independent_copy(self, cfg_files):
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "m"})
        a, b = cfgmod._load_config(), cfgmod._load_config()
        assert a == b
        assert a is not b
        assert a["agents"] is not b["agents"]
        assert a["agents"]["writer"] is not b["agents"]["writer"]

    def test_explicit_mesh_path_bypasses_the_cache(self, cfg_files, tmp_path):
        cfgmod = cfg_files["mod"]
        other = tmp_path / "other-mesh.yaml"
        other.write_text(yaml.safe_dump({"llm": {"default_model": "explicit/one"}}))
        cfg_files["mesh"].write_text(yaml.safe_dump({"llm": {"default_model": "live/one"}}))

        assert cfgmod._load_config()["llm"]["default_model"] == "live/one"
        assert cfgmod._load_config(other)["llm"]["default_model"] == "explicit/one"
        # …and the explicit read must not have become the cached answer.
        assert cfgmod._load_config()["llm"]["default_model"] == "live/one"

    def test_absent_files_are_defaults_and_a_later_write_is_seen(self, cfg_files):
        cfgmod = cfg_files["mod"]
        assert cfgmod._load_config()["agents"] == {}
        assert cfgmod._load_config()["network"] == {}
        _write_agents(cfg_files["agents"], late={"model": "m"})
        assert "late" in cfgmod._load_config()["agents"]

    def test_a_deleted_file_reverts_to_defaults(self, cfg_files):
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "m"})
        assert "writer" in cfgmod._load_config()["agents"]
        cfg_files["agents"].unlink()
        assert cfgmod._load_config()["agents"] == {}

    def test_fast_loader_agrees_with_the_pure_python_loader(self, cfg_files):
        """The cache only pays off with the C loader; pin that they agree."""
        cfgmod = cfg_files["mod"]
        doc = yaml.safe_dump({
            "agents": {
                "wörker": {
                    "role": "analyste 🎉",
                    "goal": "x" * 300,
                    "tools": ["a", "b"],
                    "n": 12345678901234,
                    "f": 1.5,
                    "on": True,
                    "off": None,
                },
            },
        }, allow_unicode=True).encode()
        assert cfgmod._yaml_loads(doc) == yaml.load(doc, Loader=yaml.SafeLoader)

    def test_concurrent_readers_parse_once(self, cfg_files, monkeypatch):
        """A cold cache under a burst must be single-flighted, not N-flighted."""
        cfgmod = cfg_files["mod"]
        _write_agents(cfg_files["agents"], writer={"model": "m"})

        parses = []
        real = cfgmod._yaml_loads
        started = threading.Barrier(8)

        def counting(raw):
            parses.append(1)
            return real(raw)

        monkeypatch.setattr(cfgmod, "_yaml_loads", counting)

        results: list[dict] = []

        def worker():
            started.wait(timeout=10)
            results.append(cfgmod._load_config())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)
            assert not t.is_alive()

        assert len(results) == 8
        assert all(r == results[0] for r in results)
        # 3 documents, parsed once between them all.
        assert len(parses) <= 3, f"cold burst parsed {len(parses)} times"


# ── (2) the stores are safe to call from a worker thread ──────────────


def _hammer(fn, *, threads: int = 6, iterations: int = 60):
    """Run ``fn(i, j)`` from several threads; return everything it raised."""
    errors: list[BaseException] = []
    ready = threading.Barrier(threads)

    def worker(i):
        ready.wait(timeout=10)
        for j in range(iterations):
            try:
                fn(i, j)
            except BaseException as e:  # noqa: BLE001 — the point is to collect them
                errors.append(e)
                return

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
        assert not t.is_alive(), "a worker wedged"
    return errors


class TestStoresAreThreadSafe:
    """``check_same_thread=False`` permits cross-thread use; it does not make
    it safe. sqlite3 only implicitly BEGINs on DML, so an INSERT from one
    thread and a commit from another share one implicit transaction and the
    loser raises ``cannot start a transaction within a transaction``. Without
    the connection lock these fail on the billing and trace paths.
    """

    def test_cost_tracker_concurrent_writes(self, tmp_path):
        ct = CostTracker(
            db_path=str(tmp_path / "costs.db"),
            budgets_path=str(tmp_path / "budgets.json"),
        )
        errors = _hammer(lambda i, j: ct.track(f"agent{i}", "openai/gpt-4o-mini", 10, 5))
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
        with ct._db_lock:
            rows = ct.db.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
        assert rows == 6 * 60, "rows were lost"
        ct.close()

    def test_cost_tracker_concurrent_reads_and_writes(self, tmp_path):
        ct = CostTracker(
            db_path=str(tmp_path / "costs.db"),
            budgets_path=str(tmp_path / "budgets.json"),
        )

        def mixed(i, j):
            if i % 2:
                ct.track(f"agent{i}", "openai/gpt-4o-mini", 10, 5)
            else:
                ct.get_spend(f"agent{i}", "today")
                ct.preflight_check(f"agent{i}", "openai/gpt-4o-mini")

        errors = _hammer(mixed)
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
        ct.close()

    def test_trace_store_concurrent_writes(self, tmp_path):
        ts = TraceStore(db_path=str(tmp_path / "traces.db"))
        errors = _hammer(
            lambda i, j: ts.record(
                trace_id=f"t{i}", source="mesh.api_proxy", agent=f"agent{i}",
                event_type="llm_call", detail="llm/chat", meta={"model": "m"},
            ),
        )
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
        assert len(ts.list_recent(limit=1000)) == 6 * 60
        ts.close()

    def test_trace_store_concurrent_reads_and_writes(self, tmp_path):
        ts = TraceStore(db_path=str(tmp_path / "traces.db"))

        def mixed(i, j):
            if i % 2:
                ts.record(
                    trace_id=f"t{i}", source="mesh.api_proxy", agent=f"agent{i}",
                    event_type="llm_call", detail="llm/chat",
                )
            else:
                ts.list_recent(limit=20)
                ts.list_trace_summaries(limit=20)

        errors = _hammer(mixed)
        assert not errors, f"{len(errors)} failures, first: {errors[0]!r}"
        ts.close()


# ── (3) the proxy's blocking work runs off the loop ───────────────────


class _ThreadRecordingCostTracker:
    """A cost tracker that records which thread each call ran on."""

    def __init__(self):
        self.threads: dict[str, int] = {}
        self.budgets: dict[str, dict] = {}
        self._team_store = None

    def _note(self, name):
        self.threads[name] = threading.get_ident()

    def check_budget(self, agent):
        self._note("check_budget")
        return {"allowed": True, "daily_used": 0.0, "daily_limit": 1.0,
                "monthly_used": 0.0, "monthly_limit": 1.0}

    def preflight_check(self, agent, model, estimated_tokens=4096):
        self._note("preflight_check")
        return {"allowed": True, "estimated_cost": 0.0, "daily_used": 0.0,
                "daily_limit": 1.0, "monthly_used": 0.0, "monthly_limit": 1.0}

    def coordination_preflight_check(self, agent, model, estimated_tokens=4096):
        self._note("coordination_preflight_check")
        return {"allowed": True, "estimated_cost": 0.0, "daily_used": 0.0,
                "daily_limit": 1.0}

    def team_envelope_check(self, agent, model, estimated_tokens=4096):
        self._note("team_envelope_check")
        return {"allowed": True, "team": None}

    def track(self, agent, model, prompt_tokens, completion_tokens, *, bill=True, kind="work"):
        self._note("track")
        return {"cost": 0.0, "over_budget": False}

    def track_fixed_cost(self, agent, model, cost_usd):
        self._note("track_fixed_cost")
        return {"cost": cost_usd, "over_budget": False}


def _vault_with(tracker):
    from src.host.credentials import CredentialVault

    vault = CredentialVault(cost_tracker=tracker)

    async def _llm(request):
        return APIProxyResponse(
            success=True,
            data={"content": "ok", "tokens_used": 12, "input_tokens": 8,
                  "output_tokens": 4, "model": request.params.get("model", "")},
        )

    vault.service_handlers["llm"] = _llm
    return vault


def _llm_request(model="openai/gpt-4o-mini"):
    return APIProxyRequest(
        service="llm", action="chat",
        params={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


class TestProxyWorkRunsOffTheLoop:
    @pytest.mark.asyncio
    async def test_budget_preflight_and_usage_write_are_off_the_loop(self):
        tracker = _ThreadRecordingCostTracker()
        vault = _vault_with(tracker)
        loop_thread = threading.get_ident()

        result = await vault.execute_api_call(_llm_request(), agent_id="writer")

        assert result.success
        for name in ("preflight_check", "team_envelope_check", "track"):
            assert name in tracker.threads, f"{name} was never called"
            assert tracker.threads[name] != loop_thread, (
                f"{name} ran on the event loop thread"
            )

    @pytest.mark.asyncio
    async def test_coordination_preflight_is_off_the_loop(self):
        tracker = _ThreadRecordingCostTracker()
        vault = _vault_with(tracker)
        vault.set_utility_model_provider(lambda: "openai/gpt-4o-mini")
        loop_thread = threading.get_ident()

        await vault.execute_api_call(_llm_request(), agent_id="writer")

        assert tracker.threads["coordination_preflight_check"] != loop_thread

    @pytest.mark.asyncio
    async def test_image_gen_budget_check_is_off_the_loop(self):
        tracker = _ThreadRecordingCostTracker()
        vault = _vault_with(tracker)
        loop_thread = threading.get_ident()

        async def _image(request):
            return APIProxyResponse(
                success=True,
                data={"content": "", "tokens_used": 0, "fixed_cost_usd": 0.04,
                      "model": "img"},
            )

        vault.service_handlers["image_gen"] = _image
        req = APIProxyRequest(service="image_gen", action="generate",
                              params={"model": "img", "prompt": "a cat"})

        await vault.execute_api_call(req, agent_id="writer")

        assert tracker.threads["check_budget"] != loop_thread
        assert tracker.threads["track_fixed_cost"] != loop_thread

    @pytest.mark.asyncio
    async def test_streaming_preflight_and_usage_write_are_off_the_loop(self, monkeypatch):
        tracker = _ThreadRecordingCostTracker()
        vault = _vault_with(tracker)
        loop_thread = threading.get_ident()

        async def _fake_stream(*a, **k):
            yield {"type": "content", "content": "hi"}
            yield {"type": "done", "content": "hi", "tokens_used": 12,
                   "input_tokens": 8, "output_tokens": 4,
                   "model": "openai/gpt-4o-mini"}

        monkeypatch.setattr(vault, "_stream_llm_chunks", _fake_stream, raising=False)

        chunks = []
        async for chunk in vault.stream_llm(_llm_request(), agent_id="writer"):
            chunks.append(chunk)

        assert "preflight_check" in tracker.threads
        assert tracker.threads["preflight_check"] != loop_thread
        assert tracker.threads["team_envelope_check"] != loop_thread

    @pytest.mark.asyncio
    async def test_a_blocking_tracker_does_not_stall_the_loop(self):
        """The end the change exists for: a slow ledger must not stop the mesh.

        The tracker sleeps 250 ms per preflight. A ticker coroutine counts
        how many times it gets scheduled meanwhile. On the loop, the ticker
        is frozen for the whole call.
        """
        tracker = _ThreadRecordingCostTracker()
        real_preflight = tracker.preflight_check

        def slow(agent, model, estimated_tokens=4096):
            import time
            time.sleep(0.25)
            return real_preflight(agent, model, estimated_tokens)

        tracker.preflight_check = slow
        vault = _vault_with(tracker)

        ticks = 0
        stop = False

        async def ticker():
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.005)
                ticks += 1

        t = asyncio.create_task(ticker())
        await vault.execute_api_call(_llm_request(), agent_id="writer")
        stop = True
        await t

        assert ticks >= 10, (
            f"the loop only got {ticks} slices while the ledger blocked for 250 ms"
        )


class TestTraceStampSurvivesTheThreadHop:
    """``track`` stamps the usage row from the ``current_trace_id``
    contextvar the endpoint seeded from ``X-Trace-Id``. ``asyncio.to_thread``
    copies the calling context; ``loop.run_in_executor`` does not. Swapping
    one for the other would leave every usage row with a NULL trace_id and
    break spend-per-task attribution silently, so pin it against the REAL
    ``execute_api_call`` — the existing coverage in ``test_mesh.py`` stubs
    that method out and never crosses the hop.
    """

    @pytest.mark.asyncio
    async def test_usage_row_keeps_the_trace_id_across_to_thread(self, tmp_path):
        from src.shared.trace import current_trace_id

        tracker = CostTracker(
            db_path=str(tmp_path / "costs.db"),
            budgets_path=str(tmp_path / "budgets.json"),
        )
        vault = _vault_with(tracker)

        token = current_trace_id.set("trace-abc")
        try:
            result = await vault.execute_api_call(_llm_request(), agent_id="writer")
        finally:
            current_trace_id.reset(token)

        assert result.success
        with tracker._db_lock:
            rows = tracker.db.execute(
                "SELECT agent, trace_id FROM usage ORDER BY id",
            ).fetchall()
        assert rows == [("writer", "trace-abc")], rows
        tracker.close()

    @pytest.mark.asyncio
    async def test_no_trace_header_stamps_null_not_a_stale_id(self, tmp_path):
        from src.shared.trace import current_trace_id

        tracker = CostTracker(
            db_path=str(tmp_path / "costs.db"),
            budgets_path=str(tmp_path / "budgets.json"),
        )
        vault = _vault_with(tracker)

        token = current_trace_id.set(None)
        try:
            await vault.execute_api_call(_llm_request(), agent_id="writer")
        finally:
            current_trace_id.reset(token)

        with tracker._db_lock:
            rows = tracker.db.execute("SELECT trace_id FROM usage").fetchall()
        assert rows == [(None,)], rows
        tracker.close()


class _ThreadRecordingTraceStore:
    """A trace store that records which thread each ``record`` ran on."""

    def __init__(self):
        self.threads: list[int] = []
        self.events: list[str] = []

    def record(self, *, trace_id, source, agent, event_type, detail="",
               duration_ms=0, status="", error="", meta=None):
        self.threads.append(threading.get_ident())
        self.events.append(event_type)


class _EchoVault:
    """Just enough vault for the mesh proxy routes."""

    def __init__(self):
        self.stream_chunks = [
            'data: {"type": "content", "content": "hi"}\n\n',
            'data: {"type": "done", "content": "hi", "tokens_used": 12, '
            '"model": "openai/gpt-4o-mini"}\n\n',
        ]

    async def execute_api_call(self, request, agent_id=""):
        return APIProxyResponse(
            success=True,
            data={"content": "ok", "tokens_used": 12, "input_tokens": 8,
                  "output_tokens": 4, "model": request.params.get("model", "")},
        )

    async def stream_llm(self, request, agent_id=""):
        for chunk in self.stream_chunks:
            yield chunk

    def is_model_compatible(self, model):
        return (True, None)


@pytest.fixture
def proxy_app(tmp_path, monkeypatch):
    """A mesh app whose only wired stores record the thread they run on."""
    import src.host.server as server_module
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import AgentPermissions, PermissionMatrix

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    server = importlib.reload(server_module)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "operator": AgentPermissions(agent_id="operator"),
        "writer": AgentPermissions(agent_id="writer", allowed_apis=["llm"]),
    }
    perms._config_path = str(tmp_path / "perms.json")

    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("writer", "http://writer:8400", [])

    trace_store = _ThreadRecordingTraceStore()
    monkeypatch.setattr(
        "src.cli.config._load_config",
        lambda *a, **k: {"llm": {}, "agents": {"writer": {}}},
    )

    app = server.create_mesh_app(
        blackboard=bb,
        pubsub=PubSub(),
        router=router,
        permissions=perms,
        credential_vault=_EchoVault(),
        auth_tokens={"writer": "writer-secret"},
        trace_store=trace_store,
    )
    yield {"app": app, "trace_store": trace_store}
    bb.close()
    monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
    importlib.reload(server_module)


def _proxy_body(model="openai/gpt-4o-mini"):
    return {
        "service": "llm",
        "action": "chat",
        "params": {"model": model, "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 16},
        "timeout": 30,
    }


class TestTraceWritesRunOffTheLoop:
    """``record`` redacts, INSERTs and commits — and every five minutes also
    sweeps the whole table. All of that used to land on the mesh loop, once
    per LLM call on the sync path and twice on the streaming one.
    """

    @pytest.mark.asyncio
    async def test_sync_proxy_trace_write_is_off_the_loop(self, proxy_app):
        from httpx import ASGITransport, AsyncClient

        store = proxy_app["trace_store"]
        loop_thread = threading.get_ident()

        transport = ASGITransport(app=proxy_app["app"])
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/mesh/api", json=_proxy_body(), params={"agent_id": "writer"},
                headers={"authorization": "Bearer writer-secret",
                         "x-trace-id": "trace-1"},
            )
        assert resp.status_code == 200, resp.text
        assert store.threads, "the trace write never happened"
        assert all(t != loop_thread for t in store.threads), (
            "a trace write ran on the event loop thread"
        )

    @pytest.mark.asyncio
    async def test_streaming_proxy_trace_writes_are_off_the_loop(self, proxy_app):
        from httpx import ASGITransport, AsyncClient

        store = proxy_app["trace_store"]
        loop_thread = threading.get_ident()

        transport = ASGITransport(app=proxy_app["app"])
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream(
                "POST", "/mesh/api/stream", json=_proxy_body(),
                params={"agent_id": "writer"},
                headers={"authorization": "Bearer writer-secret",
                         "x-trace-id": "trace-2"},
            ) as resp:
                assert resp.status_code == 200
                async for _ in resp.aiter_bytes():
                    pass

        # Both the stream-open marker and the post-stream completion row.
        assert set(store.events) >= {"llm_stream", "llm_call"}, store.events
        assert all(t != loop_thread for t in store.threads), (
            "a trace write ran on the event loop thread"
        )


# ── structural guards ─────────────────────────────────────────────────


def _module_tree(relpath: str) -> ast.Module:
    return ast.parse((REPO_ROOT / relpath).read_text())


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    out: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


class TestConnectionUseStaysLocked:
    """A missed site is a crash on the billing path under concurrency, so the
    rule is enforced structurally rather than trusted to review.
    """

    @pytest.mark.parametrize(
        ("relpath", "conn_attr", "lock_attr"),
        [
            ("src/host/costs.py", "db", "_db_lock"),
            ("src/host/traces.py", "_conn", "_db_lock"),
        ],
    )
    def test_every_connection_use_is_inside_the_lock(self, relpath, conn_attr, lock_attr):
        tree = _module_tree(relpath)
        parents = _parents(tree)

        def guarded_by_lock(node) -> bool:
            cur = node
            while cur is not None:
                if isinstance(cur, ast.With):
                    for item in cur.items:
                        ctx = item.context_expr
                        if isinstance(ctx, ast.Attribute) and ctx.attr == lock_attr:
                            return True
                # A ``*_locked`` helper documents that its caller holds it.
                if isinstance(cur, ast.FunctionDef) and cur.name.endswith("_locked"):
                    return True
                cur = parents.get(id(cur))
            return False

        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr == conn_attr):
                continue
            if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                continue
            parent = parents.get(id(node))
            # ``self.db = open_db(...)`` binds the connection; it uses nothing.
            if isinstance(parent, ast.Assign) and node in parent.targets:
                continue
            if not guarded_by_lock(node):
                offenders.append(f"{relpath}:{node.lineno}")

        assert not offenders, (
            f"self.{conn_attr} used outside `with self.{lock_attr}` at: {offenders}"
        )


class TestLedgerCallsGoThroughAThread:
    def test_no_cost_tracker_db_method_is_called_directly(self):
        """Derived from ``costs.py`` itself, so a new DB method joins the rule.

        A direct call appears as ``Call(func=Attribute(...))``; the
        ``asyncio.to_thread`` form passes the bound method as an ARGUMENT, so
        it is an ``Attribute`` that is not any ``Call``'s ``func``.
        """
        sources = {
            name: inspect.getsource(fn)
            for name, fn in inspect.getmembers(CostTracker, inspect.isfunction)
        }
        # Seed on the methods that touch the connection, then close over
        # ``self.<other>(...)`` — ``preflight_check`` never names ``self.db``,
        # it goes through ``get_spend``.
        blocking = {n for n, src in sources.items() if "self.db" in src}
        changed = True
        while changed:
            changed = False
            for name, src in sources.items():
                if name in blocking:
                    continue
                if any(f"self.{b}(" in src for b in blocking):
                    blocking.add(name)
                    changed = True
        assert {"track", "preflight_check", "team_envelope_check",
                "coordination_preflight_check", "check_budget"} <= blocking

        tree = _module_tree("src/host/credentials.py")
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr in blocking):
                continue
            inner = fn.value
            if isinstance(inner, ast.Attribute) and inner.attr == "cost_tracker":
                offenders.append(f"{fn.attr} at src/host/credentials.py:{node.lineno}")

        assert not offenders, (
            "these ledger calls run on the mesh loop; wrap them in "
            f"asyncio.to_thread: {offenders}"
        )


def test_reimporting_config_does_not_leak_a_populated_cache():
    """A reload must not resurrect another test's parse under a fresh module."""
    import src.cli.config as cfgmod

    reloaded = importlib.reload(cfgmod)
    assert reloaded._config_cache_key is None
    assert reloaded._config_cache_value is None
