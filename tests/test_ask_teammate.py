"""ask_teammate — mesh-brokered inline Q&A (Phase 2 unit 3).

Covers the AskBroker (concurrency caps, single-use resolution, billing
window mechanics), the /mesh/ask + /mesh/ask/{id}/answer endpoints
(auth matrix, busy-path steer injection, idle-path followup dispatch
with inline-response fallback, timeout/rate-limit envelopes), the
mesh-authoritative billing seam in ``credentials.execute_api_call``
(usage rows land on the ASKER during a window; asker budget preflight;
per-ask cap closes the window; recipient billed after close), and the
agent-side ``ask_teammate`` / ``answer_ask`` tool envelopes
(Constraint #10 shape on every failure path).

Broker restart semantics are IN-MEMORY BY DESIGN: a mesh restart drops
the registry, the asker's pending HTTP call dies with the process and
the tool returns a failure envelope; a late answer sees the unknown-ask
envelope. Pinned by ``test_fresh_broker_treats_old_ask_as_unknown``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from src.host.asks import AskBroker, AskLimitExceeded
from src.host.costs import CostTracker
from src.host.lanes import LaneManager
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.host.teams import TeamStore
from src.shared import limits
from src.shared.types import AgentPermissions

_OP = {"Authorization": "Bearer tok-op"}
_ASKER = {"Authorization": "Bearer tok-a"}
_HELPER = {"Authorization": "Bearer tok-h"}
_STRANGER = {"Authorization": "Bearer tok-s"}
_TOKENS = {
    "operator": "tok-op",
    "asker": "tok-a",
    "helper": "tok-h",
    "stranger": "tok-s",
    "rival": "tok-r",
}


def _perms() -> PermissionMatrix:
    p = PermissionMatrix.__new__(PermissionMatrix)
    p.permissions = {
        "asker": AgentPermissions(
            agent_id="asker", can_message=["helper", "rival", "operator"],
        ),
        "helper": AgentPermissions(agent_id="helper", can_message=["asker"]),
        # "stranger" deliberately has NO grants (deny-all default).
        "rival": AgentPermissions(agent_id="rival", can_message=["asker"]),
    }
    return p


def _teams() -> TeamStore:
    store = TeamStore(db_path=":memory:")
    store.create_team("alpha")
    store.add_member("alpha", "asker")
    store.add_member("alpha", "helper")
    store.create_team("beta")
    store.add_member("beta", "rival")
    return store


def _build_app(
    tmp_path,
    monkeypatch,
    *,
    lane_manager=None,
    broker=None,
    teams=None,
    vault=None,
    tracker=None,
):
    monkeypatch.setenv(
        "OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"),
    )
    perms = _perms()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = MessageRouter(
        permissions=perms,
        agent_registry={
            "operator": "http://op:8400",
            "asker": "http://a:8400",
            "helper": "http://h:8400",
            "rival": "http://r:8400",
            "stranger": "http://s:8400",
        },
    )
    router.agent_roles.update(
        {"asker": "researcher", "helper": "analyst", "rival": "seo"},
    )
    app = create_mesh_app(
        bb,
        PubSub(),
        router,
        perms,
        vault,
        cost_tracker=tracker,
        teams_store=teams if teams is not None else _teams(),
        lane_manager=lane_manager,
        auth_tokens=dict(_TOKENS),
        ask_broker=broker,
    )
    return app, bb


def _fast_ask_timeout(monkeypatch, seconds=1):
    """Shrink the ask wait so timeout-path tests run in ~1s."""
    monkeypatch.setitem(
        limits.LIMIT_SPECS, "ask_timeout_seconds", (seconds, 1, 10),
    )


class _FakeLanes:
    """Steer/dispatch doubles built on the REAL LaneManager (the mocks
    replace the transport-to-agent edge, never the lane machinery)."""

    def __init__(self, *, busy: bool, response: str = "", on_dispatch=None):
        self.busy = busy
        self.response = response
        self.steer_messages: list[str] = []
        self.dispatch_messages: list[str] = []
        self._on_dispatch = on_dispatch

        async def steer_fn(agent, message, system_note=False):
            self.steer_messages.append(message)
            return {"injected": self.busy}

        async def dispatch_fn(agent, message, **kwargs):
            self.dispatch_messages.append(message)
            if self._on_dispatch is not None:
                maybe = self._on_dispatch(agent, message)
                if asyncio.iscoroutine(maybe):
                    await maybe
            return self.response

        self.manager = LaneManager(dispatch_fn=dispatch_fn, steer_fn=steer_fn)


# ── Broker unit tests ────────────────────────────────────────────────


class TestAskBroker:
    async def test_per_asker_concurrency_cap(self):
        broker = AskBroker()
        for _ in range(AskBroker.MAX_ACTIVE_PER_ASKER):
            broker.create("a", "b", "q", 30)
        with pytest.raises(AskLimitExceeded):
            broker.create("a", "b", "q", 30)
        # A different asker still fits.
        broker.create("c", "b", "q", 30)

    async def test_global_cap(self):
        broker = AskBroker()
        broker.MAX_ACTIVE_GLOBAL = 2
        broker.create("a1", "b", "q", 30)
        broker.create("a2", "b", "q", 30)
        with pytest.raises(AskLimitExceeded):
            broker.create("a3", "b", "q", 30)

    async def test_resolve_single_use_and_recipient_verified(self):
        broker = AskBroker()
        rec = broker.create("a", "b", "q", 30)
        assert broker.resolve(rec.ask_id, "ans", by="mallory") == {
            "ok": False, "reason": "wrong_recipient",
        }
        assert broker.resolve(rec.ask_id, "ans", by="b") == {"ok": True}
        await asyncio.sleep(0)  # let call_soon_threadsafe land
        assert rec.future.result() == "ans"
        assert broker.resolve(rec.ask_id, "again", by="b") == {
            "ok": False, "reason": "already_resolved",
        }

    async def test_resolve_inline_noop_after_answer(self):
        broker = AskBroker()
        rec = broker.create("a", "b", "q", 30)
        assert broker.resolve(rec.ask_id, "real", by="b")["ok"]
        await asyncio.sleep(0)
        assert broker.resolve_inline(rec.ask_id, "fallback") is False
        assert rec.future.result() == "real"

    async def test_fresh_broker_treats_old_ask_as_unknown(self):
        """Restart semantics: in-memory by design — a new broker knows
        nothing about pre-restart asks; late answers are non-fatal."""
        old = AskBroker()
        rec = old.create("a", "b", "q", 30)
        restarted = AskBroker()
        assert restarted.resolve(rec.ask_id, "late", by="b") == {
            "ok": False, "reason": "unknown",
        }

    async def test_billing_window_lifecycle_and_cap(self):
        broker = AskBroker(bill_cap_usd=0.10)
        rec = broker.create("a", "b", "q", 30)
        assert broker.active_billing_for("b") is None
        assert broker.activate_billing(rec.ask_id) is True
        assert broker.active_billing_for("b") == "a"
        broker.note_billed_cost("b", 0.04)
        assert broker.active_billing_for("b") == "a"
        broker.note_billed_cost("b", 0.07)  # crosses the cap
        assert broker.active_billing_for("b") is None
        assert rec.billed_usd == pytest.approx(0.11)

    async def test_single_window_per_recipient_guard(self):
        broker = AskBroker()
        rec1 = broker.create("a1", "b", "q", 30)
        rec2 = broker.create("a2", "b", "q", 30)
        assert broker.activate_billing(rec1.ask_id) is True
        # Second concurrent window for the same recipient is refused.
        assert broker.activate_billing(rec2.ask_id) is False
        assert broker.active_billing_for("b") == "a1"

    async def test_finish_closes_window_after_grace(self):
        broker = AskBroker()
        broker.BILLING_GRACE_SECONDS = 0.0
        rec = broker.create("a", "b", "q", 30)
        broker.activate_billing(rec.ask_id)
        broker.finish(rec.ask_id)
        # Window survives until the (zero) grace timer fires.
        await asyncio.sleep(0.05)
        assert broker.active_billing_for("b") is None
        assert broker.get(rec.ask_id) is None

    async def test_thread_store_optional_and_posted(self):
        # ``ensure_dm_thread`` returns the thread ROW (a dict) — the real
        # ThreadStore contract; the id lives under ``id``. ``post_message``
        # takes ``body`` keyword-only. (The old mock returned a bare string
        # and asserted ``body`` positional, which masked the M3 seam bug.)
        store = MagicMock()
        store.ensure_dm_thread.return_value = {"id": "th_1"}
        broker = AskBroker(thread_store=store)
        rec = broker.create("a", "b", "why?", 30, scope_id="alpha")
        store.ensure_dm_thread.assert_called_once_with("alpha", "a", "b")
        assert broker.resolve(rec.ask_id, "because", by="b")["ok"]
        assert store.post_message.call_count == 2
        q_call, a_call = store.post_message.call_args_list
        assert q_call.args[:2] == ("th_1", "a")
        assert q_call.kwargs["body"] == "why?"
        assert a_call.args[:2] == ("th_1", "b")
        assert a_call.kwargs["body"] == "because"
        assert a_call.kwargs["payload"] == {"ask_id": rec.ask_id}

    async def test_thread_store_error_is_nonfatal(self):
        store = MagicMock()
        store.ensure_dm_thread.side_effect = RuntimeError("threads down")
        broker = AskBroker(thread_store=store)
        rec = broker.create("a", "b", "q", 30)  # must not raise
        assert broker.resolve(rec.ask_id, "ans", by="b")["ok"]


# ── Lane primitives the verb rides on ────────────────────────────────


class TestLanePrimitives:
    async def test_try_steer_reports_injection_without_fallback(self):
        fake = _FakeLanes(busy=True)
        assert await fake.manager.try_steer("h", "msg", system_note=True) is True
        fake.busy = False
        assert await fake.manager.try_steer("h", "msg") is False
        # No followup fallback ever fired.
        assert fake.dispatch_messages == []

    async def test_try_steer_without_steer_fn(self):
        manager = LaneManager(dispatch_fn=AsyncMock(return_value="x"))
        assert await manager.try_steer("h", "msg") is False

    async def test_on_start_fires_at_dispatch_not_enqueue(self):
        order: list[str] = []

        async def dispatch_fn(agent, message, **kwargs):
            order.append("dispatch")
            return "ok"

        manager = LaneManager(dispatch_fn=dispatch_fn)
        fut = asyncio.ensure_future(
            manager.enqueue(
                "h", "msg", mode="followup",
                on_start=lambda: order.append("on_start"),
            ),
        )
        assert await fut == "ok"
        assert order == ["on_start", "dispatch"]

    async def test_on_start_error_does_not_wedge_lane(self):
        manager = LaneManager(dispatch_fn=AsyncMock(return_value="ok"))

        def boom():
            raise RuntimeError("hook boom")

        assert await manager.enqueue("h", "msg", on_start=boom) == "ok"


# ── Endpoint auth matrix (sync TestClient — all fail before delivery) ─


class TestAskEndpointAuthMatrix:
    def test_self_ask_400(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "asker", "question": "hi me"},
            )
            assert resp.status_code == 400
            assert "self_ask" in resp.text
        finally:
            bb.close()

    def test_unknown_recipient_404_with_roster_and_roles(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "ghost", "question": "anyone there?"},
            )
            assert resp.status_code == 404
            detail = resp.json()["detail"]
            assert detail["error"] == "unknown_recipient"
            # Roster = same-team teammates WITH roles; never the operator.
            assert detail["teammates"] == [{"id": "helper", "role": "analyst"}]
        finally:
            bb.close()

    def test_operator_roster_sees_all_workers(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_OP,
                json={"to": "ghost", "question": "?"},
            )
            assert resp.status_code == 404
            ids = {t["id"] for t in resp.json()["detail"]["teammates"]}
            assert ids == {"asker", "helper", "rival", "stranger"}
        finally:
            bb.close()

    def test_cross_team_403(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "rival", "question": "psst"},
            )
            assert resp.status_code == 403
            assert "Cross-team" in resp.text
        finally:
            bb.close()

    def test_no_can_message_grant_403(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_STRANGER,
                json={"to": "helper", "question": "hello"},
            )
            assert resp.status_code == 403
        finally:
            bb.close()

    def test_worker_cannot_ask_operator_403(self, tmp_path, monkeypatch):
        """Task-2e posture: no worker-injected synchronous prompt in the
        operator's privileged loop, even with a can_message grant."""
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "operator", "question": "approve this?"},
            )
            assert resp.status_code == 403
            assert "Hand off" in resp.text
        finally:
            bb.close()

    def test_empty_question_400(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "helper", "question": "   "},
            )
            assert resp.status_code == 400
        finally:
            bb.close()

    def test_missing_auth_401(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", json={"to": "helper", "question": "q"},
            )
            assert resp.status_code == 401
        finally:
            bb.close()

    def test_rate_limit_429(self, tmp_path, monkeypatch):
        broker = AskBroker()
        broker.MAX_ACTIVE_PER_ASKER = 100
        fake = _FakeLanes(busy=True)
        _fast_ask_timeout(monkeypatch)
        app, bb = _build_app(
            tmp_path, monkeypatch, broker=broker, lane_manager=fake.manager,
        )
        app.state.rate_limits["ask"] = (1, 60)
        try:
            client = TestClient(app)
            first = client.post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "helper", "question": "q1", "timeout_seconds": 1},
            )
            assert first.status_code == 200  # timeout envelope, but 200
            second = client.post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "helper", "question": "q2", "timeout_seconds": 1},
            )
            assert second.status_code == 429
        finally:
            bb.close()

    def test_concurrency_cap_429_envelope(self, tmp_path, monkeypatch):
        broker = AskBroker()
        broker.MAX_ACTIVE_PER_ASKER = 0
        app, bb = _build_app(tmp_path, monkeypatch, broker=broker)
        try:
            resp = TestClient(app).post(
                "/mesh/ask", headers=_ASKER,
                json={"to": "helper", "question": "q"},
            )
            assert resp.status_code == 429
            detail = resp.json()["detail"]
            assert detail["error"] == "ask_concurrency_limit"
            assert "recovery_hint" in detail
        finally:
            bb.close()


# ── Delivery paths (async, same-loop ASGI transport) ─────────────────


def _asgi(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://mesh",
    )


class TestBusyPath:
    async def test_steer_injection_content_and_timeout_envelope(
        self, tmp_path, monkeypatch,
    ):
        fake = _FakeLanes(busy=True)
        broker = AskBroker()
        _fast_ask_timeout(monkeypatch)
        app, bb = _build_app(
            tmp_path, monkeypatch, lane_manager=fake.manager, broker=broker,
        )
        try:
            async with _asgi(app) as client:
                resp = await client.post(
                    "/mesh/ask", headers=_ASKER,
                    json={"to": "helper", "question": "what is our SLA?"},
                    timeout=30,
                )
            body = resp.json()
            # Timeout envelope (Constraint #10 shape).
            assert resp.status_code == 200
            assert body["answered"] is False
            assert body["timeout"] is True
            assert "ask_timeout" in body["error"]
            assert "re-ask" in body["recovery_hint"]
            # Steer injection content: tag, role suffix, ask_id,
            # answer_ask instruction, and the question itself.
            assert len(fake.steer_messages) == 1
            msg = fake.steer_messages[0]
            ask_id = body["ask_id"]
            assert msg.startswith("[Teammate question from asker (researcher) — ask ")
            assert ask_id in msg
            assert f"answer_ask(ask_id='{ask_id}'" in msg
            assert "what is our SLA?" in msg
            # Busy path: NO followup dispatch (B1 — never a second
            # parallel turn) and NO billing window (marginal tokens ride
            # the recipient's current task).
            assert fake.dispatch_messages == []
            assert broker.active_billing_for("helper") is None
        finally:
            bb.close()

    async def test_answer_ask_resolves_busy_ask(self, tmp_path, monkeypatch):
        fake = _FakeLanes(busy=True)
        broker = AskBroker()
        monkeypatch.setitem(
            limits.LIMIT_SPECS, "ask_timeout_seconds", (5, 1, 10),
        )
        app, bb = _build_app(
            tmp_path, monkeypatch, lane_manager=fake.manager, broker=broker,
        )
        try:
            async with _asgi(app) as client:
                ask_task = asyncio.ensure_future(
                    client.post(
                        "/mesh/ask", headers=_ASKER,
                        json={"to": "helper", "question": "code word?"},
                        timeout=30,
                    ),
                )
                # Wait until delivery ran and the ask is registered.
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if fake.steer_messages:
                        break
                ask_id = next(iter(broker._asks))
                ans = await client.post(
                    f"/mesh/ask/{ask_id}/answer", headers=_HELPER,
                    json={"answer": "swordfish"},
                )
                assert ans.status_code == 200
                assert ans.json() == {"delivered": True, "ask_id": ask_id}
                resp = await ask_task
            body = resp.json()
            assert body["answered"] is True
            assert body["from"] == "helper"
            assert body["provenance"] == "teammate"
            assert body["answer"] == "swordfish"
        finally:
            bb.close()


class TestIdlePath:
    async def test_followup_dispatch_with_inline_fallback(
        self, tmp_path, monkeypatch,
    ):
        broker = AskBroker()
        billing_during_turn: list[str | None] = []

        def observe(agent, message):
            billing_during_turn.append(broker.active_billing_for("helper"))

        fake = _FakeLanes(
            busy=False, response="The SLA is 24h.", on_dispatch=observe,
        )
        monkeypatch.setitem(
            limits.LIMIT_SPECS, "ask_timeout_seconds", (5, 1, 10),
        )
        app, bb = _build_app(
            tmp_path, monkeypatch, lane_manager=fake.manager, broker=broker,
        )
        try:
            async with _asgi(app) as client:
                resp = await client.post(
                    "/mesh/ask", headers=_ASKER,
                    json={"to": "helper", "question": "what is our SLA?"},
                    timeout=30,
                )
            body = resp.json()
            # Uncooperative-but-answering turn: no answer_ask call, the
            # turn's own response resolves the ask.
            assert body["answered"] is True
            assert body["answer"] == "The SLA is 24h."
            assert body["provenance"] == "teammate"
            # Idle path went through the followup lane, not steer-only.
            assert len(fake.dispatch_messages) == 1
            assert "answer_ask" in fake.dispatch_messages[0]
            # Billing window was OPEN during the recipient's turn and is
            # keyed asker-pays (mesh-held mapping).
            assert billing_during_turn == ["asker"]
        finally:
            bb.close()

    async def test_lane_failure_yields_delivery_failed_envelope(
        self, tmp_path, monkeypatch,
    ):
        async def steer_fn(agent, message, system_note=False):
            return {"injected": False}

        async def dispatch_fn(agent, message, **kwargs):
            raise RuntimeError("container unreachable")

        manager = LaneManager(dispatch_fn=dispatch_fn, steer_fn=steer_fn)
        monkeypatch.setitem(
            limits.LIMIT_SPECS, "ask_timeout_seconds", (5, 1, 10),
        )
        app, bb = _build_app(tmp_path, monkeypatch, lane_manager=manager)
        try:
            async with _asgi(app) as client:
                resp = await client.post(
                    "/mesh/ask", headers=_ASKER,
                    json={"to": "helper", "question": "q"},
                    timeout=30,
                )
            body = resp.json()
            assert body["answered"] is False
            assert "ask_delivery_failed" in body["error"]
            assert "recovery_hint" in body
        finally:
            bb.close()


class TestAnswerEndpoint:
    async def test_wrong_recipient_403_unknown_404_reused_409(
        self, tmp_path, monkeypatch,
    ):
        fake = _FakeLanes(busy=True)
        broker = AskBroker()
        monkeypatch.setitem(
            limits.LIMIT_SPECS, "ask_timeout_seconds", (5, 1, 10),
        )
        app, bb = _build_app(
            tmp_path, monkeypatch, lane_manager=fake.manager, broker=broker,
        )
        try:
            async with _asgi(app) as client:
                ask_task = asyncio.ensure_future(
                    client.post(
                        "/mesh/ask", headers=_ASKER,
                        json={"to": "helper", "question": "q"},
                        timeout=30,
                    ),
                )
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if broker._asks:
                        break
                ask_id = next(iter(broker._asks))
                # Wrong recipient — a third agent cannot answer.
                wrong = await client.post(
                    f"/mesh/ask/{ask_id}/answer", headers=_STRANGER,
                    json={"answer": "hijack"},
                )
                assert wrong.status_code == 403
                # Unknown ask id.
                missing = await client.post(
                    "/mesh/ask/ask_nope/answer", headers=_HELPER,
                    json={"answer": "?"},
                )
                assert missing.status_code == 404
                assert missing.json()["detail"]["error"] == "unknown_ask"
                # Real answer, then a second one after the asker already
                # returned → the record is gone (single-use + cleanup):
                # unknown-ask 404, the documented non-fatal late path.
                ok = await client.post(
                    f"/mesh/ask/{ask_id}/answer", headers=_HELPER,
                    json={"answer": "first"},
                )
                assert ok.status_code == 200
                resp = await ask_task
                assert resp.json()["answer"] == "first"
                dup = await client.post(
                    f"/mesh/ask/{ask_id}/answer", headers=_HELPER,
                    json={"answer": "second"},
                )
                assert dup.status_code == 404
        finally:
            bb.close()

    async def test_double_answer_before_asker_returns_409(
        self, tmp_path, monkeypatch,
    ):
        """Single-use pinned at the HTTP layer: while the record still
        exists (asker not yet returned — no finish()), a second answer
        is a 409, not a silent overwrite."""
        broker = AskBroker()
        app, bb = _build_app(tmp_path, monkeypatch, broker=broker)
        try:
            rec = broker.create("asker", "helper", "q", 30)
            async with _asgi(app) as client:
                ok = await client.post(
                    f"/mesh/ask/{rec.ask_id}/answer", headers=_HELPER,
                    json={"answer": "first"},
                )
                assert ok.status_code == 200
                dup = await client.post(
                    f"/mesh/ask/{rec.ask_id}/answer", headers=_HELPER,
                    json={"answer": "second"},
                )
                assert dup.status_code == 409
                assert dup.json()["detail"]["error"] == "already_answered"
            assert rec.future.result() == "first"
        finally:
            bb.close()


# ── Billing seam (execute_api_call, mesh-authoritative) ──────────────


def _chat_request():
    from src.shared.types import APIProxyRequest

    return APIProxyRequest(
        service="llm",
        action="chat",
        params={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def _mock_acompletion_factory(total=30_000, prompt=20_000, completion=10_000):
    async def mock_acompletion(model, messages, api_key, **kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "ok"
        resp.choices[0].message.tool_calls = None
        resp.usage = MagicMock()
        resp.usage.total_tokens = total
        resp.usage.prompt_tokens = prompt
        resp.usage.completion_tokens = completion
        return resp

    return mock_acompletion


@pytest.fixture
def vault_tracker(tmp_path, monkeypatch):
    from src.host.credentials import CredentialVault

    monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
    monkeypatch.delenv("OPENLEGION_SYSTEM_ANTHROPIC_OAUTH", raising=False)
    monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
    tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
    vault = CredentialVault(cost_tracker=tracker)
    yield vault, tracker
    tracker.close()


def _usage_rows(tracker):
    return tracker.db.execute(
        "SELECT agent, cost_usd FROM usage ORDER BY id",
    ).fetchall()


class TestAskBilling:
    async def test_usage_rows_land_on_asker_during_window(self, vault_tracker):
        vault, tracker = vault_tracker
        broker = AskBroker()
        vault.set_bill_resolver(broker)
        rec = broker.create("asker", "helper", "q", 30)
        assert broker.activate_billing(rec.ask_id)
        with patch("litellm.acompletion", side_effect=_mock_acompletion_factory()):
            result = await vault.execute_api_call(_chat_request(), agent_id="helper")
        assert result.success is True
        rows = _usage_rows(tracker)
        assert len(rows) == 1
        assert rows[0][0] == "asker"  # the asker's ledger literally pays
        assert rows[0][1] > 0
        assert rec.billed_usd == pytest.approx(rows[0][1])

    async def test_asker_budget_preflight_blocks_recipient_turn(self, vault_tracker):
        vault, tracker = vault_tracker
        broker = AskBroker()
        vault.set_bill_resolver(broker)
        # Exhaust the ASKER's budget; the recipient's own budget is
        # untouched and generous.
        tracker.set_budget("asker", 0.0001, 0.0001)
        tracker.track("asker", "openai/gpt-4o", 100_000, 50_000)
        rec = broker.create("asker", "helper", "q", 30)
        broker.activate_billing(rec.ask_id)
        result = await vault.execute_api_call(_chat_request(), agent_id="helper")
        assert result.success is False
        assert "Budget exceeded" in result.error

    async def test_asker_team_envelope_preflight(self, vault_tracker, tmp_path):
        vault, tracker = vault_tracker
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "asker")
        store.set_budget("alpha", 0.001, None)
        tracker.set_team_store(store)
        tracker.track("asker", "openai/gpt-4o", 100_000, 50_000)
        broker = AskBroker()
        vault.set_bill_resolver(broker)
        rec = broker.create("asker", "helper", "q", 30)
        broker.activate_billing(rec.ask_id)
        # Recipient "helper" is teamless — the block can only come from
        # the ASKER's team envelope.
        result = await vault.execute_api_call(_chat_request(), agent_id="helper")
        assert result.success is False
        assert "Team budget exceeded" in result.error
        assert "alpha" in result.error

    async def test_cap_closes_window_then_recipient_pays(self, vault_tracker):
        vault, tracker = vault_tracker
        broker = AskBroker(bill_cap_usd=0.01)  # < one big call's cost
        vault.set_bill_resolver(broker)
        rec = broker.create("asker", "helper", "q", 30)
        broker.activate_billing(rec.ask_id)
        mock = _mock_acompletion_factory(
            total=2_000_000, prompt=1_500_000, completion=500_000,
        )
        with patch("litellm.acompletion", side_effect=mock):
            first = await vault.execute_api_call(_chat_request(), agent_id="helper")
            assert first.success is True
            # Cap crossed → window closed by the broker.
            assert broker.active_billing_for("helper") is None
            second = await vault.execute_api_call(_chat_request(), agent_id="helper")
            assert second.success is True
        rows = _usage_rows(tracker)
        assert [r[0] for r in rows] == ["asker", "helper"]

    async def test_recipient_billed_after_window_closes(self, vault_tracker):
        vault, tracker = vault_tracker
        broker = AskBroker()
        broker.BILLING_GRACE_SECONDS = 0.0
        vault.set_bill_resolver(broker)
        rec = broker.create("asker", "helper", "q", 30)
        broker.activate_billing(rec.ask_id)
        broker.finish(rec.ask_id)
        await asyncio.sleep(0.05)  # grace timer fires
        with patch("litellm.acompletion", side_effect=_mock_acompletion_factory()):
            result = await vault.execute_api_call(_chat_request(), agent_id="helper")
        assert result.success is True
        rows = _usage_rows(tracker)
        assert [r[0] for r in rows] == ["helper"]

    async def test_no_window_no_effect(self, vault_tracker):
        vault, tracker = vault_tracker
        broker = AskBroker()
        vault.set_bill_resolver(broker)
        with patch("litellm.acompletion", side_effect=_mock_acompletion_factory()):
            result = await vault.execute_api_call(_chat_request(), agent_id="helper")
        assert result.success is True
        assert [r[0] for r in _usage_rows(tracker)] == ["helper"]


# ── Agent-side tools (mocked mesh_client; never the loop) ────────────


class TestAskTeammateTool:
    async def test_answered_passthrough_with_provenance(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(return_value={
            "answered": True, "from": "helper", "answer": "42",
            "ask_id": "ask_1", "provenance": "teammate",
        })
        result = await ask_teammate(
            to="helper", question="meaning?", mesh_client=mc,
        )
        assert result["answered"] is True
        assert result["provenance"] == "teammate"
        assert result["answer"] == "42"
        assert "not execute" in result["note"]

    async def test_self_ask_envelope_without_mesh_call(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock()
        result = await ask_teammate(to="asker", question="hm", mesh_client=mc)
        assert result["answered"] is False
        assert "self_ask" in result["error"]
        mc.ask_teammate.assert_not_called()

    async def test_unknown_recipient_envelope_includes_roster(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(return_value={
            "http_error": True, "status_code": 404,
            "detail": {
                "error": "unknown_recipient",
                "teammates": [{"id": "helper", "role": "analyst"}],
            },
        })
        result = await ask_teammate(to="ghost", question="q", mesh_client=mc)
        assert result["answered"] is False
        assert "unknown_recipient" in result["error"]
        assert result["teammates"] == [{"id": "helper", "role": "analyst"}]
        assert "role" in result["recovery_hint"]

    async def test_rate_limit_envelope(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(return_value={
            "http_error": True, "status_code": 429,
            "detail": {"error": "ask_concurrency_limit"},
        })
        result = await ask_teammate(to="helper", question="q", mesh_client=mc)
        assert result["answered"] is False
        assert "rate_limited" in result["error"]
        assert "batch" in result["recovery_hint"]

    async def test_cross_team_403_envelope(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(return_value={
            "http_error": True, "status_code": 403,
            "detail": "Cross-team asks are not allowed",
        })
        result = await ask_teammate(to="rival", question="q", mesh_client=mc)
        assert result["answered"] is False
        assert "ask_forbidden" in result["error"]
        assert "hand_off" in result["recovery_hint"]

    async def test_timeout_envelope_passthrough(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(return_value={
            "answered": False, "ask_id": "ask_9", "timeout": True,
            "error": "ask_timeout: no answer from 'helper' within 180s.",
            "recovery_hint": "Do NOT immediately re-ask.",
        })
        result = await ask_teammate(to="helper", question="q", mesh_client=mc)
        assert result["answered"] is False
        assert result["timeout"] is True
        assert "ask_timeout" in result["error"]
        assert "re-ask" in result["recovery_hint"]

    async def test_transport_exception_envelope(self):
        from src.agent.builtins.coordination_tool import ask_teammate

        mc = MagicMock()
        mc.agent_id = "asker"
        mc.ask_teammate = AsyncMock(
            side_effect=RuntimeError("boom https://u:pw@mesh/x"),
        )
        result = await ask_teammate(to="helper", question="q", mesh_client=mc)
        assert result["answered"] is False
        assert "ask_failed" in result["error"]
        assert "pw" not in result["error"]  # redacted


class TestAnswerAskTool:
    async def test_delivered(self):
        from src.agent.builtins.coordination_tool import answer_ask

        mc = MagicMock()
        mc.answer_ask = AsyncMock(
            return_value={"delivered": True, "ask_id": "ask_1"},
        )
        result = await answer_ask(ask_id="ask_1", answer="yes", mesh_client=mc)
        assert result == {"answer_delivered": True, "ask_id": "ask_1"}

    async def test_expired_is_nonfatal_envelope(self):
        from src.agent.builtins.coordination_tool import answer_ask

        mc = MagicMock()
        mc.answer_ask = AsyncMock(return_value={
            "http_error": True, "status_code": 404,
            "detail": {"error": "unknown_ask", "hint": "asker timed out"},
        })
        result = await answer_ask(ask_id="ask_1", answer="late", mesh_client=mc)
        assert result["answer_delivered"] is False
        assert result["expired"] is True
        assert "recovery_hint" in result

    async def test_wrong_recipient_envelope(self):
        from src.agent.builtins.coordination_tool import answer_ask

        mc = MagicMock()
        mc.answer_ask = AsyncMock(return_value={
            "http_error": True, "status_code": 403,
            "detail": {"error": "wrong_recipient"},
        })
        result = await answer_ask(ask_id="ask_1", answer="x", mesh_client=mc)
        assert result["answer_delivered"] is False
        assert "wrong_recipient" in result["error"]

    async def test_invalid_ask_id_rejected_locally(self):
        from src.agent.builtins.coordination_tool import answer_ask

        mc = MagicMock()
        mc.answer_ask = AsyncMock()
        result = await answer_ask(
            ask_id="../evil path", answer="x", mesh_client=mc,
        )
        assert result["answer_delivered"] is False
        mc.answer_ask.assert_not_called()


# ── Config / limits pins ─────────────────────────────────────────────


class TestAskConfig:
    def test_rate_limit_bucket_registered(self, tmp_path, monkeypatch):
        app, bb = _build_app(tmp_path, monkeypatch)
        try:
            assert app.state.rate_limits["ask"] == (20, 60)
            assert "ask_answer" in app.state.rate_limits
        finally:
            bb.close()

    def test_limits_entries(self):
        assert limits.LIMIT_SPECS["ask_timeout_seconds"] == (180, 30, 600)
        assert limits.ENV_NAMES["ask_timeout_seconds"] == "OPENLEGION_ASK_TIMEOUT_SECONDS"
        assert limits.ASK_QUESTION_MAX_CHARS == 4_000
        assert limits.ASK_ANSWER_MAX_CHARS == 8_000

    def test_bill_cap_env_override_and_clamp(self, monkeypatch):
        monkeypatch.delenv("OPENLEGION_ASK_BILL_CAP_USD", raising=False)
        assert limits.ask_bill_cap_usd() == 0.50
        monkeypatch.setenv("OPENLEGION_ASK_BILL_CAP_USD", "2.5")
        assert limits.ask_bill_cap_usd() == 2.5
        monkeypatch.setenv("OPENLEGION_ASK_BILL_CAP_USD", "99999")
        assert limits.ask_bill_cap_usd() == 100.0
        monkeypatch.setenv("OPENLEGION_ASK_BILL_CAP_USD", "junk")
        assert limits.ask_bill_cap_usd() == 0.50

    def test_operator_surfaces_include_ask_tools(self):
        from src.agent.loop import _HEARTBEAT_TOOLS
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS

        assert "ask_teammate" in _OPERATOR_ALLOWED_TOOLS
        assert "answer_ask" in _OPERATOR_ALLOWED_TOOLS
        assert "ask_teammate" in _HEARTBEAT_TOOLS
