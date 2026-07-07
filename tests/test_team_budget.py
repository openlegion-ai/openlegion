"""Team budget envelope (Phase 1 unit 2, plan B4).

Covers the mesh endpoint (`PUT /mesh/teams/{team}/budget`), the
`manage_team(action="set_budget")` operator tool routing, and the
end-to-end pre-flight block at the LLM proxy chokepoint with a real
CostTracker + TeamStore. THE B4 semantics — unset/0 envelope =
UNLIMITED — are pinned at the costs layer in test_costs.py; here we pin
the surfaces.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.host.costs import CostTracker
from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.server import create_mesh_app
from src.host.teams import TeamStore


def _build_app(tmp_path, *, cost_tracker=None, teams_store=None):
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    router = MessageRouter(permissions=perms, agent_registry={"operator": "http://op:8400"})
    app = create_mesh_app(
        bb,
        PubSub(),
        router,
        perms,
        cost_tracker=cost_tracker,
        teams_store=teams_store,
        auth_tokens={"operator": "tok-op", "worker": "tok-w"},
    )
    return app, bb


_OP = {"Authorization": "Bearer tok-op"}
_WORKER = {"Authorization": "Bearer tok-w"}


class TestBudgetEndpoint:
    def test_operator_sets_budget(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        app, bb = _build_app(tmp_path, teams_store=store)
        try:
            resp = TestClient(app).put(
                "/mesh/teams/alpha/budget",
                headers=_OP,
                json={"daily_usd": 25.0, "monthly_usd": 400.0},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {
                "team": "alpha",
                "budget_daily_usd": 25.0,
                "budget_monthly_usd": 400.0,
                "unlimited": False,
            }
            team = store.get_team("alpha")
            assert team["budget_daily_usd"] == 25.0
            assert team["budget_monthly_usd"] == 400.0
        finally:
            bb.close()

    def test_zero_normalizes_to_unlimited(self, tmp_path):
        """B4 at the API surface: 0 is stored as NULL so 'unlimited' has
        exactly one stored shape."""
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.set_budget("alpha", 25.0, 400.0)
        app, bb = _build_app(tmp_path, teams_store=store)
        try:
            resp = TestClient(app).put(
                "/mesh/teams/alpha/budget",
                headers=_OP,
                json={"daily_usd": 0, "monthly_usd": None},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["unlimited"] is True
            team = store.get_team("alpha")
            assert team["budget_daily_usd"] is None
            assert team["budget_monthly_usd"] is None
        finally:
            bb.close()

    def test_worker_403(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        app, bb = _build_app(tmp_path, teams_store=store)
        try:
            resp = TestClient(app).put(
                "/mesh/teams/alpha/budget",
                headers=_WORKER,
                json={"daily_usd": 1},
            )
            assert resp.status_code == 403
        finally:
            bb.close()

    def test_unknown_team_404(self, tmp_path):
        app, bb = _build_app(tmp_path, teams_store=TeamStore(db_path=":memory:"))
        try:
            resp = TestClient(app).put(
                "/mesh/teams/ghost/budget",
                headers=_OP,
                json={"daily_usd": 1},
            )
            assert resp.status_code == 404
        finally:
            bb.close()

    def test_validation_400s(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        app, bb = _build_app(tmp_path, teams_store=store)
        client = TestClient(app)
        try:
            for body in (
                {"daily_usd": -1},
                {"daily_usd": "ten"},
                {"daily_usd": True},
                {"daily_usd": 999_999},
                {"monthly_usd": 999_999_999},
            ):
                resp = client.put("/mesh/teams/alpha/budget", headers=_OP, json=body)
                assert resp.status_code == 400, (body, resp.text)
            # NaN/Infinity pass </> comparisons and would silently store
            # as NULL (= unlimited) then 500 on response render. Python's
            # json.loads accepts these non-standard literals, so they are
            # reachable as raw bodies even though json.dumps refuses them.
            for raw in ('{"daily_usd": NaN}', '{"daily_usd": Infinity}'):
                resp = client.put(
                    "/mesh/teams/alpha/budget",
                    headers={**_OP, "Content-Type": "application/json"},
                    content=raw,
                )
                assert resp.status_code == 400, (raw, resp.text)
        finally:
            bb.close()


class TestCostsTeamEndpoint:
    def test_team_costs_reflect_envelope(self, tmp_path):
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        tracker.set_team_store(store)
        app, bb = _build_app(tmp_path, cost_tracker=tracker, teams_store=store)
        try:
            client = TestClient(app)
            client.put(
                "/mesh/teams/alpha/budget",
                headers=_OP,
                json={"daily_usd": 30.0},
            )
            resp = client.get("/mesh/costs/team/alpha", headers=_OP)
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["daily_limit"] == 30.0
            assert data["monthly_limit"] is None
            assert data["agents"] == [{"agent": "worker", "cost": 0, "tokens": 0}]
        finally:
            bb.close()
            tracker.close()


class TestManageTeamTool:
    @pytest.mark.asyncio
    async def test_set_budget_routes_to_client(self):
        from src.agent.builtins import operator_tools

        mc = MagicMock()
        mc.set_team_budget = AsyncMock(
            return_value={"team": "alpha", "unlimited": False},
        )
        with patch.object(operator_tools, "_is_operator", return_value=True):
            result = await operator_tools.manage_team(
                action="set_budget",
                team_name="alpha",
                daily_usd=10.0,
                monthly_usd=100.0,
                mesh_client=mc,
            )
        assert result == {"team": "alpha", "unlimited": False}
        mc.set_team_budget.assert_awaited_once_with("alpha", 10.0, 100.0)

    @pytest.mark.asyncio
    async def test_set_budget_rejects_non_numeric(self):
        from src.agent.builtins import operator_tools

        with patch.object(operator_tools, "_is_operator", return_value=True):
            result = await operator_tools.manage_team(
                action="set_budget",
                team_name="alpha",
                daily_usd="lots",
                mesh_client=MagicMock(),
            )
        assert "error" in result


class TestProxyEnvelopeBlock:
    """End-to-end: the envelope blocks at the LLM proxy chokepoint with a
    real CostTracker + TeamStore, and unset/0 does NOT block (B4)."""

    def _vault_and_tracker(self, tmp_path, monkeypatch):
        from src.host.credentials import CredentialVault

        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        monkeypatch.setenv("OPENLEGION_SYSTEM_OPENAI_API_KEY", "sk-test")
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        tracker.set_team_store(store)
        return CredentialVault(cost_tracker=tracker), tracker, store

    def _chat_request(self):
        from src.shared.types import APIProxyRequest

        return APIProxyRequest(
            service="llm",
            action="chat",
            params={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    @pytest.mark.asyncio
    async def test_exhausted_envelope_blocks_member_call(self, tmp_path, monkeypatch):
        vault, tracker, store = self._vault_and_tracker(tmp_path, monkeypatch)
        store.set_budget("alpha", 0.001, None)
        # Another member burns the whole envelope.
        tracker.track("teammate-in-db", "openai/gpt-4o", 1, 1)
        store.add_member("alpha", "teammate-in-db")
        tracker.track("teammate-in-db", "openai/gpt-4o", 100_000, 50_000)

        result = await vault.execute_api_call(self._chat_request(), agent_id="worker")
        assert result.success is False
        assert "Team budget exceeded" in result.error
        assert "alpha" in result.error
        tracker.close()

    @pytest.mark.asyncio
    async def test_zero_envelope_does_not_block(self, tmp_path, monkeypatch):
        """B4 pinned at the proxy: 0 envelope = unlimited; the call reaches
        the provider handler."""
        vault, tracker, store = self._vault_and_tracker(tmp_path, monkeypatch)
        store.set_budget("alpha", 0.0, 0.0)

        async def mock_acompletion(model, messages, api_key, **kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "ok"
            resp.choices[0].message.tool_calls = None
            resp.usage = MagicMock()
            resp.usage.total_tokens = 10
            resp.usage.prompt_tokens = 6
            resp.usage.completion_tokens = 4
            return resp

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await vault.execute_api_call(self._chat_request(), agent_id="worker")
        assert result.success is True
        tracker.close()


class TestImageGenEnvelope:
    @pytest.mark.asyncio
    async def test_exhausted_envelope_blocks_image_gen(self, tmp_path, monkeypatch):
        """image_gen spend lands in the same usage ledger the envelope
        sums, so an exhausted team can't keep spending through it."""
        from src.host.credentials import CredentialVault
        from src.shared.types import APIProxyRequest

        monkeypatch.delenv("OPENLEGION_SYSTEM_OPENAI_OAUTH", raising=False)
        tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.add_member("alpha", "worker")
        store.add_member("alpha", "burner")
        store.set_budget("alpha", 0.001, None)
        tracker.set_team_store(store)
        tracker.track("burner", "openai/gpt-4o", 100_000, 50_000)
        vault = CredentialVault(cost_tracker=tracker)

        req = APIProxyRequest(
            service="image_gen",
            action="generate",
            params={"prompt": "a cat"},
        )
        result = await vault.execute_api_call(req, agent_id="worker")
        assert result.success is False
        assert "Team budget exceeded" in result.error
        tracker.close()
