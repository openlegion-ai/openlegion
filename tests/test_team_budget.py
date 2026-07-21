"""Team budget envelope (Phase 1 unit 2, plan B4).

Covers the mesh endpoint (`PUT /mesh/teams/{team}/budget`), the
`manage_team(action="set_budget")` operator tool routing, and the
end-to-end pre-flight block at the LLM proxy chokepoint with a real
CostTracker + TeamStore. THE B4 semantics — unset/0 envelope =
UNLIMITED — are pinned at the costs layer in test_costs.py; here we pin
the surfaces.
"""

import json
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

    def test_partial_update_preserves_untouched_period(self, tmp_path):
        """An OMITTED period keeps its current value — tightening the daily
        cap alone must NOT silently clear the monthly envelope. An EXPLICIT
        null/0 still clears that period to unlimited."""
        store = TeamStore(db_path=":memory:")
        store.create_team("alpha")
        store.set_budget("alpha", 25.0, 400.0)
        app, bb = _build_app(tmp_path, teams_store=store)
        try:
            client = TestClient(app)
            # Only daily_usd — monthly must round-trip unchanged.
            resp = client.put(
                "/mesh/teams/alpha/budget",
                headers=_OP,
                json={"daily_usd": 10.0},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json() == {
                "team": "alpha",
                "budget_daily_usd": 10.0,
                "budget_monthly_usd": 400.0,
                "unlimited": False,
            }
            team = store.get_team("alpha")
            assert team["budget_daily_usd"] == 10.0
            assert team["budget_monthly_usd"] == 400.0
            # An EXPLICIT null clears only that period.
            resp2 = client.put(
                "/mesh/teams/alpha/budget",
                headers=_OP,
                json={"monthly_usd": None},
            )
            assert resp2.status_code == 200, resp2.text
            team2 = store.get_team("alpha")
            assert team2["budget_daily_usd"] == 10.0  # preserved
            assert team2["budget_monthly_usd"] is None  # explicitly cleared
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
    async def test_fail_closed_envelope_renders_clear_message(self, tmp_path, monkeypatch):
        """When the envelope check fails closed (budget-store outage) the
        block is real but team/limits are null — the render must say the
        governor is unavailable, not the confusing 'team None … unlimited'
        (C-F7)."""
        vault, tracker, store = self._vault_and_tracker(tmp_path, monkeypatch)
        store.set_budget("alpha", 10.0, None)
        monkeypatch.setattr(tracker, "team_envelope_check", lambda *a, **k: {
            "allowed": False, "reason": "envelope_check_unavailable",
            "team": None, "daily_used": 0.0, "daily_limit": None,
            "monthly_used": 0.0, "monthly_limit": None, "estimated_cost": 0.0,
        })
        result = await vault.execute_api_call(self._chat_request(), agent_id="worker")
        assert result.success is False
        assert "governor temporarily unavailable" in result.error
        assert "'None'" not in result.error  # not the confusing team 'None' render
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


# ── Lead budget allocation within the human envelope (plan §8 #21) ──────


def _build_lead_app(tmp_path, *, teams_store, cost_tracker=None, extra_tokens=None):
    """Like ``_build_app`` above but with a caller-identity token per
    fixture agent (lead/teammate/other-team-lead), not just operator/worker."""
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    tokens = {"operator": "tok-op"}
    tokens.update(extra_tokens or {})
    router = MessageRouter(permissions=perms, agent_registry={"operator": "http://op:8400"})
    app = create_mesh_app(
        bb, PubSub(), router, perms,
        cost_tracker=cost_tracker, teams_store=teams_store,
        auth_tokens=tokens,
    )
    return app, bb


def _hdr(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


_LEAD_TOKENS = {
    "lead1": "tok-lead1",
    "member2": "tok-member2",
    "member3": "tok-member3",
    "lead2": "tok-lead2",
    "m1": "tok-m1",
}


def _setup_lead_teams(tmp_path):
    """alpha: lead1 (lead) + member2 + member3. beta: lead2 (lead).
    gamma: m1, no lead assigned (leaderless)."""
    store = TeamStore(db_path=":memory:")
    store.create_team("alpha")
    store.add_member("alpha", "lead1")
    store.add_member("alpha", "member2")
    store.add_member("alpha", "member3")
    store.set_lead("alpha", "lead1")
    store.create_team("beta")
    store.add_member("beta", "lead2")
    store.set_lead("beta", "lead2")
    store.create_team("gamma")
    store.add_member("gamma", "m1")
    tracker = CostTracker(db_path=str(tmp_path / "costs.db"))
    tracker.set_team_store(store)
    app, bb = _build_lead_app(
        tmp_path, teams_store=store, cost_tracker=tracker, extra_tokens=_LEAD_TOKENS,
    )
    return app, bb, store, tracker


class TestAllocateMemberBudgetGates:
    """404 unknown team -> 409 leaderless -> 403 non-lead -> 404 target
    not a member -- exactly the U5 recommend/verdict gate taxonomy."""

    def test_unknown_team_404(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/ghost/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 404
        finally:
            bb.close()
            tracker.close()

    def test_leaderless_team_409(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/gamma/members/m1/budget",
                headers=_hdr(_LEAD_TOKENS["m1"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 409
            assert "no lead" in resp.json()["detail"].lower()
        finally:
            bb.close()
            tracker.close()

    def test_non_lead_teammate_403(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member3/budget",
                headers=_hdr(_LEAD_TOKENS["member2"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 403
        finally:
            bb.close()
            tracker.close()

    def test_non_lead_operator_403(self, tmp_path):
        """The operator's own budget surfaces are untouched -- but acting
        AS the operator through this new lead-gated endpoint is denied
        exactly like any other non-lead caller."""
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr("tok-op"),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 403
        finally:
            bb.close()
            tracker.close()

    def test_lead_of_another_team_403(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead2"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 403
        finally:
            bb.close()
            tracker.close()

    def test_target_not_a_member_404(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            client = TestClient(app)
            # Unknown agent id entirely.
            resp = client.post(
                "/mesh/teams/alpha/members/ghost_agent/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 404
            # A real agent, but on a different team.
            resp = client.post(
                "/mesh/teams/alpha/members/lead2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 404
        finally:
            bb.close()
            tracker.close()


class TestAllocateMemberBudgetEnvelope:
    def test_no_envelope_at_all_409(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 5.0},
            )
            assert resp.status_code == 409
            assert "envelope" in resp.json()["detail"].lower()
        finally:
            bb.close()
            tracker.close()

    def test_daily_only_envelope_monthly_request_409(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, None)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"monthly_usd": 50.0},
            )
            assert resp.status_code == 409
            assert "monthly" in resp.json()["detail"].lower()
        finally:
            bb.close()
            tracker.close()

    def test_sigma_violation_409_names_headroom(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, None)
            tracker.set_budget("member3", 80.0, 80.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 30.0},
            )
            assert resp.status_code == 409, resp.text
            detail = resp.json()["detail"]
            assert "20.00" in detail  # headroom = 100 - 80
        finally:
            bb.close()
            tracker.close()

    def test_first_partial_allocation_rejects_default_materialization_blowout(self, tmp_path):
        """Phase-5 review finding: a first-ever DAILY-only allocation to a
        member with no prior explicit budget must NOT let ``set_budget``
        silently materialize the global default monthly cap ($200) against a
        tight monthly envelope — that blows Σ ≤ envelope and locks other
        members out. The endpoint rejects it, directing the lead to name the
        monthly period explicitly; nothing is written."""
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            # Generous daily, tight $1 monthly envelope (below any plausible
            # deployment default, so the default-materialization always breaches).
            store.set_budget("alpha", 100.0, 1.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 10.0},
            )
            assert resp.status_code == 409, resp.text
            assert "monthly" in resp.json()["detail"].lower()
            # Nothing persisted — no silent blowout.
            assert tracker.budgets.get("member2") is None
            # Naming both periods within headroom succeeds and writes exactly that.
            resp2 = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 10.0, "monthly_usd": 0.5},
            )
            assert resp2.status_code == 200, resp2.text
            assert tracker.budgets["member2"] == {"daily_usd": 10.0, "monthly_usd": 0.5}
        finally:
            bb.close()
            tracker.close()

    def test_sigma_boundary_exactly_equal_passes(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, None)
            tracker.set_budget("member3", 80.0, 80.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 20.0},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["allocation"]["daily_usd"] == 20.0
        finally:
            bb.close()
            tracker.close()

    def test_allocation_never_mutates_teams_row(self, tmp_path):
        """The surface can NEVER raise the envelope -- pin that the teams
        row is byte-unchanged by any allocation call."""
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            before = store.get_team("alpha")
            client = TestClient(app)
            resp = client.post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 30.0, "monthly_usd": 200.0},
            )
            assert resp.status_code == 200, resp.text
            resp2 = client.post(
                "/mesh/teams/alpha/members/member3/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 0.0},
            )
            assert resp2.status_code == 200, resp2.text
            after = store.get_team("alpha")
            assert before == after
        finally:
            bb.close()
            tracker.close()


class TestAllocateMemberBudgetWrites:
    def test_partial_update_preserves_other_period(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            client = TestClient(app)
            r1 = client.post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 10.0, "monthly_usd": 50.0},
            )
            assert r1.status_code == 200, r1.text
            r2 = client.post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 20.0},
            )
            assert r2.status_code == 200, r2.text
            assert r2.json()["allocation"]["daily_usd"] == 20.0
            assert r2.json()["allocation"]["monthly_usd"] == 50.0
            assert tracker.budgets["member2"]["monthly_usd"] == 50.0
        finally:
            bb.close()
            tracker.close()

    def test_zero_blocks_while_envelope_has_headroom(self, tmp_path):
        """Preflight integration: a lead-set 0 allocation blocks that
        member's own work-LLM spend even though the team envelope still
        has plenty of headroom (B4 per-agent semantics untouched)."""
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 0.0, "monthly_usd": 0.0},
            )
            assert resp.status_code == 200, resp.text
            check = tracker.check_budget("member2")
            assert check["allowed"] is False
            preflight = tracker.preflight_check("member2", "openai/gpt-4o-mini")
            assert preflight["allowed"] is False
        finally:
            bb.close()
            tracker.close()

    def test_audit_row_records_old_and_new(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            client = TestClient(app)
            client.post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 10.0},
            )
            client.post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 25.0},
            )
            log = bb.get_audit_log(action="lead_budget_allocation", agent_id="member2")
            assert log["total"] == 2
            latest = log["entries"][0]
            assert latest["actor"] == "lead1"
            before = json.loads(latest["before_value"])
            after = json.loads(latest["after_value"])
            assert before["daily_usd"] == 10.0
            assert after["daily_usd"] == 25.0
        finally:
            bb.close()
            tracker.close()

    def test_response_headroom_math(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, None)
            tracker.set_budget("member3", 20.0, 20.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": 30.0},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["envelope"]["daily_usd"] == 100.0
            assert body["envelope"]["monthly_usd"] is None
            # headroom = 100 - (20 [member3] + 30 [member2]) = 50
            assert body["headroom"]["daily_usd"] == 50.0
            assert body["headroom"]["monthly_usd"] is None
        finally:
            bb.close()
            tracker.close()


class TestAllocateMemberBudgetValidation:
    def test_missing_both_fields_400(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={},
            )
            assert resp.status_code == 400
        finally:
            bb.close()
            tracker.close()

    def test_negative_value_400(self, tmp_path):
        app, bb, store, tracker = _setup_lead_teams(tmp_path)
        try:
            store.set_budget("alpha", 100.0, 500.0)
            resp = TestClient(app).post(
                "/mesh/teams/alpha/members/member2/budget",
                headers=_hdr(_LEAD_TOKENS["lead1"]),
                json={"daily_usd": -1.0},
            )
            assert resp.status_code == 400
        finally:
            bb.close()
            tracker.close()
