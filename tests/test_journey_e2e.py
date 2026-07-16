"""End-to-end operator↔team journey against the REAL mesh app.

Design ref: ``docs/plans/2026-07-16-autonomous-team-delivery.md`` (§1/§3 —
the operator→team goal-delivery loop). Unlike the per-endpoint unit tests
(``test_team_goal.py`` / ``test_team_budget.py`` / ``test_team_brief.py``),
this walks the WHOLE journey as ONE sequential flow against a single
in-process ``create_mesh_app`` instance, asserting the COMPOSED state at
each hop — so a regression in how create → membership → goal → budget →
context → lead-lifecycle compose at runtime is caught even when every
endpoint still passes in isolation.

The journey (each hop a real endpoint call, asserting composed state):

1. create a team.
2. add real (non-operator) members → a LEAD is auto-appointed to the
   first member once the roster crosses 2 real members (#1266/#1270).
3. set the goal → (a) DB north_star/success_criteria persist, (b) a
   reserved ``## Goal`` section is written into the team's TEAM.md, (c)
   the goal is PUSHED to running members (#1268 propagation).
4. ``GET /mesh/teams`` read-back → north_star/success_criteria/budget
   columns ride the row (#1268 inspect_teams).
5. set a team budget → it reads back via ``GET /mesh/teams``.
6. update team context → a ``## Context`` section is written AND the
   pre-existing ``## Goal`` section SURVIVES (#1273 composition seam).
7. remove the LEAD → a NEW lead is auto-appointed to a remaining member
   (#1274 lead-orphan lifecycle seam).

The membership/goal/context/budget endpoints need no containers or LLM —
a fake transport observes the fleet-wide ``/team`` pushes, mirroring the
``brief_app`` / ``join_app`` fixtures in ``test_team_brief.py``.
"""

from __future__ import annotations

import importlib
import json as _json

import pytest
import yaml as _yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix

# Operator identity: auth tokens are configured (production-shape), so the
# add/remove-member endpoints — which gate on ``_resolve_agent_id`` and only
# resolve a Bearer token to "operator" when tokens are set — accept the
# caller. The goal/context/budget/list endpoints accept the same Bearer.
_OP = {"Authorization": "Bearer op-token", "X-Agent-ID": "operator"}


class _FakeTransport:
    """Records ``/team`` pushes so the propagation hops can assert the goal /
    context reached running members. Mirrors the ``_FakeTransport`` in
    ``test_team_brief.py``."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(self, agent_id, method, path, json=None, timeout=120, headers=None):
        self.calls.append(
            {"agent_id": agent_id, "method": method, "path": path, "json": json}
        )
        return {"updated": True}


@pytest.fixture
def journey_app(tmp_path, monkeypatch):
    """A REAL mesh app wired like production for the operator↔team surfaces:
    disk-backed TeamStore (so the TEAM.md scaffold + section mirrors are
    observable), a fake transport (so fleet pushes are observable), three
    RUNNING members (in the router registry), and operator auth tokens (so
    the membership endpoints' operator gate passes). Everything is pinned
    inside ``tmp_path`` (chdir + explicit DB env) so the run is hermetic."""
    monkeypatch.chdir(tmp_path)
    # Pin every store the app opens by default into tmp (belt-and-suspenders
    # over the chdir) so nothing lands in a shared ``data/`` dir.
    monkeypatch.setenv("OPENLEGION_ORCHESTRATION_TASKS_DB", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(tmp_path / "teams.db"))
    monkeypatch.setenv("OPENLEGION_THREADS_DB", str(tmp_path / "threads.db"))
    monkeypatch.setenv("OPENLEGION_PENDING_ACTIONS_DB", str(tmp_path / "pending.db"))

    import src.cli.config as config_module
    import src.host.server as server_module
    from src.host.teams import TeamStore

    importlib.reload(server_module)

    members = ["scout", "analyst", "editor"]

    # Config the add-member "known agent" gate reads (agents.yaml is the
    # create-path source of truth; the router registry + ACL matrix are the
    # other two recognized-id sources).
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        _yaml.dump(
            {"agents": {**{m: {"role": "worker"} for m in members}, "operator": {"role": "operator"}}}
        )
    )
    monkeypatch.setattr(config_module, "AGENTS_FILE", agents_file)

    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(
        _json.dumps(
            {
                "permissions": {
                    m: {"blackboard_read": [], "blackboard_write": []} for m in members
                }
            }
        )
    )
    monkeypatch.setattr(config_module, "PERMISSIONS_FILE", perms_file)

    teams_dir = tmp_path / "teams"
    monkeypatch.setattr(config_module, "TEAMS_DIR", teams_dir)

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    # Point the matrix's reload() at the tmp file too, so the membership
    # endpoints' ``permissions.reload()`` stays hermetic.
    permissions = PermissionMatrix(config_path=str(perms_file))
    # All three members are RUNNING (present in the registry) so the goal /
    # context pushes are observable; the operator is registered like the
    # sibling fixtures.
    router = MessageRouter(
        permissions,
        {
            "operator": "http://operator:8400",
            "scout": "http://scout:8400",
            "analyst": "http://analyst:8400",
            "editor": "http://editor:8400",
        },
    )
    teams_store = TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=teams_dir)
    transport = _FakeTransport()
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=teams_store,
        transport=transport,  # type: ignore[arg-type]
        auth_tokens={"operator": "op-token"},
    )
    yield app, transport, teams_store, teams_dir
    blackboard.close()
    importlib.reload(server_module)


@pytest.mark.asyncio
async def test_operator_team_journey_composes_end_to_end(journey_app):
    """Walk the whole operator↔team journey in ONE sequential flow, asserting
    the composed state at each hop. Every hop is a real mesh endpoint call;
    no hop may 500, and every state transition must land as designed."""
    app, transport, teams_store, teams_dir = journey_app
    team_md = teams_dir / "growth" / "team.md"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # ── Hop 1: create the team ────────────────────────────────────
        r = await c.post(
            "/mesh/teams",
            json={"name": "growth", "description": "growth project"},
            headers=_OP,
        )
        assert r.status_code == 200, r.text
        assert teams_store.team_exists("growth")
        # A fresh empty team commits leaderless (a solo/empty team self-leads).
        assert teams_store.get_team("growth")["lead_agent_id"] is None
        # The scaffold TEAM.md exists on disk (the shared context file).
        assert team_md.exists()

        # ── Hop 2: add members → LEAD auto-appointed to the first (#1266/#1270) ──
        r1 = await c.post("/mesh/teams/growth/members", json={"agent": "scout"}, headers=_OP)
        assert r1.status_code == 200, r1.text
        # One real member: still leaderless (a solo team self-leads).
        assert teams_store.get_team("growth")["lead_agent_id"] is None

        r2 = await c.post("/mesh/teams/growth/members", json={"agent": "analyst"}, headers=_OP)
        assert r2.status_code == 200, r2.text
        # Crossing to 2 real members auto-appoints the FIRST-added as lead.
        assert teams_store.get_team("growth")["lead_agent_id"] == "scout"

        # A third member so the later lead-removal (hop 7) leaves >=2 real
        # members and actually exercises the #1274 re-appoint seam. Adding
        # over an existing lead must NEVER re-appoint.
        r3 = await c.post("/mesh/teams/growth/members", json={"agent": "editor"}, headers=_OP)
        assert r3.status_code == 200, r3.text
        assert teams_store.get_team("growth")["lead_agent_id"] == "scout"
        assert teams_store.members("growth") == ["scout", "analyst", "editor"]

        # ── Hop 3: set the goal → DB persist + ## Goal in TEAM.md + push (#1268) ──
        # Isolate the goal push from the join-context pushes above.
        transport.calls.clear()
        rg = await c.post(
            "/mesh/teams/growth/goal",
            json={
                "north_star": "Ship $10k MRR landing page",
                "success_criteria": ["100 visits/day", "5 demos/wk"],
            },
            headers=_OP,
        )
        assert rg.status_code == 200, rg.text
        assert rg.json()["success"] is True

        # (a) DB persisted (round-trips through the store the dashboard reads).
        team = teams_store.get_team("growth")
        assert team["north_star"] == "Ship $10k MRR landing page"
        assert team["success_criteria"] == ["100 visits/day", "5 demos/wk"]

        # (b) Reserved ``## Goal`` section written into the shared TEAM.md,
        #     scaffold preamble preserved (section-scoped, not full-overwrite).
        on_disk = team_md.read_text()
        assert "## Goal" in on_disk
        assert "Ship $10k MRR landing page" in on_disk
        assert "- 100 visits/day" in on_disk
        assert "- 5 demos/wk" in on_disk
        assert "growth project" in on_disk  # scaffold survives

        # (c) The goal was PUSHED to running members via ``/team``, carrying
        #     the goal content, only to real team members (#1268 propagation).
        team_pushes = [call for call in transport.calls if call["path"] == "/team"]
        assert team_pushes, "goal must push the updated TEAM.md to running members"
        assert {call["agent_id"] for call in team_pushes} <= set(teams_store.members("growth"))
        assert all(
            "Ship $10k MRR landing page" in call["json"]["content"] for call in team_pushes
        )
        # The pushed content is exactly the composed on-disk file.
        assert all(call["json"]["content"] == on_disk for call in team_pushes)

        # ── Hop 4: GET /mesh/teams read-back (#1268 inspect_teams) ────────
        listing = await c.get("/mesh/teams", headers=_OP)
        assert listing.status_code == 200, listing.text
        row = {t["name"]: t for t in listing.json()["teams"]}["growth"]
        assert row["north_star"] == "Ship $10k MRR landing page"
        assert row["success_criteria"] == ["100 visits/day", "5 demos/wk"]
        assert row["lead_agent_id"] == "scout"
        # Budget columns ride the row (still unset at this hop — keys present).
        assert "budget_daily_usd" in row and "budget_monthly_usd" in row
        assert row["budget_daily_usd"] is None
        assert row["budget_monthly_usd"] is None

        # ── Hop 5: set a team budget → reads back via GET /mesh/teams ──────
        rb = await c.put(
            "/mesh/teams/growth/budget",
            json={"daily_usd": 25.0, "monthly_usd": 400.0},
            headers=_OP,
        )
        assert rb.status_code == 200, rb.text
        assert rb.json()["unlimited"] is False
        listing2 = await c.get("/mesh/teams", headers=_OP)
        row2 = {t["name"]: t for t in listing2.json()["teams"]}["growth"]
        assert row2["budget_daily_usd"] == 25.0
        assert row2["budget_monthly_usd"] == 400.0
        # The goal still rides the same row alongside the new budget.
        assert row2["north_star"] == "Ship $10k MRR landing page"

        # ── Hop 6: update context → ## Context written AND ## Goal survives (#1273) ──
        transport.calls.clear()
        rc = await c.put(
            "/mesh/teams/growth/context",
            json={"context": "Focus on retention."},
            headers=_OP,
        )
        assert rc.status_code == 200, rc.text
        composed = team_md.read_text()
        # New Context section written into its own reserved block.
        assert "## Context" in composed and "Focus on retention." in composed
        assert composed.count("## Context") == 1
        # The #1273 composition seam: the pre-existing ``## Goal`` SURVIVED the
        # context write (the writer used to full-overwrite and wipe it).
        assert "## Goal" in composed and "Ship $10k MRR landing page" in composed
        assert "growth project" in composed  # scaffold still intact too
        # Context also pushed to running members, carrying the composed file.
        context_pushes = [call for call in transport.calls if call["path"] == "/team"]
        assert context_pushes
        assert all(call["json"]["content"] == composed for call in context_pushes)

        # ── Hop 7: remove the LEAD → NEW lead auto-appointed (#1274) ──────
        assert teams_store.get_team("growth")["lead_agent_id"] == "scout"
        rr = await c.delete("/mesh/teams/growth/members/scout", headers=_OP)
        assert rr.status_code == 200, rr.text
        after = teams_store.get_team("growth")
        assert "scout" not in after["members"]
        assert after["members"] == ["analyst", "editor"]
        # Losing the lead with >=2 real members remaining re-appoints the
        # first remaining member — the lead-orphan lifecycle seam.
        assert after["lead_agent_id"] == "analyst"
        assert after["lead_agent_id"] in after["members"]

        # ── Whole-journey invariant: the final composed row is coherent ──
        final = {t["name"]: t for t in (await c.get("/mesh/teams", headers=_OP)).json()["teams"]}[
            "growth"
        ]
        assert final["north_star"] == "Ship $10k MRR landing page"
        assert final["budget_daily_usd"] == 25.0
        assert final["lead_agent_id"] == "analyst"
        assert final["status"] == "active"
