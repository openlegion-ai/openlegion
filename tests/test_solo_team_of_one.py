"""Solo agent = team-of-one isolation pins (ratified decision #5).

The solo→team-of-one unification is a CODE-PATH merge only: a previously
isolated solo worker gains a PRIVATE ``teams/{agent_id}/`` namespace that
only it (and the operator trust tier) can touch — never shared
reachability. These HTTP-level tests pin that security invariant:

(a) solo worker A can read/write ``teams/{A}/...`` but is 403'd on
    ``teams/{B}/...`` and on a real team's ``teams/{T}/...``;
(b) a team member is 403'd on ``teams/{soloA}/...``;
(c) the pubsub prefix gate locks a solo publisher/subscriber to its own
    ``teams/{A}/`` prefix (operator stays exempt);
(d) the cross-namespace collision guard 400s in both directions
    (team named after an agent; agent named after a team);
(e) join-then-leave leaves exactly the self pattern (no lockout, no
    residue of the old team);
(f) registration auto-watch/auto-subscribe scope to the solo worker's
    own namespace.
"""

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from httpx import ASGITransport, AsyncClient

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.host.teams import TeamStore

TOKENS = {
    "operator": "op-token",
    "solo-a": "a-token",
    "solo-b": "b-token",
    "member1": "m-token",
}


def _headers(agent: str) -> dict:
    return {"Authorization": f"Bearer {TOKENS[agent]}", "X-Agent-ID": agent}


@pytest.fixture
def solo_app(tmp_path, monkeypatch):
    """Mesh app with two solo workers (self-pattern ACLs — the posture every
    create/leave/boot path produces since ratified #5), one real team with
    a member, and bearer auth so identities are verified."""
    monkeypatch.chdir(tmp_path)
    perms_file = tmp_path / "permissions.json"
    perms_file.write_text(
        json.dumps(
            {
                "permissions": {
                    "solo-a": {
                        "blackboard_read": ["teams/solo-a/*"],
                        "blackboard_write": ["teams/solo-a/*"],
                        "can_publish": ["*"],
                        "can_subscribe": ["*"],
                    },
                    "solo-b": {
                        "blackboard_read": ["teams/solo-b/*"],
                        "blackboard_write": ["teams/solo-b/*"],
                        "can_publish": ["*"],
                        "can_subscribe": ["*"],
                    },
                    "member1": {
                        "blackboard_read": ["teams/team-x/*", "teams/member1/*"],
                        "blackboard_write": ["teams/team-x/*", "teams/member1/*"],
                        "can_publish": ["*"],
                        "can_subscribe": ["*"],
                    },
                    "operator": {
                        "blackboard_read": ["*"],
                        "blackboard_write": ["*"],
                        "can_publish": ["*"],
                        "can_subscribe": ["*"],
                    },
                },
            }
        )
    )
    import src.cli.config as cli_cfg

    monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
    agents_file = tmp_path / "agents.yaml"
    agents_file.write_text(
        yaml.dump(
            {
                "agents": {
                    "solo-a": {"role": "a"},
                    "solo-b": {"role": "b"},
                    "member1": {"role": "m"},
                    "operator": {"role": "operator"},
                },
            }
        )
    )
    monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)

    import src.host.server as server_module

    importlib.reload(server_module)

    permissions = PermissionMatrix(config_path=str(perms_file))
    router = MessageRouter(permissions, {"operator": "http://op:8400"})
    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    teams_store = TeamStore(
        db_path=str(tmp_path / "teams.db"),
        teams_dir=tmp_path / "teams",
    )
    teams_store.create_team("team-x")
    teams_store.add_member("team-x", "member1")
    app = server_module.create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=permissions,
        teams_store=teams_store,
        container_manager=MagicMock(),
        auth_tokens=dict(TOKENS),
    )
    yield app, teams_store, blackboard, pubsub, perms_file
    blackboard.close()
    importlib.reload(server_module)


async def _put(app, agent, key, value=None):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.put(
            f"/mesh/blackboard/{key}",
            params={"agent_id": agent},
            json=value or {"v": 1},
            headers=_headers(agent),
        )


async def _get(app, agent, key):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.get(
            f"/mesh/blackboard/{key}",
            params={"agent_id": agent},
            headers=_headers(agent),
        )


async def _publish(app, agent, topic):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(
            "/mesh/publish",
            json={"topic": topic, "source": agent, "payload": {}},
            headers=_headers(agent),
        )


async def _subscribe(app, agent, topic):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        return await c.post(
            "/mesh/subscribe",
            params={"topic": topic, "agent_id": agent},
            headers=_headers(agent),
        )


class TestSoloBlackboardIsolation:
    @pytest.mark.asyncio
    async def test_solo_can_write_and_read_own_namespace(self, solo_app):
        app, *_ = solo_app
        r = await _put(app, "solo-a", "teams/solo-a/notes/x", {"v": 42})
        assert r.status_code == 200, r.text
        r = await _get(app, "solo-a", "teams/solo-a/notes/x")
        assert r.status_code == 200
        assert r.json()["value"] == {"v": 42}

    @pytest.mark.asyncio
    async def test_solo_cannot_touch_other_solo_namespace(self, solo_app):
        app, *_ = solo_app
        r = await _put(app, "solo-a", "teams/solo-b/notes/x")
        assert r.status_code == 403
        r = await _get(app, "solo-a", "teams/solo-b/notes/x")
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_solo_cannot_touch_real_team_namespace(self, solo_app):
        app, *_ = solo_app
        r = await _put(app, "solo-a", "teams/team-x/context/plan")
        assert r.status_code == 403
        r = await _get(app, "solo-a", "teams/team-x/context/plan")
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_team_member_cannot_touch_solo_namespace(self, solo_app):
        app, *_ = solo_app
        r = await _put(app, "member1", "teams/solo-a/notes/x")
        assert r.status_code == 403
        r = await _get(app, "member1", "teams/solo-a/notes/x")
        assert r.status_code == 403

    @pytest.mark.asyncio
    async def test_operator_can_touch_solo_namespace(self, solo_app):
        """The operator trust tier is the ONE identity that may reach into
        a solo worker's private namespace (Constraint #12)."""
        app, *_ = solo_app
        r = await _put(app, "operator", "teams/solo-a/notes/from-op")
        assert r.status_code == 200
        r = await _get(app, "operator", "teams/solo-a/notes/from-op")
        assert r.status_code == 200


class TestSoloPubSubPrefixGate:
    @pytest.mark.asyncio
    async def test_solo_publish_inside_own_prefix_allowed(self, solo_app):
        app, *_ = solo_app
        r = await _publish(app, "solo-a", "teams/solo-a/done")
        assert r.status_code == 200, r.text

    @pytest.mark.asyncio
    async def test_solo_publish_outside_own_prefix_403(self, solo_app):
        """A solo worker is prefix-locked to teams/{its-own-id}/ — it no
        longer skips the team-prefix gate (pre-merge teamless behavior)."""
        app, *_ = solo_app
        for topic in ("global_announcements", "teams/solo-b/done", "teams/team-x/done"):
            r = await _publish(app, "solo-a", topic)
            assert r.status_code == 403, topic

    @pytest.mark.asyncio
    async def test_solo_subscribe_outside_own_prefix_403(self, solo_app):
        app, *_ = solo_app
        r = await _subscribe(app, "solo-a", "teams/solo-b/done")
        assert r.status_code == 403
        r = await _subscribe(app, "solo-a", "teams/solo-a/done")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_operator_publish_unscoped_still_allowed(self, solo_app):
        """The operator stays exempt from the prefix gate (trust tier)."""
        app, *_ = solo_app
        r = await _publish(app, "operator", "fleet_announcements")
        assert r.status_code == 200


class TestCollisionGuard:
    @pytest.mark.asyncio
    async def test_team_named_after_agent_400(self, solo_app):
        app, *_ = solo_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/teams",
                json={"name": "solo-a", "description": "", "members": []},
                headers=_headers("operator"),
            )
        assert r.status_code == 400
        assert "conflicts with an existing agent" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_agent_named_after_team_400(self, solo_app):
        app, *_ = solo_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/agents/create",
                json={"name": "team-x", "role": "helper"},
                headers=_headers("operator"),
            )
        assert r.status_code == 400
        assert "conflicts with an existing team" in r.json()["detail"]

    def test_cli_create_agent_rejects_team_collision(self, tmp_path, monkeypatch):
        from src.cli.config import _reject_agent_team_collision

        monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(tmp_path / "teams.db"))
        store = TeamStore(db_path=str(tmp_path / "teams.db"))
        store.create_team("growth")
        with pytest.raises(ValueError, match="conflicts with an existing team"):
            _reject_agent_team_collision("growth")
        # Non-colliding names pass through.
        _reject_agent_team_collision("fresh-name")


class TestJoinLeaveSelfPattern:
    def test_join_then_leave_lands_on_exactly_self_pattern(self, tmp_path, monkeypatch):
        """A leaver ends with exactly its private team-of-one pattern —
        not empty (the pre-merge lockout) and not the old team's."""
        from src.cli.config import (
            _add_team_blackboard_permissions,
            _remove_team_blackboard_permissions,
        )

        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "rover": {
                            "blackboard_read": ["teams/rover/*"],
                            "blackboard_write": ["teams/rover/*"],
                        },
                    },
                }
            )
        )
        with patch("src.cli.config.PERMISSIONS_FILE", perms_file):
            _add_team_blackboard_permissions("rover", "alpha")
            mid = json.loads(perms_file.read_text())["permissions"]["rover"]
            assert sorted(mid["blackboard_read"]) == ["teams/alpha/*", "teams/rover/*"]
            _remove_team_blackboard_permissions("rover", "alpha")

        after = json.loads(perms_file.read_text())["permissions"]["rover"]
        assert after["blackboard_read"] == ["teams/rover/*"]
        assert after["blackboard_write"] == ["teams/rover/*"]

    def test_boot_default_grants_self_pattern_to_empty_acl_solo(self, tmp_path, monkeypatch):
        """_ensure_all_agent_permissions forward-migrates a teamless worker
        with EMPTY blackboard ACLs (pre-merge standalone posture) to its
        self pattern — and leaves agents with ANY pattern untouched."""
        import src.cli.config as cli_cfg

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(tmp_path / "teams.db"))
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "lonely": {"blackboard_read": [], "blackboard_write": []},
                        "scoped": {
                            "blackboard_read": ["research/*"],
                            "blackboard_write": [],
                        },
                        "operator": {"blackboard_read": ["*"], "blackboard_write": ["*"]},
                    },
                }
            )
        )
        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(
            yaml.dump(
                {
                    "agents": {
                        "lonely": {"role": "x"},
                        "scoped": {"role": "y"},
                        "operator": {"role": "op"},
                    }
                }
            )
        )
        monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
        monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)
        monkeypatch.setattr(cli_cfg, "CONFIG_FILE", tmp_path / "mesh.yaml")

        cli_cfg._ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())["permissions"]
        assert perms["lonely"]["blackboard_read"] == ["teams/lonely/*"]
        assert perms["lonely"]["blackboard_write"] == ["teams/lonely/*"]
        # "Self always" is ADDITIVE: a teamless worker with custom patterns
        # keeps them and gains its self pattern (review F3 — legacy teamless
        # template agents must not end up all-403 with blackboard prompts).
        assert sorted(perms["scoped"]["blackboard_read"]) == ["research/*", "teams/scoped/*"]
        assert perms["scoped"]["blackboard_write"] == ["teams/scoped/*"]
        # The operator is never rewritten.
        assert perms["operator"]["blackboard_read"] == ["*"]


class TestSoloRegistrationScoping:
    @pytest.mark.asyncio
    async def test_solo_registration_scopes_watch_and_subscriptions(self, solo_app):
        """A solo worker's auto-watch/auto-subscribe land in its own
        team-of-one namespace (it now HAS blackboard perms)."""
        app, _store, blackboard, pubsub, _ = solo_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post(
                "/mesh/register",
                json={"agent_id": "solo-a", "capabilities": [], "port": 8400},
                headers=_headers("solo-a"),
            )
        assert r.status_code == 200
        watchers = blackboard.get_watchers_for_key("teams/solo-a/tasks/solo-a/ho_1")
        assert "solo-a" in watchers
        # No unscoped inbox watch survives.
        assert "solo-a" not in blackboard.get_watchers_for_key("tasks/solo-a/ho_1")
        subs = pubsub.get_agent_subscriptions("solo-a")
        assert all(t.startswith("teams/solo-a/") for t in subs), subs

    @pytest.mark.asyncio
    async def test_solo_list_agents_pseudo_team_resolves_to_self_plus_operator(self, solo_app):
        """The mesh client of a solo worker sends ?team={agent_id}; the host
        resolves the pseudo-team to a team-of-one roster (self + operator)
        instead of an empty dict — and never leaks other agents."""
        app, *_ = solo_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post(
                "/mesh/register",
                json={"agent_id": "solo-a", "capabilities": [], "port": 8400},
                headers=_headers("solo-a"),
            )
            r = await c.get(
                "/mesh/agents",
                params={"team": "solo-a"},
                headers=_headers("solo-a"),
            )
        assert r.status_code == 200
        roster = r.json()
        assert set(roster) == {"solo-a", "operator"}

    @pytest.mark.asyncio
    async def test_solo_cannot_scope_by_other_pseudo_team(self, solo_app):
        """Requesting another agent's pseudo-team returns nothing (enforce
        mode) — no cross-solo roster leak."""
        app, *_ = solo_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            await c.post(
                "/mesh/register",
                json={"agent_id": "solo-b", "capabilities": [], "port": 8400},
                headers=_headers("solo-b"),
            )
            r = await c.get(
                "/mesh/agents",
                params={"team": "solo-b"},
                headers=_headers("solo-a"),
            )
        assert r.status_code == 200
        assert r.json() == {}


class TestNoWildcardReadForWorkers:
    """Adversarial-review F1: with the agent-side standalone guards gone,
    the host ACL is THE read boundary — no create/boot path may leave a
    worker holding a blackboard_read wildcard (a never-teamed solo must
    not be WIDER than a team member, whose join strips ``*``)."""

    def _setup(self, tmp_path, monkeypatch):
        import src.cli.config as cli_cfg

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENLEGION_TEAMS_DB", str(tmp_path / "teams.db"))
        perms_file = tmp_path / "permissions.json"
        perms_file.write_text(json.dumps({"permissions": {}}))
        agents_file = tmp_path / "agents.yaml"
        agents_file.write_text(yaml.dump({"agents": {"newbie": {"role": "x"}}}))
        monkeypatch.setattr(cli_cfg, "PERMISSIONS_FILE", perms_file)
        monkeypatch.setattr(cli_cfg, "AGENTS_FILE", agents_file)
        monkeypatch.setattr(cli_cfg, "CONFIG_FILE", tmp_path / "mesh.yaml")
        return cli_cfg, perms_file

    def test_fresh_create_has_no_wildcard_read(self, tmp_path, monkeypatch):
        """The REAL create path (base + coordination merge) — the shape the
        solo_app fixture claims to reproduce — carries no ``*`` read."""
        cli_cfg, perms_file = self._setup(tmp_path, monkeypatch)
        cli_cfg._add_agent_permissions(
            "newbie",
            permissions=cli_cfg._DEFAULT_AGENT_COORDINATION_PERMS,
        )
        p = json.loads(perms_file.read_text())["permissions"]["newbie"]
        assert p["blackboard_read"] == ["teams/newbie/*"]
        assert "*" not in p["blackboard_write"]
        assert "teams/newbie/*" in p["blackboard_write"]

    def test_template_wildcard_read_is_stripped(self, tmp_path, monkeypatch):
        """Defense in depth: even a template shipping ``*`` cannot hand a
        worker fleet-wide reads."""
        cli_cfg, perms_file = self._setup(tmp_path, monkeypatch)
        cli_cfg._add_agent_permissions(
            "newbie",
            permissions={"blackboard_read": ["*", "research/*"]},
        )
        p = json.loads(perms_file.read_text())["permissions"]["newbie"]
        assert "*" not in p["blackboard_read"]
        assert "research/*" in p["blackboard_read"]

    def test_boot_narrows_untouched_default_wildcard(self, tmp_path, monkeypatch):
        """A teamless worker still holding the pre-#5 default read shape
        (["*"] or ["*"] + self) is narrowed at boot; a human-customized
        list is left alone."""
        cli_cfg, perms_file = self._setup(tmp_path, monkeypatch)
        perms_file.write_text(
            json.dumps(
                {
                    "permissions": {
                        "olddefault": {"blackboard_read": ["*"], "blackboard_write": []},
                        "olddefault2": {
                            "blackboard_read": ["*", "teams/olddefault2/*"],
                            "blackboard_write": [],
                        },
                        "custom": {
                            "blackboard_read": ["*", "research/*"],
                            "blackboard_write": [],
                        },
                    },
                }
            )
        )
        import yaml as _yaml

        (tmp_path / "agents.yaml").write_text(
            _yaml.dump(
                {
                    "agents": {
                        "olddefault": {"role": "x"},
                        "olddefault2": {"role": "y"},
                        "custom": {"role": "z"},
                    }
                }
            )
        )

        cli_cfg._ensure_all_agent_permissions()

        perms = json.loads(perms_file.read_text())["permissions"]
        assert perms["olddefault"]["blackboard_read"] == ["teams/olddefault/*"]
        assert perms["olddefault2"]["blackboard_read"] == ["teams/olddefault2/*"]
        # Human-customized pattern lists are never NARROWED (the wildcard
        # stays) — the self pattern is still added ("self always").
        assert sorted(perms["custom"]["blackboard_read"]) == ["*", "research/*", "teams/custom/*"]


class TestSelfNamespaceCarveOut:
    """Review F1/F3: the resolution-time carve-out — every agent can
    always touch its OWN teams/{id}/ namespace even with no ACL entry
    at all (ephemeral spawn agents resolve via the deny-all default
    record), and never anyone else's."""

    def test_unknown_agent_can_use_own_namespace_only(self):
        from src.host.permissions import PermissionMatrix

        pm = PermissionMatrix.__new__(PermissionMatrix)
        pm.permissions = {}  # no entries at all — spawn-agent posture
        assert pm.can_read_blackboard("spawn-abc123", "teams/spawn-abc123/notes/x") is True
        assert pm.can_write_blackboard("spawn-abc123", "teams/spawn-abc123/notes/x") is True
        # Zero shared reach: other namespaces stay denied.
        assert pm.can_read_blackboard("spawn-abc123", "teams/other-agent/notes/x") is False
        assert pm.can_write_blackboard("spawn-abc123", "teams/team-x/plan") is False
        assert pm.can_read_blackboard("spawn-abc123", "tasks/anything") is False

    def test_segment_boundary(self):
        from src.host.permissions import PermissionMatrix

        pm = PermissionMatrix.__new__(PermissionMatrix)
        pm.permissions = {}
        # "dev" must not match "teams/dev-lead/..."
        assert pm.can_read_blackboard("dev", "teams/dev-lead/notes") is False
        assert pm.can_read_blackboard("dev", "teams/dev/notes") is True
