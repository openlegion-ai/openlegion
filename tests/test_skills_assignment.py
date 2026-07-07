"""Per-agent + fleet-wide skill assignment (PR 5).

Covers ``PermissionMatrix.get_effective_skills`` (fleet ∪ per-agent) and the
per-agent filtering in the skills_list / skill_view tools (operator + solo
team-of-one workers see the full catalog; everyone else only their assignment;
fetch errors fail closed).
"""

from __future__ import annotations

import json

import pytest

from src.agent.builtins import skills_tool
from src.host.permissions import PermissionMatrix


def _write_skill(directory, name, description="d", body="Body."):
    pack = directory / name
    pack.mkdir(parents=True, exist_ok=True)
    (pack / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def _perms_file(tmp_path, data) -> str:
    p = tmp_path / "permissions.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


# ── PermissionMatrix.get_effective_skills ─────────────────────────────────

def test_effective_skills_union_fleet_and_per_agent(tmp_path):
    cfg = _perms_file(tmp_path, {
        "fleet_skills": ["common"],
        "permissions": {"alice": {"allowed_skills": ["alpha", "beta"]}},
    })
    pm = PermissionMatrix(config_path=cfg)
    assert pm.get_effective_skills("alice") == ["alpha", "beta", "common"]


def test_effective_skills_default_empty(tmp_path):
    cfg = _perms_file(tmp_path, {"permissions": {"bob": {}}})
    pm = PermissionMatrix(config_path=cfg)
    assert pm.get_effective_skills("bob") == []


def test_fleet_skills_apply_to_every_agent(tmp_path):
    cfg = _perms_file(tmp_path, {"fleet_skills": ["common"], "permissions": {}})
    pm = PermissionMatrix(config_path=cfg)
    # An agent with no explicit entry still gets the fleet-wide skill.
    assert pm.get_effective_skills("anyone") == ["common"]


def test_effective_skills_dedup_and_sorted(tmp_path):
    cfg = _perms_file(tmp_path, {
        "fleet_skills": ["dup"],
        "permissions": {"a": {"allowed_skills": ["dup", "x"]}},
    })
    pm = PermissionMatrix(config_path=cfg)
    assert pm.get_effective_skills("a") == ["dup", "x"]


def test_fleet_skills_fail_closed_on_missing_file(tmp_path):
    pm = PermissionMatrix(config_path=str(tmp_path / "nope.json"))
    assert pm.fleet_skills == []
    assert pm.get_effective_skills("a") == []


def test_assign_preserves_default_template_grants(tmp_path):
    """Materializing effective perms before a partial skill write must NOT strip
    grants an agent inherited from the 'default' template (Codex review must-fix:
    a bare {allowed_skills} write would drop the agent out of default fallback)."""
    import json as _json

    cfg = _perms_file(tmp_path, {
        "permissions": {"default": {"allowed_apis": ["svc"], "can_use_browser": True}},
    })
    pm = PermissionMatrix(config_path=cfg)
    # 'worker' has no explicit entry → inherits the default template.
    assert pm.get_permissions("worker").allowed_apis == ["svc"]
    assert pm.get_permissions("worker").can_use_browser is True

    # Replicate the endpoint's materialize-then-write.
    data = _json.loads((tmp_path / "permissions.json").read_text())
    agents = data.setdefault("permissions", {})
    if "worker" not in agents:
        agents["worker"] = pm.get_permissions("worker").model_dump(exclude={"agent_id"})
    agents["worker"]["allowed_skills"] = ["alpha"]
    (tmp_path / "permissions.json").write_text(_json.dumps(data))
    pm.reload()

    # Skill assigned AND the inherited grants survive.
    assert pm.get_permissions("worker").allowed_skills == ["alpha"]
    assert pm.get_permissions("worker").allowed_apis == ["svc"]
    assert pm.get_permissions("worker").can_use_browser is True


def test_reload_picks_up_new_assignment(tmp_path):
    cfg = _perms_file(tmp_path, {"permissions": {"a": {}}})
    pm = PermissionMatrix(config_path=cfg)
    assert pm.get_effective_skills("a") == []
    # Operator assigns a skill on disk, then the mesh reloads the matrix.
    (tmp_path / "permissions.json").write_text(
        json.dumps({"permissions": {"a": {"allowed_skills": ["alpha"]}}}),
        encoding="utf-8",
    )
    pm.reload()
    assert pm.get_effective_skills("a") == ["alpha"]


# ── skills_list / skill_view per-agent filtering ──────────────────────────

class _FakeMesh:
    # A team worker: scoped to a real team (not a team-of-one).
    agent_id = "worker"
    team_name = "alpha"

    def __init__(self, names):
        self._names = names

    async def list_my_skills(self):
        return self._names


@pytest.fixture
def store_with_three(tmp_path, monkeypatch):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    for n in ("alpha", "beta", "gamma"):
        _write_skill(bundled, n)
    monkeypatch.setenv("SKILLS_DIR", str(bundled))
    monkeypatch.setenv("SKILLS_INSTALLED_DIR", str(tmp_path / "none"))
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)  # not the operator
    return bundled


async def test_skills_list_filters_to_assigned(store_with_three):
    result = await skills_tool.skills_list(mesh_client=_FakeMesh(["alpha", "gamma"]))
    assert {s["name"] for s in result["skills"]} == {"alpha", "gamma"}
    assert result["count"] == 2


async def test_skills_list_empty_when_none_assigned(store_with_three):
    result = await skills_tool.skills_list(mesh_client=_FakeMesh([]))
    assert result["skills"] == []
    assert result["count"] == 0


async def test_skill_view_blocks_unassigned(store_with_three):
    result = await skills_tool.skill_view("beta", mesh_client=_FakeMesh(["alpha"]))
    assert "error" in result and "not found" in result["error"]


async def test_skill_view_allows_assigned(store_with_three):
    result = await skills_tool.skill_view("beta", mesh_client=_FakeMesh(["beta"]))
    assert result["name"] == "beta"
    assert "error" not in result


async def test_operator_sees_full_catalog(store_with_three, monkeypatch):
    # The operator (fleet manager) bypasses assignment — ALLOWED_TOOLS marks it.
    monkeypatch.setenv("ALLOWED_TOOLS", "skills_list,edit_agent")
    result = await skills_tool.skills_list(mesh_client=_FakeMesh([]))
    assert result["count"] == 3


async def test_solo_team_of_one_sees_full_catalog(store_with_three):
    """A solo worker (team-of-one: team_name == agent_id, ratified #5)
    sees the whole catalog without fetching an assignment — preserves the
    pre-merge standalone behavior. Sound because the collision guard
    forbids a real team named after an existing agent."""
    class Solo:
        agent_id = "solo"
        team_name = "solo"

        async def list_my_skills(self):
            raise AssertionError("a team-of-one must not fetch assignment")

    result = await skills_tool.skills_list(mesh_client=Solo())
    assert result["count"] == 3


async def test_no_mesh_client_sees_full_catalog(store_with_three):
    result = await skills_tool.skills_list(mesh_client=None)
    assert result["count"] == 3


async def test_fetch_error_fails_closed(store_with_three):
    class Boom:
        agent_id = "worker"
        team_name = "alpha"

        async def list_my_skills(self):
            raise RuntimeError("mesh down")

    # A fetch error hides skills rather than flooding the agent with all of them.
    result = await skills_tool.skills_list(mesh_client=Boom())
    assert result["count"] == 0
