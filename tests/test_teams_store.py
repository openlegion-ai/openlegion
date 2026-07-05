"""TeamStore unit tests — the first-class team entity (Phase 1).

Pins the store contract the mesh/dashboard/CLI repoint relies on:
canonical v1 schema, one-team-per-agent enforced by the schema,
membership eviction on move, file scaffold lifecycle, agent standing
goals keyed by agent alone, and the B4 budget envelope columns
(NULL/0 = unlimited is enforced at the cost layer; here we only pin
storage round-trips).
"""

import sqlite3

import pytest

from src.host.teams import (
    TeamExists,
    TeamNotFound,
    TeamStore,
    validate_team_id,
)


@pytest.fixture
def store(tmp_path):
    return TeamStore(db_path=str(tmp_path / "teams.db"), teams_dir=tmp_path / "teams")


class TestSchema:
    def test_user_version_is_1(self, tmp_path):
        db = tmp_path / "teams.db"
        TeamStore(db_path=str(db))
        conn = sqlite3.connect(str(db))
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        conn.close()

    def test_memory_mode(self):
        s = TeamStore(db_path=":memory:")
        s.create_team("alpha")
        assert s.team_exists("alpha")

    def test_reopen_preserves_rows(self, tmp_path):
        db = str(tmp_path / "teams.db")
        TeamStore(db_path=db).create_team("alpha", "desc")
        again = TeamStore(db_path=db)
        assert again.get_team("alpha")["description"] == "desc"


class TestValidation:
    def test_valid_ids(self):
        for tid in ("a", "team-1", "Team_2", "9lives"):
            assert validate_team_id(tid) == tid

    def test_invalid_ids(self):
        for tid in ("", "-lead", "a/b", "a" * 65, "a b", None):
            with pytest.raises(ValueError):
                validate_team_id(tid)

    def test_create_rejects_bad_id(self, store):
        with pytest.raises(ValueError):
            store.create_team("../escape")


class TestTeamCrud:
    def test_create_and_get(self, store):
        team = store.create_team("alpha", "does things")
        assert team["id"] == "alpha"
        assert team["name"] == "alpha"
        assert team["description"] == "does things"
        assert team["status"] == "active"
        assert team["members"] == []
        assert team["created_at"]
        assert team["budget_daily_usd"] is None
        assert team["budget_monthly_usd"] is None

    def test_create_duplicate_raises(self, store):
        store.create_team("alpha")
        with pytest.raises(TeamExists):
            store.create_team("alpha")

    def test_get_missing_returns_none(self, store):
        assert store.get_team("nope") is None

    def test_list_teams_sorted_with_members(self, store):
        store.create_team("beta")
        store.create_team("alpha")
        store.add_member("alpha", "bob")
        teams = store.list_teams()
        assert list(teams.keys()) == ["alpha", "beta"]
        assert teams["alpha"]["members"] == ["bob"]

    def test_list_excludes_archived_when_asked(self, store):
        store.create_team("alpha")
        store.create_team("beta")
        store.set_status("beta", "archived")
        assert set(store.list_teams(include_archived=False)) == {"alpha"}
        assert set(store.list_teams()) == {"alpha", "beta"}

    def test_count_teams(self, store):
        store.create_team("alpha")
        store.create_team("beta")
        store.set_status("beta", "archived")
        assert store.count_teams() == 2
        assert store.count_teams(include_archived=False) == 1

    def test_status_roundtrip(self, store):
        store.create_team("alpha")
        assert store.get_status("alpha") == "active"
        store.set_status("alpha", "archived")
        assert store.get_status("alpha") == "archived"

    def test_status_missing_team_raises(self, store):
        with pytest.raises(TeamNotFound):
            store.set_status("nope", "archived")
        with pytest.raises(TeamNotFound):
            store.get_status("nope")

    def test_delete_returns_members_and_clears(self, store):
        store.create_team("alpha")
        store.add_member("alpha", "bob")
        store.add_member("alpha", "eve")
        assert store.delete_team("alpha") == ["bob", "eve"]
        assert store.get_team("alpha") is None
        assert store.team_of("bob") is None

    def test_delete_missing_raises(self, store):
        with pytest.raises(TeamNotFound):
            store.delete_team("nope")


class TestFilesScaffold:
    def test_create_scaffolds_team_md_and_workflows(self, store, tmp_path):
        store.create_team("alpha", "my desc")
        team_dir = tmp_path / "teams" / "alpha"
        assert (team_dir / "workflows").is_dir()
        content = (team_dir / "team.md").read_text()
        assert "# alpha" in content
        assert "my desc" in content

    def test_delete_removes_dir(self, store, tmp_path):
        store.create_team("alpha")
        store.delete_team("alpha")
        assert not (tmp_path / "teams" / "alpha").exists()

    def test_team_md_path(self, store, tmp_path):
        store.create_team("alpha")
        assert store.team_md_path("alpha") == tmp_path / "teams" / "alpha" / "team.md"

    def test_pure_db_mode_skips_files(self, tmp_path):
        s = TeamStore(db_path=str(tmp_path / "t.db"))
        s.create_team("alpha")
        assert s.team_md_path("alpha") is None


class TestMetadata:
    def test_set_description(self, store):
        store.create_team("alpha")
        store.set_description("alpha", "new desc")
        assert store.get_team("alpha")["description"] == "new desc"

    def test_set_goal_roundtrip(self, store):
        store.create_team("alpha")
        store.set_goal("alpha", "ship it", ["criterion 1", "criterion 2"])
        team = store.get_team("alpha")
        assert team["north_star"] == "ship it"
        assert team["success_criteria"] == ["criterion 1", "criterion 2"]

    def test_set_goal_clears_criteria(self, store):
        store.create_team("alpha")
        store.set_goal("alpha", "ship it", ["c"])
        store.set_goal("alpha", "new star", None)
        team = store.get_team("alpha")
        assert team["north_star"] == "new star"
        assert team["success_criteria"] is None

    def test_settings_roundtrip(self, store):
        store.create_team("alpha")
        store.set_settings("alpha", {"summary_schedule": "0 9 * * *"})
        assert store.get_team("alpha")["settings"] == {"summary_schedule": "0 9 * * *"}

    def test_budget_roundtrip(self, store):
        store.create_team("alpha")
        store.set_budget("alpha", 25.0, 400.0)
        team = store.get_team("alpha")
        assert team["budget_daily_usd"] == 25.0
        assert team["budget_monthly_usd"] == 400.0

    def test_metadata_missing_team_raises(self, store):
        for call in (
            lambda: store.set_description("nope", "d"),
            lambda: store.set_goal("nope", "n"),
            lambda: store.set_settings("nope", {}),
            lambda: store.set_budget("nope", 1.0, 2.0),
        ):
            with pytest.raises(TeamNotFound):
                call()


class TestMembership:
    def test_add_and_team_of(self, store):
        store.create_team("alpha")
        assert store.add_member("alpha", "bob") is None
        assert store.team_of("bob") == "alpha"
        assert store.members("alpha") == ["bob"]

    def test_one_team_per_agent_eviction(self, store):
        store.create_team("alpha")
        store.create_team("beta")
        store.add_member("alpha", "bob")
        old = store.add_member("beta", "bob")
        assert old == "alpha"
        assert store.team_of("bob") == "beta"
        assert store.members("alpha") == []
        assert store.members("beta") == ["bob"]

    def test_readd_same_team_idempotent(self, store):
        store.create_team("alpha")
        store.add_member("alpha", "bob")
        assert store.add_member("alpha", "bob") == "alpha"
        assert store.members("alpha") == ["bob"]

    def test_operator_rejected(self, store):
        store.create_team("alpha")
        with pytest.raises(ValueError, match="system agent"):
            store.add_member("alpha", "operator")

    def test_add_to_missing_team_raises(self, store):
        with pytest.raises(TeamNotFound):
            store.add_member("nope", "bob")

    def test_remove_member(self, store):
        store.create_team("alpha")
        store.add_member("alpha", "bob")
        assert store.remove_member("alpha", "bob") is True
        assert store.team_of("bob") is None
        assert store.remove_member("alpha", "bob") is False

    def test_remove_agent_clears_membership_and_goals(self, store):
        store.create_team("alpha")
        store.add_member("alpha", "bob")
        store.set_agent_goals("bob", ["goal"])
        assert store.remove_agent("bob") == "alpha"
        assert store.team_of("bob") is None
        assert store.get_agent_goals("bob") is None
        assert store.remove_agent("bob") is None

    def test_agent_team_map(self, store):
        store.create_team("alpha")
        store.create_team("beta")
        store.add_member("alpha", "bob")
        store.add_member("beta", "eve")
        assert store.agent_team_map() == {"bob": "alpha", "eve": "beta"}

    def test_member_order_stable(self, store):
        store.create_team("alpha")
        for agent in ("charlie", "alice", "bob"):
            store.add_member("alpha", agent)
        assert store.members("alpha") == ["charlie", "alice", "bob"]


class TestAgentGoals:
    def test_unset_returns_none(self, store):
        assert store.get_agent_goals("bob") is None

    def test_set_and_get(self, store):
        store.set_agent_goals("bob", ["goal one", "goal two"], set_by="operator")
        rec = store.get_agent_goals("bob")
        assert rec["goals"] == ["goal one", "goal two"]
        assert rec["set_by"] == "operator"
        assert rec["updated_at"]

    def test_overwrite(self, store):
        store.set_agent_goals("bob", ["old"])
        store.set_agent_goals("bob", ["new"])
        assert store.get_agent_goals("bob")["goals"] == ["new"]

    def test_clear(self, store):
        store.set_agent_goals("bob", ["goal"])
        assert store.clear_agent_goals("bob") is True
        assert store.get_agent_goals("bob") is None
        assert store.clear_agent_goals("bob") is False

    def test_goals_survive_team_moves(self, store):
        store.create_team("alpha")
        store.create_team("beta")
        store.add_member("alpha", "bob")
        store.set_agent_goals("bob", ["standing goal"])
        store.add_member("beta", "bob")
        assert store.get_agent_goals("bob")["goals"] == ["standing goal"]
