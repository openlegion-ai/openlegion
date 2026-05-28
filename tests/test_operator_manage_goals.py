"""Tests for the ``manage_goals`` operator tool (PR 1 of Work tab rewrite).

Covers:

* All five actions (set / add / update / remove / list)
* `add` rejects duplicate names
* `update` / `remove` reject unknown names
* `set` enforces max 10 goals + duplicate name rejection
* Status enum validation
* `updated_at` stamped on touched goals; preserved on untouched ones
* GOALS.md and GOALS.json stay in sync
* Operator-only gating
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_TOOLS", "manage_goals")


class _FakeWorkspace:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path


def _load_goals_json(tmp_path: Path) -> list[dict]:
    path = tmp_path / "GOALS.json"
    assert path.exists()
    return json.loads(path.read_text())["goals"]


@pytest.mark.asyncio
async def test_manage_goals_non_operator_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("list", workspace_manager=ws)
    assert "error" in result
    assert "operator" in result["error"].lower()


@pytest.mark.asyncio
async def test_manage_goals_no_workspace():
    from src.agent.builtins.operator_tools import manage_goals

    result = await manage_goals("list")
    assert "error" in result
    assert "workspace_manager" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_unknown_action(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("nope", workspace_manager=ws)
    assert "error" in result
    assert "Unknown action" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_list_empty(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("list", workspace_manager=ws)
    # PR 972 Codex follow-up — list now also returns the seed_ask
    # throttle block (None when never recorded).
    assert result == {"ok": True, "goals": [], "seed_ask": None}


@pytest.mark.asyncio
async def test_manage_goals_set_writes_both_files(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "set",
        goals=[
            {"name": "Launch", "status": "in_progress", "progress_note": "running smoke"},
            {"name": "Audit", "status": "on_track"},
        ],
        workspace_manager=ws,
    )
    assert result["ok"] is True
    assert len(result["goals"]) == 2
    # JSON sidecar persisted.
    persisted = _load_goals_json(tmp_path)
    assert len(persisted) == 2
    assert persisted[0]["name"] == "Launch"
    assert persisted[0]["status"] == "in_progress"
    assert persisted[0]["progress_note"] == "running smoke"
    assert persisted[0]["updated_at"]
    # Markdown rendered.
    md = (tmp_path / "GOALS.md").read_text()
    assert "# Goals" in md
    assert "## Launch" in md
    assert "**Status:** in_progress" in md
    assert "running smoke" in md
    assert "## Audit" in md


@pytest.mark.asyncio
async def test_manage_goals_set_caps_max_entries(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    too_many = [
        {"name": f"g{i}", "status": "in_progress"} for i in range(11)
    ]
    result = await manage_goals("set", goals=too_many, workspace_manager=ws)
    assert "error" in result
    assert "max length" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_set_rejects_duplicates(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "set",
        goals=[
            {"name": "Same", "status": "in_progress"},
            {"name": "Same", "status": "on_track"},
        ],
        workspace_manager=ws,
    )
    assert "error" in result
    assert "duplicate" in result["error"].lower()


@pytest.mark.asyncio
async def test_manage_goals_status_enum_validation(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "set",
        goals=[{"name": "Bad", "status": "not_a_status"}],
        workspace_manager=ws,
    )
    assert "error" in result
    assert "status" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_add_then_list(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "add",
        goal={"name": "First", "status": "in_progress"},
        workspace_manager=ws,
    )
    result = await manage_goals("list", workspace_manager=ws)
    assert len(result["goals"]) == 1
    assert result["goals"][0]["name"] == "First"


@pytest.mark.asyncio
async def test_manage_goals_add_rejects_duplicate_name(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "add",
        goal={"name": "Same", "status": "in_progress"},
        workspace_manager=ws,
    )
    result = await manage_goals(
        "add",
        goal={"name": "Same", "status": "on_track"},
        workspace_manager=ws,
    )
    assert "error" in result
    assert "already exists" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_update_stamps_updated_at(tmp_path):
    """`update` refreshes ``updated_at`` only on the touched goal."""
    import time as _time

    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "set",
        goals=[
            {"name": "A", "status": "in_progress"},
            {"name": "B", "status": "in_progress"},
        ],
        workspace_manager=ws,
    )
    initial = _load_goals_json(tmp_path)
    initial_a = initial[0]["updated_at"]
    initial_b = initial[1]["updated_at"]
    _time.sleep(0.01)  # ensure timestamp resolution differs
    await manage_goals(
        "update",
        goal={"name": "A", "status": "done", "progress_note": "shipped"},
        workspace_manager=ws,
    )
    after = _load_goals_json(tmp_path)
    by_name = {g["name"]: g for g in after}
    # A's updated_at moved forward; B's preserved.
    assert by_name["A"]["updated_at"] != initial_a
    assert by_name["A"]["status"] == "done"
    assert by_name["A"]["progress_note"] == "shipped"
    assert by_name["B"]["updated_at"] == initial_b


@pytest.mark.asyncio
async def test_manage_goals_update_unknown_rejected(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "update",
        goal={"name": "Nope", "status": "in_progress"},
        workspace_manager=ws,
    )
    assert "error" in result
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_remove_unknown_rejected(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("remove", name="Ghost", workspace_manager=ws)
    assert "error" in result
    assert "not found" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_remove_happy_path(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "set",
        goals=[
            {"name": "Keep", "status": "in_progress"},
            {"name": "Drop", "status": "blocked"},
        ],
        workspace_manager=ws,
    )
    result = await manage_goals("remove", name="Drop", workspace_manager=ws)
    assert result["ok"] is True
    assert len(result["goals"]) == 1
    assert result["goals"][0]["name"] == "Keep"
    # File state matches.
    md = (tmp_path / "GOALS.md").read_text()
    assert "Drop" not in md
    assert "Keep" in md


@pytest.mark.asyncio
async def test_manage_goals_md_and_json_stay_in_sync(tmp_path):
    """After every mutation both files reflect the same goal list."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "set",
        goals=[{"name": "A", "status": "in_progress"}],
        workspace_manager=ws,
    )
    await manage_goals(
        "add",
        goal={"name": "B", "status": "blocked"},
        workspace_manager=ws,
    )
    json_goals = _load_goals_json(tmp_path)
    md = (tmp_path / "GOALS.md").read_text()
    names_json = {g["name"] for g in json_goals}
    assert names_json == {"A", "B"}
    assert "## A" in md
    assert "## B" in md


@pytest.mark.asyncio
async def test_manage_goals_remove_requires_name(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("remove", workspace_manager=ws)
    assert "error" in result
    assert "name" in result["error"]


@pytest.mark.asyncio
async def test_manage_goals_update_preserves_progress_note_when_unset(tmp_path):
    """Regression: update without ``progress_note`` must preserve existing.

    The first cut wiped the note silently on every update that didn't
    pass it. Verify the validator's None-sentinel + update merge keeps
    the existing note when the caller only touches status.
    """
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "add",
        goal={
            "name": "Launch",
            "status": "in_progress",
            "progress_note": "draft running",
        },
        workspace_manager=ws,
    )
    # Update only the status; do not pass progress_note.
    await manage_goals(
        "update",
        goal={"name": "Launch", "status": "done"},
        workspace_manager=ws,
    )
    persisted = _load_goals_json(tmp_path)
    assert len(persisted) == 1
    assert persisted[0]["status"] == "done"
    assert persisted[0]["progress_note"] == "draft running"


@pytest.mark.asyncio
async def test_manage_goals_update_progress_note_explicit_empty_clears(tmp_path):
    """Caller can still clear the note by passing ``progress_note: \"\"``."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "add",
        goal={
            "name": "Launch",
            "status": "in_progress",
            "progress_note": "old note",
        },
        workspace_manager=ws,
    )
    await manage_goals(
        "update",
        goal={"name": "Launch", "status": "in_progress", "progress_note": ""},
        workspace_manager=ws,
    )
    persisted = _load_goals_json(tmp_path)
    assert persisted[0]["progress_note"] == ""


@pytest.mark.asyncio
async def test_manage_goals_update_status_only_keeps_other_fields(tmp_path):
    """Updating only status leaves name + note + (other goals') updated_at intact."""
    import time as _time

    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "set",
        goals=[
            {"name": "A", "status": "in_progress", "progress_note": "alpha note"},
            {"name": "B", "status": "in_progress", "progress_note": "beta note"},
        ],
        workspace_manager=ws,
    )
    initial = _load_goals_json(tmp_path)
    initial_a_note = initial[0]["progress_note"]
    initial_b_note = initial[1]["progress_note"]
    _time.sleep(0.01)
    await manage_goals(
        "update",
        goal={"name": "A", "status": "blocked"},
        workspace_manager=ws,
    )
    after = _load_goals_json(tmp_path)
    by_name = {g["name"]: g for g in after}
    assert by_name["A"]["status"] == "blocked"
    assert by_name["A"]["progress_note"] == initial_a_note  # preserved
    assert by_name["B"]["progress_note"] == initial_b_note  # untouched


@pytest.mark.asyncio
async def test_manage_goals_add_caps_at_max_entries(tmp_path):
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    initial = [
        {"name": f"g{i}", "status": "in_progress"} for i in range(10)
    ]
    await manage_goals("set", goals=initial, workspace_manager=ws)
    overflow = await manage_goals(
        "add",
        goal={"name": "one_more", "status": "in_progress"},
        workspace_manager=ws,
    )
    assert "error" in overflow
    assert "max" in overflow["error"].lower()


# ── Cold-start seed-ask throttle (PR 972 Codex follow-up) ─────────


def _load_goals_payload(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "GOALS.json").read_text())


@pytest.mark.asyncio
async def test_list_returns_null_seed_ask_when_unset(tmp_path):
    """Fresh workspace: no GOALS.json yet → ``seed_ask`` is ``None`` so
    the heartbeat can ask immediately."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals("list", workspace_manager=ws)
    assert result["ok"] is True
    assert result["goals"] == []
    assert result["seed_ask"] is None


@pytest.mark.asyncio
async def test_record_seed_ask_writes_structured_block(tmp_path):
    """``record_seed_ask`` writes a ``{last_ts, team_names}`` block to
    GOALS.json and echoes it back to the caller."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "record_seed_ask",
        team_names=["alpha", "beta"],
        workspace_manager=ws,
    )
    assert result["ok"] is True
    seed = result["seed_ask"]
    assert seed["team_names"] == ["alpha", "beta"]
    assert seed["last_ts"]  # ISO timestamp
    # File-level check.
    payload = _load_goals_payload(tmp_path)
    assert payload["seed_ask"] == seed


@pytest.mark.asyncio
async def test_list_returns_recorded_seed_ask(tmp_path):
    """After ``record_seed_ask``, ``list`` echoes the same block back."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "record_seed_ask",
        team_names=["growth"],
        workspace_manager=ws,
    )
    result = await manage_goals("list", workspace_manager=ws)
    assert result["seed_ask"]["team_names"] == ["growth"]


@pytest.mark.asyncio
async def test_seed_ask_preserved_across_goal_mutations(tmp_path):
    """Recording a seed_ask and then mutating goals via set/add/update
    must NOT wipe the seed_ask block — the throttle has to outlive
    incremental goal edits."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    await manage_goals(
        "record_seed_ask",
        team_names=["growth"],
        workspace_manager=ws,
    )
    before = _load_goals_payload(tmp_path)["seed_ask"]
    # set
    await manage_goals(
        "set",
        goals=[{"name": "Launch landing page", "status": "in_progress"}],
        workspace_manager=ws,
    )
    assert _load_goals_payload(tmp_path)["seed_ask"] == before
    # add
    await manage_goals(
        "add",
        goal={"name": "Sign 5 design partners", "status": "not_started"},
        workspace_manager=ws,
    )
    assert _load_goals_payload(tmp_path)["seed_ask"] == before
    # update
    await manage_goals(
        "update",
        goal={"name": "Launch landing page", "status": "on_track"},
        workspace_manager=ws,
    )
    assert _load_goals_payload(tmp_path)["seed_ask"] == before


@pytest.mark.asyncio
async def test_record_seed_ask_caps_team_names_and_sanitises(tmp_path):
    """Defensive bounds: empty/whitespace strings dropped, list capped."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    raw = [
        "alpha", "", "  ", "beta", None, 42,
        *[f"team-{i}" for i in range(25)],
    ]
    result = await manage_goals(
        "record_seed_ask",
        team_names=raw,
        workspace_manager=ws,
    )
    names = result["seed_ask"]["team_names"]
    assert "alpha" in names and "beta" in names
    assert "" not in names
    assert None not in names
    assert 42 not in names
    assert len(names) <= 20  # _MAX_SEED_ASK_TEAMS


@pytest.mark.asyncio
async def test_record_seed_ask_updates_timestamp_on_replay(tmp_path):
    """A second call updates the timestamp (the throttle's whole point
    is "when did I last ask")."""
    import time as _time

    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    first = await manage_goals(
        "record_seed_ask",
        team_names=["alpha"],
        workspace_manager=ws,
    )
    _time.sleep(0.01)
    second = await manage_goals(
        "record_seed_ask",
        team_names=["alpha"],
        workspace_manager=ws,
    )
    assert second["seed_ask"]["last_ts"] > first["seed_ask"]["last_ts"]


@pytest.mark.asyncio
async def test_record_seed_ask_refuses_on_corrupt_goals_json(tmp_path):
    """Codex r3 catch — when GOALS.json is corrupt (e.g. a partial
    write from a prior crash), ``record_seed_ask`` must NOT silently
    write back ``{"goals": [], "seed_ask": {...}}`` and thereby wipe
    any goals the file would have decoded to. It must abort and
    surface an error so the operator can repair the file."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    # First, seed some goals via the happy path so we have a valid
    # baseline to "lose" — proves the corrupt-input refusal is what
    # protects the data, not just an empty-file check.
    await manage_goals(
        "set",
        goals=[{"name": "Launch landing page", "status": "in_progress"}],
        workspace_manager=ws,
    )
    # Now simulate a corrupt write — truncated JSON.
    (tmp_path / "GOALS.json").write_text('{"goals": [{"name":')
    result = await manage_goals(
        "record_seed_ask",
        team_names=["growth"],
        workspace_manager=ws,
    )
    assert "error" in result
    assert "corrupt" in result["error"].lower()
    # File is left untouched — operator can inspect and repair.
    assert (tmp_path / "GOALS.json").read_text() == '{"goals": [{"name":'


@pytest.mark.asyncio
async def test_record_seed_ask_seeds_into_empty_workspace(tmp_path):
    """No GOALS.json yet → record_seed_ask creates one with empty
    goals + the new seed_ask block. The corrupt-file guard at
    ``_safe_read_goals_for_merge`` distinguishes "missing" (legitimate
    empty) from "exists but corrupt" (raise)."""
    from src.agent.builtins.operator_tools import manage_goals

    ws = _FakeWorkspace(tmp_path)
    result = await manage_goals(
        "record_seed_ask",
        team_names=["alpha"],
        workspace_manager=ws,
    )
    assert result["ok"] is True
    payload = json.loads((tmp_path / "GOALS.json").read_text())
    assert payload["goals"] == []
    assert payload["seed_ask"]["team_names"] == ["alpha"]
