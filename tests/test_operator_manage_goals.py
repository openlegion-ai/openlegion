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
    assert result == {"ok": True, "goals": []}


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
