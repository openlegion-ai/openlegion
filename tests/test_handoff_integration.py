"""Integration test: ``hand_off`` → ``list_task_inbox``.

Canary for Bug 1: agent A calls ``hand_off`` and agent B's inbox
must contain the resulting row immediately afterwards. Exercises the
real :class:`Tasks` store (``:memory:`` shared connection) wired up to
a mock ``mesh_client`` whose write/read methods proxy to the store —
the same pattern :class:`TestHandOffParentTaskIdPropagation` in
``test_coordination.py`` uses, but driving the FULL round-trip end-to-end
(create → store → list_inbox) so a regression in the
``assignee``-normalization / post-write-verify chain would be caught
here even if individual unit tests stayed green.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_hand_off_creates_task_visible_to_recipient_inbox(monkeypatch):
    """End-to-end: ``hand_off`` from scout to bob lands a row that
    ``list_inbox(bob)`` returns immediately.

    Assertions:
      * exactly one row in bob's inbox
      * ``assignee == "bob"`` (whitespace-stripped, byte-exact)
      * ``parent_task_id`` matches the seeded contextvar
      * ``status == "pending"``
      * ``hand_off`` reports ``handed_off=True`` so the caller does not
        emit a ``create_failed`` envelope.
    """
    from src.agent.builtins.coordination_tool import hand_off
    from src.host.orchestration import Tasks
    from src.shared.trace import current_task_id

    # Real Tasks store, shared in-memory connection so writes/reads see
    # the same SQLite database across both hand_off's create + our inbox
    # readback.
    store = Tasks(db_path=":memory:")

    # Mock mesh_client whose async surface proxies the relevant calls
    # into the durable store.
    mc = MagicMock()
    mc.agent_id = "scout"
    mc.is_standalone = False
    mc.team_name = "default"
    mc.project_name = "default"

    # Roster — bob has to exist for hand_off's pre-flight validation.
    mc.list_agents = AsyncMock(return_value={
        "scout": {"role": "scout", "project": "default"},
        "bob": {"role": "analyst", "project": "default"},
    })

    # ``create_task`` proxies into ``store.create``. ``hand_off`` invokes
    # this with kw-args matching the MeshClient.create_task signature.
    async def fake_create_task(*, assignee, title, description=None,
                               project=None, parent_task_id=None,
                               priority=0, dependencies=None,
                               artifact_refs=None, origin=None):
        return store.create(
            creator=mc.agent_id,
            assignee=assignee,
            title=title,
            description=description,
            project_id=project,
            parent_task_id=parent_task_id,
            priority=priority,
            dependencies=dependencies,
            artifact_refs=artifact_refs,
            origin=None,
        )
    mc.create_task = AsyncMock(side_effect=fake_create_task)

    # Wake/write are best-effort decorations — return success-shaped dicts
    # so hand_off proceeds through the happy path.
    mc.wake_agent = AsyncMock(return_value={"woken": True})
    mc.write_blackboard = AsyncMock(return_value={"version": 1})

    # Seed a known parent task contextvar so the propagation invariant
    # can be asserted on the stored row.
    parent_token = current_task_id.set("task_root_parent")
    try:
        result = await hand_off(
            to="bob", summary="do the thing", mesh_client=mc,
        )
    finally:
        current_task_id.reset(parent_token)

    # hand_off succeeded.
    assert result.get("handed_off") is True, result
    assert result.get("create_failed") is not True

    # Recipient's inbox sees exactly one row with the right shape.
    rows = store.list_inbox("bob")
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["assignee"] == "bob"
    assert row["parent_task_id"] == "task_root_parent"
    assert row["status"] == "pending"
    assert row["creator"] == "scout"
