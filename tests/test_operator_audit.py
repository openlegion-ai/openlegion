"""Tests for operator audit trail, pending change store, and config endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.mesh import Blackboard
from src.host.server import (
    _CHANGE_TTL_SECONDS,
    _HARD_CHANGE_TTL_SECONDS,
    _cleanup_expired_changes,
    _consume_pending_change,
    _get_pending_actions_store,
    _get_pending_change,
    _store_pending_change,
)

# === Blackboard Audit Log Tests ===


@pytest.fixture
def blackboard(tmp_path):
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    yield bb
    bb.close()


def test_audit_log_table_created(blackboard):
    """The audit_log table should exist after init."""
    row = blackboard.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
    ).fetchone()
    assert row is not None


def test_log_audit_basic(blackboard):
    """log_audit inserts a record retrievable by get_audit_log."""
    blackboard.log_audit(
        action="edit_agent", target="alpha",
        field="model", before_value="gpt-4o", after_value="claude-sonnet-4-20250514",
        change_id="abc-123",
    )
    result = blackboard.get_audit_log()
    assert result["total"] == 1
    entry = result["entries"][0]
    assert entry["action"] == "edit_agent"
    assert entry["target"] == "alpha"
    assert entry["field"] == "model"
    assert entry["before_value"] == "gpt-4o"
    assert entry["after_value"] == "claude-sonnet-4-20250514"
    assert entry["change_id"] == "abc-123"
    assert entry["actor"] == "operator"
    assert entry["provenance"] == "user"


def test_log_audit_custom_actor_provenance(blackboard):
    """log_audit respects actor and provenance parameters."""
    blackboard.log_audit(
        action="restart_agent", target="beta",
        actor="admin", provenance="api",
    )
    result = blackboard.get_audit_log()
    entry = result["entries"][0]
    assert entry["actor"] == "admin"
    assert entry["provenance"] == "api"


def test_audit_log_pagination(blackboard):
    """get_audit_log supports pagination."""
    for i in range(5):
        blackboard.log_audit(action=f"action_{i}", target="alpha")

    # First page
    result = blackboard.get_audit_log(page=1, per_page=2)
    assert result["total"] == 5
    assert result["page"] == 1
    assert result["per_page"] == 2
    assert len(result["entries"]) == 2
    # Most recent first
    assert result["entries"][0]["action"] == "action_4"
    assert result["entries"][1]["action"] == "action_3"

    # Second page
    result = blackboard.get_audit_log(page=2, per_page=2)
    assert len(result["entries"]) == 2
    assert result["entries"][0]["action"] == "action_2"

    # Third page
    result = blackboard.get_audit_log(page=3, per_page=2)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["action"] == "action_0"


def test_audit_log_filter_by_agent(blackboard):
    """get_audit_log filters by agent_id (target)."""
    blackboard.log_audit(action="edit", target="alpha")
    blackboard.log_audit(action="edit", target="beta")
    blackboard.log_audit(action="restart", target="alpha")

    result = blackboard.get_audit_log(agent_id="alpha")
    assert result["total"] == 2
    assert all(e["target"] == "alpha" for e in result["entries"])


def test_audit_log_filter_by_action(blackboard):
    """get_audit_log filters by action type."""
    blackboard.log_audit(action="edit_agent", target="alpha")
    blackboard.log_audit(action="restart_agent", target="alpha")
    blackboard.log_audit(action="edit_agent", target="beta")

    result = blackboard.get_audit_log(action="edit_agent")
    assert result["total"] == 2
    assert all(e["action"] == "edit_agent" for e in result["entries"])


def test_audit_log_filter_by_since(blackboard):
    """get_audit_log filters by timestamp."""
    blackboard.log_audit(action="old_action", target="alpha")
    # Use a far-future timestamp to ensure the filter works
    result = blackboard.get_audit_log(since="2099-01-01")
    assert result["total"] == 0


def test_audit_log_combined_filters(blackboard):
    """get_audit_log supports combining filters."""
    blackboard.log_audit(action="edit_agent", target="alpha")
    blackboard.log_audit(action="restart_agent", target="alpha")
    blackboard.log_audit(action="edit_agent", target="beta")

    result = blackboard.get_audit_log(agent_id="alpha", action="edit_agent")
    assert result["total"] == 1
    assert result["entries"][0]["target"] == "alpha"
    assert result["entries"][0]["action"] == "edit_agent"


def test_audit_log_empty(blackboard):
    """get_audit_log returns empty results when no entries exist."""
    result = blackboard.get_audit_log()
    assert result["total"] == 0
    assert result["entries"] == []
    assert result["page"] == 1


# === Pending Change Store Tests ===


def _clear_pending_actions() -> None:
    """Drop every row from the pending-actions store (test helper)."""
    store = _get_pending_actions_store()
    with store._conn() as conn:
        conn.execute("DELETE FROM pending_actions")


@pytest.fixture(autouse=True)
def clear_pending_changes():
    """Clear the pending changes store before each test."""
    _clear_pending_actions()
    yield
    _clear_pending_actions()


def test_store_and_get_pending_change():
    """Store a change and retrieve it."""
    change_id = _store_pending_change("alpha", "model", "gpt-4o", "claude-sonnet-4-20250514")
    change = _get_pending_change(change_id)
    assert change is not None
    assert change["agent_id"] == "alpha"
    assert change["field"] == "model"
    assert change["old_value"] == "gpt-4o"
    assert change["new_value"] == "claude-sonnet-4-20250514"


def test_consume_pending_change():
    """Consuming removes the change."""
    change_id = _store_pending_change("alpha", "model", "old", "new")
    change = _consume_pending_change(change_id)
    assert change is not None
    assert change["field"] == "model"

    # Second consume returns None
    assert _consume_pending_change(change_id) is None


def test_consume_nonexistent_change():
    """Consuming a non-existent change returns None."""
    assert _consume_pending_change("nonexistent") is None


def test_expired_changes_cleaned_up():
    """Expired changes are removed during cleanup."""
    change_id = _store_pending_change("alpha", "model", "old", "new")
    # Force expiry by backdating ``expires_at`` directly on the store.
    store = _get_pending_actions_store()
    expired_ts = (datetime.now(timezone.utc) - timedelta(seconds=1)).timestamp()
    with store._conn() as conn:
        conn.execute(
            "UPDATE pending_actions SET expires_at=? WHERE nonce=?",
            (expired_ts, change_id),
        )

    _cleanup_expired_changes()
    assert _get_pending_change(change_id) is None


def test_max_pending_evicts_oldest():
    """When at capacity, the oldest change is evicted."""
    from src.host.server import _MAX_PENDING

    ids = []
    for i in range(_MAX_PENDING):
        cid = _store_pending_change("alpha", f"field_{i}", f"old_{i}", f"new_{i}")
        ids.append(cid)

    # Store one more — should evict the oldest
    new_id = _store_pending_change("alpha", "extra", "old", "new")
    assert len(_get_pending_actions_store().list_pending()) == _MAX_PENDING
    # The first stored should be gone
    assert _get_pending_change(ids[0]) is None
    # The new one should exist
    assert _get_pending_change(new_id) is not None


def test_get_nonexistent_change():
    """Getting a non-existent change returns None."""
    assert _get_pending_change("nonexistent") is None


def test_change_ttl_hard_field_uses_long_window():
    """Hard fields (model / permissions / budget / thinking) get the
    longer ``_HARD_CHANGE_TTL_SECONDS`` review window so the user has
    time to read the diff before confirming."""
    change_id = _store_pending_change("alpha", "model", "old", "new")
    change = _get_pending_change(change_id)
    assert change is not None
    expected = datetime.now(timezone.utc) + timedelta(seconds=_HARD_CHANGE_TTL_SECONDS)
    # Allow 5 seconds tolerance
    assert abs((change["expires_at"] - expected).total_seconds()) < 5


def test_change_ttl_soft_field_uses_default_window():
    """Soft fields fall back to the legacy ``_CHANGE_TTL_SECONDS`` window."""
    change_id = _store_pending_change("alpha", "instructions", "old", "new")
    change = _get_pending_change(change_id)
    assert change is not None
    expected = datetime.now(timezone.utc) + timedelta(seconds=_CHANGE_TTL_SECONDS)
    # Allow 5 seconds tolerance
    assert abs((change["expires_at"] - expected).total_seconds()) < 5


@pytest.mark.parametrize(
    "field,expected",
    [
        # Hard fields → 30-min review window.
        ("model", 1800),
        ("permissions", 1800),
        ("budget", 1800),
        ("thinking", 1800),
        # Soft fields → 5-min undo window.
        ("instructions", 300),
        ("soul", 300),
        ("heartbeat", 300),
        ("interface", 300),
        ("role", 300),
        # Unknown / typo / non-config action → falls through to soft.
        # Anything not in HARD_EDIT_FIELDS gets the shorter window so
        # the worst-case is "user has to retry sooner", not "user
        # waited 30 minutes for a no-op".
        ("unknown_field", 300),
        ("modle", 300),  # typo of "model" — must not earn the long window
        ("", 300),
        (None, 300),
    ],
)
def test_ttl_for_field(field, expected):
    """``_ttl_for_field`` maps a config field to the right TTL bucket.

    Hard fields earn the 30-minute review window; everything else
    (soft fields, typos, unknown action kinds, ``None``) falls through
    to the 5-minute soft window.
    """
    from src.host.server import _ttl_for_field

    assert _ttl_for_field(field) == expected


# === MeshClient Config Method Tests ===


@pytest.mark.asyncio
async def test_mesh_client_get_agent_config():
    """MeshClient.get_agent_config calls the correct endpoint."""
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"agent_id": "alpha", "config": {"model": "gpt-4o"}}

    with patch.object(client, "_get_with_retry", return_value=mock_response) as mock_get:
        result = await client.get_agent_config("alpha")
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert "mesh/agents/alpha/config" in call_args[0][0]
        assert result["agent_id"] == "alpha"
    await client.close()


@pytest.mark.asyncio
async def test_mesh_client_propose_config_change():
    """MeshClient.propose_config_change calls the correct endpoint."""
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"change_id": "abc-123", "preview_diff": "..."}

    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.propose_config_change("alpha", "model", "claude-sonnet-4-20250514")
        assert result["change_id"] == "abc-123"
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/agents/alpha/propose" in call_args[0][0]
        assert call_args[1]["json"]["field"] == "model"
        assert call_args[1]["json"]["value"] == "claude-sonnet-4-20250514"
    await client.close()


@pytest.mark.asyncio
async def test_mesh_client_confirm_config_change():
    """MeshClient.confirm_config_change calls the correct endpoint."""
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"success": True, "agent_id": "alpha", "field": "model"}

    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.confirm_config_change("abc-123")
        assert result["success"] is True
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/config/confirm" in call_args[0][0]
        assert call_args[1]["json"]["change_id"] == "abc-123"
    await client.close()
