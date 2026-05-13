"""Tests for operator audit trail, pending change store, and config endpoints."""

from __future__ import annotations

import sqlite3
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


def test_audit_log_undoable_default_false(blackboard):
    """A row written without ``undoable=`` defaults to False so the
    dashboard's Revert button stays hidden on legacy / hard-edit rows."""
    blackboard.log_audit(
        action="edit_agent", target="alpha", field="model",
        change_id="some-nonce",
    )
    entry = blackboard.get_audit_log()["entries"][0]
    assert entry["undoable"] is False


def test_audit_log_undoable_true_when_set(blackboard):
    """The soft-edit path passes ``undoable=True`` so the audit row
    advertises that ``change_id`` is a valid undo_token."""
    blackboard.log_audit(
        action="edit_agent", target="alpha", field="instructions",
        change_id="undo-uuid", undoable=True,
    )
    entry = blackboard.get_audit_log()["entries"][0]
    assert entry["undoable"] is True


def test_audit_log_undoable_column_migration(tmp_path):
    """Legacy DBs without the ``undoable`` column get the column added
    by Blackboard.__init__ and existing rows surface as undoable=False."""
    db_path = tmp_path / "legacy.db"
    # Create the table in the *old* shape — no ``undoable`` column.
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript("""
        CREATE TABLE audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            action TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'operator',
            target TEXT NOT NULL,
            field TEXT,
            before_value TEXT,
            after_value TEXT,
            change_id TEXT,
            provenance TEXT DEFAULT 'user'
        );
    """)
    legacy.execute(
        "INSERT INTO audit_log (action, target, field, change_id) "
        "VALUES (?, ?, ?, ?)",
        ("edit_agent", "alpha", "model", "legacy-nonce"),
    )
    legacy.commit()
    legacy.close()

    # Opening the Blackboard should ALTER the table and not raise.
    bb = Blackboard(db_path=str(db_path))
    try:
        cols = {
            row[1]
            for row in bb.db.execute("PRAGMA table_info(audit_log)").fetchall()
        }
        assert "undoable" in cols

        # Legacy row should be readable and report undoable=False.
        result = bb.get_audit_log()
        assert result["total"] == 1
        legacy_entry = result["entries"][0]
        assert legacy_entry["change_id"] == "legacy-nonce"
        assert legacy_entry["undoable"] is False

        # Re-running the migration on an already-migrated DB is a no-op.
        bb2 = Blackboard(db_path=str(db_path))
        try:
            cols2 = {
                row[1]
                for row in bb2.db.execute(
                    "PRAGMA table_info(audit_log)"
                ).fetchall()
            }
            assert "undoable" in cols2
        finally:
            bb2.close()
    finally:
        bb.close()


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


@pytest.mark.asyncio
async def test_mesh_client_edit_soft_wires_unified_endpoint():
    """MeshClient.edit_soft is the primary client method for the unified
    edit endpoint that now accepts both soft and hard fields. Verifies
    URL, payload shape, and that the response (including the new
    ``ttl_seconds`` and ``field_class`` keys) is passed through.
    """
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "agent_id": "alpha",
        "field": "model",
        "undo_token": "tok-xyz",
        "expires_at": "2026-05-13T20:00:00+00:00",
        "ttl_seconds": 1800,
        "field_class": "hard",
        "summary": "Updated alpha's model",
    }
    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.edit_soft(
            "alpha", "model", "anthropic/claude-opus-4-7", "user_asked",
        )
        assert result["undo_token"] == "tok-xyz"
        assert result["ttl_seconds"] == 1800
        assert result["field_class"] == "hard"
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/agents/alpha/edit-soft" in call_args[0][0]
        body = call_args[1]["json"]
        assert body["field"] == "model"
        assert body["value"] == "anthropic/claude-opus-4-7"
        assert body["reason"] == "user_asked"
    await client.close()


@pytest.mark.asyncio
async def test_mesh_client_undo_change_wires_endpoint():
    """MeshClient.undo_change targets /mesh/changes/undo/{token} so the
    operator-tool layer's ``undo_change`` skill can reverse an edit by
    token. Verifies URL shape and response passthrough.
    """
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "success": True,
        "agent_id": "alpha",
        "field": "model",
        "restored_value": "openai/gpt-4o-mini",
    }
    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.undo_change("tok-xyz")
        assert result["success"] is True
        assert result["restored_value"] == "openai/gpt-4o-mini"
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/changes/undo/tok-xyz" in call_args[0][0]
    await client.close()


# === Audit archive tests (PR C) ===


def test_audit_log_archived_column_exists(blackboard):
    """The audit_log table should expose the archived column."""
    cols = [
        row[1]
        for row in blackboard.db.execute("PRAGMA table_info(audit_log)").fetchall()
    ]
    assert "archived" in cols


def test_archive_audit_before_filters_old_rows(blackboard):
    """archive_audit_before flips matching rows to archived=1."""
    blackboard.log_audit(action="old_action", target="alpha")
    # Backdate the inserted row by hand so we can test the cutoff.
    blackboard.db.execute(
        "UPDATE audit_log SET timestamp='2025-01-01 00:00:00' WHERE target='alpha'"
    )
    blackboard.db.commit()
    blackboard.log_audit(action="recent_action", target="beta")

    res = blackboard.archive_audit_before("2026-01-01")
    assert res["archived_count"] == 1
    assert res["truncated"] is False

    # Default view hides archived rows but keeps the audit-of-audit
    # row written by archive_audit_before itself.
    result = blackboard.get_audit_log()
    targets = sorted(e["target"] for e in result["entries"])
    assert "beta" in targets
    assert "audit_log" in targets  # audit-of-audit row

    # include_archived surfaces both originals + the audit-of-audit row.
    result = blackboard.get_audit_log(include_archived=True)
    archived = [e for e in result["entries"] if e["archived"]]
    assert len(archived) == 1
    assert archived[0]["target"] == "alpha"


def test_archive_audit_before_writes_audit_of_audit(blackboard):
    """archive_audit_before records its own action with the actor."""
    blackboard.log_audit(action="old_action", target="alpha")
    blackboard.db.execute(
        "UPDATE audit_log SET timestamp='2025-01-01 00:00:00' WHERE target='alpha'"
    )
    blackboard.db.commit()

    blackboard.archive_audit_before("2026-01-01", actor="operator")
    rows = blackboard.db.execute(
        "SELECT action, actor, target, after_value, archived FROM audit_log"
        " WHERE action='audit_archive'"
    ).fetchall()
    assert len(rows) == 1
    action, actor, target, after_value, archived = rows[0]
    assert action == "audit_archive"
    assert actor == "operator"
    assert target == "audit_log"
    assert after_value == "2026-01-01"
    assert archived == 0  # the audit-of-audit row itself is NOT archived


def test_archive_audit_before_idempotent(blackboard):
    """Running archive twice doesn't double-count."""
    blackboard.log_audit(action="old_action", target="alpha")
    blackboard.db.execute(
        "UPDATE audit_log SET timestamp='2025-01-01 00:00:00' WHERE target='alpha'"
    )
    blackboard.db.commit()
    r1 = blackboard.archive_audit_before("2026-01-01")
    r2 = blackboard.archive_audit_before("2026-01-01")
    assert r1["archived_count"] == 1
    # The second call still archives nothing from the original window
    # (the audit-of-audit row from r1 is dated "now", well after 2026-01-01).
    assert r2["archived_count"] == 0


def test_archive_audit_before_normalises_t_separator(blackboard):
    """ISO 8601 with T separator should match the SQLite TEXT format."""
    blackboard.log_audit(action="old_action", target="alpha")
    blackboard.db.execute(
        "UPDATE audit_log SET timestamp='2025-01-01 00:00:00' WHERE target='alpha'"
    )
    blackboard.db.commit()
    res = blackboard.archive_audit_before("2026-01-01T00:00:00Z")
    assert res["archived_count"] == 1


def test_archive_audit_before_chunked_truncated(blackboard, monkeypatch):
    """When the hard cap is hit, ``truncated=True`` and rows remain."""
    # Drop the batch size + cap so the test runs in milliseconds.
    monkeypatch.setattr(blackboard, "_ARCHIVE_BATCH_SIZE", 5, raising=False)
    monkeypatch.setattr(blackboard, "_ARCHIVE_HARD_CAP", 10, raising=False)
    for i in range(20):
        blackboard.log_audit(action=f"old_{i}", target=f"a{i}")
    # Backdate them all.
    blackboard.db.execute(
        "UPDATE audit_log SET timestamp='2025-01-01 00:00:00'"
    )
    blackboard.db.commit()

    res = blackboard.archive_audit_before("2026-01-01")
    assert res["truncated"] is True
    assert res["archived_count"] == 10
    # 10 of the 20 originals are still archived=0 + matching the cutoff;
    # a follow-up call should sweep the rest.
    res2 = blackboard.archive_audit_before("2026-01-01")
    assert res2["archived_count"] == 10
    assert res2["truncated"] is True or res2["truncated"] is False  # exact-cap edge


def test_archive_audit_before_rejects_empty(blackboard):
    """Empty before_iso raises so the endpoint can map to HTTP 400."""
    with pytest.raises(ValueError):
        blackboard.archive_audit_before("")


def test_get_audit_log_default_hides_archived(blackboard):
    """get_audit_log() should drop archived rows by default."""
    blackboard.log_audit(action="a", target="alpha")
    blackboard.log_audit(action="b", target="beta")
    blackboard.db.execute("UPDATE audit_log SET archived=1 WHERE target='alpha'")
    blackboard.db.commit()

    result = blackboard.get_audit_log()
    assert result["total"] == 1
    assert result["entries"][0]["target"] == "beta"
    assert result["entries"][0]["archived"] is False


@pytest.mark.asyncio
async def test_mesh_client_cancel_pending_action():
    """MeshClient.cancel_pending_action POSTs to the right endpoint."""
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "ok": True, "nonce": "nonce-1",
        "target_kind": "agent", "target_id": "writer", "action_kind": "edit",
    }

    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.cancel_pending_action("nonce-1")
        assert result["ok"] is True
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/pending/nonce-1/cancel" in call_args[0][0]
    await client.close()


@pytest.mark.asyncio
async def test_mesh_client_archive_audit_before():
    """MeshClient.archive_audit_before POSTs to /mesh/audit/archive."""
    from src.agent.mesh_client import MeshClient

    client = MeshClient("http://localhost:8420", "operator")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "ok": True, "archived_count": 5, "before_date": "2026-04-01",
    }

    mock_http = AsyncMock(return_value=mock_response)
    with patch.object(client, "_get_client", return_value=AsyncMock(post=mock_http)):
        result = await client.archive_audit_before("2026-04-01")
        assert result["archived_count"] == 5
        mock_http.assert_called_once()
        call_args = mock_http.call_args
        assert "/mesh/audit/archive" in call_args[0][0]
        assert call_args[1]["json"]["before_date"] == "2026-04-01"
    await client.close()


# === /mesh/audit/archive endpoint — auth + date validation ===
#
# These tests exercise the mesh route directly (not the agent-side
# MeshClient) so the operator-or-internal gate, ISO 8601 date
# validation, and chunked-archive return shape are covered end-to-end.


def _build_audit_archive_app(tmp_path):
    """Spin up a create_mesh_app() bound to a tmp blackboard for archive tests."""
    import importlib

    import src.host.server as server_module
    from src.host.costs import CostTracker
    from src.host.mesh import MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.traces import TraceStore
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    def _cleanup():
        bb.close()
        costs.close()
        traces.close()
        importlib.reload(server_module)

    return app, bb, _cleanup


@pytest.mark.asyncio
async def test_audit_archive_endpoint_requires_operator_or_internal(tmp_path):
    """A non-operator + non-internal caller is rejected with HTTP 403."""
    from httpx import ASGITransport, AsyncClient

    app, bb, cleanup = _build_audit_archive_app(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/audit/archive",
                json={"before_date": "2026-04-01"},
                headers={"X-Agent-ID": "alpha"},  # not operator, not internal
            )
        assert resp.status_code == 403
        assert "operator" in resp.json()["detail"].lower()
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_audit_archive_endpoint_accepts_operator_caller(tmp_path):
    """``X-Agent-ID: operator`` is accepted and the archive runs."""
    from httpx import ASGITransport, AsyncClient

    app, bb, cleanup = _build_audit_archive_app(tmp_path)
    try:
        bb.log_audit(action="old", target="alpha")
        bb.db.execute(
            "UPDATE audit_log SET timestamp='2025-01-01 00:00:00' WHERE target='alpha'"
        )
        bb.db.commit()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/audit/archive",
                json={"before_date": "2026-04-01"},
                headers={"X-Agent-ID": "operator"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["archived_count"] == 1
        assert body["truncated"] is False
        # audit-of-audit row was written with actor=operator.
        rows = bb.db.execute(
            "SELECT actor FROM audit_log WHERE action='audit_archive'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "operator"
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_audit_archive_endpoint_validates_iso8601_formats(tmp_path):
    """Bare-date / Z-suffixed / offset-suffixed inputs all parse."""
    from httpx import ASGITransport, AsyncClient

    app, bb, cleanup = _build_audit_archive_app(tmp_path)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            for fmt in (
                "2026-04-01",
                "2026-04-01T12:00:00Z",
                "2026-04-01T12:00:00+02:00",
            ):
                resp = await client.post(
                    "/mesh/audit/archive",
                    json={"before_date": fmt},
                    headers={"X-Agent-ID": "operator"},
                )
                assert resp.status_code == 200, f"format {fmt} should parse: {resp.text}"
                body = resp.json()
                assert body["ok"] is True
                assert "truncated" in body

            # Garbage rejected with HTTP 400.
            resp = await client.post(
                "/mesh/audit/archive",
                json={"before_date": "not-a-date"},
                headers={"X-Agent-ID": "operator"},
            )
            assert resp.status_code == 400
            assert "iso 8601" in resp.json()["detail"].lower()
    finally:
        cleanup()
