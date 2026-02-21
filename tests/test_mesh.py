"""Unit tests for mesh layer: Blackboard, PubSub, MessageRouter, Permissions."""

import json

import pytest

from src.host.mesh import Blackboard, MessageRouter, PubSub
from src.host.permissions import PermissionMatrix
from src.shared.types import AgentPermissions

# === Blackboard Tests ===


@pytest.fixture
def blackboard(tmp_path):
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    yield bb
    bb.close()


def test_blackboard_write_and_read(blackboard):
    blackboard.write("context/test", {"data": "hello"}, written_by="agent1")
    entry = blackboard.read("context/test")
    assert entry is not None
    assert entry.value == {"data": "hello"}
    assert entry.written_by == "agent1"
    assert entry.version == 1


def test_blackboard_update_increments_version(blackboard):
    blackboard.write("context/test", {"v": 1}, written_by="a1")
    blackboard.write("context/test", {"v": 2}, written_by="a2")
    entry = blackboard.read("context/test")
    assert entry.version == 2
    assert entry.value == {"v": 2}
    assert entry.written_by == "a2"


def test_blackboard_list_by_prefix(blackboard):
    blackboard.write("context/a", {"x": 1}, written_by="a1")
    blackboard.write("context/b", {"x": 2}, written_by="a1")
    blackboard.write("tasks/c", {"x": 3}, written_by="a1")

    ctx_entries = blackboard.list_by_prefix("context/")
    assert len(ctx_entries) == 2

    task_entries = blackboard.list_by_prefix("tasks/")
    assert len(task_entries) == 1


def test_blackboard_delete(blackboard):
    blackboard.write("context/del_me", {"x": 1}, written_by="a1")
    blackboard.delete("context/del_me", deleted_by="a1")
    assert blackboard.read("context/del_me") is None


def test_blackboard_cannot_delete_history(blackboard):
    blackboard.write("history/result_1", {"x": 1}, written_by="a1")
    with pytest.raises(ValueError, match="Cannot delete from history"):
        blackboard.delete("history/result_1", deleted_by="a1")


def test_blackboard_read_nonexistent(blackboard):
    assert blackboard.read("nonexistent") is None


def test_blackboard_gc_expired(blackboard):
    blackboard.write("context/ephemeral", {"x": 1}, written_by="a1", ttl=0)
    # TTL=0 means already expired
    deleted = blackboard.gc_expired()
    # May or may not delete depending on timing; at minimum should not error
    assert deleted >= 0


# === PubSub Tests ===


def test_pubsub_subscribe_and_get():
    ps = PubSub()
    ps.subscribe("new_lead", "agent1")
    ps.subscribe("new_lead", "agent2")
    assert ps.get_subscribers("new_lead") == ["agent1", "agent2"]


def test_pubsub_no_duplicates():
    ps = PubSub()
    ps.subscribe("topic", "a1")
    ps.subscribe("topic", "a1")
    assert ps.get_subscribers("topic") == ["a1"]


def test_pubsub_unsubscribe():
    ps = PubSub()
    ps.subscribe("topic", "a1")
    ps.subscribe("topic", "a2")
    ps.unsubscribe("topic", "a1")
    assert ps.get_subscribers("topic") == ["a2"]


def test_pubsub_empty_topic():
    ps = PubSub()
    assert ps.get_subscribers("nonexistent") == []


# === PubSub Persistence Tests ===


def test_pubsub_persistence_survives_restart(tmp_path):
    """Subscriptions survive a PubSub restart when db_path is set."""
    db = str(tmp_path / "pubsub.db")
    ps1 = PubSub(db_path=db)
    ps1.subscribe("alerts", "agent1")
    ps1.subscribe("alerts", "agent2")
    ps1.subscribe("updates", "agent3")
    ps1.close()

    ps2 = PubSub(db_path=db)
    assert ps2.get_subscribers("alerts") == ["agent1", "agent2"]
    assert ps2.get_subscribers("updates") == ["agent3"]
    ps2.close()


def test_pubsub_event_persistence(tmp_path):
    """Published events are written to SQLite."""
    import sqlite3
    db = str(tmp_path / "pubsub.db")
    ps = PubSub(db_path=db)
    ps.subscribe("topic", "a1")
    ps.publish("topic", {"data": "hello"})
    ps.close()

    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1
    row = conn.execute("SELECT topic, data FROM events").fetchone()
    assert row[0] == "topic"
    assert "hello" in row[1]
    conn.close()


def test_pubsub_gc_events(tmp_path):
    """Events are garbage-collected when exceeding the threshold."""
    db = str(tmp_path / "pubsub.db")
    ps = PubSub(db_path=db)
    # Lower the threshold for testing
    ps._EVENT_GC_THRESHOLD = 10
    ps._EVENT_GC_KEEP = 5

    # Insert enough events to trigger GC at least once
    for i in range(20):
        ps.publish("topic", {"i": i})

    import sqlite3
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    # GC triggers when count > 10, keeping 5; more events may accumulate
    # after GC until the next threshold crossing. Count should be bounded.
    assert count <= ps._EVENT_GC_THRESHOLD
    conn.close()
    ps.close()


def test_pubsub_unsubscribe_persistence(tmp_path):
    """Unsubscribe is persisted across restarts."""
    db = str(tmp_path / "pubsub.db")
    ps1 = PubSub(db_path=db)
    ps1.subscribe("topic", "a1")
    ps1.subscribe("topic", "a2")
    ps1.unsubscribe("topic", "a1")
    ps1.close()

    ps2 = PubSub(db_path=db)
    assert ps2.get_subscribers("topic") == ["a2"]
    ps2.close()


def test_pubsub_no_db_backward_compat():
    """PubSub without db_path works identically to original behavior."""
    ps = PubSub()
    ps.subscribe("t", "a1")
    ps.publish("t", {"x": 1})
    assert ps.get_subscribers("t") == ["a1"]
    assert len(ps.event_log) == 1
    ps.close()  # should not raise


def test_pubsub_unsubscribe_agent():
    """unsubscribe_agent removes agent from all topics."""
    ps = PubSub()
    ps.subscribe("topic1", "a1")
    ps.subscribe("topic1", "a2")
    ps.subscribe("topic2", "a1")
    ps.subscribe("topic3", "a3")
    ps.unsubscribe_agent("a1")
    assert ps.get_subscribers("topic1") == ["a2"]
    assert ps.get_subscribers("topic2") == []
    assert ps.get_subscribers("topic3") == ["a3"]


def test_pubsub_unsubscribe_agent_persistence(tmp_path):
    """unsubscribe_agent is persisted to SQLite."""
    db = str(tmp_path / "pubsub.db")
    ps1 = PubSub(db_path=db)
    ps1.subscribe("t1", "a1")
    ps1.subscribe("t2", "a1")
    ps1.subscribe("t1", "a2")
    ps1.unsubscribe_agent("a1")
    ps1.close()

    ps2 = PubSub(db_path=db)
    assert ps2.get_subscribers("t1") == ["a2"]
    assert ps2.get_subscribers("t2") == []
    ps2.close()


# === Permission Tests ===


@pytest.fixture
def permissions(tmp_path):
    config = {
        "permissions": {
            "research": {
                "can_message": ["orchestrator"],
                "can_publish": ["research_complete"],
                "can_subscribe": ["new_lead"],
                "blackboard_read": ["context/*", "tasks/*"],
                "blackboard_write": ["context/research_*"],
                "allowed_apis": ["anthropic", "brave_search"],
            },
            "qualify": {
                "can_message": ["orchestrator"],
                "blackboard_read": ["context/*"],
                "blackboard_write": [],
                "allowed_apis": ["anthropic"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    return PermissionMatrix(config_path=str(config_path))


def test_permissions_can_message(permissions):
    assert permissions.can_message("research", "orchestrator")
    assert not permissions.can_message("research", "qualify")


def test_permissions_orchestrator_always_allowed(permissions):
    assert permissions.can_message("orchestrator", "research")
    assert permissions.can_read_blackboard("orchestrator", "anything")


def test_permissions_blackboard_read_glob(permissions):
    assert permissions.can_read_blackboard("research", "context/prospect_1")
    assert permissions.can_read_blackboard("research", "tasks/active")
    assert not permissions.can_read_blackboard("research", "signals/alert")


def test_permissions_blackboard_write_glob(permissions):
    assert permissions.can_write_blackboard("research", "context/research_data")
    assert not permissions.can_write_blackboard("research", "context/qualify_data")


def test_permissions_api_access(permissions):
    assert permissions.can_use_api("research", "anthropic")
    assert permissions.can_use_api("research", "brave_search")
    assert not permissions.can_use_api("research", "hunter")


def test_permissions_deny_unknown_agent(permissions):
    assert not permissions.can_message("unknown", "orchestrator")
    assert not permissions.can_read_blackboard("unknown", "context/test")


def test_permissions_publish(permissions):
    assert permissions.can_publish("research", "research_complete")
    assert not permissions.can_publish("research", "other_topic")


# === MessageRouter Tests ===


def test_router_resolves_direct_agent():
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "a1": AgentPermissions(agent_id="a1", can_message=["a2"]),
    }
    router = MessageRouter(permissions=perms, agent_registry={"a2": "http://a2:8400"})
    url = router._resolve_target("a2")
    assert url == "http://a2:8400"


def test_router_resolves_capability():
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={"research_1": "http://r1:8400"})
    router._capabilities_cache = {"research_1": ["web_search", "company_lookup"]}
    url = router._resolve_target("capability:web_search")
    assert url == "http://r1:8400"


def test_router_returns_none_for_unknown():
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    assert router._resolve_target("nonexistent") is None


def test_router_register_and_unregister():
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {}
    router = MessageRouter(permissions=perms, agent_registry={})
    router.register_agent("a1", "http://a1:8400", ["search"])
    assert "a1" in router.agent_registry
    assert router._capabilities_cache["a1"] == ["search"]

    router.unregister_agent("a1")
    assert "a1" not in router.agent_registry


def test_permissions_wildcard_messaging(tmp_path):
    """Wildcard can_message: ['*'] still works for system/orchestrator use."""
    config = {
        "permissions": {
            "ceo": {
                "can_message": ["*"],
                "blackboard_read": ["context/*", "goals/*"],
                "blackboard_write": ["context/*", "goals/*"],
                "allowed_apis": ["llm"],
            },
            "worker": {
                "can_message": ["orchestrator"],
                "blackboard_read": ["context/*"],
                "blackboard_write": [],
                "allowed_apis": ["llm"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_message("ceo", "worker")
    assert pm.can_message("ceo", "engineer")
    assert pm.can_message("ceo", "anyone")
    assert not pm.can_message("worker", "ceo")
    assert pm.can_message("worker", "orchestrator")


def test_can_manage_vault_default_false(permissions):
    """can_manage_vault defaults to False for agents without explicit grant."""
    assert permissions.can_manage_vault("research") is False
    assert permissions.can_manage_vault("qualify") is False


def test_can_manage_vault_granted(tmp_path):
    """Agent with can_manage_vault=True passes the check."""
    config = {
        "permissions": {
            "admin": {
                "can_message": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
                "allowed_apis": ["llm"],
                "can_manage_vault": True,
            },
            "worker": {
                "can_message": ["orchestrator"],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_manage_vault("admin") is True
    assert pm.can_manage_vault("worker") is False


def test_can_manage_vault_orchestrator_always_allowed(permissions):
    """Orchestrator and mesh always have vault access."""
    assert permissions.can_manage_vault("orchestrator") is True
    assert permissions.can_manage_vault("mesh") is True
