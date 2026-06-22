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


def test_blackboard_list_by_prefix_excludes_ttl_expired(blackboard):
    """TTL-expired rows must not be returned by list_by_prefix. The TTL GC is
    throttled (once/60s, write-triggered) so a reader could otherwise see
    entries past their TTL — the root cause of the operator inbox flood."""
    # A live entry (no TTL) and a live TTL entry that has NOT expired.
    blackboard.write("inbox/op/task_event/live", {"x": 1}, written_by="a1")
    blackboard.write(
        "inbox/op/task_event/fresh", {"x": 2}, written_by="a1", ttl=3600
    )
    # An entry with a TTL whose updated_at we backdate so it is firmly expired.
    blackboard.write(
        "inbox/op/task_event/stale", {"x": 3}, written_by="a1", ttl=60
    )
    blackboard.db.execute(
        "UPDATE entries SET updated_at = datetime('now', '-2 hours') "
        "WHERE key = ?",
        ("inbox/op/task_event/stale",),
    )
    blackboard.db.commit()

    entries = blackboard.list_by_prefix("inbox/op/task_event/")
    keys = {e.key for e in entries}
    assert keys == {"inbox/op/task_event/live", "inbox/op/task_event/fresh"}
    assert "inbox/op/task_event/stale" not in keys


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


def test_ttl_gc_runs_once_under_concurrent_writes(tmp_path):
    """Concurrent writes with TTL entries should only trigger GC once per interval."""
    import threading

    bb = Blackboard(db_path=str(tmp_path / "gc_race.db"))
    bb._TTL_GC_INTERVAL = 60  # large interval so only one GC should run

    gc_call_count = 0
    original_gc_unlocked = bb._gc_expired_unlocked
    gc_lock = threading.Lock()

    def counting_gc():
        nonlocal gc_call_count
        with gc_lock:
            gc_call_count += 1
        return original_gc_unlocked()

    # Write a TTL entry so GC has something to do (before patching,
    # so the write's own _maybe_gc_ttl call doesn't count)
    bb.write("context/ttl_entry", {"x": 1}, written_by="a1", ttl=0)

    bb._gc_expired_unlocked = counting_gc

    # Force the last GC time far in the past so the next call triggers GC.
    # ``float("-inf")`` is required because ``time.monotonic()`` is unsigned
    # and can return values < _TTL_GC_INTERVAL on cold CI runners; setting
    # ``_last_ttl_gc = 0`` would make ``now - 0 < 60`` evaluate True and the
    # GC would silently skip (matches the sentinel pattern in
    # ``src/host/traces.py:29``).
    bb._last_ttl_gc = float("-inf")

    start_event = threading.Event()

    def concurrent_gc():
        start_event.wait(timeout=5)
        bb._maybe_gc_ttl()

    threads = [threading.Thread(target=concurrent_gc) for _ in range(5)]
    for t in threads:
        t.start()
    start_event.set()
    for t in threads:
        t.join(timeout=10)

    # Only one thread should have called _gc_expired_unlocked
    assert gc_call_count == 1
    bb.close()


# === Blackboard Cleanup Tests ===


def test_blackboard_cleanup_agent_data(tmp_path):
    """cleanup_agent_data removes entries, event log, and watches for an agent."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    bb.write("context/a1", {"x": 1}, written_by="agent1")
    bb.write("context/a2", {"x": 2}, written_by="agent1")
    bb.write("context/b1", {"x": 3}, written_by="agent2")
    bb.add_watch("agent1", "context/*")

    deleted = bb.cleanup_agent_data("agent1")
    assert deleted == 2

    # agent1 entries gone
    assert bb.read("context/a1") is None
    assert bb.read("context/a2") is None

    # agent2 entry untouched
    assert bb.read("context/b1") is not None

    # agent1 watches gone
    watchers = bb.get_watchers_for_key("context/something")
    assert "agent1" not in watchers

    # event log for agent1 gone
    rows = bb.db.execute(
        "SELECT COUNT(*) FROM event_log WHERE agent_id = ?", ("agent1",),
    ).fetchone()
    assert rows[0] == 0

    bb.close()


def test_blackboard_cleanup_nonexistent_agent(tmp_path):
    """cleanup_agent_data on a non-existent agent is a no-op."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    deleted = bb.cleanup_agent_data("ghost")
    assert deleted == 0
    bb.close()


# === Blackboard Agent Profile Support ===


def test_blackboard_get_agent_watches(tmp_path):
    """get_agent_watches returns glob patterns registered by an agent."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    bb.add_watch("agent-a", "tasks/*")
    bb.add_watch("agent-a", "status/*")
    bb.add_watch("agent-b", "research/*")

    assert sorted(bb.get_agent_watches("agent-a")) == ["status/*", "tasks/*"]
    assert bb.get_agent_watches("agent-b") == ["research/*"]
    assert bb.get_agent_watches("agent-c") == []
    bb.close()


def test_blackboard_recent_keys_by_agent(tmp_path):
    """recent_keys_by_agent returns keys recently written by an agent."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    bb.write("k1", {"v": 1}, written_by="agent-a")
    bb.write("k2", {"v": 2}, written_by="agent-a")
    bb.write("k3", {"v": 3}, written_by="agent-b")
    bb.write("k4", {"v": 4}, written_by="agent-a")

    keys = bb.recent_keys_by_agent("agent-a")
    assert "k1" in keys
    assert "k2" in keys
    assert "k4" in keys
    assert "k3" not in keys  # written by agent-b

    # Test limit
    keys_limited = bb.recent_keys_by_agent("agent-a", limit=2)
    assert len(keys_limited) <= 2

    # Unknown agent returns empty
    assert bb.recent_keys_by_agent("nobody") == []
    bb.close()


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


# === PubSub Agent Profile Support ===


def test_pubsub_get_agent_subscriptions():
    """get_agent_subscriptions returns topics an agent subscribes to."""
    ps = PubSub()
    ps.subscribe("topic-a", "agent-1")
    ps.subscribe("topic-b", "agent-1")
    ps.subscribe("topic-a", "agent-2")

    subs = ps.get_agent_subscriptions("agent-1")
    assert sorted(subs) == ["topic-a", "topic-b"]

    subs2 = ps.get_agent_subscriptions("agent-2")
    assert subs2 == ["topic-a"]

    assert ps.get_agent_subscriptions("agent-3") == []
    ps.close()


def test_pubsub_get_agent_subscriptions_persistence(tmp_path):
    """get_agent_subscriptions works after restart with persistence."""
    db = str(tmp_path / "pubsub.db")
    ps1 = PubSub(db_path=db)
    ps1.subscribe("alerts", "agent-1")
    ps1.subscribe("updates", "agent-1")
    ps1.subscribe("alerts", "agent-2")
    ps1.close()

    ps2 = PubSub(db_path=db)
    subs = ps2.get_agent_subscriptions("agent-1")
    assert sorted(subs) == ["alerts", "updates"]
    assert ps2.get_agent_subscriptions("agent-2") == ["alerts"]
    ps2.close()


# === Permission Tests ===


@pytest.fixture
def permissions(tmp_path):
    config = {
        "permissions": {
            "research": {
                "can_message": ["mesh"],
                "can_publish": ["research_complete"],
                "can_subscribe": ["new_lead"],
                "blackboard_read": ["context/*", "tasks/*"],
                "blackboard_write": ["context/research_*"],
                "allowed_apis": ["anthropic", "brave_search"],
            },
            "qualify": {
                "can_message": ["mesh"],
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
    assert permissions.can_message("research", "mesh")
    assert not permissions.can_message("research", "qualify")


def test_permissions_mesh_always_allowed(permissions):
    assert permissions.can_message("mesh", "research")
    assert permissions.can_read_blackboard("mesh", "anything")


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
    assert not permissions.can_message("unknown", "mesh")
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
    """Wildcard can_message: ['*'] still works for system/mesh use."""
    config = {
        "permissions": {
            "ceo": {
                "can_message": ["*"],
                "blackboard_read": ["context/*", "goals/*"],
                "blackboard_write": ["context/*", "goals/*"],
                "allowed_apis": ["llm"],
            },
            "worker": {
                "can_message": ["mesh"],
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
    assert pm.can_message("worker", "mesh")


def test_can_manage_vault_default_false(permissions):
    """can_manage_vault defaults to False for agents without explicit grant."""
    assert permissions.can_manage_vault("research") is False
    assert permissions.can_manage_vault("qualify") is False


def test_can_manage_vault_granted(tmp_path):
    """Agent with allowed_credentials passes the can_manage_vault check."""
    config = {
        "permissions": {
            "admin": {
                "can_message": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
                "allowed_apis": ["llm"],
                "allowed_credentials": ["*"],
            },
            "worker": {
                "can_message": ["mesh"],
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


def test_can_manage_vault_mesh_always_allowed(permissions):
    """Mesh always has vault access."""
    assert permissions.can_manage_vault("mesh") is True


# === Credential Scoping Tests ===


def test_can_access_credential_system_always_denied(tmp_path):
    """System credentials are never resolvable by agents, even with wildcard."""
    config = {
        "permissions": {
            "agent": {
                "can_message": ["*"],
                "blackboard_read": ["*"],
                "blackboard_write": ["*"],
                "allowed_apis": ["llm"],
                "allowed_credentials": ["*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("agent", "anthropic_api_key") is False
    assert pm.can_access_credential("agent", "openai_api_key") is False
    assert pm.can_access_credential("agent", "openai_api_base") is False


def test_can_access_credential_wildcard(tmp_path):
    """Agent with allowed_credentials: ['*'] can access any non-system credential."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": ["*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("agent", "brightdata_cdp_url") is True
    assert pm.can_access_credential("agent", "myapp_password") is True


def test_can_access_credential_glob_pattern(tmp_path):
    """Agent with specific glob patterns can only access matching credentials."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": ["brightdata_*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("agent", "brightdata_cdp_url") is True
    assert pm.can_access_credential("agent", "myapp_password") is False


def test_can_access_credential_empty_denied(tmp_path):
    """Agent with empty allowed_credentials cannot access any credentials."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": [],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("agent", "brightdata_cdp_url") is False
    assert pm.can_access_credential("agent", "myapp_password") is False


def test_can_access_credential_mesh_always_allowed(tmp_path):
    """Mesh bypasses credential scoping."""
    config = {"permissions": {}}
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("mesh", "openai_api_key") is True


def test_allowed_credentials_controls_vault_access(tmp_path):
    """allowed_credentials patterns control both vault access and credential resolution."""
    config = {
        "permissions": {
            "full_access": {
                "can_message": ["mesh"],
                "blackboard_read": ["*"],
                "blackboard_write": [],
                "allowed_apis": ["llm"],
                "allowed_credentials": ["*"],
            },
            "restricted": {
                "can_message": ["mesh"],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_manage_vault("full_access") is True
    assert pm.can_access_credential("full_access", "brightdata_cdp_url") is True

    assert pm.can_manage_vault("restricted") is False
    assert pm.can_access_credential("restricted", "brightdata_cdp_url") is False


def test_get_allowed_credentials(tmp_path):
    """get_allowed_credentials returns the patterns list."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": ["brightdata_*", "myapp_*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.get_allowed_credentials("agent") == ["brightdata_*", "myapp_*"]
    assert pm.get_allowed_credentials("unknown") == []


def test_can_access_credential_multiple_glob_patterns(tmp_path):
    """Agent with multiple glob patterns can access matching credentials from any pattern."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": ["brightdata_*", "myapp_*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("agent", "brightdata_cdp_url") is True
    assert pm.can_access_credential("agent", "myapp_password") is True
    assert pm.can_access_credential("agent", "other_service_key") is False
    # System credentials still blocked even if pattern would match
    assert pm.can_access_credential("agent", "anthropic_api_key") is False


def test_can_access_credential_case_insensitive_patterns(tmp_path):
    """Patterns are matched case-insensitively since credential names are always lowercase."""
    config = {
        "permissions": {
            "agent": {
                "can_message": [],
                "blackboard_read": [],
                "blackboard_write": [],
                "allowed_apis": [],
                "allowed_credentials": ["BrightData_*", "MyApp_*"],
            },
        }
    }
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    # Credential names are always stored lowercase — patterns should match regardless of case
    assert pm.can_access_credential("agent", "brightdata_cdp_url") is True
    assert pm.can_access_credential("agent", "myapp_password") is True
    assert pm.can_access_credential("agent", "other_key") is False


# === Blackboard Watcher Tests ===


def test_blackboard_add_watch(blackboard):
    """add_watch registers a glob pattern for an agent."""
    blackboard.add_watch("agent1", "tasks/*")
    watchers = blackboard.get_watchers_for_key("tasks/foo")
    assert "agent1" in watchers


def test_blackboard_watch_no_duplicate(blackboard):
    """add_watch does not add duplicate patterns."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent1", "tasks/*")
    assert len(blackboard._watchers["agent1"]) == 1


def test_blackboard_watch_excludes_writer(blackboard):
    """get_watchers_for_key excludes the agent that wrote the key."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent2", "tasks/*")
    watchers = blackboard.get_watchers_for_key("tasks/foo", exclude="agent1")
    assert "agent1" not in watchers
    assert "agent2" in watchers


def test_blackboard_watch_glob_matching(blackboard):
    """Watchers only match keys that match their glob pattern."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent2", "context/*")
    assert "agent1" in blackboard.get_watchers_for_key("tasks/pending")
    assert "agent1" not in blackboard.get_watchers_for_key("context/data")
    assert "agent2" in blackboard.get_watchers_for_key("context/data")
    assert "agent2" not in blackboard.get_watchers_for_key("tasks/pending")


def test_blackboard_remove_watch_specific(blackboard):
    """remove_watch with a specific pattern removes only that pattern."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent1", "context/*")
    blackboard.remove_watch("agent1", "tasks/*")
    assert "agent1" not in blackboard.get_watchers_for_key("tasks/foo")
    assert "agent1" in blackboard.get_watchers_for_key("context/data")


def test_blackboard_remove_watch_all(blackboard):
    """remove_watch without pattern removes all watches for an agent."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent1", "context/*")
    blackboard.remove_watch("agent1")
    assert "agent1" not in blackboard.get_watchers_for_key("tasks/foo")
    assert "agent1" not in blackboard.get_watchers_for_key("context/data")


def test_blackboard_remove_agent_watches(blackboard):
    """remove_agent_watches cleans up all watches for an agent."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.remove_agent_watches("agent1")
    assert "agent1" not in blackboard._watchers


def test_blackboard_no_watchers(blackboard):
    """get_watchers_for_key returns empty list when no watches match."""
    assert blackboard.get_watchers_for_key("anything") == []


def test_blackboard_watchers_persisted(tmp_path):
    """Watcher registrations survive close/reopen."""
    db_path = str(tmp_path / "bb_persist.db")
    bb = Blackboard(db_path=db_path)
    bb.add_watch("agent1", "tasks/*")
    bb.add_watch("agent2", "context/*")
    bb.close()

    # Re-open — watchers should be loaded from SQLite
    bb2 = Blackboard(db_path=db_path)
    assert "agent1" in bb2.get_watchers_for_key("tasks/foo")
    assert "agent2" in bb2.get_watchers_for_key("context/bar")
    bb2.close()


def test_blackboard_watcher_remove_persisted(tmp_path):
    """Removed watchers are not reloaded on restart."""
    db_path = str(tmp_path / "bb_remove.db")
    bb = Blackboard(db_path=db_path)
    bb.add_watch("agent1", "tasks/*")
    bb.add_watch("agent1", "context/*")
    bb.remove_watch("agent1", "tasks/*")
    bb.close()

    bb2 = Blackboard(db_path=db_path)
    assert "agent1" not in bb2.get_watchers_for_key("tasks/foo")
    assert "agent1" in bb2.get_watchers_for_key("context/bar")
    bb2.close()


def test_blackboard_remove_agent_watches_persisted(tmp_path):
    """remove_agent_watches clears persisted watchers too."""
    db_path = str(tmp_path / "bb_agent_rm.db")
    bb = Blackboard(db_path=db_path)
    bb.add_watch("agent1", "tasks/*")
    bb.remove_agent_watches("agent1")
    bb.close()

    bb2 = Blackboard(db_path=db_path)
    assert "agent1" not in bb2._watchers
    bb2.close()


# === Blackboard Compare-and-Swap Tests ===


def test_blackboard_cas_success(blackboard):
    """write_if_version succeeds when version matches."""
    blackboard.write("tasks/claim_me", {"status": "pending"}, written_by="system")
    result = blackboard.write_if_version(
        "tasks/claim_me", {"status": "claimed", "by": "agent1"},
        written_by="agent1", expected_version=1,
    )
    assert result is not None
    assert result.version == 2
    assert result.value["status"] == "claimed"


def test_blackboard_cas_failure_stale_version(blackboard):
    """write_if_version fails when version is stale."""
    blackboard.write("tasks/claim_me", {"status": "pending"}, written_by="system")
    blackboard.write("tasks/claim_me", {"status": "updated"}, written_by="system")
    # Version is now 2, but we pass expected_version=1
    result = blackboard.write_if_version(
        "tasks/claim_me", {"status": "claimed"},
        written_by="agent1", expected_version=1,
    )
    assert result is None
    # Original value should be unchanged
    entry = blackboard.read("tasks/claim_me")
    assert entry.value["status"] == "updated"
    assert entry.version == 2


def test_blackboard_cas_concurrent_claim_one_wins(blackboard):
    """Only one CAS write succeeds when two agents race."""
    blackboard.write("tasks/race", {"status": "pending"}, written_by="system")
    # Both agents read version 1
    r1 = blackboard.write_if_version(
        "tasks/race", {"status": "claimed", "by": "agent1"},
        written_by="agent1", expected_version=1,
    )
    r2 = blackboard.write_if_version(
        "tasks/race", {"status": "claimed", "by": "agent2"},
        written_by="agent2", expected_version=1,
    )
    # Exactly one should succeed
    assert (r1 is not None) != (r2 is not None)
    entry = blackboard.read("tasks/race")
    assert entry.value["status"] == "claimed"


def test_blackboard_cas_nonexistent_key(blackboard):
    """write_if_version returns None for a key that doesn't exist."""
    result = blackboard.write_if_version(
        "tasks/nope", {"status": "claimed"},
        written_by="agent1", expected_version=1,
    )
    assert result is None


# === Cross-Project Messaging Tests ===


@pytest.mark.asyncio
async def test_router_blocks_cross_project_messaging():
    """Messages between agents in different projects are blocked."""
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(agent_id="alice", can_message=["*"]),
        "bob": AgentPermissions(agent_id="bob", can_message=["*"]),
    }
    router = MessageRouter(
        permissions=perms,
        agent_registry={"alice": "http://a:8400", "bob": "http://b:8400"},
        agent_projects={"alice": "sales", "bob": "engineering"},
    )
    from src.shared.types import AgentMessage
    msg = AgentMessage(from_agent="alice", to="bob", type="query", payload={})
    result = await router.route(msg)
    assert "error" in result
    assert "Cross-project" in result["error"]


@pytest.mark.asyncio
async def test_router_allows_same_project_messaging():
    """Messages between agents in the same project are allowed (not blocked by project check)."""
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "alice": AgentPermissions(agent_id="alice", can_message=["*"]),
        "bob": AgentPermissions(agent_id="bob", can_message=["*"]),
    }
    # Use empty registry so route() returns "No agent found" instead of attempting HTTP
    router = MessageRouter(
        permissions=perms,
        agent_registry={},
        agent_projects={"alice": "sales", "bob": "sales"},
    )
    from src.shared.types import AgentMessage
    msg = AgentMessage(from_agent="alice", to="bob", type="query", payload={})
    result = await router.route(msg)
    # Should fail with "no agent" error, NOT cross-project error
    assert "Cross-project" not in result.get("error", "")
    assert "No agent found" in result.get("error", "")


@pytest.mark.asyncio
async def test_router_allows_standalone_to_project_messaging():
    """Standalone agents can message project agents (no cross-project block)."""
    perms = PermissionMatrix.__new__(PermissionMatrix)
    perms.permissions = {
        "standalone": AgentPermissions(agent_id="standalone", can_message=["*"]),
        "bob": AgentPermissions(agent_id="bob", can_message=["*"]),
    }
    # Use empty registry so route() returns "No agent found" instead of attempting HTTP
    router = MessageRouter(
        permissions=perms,
        agent_registry={},
        agent_projects={"bob": "engineering"},  # standalone is NOT in agent_projects
    )
    from src.shared.types import AgentMessage
    msg = AgentMessage(from_agent="standalone", to="bob", type="query", payload={})
    result = await router.route(msg)
    assert "Cross-project" not in result.get("error", "")
    assert "No agent found" in result.get("error", "")


# === Watcher Thread Safety & Cleanup Tests ===


def test_blackboard_watcher_cleanup_on_deregister(blackboard):
    """remove_agent_watches called after deregister cleans up all watches."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent1", "context/*")
    blackboard.add_watch("agent2", "tasks/*")
    # Simulate deregister
    blackboard.remove_agent_watches("agent1")
    assert "agent1" not in blackboard._watchers
    assert "agent2" in blackboard._watchers
    assert blackboard.get_watchers_for_key("tasks/foo") == ["agent2"]


def test_blackboard_duplicate_watch_ignored(blackboard):
    """Adding the same pattern twice for an agent doesn't duplicate."""
    blackboard.add_watch("agent1", "tasks/*")
    blackboard.add_watch("agent1", "tasks/*")
    assert len(blackboard._watchers["agent1"]) == 1


def test_blackboard_cas_success_with_event_bus(tmp_path):
    """write_if_version emits event_bus event on success."""
    from unittest.mock import MagicMock
    bus = MagicMock()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"), event_bus=bus)
    bb.write("tasks/t1", {"status": "pending"}, written_by="system")
    bus.reset_mock()
    result = bb.write_if_version(
        "tasks/t1", {"status": "claimed"}, written_by="agent1", expected_version=1,
    )
    assert result is not None
    bus.emit.assert_called_once()
    call_kwargs = bus.emit.call_args
    assert call_kwargs[0][0] == "blackboard_write"
    bb.close()


def test_blackboard_cas_failure_no_event_bus_emission(tmp_path):
    """write_if_version does NOT emit event_bus event on CAS failure."""
    from unittest.mock import MagicMock
    bus = MagicMock()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"), event_bus=bus)
    bb.write("tasks/t1", {"status": "pending"}, written_by="system")
    bb.write("tasks/t1", {"status": "updated"}, written_by="system")
    bus.reset_mock()
    result = bb.write_if_version(
        "tasks/t1", {"status": "claimed"}, written_by="agent1", expected_version=1,
    )
    assert result is None
    bus.emit.assert_not_called()
    bb.close()


# === Registration Auto-Watch Tests ===


def test_registration_creates_inbox_watch(tmp_path):
    """Registration auto-watch creates task inbox watch scoped to project."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    agent_id = "testagent"
    project = "testproject"

    # Simulate what /mesh/register does after auto-subscription
    inbox_pattern = f"projects/{project}/tasks/{agent_id}/*"
    bb.add_watch(agent_id, inbox_pattern)

    # Verify the agent is watching its inbox
    watchers = bb.get_watchers_for_key(
        f"projects/{project}/tasks/{agent_id}/some_task"
    )
    assert agent_id in watchers
    bb.close()


def test_registration_inbox_watch_standalone(tmp_path):
    """Registration auto-watch without a project uses unscoped pattern."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    agent_id = "testagent"

    # No project — pattern is unscoped
    inbox_pattern = f"tasks/{agent_id}/*"
    bb.add_watch(agent_id, inbox_pattern)

    watchers = bb.get_watchers_for_key(f"tasks/{agent_id}/ho_abc")
    assert agent_id in watchers

    # Should NOT match another agent's inbox
    watchers_other = bb.get_watchers_for_key("tasks/otheragent/ho_abc")
    assert agent_id not in watchers_other
    bb.close()


def test_inbox_watch_fires_on_handoff_write(tmp_path):
    """Auto-watched agent is returned by get_watchers_for_key on handoff write."""
    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    agent_id = "testagent"
    writer_id = "sender"
    project = "testproject"

    # Register watcher (simulates /mesh/register auto-watch)
    inbox_pattern = f"projects/{project}/tasks/{agent_id}/*"
    bb.add_watch(agent_id, inbox_pattern)

    # Simulate a hand_off write to the agent's inbox
    task_key = f"projects/{project}/tasks/{agent_id}/ho_123"
    bb.write(task_key, {"task": "do something"}, written_by=writer_id)

    # get_watchers_for_key should return the agent (excluding the writer)
    watchers = bb.get_watchers_for_key(task_key, exclude=writer_id)
    assert agent_id in watchers
    assert writer_id not in watchers
    bb.close()


# === /mesh/agents endpoint shape ===


@pytest.mark.asyncio
async def test_list_agents_endpoint_includes_capabilities(tmp_path, monkeypatch):
    """`/mesh/agents` must include a `capabilities` key per agent.

    The list_agents builtin (src/agent/builtins/mesh_tool.py) reads
    info.get("capabilities", []) — without this the dashboard and
    coordination tools see every agent as having zero capabilities.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because this test
    hits the unscoped path with no auth tokens (caller resolves to
    ``"unknown"``); the new ``enforce`` default would filter the
    response down to {operator}.
    """
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    # Register two agents — one with capabilities, one without (the cache
    # only gets populated when capabilities is non-empty, so this exercises
    # the empty-list fallback too).
    router.register_agent("alice", "http://alice:8400", ["search", "summarize"])
    router.register_agent("bob",   "http://bob:8400",   [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/mesh/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"alice", "bob"}
        # Both entries must carry the key (even if empty list).
        assert "capabilities" in data["alice"]
        assert "capabilities" in data["bob"]
        assert sorted(data["alice"]["capabilities"]) == ["search", "summarize"]
        assert data["bob"]["capabilities"] == []
        # Existing fields preserved.
        assert data["alice"]["url"] == "http://alice:8400"
        assert "role" in data["alice"]
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_project_scope_always_includes_operator(tmp_path, monkeypatch):
    """`/mesh/agents?project=X` must include the operator entry alongside
    project members. The operator is fleet-global by design — project agents
    have to discover it to hand off back. Without this, project_agent →
    operator handoff fails at the lookup with 'Agent operator not found'.

    Also verifies the operator entry carries scope=global so coordination
    tools can route the handoff to the global namespace.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because this test
    uses an unauthenticated caller hitting ``?project=growth`` — under
    enforce, the caller is not a project member and the response would
    be empty.
    """
    import importlib
    from unittest.mock import patch

    import yaml
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    # Minimal project on disk with one member.
    projects_dir = tmp_path / "projects"
    proj_dir = projects_dir / "growth"
    proj_dir.mkdir(parents=True)
    (proj_dir / "metadata.yaml").write_text(
        yaml.dump({"name": "growth", "members": ["scout"], "created_at": "2026-05-02T00:00:00+00:00"}),
    )

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("scout", "http://scout:8400", ["recon"])
    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("analyst", "http://analyst:8400", [])  # other project

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get("/mesh/agents", params={"project": "growth"})
        assert resp.status_code == 200
        data = resp.json()
        # Project member is present.
        assert "scout" in data
        # Operator is always visible to project agents.
        assert "operator" in data
        assert data["operator"].get("scope") == "global"
        # Non-member, non-operator agents are still scoped out.
        assert "analyst" not in data
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_unscoped_marks_operator_global(tmp_path, monkeypatch):
    """The fleet-wide /mesh/agents response must also tag the operator with
    scope=global so dashboards and the operator's own list_agents call can
    distinguish it from per-project agents.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because the
    unauthenticated caller resolves to ``"unknown"`` and the unscoped
    ``enforce`` path would filter the response down to {operator}.
    """
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("scout", "http://scout:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/mesh/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["operator"].get("scope") == "global"
        # Non-operator agents do NOT carry the marker.
        assert "scope" not in data["scout"]
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


# ── Task 1 (characterization): /mesh/agents per-caller-type visibility ─


@pytest.mark.asyncio
async def test_list_agents_project_scope_excludes_other_project_members(tmp_path, monkeypatch):
    """When ``/mesh/agents?project=X`` is called, members of project Y must
    NOT appear in the response. The endpoint reads members from the project
    metadata on disk and only includes registered agents whose id is a
    member, plus the operator (which is fleet-global). This pins the
    project-isolation invariant for the response shape.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because the
    unauthenticated caller is not a member of project ``alpha`` — under
    enforce, the response would be empty rather than scoped-to-members.
    """
    import importlib
    from unittest.mock import patch

    import yaml
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    projects_dir = tmp_path / "projects"
    # Project alpha has scout.
    alpha_dir = projects_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    (alpha_dir / "metadata.yaml").write_text(
        yaml.dump({
            "name": "alpha", "members": ["scout"],
            "created_at": "2026-05-02T00:00:00+00:00",
        }),
    )
    # Project beta has analyst.
    beta_dir = projects_dir / "beta"
    beta_dir.mkdir(parents=True)
    (beta_dir / "metadata.yaml").write_text(
        yaml.dump({
            "name": "beta", "members": ["analyst"],
            "created_at": "2026-05-02T00:00:00+00:00",
        }),
    )

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("scout", "http://scout:8400", [])
    router.register_agent("analyst", "http://analyst:8400", [])
    router.register_agent("operator", "http://operator:8400", [])
    # A standalone agent that is in NO project.
    router.register_agent("loner", "http://loner:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get("/mesh/agents", params={"project": "alpha"})
        assert resp.status_code == 200
        data = resp.json()
        # Project member present.
        assert "scout" in data
        # Operator is always visible.
        assert "operator" in data
        # Other project's member is NOT visible.
        assert "analyst" not in data
        # Standalone non-member is NOT visible.
        assert "loner" not in data
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_internal_caller_sees_full_fleet(tmp_path, monkeypatch):
    """Internal callers (loopback / dashboard / mesh-internal) hitting
    ``/mesh/agents`` with no project filter receive every registered
    agent. This pins the unscoped path that dashboards rely on.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` to assert legacy
    fleet-wide behavior under the documented rollback path. The
    sibling ``test_list_agents_internal_caller_unaffected`` covers
    enforce mode.
    """
    import importlib

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    # Mixed fleet across projects + standalone.
    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("scout", "http://scout:8400", [])
    router.register_agent("analyst", "http://analyst:8400", [])
    router.register_agent("loner", "http://loner:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.get("/mesh/agents")
        assert resp.status_code == 200
        data = resp.json()
        # Full fleet visible — every registered agent appears.
        assert set(data.keys()) == {"operator", "scout", "analyst", "loner"}
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_unknown_project_returns_empty(tmp_path):
    """``/mesh/agents?project=X`` for an unknown project name returns an
    empty mapping — the endpoint logs a warning but does not 404. This
    pins the 'silent empty' contract that callers (mesh_client) rely on
    when an agent's project metadata is stale.
    """
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()  # exists but no projects inside

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("scout", "http://scout:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get("/mesh/agents", params={"project": "ghost"})
        assert resp.status_code == 200
        # Empty body (no project members + no operator carve-out — the
        # operator carve-out only fires on an existing project).
        assert resp.json() == {}
    finally:
        bb.close()
        costs.close()
        traces.close()


# ── Task 5: per-caller project scope filter on /mesh/agents ───────────


def _build_scope_test_app(tmp_path, suffix: str = ""):
    """Helper: build a mesh app + projects layout for Task 5 scope tests.

    Lays out two projects (alpha with scout, beta with analyst) plus a
    standalone ``loner`` and an ``operator``. Returns (app, projects_dir,
    cleanup_fns) — caller wraps the projects_dir with the
    ``PROJECTS_DIR`` patch and runs requests through ASGITransport.

    ``suffix`` lets a single test build multiple isolated apps off the
    same ``tmp_path`` (for tests that loop over modes).
    """
    import yaml

    from src.host.costs import CostTracker
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    base = tmp_path / f"scope{suffix}"
    base.mkdir(parents=True, exist_ok=True)
    projects_dir = base / "projects"
    alpha_dir = projects_dir / "alpha"
    alpha_dir.mkdir(parents=True)
    (alpha_dir / "metadata.yaml").write_text(
        yaml.dump({
            "name": "alpha", "members": ["scout"],
            "created_at": "2026-05-02T00:00:00+00:00",
        }),
    )
    beta_dir = projects_dir / "beta"
    beta_dir.mkdir(parents=True)
    (beta_dir / "metadata.yaml").write_text(
        yaml.dump({
            "name": "beta", "members": ["analyst"],
            "created_at": "2026-05-02T00:00:00+00:00",
        }),
    )

    bb = Blackboard(db_path=str(base / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(base / "costs.db"))
    traces = TraceStore(str(base / "traces.db"))

    router.register_agent("operator", "http://operator:8400", [])
    router.register_agent("scout", "http://scout:8400", [])
    router.register_agent("analyst", "http://analyst:8400", [])
    router.register_agent("loner", "http://loner:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    def _cleanup():
        bb.close()
        costs.close()
        traces.close()

    return app, projects_dir, _cleanup


@pytest.mark.asyncio
async def test_list_agents_worker_caller_warn_mode_logs_but_returns_legacy(
    tmp_path, monkeypatch, caplog,
):
    """Task 5 warn mode: a worker calling unscoped ``/mesh/agents`` still
    receives the full fleet (legacy behavior preserved during the soak
    window) but a structured ``scope-warn`` log line is emitted so ops
    can size the impact before flipping to enforce.
    """
    import importlib
    import logging
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)

    app, projects_dir, cleanup = _build_scope_test_app(tmp_path)

    try:
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            with caplog.at_level(logging.WARNING, logger="host.server"):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test",
                ) as client:
                    resp = await client.get(
                        "/mesh/agents",
                        headers={"X-Agent-ID": "scout"},
                    )
        assert resp.status_code == 200
        data = resp.json()
        # Warn mode: full fleet still returned (legacy preserved).
        assert set(data.keys()) == {"operator", "scout", "analyst", "loner"}
        # But a structured scope-warn was emitted.
        warn_lines = [r.message for r in caplog.records if "scope-warn" in r.message]
        assert warn_lines, f"expected a scope-warn log line, got {[r.message for r in caplog.records]}"
        assert any("caller=scout" in m for m in warn_lines)
    finally:
        cleanup()
        # Restore default mode for other tests.
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_worker_caller_enforce_mode_filters(
    tmp_path, monkeypatch,
):
    """Task 5 enforce mode: a worker call returns only members of the
    caller's own projects + the always-global operator. Other projects'
    members and standalone non-members are stripped.
    """
    import importlib
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "enforce")
    import src.host.server as server_module
    importlib.reload(server_module)

    app, projects_dir, cleanup = _build_scope_test_app(tmp_path)

    try:
        with patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/agents",
                    headers={"X-Agent-ID": "scout"},
                )
        assert resp.status_code == 200
        data = resp.json()
        # Enforce mode: scout sees only own-project members + operator.
        assert "scout" in data        # caller itself
        assert "operator" in data     # always-global
        assert "analyst" not in data  # other project
        assert "loner" not in data    # standalone non-member
    finally:
        cleanup()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_operator_caller_unaffected(
    tmp_path, monkeypatch,
):
    """The operator is fleet-global by design. Both warn and enforce
    modes return the full fleet for the operator's unscoped call.
    """
    import importlib
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    for mode in ("warn", "enforce"):
        monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", mode)
        import src.host.server as server_module
        importlib.reload(server_module)

        app, projects_dir, cleanup = _build_scope_test_app(tmp_path, suffix=f"-op-{mode}")

        try:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test",
                ) as client:
                    resp = await client.get(
                        "/mesh/agents",
                        headers={"X-Agent-ID": "operator"},
                    )
            assert resp.status_code == 200, f"mode={mode}"
            data = resp.json()
            assert set(data.keys()) == {"operator", "scout", "analyst", "loner"}, (
                f"mode={mode}: expected full fleet, got {set(data.keys())}"
            )
        finally:
            cleanup()
            monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
            importlib.reload(server_module)


@pytest.mark.asyncio
async def test_list_agents_internal_caller_unaffected(
    tmp_path, monkeypatch,
):
    """Internal callers (loopback + ``x-mesh-internal: 1``) are
    fleet-global like the operator. Both warn and enforce modes return
    the full fleet — dashboards and the CLI manager process rely on
    this for fleet rendering.
    """
    import importlib
    from unittest.mock import patch

    from httpx import ASGITransport, AsyncClient

    for mode in ("warn", "enforce"):
        monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", mode)
        import src.host.server as server_module
        importlib.reload(server_module)

        app, projects_dir, cleanup = _build_scope_test_app(tmp_path, suffix=f"-int-{mode}")

        try:
            with patch("src.cli.config.PROJECTS_DIR", projects_dir):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test",
                ) as client:
                    resp = await client.get(
                        "/mesh/agents",
                        headers={"x-mesh-internal": "1"},
                    )
            assert resp.status_code == 200, f"mode={mode}"
            data = resp.json()
            assert set(data.keys()) == {"operator", "scout", "analyst", "loner"}, (
                f"mode={mode}: expected full fleet, got {set(data.keys())}"
            )
        finally:
            cleanup()
            monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
            importlib.reload(server_module)


# === Task 8: structured routing fields on /mesh/agents and /profile ===


@pytest.mark.asyncio
async def test_list_agents_carries_structured_routing_fields(tmp_path, monkeypatch):
    """`/mesh/agents` entries surface Task-8 routing fields under
    distinct keys from the runtime tool ``capabilities`` list.

    Pinned to ``OPENLEGION_TEAM_SCOPE_MODE=warn`` because the
    unauthenticated caller resolves to ``"unknown"``; the new enforce
    default would filter out non-member agents from the response.
    """
    import importlib
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.traces import TraceStore

    monkeypatch.setenv("OPENLEGION_TEAM_SCOPE_MODE", "warn")
    import src.host.server as server_module
    importlib.reload(server_module)
    create_mesh_app = server_module.create_mesh_app

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "scout": {
            "role": "researcher", "model": "x",
            "capabilities": ["Web research", "Source analysis"],
            "preferred_inputs": ["User questions"],
            "expected_outputs": ["Research reports"],
            "escalation_to": "operator",
            "forbidden": ["Speculation as fact"],
        },
        "plain": {"role": "x", "model": "y"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    projects_dir = cfg_dir / "projects"
    projects_dir.mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router.register_agent("scout", "http://scout:8400", ["browser_navigate"])
    router.register_agent("plain", "http://plain:8400", [])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get("/mesh/agents")
        assert resp.status_code == 200
        data = resp.json()

        scout = data["scout"]
        assert scout["interface_capabilities"] == ["Web research", "Source analysis"]
        assert scout["preferred_inputs"] == ["User questions"]
        assert scout["expected_outputs"] == ["Research reports"]
        assert scout["escalation_to"] == "operator"
        assert scout["forbidden"] == ["Speculation as fact"]
        # Runtime tool capabilities preserved separately.
        assert scout["capabilities"] == ["browser_navigate"]

        plain = data["plain"]
        assert plain["interface_capabilities"] == []
        assert plain["preferred_inputs"] == []
        assert plain["expected_outputs"] == []
        assert plain["escalation_to"] is None
        assert plain["forbidden"] == []
    finally:
        bb.close()
        costs.close()
        traces.close()
        monkeypatch.delenv("OPENLEGION_TEAM_SCOPE_MODE", raising=False)
        importlib.reload(server_module)


@pytest.mark.asyncio
async def test_get_agent_profile_carries_structured_routing_fields(tmp_path):
    """`/mesh/agents/{id}/profile` surfaces Task-8 routing fields."""
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "writer": {
            "role": "writer", "model": "x",
            "capabilities": ["Long-form writing"],
            "preferred_inputs": ["Outlines"],
            "expected_outputs": ["Drafts"],
            "escalation_to": "editor",
            "forbidden": ["Plagiarism"],
        },
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    projects_dir = cfg_dir / "projects"
    projects_dir.mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router.register_agent("writer", "http://writer:8400", ["file_write"])

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/agents/writer/profile",
                    headers={"x-mesh-internal": "1"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "writer"
        assert data["interface_capabilities"] == ["Long-form writing"]
        assert data["preferred_inputs"] == ["Outlines"]
        assert data["expected_outputs"] == ["Drafts"]
        assert data["escalation_to"] == "editor"
        assert data["forbidden"] == ["Plagiarism"]
        assert data["capabilities"] == ["file_write"]
    finally:
        bb.close()
        costs.close()
        traces.close()


@pytest.mark.asyncio
async def test_get_agent_profile_carries_runtime_debug_fields(tmp_path):
    """`/mesh/agents/{id}/profile` surfaces last_heartbeat_at + spend totals."""
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.cron import CronScheduler
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "writer": {"role": "writer", "model": "x"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    projects_dir = cfg_dir / "projects"
    projects_dir.mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    cron = CronScheduler(config_path=str(tmp_path / "cron.json"))
    router.register_agent("writer", "http://writer:8400", ["file_write"])

    # Seed cost tracker with a known spend for "writer".
    costs.track_fixed_cost(agent="writer", model="gpt-4o", cost_usd=0.42)

    # Seed cron scheduler with a heartbeat job and force last_run.
    job = cron.add_job(agent="writer", schedule="every 15m", heartbeat=True)
    job.last_run = "2026-05-15T10:00:00+00:00"

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, cron_scheduler=cron,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/agents/writer/profile",
                    headers={"x-mesh-internal": "1"},
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_heartbeat_at"] == "2026-05-15T10:00:00+00:00"
        assert data["spend_today_usd"] == pytest.approx(0.42)
        assert data["spend_month_usd"] == pytest.approx(0.42)
    finally:
        bb.close()
        costs.close()
        traces.close()


@pytest.mark.asyncio
async def test_get_agent_profile_hides_runtime_fields_from_peer_agents(tmp_path):
    """Peer agents (not operator, not internal) must NOT see runtime debug
    fields on /profile — they're operator-or-internal only because they
    leak operational state across the fleet and add SQL load to the
    routing hot path."""
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.cron import CronScheduler
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "writer": {"role": "writer", "model": "x"},
        "peer": {"role": "peer", "model": "x"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    projects_dir = cfg_dir / "projects"
    projects_dir.mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    # Grant peer→writer messaging so the /profile permission check passes —
    # we want to test the runtime-field gate AFTER the message gate, not
    # short-circuit at the message gate.
    perms.permissions["peer"] = AgentPermissions(
        agent_id="peer", can_message=["writer"],
    )
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    cron = CronScheduler(config_path=str(tmp_path / "cron.json"))
    router.register_agent("writer", "http://writer:8400", ["file_write"])
    router.register_agent("peer", "http://peer:8400", [])

    costs.track_fixed_cost(agent="writer", model="gpt-4o", cost_usd=0.42)
    job = cron.add_job(agent="writer", schedule="every 15m", heartbeat=True)
    job.last_run = "2026-05-15T10:00:00+00:00"

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, cron_scheduler=cron,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                resp = await client.get(
                    "/mesh/agents/writer/profile",
                    # Peer-agent identity; NO x-mesh-internal header.
                    params={"requesting_agent": "peer"},
                )
        assert resp.status_code == 200
        data = resp.json()
        # Routing fields still present.
        assert data["agent_id"] == "writer"
        assert "capabilities" in data
        # Runtime debug fields must be ABSENT for non-operator callers.
        assert "last_heartbeat_at" not in data
        assert "spend_today_usd" not in data
        assert "spend_month_usd" not in data
    finally:
        bb.close()
        costs.close()
        traces.close()


@pytest.mark.asyncio
async def test_profile_denied_to_worker_omitting_requesting_agent(tmp_path):
    """M24: with auth tokens configured, a worker that omits the
    ``requesting_agent`` hint AND lacks ``can_message`` to the target is
    denied the profile (the verified Bearer identity drives the gate, so
    the optional hint can't be used to bypass it). An authorized peer and
    the operator still get 200."""
    from unittest.mock import patch

    import yaml as yaml_mod
    from httpx import ASGITransport, AsyncClient

    from src.host.costs import CostTracker
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True)
    agents_path = cfg_dir / "agents.yaml"
    agents_path.write_text(yaml_mod.dump({"agents": {
        "writer": {"role": "writer", "model": "x"},
        "peer": {"role": "peer", "model": "x"},
        "stranger": {"role": "stranger", "model": "x"},
    }}))
    (cfg_dir / "mesh.yaml").write_text(yaml_mod.dump({"mesh": {"port": 8420}}))
    projects_dir = cfg_dir / "projects"
    projects_dir.mkdir()

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    # peer may message writer; stranger may not message anyone.
    perms.permissions["peer"] = AgentPermissions(
        agent_id="peer", can_message=["writer"],
    )
    perms.permissions["stranger"] = AgentPermissions(
        agent_id="stranger", can_message=[],
    )
    router = MessageRouter(perms, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))
    router.register_agent("writer", "http://writer:8400", ["file_write"])
    router.register_agent("peer", "http://peer:8400", [])
    router.register_agent("stranger", "http://stranger:8400", [])

    auth_tokens = {
        "operator": "tok-operator",
        "peer": "tok-peer",
        "stranger": "tok-stranger",
    }

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        cost_tracker=costs, trace_store=traces, auth_tokens=auth_tokens,
    )

    try:
        with patch("src.cli.config.AGENTS_FILE", agents_path), \
             patch("src.cli.config.CONFIG_FILE", cfg_dir / "mesh.yaml"), \
             patch("src.cli.config.PROJECTS_DIR", projects_dir):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test",
            ) as client:
                # Stranger omits requesting_agent entirely — previously this
                # fell through to _require_any_auth and leaked the profile.
                # Now the Bearer-verified identity is gated → 403.
                denied = await client.get(
                    "/mesh/agents/writer/profile",
                    headers={"Authorization": "Bearer tok-stranger"},
                )
                assert denied.status_code == 403, denied.text

                # Authorized peer (can_message=[writer]) still gets the profile,
                # even though it also omits requesting_agent.
                peer_ok = await client.get(
                    "/mesh/agents/writer/profile",
                    headers={"Authorization": "Bearer tok-peer"},
                )
                assert peer_ok.status_code == 200, peer_ok.text
                assert peer_ok.json()["agent_id"] == "writer"

                # Operator (verified Bearer identity == "operator") always gets it.
                op_ok = await client.get(
                    "/mesh/agents/writer/profile",
                    headers={"Authorization": "Bearer tok-operator"},
                )
                assert op_ok.status_code == 200, op_ok.text
    finally:
        bb.close()
        costs.close()
        traces.close()


# ── llm_call telemetry cost: OAuth (subscription) must report $0 ──────


def _build_proxy_cost_app(tmp_path, vault):
    """Build a minimal mesh app for exercising the /mesh/api cost telemetry.

    ``vault`` is the (mocked) credential vault whose ``execute_api_call``
    return value drives what the ``llm_call`` event reports. Returns
    ``(app, event_bus, captured, cleanup)`` where ``captured`` accumulates
    every emitted ``llm_call`` event dict (EventBus listeners run
    synchronously inside emit()).
    """
    from src.dashboard.events import EventBus
    from src.host.server import create_mesh_app

    # The /mesh/api proxy runs _enforce_model_pin → is_model_compatible
    # (the model-pin gate) before dispatch; stub the documented
    # (compatible, reason) contract so the pin doesn't trip on the mock.
    vault.is_model_compatible.return_value = (True, None)

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    perms.permissions["scout"] = AgentPermissions(
        agent_id="scout", allowed_apis=["llm"],
    )
    router = MessageRouter(perms, {})
    router.register_agent("scout", "http://scout:8400", [])

    event_bus = EventBus()
    captured: list[dict] = []
    event_bus.add_listener(
        lambda e: captured.append(e) if e.get("type") == "llm_call" else None,
    )

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        credential_vault=vault, event_bus=event_bus,
    )
    return app, event_bus, captured, bb.close


async def _post_chat(app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        return await client.post(
            "/mesh/api",
            params={"agent_id": "scout"},
            json={
                "service": "llm", "action": "chat",
                "params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            },
        )


@pytest.mark.asyncio
async def test_proxy_api_oauth_call_reports_zero_cost(tmp_path):
    """OAuth (subscription) calls must emit an llm_call event with cost_usd=0.

    Pins the dashboard activity/trace-feed contract: the authoritative
    usage table already skips OAuth, and this telemetry event must agree.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.shared.types import APIProxyResponse

    vault = MagicMock()
    vault.execute_api_call = AsyncMock(return_value=APIProxyResponse(
        success=True,
        data={
            "oauth": True, "content": "hi", "tokens_used": 1000,
            "input_tokens": 700, "output_tokens": 300,
            "model": "anthropic/claude-sonnet-4-6",
        },
    ))

    app, _bus, captured, cleanup = _build_proxy_cost_app(tmp_path, vault)
    try:
        resp = await _post_chat(app)
        assert resp.status_code == 200, resp.text
        assert len(captured) == 1
        data = captured[0]["data"]
        assert data["oauth"] is True
        assert data["cost_usd"] == 0.0
        # Tokens are still reported — only the dollar cost is zeroed.
        assert data["total_tokens"] == 1000
    finally:
        cleanup()


@pytest.mark.asyncio
async def test_proxy_api_metered_call_still_reports_cost(tmp_path):
    """Regression guard (other direction): non-OAuth calls keep a real cost.

    Ensures the OAuth zeroing didn't blanket-zero metered API-key traffic.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.shared.types import APIProxyResponse

    vault = MagicMock()
    vault.execute_api_call = AsyncMock(return_value=APIProxyResponse(
        success=True,
        data={
            "content": "hi", "tokens_used": 1000,
            "input_tokens": 700, "output_tokens": 300,
            "model": "anthropic/claude-sonnet-4-6",
        },
    ))

    app, _bus, captured, cleanup = _build_proxy_cost_app(tmp_path, vault)
    try:
        resp = await _post_chat(app)
        assert resp.status_code == 200, resp.text
        assert len(captured) == 1
        data = captured[0]["data"]
        assert not data.get("oauth")
        assert data["cost_usd"] > 0.0
    finally:
        cleanup()


# ── Session observability (Phase 1): X-Trace-Id seed → stamped usage row ──
#
# Store-level tests (test_costs.py::TestTraceIdStamping) already verify
# ``CostTracker.track`` stamps ``trace_id`` when the contextvar is pre-set.
# These close the end-to-end gap: the proxy endpoint must seed
# ``current_trace_id`` from the inbound ``X-Trace-Id`` header so the cost
# write fired *inside* ``execute_api_call`` / ``stream_llm`` — which run as
# the same request task (non-stream) or a later-iterated streaming context
# (stream) — observes it. If the seed (or, for streaming, the context copy
# into the generator) regresses, the usage row silently gets a NULL
# trace_id and the session can no longer be JOINed across stores.


def _build_proxy_trace_app(tmp_path, *, streaming: bool):
    """Build a mesh app wired to a REAL CostTracker behind a fake vault.

    The fake vault mirrors production: its ``execute_api_call`` /
    ``stream_llm`` call ``cost_tracker.track`` (which reads the trace
    contextvar internally), so the row written reflects whatever trace the
    endpoint seeded. Returns ``(app, tracker, cleanup)``; read back the
    stamped trace via ``_usage_trace_ids(tracker)``.
    """
    from src.host.costs import CostTracker
    from src.host.server import create_mesh_app
    from src.shared.types import APIProxyResponse

    tracker = CostTracker(
        db_path=str(tmp_path / "costs.db"),
        budgets_path=str(tmp_path / "budgets.json"),
    )

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    pubsub = PubSub()
    perms = PermissionMatrix()
    perms.permissions["scout"] = AgentPermissions(
        agent_id="scout", allowed_apis=["llm"],
    )
    router = MessageRouter(perms, {})
    router.register_agent("scout", "http://scout:8400", [])

    from unittest.mock import MagicMock

    vault = MagicMock()
    vault.is_model_compatible.return_value = (True, None)

    if streaming:
        async def _fake_stream(api_request, agent_id=""):
            # Mirror stream_llm: emit content chunks, then track() on the
            # terminal 'done' frame. track() reads current_trace_id, which
            # the endpoint must have seeded into this generator's context.
            yield 'data: {"type": "delta", "content": "hi"}\n\n'
            tracker.track(agent_id, "anthropic/claude-sonnet-4-6", 700, 300)
            yield (
                'data: {"type": "done", "model": "anthropic/claude-sonnet-4-6", '
                '"tokens_used": 1000, "content": "hi"}\n\n'
            )
        vault.stream_llm = _fake_stream
    else:
        async def _fake_call(api_request, agent_id=""):
            # Mirror execute_api_call's cost write: track() reads the
            # contextvar the endpoint seeded from X-Trace-Id.
            tracker.track(agent_id, "anthropic/claude-sonnet-4-6", 700, 300)
            return APIProxyResponse(
                success=True,
                data={
                    "content": "hi", "tokens_used": 1000,
                    "input_tokens": 700, "output_tokens": 300,
                    "model": "anthropic/claude-sonnet-4-6",
                },
            )
        vault.execute_api_call = _fake_call

    app = create_mesh_app(
        blackboard=bb, pubsub=pubsub, router=router, permissions=perms,
        credential_vault=vault, cost_tracker=tracker,
    )

    def _cleanup():
        tracker.close()
        bb.close()

    return app, tracker, _cleanup


def _usage_trace_ids(tracker) -> list:
    return [
        r[0] for r in tracker.db.execute(
            "SELECT trace_id FROM usage ORDER BY id"
        ).fetchall()
    ]


async def _post_proxy(app, *, path: str, trace_id: str | None):
    from httpx import ASGITransport, AsyncClient

    headers = {} if trace_id is None else {"X-Trace-Id": trace_id}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        return await client.post(
            path,
            params={"agent_id": "scout"},
            json={
                "service": "llm", "action": "chat",
                "params": {
                    "model": "anthropic/claude-sonnet-4-6",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            },
            headers=headers,
        )


@pytest.mark.asyncio
async def test_proxy_api_call_stamps_usage_trace_id_from_header(tmp_path):
    """``POST /mesh/api`` with ``X-Trace-Id`` → the usage row written by the
    cost tracker inside ``execute_api_call`` carries that trace_id."""
    from src.shared.trace import current_trace_id

    app, tracker, cleanup = _build_proxy_trace_app(tmp_path, streaming=False)
    token = current_trace_id.set(None)
    try:
        resp = await _post_proxy(app, path="/mesh/api", trace_id="tr_proxy0000001")
        assert resp.status_code == 200, resp.text
        assert _usage_trace_ids(tracker) == ["tr_proxy0000001"]
    finally:
        current_trace_id.reset(token)
        cleanup()


@pytest.mark.asyncio
async def test_proxy_api_call_without_trace_id_usage_is_null(tmp_path):
    """Negative guard: no ``X-Trace-Id`` → the usage row's trace_id is NULL."""
    from src.shared.trace import current_trace_id

    app, tracker, cleanup = _build_proxy_trace_app(tmp_path, streaming=False)
    token = current_trace_id.set(None)
    try:
        resp = await _post_proxy(app, path="/mesh/api", trace_id=None)
        assert resp.status_code == 200, resp.text
        assert _usage_trace_ids(tracker) == [None]
    finally:
        current_trace_id.reset(token)
        cleanup()


@pytest.mark.asyncio
async def test_proxy_api_stream_stamps_usage_trace_id_from_header(tmp_path):
    """FRAGILE PATH: ``POST /mesh/api/stream`` seeds the contextvar before
    the SSE generator is created; the cost write fired inside the
    later-iterated ``stream_llm`` generator must still see the seeded
    trace. Guards the contextvar-survives-into-the-streaming-context
    invariant the endpoint comment promises."""
    from src.shared.trace import current_trace_id

    app, tracker, cleanup = _build_proxy_trace_app(tmp_path, streaming=True)
    token = current_trace_id.set(None)
    try:
        resp = await _post_proxy(
            app, path="/mesh/api/stream", trace_id="tr_stream0000001",
        )
        assert resp.status_code == 200, resp.text
        # Drain the SSE body so the generator (and its track() call) runs.
        body = resp.text
        assert '"type": "done"' in body
        assert _usage_trace_ids(tracker) == ["tr_stream0000001"]
    finally:
        current_trace_id.reset(token)
        cleanup()
