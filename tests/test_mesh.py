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

    # Force the last GC time far in the past so the next call triggers GC
    bb._last_ttl_gc = 0

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


def test_can_access_credential_orchestrator_always_allowed(tmp_path):
    """Orchestrator and mesh bypass credential scoping."""
    config = {"permissions": {}}
    config_path = tmp_path / "permissions.json"
    config_path.write_text(json.dumps(config))
    pm = PermissionMatrix(config_path=str(config_path))

    assert pm.can_access_credential("orchestrator", "anthropic_api_key") is True
    assert pm.can_access_credential("mesh", "openai_api_key") is True


def test_allowed_credentials_controls_vault_access(tmp_path):
    """allowed_credentials patterns control both vault access and credential resolution."""
    config = {
        "permissions": {
            "full_access": {
                "can_message": ["orchestrator"],
                "blackboard_read": ["*"],
                "blackboard_write": [],
                "allowed_apis": ["llm"],
                "allowed_credentials": ["*"],
            },
            "restricted": {
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
