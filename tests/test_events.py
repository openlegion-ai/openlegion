"""Tests for the dashboard event bus: DashboardEvent model, EventBus, and integrations."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.dashboard.events import BUFFER_SIZE, EventBus, _Subscription
from src.shared.types import DashboardEvent


# === DashboardEvent model tests ===


def test_dashboard_event_model():
    """DashboardEvent creation with defaults and all 8 valid types."""
    valid_types = [
        "agent_state", "message_sent", "message_received",
        "tool_start", "tool_result", "llm_call",
        "blackboard_write", "health_change",
    ]
    for t in valid_types:
        evt = DashboardEvent(type=t, agent="test", data={"key": "val"})
        assert evt.type == t
        assert evt.agent == "test"
        assert evt.id.startswith("evt_")
        assert evt.data == {"key": "val"}


def test_dashboard_event_invalid_type():
    """Invalid type is rejected by Pydantic validation."""
    with pytest.raises(Exception):
        DashboardEvent(type="invalid_type")


def test_dashboard_event_serialization():
    """DashboardEvent serializes to JSON-compatible dict."""
    evt = DashboardEvent(type="llm_call", agent="a1", data={"model": "gpt-4o"})
    d = evt.model_dump(mode="json")
    assert d["type"] == "llm_call"
    assert d["agent"] == "a1"
    assert d["data"]["model"] == "gpt-4o"
    assert isinstance(d["timestamp"], str)
    # Round-trip through JSON
    json.dumps(d)


def test_dashboard_event_defaults():
    """DashboardEvent has sensible defaults for agent and data."""
    evt = DashboardEvent(type="llm_call")
    assert evt.agent == ""
    assert evt.data == {}
    assert evt.timestamp is not None


# === EventBus core tests ===


def test_emit_stores_in_buffer():
    bus = EventBus()
    bus.emit("llm_call", agent="a1", data={"model": "gpt-4o"})
    assert len(bus._buffer) == 1
    assert bus._buffer[0]["type"] == "llm_call"
    assert bus._buffer[0]["agent"] == "a1"


def test_ring_buffer_eviction():
    bus = EventBus()
    for i in range(BUFFER_SIZE + 50):
        bus.emit("llm_call", agent=f"a{i}")
    assert len(bus._buffer) == BUFFER_SIZE
    # Oldest events should be evicted; newest should be present
    assert bus._buffer[-1]["agent"] == f"a{BUFFER_SIZE + 49}"
    assert bus._buffer[0]["agent"] == f"a{50}"


def test_recent_events_unfiltered():
    bus = EventBus()
    bus.emit("llm_call", agent="a1")
    bus.emit("blackboard_write", agent="a2")
    bus.emit("health_change", agent="a1")
    events = bus.recent_events()
    assert len(events) == 3


def test_recent_events_filter_by_type():
    bus = EventBus()
    bus.emit("llm_call", agent="a1")
    bus.emit("blackboard_write", agent="a2")
    bus.emit("llm_call", agent="a3")
    events = bus.recent_events(types_filter={"llm_call"})
    assert len(events) == 2
    assert all(e["type"] == "llm_call" for e in events)


def test_recent_events_filter_by_agent():
    bus = EventBus()
    bus.emit("llm_call", agent="a1")
    bus.emit("blackboard_write", agent="a2")
    bus.emit("health_change", agent="a1")
    events = bus.recent_events(agents_filter={"a1"})
    assert len(events) == 2
    assert all(e["agent"] == "a1" for e in events)


def test_recent_events_combined_filters():
    """Both agent and type filters applied together."""
    bus = EventBus()
    bus.emit("llm_call", agent="a1")
    bus.emit("llm_call", agent="a2")
    bus.emit("blackboard_write", agent="a1")
    events = bus.recent_events(agents_filter={"a1"}, types_filter={"llm_call"})
    assert len(events) == 1
    assert events[0]["agent"] == "a1"
    assert events[0]["type"] == "llm_call"


def test_set_loop_idempotent():
    """set_loop can be called multiple times without issues."""
    bus = EventBus()
    loop = asyncio.new_event_loop()
    bus.set_loop(loop)
    bus.set_loop(loop)
    assert bus._loop is loop
    loop.close()


# === Subscription tests ===


def test_subscribe_unsubscribe():
    bus = EventBus()
    ws = MagicMock()
    bus.subscribe(ws)
    assert len(bus._clients) == 1
    bus.unsubscribe(ws)
    assert len(bus._clients) == 0


def test_subscription_matches_all():
    """No filters means match everything."""
    sub = _Subscription(ws=MagicMock())
    assert sub.matches({"type": "llm_call", "agent": "a1"})
    assert sub.matches({"type": "blackboard_write", "agent": ""})


def test_subscription_type_filter():
    sub = _Subscription(ws=MagicMock(), types={"llm_call", "blackboard_write"})
    assert sub.matches({"type": "llm_call", "agent": "a1"})
    assert sub.matches({"type": "blackboard_write", "agent": "a2"})
    assert not sub.matches({"type": "health_change", "agent": "a1"})


def test_subscription_agent_filter():
    sub = _Subscription(ws=MagicMock(), agents={"a1"})
    assert sub.matches({"type": "llm_call", "agent": "a1"})
    assert not sub.matches({"type": "llm_call", "agent": "a2"})
    # Empty agent always passes agent filter
    assert sub.matches({"type": "llm_call", "agent": ""})


def test_subscription_combined_filter():
    sub = _Subscription(ws=MagicMock(), agents={"a1"}, types={"llm_call"})
    assert sub.matches({"type": "llm_call", "agent": "a1"})
    assert not sub.matches({"type": "blackboard_write", "agent": "a1"})
    assert not sub.matches({"type": "llm_call", "agent": "a2"})


# === Broadcast tests ===


@pytest.mark.asyncio
async def test_broadcast_sends_to_matching_clients():
    """Broadcast delivers to matching subscribers and skips non-matching."""
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)

    ws_all = AsyncMock()
    ws_filtered = AsyncMock()

    bus.subscribe(ws_all)
    bus.subscribe(ws_filtered, types_filter={"blackboard_write"})

    # Emit llm_call — should reach ws_all but not ws_filtered
    bus.emit("llm_call", agent="a1", data={"model": "gpt-4o"})
    await asyncio.sleep(0.05)  # let broadcast task run

    ws_all.send_text.assert_called_once()
    ws_filtered.send_text.assert_not_called()

    # Emit blackboard_write — should reach both
    ws_all.send_text.reset_mock()
    bus.emit("blackboard_write", agent="a1", data={"key": "test/k"})
    await asyncio.sleep(0.05)

    ws_all.send_text.assert_called_once()
    ws_filtered.send_text.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connections():
    """Dead WebSocket connections are cleaned up on broadcast."""
    bus = EventBus()
    loop = asyncio.get_running_loop()
    bus.set_loop(loop)

    ws_dead = AsyncMock()
    ws_dead.send_text.side_effect = ConnectionError("gone")
    ws_alive = AsyncMock()

    bus.subscribe(ws_dead)
    bus.subscribe(ws_alive)
    assert len(bus._clients) == 2

    bus.emit("llm_call", agent="a1")
    await asyncio.sleep(0.05)

    # Dead connection should be removed
    assert len(bus._clients) == 1
    ws_alive.send_text.assert_called_once()


# === Integration: Blackboard emits on write ===


def test_blackboard_emits_on_write(tmp_path):
    from src.host.mesh import Blackboard

    bus = EventBus()
    bb = Blackboard(db_path=str(tmp_path / "bb.db"), event_bus=bus)
    bb.write("test/key", {"val": 1}, written_by="agent1")
    bb.close()

    assert len(bus._buffer) == 1
    evt = bus._buffer[0]
    assert evt["type"] == "blackboard_write"
    assert evt["agent"] == "agent1"
    assert evt["data"]["key"] == "test/key"
    assert evt["data"]["version"] == 1
    assert "value_preview" in evt["data"]


def test_blackboard_no_event_bus(tmp_path):
    """Blackboard works normally when event_bus is None."""
    from src.host.mesh import Blackboard

    bb = Blackboard(db_path=str(tmp_path / "bb.db"))
    entry = bb.write("test/key", {"val": 1}, written_by="agent1")
    assert entry.version == 1
    bb.close()


# === Integration: CostTracker no longer emits ===


def test_cost_tracker_no_event_on_track(tmp_path):
    """CostTracker.track() no longer emits cost_update (merged into llm_call in server.py)."""
    from src.host.costs import CostTracker

    bus = EventBus()
    ct = CostTracker(db_path=str(tmp_path / "costs.db"), event_bus=bus)
    ct.track("agent1", "openai/gpt-4o", prompt_tokens=100, completion_tokens=50)
    ct.close()

    assert len(bus._buffer) == 0


# === Integration: HealthMonitor emits on state change ===


@pytest.mark.asyncio
async def test_health_monitor_emits_on_healthy(tmp_path):
    """HealthMonitor emits health_change when agent transitions to healthy."""
    from src.host.health import HealthMonitor

    bus = EventBus()
    runtime = MagicMock()
    transport = AsyncMock()
    transport.is_reachable = AsyncMock(return_value=True)
    router = MagicMock()

    monitor = HealthMonitor(runtime=runtime, transport=transport,
                            router=router, event_bus=bus)
    monitor.register("agent1")

    # Status starts as "unknown" — first healthy check should emit
    await monitor._check_agent("agent1")

    assert len(bus._buffer) == 1
    evt = bus._buffer[0]
    assert evt["type"] == "health_change"
    assert evt["agent"] == "agent1"
    assert evt["data"]["previous"] == "unknown"
    assert evt["data"]["current"] == "healthy"
    assert evt["data"]["failures"] == 0


@pytest.mark.asyncio
async def test_health_monitor_emits_on_unhealthy(tmp_path):
    """HealthMonitor emits health_change on transition to unhealthy."""
    from src.host.health import HealthMonitor

    bus = EventBus()
    runtime = MagicMock()
    transport = AsyncMock()
    transport.is_reachable = AsyncMock(return_value=False)
    router = MagicMock()

    monitor = HealthMonitor(runtime=runtime, transport=transport,
                            router=router, event_bus=bus)
    monitor.register("agent1")

    await monitor._check_agent("agent1")

    assert len(bus._buffer) == 1
    evt = bus._buffer[0]
    assert evt["type"] == "health_change"
    assert evt["agent"] == "agent1"
    assert evt["data"]["previous"] == "unknown"
    assert evt["data"]["current"] == "unhealthy"
    assert evt["data"]["failures"] == 1


@pytest.mark.asyncio
async def test_health_monitor_no_emit_on_stable_healthy():
    """HealthMonitor does NOT emit when status stays healthy (no transition)."""
    from src.host.health import HealthMonitor

    bus = EventBus()
    runtime = MagicMock()
    transport = AsyncMock()
    transport.is_reachable = AsyncMock(return_value=True)
    router = MagicMock()

    monitor = HealthMonitor(runtime=runtime, transport=transport,
                            router=router, event_bus=bus)
    monitor.register("agent1")

    # First check: unknown → healthy (emits)
    await monitor._check_agent("agent1")
    assert len(bus._buffer) == 1

    # Second check: healthy → healthy (no emit)
    await monitor._check_agent("agent1")
    assert len(bus._buffer) == 1  # still 1, no new event


@pytest.mark.asyncio
async def test_health_monitor_emits_on_restart_failure():
    """HealthMonitor emits health_change when agent exceeds restart limit."""
    from src.host.health import HealthMonitor

    bus = EventBus()
    runtime = MagicMock()
    runtime.agents = {}
    transport = AsyncMock()
    transport.is_reachable = AsyncMock(return_value=False)
    router = MagicMock()

    monitor = HealthMonitor(runtime=runtime, transport=transport,
                            router=router, event_bus=bus)
    monitor.register("agent1")

    # Exceed restart limit
    import time
    health = monitor.agents["agent1"]
    health.restart_timestamps = [time.time()] * monitor.RESTART_LIMIT

    # Force max failures to trigger restart attempt
    health.consecutive_failures = monitor.MAX_FAILURES - 1
    health.status = "unhealthy"

    await monitor._check_agent("agent1")

    # Should have emitted health_change with "failed" status
    failed_events = [e for e in bus._buffer if e["data"].get("current") == "failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["agent"] == "agent1"


# === WebSocket endpoint tests ===


@pytest.mark.asyncio
async def test_websocket_replay():
    """WebSocket receives recent events on connect as replay."""
    from starlette.testclient import TestClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    bus = EventBus()
    bus.emit("llm_call", agent="a1", data={"model": "gpt-4o"})
    bus.emit("blackboard_write", agent="a2", data={"key": "test/k"})

    bb = Blackboard(db_path=":memory:")
    ps = PubSub()
    pm = PermissionMatrix()
    router = MessageRouter(pm, {})
    app = create_mesh_app(bb, ps, router, pm, event_bus=bus)

    client = TestClient(app)
    with client.websocket_connect("/ws/events") as ws:
        # Should receive 2 replayed events
        msg1 = json.loads(ws.receive_text())
        assert msg1["type"] == "llm_call"
        msg2 = json.loads(ws.receive_text())
        assert msg2["type"] == "blackboard_write"

    bb.close()


@pytest.mark.asyncio
async def test_websocket_filters():
    """WebSocket query param filtering works."""
    from starlette.testclient import TestClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    bus = EventBus()
    bus.emit("llm_call", agent="a1")
    bus.emit("blackboard_write", agent="a2")
    bus.emit("llm_call", agent="a2")

    bb = Blackboard(db_path=":memory:")
    ps = PubSub()
    pm = PermissionMatrix()
    router = MessageRouter(pm, {})
    app = create_mesh_app(bb, ps, router, pm, event_bus=bus)

    client = TestClient(app)
    with client.websocket_connect("/ws/events?types=llm_call&agents=a1") as ws:
        # Only the first event should match (llm_call + a1)
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "llm_call"
        assert msg["agent"] == "a1"

    bb.close()


@pytest.mark.asyncio
async def test_websocket_no_bus():
    """WebSocket closes cleanly when event_bus is None."""
    from starlette.testclient import TestClient

    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.permissions import PermissionMatrix
    from src.host.server import create_mesh_app

    bb = Blackboard(db_path=":memory:")
    ps = PubSub()
    pm = PermissionMatrix()
    router = MessageRouter(pm, {})
    app = create_mesh_app(bb, ps, router, pm, event_bus=None)

    client = TestClient(app)
    # Connection should be closed by server when no bus
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/events") as ws:
            ws.receive_text()

    bb.close()
