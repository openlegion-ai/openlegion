"""Unit tests for Grouped Tool Search (B2).

Covers:
- the budget gate (small toolset → unchanged; large → index+defer),
- the capability index renders ALL grouped capabilities (names never hidden),
- ``load_tools`` defers the schema change to the NEXT build (not mid-turn),
- the loaded-groups state folds into the ``get_tool_definitions`` cache key,
- flag-off → identical behaviour to today.

The whole feature is default-OFF behind ``OPENLEGION_GROUPED_TOOLS``, so each
test that exercises the active path sets the env flag explicitly.
"""

import pytest

from src.agent import tool_groups
from src.agent.tool_groups import (
    TOOL_GROUPS,
    grouped_tools_enabled,
    plan_grouped_tools,
    resolve_load_request,
)
from src.agent.tools import ToolRegistry, _tool_staging, tool


def setup_function():
    _tool_staging.clear()


@pytest.fixture
def grouped_on(monkeypatch):
    monkeypatch.setenv(tool_groups.GROUPED_TOOLS_ENV, "1")
    yield


# ── Flag plumbing ───────────────────────────────────────────────────────────
def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv(tool_groups.GROUPED_TOOLS_ENV, raising=False)
    assert grouped_tools_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv(tool_groups.GROUPED_TOOLS_ENV, "true")
    assert grouped_tools_enabled() is True


def test_plan_inactive_when_flag_off(monkeypatch):
    monkeypatch.delenv(tool_groups.GROUPED_TOOLS_ENV, raising=False)
    available = {t for g in TOOL_GROUPS for t in g.tools}
    plan = plan_grouped_tools(
        available=available,
        loaded_groups=set(),
        operator=True,
        context_window=1000,  # tiny window → would trip the gate if flag were on
    )
    assert plan.active is False
    assert plan.defer == frozenset()
    assert plan.index_text == ""


# ── Budget gate ─────────────────────────────────────────────────────────────
def test_budget_gate_small_toolset_unchanged(grouped_on):
    """A small grouped surface stays under the budget → plan inactive."""
    available = {"create_agent", "set_cron"}  # only 2 grouped tools
    plan = plan_grouped_tools(
        available=available,
        loaded_groups=set(),
        operator=True,
        context_window=200_000,  # opus-class window — 2 tools is far under 10%
    )
    assert plan.active is False
    assert plan.defer == frozenset()


def test_budget_gate_large_toolset_activates(grouped_on):
    """A large grouped surface exceeding ~10% of the window → index+defer."""
    available = {t for g in TOOL_GROUPS for t in g.tools}
    # Pick a window small enough that the grouped tools exceed 10%.
    est = len(available) * tool_groups.SCHEMA_TOKENS_PER_TOOL
    window = int(est / tool_groups.BUDGET_FRACTION) - 1  # just under the gate's denom
    plan = plan_grouped_tools(
        available=available,
        loaded_groups=set(),
        operator=True,
        context_window=window,
    )
    assert plan.active is True
    assert plan.defer  # something is deferred
    assert plan.index_text


def test_budget_gate_zero_window_inactive(grouped_on):
    available = {t for g in TOOL_GROUPS for t in g.tools}
    plan = plan_grouped_tools(
        available=available, loaded_groups=set(), operator=True, context_window=0,
    )
    assert plan.active is False


# ── Capability index renders ALL capabilities (names never hidden) ──────────
def test_index_lists_every_available_grouped_tool(grouped_on):
    available = {t for g in TOOL_GROUPS for t in g.tools}
    window = int(
        (len(available) * tool_groups.SCHEMA_TOKENS_PER_TOOL)
        / tool_groups.BUDGET_FRACTION
    ) - 1
    plan = plan_grouped_tools(
        available=available, loaded_groups=set(), operator=True, context_window=window,
    )
    assert plan.active is True
    # Every grouped tool that's deferred must still appear by NAME in the index
    # — the capability is never hidden, only its full schema is lazy.
    for name in plan.defer:
        assert name in plan.index_text, f"{name} missing from capability index"


def test_index_marks_loaded_groups(grouped_on):
    available = {t for g in TOOL_GROUPS for t in g.tools}
    window = int(
        (len(available) * tool_groups.SCHEMA_TOKENS_PER_TOOL)
        / tool_groups.BUDGET_FRACTION
    ) - 1
    plan = plan_grouped_tools(
        available=available,
        loaded_groups={"scheduling"},
        operator=True,
        context_window=window,
    )
    assert "(loaded)" in plan.index_text
    # A loaded group's tools are no longer deferred.
    assert "set_cron" not in plan.defer


def test_worker_never_sees_operator_only_group(grouped_on):
    available = {t for g in TOOL_GROUPS for t in g.tools}
    # Size the window against the WORKER-eligible deferrable set (operator-only
    # groups are excluded), else the gate wouldn't trip for a worker.
    worker_deferrable = {
        t for g in TOOL_GROUPS if not g.operator_only for t in g.tools
    }
    window = int(
        (len(worker_deferrable) * tool_groups.SCHEMA_TOKENS_PER_TOOL)
        / tool_groups.BUDGET_FRACTION
    ) - 1
    plan = plan_grouped_tools(
        available=available, loaded_groups=set(), operator=False, context_window=window,
    )
    # Operator-only group tools must not be deferred/advertised for a worker.
    assert "create_agent" not in plan.defer
    assert "Fleet setup" not in plan.index_text
    # A non-operator-only group (web_browse) still appears.
    assert "Web & browse" in plan.index_text


# ── resolve_load_request ────────────────────────────────────────────────────
def test_resolve_by_group_key():
    keys, err = resolve_load_request(group="scheduling", tool=None, available=set())
    assert err is None
    assert keys == {"scheduling"}


def test_resolve_by_group_label():
    keys, err = resolve_load_request(group="Fleet setup", tool=None, available=set())
    assert err is None
    assert keys == {"fleet_setup"}


def test_resolve_by_tool_name():
    keys, err = resolve_load_request(group=None, tool="create_agent", available=set())
    assert err is None
    assert keys == {"fleet_setup"}


def test_resolve_unknown_group_errors():
    keys, err = resolve_load_request(group="nope", tool=None, available=set())
    assert keys == set()
    assert "Unknown tool group" in err


def test_resolve_core_tool_hint():
    keys, err = resolve_load_request(
        group=None, tool="notify_user", available={"notify_user"},
    )
    assert keys == set()
    assert "core surface" in err


def test_resolve_empty_errors():
    keys, err = resolve_load_request(group=None, tool=None, available=set())
    assert keys == set()
    assert err


# ── get_tool_definitions: defer folds into the memo cache key ───────────────
def _registry_with(*names: str) -> ToolRegistry:
    for n in names:
        @tool(name=n, description=f"desc {n}", parameters={"x": {"type": "string"}})
        def _fn(x: str):
            return x

    reg = ToolRegistry.__new__(ToolRegistry)
    reg.tools = dict(_tool_staging)
    reg._tool_defs_cache = {}
    reg._descriptions_cache = {}
    return reg


def test_defer_omits_schema():
    reg = _registry_with("alpha", "beta", "gamma")
    full = reg.get_tool_definitions()
    deferred = reg.get_tool_definitions(defer=frozenset({"beta"}))
    full_names = {d["function"]["name"] for d in full}
    deferred_names = {d["function"]["name"] for d in deferred}
    assert "beta" in full_names
    assert "beta" not in deferred_names
    assert deferred_names == full_names - {"beta"}


def test_defer_folds_into_cache_key():
    """Different loaded/deferred sets must yield different cached definitions."""
    reg = _registry_with("alpha", "beta", "gamma")
    a = reg.get_tool_definitions(defer=frozenset({"beta", "gamma"}))
    b = reg.get_tool_definitions(defer=frozenset({"gamma"}))
    # Different defer set → different result (different cache entry).
    assert a is not b
    assert {d["function"]["name"] for d in a} == {"alpha"}
    assert {d["function"]["name"] for d in b} == {"alpha", "beta"}
    # Same key returns the memoized object.
    a2 = reg.get_tool_definitions(defer=frozenset({"beta", "gamma"}))
    assert a2 is a


def test_defer_none_matches_today():
    """defer=None / absent must be byte-identical to the legacy build."""
    reg = _registry_with("alpha", "beta")
    legacy = reg.get_tool_definitions(exclude=frozenset())
    defer_none = reg.get_tool_definitions(exclude=frozenset(), defer=None)
    assert legacy == defer_none


def test_list_tools_ignores_defer():
    """A deferred tool is still callable → must remain in list_tools()."""
    reg = _registry_with("alpha", "beta")
    names = reg.list_tools(defer=frozenset({"beta"}))
    assert "beta" in names


# ── Loop integration: instance state + next-turn deferral ───────────────────
def _make_grouped_loop():
    """Build an AgentLoop whose operator allowlist covers every grouped tool.

    Mirrors tests/test_loop.py:_make_loop but with a tool surface large enough
    to trip the budget gate and a real (non-mock) ToolRegistry-like ``list_tools``
    that honours ``allowed``/``defer``.
    """
    from unittest.mock import AsyncMock, MagicMock

    from src.agent.loop import AgentLoop

    all_grouped = {t for g in TOOL_GROUPS for t in g.tools}
    allowed = frozenset(all_grouped | {"load_tools", "notify_user"})

    tools = MagicMock()

    def _list_tools(exclude=None, allowed=None, defer=None):
        names = list(allowed) if allowed is not None else list(all_grouped)
        return names

    tools.list_tools = MagicMock(side_effect=_list_tools)
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="")
    tools.is_parallel_safe = MagicMock(return_value=True)
    tools.get_loop_exempt_tools = MagicMock(return_value=frozenset())
    tools.operator_only_tools = MagicMock(return_value=frozenset())
    tools.tools = {n: {} for n in all_grouped}

    memory = MagicMock()
    memory.get_high_salience_facts = AsyncMock(return_value=[])
    memory.search = AsyncMock(return_value=[])

    llm = MagicMock()
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.is_standalone = False

    loop = AgentLoop(
        agent_id="operator",
        role="operator",
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
        allowed_tools=allowed,
    )
    # A roomy context window — the gate trips on the grouped-tool fraction below.
    cm = MagicMock()
    cm.max_tokens = int(
        (len(all_grouped) * tool_groups.SCHEMA_TOKENS_PER_TOOL)
        / tool_groups.BUDGET_FRACTION
    ) - 1
    loop.context_manager = cm
    return loop


def test_loop_refresh_builds_index_and_defer(grouped_on):
    loop = _make_grouped_loop()
    index = loop._refresh_grouped_plan()
    assert index  # capability index present
    assert loop._grouped_plan is not None and loop._grouped_plan.active
    # The defer set is now reflected in the tool-filter kwargs.
    assert "defer" in loop._tool_filter_kw


def test_loop_flag_off_no_index_no_defer(monkeypatch):
    monkeypatch.delenv(tool_groups.GROUPED_TOOLS_ENV, raising=False)
    loop = _make_grouped_loop()
    index = loop._refresh_grouped_plan()
    assert index == ""
    assert loop._grouped_plan is None
    assert "defer" not in loop._tool_filter_kw


def test_load_tools_defers_to_next_turn(grouped_on):
    loop = _make_grouped_loop()
    loop._refresh_grouped_plan()  # turn 1 build
    # set_cron is deferred at first.
    assert "set_cron" in loop._grouped_plan.defer

    # Request the scheduling group — must NOT mutate the loaded set mid-turn.
    res = loop.request_load_tools(group="scheduling", tool=None)
    assert res["loaded"] == ["scheduling"]
    assert loop._loaded_tool_groups == set()  # not applied yet (mid-turn)
    assert loop._pending_tool_groups == {"scheduling"}
    # The current turn's defer is unchanged (toolset stable within a turn).
    assert "set_cron" in loop._grouped_plan.defer

    # Next turn boundary: the build promotes pending → loaded.
    loop._refresh_grouped_plan()
    assert loop._loaded_tool_groups == {"scheduling"}
    assert loop._pending_tool_groups == set()
    # set_cron's schema is now loaded (no longer deferred).
    assert "set_cron" not in loop._grouped_plan.defer


def test_load_tools_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv(tool_groups.GROUPED_TOOLS_ENV, raising=False)
    loop = _make_grouped_loop()
    res = loop.request_load_tools(group="scheduling", tool=None)
    assert res["loaded"] == []
    assert loop._pending_tool_groups == set()
