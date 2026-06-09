"""Unit tests for Grouped Tool Search (B2).

Covers:
- the budget gate (small toolset → unchanged; large → index+defer),
- the capability index renders ALL grouped capabilities (names never hidden),
- ``load_tools`` defers the schema change to the NEXT build (not mid-turn),
- the loaded-groups state folds into the ``get_tool_definitions`` cache key.

Grouped Tool Search is always-on; the SOLE activation control is the budget
gate — ``plan_grouped_tools`` returns ``active=False`` when the deferrable
schemas fall under ``BUDGET_FRACTION`` of the context window (small toolset or
tiny window).
"""

import pytest

from src.agent import tool_groups
from src.agent.tool_groups import (
    TOOL_GROUPS,
    plan_grouped_tools,
    resolve_load_request,
)
from src.agent.tools import ToolRegistry, _tool_staging, tool


def setup_function():
    _tool_staging.clear()


# ── Budget gate ─────────────────────────────────────────────────────────────
def test_budget_gate_small_toolset_unchanged():
    """A small grouped surface stays under the budget → plan inactive.

    This is the only "inactive" path now that the feature is always-on: a tiny
    deferrable surface against an opus-class window stays under the budget gate.
    """
    available = {"create_agent", "set_cron"}  # only 2 grouped tools
    plan = plan_grouped_tools(
        available=available,
        loaded_groups=set(),
        operator=True,
        context_window=200_000,  # opus-class window — 2 tools is far under 10%
    )
    assert plan.active is False
    assert plan.defer == frozenset()
    assert plan.index_text == ""


def test_budget_gate_large_toolset_activates():
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


def test_budget_gate_zero_window_inactive():
    available = {t for g in TOOL_GROUPS for t in g.tools}
    plan = plan_grouped_tools(
        available=available, loaded_groups=set(), operator=True, context_window=0,
    )
    assert plan.active is False


# ── Capability index renders ALL capabilities (names never hidden) ──────────
def test_index_lists_every_available_grouped_tool():
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


def test_index_marks_loaded_groups():
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


def test_worker_never_sees_operator_only_group():
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


def test_loop_refresh_builds_index_and_defer():
    loop = _make_grouped_loop()
    index = loop._refresh_grouped_plan()
    assert index  # capability index present
    assert loop._grouped_plan is not None and loop._grouped_plan.active
    # The defer set is now reflected in the tool-filter kwargs.
    assert "defer" in loop._tool_filter_kw


def test_loop_under_budget_no_index_no_defer():
    """When the agent's deferrable surface is under the budget gate, the plan is
    inactive: no index, no ``defer`` in the tool-filter kwargs."""
    loop = _make_grouped_loop()
    # Roomy enough that the grouped tools fall under BUDGET_FRACTION → inactive.
    cm = loop.context_manager
    cm.max_tokens = 10_000_000
    index = loop._refresh_grouped_plan()
    assert index == ""
    assert loop._grouped_plan is not None
    assert loop._grouped_plan.active is False
    assert "defer" not in loop._tool_filter_kw


def test_load_tools_defers_to_next_turn():
    loop = _make_grouped_loop()
    loop._refresh_grouped_plan(promote_pending=True)  # turn 1 build
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
    loop._refresh_grouped_plan(promote_pending=True)
    assert loop._loaded_tool_groups == {"scheduling"}
    assert loop._pending_tool_groups == set()
    # set_cron's schema is now loaded (no longer deferred).
    assert "set_cron" not in loop._grouped_plan.defer


def test_mid_turn_rebuild_does_not_promote():
    """A MID-turn rebuild (promote_pending=False) must leave pending untouched.

    Promoting mid-turn would flip the toolset and bust the prompt cache — the
    whole point of deferring to the next turn boundary.
    """
    loop = _make_grouped_loop()
    loop._refresh_grouped_plan(promote_pending=True)  # turn-entry build
    assert "set_cron" in loop._grouped_plan.defer

    loop.request_load_tools(group="scheduling", tool=None)
    assert loop._pending_tool_groups == {"scheduling"}

    # Mid-turn rebuilds (hot reload, playbook change, streaming rebuild) default
    # to promote_pending=False — pending stays queued, loaded set unchanged.
    loop._refresh_grouped_plan()
    assert loop._loaded_tool_groups == set()
    assert loop._pending_tool_groups == {"scheduling"}
    # The toolset the turn started with is preserved (set_cron still deferred).
    assert "set_cron" in loop._grouped_plan.defer

    # Only the next turn boundary actually promotes.
    loop._refresh_grouped_plan(promote_pending=True)
    assert loop._loaded_tool_groups == {"scheduling"}
    assert "set_cron" not in loop._grouped_plan.defer


def test_load_tools_noop_when_plan_inactive():
    """When the budget gate hasn't tripped (plan inactive), a load_tools request
    is a no-op — there is nothing deferred to load."""
    loop = _make_grouped_loop()
    # Roomy window → grouped tools under budget → plan inactive.
    loop.context_manager.max_tokens = 10_000_000
    loop._refresh_grouped_plan()
    assert loop._grouped_plan is not None and loop._grouped_plan.active is False
    res = loop.request_load_tools(group="scheduling", tool=None)
    assert res["loaded"] == []
    assert loop._pending_tool_groups == set()


# ── Fix 3b: defer=None uses the LEGACY 2-tuple cache key ────────────────────
def test_defer_none_uses_legacy_two_tuple_cache_key():
    """With no defer set, the memo key must be the legacy ``(exclude, allowed)``
    2-tuple so flag-off cache behaviour is byte-identical to main."""
    reg = _registry_with("alpha", "beta")
    reg.get_tool_definitions(exclude=frozenset({"beta"}))
    # The cache must be keyed by the 2-tuple, NOT a 3-tuple with a trailing None.
    assert (frozenset({"beta"}), None) in reg._tool_defs_cache
    assert (frozenset({"beta"}), None, None) not in reg._tool_defs_cache


def test_defer_set_uses_three_tuple_cache_key():
    """A real defer set widens the key to the 3-tuple (separate cache entry)."""
    reg = _registry_with("alpha", "beta")
    reg.get_tool_definitions(defer=frozenset({"beta"}))
    assert (None, None, frozenset({"beta"})) in reg._tool_defs_cache


# ── Fix 1: agent_loop injection is gated by tool NAME, not signature ────────
def _registry_with_agent_loop_tool(name: str) -> ToolRegistry:
    @tool(name=name, description=f"desc {name}", parameters={})
    async def _fn(*, agent_loop=None):
        return {"got_loop": agent_loop is not None}

    reg = ToolRegistry.__new__(ToolRegistry)
    reg.tools = dict(_tool_staging)
    reg._mcp_client = None
    reg._tool_defs_cache = {}
    reg._descriptions_cache = {}
    return reg


@pytest.mark.asyncio
async def test_agent_loop_injected_only_for_allowlisted_tool():
    from src.agent.tools import _AGENT_LOOP_TOOLS

    # The allowlist must be exactly the trusted bridge tool.
    assert _AGENT_LOOP_TOOLS == frozenset({"load_tools"})

    sentinel = object()
    reg = _registry_with_agent_loop_tool("load_tools")
    result = await reg.execute("load_tools", {}, agent_loop=sentinel)
    assert result == {"got_loop": True}


@pytest.mark.asyncio
async def test_agent_loop_not_injected_for_untrusted_tool():
    """A non-allowlisted tool declaring ``agent_loop`` must NOT receive it —
    the loop is the whole sandbox runtime; custom/self-authored tools can't
    capture it just by naming the param."""
    sentinel = object()
    reg = _registry_with_agent_loop_tool("evil_custom_tool")
    result = await reg.execute("evil_custom_tool", {}, agent_loop=sentinel)
    assert result == {"got_loop": False}


# ── load_tools is always in the worker tool surface (always-on) ─────────────
def test_load_tools_present_in_worker_surface():
    """The grouped-tools bridge is always available to a worker now that the
    feature is always-on — it's never hidden from the effective surface."""
    loop = _make_worker_loop()
    excluded = loop._excluded_tools or frozenset()
    assert "load_tools" not in excluded
    assert "load_tools" in loop.tools.list_tools(**loop._tool_filter_kw)


def _make_worker_loop():
    """A worker AgentLoop (exclude-based surface) with load_tools registered."""
    from unittest.mock import AsyncMock, MagicMock

    from src.agent.loop import AgentLoop

    surface = {"notify_user", "load_tools", "read_file"}

    tools = MagicMock()

    def _list_tools(exclude=None, allowed=None, defer=None):
        names = list(surface)
        if exclude:
            names = [n for n in names if n not in exclude]
        return names

    tools.list_tools = MagicMock(side_effect=_list_tools)
    tools.get_tool_definitions = MagicMock(return_value=[])
    tools.get_descriptions = MagicMock(return_value="")
    tools.is_parallel_safe = MagicMock(return_value=True)
    tools.get_loop_exempt_tools = MagicMock(return_value=frozenset())
    tools.operator_only_tools = MagicMock(return_value=frozenset())
    tools.tools = {n: {} for n in surface}

    memory = MagicMock()
    memory.get_high_salience_facts = AsyncMock(return_value=[])
    memory.search = AsyncMock(return_value=[])

    llm = MagicMock()
    llm.default_model = "test-model"

    mesh_client = MagicMock()
    mesh_client.is_standalone = False

    return AgentLoop(
        agent_id="worker",
        role="worker",
        memory=memory,
        tools=tools,
        llm=llm,
        mesh_client=mesh_client,
    )
