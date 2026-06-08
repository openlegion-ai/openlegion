"""Tests for the allowed_tools allowlist mechanism."""
import os
import tempfile
from unittest.mock import MagicMock

from src.agent.loop import AgentLoop
from src.agent.tools import ToolRegistry


def _make_registry_with_tools():
    """Create a real ToolRegistry with test tools in a temp dir."""
    td = tempfile.mkdtemp()
    tool_file = os.path.join(td, "test_tool.py")
    with open(tool_file, "w") as f:
        f.write('''
from src.agent.tools import tool

@tool(name="tool_a", description="Tool A", parameters={})
async def tool_a(**kw): return {}

@tool(name="tool_b", description="Tool B", parameters={})
async def tool_b(**kw): return {}

@tool(name="tool_c", description="Tool C", parameters={})
async def tool_c(**kw): return {}
''')
    return ToolRegistry(tools_dir=td), td


def test_allowed_tools_filters_to_allowlist():
    reg, td = _make_registry_with_tools()
    result = reg.get_tool_definitions(allowed=frozenset({"tool_a"}))
    names = {t["function"]["name"] for t in result}
    assert names == {"tool_a"}


def test_allowed_tools_none_returns_all():
    reg, td = _make_registry_with_tools()
    result = reg.get_tool_definitions(allowed=None)
    names = {t["function"]["name"] for t in result}
    assert "tool_a" in names and "tool_b" in names and "tool_c" in names


def test_allowed_overrides_exclude():
    """When allowed is set, exclude is ignored (spec: mutually exclusive)."""
    reg, td = _make_registry_with_tools()
    result = reg.get_tool_definitions(
        exclude=frozenset({"tool_b"}),
        allowed=frozenset({"tool_a", "tool_b"})
    )
    names = {t["function"]["name"] for t in result}
    assert names == {"tool_a", "tool_b"}  # tool_b NOT excluded


def test_exclude_still_works_without_allowed():
    reg, td = _make_registry_with_tools()
    result = reg.get_tool_definitions(exclude=frozenset({"tool_b"}))
    names = {t["function"]["name"] for t in result}
    assert "tool_b" not in names
    assert "tool_a" in names


def test_list_tools_respects_allowed():
    reg, td = _make_registry_with_tools()
    result = reg.list_tools(allowed=frozenset({"tool_c"}))
    assert result == ["tool_c"]


def test_get_descriptions_respects_allowed():
    reg, td = _make_registry_with_tools()
    desc = reg.get_descriptions(allowed=frozenset({"tool_a"}))
    assert "tool_a" in desc
    assert "tool_b" not in desc


def test_get_tool_sources_respects_allowed():
    reg, td = _make_registry_with_tools()
    sources = reg.get_tool_sources(allowed=frozenset({"tool_a", "tool_c"}))
    assert "tool_a" in sources
    assert "tool_c" in sources
    assert "tool_b" not in sources


def test_get_tool_sources_allowed_overrides_exclude():
    reg, td = _make_registry_with_tools()
    sources = reg.get_tool_sources(
        exclude=frozenset({"tool_a"}),
        allowed=frozenset({"tool_a", "tool_b"}),
    )
    assert "tool_a" in sources  # NOT excluded because allowed takes precedence
    assert "tool_b" in sources
    assert "tool_c" not in sources


def test_caching_with_different_allowed_values():
    """Ensure cache distinguishes between different allowed sets."""
    reg, td = _make_registry_with_tools()
    result1 = reg.get_tool_definitions(allowed=frozenset({"tool_a"}))
    result2 = reg.get_tool_definitions(allowed=frozenset({"tool_b"}))
    names1 = {t["function"]["name"] for t in result1}
    names2 = {t["function"]["name"] for t in result2}
    assert names1 == {"tool_a"}
    assert names2 == {"tool_b"}


def test_empty_allowed_returns_nothing():
    """An empty frozenset means no tools are allowed."""
    reg, td = _make_registry_with_tools()
    result = reg.get_tool_definitions(allowed=frozenset())
    assert result == []


# --- operator_only flag: keep operator orchestration tools out of worker context ---

def _make_registry_with_operator_tool():
    """Registry with one operator_only tool and one normal tool."""
    td = tempfile.mkdtemp()
    with open(os.path.join(td, "oo_tool.py"), "w") as f:
        f.write('''
from src.agent.tools import tool

@tool(name="worker_tool", description="Worker", parameters={})
async def worker_tool(**kw): return {}

@tool(name="boss_tool", description="Boss", parameters={}, operator_only=True)
async def boss_tool(**kw): return {}
''')
    return ToolRegistry(tools_dir=td), td


def test_operator_only_flag_captured():
    reg, _ = _make_registry_with_operator_tool()
    assert reg.operator_only_tools() >= {"boss_tool"}
    assert "worker_tool" not in reg.operator_only_tools()


def test_operator_only_defaults_false():
    """Tools without the flag are never reported as operator-only."""
    reg, _ = _make_registry_with_tools()
    assert reg.operator_only_tools().isdisjoint({"tool_a", "tool_b", "tool_c"})


def test_real_registry_marks_orchestration_tools_operator_only():
    """A representative slice of the operator orchestration surface is flagged,
    and core worker tools are NOT — so the worker exclude drops the former."""
    reg = ToolRegistry(tools_dir=tempfile.mkdtemp())
    oo = reg.operator_only_tools()
    # Orchestration / management tools — must be hidden from workers.
    for name in (
        "edit_agent", "create_agent", "create_team", "manage_team",
        "manage_agent", "manage_task", "apply_template", "list_templates",
        "install_skill", "assign_skill", "rate_delivery", "workflow_snapshot",
    ):
        assert name in oo, f"{name} should be operator_only"
    # Worker tools — must stay visible to workers.
    for name in (
        "read_file", "write_file", "list_files", "run_command",
        "http_request", "web_search", "skills_list", "skill_view",
        "memory_save", "memory_search",
    ):
        assert name not in oo, f"{name} must NOT be operator_only"


def test_operator_only_flag_covers_operator_modules():
    """Drift guard: every ``@tool`` declared in the operator-only modules is
    flagged ``operator_only=True`` (so its schema is dropped from workers),
    and the flag is not set on shared worker builtins.

    Parses the decorator source directly (rather than introspecting the live
    registry, whose module-global staging dict accumulates tools across tests)
    so adding a tool to operator_tools.py / fleet_tool.py / skill_admin_tool.py
    without operator_only=True fails here.
    """
    import re
    from pathlib import Path

    builtins_dir = Path(__file__).resolve().parents[1] / "src" / "agent" / "builtins"
    declared: set[str] = set()
    for fname in ("operator_tools.py", "fleet_tool.py", "skill_admin_tool.py"):
        text = (builtins_dir / fname).read_text()
        declared |= set(re.findall(r'^\s+name="([^"]+)",\s*$', text, re.MULTILINE))
    assert declared, "parsed no operator tool names — decorator format changed?"

    oo = ToolRegistry(tools_dir=tempfile.mkdtemp()).operator_only_tools()
    missing = declared - oo
    assert not missing, f"operator-module tools missing operator_only=True: {missing}"

    # Shared worker builtins must NOT be flagged — otherwise we'd hide a tool
    # workers legitimately need.
    worker_tools = {
        "read_file", "write_file", "list_files", "run_command",
        "http_request", "web_search", "skills_list", "skill_view",
        "memory_save", "memory_search", "notify_user", "check_inbox",
        "spawn_subagent",
    }
    assert worker_tools.isdisjoint(oo), f"worker tools wrongly flagged: {worker_tools & oo}"


def _make_loop_with_real_registry(allowed_tools=None):
    """Build an AgentLoop wired to a REAL ToolRegistry to exercise the
    schema-build-time tool filtering computed in __init__."""
    reg = ToolRegistry(tools_dir=tempfile.mkdtemp())
    mesh = MagicMock()
    mesh.is_standalone = False
    loop = AgentLoop(
        agent_id="operator" if allowed_tools else "worker",
        role="operator" if allowed_tools else "research",
        memory=MagicMock(),
        tools=reg,
        llm=MagicMock(),
        mesh_client=mesh,
        allowed_tools=allowed_tools,
    )
    return loop, reg


def test_worker_loop_hides_operator_tools_from_schema():
    """A worker (no allowlist) never receives operator-only tool schemas."""
    loop, reg = _make_loop_with_real_registry()
    kw = loop._tool_filter_kw
    assert "exclude" in kw
    assert {"edit_agent", "create_team", "apply_template"} <= kw["exclude"]
    names = {t["function"]["name"] for t in reg.get_tool_definitions(**kw)}
    assert "edit_agent" not in names
    assert "create_team" not in names
    # Worker tools remain visible.
    assert "read_file" in names
    assert "run_command" in names


def test_operator_loop_still_receives_operator_tools():
    """The operator path uses the explicit allowlist, so the operator_only
    exclude never strips its orchestration tools."""
    allowed = frozenset({"edit_agent", "create_team", "read_file"})
    loop, reg = _make_loop_with_real_registry(allowed_tools=allowed)
    kw = loop._tool_filter_kw
    assert kw.get("allowed") == allowed
    assert "exclude" not in kw
    names = {t["function"]["name"] for t in reg.get_tool_definitions(**kw)}
    assert "edit_agent" in names
    assert "create_team" in names
