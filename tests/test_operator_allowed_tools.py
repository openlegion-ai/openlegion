"""Tests for the allowed_tools allowlist mechanism."""
import pytest
import tempfile
import os
from src.agent.skills import SkillRegistry


def _make_registry_with_skills():
    """Create a real SkillRegistry with test skills in a temp dir."""
    td = tempfile.mkdtemp()
    skill_file = os.path.join(td, "test_skill.py")
    with open(skill_file, "w") as f:
        f.write('''
from src.agent.skills import skill

@skill(name="tool_a", description="Tool A", parameters={})
async def tool_a(**kw): return {}

@skill(name="tool_b", description="Tool B", parameters={})
async def tool_b(**kw): return {}

@skill(name="tool_c", description="Tool C", parameters={})
async def tool_c(**kw): return {}
''')
    return SkillRegistry(skills_dir=td), td


def test_allowed_tools_filters_to_allowlist():
    reg, td = _make_registry_with_skills()
    result = reg.get_tool_definitions(allowed=frozenset({"tool_a"}))
    names = {t["function"]["name"] for t in result}
    assert names == {"tool_a"}


def test_allowed_tools_none_returns_all():
    reg, td = _make_registry_with_skills()
    result = reg.get_tool_definitions(allowed=None)
    names = {t["function"]["name"] for t in result}
    assert "tool_a" in names and "tool_b" in names and "tool_c" in names


def test_allowed_overrides_exclude():
    """When allowed is set, exclude is ignored (spec: mutually exclusive)."""
    reg, td = _make_registry_with_skills()
    result = reg.get_tool_definitions(
        exclude=frozenset({"tool_b"}),
        allowed=frozenset({"tool_a", "tool_b"})
    )
    names = {t["function"]["name"] for t in result}
    assert names == {"tool_a", "tool_b"}  # tool_b NOT excluded


def test_exclude_still_works_without_allowed():
    reg, td = _make_registry_with_skills()
    result = reg.get_tool_definitions(exclude=frozenset({"tool_b"}))
    names = {t["function"]["name"] for t in result}
    assert "tool_b" not in names
    assert "tool_a" in names


def test_list_skills_respects_allowed():
    reg, td = _make_registry_with_skills()
    result = reg.list_skills(allowed=frozenset({"tool_c"}))
    assert result == ["tool_c"]


def test_get_descriptions_respects_allowed():
    reg, td = _make_registry_with_skills()
    desc = reg.get_descriptions(allowed=frozenset({"tool_a"}))
    assert "tool_a" in desc
    assert "tool_b" not in desc


def test_get_tool_sources_respects_allowed():
    reg, td = _make_registry_with_skills()
    sources = reg.get_tool_sources(allowed=frozenset({"tool_a", "tool_c"}))
    assert "tool_a" in sources
    assert "tool_c" in sources
    assert "tool_b" not in sources


def test_get_tool_sources_allowed_overrides_exclude():
    reg, td = _make_registry_with_skills()
    sources = reg.get_tool_sources(
        exclude=frozenset({"tool_a"}),
        allowed=frozenset({"tool_a", "tool_b"}),
    )
    assert "tool_a" in sources  # NOT excluded because allowed takes precedence
    assert "tool_b" in sources
    assert "tool_c" not in sources


def test_caching_with_different_allowed_values():
    """Ensure cache distinguishes between different allowed sets."""
    reg, td = _make_registry_with_skills()
    result1 = reg.get_tool_definitions(allowed=frozenset({"tool_a"}))
    result2 = reg.get_tool_definitions(allowed=frozenset({"tool_b"}))
    names1 = {t["function"]["name"] for t in result1}
    names2 = {t["function"]["name"] for t in result2}
    assert names1 == {"tool_a"}
    assert names2 == {"tool_b"}


def test_empty_allowed_returns_nothing():
    """An empty frozenset means no tools are allowed."""
    reg, td = _make_registry_with_skills()
    result = reg.get_tool_definitions(allowed=frozenset())
    assert result == []
