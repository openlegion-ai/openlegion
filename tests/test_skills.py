"""Unit tests for agent skill registry."""

import asyncio

from src.agent.skills import SkillRegistry, _skill_registry, skill


def setup_function():
    """Clear global skill registry before each test."""
    _skill_registry.clear()


def test_skill_decorator_registers():
    @skill(name="test_skill", description="A test", parameters={"x": {"type": "string"}})
    def my_skill(x: str):
        return {"result": x}

    assert "test_skill" in _skill_registry
    assert _skill_registry["test_skill"]["description"] == "A test"


def test_skill_execution_sync():
    @skill(name="sync_skill", description="sync test", parameters={"val": {"type": "integer"}})
    def sync_fn(val: int):
        return val * 2

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_registry)

    result = asyncio.get_event_loop().run_until_complete(registry.execute("sync_skill", {"val": 5}))
    assert result == 10


def test_skill_execution_async():
    @skill(name="async_skill", description="async test", parameters={"val": {"type": "string"}})
    async def async_fn(val: str):
        return f"hello {val}"

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_registry)

    result = asyncio.get_event_loop().run_until_complete(registry.execute("async_skill", {"val": "world"}))
    assert result == "hello world"


def test_unknown_skill_raises():
    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = {}

    try:
        asyncio.get_event_loop().run_until_complete(registry.execute("nonexistent", {}))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown skill" in str(e)


def test_get_tool_definitions():
    @skill(
        name="tool_test",
        description="tool desc",
        parameters={"q": {"type": "string", "description": "query"}, "n": {"type": "integer", "default": 5}},
    )
    def dummy(q: str, n: int = 5):
        return {}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_registry)

    defs = registry.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "tool_test"
    assert "q" in defs[0]["function"]["parameters"]["required"]
    assert "n" not in defs[0]["function"]["parameters"]["required"]


def test_list_skills():
    @skill(name="a", description="a", parameters={})
    def a():
        return None

    @skill(name="b", description="b", parameters={})
    def b():
        return None

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_registry)

    names = registry.list_skills()
    assert "a" in names
    assert "b" in names
