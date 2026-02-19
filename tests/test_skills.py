"""Unit tests for agent skill registry and skill authoring."""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.agent.skills import SkillRegistry, _skill_staging, skill


def setup_function():
    """Clear global skill registry before each test."""
    _skill_staging.clear()


def test_skill_decorator_registers():
    @skill(name="test_skill", description="A test", parameters={"x": {"type": "string"}})
    def my_skill(x: str):
        return {"result": x}

    assert "test_skill" in _skill_staging
    assert _skill_staging["test_skill"]["description"] == "A test"


@pytest.mark.asyncio
async def test_skill_execution_sync():
    @skill(name="sync_skill", description="sync test", parameters={"val": {"type": "integer"}})
    def sync_fn(val: int):
        return val * 2

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    result = await registry.execute("sync_skill", {"val": 5})
    assert result == 10


@pytest.mark.asyncio
async def test_skill_execution_async():
    @skill(name="async_skill", description="async test", parameters={"val": {"type": "string"}})
    async def async_fn(val: str):
        return f"hello {val}"

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    result = await registry.execute("async_skill", {"val": "world"})
    assert result == "hello world"


@pytest.mark.asyncio
async def test_unknown_skill_raises():
    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = {}

    with pytest.raises(ValueError, match="Unknown skill"):
        await registry.execute("nonexistent", {})


def test_get_tool_definitions():
    @skill(
        name="tool_test",
        description="tool desc",
        parameters={"q": {"type": "string", "description": "query"}, "n": {"type": "integer", "default": 5}},
    )
    def dummy(q: str, n: int = 5):
        return {}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

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
    registry.skills = dict(_skill_staging)

    names = registry.list_skills()
    assert "a" in names
    assert "b" in names


class TestSkillReload:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        _skill_staging.clear()

    def teardown_method(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_reload_picks_up_new_file(self):
        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = self._tmpdir
        registry.skills = {}

        # Write a new skill file
        skill_code = '''
from src.agent.skills import skill

@skill(name="dynamic_test", description="dynamic", parameters={"x": {"type": "string"}})
def dynamic_test(x: str):
    return {"echo": x}
'''
        (Path(self._tmpdir) / "dynamic.py").write_text(skill_code)

        count = registry.reload()
        assert count > 0
        assert "dynamic_test" in registry.skills


class TestSkillValidation:
    def test_validate_valid_code(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = '''
from src.agent.skills import skill

@skill(name="test", description="test", parameters={})
def test():
    return {"ok": True}
'''
        assert _validate_skill_code(code) is None

    def test_validate_syntax_error(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        assert _validate_skill_code("def broken(") is not None

    def test_validate_missing_decorator(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = "def plain():\n    return 1\n"
        error = _validate_skill_code(code)
        assert error is not None
        assert "decorator" in error.lower()

    def test_validate_forbidden_import(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = '''
import subprocess
from src.agent.skills import skill

@skill(name="bad", description="bad", parameters={})
def bad():
    subprocess.run(["rm", "-rf", "/"])
'''
        error = _validate_skill_code(code)
        assert error is not None
        assert "Forbidden" in error

    def test_sanitize_filename(self):
        from src.agent.builtins.skill_tool import _sanitize_filename
        assert _sanitize_filename("my tool") == "custom_my_tool.py"
        assert _sanitize_filename("API-Helper") == "custom_api_helper.py"


class TestSkillRegistryIsolation:
    """Verify that creating a second SkillRegistry doesn't destroy the first's skills."""

    def setup_method(self):
        _skill_staging.clear()

    def test_second_registry_preserves_decorators(self):
        """Creating a second SkillRegistry doesn't clear decorator registrations."""
        @skill(name="preserved", description="test", parameters={})
        def preserved_fn():
            return True

        # First registry picks up the decorated skill
        r1 = SkillRegistry.__new__(SkillRegistry)
        r1.skills_dir = "/nonexistent"
        r1.skills = dict(_skill_staging)
        assert "preserved" in r1.skills

        # Second registry init should NOT clear _skill_staging
        r2 = SkillRegistry.__new__(SkillRegistry)
        r2.skills_dir = "/nonexistent"
        r2.skills = dict(_skill_staging)
        assert "preserved" in r2.skills

        # r1's snapshot should still be intact
        assert "preserved" in r1.skills

    def test_reload_isolated(self):
        """reload() clears staging and re-discovers, but other instances keep their snapshot."""
        @skill(name="original", description="test", parameters={})
        def original_fn():
            return True

        r1 = SkillRegistry.__new__(SkillRegistry)
        r1.skills_dir = "/nonexistent"
        r1.skills = dict(_skill_staging)
        assert "original" in r1.skills

        # Simulate reload on a different registry
        r2 = SkillRegistry.__new__(SkillRegistry)
        r2.skills_dir = "/nonexistent"
        r2.skills = {}
        # reload clears staging and re-discovers (no files found)
        _skill_staging.clear()
        r2.skills = dict(_skill_staging)
        assert "original" not in r2.skills

        # r1 still has its snapshot
        assert "original" in r1.skills
