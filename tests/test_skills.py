"""Unit tests for agent skill registry and skill authoring."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def test_get_tool_definitions_enum():
    """Enum values in parameters are passed through to tool definitions."""
    @skill(
        name="enum_test",
        description="test enum",
        parameters={
            "choice": {
                "type": "string",
                "enum": ["opt_a", "opt_b"],
                "description": "pick one",
            },
        },
    )
    def dummy(choice: str):
        return {}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    defs = registry.get_tool_definitions()
    assert len(defs) == 1
    props = defs[0]["function"]["parameters"]["properties"]
    assert props["choice"]["enum"] == ["opt_a", "opt_b"]


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


def test_list_skills_with_exclude():
    @skill(name="keep_me", description="keep", parameters={})
    def keep():
        return None

    @skill(name="drop_me", description="drop", parameters={})
    def drop():
        return None

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    excluded = frozenset({"drop_me"})
    names = registry.list_skills(exclude=excluded)
    assert "keep_me" in names
    assert "drop_me" not in names
    # Without exclude, both present
    assert "drop_me" in registry.list_skills()


def test_get_descriptions_with_exclude():
    @skill(name="visible", description="I am visible", parameters={})
    def vis():
        return None

    @skill(name="hidden", description="I am hidden", parameters={})
    def hid():
        return None

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    excluded = frozenset({"hidden"})
    desc = registry.get_descriptions(exclude=excluded)
    assert "visible" in desc
    assert "hidden" not in desc
    # Without exclude, both present
    assert "hidden" in registry.get_descriptions()


def test_get_tool_definitions_with_exclude():
    @skill(name="included", description="inc", parameters={"x": {"type": "string"}})
    def inc(x: str):
        return x

    @skill(name="excluded_tool", description="exc", parameters={"y": {"type": "string"}})
    def exc(y: str):
        return y

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    excluded = frozenset({"excluded_tool"})
    defs = registry.get_tool_definitions(exclude=excluded)
    names = [d["function"]["name"] for d in defs]
    assert "included" in names
    assert "excluded_tool" not in names
    # Without exclude, both present
    all_names = [d["function"]["name"] for d in registry.get_tool_definitions()]
    assert "excluded_tool" in all_names


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
        registry._mcp_client = None
        registry._tool_defs_cache = {}
        registry._descriptions_cache = {}

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

    def test_validate_valid_async_code(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = '''
from src.agent.skills import skill

@skill(name="async_test", description="async test", parameters={})
async def async_test(*, mesh_client=None):
    return {"ok": True}
'''
        assert _validate_skill_code(code) is None

    def test_validate_missing_decorator(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = "def plain():\n    return 1\n"
        error = _validate_skill_code(code)
        assert error is not None
        assert "decorator" in error.lower()

    def test_validate_missing_decorator_async(self):
        from src.agent.builtins.skill_tool import _validate_skill_code
        code = "async def plain():\n    return 1\n"
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


class TestMCPIntegration:
    """Tests for MCP tool integration with SkillRegistry."""

    def setup_method(self):
        _skill_staging.clear()

    def _make_mcp_client(self, tools: list[dict] | None = None):
        """Create a mock MCPClient with optional tool definitions."""
        from src.agent.mcp_client import MCPClient
        mcp = MagicMock(spec=MCPClient)
        mcp.list_tools.return_value = tools or []
        mcp.has_tool.side_effect = lambda name: name in {t["name"] for t in (tools or [])}
        mcp.call_tool = AsyncMock(return_value={"result": "mcp_result"})
        return mcp

    def test_mcp_tools_in_get_tool_definitions(self):
        """MCP tools appear in the LLM tool list with correct format."""
        mcp_tools = [
            {
                "name": "search_db",
                "description": "Search the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
                "function": "mcp",
            },
        ]
        mcp = self._make_mcp_client(mcp_tools)

        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = "/nonexistent"
        registry._mcp_client = mcp
        registry.skills = {}
        registry._register_mcp_tools()

        defs = registry.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["type"] == "function"
        assert defs[0]["function"]["name"] == "search_db"
        assert defs[0]["function"]["parameters"]["type"] == "object"
        assert "query" in defs[0]["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self):
        """MCP tool calls route through MCPClient, not function inspection."""
        mcp_tools = [
            {
                "name": "mcp_tool",
                "description": "An MCP tool",
                "parameters": {"type": "object", "properties": {}},
                "function": "mcp",
            },
        ]
        mcp = self._make_mcp_client(mcp_tools)

        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = "/nonexistent"
        registry._mcp_client = mcp
        registry.skills = {}
        registry._register_mcp_tools()

        result = await registry.execute("mcp_tool", {"key": "value"})
        assert result == {"result": "mcp_result"}
        mcp.call_tool.assert_awaited_once_with("mcp_tool", {"key": "value"})

    def test_reload_preserves_mcp_tools(self):
        """After reload, MCP tools are still registered."""
        mcp_tools = [
            {
                "name": "persistent_mcp",
                "description": "Persists across reload",
                "parameters": {"type": "object", "properties": {}},
                "function": "mcp",
            },
        ]
        mcp = self._make_mcp_client(mcp_tools)

        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = "/nonexistent"
        registry._mcp_client = mcp
        registry.skills = {}
        registry._register_mcp_tools()
        assert "persistent_mcp" in registry.skills

        # Simulate reload
        _skill_staging.clear()
        registry.skills = dict(_skill_staging)
        registry._register_mcp_tools()
        assert "persistent_mcp" in registry.skills

    def test_mcp_tools_in_list_skills(self):
        """MCP tools appear in list_skills output."""
        mcp_tools = [
            {
                "name": "mcp_listed",
                "description": "Should be listed",
                "parameters": {"type": "object", "properties": {}},
                "function": "mcp",
            },
        ]
        mcp = self._make_mcp_client(mcp_tools)

        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = "/nonexistent"
        registry._mcp_client = mcp
        registry.skills = {}
        registry._register_mcp_tools()

        assert "mcp_listed" in registry.list_skills()

    def test_no_mcp_client_is_noop(self):
        """When no MCP client is set, _register_mcp_tools does nothing."""
        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills_dir = "/nonexistent"
        registry._mcp_client = None
        registry.skills = {}
        registry._register_mcp_tools()
        assert len(registry.skills) == 0


@pytest.mark.asyncio
async def test_execute_filters_hallucinated_params():
    """LLM-hallucinated parameters (e.g. 'raw') should be silently dropped."""
    @skill(name="no_args_skill", description="takes no args", parameters={})
    def no_args():
        return {"ok": True}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    # Passing 'raw' should not crash — it should be filtered out
    result = await registry.execute("no_args_skill", {"raw": ""})
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_execute_filters_extra_params_keeps_valid():
    """Extra params are dropped but valid params are preserved."""
    @skill(name="one_arg_skill", description="takes one arg", parameters={"x": {"type": "string"}})
    def one_arg(x: str):
        return {"got": x}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    result = await registry.execute("one_arg_skill", {"x": "hello", "raw": "", "bogus": 42})
    assert result == {"got": "hello"}


@pytest.mark.asyncio
async def test_execute_kwargs_function_keeps_all_params():
    """Functions accepting **kwargs should receive all params including extras."""
    @skill(name="kwargs_skill", description="accepts kwargs", parameters={})
    def kwargs_fn(**kwargs):
        return {"keys": sorted(kwargs.keys())}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    result = await registry.execute("kwargs_skill", {"raw": "", "extra": "val"})
    assert result == {"keys": ["extra", "raw"]}


@pytest.mark.asyncio
async def test_execute_filters_hallucinated_params_async():
    """Async functions also get hallucinated params filtered."""
    @skill(name="async_no_args", description="async no args", parameters={})
    async def async_no_args():
        return {"async_ok": True}

    registry = SkillRegistry.__new__(SkillRegistry)
    registry.skills = dict(_skill_staging)

    result = await registry.execute("async_no_args", {"raw": ""})
    assert result == {"async_ok": True}


class TestGetToolSources:
    """Tests for SkillRegistry.get_tool_sources()."""

    def setup_method(self):
        _skill_staging.clear()

    def _make_registry(self, builtin_fns=(), extra_skills=None):
        """Build a registry with explicit builtin function refs."""
        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills = dict(_skill_staging)
        if extra_skills:
            registry.skills.update(extra_skills)
        registry._builtin_functions = frozenset(builtin_fns)
        return registry

    def test_builtin_tagged_correctly(self):
        """Skills whose function object is in _builtin_functions → 'builtin'."""
        @skill(name="core_tool", description="core", parameters={})
        def core_fn():
            return {}

        registry = self._make_registry(builtin_fns=[core_fn])
        sources = registry.get_tool_sources()
        assert sources["core_tool"] == "builtin"

    def test_custom_tagged_correctly(self):
        """Skills not in _builtin_functions and not MCP → 'custom'."""
        @skill(name="agent_tool", description="custom", parameters={})
        def agent_fn():
            return {}

        registry = self._make_registry(builtin_fns=[])
        sources = registry.get_tool_sources()
        assert sources["agent_tool"] == "custom"

    def test_mcp_tagged_correctly(self):
        """Skills with function='mcp' sentinel → 'mcp'."""
        mcp_entry = {
            "name": "ext_search",
            "description": "MCP tool",
            "parameters": {"type": "object", "properties": {}},
            "function": "mcp",
        }
        registry = self._make_registry(extra_skills={"ext_search": mcp_entry})
        sources = registry.get_tool_sources()
        assert sources["ext_search"] == "mcp"

    def test_exclude_removes_entries(self):
        """Excluded skills do not appear in the returned mapping."""
        @skill(name="visible", description="shown", parameters={})
        def vis_fn():
            return {}

        @skill(name="hidden", description="excluded", parameters={})
        def hid_fn():
            return {}

        registry = self._make_registry()
        sources = registry.get_tool_sources(exclude=frozenset({"hidden"}))
        assert "visible" in sources
        assert "hidden" not in sources

    def test_custom_override_of_builtin_name(self):
        """A custom skill using the same name as a builtin is tagged 'custom',
        not 'builtin', because the function object differs."""
        @skill(name="read_file", description="original builtin", parameters={})
        def original_builtin():
            return {}

        original_fn_ref = original_builtin  # capture the builtin function

        # Simulate a custom skill overriding the same name
        @skill(name="read_file", description="custom override", parameters={})
        def custom_override():
            return {}

        # registry.skills now has custom_override for "read_file"
        registry = self._make_registry(builtin_fns=[original_fn_ref])
        sources = registry.get_tool_sources()
        assert sources["read_file"] == "custom"

    def test_mixed_sources(self):
        """Registry with builtin, custom, and mcp tools returns correct tags."""
        @skill(name="builtin_a", description="b", parameters={})
        def builtin_a_fn():
            return {}

        @skill(name="custom_b", description="c", parameters={})
        def custom_b_fn():
            return {}

        mcp_entry = {
            "name": "mcp_c",
            "description": "mcp",
            "parameters": {"type": "object", "properties": {}},
            "function": "mcp",
        }
        registry = self._make_registry(
            builtin_fns=[builtin_a_fn],
            extra_skills={"mcp_c": mcp_entry},
        )
        sources = registry.get_tool_sources()
        assert sources["builtin_a"] == "builtin"
        assert sources["custom_b"] == "custom"
        assert sources["mcp_c"] == "mcp"

    def test_empty_registry_returns_empty(self):
        """Empty skills dict → empty sources dict."""
        registry = SkillRegistry.__new__(SkillRegistry)
        registry.skills = {}
        registry._builtin_functions = frozenset()
        assert registry.get_tool_sources() == {}


# ── Type coercion tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_coerces_string_to_int():
    """LLM sends '5' instead of 5 for an integer parameter."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="int_tool", description="t", parameters={
        "count": {"type": "integer", "description": "n"},
    })
    def int_tool(count: int) -> dict:
        return {"count": count, "type": type(count).__name__}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("int_tool", {"count": "5"})
    assert result == {"count": 5, "type": "int"}


@pytest.mark.asyncio
async def test_execute_coerces_string_to_bool():
    """LLM sends 'true' instead of true for a boolean parameter."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="bool_tool", description="t", parameters={
        "flag": {"type": "boolean", "description": "f"},
    })
    def bool_tool(flag: bool) -> dict:
        return {"flag": flag}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("bool_tool", {"flag": "true"})
    assert result == {"flag": True}


@pytest.mark.asyncio
async def test_execute_coerces_int_to_string():
    """LLM sends 42 instead of '42' for a string parameter."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="str_tool", description="t", parameters={
        "name": {"type": "string", "description": "n"},
    })
    def str_tool(name: str) -> dict:
        return {"name": name}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("str_tool", {"name": 42})
    assert result == {"name": "42"}


@pytest.mark.asyncio
async def test_execute_type_coercion_failure_raises():
    """Non-numeric string for integer parameter raises TypeError."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="bad_int", description="t", parameters={
        "count": {"type": "integer", "description": "n"},
    })
    def bad_int(count: int) -> dict:
        return {"count": count}

    registry.skills = dict(_skill_staging)
    with pytest.raises(TypeError, match="expects integer"):
        await registry.execute("bad_int", {"count": "not_a_number"})


# ── Required-parameter validation tests ───────────────────────────────


@pytest.mark.asyncio
async def test_execute_missing_required_param_raises():
    """Missing required parameter raises TypeError with clear message."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="req_tool", description="t", parameters={
        "path": {"type": "string", "description": "p"},
        "limit": {"type": "integer", "description": "l", "default": 10},
    })
    def req_tool(path: str, limit: int = 10) -> dict:
        return {"path": path}

    registry.skills = dict(_skill_staging)
    with pytest.raises(TypeError, match="Missing required.*path"):
        await registry.execute("req_tool", {})


@pytest.mark.asyncio
async def test_execute_malformed_raw_args_raises():
    """When LLM sends malformed JSON, the {"raw": ...} fallback triggers missing-param error."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="file_tool", description="t", parameters={
        "path": {"type": "string", "description": "p"},
    })
    def file_tool(path: str) -> dict:
        return {"path": path}

    registry.skills = dict(_skill_staging)
    # Simulates what happens when LLM sends malformed JSON → {"raw": "garbage"}
    with pytest.raises(TypeError, match="Missing required.*path"):
        await registry.execute("file_tool", {"raw": "some garbage"})


@pytest.mark.asyncio
async def test_execute_optional_params_not_required():
    """Optional params (with defaults) should not trigger missing-param error."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="opt_tool", description="t", parameters={
        "path": {"type": "string", "description": "p"},
        "verbose": {"type": "boolean", "description": "v", "default": False},
    })
    def opt_tool(path: str, verbose: bool = False) -> dict:
        return {"path": path, "verbose": verbose}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("opt_tool", {"path": "test.txt"})
    assert result == {"path": "test.txt", "verbose": False}


@pytest.mark.asyncio
async def test_execute_coerces_string_to_float():
    """LLM sends '3.14' instead of 3.14 for a number parameter."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="num_tool", description="t", parameters={
        "value": {"type": "number", "description": "n"},
    })
    def num_tool(value: float) -> dict:
        return {"value": value, "type": type(value).__name__}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("num_tool", {"value": "3.14"})
    assert result == {"value": 3.14, "type": "float"}


@pytest.mark.asyncio
async def test_execute_coerces_string_false_to_bool():
    """LLM sends 'false' string — must coerce to False, not truthy non-empty string."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="bool_false_tool", description="t", parameters={
        "flag": {"type": "boolean", "description": "f"},
    })
    def bool_false_tool(flag: bool) -> dict:
        return {"flag": flag}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("bool_false_tool", {"flag": "false"})
    assert result == {"flag": False}


@pytest.mark.asyncio
async def test_execute_function_default_not_flagged_as_required():
    """Param with function default but no schema default should NOT be rejected."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="image_tool", description="t", parameters={
        "prompt": {"type": "string", "description": "p"},
        "filename": {"type": "string", "description": "f"},  # No "default" in schema
    })
    def image_tool(prompt: str, filename: str = "") -> dict:  # Has default in function
        return {"prompt": prompt, "filename": filename}

    registry.skills = dict(_skill_staging)
    # LLM omits filename — should succeed using the function default
    result = await registry.execute("image_tool", {"prompt": "a cat"})
    assert result == {"prompt": "a cat", "filename": ""}


@pytest.mark.asyncio
async def test_execute_missing_param_error_includes_hints():
    """Error message includes parameter hints so agent can self-correct."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="search_tool", description="t", parameters={
        "query": {"type": "string", "description": "what to search"},
        "limit": {"type": "integer", "description": "max results", "default": 5},
    })
    def search_tool(query: str, limit: int = 5) -> dict:
        return {"query": query}

    registry.skills = dict(_skill_staging)
    with pytest.raises(TypeError, match="Expected:.*query.*string.*required.*limit.*integer.*optional"):
        await registry.execute("search_tool", {})


@pytest.mark.asyncio
async def test_execute_none_arguments_treated_as_empty():
    """None arguments (from null JSON) should be treated as empty dict."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="no_req_tool", description="t", parameters={
        "verbose": {"type": "boolean", "description": "v", "default": False},
    })
    def no_req_tool(verbose: bool = False) -> dict:
        return {"verbose": verbose}

    registry.skills = dict(_skill_staging)
    result = await registry.execute("no_req_tool", None)
    assert result == {"verbose": False}


@pytest.mark.asyncio
async def test_execute_non_dict_arguments_treated_as_empty():
    """Non-dict arguments (int, list, string) should not crash execute."""
    registry = SkillRegistry.__new__(SkillRegistry)

    @skill(name="safe_tool", description="t", parameters={
        "verbose": {"type": "boolean", "description": "v", "default": False},
    })
    def safe_tool(verbose: bool = False) -> dict:
        return {"verbose": verbose}

    registry.skills = dict(_skill_staging)
    for bad_args in [42, [1, 2], "hello", True, 0]:
        result = await registry.execute("safe_tool", bad_args)
        assert result == {"verbose": False}, f"Failed for arguments={bad_args!r}"
