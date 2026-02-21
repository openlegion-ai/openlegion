"""Tests for in-container subagent spawning."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.builtins.subagent_tool import (
    MAX_CONCURRENT,
    MAX_DEPTH,
    _active_subagents,
    _clone_skill_registry,
    _cleanup_depth,
    _depth_map,
    _get_depth,
    _parent_llm_refs,
    _set_depth,
    list_subagents,
    register_parent_llm,
    spawn_subagent,
)


def _cleanup():
    """Clean up module-level state between tests."""
    _active_subagents.clear()
    _depth_map.clear()
    _parent_llm_refs.clear()


class TestSpawnSubagentBasic:
    @pytest.mark.asyncio
    async def test_spawn_subagent_basic(self):
        """Mock LLM (immediate final answer), verify spawned + blackboard write."""
        _cleanup()

        mock_llm = MagicMock()
        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "parent-1"
        mock_mesh.write_blackboard = AsyncMock(return_value={"version": 1})

        register_parent_llm("parent-1", mock_llm)

        # Mock execute_task to return immediately
        from src.shared.types import TaskResult
        mock_result = TaskResult(
            task_id="test",
            status="complete",
            result={"answer": "42"},
            tokens_used=100,
            duration_ms=500,
        )

        with patch("src.agent.builtins.subagent_tool._run_subagent") as mock_run:
            mock_run.return_value = {
                "status": "complete",
                "result": {"answer": "42"},
                "tokens_used": 100,
                "duration_ms": 500,
            }
            result = await spawn_subagent(
                task="What is 6*7?", mesh_client=mock_mesh,
            )

        assert result["spawned"] is True
        assert "subagent_id" in result
        assert "result_key" in result
        assert result["result_key"].startswith("subagent_results/parent-1/")

        _cleanup()


class TestSpawnSubagentDepthLimit:
    @pytest.mark.asyncio
    async def test_spawn_subagent_depth_limit(self):
        """Set depth=2, verify error."""
        _cleanup()

        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "deep-agent"

        _set_depth("deep-agent", MAX_DEPTH)

        result = await spawn_subagent(
            task="some task", mesh_client=mock_mesh,
        )

        assert "error" in result
        assert "depth" in result["error"].lower()

        _cleanup()


class TestSpawnSubagentConcurrentLimit:
    @pytest.mark.asyncio
    async def test_spawn_subagent_concurrent_limit(self):
        """Spawn 3, try 4th, verify error."""
        _cleanup()

        mock_llm = MagicMock()
        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "busy-parent"
        mock_mesh.write_blackboard = AsyncMock(return_value={"version": 1})

        register_parent_llm("busy-parent", mock_llm)

        # Create 3 "active" tasks that aren't done
        _active_subagents["busy-parent"] = {}
        for i in range(MAX_CONCURRENT):
            mock_task = MagicMock()
            mock_task.done.return_value = False
            _active_subagents["busy-parent"][f"sub_{i}"] = mock_task

        result = await spawn_subagent(
            task="one more task", mesh_client=mock_mesh,
        )

        assert "error" in result
        assert "concurrent" in result["error"].lower()

        _cleanup()


class TestSpawnSubagentPrunesDoneTasks:
    @pytest.mark.asyncio
    async def test_done_tasks_pruned_on_spawn(self):
        """Completed tasks are removed from _active_subagents on next spawn."""
        _cleanup()

        mock_llm = MagicMock()
        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "prune-parent"
        mock_mesh.write_blackboard = AsyncMock(return_value={"version": 1})

        register_parent_llm("prune-parent", mock_llm)

        # Add 2 done tasks and 1 active task
        _active_subagents["prune-parent"] = {}
        for i in range(2):
            done_task = MagicMock()
            done_task.done.return_value = True
            _active_subagents["prune-parent"][f"done_{i}"] = done_task

        active_task = MagicMock()
        active_task.done.return_value = False
        _active_subagents["prune-parent"]["active_0"] = active_task

        assert len(_active_subagents["prune-parent"]) == 3

        with patch("src.agent.builtins.subagent_tool._run_subagent") as mock_run:
            mock_run.return_value = {"status": "complete", "result": "ok"}
            await spawn_subagent(task="trigger prune", mesh_client=mock_mesh)

        # Done tasks should be pruned, only active_0 + new task remain
        assert "done_0" not in _active_subagents["prune-parent"]
        assert "done_1" not in _active_subagents["prune-parent"]
        assert "active_0" in _active_subagents["prune-parent"]

        _cleanup()


class TestSpawnSubagentTTLTimeout:
    @pytest.mark.asyncio
    async def test_spawn_subagent_ttl_timeout(self):
        """TTL=1s, slow LLM, verify timeout result on blackboard."""
        _cleanup()

        mock_llm = MagicMock()
        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "timeout-parent"
        mock_mesh.write_blackboard = AsyncMock(return_value={"version": 1})

        register_parent_llm("timeout-parent", mock_llm)

        from src.agent.builtins.subagent_tool import _run_subagent

        # Mock AgentLoop.execute_task to sleep forever
        async def slow_execute(assignment):
            await asyncio.sleep(100)

        with patch("src.agent.loop.AgentLoop") as MockLoop:
            mock_loop_inst = MagicMock()
            mock_loop_inst.execute_task = slow_execute
            MockLoop.return_value = mock_loop_inst

            with patch("src.agent.memory.MemoryStore") as MockMem:
                mock_mem = MagicMock()
                mock_mem.close = MagicMock()
                MockMem.return_value = mock_mem

                with patch("src.agent.workspace.WorkspaceManager"):
                    result = await _run_subagent(
                        parent_id="timeout-parent",
                        subagent_id="sub_slow",
                        task_text="slow task",
                        role="assistant",
                        ttl_seconds=1,
                        max_iterations=10,
                        mesh_client=mock_mesh,
                    )

        assert result["status"] == "timeout"
        assert "timed out" in result["result"].lower()
        assert "duration_ms" in result  # consistent with success path
        assert "iterations" not in result  # should not have legacy field
        mock_mesh.write_blackboard.assert_called_once()

        _cleanup()


class TestCloneSkillRegistry:
    def setup_method(self):
        """Ensure skill staging is populated with builtins before each test."""
        from src.agent.skills import SkillRegistry
        # Creating a registry re-discovers builtins and populates _skill_staging
        SkillRegistry(skills_dir="/nonexistent")

    def test_clone_has_skills_minus_unsafe(self):
        """Clone has same skills but without create_skill, reload_skills, spawn_subagent."""
        _cleanup()

        clone = _clone_skill_registry()

        # Should have regular skills (builtins were discovered in setup)
        assert len(clone.skills) > 0
        assert "exec" in clone.skills

        # Should NOT have unsafe skills
        assert "create_skill" not in clone.skills
        assert "reload_skills" not in clone.skills
        assert "spawn_subagent" not in clone.skills

    def test_clone_reload_is_noop(self):
        """reload() on clone returns skill count without re-discovering."""
        clone = _clone_skill_registry()
        count = clone.reload()
        assert isinstance(count, int)
        assert count == len(clone.skills)


class TestListSubagents:
    @pytest.mark.asyncio
    async def test_list_subagents(self):
        """Verify count after spawning."""
        _cleanup()

        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "list-parent"

        # No subagents yet
        result = await list_subagents(mesh_client=mock_mesh)
        assert result["count"] == 0

        # Add some mock subagents
        mock_done = MagicMock()
        mock_done.done.return_value = True
        mock_active = MagicMock()
        mock_active.done.return_value = False

        _active_subagents["list-parent"] = {
            "sub_1": mock_done,
            "sub_2": mock_active,
        }

        result = await list_subagents(mesh_client=mock_mesh)
        assert result["count"] == 2
        assert result["active"] == 1
        assert len(result["subagents"]) == 2

        _cleanup()


class TestSubagentMemoryIsolation:
    @pytest.mark.asyncio
    async def test_subagent_memory_isolation(self):
        """Subagent memory is separate from parent (uses :memory:)."""
        _cleanup()

        mock_llm = MagicMock()
        mock_mesh = AsyncMock()
        mock_mesh.agent_id = "iso-parent"
        mock_mesh.write_blackboard = AsyncMock(return_value={"version": 1})

        register_parent_llm("iso-parent", mock_llm)

        # Track MemoryStore instantiations
        memory_paths: list[str] = []

        from src.agent.builtins.subagent_tool import _run_subagent

        with patch("src.agent.memory.MemoryStore") as MockMem:
            mock_mem = MagicMock()
            mock_mem.close = MagicMock()
            MockMem.return_value = mock_mem

            def capture_path(db_path=":memory:", **kwargs):
                memory_paths.append(db_path)
                return mock_mem
            MockMem.side_effect = capture_path

            with patch("src.agent.loop.AgentLoop") as MockLoop:
                from src.shared.types import TaskResult
                mock_result = TaskResult(
                    task_id="test", status="complete",
                    result={"answer": "done"}, tokens_used=50, duration_ms=100,
                )
                mock_loop_inst = AsyncMock()
                mock_loop_inst.execute_task = AsyncMock(return_value=mock_result)
                mock_loop_inst.MAX_ITERATIONS = 20
                MockLoop.return_value = mock_loop_inst

                with patch("src.agent.workspace.WorkspaceManager"):
                    await _run_subagent(
                        parent_id="iso-parent",
                        subagent_id="sub_iso",
                        task_text="isolated task",
                        role="assistant",
                        ttl_seconds=30,
                        max_iterations=10,
                        mesh_client=mock_mesh,
                    )

        assert len(memory_paths) == 1
        assert memory_paths[0] == ":memory:"

        _cleanup()


class TestDepthTracking:
    def test_depth_tracking(self):
        _cleanup()

        _set_depth("agent-a", 0)
        assert _get_depth("agent-a") == 0

        _set_depth("agent-b", 1)
        assert _get_depth("agent-b") == 1

        _cleanup_depth("agent-a")
        assert _get_depth("agent-a") == 0  # defaults to 0

        _cleanup()

    def test_register_parent_sets_depth_zero(self):
        _cleanup()

        mock_llm = MagicMock()
        register_parent_llm("new-parent", mock_llm)

        assert _get_depth("new-parent") == 0
        assert _parent_llm_refs["new-parent"] is mock_llm

        _cleanup()
