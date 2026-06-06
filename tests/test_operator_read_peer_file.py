"""Tests for the operator list_peer_files / read_peer_file skills.

These extend the peer-read affordance beyond ``artifacts/`` (see
``test_operator_read_peer_artifact.py``) to a worker's full /data volume,
so the operator can locate and relay a deliverable a worker built as a
plain file (e.g. ``workspace/data.md`` or a generated CSV). Without this
the manager could not get a worker's data out to the user — it would only
ever see the ``artifacts/`` folder. Mirrors the read_peer_artifact shape.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """The skills require ALLOWED_TOOLS to be set (defence-in-depth)."""
    monkeypatch.setenv(
        "ALLOWED_TOOLS",
        "read_peer_file,list_peer_files,read_peer_artifact",
    )


# ── list_peer_files ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_peer_files_happy_path():
    """Happy path returns the mesh payload unchanged and forwards args."""
    from src.agent.builtins.operator_tools import list_peer_files

    payload = {
        "agent_id": "alpha",
        "entries": [
            {"name": "data.md", "size": 4096, "is_dir": False},
            {"name": "memory", "size": 0, "is_dir": True},
        ],
        "count": 2,
    }
    mc = MagicMock()
    mc.list_peer_files = AsyncMock(return_value=payload)

    result = await list_peer_files("alpha", "workspace", True, mesh_client=mc)

    assert result == payload
    mc.list_peer_files.assert_awaited_once_with("alpha", "workspace", True)


@pytest.mark.asyncio
async def test_list_peer_files_blocked_for_non_operator(monkeypatch):
    """Non-operator agents (no ALLOWED_TOOLS) are denied at the skill layer."""
    from src.agent.builtins.operator_tools import list_peer_files

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.list_peer_files = AsyncMock()

    result = await list_peer_files("alpha", mesh_client=mc)

    assert "error" in result
    assert "operator" in result["error"].lower()
    mc.list_peer_files.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_peer_files_agent_not_found():
    """A 404 from the mesh becomes {error: agent_not_found}."""
    from src.agent.builtins.operator_tools import list_peer_files

    fake_response = MagicMock()
    fake_response.status_code = 404
    fake_response.text = "Agent 'ghost' not found"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("404 Not Found")
            self.response = fake_response

    mc = MagicMock()
    mc.list_peer_files = AsyncMock(side_effect=FakeHTTPError())

    result = await list_peer_files("ghost", mesh_client=mc)
    assert result == {"error": "agent_not_found", "agent_id": "ghost"}


@pytest.mark.asyncio
async def test_list_peer_files_no_mesh_client():
    from src.agent.builtins.operator_tools import list_peer_files

    result = await list_peer_files("alpha", mesh_client=None)
    assert "error" in result
    assert "mesh_client" in result["error"]


# ── read_peer_file ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_peer_file_happy_path():
    """Happy path returns {agent_id, path, content, ...} and forwards offset."""
    from src.agent.builtins.operator_tools import read_peer_file

    payload = {
        "agent_id": "alpha",
        "path": "workspace/data.md",
        "content": "col_a,col_b\n1,2\n",
        "size": 16,
        "encoding": "utf-8",
        "offset": 0,
        "next_offset": 16,
        "truncated": False,
    }
    mc = MagicMock()
    mc.read_peer_file = AsyncMock(return_value=payload)

    result = await read_peer_file("alpha", "workspace/data.md", mesh_client=mc)

    assert result == payload
    mc.read_peer_file.assert_awaited_once_with(
        "alpha", "workspace/data.md", offset=0,
    )


@pytest.mark.asyncio
async def test_read_peer_file_paging_offset_forwarded():
    """A non-zero offset is threaded to the mesh client for paging."""
    from src.agent.builtins.operator_tools import read_peer_file

    mc = MagicMock()
    mc.read_peer_file = AsyncMock(return_value={"content": "", "truncated": False})

    await read_peer_file("alpha", "big.csv", offset=500_000, mesh_client=mc)

    mc.read_peer_file.assert_awaited_once_with(
        "alpha", "big.csv", offset=500_000,
    )


@pytest.mark.asyncio
async def test_read_peer_file_blocked_for_non_operator(monkeypatch):
    from src.agent.builtins.operator_tools import read_peer_file

    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    mc = MagicMock()
    mc.read_peer_file = AsyncMock()

    result = await read_peer_file("alpha", "workspace/data.md", mesh_client=mc)

    assert "error" in result
    assert "operator" in result["error"].lower()
    mc.read_peer_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_peer_file_not_found():
    from src.agent.builtins.operator_tools import read_peer_file

    fake_response = MagicMock()
    fake_response.status_code = 404
    fake_response.text = "File 'workspace/nope.md' not found on agent 'alpha'"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("404 Not Found")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_file = AsyncMock(side_effect=FakeHTTPError())

    result = await read_peer_file("alpha", "workspace/nope.md", mesh_client=mc)
    assert result == {
        "error": "file_not_found", "agent_id": "alpha",
        "path": "workspace/nope.md",
    }


@pytest.mark.asyncio
async def test_read_peer_file_invalid_path():
    """A 400 (traversal / bad name) from the mesh surfaces as invalid_path."""
    from src.agent.builtins.operator_tools import read_peer_file

    fake_response = MagicMock()
    fake_response.status_code = 400
    fake_response.text = "Path traversal not allowed"

    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("400 Bad Request")
            self.response = fake_response

    mc = MagicMock()
    mc.read_peer_file = AsyncMock(side_effect=FakeHTTPError())

    result = await read_peer_file("alpha", "../etc/passwd", mesh_client=mc)
    assert result["error"] == "invalid_path"


@pytest.mark.asyncio
async def test_read_peer_file_requires_path():
    from src.agent.builtins.operator_tools import read_peer_file

    mc = MagicMock()
    mc.read_peer_file = AsyncMock()
    result = await read_peer_file("alpha", "", mesh_client=mc)
    assert "error" in result
    mc.read_peer_file.assert_not_awaited()
