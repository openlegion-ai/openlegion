"""Tests for the operator ``read_user_notifications`` tool (Bug 1).

The tool is the operator's PULL surface over the agent→user observation
log. It sanitizes each message at this boundary and tags rows
``display_only`` so the LLM treats them as untrusted observed traffic,
not as instructions.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _set_operator_env(monkeypatch):
    """read_user_notifications requires ALLOWED_TOOLS (defence-in-depth)."""
    monkeypatch.setenv("ALLOWED_TOOLS", "read_user_notifications")


@pytest.mark.asyncio
async def test_read_user_notifications_sanitizes_and_tags_display_only():
    from src.agent.builtins.operator_tools import read_user_notifications

    ts = time.time()
    mc = MagicMock()
    mc.read_user_notifications = AsyncMock(
        return_value={
            "notifications": [
                {"from": "scout", "message": "stage 2 blocked on foo", "ts": ts},
            ],
        }
    )

    result = await read_user_notifications(mesh_client=mc)

    assert result["count"] == 1
    n = result["notifications"][0]
    assert n["from"] == "scout"
    assert "foo" in n["message"]
    assert n["ts"] == ts
    assert n["display_only"] is True
    mc.read_user_notifications.assert_awaited_once_with(hours=24, limit=50)


@pytest.mark.asyncio
async def test_read_user_notifications_runs_messages_through_sanitizer():
    """Each message is sanitized at the tool boundary (untrusted data)."""
    from src.agent.builtins import operator_tools
    from src.agent.builtins.operator_tools import read_user_notifications

    seen: list[str] = []

    def _fake_sanitize(text: str) -> str:
        seen.append(text)
        return f"SANITIZED:{text}"

    mc = MagicMock()
    mc.read_user_notifications = AsyncMock(
        return_value={"notifications": [{"from": "a", "message": "raw text", "ts": 1.0}]}
    )

    orig = operator_tools.sanitize_for_prompt
    operator_tools.sanitize_for_prompt = _fake_sanitize
    try:
        result = await read_user_notifications(mesh_client=mc)
    finally:
        operator_tools.sanitize_for_prompt = orig

    assert seen == ["raw text"]
    assert result["notifications"][0]["message"] == "SANITIZED:raw text"


@pytest.mark.asyncio
async def test_read_user_notifications_passes_through_params():
    from src.agent.builtins.operator_tools import read_user_notifications

    mc = MagicMock()
    mc.read_user_notifications = AsyncMock(return_value={"notifications": []})

    result = await read_user_notifications(hours=6, limit=10, mesh_client=mc)

    assert result == {"notifications": [], "count": 0}
    mc.read_user_notifications.assert_awaited_once_with(hours=6, limit=10)


@pytest.mark.asyncio
async def test_read_user_notifications_rejects_non_operator(monkeypatch):
    """Without ALLOWED_TOOLS set, the tool self-rejects."""
    monkeypatch.delenv("ALLOWED_TOOLS", raising=False)
    from src.agent.builtins.operator_tools import read_user_notifications

    mc = MagicMock()
    mc.read_user_notifications = AsyncMock()
    result = await read_user_notifications(mesh_client=mc)
    assert "only available to the operator" in result["error"]
    mc.read_user_notifications.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_user_notifications_handles_mesh_error():
    from src.agent.builtins.operator_tools import read_user_notifications

    err = Exception("boom")
    err.response = MagicMock(status_code=403, text="forbidden")
    mc = MagicMock()
    mc.read_user_notifications = AsyncMock(side_effect=err)

    result = await read_user_notifications(mesh_client=mc)
    assert result["error"] == "mesh_error"
    assert result["status"] == 403
