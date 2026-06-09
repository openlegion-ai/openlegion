"""Tests for the execute_code (code-as-action) builtin.

Phase 2 of the operator memory/context overhaul (§7, C2). Covers: stdout
capture, multi-line snippets, error surfacing, output truncation, and the env
scrub (no credentials reach the child). The tool is always-on (no feature flag).
"""

from __future__ import annotations

import pytest

from src.agent.builtins import exec_tool
from src.agent.builtins.exec_tool import (
    _is_sensitive_env_name,
    _scrubbed_env,
    execute_code,
)

# ── execution ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_print():
    result = await execute_code(code="print(2 + 2)")
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "4"
    # Success path keeps stderr empty.
    assert result["stderr"] == ""


@pytest.mark.asyncio
async def test_multiline_code():
    code = (
        "total = 0\n"
        "for i in range(5):\n"
        "    total += i\n"
        "print('sum', total)\n"
    )
    result = await execute_code(code=code)
    assert result["exit_code"] == 0
    assert "sum 10" in result["stdout"]


@pytest.mark.asyncio
async def test_only_printed_output_returned():
    """Intermediate values never leak — only print() reaches the result."""
    code = "secret = 1234\nprint('answer:', 7 * 6)\n"
    result = await execute_code(code=code)
    assert result["exit_code"] == 0
    assert "42" in result["stdout"]
    assert "1234" not in result["stdout"]


@pytest.mark.asyncio
async def test_raise_surfaces_error():
    result = await execute_code(code="raise ValueError('boom')")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]
    assert "boom" in result["stderr"]


@pytest.mark.asyncio
async def test_empty_code_rejected():
    result = await execute_code(code="   ")
    assert result["exit_code"] == -1
    assert "non-empty" in result["stderr"]


@pytest.mark.asyncio
async def test_timeout():
    result = await execute_code(code="import time; time.sleep(10)", timeout=1)
    assert result["exit_code"] == -1
    assert "timed out" in result["stderr"]


@pytest.mark.asyncio
async def test_output_truncated():
    # Print way more than the cap; result must be clamped.
    code = "print('x' * 100000)"
    result = await execute_code(code=code)
    assert result["exit_code"] == 0
    assert len(result["stdout"]) <= exec_tool._CODE_MAX_OUTPUT


# ── env scrub ──────────────────────────────────────────────────────────────

def test_is_sensitive_env_name():
    for name in (
        "OPENLEGION_SYSTEM_ANTHROPIC_API_KEY",
        "OPENLEGION_CRED_GITHUB",
        "OL_INTERNET_ACCESS_ENABLED",
        "SOME_TOKEN",
        "my_secret_thing",
        "DB_PASSWORD",
        "AWS_ACCESS_KEY_ID",
    ):
        assert _is_sensitive_env_name(name) is True, name
    for name in ("PATH", "HOME", "LANG", "TZ"):
        assert _is_sensitive_env_name(name) is False, name


def test_scrubbed_env_drops_sensitive(monkeypatch):
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_TOKEN", "tok-123")
    monkeypatch.setenv("MY_PASSWORD", "hunter2")
    env = _scrubbed_env()
    assert "OPENLEGION_SYSTEM_ANTHROPIC_API_KEY" not in env
    assert "SOME_TOKEN" not in env
    assert "MY_PASSWORD" not in env
    # PATH (allowlisted, non-sensitive) survives so subprocess lookups work.
    assert "PATH" in env


@pytest.mark.asyncio
async def test_child_cannot_read_scrubbed_vars(monkeypatch):
    """The child process must not see credential-shaped env vars."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_ANTHROPIC_API_KEY", "sk-LEAKED")
    monkeypatch.setenv("SOME_TOKEN", "tok-LEAKED")
    code = (
        "import os\n"
        "print(os.environ.get('OPENLEGION_SYSTEM_ANTHROPIC_API_KEY'), "
        "os.environ.get('SOME_TOKEN'))\n"
    )
    result = await execute_code(code=code)
    assert result["exit_code"] == 0
    assert "LEAKED" not in result["stdout"]
    # Scrubbed vars come back as None.
    assert result["stdout"].strip() == "None None"
