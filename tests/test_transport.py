"""Tests for the Transport layer (HttpTransport and SandboxTransport)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.host.transport import HttpTransport, SandboxTransport

# ── HttpTransport ─────────────────────────────────────────────

class TestHttpTransport:
    def test_register_and_get_url(self):
        t = HttpTransport()
        t.register("alpha", "http://localhost:8401")
        assert t.get_url("alpha") == "http://localhost:8401"
        assert t.get_url("missing") is None

    @pytest.mark.asyncio
    async def test_request_unregistered_agent(self):
        t = HttpTransport()
        result = await t.request("unknown", "GET", "/status")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_is_reachable_unregistered(self):
        t = HttpTransport()
        assert await t.is_reachable("unknown") is False

    def test_request_sync_unregistered(self):
        t = HttpTransport()
        result = t.request_sync("unknown", "GET", "/status")
        assert "error" in result


# ── SandboxTransport ──────────────────────────────────────────

class TestSandboxTransport:
    @pytest.mark.asyncio
    async def test_request_builds_correct_command(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            b'{"state": "idle"}', b""
        ))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await t.request("alpha", "GET", "/status", timeout=5)

        assert result == {"state": "idle"}
        args = mock_exec.call_args[0]
        assert "docker" in args
        assert "sandbox" in args
        assert "exec" in args
        assert "openlegion_alpha" in args
        assert "curl" in args
        assert "http://localhost:8400/status" in args

    @pytest.mark.asyncio
    async def test_request_with_json_body(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            b'{"response": "hello"}', b""
        ))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            result = await t.request(
                "beta", "POST", "/chat",
                json={"message": "hi"}, timeout=10,
            )

        assert result == {"response": "hello"}
        args = mock_exec.call_args[0]
        assert "-d" in args

    @pytest.mark.asyncio
    async def test_request_nonzero_exit(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"sandbox not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await t.request("alpha", "GET", "/status", timeout=5)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_request_invalid_json(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await t.request("alpha", "GET", "/status", timeout=5)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_is_reachable_true(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(
            b'{"state": "idle"}', b""
        ))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await t.is_reachable("alpha") is True

    @pytest.mark.asyncio
    async def test_is_reachable_false(self):
        t = SandboxTransport()
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fail"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await t.is_reachable("alpha") is False

    def test_request_sync_success(self):
        t = SandboxTransport()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"state": "idle"}'

        with patch("subprocess.run", return_value=mock_result):
            result = t.request_sync("alpha", "GET", "/status", timeout=5)

        assert result == {"state": "idle"}

    def test_request_sync_failure(self):
        t = SandboxTransport()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "not found"

        with patch("subprocess.run", return_value=mock_result):
            result = t.request_sync("alpha", "GET", "/status", timeout=5)

        assert "error" in result


# ── HttpTransport Client Lifecycle ───────────────────────────

class TestHttpTransportClientLifecycle:
    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """_get_client returns the same instance on repeated calls."""
        t = HttpTransport()
        c1 = await t._get_client()
        c2 = await t._get_client()
        assert c1 is c2
        await t.close()

    @pytest.mark.asyncio
    async def test_close_and_recreate(self):
        """After close(), _get_client creates a new instance."""
        t = HttpTransport()
        c1 = await t._get_client()
        await t.close()
        assert c1.is_closed
        c2 = await t._get_client()
        assert c2 is not c1
        assert not c2.is_closed
        await t.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Calling close() twice does not raise."""
        t = HttpTransport()
        await t._get_client()
        await t.close()
        await t.close()  # should not raise


class _FakeStreamCtx:
    """Async-context-manager stand-in for ``client.stream(...)``.

    Yields a response whose ``aiter_lines()`` replays a fixed list of raw
    SSE lines (data lines and ``:`` comment/keepalive lines).
    """

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def __aenter__(self):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        lines = self._lines

        async def _aiter():
            for ln in lines:
                yield ln

        resp.aiter_lines = lambda: _aiter()
        return resp

    async def __aexit__(self, *exc):
        return False


class TestHttpTransportStreamKeepalive:
    @pytest.mark.asyncio
    async def test_stream_request_forwards_keepalive_comments(self):
        """Agent ``: keepalive`` SSE comments are forwarded as
        ``{"type": "keepalive"}`` sentinels.

        Regression: the comment used to be dropped, so a long silent tool
        call produced zero bytes on the dashboard->browser leg and the
        browser's 120s idle-abort cancelled the whole turn. Forwarding the
        agent's own keepalive lets each downstream hop reset its idle timer.
        """
        t = HttpTransport()
        t.register("a1", "http://agent")
        lines = [
            ": keepalive",
            "",  # blank line that follows an SSE comment — must be ignored
            'data: {"type": "text_delta", "content": "hi"}',
            ": keepalive",
            'data: {"type": "done", "response": "hi"}',
        ]
        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=_FakeStreamCtx(lines))
        with patch.object(t, "_get_client", AsyncMock(return_value=fake_client)):
            out = [
                ev
                async for ev in t.stream_request("a1", "POST", "/chat/stream")
            ]

        assert out.count({"type": "keepalive"}) == 2
        assert {"type": "text_delta", "content": "hi"} in out
        assert out[-1] == {"type": "done", "response": "hi"}
        # The trailing blank line must NOT have produced a spurious event.
        assert {"raw": ""} not in out
