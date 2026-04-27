"""Tests for ``BrowserManager.request_captcha_help`` (Phase 8 §11.14).

Mirrors the existing ``request_browser_login`` flow:
  * Manager method records the request and returns a structured response.
  * Mesh endpoint emits a dashboard event.
  * Dashboard completion endpoint enqueues a steer message to the agent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager, CamoufoxInstance


def _mk_inst(page_url="https://example.com/blocked"):
    mock_page = MagicMock()
    mock_page.url = page_url
    return CamoufoxInstance("agent-1", MagicMock(), MagicMock(), mock_page)


@pytest.fixture()
def mgr(tmp_path):
    return BrowserManager(profiles_dir=str(tmp_path / "profiles"))


class TestManagerRequestCaptchaHelp:
    @pytest.mark.asyncio
    async def test_returns_structured_envelope(self, mgr):
        inst = _mk_inst()
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.request_captcha_help(
            "agent-1",
            service="Cloudflare Turnstile",
            description="Please click the checkbox.",
        )
        assert result["success"] is True
        assert result["data"]["requested"] is True
        assert result["data"]["service"] == "Cloudflare Turnstile"
        assert result["data"]["description"] == "Please click the checkbox."
        assert result["data"]["url"] == "https://example.com/blocked"

    @pytest.mark.asyncio
    async def test_missing_service_returns_invalid_input(self, mgr):
        inst = _mk_inst()
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.request_captcha_help(
            "agent-1", service="", description="d",
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_missing_description_returns_invalid_input(self, mgr):
        inst = _mk_inst()
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        result = await mgr.request_captcha_help(
            "agent-1", service="X", description="",
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_truncates_long_inputs(self, mgr):
        inst = _mk_inst()
        mgr._instances["agent-1"] = inst
        mgr.get_or_start = AsyncMock(return_value=inst)

        long_service = "S" * 500
        long_desc = "D" * 2000
        result = await mgr.request_captcha_help(
            "agent-1", service=long_service, description=long_desc,
        )
        # service truncated to 128, description to 500.
        assert len(result["data"]["service"]) == 128
        assert len(result["data"]["description"]) == 500
