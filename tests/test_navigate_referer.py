"""Phase 3 §6.5 — navigate plumbs the picked referer to Playwright."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_inst(monkeypatch, current_url=""):
    """Minimal CamoufoxInstance with mocked page.goto."""
    monkeypatch.delenv("BROWSER_RECORD_BEHAVIOR", raising=False)
    import src.browser.flags as flags
    flags._operator_settings = None
    from src.browser.service import CamoufoxInstance

    page = MagicMock()
    page.goto = AsyncMock()
    page.url = current_url
    page.title = AsyncMock(return_value="title")
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value={})
    inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), page)
    return inst


class TestPickerInvokedByDefault:
    @pytest.mark.asyncio
    async def test_picker_runs_when_referer_unset(self, monkeypatch, tmp_path):
        """``referer=None`` ⇒ service picks one. Verify a search referer
        ends up on the goto call for an unknown destination."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://www.example.com/page")

        kwargs = inst.page.goto.call_args.kwargs
        assert "referer" in kwargs
        # Unknown host → search-engine referer
        assert kwargs["referer"] in (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )

    @pytest.mark.asyncio
    async def test_direct_nav_host_omits_referer_kwarg(
        self, monkeypatch, tmp_path,
    ):
        """For Gmail-class hosts the picker returns empty; the navigate
        path should NOT pass ``referer=`` to ``goto`` (Playwright's
        unset default is the right behaviour)."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://mail.google.com/")

        kwargs = inst.page.goto.call_args.kwargs
        assert "referer" not in kwargs


class TestExplicitReferer:
    @pytest.mark.asyncio
    async def test_caller_override_passes_through(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        await mgr.navigate(
            "a1", "https://example.com/",
            referer="https://news.ycombinator.com/",
        )

        kwargs = inst.page.goto.call_args.kwargs
        assert kwargs["referer"] == "https://news.ycombinator.com/"

    @pytest.mark.asyncio
    async def test_explicit_empty_string_means_no_referer(
        self, monkeypatch, tmp_path,
    ):
        """``referer=""`` is a deliberate "no referer" — picker must
        NOT run and the kwarg must be absent on goto."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://example.com/", referer="")

        kwargs = inst.page.goto.call_args.kwargs
        assert "referer" not in kwargs


class TestRollingHistory:
    @pytest.mark.asyncio
    async def test_each_nav_appends_to_recent(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        for _ in range(3):
            await mgr.navigate("a1", "https://example.com/")
        assert len(inst.recent_referers) == 3

    @pytest.mark.asyncio
    async def test_recent_referers_capped_at_5(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst

        for _ in range(12):
            await mgr.navigate("a1", "https://example.com/")
        assert len(inst.recent_referers) == 5

    @pytest.mark.asyncio
    async def test_same_origin_picked_when_previous_url_matches(
        self, monkeypatch, tmp_path,
    ):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch, current_url="https://example.com/dash")
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://example.com/orders")
        kwargs = inst.page.goto.call_args.kwargs
        assert kwargs["referer"] == "https://example.com/"
