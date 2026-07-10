"""Truth-telling for the agent browser.

Two independent gaps closed here:

1. ``navigate`` / ``snapshot`` never surfaced hard-block state, so a bare
   403 / vendor-challenge page was returned as ordinary "success" content.
2. The snapshot walk silently dropped every element past the 200-element
   cap, so an agent was confidently wrong about dense pages.

These tests pin the pure classifiers plus the navigate / snapshot
integration surfaces. No Docker / live browser needed — the page is
mocked, matching the harness in ``test_browser_service.py``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import (
    _MAX_SNAPSHOT_ELEMENTS,
    _MAX_SNAPSHOT_ELEMENTS_CEILING,
    BrowserManager,
    CamoufoxInstance,
    _classify_navigation_block,
    _snapshot_truncation_fields,
    _snapshot_truncation_notice,
)

_PROFILES_ROOT = "/tmp/ol_test_truthful"


# ── Pure classifier: _classify_navigation_block ──────────────────────────

class TestClassifyNavigationBlock:
    def test_cloudflare_hard_block_on_403(self):
        out = _classify_navigation_block({"cf-mitigated": "block"}, 403)
        assert out == {
            "status": 403,
            "vendor": "cloudflare",
            "signal": "cf-mitigated=block",
            "kind": "vendor_block",
        }

    def test_datadome_protected_is_vendor_block(self):
        out = _classify_navigation_block({"x-datadome": "protected"}, 403)
        assert out is not None
        assert out["vendor"] == "datadome"
        assert out["kind"] == "vendor_block"

    def test_bare_403_no_vendor_header_is_http_block(self):
        out = _classify_navigation_block({}, 403)
        assert out == {
            "status": 403,
            "vendor": None,
            "signal": "http_status=403",
            "kind": "http_block",
        }

    @pytest.mark.parametrize("status", [401, 403, 407, 429, 451, 503])
    def test_block_shaped_statuses_flagged(self, status):
        out = _classify_navigation_block({}, status)
        assert out is not None
        assert out["kind"] == "http_block"
        assert out["status"] == status

    def test_cf_soft_challenge_still_http_block_by_status(self):
        # ``cf-mitigated: challenge`` is NOT a hard vendor block, but a 403
        # is block-shaped on its own, so the agent must still be told.
        out = _classify_navigation_block({"cf-mitigated": "challenge"}, 403)
        assert out is not None
        assert out["kind"] == "http_block"
        assert out["vendor"] is None

    @pytest.mark.parametrize("status", [200, 204, 301, 302, 404, 410, 500])
    def test_content_and_ordinary_failures_not_flagged(self, status):
        # 404/410/500 are "not found" / "server error", not anti-bot walls.
        assert _classify_navigation_block({}, status) is None

    def test_vendor_header_on_200_is_not_a_block(self):
        # Vendor block headers can ride accepted-but-watched 200 responses;
        # the status floor keeps us from burning on those.
        assert _classify_navigation_block({"cf-mitigated": "block"}, 200) is None


# ── Pure helpers: truncation fields + notice ─────────────────────────────

class TestTruncationHelpers:
    def test_not_truncated_returns_empty(self):
        assert _snapshot_truncation_fields(shown=50, total=50, cap=200) == {}
        assert _snapshot_truncation_notice(shown=50, total=50, cap=200) == ""

    def test_truncated_reports_counts_and_hint(self):
        fields = _snapshot_truncation_fields(shown=200, total=340, cap=200)
        assert fields["truncated"] is True
        assert fields["shown_elements"] == 200
        assert fields["total_elements"] == 340
        assert fields["elements_omitted"] == 140
        assert "max_elements" in fields["truncation_hint"]
        assert str(_MAX_SNAPSHOT_ELEMENTS_CEILING) in fields["truncation_hint"]

    def test_notice_mentions_the_shortfall(self):
        notice = _snapshot_truncation_notice(shown=200, total=340, cap=200)
        assert "showing 200 of 340" in notice
        assert "140 not shown" in notice


# ── navigate() hard-block surfacing ──────────────────────────────────────

def _nav_page(status, headers):
    """A mock page whose goto returns a real Response-shaped stub."""
    page = AsyncMock()
    page.goto = AsyncMock(
        return_value=SimpleNamespace(status=status, headers=dict(headers)),
    )
    page.title = AsyncMock(return_value="Title")
    page.url = "https://example.com"
    page.evaluate = AsyncMock(return_value="page text")
    return page


class TestNavigateHardBlock:
    @pytest.mark.asyncio
    async def test_bare_403_surfaces_http_block(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/p1")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _nav_page(403, {}))
        mgr._instances["a1"] = inst

        res = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        # success stays True — the nav mechanically completed — but the
        # block is now explicit rather than masquerading as content.
        assert res["success"] is True
        assert res["data"]["http_status"] == 403
        assert res["hard_block"]["kind"] == "http_block"
        assert res["hard_block"]["status"] == 403
        assert inst.last_nav_hard_block == res["hard_block"]

    @pytest.mark.asyncio
    async def test_vendor_block_names_the_vendor(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/p2")
        page = _nav_page(403, {"cf-mitigated": "block"})
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), page)
        mgr._instances["a1"] = inst

        res = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert res["hard_block"]["kind"] == "vendor_block"
        assert res["hard_block"]["vendor"] == "cloudflare"

    @pytest.mark.asyncio
    async def test_clean_200_has_status_but_no_block(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/p3")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _nav_page(200, {}))
        mgr._instances["a1"] = inst

        res = await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert res["success"] is True
        assert res["data"]["http_status"] == 200
        assert "hard_block" not in res
        assert inst.last_nav_hard_block is None

    @pytest.mark.asyncio
    async def test_clean_nav_clears_prior_block(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/p4")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _nav_page(200, {}))
        # Pretend a previous nav had been blocked.
        inst.last_nav_hard_block = {
            "status": 403, "vendor": None,
            "signal": "http_status=403", "kind": "http_block",
        }
        mgr._instances["a1"] = inst

        await mgr.navigate("a1", "https://example.com", wait_ms=0)

        assert inst.last_nav_hard_block is None


# ── snapshot() truncation + hard-block echo ──────────────────────────────

def _snapshot_page(n_buttons):
    tree = {
        "role": "WebArea", "name": "",
        "children": [
            {"role": "button", "name": f"Btn {i}"} for i in range(n_buttons)
        ],
    }
    page = AsyncMock()
    page.url = "https://example.com"
    page.evaluate = AsyncMock(return_value=tree)
    page.query_selector_all = AsyncMock(return_value=[])
    page.viewport_size = {"width": 1280, "height": 800}
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=tree)
    return page


class TestSnapshotTruncation:
    @pytest.mark.asyncio
    async def test_dense_page_reports_truncation(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/s1")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _snapshot_page(250))
        mgr._instances["a1"] = inst

        res = await mgr.snapshot("a1", include_frames=False)
        data = res["data"]

        assert data["truncated"] is True
        assert data["shown_elements"] == _MAX_SNAPSHOT_ELEMENTS
        assert data["total_elements"] >= 250
        assert data["elements_omitted"] == data["total_elements"] - _MAX_SNAPSHOT_ELEMENTS
        assert len(data["refs"]) == _MAX_SNAPSHOT_ELEMENTS
        assert "truncated: showing 200 of" in data["snapshot"]

    @pytest.mark.asyncio
    async def test_max_elements_override_pulls_more(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/s2")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _snapshot_page(250))
        mgr._instances["a1"] = inst

        res = await mgr.snapshot("a1", include_frames=False, max_elements=300)
        data = res["data"]

        assert "truncated" not in data
        assert len(data["refs"]) >= 250

    @pytest.mark.asyncio
    async def test_max_elements_clamped_to_ceiling(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/s3")
        inst = CamoufoxInstance(
            "a1", MagicMock(), MagicMock(),
            _snapshot_page(_MAX_SNAPSHOT_ELEMENTS_CEILING + 50),
        )
        mgr._instances["a1"] = inst

        # Ask for way more than the ceiling — the effective cap is clamped.
        res = await mgr.snapshot("a1", include_frames=False, max_elements=99999)
        data = res["data"]

        assert data["truncated"] is True
        assert data["shown_elements"] == _MAX_SNAPSHOT_ELEMENTS_CEILING
        assert len(data["refs"]) == _MAX_SNAPSHOT_ELEMENTS_CEILING

    @pytest.mark.asyncio
    async def test_small_page_not_truncated(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/s4")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _snapshot_page(5))
        mgr._instances["a1"] = inst

        res = await mgr.snapshot("a1", include_frames=False)
        data = res["data"]

        assert "truncated" not in data
        assert "truncated: showing" not in data["snapshot"]

    @pytest.mark.asyncio
    async def test_snapshot_echoes_prior_nav_block(self):
        mgr = BrowserManager(profiles_dir=f"{_PROFILES_ROOT}/s5")
        inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), _snapshot_page(3))
        block = {
            "status": 403, "vendor": "datadome",
            "signal": "x-datadome=protected", "kind": "vendor_block",
        }
        inst.last_nav_hard_block = block
        mgr._instances["a1"] = inst

        res = await mgr.snapshot("a1", include_frames=False)

        # A snapshot taken while the page is a block must say so too.
        assert res["success"] is True
        assert res["hard_block"] == block
