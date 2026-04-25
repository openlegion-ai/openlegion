"""Phase 3 §6.3 — navigator self-test probe."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_inst(monkeypatch=None):
    """CamoufoxInstance with an async-mocked ``page.evaluate``."""
    if monkeypatch is not None:
        monkeypatch.delenv("BROWSER_RECORD_BEHAVIOR", raising=False)
        import src.browser.flags as flags
        flags._operator_settings = None

    from src.browser.service import CamoufoxInstance

    page = MagicMock()
    page.evaluate = AsyncMock()
    inst = CamoufoxInstance("a1", MagicMock(), MagicMock(), page)
    return inst


def _ok_signals(os_hint="windows", **overrides):
    """A clean signals dict that should produce ok=True."""
    base = {
        "webdriver": False,
        "plugins_len": 4,
        "mimeTypes_len": 2,
        "hardwareConcurrency": 8,
        "deviceMemory": None,
        "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) "
                     "Gecko/20100101 Firefox/138.0",
        "platform": {"windows": "Win32", "macos": "MacIntel",
                     "linux": "Linux x86_64"}[os_hint],
        "language": "en-US",
        "timezone": "America/Los_Angeles",
        "conn_effective": "4g",
        "conn_downlink": 12.4,
        "conn_rtt": 55,
    }
    base.update(overrides)
    return base


class TestProbeOK:
    @pytest.mark.asyncio
    async def test_clean_browser_passes(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals()

        await mgr._run_navigator_probe(inst)

        assert inst.probe_result is not None
        assert inst.probe_result["ok"] is True
        assert inst.probe_result["mismatches"] == []


class TestProbeMismatches:
    @pytest.mark.asyncio
    async def test_webdriver_true_flags_mismatch(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals(webdriver=True)

        await mgr._run_navigator_probe(inst)
        assert inst.probe_result["ok"] is False
        assert any("webdriver" in m for m in inst.probe_result["mismatches"])

    @pytest.mark.asyncio
    async def test_platform_mismatch_flagged(self, monkeypatch, tmp_path):
        """os_hint=windows but platform=Linux x86_64 — fingerprint inconsistency."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        monkeypatch.setenv("BROWSER_OS", "windows")
        inst.page.evaluate.return_value = _ok_signals(platform="Linux x86_64")

        await mgr._run_navigator_probe(inst)
        assert inst.probe_result["ok"] is False
        assert any("platform" in m for m in inst.probe_result["mismatches"])

    @pytest.mark.asyncio
    async def test_non_firefox_ua_flagged(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals(
            userAgent="Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0",
        )

        await mgr._run_navigator_probe(inst)
        assert inst.probe_result["ok"] is False
        assert any("Firefox" in m for m in inst.probe_result["mismatches"])

    @pytest.mark.asyncio
    async def test_missing_navigator_connection_flagged(
        self, monkeypatch, tmp_path,
    ):
        """§6.6 spoof not landing → conn_effective null → mismatch."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals(conn_effective=None)

        await mgr._run_navigator_probe(inst)
        assert inst.probe_result["ok"] is False
        assert any("connection" in m for m in inst.probe_result["mismatches"])


class TestProbeResilience:
    @pytest.mark.asyncio
    async def test_evaluate_failure_records_result(self, monkeypatch, tmp_path):
        """A page.evaluate raising must not crash _start_browser. The
        failure is recorded as a probe result with ok=False so operators
        still see the diagnostic signal."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.side_effect = RuntimeError("page closed")

        await mgr._run_navigator_probe(inst)
        assert inst.probe_result is not None
        assert inst.probe_result["ok"] is False
        assert "evaluate failed" in inst.probe_result["mismatches"][0]


class TestProbeEmission:
    @pytest.mark.asyncio
    async def test_probe_payload_lands_in_history(self, monkeypatch, tmp_path):
        """Mesh poll consumes _metrics_history; nav_probe events must be
        written there even when no in-process sink is wired."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals()

        await mgr._run_navigator_probe(inst)

        snap = mgr.get_recent_metrics(since_seq=0)
        kinds = [p.get("kind") for p in snap["metrics"]]
        assert "nav_probe" in kinds

    @pytest.mark.asyncio
    async def test_probe_payload_calls_sink(self, monkeypatch, tmp_path):
        from src.browser.service import BrowserManager

        seen: list[dict] = []
        mgr = BrowserManager(
            profiles_dir=str(tmp_path / "profiles"),
            metrics_sink=seen.append,
        )
        inst = _make_inst(monkeypatch)
        inst.page.evaluate.return_value = _ok_signals(webdriver=True)

        await mgr._run_navigator_probe(inst)

        probe = next(p for p in seen if p.get("kind") == "nav_probe")
        assert probe["agent_id"] == inst.agent_id
        assert probe["ok"] is False
        assert any("webdriver" in m for m in probe["mismatches"])

    @pytest.mark.asyncio
    async def test_status_endpoint_includes_probe_summary(
        self, monkeypatch, tmp_path,
    ):
        """``/browser/{agent}/status`` must surface probe ok + mismatches
        so operators can spot-check without subscribing to events."""
        from src.browser.service import BrowserManager

        mgr = BrowserManager(profiles_dir=str(tmp_path / "profiles"))
        inst = _make_inst(monkeypatch)
        mgr._instances["a1"] = inst
        inst.page.evaluate.return_value = _ok_signals()
        inst.page.url = "https://example.com/"
        await mgr._run_navigator_probe(inst)

        st = await mgr.get_status("a1")
        assert st["probe_ok"] is True
        assert st["probe_mismatches"] == []
