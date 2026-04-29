"""Tests for the agent-side ``browser_warmup`` skill.

The skill composes existing ``navigate`` and ``scroll`` browser commands
into a believable pre-task browsing trail. Tests pin:

  * the registered skill schema
  * the per-intensity action shape (light=1 nav, normal=2 navs, deep=mix)
  * apex-host extraction (``www.linkedin.com/in/x`` → ``linkedin.com/``)
  * input validation (rejects non-http URLs without making nav calls)
  * recoverability (search-engine failure ⇒ warmup continues to apex)
  * apex failure ⇒ ``success=False``
  * the returned envelope shape (``steps``, ``total_ms``, ``target_apex``)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


def _make_mesh(side_effect=None, return_value=None):
    """Build a mesh client with a mocked ``browser_command``.

    ``side_effect`` takes priority; falls back to ``return_value`` (or a
    plain success dict). Caller can inspect ``.browser_command`` to assert
    on the call sequence.
    """
    mc = AsyncMock()
    if side_effect is not None:
        mc.browser_command = AsyncMock(side_effect=side_effect)
    else:
        mc.browser_command = AsyncMock(
            return_value=return_value or {"url": "https://example.com/"},
        )
    return mc


# ── schema ─────────────────────────────────────────────────────


class TestBrowserWarmupSchema:
    def test_skill_is_registered_with_expected_schema(self):
        # Force-import the module so the @skill decorator runs.
        import src.agent.builtins.browser_tool  # noqa: F401
        from src.agent.skills import _skill_staging

        assert "browser_warmup" in _skill_staging
        info = _skill_staging["browser_warmup"]

        assert info["name"] == "browser_warmup"
        assert "warmup" in info["description"].lower()
        assert info["_parallel_safe"] is False  # browser is per-agent, sequential

        params = info["parameters"]
        assert "target_url" in params
        assert params["target_url"]["type"] == "string"
        assert "intensity" in params
        assert params["intensity"]["type"] == "string"
        assert params["intensity"].get("default") == "normal"

    def test_required_params_only_target_url(self):
        from src.agent.skills import _skill_staging

        required = _skill_staging["browser_warmup"]["_sig_required_params"]
        # ``intensity`` has a default ⇒ not required; mesh_client is
        # keyword-only auto-injected and also has a default.
        assert "target_url" in required
        assert "intensity" not in required


# ── per-intensity action shape ─────────────────────────────────


class TestBrowserWarmupIntensities:
    @pytest.mark.asyncio
    async def test_light_intensity_makes_one_navigate_call(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/someone",
            intensity="light",
            mesh_client=mc,
        )

        # light = search engine only, no apex hop, no scroll.
        nav_calls = [
            call for call in mc.browser_command.await_args_list
            if call.args[0] == "navigate"
        ]
        scroll_calls = [
            call for call in mc.browser_command.await_args_list
            if call.args[0] == "scroll"
        ]
        assert len(nav_calls) == 1
        assert len(scroll_calls) == 0
        # The single nav targets a search engine, not the apex.
        nav_url = nav_calls[0].args[1]["url"]
        assert nav_url in (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )
        assert result["success"] is True
        assert result["intensity"] == "light"
        assert len(result["steps"]) == 1

    @pytest.mark.asyncio
    async def test_normal_intensity_makes_two_navigate_calls(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/someone",
            intensity="normal",
            mesh_client=mc,
        )

        nav_calls = [
            call for call in mc.browser_command.await_args_list
            if call.args[0] == "navigate"
        ]
        scroll_calls = [
            call for call in mc.browser_command.await_args_list
            if call.args[0] == "scroll"
        ]
        assert len(nav_calls) == 2
        assert len(scroll_calls) == 0
        # First call: search engine. Second call: apex host.
        first_url = nav_calls[0].args[1]["url"]
        second_url = nav_calls[1].args[1]["url"]
        assert first_url in (
            "https://www.google.com/",
            "https://www.bing.com/",
            "https://duckduckgo.com/",
        )
        assert second_url == "https://linkedin.com/"
        assert result["success"] is True
        assert result["target_apex"] == "https://linkedin.com/"

    @pytest.mark.asyncio
    async def test_normal_is_default_intensity(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/someone",
            mesh_client=mc,
        )
        assert result["intensity"] == "normal"
        nav_calls = [
            c for c in mc.browser_command.await_args_list
            if c.args[0] == "navigate"
        ]
        assert len(nav_calls) == 2

    @pytest.mark.asyncio
    async def test_deep_intensity_mixes_nav_and_scroll(self):
        from src.agent.builtins import browser_tool as bt
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        # Stub asyncio.sleep so the SERP dwell doesn't actually sleep.
        async def _fake_sleep(s):
            return None
        with patch.object(bt.asyncio, "sleep", _fake_sleep):
            result = await browser_warmup(
                target_url="https://www.linkedin.com/in/x",
                intensity="deep",
                mesh_client=mc,
            )

        action_kinds = [c.args[0] for c in mc.browser_command.await_args_list]
        # deep = search-nav, scroll, apex-nav, scroll-on-apex (the SERP
        # dwell is an asyncio.sleep, not a mesh action).
        assert action_kinds.count("navigate") == 2
        assert action_kinds.count("scroll") == 2
        assert len(action_kinds) == 4
        # Order matters: scroll comes after the search-engine nav, before
        # the apex nav, and another scroll comes after the apex nav.
        assert action_kinds == ["navigate", "scroll", "navigate", "scroll"]
        assert result["success"] is True
        # Step kinds in the returned envelope include the SERP dwell
        # between the search-engine scroll and the apex nav.
        step_kinds = [s["kind"] for s in result["steps"]]
        assert step_kinds == [
            "search_engine", "scroll", "serp_dwell", "apex", "scroll",
        ]

    @pytest.mark.asyncio
    async def test_unknown_intensity_falls_back_to_normal(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            intensity="bogus",
            mesh_client=mc,
        )
        assert result["intensity"] == "normal"
        nav_calls = [
            c for c in mc.browser_command.await_args_list
            if c.args[0] == "navigate"
        ]
        assert len(nav_calls) == 2


# ── apex-host extraction ───────────────────────────────────────


class TestApexExtraction:
    @pytest.mark.asyncio
    async def test_strips_www_and_path(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x?foo=1",
            mesh_client=mc,
        )
        assert result["target_apex"] == "https://linkedin.com/"

    @pytest.mark.asyncio
    async def test_preserves_scheme(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="http://example.com/foo",
            mesh_client=mc,
        )
        assert result["target_apex"] == "http://example.com/"

    @pytest.mark.asyncio
    async def test_keeps_subdomains_other_than_www(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://mobile.linkedin.com/foo",
            mesh_client=mc,
        )
        # Only ``www.`` is stripped; other subdomains stay (they're real
        # apex shapes for those services).
        assert result["target_apex"] == "https://mobile.linkedin.com/"


# ── input validation ───────────────────────────────────────────


class TestBrowserWarmupValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_http_url_without_navigating(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="ftp://example.com/foo",
            mesh_client=mc,
        )
        assert result == {"error": "invalid target_url"}
        # CRITICAL: no nav calls were made.
        mc.browser_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_empty_string_without_navigating(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(target_url="", mesh_client=mc)
        assert "error" in result
        mc.browser_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_javascript_url_without_navigating(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="javascript:alert(1)", mesh_client=mc,
        )
        assert "error" in result
        mc.browser_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_mesh_client_returns_error(self):
        from src.agent.builtins.browser_tool import browser_warmup

        result = await browser_warmup(
            target_url="https://linkedin.com/", mesh_client=None,
        )
        assert "error" in result


# ── recoverability ─────────────────────────────────────────────


class TestBrowserWarmupRecoverability:
    @pytest.mark.asyncio
    async def test_search_engine_failure_does_not_abort(self):
        """Search-engine timeout is recoverable — apex nav still runs."""
        from src.agent.builtins.browser_tool import browser_warmup

        # First call (search engine) raises; second call (apex) succeeds.
        side_effect = [
            RuntimeError("search engine timed out"),
            {"url": "https://linkedin.com/", "title": "LinkedIn"},
        ]
        mc = _make_mesh(side_effect=side_effect)
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            intensity="normal",
            mesh_client=mc,
        )

        # Apex nav was still attempted despite search-engine failure.
        nav_calls = [
            c for c in mc.browser_command.await_args_list
            if c.args[0] == "navigate"
        ]
        assert len(nav_calls) == 2
        assert nav_calls[1].args[1]["url"] == "https://linkedin.com/"

        # Apex nav succeeded ⇒ overall success is True.
        assert result["success"] is True
        # The search-engine step is recorded as failed.
        steps = result["steps"]
        search_step = next(s for s in steps if s["kind"] == "search_engine")
        assert search_step["ok"] is False
        assert "error" in search_step
        # The apex step is recorded as ok.
        apex_step = next(s for s in steps if s["kind"] == "apex")
        assert apex_step["ok"] is True

    @pytest.mark.asyncio
    async def test_search_engine_error_dict_does_not_abort(self):
        """``_browser_command`` returns ``{"error": ...}`` on transport
        failures (not exceptions). Same recoverability rule applies."""
        from src.agent.builtins.browser_tool import browser_warmup

        # The error dict comes back wrapped by ``_browser_command`` only
        # when ``mesh_client.browser_command`` itself raises. Simulating
        # that here — the second call succeeds.
        side_effect = [
            Exception("transport closed"),
            {"url": "ok"},
        ]
        mc = _make_mesh(side_effect=side_effect)
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            mesh_client=mc,
        )
        assert result["success"] is True


# ── apex failure ───────────────────────────────────────────────


class TestBrowserWarmupApexFailure:
    @pytest.mark.asyncio
    async def test_apex_navigation_failure_returns_success_false(self):
        from src.agent.builtins.browser_tool import browser_warmup

        # Search engine ok, apex hop fails.
        side_effect = [
            {"url": "search-engine-ok"},
            RuntimeError("apex unreachable"),
        ]
        mc = _make_mesh(side_effect=side_effect)
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            intensity="normal",
            mesh_client=mc,
        )

        assert result["success"] is False
        apex_step = next(s for s in result["steps"] if s["kind"] == "apex")
        assert apex_step["ok"] is False
        assert "error" in apex_step

    @pytest.mark.asyncio
    async def test_apex_skip_at_deep_when_apex_fails(self):
        """At deep intensity, if the apex nav fails, the apex-scroll
        step should NOT run (no point scrolling a non-loaded page)."""
        from src.agent.builtins import browser_tool as bt
        from src.agent.builtins.browser_tool import browser_warmup

        side_effect = [
            {"url": "search-engine-ok"},
            {"data": {}},  # search-engine scroll ok
            RuntimeError("apex unreachable"),
        ]
        mc = _make_mesh(side_effect=side_effect)
        # Stub the SERP dwell sleep so the test runs quickly.
        async def _fake_sleep(s):
            return None
        with patch.object(bt.asyncio, "sleep", _fake_sleep):
            result = await browser_warmup(
                target_url="https://www.linkedin.com/in/x",
                intensity="deep",
                mesh_client=mc,
            )

        assert result["success"] is False
        # Only 3 underlying calls — the post-apex scroll was skipped.
        assert mc.browser_command.await_count == 3
        kinds = [s["kind"] for s in result["steps"]]
        # search_engine + scroll + serp_dwell + apex (apex-failed);
        # no second scroll.
        assert kinds == ["search_engine", "scroll", "serp_dwell", "apex"]


# ── returned envelope shape ────────────────────────────────────


class TestBrowserWarmupEnvelope:
    @pytest.mark.asyncio
    async def test_returns_structured_envelope(self):
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        result = await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            mesh_client=mc,
        )

        # All required keys present.
        assert "success" in result
        assert "steps" in result
        assert "total_ms" in result
        assert "target_apex" in result
        assert "intensity" in result

        assert isinstance(result["success"], bool)
        assert isinstance(result["steps"], list)
        assert isinstance(result["total_ms"], int)
        assert isinstance(result["target_apex"], str)

        # ``total_ms`` is a non-negative duration.
        assert result["total_ms"] >= 0

        # Each step records ``kind`` and ``ok``.
        for step in result["steps"]:
            assert "kind" in step
            assert "ok" in step

    @pytest.mark.asyncio
    async def test_uses_lightweight_wait_until(self):
        """Warmup navs should use ``domcontentloaded`` (not ``load`` or
        ``networkidle``) since the goal is to LOAD the page, not wait
        for full SPA hydration. Keeps the warmup under 30s budget."""
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        await browser_warmup(
            target_url="https://www.linkedin.com/in/x",
            mesh_client=mc,
        )

        nav_calls = [
            c for c in mc.browser_command.await_args_list
            if c.args[0] == "navigate"
        ]
        for call in nav_calls:
            assert call.args[1]["wait_until"] == "domcontentloaded"


# ── 8: weighted search-engine selection ───────────────────────────────


class TestWeightedSearchEngine:
    def test_weighted_distribution_skews_to_google(self):
        """Real distribution is ~85% Google / ~10% Bing / ~5% DDG.
        Verify the picker honors the weights at fleet-scale by sampling
        many times with a seeded rng — Google should dominate."""
        import random as _random

        from src.agent.builtins.browser_tool import _pick_warmup_search_engine

        rng = _random.Random(42)
        counts = {"google": 0, "bing": 0, "duckduckgo": 0}
        for _ in range(2000):
            url = _pick_warmup_search_engine(rng=rng)
            for k in counts:
                if k in url:
                    counts[k] += 1
                    break
        # Google must be the dominant pick (85% target, ±5% slack).
        assert counts["google"] / 2000 > 0.78
        # Non-Google should be present but rare.
        assert counts["bing"] > 0
        assert counts["duckduckgo"] > 0
        # No unweighted clustering — Bing should not approach Google.
        assert counts["bing"] < counts["google"] / 2

    def test_picker_reproducible_with_seeded_rng(self):
        """Operator audit: same rng seed → same engine choice. Lets a
        canary run replay an exact warmup from logs."""
        import random as _random

        from src.agent.builtins.browser_tool import _pick_warmup_search_engine

        rng_a = _random.Random(123)
        rng_b = _random.Random(123)
        urls_a = [_pick_warmup_search_engine(rng=rng_a) for _ in range(20)]
        urls_b = [_pick_warmup_search_engine(rng=rng_b) for _ in range(20)]
        assert urls_a == urls_b


# ── 9: SERP dwell ─────────────────────────────────────────────────────


class TestSerpDwell:
    def test_sample_within_clamp_window(self):
        import random as _random

        from src.agent.builtins.browser_tool import _sample_serp_dwell

        rng = _random.Random(777)
        for _ in range(500):
            d = _sample_serp_dwell(rng=rng)
            assert 2.0 <= d <= 9.0  # _SERP_DWELL_MIN_S / MAX_S

    def test_sample_clamped_to_min(self):
        from src.agent.builtins.browser_tool import _sample_serp_dwell

        class _StubRng:
            def gauss(self, mu, sigma):
                return -10.0  # below min

        d = _sample_serp_dwell(rng=_StubRng())
        assert d == 2.0

    def test_sample_clamped_to_max(self):
        from src.agent.builtins.browser_tool import _sample_serp_dwell

        class _StubRng:
            def gauss(self, mu, sigma):
                return 999.0  # above max

        d = _sample_serp_dwell(rng=_StubRng())
        assert d == 9.0

    @pytest.mark.asyncio
    async def test_warmup_normal_intensity_includes_serp_dwell_step(self):
        """At ``normal`` intensity, after the search-engine nav
        succeeds, a ``serp_dwell`` step must appear in the warmup trace
        BEFORE the apex nav."""
        from src.agent.builtins import browser_tool as bt
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})

        # Stub asyncio.sleep so the test doesn't actually wait 4s.
        slept: list[float] = []

        async def _fake_sleep(s):
            slept.append(s)
        with patch.object(bt.asyncio, "sleep", _fake_sleep):
            result = await browser_warmup(
                target_url="https://linkedin.com/in/x",
                intensity="normal",
                mesh_client=mc,
            )

        kinds = [s["kind"] for s in result["steps"]]
        # Order: search_engine → serp_dwell → apex
        assert "serp_dwell" in kinds
        assert kinds.index("serp_dwell") < kinds.index("apex")
        # Sleep was actually called with a positive duration in the
        # SERP-dwell window.
        assert len(slept) >= 1
        assert 2.0 <= slept[0] <= 9.0

    @pytest.mark.asyncio
    async def test_warmup_skips_serp_dwell_when_search_failed(self):
        """SERP-dwell should NOT fire when the search-engine nav failed
        — pausing on a failed page produces no fingerprint benefit."""
        from src.agent.builtins import browser_tool as bt
        from src.agent.builtins.browser_tool import browser_warmup

        # First call (search) returns error; second call (apex) succeeds.
        mc = _make_mesh()
        mc.browser_command = AsyncMock(side_effect=[
            {"error": "search-engine timeout"},  # SERP nav fails
            {"url": "ok"},                       # apex nav succeeds
        ])

        slept: list[float] = []

        async def _fake_sleep(s):
            slept.append(s)
        with patch.object(bt.asyncio, "sleep", _fake_sleep):
            result = await browser_warmup(
                target_url="https://linkedin.com/in/x",
                intensity="normal",
                mesh_client=mc,
            )

        kinds = [s["kind"] for s in result["steps"]]
        assert "serp_dwell" not in kinds, (
            "SERP-dwell must skip when the search engine load failed"
        )
        # No asyncio.sleep should have been called for the SERP dwell.
        assert slept == []

    @pytest.mark.asyncio
    async def test_warmup_light_intensity_no_serp_dwell(self):
        """Light intensity = search-engine only, no apex hop, no SERP
        dwell (the dwell only matters as a hand-off pause to the apex
        nav)."""
        from src.agent.builtins import browser_tool as bt
        from src.agent.builtins.browser_tool import browser_warmup

        mc = _make_mesh(return_value={"url": "ok"})
        slept: list[float] = []

        async def _fake_sleep(s):
            slept.append(s)
        with patch.object(bt.asyncio, "sleep", _fake_sleep):
            result = await browser_warmup(
                target_url="https://linkedin.com/in/x",
                intensity="light",
                mesh_client=mc,
            )

        kinds = [s["kind"] for s in result["steps"]]
        assert "serp_dwell" not in kinds
        assert slept == []
