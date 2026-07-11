"""Tests for the ``select_option`` browser action — native <select> support.

Covers:

  * Agent-tool forwarding — ``browser_select_option`` emits the
    ``select_option`` mesh action with the right params.
  * Manager primary path — ``locator.select_option`` is called with
    exactly the supplied kwarg (value/label/index) and the response
    echoes the returned selection + the FULL landed value set read back
    off the element (``el.selectedOptions``).
  * Manager error classification — an "is not a <select>" Playwright
    error is classified as ``not_a_select``; an explicit no-matching-
    option error is ``option_not_found``; a detached/not-attached
    element is ``ref_stale``; a bare timeout is the honest, ambiguous
    ``not_interactable`` (NOT option_not_found — the select may be
    hidden/disabled/loading); missing value/label/index is rejected as
    ``invalid_input`` before any locator resolution; supplying more than
    one of value/label/index is also ``invalid_input``.
  * Post-select verification — the landed set is read via
    ``locator.evaluate`` (every currently-selected option value, not the
    scalar ``input_value()`` that only reports the FIRST). When it does
    not match the requested selection multiplicity-preservingly (a change
    handler reverted one or more options, single OR multi-select) the
    result is ``selection_reverted``, not a false success. A guarded read
    off a non-select element (``select_option`` redirected through a
    ``<label>``) yields ``null`` — INCONCLUSIVE, so it trusts Playwright's
    ``selected`` and succeeds rather than fabricating a failure. A
    legitimate clear (empty ``selected`` AND empty landed set) is a
    success too. A read-back that RAISES is never a success — a detached
    element → ``ref_stale``, anything else → ``service_unavailable``.

Mirrors ``tests/test_browser_parity_actions.py`` / ``tests/test_browser_captcha_redetect.py``:
every test drives ``BrowserManager`` handler methods directly against a
real ``CamoufoxInstance`` wrapping mocked Playwright objects (patching
``get_or_start``) — no Docker, no real browser.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser.service import BrowserManager, CamoufoxInstance


def _mk_page(*, url: str = "https://example.com"):
    """Playwright-shaped page mock — ``evaluate`` backs the
    ``_with_captcha_redetect`` install/read-back calls that
    ``select_option`` is wrapped in (mirrors ``type_text``). Returning a
    non-list keeps the wrapper on its fast "no captcha-shaped mutation"
    path with no further side effects.
    """
    page = MagicMock()
    page.url = url
    page.evaluate = AsyncMock(return_value=[])
    return page


def _mk_inst(page=None, agent_id: str = "agent-select") -> CamoufoxInstance:
    page = page if page is not None else _mk_page()
    return CamoufoxInstance(agent_id, MagicMock(), MagicMock(), page)


def _make_manager() -> BrowserManager:
    root = tempfile.mkdtemp(prefix="ol_select_option_")
    return BrowserManager(profiles_dir=root)


def _patch_get_or_start(mgr: BrowserManager, inst: CamoufoxInstance) -> None:
    mgr.get_or_start = AsyncMock(return_value=inst)  # type: ignore[assignment]


_UNSET = object()


def _mk_locator(
    *,
    select_option_result=None,
    select_option_error=None,
    landed_values=_UNSET,
    evaluate_error=None,
):
    """Playwright ``Locator`` mock.

    ``select_option`` returns the requested option *values* (or raises
    ``select_option_error``). The post-select verification then reads the
    landed set via ``locator.evaluate`` — mocked here to return
    ``landed_values`` (or raise ``evaluate_error``). When ``landed_values``
    is left unset it MIRRORS ``select_option_result`` — i.e. the value stuck,
    the common success case — so callers only pass it to model a revert.
    Pass ``landed_values=None`` EXPLICITLY to model the located element having
    no ``selectedOptions`` (a label-redirect target — the read is inconclusive
    and the guard returns ``null``).
    """
    loc = MagicMock()
    if select_option_error is not None:
        loc.select_option = AsyncMock(side_effect=select_option_error)
    else:
        loc.select_option = AsyncMock(return_value=select_option_result or [])
    if evaluate_error is not None:
        loc.evaluate = AsyncMock(side_effect=evaluate_error)
    else:
        if landed_values is _UNSET:
            landed_values = list(select_option_result or [])
        loc.evaluate = AsyncMock(return_value=landed_values)
    return loc


# ── Manager: primary (native <select>) path ────────────────────────────────


class TestSelectOptionPrimaryPath:
    @pytest.mark.asyncio
    async def test_selects_by_value_and_returns_selected(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(select_option_result=["red"])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="red")

        assert result["success"] is True
        assert result["data"]["selected"] == ["red"]
        # ``value`` now carries the FULL landed set read off the element.
        assert result["data"]["value"] == ["red"]
        assert result["data"]["ref"] == "e1"
        loc.select_option.assert_awaited_once_with(value="red")

    @pytest.mark.asyncio
    async def test_selects_by_label(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # Playwright's select_option returns option *values*; the element's
        # selectedOptions report the same value — keep the mock self-consistent
        # (landed_values mirrors select_option_result) so the post-select
        # verification sees the value stuck.
        loc = _mk_locator(select_option_result=["blue"])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", label="Blue")

        assert result["success"] is True
        loc.select_option.assert_awaited_once_with(label="Blue")

    @pytest.mark.asyncio
    async def test_selects_by_index(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(select_option_result=["opt3"])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", index=2)

        assert result["success"] is True
        loc.select_option.assert_awaited_once_with(index=2)

    @pytest.mark.asyncio
    async def test_selector_path_used_when_no_ref(self):
        mgr = _make_manager()
        page = _mk_page()
        loc = _mk_locator(select_option_result=["blue"])
        locator_chain = MagicMock()
        locator_chain.first = loc
        page.locator = MagicMock(return_value=locator_chain)
        inst = _mk_inst(page=page)
        _patch_get_or_start(mgr, inst)

        result = await mgr.select_option("agent-select", selector="#color", value="blue")

        assert result["success"] is True
        page.locator.assert_called_once_with("#color")
        loc.select_option.assert_awaited_once_with(value="blue")


# ── Manager: validation ─────────────────────────────────────────────────────


class TestSelectOptionValidation:
    @pytest.mark.asyncio
    async def test_missing_value_label_index_returns_invalid_input(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)

        result = await mgr.select_option("agent-select", ref="e1")

        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_missing_ref_and_selector_rejected(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)

        result = await mgr.select_option("agent-select", value="red")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_multiple_selectors_rejected_as_invalid_input(self):
        """Playwright would treat value+label as two candidate matchers and
        select whichever DOM option matches EITHER — non-deterministic. The
        manager must reject 2+ selectors before touching the locator."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()
        mgr._locator_from_ref = AsyncMock(return_value=_mk_locator())  # type: ignore[assignment]

        result = await mgr.select_option(
            "agent-select", ref="e1", value="US", label="Canada",
        )

        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


# ── Manager: error classification ───────────────────────────────────────────


class TestSelectOptionNotASelect:
    @pytest.mark.asyncio
    async def test_non_select_element_returns_not_a_select(self):
        """A Playwright "is not a <select>" error lands on the classified
        ``not_a_select`` directive — there is no keyboard fallback."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_error=Exception("Element is not a <select> element."),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="red")

        assert result["success"] is False
        assert result["error"]["code"] == "not_a_select"

    @pytest.mark.asyncio
    async def test_detached_element_classified_as_ref_stale(self):
        """"Element is not attached to the DOM" matches the too-broad
        "element is not a" prefix — it must map to ``ref_stale`` (re-snapshot),
        NOT ``not_a_select``."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_error=Exception(
                "Element is not attached to the DOM",
            ),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="red")

        assert result["success"] is False
        assert result["error"]["code"] == "ref_stale"


class TestSelectOptionNotFound:
    @pytest.mark.asyncio
    async def test_bare_timeout_classified_as_not_interactable(self):
        """A bare Playwright timeout is AMBIGUOUS — a hidden/disabled/detached
        select produces the SAME error as a missing option. It must NOT be
        reported as ``option_not_found``; the honest code is
        ``not_interactable``."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_error=Exception("Timeout 30000ms exceeded."),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="ghost")

        assert result["success"] is False
        assert result["error"]["code"] == "not_interactable"

    @pytest.mark.asyncio
    async def test_did_not_find_some_options_classified_as_option_not_found(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_error=Exception("did not find some options"),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", label="ghost")

        assert result["success"] is False
        assert result["error"]["code"] == "option_not_found"


class TestSelectOptionRevertedValue:
    @pytest.mark.asyncio
    async def test_reverted_value_classified_as_selection_reverted(self):
        """Playwright returns the requested option in ``selected`` but a page
        change-handler reset the DOM value, so the read-back set shows the OLD
        value. A contradictory selected/landed pair must NOT report success
        (single-select regression)."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # select_option returns ["US"] but the select snapped back to "CA".
        loc = _mk_locator(select_option_result=["US"], landed_values=["CA"])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="US")

        assert result["success"] is False
        assert result["error"]["code"] == "selection_reverted"

    @pytest.mark.asyncio
    async def test_empty_landed_set_classified_as_selection_reverted(self):
        """A change handler that clears the selection entirely leaves nothing
        selected — an empty read-back set is a full revert, never a success."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(select_option_result=["US"], landed_values=[])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="US")

        assert result["success"] is False
        assert result["error"]["code"] == "selection_reverted"


class TestSelectOptionMultiSelect:
    """The bug the OLD scalar ``input_value()`` check missed: a multi-select
    whose change handler reverts option 2..N. ``input_value()`` reports only
    the FIRST selected value, so it saw the (still-present) first option and
    reported a false success; the full-set read-back catches the drop."""

    @pytest.mark.asyncio
    async def test_multiselect_partial_revert_is_selection_reverted(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # Requested US+CA+MX; the page's change handler dropped CA and MX,
        # leaving only US selected. The old scalar guard (first value == "US",
        # which IS in ``selected``) would have reported success.
        loc = _mk_locator(
            select_option_result=["US", "CA", "MX"], landed_values=["US"],
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option(
            "agent-select", ref="e1", value=["US", "CA", "MX"],
        )

        assert result["success"] is False
        assert result["error"]["code"] == "selection_reverted"
        # The honest current state is surfaced (only US survived).
        assert "'US'" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_multiselect_fully_applied_is_success(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # Landed set matches the requested set regardless of order — the
        # comparison is order-insensitive and multiplicity-preserving.
        loc = _mk_locator(
            select_option_result=["US", "CA", "MX"],
            landed_values=["MX", "US", "CA"],
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option(
            "agent-select", ref="e1", value=["US", "CA", "MX"],
        )

        assert result["success"] is True
        assert result["data"]["selected"] == ["US", "CA", "MX"]
        assert sorted(result["data"]["value"]) == ["CA", "MX", "US"]


class TestSelectOptionReadBackRaises:
    """A read-back that RAISES is never a success — the old ``except: pass``
    silently skipped verification and returned success. Now it is classified."""

    @pytest.mark.asyncio
    async def test_read_back_not_attached_is_ref_stale(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_result=["US"],
            evaluate_error=Exception("Element is not attached to the DOM"),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="US")

        assert result["success"] is False
        assert result["error"]["code"] == "ref_stale"

    @pytest.mark.asyncio
    async def test_read_back_generic_error_is_service_unavailable(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(
            select_option_result=["US"],
            evaluate_error=RuntimeError("evaluate blew up"),
        )
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="US")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"


class TestSelectOptionInconclusiveReadBack:
    """``select_option`` can redirect through a ``<label>`` to its associated
    control, so the located element isn't guaranteed to BE the ``<select>``. A
    read-back off a non-select element has no ``selectedOptions`` — the guarded
    evaluate returns ``None``. That is INCONCLUSIVE, not a failure: trust
    Playwright's ``selected`` rather than fabricate a revert."""

    @pytest.mark.asyncio
    async def test_null_read_back_is_success_not_service_unavailable(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # Playwright reports the option selected; the located element has no
        # selectedOptions (label wrapper) so the read-back is None.
        loc = _mk_locator(select_option_result=["red"], landed_values=None)
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="red")

        assert result["success"] is True
        assert result["data"]["selected"] == ["red"]
        # An inconclusive read falls back to Playwright's returned selection.
        assert result["data"]["value"] == ["red"]


class TestSelectOptionLegitimateClear:
    """A legitimate clear — deselect everything — returns an EMPTY ``selected``
    AND an empty landed set. The two are equal, so it is a success; the guard
    must NOT misflag ``landed_values == []`` as a revert on its own."""

    @pytest.mark.asyncio
    async def test_empty_selection_matching_empty_landed_is_success(self):
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        # select_option([]) deselects all in a multi-select — nothing requested
        # to stick, nothing landed.
        loc = _mk_locator(select_option_result=[], landed_values=[])
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value=[])

        assert result["success"] is True
        assert result["data"]["selected"] == []
        assert result["data"]["value"] == []


class TestSelectOptionUnclassifiedErrorPropagates:
    @pytest.mark.asyncio
    async def test_unrelated_error_falls_to_service_unavailable(self):
        """An exception that matches neither classifier re-raises out of
        the body and is caught by the outer handler as ``service_unavailable``."""
        mgr = _make_manager()
        inst = _mk_inst()
        _patch_get_or_start(mgr, inst)
        inst.refs["e1"] = object()

        loc = _mk_locator(select_option_error=RuntimeError("boom, unrelated"))
        mgr._locator_from_ref = AsyncMock(return_value=loc)  # type: ignore[assignment]

        result = await mgr.select_option("agent-select", ref="e1", value="red")

        assert result["success"] is False
        assert result["error"]["code"] == "service_unavailable"


# ── Agent-side @tool forwarding ────────────────────────────────────────────


def _mesh(action_capture: dict):
    mc = MagicMock()

    async def _cmd(action, params):
        action_capture["action"] = action
        action_capture["params"] = params
        return {"success": True}

    mc.browser_command = AsyncMock(side_effect=_cmd)
    return mc


class TestBrowserSelectOptionTool:
    @pytest.mark.asyncio
    async def test_emits_select_option_action_with_ref_and_value(self):
        from src.agent.builtins.browser_tool import browser_select_option

        cap: dict = {}
        await browser_select_option(ref="e1", value="red", mesh_client=_mesh(cap))

        assert cap["action"] == "select_option"
        assert cap["params"] == {
            "ref": "e1",
            "selector": "",
            "snapshot_after": False,
            "value": "red",
        }

    @pytest.mark.asyncio
    async def test_emits_label_and_selector(self):
        from src.agent.builtins.browser_tool import browser_select_option

        cap: dict = {}
        await browser_select_option(
            selector="#color", label="Red", mesh_client=_mesh(cap),
        )

        assert cap["action"] == "select_option"
        assert cap["params"] == {
            "ref": "",
            "selector": "#color",
            "snapshot_after": False,
            "label": "Red",
        }

    @pytest.mark.asyncio
    async def test_emits_index(self):
        from src.agent.builtins.browser_tool import browser_select_option

        cap: dict = {}
        await browser_select_option(ref="e1", index=2, mesh_client=_mesh(cap))

        assert cap["params"]["index"] == 2

    @pytest.mark.asyncio
    async def test_missing_ref_and_selector_returns_error(self):
        from src.agent.builtins.browser_tool import browser_select_option

        result = await browser_select_option(value="red")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_missing_value_label_index_returns_error(self):
        from src.agent.builtins.browser_tool import browser_select_option

        result = await browser_select_option(ref="e1")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_multiple_selectors_returns_error(self):
        """value + label together is non-deterministic in Playwright — the
        tool must reject it before emitting a mesh action."""
        from src.agent.builtins.browser_tool import browser_select_option

        cap: dict = {}
        result = await browser_select_option(
            ref="e1", value="US", label="Canada", mesh_client=_mesh(cap),
        )
        assert "error" in result
        assert "not multiple" in result["error"]
        # No mesh action should have been emitted.
        assert cap == {}
