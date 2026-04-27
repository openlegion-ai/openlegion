"""Tests for BrowserManager.fill_form (Phase 6 §9.4).

Compound find-text + fill across a sequence of form fields with
CAPTCHA-mid-flow partial-success protocol.

Style mirrors :class:`tests.test_browser_service.TestFindText` —
mocks ``_locator_from_ref``, ``_check_captcha``, ``_snapshot_impl``
on the manager, and seeds ``inst.refs`` directly.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.browser.ref_handle import from_legacy_dict as _h
from src.browser.service import BrowserManager


def _make_manager() -> BrowserManager:
    """Build a BrowserManager without invoking __init__ side effects."""
    from src.browser.redaction import CredentialRedactor
    mgr = BrowserManager.__new__(BrowserManager)
    mgr.redactor = CredentialRedactor()
    return mgr


def _make_instance(refs_dict: dict | None = None) -> MagicMock:
    """Build a mock CamoufoxInstance with the given ref-id → ref-dict map."""
    inst = MagicMock()
    inst.refs = {k: _h(v) for k, v in (refs_dict or {}).items()}
    inst.page = AsyncMock()
    inst.page.viewport_size = {"width": 1280, "height": 720}
    inst.lock = asyncio.Lock()
    inst.touch = MagicMock()
    inst._user_control = False
    return inst


def _make_locator(*, raise_on_fill: Exception | None = None) -> AsyncMock:
    """Locator stub: visible, in-viewport, fill/press succeed unless overridden."""
    loc = AsyncMock()
    loc.is_visible = AsyncMock(return_value=True)
    loc.bounding_box = AsyncMock(
        return_value={"x": 10, "y": 10, "width": 50, "height": 20},
    )
    loc.scroll_into_view_if_needed = AsyncMock()
    if raise_on_fill is not None:
        loc.fill = AsyncMock(side_effect=raise_on_fill)
    else:
        loc.fill = AsyncMock(return_value=None)
    loc.press = AsyncMock(return_value=None)
    return loc


async def _fake_snapshot(self_mgr, _inst, _agent_id, **_kw):
    return {"success": True, "data": {}}


class TestFillFormHappyPath:
    """End-to-end happy path: every field found and filled."""

    @pytest.mark.asyncio
    async def test_three_fields_all_filled(self):
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
            "e2": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "alice@example.com"},
                    {"label": "Password", "value": "hunter2"},
                    {"label": "Phone", "value": "555-1212"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert data["partial_success"] is False
        assert data["captcha_required"] is False
        assert len(data["filled"]) == 3
        assert all(f["status"] == "filled" for f in data["filled"])
        assert data["remaining"] == []
        assert data["submitted"] is False
        assert loc.fill.await_count == 3
        assert loc.press.await_count == 0

    @pytest.mark.asyncio
    async def test_top_level_submit_after_presses_enter_on_last_field(self):
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw"},
                ],
                submit_after=True,
            )

        assert result["success"] is True
        assert result["data"]["submitted"] is True
        # Enter pressed exactly once at end.
        assert loc.press.await_count == 1
        loc.press.assert_awaited_with("Enter")


class TestFillFormCaptchaMidFlow:
    """CAPTCHA detection mid-flow → partial_success envelope."""

    @pytest.mark.asyncio
    async def test_captcha_after_first_field_stops_loop(self):
        from src.browser.service import _with_legacy_fields
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
            "e2": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()
        # §11.13 envelope shape — ``_check_captcha`` always returns this.
        captcha_envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "low",
            "next_action": "notify_user",
        }

        # Captcha appears after the first fill; subsequent calls would also
        # report it, but the loop should bail before the second find_text.
        check_captcha_mock = AsyncMock(return_value=captcha_envelope)

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new=check_captcha_mock), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw"},
                    {"label": "Phone", "value": "555"},
                ],
                submit_after=True,  # Must NOT be honored on captcha exit.
            )

        assert result["success"] is True
        data = result["data"]
        assert data["partial_success"] is True
        assert data["captcha_required"] is True
        assert len(data["filled"]) == 1
        assert data["filled"][0]["status"] == "filled"
        assert data["filled"][0]["label"] == "Email"
        # Both un-attempted fields echoed back verbatim with full value.
        assert len(data["remaining"]) == 2
        assert data["remaining"][0] == {
            "label": "Password", "value": "pw", "submit_after": False,
        }
        assert data["remaining"][1] == {
            "label": "Phone", "value": "555", "submit_after": False,
        }
        # Captcha echoed as the §11.13 envelope with legacy ``type`` /
        # ``message`` fields appended for back-compat.
        assert data["captcha"] == _with_legacy_fields(captcha_envelope)
        assert data["captcha"]["captcha_found"] is True
        assert data["captcha"]["kind"] == "recaptcha-v2-checkbox"
        assert data["submitted"] is False
        # Only one fill should have happened — the loop bailed.
        assert loc.fill.await_count == 1

    @pytest.mark.asyncio
    async def test_per_field_submit_then_captcha_reports_submitted_true(self):
        """Per-field Enter triggers the CAPTCHA → ``submitted=True``.

        The plan §9.4 reviewer flagged this: pressing Enter often IS
        what triggers the CAPTCHA, and if the press succeeded the form
        may already have been submitted with partial data. The captcha
        envelope must report ``submitted: true`` so the agent doesn't
        blindly re-type the remaining fields against a navigated page.
        """
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()
        captcha_envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "low",
            "next_action": "notify_user",
        }

        # No captcha after fill, captcha appears AFTER per-field Enter.
        # _check_captcha is called once per field (after fill+optional Enter).
        check_captcha_mock = AsyncMock(return_value=captcha_envelope)

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new=check_captcha_mock), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co", "submit_after": True},
                    {"label": "Password", "value": "pw"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert data["captcha_required"] is True
        # Critical: the per-field Enter pressed → submitted=True so the
        # agent knows the form was already submitted with partial data.
        assert data["submitted"] is True
        # Enter was pressed exactly once (the per-field one).
        assert loc.press.await_count == 1
        loc.press.assert_awaited_with("Enter")
        # Remaining still echoes Password un-attempted.
        assert len(data["remaining"]) == 1
        assert data["remaining"][0]["label"] == "Password"

    @pytest.mark.asyncio
    async def test_captcha_before_any_fill_yields_empty_filled(self):
        """find_text returns no matches; captcha then detected → no fills."""
        mgr = _make_manager()
        # No matching refs — find_text returns empty matches for "Email".
        refs = {
            "e0": {"role": "button", "name": "Cancel", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()
        captcha_envelope = {
            "captcha_found": True,
            "kind": "hcaptcha",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "high",
            "next_action": "notify_user",
        }

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value=captcha_envelope), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert data["partial_success"] is True
        assert data["captcha_required"] is True
        # Email got attempted (and recorded as not_found) BEFORE captcha
        # check; Password was never touched.
        assert len(data["filled"]) == 1
        assert data["filled"][0]["status"] == "not_found"
        assert data["filled"][0]["label"] == "Email"
        assert data["remaining"] == [
            {"label": "Password", "value": "pw", "submit_after": False},
        ]
        assert loc.fill.await_count == 0


class TestFillFormFieldFailures:
    """Per-field failure modes — loop continues, doesn't abort."""

    @pytest.mark.asyncio
    async def test_field_not_found_continues_loop(self):
        """Field 1 has no matching label; field 2 fills successfully."""
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Phone", "value": "555"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        # partial_success because of the not_found, but no captcha.
        assert data["partial_success"] is True
        assert data["captcha_required"] is False
        statuses = [f["status"] for f in data["filled"]]
        assert statuses == ["not_found", "filled"]
        assert data["remaining"] == []
        assert loc.fill.await_count == 1

    @pytest.mark.asyncio
    async def test_prefers_textbox_over_matching_label_text(self):
        """When both a label element and its textbox match, fill textbox."""
        mgr = _make_manager()
        refs = {
            # DOM/snapshot order often sees the visible <label> first.
            "e0": {"role": "text", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        label_loc = _make_locator(raise_on_fill=RuntimeError("not fillable"))
        input_loc = _make_locator()

        async def fake_locator(self_mgr, _inst, ref_id):
            if ref_id == "e0":
                return label_loc
            return input_loc

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref", new=fake_locator), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": "a@b.co"}],
            )

        assert result["success"] is True
        assert result["data"]["filled"][0]["status"] == "filled"
        assert result["data"]["filled"][0]["ref"] == "e1"
        label_loc.fill.assert_not_awaited()
        input_loc.fill.assert_awaited_once_with("a@b.co")

    @pytest.mark.asyncio
    async def test_locator_from_ref_returns_none_marks_type_failed(self):
        """find_text yields a match, but locator resolution returns None.

        ``_locator_from_ref`` is called twice for the email field — once by
        ``_find_text_impl`` (visibility check) and again by ``fill_form``
        (to drive the fill). Returning ``None`` for "e0" on every call
        models a fully-stale ref. find_text gracefully tolerates None
        (in_viewport falls to False), and the fill_form path then sees
        the None and reports type_failed.
        """
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        good_loc = _make_locator()

        async def fake_locator(self_mgr, _inst, ref_id):
            if ref_id == "e0":
                return None
            return good_loc

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref", new=fake_locator), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Phone", "value": "555"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert data["filled"][0]["status"] == "type_failed"
        assert data["filled"][0]["reason"] == "ref not found"
        assert data["filled"][1]["status"] == "filled"

    @pytest.mark.asyncio
    async def test_fill_raises_timeout_marks_type_failed_and_continues(self):
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        good_loc = _make_locator()
        bad_loc = _make_locator(raise_on_fill=TimeoutError("element detached"))

        # _locator_from_ref is called by both find_text (for in-viewport
        # check) and the fill_form body itself, so route by ref-id rather
        # than call order.
        loc_map = {"e0": bad_loc, "e1": good_loc}

        async def fake_locator(self_mgr, _inst, ref_id):
            return loc_map[ref_id]

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref", new=fake_locator), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Phone", "value": "555"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert data["filled"][0]["status"] == "type_failed"
        # Structured reason code (third-pass review §9.4 concern 1) — the
        # exception type is ``TimeoutError`` so the classifier returns
        # ``"timeout"``. The original message ("element detached") is
        # preserved in ``detail`` so an operator can debug.
        assert data["filled"][0]["reason"] == "timeout"
        assert "detached" in data["filled"][0]["detail"]
        assert data["filled"][1]["status"] == "filled"


class TestFillFormValidation:
    """Input validation — rejected with invalid_input before any browser work."""

    @pytest.mark.asyncio
    async def test_empty_fields_rejected(self):
        mgr = _make_manager()
        with patch.object(BrowserManager, "get_or_start") as get_start:
            result = await mgr.fill_form("agent1", [])
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        # Should never reach get_or_start.
        get_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_too_many_fields_rejected(self):
        mgr = _make_manager()
        fields = [
            {"label": f"Field{i}", "value": "v"} for i in range(51)
        ]
        result = await mgr.fill_form("agent1", fields)
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "50" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_value_with_null_byte_rejected(self):
        mgr = _make_manager()
        result = await mgr.fill_form(
            "agent1",
            [{"label": "Email", "value": "a\x00b"}],
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "null byte" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_value_too_long_rejected(self):
        mgr = _make_manager()
        result = await mgr.fill_form(
            "agent1",
            [{"label": "Email", "value": "x" * 10001}],
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"
        assert "10000" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_label_empty_rejected(self):
        mgr = _make_manager()
        result = await mgr.fill_form(
            "agent1",
            [{"label": "", "value": "v"}],
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_label_too_long_rejected(self):
        mgr = _make_manager()
        result = await mgr.fill_form(
            "agent1",
            [{"label": "x" * 501, "value": "v"}],
        )
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"

    @pytest.mark.asyncio
    async def test_non_dict_field_rejected(self):
        mgr = _make_manager()
        result = await mgr.fill_form("agent1", ["not a dict"])
        assert result["success"] is False
        assert result["error"]["code"] == "invalid_input"


class TestFillFormUserControl:
    """User VNC control pauses agent input — return conflict."""

    @pytest.mark.asyncio
    async def test_user_control_returns_conflict(self):
        mgr = _make_manager()
        inst = _make_instance({})
        inst._user_control = True

        with patch.object(BrowserManager, "get_or_start", return_value=inst):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": "a@b.co"}],
            )

        assert result["success"] is False
        assert result["error"]["code"] == "conflict"


class TestFillFormPerFieldSubmit:
    """Per-field submit_after presses Enter without aborting the loop."""

    @pytest.mark.asyncio
    async def test_per_field_submit_does_not_stop_loop(self):
        """Field 2 of 3 has submit_after=True; field 3 still gets filled."""
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
            "e2": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw", "submit_after": True},
                    {"label": "Phone", "value": "555"},
                ],
            )

        assert result["success"] is True
        data = result["data"]
        assert all(f["status"] == "filled" for f in data["filled"])
        assert len(data["filled"]) == 3
        assert loc.fill.await_count == 3
        # Per-field submit on field 2 → one Enter press during the loop.
        # Top-level submit_after was False so no extra Enter at end.
        assert loc.press.await_count == 1
        loc.press.assert_awaited_with("Enter")


class TestFillFormResume:
    """Two-call resume: captcha → solver → resume with `remaining`."""

    @pytest.mark.asyncio
    async def test_resume_after_captcha_completes_form(self):
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
            "e1": {"role": "textbox", "name": "Password", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()
        captcha_envelope = {
            "captcha_found": True,
            "kind": "recaptcha-v2-checkbox",
            "solver_attempted": False,
            "solver_outcome": "no_solver",
            "injection_failure_reason": None,
            "solver_confidence": "low",
            "next_action": "notify_user",
        }
        no_captcha = {"captcha_found": False}

        # First call: captcha appears after Email fill.
        check_seq = iter([captcha_envelope, no_captcha, no_captcha])

        async def fake_check(_self, _inst):
            return next(check_seq)

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha", new=fake_check), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            first = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw"},
                ],
            )
            assert first["data"]["captcha_required"] is True
            remaining = first["data"]["remaining"]
            assert len(remaining) == 1

            # Resume — pass `remaining` straight back. This mirrors what the
            # agent would do after solving the CAPTCHA.
            second = await mgr.fill_form("agent1", remaining)

        assert second["success"] is True
        assert second["data"]["captcha_required"] is False
        assert second["data"]["partial_success"] is False
        assert len(second["data"]["filled"]) == 1
        assert second["data"]["filled"][0]["status"] == "filled"
        assert second["data"]["filled"][0]["label"] == "Password"
        # First call: 1 fill (Email). Second call: 1 fill (Password). Total 2.
        assert loc.fill.await_count == 2


class TestFillFormConcurrency:
    """Lock serialization — concurrent fill_form calls don't interleave."""

    @pytest.mark.asyncio
    async def test_concurrent_calls_serialize_via_lock(self):
        """Two parallel fill_form calls execute strictly one after another.

        Verifies the §2.4 lock invariant: ``inst.lock`` is the single
        serialization point. Whichever call grabs the lock first
        completes before the other proceeds — no interleaving of
        find_text and fill across the two flows.
        """
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        # Track entry/exit ordering of fill() calls. If the lock works,
        # we should see [enter, exit, enter, exit] not interleaved.
        events: list[str] = []

        async def slow_fill(_value):
            events.append("enter")
            await asyncio.sleep(0.05)
            events.append("exit")

        loc.fill = AsyncMock(side_effect=slow_fill)

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            r1, r2 = await asyncio.gather(
                mgr.fill_form("agent1", [{"label": "Email", "value": "a"}]),
                mgr.fill_form("agent1", [{"label": "Email", "value": "b"}]),
            )

        assert r1["success"] is True
        assert r2["success"] is True
        # Strict ordering — never enter/enter back-to-back.
        assert events == ["enter", "exit", "enter", "exit"]


class TestFillFormActionRegistered:
    """Action name must appear in the mesh-side known-action set."""

    def test_fill_form_in_known_browser_actions(self):
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "fill_form" in KNOWN_BROWSER_ACTIONS, (
            "fill_form missing from KNOWN_BROWSER_ACTIONS in "
            "host/permissions.py — browser_fill_form skill will silently "
            "fail with a 400 from the mesh proxy."
        )


# ── Third-pass review additions (§9.4 concerns 1, 6, 8, 10, 13, 15) ─────


class TestFillFormErrorClassification:
    """Concern 1 — Playwright fill errors classified into structured codes."""

    @pytest.mark.asyncio
    async def test_classify_not_fillable(self):
        """Find_text returned a <label> ref → fill() raises 'not an <input>'.

        Without classification the agent sees a generic ``type_failed`` and
        can't recover. With ``reason: "not_fillable"`` it knows to fall
        back to ``browser_get_elements`` to find the underlying input.
        """
        mgr = _make_manager()
        refs = {
            "e0": {"role": "text", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        # Playwright's actual message; classifier is case-insensitive.
        loc = _make_locator(
            raise_on_fill=Exception(
                "Element is not an <input>, <textarea> or "
                "[contenteditable] element"
            ),
        )

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": "a@b.co"}],
            )

        f0 = result["data"]["filled"][0]
        assert f0["status"] == "type_failed"
        assert f0["reason"] == "not_fillable"

    @pytest.mark.asyncio
    async def test_classify_disabled(self):
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator(
            raise_on_fill=Exception("element is disabled"),
        )

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": "a@b.co"}],
            )

        assert result["data"]["filled"][0]["reason"] == "disabled"

    @pytest.mark.asyncio
    async def test_classify_other_falls_through(self):
        """Unknown error message → reason='other' (no crash)."""
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator(
            raise_on_fill=Exception("some weird Playwright internal error"),
        )

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": "a@b.co"}],
            )

        assert result["data"]["filled"][0]["reason"] == "other"
        # Detail preserves the underlying message for operator debugging.
        assert "weird" in result["data"]["filled"][0]["detail"]


class TestFillFormSubmitEdgeCases:
    """Concerns 8 & 10 — submit_after only fires when there's an anchor."""

    @pytest.mark.asyncio
    async def test_top_level_submit_with_all_failed_fields_no_press(self):
        """Concern 8: ``submit_after=True`` but no field filled → no Enter.

        ``last_locator`` stays ``None``, so the final-submit branch must
        be skipped. ``submitted`` must remain ``False``.
        """
        mgr = _make_manager()
        # No matching ref for either label → both fields go ``not_found``.
        refs = {
            "e0": {"role": "button", "name": "Cancel", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Phone", "value": "555"},
                ],
                submit_after=True,
            )

        data = result["data"]
        assert all(f["status"] == "not_found" for f in data["filled"])
        # Critical: no anchor → no Enter pressed → submitted stays False.
        assert data["submitted"] is False
        assert loc.press.await_count == 0

    @pytest.mark.asyncio
    async def test_top_level_submit_with_partial_fill_no_press(self):
        """Top-level submit_after should not submit an incomplete form.

        A caller asking for final submit expects the whole requested field
        set to be present. If one field is missing, return partial_success
        and leave submission to an explicit follow-up decision.
        """
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    {"label": "Email", "value": "a@b.co"},
                    {"label": "Password", "value": "pw"},
                ],
                submit_after=True,
            )

        data = result["data"]
        assert [f["status"] for f in data["filled"]] == ["filled", "not_found"]
        assert data["partial_success"] is True
        assert data["submitted"] is False
        assert loc.press.await_count == 0

    @pytest.mark.asyncio
    async def test_per_field_submit_on_not_found_field_no_press(self):
        """Concern 10: per-field ``submit_after`` on not_found → no Enter.

        The submit-on-field branch lives below the successful-fill path.
        A ``not_found`` field must ``continue`` straight to the next
        iteration without pressing Enter on a phantom locator.
        """
        mgr = _make_manager()
        refs = {
            # Matches "Phone" but NOT "Email"
            "e0": {"role": "textbox", "name": "Phone", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [
                    # submit_after=True on a field that won't be found —
                    # must NOT press Enter on anything.
                    {"label": "Email", "value": "a@b.co", "submit_after": True},
                    {"label": "Phone", "value": "555"},
                ],
            )

        data = result["data"]
        statuses = [f["status"] for f in data["filled"]]
        assert statuses == ["not_found", "filled"]
        # Only Phone was filled; no Enter presses anywhere.
        assert loc.press.await_count == 0
        assert data["submitted"] is False


class TestFillFormValueAndLabelHandling:
    """Concerns 6 (empty string) & 13 (label redaction in response)."""

    @pytest.mark.asyncio
    async def test_empty_string_value_clears_input(self):
        """Concern 6: ``value=""`` is valid — fill('') clears the input."""
        mgr = _make_manager()
        refs = {
            "e0": {"role": "textbox", "name": "Email", "index": 0, "disabled": False},
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": "Email", "value": ""}],
            )

        assert result["success"] is True
        assert result["data"]["filled"][0]["status"] == "filled"
        loc.fill.assert_awaited_with("")

    @pytest.mark.asyncio
    async def test_label_with_secret_redacted_in_filled_response(self):
        """Concern 13: ``filled[].label`` redacted on the way out.

        If the agent passes a label like ``"sk-abc123def456..."`` (e.g. a
        prompt-injected label that contains a secret pattern), the
        echoed-back ``filled[].label`` must run through the same
        redactor that scrubs URLs, snapshot text, and find_text matches.
        ``remaining[].label`` deliberately keeps verbatim values for
        resume — that path is covered by the captcha-mid-flow tests.
        """
        mgr = _make_manager()
        # Use a label that ALSO matches a ref (so find_text → fill path
        # exercises the safe_label substitution).
        secret_label = "sk-1234567890abcdefghijklmnopqrstuv"
        refs = {
            "e0": {
                "role": "textbox", "name": secret_label,
                "index": 0, "disabled": False,
            },
        }
        inst = _make_instance(refs)
        loc = _make_locator()

        with patch.object(BrowserManager, "get_or_start", return_value=inst), \
             patch.object(BrowserManager, "_locator_from_ref",
                          new_callable=AsyncMock, return_value=loc), \
             patch.object(BrowserManager, "_check_captcha",
                          new_callable=AsyncMock,
                          return_value={"captcha_found": False}), \
             patch.object(BrowserManager, "_snapshot_impl", new=_fake_snapshot):
            result = await mgr.fill_form(
                "agent1",
                [{"label": secret_label, "value": "x"}],
            )

        echoed = result["data"]["filled"][0]["label"]
        # Some fragment of the secret (the ``sk-`` prefix + suffix) should
        # be redacted; we don't pin exact format because the redaction
        # patterns may evolve. The contract is "literal echo blocked".
        assert echoed != secret_label, (
            "filled[].label echoed user-supplied secret verbatim — "
            "expected redactor to mask it"
        )


class TestFillFormServerRoute:
    """Concern 15 — happy-path POST /browser/{agent_id}/fill_form route test."""

    def test_route_happy_path_returns_envelope(self, monkeypatch):
        """Verify the FastAPI route shape: 200 + the manager envelope."""
        from fastapi.testclient import TestClient

        from src.browser.server import create_browser_app

        monkeypatch.delenv("BROWSER_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("MESH_AUTH_TOKEN", raising=False)

        mgr = MagicMock()
        mgr.fill_form = AsyncMock(return_value={
            "success": True,
            "data": {
                "partial_success": False,
                "captcha_required": False,
                "filled": [{"label": "Email", "ref": "e0", "status": "filled"}],
                "remaining": [],
                "submitted": False,
            },
        })

        app = create_browser_app(mgr)
        with TestClient(app) as client:
            resp = client.post(
                "/browser/a1/fill_form",
                json={
                    "fields": [{"label": "Email", "value": "a@b.co"}],
                    "submit_after": False,
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["filled"][0]["status"] == "filled"
        # Manager called with the body's fields verbatim plus the keyword arg.
        mgr.fill_form.assert_awaited_once()
        args, kwargs = mgr.fill_form.await_args
        assert args[0] == "a1"
        assert args[1] == [{"label": "Email", "value": "a@b.co"}]
        assert kwargs == {"submit_after": False}
