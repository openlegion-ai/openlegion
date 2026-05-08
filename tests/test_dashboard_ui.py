"""UI-level tests for the dashboard SPA.

This file aggregates three layers of UI-contract tests:

1. Phase -1 onboarding wizard — verifies the wizard markup is present,
   the chip buttons reference the locked labels, the Alpine state
   machine has the expected handlers, and the empty-state is correctly
   suppressed when the wizard is active.

2. Phase 2 Board UX vocabulary sweep + activity translation +
   notifications bell + empty-state CTAs — verifies user-facing strings
   in the SPA template and JS-source-level checks for the activity
   translation helper.

3. Phase 3 Home single-scroll layout + Operator action chips — verifies
   ACTION-line parsing strips the trailing chip block, default Quick
   actions render, the Home single-scroll section testids are wired,
   and the kanban sub-page is gated on ``homeTab === 'tasks'``.

We can't run a real headless browser in CI, so all assertions are
string-level over the rendered template + static JS — enough to catch
regressions like "someone deleted the chip", "the state machine forgot
a transition", "a vocab string drifted", or "the chat empty state isn't
gated on wizard.step".

Phase 2 / Phase 3 of the Board UX overhaul
(`docs/plans/2026-05-08-board-ux-overhaul.md`).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _REPO_ROOT / "src/dashboard/templates/index.html"
_APP_JS = _REPO_ROOT / "src/dashboard/static/js/app.js"

# Phase 2 tests cache the file contents at import time. We expose them
# under distinct names so they don't shadow the Path objects used by
# the wizard tests.
_INDEX_HTML = _TEMPLATE.read_text(encoding="utf-8")
_APP_JS_TEXT = _APP_JS.read_text(encoding="utf-8")


# Phase 3 fixture aliases — pytest fixtures so chip-parsing tests can
# request the source as plain strings.
ROOT = _REPO_ROOT
APP_JS = _APP_JS
INDEX_HTML = _TEMPLATE


@pytest.fixture(scope="module")
def app_js() -> str:
    return APP_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index_html() -> str:
    return INDEX_HTML.read_text(encoding="utf-8")


# ── Static markup contract ───────────────────────────────────────


class TestWizardMarkup:
    """The wizard card UI is hard-coded into the SPA template."""

    def test_wizard_block_is_present(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="onboarding-wizard"' in html

    def test_wizard_renders_only_when_step_not_idle(self):
        html = _read(_TEMPLATE)
        # The outer x-if gates rendering on wizard.step !== 'idle'.
        assert "wizard.step !== 'idle'" in html

    def test_wizard_has_all_four_steps(self):
        html = _read(_TEMPLATE)
        for step in ("ask", "confirming", "building", "first-output"):
            assert f"wizard.step === '{step}'" in html, (
                f"missing wizard step: {step}"
            )

    def test_wizard_chips_use_locked_labels(self):
        html = _read(_TEMPLATE)
        # These four chip labels are frozen by the spec; the operator
        # response handler depends on them as user-facing message text.
        for chip in (
            "Monitor competitors weekly",
            "Build a content team",
            "Enrich my lead list",
        ):
            assert f"wizardChipClicked('{chip}')" in html, (
                f"missing chip handler for: {chip}"
            )
        # "Something else…" uses the literal ellipsis char — the JS
        # handler keys off this exact string.
        assert "wizardChipClicked('Something else…')" in html

    def test_wizard_confirm_buttons_present(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="wizard-confirm-go"' in html
        assert 'data-testid="wizard-confirm-customize"' in html
        assert "wizardConfirm()" in html
        assert "wizardCustomize()" in html

    def test_wizard_continue_button_calls_complete(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="wizard-continue"' in html
        assert "wizardComplete()" in html

    def test_wizard_dismiss_buttons_call_abandon(self):
        html = _read(_TEMPLATE)
        assert "wizardAbandon()" in html

    def test_wizard_card_has_dialog_role(self):
        html = _read(_TEMPLATE)
        # role="dialog" on the wizard card for screen-reader semantics.
        # We grep for the role + the data-testid on the same element by
        # scanning for the combination on adjacent characters.
        assert re.search(
            r'role="dialog"[^>]*?data-testid="onboarding-wizard"'
            r'|data-testid="onboarding-wizard"[^>]*?role="dialog"',
            html,
        ), "wizard card missing role=dialog"

    def test_existing_empty_state_gated_on_wizard_idle(self):
        html = _read(_TEMPLATE)
        # The legacy first-run empty state must not render while the
        # wizard is active — otherwise we'd stack two empty states.
        # We look for the chained guard on the existing empty state.
        assert "wizard.step === 'idle' && !(chatHistories['operator']" in html


# ── JS state machine contract ────────────────────────────────────


class TestWizardJsState:
    """The Alpine app exposes the locked state-machine handlers."""

    def test_wizard_state_object_declared(self):
        js = _read(_APP_JS)
        assert "wizard: { step: 'idle'" in js

    def test_wizard_handlers_declared(self):
        js = _read(_APP_JS)
        for handler in (
            "startWizard()",
            "wizardChipClicked(label)",
            "wizardConfirm()",
            "wizardComplete()",
            "wizardAbandon()",
            "_maybeStartWizard()",
            "_isFirstVisit()",
            "track(eventName, props)",
        ):
            assert handler in js, f"missing handler: {handler}"

    def test_telemetry_endpoint_is_called(self):
        js = _read(_APP_JS)
        assert "/telemetry" in js
        # Beacon path on unload.
        assert "navigator.sendBeacon" in js

    def test_wizard_state_persisted_to_localstorage(self):
        js = _read(_APP_JS)
        assert "localStorage" in js
        assert "ol_wizard" in js

    def test_first_visit_uses_empty_fleet_check(self):
        js = _read(_APP_JS)
        # _isFirstVisit checks two things: no fleet agents, no user msg.
        assert "_isFirstVisit" in js
        # The detector excludes the operator from the fleet check.
        assert "a.id !== 'operator'" in js

    def test_wizard_advances_to_building_on_confirm(self):
        js = _read(_APP_JS)
        # wizardConfirm should advance to 'building' and start polling.
        assert "_wizardAdvance('building'" in js
        assert "_wizardStartBuildPolling" in js

    def test_wizard_advances_to_first_output_when_fleet_populated(self):
        js = _read(_APP_JS)
        # Build poller advances to first-output when ≥1 non-operator
        # agent exists.
        assert "fleetAgents.length >= 1" in js
        assert "_wizardAdvance('first-output'" in js

    def test_telemetry_events_have_locked_names(self):
        js = _read(_APP_JS)
        # Spec freezes these event names; renaming them breaks the
        # backend dashboard query and the activation hypothesis test.
        for evt in (
            "wizard_started",
            "wizard_chip_clicked",
            "wizard_step_advanced",
            "wizard_completed",
            "wizard_abandoned",
        ):
            assert f"'{evt}'" in js, f"missing telemetry event: {evt}"


# ── Smoke test the endpoint via the dashboard router ─────────────


class TestWizardEndpointWiring:
    """Verify the dashboard serves the template with wizard markup."""

    def setup_method(self):
        from unittest.mock import MagicMock

        from src.dashboard.events import EventBus
        from src.host.costs import CostTracker
        from src.host.health import HealthMonitor
        from src.host.mesh import Blackboard
        from src.host.traces import TraceStore

        self._tmpdir = tempfile.mkdtemp()
        bb = Blackboard(db_path=os.path.join(self._tmpdir, "bb.db"))
        cost_tracker = CostTracker(
            db_path=os.path.join(self._tmpdir, "costs.db"),
        )
        trace_store = TraceStore(
            db_path=os.path.join(self._tmpdir, "traces.db"),
        )
        runtime_mock = MagicMock()
        runtime_mock.browser_vnc_url = None
        runtime_mock.browser_service_url = None
        runtime_mock.browser_auth_token = ""
        transport_mock = MagicMock()
        router_mock = MagicMock()
        health_monitor = HealthMonitor(
            runtime=runtime_mock,
            transport=transport_mock,
            router=router_mock,
        )
        self.components = {
            "blackboard": bb,
            "health_monitor": health_monitor,
            "cost_tracker": cost_tracker,
            "trace_store": trace_store,
            "event_bus": EventBus(),
            "agent_registry": {},
        }

        from src.dashboard.server import create_dashboard_router
        from src.dashboard.telemetry import DashboardTelemetry

        self.telemetry = DashboardTelemetry(
            db_path=os.path.join(self._tmpdir, "telemetry.db"),
        )
        router = create_dashboard_router(
            **self.components, mesh_port=8420, telemetry=self.telemetry,
        )
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def teardown_method(self):
        self.telemetry.close()
        self.components["cost_tracker"].close()
        self.components["trace_store"].close()
        self.components["blackboard"].close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_dashboard_index_includes_wizard_markup(self):
        resp = self.client.get("/dashboard/")
        assert resp.status_code == 200
        assert 'data-testid="onboarding-wizard"' in resp.text
        assert "wizardChipClicked" in resp.text

    def test_app_js_serves_wizard_state(self):
        resp = self.client.get("/dashboard/static/js/app.js")
        assert resp.status_code == 200
        assert "wizard: { step: 'idle'" in resp.text
        assert "_maybeStartWizard" in resp.text


# ── Tab labels (engineer-speak → user-speak) ──────────────────────────


class TestTabLabelVocabulary:
    def test_tabs_array_uses_user_speak(self):
        # The Alpine ``tabs`` array maps internal IDs to display labels.
        # Phase 2 renames Agents → Team, Board → Home, System → Settings.
        m = re.search(
            r"tabs:\s*\[(.*?)\],",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "Could not locate the tabs array in app.js"
        block = m.group(1)
        assert "id: 'fleet'" in block and "label: 'Team'" in block
        assert "id: 'workplace'" in block and "label: 'Home'" in block
        assert "id: 'system'" in block and "label: 'Settings'" in block

    def test_fleet_running_count_uses_team_word(self):
        # The dynamic-label expression in the top-nav swaps "Agents (N)"
        # for "Team (N)" once Phase 2 ships.
        assert "'Team (' + runningAgents.length" in _INDEX_HTML
        assert "'Agents (' + runningAgents.length" not in _INDEX_HTML


class TestVocabularySweepStrings:
    def test_pending_actions_renamed(self):
        # The Operator sub-tab card was titled "Pending actions"; the
        # sweep flips it to "Approvals needed".
        assert "Approvals needed" in _INDEX_HTML
        assert ">Pending actions<" not in _INDEX_HTML

    def test_audit_log_renamed_to_change_history(self):
        assert "Change history" in _INDEX_HTML
        # The legacy "Change Log" wording should be gone (not "Audit log",
        # which only ever lived in the HTML comment).
        assert "Change Log<" not in _INDEX_HTML

    def test_heartbeat_user_facing_label(self):
        # The Identity-tab file label for HEARTBEAT.md flips to
        # "Auto-checkup"; the field name itself stays heartbeat_*.
        assert "label: 'Auto-checkup'" in _APP_JS_TEXT
        assert "<span>Auto-checkup</span>" in _INDEX_HTML

    def test_failure_label_renamed_to_errors(self):
        assert ">Errors<" in _INDEX_HTML

    def test_add_agent_renamed_to_add_teammate(self):
        # All three "Add Agent" instances (sidebar header + modal title +
        # submit button) flip to "Add teammate".
        assert ">Add teammate<" in _INDEX_HTML
        assert ">Add Agent<" not in _INDEX_HTML

    def test_pending_action_toasts_renamed(self):
        # The toast strings shown on confirm/cancel.
        assert "'Approval confirmed.'" in _APP_JS_TEXT
        assert "'Approval cancelled.'" in _APP_JS_TEXT

    def test_workforce_renamed_to_team(self):
        # The chat-tab header subtitle ("Builds and manages your agent
        # workforce") drops the engineer-speak word.
        assert "Builds and manages your team" in _INDEX_HTML
        assert "agent workforce" not in _INDEX_HTML


# ── Empty-state CTAs ──────────────────────────────────────────────────


class TestEmptyStateCTAs:
    def test_workplace_empty_outputs_has_cta(self):
        # The Outputs sub-tab empty state now contains a verb-driven CTA
        # pointing users at the operator chat.
        assert "Tell the Operator what you want delivered" in _INDEX_HTML

    def test_team_empty_state_has_cta(self):
        # The fleet/team page first-run state.
        assert "Build your first team" in _INDEX_HTML

    def test_workplace_feed_empty_state_user_speak(self):
        # The activity feed empty-state text avoids "agents" — uses
        # "team" instead.
        assert "Once your team starts working" in _INDEX_HTML


# ── Notifications bell ────────────────────────────────────────────────


class TestNotificationsBellMarkup:
    def test_bell_button_in_template(self):
        # The bell sits in the top-nav, identifiable by its toggle.
        assert "toggleNotifications()" in _INDEX_HTML

    def test_bell_unread_dot_renders_when_count_above_zero(self):
        # The unread dot is gated on notificationsUnreadCount > 0.
        assert 'x-show="notificationsUnreadCount > 0"' in _INDEX_HTML

    def test_dropdown_hidden_when_empty(self):
        # Per Phase 2 §4, the dropdown hides entirely when no
        # notifications exist (empty placeholder is a no-no).
        assert (
            'x-show="notificationsOpen && notifications.length > 0"'
            in _INDEX_HTML
        )

    def test_mark_all_read_link(self):
        assert "markAllNotificationsRead()" in _INDEX_HTML

    def test_legacy_notifications_bell_removed(self):
        # The Phase 1 placeholder bell read its dropdown items from the
        # in-memory ``events`` array (mirrored the activity feed). It
        # competed visually with the Phase 2 persistent bell — both
        # rendered side-by-side in the top nav. Removing it is the
        # whole point of this fix; assert the unique markers from the
        # legacy block are gone.
        # 1) Legacy used `notificationsOpen = !notificationsOpen` for
        #    its toggle (Phase 2 uses `toggleNotifications()`).
        assert "notificationsOpen = !notificationsOpen" not in _INDEX_HTML
        # 2) Legacy listed `(events || []).slice(0, 10)` in its
        #    dropdown. The Phase 2 bell iterates `notifications` instead.
        assert "(events || []).slice(0, 10)" not in _INDEX_HTML
        # 3) Legacy showed a "No recent activity" placeholder; Phase 2
        #    hides the dropdown entirely when empty.
        assert "No recent activity" not in _INDEX_HTML

    def test_only_one_notifications_bell_renders(self):
        # The bell SVG path is identifying enough — both Phase 1 and
        # Phase 2 used the same ``M18 8A6 6 0 0...`` Feather icon. After
        # the fix exactly one bell remains in the top-nav.
        # Match the bell-shape path with whitespace-tolerance (Phase 1
        # used ``A6 6 0 0 0`` while Phase 2 collapsed to ``A6 6 0 006``;
        # any future SVG change still picks both up).
        bells = re.findall(r'M18 8A6 6 0\s*0?\s*0?6 8c0 7-3 9-3 9h18s-3-2-3-9', _INDEX_HTML)
        assert len(bells) == 1, (
            f"Expected exactly one notifications bell SVG, found {len(bells)}"
        )


# ── Activity translation (formatActivityForUser) ──────────────────────


class TestFormatActivityForUserStructure:
    """Source-level structure check on formatActivityForUser. We don't
    spin up a JS runtime; the helper's behaviour is verified indirectly
    via the cases dispatched in the switch."""

    def test_function_defined(self):
        assert "formatActivityForUser(event)" in _APP_JS_TEXT

    def test_handles_tool_start(self):
        assert "case 'tool_start':" in _APP_JS_TEXT
        assert "verbForTool" in _APP_JS_TEXT

    def test_handles_task_status_changed(self):
        assert "case 'task_status_changed':" in _APP_JS_TEXT
        assert "verbForStatus" in _APP_JS_TEXT

    def test_handles_task_outcome(self):
        assert "case 'task_outcome':" in _APP_JS_TEXT
        assert "work was" in _APP_JS_TEXT

    def test_handles_credential_request(self):
        assert "case 'credential_request':" in _APP_JS_TEXT
        assert "your call" in _APP_JS_TEXT

    def test_handles_pending_action_created(self):
        assert "case 'pending_action_created':" in _APP_JS_TEXT
        assert "wants to" in _APP_JS_TEXT

    def test_hides_implementation_events_by_default(self):
        # blackboard_write / llm_call / message_received / message_sent
        # / text_delta / agent_state should fall through to ``return null``.
        # Source-level check: each is in the same fall-through cluster.
        m = re.search(
            r"// Hidden by default.*?return null;",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "Hidden-by-default cluster missing"
        block = m.group(0)
        for case in (
            "blackboard_write",
            "llm_call",
            "message_received",
            "message_sent",
            "text_delta",
            "agent_state",
        ):
            assert f"case '{case}'" in block, f"{case} should be hidden by default"

    def test_show_tech_detail_persisted_to_localstorage(self):
        # The toggle persists to localStorage under olShowTechDetail.
        assert "olShowTechDetail" in _APP_JS_TEXT

    def test_filtered_events_respects_show_tech_detail(self):
        # The Activity feed's filtered view honors showTechDetail.
        m = re.search(
            r"get filteredEvents\(\)\s*\{(.*?)\}\s*,",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "filteredEvents getter missing"
        body = m.group(1)
        assert "showTechDetail" in body
        assert "isActivityEventVisible" in body


# ── verbForTool / verbForStatus helpers ───────────────────────────────


class TestActivityVerbs:
    """Spot-check that the verb maps cover the most-common cases.

    The helpers themselves are JS — we assert on the source so the
    mapping table doesn't quietly drift. When the dashboard ships its
    own JS test runner these can move there."""

    def test_verb_for_tool_includes_common_browser_actions(self):
        for tool in (
            "browser_navigate",
            "browser_click",
            "browser_type",
            "browser_screenshot",
            "web_search",
            "memory_save",
            "hand_off",
        ):
            assert f"{tool}:" in _APP_JS_TEXT, f"verbForTool missing {tool}"

    def test_verb_for_status_includes_terminal_states(self):
        for status in ("done", "blocked", "failed", "cancelled", "delivered"):
            assert f"{status}:" in _APP_JS_TEXT, f"verbForStatus missing {status}"


# ── Pure-Python reimplementation of the JS chip parser ──────────
#
# Mirrors ``_parseOperatorActions`` in ``app.js``. We mirror the contract
# rather than spinning up a JS runtime so the test can run in CI without
# Node. Keep this in sync if the JS implementation evolves; the tests
# below assert on the JS source string for the regex itself so a divergent
# regex still trips the test suite.
_ACTION_LINE = re.compile(r"^(?:[-*]\s+)?ACTION\s*:\s*(.+)$", re.IGNORECASE)


def parse_operator_actions(text: str) -> tuple[str, list[str]]:
    """Strip trailing ACTION lines off ``text``; return (body, labels)."""
    if not text:
        return text or "", []
    lines = text.split("\n")
    actions: list[str] = []
    trailing = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            trailing = i
            continue
        if stripped == "```" or stripped.startswith("```"):
            trailing = i
            continue
        m = _ACTION_LINE.match(stripped)
        if m:
            label = m.group(1).strip().strip("\"'`").strip()
            if label and len(label) <= 80:
                actions.insert(0, label[:60])
            trailing = i
            continue
        break
    if not actions:
        return text, []
    seen: set[str] = set()
    deduped: list[str] = []
    for label in actions:
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(label)
        if len(deduped) >= 6:
            break
    body = "\n".join(lines[:trailing]).rstrip()
    return body, deduped


class TestParseOperatorActions:
    """Phase 3 — chip parsing extracts labels and strips them off the body."""

    def test_extracts_two_chips(self):
        text = (
            "Got it — I'll get @writer on that.\n"
            "\n"
            "ACTION: Show me what we delivered\n"
            "ACTION: Add another teammate\n"
        )
        body, actions = parse_operator_actions(text)
        assert body == "Got it — I'll get @writer on that."
        assert actions == ["Show me what we delivered", "Add another teammate"]

    def test_no_chips_returns_text_untouched(self):
        text = "No chips here, just a normal message."
        body, actions = parse_operator_actions(text)
        assert body == text
        assert actions == []

    def test_dash_prefix_tolerated(self):
        text = "ok\n\n- ACTION: First\n- ACTION: Second\n"
        _, actions = parse_operator_actions(text)
        assert actions == ["First", "Second"]

    def test_fenced_block_tolerated(self):
        text = "ok\n\n```\nACTION: Run a check\nACTION: Pause everything\n```\n"
        body, actions = parse_operator_actions(text)
        assert "ok" in body
        assert actions == ["Run a check", "Pause everything"]

    def test_dedupes_case_insensitive(self):
        text = "x\n\nACTION: Do it\nACTION: do it\nACTION: Other\n"
        _, actions = parse_operator_actions(text)
        assert actions == ["Do it", "Other"]

    def test_caps_at_six(self):
        text = "x\n\n" + "\n".join(f"ACTION: Label {i}" for i in range(20))
        _, actions = parse_operator_actions(text)
        assert len(actions) == 6

    def test_action_in_middle_stays_inline(self):
        # Spec contract — ACTION block must be at the very end. An ACTION
        # token mid-message stays as plain text.
        text = "I noted ACTION: foo earlier in the message\n\nOK."
        body, actions = parse_operator_actions(text)
        assert actions == []
        assert "ACTION: foo" in body

    def test_empty_body_when_only_chips(self):
        text = "ACTION: Only thing\nACTION: Other thing\n"
        body, actions = parse_operator_actions(text)
        assert body == ""
        assert actions == ["Only thing", "Other thing"]

    def test_strips_quotes_around_label(self):
        text = "x\n\nACTION: \"Quoted thing\"\nACTION: 'Other'\n"
        _, actions = parse_operator_actions(text)
        assert actions == ["Quoted thing", "Other"]


class TestChipParserInJsSource:
    """The JS implementation in app.js must match the Python mirror."""

    def test_parser_function_present(self, app_js: str):
        assert "_parseOperatorActions" in app_js
        assert "_applyOperatorActions" in app_js
        assert "sendOperatorChip" in app_js

    def test_regex_matches_python_mirror(self, app_js: str):
        # The regex literal is embedded as a JS RegExp; assert that the
        # core anchor + ACTION token + capture pattern is present so
        # the dash/star bullet variants stay tolerated.
        assert "[-*]" in app_js or "[-\\*]" in app_js, \
            "ACTION-line regex in app.js must accept optional dash/star prefix"
        assert "ACTION" in app_js
        # Spot-check the regex literal as written in app.js. We look for
        # the ``ACTION\s*:\s*(.+)$`` body which is the load-bearing part
        # of the parser.
        assert re.search(r"ACTION\\s\*:\\s\*\(\.\+\)\$", app_js)

    def test_done_handler_invokes_parser_for_operator(self, app_js: str):
        # The streaming ``done`` event must trigger chip parsing only for
        # the operator agent — worker chats render free-text bodies.
        assert "if (agentId === 'operator') this._applyOperatorActions(entry)" in app_js

    def test_history_loader_invokes_parser_for_operator(self, app_js: str):
        # Loading history from the server should re-derive chips so a
        # page reload doesn't lose them.
        assert "for (const sm of serverMsgs) this._applyOperatorActions(sm)" in app_js


class TestQuickActionsMenu:
    """Phase 3 — pre-chip "Quick actions" menu defaults."""

    def test_default_chips_present(self, app_js: str):
        # All four labels from the plan must be present verbatim so a
        # vocabulary sweep can find them in one place.
        for label in (
            "What's happening?",
            "Show me what we delivered",
            "Add someone to my team",
            "Pause everything",
        ):
            assert label in app_js, f"Missing default Quick action chip: {label!r}"

    def test_show_chips_helper_gates_on_pause(self, app_js: str):
        # 5-min pause threshold per the plan. Use a permissive regex so
        # whitespace / comment layout can change without breaking the test.
        assert re.search(r"5\s*\*\s*60\s*\*\s*1000", app_js), \
            "showOperatorDefaultChips should gate on a 5-min pause"

    def test_quick_actions_markup_present(self, index_html: str):
        assert 'data-testid="operator-quick-actions"' in index_html
        assert "showOperatorDefaultChips()" in index_html


class TestOperatorActionChipsMarkup:
    """Operator response chips render below the bubble."""

    def test_chips_block_present(self, index_html: str):
        assert 'data-testid="operator-action-chips"' in index_html

    def test_chips_only_render_for_operator_agent(self, index_html: str):
        # The block must be gated on ``msg.role === 'agent'`` plus
        # ``suggested_actions`` populated; we look for both substrings
        # near each other.
        idx = index_html.find('data-testid="operator-action-chips"')
        assert idx > 0
        # Walk back to the enclosing template open tag and check the gate.
        slice_ = index_html[max(0, idx - 600):idx]
        assert "suggested_actions" in slice_
        assert "msg.role === 'agent'" in slice_

    def test_chip_click_uses_send_operator_chip(self, index_html: str):
        idx = index_html.find('data-testid="operator-action-chips"')
        assert idx > 0
        nearby = index_html[idx:idx + 800]
        assert "sendOperatorChip" in nearby


class TestHomeSingleScrollLayout:
    """Phase 3 — Workplace tab restructured into a single scroll page."""

    def test_main_layout_gated_on_home_tab(self, index_html: str):
        # The main scroll renders only when ``homeTab === 'main'``.
        assert "workplaceEnabled && homeTab === 'main'" in index_html

    def test_kanban_subpage_gated_on_home_tab_tasks(self, index_html: str):
        # The kanban moves to its own sub-page (homeTab === 'tasks').
        assert "workplaceEnabled && homeTab === 'tasks'" in index_html

    def test_section_testids_present(self, index_html: str):
        # Every section in the new layout has a testid so a future
        # refactor can't accidentally hide one without tripping a test.
        for testid in (
            "home-main",
            "home-just-delivered",
            "home-happening-now",
            "home-in-progress",
            "home-see-task-board",
        ):
            assert f'data-testid="{testid}"' in index_html, \
                f"Missing testid: {testid}"

    def test_subtab_bar_removed(self, index_html: str):
        # The legacy sub-tab bar must be gone — searching for the unique
        # ``workplaceTab = wt.id`` click handler (sub-tab buttons) should
        # no longer find it.
        assert "workplaceTab = wt.id; loadWorkplace()" not in index_html

    def test_back_to_home_link_in_tasks_subpage(self, index_html: str):
        # Tasks sub-page exposes a back link that returns to ``main``.
        assert 'data-testid="home-back-to-main"' in index_html
        assert "switchHomeTab('main')" in index_html

    def test_legacy_subtabs_hidden_with_back_compat_gate(self, index_html: str):
        # The old project-status / team-outputs renders are kept in
        # markup behind a ``false &&`` short-circuit so nothing visible
        # depends on ``workplaceTab`` while deep-link callers still
        # don't NPE on the missing template.
        assert "false && workplaceEnabled && workplaceTab === 'project-status'" in index_html
        assert "false && workplaceEnabled && workplaceTab === 'team-outputs'" in index_html


class TestHomeRouting:
    """Phase 3 — ``/home`` and ``/home/tasks`` URL routes."""

    def test_build_path_emits_home_route(self, app_js: str):
        # ``_buildPath`` returns ``/home`` and ``/home/tasks`` for the
        # workplace tab depending on ``homeTab``.
        assert "this.homeTab === 'tasks' ? '/home/tasks' : '/home'" in app_js

    def test_parse_path_recognizes_home_routes(self, app_js: str):
        # ``_parsePath`` accepts the new routes and maps them to the
        # workplace tab + correct sub-page.
        assert "clean === 'home'" in app_js
        assert "clean === 'home/tasks'" in app_js or "home/tasks" in app_js

    def test_switch_home_tab_helper_present(self, app_js: str):
        assert "switchHomeTab(tabId)" in app_js


# ── Phase 4: "What's new" tour for existing-fleet users ──────────


class TestWhatsNewTourMarkup:
    """The 3-step modal tour is hard-coded into the SPA template."""

    def test_modal_block_is_present(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="whats-new-modal"' in html

    def test_modal_renders_only_when_step_not_zero(self):
        html = _read(_TEMPLATE)
        assert "whatsNewTour.step !== 0" in html

    def test_modal_has_three_steps(self):
        html = _read(_TEMPLATE)
        for step in (1, 2, 3):
            assert f"whatsNewTour.step === {step}" in html, (
                f"missing whats-new step: {step}"
            )

    def test_modal_has_dialog_role_and_modal_aria(self):
        html = _read(_TEMPLATE)
        # role=dialog + aria-modal=true on the same element as the
        # data-testid hook — required for screen readers.
        assert re.search(
            r'role="dialog"[^>]*?aria-modal="true"[^>]*?data-testid="whats-new-modal"'
            r'|data-testid="whats-new-modal"[^>]*?role="dialog"',
            html,
        ), "whats-new modal missing role=dialog/aria-modal"

    def test_step1_copy_present(self):
        html = _read(_TEMPLATE)
        assert "We&rsquo;ve simplified your dashboard" in html
        assert "Skip" in html
        assert "Show me" in html

    def test_step2_copy_present(self):
        html = _read(_TEMPLATE)
        assert "Operator is one click away" in html
        assert "talk to the Operator" in html

    def test_step3_copy_present(self):
        html = _read(_TEMPLATE)
        assert "Approvals and notifications are in the top nav" in html
        assert "Got it" in html

    def test_skip_button_dismisses_tour(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="whats-new-skip"' in html
        assert "dismissWhatsNewTour(" in html

    def test_primary_button_has_focus_target(self):
        # Focus management: each step's primary advance button is
        # auto-focused on entry (via querySelector in app.js).
        html = _read(_TEMPLATE)
        assert 'data-testid="whats-new-primary"' in html


class TestWhatsNewTourJsState:
    """Alpine handlers for the tour state machine."""

    def test_tour_state_object_declared(self):
        js = _read(_APP_JS)
        assert "whatsNewTour: { step: 0 }" in js

    def test_tour_handlers_declared(self):
        js = _read(_APP_JS)
        for handler in (
            "_maybeStartWhatsNewTour()",
            "startWhatsNewTour()",
            "whatsNewTourNext()",
            "whatsNewTourBack()",
            "dismissWhatsNewTour(reason)",
            "_completeWhatsNewTour(reason)",
        ):
            assert handler in js, f"missing tour handler: {handler}"

    def test_tour_persists_seen_flag(self):
        js = _read(_APP_JS)
        # Seen flag freezes the tour after first completion / dismiss.
        assert "olSeenWhatsNew" in js
        assert "localStorage.setItem('olSeenWhatsNew', 'true')" in js

    def test_tour_is_gated_on_existing_fleet(self):
        js = _read(_APP_JS)
        # Detection: agents.length > 0 (excluding operator).
        # The detector reuses the same operator-exclusion pattern as
        # the empty-fleet wizard.
        assert "_maybeStartWhatsNewTour" in js
        assert "fleetAgents.length === 0" in js

    def test_tour_emits_locked_telemetry_events(self):
        js = _read(_APP_JS)
        for evt in (
            "whats_new_tour_started",
            "whats_new_tour_step",
            "whats_new_tour_finished",
        ):
            assert f"'{evt}'" in js, f"missing tour telemetry event: {evt}"

    def test_tour_handles_escape_key(self):
        js = _read(_APP_JS)
        # ESC dismisses the modal. Tab-trap is also wired here.
        assert "e.key === 'Escape'" in js


# ── Phase 4: wizard polish ───────────────────────────────────────


class TestWizardPolish:
    """Phase 4 refinements to the empty-fleet wizard."""

    def test_progress_dots_present(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="wizard-progress"' in html
        # Four-dot indicator iterates over the four step ordinals.
        assert "[0,1,2,3]" in html

    def test_step_index_helper_declared(self):
        js = _read(_APP_JS)
        assert "wizardStepIndex()" in js

    def test_wizard_skip_handler_tracks_reason(self):
        js = _read(_APP_JS)
        # The new skip path emits wizard_abandoned with reason='skip_link'
        # so we can split deliberate skips from page-unload abandons.
        assert "wizardSkip()" in js
        assert "reason: 'skip_link'" in js

    def test_building_step_has_skip_button(self):
        html = _read(_TEMPLATE)
        # Phase 4 added the skip-X to the building step (it was already
        # on ask/confirming via wizardAbandon).
        assert 'data-testid="wizard-skip-building"' in html
        assert "wizardSkip()" in html

    def test_first_output_has_quick_links(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="wizard-quick-links"' in html
        assert 'data-testid="wizard-link-team"' in html
        assert 'data-testid="wizard-link-workplace"' in html
        # The quick-links section keeps a "Got it, close" affordance
        # that completes the wizard without changing tabs.
        assert 'data-testid="wizard-continue"' in html
