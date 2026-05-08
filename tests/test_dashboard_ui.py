"""UI-level tests for the dashboard SPA.

This file aggregates two layers of UI-contract tests:

1. Phase -1 onboarding wizard — verifies the wizard markup is present,
   the chip buttons reference the locked labels, the Alpine state
   machine has the expected handlers, and the empty-state is correctly
   suppressed when the wizard is active.

2. Phase 2 Board UX vocabulary sweep + activity translation +
   notifications bell + empty-state CTAs — verifies user-facing strings
   in the SPA template and JS-source-level checks for the activity
   translation helper.

We can't run a real headless browser in CI, so all assertions are
string-level over the rendered template + static JS — enough to catch
regressions like "someone deleted the chip", "the state machine forgot
a transition", "a vocab string drifted", or "the chat empty state isn't
gated on wizard.step".

Phase 2 of the Board UX overhaul (`docs/plans/2026-05-08-board-ux-overhaul.md`).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path

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
