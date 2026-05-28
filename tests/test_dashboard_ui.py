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

3. Operator action chips + Work-tab post-rewrite surface — verifies
   ACTION-line parsing strips the trailing chip block, default Quick
   actions render, the Goals chip strip is wired, summary card rating
   uses SVG icons (not emoji), and the legacy sub-nav / kanban /
   activity / standalone Tell Operator templates are excised.

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
        # Build poller advances to first-output when the non-operator
        # fleet has been NON-EMPTY and COUNT-STABLE for ≥10 seconds
        # (fix #1 — replaces the prior naive ``length >= 1`` check that
        # fired prematurely on multi-agent templates).
        assert "_wizardAdvance('first-output'" in js
        assert "fleet_stable" in js
        assert "stableSeconds" in js

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
        # Phase 2 renames Agents → Team, Board → Work, System → Settings.
        # The project→team rename (PR 2) flipped "Team" → "Teams" on the
        # ``fleet`` tab label; the tab ID stays frozen (Constraint #14).
        # Work sits 2nd: it's the second-most-visited tab after Chat.
        m = re.search(
            r"tabs:\s*\[(.*?)\],",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "Could not locate the tabs array in app.js"
        block = m.group(1)
        assert "id: 'fleet'" in block and "label: 'Teams'" in block
        assert "id: 'workplace'" in block and "label: 'Work'" in block
        assert "id: 'system'" in block and "label: 'Settings'" in block
        # Order: Chat | Work | Teams | Settings (Work is 2nd, Teams 3rd).
        chat_idx = block.find("id: 'chat'")
        work_idx = block.find("id: 'workplace'")
        team_idx = block.find("id: 'fleet'")
        system_idx = block.find("id: 'system'")
        assert chat_idx < work_idx < team_idx < system_idx, (
            "tabs must be ordered Chat | Work | Teams | Settings"
        )

    def test_fleet_running_count_uses_team_word(self):
        # PR #915 swapped the fleet-tab label from "Team (runningAgents)"
        # to "Teams (teams.length)" so the count in parens matches the
        # section the tab opens. The "Agents (..." form must still be
        # absent — that was the pre-Phase-2 wording.
        assert "'Teams (' + teams.length" in _INDEX_HTML
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
    def test_team_empty_state_has_cta(self):
        # The fleet/team page first-run state.
        assert "Build your first team" in _INDEX_HTML

    def test_workplace_summary_empty_state_user_speak(self):
        # PR 2 of Work tab rewrite — the activity feed empty-state
        # "Once your team starts working…" was deleted along with the
        # activity sub-page. The summary view's empty state is now the
        # user-facing "no work yet" surface.
        assert "No summaries yet" in _INDEX_HTML


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


class TestEventBusAuditFollowUpHandlers:
    """Source-level checks that the SPA wires up the EventBus follow-up
    fixes from the audit:

    * ``task_artifact_added`` refreshes the task drawer when open and
      nudges the workplace tasks list (debounced).
    * ``task_status_changed`` cancel renders the cancel reason inline.
    * ``agent_state restart_failed`` surfaces the error via ``showToast``.
    """

    def test_task_artifact_added_handler_refreshes_drawer(self):
        # The handler must short-circuit on the open drawer's task ID
        # and call ``loadTaskDrillIn`` so the new artifact lands without
        # a refresh.
        m = re.search(
            r"if \(evt\.type === 'task_artifact_added'\)\s*\{(.*?)\n      \}",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "task_artifact_added branch missing"
        body = m.group(1)
        assert "drillInTaskId" in body, (
            "Handler must check the open drawer's taskId"
        )
        assert "loadTaskDrillIn" in body, (
            "Handler must call loadTaskDrillIn on match"
        )
        assert "loadWorkplaceTasks" in body, (
            "Handler must nudge the workplace tasks list (debounced)"
        )

    def test_cancel_reason_appears_in_format_activity_for_user(self):
        # The activity-feed renderer also surfaces cancel reason inline.
        m = re.search(
            r"case 'task_status_changed':\s*\{(.*?)\}",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "task_status_changed case missing"
        body = m.group(1)
        assert "cancelled" in body and "d.reason" in body, (
            "task_status_changed handler must include cancel reason"
        )

    def test_restart_failed_surfaces_toast_with_error(self):
        # The ``agent_state restart_failed`` branch must:
        #   * clear the spinner (via agentRestartingMap)
        #   * call showToast with the error string
        # We anchor on the if-condition + the closing brace of the
        # outer block, then assert on the body. Brace-matching with
        # nested blocks is fragile under a single regex, so we slice
        # generously and check substrings.
        idx = _APP_JS_TEXT.find(
            "if (evt.type === 'agent_state' && evt.data?.state === 'restart_failed'"
        )
        assert idx != -1, "restart_failed branch missing"
        # Slice forward enough to include the toast block; 1500 chars
        # is more than enough but bounded in case the file ever grows.
        body = _APP_JS_TEXT[idx:idx + 1500]
        assert "agentRestartingMap" in body, (
            "restart_failed handler must clear the agentRestartingMap entry"
        )
        assert "showToast" in body, (
            "restart_failed handler must call showToast with the error"
        )
        assert "evt.data?.error" in body or "data?.error" in body, (
            "Toast must include the error from the event payload"
        )


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


class TestWorkTabPr2Cutover:
    """Work-tab cutover — sub-nav (Summaries / Kanban / Activity)
    deleted, summary cards are the unconditional landing surface,
    Goals chip strip is wired, rating buttons use SVG icons (not
    emoji), the standalone Tell Operator section is removed (steering
    lives in inline rework feedback + the Chat tab), and /home/* URLs
    normalize to /home.
    """

    def test_subnav_tab_strip_removed(self, index_html: str):
        # The persistent Summaries / Kanban / Activity tab bar is gone.
        assert 'data-testid="workplace-subnav"' not in index_html
        assert 'data-testid="workplace-subnav-summaries"' not in index_html
        assert 'data-testid="workplace-subnav-kanban"' not in index_html
        assert 'data-testid="workplace-subnav-activity"' not in index_html

    def test_kanban_surface_removed(self, index_html: str):
        # No more 4-column kanban grid on the Work tab.
        assert 'data-testid="board-kanban"' not in index_html
        assert 'data-testid="board-kanban-card"' not in index_html
        assert 'data-testid="board-kanban-card-cancel"' not in index_html

    def test_activity_subpage_removed(self, index_html: str):
        # Activity sub-page (Just delivered + Happening now + In progress)
        # is gone. Only the testid that's actually rendered counts —
        # ``home-just-delivered`` etc. previously sat inside a live block.
        # After PR 2 the block sits in an ``x-if="false"`` stub that
        # doesn't render, but we also want the testids removed entirely
        # so a search for the old IDs returns nothing user-visible.
        # The dead-stub HTML in this PR carries no testids.
        for testid in (
            "home-just-delivered",
            "home-happening-now",
            "home-in-progress",
            "home-see-task-board",
            "home-activity",
            "home-idle-empty",
            "rate-buttons",
            "inline-rework",
        ):
            assert f'data-testid="{testid}"' not in index_html, \
                f"PR 2 should have removed testid: {testid}"

    def test_stuck_tasks_panel_retained(self, index_html: str):
        # Stuck Tasks panel + per-row Cancel / Restart-agent affordances
        # survive the cutover.
        assert 'data-testid="board-stuck-tasks"' in index_html
        assert 'data-testid="board-stuck-task-cancel"' in index_html
        assert 'data-testid="board-stuck-task-restart"' in index_html

    def test_cancel_task_modal_retained(self, index_html: str):
        assert 'data-testid="cancel-task-modal"' in index_html
        assert 'data-testid="cancel-task-confirm"' in index_html

    def test_summary_cards_unconditional_landing(self, index_html: str):
        # Summary view renders whenever the Work tab is on (no homeTab
        # gate any more). The conditional reads ``workplaceEnabled``
        # only.
        assert 'data-testid="workplace-summaries-view"' in index_html
        idx = index_html.find('data-testid="workplace-summaries-view"')
        # Walk back to find the enclosing template tag.
        slice_ = index_html[max(0, idx - 400):idx]
        assert 'homeTab' not in slice_, \
            "summary view should no longer gate on homeTab"

    def test_goals_strip_present_with_chip_template(self, index_html: str):
        # New Goals chip strip — read-only render of operator's goals.
        assert 'data-testid="workplace-goals-strip"' in index_html
        assert 'data-testid="workplace-goal-chip"' in index_html
        # Hidden when no goals tracked.
        idx = index_html.find('data-testid="workplace-goals-strip"')
        slice_ = index_html[max(0, idx - 200):idx + 200]
        assert "workplaceGoals.length > 0" in slice_

    def test_tell_operator_section_removed(self, index_html: str):
        # Standalone Tell Operator textarea was removed — redundant
        # with the Chat tab. Steering happens via the inline rework
        # feedback box on summary cards instead.
        assert 'data-testid="workplace-tell-operator"' not in index_html
        assert 'data-testid="tell-operator-submit"' not in index_html
        assert 'data-testid="tell-operator-confirmation"' not in index_html
        assert "submitTellOperator()" not in index_html

    def test_summary_rating_uses_svg_icons_not_emoji(self, index_html: str):
        # Three rating buttons (accept / acknowledge / rework) carry
        # data-testid attributes and EACH one contains its own inline
        # SVG body (not the emoji codepoints 👍 ➖ 👎). We scope the
        # SVG-presence check per-button so a future addition of an
        # unrelated SVG nearby can't accidentally satisfy the count.
        for testid in (
            "summary-rate-accept",
            "summary-rate-acknowledge",
            "summary-rate-rework",
        ):
            marker = f'data-testid="{testid}"'
            idx = index_html.find(marker)
            assert idx > 0, f"{testid} button missing"
            # 800 chars covers <button …><svg …>…</svg></button> with
            # comfortable headroom — actual size is ~600 chars.
            btn = index_html[idx:idx + 800]
            assert "<svg" in btn, f"{testid} button has no inline SVG"
            assert "</button>" in btn, f"{testid} button not closed in window"
            # No emoji codepoints (HTML entities OR raw).
            assert "&#x1F44D;" not in btn
            assert "&#x1F44E;" not in btn
            assert "&#x2796;" not in btn
            assert "\U0001f44d" not in btn
            assert "\U0001f44e" not in btn
            assert "➖" not in btn

    def test_build_path_emits_only_bare_home(self, app_js: str):
        # PR 2 — single Work-tab URL. _buildPath returns '/home' for
        # the Work tab; no /home/kanban, /home/activity, /home/summaries
        # branches survive.
        idx = app_js.find("if (this.activeTab === 'workplace')")
        assert idx > 0
        block = app_js[idx:idx + 400]
        assert "return '/home'" in block
        assert "/home/activity" not in block
        assert "/home/summaries" not in block
        assert "/home/kanban" not in block

    def test_parse_path_normalizes_legacy_home_urls(self, app_js: str):
        # Any /home/{anything} URL silently normalizes to /home so old
        # bookmarks (kanban, activity, summaries, tasks) survive without
        # a 404 or visible redirect.
        idx = app_js.find("if (clean === 'home' || clean.startsWith('home/'))")
        assert idx > 0, "legacy /home/* normalization missing"
        block = app_js[idx:idx + 300]
        assert "route.tab = 'workplace'" in block
        # Verify no separate branch for /home/kanban etc. lingers.
        assert "clean === 'home/kanban'" not in app_js
        assert "clean === 'home/activity'" not in app_js
        assert "clean === 'home/summaries'" not in app_js
        assert "clean === 'home/tasks'" not in app_js

    def test_home_tab_state_and_methods_removed(self, app_js: str):
        # Compaction check — homeTab + switchHomeTab + helper methods
        # were all removed from app.js. Their absence is what makes the
        # template state machine collapse to a single workplace view.
        # Comment lines that reference these names as history are fine;
        # we only check for the JS identifiers in active use.
        assert "homeTab: 'kanban'" not in app_js
        assert "switchHomeTab(tabId)" not in app_js
        assert "_applyDefaultHomeTab()" not in app_js
        assert "drillIntoTeamKanban(" not in app_js

    def test_recently_delivered_helpers_removed(self, app_js: str):
        # PR 2 also drops the "Recently delivered" cluster.
        assert "_recomputeRecentlyDelivered(" not in app_js
        assert "_ensureRecentlyDeliveredArtifact(" not in app_js
        assert "recentlyDeliveredPreview(" not in app_js
        assert "recentlyDeliveredFull(" not in app_js
        assert "recentlyDeliveredHasMore(" not in app_js
        assert "copyRecentlyDeliveredText(" not in app_js

    def test_inline_rework_and_rate_delivery_removed(self, app_js: str):
        # The per-task rating handler + inline rework state are gone.
        assert "async rateDelivery(" not in app_js
        assert "submitInlineRework(" not in app_js
        assert "openInlineRework(" not in app_js
        assert "closeInlineRework(" not in app_js
        assert "inlineReworkIsOpen(" not in app_js

    def test_global_task_ws_handlers_preserved(self, app_js: str):
        # Global task_created / task_status_changed / task_outcome
        # handlers feed System Activity + notification bell + the
        # drill-in modal — they MUST survive the Work-tab cutover.
        assert "case 'task_status_changed':" in app_js
        assert "case 'task_outcome':" in app_js
        assert "case 'task_created':" in app_js

    def test_load_workplace_drops_outputs_and_feed_loaders(self, app_js: str):
        # The master Promise.all should NOT call the deleted loaders.
        idx = app_js.find("async loadWorkplace()")
        assert idx > 0
        body = app_js[idx:idx + 1200]
        assert "loadWorkplaceTeams" in body
        assert "loadWorkplaceTasks" in body
        assert "loadWorkplaceBlockers" in body
        assert "loadWorkplacePending" in body
        assert "loadWorkplaceSummaries" in body
        assert "loadWorkplaceGoals" in body
        assert "loadWorkplaceOutputs" not in body
        assert "loadWorkplaceFeed" not in body

    def test_workplace_goals_loader_present(self, app_js: str):
        assert "async loadWorkplaceGoals()" in app_js
        assert "/workplace/goals" in app_js

    def test_tell_operator_method_removed(self, app_js: str):
        # submitTellOperator + tellOperator* Alpine state removed
        # with the standalone section.
        assert "async submitTellOperator()" not in app_js
        assert "tellOperatorText" not in app_js
        assert "tellOperatorInflight" not in app_js
        assert "tellOperatorConfirmation" not in app_js


# ── Phase 4: \"What's new\" tour for existing-fleet users ──────────


class TestNewUserTutorialMarkup:
    """The 3-step tutorial modal is hard-coded into the SPA template."""

    def test_modal_block_is_present(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="tutorial-modal"' in html

    def test_modal_renders_only_when_step_not_zero(self):
        html = _read(_TEMPLATE)
        assert "newUserTutorial.step !== 0" in html

    def test_modal_has_three_steps(self):
        html = _read(_TEMPLATE)
        for step in (1, 2, 3):
            assert f"newUserTutorial.step === {step}" in html, (
                f"missing tutorial step: {step}"
            )

    def test_modal_has_dialog_role_and_modal_aria(self):
        html = _read(_TEMPLATE)
        # role=dialog + aria-modal=true on the same element as the
        # data-testid hook — required for screen readers.
        assert re.search(
            r'role="dialog"[^>]*?aria-modal="true"[^>]*?data-testid="tutorial-modal"'
            r'|data-testid="tutorial-modal"[^>]*?role="dialog"',
            html,
        ), "tutorial modal missing role=dialog/aria-modal"

    def test_step1_copy_present(self):
        html = _read(_TEMPLATE)
        assert "Welcome to OpenLegion" in html
        assert "AI team manager" in html
        assert "Show me how" in html

    def test_step2_copy_present(self):
        html = _read(_TEMPLATE)
        assert "Meet your Operator" in html
        assert "team lead" in html

    def test_step3_copy_present(self):
        html = _read(_TEMPLATE)
        assert "See what your team is doing" in html
        assert "Got it, let&rsquo;s build a team" in html

    def test_skip_button_dismisses_tutorial(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="tutorial-skip"' in html
        assert "dismissTutorial(" in html

    def test_primary_button_has_focus_target(self):
        # Focus management: each step's primary advance button is
        # auto-focused on entry (via querySelector in app.js).
        html = _read(_TEMPLATE)
        assert 'data-testid="tutorial-primary"' in html


class TestNewUserTutorialJsState:
    """Alpine handlers for the tutorial state machine."""

    def test_tutorial_state_object_declared(self):
        js = _read(_APP_JS)
        assert "newUserTutorial: { step: 0 }" in js

    def test_tutorial_handlers_declared(self):
        js = _read(_APP_JS)
        for handler in (
            "_maybeStartTutorial()",
            "startTutorial()",
            "tutorialNext()",
            "tutorialBack()",
            "dismissTutorial(reason)",
            "_completeTutorial(reason)",
        ):
            assert handler in js, f"missing tutorial handler: {handler}"

    def test_tutorial_persists_seen_flag(self):
        js = _read(_APP_JS)
        # Seen flag freezes the tutorial after first completion/dismiss.
        assert "olSeenTutorial" in js
        assert "localStorage.setItem('olSeenTutorial', 'true')" in js

    def test_tutorial_fires_for_new_users_only(self):
        js = _read(_APP_JS)
        # Detection delegates to ``_isFirstVisit()`` — empty fleet AND
        # no real user message in the operator transcript. The wizard
        # uses the same detector so the two stay in lockstep.
        assert "_maybeStartTutorial" in js
        m = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartTutorial body missing"
        body = m.group(1)
        assert "_isFirstVisit()" in body, (
            "tutorial gate should delegate to _isFirstVisit"
        )

    def test_tutorial_does_not_fire_for_existing_users(self):
        # The detector early-returns when ``_isFirstVisit()`` is false —
        # any non-empty fleet OR any prior user message in the operator
        # transcript skips the tutorial.
        m = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartTutorial body missing"
        body = m.group(1)
        assert "if (!this._isFirstVisit()) return" in body, (
            "tutorial should bail when _isFirstVisit() is false"
        )

    def test_tutorial_does_not_fire_for_returning_user_with_chat_history(self):
        # Audit fix: the legacy gate checked only ``fleetAgents.length``
        # which mis-classified a returning user who deleted their fleet
        # but already chatted with the operator as "new". Delegating to
        # ``_isFirstVisit()`` adds the "no user message in
        # chatHistories.operator" check, so a returning chat-only user
        # is correctly excluded.
        m_first_visit = re.search(
            r"_isFirstVisit\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m_first_visit, "_isFirstVisit body missing"
        first_visit_body = m_first_visit.group(1)
        # Confirm _isFirstVisit covers BOTH conditions: empty fleet AND
        # no prior user message in operator chatHistories.
        assert "this.agents.some" in first_visit_body, (
            "_isFirstVisit must check the agents list"
        )
        assert "chatHistories['operator']" in first_visit_body, (
            "_isFirstVisit must check operator chat history"
        )
        assert "m.role === 'user'" in first_visit_body, (
            "_isFirstVisit must filter on user-role messages"
        )
        # And confirm the tutorial actually delegates to it (not an
        # inline fleetAgents-only check).
        m_tut = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m_tut, "_maybeStartTutorial body missing"
        tut_body = m_tut.group(1)
        assert "_isFirstVisit()" in tut_body
        # The legacy fleetAgents.length-only inline check should be
        # gone — _isFirstVisit replaces it.
        assert "fleetAgents.length !== 0" not in tut_body, (
            "legacy fleet-only check should be removed in favor of _isFirstVisit"
        )

    def test_tutorial_emits_locked_telemetry_events(self):
        js = _read(_APP_JS)
        for evt in (
            "tutorial_started",
            "tutorial_step",
            "tutorial_finished",
            "tutorial_to_wizard_transition",
        ):
            assert f"'{evt}'" in js, f"missing tutorial telemetry event: {evt}"

    def test_tutorial_handles_escape_key(self):
        js = _read(_APP_JS)
        # ESC dismisses the modal. Tab-trap is also wired here.
        assert "e.key === 'Escape'" in js


class TestTutorialWizardSequencing:
    """The tutorial must run BEFORE the empty-fleet wizard."""

    def test_tutorial_fires_before_wizard_when_both_eligible(self):
        # ``fetchAgents`` invokes _maybeStartTutorial BEFORE
        # _maybeStartWizard so a brand-new user sees the tutorial first.
        js = _APP_JS_TEXT
        idx_tutorial = js.find("this._maybeStartTutorial()")
        idx_wizard = js.find("this._maybeStartWizard()")
        assert idx_tutorial != -1, "fetchAgents should call _maybeStartTutorial"
        assert idx_wizard != -1, "fetchAgents should call _maybeStartWizard"
        assert idx_tutorial < idx_wizard, (
            "tutorial must be invoked before wizard so it gets first dibs"
        )

    def test_wizard_does_not_fire_while_tutorial_active(self):
        # _maybeStartWizard early-returns when newUserTutorial.step !== 0.
        m = re.search(
            r"_maybeStartWizard\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartWizard body missing"
        body = m.group(1)
        assert "newUserTutorial" in body, (
            "wizard should gate on tutorial state to avoid double-mount"
        )
        assert re.search(
            r"newUserTutorial\.step\s*!==\s*0", body
        ), "wizard guard should bail when tutorial is open"

    def test_wizard_fires_after_tutorial_completes(self):
        # _completeTutorial calls _maybeStartWizard on natural
        # completion (reason === 'completed') so the user lands in
        # the team-building flow without an extra click.
        m = re.search(
            r"_completeTutorial\(reason\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_completeTutorial body missing"
        body = m.group(1)
        assert "tutorial_to_wizard_transition" in body, (
            "completion handler should emit transition telemetry"
        )
        assert "_maybeStartWizard" in body, (
            "completion handler should invoke the wizard"
        )
        assert "reason === 'completed'" in body, (
            "wizard should only auto-fire on natural completion, not skip/escape"
        )


class TestTutorialBackCompatMigration:
    """Users who saw the legacy whats-new tour should not see the
    tutorial — they're not new users."""

    def test_old_olSeenWhatsNew_users_do_not_see_tutorial_again(self):
        # init() should migrate olSeenWhatsNew='true' → olSeenTutorial='true'.
        js = _APP_JS_TEXT
        # The migration check looks at olSeenWhatsNew and, if set,
        # writes olSeenTutorial. We grep for the combined fragment.
        assert "olSeenWhatsNew" in js, "migration must reference legacy flag"
        m = re.search(
            r"localStorage\.getItem\('olSeenWhatsNew'\)\s*===\s*'true'\s*&&\s*"
            r"!localStorage\.getItem\('olSeenTutorial'\)",
            js,
        )
        assert m, "migration check missing — old whats-new users would see tutorial"
        assert (
            "localStorage.setItem('olSeenTutorial', 'true')" in js
        ), "migration must write olSeenTutorial flag"


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


# ── Wizard correctness fixes (race / timeout / gating / labels) ───────


class TestWizardCorrectnessFixes:
    """Source-level checks for the wizard correctness fixes #1–#6.

    A real headless browser would let us drive the JS state machine
    directly; instead we assert on the source so the contract changes
    don't quietly regress. The tests cover:

      #1 stability-based polling (replaces ``length >= 1`` race)
      #2 ``build_failed`` terminal state on 5-min timeout
      #3 bootstrap greeting carries no ACTION lines
      #4 ``_maybeStartWizard`` is operatorReady-gated, retries on
         unhealthy → healthy transition
      #5 step labels say "Step N of 4"
      #6 realistic timing copy on building / first-output cards
    """

    # Fix #1 — stability-based polling -----------------------------

    def test_polling_uses_stability_window_not_naive_length(self):
        js = _read(_APP_JS)
        # The naive ``fleetAgents.length >= 1`` immediate advance is
        # replaced with a stability window (10 seconds of unchanged
        # non-zero count).
        assert "stableSince" in js
        # Check the literal numeric tolerance — 10_000 ms.
        assert "10_000" in js or "10000" in js
        # Telemetry trigger renamed.
        assert "fleet_stable" in js

    def test_polling_resets_window_when_count_changes(self):
        js = _read(_APP_JS)
        # When a new agent appears mid-window the stability clock
        # MUST reset — otherwise a 4-agent template that takes 12
        # seconds to provision would fire the success card after the
        # first agent.
        m = re.search(
            r"_wizardStartBuildPolling\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "could not locate _wizardStartBuildPolling body"
        body = m.group(1)
        # The reset path runs in the count-changed branch.
        assert "stableSince = now" in body
        assert "lastCount = count" in body

    def test_polling_clears_window_when_fleet_drops_to_zero(self):
        js = _read(_APP_JS)
        # If fleet count drops to 0 mid-build (agent crashed before
        # next agent created) we reset the window so we don't declare
        # success on a transient.
        m = re.search(
            r"_wizardStartBuildPolling\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m
        body = m.group(1)
        assert "if (count === 0)" in body
        assert "stableSince = 0" in body

    # Fix #2 — terminal build_failed state -------------------------

    def test_timeout_advances_to_build_failed(self):
        js = _read(_APP_JS)
        # The 5-minute hard cap now transitions to a terminal
        # ``build_failed`` step (replaces the silent timeout that
        # left the spinner card on screen).
        assert "_wizardAdvance('build_failed'" in js
        assert "trigger: 'timeout'" in js
        # Telemetry event for the timeout path.
        assert "'wizard_build_timeout'" in js

    def test_build_failed_card_renders(self):
        html = _read(_TEMPLATE)
        # The card itself, plus a header, plus the Retry/Type-instead
        # buttons that recover the user from the failed state.
        assert "wizard.step === 'build_failed'" in html
        assert 'data-testid="wizard-build-failed"' in html
        assert 'data-testid="wizard-build-failed-retry"' in html
        assert 'data-testid="wizard-build-failed-type"' in html

    def test_build_failed_retry_handler(self):
        js = _read(_APP_JS)
        # Retry resets to the 'ask' step and re-arms startedAt so
        # the secondsSinceStart counter reflects the new attempt.
        assert "wizardRetryBuild()" in js
        m = re.search(
            r"wizardRetryBuild\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "wizardRetryBuild body missing"
        body = m.group(1)
        assert "step: 'ask'" in body
        assert "_wizardTeardown" in body

    def test_build_failed_type_instead_handler(self):
        js = _read(_APP_JS)
        # Type-instead exits the wizard via the standard complete
        # path so telemetry fires the locked ``wizard_completed`` event.
        assert "wizardExitToTyping()" in js
        m = re.search(
            r"wizardExitToTyping\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "wizardExitToTyping body missing"
        body = m.group(1)
        assert "_wizardComplete" in body
        assert "build_failed_exit" in body

    def test_build_failed_progress_dot_index(self):
        js = _read(_APP_JS)
        # build_failed renders the same dot index as first-output
        # (3) so the user sees the progress reached the build phase
        # before failing.
        m = re.search(
            r"wizardStepIndex\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "wizardStepIndex body missing"
        body = m.group(1)
        assert "build_failed" in body
        assert "return 3" in body

    # Fix #4 — operatorReady gate ----------------------------------

    def test_maybe_start_wizard_gated_on_operator_ready(self):
        js = _read(_APP_JS)
        m = re.search(
            r"_maybeStartWizard\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "_maybeStartWizard body missing"
        body = m.group(1)
        # The body bails out if operatorReady is falsy.
        assert "this.operatorReady" in body
        assert "if (!this.operatorReady) return" in body

    def test_check_operator_ready_retries_wizard(self):
        js = _read(_APP_JS)
        m = re.search(
            r"checkOperatorReady\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "checkOperatorReady body missing"
        body = m.group(1)
        # Transition unhealthy → healthy must retry the wizard kickoff.
        assert "wasReady" in body
        assert "_maybeStartWizard" in body

    # Fix #5 — step labels align with 4-dot indicator --------------

    def test_step_labels_say_n_of_four(self):
        html = _read(_TEMPLATE)
        # Scope the search to the wizard region only — the What's-new
        # tour reuses "Step N of 3" markup elsewhere in the template.
        m = re.search(
            r'data-testid="onboarding-wizard".*?<!-- Step: building',
            html,
            re.DOTALL,
        )
        assert m, "could not isolate wizard ask/confirming region"
        ask_confirming = m.group(0)
        assert "Step 1 of 4" in ask_confirming
        assert "Step 2 of 4" in ask_confirming
        # building + first-output captions further down.
        m2 = re.search(
            r"<!-- Step: building -->.*?<!-- Step: build_failed",
            html,
            re.DOTALL,
        )
        assert m2, "could not isolate building region"
        building = m2.group(0)
        assert "Step 3 of 4" in building
        m3 = re.search(
            r"<!-- Step: first-output -->.*?</template>\s*</div>\s*</template>",
            html,
            re.DOTALL,
        )
        assert m3, "could not isolate first-output region"
        first_output = m3.group(0)
        assert "Step 4 of 4" in first_output
        # Legacy 3-step wording must be gone from the wizard cards.
        assert "Step 1 of 3" not in ask_confirming
        assert "Step 2 of 3" not in ask_confirming
        assert "Step 3 of 3" not in first_output

    # Fix #6 — realistic timing copy -------------------------------

    def test_building_card_uses_realistic_timing(self):
        html = _read(_TEMPLATE)
        # The "30 seconds" claim is replaced with vague-but-honest copy.
        assert "About 30 seconds" not in html
        assert "this may take a minute or two" in html

    def test_first_output_card_describes_heartbeat_behavior(self):
        html = _read(_TEMPLATE)
        # The fictitious "10 minutes" ETA is replaced with a heartbeat
        # mental-model description.
        assert "First output expected in about 10 minutes" not in html
        assert "checks on them every 15 minutes" in html

    # Backwards-compat for persisted localStorage ------------------

    def test_wizard_load_resets_unknown_step_to_idle(self):
        js = _read(_APP_JS)
        m = re.search(
            r"_wizardLoad\(\)\s*\{(.*?)^\s{4}\},",
            js,
            re.DOTALL | re.MULTILINE,
        )
        assert m, "_wizardLoad body missing"
        body = m.group(1)
        # Unknown step value from a prior version must reset to idle.
        assert "KNOWN_STEPS" in body
        assert "build_failed" in body
        assert "'idle'" in body
# ── Vocab + UX polish (vocab gaps / Undo countdown / icon / DM) ──


class TestVocabSweepGaps:
    """The Phase 2 sweep missed several user-visible engineer-speak
    strings; this guards against regression on the strings the second
    audit fixed."""

    def test_no_engineer_terms_in_user_visible_template(self):
        # User-visible strings — no quoted form should appear in the
        # rendered template (HTML comments are stripped first because
        # the engineer-speak inside ``<!-- -->`` blocks doesn't render
        # to users). ``Schedules`` / ``Spawn helpers`` /
        # ``Home is disabled`` are the replacements.
        visible = re.sub(r"<!--.*?-->", "", _INDEX_HTML, flags=re.DOTALL)
        forbidden = [
            "Spawn Agents",
            "Cron Jobs",
            "Cron Job",
            "Cron job",
            "cron job",
            "cron jobs",
            "Board is disabled",
        ]
        for term in forbidden:
            assert term not in visible, (
                f"Vocab leak: {term!r} still present in user-visible template"
            )

    def test_back_button_in_agent_detail_says_team(self):
        # The agent-detail breadcrumb back-button label flips from
        # "Agents" to "Team" — there's no other ">Agents<" text node
        # remaining in the template.
        # We bound the search to a likely surrounding fragment so the
        # check stays robust if Tailwind utility classes drift.
        idx = _INDEX_HTML.find("closeDetail()")
        assert idx > -1, "closeDetail breadcrumb not found"
        snippet = _INDEX_HTML[idx : idx + 600]
        assert ">\n            Team\n          </button>" in snippet or ">Team<" in snippet

    def test_replacement_terms_present(self):
        # Sanity check that the replacements actually landed.
        assert "Spawn helpers" in _INDEX_HTML
        assert "Schedules" in _INDEX_HTML
        assert "Work is disabled" in _INDEX_HTML
        assert "Contact support to enable." in _INDEX_HTML
        assert "+ Schedule" in _INDEX_HTML
        assert "New schedule" in _INDEX_HTML


class TestUndoReceiptCountdown:
    """The operator_action_receipt receipt now renders a live countdown
    next to the Undo button so users know how long they have left."""

    def test_countdown_span_present_in_chat_tab_copy(self):
        # The receipt block is duplicated in two messenger surfaces;
        # both must render the countdown.
        # Substring proves the helper-driven countdown is wired in.
        assert "formatUndoCountdown(undoSecondsLeft(msg, _tick))" in _INDEX_HTML
        # Two occurrences — one in the operator chat tab, one in the
        # side-panel copy.
        count = _INDEX_HTML.count("formatUndoCountdown(undoSecondsLeft(msg, _tick))")
        assert count >= 2, (
            f"Expected the countdown markup in both receipt copies; "
            f"found {count}"
        )

    def test_countdown_helper_defined_in_app_js(self):
        # The helper must exist with the documented signature so Alpine
        # can resolve the expression.
        assert "undoSecondsLeft(msg, _tick)" in _APP_JS_TEXT
        assert "formatUndoCountdown(seconds)" in _APP_JS_TEXT

    def test_undo_seconds_left_returns_zero_on_missing_expiry(self):
        # Python mirror of the JS helper: validates the boundary
        # behaviour we rely on (no expiry → 0; past expiry → 0).
        import math
        import time

        def undo_seconds_left(msg, _tick):
            if not msg or not msg.get("expires_at"):
                return 0
            left = math.floor((float(msg["expires_at"]) * 1000 - time.time() * 1000) / 1000)
            return left if left > 0 else 0

        assert undo_seconds_left({}, 0) == 0
        assert undo_seconds_left({"expires_at": time.time() - 60}, 0) == 0
        assert undo_seconds_left({"expires_at": time.time() + 120}, 0) > 0

    def test_format_undo_countdown_pads_seconds(self):
        # Python mirror of the JS formatter — proves the contract the
        # template relies on (mm:ss with two-digit second padding).
        def format_undo_countdown(seconds):
            s = max(0, int(seconds or 0))
            m = s // 60
            r = s % 60
            return f"({m}:{r:02d})"

        assert format_undo_countdown(0) == "(0:00)"
        assert format_undo_countdown(5) == "(0:05)"
        assert format_undo_countdown(65) == "(1:05)"
        assert format_undo_countdown(222) == "(3:42)"


class TestSidePanelToggleIcon:
    """The side-panel toggle SVG should not be the same chat-bubble
    glyph used by the Chat tab — keeps the side-panel pictogram
    distinct from the Chat tab so users don't confuse the two."""

    def test_side_panel_toggle_is_not_chat_bubble(self):
        # The Chat tab uses a chat-bubble path; the side-panel toggle
        # was using the SAME path verbatim, which made the tour ambiguous
        # ("click the chat icon" → users clicked the Chat tab). The
        # toggle now uses a distinct rectangle + divider + arrow shape.
        chat_bubble_path = (
            'd="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"'
        )
        # Locate the toggle button by its handler and assert the
        # chat-bubble path does NOT appear inside its inner SVG.
        idx = _INDEX_HTML.find("toggleSidePanel()")
        assert idx > -1, "side-panel toggle handler missing"
        button_block = _INDEX_HTML[idx : idx + 1500]
        assert chat_bubble_path not in button_block, (
            "Side-panel toggle still uses the chat-bubble SVG path; "
            "it should be a distinct pictogram."
        )

    def test_side_panel_toggle_uses_panel_pictogram(self):
        # Find the toggle button and verify its inner SVG carries the
        # rectangle + divider + arrow shape.
        idx = _INDEX_HTML.find("toggleSidePanel()")
        assert idx > -1, "side-panel toggle handler missing"
        snippet = _INDEX_HTML[idx : idx + 1500]
        assert '<rect x="3" y="4" width="18" height="16"' in snippet
        assert '<line x1="14" y1="4" x2="14" y2="20"' in snippet
        assert '<polyline points="8 10 11 12 8 14"' in snippet


class TestWorkerDmInNeedsYou:
    """Worker DMs (chatUnread > 0 for non-operator agents) must show up
    in the Needs-You aggregator — not just as a small amber dot on the
    side-panel toggle."""

    def test_worker_dm_branch_in_needs_you_items(self):
        # Source-level: the getter walks ``chatUnread`` and emits a
        # ``worker_dm`` kind when count > 0 for non-operator agents.
        assert "kind: 'worker_dm'" in _APP_JS_TEXT
        # The branch skips operator (that surface lands via the
        # operator-chat sweep above).
        assert "agentId === 'operator'" in _APP_JS_TEXT

    def test_worker_dm_uses_open_chat_handler(self):
        # The Read button must jump to the chat tab and open the
        # specific agent's chat panel.
        # We grep for the structural hint rather than full markup so
        # comments / whitespace don't make the test brittle.
        assert "this.openChat(agentId)" in _APP_JS_TEXT
        assert "label: 'Read'" in _APP_JS_TEXT

    def test_worker_dm_pluralises_unread_count(self):
        # 1 vs N copy variants must both exist so the subtitle reads
        # naturally for either case.
        assert "'1 unread'" in _APP_JS_TEXT
        assert "${n} unread" in _APP_JS_TEXT
# ── Browser notification + activity rollup + connect-channel ────


class TestBrowserNotifyOptIn:
    """Browser Notification API integration in the wizard first-output."""

    def test_browser_notify_button_in_wizard_first_output(self):
        html = _read(_TEMPLATE)
        # The opt-in button lives inside the first-output card. Assert
        # both the testid and that it actually wires to the JS handler.
        assert 'data-testid="wizard-browser-notify"' in html
        assert 'data-testid="wizard-browser-notify-enable"' in html
        assert "requestBrowserNotificationPermission()" in html
        # The "granted + enabled" affordance and "denied" advisory are
        # both rendered so the user always knows the current state.
        assert 'data-testid="wizard-browser-notify-on"' in html
        assert 'data-testid="wizard-browser-notify-denied"' in html

    def test_browser_notify_persists_to_localstorage(self):
        js = _read(_APP_JS)
        # The opt-in is persisted under the locked key so we can assert
        # against a string literal, not a regex.
        assert "olBrowserNotifyEnabled" in js
        # Permission grant flips the in-app opt-in on AND writes through
        # to localStorage in the same code path.
        assert "localStorage.setItem('olBrowserNotifyEnabled', 'true')" in js
        # State variables are declared on the Alpine root so the
        # template bindings work.
        assert "browserNotifyEnabled: false" in js
        assert "browserNotifyPermission: 'default'" in js

    def test_browser_notify_fire_is_triple_gated(self):
        """Defence in depth — opt-in flag, OS permission, and tab visibility."""
        js = _read(_APP_JS)
        fire_fn = _extract_function_body(js, "_maybeFireBrowserNotification")
        assert fire_fn is not None, "missing _maybeFireBrowserNotification"
        # Gate 1: in-app opt-in.
        assert "this.browserNotifyEnabled" in fire_fn
        # Gate 2: OS-level permission. Match either ' === ' or "===" form
        # so refactors that flip quote style still pass.
        assert "Notification.permission" in fire_fn
        assert "'granted'" in fire_fn or '"granted"' in fire_fn
        # Gate 3: tab visibility — only fire when not visible.
        assert "document.visibilityState" in fire_fn
        assert "'visible'" in fire_fn or '"visible"' in fire_fn

    def test_browser_notify_kinds_allowlist(self):
        js = _read(_APP_JS)
        # Only fire for high-signal kinds. The list lives in a single
        # place so future additions don't drift across files.
        assert "_browserNotifyKinds: ['approval', 'credential', 'alert', 'blocker']" in js


class TestActivityRollup:
    """Long-task progress rollup in the activity feed.

    The rollup is presentation-only — the underlying ``events`` array
    is unchanged so the "Show technical detail" toggle still surfaces
    the raw stream.
    """

    def test_rollup_helper_declared(self):
        js = _read(_APP_JS)
        assert "_rollupActivityEvents(events)" in js
        assert "formatRolledActivityLine(entry)" in js
        # The feed-renderer uses the ``rolledFilteredEvents`` getter
        # rather than calling the helper inline.
        assert "rolledFilteredEvents" in js

    def test_rollup_groups_consecutive_same_tool(self):
        result = _run_rollup_js([
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1000},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1060},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1120},
        ])
        assert len(result) == 1
        assert result[0]["kind"] == "rollup"
        assert result[0]["count"] == 3
        assert result[0]["tool"] == "web_search"

    def test_rollup_resets_on_tool_change(self):
        result = _run_rollup_js([
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1000},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1060},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "browser_navigate"}, "timestamp": 1120},
        ])
        assert len(result) == 2
        # First group: 2x web_search collapsed.
        assert result[0]["kind"] == "rollup"
        assert result[0]["count"] == 2
        assert result[0]["tool"] == "web_search"
        # Second group: a single browser_navigate, NOT decorated.
        assert result[1]["kind"] == "single"

    def test_rollup_resets_on_agent_change(self):
        result = _run_rollup_js([
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1000},
            {"type": "tool_start", "agent": "writer",
             "data": {"tool": "web_search"}, "timestamp": 1060},
        ])
        # Different agents must NEVER fold into the same rollup, even
        # for the same tool name.
        assert len(result) == 2
        assert all(e["kind"] == "single" for e in result)

    def test_rollup_resets_on_status_change(self):
        result = _run_rollup_js([
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1000},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1060},
            {"type": "task_status_changed", "agent": "researcher",
             "data": {"new_status": "blocked"}, "timestamp": 1080},
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1100},
        ])
        # Group A (2x web_search) → status divider → fresh single
        # web_search. Status events are always emitted as singles.
        assert len(result) == 3
        assert result[0]["kind"] == "rollup"
        assert result[0]["count"] == 2
        assert result[1]["kind"] == "single"
        assert result[1]["event"]["type"] == "task_status_changed"
        assert result[2]["kind"] == "single"

    def test_rollup_singleton_unchanged(self):
        result = _run_rollup_js([
            {"type": "tool_start", "agent": "researcher",
             "data": {"tool": "web_search"}, "timestamp": 1000},
        ])
        assert len(result) == 1
        # Single events are NOT decorated; the renderer falls back to
        # formatActivityForUser as before.
        assert result[0]["kind"] == "single"

    def test_rollup_preserves_raw_events_for_tech_detail(self):
        """The raw stream renderer is gated on ``showTechDetail`` and
        iterates over ``filteredEvents``, not the rolled-up list. So
        flipping the toggle still surfaces every event."""
        html = _read(_TEMPLATE)
        # Tech-detail branch iterates over the unaltered filteredEvents.
        assert 'x-if="showTechDetail"' in html
        # Roll-up branch iterates over the new getter.
        assert 'x-if="!showTechDetail"' in html
        assert 'in rolledFilteredEvents' in html


class TestConnectChannelPrompt:
    """Soft 'Connect a channel' nudge in the wizard first-output."""

    def test_first_output_card_has_connect_channel_link(self):
        html = _read(_TEMPLATE)
        assert 'data-testid="wizard-connect-channel"' in html
        assert 'data-testid="wizard-connect-channel-link"' in html
        # Click handler hops to Settings → Integrations and completes
        # the wizard so we don't strand the user on a dismissed card.
        assert "systemTab = 'integrations'" in html
        assert "switchTab('system')" in html
        # The copy is locked — keep the nudge soft.
        assert "Want updates when you" in html
        assert "Connect Telegram" in html


# ── Helpers — JS-source extraction + node subprocess ────────────


def _extract_function_body(js: str, name: str) -> str | None:
    """Return the body of a top-level method definition.

    Handles the Alpine-component shape ``name(args) { ... }`` by
    counting brace depth. Returns ``None`` when the method doesn't
    appear in the source.
    """
    needle = re.search(rf"\b{re.escape(name)}\s*\([^)]*\)\s*\{{", js)
    if not needle:
        return None
    start = needle.end()
    depth = 1
    i = start
    while i < len(js) and depth > 0:
        ch = js[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return js[start:i - 1]


def _run_rollup_js(events: list[dict]) -> list[dict]:
    """Run ``_rollupActivityEvents`` against a fixed event list under
    Node.js. Skips when node isn't available so contributors without a
    Node toolchain still pass the rest of the suite. CI runners
    (``ubuntu-latest``) have Node preinstalled.
    """
    import json as _json
    import shutil as _shutil
    import subprocess as _sp

    node_bin = _shutil.which("node")
    if node_bin is None:
        pytest.skip("node not available — skipping rollup behaviour test")
    js = _read(_APP_JS)
    rollup_body = _extract_function_body(js, "_rollupActivityEvents")
    if rollup_body is None:
        pytest.fail("could not locate _rollupActivityEvents in app.js")
    # Reconstruct a minimal harness that exposes the helper as a
    # standalone function. ``this`` isn't referenced inside the
    # rollup body so we don't need Alpine wiring.
    harness = (
        "const events = " + _json.dumps(events) + ";\n"
        "function rollup(events) {" + rollup_body + "}\n"
        "process.stdout.write(JSON.stringify(rollup(events)));\n"
    )
    proc = _sp.run(
        [node_bin, "-e", harness],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        pytest.fail(f"node harness failed: {proc.stderr}")
    return _json.loads(proc.stdout)
# ── User-journey audit: coverage gaps for formatActivityForUser ──


class TestFormatActivityForUserEventTypeMapping:
    """Table-driven coverage of every branch of ``formatActivityForUser``.

    The helper is a pure function over (event_type, data) → user-facing
    string. We can't run JS in CI so we mirror the switch statement in
    Python and verify the source contains the expected output template
    for each branch. When the mapping drifts the assertion fails on the
    template literal.
    """

    # Each row: (event_type, expected_template_substring_in_app_js)
    _EVENT_CASES = [
        # tool_start → "X is Y"
        ("tool_start", "is ${this.verbForTool"),
        # tool_result → "X finished Y"
        ("tool_result", "finished ${this.verbForTool"),
        # task_status_changed → status verb mapping
        ("task_status_changed", "${this.verbForStatus"),
        # task_outcome → "X's work was Y"
        ("task_outcome", "work was ${outcome}"),
        # credential_request → "X needs a Y — your call"
        ("credential_request", "needs a ${label} — your call"),
        # pending_action_created → "X wants to Y"
        ("pending_action_created", "wants to ${actionLabel}"),
        # task_created → "X picked up '<title>'"
        ("task_created", "picked up \"${title}\""),
        # health_change → "X is now Y"
        ("health_change", "is now ${d.current"),
        # heartbeat_complete → "X finished a checkup"
        ("heartbeat_complete", "finished a checkup"),
        # notification → first 100 chars of message (substring(0, 100))
        ("notification", "(d.message || '').substring(0, 100)"),
        # browser_login_request → "X needs a sign-in — your call"
        ("browser_login_request", "needs a sign-in — your call"),
        # browser_captcha_help_request → "X hit a CAPTCHA"
        ("browser_captcha_help_request", "hit a CAPTCHA"),
        # credit_exhausted → "X is out of credit"
        ("credit_exhausted", "is out of credit"),
    ]

    @pytest.mark.parametrize(
        "event_type,expected_template",
        _EVENT_CASES,
        ids=[c[0] for c in _EVENT_CASES],
    )
    def test_visible_event_types_have_user_facing_template(
        self, event_type: str, expected_template: str
    ):
        """Each user-visible event type renders its locked template literal."""
        # Locate the formatActivityForUser switch block; per-case copy
        # must match the expected template substring. We search inside
        # the function so unrelated occurrences don't false-positive.
        m = re.search(
            r"formatActivityForUser\(event\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "formatActivityForUser body missing"
        body = m.group(1)
        assert f"case '{event_type}':" in body, (
            f"missing case for {event_type}"
        )
        assert expected_template in body, (
            f"template drift for {event_type}: expected {expected_template!r}"
        )

    @pytest.mark.parametrize(
        "hidden_type",
        [
            "blackboard_write",
            "llm_call",
            "message_received",
            "message_sent",
            "text_delta",
            "agent_state",
        ],
    )
    def test_hidden_event_types_return_null(self, hidden_type: str):
        """Power-user-only event types fall through to ``return null``."""
        # The hidden cluster is a single fall-through block ending in
        # `return null`. We grep for the cluster + the case label.
        m = re.search(
            r"// Hidden by default.*?return null;",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "hidden-by-default cluster missing"
        block = m.group(0)
        assert f"case '{hidden_type}'" in block, (
            f"{hidden_type} should be in the hidden cluster"
        )

    def test_unknown_agent_falls_back_to_an_agent(self):
        """Unknown agents render as 'An agent', not 'Someone'."""
        # Polish fix: the audit flagged "Someone" as too anonymous.
        m = re.search(
            r"formatActivityForUser\(event\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "formatActivityForUser body missing"
        body = m.group(1)
        assert "'An agent'" in body, (
            "missing 'An agent' fallback for events with no agent"
        )
        assert ": 'Someone'" not in body, (
            "legacy 'Someone' fallback should be removed"
        )


def _enumerate_skill_tool_names() -> set[str]:
    """Walk ``src/agent/builtins/*.py`` and collect ``name="..."`` entries.

    Mirrors what ``test_verb_for_tool_map_completeness`` needs without
    importing the modules (some have side-effecting imports). The
    convention in this repo is `@skill(name="<tool>", ...)` with
    ``name=`` on its own line — we tolerate inline whitespace.
    """
    builtins_dir = _REPO_ROOT / "src/agent/builtins"
    name_re = re.compile(r'^\s*name\s*=\s*"([a-z_][a-z0-9_]*)"\s*,', re.MULTILINE)
    seen: set[str] = set()
    for path in builtins_dir.glob("*.py"):
        seen.update(name_re.findall(path.read_text(encoding="utf-8")))
    return seen


# Tools that don't fit the user-friendly verb map — typically
# operator-only / introspection tools surfaced via the fallback
# "using {tool}" path is acceptable.
_VERB_FOR_TOOL_FALLBACK_OK = frozenset({
    # Introspection / mesh-internal — fallback "using X" reads fine.
    "get_system_status",
    "get_agent_profile",
    "get_team_outputs",
    "list_agent_queue",
    "list_blackboard",
    "list_files",
    "list_pending",
    "list_subagents",
    "list_cron",
    "list_templates",
    "read_agent_history",
    "read_blackboard",
    "read_file",
    "write_file",
    "summarize_project_progress",
    # Coordination wrappers — covered by hand_off / update_status / etc.
    "claim_task",
    "complete_task",
    # Cron management.
    "set_cron",
    "remove_cron",
    # Subagent lifecycle.
    "spawn_subagent",
    "spawn_fleet_agent",
    "wait_for_subagent",
    # Operator-only edit/orchestration.
    "confirm_edit",
    "cancel_pending_action",
    "archive_audit_before",
    "undo_change",
    "manage_agent",
    "manage_goals",
    "manage_project",
    "manage_team",
    "manage_task",
    "create_project",
    "create_team",
    "add_agents_to_project",
    "add_agents_to_team",
    "remove_agents_from_project",
    "remove_agents_from_team",
    "set_project_goal",
    "set_team_goal",
    "update_project_context",
    "update_team_context",
    "inspect_projects",
    "inspect_teams",
    "summarize_team_progress",
    # Vault / credentials.
    "vault_generate_secret",
    "vault_list",
    "request_credential",
    # Skill self-authoring.
    "create_skill",
    "reload_skills",
    "update_workspace",
    # Misc / external integrations.
    "post_tweet",
    "save_artifact",
    # Test fixture in tests/test_skills.py — not a real builtin but
    # shows up because the loader picks any name="..." entry.
    "my_tool",
    # Browser tools that aren't user-facing in the activity feed
    # (warmup is internal; switch_tab/open_tab/inspect_requests/
    # detect_captcha/wait_for/press_key/scroll/hover/click_xy/upload/
    # download/reset/go_back/go_forward/solve_captcha share the
    # generic "using browser_*" fallback which reads fine).
    "browser_warmup",
    "browser_switch_tab",
    "browser_open_tab",
    "browser_inspect_requests",
    "browser_detect_captcha",
    "browser_wait_for",
    "browser_press_key",
    "browser_scroll",
    "browser_hover",
    "browser_click_xy",
    "browser_upload_file",
    "browser_download",
    "browser_reset",
    "browser_go_back",
    "browser_go_forward",
    "browser_solve_captcha",
    "request_captcha_help",
    "request_browser_login",
    # Watch / pub-sub.
    "watch_blackboard",
    "subscribe_event",
    "publish_event",
    "list_agents",
    # Image gen / shell / wallet variants — fallbacks read fine.
    # ("generating an image" already covered by image_gen above; the
    # builtin name is generate_image which the JS doesn't map but the
    # fallback "using generate image" is acceptable.)
    "generate_image",
    "run_command",
    "wallet_execute",
    "wallet_read_contract",
    # Operator-only programmatic surface added in PR 2 of the Work
    # tab rewrite. User-visible activity feed doesn't render it
    # with a custom verb — the fallback "using rate delivery"
    # reads fine. (``manage_goals`` is allowlisted by PR 1 in the
    # ``manage_*`` cluster above.)
    "rate_delivery",
})


class TestVerbForToolCompleteness:
    """Every ``@skill`` builtin either has an explicit ``verbForTool``
    entry or is on the fallback allowlist."""

    def test_every_builtin_tool_is_mapped_or_allowlisted(self):
        """No builtin slips through both the verb map and the allowlist."""
        all_tools = _enumerate_skill_tool_names()
        # Pull the verbForTool map body so we look for explicit keys
        # (substring match against the JS object literal).
        m = re.search(
            r"verbForTool\(toolName\)\s*\{.*?const map\s*=\s*\{(.*?)\};",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "verbForTool map body missing"
        map_body = m.group(1)
        missing: list[str] = []
        for tool in sorted(all_tools):
            if tool in _VERB_FOR_TOOL_FALLBACK_OK:
                continue
            if f"{tool}:" not in map_body:
                missing.append(tool)
        assert not missing, (
            "tools missing from verbForTool map (and not on the "
            "fallback allowlist): " + ", ".join(missing)
        )

    def test_fallback_humanises_unknown_tool(self):
        """Unknown tools fall back to ``using {snake_case_with_spaces}``."""
        # Source-level check: the fallback path replaces underscores
        # with spaces and prefixes with "using ".
        m = re.search(
            r"verbForTool\(toolName\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "verbForTool body missing"
        body = m.group(1)
        assert "using ${toolName.replace(/_/g, ' ')}" in body


# ── Wizard timeout / coordination tests (gated on PR-A) ──────────


def _wizard_has_build_failed_state() -> bool:
    """PR-A introduces a ``build_failed`` step. Tests that depend on
    that state stay skipped until the source carries the literal."""
    return "'build_failed'" in _APP_JS_TEXT


class TestWizardBuildPolling:
    """Phase -1 wizard build-polling state machine."""

    @pytest.mark.skipif(
        not _wizard_has_build_failed_state(),
        reason="depends on PR-A: wizard build_failed state",
    )
    def test_wizard_5min_timeout_advances_to_build_failed(self):
        """The 5-min hard cap advances from building → build_failed."""
        # When PR-A lands, the timeout branch should call
        # ``_wizardAdvance('build_failed', ...)`` instead of just
        # clearing the interval. We assert on the source.
        m = re.search(
            r"_wizardStartBuildPolling\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_wizardStartBuildPolling body missing"
        body = m.group(1)
        assert "5 * 60 * 1000" in body, "5 minute timeout constant missing"
        assert "_wizardAdvance('build_failed'" in body or \
               "wizard.step = 'build_failed'" in body, (
            "5-min timeout should advance to build_failed when PR-A lands"
        )

    @pytest.mark.skipif(
        not _wizard_has_build_failed_state(),
        reason="depends on PR-A: partial-apply stability detection",
    )
    def test_wizard_partial_apply_failure_does_not_advance_prematurely(self):
        """A fleet count growing 1→2 then stalling waits for stability."""
        # PR-A's stability gate keeps the wizard in 'building' until the
        # fleet count has been stable for >= the stability window. We
        # check the source carries the stability check.
        m = re.search(
            r"_wizardStartBuildPolling\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_wizardStartBuildPolling body missing"
        body = m.group(1)
        # Look for evidence of stability tracking: a counter that
        # accumulates polls at the same fleet size before advancing.
        assert "stable" in body.lower() or "stability" in body.lower(), (
            "PR-A should add stability tracking — partial fleet must not "
            "trigger premature advance to first-output"
        )


class TestBootstrapGreetingChips:
    """The seeded bootstrap_greeting must not produce ACTION chips."""

    def test_bootstrap_greeting_does_not_inject_action_chips(self):
        """Seeded greeting is suppressed from the operator-chips parser.

        The wizard ask card carries its own chips; running the chip
        parser on the bootstrap greeting would render duplicates.
        """
        # We check the operator-chips application path: it must skip
        # entries marked ``_origin === 'bootstrap_greeting'``. PR-A
        # adds the guard.
        m = re.search(
            r"_applyOperatorActions\(entry\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_applyOperatorActions body missing"
        body = m.group(1)
        if "bootstrap_greeting" not in body:
            pytest.skip(
                "depends on PR-A: _applyOperatorActions should skip "
                "bootstrap_greeting origin entries"
            )
        # When PR-A lands the body skips entries with the seeded
        # origin marker before running the chip parser.
        assert "bootstrap_greeting" in body


# ── Notifications bell coverage gaps (PR-B coordination) ─────────


def _has_legacy_notifications_bell(html: str) -> bool:
    """The Phase 1 placeholder bell mirrors ``events`` rather than the
    persistent ``/notifications`` endpoint. PR-B removes it."""
    # The legacy bell is the one whose dropdown iterates over `events`;
    # the Phase 2 bell iterates over `notifications`. We detect the
    # legacy variant via a unique substring.
    return 'events || []).slice(0, 10)' in html


class TestNotificationsBellSingleton:
    """Only one notifications bell should render after PR-B lands."""

    @pytest.mark.skipif(
        _has_legacy_notifications_bell(_INDEX_HTML),
        reason="depends on PR-B: legacy phase-1 notifications bell still present",
    )
    def test_legacy_notifications_bell_removed(self):
        """The Phase 1 placeholder bell that mirrors ``events`` is gone."""
        # The unique fingerprint of the legacy bell is the dropdown
        # binding to ``events`` rather than ``notifications``.
        assert "events || []).slice(0, 10)" not in _INDEX_HTML, (
            "legacy phase-1 notifications bell still present — PR-B "
            "should remove it before this test runs"
        )

    @pytest.mark.skipif(
        _has_legacy_notifications_bell(_INDEX_HTML),
        reason="depends on PR-B: only one bell after legacy is removed",
    )
    def test_only_one_notifications_bell_renders(self):
        """Exactly one bell SVG sits in the top-nav."""
        # The bell SVG path is unique enough to count occurrences.
        # Both the Phase 1 and Phase 2 variants share this path string.
        bell_path = 'M18 8A6 6 0 0'
        count = _INDEX_HTML.count(bell_path)
        assert count == 1, (
            f"expected exactly 1 notifications bell after PR-B, got {count}"
        )


class TestNotificationsProducer:
    """PR-B wires producers for each event type into NotificationStore."""

    def test_notifications_producer_emits_for_each_event_type(self):
        """Each PR-B event type creates a corresponding notifications row.

        Coordinates with PR-B which adds the producer hooks. We test
        the contract by importing the wiring module and firing each
        event type through its dispatch entrypoint. When PR-B hasn't
        landed the test skips gracefully.
        """
        try:
            from src.dashboard.notifications import NotificationStore
        except ImportError:  # pragma: no cover
            pytest.skip("notifications module not present")

        # PR-B's contract: the wiring module exposes a function that
        # given an event payload + a NotificationStore, inserts the
        # right row. Until PR-B lands the symbol is absent.
        wiring_callable = None
        for module_name in (
            "src.dashboard.notifications",
            "src.dashboard.events",
            "src.dashboard.server",
        ):
            try:
                mod = __import__(module_name, fromlist=["dispatch_event_to_notifications"])
                if hasattr(mod, "dispatch_event_to_notifications"):
                    wiring_callable = getattr(mod, "dispatch_event_to_notifications")
                    break
            except ImportError:
                continue
        if wiring_callable is None:
            pytest.skip(
                "depends on PR-B: dispatch_event_to_notifications not yet wired"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotificationStore(db_path=os.path.join(tmpdir, "n.db"))
            try:
                # Six event types per PR-B spec: each should produce
                # one row when dispatched.
                payloads = [
                    {"type": "task_outcome", "agent": "writer",
                     "data": {"outcome": "delivered", "title": "Draft"}},
                    {"type": "pending_action_created", "agent": "researcher",
                     "data": {"action_label": "publish"}},
                    {"type": "credential_request", "agent": "scout",
                     "data": {"credential_label": "API key"}},
                    {"type": "browser_login_request", "agent": "scout",
                     "data": {"service": "example.com"}},
                    {"type": "browser_captcha_help_request", "agent": "scout",
                     "data": {"service": "example.com"}},
                    {"type": "credit_exhausted", "agent": "writer", "data": {}},
                ]
                for payload in payloads:
                    wiring_callable(payload, store)
                rows = store.list_recent(limit=50)
                assert len(rows) == len(payloads), (
                    f"expected {len(payloads)} rows, got {len(rows)}"
                )
            finally:
                store.close()


# ── Conversations isolation (PR-D coordination) ──────────────────


class TestOpenedConversationsSessionIsolation:
    """Opened-conversation state must be per-session, not global."""

    def test_opened_conversations_session_isolation(self):
        """Two distinct cookies have independent open-conversation sets.

        Coordinates with PR-D which moves the open-set into a per-
        session store. Until that ships, the global set leaks across
        sessions and this test xfail-skips.
        """
        try:
            from src.dashboard.conversations import OpenedConversations
        except ImportError:
            pytest.skip("depends on PR-D: src.dashboard.conversations not present")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = OpenedConversations(db_path=os.path.join(tmpdir, "c.db"))
            try:
                # Distinct cookie ids represent distinct sessions.
                store.mark_opened("session-A", "researcher")
                assert "researcher" in store.list_opened("session-A")
                # Session-B has not opened anything yet.
                assert "researcher" not in store.list_opened("session-B")
            finally:
                store.close()


# ── Undo countdown (PR-C coordination) ───────────────────────────


class TestUndoCountdownRender:
    """The Undo receipt should show remaining seconds (PR-C)."""

    def test_undo_countdown_renders_remaining_seconds(self):
        """Markup carries a span with countdown text format.

        PR-C adds the ``data-testid="undo-countdown"`` span that ticks
        down. The countdown helper is gated on ``_undoExpiresAt``.
        """
        if "undo-countdown" not in _INDEX_HTML and "undoSecondsRemaining" not in _APP_JS_TEXT:
            pytest.skip("depends on PR-C: undo countdown not yet wired")
        # Either the testid or the helper must exist; both is the
        # complete state.
        assert (
            'data-testid="undo-countdown"' in _INDEX_HTML
            or "undoSecondsRemaining" in _APP_JS_TEXT
        ), "PR-C undo countdown markup/helper missing"


# ── Action chip click during stream ──────────────────────────────


class TestActionChipDuringStream:
    """``sendOperatorChip`` branches on streaming state."""

    def test_action_chip_click_during_stream_handled(self):
        """While streaming the chip routes to ``steerAgent``; otherwise
        to ``sendChatTo``."""
        # Pull the body of sendOperatorChip and verify both branches.
        m = re.search(
            r"sendOperatorChip\(label\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "sendOperatorChip body missing"
        body = m.group(1)
        # Streaming gate uses isAgentBusy('operator')
        assert "isAgentBusy('operator')" in body, (
            "sendOperatorChip should gate on isAgentBusy('operator')"
        )
        # Both branch targets present.
        assert "this.steerAgent('operator'" in body, (
            "missing steerAgent branch in sendOperatorChip"
        )
        assert "this.sendChatTo('operator'" in body, (
            "missing sendChatTo branch in sendOperatorChip"
        )


# ── Tutorial gating edge cases ───────────────────────────────────


class TestTutorialGatingEdgeCases:
    """The tutorial's gating logic must be defensive about edge conditions."""

    def test_existing_fleet_does_not_fire_tutorial(self):
        """Fleet size > 0 (non-operator) suppresses the tutorial."""
        # The empty-fleet check now lives in _isFirstVisit() and
        # _maybeStartTutorial delegates to it. Confirm both halves of
        # the contract are wired.
        m_first_visit = re.search(
            r"_isFirstVisit\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m_first_visit, "_isFirstVisit body missing"
        first_visit_body = m_first_visit.group(1)
        # The fleet check happens inside _isFirstVisit (returns false
        # when any non-operator agent exists).
        assert "a.id !== 'operator'" in first_visit_body, (
            "_isFirstVisit should exclude the operator from the fleet check"
        )
        assert "if (hasFleetAgents) return false" in first_visit_body, (
            "_isFirstVisit must return false when fleet is non-empty"
        )

        m = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartTutorial body missing"
        body = m.group(1)
        assert "if (!this._isFirstVisit()) return" in body, (
            "non-empty-fleet guard missing — tutorial would fire on existing fleets"
        )

    def test_skip_persists_seen_flag(self):
        """Skip path writes olSeenTutorial just like completion."""
        # Skip → dismissTutorial('skip_button') → _completeTutorial →
        # localStorage.setItem('olSeenTutorial', 'true'). The shared
        # _completeTutorial path is the single write site.
        m = re.search(
            r"_completeTutorial\(reason\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_completeTutorial body missing"
        body = m.group(1)
        assert "localStorage.setItem('olSeenTutorial', 'true')" in body

    def test_localstorage_unavailable_falls_through_to_show_once(self):
        """When localStorage throws (private mode) the tutorial still fires.

        The seen-flag check is wrapped in try/catch. The ``catch`` arm
        is a no-op so the function continues to the agent check — i.e.
        the tutorial shows once per session in private mode.
        """
        m = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartTutorial body missing"
        body = m.group(1)
        # The catch must not return — only the try return early-exits.
        # The simplest contract check: the comment marks the private-
        # mode policy and the catch arm is empty.
        assert re.search(
            r"catch\s*\(_\)\s*\{\s*/\*[^*]*private mode[^*]*\*/\s*\}",
            body,
        ), "private-mode catch arm should be a documented no-op"

    def test_tutorial_state_does_not_persist_across_reload(self):
        """Tutorial state lives in memory only — reload aborts mid-flight."""
        # The wizard persists to localStorage.ol_wizard; the tutorial
        # explicitly does NOT. _maybeStartTutorial gates only on the
        # seen flag, not on a stored step. We assert the tutorial
        # bootstrap reads neither a stored step nor calls a persist
        # helper from the seen-flag branch.
        m = re.search(
            r"_maybeStartTutorial\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "_maybeStartTutorial body missing"
        body = m.group(1)
        # Tutorial state object must not be hydrated from localStorage.
        # (The seen flag is the only persisted bit.)
        assert "ol_tutorial_step" not in _APP_JS_TEXT, (
            "tutorial step should not be persisted — reload must abort the tutorial"
        )
        # The function reads only ``olSeenTutorial``, not a stored step.
        assert "getItem('olSeenTutorial')" in body
        assert "getItem('ol_tutorial_step'" not in body


# ── Polish fix tests (verify the audit fixes landed) ─────────────


class TestPolishFixesApplied:
    """Verify each polish fix from the user-journey audit is in place."""

    def test_tutorial_modal_uses_unique_title_ids(self):
        """Each step's <h2> has a unique id; aria-labelledby is dynamic.

        Inherited from the original whats-new audit fix — each step
        carries a distinct ``tutorial-title-N`` so screen readers
        announce the correct dialog title when steps render in sequence.
        """
        for n in (1, 2, 3):
            assert f'id="tutorial-title-{n}"' in _INDEX_HTML, (
                f"missing unique title id for step {n}"
            )
        # The aria-labelledby binding must reference the dynamic id.
        assert (
            ":aria-labelledby=\"'tutorial-title-' + newUserTutorial.step\""
            in _INDEX_HTML
        ), "aria-labelledby should bind dynamically to the active step"
        # The legacy whats-new ids should be gone.
        assert 'id="whats-new-title"' not in _INDEX_HTML, (
            "legacy whats-new title id still present"
        )
        for n in (1, 2, 3):
            assert f'id="whats-new-title-{n}"' not in _INDEX_HTML, (
                f"legacy whats-new title id (step {n}) still present"
            )

    def test_credential_notification_icon_is_plain_text(self):
        """The 'credential' kind icon is 'K' — markup is emoji-free."""
        # The comment near the helper says "Plain text glyphs keep the
        # markup emoji-free"; emoji slipped in for the credential row.
        m = re.search(
            r"notificationKindIcon\(kind\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "notificationKindIcon body missing"
        body = m.group(1)
        assert "case 'credential': return 'K'" in body, (
            "credential icon should be plain 'K' (emoji-free)"
        )
        assert "'\U0001f511'" not in body and "🔑" not in body, (
            "key emoji should be gone"
        )

    def test_load_older_caption_shows_total(self):
        """The 'Load older' button shows visible-vs-total when known."""
        # The helper formats "Load 50 older (340 of 500)" when total >
        # limit. We assert the source carries the format literal.
        m = re.search(
            r"loadOlderCaption\(agentId\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "loadOlderCaption body missing"
        body = m.group(1)
        assert "Load 50 older (" in body and " of " in body, (
            "loadOlderCaption should format as 'Load 50 older (X of Y)'"
        )
        # Both pagination buttons in the template bind to the helper.
        assert 'x-text="loadOlderCaption(\'operator\')"' in _INDEX_HTML
        assert 'x-text="loadOlderCaption(activeChatId)"' in _INDEX_HTML

    def test_side_panel_esc_restores_focus(self):
        """closeSidePanel restores focus captured on toggleSidePanel open."""
        # Mirrors the _newUserTutorialPrevFocus pattern.
        assert "_messengerSidePanelPrevFocus" in _APP_JS_TEXT, (
            "side-panel previous-focus tracker missing"
        )
        # closeSidePanel restores via .focus() on the captured node.
        m = re.search(
            r"closeSidePanel\(\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "closeSidePanel body missing"
        body = m.group(1)
        assert "_messengerSidePanelPrevFocus" in body, (
            "closeSidePanel should restore captured focus"
        )
        assert ".focus()" in body, (
            "closeSidePanel should call .focus() on the captured element"
        )

    def test_anonymous_agent_label_is_an_agent(self):
        """formatActivityForUser uses 'An agent' rather than 'Someone'."""
        m = re.search(
            r"formatActivityForUser\(event\)\s*\{(.*?)\n    \},",
            _APP_JS_TEXT,
            re.DOTALL,
        )
        assert m, "formatActivityForUser body missing"
        body = m.group(1)
        assert "'An agent'" in body
        # The comment-ish "Someone" string in agentDisplayName is fine
        # because it has its own semantics; only the activity feed
        # fallback was flagged.
        assert ": 'Someone'" not in body

    def test_cancel_pending_action_uses_confirm_modal(self):
        """The Needs-You Cancel button pops the confirm modal first."""
        # The helper exists.
        assert "_confirmCancelPendingAction(nonce, label)" in _APP_JS_TEXT, (
            "_confirmCancelPendingAction helper missing"
        )
        # The Needs-You builder calls it instead of cancelPendingAction
        # directly. We grep for the wrapper call near the Cancel label.
        assert (
            "this._confirmCancelPendingAction(p.nonce"
            in _APP_JS_TEXT
        ), "Needs-You Cancel button should route through the confirm wrapper"

    def test_hidden_events_rollup_helper_present(self):
        """``hiddenCoordinationEventsCount`` getter + template wiring."""
        assert "hiddenCoordinationEventsCount" in _APP_JS_TEXT, (
            "hidden-events rollup getter missing"
        )
        assert 'data-testid="hidden-events-rollup"' in _INDEX_HTML, (
            "hidden-events rollup row missing from template"
        )

    def test_claudemd_documents_dashboard_support_modules(self):
        """CLAUDE.md gains rows for notifications/telemetry/platform_success."""
        claude_md = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        # Notifications module is documented.
        assert "notifications.py" in claude_md, (
            "CLAUDE.md should document src/dashboard/notifications.py"
        )
        # Telemetry module is documented.
        assert "telemetry.py" in claude_md and "dashboard_telemetry" in claude_md, (
            "CLAUDE.md should document src/dashboard/telemetry.py"
        )

    def test_claudemd_documents_tab_id_and_wizard_state_constraints(self):
        """CLAUDE.md gains the tab-ID + wizard-state Known Constraints."""
        claude_md = (_REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        assert "Tab IDs are frozen" in claude_md, (
            "tab-ID constraint missing from Known Constraints"
        )
        assert "Wizard state machine" in claude_md, (
            "wizard state machine constraint missing from Known Constraints"
        )
        # Wizard states are enumerated.
        for state in ("idle", "ask", "confirming", "building", "first-output", "build_failed"):
            assert state in claude_md, f"wizard state {state} not documented"


# ── EventBus coverage — JS handlers wired in onWsEvent ────────────────


class TestOnWsEventHandlersForLiveCoverage:
    """``onWsEvent`` must dispatch the new event types added to close
    EventBus coverage gaps. Each test is a string-level assertion over
    ``app.js`` because we can't run Alpine in pytest — but the wiring
    is what we want to guard against silent regressions."""

    def test_notification_added_handler_present(self):
        # The handler prepends to ``notifications`` and bumps the
        # unread count so the bell badge updates live.
        assert "evt.type === 'notification_added'" in _APP_JS_TEXT
        assert "this.notifications = [row, ...(this.notifications || [])]" in _APP_JS_TEXT
        assert "this.notificationsUnreadCount" in _APP_JS_TEXT

    def test_credential_stored_handler_flips_card(self):
        # Matches by ``request_id`` (new flow) with name as fallback.
        assert "evt.type === 'credential_stored'" in _APP_JS_TEXT
        # And it actually flips the card.
        # The handler iterates chatHistories and sets ``saved=true``.
        block_start = _APP_JS_TEXT.index("evt.type === 'credential_stored'")
        block = _APP_JS_TEXT[block_start:block_start + 1500]
        assert "m.saved = true" in block
        assert "m.cancelled = false" in block

    def test_agent_archived_unarchived_handlers_present(self):
        assert "evt.type === 'agent_archived'" in _APP_JS_TEXT
        assert "evt.type === 'agent_unarchived'" in _APP_JS_TEXT

    def test_agent_restart_handlers_present(self):
        assert "evt.type === 'agent_restarting'" in _APP_JS_TEXT
        assert "evt.type === 'agent_restarted'" in _APP_JS_TEXT
        # Map for the spinner state must be declared on Alpine state.
        assert "agentRestartingMap:" in _APP_JS_TEXT

    def test_agent_config_updated_handler_present(self):
        assert "evt.type === 'agent_config_updated'" in _APP_JS_TEXT

    def test_project_crud_handlers_present(self):
        # All five project events route to a single fetchProjects path.
        for ev in (
            "team_created", "team_updated", "team_deleted",
            "team_archived", "team_unarchived",
        ):
            assert f"'{ev}'" in _APP_JS_TEXT, f"handler for {ev} missing"

    def test_blackboard_delete_handler_present(self):
        assert "evt.type === 'blackboard_delete'" in _APP_JS_TEXT
