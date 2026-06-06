"""Tests for operator agent auto-creation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml


class _TempConfigMixin:
    """Mixin that redirects config files to a temp directory."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig_agents = None
        self._orig_perms = None
        self._orig_root = None
        import src.cli.config as cfg_mod

        self._orig_agents = cfg_mod.AGENTS_FILE
        self._orig_perms = cfg_mod.PERMISSIONS_FILE
        self._orig_root = cfg_mod.PROJECT_ROOT

        self._agents_path = Path(self._tmpdir) / "config" / "agents.yaml"
        self._perms_path = Path(self._tmpdir) / "config" / "permissions.json"
        self._agents_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_mod.AGENTS_FILE = self._agents_path
        cfg_mod.PERMISSIONS_FILE = self._perms_path
        cfg_mod.PROJECT_ROOT = Path(self._tmpdir)
        # Initialize empty permissions
        self._perms_path.write_text(json.dumps({"permissions": {}}, indent=2))

    def teardown_method(self):
        import src.cli.config as cfg_mod

        cfg_mod.AGENTS_FILE = self._orig_agents
        cfg_mod.PERMISSIONS_FILE = self._orig_perms
        cfg_mod.PROJECT_ROOT = self._orig_root
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _mock_config(self, *, collab=True):
        return patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
            "collaboration": collab,
        })


class TestEnsureOperatorCreates(_TempConfigMixin):
    """Creates operator in agents.yaml when it doesn't exist."""

    def test_creates_when_missing(self):
        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        # Check agents.yaml has operator
        with open(self._agents_path) as f:
            agents_cfg = yaml.safe_load(f)
        assert "operator" in agents_cfg["agents"]
        agent = agents_cfg["agents"]["operator"]
        assert "Operator" in agent["role"]
        assert agent["model"] == "openai/gpt-4o-mini"
        assert agent["initial_instructions"]
        assert agent["initial_soul"]
        assert agent["initial_heartbeat"]

        # Check permissions
        with open(self._perms_path) as f:
            perms = json.load(f)
        assert "operator" in perms["permissions"]
        assert perms["permissions"]["operator"]["can_spawn"] is True
        # Operator gets browser + internet on by default so it can navigate
        # the web directly to help users; both are togglable in Operator
        # Settings. See _ensure_operator_agent in src/cli/config.py.
        assert perms["permissions"]["operator"]["can_use_browser"] is True
        assert perms["permissions"]["operator"]["can_use_internet"] is True

        # Check skills dir created
        skills_dir = Path(self._tmpdir) / "skills" / "operator"
        assert skills_dir.is_dir()


class TestEnsureOperatorNoop(_TempConfigMixin):
    """Preserves non-model fields when operator already exists."""

    def test_preserves_role_when_exists(self):
        # Pre-create operator with same model as default
        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "openai/gpt-4o-mini",
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        # Should not overwrite role
        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        assert result["agents"]["operator"]["role"] == "Existing operator"
        assert result["agents"]["operator"]["model"] == "openai/gpt-4o-mini"


class TestEnsureOperatorModelSync(_TempConfigMixin):
    """Preserves existing operator model — does not overwrite on restart."""

    def test_preserves_existing_model(self):
        # Pre-create operator with old model
        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "anthropic/claude-3-haiku",
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="google/gemini-2.0-flash")

        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        # Model should be preserved (not overwritten by default_model)
        assert result["agents"]["operator"]["model"] == "anthropic/claude-3-haiku"
        # Role should be preserved
        assert result["agents"]["operator"]["role"] == "Existing operator"

    def test_preserves_model_when_default_not_passed(self):
        # Pre-create operator with old model
        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "anthropic/claude-3-haiku",
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "google/gemini-2.0-flash"},
            "agents": {},
            "collaboration": True,
        }):
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent()

        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        # Model should remain unchanged even when default differs
        assert result["agents"]["operator"]["model"] == "anthropic/claude-3-haiku"

    def test_no_write_when_operator_exists(self):
        """Does not rewrite agents.yaml when operator already exists.

        The post-migration ``_ensure_operator_agent`` will refresh the
        heartbeat when its sentinel is missing — to pin the pure no-write
        path we seed the heartbeat with the canonical template (which
        carries the sentinel) so the refresh branch is a no-op.
        """
        from src.cli.config import _OPERATOR_HEARTBEAT

        # Pre-create operator with matching model AND the canonical
        # heartbeat template (sentinel present → no refresh).
        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "openai/gpt-4o-mini",
            "heartbeat": _OPERATOR_HEARTBEAT,
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        mtime_before = self._agents_path.stat().st_mtime

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        # File should not have been rewritten (mtime unchanged)
        mtime_after = self._agents_path.stat().st_mtime
        assert mtime_before == mtime_after

    def test_empty_heartbeat_is_bootstrapped_to_canonical_template(self):
        """Codex r3 (PR 972) catch — the WARN for no-sentinel files
        instructs operators to clear ``heartbeat:`` in agents.yaml to
        re-bootstrap. That recovery path must actually work: an
        existing operator with an empty heartbeat field must get
        rewritten from the canonical template on next ``_ensure``
        rather than left empty forever."""
        from src.cli.config import _OPERATOR_HEARTBEAT

        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "openai/gpt-4o-mini",
            "heartbeat": "",   # operator followed the WARN advice
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        # The full canonical template — not an empty string — is now
        # in place. The first line check confirms it's the real thing.
        assert result["agents"]["operator"]["heartbeat"] == _OPERATOR_HEARTBEAT


class TestEnsureOperatorMigratesConcierge(_TempConfigMixin):
    """Renames concierge to operator in both agents.yaml and permissions."""

    def test_migrates_concierge(self):
        # Set up concierge agent
        agents_cfg = {"agents": {"concierge": {
            "role": "Fleet concierge",
            "model": "openai/gpt-4o",
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        # Set up concierge permissions
        perms = {"permissions": {"concierge": {
            "can_message": ["*"],
            "can_spawn": True,
        }}}
        with open(self._perms_path, "w") as f:
            json.dump(perms, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        # agents.yaml: concierge renamed to operator
        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        assert "operator" in result["agents"]
        assert "concierge" not in result["agents"]
        assert result["agents"]["operator"]["role"] == "Fleet concierge"

        # permissions: concierge renamed to operator
        with open(self._perms_path) as f:
            perms_result = json.load(f)
        assert "operator" in perms_result["permissions"]
        assert "concierge" not in perms_result["permissions"]
        assert perms_result["permissions"]["operator"]["can_spawn"] is True


class TestEnsureOperatorBothExist(_TempConfigMixin):
    """When both concierge and operator exist, keeps operator, removes concierge."""

    def test_handles_both_exist(self):
        agents_cfg = {"agents": {
            "concierge": {"role": "Old concierge", "model": "openai/gpt-4o"},
            "operator": {"role": "Operator agent", "model": "openai/gpt-4o-mini"},
        }}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        assert "operator" in result["agents"]
        assert "concierge" not in result["agents"]
        assert result["agents"]["operator"]["role"] == "Operator agent"


class TestOperatorConstants:
    """Verify operator constants are populated, not placeholder."""

    def test_instructions_not_empty(self):
        from src.cli.config import _OPERATOR_HEARTBEAT, _OPERATOR_SOUL
        from src.shared.operator_playbooks import _OPERATOR_CORE
        assert len(_OPERATOR_CORE) > 100
        assert len(_OPERATOR_SOUL) > 50
        assert len(_OPERATOR_HEARTBEAT) > 100
        assert "Routing Work" in _OPERATOR_CORE

    def test_allowed_tools_populated(self):
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS, _OPERATOR_HEARTBEAT_TOOLS
        # Phase 1 of the back-compat cleanup retired the propose/confirm
        # config flow and the legacy ``*_project`` operator tools. The
        # post-cleanup allowlist is checked by membership rather than
        # a magic number so additions like ``compose_work_summary`` from
        # the work-summaries backend don't require recounting.
        assert _OPERATOR_ALLOWED_TOOLS  # non-empty
        # Heartbeat tool count history (bump this assertion + the
        # ``_HEARTBEAT_TOOLS`` frozenset in ``src/agent/loop.py`` + the
        # ``_OPERATOR_HEARTBEAT_TOOLS`` doc list in ``src/cli/config.py``
        # together when adding a new heartbeat-reachable tool):
        #  * v1 (initial): 4 read-only tools (list_agents, get_agent_profile,
        #    get_system_status, notify_user).
        #  * v2 (workflow awareness): +check_inbox, workflow_snapshot,
        #    await_task_event for multi-stage chain driving.
        #  * v3 (Work-tab rewrite PR 2/3): +rate_delivery, manage_goals so the
        #    heartbeat instructions that grade up to 10 oldest unrated done
        #    tasks per cycle and steward goal staleness are reachable.
        #  * v4 (PR 972 Codex follow-up): +inspect_agents — step 5 of
        #    the heartbeat procedure already called it but the allowlist
        #    denied the call. (Note: main had previously dropped
        #    save_observations, so v3 baseline was 9 not 10.)
        assert len(_OPERATOR_HEARTBEAT_TOOLS) == 10
        # The operator-tier heartbeat tools must also be on the main
        # operator allowlist — they're the tools operator can use from
        # both /chat and heartbeat. Two heartbeat entries are
        # deliberately mesh-internal primitives that operator does NOT
        # expose in /chat (``inspect_agents`` is the operator-flavored
        # cousin used in /chat instead): ``list_agents`` and
        # ``get_agent_profile``. The rest must overlap.
        _HEARTBEAT_ONLY = {"list_agents", "get_agent_profile"}
        operator_visible = set(_OPERATOR_HEARTBEAT_TOOLS) - _HEARTBEAT_ONLY
        assert operator_visible.issubset(set(_OPERATOR_ALLOWED_TOOLS)), (
            f"heartbeat tools missing from operator allowlist: "
            f"{operator_visible - set(_OPERATOR_ALLOWED_TOOLS)}"
        )
        # Consolidated product tools (read + lifecycle) must be present.
        for tool in (
            "inspect_teams", "inspect_agents",
            "list_agent_queue", "get_team_outputs",
            "summarize_team_progress",
            "manage_team", "manage_agent", "manage_task",
            # PR 5 — north-star setter is no-confirmation meta-config.
            "set_team_goal",
            # Self-cleanup — operator can clear stale pending actions
            # and prune the audit log itself.
            "list_pending", "cancel_pending_action", "archive_audit_before",
            # PR F — vault visibility (names-only, no secret leak) so
            # the operator can check before calling request_credential.
            "vault_list",
        ):
            assert tool in _OPERATOR_ALLOWED_TOOLS
        # Dropped dead-weight + replaced tools must be gone.
        for tool in (
            "update_status",
            "list_projects", "get_project", "list_project_status",
            "list_agents", "get_agent_profile", "read_agent_history",
            "archive_project", "delete_project",
            "archive_agent", "delete_agent",
            "reroute_task", "cancel_task", "retry_failed_task",
            # Phase 1 back-compat deletions — propose/confirm flow and
            # the renamed-stub ``*_project`` tools are gone.
            "propose_edit", "confirm_edit",
            "create_project", "add_agents_to_project",
            "remove_agents_from_project", "update_project_context",
            "set_project_goal", "manage_project",
        ):
            assert tool not in _OPERATOR_ALLOWED_TOOLS

    def test_pr1_edit_tools_present(self):
        """PR 1 — edit_agent + undo_change in allowlist.

        propose_edit + confirm_edit were retired by Phase 1 of the
        back-compat cleanup (config edits now apply immediately via
        edit_agent with a built-in undo receipt, so there is nothing
        to propose or confirm). Neither name is registered as a
        @skill anymore.
        """
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS
        assert "edit_agent" in _OPERATOR_ALLOWED_TOOLS
        assert "undo_change" in _OPERATOR_ALLOWED_TOOLS
        # propose/confirm retired — neither must appear in the allowlist.
        assert "propose_edit" not in _OPERATOR_ALLOWED_TOOLS
        assert "confirm_edit" not in _OPERATOR_ALLOWED_TOOLS

    def test_seam_followup_fix1_missing_tools_added(self):
        """Seam follow-up Fix 1: 9 operator-gated tools that were missing
        from the allowlist must now be present, and write_file must
        intentionally remain out (operator orchestrates, doesn't author
        arbitrary files)."""
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS
        # Canonical inverse of edit_agent + peer-artifact reads.
        assert "read_agent_config" in _OPERATOR_ALLOWED_TOOLS
        assert "list_peer_artifacts" in _OPERATOR_ALLOWED_TOOLS
        assert "read_peer_artifact" in _OPERATOR_ALLOWED_TOOLS
        # Peer FILE reads — full /data volume, not just artifacts/. Lets the
        # operator locate + relay a worker's deliverable (CSV, data.md).
        assert "list_peer_files" in _OPERATOR_ALLOWED_TOOLS
        assert "read_peer_file" in _OPERATOR_ALLOWED_TOOLS
        # Credential-aware model discovery (new tool added by Fix 2).
        assert "list_available_models" in _OPERATOR_ALLOWED_TOOLS
        # Operator self-notes — was bouncing through hand_off-to-self.
        assert "memory_save" in _OPERATOR_ALLOWED_TOOLS
        assert "memory_search" in _OPERATOR_ALLOWED_TOOLS
        # Workspace management — caps already enforce safety.
        assert "update_workspace" in _OPERATOR_ALLOWED_TOOLS
        assert "read_file" in _OPERATOR_ALLOWED_TOOLS
        # Intentionally NOT granted — encourages anti-patterns.
        assert "write_file" not in _OPERATOR_ALLOWED_TOOLS

    def test_request_browser_login_in_allowlist(self):
        """Operator must be able to delegate browser login requests to workers.

        Even though the operator now has its own browser access, the
        delegation path (``request_browser_login(agent_id=worker)``)
        remains the way to land session cookies in a *worker's* browser
        profile.
        """
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS
        assert "request_browser_login" in _OPERATOR_ALLOWED_TOOLS

    def test_browser_tools_in_allowlist(self):
        """Operator's curated allowlist exposes the browser_* surface so the
        Browser-access toggle (can_use_browser) has tools to gate."""
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS
        assert "browser_navigate" in _OPERATOR_ALLOWED_TOOLS
        assert "browser_click" in _OPERATOR_ALLOWED_TOOLS

    def test_operator_agent_id(self):
        from src.cli.config import _OPERATOR_AGENT_ID
        assert _OPERATOR_AGENT_ID == "operator"

    def test_soul_has_key_traits(self):
        from src.cli.config import _OPERATOR_SOUL
        assert "proactive" in _OPERATOR_SOUL
        assert "action" in _OPERATOR_SOUL
        assert "confirmation" in _OPERATOR_SOUL

    def test_heartbeat_has_steps(self):
        from src.cli.config import _OPERATOR_HEARTBEAT
        assert "get_system_status" in _OPERATOR_HEARTBEAT
        assert "notify_user" in _OPERATOR_HEARTBEAT
        assert "check_inbox" in _OPERATOR_HEARTBEAT

    def test_heartbeat_references_prj_metric_keys(self):
        """PR-J' — heartbeat keys on the new metric fields, not failure_rate."""
        from src.cli.config import _OPERATOR_HEARTBEAT
        # New keys must appear so the LLM knows to look at them.
        assert "outcome_rejected_24h_count" in _OPERATOR_HEARTBEAT
        assert "stale_tasks_24h_count" in _OPERATOR_HEARTBEAT
        assert "execution_failures_24h_count" in _OPERATOR_HEARTBEAT
        assert "chain_breaks_24h_count" in _OPERATOR_HEARTBEAT
        # Bug 6 — operator's own untriaged-inbox depth surfaces here.
        assert "inbox_stale_count" in _OPERATOR_HEARTBEAT
        assert "per_agent_cost_vs_yesterday_ratio" in _OPERATOR_HEARTBEAT
        # The dead failure_rate threshold rule must NOT be back.
        assert "failure_rate > 0.30" not in _OPERATOR_HEARTBEAT
        # Stale-task drill-in references the new inspect_agents parameter.
        assert "stale_threshold_hours=24" in _OPERATOR_HEARTBEAT

    def test_heartbeat_step_count_within_budget(self):
        """Total numbered steps stay within the HEARTBEAT_MAX_ITERATIONS=12 budget.

        Counting numbered top-level steps (``1.`` through ``N.``) — each
        step that calls a tool burns one iteration. PR-V capped per-cycle
        drill-ins at 3, and bumped the loop ceiling from 10 to 12 to give
        headroom. The worst-case fan-out is now:
        1 status + 1 roster + 3 drill-ins + 1 stale fanout + 1 notify_user
        = 7 tool calls, plus the final assistant turn that emits the
        heartbeat summary.
        """
        import re

        from src.cli.config import _OPERATOR_HEARTBEAT
        steps = re.findall(r"^\s*(\d+)\.\s", _OPERATOR_HEARTBEAT, re.MULTILINE)
        max_step = max(int(s) for s in steps)
        assert max_step <= 11, (
            f"heartbeat has {max_step} numbered steps; budget is 11 "
            f"(HEARTBEAT_MAX_ITERATIONS=12 minus the final assistant turn)"
        )

    def test_heartbeat_caps_drill_ins_at_three(self):
        """PR-V — drill-ins capped at 3 per cycle to stay under the budget.

        Without a cap, N concerning agents → N drill calls + 4 fixed steps,
        which blows past HEARTBEAT_MAX_ITERATIONS at N>=6. The playbook now
        instructs the operator to focus on the top-3 worst offenders and
        defer the rest to the next cycle.
        """
        from src.cli.config import _OPERATOR_HEARTBEAT
        # Cap-3 wording must be unambiguous.
        assert "at most THREE" in _OPERATOR_HEARTBEAT or "at most 3" in _OPERATOR_HEARTBEAT
        # Overflow guidance must reference AGENTS, not THRESHOLDS, so the LLM
        # cannot misread the subject of the count (audit follow-up).
        assert "more than 3 agents trigger" in _OPERATOR_HEARTBEAT
        assert "top-3 worst" in _OPERATOR_HEARTBEAT
        # Overflow agents must be deferred to the next cycle.
        assert "next cycle" in _OPERATOR_HEARTBEAT

    def test_heartbeat_budget_header_matches_loop_constant(self):
        """PR-V — the playbook's stated budget tracks HEARTBEAT_MAX_ITERATIONS."""
        from src.agent.loop import HEARTBEAT_MAX_ITERATIONS
        from src.cli.config import _OPERATOR_HEARTBEAT
        assert f"HEARTBEAT_MAX_ITERATIONS={HEARTBEAT_MAX_ITERATIONS}" in _OPERATOR_HEARTBEAT

    def test_heartbeat_imports_sentinel_from_shared_types(self):
        """HEARTBEAT_SENTINELS lives in ``src.shared.types`` and the
        operator-config refresh path imports it from there. Verifies
        the central constant resolves and contains the canonical
        marker actually embedded in ``_OPERATOR_HEARTBEAT``."""
        from src.cli.config import _OPERATOR_HEARTBEAT
        from src.shared.types import HEARTBEAT_SENTINELS
        assert isinstance(HEARTBEAT_SENTINELS, tuple)
        assert "heartbeat_v2_workflow_aware" in HEARTBEAT_SENTINELS
        # Every sentinel in the tuple must appear as an HTML comment
        # somewhere in the canonical template — otherwise the
        # ``new_has_sentinel`` check in ``_ensure_operator_agent``
        # would silently fail to roll the heartbeat forward.
        present = [
            f"<!-- {m} -->" in _OPERATOR_HEARTBEAT
            for m in HEARTBEAT_SENTINELS
        ]
        assert any(present), (
            "no HEARTBEAT_SENTINELS marker present in _OPERATOR_HEARTBEAT — "
            "operator heartbeat refresh would never fire"
        )

    def test_heartbeat_step4_mentions_inline_blocker_note(self):
        """Fix 4 — step 4 must surface the new inline ``blocker_note``
        contract and the 3-call cap so the LLM doesn't loop snapshots."""
        from src.cli.config import _OPERATOR_HEARTBEAT
        # Snapshot now carries blocker_note inline — no follow-up
        # get_task call needed.
        assert "blocker_note" in _OPERATOR_HEARTBEAT
        assert "inline" in _OPERATOR_HEARTBEAT
        # Cap of 3 snapshot calls per heartbeat.
        assert (
            "Cap at 3 snapshot calls" in _OPERATOR_HEARTBEAT
            or "cap at 3 snapshot calls" in _OPERATOR_HEARTBEAT
        )

    def test_core_has_key_sections(self):
        from src.shared.operator_playbooks import _OPERATOR_CORE
        assert "Routing Work" in _OPERATOR_CORE
        assert "Plan Limits" in _OPERATOR_CORE
        assert "Assessment" in _OPERATOR_CORE
        assert "Workflow Overview" in _OPERATOR_CORE
        assert "playbook_v2" in _OPERATOR_CORE

    def test_playbooks_have_key_tools(self):
        from src.shared.operator_playbooks import (
            _PLAYBOOK_CREDENTIALS,
            _PLAYBOOK_EDIT,
            _PLAYBOOK_MONITOR,
            _PLAYBOOK_TEAM_BUILD,
        )

        # PR 1 — playbook references edit_agent for all config edits
        # (the propose/confirm flow was retired in Phase 1 of the
        # back-compat cleanup; edits apply immediately with an undo
        # receipt, so confirm_edit no longer appears in any playbook).
        assert "edit_agent" in _PLAYBOOK_TEAM_BUILD
        assert "edit_agent" in _PLAYBOOK_EDIT
        assert "inspect_agents" in _PLAYBOOK_MONITOR
        assert "request_credential" in _PLAYBOOK_CREDENTIALS

    def test_core_has_plan_tiers(self):
        from src.shared.operator_playbooks import _OPERATOR_CORE
        assert "Basic" in _OPERATOR_CORE
        assert "Growth" in _OPERATOR_CORE
        assert "Pro" in _OPERATOR_CORE
        assert "Self-hosted" in _OPERATOR_CORE


class TestEnsureOperatorUsesConfigModel(_TempConfigMixin):
    """When no default_model is passed, reads from config."""

    def test_reads_model_from_config(self):
        with patch("src.cli.config._load_config", return_value={
            "llm": {"default_model": "anthropic/claude-3-haiku"},
            "agents": {},
            "collaboration": True,
        }):
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent()

        with open(self._agents_path) as f:
            agents_cfg = yaml.safe_load(f)
        assert agents_cfg["agents"]["operator"]["model"] == "anthropic/claude-3-haiku"
