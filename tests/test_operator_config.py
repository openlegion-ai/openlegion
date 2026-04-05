"""Tests for operator agent auto-creation."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
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
        assert perms["permissions"]["operator"]["can_use_browser"] is False

        # Check skills dir created
        skills_dir = Path(self._tmpdir) / "skills" / "operator"
        assert skills_dir.is_dir()


class TestEnsureOperatorNoop(_TempConfigMixin):
    """Does nothing when operator already exists."""

    def test_noop_when_exists(self):
        # Pre-create operator
        agents_cfg = {"agents": {"operator": {
            "role": "Existing operator",
            "model": "anthropic/claude-3-haiku",
        }}}
        with open(self._agents_path, "w") as f:
            yaml.dump(agents_cfg, f)

        with self._mock_config():
            from src.cli.config import _ensure_operator_agent
            _ensure_operator_agent(default_model="openai/gpt-4o-mini")

        # Should not overwrite
        with open(self._agents_path) as f:
            result = yaml.safe_load(f)
        assert result["agents"]["operator"]["role"] == "Existing operator"
        assert result["agents"]["operator"]["model"] == "anthropic/claude-3-haiku"


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
        from src.cli.config import _OPERATOR_INSTRUCTIONS, _OPERATOR_SOUL, _OPERATOR_HEARTBEAT
        assert len(_OPERATOR_INSTRUCTIONS) > 100
        assert len(_OPERATOR_SOUL) > 50
        assert len(_OPERATOR_HEARTBEAT) > 100
        assert "FIRST RUN" in _OPERATOR_INSTRUCTIONS

    def test_allowed_tools_populated(self):
        from src.cli.config import _OPERATOR_ALLOWED_TOOLS, _OPERATOR_HEARTBEAT_TOOLS
        assert len(_OPERATOR_ALLOWED_TOOLS) == 20
        assert len(_OPERATOR_HEARTBEAT_TOOLS) == 5
        # Heartbeat tools should be a subset of allowed tools
        assert set(_OPERATOR_HEARTBEAT_TOOLS).issubset(set(_OPERATOR_ALLOWED_TOOLS))

    def test_operator_agent_id(self):
        from src.cli.config import _OPERATOR_AGENT_ID
        assert _OPERATOR_AGENT_ID == "operator"

    def test_soul_has_key_traits(self):
        from src.cli.config import _OPERATOR_SOUL
        assert "fleet architect" in _OPERATOR_SOUL
        assert "security-conscious" in _OPERATOR_SOUL
        assert "confirmation" in _OPERATOR_SOUL

    def test_heartbeat_has_steps(self):
        from src.cli.config import _OPERATOR_HEARTBEAT
        assert "get_system_status" in _OPERATOR_HEARTBEAT
        assert "save_observations" in _OPERATOR_HEARTBEAT
        assert "notify_user" in _OPERATOR_HEARTBEAT

    def test_instructions_has_decision_tree(self):
        from src.cli.config import _OPERATOR_INSTRUCTIONS
        assert "BUILD REQUEST" in _OPERATOR_INSTRUCTIONS
        assert "EDIT REQUEST" in _OPERATOR_INSTRUCTIONS
        assert "WORK REQUEST" in _OPERATOR_INSTRUCTIONS
        assert "STATUS REQUEST" in _OPERATOR_INSTRUCTIONS
        assert "PROJECT REQUEST" in _OPERATOR_INSTRUCTIONS

    def test_instructions_has_propose_edit(self):
        from src.cli.config import _OPERATOR_INSTRUCTIONS
        assert "propose_edit" in _OPERATOR_INSTRUCTIONS
        assert "confirm_edit" in _OPERATOR_INSTRUCTIONS
        assert "SHOW" in _OPERATOR_INSTRUCTIONS
        assert "PROPOSE" in _OPERATOR_INSTRUCTIONS
        assert "CONFIRM" in _OPERATOR_INSTRUCTIONS
        assert "APPLY" in _OPERATOR_INSTRUCTIONS

    def test_instructions_has_plan_tiers(self):
        from src.cli.config import _OPERATOR_INSTRUCTIONS
        assert "Basic Plan" in _OPERATOR_INSTRUCTIONS
        assert "Growth Plan" in _OPERATOR_INSTRUCTIONS
        assert "Pro Plan" in _OPERATOR_INSTRUCTIONS
        assert "Self-hosted" in _OPERATOR_INSTRUCTIONS


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
