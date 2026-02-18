"""Tests for CLI commands: agent create/list, config set-key."""

import json
import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.cli import cli


class TestConfigSetKey:
    @staticmethod
    def _save_and_restore_env(key: str):
        """Context manager to save and restore an env var around a test."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            original = os.environ.get(key)
            try:
                yield
            finally:
                if original is not None:
                    os.environ[key] = original
                else:
                    os.environ.pop(key, None)

        return _ctx()

    def test_set_key_creates_env(self, tmp_path):
        env_file = tmp_path / ".env"
        with self._save_and_restore_env("OPENLEGION_CRED_OPENAI_API_KEY"):
            with patch("src.cli.ENV_FILE", env_file):
                runner = CliRunner()
                result = runner.invoke(cli, ["config", "set-key", "openai", "sk-test-key"])
                assert result.exit_code == 0
                assert "Saved" in result.output
                content = env_file.read_text()
                assert "OPENLEGION_CRED_OPENAI_API_KEY=sk-test-key" in content

    def test_set_key_updates_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENLEGION_CRED_OPENAI_API_KEY=old-key\n")
        with self._save_and_restore_env("OPENLEGION_CRED_OPENAI_API_KEY"):
            with patch("src.cli.ENV_FILE", env_file):
                runner = CliRunner()
                runner.invoke(cli, ["config", "set-key", "openai", "new-key"])
                content = env_file.read_text()
                assert "new-key" in content
                assert "old-key" not in content


class TestAgentCreate:
    def test_create_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4o-mini"},
            "agents": {},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.CONFIG_FILE", config_file),
            patch("src.cli.PERMISSIONS_FILE", perms_file),
            patch("src.cli.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "create", "mybot"],
                input="coder\nopenai/gpt-4o-mini\nYou are a coder.\n",
            )
            assert result.exit_code == 0
            assert "mybot" in result.output

            cfg = yaml.safe_load(config_file.read_text())
            assert "mybot" in cfg["agents"]
            assert cfg["agents"]["mybot"]["role"] == "coder"

            perms = json.loads(perms_file.read_text())
            assert "mybot" in perms["permissions"]
            assert "llm" in perms["permissions"]["mybot"]["allowed_apis"]

    def test_create_duplicate_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "agents": {"existing": {"role": "test"}},
        }))

        with patch("src.cli.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "create", "existing"])
            assert "already exists" in result.output


class TestAgentList:
    def test_list_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "agents": {
                "bot1": {"role": "assistant", "model": "openai/gpt-4o-mini"},
                "bot2": {"role": "coder", "model": "anthropic/claude-sonnet-4-5-20250929"},
            },
        }))

        with patch("src.cli.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert result.exit_code == 0
            assert "bot1" in result.output
            assert "bot2" in result.output
            assert "assistant" in result.output
            assert "coder" in result.output

    def test_list_no_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        config_file.write_text(yaml.dump({"mesh": {}, "agents": {}}))

        with patch("src.cli.CONFIG_FILE", config_file):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert "No agents" in result.output


class TestPermissionsDefault:
    def test_new_agent_gets_default_permissions(self):
        """Verify the permissions module falls back to 'default' template."""
        from src.host.permissions import PermissionMatrix

        # Write a temp permissions file with only "default"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "permissions": {
                    "default": {
                        "can_message": ["orchestrator"],
                        "allowed_apis": ["llm"],
                        "blackboard_read": ["context/*"],
                    }
                }
            }, f)
            f.flush()
            pm = PermissionMatrix(config_path=f.name)

        perms = pm.get_permissions("some_unknown_agent")
        assert perms.agent_id == "some_unknown_agent"
        assert "orchestrator" in perms.can_message
        assert "llm" in perms.allowed_apis
        assert pm.can_use_api("some_unknown_agent", "llm")
        os.unlink(f.name)
