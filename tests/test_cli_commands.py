"""Tests for CLI commands: agent add/list/model/browser/remove, setup helpers."""

import json
import os
import tempfile
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.cli import cli


class TestAgentAdd:
    def test_add_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot"],
                input="Code review specialist\n\n\n",  # description + model + browser
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert "mybot" in agents_cfg["agents"]
            assert agents_cfg["agents"]["mybot"]["role"] == "Code review specialist"
            assert "openai/" in agents_cfg["agents"]["mybot"]["model"]
            assert agents_cfg["agents"]["mybot"].get("browser_backend", "basic") in ("basic", "")

            perms = json.loads(perms_file.read_text())
            assert "mybot" in perms["permissions"]
            assert "llm" in perms["permissions"]["mybot"]["allowed_apis"]

    def test_add_agent_with_model_flag(self, tmp_path):
        """--model flag skips interactive model selection."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot", "--model", "anthropic/claude-haiku-4-5-20251001"],
                input="Code review specialist\n\n",  # description + browser
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["model"] == "anthropic/claude-haiku-4-5-20251001"

    def test_add_agent_with_browser_flag(self, tmp_path):
        """--browser flag skips interactive browser selection."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"
        project_root = tmp_path

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        perms_file.write_text(json.dumps({"permissions": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
            patch("src.cli.config.PROJECT_ROOT", project_root),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "add", "mybot", "--model", "openai/gpt-4.1", "--browser", "stealth"],
                input="Web scraper agent\n",  # only description needed
            )
            assert result.exit_code == 0, result.output
            assert "mybot" in result.output
            assert "stealth" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["browser_backend"] == "stealth"

    def test_add_duplicate_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"existing": {"role": "test"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "add", "existing"])
            assert "already exists" in result.output


class TestAgentModel:
    def test_change_model_direct(self, tmp_path):
        """Direct model argument updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "model", "mybot", "anthropic/claude-sonnet-4-6"],
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["model"] == "anthropic/claude-sonnet-4-6"

    def test_change_model_interactive(self, tmp_path):
        """Interactive model picker updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "model", "mybot"],
                input="2\n",  # pick second model
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

    def test_change_model_nonexistent_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))
        agents_file.write_text(yaml.dump({
            "agents": {"other": {"role": "test", "model": "openai/gpt-4.1"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "model", "ghost", "some/model"])
            assert "not found" in result.output

    def test_change_model_same_noop(self, tmp_path):
        """Same model should report no change."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
            "llm": {"default_model": "anthropic/claude-haiku-4-5-20251001"},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "anthropic/claude-haiku-4-5-20251001",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "model", "mybot", "anthropic/claude-haiku-4-5-20251001"],
            )
            assert result.exit_code == 0
            assert "already uses" in result.output


class TestAgentBrowser:
    def test_change_browser_direct(self, tmp_path):
        """Direct backend argument updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "openai/gpt-4.1",
                "browser_backend": "basic",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "browser", "mybot", "stealth"],
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert agents_cfg["agents"]["mybot"]["browser_backend"] == "stealth"

    def test_change_browser_interactive(self, tmp_path):
        """Interactive browser picker updates agents.yaml."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "openai/gpt-4.1",
                "browser_backend": "basic",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "browser", "mybot"],
                input="2\n",  # pick stealth
            )
            assert result.exit_code == 0, result.output
            assert "->" in result.output

    def test_change_browser_nonexistent_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))
        agents_file.write_text(yaml.dump({
            "agents": {"other": {"role": "test", "model": "openai/gpt-4.1"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "browser", "ghost", "stealth"])
            assert "not found" in result.output

    def test_change_browser_same_noop(self, tmp_path):
        """Same browser should report no change."""
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {
                "role": "test",
                "model": "openai/gpt-4.1",
                "browser_backend": "basic",
            }},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["agent", "browser", "mybot", "basic"],
            )
            assert result.exit_code == 0
            assert "already uses" in result.output


class TestAgentList:
    def test_list_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {
                "bot1": {"role": "assistant", "model": "openai/gpt-4o-mini"},
                "bot2": {"role": "coder", "model": "anthropic/claude-sonnet-4-5-20250929"},
            },
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert result.exit_code == 0
            assert "bot1" in result.output
            assert "bot2" in result.output
            assert "openai/gpt-4o-mini" in result.output
            assert "anthropic/claude-sonnet-4-5-20250929" in result.output

    def test_list_no_agents(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "list"])
            assert "No agents" in result.output


class TestAgentRemove:
    def test_remove_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        perms_file = tmp_path / "permissions.json"

        config_file.write_text(yaml.dump({
            "mesh": {"host": "0.0.0.0", "port": 8420},
        }))
        agents_file.write_text(yaml.dump({
            "agents": {"mybot": {"role": "test"}},
        }))
        perms_file.write_text(json.dumps({
            "permissions": {"mybot": {"allowed_apis": ["llm"]}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
            patch("src.cli.config.PERMISSIONS_FILE", perms_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "remove", "mybot", "--yes"])
            assert result.exit_code == 0
            assert "Removed" in result.output

            agents_cfg = yaml.safe_load(agents_file.read_text())
            assert "mybot" not in agents_cfg.get("agents", {})

            perms = json.loads(perms_file.read_text())
            assert "mybot" not in perms["permissions"]

    def test_remove_nonexistent_agent(self, tmp_path):
        config_file = tmp_path / "mesh.yaml"
        agents_file = tmp_path / "agents.yaml"
        config_file.write_text(yaml.dump({"mesh": {}}))
        agents_file.write_text(yaml.dump({
            "agents": {"other": {"role": "test"}},
        }))

        with (
            patch("src.cli.config.CONFIG_FILE", config_file),
            patch("src.cli.config.AGENTS_FILE", agents_file),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["agent", "remove", "ghost"])
            assert "not found" in result.output


class TestChatNoMesh:
    def test_chat_fails_gracefully_when_mesh_not_running(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "testbot", "--port", "19999"])
        assert result.exit_code == 0
        assert "not running" in result.output


class TestPermissionsDefault:
    def test_new_agent_gets_default_permissions(self):
        """Verify the permissions module falls back to 'default' template."""
        from src.host.permissions import PermissionMatrix

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
