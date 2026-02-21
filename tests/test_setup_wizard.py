"""Tests for the setup wizard: full setup, API key validation, summary."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from click.testing import CliRunner

from src.cli import cli
from src.setup_wizard import SetupWizard


def _make_project(tmp_path: Path) -> dict:
    """Create patching targets for a temporary project root."""
    config_file = tmp_path / "config" / "mesh.yaml"
    agents_file = tmp_path / "config" / "agents.yaml"
    perms_file = tmp_path / "config" / "permissions.json"
    project_file = tmp_path / "PROJECT.md"
    templates_dir = Path(__file__).resolve().parent.parent / "src" / "templates"

    config_file.parent.mkdir(parents=True, exist_ok=True)

    return {
        "config_file": config_file,
        "agents_file": agents_file,
        "perms_file": perms_file,
        "project_file": project_file,
        "templates_dir": templates_dir,
        "patches": {
            "src.cli.config.CONFIG_FILE": config_file,
            "src.cli.config.AGENTS_FILE": agents_file,
            "src.cli.config.PERMISSIONS_FILE": perms_file,
            "src.cli.config.PROJECT_FILE": project_file,
            "src.cli.config.PROJECT_ROOT": tmp_path,
            "src.cli.config.ENV_FILE": tmp_path / ".env",
        },
    }


def _clean_env() -> dict:
    """Build env dict with all OPENLEGION_CRED_* vars removed."""
    return {k: v for k, v in os.environ.items() if not k.startswith("OPENLEGION_CRED_")}


def _patch_all(project: dict):
    """Return a combined context manager patching all cli globals.

    Also clears OPENLEGION_CRED_* env vars so tests don't pick up the
    real .env file loaded at import time.
    """
    from contextlib import ExitStack
    stack = ExitStack()
    for target, value in project["patches"].items():
        stack.enter_context(patch(target, value))
    # Clear credential env vars — patch.dict restores them on exit
    stack.enter_context(patch.dict(os.environ, _clean_env(), clear=True))
    return stack


class TestSetupFull:
    def test_full_setup_creates_config(self, tmp_path):
        """Full setup with piped input creates all config files."""
        project = _make_project(tmp_path)

        # Input: provider=1 (anthropic), model=1, API key, project desc, agent name, description, browser
        piped_input = (
            "1\n"           # provider: Anthropic
            "1\n"           # model: first option
            "sk-test-key\n" # API key
            "My project\n"  # project description
            "none\n"        # template: none (custom)
            "myagent\n"     # agent name
            "Test agent\n"  # agent description
            "\n"            # browser: default (basic)
        )

        with _patch_all(project):
            with patch("src.cli.config._check_docker_running", return_value=True):
                with patch.object(SetupWizard, "_validate_api_key", return_value=True):
                    with patch("src.cli.config._set_env_key"):
                        runner = CliRunner()
                        result = runner.invoke(cli, ["setup"], input=piped_input)

        assert result.exit_code == 0, result.output
        assert "Setup Complete" in result.output

        # mesh.yaml created with model
        assert project["config_file"].exists()
        mesh_cfg = yaml.safe_load(project["config_file"].read_text())
        assert "anthropic/" in mesh_cfg["llm"]["default_model"]

        # agents.yaml created
        assert project["agents_file"].exists()
        agents_cfg = yaml.safe_load(project["agents_file"].read_text())
        assert "myagent" in agents_cfg["agents"]

        # permissions.json created
        assert project["perms_file"].exists()
        perms = json.loads(project["perms_file"].read_text())
        assert "myagent" in perms["permissions"]

        # PROJECT.md created
        assert project["project_file"].exists()
        assert "My project" in project["project_file"].read_text()

    def test_full_setup_existing_config_detected(self, tmp_path):
        """Pre-populated config shows 'Existing configuration found' message."""
        project = _make_project(tmp_path)

        # Pre-populate config
        project["config_file"].parent.mkdir(parents=True, exist_ok=True)
        project["config_file"].write_text(yaml.dump({
            "llm": {"default_model": "openai/gpt-4.1"},
        }))
        project["agents_file"].write_text(yaml.dump({
            "agents": {"oldbot": {"role": "test"}},
        }))

        # Input: 'n' to decline overwrite
        piped_input = "n\n"

        with _patch_all(project):
            with patch("src.cli.config._check_docker_running", return_value=True):
                runner = CliRunner()
                result = runner.invoke(cli, ["setup"], input=piped_input)

        assert result.exit_code == 0, result.output
        assert "Existing configuration found" in result.output
        assert "openai" in result.output.lower() or "gpt-4.1" in result.output
        assert "oldbot" in result.output

    def test_full_setup_template_selection(self, tmp_path):
        """Selecting a template creates agents from the template."""
        project = _make_project(tmp_path)

        # Input: provider=1, model=1, key, skip project, template=starter
        piped_input = (
            "1\n"           # provider
            "1\n"           # model
            "sk-test-key\n" # API key
            "\n"            # skip project
            "starter\n"    # template name
        )

        with _patch_all(project):
            with patch("src.cli.config._check_docker_running", return_value=True):
                with patch.object(SetupWizard, "_validate_api_key", return_value=True):
                    with patch("src.cli.config._set_env_key"):
                        runner = CliRunner()
                        result = runner.invoke(cli, ["setup"], input=piped_input)

        assert result.exit_code == 0, result.output
        assert "assistant" in result.output
        assert "Setup Complete" in result.output

    def test_full_setup_invalid_key_retry(self, tmp_path):
        """Invalid key on first attempt triggers retry, valid on second."""
        project = _make_project(tmp_path)

        # Input: provider=1, model=1, bad key, good key, skip project, agent, browser
        piped_input = (
            "1\n"                # provider
            "1\n"                # model
            "bad-key\n"          # first attempt (will fail validation)
            "good-key\n"         # second attempt (will pass)
            "\n"                 # skip project
            "none\n"             # no template
            "bot\n"              # agent name
            "Test bot\n"         # agent description
            "\n"                 # browser: default (basic)
        )

        call_count = {"n": 0}

        def _mock_validate(self_arg, provider, api_key):
            call_count["n"] += 1
            return call_count["n"] >= 2  # Fail first, pass second

        with _patch_all(project):
            with patch("src.cli.config._check_docker_running", return_value=True):
                with patch.object(SetupWizard, "_validate_api_key", _mock_validate):
                    with patch("src.cli.config._set_env_key"):
                        runner = CliRunner()
                        result = runner.invoke(cli, ["setup"], input=piped_input)

        assert result.exit_code == 0, result.output
        assert "invalid" in result.output.lower()
        assert "Setup Complete" in result.output


class TestValidateApiKey:
    def test_validate_key_success(self):
        """Valid key returns True."""
        wizard = SetupWizard(Path("/tmp/test"))

        mock_response = MagicMock()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = wizard._validate_api_key("anthropic", "sk-valid-key")

        assert result is True

    def test_validate_key_failure(self):
        """Authentication error returns False."""
        wizard = SetupWizard(Path("/tmp/test"))

        import litellm
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.AuthenticationError(
                message="Invalid API Key",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
            ),
        ):
            result = wizard._validate_api_key("anthropic", "sk-bad-key")

        assert result is False

    def test_validate_key_timeout(self):
        """Timeout returns True (don't block setup on network issues)."""
        wizard = SetupWizard(Path("/tmp/test"))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=TimeoutError("Connection timed out"),
        ):
            result = wizard._validate_api_key("anthropic", "sk-some-key")

        assert result is True

    def test_validate_unknown_provider(self):
        """Unknown provider skips validation and returns True."""
        wizard = SetupWizard(Path("/tmp/test"))
        result = wizard._validate_api_key("unknown_provider", "some-key")
        assert result is True


class TestSummaryCard:
    def test_summary_card_output(self, capsys):
        """Summary card contains provider, model, and agents."""
        wizard = SetupWizard(Path("/tmp/test"))
        wizard._print_summary("anthropic", "anthropic/claude-sonnet-4-6", ["assistant", "coder"])

        captured = capsys.readouterr()
        assert "Anthropic" in captured.out
        assert "claude-sonnet-4-6" in captured.out
        assert "assistant, coder" in captured.out
        assert "openlegion start" in captured.out
        assert "Setup Complete" in captured.out
        # Check box drawing chars
        assert "┌" in captured.out
        assert "└" in captured.out

    def test_summary_card_single_agent(self, capsys):
        """Summary card works with a single agent."""
        wizard = SetupWizard(Path("/tmp/test"))
        wizard._print_summary("openai", "openai/gpt-4.1", ["mybot"])

        captured = capsys.readouterr()
        assert "Openai" in captured.out
        assert "gpt-4.1" in captured.out
        assert "mybot" in captured.out


class TestDetectExistingConfig:
    def test_detect_no_config(self, tmp_path):
        """No config files returns None."""
        wizard = SetupWizard(tmp_path)
        assert wizard._detect_existing_config() is None

    def test_detect_with_config(self, tmp_path):
        """Existing config files return summary."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "mesh.yaml").write_text(yaml.dump({
            "llm": {"default_model": "anthropic/claude-sonnet-4-6"},
        }))
        (config_dir / "agents.yaml").write_text(yaml.dump({
            "agents": {"bot1": {"role": "test"}, "bot2": {"role": "coder"}},
        }))

        wizard = SetupWizard(tmp_path)
        result = wizard._detect_existing_config()

        assert result is not None
        assert result["provider"] == "anthropic"
        assert result["model"] == "anthropic/claude-sonnet-4-6"
        assert "bot1" in result["agents"]
        assert "bot2" in result["agents"]

    def test_detect_empty_config(self, tmp_path):
        """Empty config files return None."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "mesh.yaml").write_text("")
        (config_dir / "agents.yaml").write_text("")

        wizard = SetupWizard(tmp_path)
        assert wizard._detect_existing_config() is None


class TestStepHeader:
    def test_step_header_output(self, capsys):
        """Step header prints formatted header."""
        wizard = SetupWizard(Path("/tmp/test"))
        wizard._print_step_header(2, 4, "Your Project")

        captured = capsys.readouterr()
        assert "[2/4]" in captured.out
        assert "Your Project" in captured.out
