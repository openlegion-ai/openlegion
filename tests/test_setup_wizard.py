"""Tests for the setup wizard: API key validation, summary, inline setup."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click
import yaml

from src.setup_wizard import InlineSetup, SetupWizard


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

        import litellm
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.Timeout(
                message="Connection timed out",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
            ),
        ):
            result = wizard._validate_api_key("anthropic", "sk-some-key")

        assert result is True

    def test_validate_unknown_provider(self):
        """Unknown provider skips validation and returns True."""
        wizard = SetupWizard(Path("/tmp/test"))
        result = wizard._validate_api_key("unknown_provider", "some-key")
        assert result is True

    def test_validate_permission_denied_returns_false(self):
        """PermissionDeniedError returns False."""
        wizard = SetupWizard(Path("/tmp/test"))

        import httpx
        import litellm
        mock_resp = httpx.Response(status_code=403, request=httpx.Request("POST", "https://api.anthropic.com"))
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.PermissionDeniedError(
                message="Permission denied",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
                response=mock_resp,
            ),
        ):
            result = wizard._validate_api_key("anthropic", "sk-bad-key")

        assert result is False

    def test_validate_unknown_error_returns_false(self):
        """Unknown exception returns False (safe default)."""
        wizard = SetupWizard(Path("/tmp/test"))

        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Something unexpected"),
        ):
            result = wizard._validate_api_key("anthropic", "sk-some-key")

        assert result is False

    def test_validate_rate_limit_returns_true(self):
        """RateLimitError returns True (transient, don't block setup)."""
        wizard = SetupWizard(Path("/tmp/test"))

        import litellm
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                llm_provider="anthropic",
                model="anthropic/claude-haiku-4-5-20251001",
            ),
        ):
            result = wizard._validate_api_key("anthropic", "sk-some-key")

        assert result is True

    def test_validate_uses_api_key_kwarg(self):
        """Validation passes api_key directly (no env var mutation)."""
        wizard = SetupWizard(Path("/tmp/test"))

        mock_response = MagicMock()
        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            wizard._validate_api_key("anthropic", "sk-test-key-123")

        # Verify api_key was passed as a kwarg, not via env vars
        call_kwargs = mock_call.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-key-123"


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


class TestPromptWithBack:
    def test_back_returns_none(self):
        """Typing 'back' returns None."""
        with patch("click.prompt", return_value="back"):
            result = SetupWizard._prompt_with_back("prompt")
            assert result is None

    def test_back_case_insensitive(self):
        """'Back', 'BACK', ' back ' all return None."""
        for val in ["Back", "BACK", " back "]:
            with patch("click.prompt", return_value=val):
                result = SetupWizard._prompt_with_back("prompt")
                assert result is None

    def test_normal_input_passes_through(self):
        """Normal input is returned unchanged."""
        with patch("click.prompt", return_value="hello"):
            result = SetupWizard._prompt_with_back("prompt")
            assert result == "hello"

    def test_back_with_int_range_type(self):
        """'back' works even when type=IntRange is provided."""
        with patch("click.prompt", return_value="back"):
            result = SetupWizard._prompt_with_back(
                "prompt", type=click.IntRange(1, 5), default=1,
            )
            assert result is None

    def test_valid_int_with_int_range_type(self):
        """Valid integer input is converted when type=IntRange is provided."""
        with patch("click.prompt", return_value="3"):
            result = SetupWizard._prompt_with_back(
                "prompt", type=click.IntRange(1, 5), default=1,
            )
            assert result == 3


class TestInlineSetup:
    def test_needs_setup_true_when_no_vault(self):
        """needs_setup returns True when credential_vault is None."""
        assert InlineSetup.needs_setup(None) is True

    def test_needs_setup_true_when_empty_vault(self):
        """needs_setup returns True when vault has no credentials in either tier."""
        vault = MagicMock()
        vault.credentials = {}
        vault.system_credentials = {}
        assert InlineSetup.needs_setup(vault) is True

    def test_needs_setup_false_when_creds_exist(self):
        """needs_setup returns False when vault has credentials."""
        vault = MagicMock()
        vault.credentials = {"myapp_key": "val"}
        vault.system_credentials = {}
        assert InlineSetup.needs_setup(vault) is False

    def test_needs_setup_false_when_system_creds_exist(self):
        """needs_setup returns False when vault has system credentials."""
        vault = MagicMock()
        vault.credentials = {}
        vault.system_credentials = {"anthropic_api_key": "sk-test"}
        assert InlineSetup.needs_setup(vault) is False

    def test_run_stores_credential(self, tmp_path):
        """run() stores the credential via vault and writes mesh.yaml."""
        vault = MagicMock()
        vault.credentials = {}
        setup = InlineSetup(tmp_path, credential_vault=vault)

        # Input: provider=1 (Anthropic), auth_type="1" (API key), key, model=1
        with patch("click.prompt", side_effect=[1, "1", "sk-test-key", 1]):
            with patch.object(SetupWizard, "_validate_api_key", return_value=True):
                with patch("src.cli.config._set_env_key"):
                    setup.run()

        vault.add_credential.assert_called_once()
        call_args = vault.add_credential.call_args[0]
        assert "api_key" in call_args[0]
        assert call_args[1] == "sk-test-key"

        # mesh.yaml should be written
        config_file = tmp_path / "config" / "mesh.yaml"
        assert config_file.exists()
        cfg = yaml.safe_load(config_file.read_text())
        assert "default_model" in cfg.get("llm", {})


class TestOAuthTokenValidation:
    """Tests for OAuth setup-token format validation."""

    def test_valid_token_format(self):
        token = "sk-ant-oat01-" + "x" * 80
        assert SetupWizard._validate_oauth_token_format(token) is None

    def test_wrong_prefix(self):
        err = SetupWizard._validate_oauth_token_format("sk-ant-api03-regular")
        assert err is not None
        assert "prefix" in err.lower()

    def test_too_short(self):
        err = SetupWizard._validate_oauth_token_format("sk-ant-oat01-short")
        assert err is not None
        assert "truncated" in err.lower()

    def test_newline_in_token(self):
        token = "sk-ant-oat01-" + "x" * 40 + "\n" + "x" * 40
        err = SetupWizard._validate_oauth_token_format(token)
        assert err is not None
        assert "line break" in err.lower()

    def test_inline_setup_autodetects_oauth_token(self, tmp_path):
        """InlineSetup detects pasted OAuth token and validates it."""
        vault = MagicMock()
        vault.credentials = {}
        setup = InlineSetup(tmp_path, credential_vault=vault)
        oauth_token = "sk-ant-oat01-" + "x" * 80

        # Input: provider=1 (Anthropic), auth_type="1" (API key),
        # but user pastes an OAuth token, model=1
        with patch("click.prompt", side_effect=[1, "1", oauth_token, 1]):
            with patch.object(
                SetupWizard, "_validate_oauth_token_live", return_value=True,
            ):
                with patch("src.cli.config._set_env_key"):
                    setup.run()

        vault.add_credential.assert_called_once()
        call_args = vault.add_credential.call_args[0]
        assert call_args[1] == oauth_token


