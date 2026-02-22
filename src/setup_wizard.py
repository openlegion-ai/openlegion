"""Setup wizard for OpenLegion first-run experience.

Extracted from cli.py to keep module size manageable. Provides:
  - SetupWizard.run_full()      Interactive multi-step setup
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
import yaml

# Validation models: cheapest model per provider for key checks
_VALIDATION_MODELS = {
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "openai": "openai/gpt-4.1-mini",
    "gemini": "gemini/gemini-2.5-flash",
    "deepseek": "deepseek/deepseek-chat",
    "moonshot": "moonshot/moonshot-v1-128k",
    "xai": "xai/grok-3-mini",
    "groq": "groq/llama-3.1-8b-instant",
}


class SetupWizard:
    """Interactive setup wizard with validation and visual polish."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_file = project_root / ".env"
        self.config_file = project_root / "config" / "mesh.yaml"
        self.agents_file = project_root / "config" / "agents.yaml"
        self.permissions_file = project_root / "config" / "permissions.json"
        self.project_file = project_root / "PROJECT.md"
        self.templates_dir = Path(__file__).parent / "templates"

    # ── Public entry points ──────────────────────────────────

    def run_full(self) -> None:
        """Full interactive setup (replaces original cli.py setup body).

        Supports Ctrl+C to exit cleanly and 'back' to return to the previous step.
        """
        try:
            self._run_full_inner()
        except KeyboardInterrupt:
            click.echo("\n\n  Setup cancelled.")
            sys.exit(0)

    def _run_full_inner(self) -> None:
        """Inner setup logic with step-loop for back navigation."""
        from src.cli.config import (
            _PROVIDER_MODELS,
            _PROVIDERS,
            _apply_template,
            _check_docker_running,
            _load_config,
            _load_templates,
            _set_collaborative_permissions,
            _set_env_key,
            _setup_agent_wizard,
        )

        click.echo("=== OpenLegion Setup ===")
        click.echo("  (Ctrl+C to quit, 'back' to return to previous step)\n")

        # Check prerequisites
        if not _check_docker_running():
            click.echo(
                "Docker is not running or not accessible.\n"
                "Please start Docker and ensure your user has permission to use it.\n"
                "  - Linux: sudo systemctl start docker && sudo usermod -aG docker $USER\n"
                "  - macOS/Windows: Start Docker Desktop",
                err=True,
            )
            sys.exit(1)

        # Detect existing config
        existing = self._detect_existing_config()
        if existing:
            click.echo("Existing configuration found:")
            click.echo(f"  Provider: {existing['provider']}")
            click.echo(f"  Model:    {existing['model']}")
            if existing["agents"]:
                click.echo(f"  Agents:   {', '.join(existing['agents'])}")
            click.echo("")
            if not click.confirm("Overwrite existing configuration?", default=False):
                click.echo("Setup cancelled. Existing config kept.")
                return

        total_steps = 4
        # State accumulated across steps
        state: dict = {}

        step = 1
        while step <= total_steps:
            if step == 1:
                result = self._step_provider(total_steps, _PROVIDERS, _PROVIDER_MODELS, _set_env_key)
                if result is None:
                    click.echo("\n  Already at the first step.\n")
                    continue
                state.update(result)
                step = 2

            elif step == 2:
                result = self._step_project(total_steps)
                if result is None:
                    step = 1
                    continue
                step = 3

            elif step == 3:
                result = self._step_agents(
                    total_steps,
                    state["selected_model"],
                    _load_config,
                    _load_templates,
                    _apply_template,
                    _setup_agent_wizard,
                )
                if result is None:
                    step = 2
                    continue
                state["created_agents"] = result["created_agents"]
                step = 4

            elif step == 4:
                result = self._step_collaboration(total_steps, _set_collaborative_permissions)
                if result is None:
                    step = 3
                    continue
                step = 5

        # Summary
        self._print_summary(state["provider"], state["selected_model"], state.get("created_agents", []))

    @staticmethod
    def _prompt_with_back(prompt_text: str, **kwargs):
        """Wrapper around click.prompt that returns None when user types 'back'.

        When a ``type`` kwarg with validation (e.g. ``click.IntRange``) is
        provided, we strip it and validate manually *after* checking for
        'back'.  This avoids Click rejecting 'back' as invalid input.
        """
        click_type = kwargs.pop("type", None)
        value = click.prompt(prompt_text, **kwargs)
        if isinstance(value, str) and value.strip().lower() == "back":
            return None
        # Re-apply type validation if one was provided
        if click_type is not None and value is not None:
            try:
                value = click_type.convert(value, None, None)
            except (click.BadParameter, Exception):
                click.echo(f"  Error: invalid input. Try again or type 'back'.")
                return SetupWizard._prompt_with_back(prompt_text, type=click_type, **kwargs)
        return value

    def _step_provider(self, total_steps, _PROVIDERS, _PROVIDER_MODELS, _set_env_key) -> dict | None:
        """Step 1: LLM provider + model + API key. Returns None for 'back' (no-op at step 1)."""
        self._print_step_header(1, total_steps, "LLM Provider")

        for i, p in enumerate(_PROVIDERS, 1):
            click.echo(f"  {i}. {p['label']}")
        click.echo(
            "\n  Tip: Anthropic Claude and Moonshot Kimi are recommended for agentic\n"
            "  tasks (browser automation, web interaction, tool use). They have\n"
            "  built-in computer use training and strong tool-calling support.\n"
        )
        raw = self._prompt_with_back(
            "  Select provider",
            type=click.IntRange(1, len(_PROVIDERS)),
            default=1,
        )
        if raw is None:
            return None  # 'back' at step 1
        choice = raw
        provider = _PROVIDERS[choice - 1]["name"]
        click.echo(f"  Selected: {_PROVIDERS[choice - 1]['label']}\n")

        # Model selection
        models = _PROVIDER_MODELS[provider]
        click.echo("  Available models:")
        for i, m in enumerate(models, 1):
            click.echo(f"  {i}. {m}")
        raw = self._prompt_with_back(
            "\n  Select model",
            type=click.IntRange(1, len(models)),
            default=1,
        )
        if raw is None:
            return None
        model_choice = raw
        selected_model = models[model_choice - 1]
        click.echo(f"  Selected: {selected_model}\n")

        # API key with validation
        key_name = f"{provider}_api_key"
        existing_key = os.environ.get(f"OPENLEGION_CRED_{key_name.upper()}", "")
        if existing_key:
            click.echo(f"  API key already set for {provider}.")
            if click.confirm("  Replace it?", default=False):
                api_key = self._prompt_and_validate_key(provider, _PROVIDERS[choice - 1]["label"])
                if api_key:
                    _set_env_key(key_name, api_key)
        else:
            api_key = self._prompt_and_validate_key(provider, _PROVIDERS[choice - 1]["label"])
            if api_key:
                _set_env_key(key_name, api_key)

        # Update default model in mesh config
        mesh_cfg: dict = {}
        if self.config_file.exists():
            with open(self.config_file) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        mesh_cfg.setdefault("llm", {})["default_model"] = selected_model
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w") as f:
            yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)

        return {"provider": provider, "selected_model": selected_model, "choice": choice}

    def _step_project(self, total_steps) -> dict | None:
        """Step 2: Project definition. Returns None for 'back'."""
        self._print_step_header(2, total_steps, "Your Project (optional)")

        project_desc = self._prompt_with_back(
            "  What are you building? (press Enter to skip, 'back' for previous step)",
            default="",
            show_default=False,
        )
        if project_desc is None:
            return None
        if project_desc:
            self.project_file.write_text(
                f"# PROJECT.md\n\n"
                f"## What We're Building\n{project_desc}\n\n"
                f"## Current Priority\n[Define your current focus]\n\n"
                f"## Hard Constraints\n[Budget limits, deadlines, compliance rules]\n"
            )
            click.echo("  Saved to PROJECT.md. Every agent will see this as their north star.")
        elif not self.project_file.exists():
            click.echo("  Skipped. You can define it later by editing PROJECT.md.")

        return {}

    def _step_agents(self, total_steps, selected_model, _load_config, _load_templates, _apply_template, _setup_agent_wizard) -> dict | None:
        """Step 3: Agent setup. Returns None for 'back'."""
        self._print_step_header(3, total_steps, "Your Agents")

        cfg = _load_config()
        existing_agents = list(cfg.get("agents", {}).keys())
        created_agents: list[str] = []

        if existing_agents:
            click.echo(f"  Existing agents: {', '.join(existing_agents)}")

            from src.cli.config import AGENTS_FILE
            if AGENTS_FILE.exists():
                with open(AGENTS_FILE) as f:
                    agents_data = yaml.safe_load(f) or {}
                stale = [
                    n for n, a in agents_data.get("agents", {}).items()
                    if a.get("model") != selected_model
                ]
                if stale:
                    click.echo(f"\n  These agents use a different model: {', '.join(stale)}")
                    if click.confirm(f"  Update all agents to {selected_model}?", default=True):
                        for n in stale:
                            agents_data["agents"][n]["model"] = selected_model
                        with open(AGENTS_FILE, "w") as f:
                            yaml.dump(agents_data, f, default_flow_style=False, sort_keys=False)
                        click.echo(f"  Updated {len(stale)} agent(s).")

            if not click.confirm("  Add another agent?", default=False):
                click.echo("  Keeping existing agents.")
                created_agents = existing_agents
            else:
                name = _setup_agent_wizard(selected_model)
                created_agents = existing_agents + [name]
        else:
            templates = _load_templates()
            if templates:
                tpl_names = list(templates.keys())
                tpl_display = ", ".join(tpl_names)
                raw = self._prompt_with_back(
                    f"  Start from a template? ({tpl_display}) or 'none' for custom",
                    default="none",
                )
                if raw is None:
                    return None
                use_template = raw
                if use_template != "none" and use_template in templates:
                    created_agents = _apply_template(use_template, templates[use_template])
                    click.echo(f"  Created agents: {', '.join(created_agents)}")
                else:
                    name = _setup_agent_wizard(selected_model)
                    created_agents = [name]
            else:
                name = _setup_agent_wizard(selected_model)
                created_agents = [name]

        return {"created_agents": created_agents}

    def _step_collaboration(self, total_steps, _set_collaborative_permissions) -> dict | None:
        """Step 4: Collaboration (auto-completes, no user prompt)."""
        self._print_step_header(4, total_steps, "Collaboration")

        mesh_cfg = {}
        if self.config_file.exists():
            with open(self.config_file) as f:
                mesh_cfg = yaml.safe_load(f) or {}
        if "collaboration" not in mesh_cfg:
            mesh_cfg["collaboration"] = True
            with open(self.config_file, "w") as f:
                yaml.dump(mesh_cfg, f, default_flow_style=False, sort_keys=False)
            _set_collaborative_permissions()
        click.echo("  Inter-agent collaboration enabled.\n")

        return {}

    # ── API key validation ───────────────────────────────────

    def _prompt_and_validate_key(self, provider: str, label: str) -> str:
        """Prompt for API key with validation and retry loop. Returns the valid key."""
        max_attempts = 3
        for attempt in range(max_attempts):
            api_key = click.prompt(f"  {label} API key", hide_input=True)
            if not api_key.strip():
                click.echo("  No key provided.")
                continue

            click.echo("  Validating API key...", nl=False)
            valid = self._validate_api_key(provider, api_key.strip())
            if valid:
                click.echo(" valid.\n")
                return api_key.strip()
            else:
                remaining = max_attempts - attempt - 1
                if remaining > 0:
                    click.echo(f" invalid. {remaining} attempt(s) remaining.")
                else:
                    click.echo(" invalid.")
                    click.echo("  Saving key anyway — you can fix it later in .env")
                    return api_key.strip()
        return ""

    def _validate_api_key(self, provider: str, api_key: str) -> bool:
        """Lightweight key check using litellm with minimal token usage."""
        validation_model = _VALIDATION_MODELS.get(provider)
        if not validation_model:
            return True  # Unknown provider — skip validation

        try:
            import litellm

            # Map provider to the env var litellm expects
            env_mapping = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "moonshot": "MOONSHOT_API_KEY",
                "xai": "XAI_API_KEY",
                "groq": "GROQ_API_KEY",
            }
            env_var = env_mapping.get(provider)
            old_val = os.environ.get(env_var, "") if env_var else ""

            try:
                if env_var:
                    os.environ[env_var] = api_key
                asyncio.run(
                    litellm.acompletion(
                        model=validation_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    )
                )
                return True
            except litellm.AuthenticationError:
                return False
            except Exception:
                # Network errors, rate limits, etc. — don't block setup
                return True
            finally:
                if env_var:
                    if old_val:
                        os.environ[env_var] = old_val
                    else:
                        os.environ.pop(env_var, None)
        except ImportError:
            # litellm not installed — skip validation
            return True

    # ── Config detection ─────────────────────────────────────

    def _detect_existing_config(self) -> dict | None:
        """Check for existing config and return summary if found."""
        if not self.config_file.exists() and not self.agents_file.exists():
            return None

        result: dict = {"provider": "unknown", "model": "unknown", "agents": []}

        if self.config_file.exists():
            with open(self.config_file) as f:
                cfg = yaml.safe_load(f) or {}
            model = cfg.get("llm", {}).get("default_model", "")
            if model:
                result["model"] = model
                result["provider"] = model.split("/")[0] if "/" in model else "unknown"

        if self.agents_file.exists():
            with open(self.agents_file) as f:
                agents_cfg = yaml.safe_load(f) or {}
            result["agents"] = list(agents_cfg.get("agents", {}).keys())

        # Only report if there's something meaningful
        if result["model"] == "unknown" and not result["agents"]:
            return None
        return result

    # ── Display helpers ──────────────────────────────────────

    def _print_step_header(self, step_num: int, total: int, title: str) -> None:
        """Print consistent step header."""
        click.echo(f"\n[{step_num}/{total}] {title}\n")

    def _print_summary(self, provider: str, model: str, agents: list[str]) -> None:
        """Print ASCII summary card showing what was configured."""
        # Strip provider prefix from model for display
        display_model = model.split("/", 1)[1] if "/" in model else model
        display_provider = provider.title()
        agents_str = ", ".join(agents) if agents else "(none)"

        # Calculate width based on content
        lines = [
            f"  Provider:  {display_provider}",
            f"  Model:     {display_model}",
            f"  Agents:    {agents_str}",
            "  Next: openlegion start",
        ]
        inner_width = max(len(line) for line in lines) + 2
        inner_width = max(inner_width, 33)  # minimum width

        border = "─" * inner_width
        click.echo(f"┌{border}┐")
        click.echo(f"│{'  OpenLegion Setup Complete':<{inner_width}}│")
        click.echo(f"├{border}┤")
        for line in lines[:3]:
            click.echo(f"│{line:<{inner_width}}│")
        click.echo(f"├{border}┤")
        click.echo(f"│{lines[3]:<{inner_width}}│")
        click.echo(f"└{border}┘")
