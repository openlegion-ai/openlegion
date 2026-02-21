"""CLI package for OpenLegion.

Re-exports for backward compatibility with tests and pyproject.toml entry point.
"""

from src.cli.config import (  # noqa: F401
    _PROVIDER_MODELS,
    _PROVIDERS,
    AGENTS_FILE,
    CONFIG_FILE,
    ENV_FILE,
    PERMISSIONS_FILE,
    PROJECT_FILE,
    PROJECT_ROOT,
    TEMPLATES_DIR,
    _add_agent_permissions,
    _add_agent_to_config,
    _apply_template,
    _check_docker_image,
    _check_docker_running,
    _create_agent,
    _default_description,
    _edit_agent_interactive,
    _get_default_model,
    _load_config,
    _load_permissions,
    _load_templates,
    _prompt_brightdata_key,
    _save_permissions,
    _set_collaborative_permissions,
    _set_env_key,
    _setup_agent_wizard,
    _update_agent_field,
)
from src.cli.formatting import _format_tool_result_hint, _format_tool_summary  # noqa: F401
from src.cli.main import cli  # noqa: F401
