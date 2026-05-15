"""Static guardrails on the agent / shared / host trust boundary.

The agent container ships only ``src/agent`` + ``src/shared`` (see
``Dockerfile.agent``). Any import from ``src.host`` inside agent or
shared code raises ``ModuleNotFoundError`` at runtime inside the
container — the bug that broke every ``hand_off`` call once
orchestration v2 was enabled.

These tests parse each ``.py`` under ``src/agent`` and ``src/shared``
and assert no import targets ``src.host`` (top-level imports and
function-level imports alike). Both branches of ``if TYPE_CHECKING:``
are checked too — even though those don't execute at runtime, allowing
them encodes a phantom dependency that mypy / IDE tooling will follow
into a module the agent container can't open.
"""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _host_imports_in(directory: pathlib.Path) -> list[str]:
    """Return ``path:line: <import>`` for every ``src.host`` import found."""
    offenders: list[str] = []
    for py_file in sorted(directory.rglob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        rel = py_file.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "src.host" or module.startswith("src.host."):
                    offenders.append(f"{rel}:{node.lineno}: from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src.host" or alias.name.startswith("src.host."):
                        offenders.append(
                            f"{rel}:{node.lineno}: import {alias.name}",
                        )
    return offenders


def test_src_agent_does_not_import_from_src_host():
    """Agents run in a container that only ships src/agent + src/shared.

    Any ``from src.host…`` (top-level or function-level) raises
    ``ModuleNotFoundError`` at runtime. This guardrail caught the
    ``hand_off`` outage that motivated extracting title-policy helpers
    into ``src/shared/task_titles``.
    """
    offenders = _host_imports_in(REPO_ROOT / "src" / "agent")
    assert not offenders, (
        "Agent code must not import from src.host — the agent container "
        "doesn't ship src/host. Move the helper into src/shared instead. "
        "Found:\n  " + "\n  ".join(offenders)
    )


def test_src_shared_does_not_import_from_src_host():
    """Shared code is reachable from both zones — must stay decoupled.

    A ``src.host`` import inside ``src/shared`` would propagate the same
    runtime failure to every shared-module caller in the agent.
    """
    offenders = _host_imports_in(REPO_ROOT / "src" / "shared")
    assert not offenders, (
        "Shared code must not import from src.host — it's imported by "
        "the agent container which doesn't ship src/host. Found:\n  "
        + "\n  ".join(offenders)
    )
