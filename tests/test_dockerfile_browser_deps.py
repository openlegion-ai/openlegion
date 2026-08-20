"""Static-check tests for the browser image's Python dependencies.

PR #815 shipped to production with a broken browser image because the
browser install list used plain ``uvicorn`` instead of
``uvicorn[standard]``. The ``[standard]`` extra is what pulls the
``websockets`` package — without it, uvicorn rejects every WebSocket
upgrade with HTTP 404 at the protocol layer, before any FastAPI route
handler runs. The per-agent VNC iframe needs WS upgrades to work, so
this regression silently broke the dashboard.

That list used to be written literally in ``Dockerfile.browser`` — a
second, invisible dependency manifest that no packaging tool could see
and that had to be kept in sync with ``pyproject.toml`` by hand. It now
lives in ``pyproject.toml``'s ``[project.optional-dependencies]
browser`` extra, and the Dockerfile installs from there. These tests
therefore check two things: the extra declares what the browser service
imports at runtime, and the Dockerfile still sources its packages from
that extra rather than re-introducing a literal list.

These are static checks against file content. They don't build an image
(too slow for unit-test cadence). The ``build-and-smoke`` job in
.github/workflows/browser-image.yml closes the remaining gap by actually
building and importing.

``TestBuildTriggerCoverage`` covers the other half of that gap: the
``build-and-smoke`` job is path-filtered, so it only defends the paths it
watches. Those tests pin the filter against the Dockerfile's own ``COPY``
list, so adding a ``COPY`` without widening the trigger fails here rather
than shipping unbuilt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

try:  # tomllib is stdlib on 3.11+; the project floor is 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.browser"
WORKFLOW = REPO_ROOT / ".github/workflows/browser-image.yml"
PYPROJECT = REPO_ROOT / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    tomllib is None, reason="tomllib requires Python 3.11+ (CI runs 3.11/3.12)"
)


@pytest.fixture(scope="module")
def browser_extra() -> list[str]:
    """The ``browser`` extra from pyproject — the browser image's manifest."""
    data = tomllib.loads(PYPROJECT.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "browser" in extras, (
        "pyproject.toml declares no 'browser' extra. Dockerfile.browser "
        "installs from it — without the extra the image build fails with "
        "a KeyError."
    )
    return extras["browser"]


@pytest.fixture(scope="module")
def browser_requirements(browser_extra: list[str]) -> str:
    """The extra joined into one searchable blob."""
    return "\n".join(browser_extra)


def _dockerfile_install_block() -> str:
    """Return the concatenated text of every ``RUN pip install ...`` line.

    Joins continuation lines (``\\``) so a command spread across multiple
    lines is visible to a single substring search.
    """
    text = DOCKERFILE.read_text()
    # Collapse line continuations.
    collapsed = re.sub(r"\\\s*\n\s*", " ", text)
    install_lines = [
        line for line in collapsed.splitlines()
        if "pip install" in line and not line.lstrip().startswith("#")
    ]
    return "\n".join(install_lines)


@pytest.fixture(scope="module")
def install_block() -> str:
    block = _dockerfile_install_block()
    assert block, (
        "Couldn't find any 'pip install' line in Dockerfile.browser — "
        "test needs an update if the install pattern moved."
    )
    return block


class TestWebSocketSupport:
    """Regression for #815: WS upgrades must work."""

    def test_websocket_support_installed(self, browser_requirements: str):
        """uvicorn needs the ``[standard]`` extra (which pulls in
        ``websockets``) OR an explicit ``websockets`` install.

        Without one of these, ``uvicorn`` rejects every WS upgrade
        request with HTTP 404 at the protocol layer — there is no way
        for our route handler to run, no log, no clear failure mode.
        It just looks like ``/agent-vnc/{agent_id}/websockify`` doesn't
        exist. (See PR #815 for the production incident.)
        """
        has_uvicorn_standard = "uvicorn[standard]" in browser_requirements
        # Token-boundary match so ``websockets`` doesn't false-positive
        # on ``uvicorn[standard]``. Either an exact ``websockets``
        # standalone requirement or the ``[standard]`` extra is fine.
        has_explicit_websockets = bool(
            re.search(r"(^|\s)websockets(\s|>=|==|$)", browser_requirements, re.MULTILINE)
        )
        assert has_uvicorn_standard or has_explicit_websockets, (
            "The [browser] extra MUST require uvicorn[standard] OR an "
            "explicit websockets package. Without it, every WebSocket "
            "upgrade to /agent-vnc/{agent_id}/{path} returns HTTP 404 "
            "from uvicorn before any route handler runs.\n\n"
            f"Current browser extra:\n{browser_requirements}"
        )


class TestRequiredPackages:
    """The browser extra must include the packages the browser service
    imports at runtime. Catches a partial-rewrite regression class
    earlier than container-startup failure."""

    @pytest.mark.parametrize(
        "package",
        [
            "fastapi",       # Web framework
            "pydantic",      # Validation; agent <-> service contract
            "httpx",         # Mesh + browser-service HTTP clients
            "Pillow",        # WebP encoding for screenshots
            "camoufox",      # The whole point of the image
        ],
    )
    def test_package_in_browser_extra(self, package: str, browser_requirements: str):
        # Case-insensitive substring is enough — the list is short and
        # these tokens don't appear elsewhere in it.
        assert package.lower() in browser_requirements.lower(), (
            f"pyproject.toml's [browser] extra is missing required package "
            f"'{package}'. Current browser extra:\n{browser_requirements}"
        )

    def test_camoufox_is_exactly_pinned(self, browser_extra: list[str]):
        """Camoufox pins ``playwright="*"``, so an unpinned rebuild can
        drift onto a Playwright release that breaks the Juggler bridge
        (camoufox#617) and swaps the fingerprint DB. The ``==`` pin is
        load-bearing; bump it deliberately, not by resolver accident."""
        camoufox = [req for req in browser_extra if req.lower().startswith("camoufox")]
        assert camoufox, "No camoufox requirement in the [browser] extra."
        assert "==" in camoufox[0], (
            "camoufox must be pinned with '==' in the [browser] extra — an "
            "unpinned rebuild silently drifts Playwright and the fingerprint "
            f"DB. Got: {camoufox[0]!r}"
        )


class TestDockerfileUsesTheManifest:
    """The Dockerfile must install FROM the pyproject extra. If it ever
    grows its own literal package list again we are back to two manifests
    that drift apart silently — the exact failure this consolidation
    removed."""

    def test_dockerfile_copies_pyproject(self):
        text = DOCKERFILE.read_text()
        assert re.search(r"^COPY\s+pyproject\.toml", text, re.MULTILINE), (
            "Dockerfile.browser must COPY pyproject.toml before installing — "
            "the dependency list is read out of it at build time."
        )

    def test_dockerfile_installs_from_the_browser_extra(self, install_block: str):
        assert "optional-dependencies" in install_block and "browser" in install_block, (
            "Dockerfile.browser must install the packages declared in "
            "pyproject.toml's [project.optional-dependencies] browser extra.\n\n"
            f"Current install block:\n{install_block}"
        )

    def test_dockerfile_has_no_shadow_package_list(self, install_block: str):
        """No hand-written runtime package names in the pip commands."""
        shadowed = [
            name for name in ("camoufox", "fastapi", "uvicorn", "pydantic", "httpx", "pillow")
            if name in install_block.lower()
        ]
        assert not shadowed, (
            "Dockerfile.browser names runtime packages directly "
            f"({', '.join(shadowed)}) instead of installing the [browser] "
            "extra from pyproject.toml. That re-creates the second, "
            "invisible dependency manifest.\n\n"
            f"Current install block:\n{install_block}"
        )


class TestBuildTriggerCoverage:
    """The image-build workflow must actually run when the image changes.

    ``.github/workflows/browser-image.yml`` is path-filtered, and its
    filter used to watch only ``Dockerfile.browser``,
    ``docker/browser-entrypoint.sh`` and ``pyproject.toml``. But the
    Dockerfile also ``COPY``s ``src/browser/``, ``src/shared/``,
    ``src/__init__.py`` and the fontconfig alias file into the image —
    so a change to any of those produced a different image that the
    only automated build-and-smoke gate never rebuilt.

    These tests derive the expectation from the Dockerfile itself
    rather than restating a list, so adding a new ``COPY`` without
    widening the trigger fails here instead of silently shipping.
    """

    @staticmethod
    def _copy_sources() -> list[str]:
        """Return every build-context path ``Dockerfile.browser`` COPYs in."""
        sources: list[str] = []
        for raw in DOCKERFILE.read_text().splitlines():
            line = raw.strip()
            if not line.upper().startswith("COPY "):
                continue
            args = [a for a in line.split()[1:] if not a.startswith("--")]
            # Last arg is the destination inside the image.
            sources.extend(args[:-1])
        return sources

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        """Match one path against a GitHub ``paths:`` pattern.

        NOT ``fnmatch``: Python's ``*`` crosses ``/``, GitHub's does not.
        Using fnmatch directly would accept ``src/browser/*`` as covering
        ``src/browser/nested/new_file.py`` — the exact shallow-filter gap
        these tests exist to catch — so the check would pass while the
        build trigger stayed blind to new subpackages.
        """
        if pattern.endswith("/**"):
            prefix = pattern[: -len("/**")]
            return path == prefix or path.startswith(prefix + "/")
        # Translate the remaining wildcards segment-wise: ``**`` spans
        # separators, a single ``*`` (and ``?``) never does.
        parts = []
        i = 0
        while i < len(pattern):
            c = pattern[i]
            if pattern.startswith("**", i):
                parts.append(".*")
                i += 2
            elif c == "*":
                parts.append("[^/]*")
                i += 1
            elif c == "?":
                parts.append("[^/]")
                i += 1
            else:
                parts.append(re.escape(c))
                i += 1
        return re.fullmatch("".join(parts), path) is not None

    @pytest.fixture(scope="class")
    def triggers(self) -> dict[str, list[str]]:
        workflow = yaml.safe_load(WORKFLOW.read_text())
        # ``on`` is parsed as the boolean True by YAML 1.1.
        on = workflow.get("on", workflow.get(True))
        assert on, "browser-image.yml has no trigger block"
        return {event: cfg.get("paths", []) for event, cfg in on.items()}

    def test_push_and_pull_request_filters_agree(self, triggers: dict[str, list[str]]):
        """A filter that fires on one event but not the other means main
        and PRs are gated differently — the gap just moves rather than
        closing."""
        assert triggers["push"] == triggers["pull_request"], (
            "browser-image.yml's push and pull_request paths filters have "
            f"drifted:\npush: {triggers['push']}\npull_request: {triggers['pull_request']}"
        )

    def test_every_copied_path_triggers_a_rebuild(self, triggers: dict[str, list[str]]):
        """Everything baked into the image must be watched by the filter."""
        patterns = triggers["pull_request"]
        uncovered = []
        for source in self._copy_sources():
            cleaned = source.rstrip("/")
            if (REPO_ROOT / cleaned).is_dir():
                # A directory COPY bakes in the whole subtree, so a
                # shallow pattern (``src/browser/*.py``) is not enough.
                probe = f"{cleaned}/nested/new_file.py"
            else:
                probe = cleaned
            if not any(self._matches(p, probe) for p in patterns):
                uncovered.append(source)
        assert not uncovered, (
            "Dockerfile.browser COPYs these paths into the image, but "
            "browser-image.yml's paths filter does not watch them — a change "
            f"to any of them ships without a build-and-smoke run: {uncovered}\n"
            f"Current filter: {patterns}"
        )

    def test_build_definition_files_trigger_a_rebuild(self, triggers: dict[str, list[str]]):
        """The Dockerfile and the workflow itself are build inputs too."""
        patterns = triggers["pull_request"]
        for required in ("Dockerfile.browser", ".github/workflows/browser-image.yml"):
            assert any(self._matches(p, required) for p in patterns), (
                f"browser-image.yml's paths filter does not watch '{required}'. Current filter: {patterns}"
            )
