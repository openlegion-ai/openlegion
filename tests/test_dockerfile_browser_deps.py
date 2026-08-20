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
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:  # tomllib is stdlib on 3.11+; the project floor is 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.browser"
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
