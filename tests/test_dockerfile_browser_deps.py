"""Static-check tests for ``Dockerfile.browser`` Python dependencies.

PR #815 shipped to production with a broken browser image because
``Dockerfile.browser`` installed plain ``uvicorn`` instead of
``uvicorn[standard]``. The ``[standard]`` extra is what pulls the
``websockets`` package — without it, uvicorn rejects every WebSocket
upgrade with HTTP 404 at the protocol layer, before any FastAPI route
handler runs. The per-agent VNC iframe needs WS upgrades to work, so
this regression silently broke the dashboard.

These tests are static checks against the Dockerfile content. They
don't build an image (too slow for unit-test cadence) — they just
confirm the dependency list contains what the application requires.
A full ``docker build`` smoke would close the gap completely; these
tests are the cheap first defense.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile.browser"


def _pip_install_block() -> str:
    """Return the concatenated text of every ``RUN pip install ...`` line.

    Joins continuation lines (``\\``) so packages spread across multiple
    lines are visible to a single substring search.
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
    block = _pip_install_block()
    assert block, (
        "Couldn't find any 'pip install' line in Dockerfile.browser — "
        "test needs an update if the install pattern moved."
    )
    return block


class TestWebSocketSupport:
    """Regression for #815: WS upgrades must work."""

    def test_websocket_support_installed(self, install_block: str):
        """uvicorn needs the ``[standard]`` extra (which pulls in
        ``websockets``) OR an explicit ``websockets`` install.

        Without one of these, ``uvicorn`` rejects every WS upgrade
        request with HTTP 404 at the protocol layer — there is no way
        for our route handler to run, no log, no clear failure mode.
        It just looks like ``/agent-vnc/{agent_id}/websockify`` doesn't
        exist. (See PR #815 for the production incident.)
        """
        has_uvicorn_standard = (
            '"uvicorn[standard]"' in install_block
            or "'uvicorn[standard]'" in install_block
            or "uvicorn[standard]" in install_block
        )
        # Token-boundary match so ``websockets`` doesn't false-positive
        # on ``uvicorn[standard]``-implied dependency mentions in
        # comments. Either an exact ``websockets`` standalone install
        # or the ``[standard]`` extra is acceptable.
        has_explicit_websockets = bool(
            re.search(r"(^|\s)['\"]?websockets['\"]?(\s|>=|==|$)", install_block)
        )
        assert has_uvicorn_standard or has_explicit_websockets, (
            "Dockerfile.browser MUST install uvicorn[standard] OR an "
            "explicit websockets package. Without it, every WebSocket "
            "upgrade to /agent-vnc/{agent_id}/{path} returns HTTP 404 "
            "from uvicorn before any route handler runs.\n\n"
            f"Current install block:\n{install_block}"
        )


class TestRequiredPackages:
    """The browser image's pip install must include the packages the
    application imports at runtime. Catches a partial-rewrite-of-Dockerfile
    regression class earlier than container-startup failure."""

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
    def test_package_in_install_block(self, package: str, install_block: str):
        # Case-insensitive substring is enough — the install line is
        # short and these tokens don't appear elsewhere.
        assert package.lower() in install_block.lower(), (
            f"Dockerfile.browser is missing required package '{package}'. "
            f"Current install block:\n{install_block}"
        )
