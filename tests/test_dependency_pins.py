"""Upper-bound guards on dependencies whose next major release breaks us.

The project deliberately uses ``>=`` bounds and ships no lock file (see
CLAUDE.md, "Stack"), so a rebuild always resolves to the newest
compatible release. That is fine for dependencies that keep their API —
and actively dangerous for one that doesn't: a container or CI rebuild
can pick up a breaking major with no commit on our side, and the failure
surfaces at runtime rather than at review time.

``mcp`` is that case today. The 2.0 SDK renamed or removed every symbol
the engine imports:

  - ``mcp.client.streamable_http.streamablehttp_client`` (used by
    ``src/host/mcp_gateway.py``) is now ``streamable_http_client``.
  - ``mcp.server.fastmcp`` — and ``FastMCP`` on ``mcp.server`` — used by
    ``tests/fixtures/echo_mcp_server.py``, is gone entirely
    (``mcp.server.mcpserver.MCPServer`` replaces it).

Both import sites catch ``ImportError`` and fail CLOSED (the gateway
raises ``GatewayUnavailable``; the agent client marks every configured
server ``failed``), so an unbounded resolve doesn't crash — it silently
turns every MCP path off. Hence the ceiling, and hence this test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:  # tomllib is stdlib on 3.11+; the project floor is 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only on Python 3.10
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    tomllib is None, reason="tomllib requires Python 3.11+ (CI runs 3.11/3.12)"
)


def _mcp_requirements() -> dict[str, Requirement]:
    """Every declared ``mcp`` requirement, keyed by where it's declared."""
    data = tomllib.loads(PYPROJECT.read_text())
    project = data["project"]
    found: dict[str, Requirement] = {}
    for spec in project["dependencies"]:
        req = Requirement(spec)
        if req.name == "mcp":
            found["dependencies"] = req
    for extra, specs in project.get("optional-dependencies", {}).items():
        for spec in specs:
            req = Requirement(spec)
            if req.name == "mcp":
                found[f"optional-dependencies.{extra}"] = req
    return found


def test_mcp_is_declared_in_core_dependencies():
    """The mesh gateway makes the SDK host-critical (it was promoted out
    of the [mcp] extra for exactly that reason)."""
    assert "dependencies" in _mcp_requirements(), (
        "mcp must stay a CORE dependency — src/host/mcp_gateway.py needs it "
        "on the mesh host, and install.sh never installs the [mcp] extra."
    )


# Probe points spanning the 2.x line. Testing only "2.0.0" would let a
# spec like ``!=2.0.0`` pass while still admitting 2.1+ — the ceiling has
# to exclude the whole major, not one release of it.
_2X_PROBES = ["2.0.0", "2.0.1", "2.5.0", "2.99.0"]


@pytest.mark.parametrize("where", ["dependencies", "optional-dependencies.mcp"])
@pytest.mark.parametrize("version", _2X_PROBES)
def test_mcp_excludes_2x(where: str, version: str):
    """Every declared mcp requirement must exclude the WHOLE 2.x line.

    Checked by evaluating the specifier rather than string-matching, so
    any equivalent spelling of the ceiling passes — but probed at several
    points across the major, so a spec that excludes only the first 2.x
    release doesn't slip through.
    """
    reqs = _mcp_requirements()
    assert where in reqs, f"No mcp requirement declared in [{where}]."
    specifier = reqs[where].specifier
    assert not specifier.contains(version), (
        f"mcp requirement in [{where}] is {str(specifier)!r}, which still "
        f"admits mcp {version}. The 2.x SDK renamed streamablehttp_client and "
        "removed mcp.server.fastmcp — both engine import sites fail closed, "
        "so an unbounded rebuild silently disables every MCP path. Add a "
        "'<2' ceiling, or lift it deliberately together with "
        "src/host/mcp_gateway.py, src/agent/mcp_client.py and "
        "tests/fixtures/echo_mcp_server.py."
    )


@pytest.mark.parametrize("where", ["dependencies", "optional-dependencies.mcp"])
def test_mcp_still_admits_the_1x_line(where: str):
    """The ceiling must not be so tight it excludes current 1.x releases —
    the floor is 1.9 (where streamablehttp_client stabilized)."""
    specifier = _mcp_requirements()[where].specifier
    assert specifier.contains("1.29.0"), (
        f"mcp requirement in [{where}] is {str(specifier)!r}, which excludes "
        "the current 1.x line. The supported range is >=1.9,<2."
    )


class TestAgentImageCarriesTheSameCeiling:
    """``Dockerfile.agent`` installs mcp DIRECTLY, not via pyproject.

    The agent image does not ``pip install`` the project — it names its
    runtime packages literally (litellm, pyyaml, click and docker are
    host-only, so installing the full core set there would be waste). That
    makes it a SECOND declaration site which no dependency tool can see and
    which the pyproject tests above do not cover: pinning pyproject alone
    would leave the agent image resolving mcp 2.x on its next rebuild, and
    ``src/agent/mcp_client.py`` fails closed — every configured server just
    goes "failed" with no crash to notice.
    """

    DOCKERFILE = REPO_ROOT / "Dockerfile.agent"

    def _mcp_requirement(self) -> Requirement:
        text = self.DOCKERFILE.read_text()
        # Collapse line continuations so a multi-line pip install is one line.
        collapsed = re.sub(r"\\\s*\n", " ", text)
        for raw in collapsed.splitlines():
            if "pip install" not in raw:
                continue
            # Requirement tokens are quoted when they carry a specifier.
            for token in re.findall(r"[\"']([^\"']+)[\"']|(\S+)", raw):
                spec = token[0] or token[1]
                try:
                    req = Requirement(spec)
                except Exception:
                    continue
                if req.name == "mcp":
                    return req
        raise AssertionError(
            "No mcp requirement found in any Dockerfile.agent pip install "
            "line. If the agent image stopped installing mcp directly, drop "
            "this test; if the install pattern moved, update it."
        )

    @pytest.mark.parametrize("version", _2X_PROBES)
    def test_agent_image_excludes_2x(self, version: str):
        req = self._mcp_requirement()
        assert not req.specifier.contains(version), (
            f"Dockerfile.agent installs {str(req)!r}, which still admits mcp "
            f"{version}. The agent image is a separate declaration site from "
            "pyproject.toml — pinning one does not pin the other."
        )

    def test_agent_image_still_admits_the_1x_line(self):
        req = self._mcp_requirement()
        assert req.specifier.contains("1.29.0"), (
            f"Dockerfile.agent installs {str(req)!r}, which excludes the "
            "current 1.x line. The supported range is >=1.9,<2."
        )
