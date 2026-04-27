"""Tests for wallet permission checks in PermissionMatrix."""

from __future__ import annotations

import json

import pytest

from src.host.permissions import PermissionMatrix


@pytest.fixture()
def matrix(tmp_path):
    """PermissionMatrix loaded from a temp config with wallet permissions."""
    cfg = {
        "permissions": {
            "default": {
                "can_use_wallet": False,
            },
            "trader": {
                "can_use_wallet": True,
                "wallet_allowed_chains": ["evm:base", "solana:mainnet"],
                "wallet_spend_limit_per_tx_usd": 50,
                "wallet_spend_limit_daily_usd": 500,
                "wallet_rate_limit_per_hour": 30,
                "wallet_allowed_contracts": [],
            },
            "restricted": {
                "can_use_wallet": True,
                "wallet_allowed_chains": ["evm:sepolia"],
                "wallet_allowed_contracts": ["0xABCD1234567890abcdef1234567890ABcDeF1234"],
            },
            "wildcard": {
                "can_use_wallet": True,
                "wallet_allowed_chains": ["*"],
            },
        },
    }
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps(cfg))
    return PermissionMatrix(config_path=str(path))


class TestCanUseWallet:
    def test_default_deny(self, matrix):
        assert matrix.can_use_wallet("unknown-agent") is False

    def test_trader_allowed(self, matrix):
        assert matrix.can_use_wallet("trader") is True

    def test_trusted_always_allowed(self, matrix):
        assert matrix.can_use_wallet("mesh") is True


class TestCanUseWalletChain:
    def test_requires_wallet_enabled(self, matrix):
        # default has can_use_wallet=False
        assert matrix.can_use_wallet_chain("unknown-agent", "evm:base") is False

    def test_allowed_chain(self, matrix):
        assert matrix.can_use_wallet_chain("trader", "evm:base") is True
        assert matrix.can_use_wallet_chain("trader", "solana:mainnet") is True

    def test_disallowed_chain(self, matrix):
        assert matrix.can_use_wallet_chain("trader", "evm:ethereum") is False

    def test_wildcard(self, matrix):
        assert matrix.can_use_wallet_chain("wildcard", "evm:anything") is True
        assert matrix.can_use_wallet_chain("wildcard", "solana:devnet") is True

    def test_trusted_bypasses(self, matrix):
        assert matrix.can_use_wallet_chain("mesh", "evm:whatever") is True


class TestGetWalletLimits:
    def test_custom_limits(self, matrix):
        per_tx, daily, rate = matrix.get_wallet_limits("trader")
        assert per_tx == 50
        assert daily == 500
        assert rate == 30

    def test_zero_means_default(self, matrix):
        per_tx, daily, rate = matrix.get_wallet_limits("wildcard")
        assert per_tx == 0
        assert daily == 0
        assert rate == 0


class TestCanAccessWalletContract:
    def test_empty_allows_all(self, matrix):
        assert matrix.can_access_wallet_contract("trader", "0xAnything") is True

    def test_populated_rejects_unlisted(self, matrix):
        assert matrix.can_access_wallet_contract("restricted", "0xOther") is False

    def test_populated_allows_listed(self, matrix):
        assert matrix.can_access_wallet_contract(
            "restricted",
            "0xABCD1234567890abcdef1234567890ABcDeF1234",
        ) is True

    def test_case_insensitive(self, matrix):
        assert matrix.can_access_wallet_contract(
            "restricted",
            "0xabcd1234567890abcdef1234567890abcdef1234",
        ) is True

    def test_trusted_bypasses(self, matrix):
        assert matrix.can_access_wallet_contract("mesh", "0xAnything") is True


class TestDefaultTemplatePropagation:
    def test_wallet_fields_propagated(self, matrix):
        """Default template fields should propagate to unknown agents."""
        perms = matrix.get_permissions("new-agent")
        assert perms.can_use_wallet is False
        assert perms.wallet_allowed_chains == []
        assert perms.wallet_spend_limit_per_tx_usd == 0.0
        assert perms.wallet_spend_limit_daily_usd == 0.0
        assert perms.wallet_rate_limit_per_hour == 0
        assert perms.wallet_allowed_contracts == []


# ── Per-browser-action permissions (Phase 1.1) ─────────────────────────────


@pytest.fixture()
def browser_matrix(tmp_path):
    """PermissionMatrix with a range of browser_actions configurations."""
    cfg = {
        "permissions": {
            "default": {
                "can_use_browser": False,
            },
            # Legacy-style agent: `browser_actions` unset → None → legacy set
            "legacy-agent": {
                "can_use_browser": True,
            },
            # Explicit wildcard: every known + future action allowed
            "wildcard-agent": {
                "can_use_browser": True,
                "browser_actions": ["*"],
            },
            # Specific allowlist only
            "readonly-agent": {
                "can_use_browser": True,
                "browser_actions": ["navigate", "snapshot", "screenshot"],
            },
            # Explicit empty list: equivalent to can_use_browser=False
            "empty-list-agent": {
                "can_use_browser": True,
                "browser_actions": [],
            },
            # can_use_browser=False but browser_actions populated — must deny
            "no-browser-agent": {
                "can_use_browser": False,
                "browser_actions": ["*"],
            },
            # Agent with a future action opt-in (simulating post-upgrade grant)
            "uploader-agent": {
                "can_use_browser": True,
                "browser_actions": [
                    "navigate", "click", "type", "upload_file",
                ],
            },
        },
    }
    path = tmp_path / "permissions.json"
    path.write_text(json.dumps(cfg))
    return PermissionMatrix(config_path=str(path))


class TestCanBrowserAction:
    def test_default_grants_all_known_actions(self, browser_matrix):
        """browser_actions=None (default) → all known actions allowed."""
        for action in ("navigate", "click", "type", "screenshot", "scroll",
                       "focus", "status", "detect_captcha", "upload_file",
                       "download", "find_text", "open_tab",
                       "fill_form", "solve_captcha"):
            assert browser_matrix.can_browser_action("legacy-agent", action) is True, action

    def test_default_also_allows_future_actions(self, browser_matrix):
        """The default None is an unbounded allow — future actions pass too.

        The mesh validates unknown action names as 400 input errors before
        calling can_browser_action; this method is permission-only.
        """
        for action in ("upload_file", "download", "brand_new_future_action"):
            assert browser_matrix.can_browser_action("legacy-agent", action) is True, action

    def test_wildcard_equivalent_to_default(self, browser_matrix):
        """Explicit ['*'] is equivalent to None for permission purposes."""
        for action in ("navigate", "upload_file", "brand_new_action"):
            assert browser_matrix.can_browser_action("wildcard-agent", action) is True

    def test_specific_list_is_opt_out_restriction(self, browser_matrix):
        """Operators can narrow to a specific list — opt-out default-allow."""
        assert browser_matrix.can_browser_action("readonly-agent", "navigate") is True
        assert browser_matrix.can_browser_action("readonly-agent", "snapshot") is True
        assert browser_matrix.can_browser_action("readonly-agent", "click") is False
        assert browser_matrix.can_browser_action("readonly-agent", "upload_file") is False

    def test_empty_list_denies_all(self, browser_matrix):
        for action in ("navigate", "click", "upload_file"):
            assert browser_matrix.can_browser_action("empty-list-agent", action) is False

    def test_can_use_browser_false_overrides(self, browser_matrix):
        """Even with browser_actions=['*'], can_use_browser=False must deny."""
        for action in ("navigate", "click", "upload_file"):
            assert browser_matrix.can_browser_action("no-browser-agent", action) is False

    def test_curated_list_honored_even_for_legacy_actions(self, browser_matrix):
        """Explicit curation is respected — listed actions only, nothing else."""
        assert browser_matrix.can_browser_action("uploader-agent", "upload_file") is True
        assert browser_matrix.can_browser_action("uploader-agent", "navigate") is True
        # Not granted explicitly → denied (curated = strict opt-out default)
        assert browser_matrix.can_browser_action("uploader-agent", "download") is False
        # Even legacy action is denied when caller specified a curated list
        assert browser_matrix.can_browser_action("uploader-agent", "screenshot") is False

    def test_trusted_always_allowed(self, browser_matrix):
        """The 'mesh' internal identity bypasses all checks."""
        for action in ("navigate", "upload_file", "brand_new_action"):
            assert browser_matrix.can_browser_action("mesh", action) is True

    def test_unknown_agent_falls_through_to_default(self, browser_matrix):
        """Default here is can_use_browser=False → deny."""
        assert browser_matrix.can_browser_action("ghost-agent", "navigate") is False

    def test_can_use_browser_unchanged_for_legacy_callers(self, browser_matrix):
        """The old binary check still works for callers that haven't migrated."""
        assert browser_matrix.can_use_browser("legacy-agent") is True
        assert browser_matrix.can_use_browser("no-browser-agent") is False
        assert browser_matrix.can_use_browser("empty-list-agent") is True  # flag itself is True


class TestKnownBrowserActionsSet:
    """KNOWN_BROWSER_ACTIONS is the mesh-side input validation set.
    Catches typo'd action names with a clean 400 before reaching the
    browser service. Does NOT gate permissions.

    Regression-proof against accidental REMOVAL — every action the mesh
    is expected to recognize must appear here.
    """

    def test_legacy_actions_present(self):
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        legacy = frozenset({
            "navigate", "snapshot", "click", "type", "hover",
            "screenshot", "reset", "focus", "status", "detect_captcha",
            "scroll", "wait_for", "press_key", "go_back", "go_forward",
            "switch_tab",
        })
        assert legacy.issubset(KNOWN_BROWSER_ACTIONS), (
            "Legacy actions disappeared from KNOWN_BROWSER_ACTIONS — this "
            "would silently break every pre-upgrade agent."
        )

    def test_phase_1_5_file_transfer_actions_reserved(self):
        """upload_file + download must be recognized by the mesh input
        validator even before Phase 1.5 endpoints exist, so the PRs can
        merge in any order without cross-branch action-name coordination.
        """
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "upload_file" in KNOWN_BROWSER_ACTIONS
        assert "download" in KNOWN_BROWSER_ACTIONS

    def test_phase_5_find_text_and_open_tab_present(self):
        """Phase 5 §8.5 / §8.6 default-allow actions registered with the
        mesh input validator."""
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "find_text" in KNOWN_BROWSER_ACTIONS
        assert "open_tab" in KNOWN_BROWSER_ACTIONS

    def test_phase_8_captcha_actions_present(self):
        """Phase 8 §11.14 explicit-trigger captcha skills must be in
        KNOWN_BROWSER_ACTIONS so the mesh accepts ``solve_captcha`` and
        ``request_captcha_help`` routed via /mesh/browser/command."""
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "solve_captcha" in KNOWN_BROWSER_ACTIONS
        assert "request_captcha_help" in KNOWN_BROWSER_ACTIONS

    def test_legacy_alias_still_exported(self):
        """Back-compat alias for callers that imported the old name."""
        from src.host.permissions import (
            KNOWN_BROWSER_ACTIONS,
            LEGACY_BROWSER_ACTIONS,
        )
        assert LEGACY_BROWSER_ACTIONS is KNOWN_BROWSER_ACTIONS


class TestDefaultFallbackPropagatesBrowserActions:
    """The 'default' template's browser_actions must flow into fallback perms
    for unknown agents; otherwise per-action grants on 'default' silently lose."""

    def test_default_browser_actions_curation_propagates(self, tmp_path):
        """When 'default' template curates a specific list, unknown agents
        inherit that curation — the restriction flows through the fallback."""
        cfg = {
            "permissions": {
                "default": {
                    "can_use_browser": True,
                    "browser_actions": ["navigate", "snapshot"],
                },
            },
        }
        path = tmp_path / "permissions.json"
        path.write_text(json.dumps(cfg))
        matrix = PermissionMatrix(config_path=str(path))
        # Unknown agent falls through to default
        assert matrix.can_browser_action("unknown", "navigate") is True
        assert matrix.can_browser_action("unknown", "click") is False

    def test_default_unspecified_inherits_default_allow(self, tmp_path):
        """When 'default' template has can_use_browser=True but no
        browser_actions key, the full default-allow semantic propagates:
        unknown agents get ALL known actions."""
        cfg = {
            "permissions": {
                "default": {"can_use_browser": True},
            },
        }
        path = tmp_path / "permissions.json"
        path.write_text(json.dumps(cfg))
        matrix = PermissionMatrix(config_path=str(path))
        for action in ("navigate", "upload_file", "download", "solve_captcha"):
            assert matrix.can_browser_action("unknown", action) is True, action


# ── Endpoint-level gates on the dedicated handoff routes ───────────────────


def _build_handoff_app(tmp_path, *, perms_map):
    """Mesh app fixture for the two dedicated handoff endpoints.

    Mirrors :func:`tests.test_browser_delegation._build_app` but lives here
    so the permission tests stay co-located with the rest of the matrix
    coverage. Both endpoints (``/mesh/browser-login-request`` and
    ``/mesh/browser-captcha-help-request``) are gated by
    ``can_browser_action`` post-PR, so we exercise that gate directly
    here — the ``browser_command`` path test in ``test_browser_delegation``
    catches the *other* bypass surface but not these two dedicated routes.
    """
    from unittest.mock import MagicMock

    from src.host.costs import CostTracker
    from src.host.mesh import Blackboard, MessageRouter, PubSub
    from src.host.server import create_mesh_app
    from src.host.traces import TraceStore
    from src.shared.types import AgentPermissions

    blackboard = Blackboard(str(tmp_path / "bb.db"))
    pubsub = PubSub()
    matrix = PermissionMatrix()
    for aid, perms in perms_map.items():
        matrix.permissions[aid] = AgentPermissions(agent_id=aid, **perms)
    router = MessageRouter(matrix, {})
    costs = CostTracker(str(tmp_path / "costs.db"))
    traces = TraceStore(str(tmp_path / "traces.db"))

    container_manager = MagicMock()
    container_manager.browser_service_url = "http://browser-svc:8500"
    container_manager.browser_auth_token = ""

    event_bus = MagicMock()
    app = create_mesh_app(
        blackboard=blackboard,
        pubsub=pubsub,
        router=router,
        permissions=matrix,
        cost_tracker=costs,
        trace_store=traces,
        event_bus=event_bus,
        container_manager=container_manager,
    )
    return app, event_bus


class TestBrowserCaptchaHelpRequestPermissionGate:
    """``/mesh/browser-captcha-help-request`` enforces
    ``can_browser_action(agent_id, "request_captcha_help")``.

    Pre-PR (Phase 8 §11.14 in PR #769) the dedicated endpoint emitted
    the dashboard handoff card without consulting the per-action gate.
    An operator who narrowed a template to ``browser_actions=["navigate"]``
    saw the FIRST call (``browser_command`` → ``request_captcha_help``)
    rejected, but the SECOND call to this endpoint succeeded — defeating
    permission narrowing.  These tests pin that the gate now fires here.
    """

    @pytest.mark.asyncio
    async def test_narrowed_browser_actions_rejected_403(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "narrow-agent": {
                    "can_use_browser": True,
                    "browser_actions": ["navigate"],
                },
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-captcha-help-request",
                json={
                    "agent_id": "narrow-agent",
                    "service": "Cloudflare",
                    "description": "Help me solve this CAPTCHA.",
                },
                headers={"X-Agent-ID": "narrow-agent"},
            )
        assert resp.status_code == 403, resp.text
        assert "request_captcha_help" in resp.text
        # Critical: the dashboard event was NOT emitted.
        event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_allow_browser_actions_succeeds(self, tmp_path):
        """``browser_actions=None`` → default-allow → endpoint succeeds."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "open-agent": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-captcha-help-request",
                json={
                    "agent_id": "open-agent",
                    "service": "Cloudflare",
                    "description": "Help me solve this CAPTCHA.",
                },
                headers={"X-Agent-ID": "open-agent"},
            )
        assert resp.status_code == 200, resp.text
        event_bus.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_explicit_grant_succeeds(self, tmp_path):
        """``browser_actions=["request_captcha_help"]`` (explicitly
        granted) → endpoint succeeds even though everything else is
        blocked."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "captcha-only-agent": {
                    "can_use_browser": True,
                    "browser_actions": ["request_captcha_help"],
                },
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-captcha-help-request",
                json={
                    "agent_id": "captcha-only-agent",
                    "service": "Cloudflare",
                    "description": "Help me.",
                },
                headers={"X-Agent-ID": "captcha-only-agent"},
            )
        assert resp.status_code == 200, resp.text
        event_bus.emit.assert_called_once()


class TestBrowserLoginRequestPermissionGate:
    """``/mesh/browser-login-request`` enforces
    ``can_browser_action(agent_id, "request_browser_login")``.

    Same bypass surface as the captcha-help endpoint.  Pre-PR, an
    operator who restricted the agent's ``browser_actions`` could not
    forbid this endpoint because the action name wasn't in
    ``KNOWN_BROWSER_ACTIONS`` and the route did not consult the gate.
    """

    @pytest.mark.asyncio
    async def test_narrowed_browser_actions_rejected_403(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "narrow-agent": {
                    "can_use_browser": True,
                    "browser_actions": ["navigate"],
                },
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "narrow-agent",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log me in.",
                },
                headers={"X-Agent-ID": "narrow-agent"},
            )
        assert resp.status_code == 403, resp.text
        assert "request_browser_login" in resp.text
        event_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_allow_browser_actions_succeeds(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "open-agent": {"can_use_browser": True},
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            resp = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "open-agent",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log me in.",
                },
                headers={"X-Agent-ID": "open-agent"},
            )
        assert resp.status_code == 200, resp.text
        event_bus.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_captcha_only_grant_blocks_login(self, tmp_path):
        """Mirrors the spec scenario: an agent granted ONLY
        ``request_captcha_help`` can use the captcha endpoint but the
        login endpoint must 403 — separate per-action grants."""
        from httpx import ASGITransport, AsyncClient

        app, event_bus = _build_handoff_app(
            tmp_path,
            perms_map={
                "captcha-only-agent": {
                    "can_use_browser": True,
                    "browser_actions": ["request_captcha_help"],
                },
            },
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            # Captcha succeeds.
            resp_captcha = await client.post(
                "/mesh/browser-captcha-help-request",
                json={
                    "agent_id": "captcha-only-agent",
                    "service": "Cloudflare",
                    "description": "Help me.",
                },
                headers={"X-Agent-ID": "captcha-only-agent"},
            )
            assert resp_captcha.status_code == 200, resp_captcha.text
            # Login 403s.
            resp_login = await client.post(
                "/mesh/browser-login-request",
                json={
                    "agent_id": "captcha-only-agent",
                    "url": "https://x.com/login",
                    "service": "X",
                    "description": "Log me in.",
                },
                headers={"X-Agent-ID": "captcha-only-agent"},
            )
            assert resp_login.status_code == 403, resp_login.text
            assert "request_browser_login" in resp_login.text


class TestRequestBrowserLoginInKnownActions:
    """``request_browser_login`` must be in ``KNOWN_BROWSER_ACTIONS`` so
    operators can grant it explicitly via ``browser_actions=[...]`` and
    the mesh-side input validator on ``/mesh/browser/command`` accepts
    it as a known action name (although the dedicated endpoint is the
    primary call site)."""

    def test_present(self):
        from src.host.permissions import KNOWN_BROWSER_ACTIONS
        assert "request_browser_login" in KNOWN_BROWSER_ACTIONS
