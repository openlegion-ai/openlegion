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
