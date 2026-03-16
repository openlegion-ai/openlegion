"""Tests for WalletService (src/host/wallet.py)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Set a test mnemonic before importing WalletService
_TEST_MNEMONIC = (
    "abandon abandon abandon abandon abandon abandon abandon abandon "
    "abandon abandon abandon abandon abandon abandon abandon abandon "
    "abandon abandon abandon abandon abandon abandon abandon art"
)


@pytest.fixture()
def wallet_service(tmp_path, monkeypatch):
    """Create a WalletService with a test mnemonic."""
    monkeypatch.setenv("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", _TEST_MNEMONIC)
    from src.host.wallet import WalletService
    ws = WalletService(db_path=str(tmp_path / "wallet.db"))
    yield ws
    ws.close()


@pytest.fixture()
def unconfigured_service(tmp_path, monkeypatch):
    """WalletService with no master seed."""
    monkeypatch.delenv("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", raising=False)
    from src.host.wallet import WalletService
    ws = WalletService(db_path=str(tmp_path / "wallet.db"))
    yield ws
    ws.close()


# ── Configuration ─────────────────────────────────────────────


class TestConfiguration:
    def test_is_configured_true(self, wallet_service):
        assert wallet_service.is_configured is True

    def test_is_configured_false(self, unconfigured_service):
        assert unconfigured_service.is_configured is False

    def test_chains_loaded(self, wallet_service):
        assert "evm:ethereum" in wallet_service.chains
        assert "evm:base" in wallet_service.chains
        assert "solana:mainnet" in wallet_service.chains
        assert "solana:devnet" in wallet_service.chains

    def test_chain_rpc_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", _TEST_MNEMONIC)
        monkeypatch.setenv("OPENLEGION_SYSTEM_WALLET_RPC_BASE", "https://custom-rpc.com")
        from src.host.wallet import WalletService
        ws = WalletService(db_path=str(tmp_path / "wallet.db"))
        assert ws.chains["evm:base"]["rpc_url"] == "https://custom-rpc.com"
        ws.close()

    def test_chain_rpc_default_used(self, wallet_service):
        assert "publicnode" in wallet_service.chains["evm:ethereum"]["rpc_url"]


# ── Require configured ────────────────────────────────────────


class TestRequireConfigured:
    @pytest.mark.asyncio
    async def test_get_address_unconfigured(self, unconfigured_service):
        with pytest.raises(ValueError, match="wallet init"):
            await unconfigured_service.get_address("agent-1", "evm:ethereum")

    @pytest.mark.asyncio
    async def test_transfer_unconfigured(self, unconfigured_service):
        with pytest.raises(ValueError, match="wallet init"):
            await unconfigured_service.transfer("agent-1", "evm:base", "0x1", "0.1")


# ── Chain validation ──────────────────────────────────────────


class TestChainValidation:
    @pytest.mark.asyncio
    async def test_unknown_chain_rejected(self, wallet_service):
        with pytest.raises(ValueError, match="Unknown chain"):
            await wallet_service.get_address("agent-1", "btc:mainnet")

    @pytest.mark.asyncio
    async def test_error_lists_supported_chains(self, wallet_service):
        with pytest.raises(ValueError, match="evm:base"):
            await wallet_service.get_address("agent-1", "invalid")


# ── Key derivation ────────────────────────────────────────────


class TestKeyDerivation:
    def test_evm_deterministic(self, wallet_service):
        a1 = wallet_service._derive_evm_account(0)
        a2 = wallet_service._derive_evm_account(0)
        assert a1.address == a2.address

    def test_evm_different_index_different_address(self, wallet_service):
        a0 = wallet_service._derive_evm_account(0)
        a1 = wallet_service._derive_evm_account(1)
        assert a0.address != a1.address

    def test_evm_valid_address(self, wallet_service):
        a = wallet_service._derive_evm_account(0)
        assert a.address.startswith("0x")
        assert len(a.address) == 42

    def test_solana_deterministic(self, wallet_service):
        k1 = wallet_service._derive_solana_keypair(0)
        k2 = wallet_service._derive_solana_keypair(0)
        assert str(k1.pubkey()) == str(k2.pubkey())

    def test_solana_different_index_different_pubkey(self, wallet_service):
        k0 = wallet_service._derive_solana_keypair(0)
        k1 = wallet_service._derive_solana_keypair(1)
        assert str(k0.pubkey()) != str(k1.pubkey())

    def test_derive_account_routes(self, wallet_service):
        evm = wallet_service._derive_account(0, "evm")
        sol = wallet_service._derive_account(0, "solana")
        assert hasattr(evm, "address")  # eth_account.Account
        assert hasattr(sol, "pubkey")   # solders.Keypair

    def test_derive_account_unknown_ecosystem(self, wallet_service):
        with pytest.raises(ValueError, match="Unknown ecosystem"):
            wallet_service._derive_account(0, "cosmos")


# ── Index assignment ──────────────────────────────────────────


class TestIndexAssignment:
    def test_monotonic(self, wallet_service):
        i0 = wallet_service._get_or_assign_index("agent-a")
        i1 = wallet_service._get_or_assign_index("agent-b")
        assert i1 == i0 + 1

    def test_idempotent(self, wallet_service):
        i1 = wallet_service._get_or_assign_index("agent-x")
        i2 = wallet_service._get_or_assign_index("agent-x")
        assert i1 == i2

    def test_never_reused(self, wallet_service):
        wallet_service._get_or_assign_index("agent-1")
        wallet_service._get_or_assign_index("agent-2")
        i3 = wallet_service._get_or_assign_index("agent-3")
        assert i3 == 2  # 0, 1, 2


# ── Address derivation ────────────────────────────────────────


class TestGetAddress:
    @pytest.mark.asyncio
    async def test_evm_address(self, wallet_service):
        addr = await wallet_service.get_address("agent-1", "evm:ethereum")
        assert addr.startswith("0x")
        assert len(addr) == 42

    @pytest.mark.asyncio
    async def test_solana_address(self, wallet_service):
        addr = await wallet_service.get_address("agent-1", "solana:mainnet")
        assert len(addr) >= 32  # base58 pubkey

    @pytest.mark.asyncio
    async def test_same_agent_same_address(self, wallet_service):
        a1 = await wallet_service.get_address("agent-1", "evm:base")
        a2 = await wallet_service.get_address("agent-1", "evm:base")
        assert a1 == a2

    @pytest.mark.asyncio
    async def test_different_agents_different_addresses(self, wallet_service):
        a1 = await wallet_service.get_address("agent-1", "evm:base")
        a2 = await wallet_service.get_address("agent-2", "evm:base")
        assert a1 != a2


# ── Input validation ──────────────────────────────────────────


class TestValidation:
    def test_validate_evm_address_valid(self, wallet_service):
        result = wallet_service._validate_evm_address(
            "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        )
        assert result.startswith("0x")

    def test_validate_evm_address_invalid(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid EVM address"):
            wallet_service._validate_evm_address("not-an-address")

    def test_validate_evm_address_short(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid EVM address"):
            wallet_service._validate_evm_address("0x123")

    def test_validate_amount_valid(self, wallet_service):
        d = wallet_service._validate_amount("0.1")
        assert d == Decimal("0.1")

    def test_validate_amount_zero(self, wallet_service):
        with pytest.raises(ValueError, match="positive"):
            wallet_service._validate_amount("0")

    def test_validate_amount_negative(self, wallet_service):
        with pytest.raises(ValueError, match="positive"):
            wallet_service._validate_amount("-1")

    def test_validate_amount_non_numeric(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid amount"):
            wallet_service._validate_amount("abc")

    def test_validate_amount_excessive_decimals(self, wallet_service):
        with pytest.raises(ValueError, match="decimal places"):
            wallet_service._validate_amount("0." + "1" * 19)

    def test_validate_amount_infinity(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid amount"):
            wallet_service._validate_amount("Infinity")

    def test_validate_amount_nan(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid amount"):
            wallet_service._validate_amount("NaN")

    def test_validate_function_sig_valid(self, wallet_service):
        wallet_service._validate_function_signature("transfer(address,uint256)")

    def test_validate_function_sig_no_parens(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid function signature"):
            wallet_service._validate_function_signature("transfer")

    def test_validate_function_sig_empty(self, wallet_service):
        with pytest.raises(ValueError, match="Invalid function signature"):
            wallet_service._validate_function_signature("")


# ── ABI encoding ──────────────────────────────────────────────


class TestAbiEncoding:
    def test_split_simple(self, wallet_service):
        result = wallet_service._split_abi_params("address,uint256")
        assert result == ["address", "uint256"]

    def test_split_tuple(self, wallet_service):
        result = wallet_service._split_abi_params("(address,uint256),bool")
        assert result == ["(address,uint256)", "bool"]

    def test_split_nested(self, wallet_service):
        result = wallet_service._split_abi_params(
            "((address,uint256),address),uint256[]",
        )
        assert result == ["((address,uint256),address)", "uint256[]"]

    def test_split_empty(self, wallet_service):
        result = wallet_service._split_abi_params("")
        assert result == []

    def test_encode_no_args(self, wallet_service):
        data = wallet_service._encode_function_call("getReserves()", [])
        assert len(data) == 4  # just selector

    def test_encode_simple(self, wallet_service):
        data = wallet_service._encode_function_call(
            "transfer(address,uint256)",
            ["0x" + "00" * 19 + "01", 1000],
        )
        assert len(data) == 4 + 64  # selector + 2 * 32-byte words


# ── Policy engine ─────────────────────────────────────────────


class TestPolicyEngine:
    def test_allows_within_limits(self, wallet_service):
        result = wallet_service._check_policy("agent-1", 5.0)
        assert result is None

    def test_per_tx_exceeded(self, wallet_service):
        result = wallet_service._check_policy("agent-1", 15.0)
        assert result is not None
        assert "Per-transaction" in result

    def test_daily_exceeded(self, wallet_service):
        # Insert transactions dated TODAY to fill the daily budget.
        # Default daily limit is $100.  Insert 11 x $9.50 = $104.50.
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for _ in range(11):
            wallet_service.db.execute(
                "INSERT INTO transactions (agent_id, chain, to_address, value, "
                "value_usd, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("agent-daily", "evm:base", "0x1", "1", 9.50, "broadcast", today),
            )
        wallet_service.db.commit()
        # $104.50 already spent + $5.00 new = $109.50 > $100 daily limit
        result = wallet_service._check_policy("agent-daily", 5.0)
        assert result is not None
        assert "Daily limit" in result

    def test_rate_limit_exceeded(self, wallet_service):
        wallet_service._default_rate_per_hour = 3
        for _ in range(3):
            wallet_service._check_policy("agent-rate", 1.0)
        result = wallet_service._check_policy("agent-rate", 1.0)
        assert result is not None
        assert "Rate limit" in result

    def test_custom_permissions_limits(self, wallet_service):
        mock_perms = MagicMock()
        mock_perms.get_wallet_limits.return_value = (5.0, 50.0, 100)
        result = wallet_service._check_policy("agent-1", 6.0, mock_perms)
        assert result is not None
        assert "Per-transaction" in result


# ── Price feed ────────────────────────────────────────────────


class TestPriceFeed:
    @pytest.mark.asyncio
    async def test_estimate_zero_value(self, wallet_service):
        result = await wallet_service._estimate_value_usd("evm:base", 0)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_fallback_price(self, wallet_service):
        price = await wallet_service._get_price("ETH", "ethereum")
        # Should return fallback since we're not hitting real CoinGecko in tests
        assert price > 0

    @pytest.mark.asyncio
    async def test_cache_hit(self, wallet_service):
        import time
        wallet_service._price_cache["ETH"] = (1234.0, time.time())
        price = await wallet_service._get_price("ETH", "ethereum")
        assert price == 1234.0


# ── Explorer URL ──────────────────────────────────────────────


class TestExplorerUrl:
    def test_evm_explorer_format(self, wallet_service):
        fmt = wallet_service.chains["evm:base"]["explorer_tx_fmt"]
        url = fmt.format(tx_hash="0xDEAD")
        assert url == "https://basescan.org/tx/0xDEAD"

    def test_solana_devnet_format(self, wallet_service):
        fmt = wallet_service.chains["solana:devnet"]["explorer_tx_fmt"]
        url = fmt.format(tx_hash="5abc")
        assert url == "https://solscan.io/tx/5abc?cluster=devnet"


# ── Audit log ─────────────────────────────────────────────────


class TestAuditLog:
    def test_audit_writes_to_db(self, wallet_service):
        wallet_service._audit(
            "agent-1", "evm:base", "broadcast",
            tx_hash="0xABC", to_address="0x1", value="0.1",
        )
        row = wallet_service.db.execute(
            "SELECT agent_id, chain, tx_hash, status FROM transactions",
        ).fetchone()
        assert row == ("agent-1", "evm:base", "0xABC", "broadcast")

    def test_audit_rejected(self, wallet_service):
        wallet_service._audit(
            "agent-1", "evm:base", "rejected",
            error="daily limit",
        )
        row = wallet_service.db.execute(
            "SELECT status, error FROM transactions",
        ).fetchone()
        assert row == ("rejected", "daily limit")


# ── Close ─────────────────────────────────────────────────────


class TestClose:
    def test_close(self, wallet_service):
        wallet_service.close()
        # DB should be closed — operations raise
        with pytest.raises(Exception):
            wallet_service.db.execute("SELECT 1")
