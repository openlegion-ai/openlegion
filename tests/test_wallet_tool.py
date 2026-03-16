"""Tests for agent wallet tools (wallet_tool.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agent.builtins.wallet_tool import (
    wallet_execute,
    wallet_get_address,
    wallet_get_balance,
    wallet_read_contract,
    wallet_transfer,
)


@pytest.fixture()
def mock_mesh():
    m = AsyncMock()
    m.wallet_get_address.return_value = {"address": "0xABC", "chain": "evm:base"}
    m.wallet_get_balance.return_value = {"balance": "1.5", "symbol": "ETH", "decimals": 18, "raw": "1500000000000000000"}
    m.wallet_read_contract.return_value = {"result": "0x01", "chain": "evm:base"}
    m.wallet_transfer.return_value = {"tx_hash": "0xDEAD", "chain": "evm:base", "status": "broadcast"}
    m.wallet_execute.return_value = {"tx_hash": "0xBEEF", "chain": "evm:base", "status": "broadcast"}
    return m


# ── wallet_get_address ──


@pytest.mark.asyncio
async def test_get_address_no_mesh():
    result = await wallet_get_address("evm:base", mesh_client=None)
    assert "error" in result


@pytest.mark.asyncio
async def test_get_address_no_chain():
    result = await wallet_get_address("", mesh_client=AsyncMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_get_address_success(mock_mesh):
    result = await wallet_get_address("evm:base", mesh_client=mock_mesh)
    assert result["address"] == "0xABC"
    mock_mesh.wallet_get_address.assert_awaited_once_with("evm:base")


@pytest.mark.asyncio
async def test_get_address_exception(mock_mesh):
    mock_mesh.wallet_get_address.side_effect = Exception("rpc down")
    result = await wallet_get_address("evm:base", mesh_client=mock_mesh)
    assert "error" in result
    assert "rpc down" in result["error"]


# ── wallet_get_balance ──


@pytest.mark.asyncio
async def test_get_balance_no_mesh():
    result = await wallet_get_balance("evm:base", mesh_client=None)
    assert "error" in result


@pytest.mark.asyncio
async def test_get_balance_success(mock_mesh):
    result = await wallet_get_balance("evm:base", token="native", mesh_client=mock_mesh)
    assert result["balance"] == "1.5"
    mock_mesh.wallet_get_balance.assert_awaited_once_with("evm:base", "native")


# ── wallet_read_contract ──


@pytest.mark.asyncio
async def test_read_contract_no_mesh():
    result = await wallet_read_contract("evm:base", "0x1", "balanceOf(address)", mesh_client=None)
    assert "error" in result


@pytest.mark.asyncio
async def test_read_contract_missing_params():
    result = await wallet_read_contract("evm:base", "", "balanceOf(address)", mesh_client=AsyncMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_read_contract_evm_requires_function():
    result = await wallet_read_contract("evm:base", "0x1", "", mesh_client=AsyncMock())
    assert "error" in result
    assert "function" in result["error"].lower()


@pytest.mark.asyncio
async def test_read_contract_solana_no_function_ok(mock_mesh):
    """Solana reads don't require a function signature."""
    result = await wallet_read_contract(
        "solana:mainnet", "SomeAccount", "", mesh_client=mock_mesh,
    )
    assert "error" not in result


@pytest.mark.asyncio
async def test_read_contract_success(mock_mesh):
    result = await wallet_read_contract(
        "evm:base", "0x1", "balanceOf(address)", args=["0x2"], mesh_client=mock_mesh,
    )
    assert result["result"] == "0x01"
    mock_mesh.wallet_read_contract.assert_awaited_once_with(
        "evm:base", "0x1", "balanceOf(address)", ["0x2"],
    )


# ── wallet_transfer ──


@pytest.mark.asyncio
async def test_transfer_no_mesh():
    result = await wallet_transfer("evm:base", "0x1", "0.1", mesh_client=None)
    assert "error" in result


@pytest.mark.asyncio
async def test_transfer_missing_params():
    result = await wallet_transfer("evm:base", "", "0.1", mesh_client=AsyncMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_transfer_success(mock_mesh):
    result = await wallet_transfer("evm:base", "0x1", "0.1", "native", mesh_client=mock_mesh)
    assert result["tx_hash"] == "0xDEAD"
    mock_mesh.wallet_transfer.assert_awaited_once_with("evm:base", "0x1", "0.1", "native")


@pytest.mark.asyncio
async def test_transfer_exception(mock_mesh):
    mock_mesh.wallet_transfer.side_effect = Exception("policy denied")
    result = await wallet_transfer("evm:base", "0x1", "0.1", mesh_client=mock_mesh)
    assert "error" in result
    assert "policy denied" in result["error"]


# ── wallet_execute ──


@pytest.mark.asyncio
async def test_execute_no_mesh():
    result = await wallet_execute("evm:base", mesh_client=None)
    assert "error" in result


@pytest.mark.asyncio
async def test_execute_no_chain():
    result = await wallet_execute("", mesh_client=AsyncMock())
    assert "error" in result


@pytest.mark.asyncio
async def test_execute_evm_requires_contract_and_function():
    result = await wallet_execute("evm:base", contract="", function="", mesh_client=AsyncMock())
    assert "error" in result
    assert "contract" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_solana_requires_transaction():
    result = await wallet_execute("solana:mainnet", mesh_client=AsyncMock())
    assert "error" in result
    assert "transaction" in result["error"].lower()


@pytest.mark.asyncio
async def test_execute_evm_success(mock_mesh):
    result = await wallet_execute(
        "evm:base",
        contract="0xRouter",
        function="swap(address,uint256)",
        args=["0x1", 100],
        value="0.1",
        mesh_client=mock_mesh,
    )
    assert result["tx_hash"] == "0xBEEF"
    mock_mesh.wallet_execute.assert_awaited_once_with(
        "evm:base", "0xRouter", "swap(address,uint256)", ["0x1", 100], "0.1", "",
    )


@pytest.mark.asyncio
async def test_execute_solana_success(mock_mesh):
    result = await wallet_execute(
        "solana:mainnet",
        transaction="base64encodedtx",
        mesh_client=mock_mesh,
    )
    assert result["tx_hash"] == "0xBEEF"
    mock_mesh.wallet_execute.assert_awaited_once_with(
        "solana:mainnet", "", "", [], "0", "base64encodedtx",
    )


@pytest.mark.asyncio
async def test_execute_exception(mock_mesh):
    mock_mesh.wallet_execute.side_effect = Exception("sim failed")
    result = await wallet_execute(
        "evm:base", contract="0x1", function="foo()", mesh_client=mock_mesh,
    )
    assert "error" in result
    assert "sim failed" in result["error"]
