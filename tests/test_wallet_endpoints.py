"""Tests for wallet mesh endpoints in server.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.host.server import create_mesh_app


@pytest.fixture()
def mock_wallet_service():
    ws = AsyncMock()
    ws.get_address = AsyncMock(return_value="0xABCDEF1234567890abcdef1234567890AbCdEf12")
    ws.get_balance = AsyncMock(return_value={
        "balance": "1.5", "symbol": "ETH", "decimals": 18, "raw": "1500000000000000000",
    })
    ws.read_contract = AsyncMock(return_value={"result": "0x01", "chain": "evm:base"})
    ws.transfer = AsyncMock(return_value={
        "tx_hash": "0xDEAD", "chain": "evm:base",
        "status": "broadcast", "explorer_url": "https://basescan.org/tx/0xDEAD",
    })
    ws.execute_contract = AsyncMock(return_value={
        "tx_hash": "0xBEEF", "chain": "evm:base",
        "status": "broadcast", "explorer_url": "https://basescan.org/tx/0xBEEF",
    })
    return ws


@pytest.fixture()
def mock_permissions():
    p = MagicMock()
    p.can_use_wallet.return_value = True
    p.can_use_wallet_chain.return_value = True
    p.can_access_wallet_contract.return_value = True
    p.get_permissions.return_value = MagicMock(
        can_use_wallet=True,
        wallet_allowed_chains=["*"],
    )
    return p


@pytest.fixture()
def mock_blackboard():
    bb = MagicMock()
    bb.remove_agent_watches = MagicMock()
    return bb


@pytest.fixture()
def client(mock_wallet_service, mock_permissions, mock_blackboard):
    """TestClient with wallet service configured and auth disabled (dev mode)."""
    app = create_mesh_app(
        blackboard=mock_blackboard,
        pubsub=MagicMock(),
        router=MagicMock(),
        permissions=mock_permissions,
        wallet_service=mock_wallet_service,
    )
    return TestClient(app)


@pytest.fixture()
def denied_client(mock_wallet_service, mock_blackboard):
    """TestClient where wallet permissions are denied."""
    denied_perms = MagicMock()
    denied_perms.can_use_wallet.return_value = False
    denied_perms.can_use_wallet_chain.return_value = False
    app = create_mesh_app(
        blackboard=mock_blackboard,
        pubsub=MagicMock(),
        router=MagicMock(),
        permissions=denied_perms,
        wallet_service=mock_wallet_service,
    )
    return TestClient(app)


@pytest.fixture()
def no_wallet_client(mock_permissions, mock_blackboard):
    """TestClient with no wallet service (not configured)."""
    app = create_mesh_app(
        blackboard=mock_blackboard,
        pubsub=MagicMock(),
        router=MagicMock(),
        permissions=mock_permissions,
        wallet_service=None,
    )
    return TestClient(app)


# ── Address endpoint ──────────────────────────────────────────


class TestWalletAddress:
    def test_success(self, client):
        resp = client.get("/mesh/wallet/address", params={"chain": "evm:base", "agent_id": "test"})
        assert resp.status_code == 200
        assert resp.json()["address"] == "0xABCDEF1234567890abcdef1234567890AbCdEf12"

    def test_permission_denied(self, denied_client):
        resp = denied_client.get("/mesh/wallet/address", params={"chain": "evm:base", "agent_id": "test"})
        assert resp.status_code == 403

    def test_not_configured(self, no_wallet_client):
        resp = no_wallet_client.get("/mesh/wallet/address", params={"chain": "evm:base", "agent_id": "test"})
        assert resp.status_code == 503

    def test_bad_chain(self, client, mock_wallet_service):
        mock_wallet_service.get_address.side_effect = ValueError("Unknown chain")
        resp = client.get("/mesh/wallet/address", params={"chain": "btc:main", "agent_id": "test"})
        assert resp.status_code == 400


# ── Balance endpoint ──────────────────────────────────────────


class TestWalletBalance:
    def test_success(self, client):
        resp = client.get("/mesh/wallet/balance", params={"chain": "evm:base", "agent_id": "test"})
        assert resp.status_code == 200
        assert resp.json()["balance"] == "1.5"

    def test_with_token(self, client):
        resp = client.get("/mesh/wallet/balance", params={
            "chain": "evm:base", "agent_id": "test", "token": "0xUSDC",
        })
        assert resp.status_code == 200


# ── Read endpoint ─────────────────────────────────────────────


class TestWalletRead:
    def test_success(self, client):
        resp = client.post("/mesh/wallet/read", json={
            "agent_id": "test", "chain": "evm:base",
            "contract": "0x1", "function": "balanceOf(address)", "args": ["0x2"],
        })
        assert resp.status_code == 200
        assert resp.json()["result"] == "0x01"

    def test_value_error(self, client, mock_wallet_service):
        mock_wallet_service.read_contract.side_effect = ValueError("bad function")
        resp = client.post("/mesh/wallet/read", json={
            "agent_id": "test", "chain": "evm:base",
            "contract": "0x1", "function": "bad",
        })
        assert resp.status_code == 400


# ── Transfer endpoint ─────────────────────────────────────────


class TestWalletTransfer:
    def test_success(self, client):
        resp = client.post("/mesh/wallet/transfer", json={
            "agent_id": "test", "chain": "evm:base",
            "to": "0x1234", "amount": "0.1", "token": "native",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["tx_hash"] == "0xDEAD"
        assert data["status"] == "broadcast"

    def test_value_error(self, client, mock_wallet_service):
        mock_wallet_service.transfer.side_effect = ValueError("bad address")
        resp = client.post("/mesh/wallet/transfer", json={
            "agent_id": "test", "chain": "evm:base",
            "to": "bad", "amount": "0.1",
        })
        assert resp.status_code == 400

    def test_permission_error(self, client, mock_wallet_service):
        mock_wallet_service.transfer.side_effect = PermissionError("daily limit")
        resp = client.post("/mesh/wallet/transfer", json={
            "agent_id": "test", "chain": "evm:base",
            "to": "0x1", "amount": "999",
        })
        assert resp.status_code == 403

    def test_permission_denied(self, denied_client):
        resp = denied_client.post("/mesh/wallet/transfer", json={
            "agent_id": "test", "chain": "evm:base",
            "to": "0x1", "amount": "0.1",
        })
        assert resp.status_code == 403


# ── Execute endpoint ──────────────────────────────────────────


class TestWalletExecute:
    def test_evm_success(self, client):
        resp = client.post("/mesh/wallet/execute", json={
            "agent_id": "test", "chain": "evm:base",
            "contract": "0xRouter", "function": "swap(address,uint256)",
            "args": ["0x1", 100], "value": "0.1",
        })
        assert resp.status_code == 200
        assert resp.json()["tx_hash"] == "0xBEEF"

    def test_solana_success(self, client):
        resp = client.post("/mesh/wallet/execute", json={
            "agent_id": "test", "chain": "solana:mainnet",
            "transaction": "base64encodedtx",
        })
        assert resp.status_code == 200

    def test_contract_denied(self, client, mock_permissions):
        mock_permissions.can_access_wallet_contract.return_value = False
        resp = client.post("/mesh/wallet/execute", json={
            "agent_id": "test", "chain": "evm:base",
            "contract": "0xEvil", "function": "drain()",
        })
        assert resp.status_code == 403

    def test_solana_skips_contract_check(self, client, mock_permissions):
        """Solana path sends empty contract — should not trigger contract check."""
        mock_permissions.can_access_wallet_contract.return_value = False
        resp = client.post("/mesh/wallet/execute", json={
            "agent_id": "test", "chain": "solana:mainnet",
            "transaction": "base64tx",
        })
        # Contract is empty string, so can_access_wallet_contract isn't called
        assert resp.status_code == 200

    def test_not_configured(self, no_wallet_client):
        resp = no_wallet_client.post("/mesh/wallet/execute", json={
            "agent_id": "test", "chain": "evm:base",
            "contract": "0x1", "function": "foo()",
        })
        assert resp.status_code == 503
