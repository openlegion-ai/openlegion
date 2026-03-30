"""Wallet signing service — extends the vault with onchain transaction signing.

Same trust model as CredentialVault: master seed is a system-tier credential
loaded from .env.  Agents request transactions through the mesh.  This service
derives the per-agent key, enforces policy, signs, and broadcasts.  Only the
transaction hash is returned — never keys or raw signatures.

Key hierarchy:
  OPENLEGION_SYSTEM_WALLET_MASTER_SEED (BIP-39 mnemonic in .env)
    ├─ m/44'/60'/0'/0/{agent_index}    (EVM — all EVM chains share one address)
    └─ m/44'/501'/{agent_index}'/0'    (Solana — HMAC-derived Ed25519 seed)
"""

from __future__ import annotations

import base64
import hashlib
import hmac as _hmac
import os
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

import httpx

from src.shared.utils import setup_logging

if TYPE_CHECKING:
    from src.dashboard.events import EventBus
    from src.host.permissions import PermissionMatrix

logger = setup_logging("host.wallet")

# ── Chain registry ────────────────────────────────────────────

_CHAINS: dict[str, dict[str, Any]] = {
    # ── EVM ──
    "evm:ethereum": {
        "ecosystem": "evm",
        "chain_id": 1,
        "rpc_default": "https://ethereum.publicnode.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_ETHEREUM",
        "explorer_tx_fmt": "https://etherscan.io/tx/{tx_hash}",
        "symbol": "ETH",
        "decimals": 18,
        "coingecko_id": "ethereum",
    },
    "evm:base": {
        "ecosystem": "evm",
        "chain_id": 8453,
        "rpc_default": "https://mainnet.base.org",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_BASE",
        "explorer_tx_fmt": "https://basescan.org/tx/{tx_hash}",
        "symbol": "ETH",
        "decimals": 18,
        "coingecko_id": "ethereum",
    },
    "evm:arbitrum": {
        "ecosystem": "evm",
        "chain_id": 42161,
        "rpc_default": "https://arb1.arbitrum.io/rpc",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_ARBITRUM",
        "explorer_tx_fmt": "https://arbiscan.io/tx/{tx_hash}",
        "symbol": "ETH",
        "decimals": 18,
        "coingecko_id": "ethereum",
    },
    "evm:polygon": {
        "ecosystem": "evm",
        "chain_id": 137,
        "rpc_default": "https://polygon-bor-rpc.publicnode.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_POLYGON",
        "explorer_tx_fmt": "https://polygonscan.com/tx/{tx_hash}",
        "symbol": "POL",
        "decimals": 18,
        "coingecko_id": "polygon-ecosystem-token",
    },
    "evm:sepolia": {
        "ecosystem": "evm",
        "chain_id": 11155111,
        "rpc_default": "https://ethereum-sepolia.publicnode.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_SEPOLIA",
        "explorer_tx_fmt": "https://sepolia.etherscan.io/tx/{tx_hash}",
        "symbol": "ETH",
        "decimals": 18,
        "coingecko_id": "ethereum",
    },
    # ── Solana ──
    "solana:mainnet": {
        "ecosystem": "solana",
        "rpc_default": "https://api.mainnet.solana.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_SOLANA",
        "explorer_tx_fmt": "https://solscan.io/tx/{tx_hash}",
        "symbol": "SOL",
        "decimals": 9,
        "coingecko_id": "solana",
    },
    "solana:devnet": {
        "ecosystem": "solana",
        "rpc_default": "https://api.devnet.solana.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_SOLANA_DEVNET",
        "explorer_tx_fmt": "https://solscan.io/tx/{tx_hash}?cluster=devnet",
        "symbol": "SOL",
        "decimals": 9,
        "coingecko_id": "solana",
    },
}

# ── Price fallbacks ───────────────────────────────────────────

_FALLBACK_PRICES: dict[str, float] = {"ETH": 3000.0, "POL": 0.10, "SOL": 150.0}
_PRICE_CACHE_TTL = 300  # 5 minutes


# ── WalletService ─────────────────────────────────────────────


class WalletService:
    """Secure transaction signing for agents.

    Private keys NEVER leave this class.  All public methods return only
    transaction hashes, addresses, or balances — never key material.
    """

    def __init__(
        self,
        db_path: str = "data/wallet.db",
        event_bus: EventBus | None = None,
    ) -> None:
        from pathlib import Path

        self._master_seed: str | None = os.environ.get(
            "OPENLEGION_SYSTEM_WALLET_MASTER_SEED",
        )
        self._chains = self._load_chains()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

        # Policy defaults (overridable via env vars)
        self._default_per_tx_usd = float(
            os.environ.get("OPENLEGION_WALLET_LIMIT_PER_TX_USD", "10.0"),
        )
        self._default_daily_usd = float(
            os.environ.get("OPENLEGION_WALLET_LIMIT_DAILY_USD", "100.0"),
        )
        self._default_rate_per_hour = int(
            os.environ.get("OPENLEGION_WALLET_RATE_LIMIT_PER_HOUR", "10"),
        )

        # In-memory caches
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._tx_timestamps: dict[str, list[float]] = {}

        # Lazy-init RPC providers (one per chain)
        self._evm_providers: dict[str, Any] = {}
        self._solana_clients: dict[str, Any] = {}

        self._event_bus = event_bus

    # ── Properties ────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return self._master_seed is not None

    @property
    def chains(self) -> dict[str, dict[str, Any]]:
        return self._chains

    # ── Lifecycle ─────────────────────────────────────────────

    def close(self) -> None:
        """Shutdown: release RPC connections and database."""
        self._evm_providers.clear()
        self._solana_clients.clear()
        self.db.close()

    def cleanup_agent(self, agent_id: str) -> int:
        """Delete wallet records for an agent. Returns transaction rows deleted.

        Does NOT reuse the key derivation index — that would create a
        different agent with the same wallet key (security issue).
        """
        cursor = self.db.execute(
            "DELETE FROM transactions WHERE agent_id = ?", (agent_id,),
        )
        deleted = cursor.rowcount
        self.db.execute("DELETE FROM agent_index WHERE agent_id = ?", (agent_id,))
        self.db.commit()
        return deleted

    # ── Schema ────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_index (
                agent_id TEXT PRIMARY KEY,
                idx INTEGER UNIQUE NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS index_counter (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_idx INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                chain TEXT NOT NULL,
                tx_hash TEXT,
                to_address TEXT NOT NULL,
                value TEXT NOT NULL,
                token TEXT DEFAULT 'native',
                value_usd REAL DEFAULT 0.0,
                function TEXT,
                status TEXT NOT NULL,
                error TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tx_agent_ts
                ON transactions(agent_id, timestamp);
        """)
        self.db.commit()

    # ── Chain loading ─────────────────────────────────────────

    @staticmethod
    def _load_chains() -> dict[str, dict[str, Any]]:
        """Return chain configs with RPC URLs resolved (env override → default)."""
        loaded: dict[str, dict[str, Any]] = {}
        for chain_id, cfg in _CHAINS.items():
            entry = dict(cfg)
            entry["rpc_url"] = os.environ.get(cfg["rpc_env"], cfg["rpc_default"])
            loaded[chain_id] = entry
        return loaded

    # ── Validation helpers ────────────────────────────────────

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise ValueError(
                "Wallet not configured — run 'openlegion wallet init' "
                "to generate a master seed"
            )

    def _validate_chain(self, chain: str) -> dict[str, Any]:
        """Return chain config or raise ValueError."""
        cfg = self._chains.get(chain)
        if cfg is None:
            supported = ", ".join(sorted(self._chains))
            raise ValueError(f"Unknown chain: {chain}. Supported: {supported}")
        return cfg

    @staticmethod
    def _validate_evm_address(address: str) -> str:
        """Validate and return checksummed EVM address."""
        from web3 import Web3

        if not address or len(address) != 42 or not address.startswith("0x"):
            raise ValueError(f"Invalid EVM address format: {address}")
        try:
            return Web3.to_checksum_address(address)
        except Exception:
            raise ValueError(f"Invalid EVM address: {address}")

    @staticmethod
    def _validate_solana_address(address: str) -> str:
        """Validate Solana base58 address."""
        from solders.pubkey import Pubkey

        try:
            Pubkey.from_string(address)
            return address
        except Exception:
            raise ValueError(f"Invalid Solana address: {address}")

    @staticmethod
    def _validate_amount(amount: str) -> Decimal:
        """Validate amount string and return as Decimal."""
        if not amount:
            raise ValueError("Amount is required")
        try:
            d = Decimal(amount)
        except InvalidOperation:
            raise ValueError(f"Invalid amount: {amount}")
        if not d.is_finite():
            raise ValueError(f"Invalid amount: {amount}")
        if d <= 0:
            raise ValueError(f"Amount must be positive: {amount}")
        exp = d.as_tuple().exponent
        if isinstance(exp, int) and exp < -18:
            raise ValueError(f"Too many decimal places: {amount}")
        return d

    @staticmethod
    def _validate_function_signature(sig: str) -> None:
        """Validate Solidity function signature format."""
        if not sig or "(" not in sig or not sig.endswith(")"):
            raise ValueError(
                f"Invalid function signature: {sig}. "
                f"Expected format: 'functionName(type1,type2)'"
            )

    def _validate_address(self, address: str, ecosystem: str) -> str:
        """Validate and normalise an address for the given ecosystem."""
        if ecosystem == "evm":
            return self._validate_evm_address(address)
        if ecosystem == "solana":
            return self._validate_solana_address(address)
        raise ValueError(f"Unknown ecosystem: {ecosystem}")

    # ── Key derivation ────────────────────────────────────────

    def _derive_evm_account(self, agent_index: int):
        """BIP-44: m/44'/60'/0'/0/{index}."""
        from eth_account import Account

        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(
            self._master_seed,
            account_path=f"m/44'/60'/0'/0/{agent_index}",
        )

    def _derive_solana_keypair(self, agent_index: int):
        """Derive Solana keypair via HMAC-SHA512 of BIP-39 seed.

        Uses a project-specific derivation (deterministic, secure) rather
        than full SLIP-0010.  Addresses won't match Phantom/Solflare —
        fine for auto-generated agent wallets.
        """
        from solders.keypair import Keypair

        seed = hashlib.pbkdf2_hmac(
            "sha512",
            self._master_seed.encode("utf-8"),
            b"mnemonic",
            2048,
        )
        path = f"m/44'/501'/{agent_index}'/0'"
        derived = _hmac.new(seed, path.encode("utf-8"), hashlib.sha512).digest()
        return Keypair.from_seed(derived[:32])

    def _derive_account(self, agent_index: int, ecosystem: str):
        if ecosystem == "evm":
            return self._derive_evm_account(agent_index)
        if ecosystem == "solana":
            return self._derive_solana_keypair(agent_index)
        raise ValueError(f"Unknown ecosystem: {ecosystem}")

    def _get_or_assign_index(self, agent_id: str) -> int:
        """Get agent's HD index, assigning next available if new.

        Double-check locking inside a DB transaction prevents races.
        """
        row = self.db.execute(
            "SELECT idx FROM agent_index WHERE agent_id = ?", (agent_id,),
        ).fetchone()
        if row:
            return row[0]

        with self.db:
            row = self.db.execute(
                "SELECT idx FROM agent_index WHERE agent_id = ?", (agent_id,),
            ).fetchone()
            if row:
                return row[0]
            self.db.execute(
                "INSERT OR IGNORE INTO index_counter(id, next_idx) VALUES (1, 0)",
            )
            self.db.execute(
                "UPDATE index_counter SET next_idx = next_idx + 1 WHERE id = 1",
            )
            idx = self.db.execute(
                "SELECT next_idx - 1 FROM index_counter WHERE id = 1",
            ).fetchone()[0]
            self.db.execute(
                "INSERT INTO agent_index(agent_id, idx) VALUES (?, ?)",
                (agent_id, idx),
            )
        return idx

    # ── RPC providers (lazy, cached) ──────────────────────────

    _RPC_TIMEOUT = 30  # seconds

    async def _get_evm_provider(self, chain: str):
        if chain in self._evm_providers:
            return self._evm_providers[chain]
        from web3 import AsyncHTTPProvider, AsyncWeb3

        cfg = self._chains[chain]
        provider = AsyncWeb3(AsyncHTTPProvider(
            cfg["rpc_url"],
            request_kwargs={"timeout": self._RPC_TIMEOUT},
        ))
        self._evm_providers[chain] = provider
        return provider

    async def _get_solana_client(self, chain: str):
        if chain in self._solana_clients:
            return self._solana_clients[chain]
        from solana.rpc.async_api import AsyncClient

        cfg = self._chains[chain]
        client = AsyncClient(cfg["rpc_url"], timeout=self._RPC_TIMEOUT)
        self._solana_clients[chain] = client
        return client

    # ── Public API ────────────────────────────────────────────

    async def get_address(self, agent_id: str, chain: str) -> str:
        """Derive and return the agent's wallet address."""
        self._require_configured()
        cfg = self._validate_chain(chain)
        eco = cfg["ecosystem"]
        index = self._get_or_assign_index(agent_id)
        account = self._derive_account(index, eco)
        if eco == "evm":
            return account.address
        return str(account.pubkey())

    async def get_balance(
        self, agent_id: str, chain: str, token: str = "native",
    ) -> dict:
        """Query balance via public RPC."""
        self._require_configured()
        cfg = self._validate_chain(chain)
        eco = cfg["ecosystem"]
        address = await self.get_address(agent_id, chain)

        if eco == "evm":
            return await self._evm_get_balance(chain, address, token, cfg)
        return await self._solana_get_balance(chain, address, token, cfg)

    async def read_contract(
        self,
        agent_id: str,
        chain: str,
        contract: str,
        function: str = "",
        args: list | None = None,
    ) -> dict:
        """Read-only call.  EVM: eth_call.  Solana: getAccountInfo."""
        cfg = self._validate_chain(chain)
        eco = cfg["ecosystem"]

        if eco == "evm":
            if not function:
                raise ValueError("EVM chains require a function signature")
            self._validate_evm_address(contract)
            self._validate_function_signature(function)
            return await self._evm_read_contract(chain, contract, function, args or [])
        # Solana: read account data
        self._validate_solana_address(contract)
        return await self._solana_read_account(chain, contract)

    async def transfer(
        self,
        agent_id: str,
        chain: str,
        to: str,
        amount: str,
        token: str = "native",
        permissions: PermissionMatrix | None = None,
    ) -> dict:
        """Send native or fungible tokens."""
        self._require_configured()
        cfg = self._validate_chain(chain)
        eco = cfg["ecosystem"]
        to = self._validate_address(to, eco)
        parsed_amount = self._validate_amount(amount)

        if eco == "evm" and token != "native":
            # ERC-20 transfer: encode transfer(to, amount) and send to the
            # token contract, not the recipient.
            token_addr = self._validate_evm_address(token)
            # Fetch decimals from the token contract to correctly convert
            # human-readable amounts.  Critical for USDC (6), USDT (6),
            # WBTC (8) — hardcoding 18 would make transfers fail or send
            # wildly wrong amounts.
            token_decimals = await self._evm_token_decimals(chain, token_addr)
            raw_amount = int(parsed_amount * 10 ** token_decimals)
            calldata = self._encode_function_call(
                "transfer(address,uint256)", [to, raw_amount],
            )
            tx_params: dict[str, Any] = {
                "to": token_addr,
                "data": calldata,
                "native_value": 0,
                "value_wei": 0,
                "amount": str(parsed_amount),
                "token": token,
            }
        else:
            # Native token transfer (ETH/SOL/POL)
            if eco == "solana" and token != "native":
                raise ValueError(
                    "SPL token transfers via wallet_transfer are not yet supported. "
                    "Use wallet_execute with a base64 unsigned transaction from "
                    "Jupiter or another DEX API instead."
                )
            native_value = int(parsed_amount * 10 ** cfg["decimals"]) if token == "native" else 0
            tx_params = {
                "to": to,
                "amount": str(parsed_amount),
                "token": token,
                "native_value": native_value,
            }

        audit = {"to_address": to, "value": amount, "token": token}
        return await self._sign_and_send(agent_id, chain, tx_params, permissions, audit)

    async def execute_contract(
        self,
        agent_id: str,
        chain: str,
        contract: str = "",
        function: str = "",
        args: list | None = None,
        value: str = "0",
        transaction: str = "",
        permissions: PermissionMatrix | None = None,
    ) -> dict:
        """EVM: contract + function + args.  Solana: base64 unsigned tx."""
        self._require_configured()
        cfg = self._validate_chain(chain)
        eco = cfg["ecosystem"]

        if eco == "evm":
            if not contract or not function:
                raise ValueError("EVM wallet_execute requires contract and function")
            contract = self._validate_evm_address(contract)
            self._validate_function_signature(function)
            parsed_value = self._validate_amount(value) if value and value != "0" else Decimal(0)
            native_value = int(parsed_value * 10 ** cfg["decimals"])
            calldata = self._encode_function_call(function, args or [])
            tx_params: dict[str, Any] = {
                "to": contract,
                "data": calldata,
                "native_value": native_value,
                "value_wei": native_value,
            }
            audit = {
                "to_address": contract, "value": value,
                "token": "native", "function": function,
            }
        elif eco == "solana":
            if not transaction:
                raise ValueError(
                    "Solana wallet_execute requires a base64-encoded unsigned transaction"
                )
            tx_params = {"unsigned_transaction": transaction, "native_value": 0}
            audit = {"to_address": contract or "program", "value": "0", "token": "native"}
        else:
            raise ValueError(f"Unknown ecosystem: {eco}")

        return await self._sign_and_send(agent_id, chain, tx_params, permissions, audit)

    # ── Signing core ──────────────────────────────────────────

    async def _sign_and_send(
        self,
        agent_id: str,
        chain: str,
        tx_params: dict,
        permissions: PermissionMatrix | None,
        audit_info: dict,
    ) -> dict:
        cfg = self._chains[chain]
        native_value = tx_params.get("native_value", 0)
        value_usd = await self._estimate_value_usd(chain, native_value)

        denial = self._check_policy(agent_id, value_usd, permissions)
        if denial:
            self._audit(agent_id, chain, "rejected", error=denial, **audit_info)
            raise PermissionError(denial)

        eco = cfg["ecosystem"]
        if eco == "evm":
            tx_hash = await self._evm_sign_and_send(agent_id, chain, tx_params)
        elif eco == "solana":
            tx_hash = await self._solana_sign_and_send(agent_id, chain, tx_params)
        else:
            raise ValueError(f"Unknown ecosystem: {eco}")

        self._audit(
            agent_id, chain, "broadcast",
            tx_hash=tx_hash, value_usd=value_usd, **audit_info,
        )
        explorer = cfg["explorer_tx_fmt"]
        return {
            "tx_hash": tx_hash,
            "chain": chain,
            "status": "broadcast",
            "explorer_url": explorer.format(tx_hash=tx_hash),
        }

    # ── EVM backend ───────────────────────────────────────────

    async def _evm_sign_and_send(self, agent_id: str, chain: str, tx_params: dict) -> str:
        w3 = await self._get_evm_provider(chain)
        account = self._derive_evm_account(self._get_or_assign_index(agent_id))
        tx = await self._fill_evm_tx(w3, account.address, tx_params, chain)

        has_calldata = bool(tx.get("data"))
        if has_calldata:
            try:
                await w3.eth.call(tx)
            except Exception as e:
                raise ValueError(f"Transaction simulation failed: {e}")
        else:
            balance = await w3.eth.get_balance(account.address)
            gas_cost = tx.get("gas", 21000) * tx.get("maxFeePerGas", 0)
            total = tx.get("value", 0) + gas_cost
            if balance < total:
                from web3 import Web3
                raise ValueError(
                    f"Insufficient balance: have {Web3.from_wei(balance, 'ether')} "
                    f"{self._chains[chain]['symbol']}, "
                    f"need ~{Web3.from_wei(total, 'ether')} (value + gas)"
                )

        signed = account.sign_transaction(tx)
        raw_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
        return raw_hash.hex()

    async def _fill_evm_tx(
        self, w3, from_address: str, tx_params: dict, chain: str,
    ) -> dict:
        cfg = self._chains[chain]
        tx: dict[str, Any] = {}
        tx["from"] = from_address
        tx["to"] = tx_params["to"]
        tx["chainId"] = cfg["chain_id"]
        tx["value"] = tx_params.get("value_wei", tx_params.get("native_value", 0))
        if tx_params.get("data"):
            tx["data"] = tx_params["data"]
        tx["nonce"] = await w3.eth.get_transaction_count(from_address)

        latest = await w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", 0)
        priority_fee = await w3.eth.max_priority_fee
        tx["maxFeePerGas"] = int(base_fee * 1.2) + priority_fee
        tx["maxPriorityFeePerGas"] = priority_fee

        try:
            estimated = await w3.eth.estimate_gas(tx)
            tx["gas"] = max(int(estimated * 1.1), 21000)
        except Exception:
            tx["gas"] = 21000

        tx.pop("from", None)
        return tx

    async def _evm_get_balance(
        self, chain: str, address: str, token: str, cfg: dict,
    ) -> dict:
        w3 = await self._get_evm_provider(chain)
        if token == "native":
            raw = await w3.eth.get_balance(address)
            from web3 import Web3
            return {
                "balance": str(Web3.from_wei(raw, "ether")),
                "symbol": cfg["symbol"],
                "decimals": cfg["decimals"],
                "raw": str(raw),
            }
        # ERC-20
        from web3 import Web3
        token_addr = self._validate_evm_address(token)
        # balanceOf(address)
        selector = Web3.keccak(text="balanceOf(address)")[:4]
        from eth_abi import encode
        calldata = selector + encode(["address"], [address])
        result = await w3.eth.call({"to": token_addr, "data": calldata})
        from eth_abi import decode
        (raw_balance,) = decode(["uint256"], result)
        # decimals()
        dec_selector = Web3.keccak(text="decimals()")[:4]
        dec_result = await w3.eth.call({"to": token_addr, "data": dec_selector})
        (decimals,) = decode(["uint8"], dec_result)
        human = str(Decimal(raw_balance) / Decimal(10 ** decimals))
        return {
            "balance": human,
            "symbol": token[:8],
            "decimals": decimals,
            "raw": str(raw_balance),
        }

    async def _evm_token_decimals(self, chain: str, token_addr: str) -> int:
        """Fetch ERC-20 decimals from the token contract.

        Raises ValueError if the RPC call fails — silently guessing
        decimals would cause catastrophically wrong transfer amounts
        (e.g. 1 USDC interpreted as 10^12 USDC with 18 vs 6 decimals).
        """
        from eth_abi import decode
        from web3 import Web3

        w3 = await self._get_evm_provider(chain)
        try:
            selector = Web3.keccak(text="decimals()")[:4]
            result = await w3.eth.call({"to": token_addr, "data": selector})
            (decimals,) = decode(["uint8"], result)
            return decimals
        except Exception as e:
            raise ValueError(
                f"Could not fetch decimals for token {token_addr} on {chain}: {e}. "
                f"Cannot safely determine transfer amount."
            )

    async def _evm_read_contract(
        self, chain: str, contract: str, function: str, args: list,
    ) -> dict:
        w3 = await self._get_evm_provider(chain)
        calldata = self._encode_function_call(function, args)
        result = await w3.eth.call({
            "to": self._validate_evm_address(contract),
            "data": calldata,
        })
        return {"result": "0x" + result.hex(), "chain": chain}

    # ── Solana backend ────────────────────────────────────────

    async def _solana_sign_and_send(self, agent_id: str, chain: str, tx_params: dict) -> str:
        from solana.rpc.types import TxOpts
        from solders.transaction import VersionedTransaction

        client = await self._get_solana_client(chain)
        keypair = self._derive_solana_keypair(self._get_or_assign_index(agent_id))

        if "unsigned_transaction" in tx_params:
            tx_bytes = base64.b64decode(tx_params["unsigned_transaction"])
            unsigned = VersionedTransaction.from_bytes(tx_bytes)
            msg = unsigned.message

            agent_pubkey = keypair.pubkey()
            signer_keys = list(msg.account_keys)[: msg.header.num_required_signatures]
            if agent_pubkey not in signer_keys:
                raise ValueError(
                    f"Agent wallet {agent_pubkey} is not a required signer in this "
                    f"transaction. Ensure the protocol API was called with this "
                    f"agent's public key."
                )
            tx = VersionedTransaction(msg, [keypair])
        else:
            tx = await self._build_solana_transfer(keypair, tx_params, client)

        sim = await client.simulate_transaction(tx)
        if sim.value.err:
            raise ValueError(f"Transaction simulation failed: {sim.value.err}")

        result = await client.send_transaction(
            tx, opts=TxOpts(skip_preflight=True),
        )
        return str(result.value)

    async def _build_solana_transfer(self, keypair, tx_params: dict, client) -> Any:
        from solders.message import MessageV0
        from solders.pubkey import Pubkey
        from solders.system_program import TransferParams, transfer
        from solders.transaction import VersionedTransaction

        to_pubkey = Pubkey.from_string(tx_params["to"])
        amount_lamports = int(Decimal(tx_params["amount"]) * 10**9)

        if tx_params.get("token", "native") != "native":
            raise NotImplementedError(
                "SPL token transfers not yet supported. "
                "Use wallet_execute with a base64 unsigned transaction from "
                "Jupiter or another DEX API as a workaround."
            )

        ix = transfer(TransferParams(
            from_pubkey=keypair.pubkey(),
            to_pubkey=to_pubkey,
            lamports=amount_lamports,
        ))
        blockhash_resp = await client.get_latest_blockhash()
        blockhash = blockhash_resp.value.blockhash
        msg = MessageV0.try_compile(
            payer=keypair.pubkey(),
            instructions=[ix],
            address_lookup_table_accounts=[],
            recent_blockhash=blockhash,
        )
        return VersionedTransaction(msg, [keypair])

    async def _solana_get_balance(
        self, chain: str, address: str, token: str, cfg: dict,
    ) -> dict:
        from solders.pubkey import Pubkey

        client = await self._get_solana_client(chain)
        if token == "native":
            resp = await client.get_balance(Pubkey.from_string(address))
            raw = resp.value
            human = str(Decimal(raw) / Decimal(10**9))
            return {
                "balance": human,
                "symbol": cfg["symbol"],
                "decimals": cfg["decimals"],
                "raw": str(raw),
            }
        # SPL token balance
        from solders.pubkey import Pubkey as Pk
        mint = Pk.from_string(token)
        owner = Pk.from_string(address)
        resp = await client.get_token_accounts_by_owner_json_parsed(
            owner, {"mint": mint},
        )
        if resp.value:
            info = resp.value[0].account.data.parsed["info"]["tokenAmount"]
            return {
                "balance": info["uiAmountString"],
                "symbol": token[:8],
                "decimals": info["decimals"],
                "raw": info["amount"],
            }
        return {"balance": "0", "symbol": token[:8], "decimals": 0, "raw": "0"}

    async def _solana_read_account(self, chain: str, account: str) -> dict:
        from solders.pubkey import Pubkey

        client = await self._get_solana_client(chain)
        resp = await client.get_account_info(Pubkey.from_string(account))
        if resp.value is None:
            return {"error": "Account not found", "chain": chain}
        acct = resp.value
        return {
            "lamports": acct.lamports,
            "owner": str(acct.owner),
            "data": base64.b64encode(bytes(acct.data)).decode("ascii"),
            "executable": acct.executable,
            "chain": chain,
        }

    # ── EVM ABI encoding ──────────────────────────────────────

    def _encode_function_call(self, function_sig: str, args: list) -> bytes:
        """Encode a Solidity function call from signature + args."""
        from eth_abi import encode
        from web3 import Web3

        self._validate_function_signature(function_sig)
        selector = Web3.keccak(text=function_sig)[:4]

        params_str = function_sig[function_sig.index("(") + 1: -1]
        if not params_str:
            return selector

        param_types = self._split_abi_params(params_str)
        return selector + encode(param_types, args)

    @staticmethod
    def _split_abi_params(params_str: str) -> list[str]:
        """Split ABI param string respecting nested parentheses.

        "address,uint256" → ["address", "uint256"]
        "(address,uint256),bool" → ["(address,uint256)", "bool"]
        """
        result: list[str] = []
        depth = 0
        current: list[str] = []
        for char in params_str:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                result.append("".join(current))
                current = []
            else:
                current.append(char)
        if current:
            result.append("".join(current))
        return result

    # ── Price feed ────────────────────────────────────────────

    async def _estimate_value_usd(self, chain: str, value_smallest_unit: int) -> float:
        if value_smallest_unit == 0:
            return 0.0
        cfg = self._chains[chain]
        price = await self._get_price(cfg["symbol"], cfg["coingecko_id"])
        return float(Decimal(value_smallest_unit) / Decimal(10 ** cfg["decimals"])) * price

    async def _get_price(self, symbol: str, coingecko_id: str) -> float:
        cached = self._price_cache.get(symbol)
        if cached and time.time() - cached[1] < _PRICE_CACHE_TTL:
            return cached[0]
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                resp = await c.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={"ids": coingecko_id, "vs_currencies": "usd"},
                )
                if resp.status_code == 200:
                    price = resp.json()[coingecko_id]["usd"]
                    self._price_cache[symbol] = (price, time.time())
                    return price
        except (httpx.HTTPError, OSError, KeyError, ValueError):
            pass
        return _FALLBACK_PRICES.get(symbol, 3000.0)

    # ── Policy engine ─────────────────────────────────────────

    def _check_policy(
        self,
        agent_id: str,
        value_usd: float,
        permissions: PermissionMatrix | None = None,
    ) -> str | None:
        """Returns None if allowed, error string if denied."""
        per_tx, daily, rate = 0.0, 0.0, 0
        if permissions:
            per_tx, daily, rate = permissions.get_wallet_limits(agent_id)
        per_tx = per_tx or self._default_per_tx_usd
        daily = daily or self._default_daily_usd
        rate = rate or self._default_rate_per_hour

        if value_usd > per_tx:
            return f"Per-transaction limit exceeded: ${value_usd:.2f} > ${per_tx:.2f}"

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(value_usd), 0) FROM transactions "
            "WHERE agent_id = ? AND timestamp >= ? AND status = 'broadcast'",
            (agent_id, today),
        ).fetchone()
        daily_used = row[0] if row else 0.0
        if daily_used + value_usd > daily:
            return (
                f"Daily limit exceeded: ${daily_used + value_usd:.2f} > ${daily:.2f}"
            )

        now = time.time()
        ts = self._tx_timestamps.setdefault(agent_id, [])
        ts[:] = [t for t in ts if now - t < 3600]
        if len(ts) >= rate:
            return (
                f"Rate limit exceeded: {len(ts)} transactions in the last hour "
                f"(limit: {rate})"
            )
        ts.append(now)
        return None

    # ── Audit log ─────────────────────────────────────────────

    def _audit(
        self,
        agent_id: str,
        chain: str,
        status: str,
        *,
        tx_hash: str = "",
        to_address: str = "",
        value: str = "0",
        token: str = "native",
        value_usd: float = 0.0,
        function: str = "",
        error: str = "",
    ) -> None:
        try:
            self.db.execute(
                "INSERT INTO transactions "
                "(agent_id, chain, tx_hash, to_address, value, token, "
                " value_usd, function, status, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id, chain, tx_hash or None, to_address,
                    value, token, value_usd, function or None,
                    status, error or None,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.warning("Audit log write failed: %s", e)
