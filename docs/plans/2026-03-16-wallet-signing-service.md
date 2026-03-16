# Wallet Signing Service — Implementation Plan

**Goal:** Give every agent in the fleet secure access to EVM and Solana wallets for onchain transactions, with private keys that never enter agent containers.

**Core invariant:** Private keys exist ONLY in `WalletService` memory within the mesh process. Agents describe what they want; the mesh decides whether to allow it and executes it.

---

## How This Extends the Vault

This is **not** a parallel system. It extends the existing vault architecture with a new operation type.

The vault has two credential patterns today:

| Pattern | Example | Agent sees key? | How it works |
|---|---|---|---|
| **A — Proxy** | LLM calls | NO | Agent → `api_call("llm", params)` → mesh uses system key internally → returns response |
| **B — $CRED{}** | HTTP tool | Code only (not LLM) | Agent code resolves `$CRED{name}` → gets raw value → uses it, redacts output |

Wallet signing uses **Pattern A** — the proxy pattern:

```
Agent → wallet_transfer(chain, to, amount) → mesh verifies identity + permissions
  → WalletService derives key from master seed → signs → broadcasts
  → returns tx_hash (NEVER key, NEVER signature)
```

The master seed is a system-tier credential (`OPENLEGION_SYSTEM_WALLET_MASTER_SEED`), stored in `.env`, loaded at startup — same as LLM API keys. WalletService is to transaction signing what `_handle_llm` is to LLM calls: a handler that uses a system credential internally and only returns the result.

More secure than Pattern B ($CRED{}): the key never reaches agent code at all.

---

## Architecture

```
Agent Container (untrusted, no keys)
  │
  │  wallet_get_address(chain)
  │  wallet_get_balance(chain, token?)
  │  wallet_read_contract(chain, contract, function, args)
  │  wallet_transfer(chain, to, amount, token?)
  │  wallet_execute(chain, contract, function, args, value?)
  │
  ▼
Mesh Host (FastAPI :8420, trusted)
  ├─ GET  /mesh/wallet/address       ── identity + permission
  ├─ GET  /mesh/wallet/balance       ── identity + permission
  ├─ POST /mesh/wallet/read          ── identity + permission (no signing)
  ├─ POST /mesh/wallet/transfer      ── identity + permission + policy
  ├─ POST /mesh/wallet/execute       ── identity + permission + policy
  │
  ▼
WalletService (in-process, same trust as CredentialVault)
  ├─ HD key derivation (master seed → per-agent key, BIP-44)
  ├─ EVM backend (web3.py) + Solana backend (solders)
  ├─ Policy check (spend limits, rate limits)
  ├─ Simulate transaction before signing
  ├─ Sign + broadcast
  └─ Return tx_hash / signature only
```

**Why in-process:** Same trust level as CredentialVault. No new container, no new network boundary, no new attack surface. The mesh process already holds all system secrets.

### Agent Tools (5 tools, chain-agnostic)

| Tool | Signing? | Purpose |
|---|---|---|
| `wallet_get_address` | No | Get agent's wallet address for a chain |
| `wallet_get_balance` | No | Check native or token balance |
| `wallet_read_contract` | No | Read-only call (EVM: eth_call, Solana: getAccountInfo) |
| `wallet_transfer` | Yes | Send native tokens or fungible tokens (ERC-20 / SPL) |
| `wallet_execute` | Yes | EVM: call any contract function. Solana: sign an unsigned transaction. |

All 5 tools work on both EVM and Solana. The `chain` parameter (`"evm:base"` vs `"solana:mainnet"`) determines which backend handles the request. The tools, mesh endpoints, permissions, and policy engine are ecosystem-agnostic — only WalletService internals differ.

### Multi-Ecosystem Design

EVM and Solana have fundamentally different programming models. Rather than forcing them into one abstraction, we let each backend handle its ecosystem naturally:

| Aspect | EVM | Solana |
|---|---|---|
| Key derivation | `m/44'/60'/0'/0/{index}` | `m/44'/501'/{index}'/0'` |
| `wallet_transfer` | Native ETH send / ERC-20 `transfer()` | System Program transfer / SPL Token transfer |
| `wallet_execute` | `contract` + `function` signature + `args` | `transaction` (base64 unsigned tx from protocol APIs) |
| `wallet_read_contract` | `eth_call` with function sig → decoded result | `getAccountInfo` → account data (lamports, owner, data) |
| Simulation | `eth_call` (contract calls only) | `simulateTransaction` with `sigVerify: false` |
| Signing lib | `eth-account` | `solders` |

**Why Solana `wallet_execute` takes a base64 transaction:** Solana programs don't have a universal ABI like Solidity. Each program has its own instruction format. DeFi protocols (Jupiter, Raydium, Drift) provide APIs that return unsigned transactions ready to sign — the agent calls the API, gets the transaction, and hands it to the wallet service for signing. This is the standard Solana DeFi pattern and the cleanest abstraction.

### DeFi Readiness

**EVM Example — Agent swaps ETH for USDC on Uniswap (Base):**
```
1. wallet_read_contract("evm:base", QUOTER, "quoteExactInputSingle((address,address,uint256,uint24,uint160))", [...])
   → "Expected output: 250.43 USDC"

2. wallet_execute("evm:base", contract=ROUTER, function="exactInputSingle((address,address,uint24,address,uint256,uint256,uint160))", args=[...], value="0.1")
   → {tx_hash: "0x...", status: "broadcast"}

3. wallet_get_balance("evm:base", token="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
   → {balance: "250.43", symbol: "USDC"}
```

**Solana Example — Agent swaps SOL for USDC on Jupiter:**
```
1. http_request("https://quote-api.jup.ag/v6/quote?inputMint=So1...&outputMint=EPJF...&amount=100000000")
   → {routePlan: [...], outAmount: "25043000"}

2. http_request("https://quote-api.jup.ag/v6/swap", method="POST", body={quoteResponse, userPublicKey: MY_ADDRESS})
   → {swapTransaction: "base64-encoded-unsigned-tx..."}

3. wallet_execute("solana:mainnet", transaction="base64-encoded-unsigned-tx...")
   → {tx_hash: "5Uf2...", status: "broadcast"}

4. wallet_get_balance("solana:mainnet", token="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
   → {balance: "250.43", symbol: "USDC"}
```

**EVM Example — Prediction market on Polymarket:**
```
1. wallet_read_contract → check market state, odds, resolution status
2. wallet_execute → approve USDC spending by market contract
3. wallet_execute → buy outcome tokens
4. wallet_read_contract → verify position
```

The EVM function signature approach handles complex types (structs as tuples, arrays, nested types) via `eth_abi` encoding — same syntax Etherscan and Solidity docs use, natural for LLMs.

For Solana, the "protocol API → unsigned tx → sign" pattern works with any program — Jupiter, Raydium, Drift, Marinade, Tensor, whatever ships next. No WalletService changes needed per protocol.

### Adding Future Chains

Adding a new ecosystem (e.g. `sui:mainnet`, `aptos:mainnet`, `cosmos:osmosis`) requires:
1. A new backend class in `wallet.py` (key derivation + signing + RPC)
2. An entry in the chain registry
3. No changes to tools, endpoints, permissions, or policy engine

---

## Key Management

**HD derivation from single master seed — one seed, all ecosystems:**

```
OPENLEGION_SYSTEM_WALLET_MASTER_SEED  (BIP-39 mnemonic, stored in .env)
  │
  ├─ EVM:    m/44'/60'/0'/0/{agent_index}   (BIP-44 standard, all EVM chains share one address)
  └─ Solana: m/44'/501'/{agent_index}'/0'   (Solana standard, separate address per agent)
```

- One backup (the mnemonic) protects all agent wallets on all chains
- Deterministic — re-derivable from seed + index
- Each agent gets one EVM address (works on all EVM chains) and one Solana address
- Agent index assigned at registration, stored in `wallet.db`
- Indices monotonically increase, never reused
- Derived keys held only in WalletService memory — never written to disk

**Lifecycle:**

| Phase | What happens |
|---|---|
| `openlegion wallet init` | Generate BIP-39 mnemonic → store in `.env` → show once for backup |
| Agent starts | Next index assigned in `wallet.db` → address derived |
| Transaction | Key derived on-demand → sign → discard. Never persisted. |
| Agent removed | Index retired (never reused). Key no longer derivable. |

---

## Chain Configuration

Public RPCs baked in as defaults. User can override with env vars for better rate limits / reliability.

```python
_CHAINS = {
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
        "rpc_default": "https://polygon-rpc.com",
        "rpc_env": "OPENLEGION_SYSTEM_WALLET_RPC_POLYGON",
        "explorer_tx_fmt": "https://polygonscan.com/tx/{tx_hash}",
        "symbol": "POL",
        "decimals": 18,
        "coingecko_id": "matic-network",
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
        "rpc_default": "https://api.mainnet-beta.solana.com",
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
```

**Explorer URL:** Uses `explorer_tx_fmt` with `{tx_hash}` placeholder instead of string concatenation — handles Solana devnet's query parameter format correctly.

User overrides: set `OPENLEGION_SYSTEM_WALLET_RPC_ETHEREUM=https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY` or `OPENLEGION_SYSTEM_WALLET_RPC_SOLANA=https://your-helius-rpc.com` in `.env`. If present, takes priority over the default.

---

## Permission Model

Extend `AgentPermissions` in `types.py`:

```python
can_use_wallet: bool = False                # Deny-all default
wallet_allowed_chains: list[str] = []       # e.g. ["evm:base", "evm:ethereum"]
wallet_spend_limit_per_tx_usd: float = 0.0  # 0 = global default ($10)
wallet_spend_limit_daily_usd: float = 0.0   # 0 = global default ($100)
wallet_rate_limit_per_hour: int = 0          # 0 = global default (10)
wallet_allowed_contracts: list[str] = []     # Empty = allow all (Phase 2: restrict)
```

Config example:
```json
{
  "permissions": {
    "trader": {
      "can_use_wallet": true,
      "wallet_allowed_chains": ["evm:base", "solana:mainnet"],
      "wallet_spend_limit_per_tx_usd": 50,
      "wallet_spend_limit_daily_usd": 500
    },
    "researcher": {
      "can_use_wallet": true,
      "wallet_allowed_chains": ["evm:sepolia", "solana:devnet"],
      "wallet_spend_limit_per_tx_usd": 1,
      "wallet_spend_limit_daily_usd": 10
    }
  }
}
```

---

## Guardrails

| Guardrail | In MVP? | Default |
|---|---|---|
| Per-tx USD spend limit | Yes | $10 |
| Daily USD spend limit | Yes | $100 per agent |
| Rate limiting (tx/hour) | Yes | 10 per agent |
| Chain allowlist | Yes | Deny-all, explicit opt-in |
| Address format validation | Yes | Checksum (EVM) / base58 (Solana) |
| Amount validation | Yes | Reject negative, zero, non-numeric, excessive precision |
| Transaction simulation | Yes | `eth_call` for contract calls (EVM) / `simulateTransaction` with `sigVerify: false` (Solana) |
| Chain ID validation | Yes | Reject chains not in `_CHAINS` registry |
| Contract allowlist | Phase 2 | Allow all (field exists but unused) |
| Unlimited approval detection | Phase 2 | — |
| Human-in-the-loop approval | Phase 3 | — |

**Price feed for USD limits:** CoinGecko free API with 5-minute cache + hardcoded fallbacks ($3000 ETH, $0.50 POL, $150 SOL). Simple, no API key needed.

**Spend limit scope:** Limits apply to the **native token value** of each transaction (msg.value for EVM, lamports for Solana). For pure ERC-20/SPL token transfers with zero native value, rate limiting still applies but USD value estimation is best-effort via the token's CoinGecko price if available, otherwise the transaction is allowed but capped by rate limits. This is a known limitation — Phase 2 adds token-aware spend limits.

---

## Threat Model

| Attacker | Vector | Mitigation | Residual risk |
|---|---|---|---|
| Prompt-injected agent | Drain wallet | Per-tx + daily spend limits, simulation | Can send within limits to arbitrary addresses |
| Prompt-injected agent | Exfiltrate private key | Key never in container — architectural guarantee | None |
| Prompt-injected agent | Malicious contract calls | Spend limits cap damage, simulation catches reverts | Bad trades within limits |
| Prompt-injected agent | Inject malicious Solana tx | Simulation catches failures, spend limits cap native value | Malicious instructions within limits (Phase 2: program allowlist) |
| Compromised container | Read key from memory/disk | Key not in container at all | None |
| Compromised container | Call signing endpoint directly | Same auth + permissions + policy apply | Same capability as tools (bounded by policy) |
| Malicious user | Access other users' keys | Separate VPS per user, separate master seed | None (infra isolation) |

---

## Implementation — 5 Tasks

### Task 1: Types + Permissions

**Files:** `src/shared/types.py`, `src/host/permissions.py`
**Tests:** `tests/test_types.py` (extend), `tests/test_permissions.py` (new)

**1a. Extend `AgentPermissions` in `types.py`**

Add after `can_manage_cron` (line 179):

```python
    can_use_wallet: bool = False
    wallet_allowed_chains: list[str] = []
    wallet_spend_limit_per_tx_usd: float = 0.0
    wallet_spend_limit_daily_usd: float = 0.0
    wallet_rate_limit_per_hour: int = 0
    wallet_allowed_contracts: list[str] = []
```

**1b. Add permission checks in `permissions.py`**

Add methods to `PermissionMatrix`:

```python
def can_use_wallet(self, agent_id: str) -> bool:
    if self._is_trusted(agent_id):
        return True
    return self.get_permissions(agent_id).can_use_wallet

def can_use_wallet_chain(self, agent_id: str, chain: str) -> bool:
    if self._is_trusted(agent_id):
        return True
    perms = self.get_permissions(agent_id)
    if not perms.can_use_wallet:
        return False
    return "*" in perms.wallet_allowed_chains or chain in perms.wallet_allowed_chains

def get_wallet_limits(self, agent_id: str) -> tuple[float, float, int]:
    """Return (per_tx_usd, daily_usd, rate_per_hour). 0 = use global default."""
    perms = self.get_permissions(agent_id)
    return (
        perms.wallet_spend_limit_per_tx_usd,
        perms.wallet_spend_limit_daily_usd,
        perms.wallet_rate_limit_per_hour,
    )

def can_access_wallet_contract(self, agent_id: str, contract: str) -> bool:
    if self._is_trusted(agent_id):
        return True
    contracts = self.get_permissions(agent_id).wallet_allowed_contracts
    if not contracts:
        return True  # Empty = allow all
    return "*" in contracts or contract.lower() in [c.lower() for c in contracts]
```

Update `get_permissions` default fallback to propagate new fields.

**Tests:**
- Default permissions: `can_use_wallet=False`, empty chain list
- `can_use_wallet_chain` requires both `can_use_wallet=True` AND chain in list
- Wildcard `"*"` in chains allows all
- `can_access_wallet_contract` allows all when list is empty
- `can_access_wallet_contract` with populated list rejects unlisted contracts
- Trusted agents bypass all checks
- Default template propagates wallet fields

---

### Task 2: WalletService

**Files:** `src/host/wallet.py` (new)
**Tests:** `tests/test_wallet.py` (new)

This is the security-critical module. All private key material is confined here.

**Module structure:**

```python
"""Wallet signing service — extends the vault with onchain transaction signing.

Same trust model as CredentialVault: master seed is a system-tier credential
loaded from .env. Agents request transactions through the mesh. This service
derives the per-agent key, enforces policy, signs, and broadcasts. Only the
transaction hash is returned.

Key hierarchy:
  OPENLEGION_SYSTEM_WALLET_MASTER_SEED (BIP-39 mnemonic in .env)
    ├─ m/44'/60'/0'/0/{agent_index}    (EVM — all EVM chains share one address)
    └─ m/44'/501'/{agent_index}'/0'    (Solana — separate address space)
"""
```

**Chain registry:** `_CHAINS` dict (as shown above, both EVM and Solana). RPC URL resolved as: env var override → built-in default. The `ecosystem` field routes to the correct backend.

**SQLite schema (`data/wallet.db`):**

```sql
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
CREATE INDEX IF NOT EXISTS idx_tx_agent_ts ON transactions(agent_id, timestamp);
```

**WalletService class:**

```python
class WalletService:
    def __init__(self, db_path="data/wallet.db", event_bus=None):
        self._master_seed = os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED")
        self._chains = self._load_chains()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self._init_schema()
        self._default_per_tx_usd = 10.0
        self._default_daily_usd = 100.0
        self._default_rate_per_hour = 10
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol → (usd, timestamp)
        self._tx_timestamps: dict[str, list[float]] = {}
        self._evm_providers: dict[str, AsyncWeb3] = {}  # lazy, per-chain
        self._solana_clients: dict[str, AsyncClient] = {}  # lazy, per-chain
        self._event_bus = event_bus

    @property
    def is_configured(self) -> bool:
        return self._master_seed is not None

    @property
    def chains(self) -> dict:
        return self._chains

    def close(self) -> None:
        """Shutdown: close all RPC connections and database."""
        for provider in self._evm_providers.values():
            # AsyncWeb3 providers are closed via their session
            pass
        self._evm_providers.clear()
        # Solana AsyncClient instances should be closed
        self._solana_clients.clear()
        self.db.close()
```

**Key derivation (both ecosystems from same seed):**

```python
def _derive_evm_account(self, agent_index: int):
    """BIP-44: m/44'/60'/0'/0/{index}. Returns eth_account.Account (LocalAccount)."""
    from eth_account import Account
    Account.enable_unaudited_hdwallet_features()
    return Account.from_mnemonic(
        self._master_seed,
        account_path=f"m/44'/60'/0'/0/{agent_index}",
    )

def _derive_solana_keypair(self, agent_index: int):
    """Derive Solana keypair from mnemonic using standard derivation.

    Uses mnemonic → seed → HMAC-SHA512 derivation keyed by agent index.
    This produces a deterministic 32-byte Ed25519 seed per agent.

    NOTE: This uses a project-specific derivation (HMAC of BIP-39 seed
    keyed by path string) rather than full SLIP-0010 Ed25519 derivation.
    The mnemonic won't produce the same addresses as Phantom/Solflare —
    which is fine since these are auto-generated agent wallets, never
    imported into external wallets.
    """
    import hashlib
    import hmac as _hmac
    from solders.keypair import Keypair

    # BIP-39 seed from mnemonic (standard 64-byte seed)
    seed = hashlib.pbkdf2_hmac(
        "sha512",
        self._master_seed.encode("utf-8"),
        b"mnemonic",  # BIP-39 standard salt
        2048,
    )
    # Derive per-agent seed using HMAC-SHA512 keyed by derivation path
    path = f"m/44'/501'/{agent_index}'/0'"
    derived = _hmac.new(seed, path.encode("utf-8"), hashlib.sha512).digest()
    # First 32 bytes → Ed25519 seed
    return Keypair.from_seed(derived[:32])

def _derive_account(self, agent_index: int, ecosystem: str):
    """Route to correct derivation based on ecosystem."""
    if ecosystem == "evm":
        return self._derive_evm_account(agent_index)
    elif ecosystem == "solana":
        return self._derive_solana_keypair(agent_index)
    raise ValueError(f"Unknown ecosystem: {ecosystem}")

def _get_or_assign_index(self, agent_id: str) -> int:
    """Get agent's index, assigning next available if new.

    Indices are monotonically increasing and never reused.
    Uses a database transaction to prevent race conditions between
    concurrent agent registrations.
    """
    # Fast path: check existing assignment (no write lock needed)
    row = self.db.execute(
        "SELECT idx FROM agent_index WHERE agent_id = ?", (agent_id,),
    ).fetchone()
    if row:
        return row[0]

    # Slow path: assign new index under write transaction
    with self.db:
        # Re-check inside transaction (another coroutine may have assigned)
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
```

**Public methods (ecosystem-agnostic interface):**

```python
def _require_configured(self) -> None:
    """Guard: raise early if master seed is not set."""
    if not self.is_configured:
        raise ValueError(
            "Wallet not configured — run 'openlegion wallet init' to generate a master seed"
        )

async def get_address(self, agent_id: str, chain: str) -> str:
    """Derive and return the agent's address. Routes to EVM or Solana backend."""
    self._require_configured()
    self._validate_chain(chain)
    ecosystem = self._chains[chain]["ecosystem"]
    index = self._get_or_assign_index(agent_id)
    account = self._derive_account(index, ecosystem)
    if ecosystem == "evm":
        return account.address  # 0x-prefixed checksummed
    elif ecosystem == "solana":
        return str(account.pubkey())  # base58

async def get_balance(self, agent_id: str, chain: str, token: str = "native") -> dict:
    """Query balance via RPC. Returns {balance, symbol, decimals, raw}.
    Does NOT require master seed (balance is public data, only needs address)."""
    self._require_configured()  # need seed to derive the address to check
    self._validate_chain(chain)

async def read_contract(self, agent_id: str, chain: str, contract: str,
                         function: str = "", args: list = None) -> dict:
    """EVM: eth_call with function sig → decoded result.
    Solana: getAccountInfo → {lamports, owner, data, executable}.
    Does not require agent's key — read-only operations."""
    self._validate_chain(chain)

async def transfer(self, agent_id: str, chain: str, to: str, amount: str,
                    token: str = "native", permissions=None) -> dict:
    """Send native/fungible tokens. Routes to EVM or Solana backend."""
    self._require_configured()
    self._validate_chain(chain)
    self._validate_amount(amount)
    # Validate address format per ecosystem
    ecosystem = self._chains[chain]["ecosystem"]
    if ecosystem == "evm":
        to = self._validate_evm_address(to)
    elif ecosystem == "solana":
        self._validate_solana_address(to)

async def execute_contract(self, agent_id: str, chain: str, contract: str = "",
                            function: str = "", args: list = None,
                            value: str = "0", transaction: str = "",
                            permissions=None) -> dict:
    """EVM: contract + function + args → sign → broadcast.
    Solana: transaction (base64 unsigned tx) → sign → broadcast."""
    self._require_configured()
    self._validate_chain(chain)
    ecosystem = self._chains[chain]["ecosystem"]
    if ecosystem == "evm":
        self._validate_evm_address(contract)
        self._validate_function_signature(function)
    elif ecosystem == "solana" and not transaction:
        raise ValueError("Solana wallet_execute requires a base64-encoded unsigned transaction")
```

**Input validation:**

```python
def _validate_chain(self, chain: str) -> None:
    """Reject unknown chain identifiers."""
    if chain not in self._chains:
        raise ValueError(
            f"Unknown chain: {chain}. "
            f"Supported: {', '.join(sorted(self._chains.keys()))}"
        )

def _validate_evm_address(self, address: str) -> str:
    """Validate and return checksummed EVM address. Raises ValueError."""
    from web3 import Web3
    if not address or len(address) != 42 or not address.startswith("0x"):
        raise ValueError(f"Invalid EVM address format: {address}")
    try:
        return Web3.to_checksum_address(address)
    except Exception:
        raise ValueError(f"Invalid EVM address: {address}")

def _validate_solana_address(self, address: str) -> str:
    """Validate Solana base58 address. Raises ValueError."""
    from solders.pubkey import Pubkey
    try:
        Pubkey.from_string(address)
        return address
    except Exception:
        raise ValueError(f"Invalid Solana address: {address}")

def _validate_amount(self, amount: str) -> None:
    """Validate human-readable amount string. Raises ValueError."""
    from decimal import Decimal, InvalidOperation
    if not amount:
        raise ValueError("Amount is required")
    try:
        d = Decimal(amount)
    except InvalidOperation:
        raise ValueError(f"Invalid amount: {amount}")
    if d <= 0:
        raise ValueError(f"Amount must be positive: {amount}")
    if d.as_tuple().exponent < -18:
        raise ValueError(f"Too many decimal places: {amount}")

def _validate_function_signature(self, function_sig: str) -> None:
    """Validate a Solidity function signature format."""
    if not function_sig or "(" not in function_sig or not function_sig.endswith(")"):
        raise ValueError(
            f"Invalid function signature: {function_sig}. "
            f"Expected format: 'functionName(type1,type2)'"
        )
```

**Signing flow (shared by `transfer` and `execute_contract`, routes to correct backend):**

```python
async def _sign_and_send(self, agent_id, chain, tx_params, permissions, audit_info):
    # 1. Estimate USD value (native token only — see spend limit scope note)
    native_value = tx_params.get("native_value", 0)
    value_usd = await self._estimate_value_usd(chain, native_value)

    # 2. Policy check (spend limit, daily limit, rate limit)
    denial = self._check_policy(agent_id, value_usd, permissions)
    if denial:
        self._audit(agent_id, chain, "rejected", error=denial, **audit_info)
        raise PermissionError(denial)

    ecosystem = self._chains[chain]["ecosystem"]

    # 3. Route to ecosystem-specific signing
    if ecosystem == "evm":
        tx_hash = await self._evm_sign_and_send(agent_id, chain, tx_params)
    elif ecosystem == "solana":
        tx_hash = await self._solana_sign_and_send(agent_id, chain, tx_params)
    else:
        raise ValueError(f"Unknown ecosystem: {ecosystem}")

    # 4. Audit log
    self._audit(agent_id, chain, "broadcast", tx_hash=tx_hash,
                value_usd=value_usd, **audit_info)

    # 5. Return (NEVER key, NEVER signature bytes)
    explorer_fmt = self._chains[chain]["explorer_tx_fmt"]
    return {"tx_hash": tx_hash, "chain": chain, "status": "broadcast",
            "explorer_url": explorer_fmt.format(tx_hash=tx_hash)}


async def _evm_sign_and_send(self, agent_id, chain, tx_params) -> str:
    w3 = await self._get_evm_provider(chain)
    account = self._derive_evm_account(self._get_or_assign_index(agent_id))

    # Fill nonce, gas, EIP-1559 fees
    tx = await self._fill_evm_tx(w3, account.address, tx_params, chain)

    # Simulate contract calls (skip for simple value transfers — eth_call
    # doesn't meaningfully validate plain ETH sends, just check balance)
    if tx.get("data") and tx["data"] != b"":
        try:
            await w3.eth.call(tx)
        except Exception as e:
            raise ValueError(f"Transaction simulation failed: {e}")
    else:
        # Simple transfer: verify sufficient balance
        balance = await w3.eth.get_balance(account.address)
        total_cost = tx.get("value", 0) + tx.get("gas", 21000) * tx.get("maxFeePerGas", 0)
        if balance < total_cost:
            raise ValueError(
                f"Insufficient balance: have {w3.from_wei(balance, 'ether')} ETH, "
                f"need ~{w3.from_wei(total_cost, 'ether')} ETH (value + gas)"
            )

    # Sign + broadcast
    signed = account.sign_transaction(tx)
    return (await w3.eth.send_raw_transaction(signed.raw_transaction)).hex()


async def _solana_sign_and_send(self, agent_id, chain, tx_params) -> str:
    """Sign and broadcast a Solana transaction.

    Two paths:
    1. DeFi: agent provides base64 unsigned transaction (from protocol APIs).
       We deserialize the message, sign with agent's keypair, and broadcast.
    2. Transfer: we build the transaction from parameters.
    """
    from solana.rpc.async_api import AsyncClient
    from solana.rpc.types import TxOpts
    from solders.message import MessageV0, Message as LegacyMessage
    from solders.transaction import VersionedTransaction

    client = await self._get_solana_client(chain)
    keypair = self._derive_solana_keypair(self._get_or_assign_index(agent_id))

    if "unsigned_transaction" in tx_params:
        # DeFi path: deserialize message, create signed transaction
        import base64
        tx_bytes = base64.b64decode(tx_params["unsigned_transaction"])
        # Unsigned tx from protocol APIs is a serialized VersionedTransaction
        # with empty/placeholder signatures. We need the message to re-sign.
        unsigned = VersionedTransaction.from_bytes(tx_bytes)
        msg = unsigned.message

        # Validate that the agent's pubkey is a required signer in the message.
        # Prevents confusing errors from submitting a tx constructed for a
        # different wallet (e.g., agent called Jupiter API with wrong pubkey).
        agent_pubkey = keypair.pubkey()
        signer_keys = list(msg.account_keys)[:msg.header.num_required_signatures]
        if agent_pubkey not in signer_keys:
            raise ValueError(
                f"Agent wallet {agent_pubkey} is not a required signer in this transaction. "
                f"Ensure the protocol API was called with this agent's public key."
            )

        # Create new signed transaction
        tx = VersionedTransaction(msg, [keypair])
    else:
        # Simple transfer path: construct transaction from params
        tx = await self._build_solana_transfer(keypair, tx_params, client)

    # Simulate the signed transaction to catch program errors before broadcast
    sim = await client.simulate_transaction(tx)
    if sim.value.err:
        raise ValueError(f"Transaction simulation failed: {sim.value.err}")

    # Broadcast (skip_preflight=True since we already simulated)
    result = await client.send_transaction(tx, opts=TxOpts(skip_preflight=True))
    return str(result.value)


async def _build_solana_transfer(self, keypair, tx_params, client) -> VersionedTransaction:
    """Build a SOL or SPL token transfer transaction."""
    from solders.system_program import TransferParams, transfer
    from solders.message import MessageV0
    from solders.transaction import VersionedTransaction
    from solders.hash import Hash as Blockhash

    to_pubkey = Pubkey.from_string(tx_params["to"])
    amount_lamports = int(Decimal(tx_params["amount"]) * 10**9)

    if tx_params.get("token", "native") == "native":
        # SOL transfer
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
    else:
        # SPL token transfer — uses Associated Token Account pattern
        # Build transfer instruction using spl-token program
        # (implementation handles ATA creation if needed)
        raise NotImplementedError(
            "SPL token transfers not yet supported. "
            "Use wallet_execute with a base64 unsigned transaction from "
            "Jupiter or another DEX API as a workaround."
        )
```

**Provider lazy-init:**

```python
async def _get_evm_provider(self, chain: str):
    """Get or create an AsyncWeb3 provider for the chain. Cached per-chain."""
    if chain in self._evm_providers:
        return self._evm_providers[chain]
    cfg = self._chains[chain]
    rpc_url = os.environ.get(cfg["rpc_env"], cfg["rpc_default"])
    from web3 import AsyncWeb3, AsyncHTTPProvider
    provider = AsyncWeb3(AsyncHTTPProvider(rpc_url))
    self._evm_providers[chain] = provider
    return provider

async def _get_solana_client(self, chain: str):
    """Get or create an AsyncClient for the chain. Cached per-chain."""
    if chain in self._solana_clients:
        return self._solana_clients[chain]
    cfg = self._chains[chain]
    rpc_url = os.environ.get(cfg["rpc_env"], cfg["rpc_default"])
    from solana.rpc.async_api import AsyncClient
    client = AsyncClient(rpc_url)
    self._solana_clients[chain] = client
    return client
```

**EVM transaction building:**

```python
async def _fill_evm_tx(self, w3, from_address: str, tx_params: dict, chain: str) -> dict:
    """Fill in nonce, gas, and EIP-1559 fee fields for an EVM transaction."""
    cfg = self._chains[chain]
    tx = dict(tx_params)
    tx["from"] = from_address
    tx["chainId"] = cfg["chain_id"]

    # Nonce
    tx["nonce"] = await w3.eth.get_transaction_count(from_address)

    # EIP-1559 gas pricing
    latest = await w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", 0)
    priority_fee = await w3.eth.max_priority_fee
    tx["maxFeePerGas"] = int(base_fee * 1.2) + priority_fee  # 20% buffer over base
    tx["maxPriorityFeePerGas"] = priority_fee

    # Gas estimation (10% buffer, minimum 21000 for simple transfers)
    if "gas" not in tx:
        try:
            estimated = await w3.eth.estimate_gas(tx)
            tx["gas"] = max(int(estimated * 1.1), 21000)
        except Exception:
            tx["gas"] = 21000  # fallback for simple transfers

    # Remove 'from' — not part of signed tx, was only needed for estimation
    tx.pop("from", None)
    return tx
```

**Contract call encoding (EVM):**

```python
def _encode_function_call(self, function_sig: str, args: list) -> bytes:
    """Encode a Solidity function call from signature + args.

    function_sig: "transfer(address,uint256)" or "swap((address,address,uint24),uint256)"
    Returns: 4-byte selector + ABI-encoded args

    Uses eth_abi.grammar to parse complex type signatures (nested tuples,
    arrays, etc.) rather than hand-rolled parsing.
    """
    from eth_abi import encode
    from eth_abi.grammar import parse as parse_abi_type
    from web3 import Web3

    self._validate_function_signature(function_sig)

    # Function selector: keccak256 of the canonical signature
    selector = Web3.keccak(text=function_sig)[:4]

    # Extract param types string from "funcName(type1,type2,...)"
    params_str = function_sig[function_sig.index("(") + 1 : -1]
    if not params_str:
        return selector  # No arguments

    # Parse types using eth_abi's grammar parser (handles tuples, arrays, nested types)
    param_types = self._split_abi_params(params_str)
    encoded_args = encode(param_types, args)
    return selector + encoded_args

def _split_abi_params(self, params_str: str) -> list[str]:
    """Split ABI parameter string respecting nested parentheses.

    "address,uint256" → ["address", "uint256"]
    "(address,uint256),bool" → ["(address,uint256)", "bool"]
    "((address,uint256),address),uint256[]" → ["((address,uint256),address)", "uint256[]"]
    """
    result = []
    depth = 0
    current = []
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
```

**Price feed:**

```python
_FALLBACK_PRICES = {"ETH": 3000.0, "POL": 0.50, "SOL": 150.0}
_PRICE_CACHE_TTL = 300  # 5 minutes

async def _estimate_value_usd(self, chain: str, value_smallest_unit) -> float:
    """Convert native token amount (wei/lamports) to USD."""
    cfg = self._chains[chain]
    price = await self._get_price(cfg["symbol"], cfg["coingecko_id"])
    return float(value_smallest_unit / (10 ** cfg["decimals"])) * price

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
    except Exception:
        pass
    return _FALLBACK_PRICES.get(symbol, 3000.0)
```

**Policy engine:**

```python
def _check_policy(self, agent_id: str, value_usd: float, permissions=None) -> str | None:
    """Returns None if allowed, error string if denied."""
    # 1. Per-tx limit
    per_tx, daily, rate = (0, 0, 0)
    if permissions:
        per_tx, daily, rate = permissions.get_wallet_limits(agent_id)
    per_tx = per_tx or self._default_per_tx_usd
    daily = daily or self._default_daily_usd
    rate = rate or self._default_rate_per_hour

    if value_usd > per_tx:
        return f"Per-transaction limit exceeded: ${value_usd:.2f} > ${per_tx:.2f}"

    # 2. Daily limit (sum from wallet.db)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = self.db.execute(
        "SELECT COALESCE(SUM(value_usd), 0) FROM transactions "
        "WHERE agent_id = ? AND timestamp >= ? AND status = 'broadcast'",
        (agent_id, today),
    ).fetchone()
    daily_used = row[0] if row else 0
    if daily_used + value_usd > daily:
        return f"Daily limit exceeded: ${daily_used + value_usd:.2f} > ${daily:.2f}"

    # 3. Rate limit
    now = time.time()
    ts = self._tx_timestamps.setdefault(agent_id, [])
    ts[:] = [t for t in ts if now - t < 3600]
    if len(ts) >= rate:
        return f"Rate limit exceeded: {len(ts)} transactions in the last hour (limit: {rate})"
    ts.append(now)

    return None
```

**Tests:**
- `_derive_evm_account`: deterministic (same seed + index = same address, always)
- `_derive_evm_account`: different index = different address
- `_derive_solana_keypair`: deterministic, produces valid pubkey
- `_derive_solana_keypair`: different index = different pubkey
- `_derive_solana_keypair`: same seed + same index = same keypair always
- `_derive_account` routes correctly based on ecosystem
- `_get_or_assign_index`: monotonic, no reuse, idempotent for same agent
- `_get_or_assign_index`: double-check inside transaction (concurrent safety)
- `_encode_function_call`: simple types (`transfer(address,uint256)`)
- `_encode_function_call`: tuple types for structs (`swap((address,uint256),bool)`)
- `_split_abi_params`: nested parens handled correctly
- `_validate_chain`: rejects unknown chains with helpful error
- `_validate_evm_address`: rejects short, non-hex, missing prefix
- `_validate_solana_address`: rejects invalid base58
- `_validate_amount`: rejects negative, zero, non-numeric, excessive decimals
- `_validate_function_signature`: rejects missing parens, empty string
- `_check_policy`: per-tx limit enforced
- `_check_policy`: daily limit from DB aggregation
- `_check_policy`: rate limit enforced
- `_check_policy`: returns None when all pass
- `_load_chains`: env var override takes priority over default RPC
- `_load_chains`: default RPC used when no env var
- `_load_chains`: both EVM and Solana chains loaded
- `get_address`: valid EVM address (0x, 42 chars) for EVM chains
- `get_address`: valid Solana address (base58) for Solana chains
- `get_balance`: returns formatted balance (mock RPC, both ecosystems)
- `read_contract`: EVM: decodes return value (mock RPC)
- `read_contract`: Solana: returns account info (mock RPC)
- `transfer`: rejects invalid address format (both ecosystems)
- `transfer`: rejects unsupported chain
- `transfer`: rejects negative/zero/non-numeric amount
- `transfer`: EVM simulation failure → ValueError, no signing
- `transfer`: EVM simple transfer checks balance sufficiency
- `transfer`: EVM success → tx_hash + explorer URL (mock RPC)
- `transfer`: Solana SOL transfer → signature + explorer URL (mock RPC)
- `transfer`: Solana simulation failure → ValueError
- `transfer`: audit log written to wallet.db
- `execute_contract`: EVM: encodes function + args correctly
- `execute_contract`: Solana: deserializes unsigned tx message, signs, broadcasts
- `execute_contract`: Solana: rejects when no transaction provided
- `execute_contract`: EVM: rejects when no contract/function provided
- `close()` cleans up providers and database
- Price cache: hit (no HTTP), miss (HTTP + update), fallback on failure
- Explorer URL: format string produces correct URLs (including Solana devnet query param)
- `is_configured` returns False when no seed
- `_require_configured` raises clear error when seed is None
- `transfer` with unconfigured seed → ValueError before any signing
- `execute_contract` with unconfigured seed → ValueError before any signing
- `transfer` calls `_validate_amount` before signing (bad amount never reaches RPC)
- `transfer` calls address validation before signing (bad address never reaches RPC)
- Solana `execute_contract` rejects unsigned tx where agent is not a signer
- Solana `execute_contract` accepts unsigned tx where agent IS a signer
- `_get_evm_provider` uses env var override when set
- `_get_evm_provider` uses default RPC when env var not set
- `_get_evm_provider` caches provider per chain (second call returns same instance)
- `_fill_evm_tx` sets chainId, nonce, EIP-1559 fees
- `_fill_evm_tx` estimates gas with buffer
- SPL token transfer raises NotImplementedError (documented MVP limitation)

---

### Task 3: Mesh Endpoints + MeshClient

**Files:** `src/host/server.py`, `src/agent/mesh_client.py`
**Tests:** `tests/test_wallet_endpoints.py` (new)

**3a. MeshClient methods** (add after vault section in `mesh_client.py`):

```python
# === Wallet (blockchain transactions via mesh signing service) ===

async def wallet_get_address(self, chain: str) -> dict:
    response = await self._get_with_retry(
        f"{self.mesh_url}/mesh/wallet/address",
        params={"agent_id": self.agent_id, "chain": chain},
    )
    response.raise_for_status()
    return response.json()

async def wallet_get_balance(self, chain: str, token: str = "native") -> dict:
    response = await self._get_with_retry(
        f"{self.mesh_url}/mesh/wallet/balance",
        params={"agent_id": self.agent_id, "chain": chain, "token": token},
    )
    response.raise_for_status()
    return response.json()

async def wallet_read_contract(self, chain: str, contract: str,
                                function: str, args: list) -> dict:
    client = await self._get_client()
    response = await client.post(
        f"{self.mesh_url}/mesh/wallet/read",
        json={"agent_id": self.agent_id, "chain": chain, "contract": contract,
              "function": function, "args": args},
        timeout=30, headers=self._trace_headers(),
    )
    response.raise_for_status()
    return response.json()

async def wallet_transfer(self, chain: str, to: str, amount: str,
                           token: str = "native") -> dict:
    client = await self._get_client()
    response = await client.post(
        f"{self.mesh_url}/mesh/wallet/transfer",
        json={"agent_id": self.agent_id, "chain": chain, "to": to,
              "amount": amount, "token": token},
        timeout=60, headers=self._trace_headers(),
    )
    response.raise_for_status()
    return response.json()

async def wallet_execute(self, chain: str, contract: str = "", function: str = "",
                          args: list = None, value: str = "0",
                          transaction: str = "") -> dict:
    """EVM: contract + function + args. Solana: transaction (base64 unsigned tx)."""
    client = await self._get_client()
    body: dict = {"agent_id": self.agent_id, "chain": chain}
    if transaction:
        body["transaction"] = transaction
    else:
        body.update({"contract": contract, "function": function,
                     "args": args or [], "value": value})
    response = await client.post(
        f"{self.mesh_url}/mesh/wallet/execute",
        json=body, timeout=60, headers=self._trace_headers(),
    )
    response.raise_for_status()
    return response.json()
```

**3b. Mesh server endpoints** (add to `create_mesh_app` in `server.py`):

Add `wallet_service` parameter to `create_mesh_app()`.

Add rate limit entries:
```python
"wallet_transfer": (10, 3600),
"wallet_execute": (10, 3600),
```

Add 5 endpoints:

```python
# === Wallet Signing Service ===

@app.get("/mesh/wallet/address")
async def wallet_address(chain: str, agent_id: str, request: Request) -> dict:
    agent_id = _resolve_agent_id(agent_id, request)
    if not permissions.can_use_wallet(agent_id):
        raise HTTPException(403, "Wallet access denied")
    if not permissions.can_use_wallet_chain(agent_id, chain):
        raise HTTPException(403, f"Chain not allowed: {chain}")
    if wallet_service is None:
        raise HTTPException(503, "Wallet service not configured")
    try:
        address = await wallet_service.get_address(agent_id, chain)
        return {"address": address, "chain": chain}
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.get("/mesh/wallet/balance")
async def wallet_balance(chain: str, agent_id: str, request: Request,
                         token: str = "native") -> dict:
    agent_id = _resolve_agent_id(agent_id, request)
    if not permissions.can_use_wallet(agent_id):
        raise HTTPException(403, "Wallet access denied")
    if not permissions.can_use_wallet_chain(agent_id, chain):
        raise HTTPException(403, f"Chain not allowed: {chain}")
    if wallet_service is None:
        raise HTTPException(503, "Wallet service not configured")
    try:
        return await wallet_service.get_balance(agent_id, chain, token)
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/mesh/wallet/read")
async def wallet_read(data: dict, request: Request) -> dict:
    agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
    chain = data.get("chain", "")
    if not permissions.can_use_wallet(agent_id):
        raise HTTPException(403, "Wallet access denied")
    if not permissions.can_use_wallet_chain(agent_id, chain):
        raise HTTPException(403, f"Chain not allowed: {chain}")
    if wallet_service is None:
        raise HTTPException(503, "Wallet service not configured")
    try:
        return await wallet_service.read_contract(
            agent_id, chain, data.get("contract", ""),
            data.get("function", ""), data.get("args", []))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/mesh/wallet/transfer")
async def wallet_transfer(data: dict, request: Request) -> dict:
    agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
    chain = data.get("chain", "")
    if not permissions.can_use_wallet(agent_id):
        raise HTTPException(403, "Wallet access denied")
    if not permissions.can_use_wallet_chain(agent_id, chain):
        raise HTTPException(403, f"Chain not allowed: {chain}")
    if wallet_service is None:
        raise HTTPException(503, "Wallet service not configured")
    await _check_rate_limit("wallet_transfer", agent_id)
    _server_logger.info("Wallet transfer", extra={"extra_data": {
        "agent_id": agent_id, "chain": chain,
        "to": data.get("to", ""), "amount": data.get("amount", "")}})
    try:
        return await wallet_service.transfer(
            agent_id, chain, data.get("to", ""), data.get("amount", ""),
            data.get("token", "native"), permissions)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))

@app.post("/mesh/wallet/execute")
async def wallet_execute(data: dict, request: Request) -> dict:
    agent_id = _resolve_agent_id(data.get("agent_id", ""), request)
    chain = data.get("chain", "")
    contract = data.get("contract", "")
    if not permissions.can_use_wallet(agent_id):
        raise HTTPException(403, "Wallet access denied")
    if not permissions.can_use_wallet_chain(agent_id, chain):
        raise HTTPException(403, f"Chain not allowed: {chain}")
    # Contract allowlist only checked for EVM (Solana uses transaction blobs;
    # program allowlisting requires tx deserialization — Phase 2)
    if contract and not permissions.can_access_wallet_contract(agent_id, contract):
        raise HTTPException(403, f"Contract not allowed: {contract}")
    if wallet_service is None:
        raise HTTPException(503, "Wallet service not configured")
    await _check_rate_limit("wallet_execute", agent_id)
    _server_logger.info("Wallet execute", extra={"extra_data": {
        "agent_id": agent_id, "chain": chain,
        "contract": contract, "function": data.get("function", "")}})
    try:
        return await wallet_service.execute_contract(
            agent_id, chain, contract, data.get("function", ""),
            data.get("args", []), data.get("value", "0"),
            data.get("transaction", ""), permissions)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except PermissionError as e:
        raise HTTPException(403, str(e))
```

**Tests (`tests/test_wallet_endpoints.py`):**
- Each endpoint: auth verified (401 without token)
- Each endpoint: permission denied → 403
- Each endpoint: wallet_service=None → 503
- Chain not allowed → 403
- Address/balance: ValueError from WalletService (bad chain) → 400
- Transfer: rate limited (429 after exceeding)
- Transfer: success returns tx_hash
- Transfer: ValueError from WalletService → 400
- Transfer: PermissionError from WalletService → 403
- Execute: contract not allowed → 403 (EVM with populated allowlist)
- Execute: contract check skipped when contract is empty (Solana path)
- Execute: passes `transaction` field through for Solana
- Read: returns decoded result
- Read: EVM with empty function → WalletService rejects (400)
- Use mock WalletService (AsyncMock)

---

### Task 4: Agent Tools

**Files:** `src/agent/builtins/wallet_tool.py` (new)
**Tests:** `tests/test_wallet_tool.py` (new)

5 `@skill` functions. Each is thin — validates required params, calls mesh_client, catches exceptions.

```python
"""Blockchain wallet tools for agents.

All signing is performed by the mesh wallet service — private keys
never enter the agent container. Uses the same vault proxy pattern
as LLM calls: agent describes what, mesh decides whether and how.
"""

from __future__ import annotations
from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.wallet")


@skill(
    name="wallet_get_address",
    description="Get your wallet address for a blockchain.",
    parameters={
        "chain": {"type": "string", "description": "Chain ID (e.g. 'evm:ethereum', 'evm:base', 'evm:arbitrum', 'evm:polygon', 'solana:mainnet', 'solana:devnet')"},
    },
)
async def wallet_get_address(chain: str, *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain:
        return {"error": "chain is required"}
    try:
        return await mesh_client.wallet_get_address(chain)
    except Exception as e:
        return {"error": f"Failed to get address: {e}"}


@skill(
    name="wallet_get_balance",
    description="Check wallet balance for native tokens (ETH, SOL, POL) or fungible tokens (ERC-20, SPL).",
    parameters={
        "chain": {"type": "string", "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')"},
        "token": {"type": "string", "description": "'native' for ETH/SOL/POL, or token address (ERC-20 contract / SPL mint)", "default": "native"},
    },
)
async def wallet_get_balance(chain: str, token: str = "native", *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain:
        return {"error": "chain is required"}
    try:
        return await mesh_client.wallet_get_balance(chain, token)
    except Exception as e:
        return {"error": f"Failed to get balance: {e}"}


@skill(
    name="wallet_read_contract",
    description=(
        "Read data from a smart contract or account (no transaction, no gas). "
        "EVM: call a view/pure function — provide contract, function signature, and args. "
        "Solana: read account data — provide the account address as 'contract'."
    ),
    parameters={
        "chain": {"type": "string", "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')"},
        "contract": {"type": "string", "description": "EVM: contract address. Solana: account address to read."},
        "function": {"type": "string", "description": "EVM: function signature (e.g. 'balanceOf(address)'). Solana: not required.", "default": ""},
        "args": {"type": "array", "description": "EVM: function arguments in order. Solana: not required.", "default": []},
    },
)
async def wallet_read_contract(chain: str, contract: str, function: str = "",
                                args: list | None = None, *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain or not contract:
        return {"error": "chain and contract are required"}
    # EVM requires function signature; Solana reads account data directly
    if chain.startswith("evm:") and not function:
        return {"error": "EVM chains require a function signature"}
    try:
        return await mesh_client.wallet_read_contract(chain, contract, function, args or [])
    except Exception as e:
        return {"error": f"Contract read failed: {e}"}


@skill(
    name="wallet_transfer",
    description=(
        "Send native tokens or fungible tokens to an address. "
        "Works on both EVM (ETH/ERC-20) and Solana (SOL/SPL). "
        "The transaction is signed by the mesh wallet service — your private key never leaves the vault. "
        "Subject to spend limits and rate limits."
    ),
    parameters={
        "chain": {"type": "string", "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')"},
        "to": {"type": "string", "description": "Recipient address"},
        "amount": {"type": "string", "description": "Amount in human-readable form (e.g. '0.1' for 0.1 ETH, '1.5' for 1.5 SOL)"},
        "token": {"type": "string", "description": "'native' for ETH/SOL/POL, or token address (ERC-20 contract / SPL mint)", "default": "native"},
    },
)
async def wallet_transfer(chain: str, to: str, amount: str,
                           token: str = "native", *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain or not to or not amount:
        return {"error": "chain, to, and amount are required"}
    try:
        return await mesh_client.wallet_transfer(chain, to, amount, token)
    except Exception as e:
        return {"error": f"Transfer failed: {e}"}


@skill(
    name="wallet_execute",
    description=(
        "Execute an onchain transaction. The mesh signs and broadcasts it. "
        "EVM: provide contract, function signature, and args. "
        "Solana: provide a base64-encoded unsigned transaction (from protocol APIs like Jupiter). "
        "Use this for swaps, mints, approvals, staking, lending, or any onchain interaction."
    ),
    parameters={
        "chain": {"type": "string", "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')"},
        "contract": {"type": "string", "description": "EVM: contract address. Not used for Solana.", "default": ""},
        "function": {"type": "string", "description": "EVM: Solidity function signature (e.g. 'approve(address,uint256)'). Not used for Solana.", "default": ""},
        "args": {"type": "array", "description": "EVM: function arguments in order. Not used for Solana.", "default": []},
        "value": {"type": "string", "description": "EVM: native token to send with call. Not used for Solana.", "default": "0"},
        "transaction": {"type": "string", "description": "Solana: base64-encoded unsigned transaction. Not used for EVM.", "default": ""},
    },
)
async def wallet_execute(chain: str, contract: str = "", function: str = "",
                          args: list = None, value: str = "0",
                          transaction: str = "", *, mesh_client=None) -> dict:
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain:
        return {"error": "chain is required"}
    if chain.startswith("evm:") and (not contract or not function):
        return {"error": "EVM chains require contract and function"}
    if chain.startswith("solana:") and not transaction:
        return {"error": "Solana chains require transaction (base64 unsigned tx)"}
    try:
        return await mesh_client.wallet_execute(
            chain, contract, function, args or [], value, transaction)
    except Exception as e:
        return {"error": f"Transaction failed: {e}"}
```

**Tests (`tests/test_wallet_tool.py`):**
- Each tool: returns error when `mesh_client=None`
- Each tool: returns error when required params missing
- Each tool: calls correct mesh_client method with correct args
- Each tool: catches exception, returns error dict
- `wallet_read_contract`: EVM requires function, Solana does not
- `wallet_execute`: EVM requires contract+function, Solana requires transaction
- `wallet_execute`: unknown chain prefix gets through to mesh (validated server-side)
- Use `AsyncMock` for mesh_client

---

### Task 5: CLI + Startup Wiring + Docs

**Files:** `src/cli/main.py`, `src/cli/runtime.py`, `CLAUDE.md`
**Tests:** `tests/test_cli_commands.py` (extend)

**5a. CLI commands** (`src/cli/main.py`):

```python
@cli.group()
def wallet():
    """Manage agent wallets."""

@wallet.command()
def init():
    """Generate a master seed and store in .env. Shows mnemonic ONCE for backup."""
    # Check if seed already exists — refuse to overwrite
    # Generate 24-word BIP-39 mnemonic via mnemonic library
    # Store via _persist_to_env("OPENLEGION_SYSTEM_WALLET_MASTER_SEED", mnemonic)
    # Display mnemonic with clear WARNING to back it up
    # Derive and display first agent's EVM + Solana address as confirmation

@wallet.command()
@click.argument("agent_id", required=False)
def show(agent_id):
    """Show wallet addresses for agents (or a specific agent).
    Displays both EVM and Solana addresses."""
    # Load seed from env, instantiate WalletService
    # If agent_id: show that agent's addresses on all chains
    # If no agent_id: list all agents from wallet.db with their addresses
```

**5b. Startup wiring** (`src/cli/runtime.py`):

```python
wallet_service = None
if os.environ.get("OPENLEGION_SYSTEM_WALLET_MASTER_SEED"):
    from src.host.wallet import WalletService
    wallet_service = WalletService(db_path="data/wallet.db", event_bus=event_bus)
    logger.info("Wallet service initialized (%d chains)", len(wallet_service.chains))
```

Pass `wallet_service=wallet_service` to `create_mesh_app()`.

Add `wallet_service.close()` to shutdown sequence.

Zero impact when not configured — wallet_service is None, endpoints return 503.

**5c. CLAUDE.md updates:**

Add to Module Map:
```
| `src/host/wallet.py` | HD wallet derivation, transaction signing, policy engine |
| `src/agent/builtins/wallet_tool.py` | Agent wallet tools (address, balance, read, transfer, execute) |
```

Add to Security Boundaries:
```
- **Private keys never in agent containers.** Master seed loaded by mesh process only (system-tier). Per-agent keys derived in-memory via BIP-44. Tools return tx_hash, never keys or signatures.
- **Transaction policy enforcement.** Per-tx/daily USD limits, rate limiting, chain allowlists, simulation before every signing.
```

Add to Test File Mapping:
```
| `src/host/wallet.py` | `tests/test_wallet.py` |
| `src/agent/builtins/wallet_tool.py` | `tests/test_wallet_tool.py` |
| Wallet mesh endpoints | `tests/test_wallet_endpoints.py` |
| Wallet permissions | `tests/test_permissions.py` |
```

---

## Dependencies

Add to host requirements (NOT agent container — signing happens in mesh process):

```
web3>=7.0,<8.0
eth-account>=0.13,<1.0
solders>=0.21,<1.0
solana>=0.34,<1.0
```

`eth-account` includes `mnemonic` and `eth_abi` as transitive deps.
`solders` is the Rust-backed Solana types library (fast, no heavy SDK).
`solana` is the Python RPC client.

---

## File Summary

| Action | File | Task |
|---|---|---|
| Modify | `src/shared/types.py` | 1 |
| Modify | `src/host/permissions.py` | 1 |
| Create | `src/host/wallet.py` | 2 |
| Modify | `src/agent/mesh_client.py` | 3 |
| Modify | `src/host/server.py` | 3 |
| Create | `src/agent/builtins/wallet_tool.py` | 4 |
| Modify | `src/cli/main.py` | 5 |
| Modify | `src/cli/runtime.py` | 5 |
| Modify | `CLAUDE.md` | 5 |
| Create | `tests/test_wallet.py` | 2 |
| Create | `tests/test_wallet_tool.py` | 4 |
| Create | `tests/test_wallet_endpoints.py` | 3 |
| Create | `tests/test_permissions.py` | 1 |

---

## Open Questions

1. **Default spend limits** — $10/tx and $100/day. Adjust?
2. **Gas funding** — User manually funds agent wallets. Add a `wallet fund` CLI command later?
3. **Audit retention** — Keep transaction logs forever (user can truncate). OK?
4. **User wallet import** — Defer to Phase 2?

## Known Limitations (MVP)

1. **SPL token transfers** — Not implemented in MVP `_build_solana_transfer`. Agents can use Jupiter API + `wallet_execute` with unsigned transactions as a workaround for any token operation.
2. **Token-aware spend limits** — USD limits apply to native token value only. Pure ERC-20/SPL transfers with zero native value bypass spend limits (but not rate limits). Phase 2 adds token price lookups.
3. **Solana program allowlists** — Contract allowlist only works for EVM (where `contract` is explicit). Solana `wallet_execute` accepts opaque transaction blobs; inspecting program IDs requires deserialization. Phase 2.
4. **Solana key derivation** — Uses HMAC-SHA512 derivation (deterministic, secure) rather than full SLIP-0010 Ed25519 derivation. Addresses won't match Phantom/Solflare for the same mnemonic. Fine for auto-generated agent wallets.

## Future Phases

**Phase 2:** Contract/program allowlists (EVM + Solana), token-aware spend limits, unlimited approval detection, SPL token transfer builder, human-in-the-loop approval, dashboard wallet panel.
**Phase 3:** User wallet import, multi-sig, DeFi helpers (swap router wrappers), additional ecosystems (SUI, Aptos, Cosmos).
