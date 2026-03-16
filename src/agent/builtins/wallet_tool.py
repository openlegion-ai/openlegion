"""Blockchain wallet tools for agents.

All signing is performed by the mesh wallet service — private keys
never enter the agent container.  Uses the same vault proxy pattern
as LLM calls: agent describes what, mesh decides whether and how.
"""

from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("agent.wallet")


@skill(
    name="wallet_get_address",
    description="Get your wallet address for a blockchain.",
    parameters={
        "chain": {
            "type": "string",
            "description": (
                "Chain ID (e.g. 'evm:ethereum', 'evm:base', 'evm:arbitrum', "
                "'evm:polygon', 'solana:mainnet', 'solana:devnet')"
            ),
        },
    },
)
async def wallet_get_address(chain: str, *, mesh_client=None) -> dict:
    """Get agent's wallet address for a chain."""
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain:
        return {"error": "chain is required"}
    try:
        return await mesh_client.wallet_get_address(chain)
    except Exception as e:
        return {"error": sanitize_for_prompt(f"Failed to get address: {e}")}


@skill(
    name="wallet_get_balance",
    description=(
        "Check wallet balance for native tokens (ETH, SOL, POL) "
        "or fungible tokens (ERC-20, SPL)."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')",
        },
        "token": {
            "type": "string",
            "description": (
                "'native' for ETH/SOL/POL, or token address "
                "(ERC-20 contract / SPL mint)"
            ),
            "default": "native",
        },
    },
)
async def wallet_get_balance(
    chain: str, token: str = "native", *, mesh_client=None,
) -> dict:
    """Check wallet balance."""
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain:
        return {"error": "chain is required"}
    try:
        return await mesh_client.wallet_get_balance(chain, token)
    except Exception as e:
        return {"error": sanitize_for_prompt(f"Failed to get balance: {e}")}


@skill(
    name="wallet_read_contract",
    description=(
        "Read data from a smart contract or account (no transaction, no gas). "
        "EVM: call a view/pure function — provide contract, function signature, and args. "
        "Solana: read account data — provide the account address as 'contract'."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')",
        },
        "contract": {
            "type": "string",
            "description": "EVM: contract address. Solana: account address to read.",
        },
        "function": {
            "type": "string",
            "description": (
                "EVM: function signature (e.g. 'balanceOf(address)'). "
                "Solana: not required."
            ),
            "default": "",
        },
        "args": {
            "type": "array",
            "description": "EVM: function arguments in order. Solana: not required.",
            "default": [],
        },
    },
)
async def wallet_read_contract(
    chain: str,
    contract: str,
    function: str = "",
    args: list | None = None,
    *,
    mesh_client=None,
) -> dict:
    """Read-only contract/account data."""
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain or not contract:
        return {"error": "chain and contract are required"}
    if chain.startswith("evm:") and not function:
        return {"error": "EVM chains require a function signature"}
    try:
        return await mesh_client.wallet_read_contract(
            chain, contract, function, args or [],
        )
    except Exception as e:
        return {"error": sanitize_for_prompt(f"Contract read failed: {e}")}


@skill(
    name="wallet_transfer",
    description=(
        "Send native tokens or fungible tokens to an address. "
        "Works on both EVM (ETH/ERC-20) and Solana (SOL/SPL). "
        "The transaction is signed by the mesh wallet service — "
        "your private key never leaves the vault. "
        "Subject to spend limits and rate limits."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')",
        },
        "to": {
            "type": "string",
            "description": "Recipient address",
        },
        "amount": {
            "type": "string",
            "description": (
                "Amount in human-readable form "
                "(e.g. '0.1' for 0.1 ETH, '1.5' for 1.5 SOL)"
            ),
        },
        "token": {
            "type": "string",
            "description": (
                "'native' for ETH/SOL/POL, or token address "
                "(ERC-20 contract / SPL mint)"
            ),
            "default": "native",
        },
    },
)
async def wallet_transfer(
    chain: str,
    to: str,
    amount: str,
    token: str = "native",
    *,
    mesh_client=None,
) -> dict:
    """Send tokens.  Signed by mesh, never the agent."""
    if not mesh_client:
        return {"error": "Wallet tools require mesh connectivity"}
    if not chain or not to or not amount:
        return {"error": "chain, to, and amount are required"}
    try:
        return await mesh_client.wallet_transfer(chain, to, amount, token)
    except Exception as e:
        return {"error": sanitize_for_prompt(f"Transfer failed: {e}")}


@skill(
    name="wallet_execute",
    description=(
        "Execute an onchain transaction. The mesh signs and broadcasts it. "
        "EVM: provide contract, function signature, and args. "
        "Solana: provide a base64-encoded unsigned transaction "
        "(from protocol APIs like Jupiter). "
        "Use this for swaps, mints, approvals, staking, lending, "
        "or any onchain interaction."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": "Chain ID (e.g. 'evm:base', 'solana:mainnet')",
        },
        "contract": {
            "type": "string",
            "description": "EVM: contract address. Not used for Solana.",
            "default": "",
        },
        "function": {
            "type": "string",
            "description": (
                "EVM: Solidity function signature "
                "(e.g. 'approve(address,uint256)'). Not used for Solana."
            ),
            "default": "",
        },
        "args": {
            "type": "array",
            "description": "EVM: function arguments in order. Not used for Solana.",
            "default": [],
        },
        "value": {
            "type": "string",
            "description": "EVM: native token to send with call. Not used for Solana.",
            "default": "0",
        },
        "transaction": {
            "type": "string",
            "description": (
                "Solana: base64-encoded unsigned transaction. Not used for EVM."
            ),
            "default": "",
        },
    },
)
async def wallet_execute(
    chain: str,
    contract: str = "",
    function: str = "",
    args: list | None = None,
    value: str = "0",
    transaction: str = "",
    *,
    mesh_client=None,
) -> dict:
    """Execute an onchain transaction.  Signed by mesh, never the agent."""
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
            chain, contract, function, args or [], value, transaction,
        )
    except Exception as e:
        return {"error": sanitize_for_prompt(f"Transaction failed: {e}")}
