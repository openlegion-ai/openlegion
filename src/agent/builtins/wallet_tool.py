"""Blockchain wallet tools for agents.

All signing is performed by the mesh wallet service — private keys
never enter the agent container.  Uses the same vault proxy pattern
as LLM calls: agent describes what, mesh decides whether and how.
"""

from __future__ import annotations

from src.agent.skills import skill
from src.shared.utils import sanitize_for_prompt, setup_logging

logger = setup_logging("agent.wallet")


_CHAIN_DESC = (
    "One of: 'evm:ethereum', 'evm:base', 'evm:arbitrum', "
    "'evm:polygon', 'evm:sepolia', 'solana:mainnet', 'solana:devnet'"
)


@skill(
    name="wallet_get_address",
    description=(
        "Returns your wallet address on a specific chain. "
        "Use this to share your address, verify which wallet you control, "
        "or pass your address to protocol APIs."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": _CHAIN_DESC,
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
        "Check your wallet balance. Returns the amount in human-readable form. "
        "Use token='native' for ETH/SOL, or pass a token contract address "
        "for tokens like USDC, WETH, etc."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": _CHAIN_DESC,
        },
        "token": {
            "type": "string",
            "description": (
                "'native' for the chain's currency (ETH, SOL, POL), "
                "or a token contract address (e.g. USDC address)"
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
        "Read onchain data without sending a transaction. "
        "EVM: call a contract read function (e.g. check prices, allowances, pool reserves). "
        "Solana: read account data (lamports, owner, raw data)."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": _CHAIN_DESC,
        },
        "contract": {
            "type": "string",
            "description": "EVM: contract address. Solana: account address.",
        },
        "function": {
            "type": "string",
            "description": (
                "EVM only. Solidity function signature, "
                "e.g. 'balanceOf(address)', 'getReserves()', 'allowance(address,address)'. "
                "Leave empty for Solana."
            ),
            "default": "",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "EVM only. Arguments matching the function signature, in order. Leave empty for Solana.",
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
        "Send tokens to an address. Use this for simple sends — "
        "ETH, SOL, USDC, or any token. For complex operations like "
        "swaps or contract calls, use wallet_execute instead."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": _CHAIN_DESC,
        },
        "to": {
            "type": "string",
            "description": "Recipient wallet address (0x... for EVM, base58 for Solana)",
        },
        "amount": {
            "type": "string",
            "description": "Amount as a decimal string (e.g. '0.1' for 0.1 ETH, '100' for 100 USDC)",
        },
        "token": {
            "type": "string",
            "description": (
                "'native' for chain currency (ETH, SOL, POL), "
                "or token contract address for other tokens"
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
        "Call a smart contract or sign a protocol transaction. "
        "Use this for swaps, approvals, mints, staking, lending — "
        "anything beyond a simple token transfer. "
        "EVM: provide the contract address and Solidity function signature. "
        "Solana: provide the base64 unsigned transaction from a protocol API (e.g. Jupiter swap API)."
    ),
    parameters={
        "chain": {
            "type": "string",
            "description": _CHAIN_DESC,
        },
        "contract": {
            "type": "string",
            "description": "EVM only. Target contract address.",
            "default": "",
        },
        "function": {
            "type": "string",
            "description": (
                "EVM only. Solidity function signature, "
                "e.g. 'approve(address,uint256)', 'swap(address,uint256,uint256,address,uint256)'"
            ),
            "default": "",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "EVM only. Arguments matching the function signature, in order.",
            "default": [],
        },
        "value": {
            "type": "string",
            "description": "EVM only. Native token (ETH) to send with the call, as a decimal string. Default '0'.",
            "default": "0",
        },
        "transaction": {
            "type": "string",
            "description": "Solana only. Base64-encoded unsigned transaction from a protocol API.",
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
