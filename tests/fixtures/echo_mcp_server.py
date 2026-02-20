"""Minimal MCP server for testing.

Exposes three tools:
  - echo: returns the input text back
  - add: adds two numbers
  - fail: always returns an error

Run directly: python tests/fixtures/echo_mcp_server.py
"""

from mcp.server import FastMCP

server = FastMCP("echo-test-server")


@server.tool()
def echo(text: str) -> str:
    """Echo the input text back."""
    return text


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


@server.tool()
def fail() -> str:
    """Always raises an error for testing error handling."""
    raise ValueError("intentional test error")


if __name__ == "__main__":
    server.run(transport="stdio")
