"""Holds the FastMCP instance that every tool registers on, the host and port, and the `log` function."""

import os

from fastmcp import FastMCP

#: Port from `MCP_PORT`, default 26611.
PORT = int(os.environ.get("MCP_PORT", "26611"))
HOST = "0.0.0.0"
mcp = FastMCP("SURIMI Model Output")


def log(msg: str) -> None:
    """Print `msg` to stdout with the `[surimi-model-mcp]` prefix, flushed at once."""
    print(f"[surimi-model-mcp] {msg}", flush=True)
