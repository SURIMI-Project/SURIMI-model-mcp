"""describe_model_capabilities: what the server serves, for a client discovering it."""
from surimi_model_mcp import capabilities
from surimi_model_mcp.mcp_server import log, mcp


async def describe_model_capabilities() -> dict:
    """List data sources, tools, and usage hints for this SURIMI model-output MCP server."""
    log("[meta] describe_model_capabilities")
    tools = await mcp.list_tools()
    return capabilities.describe([t.name for t in tools])
