"""The MCP tools, by group: gridded data, simulation runs, reference lookups and the overview."""


def register_all(mcp) -> None:
    """Register the 26 tools in the order the server has always listed them."""
    from surimi_model_mcp.tools import gridded, lookups, overview, runs

    mcp.tool()(overview.describe_model_capabilities)
    for module in (gridded, runs, lookups):
        for name in module.TOOLS:
            mcp.tool(name=name)(getattr(module, name))
