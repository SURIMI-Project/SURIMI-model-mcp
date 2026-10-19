"""The HTTP routes the server answers on (2026-10-13).

The chat connects to /sse, other MCP clients to /mcp, and run artifacts are served under /artifact.
Pinned before the package restructure so the move cannot drop or rename one.
"""
from surimi_model_mcp import server as server
from surimi_model_mcp.app_factory import build_asgi_app


def test_the_server_answers_on_the_same_paths():
    app = build_asgi_app(server.mcp, http_path="/mcp", sse_path="/sse")
    paths = sorted(getattr(r, "path", "") for r in app.routes)
    assert paths == ["/artifact/{simulation_id}/{filename}", "/mcp", "/messages", "/sse"]
