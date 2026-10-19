"""Builds the ASGI app that serves both MCP transports and the artifact route from one process.

The app serves Streamable HTTP at `/mcp` and SSE at `/sse` with `/messages`, and a combined lifespan starts and stops both FastMCP session managers.
"""
from __future__ import annotations

import contextlib
import mimetypes

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

#: URL prefix of the artifact route. The server streams binary artifacts from
#: the local cache here, so a link keeps working after the STS token expires,
#: which a presigned MinIO URL would not.
ARTIFACT_PREFIX = "/artifact"


async def _serve_artifact(request):
    """Return one artifact of a run from the local cache, or a plain-text error.

    The server answers 404 for an unknown run and for a `filename` that does not match exactly one key in the run's own key list, so no other path is read. It answers 502 with the error text when the fetch fails.
    """
    from surimi_model_mcp.tools import runs as sim_tools
    from surimi_model_mcp.simulations import cache

    simulation_id = request.path_params["simulation_id"]
    filename = request.path_params["filename"]

    run, _deg = sim_tools._find_run(simulation_id)
    if not run:
        return PlainTextResponse(
            f"no simulation with id {simulation_id!r}", status_code=404)

    match = [k for k in run["keys"] if k.rsplit("/", 1)[-1] == filename]
    if len(match) != 1:
        detail = "is ambiguous in" if match else "is not an artifact of"
        return PlainTextResponse(
            f"{filename!r} {detail} {simulation_id}", status_code=404)

    try:
        raw = cache.fetch(match[0])
    except Exception as exc:                          # noqa: BLE001
        # This covers an uncached artifact requested after the token expired.
        # The caller gets the error text and not a bare 500.
        return PlainTextResponse(str(exc), status_code=502)

    media, _ = mimetypes.guess_type(filename)
    return Response(raw, media_type=media or "application/octet-stream")


def build_artifact_routes() -> list[Route]:
    """Return the route that serves cached artifact bytes, apart from the MCP transports so that it can be mounted without FastMCP."""
    return [Route(ARTIFACT_PREFIX + "/{simulation_id}/{filename}",
                  _serve_artifact, methods=["GET"])]


def build_asgi_app(mcp, *, http_path: str = "/mcp", sse_path: str = "/sse") -> Starlette:
    """Build a Starlette app that serves `mcp` over streamable HTTP at `http_path` and SSE at `sse_path`, plus the artifact route."""
    http_app = mcp.http_app(path=http_path, transport="http")
    sse_app = mcp.http_app(path=sse_path, transport="sse")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with http_app.router.lifespan_context(app):
            async with sse_app.router.lifespan_context(app):
                yield

    return Starlette(
        routes=[*http_app.routes, *sse_app.routes, *build_artifact_routes()],
        lifespan=lifespan,
    )
