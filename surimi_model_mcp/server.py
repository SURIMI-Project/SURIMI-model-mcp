"""Starts the SURIMI model server, which serves gridded NetCDF analytics, simulation runs and the reference layer over FastMCP.

The server has no database. It reads gridded NetCDF files and region masks from S3 or MinIO and returns aggregated results, so the LLM never sees raw grid cells. Each analytical result carries an HMAC-signed receipt under `SURIMI_RECEIPT_SECRET`, which `model_verify` checks.

Tabular fisheries and economics data is served by SURIMI-data-mcp, and a chat client can attach both servers.
"""
import faulthandler
import os

# A segfault in the NetCDF/HDF5 C libraries ends the process before any Python
# logging runs. This writes the Python stack of every thread to stderr.
faulthandler.enable(all_threads=True)

from surimi_model_mcp import capabilities, sources  # noqa: E402,F401  (sources registers on import)
from surimi_model_mcp.mcp_server import HOST, PORT, log, mcp  # noqa: E402,F401
from surimi_model_mcp.tools import gridded as _nc  # noqa: E402
from surimi_model_mcp.tools import register_all  # noqa: E402

register_all(mcp)


def _prefetch_simulations() -> None:
    """Download the artifacts of every discovered simulation run into the local cache.

    It runs in a background thread at launch, like `_prefetch_cache`, and skips the download when `AWS_ENDPOINT_URL` is not set.
    """
    from surimi_model_mcp.simulations import cache, discovery

    if not os.environ.get("AWS_ENDPOINT_URL"):
        log("sim prefetch skipped: no AWS_ENDPOINT_URL (local mode)")
        return
    payload = discovery.scan()
    keys = [k for run in payload.get("simulations", []) for k in run.get("keys", [])]
    log(f"sim prefetch: warming local cache with {len(keys)} artifact(s) "
         f"from {len(payload.get('simulations', []))} run(s) ...")
    res = cache.prefetch_keys(keys)
    log(f"sim prefetch done: {res['cached']} cached, {res['skipped']} already "
         f"present, {len(res['errors'])} failed")
    for e in res["errors"][:10]:
        log(f"sim prefetch WARN {e['key']}: {e['error']}")


def _prefetch_reference() -> None:
    """Download the reference files (species codes, ecosystem models, contracts) into the local cache.

    It runs in a background thread at launch through `reference/warm.py`, and skips the download when `AWS_ENDPOINT_URL` is not set.
    """
    if not os.environ.get("AWS_ENDPOINT_URL"):
        log("reference prefetch skipped: no AWS_ENDPOINT_URL (local mode)")
        return
    from surimi_model_mcp.reference import warm

    res = warm.warm()
    log(f"reference prefetch done: {res['cached']} cached, {res['skipped']} already "
         f"present, {len(res['errors'])} failed")
    for e in res["errors"][:10]:
        log(f"reference prefetch WARN {e['key']}: {e['error']}")


def _prefetch_cache() -> None:
    """Download every NetCDF file under the default prefix into the local cache.

    It runs in a background thread, so the server starts at once, and tools download on demand until it finishes. The cache directory is set by `SURIMI_NC_CACHE`. The function skips the download when the prefix is not an `s3://` URI or `AWS_ENDPOINT_URL` is not set.
    """
    from surimi_model_mcp.netcdf import io as ncio

    prefix = _nc.DEFAULT_NETCDF_PREFIX
    if not ncio.is_s3_uri(prefix) or not os.environ.get("AWS_ENDPOINT_URL"):
        log("prefetch skipped: no s3:// prefix / AWS_ENDPOINT_URL (local mode)")
        return
    log(f"prefetch: warming local cache from {prefix} ...")
    res = ncio.prefetch(prefix)
    if res.get("auth_error"):
        # The credentials failed and the contents did not, so the legacy
        # prefix would fail the same way. Report the credential problem.
        log(f"prefetch ABORTED: {prefix} could not be listed with the "
             f"credentials this pod holds (expired or invalid STS token). "
             f"Cached files and the last saved listing still serve; a fresh "
             f"launch is needed for new objects.")
    elif (
        not os.environ.get("SURIMI_NETCDF_PREFIX")
        and res["localized"] == 0
        and res["skipped"] == 0
    ):
        # The default prefix holds no files, so warm from the legacy prefix.
        log(f"prefetch: {prefix} empty; falling back to {_nc.LEGACY_NETCDF_PREFIX}")
        res = ncio.prefetch(_nc.LEGACY_NETCDF_PREFIX)
    log(
        f"prefetch done: {res['localized']} file(s) cached, "
        f"{res['skipped']} skipped, {len(res['errors'])} error(s)"
    )
    for e in res["errors"]:
        log(f"prefetch WARN {e['uri']}: {e['error']}")


def main() -> None:
    """Check the tool registry for drift, start the three prefetch threads and serve the app with uvicorn on `HOST` and `PORT`."""
    import asyncio
    import threading

    import uvicorn

    from surimi_model_mcp.app_factory import build_asgi_app

    log("running capabilities drift check...")
    tools = asyncio.run(mcp.list_tools())
    drift = capabilities.drift_check({t.name for t in tools})
    if drift["missing_from_registry"]:
        log(f"WARN drift: tools not claimed by any source: {drift['missing_from_registry']}")
    if drift["missing_from_fastmcp"]:
        log(f"WARN drift: claimed tools missing from FastMCP: {drift['missing_from_fastmcp']}")
    if not drift["missing_from_registry"] and not drift["missing_from_fastmcp"]:
        log(f"drift check OK ({len(tools)} tools, {len(capabilities.all_capabilities())} sources)")

    threading.Thread(target=_prefetch_cache, name="nc-prefetch", daemon=True).start()
    threading.Thread(target=_prefetch_simulations, name="sim-prefetch", daemon=True).start()
    threading.Thread(target=_prefetch_reference, name="ref-prefetch", daemon=True).start()

    log(f"starting SURIMI model MCP on {HOST}:{PORT} (streamable-http at /mcp, sse at /sse)")
    uvicorn.run(build_asgi_app(mcp, http_path="/mcp", sse_path="/sse"), host=HOST, port=PORT)
