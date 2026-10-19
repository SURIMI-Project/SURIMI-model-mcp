"""Every NetCDF tool runs under one process-wide lock.

The NetCDF-C / HDF5 libraries are not thread-safe, and FastMCP runs sync
tools in a thread pool. On 2026-09-28 at 07:29 UTC model-mcp 1.2.8 died with
a segfault; faulthandler printed the stack inside xarray -> netCDF4 opening
a file from inspect._open. After the restart the process was left unable to
open bd30cm at all ('NetCDF: HDF error' on every call), while the same file
opened cleanly in a fresh process. Serialising the NetCDF work removes the
concurrent access behind both.
"""
import inspect
import threading
import time

from surimi_model_mcp.tools import gridded as nc_tools
from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.netcdf import io as nc_io


def test_every_netcdf_tool_is_serialised():
    for name in nc_tools.TOOLS:
        assert getattr(getattr(nc_tools, name), "_serialized_by_nc_lock", False), name
    assert getattr(sim_tools.model_run_stock_status, "_serialized_by_nc_lock", False)


def test_serialising_keeps_the_signature_fastmcp_reads():
    params = inspect.signature(nc_tools.model_trend).parameters
    assert {"file", "var", "region", "mask_file"} <= set(params)
    assert "simulation_id" in inspect.signature(sim_tools.model_run_stock_status).parameters


def test_concurrent_calls_never_overlap_inside_the_library(tiny_nc, monkeypatch):
    real_open = nc_io.xr.open_dataset
    state = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def tracking_open(*a, **kw):
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.02)
            return real_open(*a, **kw)
        finally:
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(nc_io.xr, "open_dataset", tracking_open)
    errors = []

    def call(i):
        try:
            if i % 2:
                nc_tools.model_describe_file(tiny_nc)
            else:
                nc_tools.model_list_variables(tiny_nc)
        except Exception as exc:          # noqa: BLE001 - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=call, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert state["peak"] == 1
