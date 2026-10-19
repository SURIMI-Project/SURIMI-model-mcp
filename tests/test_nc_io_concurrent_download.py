"""Concurrent first reads of one uncached s3:// file.

Every caller used to download into the same "<cache>.part" path. The first
to finish renamed it, the rest failed with ENOENT at their own rename, and a
late starter's "wb" open truncated the shared file under the one still
writing. On 2026-09-25 at 08:53 the model server died with a segfault while
four questions read the run NetCDF concurrently; the file was not yet in its
cache (its cache copy is timestamped 08:55, written after the restart).
Measured locally: 8 concurrent first reads started 8 downloads and 7 failed.
"""
import os
import threading
import time

import numpy as np
import pytest
import xarray as xr

from surimi_model_mcp.netcdf import io as ncio


@pytest.fixture
def slow_s3(tmp_path, monkeypatch):
    src = tmp_path / "src.nc"
    xr.Dataset({"v": (("t", "y"), np.arange(2000, dtype="float32").reshape(100, 20))}
               ).to_netcdf(src, format="NETCDF4")
    calls = {"n": 0}

    class SlowFS:
        def get_file(self, uri, dst):
            calls["n"] += 1
            with open(src, "rb") as fi, open(dst, "wb") as fo:
                while chunk := fi.read(4096):
                    fo.write(chunk)
                    fo.flush()
                    time.sleep(0.005)

    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(ncio, "_s3_filesystem", lambda: SlowFS())
    return calls


def test_concurrent_first_reads_download_once_and_all_succeed(slow_s3):
    errors, sums = [], []

    def read(i):
        time.sleep(0.01 * i)
        try:
            ds = ncio.open_dataset("s3://bucket/run.nc")
            sums.append(float(ds["v"].sum()))
            ds.close()
        except Exception as exc:          # noqa: BLE001 - the test reports it
            errors.append(repr(exc))

    threads = [threading.Thread(target=read, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert slow_s3["n"] == 1
    assert sums == [float(np.arange(2000).sum())] * 8


def test_a_failed_download_leaves_nothing_behind_and_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path / "cache"))

    class Broken:
        def get_file(self, uri, dst):
            with open(dst, "wb") as fo:
                fo.write(b"half")
            raise OSError("connection reset")

    monkeypatch.setattr(ncio, "_s3_filesystem", lambda: Broken())
    with pytest.raises(ncio.NetCDFIOError):
        ncio.open_dataset("s3://bucket/run.nc")
    assert os.listdir(tmp_path / "cache") == []


def test_the_server_prints_a_stack_if_it_segfaults():
    """The 08:53 crash left no trace: a segfault kills the process before any
    Python logging runs. faulthandler writes every thread's stack to stderr.
    Checked in a fresh interpreter, because pytest enables faulthandler
    itself and would make this pass on its own."""
    import subprocess
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run(
        [sys.executable, "-c",
         "import faulthandler; from surimi_model_mcp import server; print(faulthandler.is_enabled())"],
        cwd=here, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": here, "PYTHONFAULTHANDLER": ""})
    assert out.stdout.strip().splitlines()[-1] == "True", out.stderr[-500:]
