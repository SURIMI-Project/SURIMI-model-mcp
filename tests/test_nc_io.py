"""Tests for the netcdf IO layer that abstracts local-vs-S3 access.

The deployed MCP must read netcdf from MinIO (s3://project-surimi/netcdf/...),
not just local disk. netcdf.io centralizes open + hashing + existence so the
inspect/analytics/regions modules do not each reimplement S3 handling.
"""
from __future__ import annotations

import hashlib

import pytest
import xarray as xr

from surimi_model_mcp.netcdf import io as ncio


# ---------- local path handling (regression) ----------

def test_open_dataset_opens_local_path(tiny_nc):
    with ncio.open_dataset(tiny_nc) as ds:
        assert "biomass" in ds.data_vars


def test_open_dataset_accepts_file_uri(tiny_nc):
    with ncio.open_dataset("file://" + tiny_nc) as ds:
        assert "biomass" in ds.data_vars


def test_resource_sha256_matches_local_file(tiny_nc):
    expected = hashlib.sha256(open(tiny_nc, "rb").read()).hexdigest()
    assert ncio.resource_sha256(tiny_nc) == expected


def test_resource_sha256_stable_across_calls(tiny_nc):
    assert ncio.resource_sha256(tiny_nc) == ncio.resource_sha256(tiny_nc)


def test_exists_true_for_local(tiny_nc):
    assert ncio.exists(tiny_nc) is True


def test_exists_false_for_missing_local(tmp_path):
    assert ncio.exists(str(tmp_path / "nope.nc")) is False


def test_open_dataset_raises_on_missing_local(tmp_path):
    with pytest.raises(ncio.NetCDFIOError):
        ncio.open_dataset(str(tmp_path / "nope.nc"))


# ---------- S3 dispatch ----------

def test_is_s3_uri():
    assert ncio.is_s3_uri("s3://bucket/key.nc") is True
    assert ncio.is_s3_uri("/tmp/local.nc") is False
    assert ncio.is_s3_uri("file:///tmp/local.nc") is False


def _fake_s3fs(monkeypatch, tiny_nc, captured, cache_dir):
    """Install a fake s3fs whose get_file copies the local fixture, and point
    the cache at a tmp dir so downloads land there."""
    import shutil
    import types

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(cache_dir))

    class _FakeFS:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def get_file(self, uri, local):
            captured["downloaded"] = uri
            shutil.copy(tiny_nc, local)

        def exists(self, uri):
            return True

    monkeypatch.setitem(
        __import__("sys").modules, "s3fs",
        types.SimpleNamespace(S3FileSystem=_FakeFS),
    )


def test_open_dataset_s3_downloads_then_opens(monkeypatch, tiny_nc, tmp_path):
    """s3:// should download via s3fs.get_file to the local cache, then open."""
    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")

    with ncio.open_dataset("s3://project-surimi/NetCDF/x.nc") as ds:
        assert "biomass" in ds.data_vars
    assert captured["downloaded"] == "s3://project-surimi/NetCDF/x.nc"
    assert captured["kwargs"]["client_kwargs"]["endpoint_url"] == "https://minio.test"


def test_open_dataset_s3_caches_second_call(monkeypatch, tiny_nc, tmp_path):
    """Second open of the same URI must NOT re-download (cache hit)."""
    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")
    with ncio.open_dataset("s3://b/k.nc"):
        pass
    captured["downloaded"] = None  # reset
    with ncio.open_dataset("s3://b/k.nc"):
        pass
    assert captured["downloaded"] is None  # no second download


def test_open_dataset_s3_raises_without_endpoint(monkeypatch):
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    with pytest.raises(ncio.NetCDFIOError):
        ncio.open_dataset("s3://bucket/key.nc")


def test_resource_sha256_s3(monkeypatch, tiny_nc, tmp_path):
    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")
    expected = hashlib.sha256(open(tiny_nc, "rb").read()).hexdigest()
    assert ncio.resource_sha256("s3://b/k.nc") == expected


# ---------- prefetch (warm the cache at launch) ----------

def _fake_s3fs_with_ls(monkeypatch, tiny_nc, captured, cache_dir, listing):
    """Like _fake_s3fs but the fake also answers ls() so prefetch can list."""
    import shutil
    import types

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(cache_dir))
    captured.setdefault("downloads", [])

    class _FakeFS:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def ls(self, path, detail=True):
            return listing

        def get_file(self, uri, local):
            captured["downloads"].append(uri)
            shutil.copy(tiny_nc, local)

        def exists(self, uri):
            return True

    monkeypatch.setitem(
        __import__("sys").modules, "s3fs",
        types.SimpleNamespace(S3FileSystem=_FakeFS),
    )


def test_prefetch_downloads_all_nc_under_prefix(monkeypatch, tiny_nc, tmp_path):
    """prefetch localizes every .nc under the prefix and skips non-.nc files."""
    captured = {}
    listing = [
        {"name": "project-surimi/NetCDF/ecoocean.nc", "size": 100, "type": "file"},
        {"name": "project-surimi/NetCDF/mask.nc", "size": 50, "type": "file"},
        {"name": "project-surimi/NetCDF/readme.txt", "size": 5, "type": "file"},
    ]
    _fake_s3fs_with_ls(monkeypatch, tiny_nc, captured, tmp_path / "cache", listing)

    res = ncio.prefetch("s3://project-surimi/NetCDF/")
    assert res["localized"] == 2
    assert res["skipped"] == 1
    assert res["errors"] == []
    assert sorted(captured["downloads"]) == [
        "s3://project-surimi/NetCDF/ecoocean.nc",
        "s3://project-surimi/NetCDF/mask.nc",
    ]


def test_prefetch_second_run_is_cache_hit(monkeypatch, tiny_nc, tmp_path):
    """A second prefetch over the same prefix re-downloads nothing."""
    captured = {}
    listing = [{"name": "b/NetCDF/a.nc", "size": 1, "type": "file"}]
    _fake_s3fs_with_ls(monkeypatch, tiny_nc, captured, tmp_path / "cache", listing)
    ncio.prefetch("s3://b/NetCDF/")
    captured["downloads"].clear()
    res = ncio.prefetch("s3://b/NetCDF/")
    assert res["localized"] == 1
    assert captured["downloads"] == []


def test_prefetch_records_errors_without_raising(monkeypatch, tiny_nc, tmp_path):
    """A failed download is recorded in errors, not raised; others still cache."""
    import shutil
    import types

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.test")
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path / "cache"))

    class _FlakyFS:
        def __init__(self, **kwargs):
            pass

        def ls(self, path, detail=True):
            return [
                {"name": "b/NetCDF/good.nc", "size": 1, "type": "file"},
                {"name": "b/NetCDF/bad.nc", "size": 1, "type": "file"},
            ]

        def get_file(self, uri, local):
            if uri.endswith("bad.nc"):
                raise RuntimeError("boom")
            shutil.copy(tiny_nc, local)

    monkeypatch.setitem(
        __import__("sys").modules, "s3fs",
        types.SimpleNamespace(S3FileSystem=_FlakyFS),
    )

    res = ncio.prefetch("s3://b/NetCDF/")
    assert res["localized"] == 1
    assert len(res["errors"]) == 1
    assert res["errors"][0]["uri"] == "s3://b/NetCDF/bad.nc"


def test_prefetch_local_prefix_is_noop(tmp_path):
    """A non-s3 prefix needs no warm-up (files are already local)."""
    res = ncio.prefetch(str(tmp_path))
    assert res["localized"] == 0
    assert res["local"] is True


# ---------- read freshness (the cache never re-checks the remote) ----------

def test_read_freshness_of_local_file_is_not_stale(tiny_nc):
    """A local path is read from disk on every call, so it is never stale."""
    fr = ncio.read_freshness(tiny_nc)
    assert fr["stale"] is False
    assert fr["note"] is None
    assert fr["as_of"].endswith("Z")


def test_read_freshness_reports_when_the_cache_copy_was_fetched(
    monkeypatch, tiny_nc, tmp_path
):
    """as_of must be the cache file's fetch time, not the time of the call.

    The whole defect is that a cache hit serves bytes downloaded long ago, so
    an as_of taken from the clock would restate the lie the receipt already
    tells.
    """
    import datetime as dt
    import os as _os

    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")
    ncio._localize("s3://b/k.nc")

    local = ncio._cache_path("s3://b/k.nc")
    old = dt.datetime(2026, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc).timestamp()
    _os.utime(local, (old, old))

    fr = ncio.read_freshness("s3://b/k.nc")
    assert fr["as_of"] == "2026-01-02T03:04:05Z"


def test_read_freshness_fresh_s3_read_is_not_stale(monkeypatch, tiny_nc, tmp_path):
    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")
    ncio._localize("s3://b/fresh.nc")

    fr = ncio.read_freshness("s3://b/fresh.nc")
    assert fr["stale"] is False
    assert fr["note"] is None
    assert fr["cached"] is True


def test_read_freshness_old_cache_copy_is_stale(monkeypatch, tiny_nc, tmp_path):
    """An aged cache file is the exact condition the receipt used to hide."""
    import os as _os
    import time as _time

    captured = {}
    _fake_s3fs(monkeypatch, tiny_nc, captured, tmp_path / "cache")
    ncio._localize("s3://b/old.nc")

    local = ncio._cache_path("s3://b/old.nc")
    aged = _time.time() - (ncio.CACHE_STALE_AFTER_S + 60)
    _os.utime(local, (aged, aged))

    fr = ncio.read_freshness("s3://b/old.nc")
    assert fr["stale"] is True
    assert "s3://b/old.nc" in fr["note"]
    assert fr["age_s"] >= ncio.CACHE_STALE_AFTER_S


def test_read_freshness_of_missing_uri_is_not_cached(tmp_path):
    """No file, no claim about freshness: as_of stays None rather than lying."""
    fr = ncio.read_freshness(str(tmp_path / "nope.nc"))
    assert fr["cached"] is False
    assert fr["as_of"] is None
