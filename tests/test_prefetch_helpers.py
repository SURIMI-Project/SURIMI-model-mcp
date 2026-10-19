"""The three start-up copies run their real bodies, with only the storage faked (2026-10-13).

They run in daemon threads, so an exception there prints a traceback while the server keeps serving:
the same silent failure the data server's restructure produced. The entry-point test starts the server
without storage, where each helper returns before its lazy imports. This runs them with storage
configured and each storage call replaced, so a broken import or a renamed function fails here.
"""
import pytest

from surimi_model_mcp import server
from surimi_model_mcp.netcdf import io as ncio
from surimi_model_mcp.reference import warm
from surimi_model_mcp.simulations import cache, discovery


@pytest.fixture(autouse=True)
def _storage_configured(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example")
    monkeypatch.delenv("SURIMI_NETCDF_PREFIX", raising=False)
    # Other tests reload the gridded module with a local prefix and leave it set; pin the bucket
    # prefixes here so these tests do not depend on test order.
    monkeypatch.setattr(server._nc, "DEFAULT_NETCDF_PREFIX", "s3://project-surimi/netcdf/")
    monkeypatch.setattr(server._nc, "LEGACY_NETCDF_PREFIX", "s3://project-surimi/NetCDF/")


def test_the_simulation_copy_reads_the_run_listing(monkeypatch, capsys):
    monkeypatch.setattr(discovery, "scan", lambda: {"simulations": [{"keys": ["a/1.csv", "a/2.csv"]}]})
    monkeypatch.setattr(cache, "prefetch_keys", lambda keys: {"cached": len(keys), "skipped": 0, "errors": []})
    server._prefetch_simulations()
    out = capsys.readouterr().out
    assert "warming local cache with 2 artifact(s) from 1 run(s)" in out
    assert "sim prefetch done: 2 cached" in out


def test_the_reference_copy_runs_warm(monkeypatch, capsys):
    monkeypatch.setattr(warm, "warm", lambda: {"cached": 3, "skipped": 1, "errors": [{"key": "x", "error": "gone"}]})
    server._prefetch_reference()
    out = capsys.readouterr().out
    assert "reference prefetch done: 3 cached, 1 already present, 1 failed" in out
    assert "reference prefetch WARN x: gone" in out


def test_the_gridded_copy_falls_back_to_the_legacy_prefix_when_the_canonical_one_is_empty(monkeypatch, capsys):
    seen = []

    def _prefetch(prefix):
        seen.append(prefix)
        empty = prefix == server._nc.DEFAULT_NETCDF_PREFIX
        return {"localized": 0 if empty else 4, "skipped": 0, "errors": []}

    monkeypatch.setattr(ncio, "prefetch", _prefetch)
    server._prefetch_cache()
    out = capsys.readouterr().out
    assert seen == [server._nc.DEFAULT_NETCDF_PREFIX, server._nc.LEGACY_NETCDF_PREFIX]
    assert "prefetch done: 4 file(s) cached" in out


def test_the_gridded_copy_names_the_credentials_when_listing_is_refused(monkeypatch, capsys):
    monkeypatch.setattr(ncio, "prefetch", lambda prefix: {"auth_error": True, "localized": 0, "skipped": 0, "errors": []})
    server._prefetch_cache()
    assert "prefetch ABORTED" in capsys.readouterr().out
