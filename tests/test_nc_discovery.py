"""Tests for netcdf discovery: list the catalog so the LLM can find data
from a plain-English question instead of being handed s3:// paths."""
from __future__ import annotations

import pytest

from surimi_model_mcp.netcdf import io as ncio
from surimi_model_mcp.netcdf.discovery import classify, list_netcdf_files


# ---------- classify ----------

def test_classify_data_file():
    assert classify("s3://b/NetCDF/ecoocean_..._bd30cm_global_monthly_1961_2015.nc") == "data"


def test_classify_mask_file():
    assert classify("s3://b/NetCDF/ar6_ocean_mask_0.25deg.nc") == "mask"


def test_classify_non_netcdf_is_other():
    assert classify("s3://b/NetCDF/.keep") == "other"
    assert classify("s3://b/NetCDF/readme.txt") == "other"


# ---------- io.list_objects (local) ----------

def test_list_objects_local_dir(tmp_path):
    (tmp_path / "a.nc").write_bytes(b"x")
    (tmp_path / "b_mask.nc").write_bytes(b"yy")
    (tmp_path / "ignore.txt").write_bytes(b"z")
    objs = ncio.list_objects(str(tmp_path))
    uris = sorted(o["uri"].rsplit("/", 1)[-1] for o in objs)
    assert uris == ["a.nc", "b_mask.nc", "ignore.txt"]
    a = next(o for o in objs if o["uri"].endswith("a.nc"))
    assert a["size"] == 1


def test_list_objects_local_missing_dir(tmp_path):
    with pytest.raises(ncio.NetCDFIOError):
        ncio.list_objects(str(tmp_path / "nope"))


# ---------- io.list_objects (s3, mocked) ----------

def test_list_objects_s3(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.test")
    import types

    class _FakeFS:
        def __init__(self, **kw):
            pass

        def ls(self, path, detail=True):
            assert path == "project-surimi/NetCDF"
            return [
                {"name": "project-surimi/NetCDF/eco_bd30cm.nc", "size": 500, "type": "file"},
                {"name": "project-surimi/NetCDF/ar6_ocean_mask.nc", "size": 60, "type": "file"},
                {"name": "project-surimi/NetCDF/sub", "size": 0, "type": "directory"},
            ]

    monkeypatch.setitem(__import__("sys").modules, "s3fs",
                        types.SimpleNamespace(S3FileSystem=_FakeFS))
    objs = ncio.list_objects("s3://project-surimi/NetCDF/")
    uris = sorted(o["uri"] for o in objs)
    assert uris == [
        "s3://project-surimi/NetCDF/ar6_ocean_mask.nc",
        "s3://project-surimi/NetCDF/eco_bd30cm.nc",
    ]  # the directory entry is excluded


# ---------- list_netcdf_files ----------

def test_list_netcdf_files_classifies(tmp_path):
    (tmp_path / "eco_bd30cm.nc").write_bytes(b"x")
    (tmp_path / "ar6_ocean_mask.nc").write_bytes(b"y")
    (tmp_path / ".keep").write_bytes(b"")
    out = list_netcdf_files(str(tmp_path))
    assert out["count"] == 3
    kinds = {f["uri"].rsplit("/", 1)[-1]: f["kind"] for f in out["files"]}
    assert kinds["eco_bd30cm.nc"] == "data"
    assert kinds["ar6_ocean_mask.nc"] == "mask"
    assert kinds[".keep"] == "other"


def test_list_netcdf_files_records_prefix(tmp_path):
    out = list_netcdf_files(str(tmp_path))
    assert out["prefix"] == str(tmp_path)
    assert out["files"] == []
    assert out["count"] == 0


# ---------- nc_tools.model_list_files ----------

def test_nc_list_files_tool_default_prefix(monkeypatch, tmp_path):
    # point the default prefix at a local dir via env
    (tmp_path / "eco_bd30cm.nc").write_bytes(b"x")
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(tmp_path))
    import importlib
    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "no-bundled-samples"))
    out = nc_tools.model_list_files()
    assert out["count"] == 1
    assert out["files"][0]["kind"] == "data"


def test_nc_list_files_registered():
    from surimi_model_mcp.tools import gridded as nc_tools
    assert "model_list_files" in nc_tools.TOOLS


# ---------- snake_case default prefix + legacy fallback ----------
# SURIMI naming convention: shared-bucket directories are lower_snake_case.
# The canonical prefix is netcdf/; NetCDF/ is the legacy location kept as a
# transitional fallback until the bucket objects are migrated.

def test_default_prefix_is_snake_case(monkeypatch):
    monkeypatch.delenv("SURIMI_NETCDF_PREFIX", raising=False)
    import importlib
    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    assert nc_tools.DEFAULT_NETCDF_PREFIX == "s3://project-surimi/netcdf/"
    assert nc_tools.LEGACY_NETCDF_PREFIX == "s3://project-surimi/NetCDF/"


def test_fallback_prefers_primary_when_it_has_files(tmp_path):
    from surimi_model_mcp.netcdf.discovery import list_netcdf_files_with_fallback
    primary = tmp_path / "netcdf"
    legacy = tmp_path / "NetCDF"
    primary.mkdir()
    legacy.mkdir()
    (primary / "eco.nc").write_bytes(b"x")
    (legacy / "old.nc").write_bytes(b"y")
    out = list_netcdf_files_with_fallback(str(primary), str(legacy))
    assert out["prefix"] == str(primary)
    assert out["count"] == 1
    assert "note" not in out


def test_fallback_uses_legacy_when_primary_empty(tmp_path):
    from surimi_model_mcp.netcdf.discovery import list_netcdf_files_with_fallback
    primary = tmp_path / "netcdf"
    legacy = tmp_path / "NetCDF"
    primary.mkdir()
    legacy.mkdir()
    (legacy / "eco.nc").write_bytes(b"x")
    out = list_netcdf_files_with_fallback(str(primary), str(legacy))
    assert out["prefix"] == str(legacy)
    assert out["count"] == 1
    assert str(primary) in out["note"]


def test_fallback_uses_legacy_when_primary_missing(tmp_path):
    from surimi_model_mcp.netcdf.discovery import list_netcdf_files_with_fallback
    legacy = tmp_path / "NetCDF"
    legacy.mkdir()
    (legacy / "eco.nc").write_bytes(b"x")
    out = list_netcdf_files_with_fallback(str(tmp_path / "netcdf"), str(legacy))
    assert out["prefix"] == str(legacy)
    assert out["count"] == 1


def test_fallback_returns_empty_primary_when_both_empty(tmp_path):
    from surimi_model_mcp.netcdf.discovery import list_netcdf_files_with_fallback
    primary = tmp_path / "netcdf"
    legacy = tmp_path / "NetCDF"
    primary.mkdir()
    legacy.mkdir()
    out = list_netcdf_files_with_fallback(str(primary), str(legacy))
    assert out["prefix"] == str(primary)
    assert out["count"] == 0


def test_nc_list_files_without_env_falls_back_to_legacy(monkeypatch, tmp_path):
    monkeypatch.delenv("SURIMI_NETCDF_PREFIX", raising=False)
    import importlib
    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    primary = tmp_path / "netcdf"
    legacy = tmp_path / "NetCDF"
    primary.mkdir()
    legacy.mkdir()
    (legacy / "eco_bd30cm.nc").write_bytes(b"x")
    monkeypatch.setattr(nc_tools, "DEFAULT_NETCDF_PREFIX", str(primary))
    monkeypatch.setattr(nc_tools, "LEGACY_NETCDF_PREFIX", str(legacy))
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "no-bundled-samples"))
    out = nc_tools.model_list_files()
    assert out["count"] == 1
    assert out["prefix"] == str(legacy)


def test_nc_list_files_env_override_disables_fallback(monkeypatch, tmp_path):
    # An explicit SURIMI_NETCDF_PREFIX pins the catalog; no legacy fallback.
    primary = tmp_path / "pinned"
    primary.mkdir()
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(primary))
    import importlib
    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "no-bundled-samples"))
    out = nc_tools.model_list_files()
    assert out["prefix"] == str(primary)
    assert out["count"] == 0


# ---------- experiment output is a second catalog location ----------
# The aggregator writes a run to surimi-aggregator/experiments/<id>/ (see
# SURIMI-aggregator/server/netcdf_worker_main.py:36) while this server listed
# only netcdf/. Every experiment output was therefore invisible to the server
# whose job is to serve it. The two are not primary-and-fallback: netcdf/ holds
# the forcing and mask files, experiments/ holds model output, and a question
# about a run needs both listed.

def test_an_explicit_prefix_lists_only_that_prefix(monkeypatch, tmp_path):
    """Asking for one location must not silently union in another."""
    catalog = tmp_path / "catalog"
    runs = tmp_path / "runs"
    catalog.mkdir()
    runs.mkdir()
    (catalog / "a.nc").write_bytes(b"x")
    (runs / "b.nc").write_bytes(b"y")
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    out = nc_tools.model_list_files(prefix=str(catalog))
    assert out["count"] == 1
    assert out["files"][0]["uri"].endswith("a.nc")


# ---------- run output, found where the runs actually are ----------
# The first SURIMI run landed on 2026-09-15 under output_creator/<uuid>/, not
# under the aggregator's experiments/ prefix, so the prefix union alone left a
# 74 MB result invisible to the file catalog. The simulation layer had already
# found it: it scans the whole bucket and groups on any UUID directory segment.
# Reuse that scan rather than guessing a second prefix.

def _fake_scan(keys_and_sizes, model="output_creator",
               run_id="9eeadb70-6ef8-43ea-93ad-ee7b32dceca0"):
    return {"simulations": [{
        "model": model, "simulation_id": run_id,
        "n_artifacts": len(keys_and_sizes),
        "keys": list(keys_and_sizes), "sizes": dict(keys_and_sizes),
        "kinds": {}, "last_modified": "2026-09-15T21:55:50Z",
    }]}


def test_list_files_includes_netcdf_a_run_wrote(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "forcing_bd30cm.nc").write_bytes(b"x")
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(catalog))
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: _fake_scan({
        "output_creator/9eeadb70-6ef8-43ea-93ad-ee7b32dceca0/out.nc": 74723971,
        "output_creator/9eeadb70-6ef8-43ea-93ad-ee7b32dceca0/out.bin": 330438823,
    }))
    out = nc_tools.model_list_files()
    uris = {f["uri"] for f in out["files"]}
    assert "s3://project-surimi/output_creator/9eeadb70-6ef8-43ea-93ad-ee7b32dceca0/out.nc" in uris
    assert not any(u.endswith(".bin") for u in uris), "only netcdf belongs in this catalog"
    assert any(u.endswith("forcing_bd30cm.nc") for u in uris), "catalog must survive"


def test_a_run_file_names_the_run_that_wrote_it(monkeypatch, tmp_path):
    """A uri alone strands the caller: they can see the file and not ask the
    simulation tools anything about it."""
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(catalog))
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: _fake_scan(
        {"output_creator/9eeadb70-6ef8-43ea-93ad-ee7b32dceca0/out.nc": 5}))
    entry = [f for f in nc_tools.model_list_files()["files"]
             if f["uri"].endswith("out.nc")][0]
    assert entry["simulation_id"] == "9eeadb70-6ef8-43ea-93ad-ee7b32dceca0"
    assert entry["model"] == "output_creator"
    assert entry["kind"] == "data"
    assert entry["size"] == 5


def test_a_failed_run_scan_does_not_take_the_catalog_down(monkeypatch, tmp_path):
    """The catalog is the older, more reliable half. A scan outage must degrade
    the listing and name the degradation, not empty it."""
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "forcing_bd30cm.nc").write_bytes(b"x")
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(catalog))
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)

    def boom():
        raise RuntimeError("listing refused")

    monkeypatch.setattr(nc_tools, "_scan_runs", boom)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "no-bundled-samples"))
    out = nc_tools.model_list_files()
    assert out["count"] == 1
    assert any("run output" in n for n in out.get("notes") or []), out.get("notes")


def test_a_run_file_already_in_the_catalog_is_listed_once(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "out.nc").write_bytes(b"x")
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(catalog))
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    uri = f"{catalog}/out.nc"
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: _fake_scan({uri: 1}))
    out = nc_tools.model_list_files()
    assert [f["uri"] for f in out["files"]].count(uri) == 1


def test_a_healthy_listing_carries_no_notes(monkeypatch, tmp_path):
    """Live on 1.1.0, every model_list_files reply carried a note saying the
    aggregator's experiments prefix could not be listed. Nothing was missing:
    the run scan had already found the output. A standing false alarm on the
    most-called tool is worse than cosmetic, because the model quotes it back
    to the user as a caveat.
    """
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "forcing_bd30cm.nc").write_bytes(b"x")
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(catalog))
    import importlib

    from surimi_model_mcp.tools import gridded as nc_tools
    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: _fake_scan(
        {"output_creator/experiments/9eeadb70-6ef8-43ea-93ad-ee7b32dceca0/o.nc": 7}))
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "no-bundled-samples"))
    out = nc_tools.model_list_files()
    assert out["count"] == 2
    assert not out.get("notes"), out["notes"]


# ---------- the bundled western-Med sample answers a size question ----------
# westmed-2013-2017-biomass-catch-sales.nc is bundled under
# data/netcdf-samples/ and discovered by _reference_sample_files() -- see
# tests/test_reference_samples.py for that discovery mechanism, and
# tests/test_listing_marks_origin.py for the surimi-sample origin override.
# This one just holds the explicit-prefix isolation rule, the same rule run
# output is already held to.

def test_known_sample_not_listed_for_an_explicit_prefix(tmp_path):
    """Asking for one location lists exactly that location -- same rule
    test_an_explicit_prefix_lists_only_that_prefix already holds run output to."""
    from surimi_model_mcp.tools import gridded as nc_tools
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    out = nc_tools.model_list_files(prefix=str(catalog))
    assert not any(f["uri"].endswith("westmed-2013-2017-biomass-catch-sales.nc")
                   for f in out["files"])


# ---------- listings survive an expired token ----------
# The cold-copy design caches FILE CONTENTS at boot so reads survive the ~24h
# STS expiry. It never cached the LISTING, so model_list_files -- a live LIST
# against S3 -- went dark while the files it would have named sat on local
# disk. Observed 2026-09-17: "The Access Key Id you provided does not exist".

def test_listing_is_persisted_at_prefetch(tmp_path, monkeypatch):
    """A successful listing must be written beside the cached files."""
    from surimi_model_mcp.netcdf import io as ncio
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path))
    ncio.save_listing("s3://b/netcdf/", [
        {"uri": "s3://b/netcdf/a.nc", "size": 10},
        {"uri": "s3://b/netcdf/m.nc", "size": 20},
    ])
    got = ncio.load_listing("s3://b/netcdf/")
    assert got is not None
    assert [f["uri"] for f in got["files"]] == ["s3://b/netcdf/a.nc", "s3://b/netcdf/m.nc"]
    assert got["as_of"]


def test_load_listing_absent_returns_none(tmp_path, monkeypatch):
    from surimi_model_mcp.netcdf import io as ncio
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path))
    assert ncio.load_listing("s3://b/never-listed/") is None


def test_list_files_serves_the_cached_listing_when_s3_is_unreachable(
    tmp_path, monkeypatch
):
    """The whole point: an auth failure must degrade to the last good listing,
    dated, rather than erroring. The files are on disk; refusing to name them
    is the worst of both."""
    from surimi_model_mcp.netcdf import io as ncio
    from surimi_model_mcp.netcdf import discovery
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path))
    ncio.save_listing("s3://b/netcdf/", [{"uri": "s3://b/netcdf/a.nc", "size": 10}])

    def boom(prefix):
        raise ncio.NetCDFIOError(
            "could not list s3://b/netcdf/: The Access Key Id you provided "
            "does not exist in our records.")

    monkeypatch.setattr(ncio, "list_objects", boom)
    out = discovery.list_netcdf_files("s3://b/netcdf/")
    assert out["count"] == 1
    assert out["stale"] is True
    assert out["as_of"]
    assert "credential" in " ".join(out.get("notes") or []).lower() \
        or "unreachable" in " ".join(out.get("notes") or []).lower()


def test_an_auth_failure_is_not_reported_as_an_empty_prefix(tmp_path, monkeypatch):
    """Live on 2026-09-17 the canonical listing failed on auth, the code read
    that as 'empty or unlistable', fell through to the LEGACY prefix, and the
    user saw an error naming a prefix they never asked about."""
    from surimi_model_mcp.netcdf import io as ncio
    assert ncio.is_auth_error("The Access Key Id you provided does not exist")
    assert ncio.is_auth_error("ExpiredToken: the security token has expired")
    assert not ncio.is_auth_error("NoSuchKey: the specified key does not exist")
    assert not ncio.is_auth_error("prefix is empty")
