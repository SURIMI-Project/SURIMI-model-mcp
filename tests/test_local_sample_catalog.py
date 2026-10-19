"""Q-E5-4: `SURIMI_LOCAL_NETCDF_SAMPLES` merges a local directory into `model_list_files`.

The westmed sample, westmed-2013-2017-biomass-catch-sales.nc, has 49 monthly steps from 2013-01
to 2017-01 over 94 species and 9 fleets, on a 109x82 western-Med grid that the served EcoOcean
file and the AR6 mask do not share. `_local_sample_files()` in `tools/gridded.py` lists the
files of an operator-supplied directory, in the same shape as `_run_netcdf_files` uses for
simulation output. A file found this way is not marked as a SURIMI run.

The species-level `select` mechanism already works on such a file (see
test_nc_tools.py::test_model_time_series_can_select_one_species).
"""
import importlib

from surimi_model_mcp.tools import gridded as nc_tools


def test_no_local_sample_dir_configured_changes_nothing(monkeypatch):
    monkeypatch.delenv("SURIMI_LOCAL_NETCDF_SAMPLES", raising=False)
    importlib.reload(nc_tools)
    files, note = nc_tools._local_sample_files()
    assert files == []
    assert note is None


def test_local_sample_dir_is_merged_into_model_list_files(monkeypatch, tmp_path):
    sample_dir = tmp_path / "netcdf-samples"
    sample_dir.mkdir()
    (sample_dir / "westmed-2013-2017-biomass-catch-sales.nc").write_bytes(b"x" * 100)
    monkeypatch.setenv("SURIMI_LOCAL_NETCDF_SAMPLES", str(sample_dir))
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(tmp_path / "empty-catalog"))
    (tmp_path / "empty-catalog").mkdir()

    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    matches = [f for f in out["files"]
               if f["uri"].endswith("westmed-2013-2017-biomass-catch-sales.nc")]
    assert len(matches) == 1, out["files"]
    entry = matches[0]
    assert entry["kind"] == "data"
    assert entry["size"] == 100


def test_local_sample_file_is_marked_as_not_a_surimi_run(monkeypatch, tmp_path):
    """It is sample or reference data, not a signed SURIMI run output, so the
    listing must say so, as it does for the external EcoOcean file."""
    sample_dir = tmp_path / "netcdf-samples"
    sample_dir.mkdir()
    (sample_dir / "westmed-2013-2017-biomass-catch-sales.nc").write_bytes(b"x" * 100)
    monkeypatch.setenv("SURIMI_LOCAL_NETCDF_SAMPLES", str(sample_dir))
    monkeypatch.setenv("SURIMI_NETCDF_PREFIX", str(tmp_path / "empty-catalog"))
    (tmp_path / "empty-catalog").mkdir()

    importlib.reload(nc_tools)
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    entry = next(f for f in out["files"]
                 if f["uri"].endswith("westmed-2013-2017-biomass-catch-sales.nc"))
    assert entry["origin"] == "external-or-reference"
    assert "simulation_id" not in entry


def test_a_bad_local_sample_dir_does_not_take_the_catalog_down(monkeypatch, tmp_path):
    monkeypatch.setenv("SURIMI_LOCAL_NETCDF_SAMPLES", str(tmp_path / "does-not-exist"))
    importlib.reload(nc_tools)
    files, note = nc_tools._local_sample_files()
    assert files == []
    assert note is not None and "does-not-exist" in note
