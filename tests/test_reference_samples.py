"""Q-E6-5: the westmed sample is listed by `model_list_files` as a bundled reference sample.

The sample is `westmed-2013-2017-biomass-catch-sales.nc` (94 species, adult and juvenile life
stages). `_apply_select` in `netcdf/analytics.py` already resolves a named species, so the only
gap was that `model_list_files` never listed the file. The repository now carries it in
`data/netcdf-samples/` next to a mask on its own grid, and `_reference_sample_files()` in
`tools/gridded.py` merges that directory into the listing the way the simulation scan is merged.

The tests below use the generic filename `reference-sample.nc`, so they cover the scan itself
(merge, classify, degrade) and not the real file. `test_westmed_sample_bundled.py` covers the
real file.
"""
from __future__ import annotations

from surimi_model_mcp.tools import gridded as nc_tools


def _empty_s3_catalog(tmp_path):
    d = tmp_path / "s3catalog"
    d.mkdir()
    return str(d)


def test_reference_sample_dir_is_merged_into_model_list_files(monkeypatch, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "reference-sample.nc").write_bytes(b"x" * 100)
    (samples / "reference-sample_mask.nc").write_bytes(b"y" * 20)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(samples))
    monkeypatch.setattr(nc_tools, "_ENV_PREFIX", _empty_s3_catalog(tmp_path))
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    names = {f["uri"].rsplit("/", 1)[-1] for f in out["files"]}
    assert "reference-sample.nc" in names
    assert "reference-sample_mask.nc" in names


def test_reference_sample_data_file_is_kind_data_and_mask_is_kind_mask(monkeypatch, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "reference-sample.nc").write_bytes(b"x" * 100)
    (samples / "reference-sample_mask.nc").write_bytes(b"y" * 20)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(samples))
    monkeypatch.setattr(nc_tools, "_ENV_PREFIX", _empty_s3_catalog(tmp_path))
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    by_name = {f["uri"].rsplit("/", 1)[-1]: f for f in out["files"]}
    assert by_name["reference-sample.nc"]["kind"] == "data"
    assert by_name["reference-sample_mask.nc"]["kind"] == "mask"


def test_missing_reference_dir_degrades_to_an_empty_addition_not_a_crash(monkeypatch, tmp_path):
    """The bundled dir is optional infrastructure, exactly like the S3 catalog
    and the simulation scan already degrade -- one missing source must never
    sink the whole listing."""
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(nc_tools, "_ENV_PREFIX", _empty_s3_catalog(tmp_path))
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()  # must not raise
    assert out["files"] == []


def test_any_data_file_bundled_under_reference_samples_dir_is_surimi_sample(
    monkeypatch, tmp_path
):
    """Trust is by SOURCE
    (checked into this repo's data/netcdf-samples/, so only a maintainer
    commit puts a file there), not by filename. Every 'data' entry this scan
    finds gets origin 'surimi-sample' from _reference_sample_files() itself
    -- a preset mark_origin leaves alone -- regardless of basename. This is
    what lets a future second bundled sample "just work" with no name
    registry to update.

    Contrast test_local_sample_catalog.py: SURIMI_LOCAL_NETCDF_SAMPLES points
    at an OPERATOR-supplied directory, not a repo commit, so a same-named
    file found there is NOT trusted this way and keeps the ordinary
    simulation_id-or-external default (external-or-reference)."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "reference-sample.nc").write_bytes(b"x" * 100)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(samples))
    monkeypatch.setattr(nc_tools, "_ENV_PREFIX", _empty_s3_catalog(tmp_path))
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    entry = next(f for f in out["files"] if f["uri"].endswith("reference-sample.nc"))
    assert entry["origin"] == "surimi-sample"


def test_the_real_westmed_basename_is_marked_surimi_sample_not_external(monkeypatch, tmp_path):
    """The real file, end to end through model_list_files: same rule as
    above, exercised with the actual westmed basename this scan finds in
    production. See test_listing_marks_origin.py for mark_origin's own
    preset-is-not-overwritten unit test."""
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "westmed-2013-2017-biomass-catch-sales.nc").write_bytes(b"x" * 100)
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR", str(samples))
    monkeypatch.setattr(nc_tools, "_ENV_PREFIX", _empty_s3_catalog(tmp_path))
    monkeypatch.setattr(nc_tools, "_scan_runs", lambda: {"simulations": []})

    out = nc_tools.model_list_files()
    entry = next(f for f in out["files"]
                 if f["uri"].endswith("westmed-2013-2017-biomass-catch-sales.nc"))
    assert entry["origin"] == "surimi-sample"
