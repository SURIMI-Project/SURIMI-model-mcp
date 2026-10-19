"""Tests for the netcdf inspection layer.

These tools mirror dogukanteber/netcdf-mcp's metadata tools but stay scoped to
what surimi-mcp's LLM workflow actually needs: file structure, variable
inventory, time/space bounds, CF-compliance check, NaN/coverage summary.
"""
from __future__ import annotations

import pytest

from surimi_model_mcp.netcdf.inspect import (
    NetCDFInspectError,
    check_cf_compliance,
    describe_file,
    get_coverage_summary,
    get_spatial_bounds,
    get_time_range,
    list_variables,
)


# ---------- describe_file ----------

def test_describe_file_returns_structure_dict(tiny_nc):
    out = describe_file(tiny_nc)
    assert "dimensions" in out
    assert "variables" in out
    assert "global_attributes" in out
    assert "file_path" in out
    assert "file_sha256" in out


def test_describe_file_has_correct_dimension_sizes(tiny_nc):
    out = describe_file(tiny_nc)
    assert out["dimensions"]["time"] == 4
    assert out["dimensions"]["lat"] == 3
    assert out["dimensions"]["lon"] == 4


def test_describe_file_lists_data_variables(tiny_nc):
    out = describe_file(tiny_nc)
    var_names = [v["name"] for v in out["variables"]]
    assert "biomass" in var_names


def test_describe_file_includes_global_attributes(tiny_nc):
    out = describe_file(tiny_nc)
    assert out["global_attributes"].get("Conventions") == "CF-1.8"


def test_describe_file_sha256_is_64_hex_chars(tiny_nc):
    out = describe_file(tiny_nc)
    assert len(out["file_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in out["file_sha256"])


def test_describe_file_raises_on_missing_file(tmp_path):
    with pytest.raises(NetCDFInspectError):
        describe_file(str(tmp_path / "nope.nc"))


# ---------- list_variables ----------

def test_list_variables_returns_name_shape_dtype_units(tiny_nc):
    vars_ = list_variables(tiny_nc)
    biomass = next(v for v in vars_ if v["name"] == "biomass")
    assert biomass["shape"] == [4, 3, 4]
    assert biomass["dtype"].startswith("float32")
    assert biomass["units"] == "kg m-2"
    assert biomass["long_name"] == "Biomass density"


def test_list_variables_excludes_coordinate_dims(tiny_nc):
    """Coordinates (time, lat, lon) should NOT be listed as data variables."""
    vars_ = list_variables(tiny_nc)
    names = {v["name"] for v in vars_}
    assert "biomass" in names
    assert "time" not in names
    assert "lat" not in names
    assert "lon" not in names


# ---------- get_time_range ----------

def test_get_time_range_returns_start_and_end_iso(tiny_nc):
    out = get_time_range(tiny_nc)
    assert "start" in out
    assert "end" in out
    assert out["n_steps"] == 4


def test_get_time_range_includes_units(tiny_nc):
    out = get_time_range(tiny_nc)
    assert "days since 2010" in out["units"]


# ---------- get_spatial_bounds ----------

def test_get_spatial_bounds_returns_lat_lon_extents(tiny_nc):
    out = get_spatial_bounds(tiny_nc)
    assert out["lat_min"] == pytest.approx(-30.0)
    assert out["lat_max"] == pytest.approx(30.0)
    assert out["lon_min"] == pytest.approx(-180.0)
    assert out["lon_max"] < 180.0  # endpoint=False
    assert out["lat_n"] == 3
    assert out["lon_n"] == 4


# ---------- check_cf_compliance ----------

def test_check_cf_compliance_passes_on_cf_file(tiny_nc):
    out = check_cf_compliance(tiny_nc)
    assert out["compliant"] is True
    assert out["convention"] == "CF-1.8"


def test_check_cf_compliance_flags_missing_conventions(tiny_nc_no_cf):
    out = check_cf_compliance(tiny_nc_no_cf)
    assert out["compliant"] is False
    assert "convention" in out["missing"]


def test_check_cf_compliance_reports_missing_units(tiny_nc_no_cf):
    out = check_cf_compliance(tiny_nc_no_cf)
    # tiny_nc_no_cf has no units on its variable
    assert any("units" in m for m in out["warnings"])


# ---------- get_coverage_summary ----------

def test_get_coverage_summary_returns_nan_fraction_per_variable(tiny_nc):
    out = get_coverage_summary(tiny_nc)
    biomass = next(v for v in out["variables"] if v["name"] == "biomass")
    assert biomass["nan_fraction"] == 0.0
    assert biomass["n_total"] == 4 * 3 * 4


def test_get_coverage_summary_detects_actual_nans(tmp_path):
    """Synthesize a file with explicit NaNs and check fraction."""
    import numpy as np
    import xarray as xr

    arr = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype="float32")
    ds = xr.Dataset(
        data_vars={"v": (("y", "x"), arr)},
        coords={"y": [0, 1], "x": [0, 1]},
    )
    p = tmp_path / "nans.nc"
    ds.to_netcdf(p)
    out = get_coverage_summary(str(p))
    v = next(x for x in out["variables"] if x["name"] == "v")
    assert v["nan_fraction"] == pytest.approx(0.25)
    assert v["n_nan"] == 1


def test_get_coverage_summary_samples_large_variables(tmp_path, monkeypatch):
    """A variable above the full-scan cell limit is sampled, not materialized."""
    import numpy as np
    import xarray as xr

    from surimi_model_mcp.netcdf import inspect as nci

    # Force a tiny limit so a small test file takes the sampled path.
    monkeypatch.setattr(nci, "_COVERAGE_FULL_SCAN_LIMIT", 10)
    # land-mask style: first lat row all-NaN, rest present; constant over time.
    arr = np.ones((6, 3, 4), dtype="float32")
    arr[:, 0, :] = np.nan
    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), arr)},
        coords={"time": range(6), "lat": [0, 1, 2], "lon": [0, 1, 2, 3]},
    )
    p = tmp_path / "big.nc"
    ds.to_netcdf(p)
    out = nci.get_coverage_summary(str(p))
    v = next(x for x in out["variables"] if x["name"] == "biomass")
    assert v["sampled"] is True
    assert v["n_total"] == 6 * 3 * 4
    # one of three lat rows is NaN -> ~1/3, estimated from the sample.
    assert v["nan_fraction"] == pytest.approx(1 / 3, abs=0.01)
    assert "sample_note" in v


def test_get_coverage_summary_handles_non_float_dtypes(tmp_path):
    """Integer and object dtype variables should not raise; n_nan must be 0."""
    import numpy as np
    import xarray as xr

    int_arr = np.array([[1, 2], [3, 4]], dtype="int32")
    str_arr = np.array(["a", "b", "c"], dtype="<U4")
    ds = xr.Dataset(
        data_vars={
            "counts": (("y", "x"), int_arr),
            "labels": (("k",), str_arr),
        },
        coords={"y": [0, 1], "x": [0, 1], "k": [0, 1, 2]},
    )
    p = tmp_path / "mixed.nc"
    ds.to_netcdf(p)
    out = get_coverage_summary(str(p))
    counts = next(v for v in out["variables"] if v["name"] == "counts")
    labels = next(v for v in out["variables"] if v["name"] == "labels")
    assert counts["n_nan"] == 0
    assert labels["n_nan"] == 0


def test_coverage_sampling_does_not_claim_the_mask_is_constant(
    tiny_nc_varying_nan, monkeypatch
):
    """The sample note must not assert constancy it has not checked.

    The old wording said "(NaN/land mask is constant over this dimension)".
    On this file that is false: rows 0 of every slice are land, but the last
    four steps are empty as well, so the pattern varies along time.
    """
    from surimi_model_mcp.netcdf import inspect as nci

    monkeypatch.setattr(nci, "_COVERAGE_FULL_SCAN_LIMIT", 10)
    out = nci.get_coverage_summary(tiny_nc_varying_nan)
    v = next(x for x in out["variables"] if x["name"] == "biomass")
    assert "constant over this dimension" not in v.get("sample_note", "")


def test_coverage_sampling_estimate_is_labelled_as_an_estimate(
    tiny_nc_varying_nan, monkeypatch
):
    """A sampled count is extrapolated, and must say so rather than pose as a count."""
    from surimi_model_mcp.netcdf import inspect as nci

    monkeypatch.setattr(nci, "_COVERAGE_FULL_SCAN_LIMIT", 10)
    out = nci.get_coverage_summary(tiny_nc_varying_nan)
    v = next(x for x in out["variables"] if x["name"] == "biomass")
    assert v["sampled"] is True
    # The extrapolated figure is reported under an estimate-named key, and the
    # exact key `n_nan` is not used to carry a number nobody counted.
    assert "n_nan_estimated" in v
    assert v.get("n_nan") is None


def test_coverage_sampling_separates_empty_steps_from_the_land_mask(
    tiny_nc_varying_nan, monkeypatch
):
    """Land and "this step has nothing" are different facts and reported apart.

    Fixture: 1/4 of every slice is land, and 4 of 20 steps are wholly empty.
    """
    from surimi_model_mcp.netcdf import inspect as nci

    monkeypatch.setattr(nci, "_COVERAGE_FULL_SCAN_LIMIT", 10)
    out = nci.get_coverage_summary(tiny_nc_varying_nan)
    v = next(x for x in out["variables"] if x["name"] == "biomass")
    assert v["empty_step_fraction"] == pytest.approx(4 / 20)
    assert v["nan_fraction_spatial"] == pytest.approx(0.25)


def test_coverage_sampling_estimate_is_close_to_the_truth(
    tiny_nc_varying_nan, monkeypatch
):
    """True fraction is 0.40. The old 3-step sampler lands on 0/9/19 and reads 0.50."""
    from surimi_model_mcp.netcdf import inspect as nci

    monkeypatch.setattr(nci, "_COVERAGE_FULL_SCAN_LIMIT", 10)
    out = nci.get_coverage_summary(tiny_nc_varying_nan)
    v = next(x for x in out["variables"] if x["name"] == "biomass")
    assert v["nan_fraction"] == pytest.approx(0.40, abs=0.02)


# ---------- get_time_range, decoded ----------

def test_get_time_range_decodes_the_axis_to_calendar_dates(tiny_nc):
    """Raw offsets are unreadable on their own.

    tiny_nc is 4 monthly steps from 2010-01-01, stored as 0/31/59/90 "days
    since 2010-01-01". Reporting 0 and 90 leaves the reader to do calendar
    arithmetic against an epoch, which is how a period gets misreported.
    """
    out = get_time_range(tiny_nc)
    assert out["start_iso"].startswith("2010-01-01")
    assert out["end_iso"].startswith("2010-04-01")
    assert out["start_year"] == 2010
    assert out["end_year"] == 2010


def test_get_time_range_keeps_the_raw_values(tiny_nc):
    """Decoding adds to the answer, it does not replace what is on disk."""
    out = get_time_range(tiny_nc)
    assert out["start"] == 0
    assert out["end"] == 90
    assert "days since 2010" in out["units"]


def test_get_time_range_spans_years_when_the_axis_does(tiny_nc_multiyear):
    out = get_time_range(tiny_nc_multiyear)
    assert out["start_year"] == 2009
    assert out["end_year"] == 2010


def test_get_time_range_survives_an_axis_it_cannot_decode(tiny_nc_no_cf):
    """No CF units: report the raw axis and say plainly that it is not decoded."""
    out = get_time_range(tiny_nc_no_cf)
    assert out["start"] == 0
    assert out["start_iso"] is None
    assert out["start_year"] is None
    assert out["decoded"] is False


def test_get_time_range_decodes_a_noleap_calendar_correctly(tmp_path):
    """A non-standard calendar decodes to cftime objects, not datetime64.

    The real gridded file uses "days since 1661-01-01" on a noleap calendar.
    An earlier `.astype("datetime64[Y]")` path read such an axis off by years,
    so pin the calendar-aware result here.
    """
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        data_vars={"v": (("time",), np.arange(3, dtype="float32"))},
        coords={"time": ("time", np.array([0.0, 365.0, 730.0]),
                         {"units": "days since 1661-01-01", "calendar": "noleap"})},
    )
    p = tmp_path / "noleap.nc"
    ds.to_netcdf(p)
    out = get_time_range(str(p))
    assert out["decoded"] is True
    assert out["calendar"] == "noleap"
    # 365 noleap days per year: 1661, 1662, 1663.
    assert out["start_year"] == 1661
    assert out["end_year"] == 1663
