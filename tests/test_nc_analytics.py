"""Tests for analytical primitives.

The fixture `tiny_nc` produces deterministic biomass values:
biomass[t, lat_idx, lon_idx] = t*100 + lat_idx*10 + lon_idx

For t=0 only:
  biomass[0, :, :] =
    [[ 0,  1,  2,  3],
     [10, 11, 12, 13],
     [20, 21, 22, 23]]

`tiny_region_mask` splits the lon axis:
  Region "West" (region_id=1): lon indices 0, 1
  Region "East" (region_id=2): lon indices 2, 3

At t=0, unweighted mean per region:
  West = mean([0,1, 10,11, 20,21]) = 63/6  = 10.5
  East = mean([2,3, 12,13, 22,23]) = 75/6  = 12.5

Averaged across 4 timesteps the means grow by 150 each (each step adds 100
per cell, so mean adds 100 too; t=0..3 mean offset = 150).
Expected over all 4 timesteps:
  West mean = 10.5 + 150 = 160.5
  East mean = 12.5 + 150 = 162.5
"""
from __future__ import annotations

import pytest

from surimi_model_mcp.netcdf.analytics import (
    MAX_SERIES_POINTS,
    AnalyticsError,
    compare_periods,
    nc_trend,
    select_points,
    time_series,
    top_regions,
)


# ---------- top_regions ----------

def test_top_regions_returns_correct_order_unweighted(tiny_nc, tiny_region_mask):
    out = top_regions(
        file=tiny_nc,
        var="biomass",
        n=2,
        agg="mean",
        mask_file=tiny_region_mask,
    )
    assert len(out["result"]) == 2
    # East has slightly higher mean (lon indices 2,3 vs 0,1)
    assert out["result"][0]["region"] == "East"
    assert out["result"][0]["value"] == pytest.approx(162.5, rel=1e-3)
    assert out["result"][1]["region"] == "West"
    assert out["result"][1]["value"] == pytest.approx(160.5, rel=1e-3)


def test_top_regions_respects_n(tiny_nc, tiny_region_mask):
    out = top_regions(
        file=tiny_nc, var="biomass", n=1, agg="mean", mask_file=tiny_region_mask,
    )
    assert len(out["result"]) == 1
    assert out["result"][0]["region"] == "East"


def test_top_regions_provenance_includes_file_and_mask_sha(tiny_nc, tiny_region_mask):
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    p = out["provenance"]
    assert len(p["file_sha256"]) == 64
    assert len(p["mask_sha256"]) == 64
    assert p["var"] == "biomass"
    assert p["agg"] == "mean"


def test_top_regions_includes_coverage(tiny_nc, tiny_region_mask):
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    cov = out["coverage"]
    assert cov["nan_fraction"] == pytest.approx(0.0)
    assert cov["n_regions_evaluated"] == 2


def test_top_regions_raises_on_unknown_variable(tiny_nc, tiny_region_mask):
    with pytest.raises(AnalyticsError):
        top_regions(
            file=tiny_nc, var="nonexistent", n=2,
            agg="mean", mask_file=tiny_region_mask,
        )


def test_top_regions_raises_on_unknown_agg(tiny_nc, tiny_region_mask):
    with pytest.raises(AnalyticsError):
        top_regions(
            file=tiny_nc, var="biomass", n=2,
            agg="bogus", mask_file=tiny_region_mask,
        )


def test_top_regions_year_filter_selects_only_that_year(tiny_nc_multiyear, tiny_region_mask):
    """year=2010 must aggregate only the 2010 timesteps (mean 500), not all 6."""
    out2010 = top_regions(
        file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010,
    )
    assert out2010["result"][0]["value"] == pytest.approx(500.0, rel=1e-3)

    out2009 = top_regions(
        file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2009,
    )
    assert out2009["result"][0]["value"] == pytest.approx(200.0, rel=1e-3)


def test_top_regions_year_filter_noleap_calendar(tiny_nc_noleap_multiyear, tiny_region_mask):
    """Year filtering must be correct for non-standard (cftime) calendars.

    Regression for the EcoOcean bug: a "days since 1661" + noleap axis decodes
    to cftime objects; the old .astype('datetime64[Y]') mis-assigned years and
    year=2010 selected the wrong/empty slice (all-NaN). Now year=2010 must
    pick only the 2010 steps (mean 500) and 2009 only its steps (mean 200).
    """
    out2010 = top_regions(
        file=tiny_nc_noleap_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010,
    )
    assert out2010["coverage"]["n_timesteps"] == 3
    assert out2010["result"][0]["value"] == pytest.approx(500.0, rel=1e-3)

    out2009 = top_regions(
        file=tiny_nc_noleap_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2009,
    )
    assert out2009["coverage"]["n_timesteps"] == 3
    assert out2009["result"][0]["value"] == pytest.approx(200.0, rel=1e-3)


def test_top_regions_without_year_defaults_to_recent_year(tiny_nc_multiyear, tiny_region_mask):
    """No year on a MULTI-year file => default to the most recent year (2010),
    not the whole cube. Aggregating all years on a global multi-decade file is
    slow enough to time out a client, so the server picks the latest year and
    reports the substitution. 2010 mean = mean(400,500,600) = 500 (all-time
    would be 350)."""
    out = top_regions(
        file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask,
    )
    assert out["provenance"]["year"] == 2010
    assert out["provenance"]["year_defaulted"] is True
    assert "note" in out["coverage"]
    assert out["result"][0]["value"] == pytest.approx(500.0, rel=1e-3)


def test_top_regions_explicit_year_not_flagged_defaulted(tiny_nc_multiyear, tiny_region_mask):
    """An explicit year is honored as-is and never flagged as defaulted."""
    out = top_regions(
        file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2009,
    )
    assert out["provenance"]["year"] == 2009
    assert out["provenance"]["year_defaulted"] is False
    assert "note" not in out["coverage"]


def test_top_regions_single_year_file_no_default(tiny_nc, tiny_region_mask):
    """A single-year file is small; no-year stays the full (single-year) cube
    and is NOT flagged as defaulted (the guard only triggers on multi-year)."""
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask,
    )
    assert out["provenance"]["year"] is None
    assert out["provenance"]["year_defaulted"] is False


def test_top_regions_year_records_in_provenance(tiny_nc_multiyear, tiny_region_mask):
    out = top_regions(
        file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010,
    )
    assert out["provenance"]["year"] == 2010
    assert out["coverage"]["n_timesteps"] == 3


def test_top_regions_year_not_present_raises(tiny_nc_multiyear, tiny_region_mask):
    with pytest.raises(AnalyticsError):
        top_regions(
            file=tiny_nc_multiyear, var="biomass", n=2, agg="mean",
            mask_file=tiny_region_mask, year=1999,
        )


def test_top_regions_supports_sum_aggregation(tiny_nc, tiny_region_mask):
    """sum should add all cells * timesteps in each region."""
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, agg="sum", mask_file=tiny_region_mask,
    )
    # Per t, West sum = 0+1 +10+11 +20+21 = 63; East = 2+3+12+13+22+23 = 75
    # Over 4 timesteps: West adds (per step) 600 cells*increment, but
    # simpler: sum over t of sum over region = sum_region_t = 4 * region_sum
    # plus the t-offset contribution: per region 6 cells, offset (0+100+200+300)*6 = 3600
    # West: 4*63 + 3600 = 252 + 3600 = 3852
    # East: 4*75 + 3600 = 300 + 3600 = 3900
    res = {r["region"]: r["value"] for r in out["result"]}
    assert res["East"] == pytest.approx(3900.0, rel=1e-3)
    assert res["West"] == pytest.approx(3852.0, rel=1e-3)


# ---------- time_series year rollup (cube slice) ----------

def test_rollup_by_year_groups_and_aggregates():
    """A per-timestep series rolls up into one row per calendar year (the cube
    dimension), aggregated to a measure. NaN steps are excluded from the
    aggregate but still counted in n_steps."""
    from surimi_model_mcp.netcdf.analytics import _rollup_by_year

    years = [2009, 2009, 2010]
    values = [10.0, float("nan"), 30.0]
    rows = _rollup_by_year(values, years, "mean")
    by = {r["year"]: r for r in rows}
    assert set(by) == {2009, 2010}
    assert by[2009]["value"] == pytest.approx(10.0)  # NaN excluded from mean
    assert by[2009]["n_steps"] == 2                  # both steps still reported
    assert by[2010]["value"] == pytest.approx(30.0)
    assert by[2010]["n_steps"] == 1


def test_rollup_by_year_all_missing_year_reports_none():
    """A year with no finite steps reports value=None, not NaN or a crash."""
    from surimi_model_mcp.netcdf.analytics import _rollup_by_year

    years = [2009, 2009]
    values = [float("nan"), float("nan")]
    rows = _rollup_by_year(values, years, "mean")
    assert rows == [{"year": 2009, "value": None, "n_steps": 2}]


def test_time_series_returns_year_rollup(tiny_nc_multiyear, tiny_region_mask):
    """time_series must return a pivotable per-year rollup so the model
    summarizes a handful of yearly rows instead of every raw monthly point.
    East is uniform (t+1)*100 -> 2009 mean(100,200,300)=200,
    2010 mean(400,500,600)=500, three steps each."""
    out = time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    by = {r["year"]: r for r in out["by_year"]}
    assert set(by) == {2009, 2010}
    assert by[2009]["value"] == pytest.approx(200.0)
    assert by[2009]["n_steps"] == 3
    assert by[2010]["value"] == pytest.approx(500.0)
    assert by[2010]["n_steps"] == 3


# ---------- time_series ----------

def test_time_series_returns_one_point_per_step(tiny_nc, tiny_region_mask):
    out = time_series(
        file=tiny_nc, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert len(out["points"]) == 4
    # East at t=0: mean of [2,3,12,13,22,23] = 12.5
    # Each subsequent step adds 100
    expected = [12.5, 112.5, 212.5, 312.5]
    for got, exp in zip(out["points"], expected):
        assert got["value"] == pytest.approx(exp, rel=1e-3)


def test_time_series_provenance_records_region(tiny_nc, tiny_region_mask):
    out = time_series(
        file=tiny_nc, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert out["provenance"]["region"] == "East"


def test_time_series_raises_on_unknown_region(tiny_nc, tiny_region_mask):
    with pytest.raises(AnalyticsError):
        time_series(
            file=tiny_nc, var="biomass", region="Atlantis",
            agg_per_step="mean", mask_file=tiny_region_mask,
        )


def test_time_series_includes_compact_summary(tiny_nc, tiny_region_mask):
    """time_series must return a compact `summary` so the model can synthesize a
    final answer without iterating every raw point (the S3 no-answer bug: 240
    monthly points came back and the model never summarized them)."""
    out = time_series(
        file=tiny_nc, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    s = out["summary"]
    assert s["n_points"] == 4
    assert s["n_valid"] == 4
    assert s["value_start"] == pytest.approx(12.5, rel=1e-3)
    assert s["value_end"] == pytest.approx(312.5, rel=1e-3)
    assert s["value_min"] == pytest.approx(12.5, rel=1e-3)
    assert s["value_max"] == pytest.approx(312.5, rel=1e-3)
    assert s["value_mean"] == pytest.approx(162.5, rel=1e-3)
    assert s["slope_per_step"] == pytest.approx(100.0, rel=1e-3)
    assert s["direction"] == "increasing"


def test_series_summary_handles_all_missing():
    """All-NaN series must report n_valid=0 and omit value_* fields, not raise."""
    from surimi_model_mcp.netcdf.analytics import _series_summary
    pts = [{"t": 0.0, "value": float("nan")}, {"t": 1.0, "value": float("nan")}]
    s = _series_summary(pts)
    assert s["n_points"] == 2
    assert s["n_valid"] == 0
    assert "value_mean" not in s


def test_series_summary_partial_missing_uses_only_valid():
    """Gaps (NaN steps) must be skipped; start/end are the first/last VALID values."""
    from surimi_model_mcp.netcdf.analytics import _series_summary
    pts = [
        {"t": 0.0, "value": 10.0},
        {"t": 1.0, "value": float("nan")},
        {"t": 2.0, "value": 30.0},
    ]
    s = _series_summary(pts)
    assert s["n_points"] == 3
    assert s["n_valid"] == 2
    assert s["value_start"] == pytest.approx(10.0)
    assert s["value_end"] == pytest.approx(30.0)
    assert s["direction"] == "increasing"


# ---------- compare_periods ----------

def test_compare_periods_returns_diff_per_region(tiny_nc, tiny_region_mask):
    """Period A: steps 0..1, Period B: steps 2..3."""
    out = compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="diff",
    )
    # Period A mean per region: (region_offset + 50)
    # Period B mean per region: (region_offset + 250)
    # diff = 200 for both
    diffs = {r["region"]: r["value"] for r in out["result"]}
    assert diffs["West"] == pytest.approx(200.0, rel=1e-3)
    assert diffs["East"] == pytest.approx(200.0, rel=1e-3)


def test_compare_periods_supports_ratio(tiny_nc, tiny_region_mask):
    out = compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="ratio",
    )
    for r in out["result"]:
        assert r["value"] > 1.0  # period_b > period_a


# ---------- nc_trend (Mann-Kendall) ----------

def test_trend_returns_slope_and_pvalue(tiny_nc, tiny_region_mask):
    out = nc_trend(
        file=tiny_nc, var="biomass", region="East",
        mask_file=tiny_region_mask,
    )
    # Monotonically increasing, so slope > 0, p-value low
    assert out["slope"] > 0
    assert out["p_value"] < 0.5
    assert "direction" in out
    assert out["direction"] in ("increasing", "decreasing", "no_trend")


def test_trend_provenance_includes_method(tiny_nc, tiny_region_mask):
    out = nc_trend(
        file=tiny_nc, var="biomass", region="East",
        mask_file=tiny_region_mask,
    )
    assert out["provenance"]["method"] == "mann_kendall"


def test_mann_kendall_trend_ignores_nan_gaps():
    """NaN steps must be dropped, not crash with 'cannot convert float NaN to
    integer' (the real EcoOcean Mediterranean series has missing months)."""
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    out = _mann_kendall_trend(np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0]))
    assert out["n"] == 5  # only the 5 finite values count
    assert out["slope"] > 0


def test_mann_kendall_trend_too_few_valid_raises():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    with pytest.raises(AnalyticsError):
        _mann_kendall_trend(np.array([1.0, np.nan, np.nan]))


def test_nc_trend_with_gap_does_not_crash(tiny_nc_gap, tiny_region_mask):
    """End-to-end: a region series with a missing timestep must return a trend,
    not an error, and report how many steps were missing."""
    out = nc_trend(
        file=tiny_nc_gap, var="biomass", region="East",
        mask_file=tiny_region_mask,
    )
    assert out["n"] == 3            # 3 valid of 4
    assert out["n_total"] == 4
    assert out["n_missing"] == 1
    assert out["slope"] > 0         # 12.5 -> 212.5 -> 312.5 (t=1 dropped)
    assert out["provenance"]["method"] == "mann_kendall"


# --- regressions found live 2026-08-05 (audit) --------------------------------

def test_top_regions_agg_defaults_to_mean(tiny_nc, tiny_region_mask):
    """`agg` had no default, so the natural LLM call died on a pydantic
    validation error before reaching the data."""
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, mask_file=tiny_region_mask,
    )
    explicit = top_regions(
        file=tiny_nc, var="biomass", n=2, mask_file=tiny_region_mask, agg="mean",
    )
    assert out["result"] == explicit["result"]
    assert out["provenance"]["agg"] == "mean"


def test_year_default_skips_an_all_nan_tail(tiny_nc_empty_tail, tiny_region_mask):
    """Defaulting to "most recent year in the file" landed on the empty tail and
    returned null for every region. It must pick the most recent year WITH data."""
    out = top_regions(
        file=tiny_nc_empty_tail, var="biomass", n=2, mask_file=tiny_region_mask,
    )
    assert out["provenance"]["year"] == 2009, out["provenance"]
    assert out["provenance"]["year_defaulted"] is True
    assert out["coverage"]["nan_fraction"] < 1.0
    assert all(r["value"] is not None for r in out["result"]), out["result"]


def test_explicit_empty_year_is_still_honoured(tiny_nc_empty_tail, tiny_region_mask):
    """Only the DEFAULT skips empty years. An explicitly requested empty year
    must still report honestly rather than silently substituting."""
    out = top_regions(
        file=tiny_nc_empty_tail, var="biomass", n=2,
        mask_file=tiny_region_mask, year=2010,
    )
    assert out["provenance"]["year"] == 2010
    assert out["provenance"].get("year_defaulted") is False
    assert out["coverage"]["nan_fraction"] == 1.0


# --- defects found by expert review, 2026-09-14 -----------------------------
# Each of these returns a wrong answer on the live server today. The reviewers
# who hit them are identified by label in the project's private benchmark.

def test_sum_of_an_all_missing_region_is_not_zero():
    """An empty period must report unavailable, never 0.0.

    Live on 2026-09-14, model_top_regions with agg="sum" and year=2012, a year
    with no data, returned value 0.0 for all fifteen regions and ranked them.
    Reviewer E5 names "treats missing values as zero" as a trust-breaker.
    """
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _apply_agg
    assert _apply_agg(np.array([np.nan, np.nan]), "sum") is None


def test_sum_of_a_partly_missing_region_still_sums():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _apply_agg
    assert _apply_agg(np.array([1.0, np.nan, 2.0]), "sum") == 3.0


def test_every_all_missing_aggregation_reports_unavailable():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _apply_agg
    empty = np.array([np.nan, np.nan])
    for agg in ("mean", "sum", "max", "min"):
        assert _apply_agg(empty, agg) is None, agg


def test_a_real_aggregation_is_unaffected():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _apply_agg
    vals = np.array([1.0, 2.0, 3.0])
    assert _apply_agg(vals, "mean") == 2.0
    assert _apply_agg(vals, "max") == 3.0
    assert _apply_agg(vals, "min") == 1.0


def test_every_analytic_result_states_its_unit(tiny_nc, tiny_region_mask):
    """A number without its unit is unusable, and three of four tools omit it.

    Reviewer E6 names "if the answers given do not contain units of
    measurment" as a trust-breaker. Live on 2026-09-14 only top_regions
    returned a unit; time_series, nc_trend and compare_periods did not.
    """
    from surimi_model_mcp.netcdf import analytics
    common = dict(file=tiny_nc, var="biomass", mask_file=tiny_region_mask)

    ts = analytics.time_series(region="East", agg_per_step="mean", **common)
    assert ts["units"] == "kg m-2"

    tr = analytics.nc_trend(region="East", **common)
    assert tr["units"] == "kg m-2"

    cp = analytics.compare_periods(period_a=[0, 1], period_b=[1, 2], **common)
    assert cp["units"] == "kg m-2"


def test_unit_is_stated_as_unknown_when_the_file_omits_it():
    """An absent unit is named, never silently dropped as an empty string."""
    import types
    from surimi_model_mcp.netcdf.analytics import _units_of
    assert _units_of(types.SimpleNamespace(attrs={})) == "unknown"
    assert _units_of(types.SimpleNamespace(attrs={"units": ""})) == "unknown"
    assert _units_of(types.SimpleNamespace(attrs={"units": "  "})) == "unknown"
    assert _units_of(types.SimpleNamespace(attrs={"units": "g m-2"})) == "g m-2"


# --- the factor of twelve ---------------------------------------------------
# A slope came back per monthly step with nothing saying so, and the correct
# annual figure is about twelve times larger. Reviewer E6 wrote that she could
# not understand what "wrong by about 12 times" meant. This makes the step
# explicit and supplies the annual figure rather than leaving it to be derived.

def test_slope_names_its_step_and_supplies_an_annual_figure(tiny_nc_multiyear, tiny_region_mask):
    from surimi_model_mcp.netcdf.analytics import time_series
    out = time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    s = out["summary"]
    # the existing field keeps its name and meaning
    assert "slope_per_step" in s
    # and the step is now named, with the annual conversion beside it
    assert s["step_unit"] in {"month", "year", "quarter", "day", "step"}
    assert s["steps_per_year"] is not None
    assert s["slope_per_year_scaled"] == pytest.approx(
        s["slope_per_step"] * s["steps_per_year"]
    )
    assert s["slope_unit"].endswith(" per " + s["step_unit"])


def test_the_two_slope_paths_agree_across_a_gap(tiny_nc_gap, tiny_region_mask):
    """_series_summary fitted on compressed indices, nc_trend on real positions.

    With an interior gap the two disagreed about the same series. On the live
    file the gap is a tail, so the published number is unaffected, but the
    divergence is real and a file with an interior gap would expose it.
    """
    from surimi_model_mcp.netcdf.analytics import time_series, nc_trend
    common = dict(file=tiny_nc_gap, var="biomass", region="East",
                  mask_file=tiny_region_mask)
    ts = time_series(agg_per_step="mean", **common)
    tr = nc_trend(**common)
    assert ts["summary"]["slope_per_step"] == pytest.approx(tr["slope"], rel=1e-9)


def test_step_unit_is_month_on_a_monthly_axis():
    from surimi_model_mcp.netcdf.analytics import _step_unit
    assert _step_unit(12.0) == "month"
    assert _step_unit(1.0) == "year"
    assert _step_unit(4.0) == "quarter"
    assert _step_unit(365.0) == "day"
    assert _step_unit(None) == "step"
    assert _step_unit(7.0) == "step"


def test_trend_also_names_its_step(tiny_nc_multiyear, tiny_region_mask):
    from surimi_model_mcp.netcdf.analytics import nc_trend
    out = nc_trend(file=tiny_nc_multiyear, var="biomass", region="East",
                   mask_file=tiny_region_mask)
    assert out["step_unit"] in {"month", "year", "quarter", "day", "step"}
    assert out["slope_unit"].endswith(" per " + out["step_unit"])
    if out["steps_per_year"] is not None:
        assert out["slope_per_year_scaled"] == pytest.approx(
            out["slope"] * out["steps_per_year"])


def test_a_trailing_gap_leaves_the_slope_unchanged(tiny_nc_empty_tail, tiny_region_mask):
    """The published live figure must not move because of the basis fix.

    The live file's 60 missing steps are a trailing tail, so the finite
    positions are still 0..n-1 and the compacted and real bases coincide. The
    consortium cases ground slope_per_step at -0.0003194165861383859; this test
    is what says that number is safe.
    """
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import time_series
    out = time_series(file=tiny_nc_empty_tail, var="biomass", region="East",
                      agg_per_step="mean", mask_file=tiny_region_mask)
    vals = np.array([p["value"] for p in out["points"]], dtype=float)
    finite = np.isfinite(vals)
    compacted = float(np.polyfit(np.arange(int(finite.sum()), dtype=float),
                                 vals[finite], 1)[0])
    assert out["summary"]["slope_per_step"] == pytest.approx(compacted, rel=1e-12)


# --- the significance value that underflowed to zero ------------------------
# Live on 2026-09-14 the Mediterranean trend returned p_value exactly 0.0 on 588
# serially correlated monthly values. Zero is not a probability, it is a
# floating-point underflow in 2*(1 - CDF(|z|)). Reviewer E5 lists presenting a
# p-value as a confidence interval among his trust-breakers; presenting an
# underflow as certainty is worse.

def test_a_strong_trend_does_not_report_probability_zero():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    out = _mann_kendall_trend(np.arange(600, dtype=float))
    assert out["p_value"] > 0.0, "p must never underflow to exactly zero"
    assert out["p_value"] < 1e-50
    assert out["z"] < -0.0 or out["z"] > 0.0


def test_the_statistic_and_its_assumption_are_reported():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    out = _mann_kendall_trend(np.arange(600, dtype=float))
    assert "z" in out
    assert out["independence_assumed"] is True
    assert out["alpha"] == 0.1
    assert out["n"] == 600


def test_effective_sample_size_is_reported_for_correlated_data():
    """Monthly biomass is serially correlated; n is not the information count."""
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    out = _mann_kendall_trend(np.arange(600, dtype=float))
    assert out["lag1_autocorrelation"] == pytest.approx(1.0, abs=0.02)
    assert out["n_effective"] is not None
    assert out["n_effective"] < out["n"]


def test_an_uncorrelated_series_keeps_its_full_sample_size():
    import numpy as np
    from surimi_model_mcp.netcdf.analytics import _mann_kendall_trend
    rng = np.random.default_rng(0)
    out = _mann_kendall_trend(rng.normal(size=400))
    assert out["n_effective"] == pytest.approx(out["n"], rel=0.4)


# --- periods given as years --------------------------------------------------
# compare_periods takes STEP INDICES. Live on 2026-09-14, passing calendar years
# sliced past the end of the axis and returned a missing value for all fifteen
# regions with no error at all, which reads as "the data is empty" rather than
# "you passed the wrong kind of number".

def test_periods_outside_the_axis_raise_rather_than_return_nothing(
    tiny_nc_multiyear, tiny_region_mask
):
    from surimi_model_mcp.netcdf.analytics import compare_periods, AnalyticsError
    with pytest.raises(AnalyticsError) as e:
        compare_periods(
            file=tiny_nc_multiyear, var="biomass",
            period_a=(2009, 2009), period_b=(2010, 2010),
            mask_file=tiny_region_mask,
        )
    msg = str(e.value).lower()
    assert "step" in msg and "year" in msg, msg


def test_the_error_names_the_axis_length_and_the_year_range(
    tiny_nc_multiyear, tiny_region_mask
):
    from surimi_model_mcp.netcdf.analytics import compare_periods, AnalyticsError
    with pytest.raises(AnalyticsError) as e:
        compare_periods(
            file=tiny_nc_multiyear, var="biomass",
            period_a=(2009, 2009), period_b=(2010, 2010),
            mask_file=tiny_region_mask,
        )
    msg = str(e.value)
    assert "2009" in msg and "2010" in msg, msg


def test_valid_step_indices_still_work(tiny_nc_multiyear, tiny_region_mask):
    from surimi_model_mcp.netcdf.analytics import compare_periods
    out = compare_periods(
        file=tiny_nc_multiyear, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask,
    )
    assert out["result"]
    assert out["provenance"]["period_unit"] == "step"


# --- extra non-spatial axes (species / fleet), 2026-09-14 -------------------
# Two linked defects on any variable that carries a functional-group axis:
# the year filter indexed dims[0] whatever it was, and the region aggregation
# broadcast the mask over every leading dim, averaging cod and herring
# together. Fixtures tiny_nc_species (time, species, lat, lon) and
# tiny_nc_species_leading (species, time, lat, lon) carry cod=100*(t+1) and
# herring=900*(t+1), so cod 2010=500, herring 2010=4500, cross-species=2500.

def test_year_filter_on_species_leading_var_selects_year_not_species(
    tiny_nc_species_leading, tiny_region_mask
):
    """year=2010 on a (species, time, lat, lon) variable must slice TIME.

    dims[0] is `species` here, so the old code passed time indices [3,4,5] to
    an axis of size 2 and got an IndexError, or on a longer species axis
    silently returned species 3-5 labelled as the year 2010.
    """
    out = top_regions(
        file=tiny_nc_species_leading, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010, select={"species": "cod"},
    )
    assert out["coverage"]["n_timesteps"] == 3
    assert out["result"][0]["value"] == pytest.approx(500.0, rel=1e-3)


def test_year_default_probe_uses_the_real_time_dim(
    tiny_nc_species_leading, tiny_region_mask
):
    """The no-year default probes years through the same dim lookup.

    _year_has_data sliced dims[0] too, so on a species-leading variable the
    probe read species instead of years and the default year was decided from
    the wrong axis.
    """
    out = top_regions(
        file=tiny_nc_species_leading, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, select={"species": "herring"},
    )
    assert out["provenance"]["year"] == 2010
    assert out["provenance"]["year_defaulted"] is True
    assert out["result"][0]["value"] == pytest.approx(4500.0, rel=1e-3)


def test_selecting_one_species_returns_that_species_not_the_mean(
    tiny_nc_species, tiny_region_mask
):
    """select={"species": "cod"} must return 500 for 2010, not 2500.

    2500 is the mean of cod (500) and herring (4500) and is the catch of no
    species that exists: it is the number the silent broadcast reported.
    """
    cod = top_regions(
        file=tiny_nc_species, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010, select={"species": "cod"},
    )
    assert cod["result"][0]["value"] == pytest.approx(500.0, rel=1e-3)
    assert cod["provenance"]["select"] == {"species": "cod"}

    herring = top_regions(
        file=tiny_nc_species, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010, select={"species": "herring"},
    )
    assert herring["result"][0]["value"] == pytest.approx(4500.0, rel=1e-3)


def test_omitting_the_selector_names_the_species_dim_and_its_values(
    tiny_nc_species, tiny_region_mask
):
    """No selector on a species-dimensioned variable must SAY so.

    Averaging cod and herring into one "region" number with no warning is the
    defect. The note must name the dimension and list its coordinate values so
    the caller can re-ask for one of them.
    """
    out = top_regions(
        file=tiny_nc_species, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010,
    )
    note = out["coverage"].get("unselected_dims_note", "")
    assert "species" in note
    assert "cod" in note and "herring" in note
    assert out["coverage"]["unselected_dims"] == {
        "species": ["cod", "herring"],
    }


def test_selecting_a_species_removes_the_note(tiny_nc_species, tiny_region_mask):
    """A resolved selection leaves nothing to warn about."""
    out = top_regions(
        file=tiny_nc_species, var="catch", n=2, agg="mean",
        mask_file=tiny_region_mask, year=2010, select={"species": "cod"},
    )
    assert "unselected_dims_note" not in out["coverage"]
    assert "unselected_dims" not in out["coverage"]


def test_unknown_select_dim_or_value_raises(tiny_nc_species, tiny_region_mask):
    """A typo must fail loudly; silently ignoring it restores the wrong mean."""
    with pytest.raises(AnalyticsError):
        top_regions(
            file=tiny_nc_species, var="catch", n=2, agg="mean",
            mask_file=tiny_region_mask, year=2010, select={"fleet": "trawl"},
        )
    with pytest.raises(AnalyticsError):
        top_regions(
            file=tiny_nc_species, var="catch", n=2, agg="mean",
            mask_file=tiny_region_mask, year=2010, select={"species": "tuna"},
        )


def test_time_series_selects_one_species(tiny_nc_species, tiny_region_mask):
    """time_series must thread the selector too, or the trend mixes species."""
    out = time_series(
        file=tiny_nc_species, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
        select={"species": "cod"},
    )
    assert [p["value"] for p in out["points"]] == pytest.approx(
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    )
    assert out["provenance"]["select"] == {"species": "cod"}


def test_time_series_without_selector_names_the_species_dim(
    tiny_nc_species, tiny_region_mask
):
    out = time_series(
        file=tiny_nc_species, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert out["unselected_dims"] == {"species": ["cod", "herring"]}
    assert "species" in out["unselected_dims_note"]


def test_trend_threads_the_selector(tiny_nc_species, tiny_region_mask):
    """nc_trend delegates to time_series, so the selector must survive the hop."""
    out = nc_trend(
        file=tiny_nc_species, var="catch", region="East",
        mask_file=tiny_region_mask, select={"species": "herring"},
    )
    assert out["direction"] == "increasing"
    assert out["provenance"]["select"] == {"species": "herring"}


def test_compare_periods_selects_one_species(tiny_nc_species, tiny_region_mask):
    """Periods are time-index ranges, so they must index TIME on a 4D variable.

    Steps 0-1 of cod are 100 and 200 (mean 150); steps 4-5 are 500 and 600
    (mean 550); diff 400. Across both species the diff would be 2000.
    """
    out = compare_periods(
        file=tiny_nc_species, var="catch",
        period_a=(0, 1), period_b=(4, 5),
        mask_file=tiny_region_mask, select={"species": "cod"},
    )
    east = {r["region"]: r for r in out["result"]}["East"]
    assert east["a"] == pytest.approx(150.0, rel=1e-3)
    assert east["b"] == pytest.approx(550.0, rel=1e-3)
    assert east["value"] == pytest.approx(400.0, rel=1e-3)


def test_compare_periods_without_selector_names_the_species_dim(
    tiny_nc_species, tiny_region_mask
):
    out = compare_periods(
        file=tiny_nc_species, var="catch",
        period_a=(0, 1), period_b=(4, 5), mask_file=tiny_region_mask,
    )
    assert out["unselected_dims"] == {"species": ["cod", "herring"]}


def test_plain_3d_variables_get_no_note(tiny_nc, tiny_region_mask):
    """A (time, lat, lon) variable has no extra axis, so nothing is warned."""
    out = top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    assert "unselected_dims" not in out["coverage"]
    assert out["result"][0]["value"] == pytest.approx(162.5, rel=1e-3)


def test_time_dim_of_prefers_a_named_time_axis():
    """The helper must find `time` wherever it sits, and keep the dims[0]
    fallback for files whose axis is unnamed."""
    import numpy as np
    import xarray as xr
    from surimi_model_mcp.netcdf.analytics import _time_dim_of

    da = xr.DataArray(np.zeros((2, 6, 3, 4)), dims=("species", "time", "lat", "lon"))
    assert _time_dim_of(da) == "time"

    da_t = xr.DataArray(np.zeros((2, 6, 3)), dims=("species", "t", "lat"))
    assert _time_dim_of(da_t) == "t"

    da_none = xr.DataArray(np.zeros((6, 3, 4)), dims=("step", "lat", "lon"))
    assert _time_dim_of(da_none) == "step"


# ---------- compare_periods: single-region selection ----------

def test_compare_periods_without_region_still_returns_every_region(
    tiny_nc, tiny_region_mask
):
    """Regression guard for existing callers: no region means all regions.

    The `region` parameter is optional, and every caller written before it
    existed passes nothing. If omitting it ever narrowed the result, those
    callers would silently lose regions from a ranking they still present in
    full.
    """
    out = compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="diff",
    )
    assert [r["region"] for r in out["result"]] == ["West", "East"]
    assert out["provenance"]["region"] is None


def test_compare_periods_with_region_returns_only_that_region(
    tiny_nc, tiny_region_mask
):
    """One named region, carrying the same numbers it had in the full result.

    A narrowed computation that changes the value is worse than a slow one: the
    caller cannot tell a cheaper answer from a different answer.
    """
    common = dict(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="diff",
    )
    full = {r["region"]: r for r in compare_periods(**common)["result"]}
    out = compare_periods(region="East", **common)

    assert [r["region"] for r in out["result"]] == ["East"]
    assert out["result"][0]["value"] == pytest.approx(full["East"]["value"])
    assert out["result"][0]["a"] == pytest.approx(full["East"]["a"])
    assert out["result"][0]["b"] == pytest.approx(full["East"]["b"])
    assert out["provenance"]["region"] == "East"


def test_compare_periods_unknown_region_raises_and_lists_the_valid_names(
    tiny_nc, tiny_region_mask
):
    """An unrecognised name must not fall back to all regions.

    Silently widening a request for 'Atlantic' into every region hands back a
    plausible list of numbers that answers a question nobody asked.
    """
    with pytest.raises(AnalyticsError) as exc:
        compare_periods(
            file=tiny_nc, var="biomass",
            period_a=(0, 1), period_b=(2, 3),
            mask_file=tiny_region_mask, region="Atlantic",
        )
    msg = str(exc.value)
    assert "Atlantic" in msg
    assert "West" in msg and "East" in msg


def test_compare_periods_region_skips_the_other_regions_aggregation(
    tiny_nc, tiny_region_mask, monkeypatch
):
    """The point of `region` is not computing the other regions at all.

    Aggregating all fifteen live regions and filtering at the end is what times
    out on thirty-year windows, so assert on the mask actually handed to the
    aggregator rather than on the shape of the result.
    """
    from surimi_model_mcp.netcdf import analytics as analytics_mod

    seen: list[int] = []
    real = analytics_mod._aggregate_per_region

    def spy(data, mask, agg):
        seen.append(mask.shape[0])
        return real(data, mask, agg)

    monkeypatch.setattr(analytics_mod, "_aggregate_per_region", spy)
    analytics_mod.compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, region="East",
    )
    assert seen == [1, 1]


def test_compare_periods_region_composes_with_select(
    tiny_nc_species, tiny_region_mask
):
    """Region and species narrow independently, so both must apply together.

    Cod steps 0-1 mean 150 and steps 4-5 mean 550; across both species the diff
    would be 2000 instead of 400.
    """
    out = compare_periods(
        file=tiny_nc_species, var="catch",
        period_a=(0, 1), period_b=(4, 5),
        mask_file=tiny_region_mask,
        region="East", select={"species": "cod"},
    )
    assert [r["region"] for r in out["result"]] == ["East"]
    assert out["result"][0]["a"] == pytest.approx(150.0, rel=1e-3)
    assert out["result"][0]["b"] == pytest.approx(550.0, rel=1e-3)
    assert out["result"][0]["value"] == pytest.approx(400.0, rel=1e-3)
    assert out["provenance"]["region"] == "East"
    assert out["provenance"]["select"] == {"species": "cod"}
    assert "unselected_dims" not in out


def test_compare_periods_region_composes_with_select_on_a_species_leading_file(
    tiny_nc_species_leading, tiny_region_mask
):
    """Same two narrowings where `dims[0]` is species, not time.

    Periods are time-index ranges, so a file shaped (species, time, lat, lon)
    must still slice time; indexing dims[0] would slice species and return
    herring under a period label.
    """
    out = compare_periods(
        file=tiny_nc_species_leading, var="catch",
        period_a=(0, 1), period_b=(4, 5),
        mask_file=tiny_region_mask,
        region="West", select={"species": "cod"},
    )
    assert [r["region"] for r in out["result"]] == ["West"]
    assert out["result"][0]["value"] == pytest.approx(400.0, rel=1e-3)


# ---------- period bounds must not silently clamp ----------

def test_compare_periods_rejects_a_period_past_the_end_of_the_axis(
        tiny_nc_multiyear, tiny_region_mask):
    """Out-of-range indices used to clamp through numpy slice semantics.

    A 6-step file asked for steps (0, 500) returned a mean over the 6 that
    exist, and provenance recorded period_b as [0, 500]: a signed receipt
    claiming a period the data never covered.
    """
    with pytest.raises(AnalyticsError) as exc:
        compare_periods(
            file=tiny_nc_multiyear, var="biomass",
            period_a=(0, 2), period_b=(0, 500), mask_file=tiny_region_mask)
    msg = str(exc.value)
    assert "500" in msg
    assert "6" in msg


def test_compare_periods_rejects_a_negative_period_bound(
        tiny_nc_multiyear, tiny_region_mask):
    with pytest.raises(AnalyticsError):
        compare_periods(
            file=tiny_nc_multiyear, var="biomass",
            period_a=(-3, 2), period_b=(3, 5), mask_file=tiny_region_mask)


def test_compare_periods_rejects_an_inverted_period(
        tiny_nc_multiyear, tiny_region_mask):
    """end before start selects nothing and used to return an empty mean."""
    with pytest.raises(AnalyticsError):
        compare_periods(
            file=tiny_nc_multiyear, var="biomass",
            period_a=(4, 1), period_b=(0, 2), mask_file=tiny_region_mask)


def test_compare_periods_accepts_the_last_valid_step(
        tiny_nc_multiyear, tiny_region_mask):
    """The bound is inclusive, so the final index must still be allowed."""
    out = compare_periods(
        file=tiny_nc_multiyear, var="biomass",
        period_a=(0, 2), period_b=(3, 5), mask_file=tiny_region_mask)
    assert out["result"]


# ---------- per-step series, opt-in ----------

def test_time_series_points_carry_their_calendar_year(
    tiny_nc_multiyear, tiny_region_mask
):
    """Each point must name the year it falls in.

    Without it a caller who wants one season has to re-decode the raw `t`
    values (days since an epoch the reply does not repeat) to find out which
    points those are, which is the decoding this module already does once.
    """
    out = time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert [p["year"] for p in out["points"]] == [2009, 2009, 2009,
                                                  2010, 2010, 2010]


def test_select_points_windows_to_the_named_years(
    tiny_nc_multiyear, tiny_region_mask
):
    out = time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    window = select_points(out, (2010, 2010))
    assert [p["value"] for p in window["points"]] == pytest.approx(
        [400.0, 500.0, 600.0])
    assert window["window"] == {"years": [2010, 2010], "n_points": 3}


def test_select_points_refuses_a_window_wider_than_the_cap():
    """A cap that truncates silently would sign a partial series as a whole one,
    so the window is refused and the caller narrows it."""
    pts = [{"t": float(i), "value": 1.0, "year": 2000 + i // 12}
           for i in range(MAX_SERIES_POINTS + 12)]
    with pytest.raises(AnalyticsError) as e:
        select_points({"points": pts}, (2000, 2000 + len(pts) // 12))
    msg = str(e.value)
    assert str(MAX_SERIES_POINTS) in msg
    assert str(len(pts)) in msg


def test_select_points_names_the_available_range_when_window_is_empty():
    pts = [{"t": 0.0, "value": 1.0, "year": 2009},
           {"t": 1.0, "value": 2.0, "year": 2010}]
    with pytest.raises(AnalyticsError) as e:
        select_points({"points": pts}, (2020, 2021))
    assert "2009" in str(e.value) and "2010" in str(e.value)


def test_select_points_refuses_an_undecodable_time_axis():
    """No year on the points means the axis did not CF-decode. Saying so beats
    returning every step under a window the caller asked to be narrow."""
    pts = [{"t": 0.0, "value": 1.0}, {"t": 1.0, "value": 2.0}]
    with pytest.raises(AnalyticsError) as e:
        select_points({"points": pts}, (2009, 2009))
    assert "calendar year" in str(e.value)


def test_select_points_refuses_an_inverted_window():
    pts = [{"t": 0.0, "value": 1.0, "year": 2009}]
    with pytest.raises(AnalyticsError):
        select_points({"points": pts}, (2010, 2009))


# ---------- selecting by code, not by position ----------

def test_a_dim_with_no_coordinate_is_labelled_from_its_code_variable(
    tiny_nc_coded_species, tiny_region_mask
):
    """Reported as bare integers, a species axis is unusable.

    The reviewers ask about HKE and PIL. Offering them 0, 1, 2, 3 asks them to
    guess an index whose meaning is in a variable the reply never shows.
    """
    out = time_series(
        file=tiny_nc_coded_species, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert out["unselected_dims"]["species"] == ["HKE", "ANE", "ANE", "PIL"]


def test_select_by_code_picks_that_species(tiny_nc_coded_species, tiny_region_mask):
    out = time_series(
        file=tiny_nc_coded_species, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
        select={"species": "PIL"},
    )
    assert [p["value"] for p in out["points"]] == pytest.approx(
        [400.0, 800.0, 1200.0, 1600.0, 2000.0, 2400.0])


def test_an_ambiguous_code_raises_and_says_what_distinguishes_them(
    tiny_nc_coded_species, tiny_region_mask
):
    """ANE is three rows in the real file, one per life stage. Taking the first
    would answer an adult question with a juvenile number and sign it."""
    with pytest.raises(AnalyticsError) as e:
        time_series(
            file=tiny_nc_coded_species, var="catch", region="East",
            agg_per_step="mean", mask_file=tiny_region_mask,
            select={"species": "ANE"},
        )
    msg = str(e.value)
    assert "1" in msg and "2" in msg           # the positions that match
    assert "species_life_stage" in msg          # what tells them apart
    assert "adult" in msg and "juvenile" in msg


def test_an_integer_position_still_selects(tiny_nc_coded_species, tiny_region_mask):
    """Labels are an addition. A caller who read the positions off an earlier
    reply must not be broken by them, and it is the escape hatch from the
    ambiguity above."""
    out = time_series(
        file=tiny_nc_coded_species, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
        select={"species": 2},
    )
    assert out["points"][0]["value"] == pytest.approx(300.0)


def test_an_unknown_code_names_the_labels_not_the_positions(
    tiny_nc_coded_species, tiny_region_mask
):
    with pytest.raises(AnalyticsError) as e:
        time_series(
            file=tiny_nc_coded_species, var="catch", region="East",
            agg_per_step="mean", mask_file=tiny_region_mask,
            select={"species": "SWO"},
        )
    assert "HKE" in str(e.value)


def test_a_long_enumeration_is_capped_in_the_note():
    """The real file has 94 species and 9 fleets. Listing all of them restores
    the flood the cube shape exists to prevent."""
    from surimi_model_mcp.netcdf.analytics import _unselected_note
    note = _unselected_note({"species": [f"S{i}" for i in range(94)]})
    assert "S0" in note
    assert "S93" not in note
    assert "94" in note


def test_a_dim_with_two_candidate_labels_says_what_describes_it(
    tiny_nc_two_labels, tiny_region_mask
):
    """fleet is a gear AND a country in the real file, so nothing is picked.
    Positions alone leave the caller stuck; naming the describing variables
    tells them what a fleet is and lets them read the codes off the file.
    """
    out = time_series(
        file=tiny_nc_two_labels, var="catch", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    note = out["unselected_dims_note"]
    assert "fleet_gear_code" in note and "fleet_country_code" in note
