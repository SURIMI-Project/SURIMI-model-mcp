"""Tests for `surimi_model_mcp.tools.gridded` (imported here as `nc_tools`), which wraps the
gridded-data functions with receipts. `register_all` in `tools/__init__.py` registers each
name in its `TOOLS` list as an MCP tool."""
from __future__ import annotations

import pytest

from surimi_model_mcp.tools import gridded as nc_tools
from surimi_model_mcp.provenance.receipts import verify_receipt


SECRET = b"unit-test-secret"


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", SECRET.decode())


# ---------- inspection tools (no receipts) ----------

def test_nc_list_regions_returns_named_regions(tiny_region_mask):
    out = nc_tools.model_list_regions(tiny_region_mask)
    assert out["n_regions"] == 2
    names = {r["name"] for r in out["regions"]}
    assert names == {"West", "East"}
    # each region in the tiny mask covers half the lon axis: 2 lon x 3 lat = 6 cells
    for r in out["regions"]:
        assert r["n_cells"] == 6


def test_nc_describe_file_exposed_and_works(tiny_nc):
    out = nc_tools.model_describe_file(tiny_nc)
    assert "dimensions" in out
    assert out["dimensions"]["time"] == 4


def test_nc_list_variables_exposed(tiny_nc):
    out = nc_tools.model_list_variables(tiny_nc)
    assert any(v["name"] == "biomass" for v in out)


def test_nc_get_time_range_exposed(tiny_nc):
    out = nc_tools.model_get_time_range(tiny_nc)
    assert out["n_steps"] == 4


def test_nc_get_spatial_bounds_exposed(tiny_nc):
    out = nc_tools.model_get_spatial_bounds(tiny_nc)
    assert out["lat_n"] == 3


def test_nc_check_cf_compliance_exposed(tiny_nc):
    out = nc_tools.model_check_cf_compliance(tiny_nc)
    assert out["compliant"] is True


def test_nc_get_coverage_summary_exposed(tiny_nc):
    out = nc_tools.model_get_coverage_summary(tiny_nc)
    assert any(v["name"] == "biomass" for v in out["variables"])


# ---------- analytical tools (with receipts) ----------

def test_nc_top_regions_returns_value_and_receipt(tiny_nc, tiny_region_mask):
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    assert "value" in out
    assert "receipt" in out
    assert out["receipt"]["tool_id"] == "model_top_regions"


def test_nc_top_regions_receipt_verifies(tiny_nc, tiny_region_mask):
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    v = verify_receipt(out["receipt"], secret=SECRET)
    assert v["verified"] is True


def test_nc_time_series_signed(tiny_nc, tiny_region_mask):
    out = nc_tools.model_time_series(
        file=tiny_nc, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert out["receipt"]["tool_id"] == "model_time_series"
    v = verify_receipt(out["receipt"], secret=SECRET)
    assert v["verified"] is True


def test_nc_time_series_omits_raw_points_returns_cube(tiny_nc_multiyear, tiny_region_mask):
    """The LLM-facing model_time_series must NOT return the raw per-timestep
    points array (the 30B model dumped hundreds of floats into chat). It returns
    the shaped cube slice: compact summary + per-year rollup + provenance, signed
    by one receipt. model_trend keeps its own internal access to the full series."""
    out = nc_tools.model_time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    val = out["value"]
    assert "points" not in val
    assert "summary" in val
    assert "provenance" in val
    assert {r["year"] for r in val["by_year"]} == {2009, 2010}
    assert out["receipt"]["tool_id"] == "model_time_series"
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_nc_compare_periods_signed(tiny_nc, tiny_region_mask):
    out = nc_tools.model_compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="diff",
    )
    assert out["receipt"]["tool_id"] == "model_compare_periods"


def test_nc_trend_signed(tiny_nc, tiny_region_mask):
    out = nc_tools.model_trend(
        file=tiny_nc, var="biomass", region="East",
        mask_file=tiny_region_mask,
    )
    assert out["receipt"]["tool_id"] == "model_trend"


# ---------- verify tool ----------

def test_nc_verify_accepts_good_receipt(tiny_nc, tiny_region_mask):
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    v = nc_tools.model_verify(out["receipt"])
    assert v["verified"] is True


def test_nc_verify_rejects_tampered_receipt(tiny_nc, tiny_region_mask):
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    out["receipt"]["output_value"] = "TAMPERED"
    v = nc_tools.model_verify(out["receipt"])
    assert v["verified"] is False


def test_nc_verify_says_how_it_verified(tiny_nc, tiny_region_mask):
    """A signature is not a recomputation, and the reply has to say so.

    Nothing on either server re-derives a model_* receipt. The data server
    reports that distinction as verification_method; this one returned a bare
    {verified, reason}, so a reviewer asking "can I reproduce this number"
    got an answer whose meaning depended on which server the receipt came
    from. The strings match surimi_data_mcp/provenance/verification.py in SURIMI-data-mcp exactly.
    """
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    v = nc_tools.model_verify(out["receipt"])
    assert v["verified"] is True
    assert v["verification_method"] == "signature_only"
    assert v["recompute_supported"] is False
    assert "not re-derived" in v["recompute_note"]


def test_nc_verify_reports_the_method_on_a_tampered_receipt_too(
        tiny_nc, tiny_region_mask):
    """The method describes what this server does, not how the check came out."""
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    out["receipt"]["output_value"] = "TAMPERED"
    v = nc_tools.model_verify(out["receipt"])
    assert v["verified"] is False
    assert v["verification_method"] == "signature_only"


def test_nc_verify_never_claims_a_recompute(tiny_nc, tiny_region_mask):
    """Guard: this server has no recompute path, so the data server's other
    value must never appear here."""
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    v = nc_tools.model_verify(out["receipt"])
    assert v["verification_method"] != "signature_and_recompute"


# ---------- registry ----------

def test_all_tools_listed_in_registry():
    """`nc_tools.TOOLS` names every callable that `register_all` registers."""
    expected = {
        "model_list_files",
        "model_list_regions",
        "model_describe_file", "model_list_variables", "model_get_time_range",
        "model_get_spatial_bounds", "model_check_cf_compliance",
        "model_get_coverage_summary",
        "model_validate_alignment",
        "model_top_regions", "model_time_series", "model_compare_periods", "model_trend",
        "model_verify",
    }
    assert set(nc_tools.TOOLS) == expected


def test_tools_registry_uses_model_prefix():
    """Every exposed tool is model_*; no nc_* name survives the rename."""
    assert len(nc_tools.TOOLS) == 14
    assert all(name.startswith("model_") for name in nc_tools.TOOLS)
    assert not any(name.startswith("nc_") for name in nc_tools.TOOLS)


def test_every_registered_tool_is_a_module_callable():
    """`register_all` calls `getattr(module, name)` for each entry, so each name must exist."""
    for name in nc_tools.TOOLS:
        assert callable(getattr(nc_tools, name)), f"{name} missing from module"


def test_signed_tools_stamp_model_tool_id(tiny_nc, tiny_region_mask, monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", "test-secret")
    out = nc_tools.model_top_regions(
        file=tiny_nc, var="biomass", n=2, agg="mean", mask_file=tiny_region_mask,
    )
    assert out["receipt"]["tool_id"] == "model_top_regions"


# ---------- alignment guard (signed) ----------

def test_nc_validate_alignment_reports_aligned_grid(tiny_nc, tiny_region_mask):
    """The mask shipped with a data file shares its grid, so the verdict is
    aligned and the caller may trust regional numbers computed with it."""
    out = nc_tools.model_validate_alignment(
        data_file=tiny_nc, mask_file=tiny_region_mask,
    )
    assert out["value"]["aligned"] is True
    assert out["value"]["mismatches"] == []


def test_nc_validate_alignment_reports_misaligned_grid(tiny_nc, tmp_path):
    """A user-supplied mask on a different grid must come back misaligned with
    the offending axes named. Without this tool the same mask reaches
    model_top_regions and returns numbers that look ordinary and are wrong."""
    import numpy as np
    import xarray as xr
    arr = np.zeros((1, 5, 5), dtype="float32")
    ds = xr.Dataset(
        data_vars={"region": (("region_id", "lat", "lon"), arr)},
        coords={"region_id": [1], "lat": np.linspace(0, 10, 5),
                "lon": np.linspace(0, 10, 5)},
    )
    p = tmp_path / "wrongmask.nc"
    ds.to_netcdf(p)
    out = nc_tools.model_validate_alignment(data_file=tiny_nc, mask_file=str(p))
    assert out["value"]["aligned"] is False
    assert set(out["value"]["mismatches"]) == {"lat", "lon"}


def test_nc_validate_alignment_is_receipt_signed(tiny_nc, tiny_region_mask):
    """An alignment verdict is the basis for trusting or discarding a figure, so
    it is signed like the figures themselves and can be re-checked later."""
    out = nc_tools.model_validate_alignment(
        data_file=tiny_nc, mask_file=tiny_region_mask,
    )
    assert out["receipt"]["tool_id"] == "model_validate_alignment"
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_nc_validate_alignment_verdict_survives_model_verify(tiny_nc, tiny_region_mask):
    """A misaligned verdict must be verifiable through the same model_verify
    path as analytical results, so a reviewer can confirm the warning was real."""
    out = nc_tools.model_validate_alignment(
        data_file=tiny_nc, mask_file=tiny_region_mask,
    )
    assert nc_tools.model_verify(out["receipt"])["verified"] is True


# ---------- server registration ----------

def test_validate_alignment_is_registered_and_claimed_in_server():
    """`register_all` registers every `nc_tools.TOOLS` entry with FastMCP, and the server
    runs `drift_check` against the capability registry at startup. A tool missing
    from either side is invisible to the client."""
    import asyncio

    from surimi_model_mcp import capabilities as capabilities
    from surimi_model_mcp import server as server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert "model_validate_alignment" in names

    drift = capabilities.drift_check(names)
    assert drift["missing_from_registry"] == []
    assert drift["missing_from_fastmcp"] == []


def test_nc_compare_periods_accepts_region_and_signs_it(tiny_nc, tiny_region_mask):
    """The MCP wrapper is a direct wrap, so `region` must reach it unaltered.

    with_receipt binds the wrapped function's own signature, so a parameter the
    analytics function grows is exposed and signed without a hand-written
    passthrough. Assert that rather than trusting it.
    """
    out = nc_tools.model_compare_periods(
        file=tiny_nc, var="biomass",
        period_a=(0, 1), period_b=(2, 3),
        mask_file=tiny_region_mask, op="diff", region="East",
    )
    assert [r["region"] for r in out["value"]["result"]] == ["East"]
    assert out["value"]["provenance"]["region"] == "East"
    assert out["receipt"]["tool_id"] == "model_compare_periods"
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_model_time_series_can_select_one_species(tiny_nc_species, tiny_region_mask,
                                                  monkeypatch):
    """model_time_series is the one analytics wrapper written by hand.

    The others wrap the analytics function directly, so `select` reached them
    for free. This one rebuilds the signature, so it silently dropped the
    selector and averaged across species with no way for a caller to say which
    one they meant.
    """
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", "test-secret")
    out = nc_tools.model_time_series(
        file=tiny_nc_species, var="catch", region="West",
        agg_per_step="mean", mask_file=tiny_region_mask,
        select={"species": "herring"},
    )
    assert out["value"]["provenance"]["select"] == {"species": "herring"}
    assert "unselected_dims" not in out["value"]


def test_model_time_series_without_select_names_the_species_dim(
        tiny_nc_species, tiny_region_mask, monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", "test-secret")
    out = nc_tools.model_time_series(
        file=tiny_nc_species, var="catch", region="West",
        agg_per_step="mean", mask_file=tiny_region_mask,
    )
    assert "species" in out["value"]["unselected_dims"]


def test_nc_time_series_returns_points_only_when_a_window_is_asked_for(
    tiny_nc_multiyear, tiny_region_mask
):
    """The raw series stays off by default and comes back bounded when asked.

    Three expert questions need within-year resolution (a seasonal cycle, a
    spike in one year) and the year rollup cannot show it. Removing the filter
    would put hundreds of monthly floats back in the reply, so the series is
    opt-in and windowed to named years.
    """
    out = nc_tools.model_time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
        points_for_years=[2010, 2010],
    )
    val = out["value"]
    assert [p["value"] for p in val["points"]] == pytest.approx(
        [400.0, 500.0, 600.0])
    assert val["points_window"] == {"years": [2010, 2010], "n_points": 3}
    # the shaped cube is still there: a window narrows, it does not replace
    assert "summary" in val
    assert {r["year"] for r in val["by_year"]} == {2009, 2010}
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_nc_time_series_window_is_recorded_in_signed_provenance(
    tiny_nc_multiyear, tiny_region_mask
):
    """The window changes which points are shown, so it belongs inside the
    signature: a receipt must not be re-readable as covering the whole series."""
    out = nc_tools.model_time_series(
        file=tiny_nc_multiyear, var="biomass", region="East",
        agg_per_step="mean", mask_file=tiny_region_mask,
        points_for_years=[2009, 2009],
    )
    assert out["value"]["provenance"]["points_for_years"] == [2009, 2009]
    assert out["receipt"]["input_params"]["points_for_years"] == [2009, 2009]


def test_nc_time_series_rejects_a_malformed_window(
    tiny_nc_multiyear, tiny_region_mask
):
    with pytest.raises(Exception) as e:
        nc_tools.model_time_series(
            file=tiny_nc_multiyear, var="biomass", region="East",
            agg_per_step="mean", mask_file=tiny_region_mask,
            points_for_years=[2010],
        )
    assert "two" in str(e.value)
