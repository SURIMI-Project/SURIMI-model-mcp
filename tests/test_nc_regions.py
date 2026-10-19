"""Tests for the regional mask module."""
from __future__ import annotations

import numpy as np
import pytest

from surimi_model_mcp.netcdf.regions import (
    RegionMaskError,
    load_region_mask,
    validate_alignment,
)


def test_load_region_mask_returns_mask_and_names(tiny_region_mask):
    mask = load_region_mask(tiny_region_mask)
    assert mask["n_regions"] == 2
    assert set(mask["region_names"]) == {"West", "East"}
    assert mask["mask"].shape == (2, 3, 4)  # (region, lat, lon)


def test_load_region_mask_includes_sha256(tiny_region_mask):
    mask = load_region_mask(tiny_region_mask)
    assert len(mask["mask_sha256"]) == 64


def test_load_region_mask_raises_on_missing_file(tmp_path):
    with pytest.raises(RegionMaskError):
        load_region_mask(str(tmp_path / "nope.nc"))


def test_validate_alignment_passes_when_grids_match(tiny_nc, tiny_region_mask):
    """When data and mask share the lat/lon grid, alignment passes."""
    out = validate_alignment(tiny_nc, tiny_region_mask)
    assert out["aligned"] is True


def test_validate_alignment_fails_when_grids_differ(tiny_nc, tmp_path):
    """A mask with different lat/lon should fail alignment."""
    import xarray as xr
    arr = np.zeros((1, 5, 5), dtype="float32")
    ds = xr.Dataset(
        data_vars={"region": (("region_id", "lat", "lon"), arr)},
        coords={"region_id": [1], "lat": np.linspace(0, 10, 5),
                "lon": np.linspace(0, 10, 5)},
    )
    p = tmp_path / "wrongmask.nc"
    ds.to_netcdf(p)
    out = validate_alignment(tiny_nc, str(p))
    assert out["aligned"] is False
    assert "lat" in out["mismatches"] or "lon" in out["mismatches"]


# ---------- a misaligned mask must not produce a signed answer ----------

def _shifted_mask(mask_file, tmp_path):
    """The same mask, same shape, on a different lat/lon grid."""
    import numpy as np
    import xarray as xr

    good = xr.open_dataset(mask_file)
    nlat, nlon = good.sizes["lat"], good.sizes["lon"]
    shifted = good.assign_coords(
        lat=("lat", np.linspace(35.0, 45.0, nlat).astype("float32")),
        lon=("lon", np.linspace(0.0, 20.0, nlon).astype("float32")),
    )
    p = tmp_path / "shifted_mask.nc"
    shifted.to_netcdf(p)
    return str(p)


def test_top_regions_refuses_a_misaligned_mask(tiny_nc, tiny_region_mask, tmp_path):
    """A mask of the right shape on the wrong grid used to answer silently.

    A Mediterranean mask against a global file returned East 162.5 and West
    160.5: plausible numbers, full provenance, and a valid HMAC receipt. The
    signature lent credibility to a number computed over the wrong cells.
    """
    from surimi_model_mcp.netcdf import analytics

    bad = _shifted_mask(tiny_region_mask, tmp_path)
    with pytest.raises(analytics.AnalyticsError) as exc:
        analytics.top_regions(file=tiny_nc, var="biomass", mask_file=bad, n=5)
    msg = str(exc.value).lower()
    assert "align" in msg
    assert "lat" in msg or "lon" in msg


def test_time_series_refuses_a_misaligned_mask(tiny_nc, tiny_region_mask, tmp_path):
    from surimi_model_mcp.netcdf import analytics

    bad = _shifted_mask(tiny_region_mask, tmp_path)
    with pytest.raises(analytics.AnalyticsError):
        analytics.time_series(file=tiny_nc, var="biomass", region="West",
                              mask_file=bad, agg_per_step="mean")


def test_an_aligned_mask_is_still_accepted(tiny_nc, tiny_region_mask):
    """The guard must not reject the masks that already work."""
    from surimi_model_mcp.netcdf import analytics

    out = analytics.top_regions(file=tiny_nc, var="biomass",
                                mask_file=tiny_region_mask, n=5)
    assert out["result"]
