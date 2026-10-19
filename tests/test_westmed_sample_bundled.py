"""Q-E6-5: the bundled westmed sample and its mask stay usable.

Unlike test_reference_samples.py (which exercises the merge logic with tiny
fake files), this pins the REAL bundled pair: they open, their lat/lon grids
agree exactly, and a real species (Lepidotrigla cavillone, the gurnard named
in Q-E6-5's certified answer, code LDV) resolves through the existing
`select` mechanism with no analytics change. Opening a local netcdf file with
xarray is a lazy header/coordinate read, not a full load, so this test stays
fast despite the file being 53 MB on disk.
"""
from __future__ import annotations

import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLE = os.path.join(_HERE, "..", "data", "netcdf-samples",
                        "westmed-2013-2017-biomass-catch-sales.nc")
_MASK = os.path.join(_HERE, "..", "data", "netcdf-samples",
                      "westmed_full_domain_mask.nc")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_SAMPLE) and os.path.exists(_MASK)),
    reason="bundled westmed sample/mask not present in this checkout",
)


def test_bundled_sample_and_mask_are_aligned():
    from surimi_model_mcp.netcdf.regions import validate_alignment

    result = validate_alignment(_SAMPLE, _MASK)
    assert result == {"aligned": True, "mismatches": []}


def test_ldv_gurnard_biomass_resolves_through_the_generic_species_selector():
    """LDV = Lepidotrigla cavillone, the gurnard whose MEDITS decline Q-E6-5's
    certified answer names. No netcdf analytics code changed for this to
    work -- `_apply_select` already handled a `species_code`-labelled axis
    before this fix; the fix was making the file findable at all."""
    from surimi_model_mcp.netcdf import analytics

    result = analytics.time_series(
        file=_SAMPLE, var="biomass", region="westmed_full_domain",
        agg_per_step="mean", mask_file=_MASK, select={"species": "LDV"},
    )
    assert result["provenance"]["select"] == {"species": "LDV"}
    assert result["summary"]["n_points"] == 49  # 2013-01 .. 2017-01, monthly
    assert result["summary"]["n_valid"] == 49
    assert len(result["by_year"]) == 5  # 2013..2017


def test_an_unlisted_species_code_is_a_named_error_not_a_silent_average():
    from surimi_model_mcp.netcdf import analytics

    with pytest.raises(analytics.AnalyticsError, match="not in dimension"):
        analytics.time_series(
            file=_SAMPLE, var="biomass", region="westmed_full_domain",
            agg_per_step="mean", mask_file=_MASK,
            select={"species": "NOT-A-REAL-CODE"},
        )
