"""Build the whole-domain region mask for the western Mediterranean reference sample.

The regional tools (`model_top_regions`, `model_time_series`, `model_compare_periods`
and `model_trend`) refuse a mask whose lat and lon differ from those of the data file
(`_load_mask_checked` in `netcdf/analytics.py`). The AR6 mask the server serves is global at
0.25 degrees, and the sample is on a 1/12-degree western Mediterranean grid. The repository has no
GFCM sub-area mask. This script therefore writes one region that covers every cell,
using the sample's own coordinates.

The output file is checked in. To rebuild it, run `python3 scripts/build_westmed_reference_mask.py`.
"""
from __future__ import annotations

import os

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLE = os.path.join(_HERE, "..", "data", "netcdf-samples",
                        "westmed-2013-2017-biomass-catch-sales.nc")
_OUT = os.path.join(_HERE, "..", "data", "netcdf-samples",
                     "westmed_full_domain_mask.nc")


def build(sample_path: str = _SAMPLE, out_path: str = _OUT) -> str:
    """Write the whole-domain mask to `out_path`, on the lat/lon grid of `sample_path`, and return `out_path`."""
    with xr.open_dataset(sample_path) as src:
        lat = src["lat"].values.copy()
        lon = src["lon"].values.copy()

    mask = np.ones((1, lat.shape[0], lon.shape[0]), dtype="float32")
    ds = xr.Dataset(
        data_vars={"region": (("region_id", "lat", "lon"), mask)},
        coords={
            "region_id": ("region_id", [1]),
            "region_name": ("region_id", np.array(["westmed_full_domain"], dtype="<U32")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs={
            "title": "Whole-domain region mask for the westmed reference sample",
            "note": (
                "Every cell is region 1. Built to match "
                "westmed-2013-2017-biomass-catch-sales.nc's own grid exactly "
                "(1/12-degree, 82x109) so model_validate_alignment passes "
                "with no GFCM sub-area data. NOT a sub-basin mask: it cannot "
                "answer an Aegean-only question, only a whole-western-Med one."
            ),
        },
    )
    ds.to_netcdf(out_path)
    return out_path


if __name__ == "__main__":
    print(build())
