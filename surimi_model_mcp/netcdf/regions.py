"""Loads region masks and checks them against a data file's grid.

A region mask is a NetCDF file whose 3D variable `region` has the shape (region, lat, lon). The analytics treat every cell with a weight above 0 as inside the region. Region names come from `region_name` and fall back to `region_<id>`. `validate_alignment` checks that the `lat` and `lon` of a mask match those of a data file.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from surimi_model_mcp.netcdf import io as ncio


class RegionMaskError(Exception):
    """Raised on mask load / alignment failure."""


def load_region_mask(path: str) -> dict[str, Any]:
    """Load a region mask NetCDF from a local path or `s3://` URI.

    The file needs a 3D variable `region`. Names come from the coordinate or variable `region_name`, else `region_<id>` from `region_id` or the position of the region.

    Returns `{n_regions, region_names, mask, lat, lon, mask_sha256}`. Raises `RegionMaskError` when the file cannot be opened, has no `region` variable or the variable is not 3D.
    """
    try:
        ds = ncio.open_dataset(path)
    except ncio.NetCDFIOError as exc:
        raise RegionMaskError(str(exc)) from exc

    with ds:
        if "region" not in ds:
            raise RegionMaskError("mask file must contain a 'region' variable")
        mask = ds["region"].values
        if mask.ndim != 3:
            raise RegionMaskError(
                f"region mask must be 3D (region, lat, lon), got shape {mask.shape}"
            )
        n_regions = mask.shape[0]

        # Resolve region names: prefer coord `region_name`, else use ids
        if "region_name" in ds.coords:
            names = [_decode(v) for v in ds["region_name"].values]
        elif "region_name" in ds:
            names = [_decode(v) for v in ds["region_name"].values]
        else:
            ids = ds["region_id"].values if "region_id" in ds else range(n_regions)
            names = [f"region_{i}" for i in ids]

        lat = ds["lat"].values if "lat" in ds else None
        lon = ds["lon"].values if "lon" in ds else None

        return {
            "n_regions": int(n_regions),
            "region_names": names,
            "mask": mask,  # ndarray (region, lat, lon)
            "lat": lat,
            "lon": lon,
            "mask_sha256": ncio.resource_sha256(path),
        }


def validate_alignment(data_file: str, mask_file: str) -> dict[str, Any]:
    """Compare the `lat` and `lon` of the data file with those of the mask and return `{aligned, mismatches}`.

    An axis matches when its shape is equal and its values agree within 1e-3. `mismatches` holds `lat` or `lon` for an axis that differs, and `lat_missing` or `lon_missing` when either file has no such variable. Raises `RegionMaskError` when a file cannot be opened.
    """
    try:
        d = ncio.open_dataset(data_file)
        m = ncio.open_dataset(mask_file)
    except ncio.NetCDFIOError as exc:
        raise RegionMaskError(str(exc)) from exc

    with d, m:
        mismatches: list[str] = []
        if "lat" in d and "lat" in m:
            if d["lat"].shape != m["lat"].shape or not np.allclose(
                d["lat"].values, m["lat"].values, atol=1e-3
            ):
                mismatches.append("lat")
        else:
            mismatches.append("lat_missing")
        if "lon" in d and "lon" in m:
            if d["lon"].shape != m["lon"].shape or not np.allclose(
                d["lon"].values, m["lon"].values, atol=1e-3
            ):
                mismatches.append("lon")
        else:
            mismatches.append("lon_missing")

        return {"aligned": not mismatches, "mismatches": mismatches}


def _decode(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    return str(v)
