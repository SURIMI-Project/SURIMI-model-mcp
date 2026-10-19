"""Reports what a NetCDF file contains by opening it and reading it.

No function assumes a layout. Names such as `time`, `lat` and `lon` are looked up in the file, and the reply states what was found.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from surimi_model_mcp.netcdf import io as ncio


class NetCDFInspectError(Exception):
    """Raised on inspection failure (missing file, unreadable, etc.)."""


def _open(uri: str) -> xr.Dataset:
    try:
        return ncio.open_dataset(uri, decode_times=False)
    except ncio.NetCDFIOError as exc:
        raise NetCDFInspectError(str(exc)) from exc


def describe_file(path: str) -> dict[str, Any]:
    """Return `file_path`, `file_sha256`, `dimensions` (name to size), `variables` and `global_attributes` of a NetCDF file.

    `variables` lists the data variables in the form returned by `list_variables`.
    """
    with _open(path) as ds:
        return {
            "file_path": path,
            "file_sha256": ncio.resource_sha256(path),
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "variables": _variable_list(ds),
            "global_attributes": _coerce_attrs(ds.attrs),
        }


def list_variables(path: str) -> list[dict[str, Any]]:
    """List the data variables, without the coordinates, with `name`, `shape`, `dims`, `dtype`, `units` and `long_name`."""
    with _open(path) as ds:
        return _variable_list(ds)


def _decode_endpoints(t: xr.DataArray) -> tuple[Any, Any] | None:
    """Return the first and last decoded timestamp of a CF time axis, or None when the axis has no `units` or cannot be decoded.

    The decoding is the one `analytics` uses for its year filter, so a noleap or 360_day calendar gives the same years in both. `get_time_range` then reports the raw values with `decoded` False.
    """
    units = t.attrs.get("units")
    if not units:
        return None
    calendar = t.attrs.get("calendar", "standard")
    try:
        decoded = xr.coding.times.decode_cf_datetime(t.values, units, calendar)
    except Exception:
        return None
    if len(decoded) == 0:
        return None
    return decoded[0], decoded[-1]


def get_time_range(path: str) -> dict[str, Any]:
    """Return the start and end of the time axis, raw and decoded to calendar dates.

    `start` and `end` are the raw values on disk. `start_iso`, `end_iso`, `start_year` and `end_year` hold the decoded dates, and `decoded` says whether decoding worked. Raises `NetCDFInspectError` when the file has no time coordinate.
    """
    from surimi_model_mcp.netcdf.analytics import _calendar_years

    with _open(path) as ds:
        time_name = _find_time_coord(ds)
        if time_name is None:
            raise NetCDFInspectError("no time coordinate found")
        t = ds[time_name]
        arr = t.values
        out: dict[str, Any] = {
            "name": time_name,
            "units": str(t.attrs.get("units", "")),
            "calendar": str(t.attrs.get("calendar", "standard")),
            "start": _coerce_value(arr[0]),
            "end": _coerce_value(arr[-1]),
            "n_steps": int(len(arr)),
            "decoded": False,
            "start_iso": None,
            "end_iso": None,
            "start_year": None,
            "end_year": None,
        }
        ends = _decode_endpoints(t)
        if ends is None:
            out["decode_note"] = (
                "time axis carries no usable CF 'units', so start/end are raw "
                "axis values and no calendar date is reported"
            )
            return out
        first, last = ends
        # _calendar_years handles both datetime64 and cftime object arrays.
        years = _calendar_years(np.asarray([first, last]))
        out.update({
            "decoded": True,
            "start_iso": str(first),
            "end_iso": str(last),
            "start_year": int(years[0]),
            "end_year": int(years[-1]),
        })
        return out


def get_spatial_bounds(path: str) -> dict[str, Any]:
    """Return the names, minimum, maximum and cell count of the lat and lon coordinates.

    The names tried are `lat`, `latitude`, `y` and `lon`, `longitude`, `x`. Raises `NetCDFInspectError` when either coordinate is missing.
    """
    with _open(path) as ds:
        lat_name = _find_coord(ds, ["lat", "latitude", "y"])
        lon_name = _find_coord(ds, ["lon", "longitude", "x"])
        if lat_name is None or lon_name is None:
            raise NetCDFInspectError("lat/lon coordinates not found")
        lat = ds[lat_name].values
        lon = ds[lon_name].values
        return {
            "lat_name": lat_name,
            "lon_name": lon_name,
            "lat_min": float(np.min(lat)),
            "lat_max": float(np.max(lat)),
            "lon_min": float(np.min(lon)),
            "lon_max": float(np.max(lon)),
            "lat_n": int(len(lat)),
            "lon_n": int(len(lon)),
        }


def check_cf_compliance(path: str) -> dict[str, Any]:
    """Check a file against the CF conventions with a heuristic.

    `compliant` is True when the `Conventions` attribute starts with "CF" and the file has a latitude and a longitude coordinate. A data variable without `units` adds a warning and does not change `compliant`. Non-compliance is reported in `missing` and `warnings`, not raised.
    """
    with _open(path) as ds:
        missing: list[str] = []
        warnings: list[str] = []

        conv = ds.attrs.get("Conventions")
        if not conv or not str(conv).upper().startswith("CF"):
            missing.append("convention")

        for var_name, var in ds.data_vars.items():
            if "units" not in var.attrs:
                warnings.append(f"variable '{var_name}' missing units")

        # Check lat/lon presence with units
        if _find_coord(ds, ["lat", "latitude"]) is None:
            missing.append("latitude_coord")
        if _find_coord(ds, ["lon", "longitude"]) is None:
            missing.append("longitude_coord")

        return {
            "compliant": not missing,
            "convention": str(conv) if conv else None,
            "missing": missing,
            "warnings": warnings,
        }


# Variables with more cells than this are estimated from sampled steps instead of
# being read in full, because a global 0.25 degree monthly cube holds too many
# cells to load and test for NaN in one tool call.
# The sample is _COVERAGE_SAMPLE_STEPS steps along the leading dimension. The
# missing pattern is not assumed constant along that dimension, so cells that are
# always missing and steps that hold no data are reported apart.
_COVERAGE_FULL_SCAN_LIMIT = 20_000_000
_COVERAGE_SAMPLE_STEPS = 24


def get_coverage_summary(path: str) -> dict[str, Any]:
    """Report the cell count, NaN count and NaN fraction of each data variable.

    A variable with more than `_COVERAGE_FULL_SCAN_LIMIT` cells is estimated from up to `_COVERAGE_SAMPLE_STEPS` evenly spaced slices along its leading dimension and carries `sampled` True. For those, `n_nan` is None, and `nan_fraction_spatial` and `empty_step_fraction` separate cells that are always missing from steps that hold no data. A variable that is not a float counts as fully present.
    """
    with _open(path) as ds:
        vars_out: list[dict[str, Any]] = []
        for name, var in ds.data_vars.items():
            n_total = int(var.size)
            # isnan only valid for float dtypes; treat non-numeric as fully present.
            if var.dtype.kind != "f":
                vars_out.append({
                    "name": name, "n_total": n_total,
                    "n_nan": 0, "nan_fraction": 0.0, "sampled": False,
                })
                continue
            if n_total <= _COVERAGE_FULL_SCAN_LIMIT or not var.dims:
                arr = var.values
                n_nan = int(np.isnan(arr).sum())
                vars_out.append({
                    "name": name, "n_total": n_total,
                    "n_nan": n_nan,
                    "nan_fraction": (n_nan / n_total) if n_total else 0.0,
                    "sampled": False,
                })
                continue
            # Sample slices along the leading dimension and keep two kinds of
            # missingness apart: cells absent at every step (such as a land mask)
            # and steps that carry no data at all.
            lead = var.dims[0]
            length = int(var.sizes[lead])
            n_steps = min(length, _COVERAGE_SAMPLE_STEPS)
            idx = sorted({int(round(i)) for i in np.linspace(0, length - 1, n_steps)})

            per_step_nan: list[float] = []
            for i in idx:
                slice_arr = np.isnan(var.isel({lead: i}).values)
                per_step_nan.append(float(slice_arr.mean()) if slice_arr.size else 0.0)

            empty = [f for f in per_step_nan if f >= 1.0]
            non_empty = [f for f in per_step_nan if f < 1.0]
            empty_step_fraction = len(empty) / len(per_step_nan) if per_step_nan else 0.0
            # The spatial gap is read off the steps that hold something. A step
            # with no data says how complete the axis is, not how much of the
            # grid is missing.
            spatial = float(np.mean(non_empty)) if non_empty else None
            frac = float(np.mean(per_step_nan)) if per_step_nan else 0.0

            vars_out.append({
                "name": name, "n_total": n_total,
                # Nothing counted every cell, so no count is reported.
                "n_nan": None,
                "n_nan_estimated": int(round(frac * n_total)),
                "nan_fraction": frac,
                "nan_fraction_spatial": spatial,
                "empty_step_fraction": empty_step_fraction,
                "sampled": True,
                "n_steps_sampled": len(idx),
                "n_steps_total": length,
                "sample_note": (
                    f"estimated from {len(idx)} of {length} '{lead}' steps; "
                    f"{len(empty)} of those steps held no data at all. "
                    "The missing pattern is not assumed constant along this "
                    "dimension, so the fraction is an estimate, not a count."
                ),
            })
        return {"file_path": path, "variables": vars_out}


# ---------- helpers ----------

def _variable_list(ds: xr.Dataset) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, var in ds.data_vars.items():
        out.append({
            "name": str(name),
            "shape": [int(s) for s in var.shape],
            "dims": list(var.dims),
            "dtype": str(var.dtype),
            "units": str(var.attrs.get("units", "")),
            "long_name": str(var.attrs.get("long_name", "")),
        })
    return out


def _find_time_coord(ds: xr.Dataset) -> str | None:
    for cand in ("time", "t", "T"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    # Look for any coord with time-like units
    for name, coord in ds.coords.items():
        u = str(coord.attrs.get("units", "")).lower()
        if "since" in u and ("day" in u or "hour" in u or "second" in u):
            return str(name)
    return None


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str | None:
    keys = set(ds.coords.keys()) | set(ds.dims)
    for cand in candidates:
        if cand in keys:
            return cand
    return None


def _coerce_attrs(attrs: Any) -> dict[str, Any]:
    """Coerce xarray attrs into JSON-serializable dict (no numpy scalars)."""
    return {str(k): _coerce_value(v) for k, v in dict(attrs).items()}


def _coerce_value(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
