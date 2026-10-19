"""Analytical functions over NetCDF data and a region mask.

`top_regions`, `time_series`, `compare_periods` and `nc_trend` each open the file, apply the region mask, aggregate on the server and return the result with its provenance. The tools in `tools/gridded.py` wrap them with `with_receipt`, which signs the calculation record of the result. `select_points` picks the points of a `time_series` result inside a year window.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from surimi_model_mcp.netcdf import io as ncio
from surimi_model_mcp.netcdf.regions import (RegionMaskError, load_region_mask,
                            validate_alignment)

_log = logging.getLogger(__name__)


class AnalyticsError(Exception):
    """Raised on primitive execution failure."""


_VALID_AGGS = {"mean", "sum", "max", "min"}

#: Most per-step points `select_points` returns for one window (ten years of
#: monthly steps). A window over the limit raises an error and is not cut,
#: because a cut series would carry the same signed calculation record as a whole one.
MAX_SERIES_POINTS = 120


def _file_sha256(path: str) -> str:
    try:
        return ncio.resource_sha256(path)
    except ncio.NetCDFIOError as exc:
        raise AnalyticsError(str(exc)) from exc


def _open(path: str) -> xr.Dataset:
    try:
        return ncio.open_dataset(path, decode_times=False)
    except ncio.NetCDFIOError as exc:
        raise AnalyticsError(str(exc)) from exc


def _calendar_years(decoded: np.ndarray) -> np.ndarray:
    """Return the calendar year of each decoded timestep.

    `decode_cf_datetime` returns `datetime64` for standard calendars and an object array of `cftime` objects for noleap, 365_day, 360_day and similar calendars. An object array takes the `.year` of each value, because converting it with `datetime64[Y]` gives wrong years. Raises `AnalyticsError` when a value has no `.year`.
    """
    decoded = np.asarray(decoded)
    if decoded.dtype == object:
        try:
            return np.array(
                [d.year for d in decoded.ravel()]
            ).reshape(decoded.shape)
        except AttributeError as exc:
            raise AnalyticsError(
                f"could not extract calendar year from decoded time axis: {exc}"
            ) from exc
    return decoded.astype("datetime64[Y]").astype(int) + 1970


def _find_time_name(ds: xr.Dataset) -> str | None:
    """Name of the dataset's time coordinate/dimension, or None if absent."""
    for cand in ("time", "t", "T"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    return None


def _time_dim_of(var_da: xr.DataArray) -> str:
    """Return the variable's time dimension (`time`, `t` or `T`), else its first dimension.

    The fallback covers files whose time axis has another name, such as `step`. A variable shaped (species, time, lat, lon) has time as its second dimension, so the first dimension cannot be assumed to be time.
    """
    for cand in ("time", "t", "T"):
        if cand in var_da.dims:
            return cand
    return str(var_da.dims[0])


#: Most coordinate values named in one message. The total is always stated, so a
#: cut list does not read as the whole one.
_MAX_ENUMERATED = 12


def _decode_label(v: Any) -> Any:
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="replace")
    return v.item() if hasattr(v, "item") else v


def _label_variables(ds: xr.Dataset, dim: str) -> list[str]:
    """1-D text variables defined along `dim`, which name its positions."""
    return sorted(
        str(name) for name, da in ds.variables.items()
        if str(name) != dim and da.dims == (dim,) and da.dtype.kind in "USO"
    )


def _label_source(ds: xr.Dataset, dim: str) -> str | None:
    """Return the variable that labels `dim`, or None when no single variable does.

    The labels sit in 1-D text variables along `dim`, for example `species_code` along `species`. `<dim>_code` wins where it exists. Several candidates without it give None, and the positions stay integers.
    """
    candidates = _label_variables(ds, dim)
    if f"{dim}_code" in candidates:
        return f"{dim}_code"
    return candidates[0] if len(candidates) == 1 else None


def _coord_values(ds: xr.Dataset, dim: str) -> list[Any]:
    """Labels for `dim`: its coordinate, else its code variable, else positions."""
    if dim in ds.coords:
        return [_decode_label(v)
                for v in np.asarray(ds[dim].values).ravel().tolist()]
    source = _label_source(ds, dim)
    if source is not None:
        return [_decode_label(v)
                for v in np.asarray(ds[source].values).ravel().tolist()]
    return list(range(int(ds.sizes[dim])))


def _abbreviate(values: list[Any]) -> str:
    """Join the first `_MAX_ENUMERATED` values with commas, and state the total when there are more."""
    shown = ", ".join(str(v) for v in values[:_MAX_ENUMERATED])
    if len(values) <= _MAX_ENUMERATED:
        return shown
    return f"{shown}, ... ({len(values)} in total)"


def _extra_dims_of(
    ds: xr.Dataset, var_da: xr.DataArray, tdim: str | None
) -> dict[str, list[Any]]:
    """Return the non-time, non-spatial dimensions of the variable with their coordinate values.

    The region mask does not cover these axes, such as species, functional group, fleet or depth layer. A dimension left unselected is averaged into the region value.
    """
    spatial = {"lat", "lon", "latitude", "longitude", "x", "y"}
    extra: dict[str, list[Any]] = {}
    for d in var_da.dims:
        d = str(d)
        if d == tdim or d in spatial:
            continue
        extra[d] = _coord_values(ds, d)
    return extra


def _apply_select(
    ds: xr.Dataset,
    var_da: xr.DataArray,
    select: dict[str, Any] | None,
    tdim: str | None,
) -> tuple[xr.DataArray, dict[str, list[Any]]]:
    """Reduce `var_da` to one coordinate value for each dimension in `select`.

    Returns the lazily selected variable and the extra dimensions that remain unselected. A dimension or value the variable does not have raises `AnalyticsError`, and so does a value found at several positions. An integer that matches no label selects that position.
    """
    if select:
        for dim, value in select.items():
            if dim not in var_da.dims:
                raise AnalyticsError(
                    f"cannot select on '{dim}': variable dims are "
                    f"{[str(d) for d in var_da.dims]}"
                )
            if dim == tdim:
                raise AnalyticsError(
                    f"cannot select on the time dimension '{dim}'; use `year` "
                    f"or a period instead"
                )
            available = _coord_values(ds, dim)
            matches = [i for i, v in enumerate(available) if v == value]
            if len(matches) > 1:
                # A code is not always a key: it can appear at several positions,
                # for example once per life stage. Taking the first would answer
                # for the wrong group.
                raise AnalyticsError(
                    f"'{value}' is not unique on dimension '{dim}': it is at "
                    f"positions {matches}. {_disambiguators(ds, dim, matches)} "
                    f"Select by position to choose one."
                )
            if not matches:
                # An integer with no label match selects the position it names.
                # This is the way out of the ambiguity above.
                if isinstance(value, int) and not isinstance(value, bool) \
                        and 0 <= value < int(ds.sizes[dim]):
                    var_da = var_da.isel({dim: value})  # lazy
                    continue
                raise AnalyticsError(
                    f"'{value}' not in dimension '{dim}'; available: "
                    f"{_abbreviate(available)}"
                )
            var_da = var_da.isel({dim: matches[0]})  # lazy
    return var_da, _extra_dims_of(ds, var_da, tdim)


def _disambiguators(ds: xr.Dataset, dim: str, positions: list[int]) -> str:
    """Name the other label variables and their values at the tied `positions`.

    The caller can then tell the matches apart, for example `species_life_stage (1='adult', 2='juvenile')`.
    """
    others = [v for v in _label_variables(ds, dim) if v != _label_source(ds, dim)]
    if not others:
        return "Nothing else on this dimension tells them apart."
    parts = []
    for name in others:
        vals = np.asarray(ds[name].values).ravel().tolist()
        pairs = ", ".join(f"{i}={_decode_label(vals[i])!r}" for i in positions)
        parts.append(f"{name} ({pairs})")
    return "They differ by " + "; ".join(parts) + "."


def _unselected_note(
    extra: dict[str, list[Any]], ds: xr.Dataset | None = None
) -> str:
    """Return the sentence that names the dimensions averaged together and says how to select one.

    Where `ds` is given, a dimension with no label of its own is named together with the variables that describe it.
    """
    parts = []
    for dim, values in extra.items():
        part = f"'{dim}' ({_abbreviate(values)})"
        if ds is not None and dim not in ds.coords and _label_source(ds, dim) is None:
            described_by = _label_variables(ds, dim)
            if described_by:
                part += f", described by {' and '.join(described_by)}"
        parts.append(part)
    return (
        f"This variable also has the dimension(s) {'; '.join(parts)}, and no "
        f"selection was made, so the reported value AVERAGES across them. That "
        f"average belongs to no single group. Pass select={{dim: value}} to get "
        f"one of them."
    )


def _decode_years(ds: xr.Dataset) -> np.ndarray | None:
    """Return the calendar year of each timestep on the dataset's time axis.

    Returns None when there is no time coordinate, it has no CF `units` or decoding fails (logged as a warning), and the caller then skips the year rollup. The decoding is the one `_time_indices_for_year` uses.
    """
    tname = _find_time_name(ds)
    if tname is None:
        return None
    t = ds[tname]
    units = t.attrs.get("units")
    if not units:
        return None
    calendar = t.attrs.get("calendar", "standard")
    try:
        decoded = xr.coding.times.decode_cf_datetime(t.values, units, calendar)
    except Exception as exc:
        # Graceful fallback to no rollup, but make the cause observable.
        _log.warning("could not decode time axis for year rollup: %s", exc)
        return None
    return _calendar_years(decoded)


def _time_indices_for_year(ds: xr.Dataset, year: int) -> np.ndarray:
    """Return the indices along the time axis whose decoded calendar year is `year`.

    The dataset is opened with `decode_times=False`, so the raw time coordinate is decoded here with its CF `units`. Raises `AnalyticsError` when there is no time coordinate or `units`, when decoding fails or when `year` is not on the axis.
    """
    tname = _find_time_name(ds)
    if tname is None:
        raise AnalyticsError("no time coordinate to filter by year")
    t = ds[tname]
    units = t.attrs.get("units")
    if not units:
        raise AnalyticsError(f"time coord '{tname}' has no units; cannot filter by year")
    calendar = t.attrs.get("calendar", "standard")
    try:
        decoded = xr.coding.times.decode_cf_datetime(t.values, units, calendar)
    except Exception as exc:
        raise AnalyticsError(f"could not decode time axis: {exc}") from exc
    years = _calendar_years(decoded)
    idx = np.where(years == year)[0]
    if idx.size == 0:
        raise AnalyticsError(
            f"year {year} not present in time axis (range "
            f"{int(years.min())}-{int(years.max())})"
        )
    return idx


def _steps_per_year(years) -> float | None:
    """Return the median number of time steps per calendar year, or None when it cannot be determined.

    It is measured on the decoded axis. The first and last year are left out when the axis has three or more years, because they are usually partial.
    """
    if years is None:
        return None
    arr = np.asarray(years)
    if arr.size == 0:
        return None
    uniq, counts = np.unique(arr, return_counts=True)
    if uniq.size < 3:
        return float(np.median(counts)) if counts.size else None
    interior = counts[1:-1]
    return float(np.median(interior)) if interior.size else None


def _step_unit(steps_per_year: float | None) -> str:
    """Name the time step from how many of them make a year: `year`, `quarter`, `month` or `day`, else `step`."""
    if steps_per_year is None:
        return "step"
    for n, name in ((1, "year"), (4, "quarter"), (12, "month"), (365, "day"), (366, "day")):
        if abs(steps_per_year - n) < 0.5:
            return name
    return "step"


def _series_summary(
    points: list[dict[str, Any]],
    years=None,
    units: str = "unknown",
) -> dict[str, Any]:
    """Summarise a time series without its points.

    Returns `n_points`, `n_valid` and, for the valid steps, `value_start`, `value_end`, `value_min`, `value_max`, `value_mean`, `slope_per_step`, `direction`, `steps_per_year`, `step_unit`, `slope_unit` and `slope_per_year_scaled`. NaN steps are excluded. When every step is missing, only the counts and a `note` are returned.
    """
    n_points = len(points)
    vals = np.array([p["value"] for p in points], dtype=float)
    finite = np.isfinite(vals)
    n_valid = int(finite.sum())
    summary: dict[str, Any] = {"n_points": n_points, "n_valid": n_valid}
    if n_valid == 0:
        summary["note"] = "all timesteps are missing (NaN)"
        return summary

    v = vals[finite]
    if n_valid >= 2:
        # The fit uses the original step positions, so a gap inside the series
        # counts as elapsed time, as it does in `_mann_kendall_trend`.
        positions = np.arange(len(vals), dtype=float)[finite]
        slope = float(np.polyfit(positions, v, 1)[0])
    else:
        slope = 0.0
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    summary.update(
        {
            "value_start": float(v[0]),
            "value_end": float(v[-1]),
            "value_min": float(v.min()),
            "value_max": float(v.max()),
            "value_mean": float(v.mean()),
            "slope_per_step": slope,
            "direction": direction,
        }
    )
    # Name the step and give the slope per year, so a reader does not scale it by hand.
    spy = _steps_per_year(years)
    step_unit = _step_unit(spy)
    summary["steps_per_year"] = spy
    summary["step_unit"] = step_unit
    summary["slope_unit"] = f"{units} per {step_unit}"
    summary["slope_per_year_scaled"] = (
        float(slope * spy) if spy is not None else None
    )
    return summary


def _rollup_by_year(
    values: np.ndarray | Sequence[float],
    years: np.ndarray | Sequence[int],
    agg: str,
) -> list[dict[str, Any]]:
    """Group a per-timestep series by calendar year and aggregate each year with `agg`.

    Returns one `{year, value, n_steps}` row per year, in ascending order. NaN steps are left out of the aggregate and still counted in `n_steps`. A year with no finite step has `value` None.
    """
    vals = np.asarray(values, dtype=float)
    yrs = np.asarray(years)
    rows: list[dict[str, Any]] = []
    for y in sorted(np.unique(yrs).tolist()):
        sel = yrs == y
        subset = vals[sel]
        finite = subset[np.isfinite(subset)]
        value = _apply_agg(finite, agg) if finite.size else None
        rows.append({"year": int(y), "value": value, "n_steps": int(sel.sum())})
    return rows


def _load_mask_checked(file: str, mask_file: str) -> dict[str, Any]:
    """Load a region mask after checking that it sits on the data file's grid.

    Raises `AnalyticsError` when `validate_alignment` reports a mismatch, when a file cannot be opened or when the mask cannot be loaded. The mismatch message points to `model_validate_alignment`.
    """
    try:
        check = validate_alignment(file, mask_file)
    except RegionMaskError as exc:
        raise AnalyticsError(str(exc)) from exc
    if not check["aligned"]:
        raise AnalyticsError(
            f"region mask '{mask_file}' is not aligned with data file "
            f"'{file}': {', '.join(check['mismatches'])} differ. Regional "
            f"values computed with it would be read off the wrong cells. "
            f"Call model_validate_alignment for the detail."
        )
    try:
        return load_region_mask(mask_file)
    except RegionMaskError as exc:
        raise AnalyticsError(str(exc)) from exc


def _region_index(mask: dict[str, Any], region: str) -> int:
    try:
        return mask["region_names"].index(region)
    except ValueError as exc:
        raise AnalyticsError(
            f"region '{region}' not in mask; valid: {mask['region_names']}"
        ) from exc


def _units_of(var_da) -> str:
    """Return the `units` attribute of the variable, or "unknown" when it is absent or empty."""
    u = str(var_da.attrs.get("units", "") or "").strip()
    return u or "unknown"


def _apply_agg(values: np.ndarray, agg: str) -> float | None:
    """Apply `mean`, `sum`, `max` or `min` to the non-NaN values, or return None when there are none.

    `np.nansum` of an all-NaN array returns 0.0, which would read as a measured zero, so a selection with no values returns None. Callers must not sort or calculate with None. Raises `AnalyticsError` for any other `agg`.
    """
    if agg not in _VALID_AGGS:
        raise AnalyticsError(f"unknown agg '{agg}', valid: {sorted(_VALID_AGGS)}")
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return None
    if agg == "mean":
        return float(np.nanmean(values))
    if agg == "sum":
        return float(np.nansum(values))
    if agg == "max":
        return float(np.nanmax(values))
    if agg == "min":
        return float(np.nanmin(values))
    raise AnalyticsError(f"unreachable: {agg}")


def _aggregate_per_region(
    data: np.ndarray, mask: np.ndarray, agg: str
) -> tuple[list[float | None], list[int]]:
    """For each region, aggregate `data` over cells where mask > 0.

    Returns (values, n_cells_per_region).
    `data` has shape (..., lat, lon); mask has shape (region, lat, lon).
    """
    n_regions = mask.shape[0]
    values: list[float | None] = []
    n_cells: list[int] = []
    for r in range(n_regions):
        weights = mask[r]
        # Broadcast mask over leading dims of data
        weighted = np.where(weights > 0, data, np.nan)
        values.append(_apply_agg(weighted, agg))
        n_cells.append(int(np.sum(weights > 0)))
    return values, n_cells


#: Most years `_latest_year_with_data` tests, newest first, before it gives up.
#: The limit bounds the cost on a file that is empty throughout.
_YEAR_PROBE_LIMIT = 30

#: Stride along every non-time dimension for the cheap probe of whether a year
#: holds data. A probe that finds nothing is repeated on the full year, so a year
#: with sparse data is never skipped.
_YEAR_PROBE_STRIDE = 8


def _year_has_data(ds: xr.Dataset, var_da: xr.DataArray, year: int) -> bool:
    """True if `year` holds at least one finite value for this variable."""
    idx = _time_indices_for_year(ds, year)
    if idx.size == 0:
        return False
    tdim = _time_dim_of(var_da)
    sliced = var_da.isel({tdim: idx})

    # Stride every dimension except time, so the probe thins the grid and not
    # the year being tested.
    probe = sliced.isel(
        {d: slice(None, None, _YEAR_PROBE_STRIDE)
         for d in sliced.dims if d != tdim}
    )
    if bool(np.isfinite(probe.values).any()):
        return True
    # The probe found nothing, so check the full year before rejecting it.
    return bool(np.isfinite(sliced.values).any())


def _latest_year_with_data(
    ds: xr.Dataset, var_da: xr.DataArray, years_axis: np.ndarray
) -> int:
    """Return the newest year in `years_axis` that holds data, used when no year is given.

    It tests up to `_YEAR_PROBE_LIMIT` years, newest first. When none of them has data it returns the newest year, and the result then reports a `nan_fraction` of 1.0.
    """
    candidates = sorted({int(y) for y in years_axis}, reverse=True)
    for year in candidates[:_YEAR_PROBE_LIMIT]:
        if _year_has_data(ds, var_da, year):
            return year
    return int(candidates[0])


def top_regions(
    file: str,
    var: str,
    n: int,
    mask_file: str,
    agg: str = "mean",
    year: int | None = None,
    select: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top N regions by aggregated value of `var`, optionally within one year.

    When `year` is given, only that calendar year's timesteps are loaded
    (lazy .isel before .values), so global multi-decade files do not blow
    memory. When `year` is omitted on a file that spans MULTIPLE years, the
    most recent year THAT ACTUALLY HAS DATA is used automatically (reported in
    provenance.year with year_defaulted=true + a coverage note) -- aggregating
    every year of a global file is slow enough to time out. Pass an explicit
    year to cover a different period; an explicit year is always honoured as
    given, even if it is empty. Returns a rank-ordered list with values,
    provenance, and coverage.

    `agg` defaults to "mean": it previously had no default, so the natural
    tool call omitted it and died on argument validation before reaching data.

    `select` is {dimension: coordinate_value} for any extra non-time,
    non-spatial axis the variable carries (species, functional group, fleet).
    Without it a catch variable shaped (time, species, lat, lon) is averaged
    across cod and herring into one number belonging to neither, so when such a
    dimension is left unselected the result carries coverage.unselected_dims
    and a note listing the dimension and its values.
    """
    if agg not in _VALID_AGGS:
        raise AnalyticsError(f"unknown agg '{agg}', valid: {sorted(_VALID_AGGS)}")

    mask_info = _load_mask_checked(file, mask_file)

    ds = _open(file)
    with ds:
        if var not in ds.data_vars:
            raise AnalyticsError(
                f"variable '{var}' not found; available: {list(ds.data_vars)}"
            )
        var_da = ds[var]
        units = _units_of(var_da)
        tdim_name = _time_dim_of(var_da) if var_da.ndim else None
        var_da, extra_dims = _apply_select(ds, var_da, select, tdim_name)

        # With no year given on a file that spans more than one year, use the
        # most recent year with data (a lazy .isel slice) and report the
        # substitution. Aggregating every year of a global file is slow enough
        # to time out. Files with one year or no time axis keep the full path.
        year_defaulted = False
        if year is None:
            years_axis = _decode_years(ds)
            if years_axis is not None and years_axis.size and \
                    np.unique(years_axis).size > 1:
                year = _latest_year_with_data(ds, var_da, years_axis)
                year_defaulted = True

        n_timesteps = None
        if year is not None:
            idx = _time_indices_for_year(ds, year)
            tdim = _time_dim_of(var_da)
            var_da = var_da.isel({tdim: idx})  # lazy
            n_timesteps = int(idx.size)

        data = var_da.values  # materialize only the (optionally subset) slice

    values, _ = _aggregate_per_region(data, mask_info["mask"], agg)
    nan_frac = float(np.isnan(data).sum() / max(1, data.size))

    pairs = list(zip(mask_info["region_names"], values))
    # Regions with no data sort last and keep their None.
    def _rank_key(pair):
        v = pair[1]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return (1, 0.0)
        return (0, -v)

    pairs.sort(key=_rank_key)
    top = pairs[: max(0, n)]

    coverage: dict[str, Any] = {
        "nan_fraction": nan_frac,
        "n_regions_evaluated": mask_info["n_regions"],
    }
    if n_timesteps is not None:
        coverage["n_timesteps"] = n_timesteps
    if year_defaulted:
        coverage["note"] = (
            f"No year was given, so this covers the most recent available "
            f"year ({year}), not all years. Pass an explicit year to "
            f"aggregate a different period."
        )
    if extra_dims:
        coverage["unselected_dims"] = extra_dims
        coverage["unselected_dims_note"] = _unselected_note(extra_dims, ds)

    return {
        "result": [
            {"region": name, "value": value, "unit": units}
            for name, value in top
        ],
        "provenance": {
            "file_sha256": _file_sha256(file),
            "mask_sha256": mask_info["mask_sha256"],
            "var": var,
            "agg": agg,
            "n_requested": n,
            "year": year,
            "year_defaulted": year_defaulted,
            "select": select,
        },
        "coverage": coverage,
    }


def time_series(
    file: str,
    var: str,
    region: str,
    agg_per_step: str,
    mask_file: str,
    select: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate `var` with `agg_per_step` over one region at each timestep.

    Returns `summary`, `units`, `by_year`, `points` and `provenance`. Each point is `{t, value}` plus `year` when the time axis decodes, and `select_points` windows the points on that year. The `model_time_series` tool drops `points` unless a year window is requested. Raises `AnalyticsError` when the variable has fewer than three dimensions.

    `select` is {dimension: coordinate_value} for an extra non-time, non-spatial axis. An unselected one is averaged into each point and reported in `unselected_dims`.
    """
    mask_info = _load_mask_checked(file, mask_file)
    r_idx = _region_index(mask_info, region)

    ds = _open(file)
    with ds:
        if var not in ds.data_vars:
            raise AnalyticsError(f"variable '{var}' not found")
        var_da = ds[var]
        if var_da.ndim < 3:
            raise AnalyticsError(
                f"variable '{var}' must have at least 3 dims (time, lat, lon); "
                f"got {var_da.ndim}"
            )
        tdim = _time_dim_of(var_da)
        units = _units_of(var_da)
        var_da, extra_dims = _apply_select(ds, var_da, select, tdim)
        n_steps = var_da.sizes[tdim]
        time_values = ds["time"].values if "time" in ds else None
        years = _decode_years(ds)
        # If the time coordinate and the variable differ in length, the years
        # would not line up with the points, so the year rollup is skipped.
        if years is not None and len(years) != n_steps:
            _log.warning(
                "time axis length %d != variable steps %d; skipping year rollup",
                len(years), n_steps,
            )
            years = None

        region_mask = mask_info["mask"][r_idx]
        points: list[dict[str, Any]] = []
        # Load one timestep at a time instead of the whole variable, so a
        # global multi-decade file stays within memory.
        for t in range(n_steps):
            slab = var_da.isel({tdim: t}).values
            weighted = np.where(region_mask > 0, slab, np.nan)
            val = _apply_agg(weighted, agg_per_step)
            ts = float(time_values[t]) if time_values is not None else float(t)
            point: dict[str, Any] = {"t": ts, "value": val}
            # `t` is the raw axis value, so `year` is added when the axis
            # decodes. `select_points` relies on it, and it is absent when the
            # axis did not decode.
            if years is not None:
                point["year"] = int(years[t])
            points.append(point)

    by_year = (
        _rollup_by_year(
            np.array([p["value"] for p in points], dtype=float),
            years,
            agg_per_step,
        )
        if years is not None
        else []
    )

    out: dict[str, Any] = {
        "summary": _series_summary(points, years=years, units=units),
        "units": units,
        "by_year": by_year,
        "points": points,
        "provenance": {
            "file_sha256": _file_sha256(file),
            "mask_sha256": mask_info["mask_sha256"],
            "var": var,
            "region": region,
            "agg_per_step": agg_per_step,
            "select": select,
        },
    }
    if extra_dims:
        out["unselected_dims"] = extra_dims
        out["unselected_dims_note"] = _unselected_note(extra_dims, ds)
    return out


def select_points(
    result: dict[str, Any], years: tuple[int, int]
) -> dict[str, Any]:
    """Pick the per-step points of a time_series result inside a year window.

    `years` is (first_year, last_year), both inclusive. Returns
    {points, window:{years, n_points}}.

    Raises `AnalyticsError` on an axis that did not decode to calendar years, on a window the series does not reach, on an inverted window and on a window holding more than `MAX_SERIES_POINTS` steps.
    """
    lo, hi = int(years[0]), int(years[1])
    if hi < lo:
        raise AnalyticsError(
            f"year window [{lo}, {hi}] ends before it starts; give "
            f"(first_year, last_year) with both inclusive")

    points = list(result.get("points") or [])
    if not points:
        raise AnalyticsError("the series has no timesteps to window")
    if any("year" not in p for p in points):
        raise AnalyticsError(
            "this file's time axis did not decode to calendar years, so a "
            "year window cannot be applied; use by_year or summary instead")

    selected = [dict(p) for p in points if lo <= p["year"] <= hi]
    if not selected:
        covered = [p["year"] for p in points]
        raise AnalyticsError(
            f"no timestep falls in [{lo}, {hi}]; this series covers "
            f"{min(covered)} to {max(covered)}")
    if len(selected) > MAX_SERIES_POINTS:
        raise AnalyticsError(
            f"[{lo}, {hi}] holds {len(selected)} timesteps, over the "
            f"{MAX_SERIES_POINTS} this returns at once; narrow the window, or "
            f"read the per-year rollup for the longer span")

    return {
        "points": selected,
        "window": {"years": [lo, hi], "n_points": len(selected)},
    }


def _check_period(
    label: str,
    period: tuple[int, int],
    n_steps: int,
    year_range: tuple[int | None, int | None] = (None, None),
) -> None:
    """Raise `AnalyticsError` when `period` is not a valid range of step indices on an axis of `n_steps` steps.

    A negative index, an inverted range and an end past the axis are refused. `.isel(slice(...))` clamps, so an end past the axis would average only the steps that exist while the provenance recorded the requested range. When every bound lies inside `year_range`, the message says that calendar years were probably passed.
    """
    lo, hi = period
    yr_lo, yr_hi = year_range
    if lo < 0 or hi < 0:
        raise AnalyticsError(
            f"{label} {list(period)} has a negative step index; steps are "
            f"0-based positions on the time axis, which has {n_steps} of them")
    if hi < lo:
        raise AnalyticsError(
            f"{label} {list(period)} ends before it starts; give "
            f"(start_step, end_step_inclusive)")
    if hi >= n_steps:
        _looks_like_years = (
            yr_lo is not None and all(yr_lo <= x <= yr_hi for x in period)
        )
        _hint = (
            " Those look like calendar years, not step indices."
            if _looks_like_years else ""
        )
        _range = (
            f" This file covers years {yr_lo} to {yr_hi}."
            if yr_lo is not None else ""
        )
        raise AnalyticsError(
            f"{label} {list(period)} runs past the end of the time axis, "
            f"which has {n_steps} steps (valid 0 to {n_steps - 1}).{_hint} "
            f"These are step indices, not years.{_range}")


def compare_periods(
    file: str,
    var: str,
    period_a: tuple[int, int],
    period_b: tuple[int, int],
    mask_file: str,
    op: str = "diff",
    select: dict[str, Any] | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Compare two time-index ranges per region.

    period_a and period_b are (start_step, end_step_inclusive) indices.
    op is 'diff' (B - A) or 'ratio' (B / A).
    `select` picks one coordinate value on an extra non-time, non-spatial axis;
    an unselected one is reported in `unselected_dims`.
    `region` restricts the comparison to one region name; omitted, every region
    in the mask is compared.
    """
    if op not in ("diff", "ratio"):
        raise AnalyticsError(f"op must be 'diff' or 'ratio', got '{op}'")

    mask_info = _load_mask_checked(file, mask_file)

    # The mask drives the per-region loop, so slicing it here skips the other
    # regions before any aggregation. _region_index raises AnalyticsError naming
    # the valid names, so an unknown region never falls back to the full set.
    if region is not None:
        r_idx = _region_index(mask_info, region)
        region_names = [mask_info["region_names"][r_idx]]
        region_mask = mask_info["mask"][r_idx:r_idx + 1]
    else:
        region_names = mask_info["region_names"]
        region_mask = mask_info["mask"]

    a0, a1 = period_a
    b0, b1 = period_b

    ds = _open(file)
    with ds:
        if var not in ds.data_vars:
            raise AnalyticsError(f"variable '{var}' not found")
        var_da = ds[var]
        units = _units_of(var_da)
        tdim = _time_dim_of(var_da)
        var_da, extra_dims = _apply_select(ds, var_da, select, tdim)

        # The periods are step indices, not years. Slicing past the end of an
        # axis is not an error in numpy, so _check_period refuses it and gives
        # the real range.
        n_steps = int(var_da.sizes[tdim])
        _years = _decode_years(ds)
        _year_range = (
            (int(np.min(_years)), int(np.max(_years)))
            if _years is not None and len(_years) else (None, None)
        )
        _check_period("period_a", (a0, a1), n_steps, _year_range)
        _check_period("period_b", (b0, b1), n_steps, _year_range)
        # Load only the two period slabs, time-averaged, not the whole var.
        slab_a = var_da.isel({tdim: slice(a0, a1 + 1)}).mean(dim=tdim).values
        slab_b = var_da.isel({tdim: slice(b0, b1 + 1)}).mean(dim=tdim).values

    a_vals, _ = _aggregate_per_region(slab_a, region_mask, "mean")
    b_vals, _ = _aggregate_per_region(slab_b, region_mask, "mean")

    out: list[dict[str, Any]] = []
    for name, a, b in zip(region_names, a_vals, b_vals):
        if op == "diff":
            v = b - a
        else:  # ratio
            v = (b / a) if a != 0 else float("nan")
        out.append({"region": name, "value": v, "a": a, "b": b})

    result: dict[str, Any] = {
        "result": out,
        "units": units,
        "provenance": {
            "file_sha256": _file_sha256(file),
            "mask_sha256": mask_info["mask_sha256"],
            "var": var,
            "period_a": list(period_a),
            "period_b": list(period_b),
            # The pair of integers is a range of step indices, not years.
            "period_unit": "step",
            "op": op,
            "select": select,
            "region": region,
        },
    }
    if extra_dims:
        result["unselected_dims"] = extra_dims
        result["unselected_dims_note"] = _unselected_note(extra_dims, ds)
    return result


def _mann_kendall_trend(values: np.ndarray) -> dict[str, Any]:
    """Calculate the Mann-Kendall trend test and the linear slope of `values`, ignoring NaN.

    Non-finite values are dropped first, and the slope regresses on the original index positions, so a missing step still counts as elapsed time. Returns `{slope, p_value, z, direction, n, alpha, n_tie_groups, lag1_autocorrelation, n_effective, independence_assumed, method}`, where `n` counts the valid points used. `direction` is `no_trend` unless `p_value` is below `alpha` (0.1). Raises `AnalyticsError` for fewer than three valid points.
    """
    arr = np.asarray(values, dtype=float)
    positions = np.arange(arr.size, dtype=float)
    finite = np.isfinite(arr)
    v = arr[finite]
    x = positions[finite]
    n = int(v.size)
    if n < 3:
        raise AnalyticsError("need at least 3 valid (non-NaN) timesteps for trend")

    slope = float(np.polyfit(x, v, 1)[0])

    s_stat = 0
    for i in range(n):
        for j in range(i + 1, n):
            s_stat += int(np.sign(v[j] - v[i]))

    # Variance of the statistic with the tie correction.
    _, tie_counts = np.unique(v, return_counts=True)
    ties = tie_counts[tie_counts > 1]
    tie_term = float(np.sum(ties * (ties - 1) * (2 * ties + 5))) if ties.size else 0.0
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s_stat > 0:
        z = (s_stat - 1) / math.sqrt(var_s)
    elif s_stat < 0:
        z = (s_stat + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value. The form 2*(1 - 0.5*(1 + erf(|z|/sqrt(2)))) cancels and
    # underflows to exactly 0.0 for |z| above about 8.3. erfc gives the same
    # quantity without the cancellation and stays finite to around 1e-308.
    p_value = math.erfc(abs(z) / math.sqrt(2.0))

    # Lag-1 autocorrelation and the effective sample size it implies. Serial
    # correlation makes n a count of observations and not of independent ones,
    # so the p-value above is too small. Both numbers are reported and no
    # correction is applied.
    lag1 = None
    n_eff = None
    if n >= 3:
        d = v - v.mean()
        denom = float(np.dot(d, d))
        if denom > 0:
            lag1 = float(np.dot(d[:-1], d[1:]) / denom)
            if lag1 < 1.0:
                n_eff = float(n * (1.0 - lag1) / (1.0 + lag1))
                n_eff = max(1.0, min(float(n), n_eff))
            else:
                n_eff = 1.0

    alpha = 0.1
    if p_value < alpha:
        direction = "increasing" if slope > 0 else "decreasing"
    else:
        direction = "no_trend"

    return {
        "slope": slope,
        "p_value": p_value,
        "z": z,
        "direction": direction,
        "n": n,
        "alpha": alpha,
        "n_tie_groups": int(ties.size),
        "lag1_autocorrelation": lag1,
        "n_effective": n_eff,
        "independence_assumed": True,
        "method": "mann_kendall",
    }


def nc_trend(
    file: str,
    var: str,
    region: str,
    mask_file: str,
    select: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate a monotonic trend with Mann-Kendall + linear slope.

    PREFERRED for "is X increasing/decreasing", "what is the trend", or "how has
    X changed over time" questions. Returns a compact result: slope (linear
    regression on time index), Mann-Kendall p_value, a categorical direction
    (increasing/decreasing/no_trend), n (valid points), and n_missing (NaN gaps
    skipped). Answer directly from these fields.
    """
    ts = time_series(
        file=file, var=var, region=region,
        agg_per_step="mean", mask_file=mask_file, select=select,
    )
    all_values = [p["value"] for p in ts["points"]]
    n_total = len(all_values)
    result = _mann_kendall_trend(np.array(all_values, dtype=float))

    # Carry the step naming through, so the answer states whether the slope is
    # per month or per year.
    _summary = ts.get("summary", {})
    out: dict[str, Any] = {
        **result,
        "units": ts.get("units", "unknown"),
        "steps_per_year": _summary.get("steps_per_year"),
        "step_unit": _summary.get("step_unit", "step"),
        "slope_unit": _summary.get("slope_unit"),
        "slope_per_year_scaled": (
            float(result["slope"] * _summary["steps_per_year"])
            if _summary.get("steps_per_year") is not None else None
        ),
        "n_total": n_total,
        "n_missing": n_total - result["n"],
        "provenance": {
            **ts["provenance"],
            "method": "mann_kendall",
        },
    }
    for key in ("unselected_dims", "unselected_dims_note"):
        if key in ts:
            out[key] = ts[key]
    return out
