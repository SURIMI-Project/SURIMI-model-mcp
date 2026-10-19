"""The gridded tools: list and inspect NetCDF files, calculate regional numbers, and check a calculation record.

The listing and inspection tools return no signature. The five analytical tools are wrapped
with `with_receipt`, so each number they return comes with a calculation record, called a
receipt in the code, signed with HMAC. `model_verify` checks that signature. `register_all` in
`tools/__init__.py` registers every name in `TOOLS` with `mcp.tool()`.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from surimi_model_mcp.netcdf import analytics, discovery, inspect as nc_inspect, io as ncio, regions as nc_regions
from surimi_model_mcp.provenance.receipts import verify_receipt
from surimi_model_mcp.provenance.signed_tool import with_receipt
from surimi_model_mcp.paths import APP_ROOT


# Catalog prefix that `model_list_files` lists. `SURIMI_NETCDF_PREFIX` replaces it.
# The SURIMI naming convention uses lower_snake_case bucket directories, so the
# canonical prefix is `netcdf/`. Without the environment variable, discovery falls
# back to `LEGACY_NETCDF_PREFIX` (with a note) when the canonical prefix is empty or cannot be listed.
_ENV_PREFIX = os.environ.get("SURIMI_NETCDF_PREFIX")
DEFAULT_NETCDF_PREFIX = _ENV_PREFIX or "s3://project-surimi/netcdf/"
LEGACY_NETCDF_PREFIX = "s3://project-surimi/NetCDF/"

# Directory of the two sample files bundled with this repository: the western
# Mediterranean sample `westmed-2013-2017-biomass-catch-sales.nc` and a region mask
# on its own grid, `westmed_full_domain_mask.nc`. `SURIMI_NETCDF_REFERENCE_DIR`
# replaces it. `model_list_files` lists both, and the mask lets the regional
# tools run on the sample because the repository holds no GFCM sub-area mask.
_REFERENCE_SAMPLES_DIR = os.environ.get(
    "SURIMI_NETCDF_REFERENCE_DIR", str(APP_ROOT / "data" / "netcdf-samples"))


# ---------- Discovery (unsigned) ----------

def _scan_runs() -> dict[str, Any]:
    """The simulation layer's live inventory of runs.

    Imported inside the function: this module is imported by tests that have no
    S3 stack, and the simulation store reaches for one at import time.
    """
    from surimi_model_mcp.simulations import discovery as sim_discovery

    return sim_discovery.scan()


def _run_netcdf_files() -> tuple[list[dict[str, Any]], str | None]:
    """NetCDF written by SURIMI runs, taken from the simulation scan.

    A second PREFIX is a guess about where output will land, and the guess was
    wrong: the first real run (2026-09-15) went to output_creator/<uuid>/ while
    EXPERIMENTS_PREFIX pointed at the aggregator's experiments/, so a 74 MB
    result stayed invisible to the catalog. The simulation layer had already
    found it, because it lists the whole bucket and groups on any UUID-shaped
    directory segment, so it needs no per-model configuration. Reuse that
    answer rather than adding prefixes one incident at a time.

    Costs no extra listing: scan() is cached for 300s and the tool layer calls
    it anyway. Returns (files, note) where note is set only on failure.
    """
    from surimi_model_mcp.simulations import store

    try:
        payload = _scan_runs()
    except Exception as exc:                        # scan raises on S3 outage
        return [], (f"the simulation scan failed ({exc}); run output is not "
                    f"included in this listing")
    files: list[dict[str, Any]] = []
    for run in payload.get("simulations") or []:
        sizes = run.get("sizes") or {}
        for key in run.get("keys") or []:
            if not key.lower().endswith(".nc"):
                continue
            uri = key if "://" in key or key.startswith("/") else \
                f"s3://{store.BUCKET}/{key}"
            files.append({
                "uri": uri,
                "size": int(sizes.get(key) or 0),
                "kind": "data",
                # The simulation tools take this id and model, so the caller can
                # go from the file to the run that wrote it.
                "simulation_id": run.get("simulation_id"),
                "model": run.get("model"),
            })
    return files, None


#: Origin note for the samples bundled in this repository (currently the western
#: Mediterranean sample). Only files found there get it. A same-named file in a
#: directory named by `SURIMI_LOCAL_NETCDF_SAMPLES` (see `_local_sample_files`)
#: is not checked against the bundled sample, so `mark_origin` labels it
#: "external-or-reference".
_SAMPLE_ORIGIN_NOTE = (
    "A real SURIMI-format sample bundled with this server rather than held "
    "in s3://project-surimi. It carries no simulation_id because it is not "
    "a tracked run, not because it is external data."
)


def _reference_sample_files() -> tuple[list[dict[str, Any]], str | None]:
    """Local reference/sample netcdf files bundled with THIS repo.

    Degrades to an empty list (no note) when the directory does not exist --
    it is optional infrastructure, exactly like the S3 catalog and the
    simulation scan already degrade elsewhere in this function, so one
    missing source must never sink the whole listing. A directory that
    exists but fails to list (permissions, corrupt entry) reports why rather
    than raising, matching `_run_netcdf_files`'s failure shape.

    Every 'data' entry found here carries a preset `origin`/`origin_note`
    (see _SAMPLE_ORIGIN_NOTE): mark_origin leaves a preset origin alone, so
    these read "surimi-sample" instead of falling into the simulation_id-or-
    external default that _local_sample_files's entries below deliberately
    still fall into.
    """
    if not os.path.isdir(_REFERENCE_SAMPLES_DIR):
        return [], None
    try:
        listing = discovery.list_netcdf_files(_REFERENCE_SAMPLES_DIR)
    except ncio.NetCDFIOError as exc:
        return [], (f"reference sample directory scan failed ({exc}); not "
                    f"included in this listing")
    files = []
    for entry in listing["files"]:
        item = dict(entry)
        if item.get("kind") == "data":
            item["origin"] = "surimi-sample"
            item["origin_note"] = _SAMPLE_ORIGIN_NOTE
        files.append(item)
    return files, None


# Fields copied for each variable in the listing. Three fields are enough for the
# LLM to choose a file by its content, and a full `model_describe_file` per entry
# would make the reply too long.
_VAR_FIELDS = ("name", "units", "long_name")


# Note attached to every data file that is not bundled and carries no
# `simulation_id`. Only files written by a SURIMI run carry that id, so the note
# keeps external data such as the EcoOcean/FishMIP file from being presented as a
# SURIMI result.
_EXTERNAL_NOTE = (
    "NOT SURIMI model output. This is external or reference data served for "
    "comparison. An answer built on it must say which file it came from and "
    "must not be presented as a SURIMI result."
)


def mark_origin(listing: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `listing` with every data file's origin named.

    A source that already knows a file's origin (currently only
    _reference_sample_files, for the bundled-in-this-repo samples) sets
    `origin` itself; that preset is left alone here rather than overwritten
    by the simulation_id-or-external default below.
    """
    files_out = []
    for entry in listing.get("files") or []:
        item = dict(entry)
        if item.get("kind") == "data" and "origin" not in item:
            if item.get("simulation_id"):
                item["origin"] = "surimi-run"
            else:
                item["origin"] = "external-or-reference"
                item["origin_note"] = _EXTERNAL_NOTE
        files_out.append(item)
    out = dict(listing)
    out["files"] = files_out
    return out


def describe_listing_contents(listing: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `listing` with each data file's variables named.

    Masks are left alone: they hold regions, not measured variables. A file
    that cannot be opened carries `variables_error` and does not sink the
    catalogue, because one truncated object must not hide every other file.
    """
    files_out = []
    for entry in listing.get("files") or []:
        if entry.get("kind") != "data":
            files_out.append(dict(entry))
            continue
        item = dict(entry)
        # The S3 cache downloads the whole object on first open, so opening an
        # uncached file here would turn a listing into a large download. Only
        # files already in the cache get their variables, and the rest get a note.
        try:
            cached = bool(ncio.read_freshness(entry["uri"]).get("cached"))
        except Exception:
            cached = False
        if not cached:
            item["variables_note"] = (
                "not cached on this server yet, so its variables are not listed "
                "here; call model_list_variables on it to fetch and read it")
            files_out.append(item)
            continue
        try:
            item["variables"] = [
                {k: v.get(k) for k in _VAR_FIELDS}
                for v in nc_inspect.list_variables(entry["uri"])
            ]
        except Exception as exc:
            item["variables_error"] = f"could not read variables ({exc})"
        files_out.append(item)
    out = dict(listing)
    out["files"] = files_out
    return out


#: Environment variable naming a local directory of NetCDF files to merge into the
#: catalog, for example a mounted copy of this repository's `data/netcdf-samples/`.
#: When it is unset, `_local_sample_files` adds nothing.
_ENV_LOCAL_SAMPLES = "SURIMI_LOCAL_NETCDF_SAMPLES"


def _local_sample_files() -> tuple[list[dict[str, Any]], str | None]:
    """Extra netcdf files from a local sample directory, merged the same way
    simulation-run output is.

    Closes the Q-E5-4 gap: a 53 MB westmed sample
    (westmed-2013-2017-biomass-catch-sales.nc, 94 species, 9 fleets,
    per-species biomass/catch/discards/price/sales, 2013-01 to 2017-01) sits
    in data/netcdf-samples/ of this repository (it used to sit only in the data
    server's repository), and no catalog on either server referenced it. The species-level `select`
    mechanism it needs already works generically (top_regions/time_series/
    compare_periods all accept select={dim: value}); the only real gap was
    that model_list_files had no path from an s3:// prefix or a
    simulation-run scan to a file that lives on local disk. A per-file
    listing failure (bad path, not a directory) degrades to an empty list
    with a note, exactly like a failed simulation scan: one bad mount must
    not take the whole catalog down.
    """
    local_dir = os.environ.get(_ENV_LOCAL_SAMPLES)
    if not local_dir:
        return [], None
    try:
        objs = ncio.list_objects(local_dir)
    except ncio.NetCDFIOError as exc:
        return [], f"local sample directory {local_dir} could not be listed ({exc})"
    files = [
        {"uri": o["uri"], "size": o.get("size"), "kind": discovery.classify(o["uri"])}
        for o in objs
        if o["uri"].lower().endswith(".nc")
    ]
    return files, None


def model_list_files(prefix: str | None = None) -> dict[str, Any]:
    """List the netcdf catalog so you can find a file from a plain question.

    Returns {prefix, count, files:[{uri, size, kind}]} where kind is
    'data', 'mask', or 'other', plus `notes` when a source could not be
    listed and `stale` and `as_of` when a saved listing was served. Call this FIRST when asked about ocean/
    biomass/gridded data, then model_describe_file on a chosen 'data' file to
    see its variables, then an analytical tool (e.g. model_top_regions) with a
    matching 'mask' file. Defaults to the SURIMI netcdf bucket prefix.

    A file a SURIMI run wrote also carries `simulation_id` and `model`, so you
    can go from the file to model_describe_simulation without guessing.

    A 'data' entry that is already cached on this server also carries
    `variables`, each with its name, units and long_name, so you can pick the
    file that holds what was asked for without opening them one by one. An
    entry that is not cached yet carries `variables_note` instead, which says
    to call model_list_variables on it, and an entry that could not be read
    carries `variables_error`. Judge a file by its variables, not by its
    filename: a file named for an external model can still hold the variable
    the question needs.

    Each 'data' entry also carries `origin`. `surimi-run` is output a SURIMI run
    wrote. `external-or-reference` is data served for comparison and is NOT a
    SURIMI result: an answer built on it must name the file it came from and
    must not present it as what SURIMI found. `surimi-sample` is real SURIMI
    output bundled with this server rather than held in S3 -- currently just
    the western-Med sample -- readable through this server exactly like any
    other 'data' entry.

    The default listing (no `prefix`) also carries every file bundled under
    this server's own data/netcdf-samples/ directory, regardless of what the
    S3 prefix scan finds. When SURIMI_LOCAL_NETCDF_SAMPLES additionally names
    an operator-supplied local directory, its .nc files are merged in too --
    origin `external-or-reference`, not `surimi-sample`: unlike the bundled
    directory, a file's presence there is not verified to be the genuine
    sample just because it shares a name with one.
    """
    if prefix:
        # With an explicit prefix, only that location is listed.
        return mark_origin(describe_listing_contents(
            discovery.list_netcdf_files(prefix)))
    if _ENV_PREFIX:
        catalog = discovery.list_netcdf_files(_ENV_PREFIX)
    else:
        catalog = discovery.list_netcdf_files_with_fallback(
            DEFAULT_NETCDF_PREFIX, LEGACY_NETCDF_PREFIX
        )
    run_files, note = _run_netcdf_files()
    merged = discovery.merge_files(catalog, run_files, note)
    ref_files, ref_note = _reference_sample_files()
    merged = discovery.merge_files(merged, ref_files, ref_note)
    local_files, local_note = _local_sample_files()
    merged = discovery.merge_files(merged, local_files, local_note)
    return mark_origin(describe_listing_contents(merged))


# ---------- Inspection (unsigned) ----------

def model_list_regions(mask_file: str) -> dict[str, Any]:
    """List the named regions in a mask file so you can pick a region by name.

    Returns {mask_file, n_regions, regions:[{name, n_cells}]}. Call this BEFORE
    model_time_series or model_trend (which take a region NAME), and to know
    which regions model_top_regions will rank. Do NOT guess region names or call
    model_top_regions on the mask itself to discover them.
    """
    mask = nc_regions.load_region_mask(mask_file)
    regions = [
        {"name": name, "n_cells": int(np.sum(mask["mask"][i] > 0))}
        for i, name in enumerate(mask["region_names"])
    ]
    return {
        "mask_file": mask_file,
        "n_regions": mask["n_regions"],
        "regions": regions,
    }


# The six inspection tools below take the file as `path` or as its alias `file`,
# the keyword most analytical tools use, so the LLM can pass the same keyword to both.
def _resolve_path(path: str | None, file: str | None) -> str:
    """Take the one file argument, under either spelling."""
    if path is not None and file is not None and path != file:
        raise ValueError(
            f"got both path={path!r} and file={file!r}; pass one (they are aliases)")
    chosen = path if path is not None else file
    if chosen is None:
        raise ValueError("a netcdf path is required: pass path= (or its alias file=)")
    return chosen


def model_describe_file(path: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Return the full structure of a netcdf file: dimensions, variables, global attributes and the file's sha256.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.describe_file(_resolve_path(path, file))


def model_list_variables(path: str | None = None, file: str | None = None) -> list[dict[str, Any]]:
    """List data variables in a netcdf file with shapes, dtypes, units.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.list_variables(_resolve_path(path, file))


def model_get_time_range(path: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Return start/end of the time axis as raw values with their units, and decoded to calendar dates (`start_iso`, `end_iso`, `start_year`, `end_year`) when the axis has usable CF units.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.get_time_range(_resolve_path(path, file))


def model_get_spatial_bounds(path: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Return lat/lon extents and grid sizes.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.get_spatial_bounds(_resolve_path(path, file))


def model_check_cf_compliance(path: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Check CF-conventions compliance with a heuristic: a Conventions attribute that starts with CF, and latitude and longitude coordinates. A data variable without units gives a warning only.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.check_cf_compliance(_resolve_path(path, file))


def model_get_coverage_summary(path: str | None = None, file: str | None = None) -> dict[str, Any]:
    """Per-variable NaN counts and fraction. A very large variable is estimated from sampled steps and carries `sampled` true, with no exact count.

    `file` is accepted as an alias for `path`.
    """
    return nc_inspect.get_coverage_summary(_resolve_path(path, file))


# ---------- Analytical (signed) ----------

def _nc_time_series_cube(
    file: str,
    var: str,
    region: str,
    agg_per_step: str,
    mask_file: str,
    select: dict[str, Any] | None = None,
    points_for_years: list[int] | None = None,
) -> dict[str, Any]:
    """Time series of `var` within one region, returned as a compact cube slice.

    Returns a `summary` (n_points, n_valid, value_start/end/min/max/mean,
    slope_per_step, direction) and `by_year` (one pivotable row per calendar
    year: {year, value, n_steps}) plus provenance. Present the summary and the
    by_year rows (a small table or chart); cite the single calculation record
    (receipt) for the whole block. By default the raw per-step series is NOT returned -- a global file
    has hundreds of monthly steps and enumerating them floods the reply. For
    "is it increasing/decreasing" or "what is the trend" use model_trend (it
    adds a significance test and reads the full series internally).

    `points_for_years` is [first_year, last_year], both inclusive, and adds
    `points` (one {t, value, year} per timestep in that window) plus
    `points_window`. Ask for it ONLY when the question needs resolution finer
    than a year -- a seasonal cycle, which month a peak fell in, a spike inside
    one year -- because by_year has already averaged that away. Keep the window
    to the years actually asked about: a window over 120 timesteps is refused,
    not trimmed. The summary and by_year still cover the WHOLE series, so do
    not read a windowed `points` as the full record.

    `select` picks one coordinate on an extra dimension, e.g.
    {"species": "herring"}. A value found at several positions is refused,
    and the error names the positions, so pass the position number instead.
    Omitted on a variable that has such a dimension, the value AVERAGES
    across it and the reply names the dimension in `unselected_dims`.
    """
    window: tuple[int, int] | None = None
    if points_for_years is not None:
        # Checked before the read, because the series loads one slab per timestep.
        if len(points_for_years) != 2:
            raise ValueError(
                f"points_for_years takes exactly two years, "
                f"[first_year, last_year]; got {points_for_years}")
        window = (int(points_for_years[0]), int(points_for_years[1]))

    result = analytics.time_series(
        file=file, var=var, region=region,
        agg_per_step=agg_per_step, mask_file=mask_file, select=select,
    )
    cube = {k: v for k, v in result.items() if k != "points"}
    if window is None:
        return cube

    selection = analytics.select_points(result, window)
    # The window decides which steps are returned, so it goes into the signed
    # provenance. Without it, the receipt of a windowed series would look the same
    # as the receipt of the whole series.
    provenance = {**cube["provenance"], "points_for_years": list(window)}
    return {
        **cube,
        "provenance": provenance,
        "points": selection["points"],
        "points_window": selection["window"],
    }


def _nc_validate_alignment(data_file: str, mask_file: str) -> dict[str, Any]:
    """Check that a region mask sits on the same lat/lon grid as the data file.

    Returns {aligned: bool, mismatches: [...]} naming each axis that differs
    ('lat'/'lon' for a differing axis, 'lat_missing'/'lon_missing' when a file
    carries no such coordinate). An axis matches when its shape is equal and its
    values agree within 1e-3. model_top_regions, model_time_series,
    model_compare_periods and model_trend apply the mask cell by cell without
    re-projecting it, so each runs this check first and refuses a mask on a
    different grid. Call this to see which axis differs, and whenever a data
    file and a mask come from different sources. If aligned is False, report
    the mismatch and give no regional numbers for that pair.
    """
    return nc_regions.validate_alignment(data_file, mask_file)


model_validate_alignment = with_receipt("model_validate_alignment")(
    _nc_validate_alignment
)
model_top_regions = with_receipt("model_top_regions")(analytics.top_regions)
model_time_series = with_receipt("model_time_series")(_nc_time_series_cube)
model_compare_periods = with_receipt("model_compare_periods")(analytics.compare_periods)
model_trend = with_receipt("model_trend")(analytics.nc_trend)


# ---------- Verify ----------

#: `model_verify` has one verification method and reports it on every reply,
#: because no tool on this server has a recompute path.
_VERIFICATION_METHOD = "signature_only"
_RECOMPUTE_NOTE = (
    "Verified by signature only: no recompute path exists for this calculation, "
    "so the values were not re-derived. The signature shows the calculation "
    "record (receipt) is unaltered since it was issued."
)


def model_verify(receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify a signed calculation record (the `receipt`) against the server's signing key.

    Returns {verified, reason, verification_method, recompute_supported,
    recompute_note}. `verification_method` is always "signature_only" here:
    this server has no recompute path for any tool, so a passing check means
    the record is unaltered since issue, NOT that the number was recomputed.
    That is a weaker claim and the two must not be confused.

    A signature mismatch returns verified false and does not raise, and so does
    a server with no signing key (reason "secret_not_configured"). Records
    issued before the model/data rename carry legacy nc_* tool_ids; those still
    verify unchanged, because the signature covers the stored payload, not the
    current tool name.
    """
    # The method states what this server can do, so it is also returned when the
    # check fails or the secret is missing.
    envelope = {
        "verification_method": _VERIFICATION_METHOD,
        "recompute_supported": False,
        "recompute_note": _RECOMPUTE_NOTE,
    }
    raw = os.environ.get("SURIMI_RECEIPT_SECRET", "")
    if not raw:
        return {"verified": False, "reason": "secret_not_configured", **envelope}
    return {**verify_receipt(receipt, secret=raw.encode("utf-8")), **envelope}


# ---------- Registry ----------

TOOLS = [
    "model_list_files",
    "model_list_regions",
    "model_describe_file",
    "model_list_variables",
    "model_get_time_range",
    "model_get_spatial_bounds",
    "model_check_cf_compliance",
    "model_get_coverage_summary",
    "model_validate_alignment",
    "model_top_regions",
    "model_time_series",
    "model_compare_periods",
    "model_trend",
    "model_verify",
]

# Every tool above opens or lists NetCDF through the C libraries, so each runs
# under `NC_LOCK` from `netcdf/io.py`, which allows one NetCDF operation at a time.
for _name in TOOLS:
    globals()[_name] = ncio.serialized(globals()[_name])
