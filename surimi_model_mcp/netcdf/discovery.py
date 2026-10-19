"""Lists the NetCDF files under a prefix and classifies each one.

The `model_list_files` tool uses it, so the LLM can find a data file from a question without the user giving an `s3://` path. The `uri` of a file then goes to `model_describe_file` and the analytics tools.
"""
from __future__ import annotations

from typing import Any

from surimi_model_mcp.netcdf import io as ncio


def classify(uri: str) -> str:
    """Return `data` for a `.nc` file, `mask` for a `.nc` file whose name contains "mask", and `other` for any other object."""
    name = uri.rsplit("/", 1)[-1].lower()
    if not name.endswith(".nc"):
        return "other"
    if "mask" in name:
        return "mask"
    return "data"


def list_netcdf_files(prefix: str) -> dict[str, Any]:
    """List the objects under `prefix` as `{prefix, count, files}`, with `uri`, `size` and `kind` for each file.

    When S3 cannot be listed and a listing was saved earlier, the reply is that saved listing with `stale` True, its `as_of` and an entry in `notes`. Without a saved listing the `NetCDFIOError` is raised.
    """
    try:
        objs = ncio.list_objects(prefix)
    except ncio.NetCDFIOError as exc:
        cached = ncio.load_listing(prefix)
        if cached is None:
            raise
        note = (
            f"S3 is unreachable ({'expired or invalid credentials' if ncio.is_auth_error(str(exc)) else str(exc)}); "
            f"serving the listing captured at {cached['as_of']}"
        )
        files = [
            {"uri": f["uri"], "size": f.get("size"), "kind": classify(f["uri"])}
            for f in cached.get("files") or []
        ]
        return {"prefix": prefix, "count": len(files), "files": files,
                "stale": True, "as_of": cached["as_of"], "notes": [note]}
    files = [
        {"uri": o["uri"], "size": o["size"], "kind": classify(o["uri"])}
        for o in objs
    ]
    return {"prefix": prefix, "count": len(files), "files": files}

def merge_files(
    listing: dict[str, Any], extra: list[dict[str, Any]], note: str | None = None
) -> dict[str, Any]:
    """Return a copy of `listing` with the `extra` entries added, skipping any `uri` already present, and `count` updated.

    A `note` is appended to `notes`, so a source that added nothing can say why and an empty result does not read as an empty bucket.
    """
    out = dict(listing)
    notes = list(listing.get("notes") or [])
    if note:
        notes.append(note)
    seen = {f["uri"] for f in out.get("files") or []}
    merged = list(out.get("files") or [])
    for f in extra:
        if f["uri"] in seen:
            continue
        seen.add(f["uri"])
        merged.append(f)
    out["files"] = merged
    out["count"] = len(merged)
    if notes:
        out["notes"] = notes
    return out


def list_netcdf_files_with_fallback(primary: str, legacy: str) -> dict[str, Any]:
    """List `primary`, and fall back to `legacy` when `primary` is empty or cannot be listed.

    `primary` is the snake_case prefix (`netcdf/`) and `legacy` the PascalCase prefix (`NetCDF/`) that existing deployments still hold objects under. A result taken from `legacy` carries a `note` naming `primary`. The error from `legacy` is raised only when `primary` also failed.
    """
    primary_result: dict[str, Any] | None
    try:
        primary_result = list_netcdf_files(primary)
    except ncio.NetCDFIOError:
        primary_result = None
    if primary_result is not None and primary_result["count"] > 0:
        return primary_result

    try:
        legacy_result = list_netcdf_files(legacy)
    except ncio.NetCDFIOError:
        if primary_result is not None:
            return primary_result
        raise
    if legacy_result["count"] > 0:
        return {
            **legacy_result,
            "note": f"served from legacy prefix; objects should migrate to {primary}",
        }
    return primary_result if primary_result is not None else legacy_result
