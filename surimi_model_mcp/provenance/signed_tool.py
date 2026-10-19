"""Wraps a tool function so that each call returns its value together with an HMAC-signed receipt.

Usage:
    @with_receipt("model_top_regions")
    def top_regions(file, var, n, agg, mask_file):
        return {... "provenance": {...}}

    # call:
    out = top_regions(...)
    # out == {"value": {...}, "receipt": {...signed...}}

The wrapper reads the secret from `SURIMI_RECEIPT_SECRET` on each call.
"""
from __future__ import annotations

import functools
import inspect
import os
from typing import Any, Callable

from surimi_model_mcp.provenance.receipts import issue_receipt


class ToolError(RuntimeError):
    """Raised when the wrapper cannot sign a call, for example when `SURIMI_RECEIPT_SECRET` is not set."""


def _get_secret() -> bytes:
    raw = os.environ.get("SURIMI_RECEIPT_SECRET", "")
    if not raw:
        raise ToolError(
            "SURIMI_RECEIPT_SECRET env var not set; cannot sign receipts"
        )
    return raw.encode("utf-8")


def _bind_args(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Map the call's arguments to parameter names, with defaults filled in."""
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


#: Parameter names that carry a readable resource URI. `_read_freshness` looks
#: only at these parameters, because tool calls run on several threads and a
#: shared read log would mix up the reads of different calls.
_URI_PARAMS = ("file", "data_file", "mask_file")


def _read_freshness(params: dict[str, Any]) -> dict[str, Any] | None:
    """Return when the resources named in `params` were read and whether any is stale, or None if none has a read date.

    The S3 cache never re-checks the remote, so the receipt carries the read date through `ncio.read_freshness`. The keys are `read_as_of`, `read_stale` and `read_sources`, plus `read_note` when a source is stale. With several inputs, `read_as_of` is the oldest read and `read_stale` is True if any input is stale.
    """
    from surimi_model_mcp.netcdf import io as ncio

    uris = [
        params[name] for name in _URI_PARAMS
        if isinstance(params.get(name), str) and params[name]
    ]
    if not uris:
        return None

    per_uri = {u: ncio.read_freshness(u) for u in uris}
    dated = {u: f for u, f in per_uri.items() if f["as_of"] is not None}
    if not dated:
        return None

    oldest_uri = min(dated, key=lambda u: dated[u]["as_of"])
    oldest = dated[oldest_uri]
    stale = [u for u, f in dated.items() if f["stale"]]

    out: dict[str, Any] = {
        "read_as_of": oldest["as_of"],
        "read_stale": bool(stale),
        "read_sources": sorted(dated),
    }
    if stale:
        # One note per stale source, so a reader can see which input is old.
        out["read_note"] = "; ".join(dated[u]["note"] for u in sorted(stale))
    return out


def with_receipt(tool_id: str) -> Callable:
    """Return a decorator that makes each call return `{value, receipt}`, signed under `tool_id`.

    The receipt's provenance is the one the tool returns under `provenance`, with the read freshness added. A tool that returns no provenance gets none.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            secret = _get_secret()
            params = _bind_args(fn, args, kwargs)
            value = fn(*args, **kwargs)

            provenance: dict[str, Any] | None = None
            if isinstance(value, dict) and "provenance" in value:
                provenance = value["provenance"]
                freshness = _read_freshness(params)
                if freshness and isinstance(provenance, dict):
                    # Copy the provenance instead of writing into `value`, so
                    # the returned value stays as the tool computed it.
                    provenance = {**provenance, **freshness}

            receipt = issue_receipt(
                tool_id=tool_id,
                input_params=params,
                output_value=value,
                secret=secret,
                provenance=provenance,
            )
            return {"value": value, "receipt": receipt}

        return wrapper

    return deco
