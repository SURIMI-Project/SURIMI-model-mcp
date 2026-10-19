"""Loads the external vocabularies that SURIMI data is coded against: FAO/ASFIS species codes, ISSCFG gear and fishery codes, WoRMS taxonomy, FishBase names and ISO 3166 countries.

These are external standards and not SURIMI output. The server loads each source on the first call that needs it, indexes it once and caches the index for the life of the process.

The five sources load and cache independently, so a failure in one leaves the other four usable. Two kinds of failure are handled differently.

- A `store.SimStoreError` comes from the fetch itself, for example a network error or an expired STS token. `_get` does not catch it, so it reaches the caller uncached and the next call retries.
- A `VocabError` comes from the file's own content, such as a missing expected column, no data rows or a read cut short at the byte cap. `_get` caches it, because the same bytes fail the same way.
"""
from __future__ import annotations

import csv
import io
import threading
from typing import Any, Callable

from surimi_model_mcp.simulations import store

_PREFIX = "ecopath"
ASFIS_KEY = f"{_PREFIX}/ASFIS_sp_2024.csv"
ISSCFG_KEY = f"{_PREFIX}/ISSCFG_2024.csv"
WORMS_KEY = f"{_PREFIX}/WoRMS_taxon.csv"
FISHBASE_KEY = f"{_PREFIX}/speciesoccursum_ver0816c.csv"
ISO3166_KEY = f"{_PREFIX}/ISO-3166-Countries-with-Regional-Codes.csv"

#: Read cap in bytes for the ASFIS, ISSCFG, WoRMS and FishBase files. `_rows`
#: refuses a file that `store.get_text` reports as truncated at this cap.
VOCAB_MAX_BYTES = 4 * 1024 * 1024

#: Read cap in bytes for the ISO 3166 file, set here instead of relying on the
#: default of `store.get_text`.
ISO3166_MAX_BYTES = 131072


class VocabError(Exception):
    """A vocabulary file's own content is unusable. `_get` caches this error, unlike `store.SimStoreError`."""


def _rows(key: str, max_bytes: int, label: str) -> list[dict[str, str]]:
    """Fetch one CSV and return its rows as dicts keyed by header.

    The function strips whitespace and a UTF-8 byte order mark from every header, because `store.get_text` decodes as plain `utf-8`, which keeps a byte order mark. It raises `VocabError` for a truncated, empty or header-only file.
    """
    text, truncated = store.get_text(key, max_bytes=max_bytes)   # store.SimStoreError propagates
    if truncated:
        raise VocabError(
            f"{label} ({key}) exceeds the configured read cap of "
            f"{max_bytes} bytes and was truncated mid-file; refusing to "
            "build an index from a partial read that could silently drop "
            "or corrupt rows"
        )
    reader = csv.reader(io.StringIO(text))
    try:
        header = [h.strip().lstrip("﻿") for h in next(reader)]
    except StopIteration:
        raise VocabError(f"{label} ({key}) is empty") from None
    rows = [dict(zip(header, row)) for row in reader if row]
    if not rows:
        raise VocabError(f"{label} ({key}) has no data rows")
    return rows


def _require_columns(rows: list[dict[str, str]], required: set[str], label: str, key: str) -> None:
    have = set(rows[0])
    missing = required - have
    if missing:
        raise VocabError(
            f"{label} ({key}) is missing expected column(s): {sorted(missing)} "
            f"- has {sorted(have)}"
        )


def _load_asfis() -> dict[str, Any]:
    rows = _rows(ASFIS_KEY, VOCAB_MAX_BYTES, "ASFIS species codes")
    _require_columns(rows, {"Alpha3_Code", "Scientific_Name"}, "ASFIS species codes", ASFIS_KEY)
    by_code: dict[str, dict[str, str]] = {}
    by_sci_name: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        code = r.get("Alpha3_Code", "").strip()
        if code:
            by_code[code] = r
        sci = r.get("Scientific_Name", "").strip()
        if sci:
            by_sci_name.setdefault(sci.lower(), []).append(r)
    return {"rows": rows, "by_code": by_code, "by_sci_name": by_sci_name}


def _load_isscfg() -> dict[str, Any]:
    rows = _rows(ISSCFG_KEY, VOCAB_MAX_BYTES, "ISSCFG gear/fishery codes")
    _require_columns(rows, {"FISHERY_CODE", "GEAR_CODE"}, "ISSCFG gear/fishery codes", ISSCFG_KEY)
    by_fishery: dict[str, dict[str, str]] = {}
    by_gear: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        fc = r.get("FISHERY_CODE", "").strip()
        if fc:
            by_fishery[fc] = r
        gc = r.get("GEAR_CODE", "").strip()
        if gc:
            by_gear.setdefault(gc, []).append(r)
    return {"rows": rows, "by_fishery_code": by_fishery, "by_gear_code": by_gear}


def _load_worms() -> dict[str, Any]:
    rows = _rows(WORMS_KEY, VOCAB_MAX_BYTES, "WoRMS taxonomy")
    _require_columns(rows, {"scientificName", "FAO_Code"}, "WoRMS taxonomy", WORMS_KEY)
    by_fao_code: dict[str, dict[str, str]] = {}
    for r in rows:
        code = r.get("FAO_Code", "").strip()
        if code:
            by_fao_code[code] = r
    return {"rows": rows, "by_fao_code": by_fao_code}


def _load_fishbase() -> dict[str, Any]:
    rows = _rows(FISHBASE_KEY, VOCAB_MAX_BYTES, "FishBase species occurrence summary")
    _require_columns(
        rows, {"Genus", "Species", "FBname"}, "FishBase species occurrence summary", FISHBASE_KEY,
    )
    by_binomial: dict[str, dict[str, str]] = {}
    for r in rows:
        genus, species = r.get("Genus", "").strip(), r.get("Species", "").strip()
        if genus and species:
            by_binomial[f"{genus} {species}".lower()] = r
    return {"rows": rows, "by_binomial": by_binomial}


def _load_iso3166() -> dict[str, Any]:
    rows = _rows(ISO3166_KEY, ISO3166_MAX_BYTES, "ISO 3166 countries")
    _require_columns(rows, {"alpha-3", "name"}, "ISO 3166 countries", ISO3166_KEY)
    by_code: dict[str, dict[str, str]] = {}
    for r in rows:
        code = r.get("alpha-3", "").strip()
        if code:
            by_code[code] = r
    return {"rows": rows, "by_code": by_code}


_LOADERS: dict[str, Callable[[], dict[str, Any]]] = {
    "asfis": _load_asfis,
    "isscfg": _load_isscfg,
    "worms": _load_worms,
    "fishbase": _load_fishbase,
    "iso3166": _load_iso3166,
}

#: Loaded index per source, kept for the life of the process. `_errors` holds
#: the `VocabError` message instead for a source whose file is unusable. A
#: source is in at most one of the two.
_cache: dict[str, dict[str, Any]] = {}
_errors: dict[str, str] = {}
_lock = threading.Lock()


def reset_cache() -> None:
    """Clear both caches so that tests can re-read a patched store."""
    with _lock:
        _cache.clear()
        _errors.clear()


def _get(name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the named source, or `(None, error)` when its file raised `VocabError`.

    A `store.SimStoreError` from the loader propagates uncached, so the caller decides how to degrade and the next call retries.
    """
    with _lock:
        if name in _cache:
            return _cache[name], None
        if name in _errors:
            return None, _errors[name]
    try:
        idx = _LOADERS[name]()
    except VocabError as exc:
        with _lock:
            _errors[name] = str(exc)
        return None, str(exc)
    with _lock:
        _cache[name] = idx
    return idx, None


def asfis() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the FAO ASFIS species index, or `(None, error)`; see `_get`."""
    return _get("asfis")


def isscfg() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the ISSCFG gear and fishery index, or `(None, error)`; see `_get`."""
    return _get("isscfg")


def worms() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the WoRMS taxonomy index, or `(None, error)`; see `_get`."""
    return _get("worms")


def fishbase() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the FishBase species index, or `(None, error)`; see `_get`."""
    return _get("fishbase")


def iso3166() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(index, None)` for the ISO 3166 country index, or `(None, error)`; see `_get`."""
    return _get("iso3166")
