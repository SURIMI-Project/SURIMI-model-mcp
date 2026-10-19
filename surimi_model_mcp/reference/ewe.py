"""Reads the structure of an EwE (Ecopath with Ecosim) model from its `.eiixml` export, which is model configuration and not a simulation result.

The server scans the `<Table Name="..." Columns="..." Num="...">` header tags with a regular expression, so it reports every table's declared row and column count without parsing the table rows. The one exception is `EcopathGroup`, whose `<Row>` elements the server reads to return the functional-group names.

The `Num` value in a header can differ from the number of `<Row>` elements in the file. `declared_row_count` is the header value and `names_found` is the number of names actually read, and `_parse_model` adds a note when the two differ.

`list_models` finds the models by listing `ecopath/` for `.eiixml` files, so a new model appears without a code change. Parsed models and errors are cached for the life of the process.
"""
from __future__ import annotations

import csv
import io
import re
import threading
from typing import Any
from xml.sax.saxutils import unescape as _xml_unescape

from surimi_model_mcp.simulations import store

# Models sit at ecopath/<case study>/<name>.eiixml, and `store.find` walks the subfolders.
_PREFIX = "ecopath"

#: Read cap in bytes for one model file. `_parse_model` reports a file cut at
#: this cap as truncated.
EIIXML_MAX_BYTES = 8 * 1024 * 1024

#: Most group names read from one EcopathGroup table. `_parse_model` adds
#: `names_note` when the server stops at this number.
_MAX_GROUP_NAMES = 5000

_ROOT_RE = re.compile(r'<EwEModel Name="([^"]*)" DBVersion="([^"]*)" EwEVersion="([^"]*)"')
_TABLE_RE = re.compile(r'<Table Name="([^"]*)" Columns="([^"]*)" Num="(\d+)"')
_ROW_RE = re.compile(r"<Row>(.*?)</Row>", re.S)
_ENTITIES = {"&apos;": "'", "&quot;": '"'}


def _unescape(s: str) -> str:
    return _xml_unescape(s, _ENTITIES)


class EweError(Exception):
    """A model file has no recognisable `<EwEModel>` root tag. `describe_model` caches this error, as `vocab.VocabError` is cached."""


_list_cache: dict[str, list[dict[str, Any]]] = {}
_list_error: dict[str, str] = {}
_list_lock = threading.Lock()

_describe_cache: dict[str, dict[str, Any]] = {}
_describe_error: dict[str, str] = {}
_describe_lock = threading.Lock()


def reset_cache() -> None:
    """Clear every cache so that tests can re-read a patched store."""
    with _list_lock:
        _list_cache.clear()
        _list_error.clear()
    with _describe_lock:
        _describe_cache.clear()
        _describe_error.clear()


def list_models() -> tuple[list[dict[str, Any]] | None, str | None]:
    """Return `[{name, file, key, size_bytes}, ...]` for every `.eiixml` under `ecopath/`, sorted by name, or `(None, error)` when there are none.

    A `store.SimStoreError` from `store.find` propagates uncached.
    """
    with _list_lock:
        if "models" in _list_cache:
            return _list_cache["models"], None
        if "models" in _list_error:
            return None, _list_error["models"]
    objects, _failures = store.find(_PREFIX)   # store.SimStoreError propagates
    models = []
    for o in objects:
        key = o["key"]
        if not key.lower().endswith(".eiixml"):
            continue
        basename = key.rsplit("/", 1)[-1]
        models.append({
            "name": basename[: -len(".eiixml")],
            "file": basename, "key": key, "size_bytes": o["size"],
        })
    models.sort(key=lambda m: m["name"])
    if not models:
        err = f"no .eiixml EwE model files found under {_PREFIX}/"
        with _list_lock:
            _list_error["models"] = err
        return None, err
    with _list_lock:
        _list_cache["models"] = models
    return models, None


def _extract_group_names(
    content: str, tables: list[re.Match],
) -> tuple[list[str], int, bool]:
    """Return `(names, names_found, capped)` from the `<Row>` elements of the `EcopathGroup` table.

    The server reads no other table's rows. `capped` is True only when `_MAX_GROUP_NAMES` was reached.
    """
    idx = next((i for i, m in enumerate(tables) if m.group(1) == "EcopathGroup"), None)
    if idx is None:
        return [], 0, False
    columns = [c.split(":", 1)[0] for c in tables[idx].group(2).split(",") if c]
    if "GroupName" not in columns:
        return [], 0, False
    name_idx = columns.index("GroupName")
    seg_start = tables[idx].end()
    seg_end = tables[idx + 1].start() if idx + 1 < len(tables) else len(content)
    segment = content[seg_start:seg_end]

    names: list[str] = []
    capped = False
    for i, rm in enumerate(_ROW_RE.finditer(segment)):
        if i >= _MAX_GROUP_NAMES:
            capped = True
            break
        # `csv.reader` keeps a quoted field that contains a comma or newline
        # in one piece, so it cannot shift the GroupName column.
        fields = next(csv.reader(io.StringIO(_unescape(rm.group(1)))), [])
        if name_idx < len(fields):
            names.append(fields[name_idx])
    return names, len(names), capped


def _parse_model(content: str, truncated: bool) -> dict[str, Any]:
    root = _ROOT_RE.search(content)
    if not root:
        raise EweError("no <EwEModel> root tag found - not a recognised .eiixml export")

    tables = list(_TABLE_RE.finditer(content))
    table_list = [
        {
            "name": m.group(1),
            "row_count": int(m.group(3)),
            "column_count": len(m.group(2).split(",")) if m.group(2) else 0,
        }
        for m in tables
    ]
    names, names_found, capped = _extract_group_names(content, tables)
    group_table = next((t for t in table_list if t["name"] == "EcopathGroup"), None)
    declared = group_table["row_count"] if group_table else None

    functional_groups: dict[str, Any] = {
        "table": "EcopathGroup",
        "declared_row_count": declared,
        "names_found": names_found,
        "names": names,
    }
    if declared is not None and declared != names_found:
        functional_groups["note"] = (
            f"the file declares EcopathGroup Num={declared} but only "
            f"{names_found} <Row> element(s) with a GroupName are actually "
            "present in this export - report names_found/names as the "
            "usable functional-group count, not declared_row_count; see "
            "this module's docstring for why the two diverge"
        )
    if capped:
        functional_groups["names_note"] = (
            f"stopped after {_MAX_GROUP_NAMES} group row(s) to bound cost; "
            "more rows may exist in the file"
        )

    result: dict[str, Any] = {
        "model_name": root.group(1),
        "db_version": root.group(2),
        "ewe_version": root.group(3),
        "table_count": len(table_list),
        "tables": sorted(table_list, key=lambda t: t["name"]),
        "functional_groups": functional_groups,
        "truncated": truncated,
    }
    if truncated:
        result["truncated_note"] = (
            "this file exceeded the configured read cap and was cut off; "
            "table_count/tables/functional_groups above reflect only the "
            "untruncated prefix and may be short - never treat them as the "
            "complete model"
        )
    return result


def describe_model(name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the structure of one EwE model, or `(None, error)` when no model matches or the file cannot be loaded or parsed.

    The match is case-insensitive and exact, against the file name stem or the file name. A partial name matches nothing.
    """
    models, err = list_models()
    if err:
        return None, err
    query = name.strip().lower()
    match = next(
        (m for m in models if m["name"].lower() == query or m["file"].lower() == query),
        None,
    )
    if match is None:
        available = ", ".join(m["name"] for m in models)
        return None, f"no EwE model named {name!r}; available model(s): {available}"

    key = match["key"]
    with _describe_lock:
        if key in _describe_cache:
            return _describe_cache[key], None
        if key in _describe_error:
            return None, _describe_error[key]

    text, truncated = store.get_text(key, max_bytes=EIIXML_MAX_BYTES)   # SimStoreError propagates
    try:
        parsed = _parse_model(text, truncated)
    except EweError as exc:
        with _describe_lock:
            _describe_error[key] = str(exc)
        return None, str(exc)

    parsed["file"] = match["file"]
    parsed["size_bytes"] = match["size_bytes"]
    with _describe_lock:
        _describe_cache[key] = parsed
    return parsed, None
