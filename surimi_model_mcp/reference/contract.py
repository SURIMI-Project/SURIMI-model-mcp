"""Reads the northwestern_med case-study contract, which states what SURIMI can model, over what area and period, and how it is coded.

The module reads `KEY` with the default `store.get_text` cap and refuses a truncated file. It parses the file once and caches the result, or the parse error, for the life of the process.
"""
from __future__ import annotations

import threading
from typing import Any

from surimi_model_mcp.simulations import store

KEY = "controller/northwestern_med/northwestern_med_contract.yaml"

DIVERGENCE_NOTE = (
    "this is the northwestern_med case-study contract in the project bucket "
    "(controller/northwestern_med/northwestern_med_contract.yaml). The bucket "
    "also holds aegean_sea and anchovy_bay contracts, which this tool does not "
    "read, so do not take this one for the only case study. The older copy at "
    "surimi-controller/config/western_med_contract.yaml (named \"Western Med!\", "
    "ending 2024-01-01) no longer exists."
)


class ContractError(Exception):
    """The contract file's own content is unusable: a required key is missing or the YAML is invalid.

    `get` caches this error, as `reference/vocab.py` does for `VocabError`.
    """


_cache: dict[str, Any] = {}
_error: dict[str, str] = {}
_lock = threading.Lock()


def reset_cache() -> None:
    """Clear the cached contract and error so that tests can re-read a patched store."""
    with _lock:
        _cache.clear()
        _error.clear()


def _parse(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:                                   # pragma: no cover
        raise ContractError(f"PyYAML unavailable: {exc}") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"{KEY} is not valid YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise ContractError(f"{KEY} did not parse to a mapping")

    sim = doc.get("simulation") or {}
    geo = sim.get("geography") or {}
    crs = geo.get("crs") or {}
    standards = doc.get("standards") or {}
    measurements = standards.get("measurements") or {}
    items = doc.get("items") or {}

    required = {"case_study_name", "start_date_time", "maximum_end_date_time"}
    missing = required - set(sim)
    if missing:
        raise ContractError(f"{KEY}: simulation block missing {sorted(missing)}")

    def _iso(value: Any) -> Any:
        # `yaml.safe_load` turns an unquoted ISO 8601 value such as
        # 2013-01-01T00:00:00 into a datetime, which JSON cannot carry.
        # Return a string whichever form the YAML held.
        return value.isoformat() if hasattr(value, "isoformat") else value

    return {
        "case_study_name": sim.get("case_study_name"),
        "start_date_time": _iso(sim.get("start_date_time")),
        "maximum_end_date_time": _iso(sim.get("maximum_end_date_time")),
        "time_step": sim.get("time_step"),
        "grid": {
            "crs": f"{crs.get('authority')}:{crs.get('code')}" if crs.get("authority") else crs.get("name"),
            "xres": geo.get("xres"), "yres": geo.get("yres"),
            "ncol": geo.get("ncol"), "nrow": geo.get("nrow"),
            "bbox": {
                "xmin": geo.get("xmin"), "xmax": geo.get("xmax"),
                "ymin": geo.get("ymin"), "ymax": geo.get("ymax"),
            },
        },
        "standards": {
            "date_and_time": standards.get("date_and_time"),
            "species_code": standards.get("species_code"),
            "gear_code": standards.get("gear_code"),
            "life_stage": standards.get("life_stage"),
            "market_code": standards.get("market_code"),
            "currency": standards.get("currency"),
            "country_code": standards.get("country_code"),
            "measurement_system": measurements.get("system"),
            "measurement_units": [u.get("unit") for u in (measurements.get("units") or [])],
        },
        "markets": [m.get("market_code") for m in (items.get("markets") or [])],
        "currencies": [c.get("currency_code") for c in (items.get("currencies") or [])],
        "species_count": len(items.get("species") or []),
        "fleet_segment_count": len(items.get("fleet_segments") or []),
        # Each fleet segment is passed through as written, so a field added to
        # the contract is not dropped.
        "fleet_segments": [dict(fs) for fs in (items.get("fleet_segments") or [])],
    }


def get() -> tuple[dict[str, Any] | None, str | None]:
    """Return `(contract, None)`, or `(None, error)` when the file is truncated or fails to parse.

    Both outcomes are cached. A `store.SimStoreError` from the fetch propagates uncached, so the next call retries the read.
    """
    with _lock:
        if "contract" in _cache:
            return _cache["contract"], None
        if "contract" in _error:
            return None, _error["contract"]
    text, truncated = store.get_text(KEY)   # store.SimStoreError propagates
    if truncated:
        err = (
            f"{KEY} was truncated at the read cap; refusing to parse a "
            "contract from a partial file"
        )
        with _lock:
            _error["contract"] = err
        return None, err
    try:
        parsed = _parse(text)
    except ContractError as exc:
        with _lock:
            _error["contract"] = str(exc)
        return None, str(exc)
    with _lock:
        _cache["contract"] = parsed
    return parsed, None
