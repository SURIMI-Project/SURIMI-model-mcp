"""Find simulation runs generically, so a new model needs no code change.

`scan` lists the bucket and groups the objects into runs. A run is any object whose key has a
UUID-shaped directory segment, or a file name that starts with a UUID and an underscore. The model is
the first path segment and the run id is the first such UUID, so a new model needs no configuration.

In a key such as `output_creator/<scenario>/<uuid>/<uuid>.nc` only the directory segment starts the
run.
"""
from __future__ import annotations

import copy
import os
import json
import datetime as _dt
import re
import time
from collections import Counter
from typing import Any

from . import store

_UUID = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

#: A flat run file name, "<uuid>_<name>", where the run id is a prefix of the file name.
_FLAT_RUN = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})_.+"
)

#: Key prefixes that `scan` leaves out of the run grouping. `scan` still lists the whole bucket with
#: `store.find("")` and filters the keys in Python.
DENY_PREFIXES: tuple[str, ...] = (
    "mcp_data/", "surimi_data/", "surimi_docs/", "hive/",
    "processed/", "raw/", "surimi_data_dummy/",
)

CACHE_TTL_S = 300

#: Seconds after a failed listing during which `scan` serves the stale listing without listing again.
#: It is shorter than `CACHE_TTL_S`, so a long outage causes one retry per window and not one per call.
_FAILURE_BACKOFF_S = 30

#: Directory of `sim-listing.json`, the last good run listing. `_cache` is in memory only, and this
#: file lets a restarted server serve a stale listing while the storage token is invalid.
_STATE_DIR = os.environ.get("SURIMI_SIM_STATE", os.environ.get(
    "SURIMI_NC_CACHE", "/tmp/surimi-s3cache"))


def _state_path() -> str:
    return os.path.join(_STATE_DIR, "sim-listing.json")


def _save_payload(payload: dict[str, Any]) -> None:
    """Write a good scan to `sim-listing.json`, and ignore a failed write."""
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        tmp = _state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, _state_path())
    except OSError:
        pass          # a failed write must not fail the scan


def _load_payload() -> dict[str, Any] | None:
    try:
        with open(_state_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


_cache: dict[str, Any] = {"at": 0.0, "payload": None, "failed_at": 0.0}


def _reset_cache(forget_disk: bool = True) -> None:
    """Clear the in-memory cache, and also delete the saved listing when `forget_disk` is True."""
    _cache["at"] = 0.0
    _cache["payload"] = None
    _cache["failed_at"] = 0.0
    if forget_disk:
        try:
            os.remove(_state_path())
        except OSError:
            pass


def _prime_from_disk() -> None:
    """Load the saved listing into the in-memory cache when the cache is empty."""
    if _cache["payload"] is None:
        saved = _load_payload()
        if saved:
            _cache["payload"] = saved


def _serve_stale() -> dict[str, Any]:
    """Return a copy of the last good listing with `stale` set to True and a `note` that says why."""
    stale = copy.deepcopy(_cache["payload"])
    stale["stale"] = True
    stale["note"] = (
        "object storage unreachable (the STS token expires ~24h after "
        "launch); serving the last good listing from as_of"
    )
    return stale


def parse_run(key: str) -> tuple[str, str, str] | None:
    """Return `(model, run_id, relative_path)` when `key` belongs to a run, and None otherwise.

    When the UUID is the first path segment, as in `<uuid>/report.csv`, `model` and `run_id` are equal.
    """
    parts = key.split("/")
    for i, seg in enumerate(parts[:-1]):          # [:-1] => directories only
        if _UUID.match(seg):
            return parts[0], seg, "/".join(parts[i + 1:])
    # A flat run has its id as the file-name prefix, as in
    # fisheries_authority/output/<uuid>_mse-assessment.csv.
    m = _FLAT_RUN.match(parts[-1])
    if m and len(parts) > 1:
        return parts[0], m.group(1), parts[-1]
    return None


def _kind(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name.rsplit("/", 1)[-1] else "none"


def scan(force: bool = False) -> dict[str, Any]:
    """Return the inventory of runs as a dictionary with `simulations`, `scanned`, `as_of`, `stale` and `partial`.

    The listing is cached for `CACHE_TTL_S` seconds unless `force` is True. When storage cannot be listed,
    the last good listing is returned with `stale` set to True, and `store.SimStoreError` is raised only
    when there is none.
    """
    _prime_from_disk()
    now = time.time()
    if not force and _cache["payload"] and now - _cache["at"] < CACHE_TTL_S:
        # Return a deep copy, so a caller that changes a returned run cannot change the cache.
        return copy.deepcopy(_cache["payload"])

    if (not force and _cache["payload"] and _cache["failed_at"]
            and now - _cache["failed_at"] < _FAILURE_BACKOFF_S):
        # A listing failed recently, so serve the stale listing until the backoff window ends.
        return _serve_stale()

    try:
        objects, failures = store.find("")
    except store.SimStoreError:
        _cache["failed_at"] = now
        if _cache["payload"]:
            return _serve_stale()
        raise
    _cache["failed_at"] = 0.0

    runs: dict[tuple[str, str], dict[str, Any]] = {}
    denied_run_like = 0
    for o in objects:
        key = o["key"]
        if any(key.startswith(p) for p in DENY_PREFIXES):
            # Count the keys under a denied prefix that parse as runs, for `denied_note`.
            if parse_run(key):
                denied_run_like += 1
            continue
        parsed = parse_run(key)
        if not parsed:
            continue
        model, run_id, rel = parsed
        if not rel or rel.endswith("/"):
            # A directory-marker key, either the run root (rel == "") or a subdirectory such as
            # "config/". It names no file, so it is not counted as an artifact.
            continue
        r = runs.setdefault((model, run_id), {
            "model": model, "simulation_id": run_id,
            "n_artifacts": 0, "last_modified": None,
            "kinds": Counter(), "keys": [], "sizes": {},
        })
        r["n_artifacts"] += 1
        r["keys"].append(key)
        r["sizes"][key] = int(o.get("size") or 0)
        r["kinds"][_kind(rel)] += 1
        lm = o.get("last_modified")
        if lm and (r["last_modified"] is None or lm > r["last_modified"]):
            r["last_modified"] = lm

    sims = []
    for r in runs.values():
        r["kinds"] = dict(r["kinds"])
        sims.append(r)
    sims.sort(key=lambda s: (s["last_modified"] or "", s["simulation_id"]), reverse=True)

    payload = {
        "simulations": sims,
        "scanned": sorted({s["model"] for s in sims}),
        "as_of": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stale": False,
        # "partial" means some objects of this fresh listing could not be read, and "stale"
        # means an old listing is being served.
        "partial": bool(failures),
        "denied_run_like_keys": denied_run_like,
    }
    if failures:
        payload["partial_note"] = (
            f"{len(failures)} object(s) in object storage could not be "
            "listed and are omitted from this inventory (see server log "
            "for which) - artifact counts and manifests here may be short"
        )
    if denied_run_like:
        payload["denied_note"] = (
            f"{denied_run_like} object(s) under a denylisted prefix "
            f"({', '.join(DENY_PREFIXES)}) parse as run directories and were "
            "excluded from this inventory - if a model has started "
            "publishing under one of these prefixes it is invisible here, "
            "not degraded; check DENY_PREFIXES"
        )
    _cache["at"] = now
    _cache["payload"] = payload
    _save_payload(payload)
    # `payload` is the object now stored in the cache, so return a copy.
    return copy.deepcopy(payload)
