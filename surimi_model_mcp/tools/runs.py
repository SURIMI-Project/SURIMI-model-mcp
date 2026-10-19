"""The simulation tools: list runs, describe them, fetch their files and read CMSY++ and stock-status results.

These tools return no signature, like the listing and inspection tools in `tools/gridded.py`.
`register_all` in `tools/__init__.py` registers every name in `TOOLS` with `mcp.tool()`.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from surimi_model_mcp.netcdf import io as nc_io
from surimi_model_mcp.simulations import cmsy_result, discovery, parsers, run_status, store

logger = logging.getLogger(__name__)

_TEXT_EXT = {"csv", "txt", "r", "yaml", "yml", "json", "tex", "rnw", "bug", "md"}


def _artifact_base() -> str:
    """Absolute base for artifact URLs when the deployment advertises one.

    SURIMI_PUBLIC_BASE_URL is set by the chart from the ingress hostname. With
    it unset (local runs, tests) the URL is site-relative, which is correct for
    a caller talking to this same origin and avoids inventing a hostname that
    would 404 for everyone else.
    """
    return os.environ.get("SURIMI_PUBLIC_BASE_URL", "").rstrip("/")

#: Largest number of runs that `model_list_simulations` summarises in one call.
#: Each summary (and, for CMSY, `stock`) needs one S3 read of the run's own
#: metadata, while the rest of the listing comes from the object-key scan. Runs
#: past the cap get no `summary` key, and the reply reports how many.
_SUMMARY_CAP = 50

#: Number of threads that fetch run summaries at the same time, so a cold call
#: makes up to `_SUMMARY_CAP` S3 reads in parallel instead of one after another.
_SUMMARY_WORKERS = 8

#: Summary cache, keyed by `simulation_id` and kept for the life of the process.
#: A run's files do not change once written, so an entry never expires. Each
#: value is the `{"summary": ..., "stock": ...}` dict that `_summarize_run`
#: returns. A parser refusal (a truncated or ambiguous `id_file.csv`, or an
#: unknown model) is cached, because it is the same on every read. A storage
#: error while fetching is not cached (see `_TransientFetchFailure` and
#: `_cached_summary`), so the next listing tries again.
#:
#: The worker threads share this dict. `_summary_cache_lock` covers each read and
#: each write in `_cached_summary`. Two threads that miss the cache for the same
#: new `simulation_id` both fetch it, which wastes one read and stores the same value.
_summary_cache: dict[str, dict[str, Any]] = {}
_summary_cache_lock = threading.Lock()


class _TransientFetchFailure(Exception):
    """Raised by `_summarize_run` when a fetch inside parsers.describe hit a
    store.SimStoreError, rather than a parser's own deliberate refusal.
    `parsers.describe`'s outer except would otherwise flatten this into the
    exact same {"parse_error": ...} shape as a genuine truncated/ambiguous
    id_file.csv, making the two indistinguishable from the caller's side -
    this exists solely so `_cached_summary` can tell them apart and skip
    caching only this one.
    """


def _summarize_cmsy(meta: dict[str, Any]) -> dict[str, Any]:
    stock = meta.get("stock")
    if not stock:
        return {"summary": None}
    bits = [stock]
    start, end = meta.get("start_year"), meta.get("end_year")
    if start is not None and end is not None:
        bits.append(f"{start}-{end}")
    if meta.get("biomass_type"):
        bits.append(meta["biomass_type"])
    return {"summary": ", ".join(bits), "stock": stock}


def _summarize_market(meta: dict[str, Any]) -> dict[str, Any]:
    cfg = meta.get("config_files")
    local_prices = meta.get("local_prices")
    if not cfg and not local_prices:
        return {"summary": None}
    parts = []
    if cfg:
        rows = meta.get("rows", {})
        parts.append(
            f"{len(cfg)} config file(s): " + ", ".join(
                f"{name} ({rows[name]} rows)" if rows.get(name) is not None
                else f"{name} (truncated)"
                for name in cfg
            )
        )
    if local_prices:
        # Q-E3-10: tick 0 is the market model's initialisation step, not an
        # observed price/quantity pair - say so here too, not only in the
        # raw metadata a caller might not read past.
        parts.append(
            f"local_prices.csv: {local_prices['rows_read']} row(s) read"
            f"{' (truncated)' if local_prices.get('truncated') else ''}, "
            "tick 0 is a cold-start artefact, not an observed price/quantity pair"
        )
    return {"summary": "; ".join(parts)}


#: One builder per model with a parser, keyed like `parsers.PARSERS`. A builder
#: turns the parsed metadata into the `summary` line.
_SUMMARY_BUILDERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "surimi-cmsy": _summarize_cmsy,
    "surimi-market": _summarize_market,
}


def _summarize_run(model: str, keys: list[str]) -> dict[str, Any]:
    """{"summary": str | None, ["stock": str]} for one run.

    Reuses parsers.describe - the SAME parser model_describe_simulation
    calls, unmodified and not duplicated - so its refusal behaviour carries
    over unchanged: a genuine parse_error (truncated id_file.csv, or an
    ambiguous/missing "Out_"-prefixed one) means summary is None here too,
    never a value read from a partial fetch.

    `fetch_text` is NOT passed straight through as store.get_text: it is
    wrapped so a store.SimStoreError raised while fetching can be told
    apart from a parser's own deliberate refusal, even though
    parsers.describe's outer except would otherwise flatten both into the
    identical {"parse_error": ...} shape. `_market` also catches per-file
    fetch exceptions itself and keeps going (see parsers.py) - the wrapper
    still records the failure even when describe()'s return value hides it
    behind a merely-partial-looking result, so a run degraded by even one
    transient fetch failure is treated as transient overall, not cached as
    if the partial result were a deterministic fact about the run.

    An unknown model (no parser) and a known model whose parser legitimately
    found nothing (no id_file.csv, no config/*.csv) both make parsers.
    describe return {} - both cases fall through to `summary: None` without
    needing to tell them apart, because in either case there is genuinely
    nothing to report.
    """
    transient: dict[str, store.SimStoreError | None] = {"exc": None}

    def fetch(key: str) -> tuple[str, bool]:
        try:
            return store.get_text(key)
        except store.SimStoreError as exc:
            transient["exc"] = exc
            raise

    meta = parsers.describe(model, keys, fetch)
    if transient["exc"] is not None:
        raise _TransientFetchFailure(str(transient["exc"]))
    if not meta or meta.get("parse_error"):
        return {"summary": None}
    builder = _SUMMARY_BUILDERS.get(model)
    if not builder:
        return {"summary": None}
    return builder(meta)


def _cached_summary(
    model: str, simulation_id: str, keys: list[str],
) -> tuple[dict[str, Any], bool]:
    """Cache-or-compute wrapper around `_summarize_run`.

    Returns (fields, failed_this_call). `failed_this_call` is True whenever
    the result was NOT written to `_summary_cache` - a caller uses it to
    count/report runs whose summary could not be computed this call, so
    that degradation is visible the same way stale/partial already are,
    rather than being silently indistinguishable from a deliberate
    `summary: null`.

    Two failure classes reach here, both uncached, both logged, both
    reported the same way to the caller:

    - `_TransientFetchFailure` - a store.SimStoreError during fetch; a
      property of the moment (network blip, expiring STS token), not of the
      run. Retried on the next call.
    - any other Exception - a second line of defence for a bug in a summary
      BUILDER (_summarize_cmsy/_summarize_market), which runs after
      parsers.describe and outside its own protection. Not cached either,
      so a fixed builder gets a fresh attempt rather than a result frozen
      at the moment of the bug.

    A genuine parser refusal (truncated id_file.csv, ambiguous/missing
    "Out_" candidate, unknown model) never raises at all - `_summarize_run`
    returns `{"summary": None}` normally for those, which IS cached below,
    because that null is a deterministic fact about the run's own immutable
    content, not a failure of this call.

    Called from a worker-pool thread (see model_list_simulations), so the
    cache check and the eventual write are each taken under
    `_summary_cache_lock` - NOT the fetch/parse in between, which stays
    unlocked on purpose: serialising that would defeat the point of running
    it concurrently in the first place. See `_summary_cache`'s module
    comment for why the resulting race (two threads both missing the cache
    for the same brand-new id) is harmless rather than corrupting.
    """
    with _summary_cache_lock:
        if simulation_id in _summary_cache:
            return _summary_cache[simulation_id], False
    try:
        result = _summarize_run(model, keys)
    except _TransientFetchFailure as exc:
        logger.warning(
            "transient failure summarising simulation %s (%s): %s",
            simulation_id, model, exc,
        )
        return {"summary": None}, True
    except Exception as exc:
        logger.warning(
            "summarising simulation %s (%s) failed: %s: %s",
            simulation_id, model, type(exc).__name__, exc,
        )
        return {"summary": None}, True
    with _summary_cache_lock:
        _summary_cache[simulation_id] = result
    return result, False


def _find_run(
    simulation_id: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(run, degradation) - degradation carries stale/note/partial/
    partial_note from the scan() payload wrapper, not the run dict itself,
    so every caller - success OR error - can hedge what it reports the same
    way model_list_simulations already hedges its listing.

    Returned even when no run matches: during a stale or partial window,
    "no simulation with id X" is a claim about our current listing, not
    proof the run does not exist - it may simply be missing from a
    degraded or short listing. That is the same wrong-rather-than-absent
    failure the numeric fields are protected from, so the error path needs
    the same hedge as the success path.
    """
    payload = discovery.scan()
    degradation = {
        "stale": payload.get("stale", False),
        "note": payload.get("note"),
        "partial": payload.get("partial", False),
        "partial_note": payload.get("partial_note"),
    }
    for s in payload.get("simulations", []):
        if s["simulation_id"] == simulation_id:
            return s, degradation
    return None, degradation

#: Returned in the result of `model_list_simulations`, so the LLM reads the
#: referral together with the listing.
_STOCK_STATUS_NOTE = (
    "Western Mediterranean stock status IS in these runs: an output_creator run "
    "whose NetCDF carries stock_status and exploitation holds B/Bmsy and F/Fmsy "
    "per stock and year (47 WMed stocks, up to 2013). Call "
    "model_run_stock_status(simulation_id) for every stock's Kobe position, or "
    "with stock= for one stock's trajectory. For a NORTHEAST ATLANTIC stock the "
    "answer is on the sibling DATA server: dataset ices-stock-assessment-latest "
    "(ICES Stock Database, FAO Area 27, 538 stocks) carries stock_size with "
    "stock_size_description, the blim, bpa, fmsy and msy_b_trigger reference "
    "points, and low_stock_size / high_stock_size as the published confidence "
    "interval. That table does NOT cover the Mediterranean, so a Mediterranean "
    "stock must not be answered from ICES."
)

def model_list_simulations(model: str | None = None) -> dict[str, Any]:
    """List SURIMI simulation runs found in object storage, newest first.

    Call this FIRST for any question about a model RUN or experiment. Returns
    {count, simulations:[{model, simulation_id, n_artifacts, last_modified,
    kinds, summary, stock}], scanned, as_of, stale, note, partial,
    partial_note, denied_run_like_keys, denied_note, summaries_omitted,
    summaries_omitted_note, summaries_failed, summaries_failed_note}.
    `scanned` lists which models actually have at least one run right now -
    that is what answers "which models have runs". `model` filters to one
    model (e.g. 'surimi-cmsy'). Then call model_describe_simulation on a
    simulation_id to see full artifact/metadata detail for just that run.

    `summary` is a short, human-readable line built from the run's own
    parsed metadata - for CMSY, the STOCK plus year range and biomass type
    (e.g. "Hogfish - Florida Keys, 1986-2012, CPUE"); for market, its
    config-file coverage. Read `summary` (and `stock`) here to answer "what
    is run X about" / "which stock did it assess" WITHOUT calling
    model_describe_simulation on every run in the list - that is the whole
    point of this field. `summary` is null when the model has no parser,
    the parser found nothing to say, or the parse was refused (a truncated
    or ambiguous id_file.csv) - null always means unknown, never a
    placeholder or a guess; do not infer a stock from a filename yourself.
    `stock` sits alongside `summary` only for runs where a stock was
    actually determined (CMSY today) - read `stock`, not `summary`, when
    asked specifically "which stock". A run beyond the newest ones
    summarised in this call does not carry a `summary` key AT ALL - that is
    different from `summary: null` ("attempted, nothing to report") - the
    top-level `summaries_omitted` count says how many were skipped to bound
    cost this call, and `summaries_omitted_note` explains; call
    model_describe_simulation directly on any of those by simulation_id.

    `summaries_failed` counts runs (within this call's summarised set)
    whose summary could NOT be computed this call because of a storage
    error while fetching - a property of the moment (a network blip, an
    expiring STS token), not a fact about the run - rather than a
    deliberate parser refusal. Those runs show `summary: null` exactly like
    a genuine refusal, but are retried on the next call rather than staying
    null forever - do not treat their null as a final answer the way a real
    refusal's null is; say the summary was unavailable this call rather
    than that the run has none. `summaries_failed_note` explains when it is
    above zero.

    A COLD call (nothing cached yet) fetches summaries for up to
    _SUMMARY_CAP runs concurrently, which can still take a few seconds even
    though it is no longer one round trip per run done in series - mention
    that a first call may be slower than a later one rather than assuming
    every call is instant.

    stale=true means object storage was unreachable and this is the
    last good listing - say so rather than presenting it as current.
    partial=true means some objects could not be listed even in a fresh
    scan, so counts here (n_artifacts, kinds) may be short - say that too,
    do not present them as exact. denied_run_like_keys is a COVERAGE hint,
    not a degradation - this listing itself is correct and current - it
    counts objects that look like simulation runs but sit under a
    deliberately-excluded area of storage (housekeeping/lake prefixes, e.g.
    "raw/", "processed/") and so are never scanned for runs at all. When it
    is above zero, mention that some run-like objects exist outside the
    scanned area (see denied_note) rather than implying this listing is the
    entire ensemble's coverage.
    """
    payload = discovery.scan()
    sims = payload.get("simulations", [])
    if model:
        sims = [s for s in sims if s["model"] == model]

    # `discovery.scan` sorts `sims` newest first, so the cap keeps the newest runs.
    # Runs past the cap get no `summary` or `stock` key in this call.
    n_summarized = min(len(sims), _SUMMARY_CAP)
    omitted = len(sims) - n_summarized
    n_failed = 0

    # `pool.map` returns results in input order, so `summary_results[i]` belongs
    # to `sims[i]`.
    summary_results: list[tuple[dict[str, Any], bool]] = []
    if n_summarized:
        with ThreadPoolExecutor(max_workers=min(_SUMMARY_WORKERS, n_summarized)) as pool:
            summary_results = list(pool.map(
                lambda i: _cached_summary(
                    sims[i]["model"], sims[i]["simulation_id"], sims[i]["keys"],
                ),
                range(n_summarized),
            ))

    listed = []
    for i, s in enumerate(sims):
        # "keys" and "sizes" are the run's file list, which `model_describe_simulation`
        # returns, so the listing leaves them out.
        entry = {k: v for k, v in s.items() if k not in ("keys", "sizes")}
        if i < n_summarized:
            fields, failed = summary_results[i]
            entry.update(fields)
            if failed:
                n_failed += 1
        listed.append(entry)

    result = {
        "count": len(sims),
        "simulations": listed,
        "scanned": payload.get("scanned", []),
        "as_of": payload.get("as_of"),
        "stale": payload.get("stale", False),
        "note": payload.get("note"),
        "partial": payload.get("partial", False),
        "partial_note": payload.get("partial_note"),
        # Coverage hint, not a degradation flag. Only this listing returns it,
        # because the other tools answer for one known run.
        "denied_run_like_keys": payload.get("denied_run_like_keys", 0),
        "denied_note": payload.get("denied_note"),
        "summaries_omitted": omitted,
        # Counts runs that were tried and hit a storage error (see `_cached_summary`).
        # `summaries_omitted` counts runs past the cap that were not tried.
        "summaries_failed": n_failed,
    }
    if omitted:
        result["summaries_omitted_note"] = (
            f"{omitted} run(s) beyond the newest {n_summarized} were not "
            "summarised in this call to bound cost - they carry no "
            "`summary` key at all (not `summary: null`); call "
            "model_describe_simulation on their simulation_id directly"
        )
    if n_failed:
        result["summaries_failed_note"] = (
            f"{n_failed} run(s) show summary: null because of a storage "
            "error while fetching this call, not a deliberate refusal - "
            "this is transient (see server log for which); a later call to "
            "model_list_simulations will retry them, it was not cached"
        )
    result["stock_status_note"] = _STOCK_STATUS_NOTE
    return result


def model_describe_simulation(simulation_id: str) -> dict[str, Any]:
    """Describe one simulation run: its artifacts plus parsed metadata.

    Returns {model, simulation_id, n_artifacts, last_modified, metadata,
    artifacts:[{name, size, kind, access}], stale, note, partial,
    partial_note} - or {error, stale, note, partial, partial_note} if no
    matching run is found. `access` is 'inline' for text you can fetch with
    model_get_artifact, or 'link' for any other file, which
    model_get_artifact returns as a download link.
    `metadata` is model-specific: for CMSY it names the STOCK the run
    assessed - read it and report it, because the stock decides what the
    run is about. stale=true means object storage was unreachable and
    n_artifacts, last_modified, and every artifact `size` come from the
    last good listing, not a fresh one - say so rather than presenting them
    as current. partial=true means some objects were dropped from even a
    fresh listing, so this run's artifact count/manifest may be short - say
    that too. An "error: no simulation with id ..." while stale or partial
    is true is NOT proof the run does not exist - it may simply be missing
    from a degraded or short listing; say that rather than asserting the
    run does not exist.
    """
    run, deg = _find_run(simulation_id)
    if not run:
        return {
            "error": f"no simulation with id {simulation_id!r}; "
                     "call model_list_simulations to see what exists",
            **deg,
        }

    sizes = run.get("sizes", {})
    artifacts = []
    for key in run["keys"]:
        name = key.rsplit("/", 1)[-1]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        artifacts.append({
            "name": name,
            "size": sizes.get(key),
            "kind": "text" if ext in _TEXT_EXT else "binary",
            "access": "inline" if ext in _TEXT_EXT else "link",
        })
    return {
        "model": run["model"],
        "simulation_id": simulation_id,
        "n_artifacts": run["n_artifacts"],
        "last_modified": run["last_modified"],
        # `store.get_text` returns the (content, truncated) tuple that
        # `parsers.describe` expects from its `fetch_text` argument.
        "metadata": parsers.describe(run["model"], run["keys"], store.get_text),
        "artifacts": sorted(artifacts, key=lambda a: a["name"]),
        **deg,
    }


def model_get_artifact(simulation_id: str, filename: str) -> dict[str, Any]:
    """Fetch one artifact from a run.

    Text (csv, R, yaml, tex, ...) comes back inline as {kind:'text', name,
    content, truncated, stale, note, partial, partial_note} - plus
    `truncation_note` when truncated=true. Any other file (pdf, jpg, nc) comes
    back as {kind:'link', name, url, stale, note, partial, partial_note}. The
    `url` points to this server's /artifact route, which serves the file from
    the server's local cache, so it is a download link to give the user and
    NOT something to describe the contents of. The `url` is absolute when the
    deployment sets SURIMI_PUBLIC_BASE_URL and relative to this server
    otherwise. Never claim to have read a PDF. stale=true means the run
    was looked up in a degraded (last-good) listing because object storage
    was unreachable when it was scanned - the artifact itself is still
    read from the server, but say the listing that found it may be outdated
    rather than presenting it as current. partial=true means some objects
    were dropped from even a fresh listing, so a "not an artifact of this
    run" error may reflect a short manifest rather than the file truly being
    absent - say so instead of asserting it does not exist. truncated=true
    means `content` is a PREFIX of the file, cut at a fixed byte cap,
    possibly mid-field or mid-row: never count, total, or summarise it as if
    it were the whole file - a 46,436-row CSV can come back showing ~3,761
    rows, and counting those and reporting "3,761 rows" states a number that
    is wrong, not merely incomplete. Only quote from what is shown, and say
    the file was truncated.
    """
    run, deg = _find_run(simulation_id)
    if not run:
        return {"error": f"no simulation with id {simulation_id!r}", **deg}

    match = [k for k in run["keys"] if k.rsplit("/", 1)[-1] == filename]
    if not match:
        return {
            "error": f"{filename!r} is not an artifact of {simulation_id}; "
                     "call model_describe_simulation to list them",
            **deg,
        }
    if len(match) > 1:
        # Several keys in this run share the requested file name, so the reply
        # names them instead of choosing one.
        return {
            "error": f"{filename!r} is ambiguous in simulation "
                     f"{simulation_id}: {len(match)} artifacts share this "
                     f"name - {', '.join(sorted(match))}",
            **deg,
        }
    key = match[0]

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in _TEXT_EXT:
        content, truncated = store.get_text(key)
        result = {"kind": "text", "name": filename,
                  "content": content, "truncated": truncated,
                  **deg}
        if truncated:
            # A read cut at the cap can end inside a field and still look like
            # complete CSV, so the reply carries a sentence the LLM can act on.
            result["truncation_note"] = (
                "this content was cut off at the read cap and is a PREFIX "
                "of the file, possibly mid-field or mid-row - do not count, "
                "total, or summarise it as if it were the whole file; only "
                "quote from what is shown"
            )
        return result
    # The server serves binary files itself under `ARTIFACT_PREFIX` (see
    # `app_factory.py`), so the link carries no expiry fields.
    return {"kind": "link", "name": filename,
            "url": f"{_artifact_base()}/artifact/{simulation_id}/{filename}",
            **deg}


def _basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _cmsy_result_lookup(
    simulation_id: str, stock: str, phase: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(row, error_response). Exactly one is non-None.

    Shared by model_get_stock_status and model_get_stock_trajectory: both
    need the same run lookup, phase resolution (never guessed - see
    simulations/cmsy_result.resolve_result_file) and stock-row selection,
    and both must fail the identical way for the identical reason.
    """
    run, deg = _find_run(simulation_id)
    if not run:
        return None, {
            "error": f"no simulation with id {simulation_id!r}; "
                     "call model_list_simulations to see what exists",
            **deg,
        }
    if phase not in cmsy_result.PHASES:
        return None, {
            "error": f"phase must be one of {cmsy_result.PHASES}, got {phase!r}",
            **deg,
        }

    key, refusal = cmsy_result.resolve_result_file(run["keys"], phase)
    if refusal:
        return None, {"error": refusal, **deg}

    content, truncated = store.get_text(key)
    if truncated:
        return None, {
            "error": f"{_basename(key)} was truncated at the read cap; refusing to "
                     "report status/trajectory values that may be cut mid-field",
            **deg,
        }

    row = cmsy_result.read_stock_row(content, stock)
    if row is None:
        stocks = cmsy_result.known_stocks(content)
        shown = ", ".join(stocks[:20])
        more = f" (+{len(stocks) - 20} more)" if len(stocks) > 20 else ""
        return None, {
            "error": f"no stock {stock!r} in {_basename(key)}; known stocks: {shown}{more}",
            **deg,
        }

    return {"row": row, "source_file": _basename(key), **deg}, None


def model_get_stock_status(simulation_id: str, stock: str, phase: str) -> dict[str, Any]:
    """Current Kobe status (B/Bmsy, F/Fmsy, reference points) from a CMSY++
    result table.

    `phase` has NO default - it must be 'historical' (assessed from real
    catch/abundance data) or 'simulation' (assessed from ecosystem-model-
    simulated data, SURIMI-CMSY/server/StockAssessment.py's second run). A
    run that carries no CMSY_output_historical.csv/CMSY_output.csv split at
    all, only one unlabelled Out_..._id_file.csv result table, is refused for
    EITHER phase rather than guessed. The error names that file so the
    caller can fetch it directly with model_get_artifact instead of trusting
    a label this tool invented.

    Returns {stock, phase, source_file, btype, n_abundance_years, estimator,
    estimator_note, biomass_ratio_to_bmsy, fishing_pressure_ratio_to_fmsy,
    reference_points: {Bmsy, Fmsy, MSY}, stale, note, partial, partial_note}
    on success, or {error, stale, note, partial, partial_note} otherwise.

    Every point estimate (biomass_ratio_to_bmsy, fishing_pressure_ratio_to_fmsy,
    and each reference point) comes back as {point, lcl, ucl, bound_meaning} -
    NEVER the point alone. `bound_meaning` says outright that lcl/ucl are the
    2.5th/97.5th percentile of the MCMC posterior, not a confidence interval
    at any stated level - the CMSY++ column names ("lcl"/"ucl") suggest a
    confidence limit, but AA_CMSY++.R never states one, so report the bound
    that way rather than repeating the column's own naming.

    `estimator` names which of AA_CMSY++.R's two fits (CMSY, catch only, or
    BSM, catch + abundance index) the reported numbers came from, decided by
    the SAME rule model_explain_assessment_field('btype') documents: BSM
    only when btype != 'None' AND at least 5 usable abundance-index years
    exist in the assessment window, CMSY otherwise regardless of the btype
    label. `estimator_note` explains the specific reason for this run, and
    says outright that a run's id_file.csv force.cmsy=TRUE would override
    this even when BSM's own gate is met - a fact this tool cannot verify
    because force.cmsy is an id_file.csv INPUT column, never written into
    the 349-column result row this tool reads.
    """
    result, error = _cmsy_result_lookup(simulation_id, stock, phase)
    if error is not None:
        return error
    status = cmsy_result.extract_status(result["row"])
    return {
        "stock": stock, "phase": phase, "source_file": result["source_file"],
        **status,
        "stale": result["stale"], "note": result["note"],
        "partial": result["partial"], "partial_note": result["partial_note"],
    }


def model_get_stock_trajectory(simulation_id: str, stock: str, phase: str) -> dict[str, Any]:
    """Year-indexed catch/F.Fmsy/B series from a CMSY++ result table.

    Same `simulation_id`/`stock`/`phase` contract, lookup and refusal
    behaviour as model_get_stock_status (see its docstring) - `phase` has no
    default and a run with no historical/simulated split is refused by name
    rather than guessed.

    Returns {stock, phase, source_file, trajectory: [{year, catch, f_fmsy,
    b}, ...], stale, note, partial, partial_note}. `trajectory` is sorted by
    the DECODED year, oldest first - AA_CMSY++.R names its year columns
    c50..c99 (1950-1999) then c00..c30 (2000-2030) in the SAME row, so
    sorting the column names themselves would put 2000 before 1950. Years
    where catch, F/Fmsy and B are all NA (the schema reaches to 2030
    unconditionally, but a stock assessed only through, say, 2012 leaves
    every later column empty) are dropped rather than returned as a row of
    nulls - do not report "data through 2030" from this response's shape.
    """
    result, error = _cmsy_result_lookup(simulation_id, stock, phase)
    if error is not None:
        return error
    trajectory = cmsy_result.extract_trajectory(result["row"])
    return {
        "stock": stock, "phase": phase, "source_file": result["source_file"],
        "trajectory": trajectory,
        "stale": result["stale"], "note": result["note"],
        "partial": result["partial"], "partial_note": result["partial_note"],
    }


#: Definitions of the CMSY++ fields that `model_explain_assessment_field` knows.
#: Each entry cites the GSA06 CMSY++ input `README.txt` or `AA_CMSY++.R`, and the
#: line numbers are from `AA_CMSY++.R`. `_normalise_field` lowercases a key and turns
#: spaces and underscores into '.', so 'N bt', 'n_bt' and 'n.bt' find the same entry.
_ASSESSMENT_FIELDS: dict[str, dict[str, Any]] = {
    "btype": {
        "meaning": (
            "Type of abundance information paired with catch in catch_file.csv's "
            "bt column (GSA06_CMSY++_input files/README.txt): 'None' = catch data "
            "only; 'CPUE' = catch plus a relative abundance index (GSA06's is "
            "MEDITS-derived, kg/km^2); 'biomass' = catch plus an absolute biomass "
            "index."
        ),
        "decides": (
            "Which AA_CMSY++.R estimator runs. btype='None' -> catch-only CMSY "
            "(Monte Carlo). btype in {'CPUE','biomass'} -> the Bayesian Schaefer "
            "Model (BSM, catch + abundance index) is attempted, but ONLY if the "
            "gate at AA_CMSY++.R:1135 is met: at least nab=5 years with a non-NA "
            "abundance value in the assessment window. Below that threshold the "
            "run falls back to CMSY regardless of the btype label - a stock "
            "labelled CPUE with fewer than 5 usable index years is run exactly "
            "as if btype were 'None'."
        ),
        "validation": (
            "AA_CMSY++.R:620-621 rejects any other value: "
            "\"ERROR: In ID file, btype must be None, CPUE, or biomass.\""
        ),
        "source": (
            "GSA06_CMSY++_input files/README.txt schema section; "
            "AA_CMSY++.R:596-621,1135"
        ),
    },
    "force.cmsy": {
        "meaning": (
            "id_file.csv logical column (GSA06_CMSY++_input files/README.txt: "
            "'If TRUE, forces CMSY results to be used instead of BSM when "
            "abundance data are unreliable'). GSA06's own id_file.csv sets it "
            "FALSE for all 153 stocks."
        ),
        "decides": (
            "AA_CMSY++.R:1267 picks which fitted model's numbers populate the "
            "run's final management columns (Bmsy, Fmsy, MSY, last.B_Bmsy, "
            "last.F_Fmsy, ...): BSM, unless FullSchaefer is False (BSM's own "
            "nab gate was not met) OR force.cmsy is TRUE - in which case CMSY is "
            "used even though the BSM-suffixed columns (MSY_BSM, r_BSM, ...) "
            "were still computed and remain present in the output row."
        ),
        "caveat": (
            "force.cmsy is an id_file.csv INPUT column - it is never written "
            "into the 349-column CMSY++ result row, so model_get_stock_status "
            "cannot tell from that row alone whether it was set for a given run."
        ),
        "source": "GSA06_CMSY++_input files/README.txt:101; AA_CMSY++.R:1267,1352",
    },
    "resilience": {
        "meaning": (
            "id_file.csv column naming the stock's prior productivity class: "
            "one of 'High', 'Medium', 'Low', 'Very low' "
            "(GSA06_CMSY++_input files/README.txt). Any other value is refused."
        ),
        "decides": (
            "Sets the initial prior range for the intrinsic growth rate r when "
            "the id_file's own r.low/r.hi are not given (AA_CMSY++.R:679-684): "
            "High -> r in [0.6, 1.5]; Medium -> [0.2, 0.8]; Low -> [0.05, 0.5]; "
            "Very low -> [0.015, 0.1]. This changes the MSY/Bmsy/Fmsy estimates "
            "through the r prior - Resilience is not just descriptive metadata."
        ),
        "validation": (
            "AA_CMSY++.R:674-675: \"ERROR: Resilience not High, Medium, Low, or "
            "Very low in ID input file\""
        ),
        "source": "GSA06_CMSY++_input files/README.txt:89; AA_CMSY++.R:673-684",
    },
    "n.bt": {
        "meaning": (
            "Output column 'N bt': the number of non-NA abundance-index (bt) "
            "values inside the assessment window for this stock."
        ),
        "decides": (
            "Compared against AA_CMSY++.R:1135's nab=5 threshold to decide "
            "CMSY vs BSM - see model_explain_assessment_field('btype')."
        ),
        "source": "AA_CMSY++.R:363 (outheaders), :1135 (the gate)",
    },
    "last.b.bmsy": {
        "meaning": (
            "Output column 'last.B_Bmsy': the stock's biomass in the last "
            "assessed year, relative to Bmsy - the vertical axis of the Kobe "
            "plot. Always reported here with lcl.last.B_Bmsy/ucl.last.B_Bmsy, "
            "the 2.5th/97.5th percentile of the MCMC posterior, never alone."
        ),
        "source": "AA_CMSY++.R:378 (outheaders), :2302 (written from bk.cmsy/bk.bsm)",
    },
    "last.f.fmsy": {
        "meaning": (
            "Output column 'last.F_Fmsy': fishing pressure in the last "
            "assessed year, relative to Fmsy - the horizontal axis of the Kobe "
            "plot. Always reported here with its own lcl/ucl percentile pair."
        ),
        "source": "AA_CMSY++.R:379 (outheaders)",
    },
}

_YEAR_COLUMN_FIELD = re.compile(r"^(c|f\.fmsy|b)(\d{2})$")
_YEAR_SERIES_LABEL = {
    "c": "reconstructed annual catch",
    "f.fmsy": "fishing pressure relative to Fmsy",
    "b": "biomass (absolute, not relative to Bmsy)",
}


def _normalise_field(field: str) -> str:
    return re.sub(r"[\s_]+", ".", field.strip()).lower()


def model_explain_assessment_field(field: str) -> dict[str, Any]:
    """What a CMSY++ id_file.csv input or result-table column means and
    what it decides - the vocabulary model_get_stock_status/
    model_get_stock_trajectory assume the caller already has.

    Covers, at minimum, 'btype', 'force.cmsy' and 'Resilience' (the three
    id_file.csv fields that change which estimator runs or how), plus the
    year-encoded result columns ('c50', 'F.Fmsy00', 'B30', ...: decodes the
    two-digit year the same way model_get_stock_trajectory does, including
    the century wrap at 50) and a handful of named result-table columns
    ('N bt', 'last.B_Bmsy', 'last.F_Fmsy'). Definitions are sourced from
    GSA06_CMSY++_input files/README.txt and AA_CMSY++.R's own validation -
    not invented - and each entry cites where.

    Field names are matched case-insensitively with spaces/underscores
    treated as '.', so 'btype', 'BType' and 'b_type' all resolve the same
    entry. An unmatched field returns {error, known_fields} naming every
    field this tool DOES know, rather than a silent guess.
    """
    norm = _normalise_field(field)
    if norm in _ASSESSMENT_FIELDS:
        return {"field": field, **_ASSESSMENT_FIELDS[norm]}

    m = _YEAR_COLUMN_FIELD.match(norm)
    if m:
        series, suffix = m.groups()
        year = cmsy_result.decode_cmsy_year(suffix)
        return {
            "field": field,
            "meaning": (
                f"Year-encoded output column: {_YEAR_SERIES_LABEL[series]} for "
                f"{year}. The two-digit suffix wraps by century - AA_CMSY++.R "
                "names c50..c99 for 1950-1999 and c00..c30 for 2000-2030 in the "
                "SAME row, so suffixes below 50 mean 20xx and 50-99 mean 19xx; "
                "sorting these column names as text (not by decoded year) puts "
                "2000 before 1950."
            ),
            "decoded_year": year,
            "source": "AA_CMSY++.R:381-399 (outheaders' c/F.Fmsy/B blocks)",
        }

    known = sorted(_ASSESSMENT_FIELDS) + [
        "<series><YY> e.g. c50, F.Fmsy00, B30 (year-encoded result columns)",
    ]
    return {
        "error": f"{field!r} is not a known CMSY++ assessment field",
        "known_fields": known,
    }


@nc_io.serialized
def model_run_stock_status(
    simulation_id: str, stock: str | None = None, year: int | None = None,
) -> dict[str, Any]:
    """B/Bmsy and F/Fmsy per stock from a run's NetCDF (stock_status,
    exploitation on year x species), with each stock's Kobe position.

    Use this for the status of WESTERN MEDITERRANEAN stocks in SURIMI's
    runs: whether a stock is overfished, has improved or deteriorated, or is
    recovering. It reads the first NetCDF file of the run. Without `stock` it
    returns every stock's position in `year` (default: the last year) and the
    count per Kobe quadrant, which answers "the stock" when no stock was
    named. With `stock` (a code such as "HKE", or a label such as "HKE
    adult") it returns that stock's position in `year` (default: its latest
    year), its first assessed year and its full trajectory.

    Returns {simulation_id, source_file, semantics_note, year, n_stocks,
    counts, stocks} or {simulation_id, source_file, semantics_note, stock,
    first_assessed_year, latest: {year, b_bmsy, f_fmsy, kobe}, trajectory},
    plus stale/note/partial/partial_note; or {error, ...}. Read
    semantics_note before reporting: there are no uncertainty bounds, and
    historical and simulated years are not distinguished.
    """
    run, deg = _find_run(simulation_id)
    if not run:
        return {"error": f"no simulation with id {simulation_id!r}; "
                         "call model_list_simulations to see what exists", **deg}
    nc_keys = [k for k in run["keys"] if k.lower().endswith(".nc")]
    if not nc_keys:
        return {"error": f"simulation {simulation_id} has no NetCDF, so no "
                         "stock_status/exploitation series to read", **deg}
    key = nc_keys[0]
    try:
        ds = nc_io.open_dataset(f"s3://{store.BUCKET}/{key}")
    except nc_io.NetCDFIOError as exc:
        return {"error": f"could not open {_basename(key)}: {exc}", **deg}
    try:
        if not run_status.has_status(ds):
            return {"error": f"{_basename(key)} carries no stock_status/exploitation "
                             "variables", **deg}
        series = run_status.read(ds)
    finally:
        close = getattr(ds, "close", None)
        if close:
            close()
    base = {"simulation_id": simulation_id, "source_file": _basename(key),
            "semantics_note": run_status.SEMANTICS_NOTE}
    if stock is None:
        return {**base, **run_status.overview(series, year), **deg}
    label = run_status.resolve(series, stock)
    if label is None:
        return {"error": f"no stock {stock!r} in {_basename(key)}; known stocks: "
                         f"{', '.join(sorted(series))}", **deg}
    rows = series[label]
    last = rows[-1] if year is None else next((r for r in rows if r["year"] == year), None)
    if last is None:
        return {"error": f"{label} has no status in {year}; assessed "
                         f"{rows[0]['year']}-{rows[-1]['year']}", **deg}
    return {**base, "stock": label, "first_assessed_year": rows[0]["year"],
            "latest": {**last, "kobe": run_status.kobe(last["b_bmsy"], last["f_fmsy"])},
            "trajectory": rows, **deg}


TOOLS = [
    "model_list_simulations",
    "model_describe_simulation",
    "model_get_artifact",
    "model_get_stock_status",
    "model_get_stock_trajectory",
    "model_explain_assessment_field",
    "model_run_stock_status",
]
