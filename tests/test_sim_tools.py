import logging
import threading
import time

import pytest

from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.simulations import discovery


@pytest.fixture(autouse=True)
def _isolated_summary_cache():
    """sim_tools._summary_cache is a module-level, life-of-process cache
    (per-run listing summaries). Without this it leaks between
    tests the same way discovery's scan cache would - a later test's
    "no re-fetch" assertion would pass for the wrong reason (a PRIOR test
    already populated the entry), and a get_text call-count assertion would
    be comparing against stale state."""
    sim_tools._summary_cache.clear()
    yield
    sim_tools._summary_cache.clear()


RUN = "0880dde9-b43c-4888-af62-34783a6b6239"
# CMSY runs carry TWO keys ending in "id_file.csv" in real life: the run's
# own output (Out_<region><date>_id_file.csv) and the CMSY input stock
# database (bare id_file.csv, unrelated schema). parsers._select_cmsy_id_file
# refuses to parse unless exactly one "Out_"-prefixed candidate exists, so
# fixtures that want a successful parse must use the "Out_" name.
ID_FILE_NAME = "Out_HogfishFloridaKeys20260107_id_file.csv"
ID_FILE_KEY = f"surimi-cmsy/Simulations/{RUN}/{ID_FILE_NAME}"
REPORT_KEY = f"surimi-cmsy/Simulations/{RUN}/report.pdf"
KEYS = [ID_FILE_KEY, REPORT_KEY]
PAYLOAD = {
    "simulations": [{"model": "surimi-cmsy", "simulation_id": RUN,
                     "n_artifacts": 2, "last_modified": "2026-01-07T12:54:00",
                     "kinds": {"csv": 1, "pdf": 1}, "keys": KEYS,
                     "sizes": {ID_FILE_KEY: 812, REPORT_KEY: 204800}}],
    "scanned": ["surimi-cmsy"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}
CMSY_ID = ('"Name","SciName","Stock","start.yr","end.yr","btype"\n'
           '"Hogfish","Lachnolaimus maximus","Hogfish - Florida Keys",1986,2012,"CPUE"\n')

# Fixture for addition 4: two artifacts in one run sharing a basename.
DUP_RUN = "11111111-2222-3333-4444-555555555555"
DUP_KEY_A = f"surimi-market/simulations/{DUP_RUN}/scenario_a/summary.csv"
DUP_KEY_B = f"surimi-market/simulations/{DUP_RUN}/scenario_b/summary.csv"
DUP_PAYLOAD = {
    "simulations": [{"model": "surimi-market", "simulation_id": DUP_RUN,
                     "n_artifacts": 2, "last_modified": "2026-02-01T00:00:00",
                     "kinds": {"csv": 2}, "keys": [DUP_KEY_A, DUP_KEY_B],
                     "sizes": {DUP_KEY_A: 10, DUP_KEY_B: 20}}],
    "scanned": ["surimi-market"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}

# Fixtures for the model_list_simulations `summary`/`stock` addition.
MARKET_RUN = "22222222-3333-4444-5555-666666666666"
MARKET_CFG_KEY = f"surimi-market/simulations/{MARKET_RUN}/config/species.csv"
MARKET_PAYLOAD = {
    "simulations": [{"model": "surimi-market", "simulation_id": MARKET_RUN,
                     "n_artifacts": 1, "last_modified": "2026-03-01T00:00:00",
                     "kinds": {"csv": 1}, "keys": [MARKET_CFG_KEY],
                     "sizes": {MARKET_CFG_KEY: 40}}],
    "scanned": ["surimi-market"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}
MARKET_CSV = ("species_code,life_stage,species_name\n"
              "ANE,adult,Engraulis\nHKE,NA,Merluccius\n")

UNKNOWN_RUN = "33333333-4444-5555-6666-777777777777"
UNKNOWN_KEY = f"poseidon/Runs/{UNKNOWN_RUN}/output.nc"
UNKNOWN_PAYLOAD = {
    "simulations": [{"model": "poseidon", "simulation_id": UNKNOWN_RUN,
                     "n_artifacts": 1, "last_modified": "2026-04-01T00:00:00",
                     "kinds": {"nc": 1}, "keys": [UNKNOWN_KEY],
                     "sizes": {UNKNOWN_KEY: 100}}],
    "scanned": ["poseidon"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}

AMBIG_RUN = "44444444-5555-6666-7777-888888888888"
AMBIG_KEY_A = f"surimi-cmsy/Simulations/{AMBIG_RUN}/Out_North_id_file.csv"
AMBIG_KEY_B = f"surimi-cmsy/Simulations/{AMBIG_RUN}/Out_South_id_file.csv"
AMBIG_PAYLOAD = {
    "simulations": [{"model": "surimi-cmsy", "simulation_id": AMBIG_RUN,
                     "n_artifacts": 2, "last_modified": "2026-05-01T00:00:00",
                     "kinds": {"csv": 2}, "keys": [AMBIG_KEY_A, AMBIG_KEY_B],
                     "sizes": {AMBIG_KEY_A: 100, AMBIG_KEY_B: 100}}],
    "scanned": ["surimi-cmsy"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}


def test_tools_registry_is_the_seven_names():
    assert sim_tools.TOOLS == ["model_list_simulations",
                               "model_describe_simulation",
                               "model_get_artifact",
                               "model_get_stock_status",
                               "model_get_stock_trajectory",
                               "model_explain_assessment_field",
                               "model_run_stock_status"]
    for name in sim_tools.TOOLS:
        assert callable(getattr(sim_tools, name))


def test_list_simulations_returns_runs(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    out = sim_tools.model_list_simulations()
    assert out["count"] == 1
    assert out["simulations"][0]["simulation_id"] == RUN
    assert out["stale"] is False
    assert "keys" not in out["simulations"][0]      # manifest belongs to describe
    assert "sizes" not in out["simulations"][0]      # ditto


def test_list_simulations_filters_by_model(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    assert sim_tools.model_list_simulations(model="surimi-market")["count"] == 0
    assert sim_tools.model_list_simulations(model="surimi-cmsy")["count"] == 1


def test_list_simulations_surfaces_denied_run_like_keys_as_a_coverage_hint(monkeypatch):
    """A future model landing in an excluded prefix (raw/, processed/, ...)
    produces run-shaped keys discovery.scan() deliberately drops. The count
    must reach model_list_simulations - the tool answering "what runs/models
    exist" - or nobody who could act on it ever learns."""
    denied_payload = {**PAYLOAD, "denied_run_like_keys": 3,
                       "denied_note": "3 object(s) under a denylisted prefix "
                                      "(raw/, processed/) parse as run "
                                      "directories and were excluded"}
    monkeypatch.setattr(discovery, "scan", lambda force=False: denied_payload)
    out = sim_tools.model_list_simulations()
    assert out["denied_run_like_keys"] == 3
    assert out["denied_note"] == denied_payload["denied_note"]


def test_list_simulations_denied_run_like_keys_is_zero_when_none_exist(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    out = sim_tools.model_list_simulations()
    assert out["denied_run_like_keys"] == 0
    assert out["denied_note"] is None


def test_describe_and_get_artifact_do_not_carry_denied_run_like_keys(monkeypatch):
    """Scoped deliberately to model_list_simulations only: describe/
    get_artifact answer questions about one already-known run, where this
    count is irrelevant noise and does not affect trustworthiness the way
    stale/partial do."""
    denied_payload = {**PAYLOAD, "denied_run_like_keys": 3, "denied_note": "3 object(s) ..."}
    monkeypatch.setattr(discovery, "scan", lambda force=False: denied_payload)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: (CMSY_ID, False))
    described = sim_tools.model_describe_simulation(RUN)
    assert "denied_run_like_keys" not in described
    assert "denied_note" not in described

    monkeypatch.setattr(sim_tools.store, "presign", lambda k, expires_in=900: "https://signed")
    fetched = sim_tools.model_get_artifact(RUN, "report.pdf")
    assert "denied_run_like_keys" not in fetched
    assert "denied_note" not in fetched


def test_describe_simulation_includes_parsed_metadata(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_describe_simulation(RUN)
    assert out["model"] == "surimi-cmsy"
    assert out["metadata"]["species"] == "Lachnolaimus maximus"
    names = {a["name"]: a for a in out["artifacts"]}
    assert names[ID_FILE_NAME]["access"] == "inline"
    assert names["report.pdf"]["access"] == "link"


def test_describe_simulation_reports_artifact_sizes(monkeypatch):
    """Discovery carries per-artifact sizes; sim_tools must
    surface them so the model can decide inline-fetch vs link without an
    extra round trip."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_describe_simulation(RUN)
    names = {a["name"]: a for a in out["artifacts"]}
    assert names[ID_FILE_NAME]["size"] == 812
    assert names["report.pdf"]["size"] == 204800


def test_describe_unknown_simulation_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    out = sim_tools.model_describe_simulation("no-such-run")
    assert "error" in out and "no-such-run" in out["error"]


def test_get_artifact_returns_text_inline(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: ("a,b\n1,2\n", False))
    out = sim_tools.model_get_artifact(RUN, ID_FILE_NAME)
    assert out["kind"] == "text"
    assert out["content"].startswith("a,b")
    assert out["truncated"] is False
    assert "truncation_note" not in out


def test_get_artifact_flags_truncated_text_with_a_natural_language_note(monkeypatch):
    """A bare `truncated: True` boolean is not a signal a model reliably
    hedges on - the reviewer reproduced a 20,000-row CSV read back as 3,761
    rows, final line cut mid-field, with the model then reporting 3,761 as
    the row count. truncation_note must tell it never to count/total/
    summarise, only quote."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: ("a,b\n1,2\n3,", True))
    out = sim_tools.model_get_artifact(RUN, ID_FILE_NAME)
    assert out["truncated"] is True
    assert "truncation_note" in out
    note = out["truncation_note"].lower()
    assert "count" in note and "quote" in note


def test_get_artifact_returns_link_for_binary(monkeypatch):
    """The link points at THIS server, not at a presigned MinIO URL.

    A presigned URL inherited the ~24h STS token's life, so every binary link
    died with it while cached text artifacts kept being served. The bytes now
    come from the local cache over /artifact/..., which does not expire - so
    expires_in_s and expiry_note are gone rather than merely unused.
    """
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "presign",
                        lambda *a, **k: pytest.fail("presign must not be called"))
    out = sim_tools.model_get_artifact(RUN, "report.pdf")
    assert out["kind"] == "link"
    assert out["url"] == f"/artifact/{RUN}/report.pdf"
    assert "expires_in_s" not in out


def test_get_artifact_rejects_a_file_outside_the_run(monkeypatch):
    """filename must resolve within the run - never let a caller reach
    arbitrary bucket keys through this tool."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    out = sim_tools.model_get_artifact(RUN, "../../mcp_data/tables/secret.csv")
    assert "error" in out


# Fixture for the fix-round-1 finding: stale/note from the scan() wrapper
# must reach model_describe_simulation and model_get_artifact, not just
# model_list_simulations.
STALE_NOTE = ("object storage unreachable (the STS token expires ~24h after "
              "launch); serving the last good listing from as_of")
STALE_PAYLOAD = {**PAYLOAD, "stale": True, "note": STALE_NOTE}


def test_describe_simulation_surfaces_staleness(monkeypatch):
    """A stale listing means n_artifacts/last_modified/artifact size all
    come from the last good scan, not a fresh one - the response must say
    so, the same way model_list_simulations already does."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: STALE_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_describe_simulation(RUN)
    assert out["stale"] is True
    assert out["note"] == STALE_NOTE


def test_get_artifact_surfaces_staleness_for_text(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: STALE_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: ("a,b\n1,2\n", False))
    out = sim_tools.model_get_artifact(RUN, ID_FILE_NAME)
    assert out["stale"] is True
    assert out["note"] == STALE_NOTE


def test_get_artifact_surfaces_staleness_for_link(monkeypatch):
    """Discovery staleness still rides on the link branch.

    The URL-expiry caveat that used to sit beside it is deliberately absent
    (the link no longer expires); staleness is a separate concern and
    must still be reported.
    """
    monkeypatch.setattr(discovery, "scan", lambda force=False: STALE_PAYLOAD)
    out = sim_tools.model_get_artifact(RUN, "report.pdf")
    assert out["stale"] is True
    assert out["note"] == STALE_NOTE
    assert "expiry_note" not in out


def test_get_artifact_rejects_an_ambiguous_basename(monkeypatch):
    """Two artifacts in the same run sharing a basename must not
    be silently resolved by picking match[0] - that is a guess, not a
    lookup. Report an error naming both candidates instead."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: DUP_PAYLOAD)
    out = sim_tools.model_get_artifact(DUP_RUN, "summary.csv")
    assert "error" in out
    assert "summary.csv" in out["error"]
    assert DUP_KEY_A in out["error"]
    assert DUP_KEY_B in out["error"]


# Fixture for the store.find() short-listing fix: "partial" must reach all
# three tools, on both success and error paths, alongside "stale".
PARTIAL_NOTE = ("1 object(s) in object storage could not be listed and are "
                "omitted from this inventory (see server log for which) - "
                "artifact counts and manifests here may be short")
PARTIAL_PAYLOAD = {**PAYLOAD, "partial": True, "partial_note": PARTIAL_NOTE}


def test_list_simulations_surfaces_partial(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PARTIAL_PAYLOAD)
    out = sim_tools.model_list_simulations()
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_describe_simulation_surfaces_partial(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PARTIAL_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_describe_simulation(RUN)
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_get_artifact_surfaces_partial_for_text(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PARTIAL_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text", lambda k, max_bytes=65536: ("a,b\n1,2\n", False))
    out = sim_tools.model_get_artifact(RUN, ID_FILE_NAME)
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_get_artifact_surfaces_partial_for_link(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PARTIAL_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "presign", lambda k, expires_in=900: "https://signed")
    out = sim_tools.model_get_artifact(RUN, "report.pdf")
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_describe_unknown_simulation_error_carries_staleness_and_partial(monkeypatch):
    """A false negative during a stale/partial window is the same
    wrong-rather-than-absent failure as an unhedged number: 'no simulation
    with id X' during a degraded window is a claim about our listing, not
    proof the run does not exist."""
    degraded = {**PAYLOAD, "stale": True, "note": STALE_NOTE,
                "partial": True, "partial_note": PARTIAL_NOTE}
    monkeypatch.setattr(discovery, "scan", lambda force=False: degraded)
    out = sim_tools.model_describe_simulation("no-such-run")
    assert "error" in out
    assert out["stale"] is True
    assert out["note"] == STALE_NOTE
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_get_artifact_unknown_simulation_error_carries_staleness(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: STALE_PAYLOAD)
    out = sim_tools.model_get_artifact("no-such-run", ID_FILE_NAME)
    assert "error" in out
    assert out["stale"] is True
    assert out["note"] == STALE_NOTE


def test_get_artifact_missing_filename_error_carries_partial(monkeypatch):
    """The scenario named in review: a run whose keys were dropped from a
    short listing can produce a spurious 'not an artifact of this run'
    error - partial must be visible on this error too, not just on
    successful fetches."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PARTIAL_PAYLOAD)
    out = sim_tools.model_get_artifact(RUN, "does-not-exist.csv")
    assert "error" in out
    assert out["partial"] is True
    assert out["partial_note"] == PARTIAL_NOTE


def test_get_artifact_ambiguous_basename_error_carries_staleness_and_partial(monkeypatch):
    degraded = {**DUP_PAYLOAD, "stale": True, "note": STALE_NOTE,
                "partial": True, "partial_note": PARTIAL_NOTE}
    monkeypatch.setattr(discovery, "scan", lambda force=False: degraded)
    out = sim_tools.model_get_artifact(DUP_RUN, "summary.csv")
    assert "error" in out
    assert out["stale"] is True
    assert out["partial"] is True


# ── model_list_simulations `summary`/`stock` addition ──

def test_list_simulations_includes_cmsy_summary_and_stock(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert sim["stock"] == "Hogfish - Florida Keys"
    assert sim["summary"] == "Hogfish - Florida Keys, 1986-2012, CPUE"


def test_list_simulations_includes_market_summary_without_a_stock(monkeypatch):
    """Market has no stock concept - `stock` must not appear at all, only
    `summary`, built from what _market's parser already yields (config
    coverage / row counts)."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: MARKET_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (MARKET_CSV, False))
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert sim["summary"] is not None
    assert "species.csv" in sim["summary"]
    assert "2" in sim["summary"]          # row count surfaced
    assert "stock" not in sim


LOCAL_PRICES_KEY = f"surimi-market/simulations/{MARKET_RUN}/local_prices.csv"
LOCAL_PRICES_CSV = (
    "tick,market,species,price,catch\n0,ES-CAT,ANE,3.5,120.0\n1,ES-CAT,ANE,3.6,130.0\n"
)
MARKET_WITH_LOCAL_PRICES_PAYLOAD = {
    "simulations": [{"model": "surimi-market", "simulation_id": MARKET_RUN,
                     "n_artifacts": 2, "last_modified": "2026-03-01T00:00:00",
                     "kinds": {"csv": 2},
                     "keys": [MARKET_CFG_KEY, LOCAL_PRICES_KEY],
                     "sizes": {MARKET_CFG_KEY: 40, LOCAL_PRICES_KEY: 80}}],
    "scanned": ["surimi-market"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}


def test_list_simulations_market_summary_flags_tick_0_as_a_cold_start_artefact(monkeypatch):
    """Q-E3-10: local_prices.csv's tick 0 is the market model's own
    initialisation step, not an observed value - the listing-level summary
    must say so rather than let tick 0 read as a real data point."""
    monkeypatch.setattr(discovery, "scan",
                         lambda force=False: MARKET_WITH_LOCAL_PRICES_PAYLOAD)

    def get_text(k, max_bytes=65536):
        return (MARKET_CSV, False) if k == MARKET_CFG_KEY else (LOCAL_PRICES_CSV, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert "local_prices.csv" in sim["summary"]
    assert "cold-start" in sim["summary"].lower()
    assert "tick 0" in sim["summary"]


def test_list_simulations_summary_is_null_for_an_unknown_model(monkeypatch):
    """No parser exists for this model at all - null, not a guess, and the
    absence of a parser means parsers.describe never even calls fetch_text."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: UNKNOWN_PAYLOAD)
    calls = {"n": 0}

    def get_text(k, max_bytes=65536):
        calls["n"] += 1
        return ("", False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert sim["summary"] is None
    assert "stock" not in sim
    assert calls["n"] == 0


def test_list_simulations_summary_is_null_when_id_file_is_truncated(monkeypatch):
    """A truncated id_file.csv is parsers._cmsy's parse_error refusal - the
    summary must be null, never a value scraped from the partial read."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID[:20], True))
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert sim["summary"] is None
    assert "stock" not in sim


def test_list_simulations_summary_is_null_when_id_file_is_ambiguous(monkeypatch):
    """Two "Out_"-prefixed id_file.csv candidates in one run is
    parsers._select_cmsy_id_file's other refusal path - same null."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: AMBIG_PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_list_simulations()
    sim = out["simulations"][0]
    assert sim["summary"] is None
    assert "stock" not in sim


def test_list_simulations_summary_cache_avoids_refetch_on_second_call(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    calls = {"n": 0}

    def get_text(k, max_bytes=65536):
        calls["n"] += 1
        return (CMSY_ID, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    first = sim_tools.model_list_simulations()
    assert calls["n"] == 1
    second = sim_tools.model_list_simulations()
    assert calls["n"] == 1   # cached: no re-fetch on the second listing
    assert second["simulations"][0]["summary"] == first["simulations"][0]["summary"]
    assert second["simulations"][0]["stock"] == first["simulations"][0]["stock"]


def test_list_simulations_transient_store_error_yields_null_and_is_not_cached(monkeypatch):
    """store.get_text raising mid-fetch (the CMSY parser's own fetch(idk)
    call, see parsers._cmsy) is a property of the MOMENT - a network blip,
    an expiring STS token - not a fact about the run's own immutable
    content. Unlike a genuine parser refusal it must NOT be cached: the
    listing must not raise, summary must be null and summaries_failed must
    say so THIS call, but a later call (once storage recovers) must get a
    real summary and re-fetch, not a permanently-stuck null."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    state = {"fail": True}
    calls = {"n": 0}

    def get_text(k, max_bytes=65536):
        calls["n"] += 1
        if state["fail"]:
            raise sim_tools.store.SimStoreError("boom")
        return (CMSY_ID, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    first = sim_tools.model_list_simulations()
    assert first["count"] == 1
    assert first["simulations"][0]["summary"] is None
    assert "stock" not in first["simulations"][0]
    assert first["summaries_failed"] == 1
    assert "summaries_failed_note" in first
    assert calls["n"] == 1

    state["fail"] = False
    second = sim_tools.model_list_simulations()
    assert second["simulations"][0]["summary"] == "Hogfish - Florida Keys, 1986-2012, CPUE"
    assert second["simulations"][0]["stock"] == "Hogfish - Florida Keys"
    assert second["summaries_failed"] == 0
    assert "summaries_failed_note" not in second
    assert calls["n"] == 2   # retried this time - the failed attempt was not cached


def test_list_simulations_non_simstoreerror_from_the_fetch_path_is_treated_as_transient(monkeypatch, caplog):
    """Reproduces the review finding directly: store.get_text does not
    guarantee every failure arrives as SimStoreError UNLESS store._fs()
    itself is guarded - the realistic trigger being an STS token expiring
    during S3 client construction (s3fs.S3FileSystem(...) inside _fs(),
    previously unguarded). Only store._fs is monkeypatched here (to raise a
    bare RuntimeError, NOT SimStoreError) - store.get_text/find/presign
    themselves are the REAL functions, so this exercises store's own fixed
    contract end-to-end: _fs() (and get_text() calling it inside its own
    try) convert the RuntimeError to SimStoreError before it ever reaches
    sim_tools, so sim_tools' closure - which deliberately catches ONLY
    store.SimStoreError, not a broad Exception, so a genuine parser bug
    still surfaces as a bug rather than being retried forever - correctly
    treats it as transient: not cached, counted in summaries_failed, and
    logged."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)

    def boom():
        raise RuntimeError("STS token expired mid-construction")

    monkeypatch.setattr(sim_tools.store, "_fs", boom)

    with caplog.at_level(logging.WARNING):
        out = sim_tools.model_list_simulations()

    assert out["simulations"][0]["summary"] is None
    assert "stock" not in out["simulations"][0]
    assert out["summaries_failed"] == 1
    assert "summaries_failed_note" in out
    assert RUN not in sim_tools._summary_cache   # the exact leak the reviewer reproduced
    assert any("transient" in r.getMessage().lower() for r in caplog.records)


def test_list_simulations_genuine_parse_error_is_cached_and_not_refetched(monkeypatch):
    """A truncated id_file.csv is parsers._cmsy's OWN deliberate refusal
    (not a store error) - a deterministic fact about this run's content
    that will read the same way forever, so unlike a transient failure it
    IS cached: summaries_failed must stay 0 (this is not "storage hiccuped,
    ask again"), and a second listing must not re-fetch at all."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    calls = {"n": 0}

    def get_text(k, max_bytes=65536):
        calls["n"] += 1
        return (CMSY_ID[:20], True)   # truncated -> deliberate parser refusal

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    first = sim_tools.model_list_simulations()
    assert first["simulations"][0]["summary"] is None
    assert first["summaries_failed"] == 0
    assert "summaries_failed_note" not in first
    assert calls["n"] == 1

    second = sim_tools.model_list_simulations()
    assert second["simulations"][0]["summary"] is None
    assert second["summaries_failed"] == 0
    assert calls["n"] == 1   # cached: no re-fetch


def test_cached_summary_survives_a_bug_in_the_summary_builder_itself(monkeypatch):
    """_cached_summary's own try/except is a second line of defence beyond
    parsers.describe's: it covers a bug in the summary-BUILDING step
    (_summarize_cmsy/_summarize_market), which runs after parsers.describe
    returns and is outside that function's protection. A raise there must
    not sink the listing, and the failed attempt must not be cached, so a
    later call (once the builder is fixed) gets a fresh attempt rather than
    a result frozen at the moment of the bug."""
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID, False))

    def broken_builder(meta):
        raise RuntimeError("builder bug")

    monkeypatch.setattr(sim_tools, "_SUMMARY_BUILDERS", {"surimi-cmsy": broken_builder})
    out = sim_tools.model_list_simulations()
    assert out["count"] == 1
    assert out["simulations"][0]["summary"] is None
    assert out["summaries_failed"] == 1

    monkeypatch.setattr(sim_tools, "_SUMMARY_BUILDERS",
                         {"surimi-cmsy": sim_tools._summarize_cmsy})
    out2 = sim_tools.model_list_simulations()
    assert out2["simulations"][0]["summary"] == "Hogfish - Florida Keys, 1986-2012, CPUE"
    assert out2["summaries_failed"] == 0


def test_list_simulations_one_failing_summary_does_not_sink_the_listing(monkeypatch):
    two_run_payload = {
        "simulations": [PAYLOAD["simulations"][0], MARKET_PAYLOAD["simulations"][0]],
        "scanned": ["surimi-cmsy", "surimi-market"],
        "as_of": "2026-08-06T12:00:00Z", "stale": False,
    }
    monkeypatch.setattr(discovery, "scan", lambda force=False: two_run_payload)

    def get_text(k, max_bytes=65536):
        if k == ID_FILE_KEY:
            raise sim_tools.store.SimStoreError("boom")
        return (MARKET_CSV, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)
    out = sim_tools.model_list_simulations()
    assert out["count"] == 2
    by_id = {s["simulation_id"]: s for s in out["simulations"]}
    assert by_id[RUN]["summary"] is None
    assert by_id[MARKET_RUN]["summary"] is not None
    assert out["summaries_failed"] == 1   # the CMSY run's store error, not cached


def _cmsy_run_fixture(i: int) -> dict:
    run_id = f"{i:08d}-0000-0000-0000-000000000000"
    key = f"surimi-cmsy/Simulations/{run_id}/Out_Stock{i}_id_file.csv"
    return {"model": "surimi-cmsy", "simulation_id": run_id,
            "n_artifacts": 1, "last_modified": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
            "kinds": {"csv": 1}, "keys": [key], "sizes": {key: 100}}


def test_list_simulations_caps_cold_summaries_and_reports_the_cap(monkeypatch):
    """More runs than the cap must not turn a listing into an unbounded
    fan-out of fetches. The overflow is reported (summaries_omitted +
    note), never silently dropped, and the omitted entries carry no
    `summary` key at all - distinct from `summary: null`."""
    n = sim_tools._SUMMARY_CAP + 2
    runs = [_cmsy_run_fixture(i) for i in range(n)]
    big_payload = {"simulations": runs, "scanned": ["surimi-cmsy"],
                   "as_of": "2026-08-06T12:00:00Z", "stale": False}
    monkeypatch.setattr(discovery, "scan", lambda force=False: big_payload)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID, False))

    out = sim_tools.model_list_simulations()
    assert out["count"] == n
    assert out["summaries_omitted"] == 2
    assert "summaries_omitted_note" in out
    assert out["summaries_failed"] == 0
    assert "summaries_failed_note" not in out

    summarized = [s for s in out["simulations"] if "summary" in s]
    not_summarized = [s for s in out["simulations"] if "summary" not in s]
    assert len(summarized) == sim_tools._SUMMARY_CAP
    assert len(not_summarized) == 2
    for s in not_summarized:
        assert "stock" not in s


def test_list_simulations_summaries_are_fetched_concurrently_not_sequentially(monkeypatch):
    """Proves the fan-out is actually concurrent, not merely correctly
    bounded - a naive sequential loop would pass every other cap/summary
    test unchanged. Each fake fetch blocks briefly and records how many
    calls were in flight at once; a sequential implementation would never
    see more than 1, and wall-clock time would scale with n rather than
    n / _SUMMARY_WORKERS."""
    n = sim_tools._SUMMARY_WORKERS
    runs = [_cmsy_run_fixture(i) for i in range(n)]
    payload = {"simulations": runs, "scanned": ["surimi-cmsy"],
               "as_of": "2026-08-06T12:00:00Z", "stale": False}
    monkeypatch.setattr(discovery, "scan", lambda force=False: payload)

    lock = threading.Lock()
    state = {"in_flight": 0, "max_in_flight": 0}

    def get_text(k, max_bytes=65536):
        with lock:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        time.sleep(0.05)
        with lock:
            state["in_flight"] -= 1
        return (CMSY_ID, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)

    start = time.monotonic()
    out = sim_tools.model_list_simulations()
    elapsed = time.monotonic() - start

    assert out["count"] == n
    assert all(s["summary"] is not None for s in out["simulations"])
    assert state["max_in_flight"] > 1          # actually overlapped
    assert elapsed < 0.05 * n                  # well under the sequential bound


def test_list_simulations_concurrent_cache_writes_are_all_present_afterwards(monkeypatch):
    """After a concurrent cold call spanning several full rounds through
    the worker pool, every summarised run's result must be in
    _summary_cache - proving the lock-guarded check-then-write does not
    drop or corrupt an entry under real concurrent access - and a second
    call must not re-fetch any of them."""
    n = sim_tools._SUMMARY_WORKERS * 3
    runs = [_cmsy_run_fixture(i) for i in range(n)]
    payload = {"simulations": runs, "scanned": ["surimi-cmsy"],
               "as_of": "2026-08-06T12:00:00Z", "stale": False}
    monkeypatch.setattr(discovery, "scan", lambda force=False: payload)

    calls_lock = threading.Lock()
    calls = {"n": 0}

    def get_text(k, max_bytes=65536):
        with calls_lock:
            calls["n"] += 1
        return (CMSY_ID, False)

    monkeypatch.setattr(sim_tools.store, "get_text", get_text)

    first = sim_tools.model_list_simulations()
    assert first["count"] == n
    assert all(s["summary"] is not None for s in first["simulations"])
    assert calls["n"] == n
    assert len(sim_tools._summary_cache) == n

    second = sim_tools.model_list_simulations()
    assert calls["n"] == n                     # fully cached: zero additional fetches
    assert all(s["summary"] is not None for s in second["simulations"])


def test_list_simulations_summaries_omitted_is_zero_and_note_absent_under_the_cap(monkeypatch):
    monkeypatch.setattr(discovery, "scan", lambda force=False: PAYLOAD)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (CMSY_ID, False))
    out = sim_tools.model_list_simulations()
    assert out["summaries_omitted"] == 0
    assert "summaries_omitted_note" not in out
    assert out["summaries_failed"] == 0
    assert "summaries_failed_note" not in out
