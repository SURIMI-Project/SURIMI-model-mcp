"""model_get_stock_status, model_get_stock_trajectory, model_explain_assessment_field.

Q-E7-1/2/3/4 from the SURIMI expert benchmark (E7, stock-assessment
scientist): the CMSY++ result table was already fetchable one field at a
time via model_get_artifact, but nothing unpivoted its year blocks, paired
its bounds, named its estimator, or explained what its input flags mean.
"""
from __future__ import annotations

import pytest

from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.simulations import discovery
from test_cmsy_result import HOGFISH_CSV, HOGFISH_ROW, _csv_text, _row

RUN = "0880dde9-b43c-4888-af62-34783a6b6239"
RESULT_KEY = f"surimi-cmsy/experiments/{RUN}/CMSY_output_historical.csv"
SIM_KEY = f"surimi-cmsy/experiments/{RUN}/CMSY_output.csv"
PAYLOAD = {
    "simulations": [{
        "model": "surimi-cmsy", "simulation_id": RUN,
        "n_artifacts": 1, "last_modified": "2026-01-07T12:54:00",
        "kinds": {"csv": 1}, "keys": [RESULT_KEY],
        "sizes": {RESULT_KEY: len(HOGFISH_CSV)},
    }],
    "scanned": ["surimi-cmsy"], "as_of": "2026-08-06T12:00:00Z", "stale": False,
}
STOCK = "Hogfish - Florida Keys / East Florida"


@pytest.fixture(autouse=True)
def _isolated_summary_cache():
    sim_tools._summary_cache.clear()
    yield
    sim_tools._summary_cache.clear()


def _patch(monkeypatch, payload=PAYLOAD, text=HOGFISH_CSV, truncated=False):
    monkeypatch.setattr(discovery, "scan", lambda force=False: payload)
    monkeypatch.setattr(sim_tools.store, "get_text",
                         lambda k, max_bytes=65536: (text, truncated))


# ── model_get_stock_status ────────────────────────────────────────────────

def test_status_requires_an_explicit_phase(monkeypatch):
    """'a phase that cannot be defaulted' - the parameter has no default, so
    omitting it is a TypeError at the call site, not a silent guess."""
    import inspect
    sig = inspect.signature(sim_tools.model_get_stock_status)
    assert sig.parameters["phase"].default is inspect.Parameter.empty


def test_status_returns_bmsy_fmsy_and_the_current_kobe_pair(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "historical")
    assert out["stock"] == STOCK
    assert out["phase"] == "historical"
    assert out["source_file"] == "CMSY_output_historical.csv"
    assert out["reference_points"]["Bmsy"]["point"] == 1200.0
    assert out["biomass_ratio_to_bmsy"]["point"] == 0.55
    assert out["fishing_pressure_ratio_to_fmsy"]["point"] == 0.88


def test_status_unknown_simulation_id_errors(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_status("does-not-exist", STOCK, "historical")
    assert "error" in out
    assert "no simulation" in out["error"]


def test_status_unknown_stock_lists_known_stocks(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_status(RUN, "Nephrops norvegicus - GSA06", "historical")
    assert "error" in out
    assert STOCK in out["error"]


def test_status_refuses_when_run_has_no_historical_simulated_split(monkeypatch):
    """The live-observed shape: a legacy Out_..._id_file.csv result table
    and no CMSY_output*.csv at all."""
    legacy_key = f"surimi-cmsy/Simulations/{RUN}/Out_South_Med_January082026_id_file.csv"
    payload = {**PAYLOAD, "simulations": [{
        **PAYLOAD["simulations"][0], "keys": [legacy_key], "sizes": {legacy_key: len(HOGFISH_CSV)},
    }]}
    _patch(monkeypatch, payload=payload)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "historical")
    assert "error" in out
    assert "Out_South_Med_January082026_id_file.csv" in out["error"]


def test_status_refuses_a_truncated_result_file(monkeypatch):
    _patch(monkeypatch, truncated=True)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "historical")
    assert "error" in out
    assert "truncat" in out["error"].lower()


def test_status_bad_phase_value_errors(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "current")
    assert "error" in out
    assert "phase" in out["error"].lower()


def test_status_carries_stale_degradation_flag(monkeypatch):
    stale_payload = {**PAYLOAD, "stale": True, "note": "object storage unreachable"}
    _patch(monkeypatch, payload=stale_payload)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "historical")
    assert out["stale"] is True


def test_status_bound_meaning_is_percentile_not_confidence_interval(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_status(RUN, STOCK, "historical")
    assert "percentile" in out["reference_points"]["Bmsy"]["bound_meaning"].lower()


# ── model_get_stock_trajectory ────────────────────────────────────────────

def test_trajectory_requires_an_explicit_phase():
    import inspect
    sig = inspect.signature(sim_tools.model_get_stock_trajectory)
    assert sig.parameters["phase"].default is inspect.Parameter.empty


def test_trajectory_returns_year_indexed_series_in_order(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_trajectory(RUN, STOCK, "historical")
    years = [p["year"] for p in out["trajectory"]]
    assert years == sorted(years)
    assert years[0] == 1986 and years[-1] == 2012


def test_trajectory_unknown_stock_errors(monkeypatch):
    _patch(monkeypatch)
    out = sim_tools.model_get_stock_trajectory(RUN, "nope", "historical")
    assert "error" in out


# ── model_explain_assessment_field ────────────────────────────────────────

def test_explains_btype():
    out = sim_tools.model_explain_assessment_field("btype")
    text = str(out).lower()
    assert "cpue" in text and "none" in text and "biomass" in text
    assert "bsm" in text or "bayesian" in text


def test_explains_btype_names_the_nab_fallback():
    """The Q-E7-1 answer's decisive point: CPUE does not always mean BSM."""
    out = sim_tools.model_explain_assessment_field("btype")
    text = str(out).lower()
    assert "5" in text or "nab" in text


def test_explains_force_cmsy(monkeypatch=None):
    out = sim_tools.model_explain_assessment_field("force.cmsy")
    text = str(out).lower()
    assert "cmsy" in text and "bsm" in text


def test_explains_force_cmsy_accepts_underscore_alias():
    out = sim_tools.model_explain_assessment_field("force_cmsy")
    assert "error" not in out


def test_explains_resilience():
    out = sim_tools.model_explain_assessment_field("Resilience")
    text = str(out).lower()
    assert "high" in text and "medium" in text and "low" in text
    assert "0.6" in text or "1.5" in text


def test_explains_a_year_encoded_output_column():
    out = sim_tools.model_explain_assessment_field("c50")
    text = str(out).lower()
    assert "1950" in text


def test_explains_a_year_encoded_column_past_the_century_wrap():
    out = sim_tools.model_explain_assessment_field("F.Fmsy00")
    text = str(out).lower()
    assert "2000" in text


def test_unknown_field_errors_and_lists_known_fields():
    out = sim_tools.model_explain_assessment_field("not_a_real_field")
    assert "error" in out
    assert "known_fields" in out
    assert "btype" in out["known_fields"]


# ── TOOLS registry ─────────────────────────────────────────────────────────

def test_new_tools_are_registered():
    for name in ("model_get_stock_status", "model_get_stock_trajectory",
                 "model_explain_assessment_field"):
        assert name in sim_tools.TOOLS
        assert callable(getattr(sim_tools, name))
