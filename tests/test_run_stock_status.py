"""model_run_stock_status: B/Bmsy and F/Fmsy from a run's NetCDF.

The WMed output_creator run 9eeadb70 carries stock_status and exploitation on
(year, species) for 47 stocks, 1950-2013. They were unreachable: the gridded
analytics refuse a variable with fewer than 3 dimensions, and the CMSY++
tools read only CMSY_output tables, which no run has. So an expert asking
whether "the stock" in SURIMI's CMSY++ results has improved got a refusal
built on the server's own note that stock status "is not in these runs".
"""
import numpy as np
import pytest
import xarray as xr

from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.simulations import run_status

RUN = "9eeadb70-6ef8-43ea-93ad-ee7b32dceca0"
NC_KEY = f"output_creator/experiments/{RUN}/{RUN}.nc"


def _ds():
    years = np.arange(2008, 2014, dtype="int32")          # 2008..2013
    codes = np.array(["HKE", "PIL", "BFT", "ZZZ"])
    stages = np.array(["adult", "", "", ""])
    b = np.full((6, 4), np.nan, dtype="float32")
    f = np.full((6, 4), np.nan, dtype="float32")
    # HKE: fill (2.0, 0.0) in 2008-2009, assessed from 2010, improving, healthy.
    b[:, 0] = [2.0, 2.0, 0.90, 1.00, 1.10, 1.24]
    f[:, 0] = [0.0, 0.0, 1.20, 0.90, 0.70, 0.51]
    # PIL: overfished and subject to overfishing throughout.
    b[:, 1] = [0.60, 0.55, 0.50, 0.48, 0.45, 0.40]
    f[:, 1] = [1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
    # BFT: overfished only in 2013.
    b[:, 2] = [1.07, 1.00, 0.80, 0.60, 0.50, 0.35]
    f[:, 2] = [0.90, 0.80, 0.70, 0.60, 0.50, 0.42]
    # ZZZ: never assessed (all NaN).
    return xr.Dataset(
        {"stock_status": (("year", "species"), b),
         "exploitation": (("year", "species"), f),
         "species_code": (("species",), codes),
         "species_life_stage": (("species",), stages)},
        coords={"year": years})


# ── pure reading ───────────────────────────────────────────────────────────

def test_the_leading_fill_years_are_dropped():
    s = run_status.read(_ds())
    hke = s["HKE adult"]
    assert [r["year"] for r in hke] == [2010, 2011, 2012, 2013]
    assert hke[0] == {"year": 2010, "b_bmsy": pytest.approx(0.90), "f_fmsy": pytest.approx(1.20)}


def test_a_stock_with_no_values_is_not_listed():
    assert "ZZZ" not in run_status.read(_ds())


def test_the_kobe_quadrants():
    assert run_status.kobe(1.24, 0.51) == "healthy"
    assert run_status.kobe(0.40, 2.00) == "overfished and subject to overfishing"
    assert run_status.kobe(0.35, 0.42) == "overfished"
    assert run_status.kobe(1.10, 1.30) == "subject to overfishing"


def test_the_overview_counts_the_quadrants_in_the_last_year():
    o = run_status.overview(run_status.read(_ds()))
    assert o["year"] == 2013
    assert o["n_stocks"] == 3
    assert o["counts"] == {"healthy": 1, "overfished": 1,
                           "subject to overfishing": 0,
                           "overfished and subject to overfishing": 1}


def test_the_overview_can_be_asked_for_an_earlier_year():
    o = run_status.overview(run_status.read(_ds()), year=2010)
    assert o["year"] == 2010
    assert o["counts"]["overfished and subject to overfishing"] == 2   # HKE and PIL


def test_a_stock_is_found_by_its_code_alone_when_unambiguous():
    s = run_status.read(_ds())
    assert run_status.resolve(s, "hke") == "HKE adult"
    assert run_status.resolve(s, "PIL") == "PIL"
    assert run_status.resolve(s, "XXX") is None


# ── the tool ───────────────────────────────────────────────────────────────

@pytest.fixture
def served(monkeypatch):
    run = {"model": "output_creator", "simulation_id": RUN, "keys": [NC_KEY]}
    deg = {"stale": False, "note": None, "partial": False, "partial_note": None}
    monkeypatch.setattr(sim_tools, "_find_run", lambda sid: (run if sid == RUN else None, deg))
    opened = []
    def fake_open(uri, **kw):
        opened.append(uri)
        return _ds()
    monkeypatch.setattr(sim_tools.nc_io, "open_dataset", fake_open)
    return opened


def test_no_stock_gives_the_overview(served):
    out = sim_tools.model_run_stock_status(RUN)
    assert out["year"] == 2013 and out["n_stocks"] == 3
    assert {r["stock"] for r in out["stocks"]} == {"HKE adult", "PIL", "BFT"}
    assert served == [f"s3://project-surimi/{NC_KEY}"]


def test_a_named_stock_gives_its_status_and_trajectory(served):
    out = sim_tools.model_run_stock_status(RUN, stock="HKE")
    assert out["stock"] == "HKE adult"
    assert out["latest"] == {"year": 2013, "b_bmsy": pytest.approx(1.24),
                             "f_fmsy": pytest.approx(0.51), "kobe": "healthy"}
    assert out["first_assessed_year"] == 2010
    assert [r["year"] for r in out["trajectory"]] == [2010, 2011, 2012, 2013]


def test_every_answer_says_what_the_numbers_are_and_are_not(served):
    out = sim_tools.model_run_stock_status(RUN)
    n = out["semantics_note"].lower()
    assert "b/bmsy" in n and "f/fmsy" in n
    assert "no uncertainty" in n
    assert "historical" in n and "simulated" in n


def test_an_unknown_stock_names_the_known_ones(served):
    out = sim_tools.model_run_stock_status(RUN, stock="COD")
    assert "error" in out and "HKE adult" in out["error"]


def test_an_unknown_run_is_an_error(served):
    assert "error" in sim_tools.model_run_stock_status("nope")


def test_a_run_without_status_variables_says_so(monkeypatch):
    run = {"model": "market", "simulation_id": "r", "keys": ["market/r/out.csv"]}
    deg = {"stale": False, "note": None, "partial": False, "partial_note": None}
    monkeypatch.setattr(sim_tools, "_find_run", lambda sid: (run, deg))
    out = sim_tools.model_run_stock_status("r")
    assert "error" in out and "stock_status" in out["error"]


def test_the_tool_is_registered():
    assert "model_run_stock_status" in sim_tools.TOOLS
