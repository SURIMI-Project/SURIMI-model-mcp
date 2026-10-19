"""simulations.cmsy_result: the wide (349-column) CMSY++ result row that
parsers._cmsy deliberately ignores past its first 9 columns.

Fixture header/row below is transcribed from AA_CMSY++.R:361-399 (the
`outheaders` data.frame CMSY++ writes), not invented - see
simulations/cmsy_result.py's module docstring for the source.
"""
from __future__ import annotations

import csv
import io

from surimi_model_mcp.simulations import cmsy_result as cr

_LEAD_COLS = [
    "Group", "Region", "Subregion", "Name", "SciName", "Stock",
    "start.yr", "end.yr", "start.yr.new", "btype",
    "N bt", "start.yr.cpue", "end.yr.cpue", "min.cpue", "max.cpue",
    "min.yr.cpue", "max.yr.cpue",
    "endbio.low", "endbio.hi", "q.prior.low", "q.prior.hi",
    "MaxCatch", "MSY_prior", "MeanLast5RawCatch", "SDLast5RawCatch", "LastCatch",
    "MinSmoothCatch", "MaxSmoothCatch", "MeanSmoothCatch", "gMeanPrior_r",
    "MSY_BSM", "lcl.MSY_BSM", "ucl.MSY_BSM", "r_BSM", "lcl.r_BSM", "ucl.r_BSM", "log.r_var",
    "k_BSM", "lcl.k_BSM", "ucl.k_BSM", "log.k_var", "log.kr_cor", "log.kr_cov",
    "q_BSM", "lcl.q_BSM", "ucl.q_BSM",
    "rel_B_BSM", "lcl.rel_B_BSM", "ucl.rel_B_BSM",
    "rel_start_B_BSM", "lcl.rel_start_B_BSM", "ucl.rel_start_B_BSM",
    "rel_int_B_BSM", "lcl.rel_int_B_BSM", "ucl.rel_int_B_BSM", "int.yr", "rel_F_BSM",
    "r_CMSY", "lcl.r_CMSY", "ucl.r_CMSY", "k_CMSY", "lcl.k_CMSY", "ucl.k_CMSY",
    "MSY_CMSY", "lcl.MSY_CMSY", "ucl.MSY_CMSY",
    "rel_B_CMSY", "2.5th.rel_B_CMSY", "97.5th.rel_B_CMSY",
    "rel_start_B_CMSY", "2.5th.rel_start_B_CMSY", "97.5th.rel_start_B_CMSY",
    "rel_int_B_CMSY", "2.5th.rel_int_B_CMSY", "97.5th.rel_int_B_CMSY",
    "rel_F_CMSY", "2.5th.rel_F_CMSY", "97.5th.rel_F_CMSY",
    "F_msy", "lcl.F_msy", "ucl.F_msy", "curF_msy", "lcl.curF_msy", "ucl.curF_msy",
    "MSY", "lcl.MSY", "ucl.MSY", "Bmsy", "lcl.Bmsy", "ucl.Bmsy",
    "last.B", "lcl.last.B", "ucl.last.B", "last.B_Bmsy", "lcl.last.B_Bmsy", "ucl.last.B_Bmsy",
    "last.F", "lcl.last.F", "ucl.last.F", "last.F_Fmsy", "lcl.last.F_Fmsy", "ucl.last.F_Fmsy",
    "sel_B", "sel_B_Bmsy", "sel_F", "sel_F_Fmsy",
]


def _year_suffixes() -> list[str]:
    # AA_CMSY++.R's own wrap: c50..c99 (1950-1999), then c00..c30 (2000-2030),
    # all in the SAME row - "c00" sorts before "c50" alphabetically.
    return [f"{d}{i}" for d in range(5, 10) for i in range(10)] + \
           [f"{d}{i}" for d in range(0, 3) for i in range(10)] + ["30"]


_SUFFIXES = _year_suffixes()
_HEADER = _LEAD_COLS + [f"c{s}" for s in _SUFFIXES] \
    + [f"F.Fmsy{s}" for s in _SUFFIXES] + [f"B{s}" for s in _SUFFIXES]

assert len(_HEADER) == 349, len(_HEADER)  # matches the live-verified column count


def _row(stock="Hogfish - Florida Keys / East Florida", btype="CPUE", n_bt=27, **overrides):
    """One realistic (fabricated, not real Hogfish numbers) CMSY++ result
    row. Catch data only for 1986-2012; every other year column is NA,
    matching a real run (schema reaches to 2030, but only years with real
    catch data are filled)."""
    values = {c: "NA" for c in _HEADER}
    values.update({
        "Group": "NA", "Region": "Southeast Shelf", "Subregion": "Florida Keys / East Florida",
        "Name": "Hogfish", "SciName": "Lachnolaimus maximus", "Stock": stock,
        "start.yr": "1986", "end.yr": "2012", "start.yr.new": "NA", "btype": btype,
        "N bt": str(n_bt),
        "Bmsy": "1200", "lcl.Bmsy": "900", "ucl.Bmsy": "1600",
        "F_msy": "0.3", "lcl.F_msy": "0.2", "ucl.F_msy": "0.42",
        "MSY": "310", "lcl.MSY": "250", "ucl.MSY": "390",
        "last.B": "660", "lcl.last.B": "480", "ucl.last.B": "870",
        "last.B_Bmsy": "0.55", "lcl.last.B_Bmsy": "0.4", "ucl.last.B_Bmsy": "0.72",
        "last.F": "0.264", "lcl.last.F": "0.18", "ucl.last.F": "0.35",
        "last.F_Fmsy": "0.88", "lcl.last.F_Fmsy": "0.6", "ucl.last.F_Fmsy": "1.17",
    })
    for i, year in enumerate(range(1986, 2013)):
        suffix = f"{year % 100:02d}"
        values[f"c{suffix}"] = str(1000 + i * 5)
        values[f"F.Fmsy{suffix}"] = str(round(0.5 + i * 0.02, 3))
        values[f"B{suffix}"] = str(2000 - i * 10)
    values.update(overrides)
    return values


def _csv_text(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_HEADER)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


HOGFISH_ROW = _row()
HOGFISH_CSV = _csv_text([HOGFISH_ROW])


# ── decode_cmsy_year ──────────────────────────────────────────────────────

def test_decode_year_wraps_at_the_century():
    assert cr.decode_cmsy_year("50") == 1950
    assert cr.decode_cmsy_year("99") == 1999
    assert cr.decode_cmsy_year("00") == 2000
    assert cr.decode_cmsy_year("30") == 2030


# ── resolve_result_file ──────────────────────────────────────────────────

def test_resolves_the_named_historical_file():
    keys = ["surimi-cmsy/experiments/x/CMSY_output_historical.csv",
            "surimi-cmsy/experiments/x/id_file.csv"]
    key, refusal = cr.resolve_result_file(keys, "historical")
    assert refusal is None
    assert key == "surimi-cmsy/experiments/x/CMSY_output_historical.csv"


def test_resolves_the_named_simulation_file():
    keys = ["surimi-cmsy/experiments/x/CMSY_output_historical.csv",
            "surimi-cmsy/experiments/x/CMSY_output.csv"]
    key, refusal = cr.resolve_result_file(keys, "simulation")
    assert refusal is None
    assert key == "surimi-cmsy/experiments/x/CMSY_output.csv"


def test_phase_argument_is_validated_not_defaulted():
    import pytest
    with pytest.raises(ValueError):
        cr.resolve_result_file(["a/CMSY_output.csv"], "current")


def test_legacy_single_file_run_refuses_rather_than_guesses_the_phase():
    """The real, live-observed shape (a stock-assessment case from the expert
    review): one unlabelled Out_..._id_file.csv result table, no CMSY_output*.csv
    at all. Must name that file and refuse, never silently call it either
    phase."""
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_January082026_id_file.csv",
            "surimi-cmsy/Simulations/abc/id_file.csv",
            "surimi-cmsy/Simulations/abc/catch_file.csv"]
    key, refusal = cr.resolve_result_file(keys, "historical")
    assert key is None
    assert "Out_South_Med_January082026_id_file.csv" in refusal
    assert "historical" not in refusal.split("carries")[0]  # not asserted as fact


def test_legacy_single_file_run_also_refuses_the_simulation_phase():
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_January082026_id_file.csv"]
    key, refusal = cr.resolve_result_file(keys, "simulation")
    assert key is None
    assert "Out_South_Med_January082026_id_file.csv" in refusal


def test_no_result_file_at_all_refuses_by_name():
    key, refusal = cr.resolve_result_file(["surimi-cmsy/x/id_file.csv"], "historical")
    assert key is None
    assert "cmsy_output_historical.csv" in refusal.lower()


def test_ambiguous_named_files_refuse_rather_than_pick_one():
    keys = ["a/CMSY_output.csv", "b/CMSY_output.csv"]
    key, refusal = cr.resolve_result_file(keys, "simulation")
    assert key is None
    assert "2" in refusal and "CMSY_output.csv" in refusal


def test_asking_for_simulation_when_only_historical_exists_names_it():
    keys = ["x/CMSY_output_historical.csv"]
    key, refusal = cr.resolve_result_file(keys, "simulation")
    assert key is None
    assert "CMSY_output_historical.csv" in refusal
    assert "historical" in refusal


# ── row lookup ────────────────────────────────────────────────────────────

def test_read_stock_row_exact_match():
    row = cr.read_stock_row(HOGFISH_CSV, "Hogfish - Florida Keys / East Florida")
    assert row is not None
    assert row["SciName"] == "Lachnolaimus maximus"


def test_read_stock_row_missing_returns_none():
    assert cr.read_stock_row(HOGFISH_CSV, "Nephrops norvegicus - GSA06") is None


def test_known_stocks_lists_every_distinct_stock():
    two_row_csv = _csv_text([HOGFISH_ROW, _row(stock="Other Stock - Region")])
    assert cr.known_stocks(two_row_csv) == ["Hogfish - Florida Keys / East Florida", "Other Stock - Region"]


# ── extract_status ────────────────────────────────────────────────────────

def test_status_pairs_every_point_with_lcl_and_ucl():
    status = cr.extract_status(HOGFISH_ROW)
    bmsy = status["reference_points"]["Bmsy"]
    assert bmsy["point"] == 1200.0
    assert bmsy["lcl"] == 900.0
    assert bmsy["ucl"] == 1600.0


def test_status_never_asserts_the_bound_is_a_confidence_interval():
    """The bound must be labelled a posterior percentile. It may explain
    that the column names ('lcl'/'ucl') sound like a confidence interval,
    as long as it does not itself assert one - i.e. "confidence interval"
    may appear only alongside a disclaiming word (not/never/no stated)."""
    disclaimers = ("not", "never", "no stated", "does not")
    for triplet in (
        HOGFISH_ROW and cr.extract_status(HOGFISH_ROW)["biomass_ratio_to_bmsy"],
        cr.extract_status(HOGFISH_ROW)["fishing_pressure_ratio_to_fmsy"],
        cr.extract_status(HOGFISH_ROW)["reference_points"]["Bmsy"],
    ):
        meaning = triplet["bound_meaning"].lower()
        assert "percentile" in meaning
        if "confidence interval" in meaning:
            before = meaning.split("confidence interval")[0][-40:]
            assert any(d in before for d in disclaimers), meaning


def test_status_selects_bsm_when_btype_and_nab_gate_are_met():
    status = cr.extract_status(_row(btype="CPUE", n_bt=27))
    assert status["estimator"] == "BSM"


def test_status_falls_back_to_cmsy_when_abundance_years_are_below_nab():
    """The Q-E7-1 rule: btype=CPUE alone is not enough - AA_CMSY++.R falls
    back to CMSY when fewer than nab=5 usable abundance years exist in the
    assessment window."""
    status = cr.extract_status(_row(btype="CPUE", n_bt=3))
    assert status["estimator"] == "CMSY"
    assert "nab" in status["estimator_note"].lower() or "5" in status["estimator_note"]


def test_status_is_cmsy_when_btype_is_none():
    status = cr.extract_status(_row(btype="None", n_bt=0))
    assert status["estimator"] == "CMSY"


def test_status_biomass_ratio_reads_last_b_bmsy():
    status = cr.extract_status(HOGFISH_ROW)
    assert status["biomass_ratio_to_bmsy"]["point"] == 0.55
    assert status["fishing_pressure_ratio_to_fmsy"]["point"] == 0.88


# ── extract_trajectory ──────────────────────────────────────────────────

def test_trajectory_decodes_and_sorts_by_real_year_not_column_text():
    traj = cr.extract_trajectory(HOGFISH_ROW)
    years = [e["year"] for e in traj]
    assert years == sorted(years)
    assert years[0] == 1986
    assert years[-1] == 2012
    # c00 (2000) must land after c99 (1999) once decoded, never before c50
    assert years.index(1999) < years.index(2000)


def test_trajectory_drops_years_with_no_data_in_any_series():
    """Schema reaches 2030 (AA_CMSY++.R writes c50..c30 unconditionally) but
    a stock assessed only to 2012 leaves 2013-2030 entirely NA - those years
    must not appear as rows of nulls."""
    traj = cr.extract_trajectory(HOGFISH_ROW)
    assert all(e["year"] <= 2012 for e in traj)
    assert 2020 not in [e["year"] for e in traj]


def test_trajectory_rows_carry_catch_f_fmsy_and_b_together():
    traj = cr.extract_trajectory(HOGFISH_ROW)
    row_1986 = next(e for e in traj if e["year"] == 1986)
    assert row_1986["catch"] == 1000.0
    assert row_1986["f_fmsy"] == 0.5
    assert row_1986["b"] == 2000.0
