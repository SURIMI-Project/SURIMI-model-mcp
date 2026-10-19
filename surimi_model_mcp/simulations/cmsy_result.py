"""Read the wide CMSY++ result table that `parsers._cmsy` does not use.

`parsers._cmsy` takes a few run-identification columns from a bundle's `Out_<region><date>_id_file.csv`.
This module reads the full CMSY++ result row, which has r, k, MSY, Bmsy and Fmsy for the CMSY and BSM
fits, the current B/Bmsy and F/Fmsy with their 2.5th and 97.5th posterior percentiles, and three
year-indexed blocks (catch, F/Fmsy and B).

The year columns carry two-digit suffixes that wrap, `c50` to `c99` for 1950-1999 and `c00` to `c30` for
2000-2030. `decode_cmsy_year` turns each suffix into a year, and callers must decode before sorting,
because sorting the column names puts `c00` before `c50`.

A run can hold one result file per phase, `CMSY_output_historical.csv` for `historical` and
`CMSY_output.csv` for `simulation`. `resolve_result_file` finds the file for a phase. A run that has
only the unlabelled `Out_..._id_file.csv` gets a refusal that names the file, because the table does
not say which phase it belongs to.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

PHASES = ("historical", "simulation")

#: Minimum number of years with abundance data for BSM, from `nab <- 5` in AA_CMSY++.R.
#: `estimator_for` applies it together with `btype`, as the gate at AA_CMSY++.R:1135 does.
NAB = 5

_PHASE_FILENAMES = {
    "historical": "cmsy_output_historical.csv",
    "simulation": "cmsy_output.csv",
}

_YEAR_COLUMN = re.compile(r"^(c|F\.Fmsy|B)(\d{2})$")


def _basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def decode_cmsy_year(suffix: str) -> int:
    """Return the year for a two-digit column suffix: 50 and above is 19xx, below 50 is 20xx."""
    n = int(suffix)
    return 1900 + n if n >= 50 else 2000 + n


def resolve_result_file(keys: list[str], phase: str) -> tuple[str | None, str | None]:
    """Return `(key, None)` for the result file of `phase`, or `(None, refusal)` with the reason.

    It never guesses a phase. A run with no `CMSY_output*.csv` and one `Out_..._id_file.csv` gets a refusal
    that names that file.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    wanted = _PHASE_FILENAMES[phase]
    named = [k for k in keys if _basename(k).lower() == wanted]
    if len(named) == 1:
        return named[0], None
    if len(named) > 1:
        names = ", ".join(sorted(_basename(k) for k in named))
        return None, (
            f"{len(named)} artifacts in this run are named {wanted!r}: {names} - "
            "refusing to guess which one is authoritative"
        )

    other_phase = "simulation" if phase == "historical" else "historical"
    other_named = [k for k in keys if _basename(k).lower() == _PHASE_FILENAMES[other_phase]]
    legacy = sorted(
        k for k in keys
        if _basename(k).startswith("Out_") and _basename(k).endswith("id_file.csv")
    )

    if legacy and not other_named:
        if len(legacy) > 1:
            names = ", ".join(_basename(k) for k in legacy)
            return None, (
                f"no {wanted!r} in this run, and {len(legacy)} legacy result-table "
                f"candidates instead of one: {names} - refusing to guess"
            )
        return None, (
            f"this run carries no {_PHASE_FILENAMES['historical']!r}/"
            f"{_PHASE_FILENAMES['simulation']!r} split at all; its only CMSY++ result "
            f"table is {_basename(legacy[0])!r}. Call model_get_artifact with that name "
            f"directly instead of phase={phase!r} - do not assume it is the historical "
            "or the simulated run"
        )

    if other_named:
        return None, (
            f"no {wanted!r} in this run; it carries {_basename(other_named[0])!r} "
            f"(phase={other_phase!r}) instead"
        )
    return None, f"no {wanted!r} artifact in this run"


def read_stock_row(text: str, stock: str) -> dict[str, str] | None:
    """Return the row whose `Stock` column equals `stock`, or None when no row matches."""
    for row in csv.DictReader(io.StringIO(text)):
        if (row.get("Stock") or "").strip() == stock:
            return row
    return None


def known_stocks(text: str) -> list[str]:
    """Return the sorted, non-empty `Stock` values of the table."""
    return sorted({
        (row.get("Stock") or "").strip()
        for row in csv.DictReader(io.StringIO(text))
        if (row.get("Stock") or "").strip()
    })


def _num(row: dict[str, str], col: str) -> float | None:
    v = row.get(col)
    if v is None:
        return None
    v = v.strip()
    if v == "" or v.upper() == "NA":
        return None
    try:
        return float(v)
    except ValueError:
        return None


#: Meaning of every lcl/ucl pair. AA_CMSY++.R derives them as the 2.5th and 97.5th quantiles of the
#: MCMC posterior (for example `r.quant.cmsy[2:3]`) and states no confidence level.
#: Each bound this module returns carries this text.
_BOUND_MEANING = (
    "2.5th and 97.5th percentile of the MCMC posterior - the CMSY++ column "
    "names call this lcl/ucl ('confidence limit'), but AA_CMSY++.R never "
    "states what confidence level it represents; report it as a posterior "
    "percentile band, not as a confidence interval"
)


def _bound(row: dict[str, str], point: str, lcl: str, ucl: str) -> dict[str, Any]:
    return {
        "point": _num(row, point),
        "lcl": _num(row, lcl),
        "ucl": _num(row, ucl),
        "bound_meaning": _BOUND_MEANING,
    }


def estimator_for(row: dict[str, str]) -> tuple[str, str]:
    """Return `("BSM" or "CMSY", reason)` for the estimator that the gate at AA_CMSY++.R:1135 selects.

    BSM needs `btype` other than "None" and at least `NAB` years in `N bt`, and CMSY is used otherwise.
    The result table has no `force.cmsy` column, so the BSM reason states that the setting cannot be
    checked from this row.
    """
    btype = (row.get("btype") or "").strip()
    n_bt = _num(row, "N bt")
    if btype and btype != "None" and n_bt is not None and n_bt >= NAB:
        return "BSM", (
            f"btype={btype!r} and N bt={int(n_bt)} (>= AA_CMSY++.R's nab={NAB}) meet the "
            "gate at AA_CMSY++.R:1135, so the Bayesian Schaefer Model (catch + abundance "
            "index) is used - UNLESS this run's id_file.csv sets force.cmsy=TRUE, which "
            "is not a column in this result table and cannot be checked from it alone"
        )
    reason = (
        f"btype={btype!r}" if btype == "None" else
        f"btype={btype!r} but only "
        f"{int(n_bt) if n_bt is not None else 'an unreported number of'} usable "
        f"abundance-index year(s), below AA_CMSY++.R's nab={NAB} threshold"
    )
    return "CMSY", f"{reason}, so the catch-only CMSY estimator is used (AA_CMSY++.R:1135)"


def extract_status(row: dict[str, str]) -> dict[str, Any]:
    """Return the current Kobe status of one result row, with every point paired with its bounds.

    The dictionary holds the estimator, the current B/Bmsy and F/Fmsy, and the Bmsy, Fmsy and MSY
    reference points.
    """
    estimator, estimator_note = estimator_for(row)
    return {
        "btype": row.get("btype"),
        "n_abundance_years": _num(row, "N bt"),
        "estimator": estimator,
        "estimator_note": estimator_note,
        "biomass_ratio_to_bmsy": _bound(row, "last.B_Bmsy", "lcl.last.B_Bmsy", "ucl.last.B_Bmsy"),
        "fishing_pressure_ratio_to_fmsy": _bound(
            row, "last.F_Fmsy", "lcl.last.F_Fmsy", "ucl.last.F_Fmsy"),
        "reference_points": {
            "Bmsy": _bound(row, "Bmsy", "lcl.Bmsy", "ucl.Bmsy"),
            "Fmsy": _bound(row, "F_msy", "lcl.F_msy", "ucl.F_msy"),
            "MSY": _bound(row, "MSY", "lcl.MSY", "ucl.MSY"),
        },
    }


def extract_trajectory(row: dict[str, str]) -> list[dict[str, Any]]:
    """Return `[{"year", "catch", "f_fmsy", "b"}, ...]` from one result row, oldest year first.

    It reads the `c`, `F.Fmsy` and `B` column blocks and decodes each suffix with `decode_cmsy_year`
    before sorting. Years where all three series are empty are left out.
    """
    years: dict[int, dict[str, Any]] = {}
    for col, _ in row.items():
        m = _YEAR_COLUMN.match(col)
        if not m:
            continue
        series, suffix = m.groups()
        year = decode_cmsy_year(suffix)
        entry = years.setdefault(year, {"year": year, "catch": None, "f_fmsy": None, "b": None})
        value = _num(row, col)
        if series == "c":
            entry["catch"] = value
        elif series == "F.Fmsy":
            entry["f_fmsy"] = value
        else:  # "B"
            entry["b"] = value

    return [
        entry for _, entry in sorted(years.items())
        if entry["catch"] is not None or entry["f_fmsy"] is not None or entry["b"] is not None
    ]
