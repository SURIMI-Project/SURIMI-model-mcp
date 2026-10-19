"""Stock status stored inside a run's NetCDF, as (year, species) series.

A run's NetCDF can hold two variables without units or labels, `stock_status` and `exploitation`,
on the dimensions (year, species). SURIMI-CMSY's `server/StockAssessment.py` writes them as B/Bmsy
and F/Fmsy, with the historical and the simulated assessment merged into one series and no
uncertainty bounds. Every year before a stock's first assessed year holds a fill pair, B/Bmsy 2.0
and F/Fmsy 0.0, which is not an estimate.

The functions take an xarray Dataset, so the tool layer only has to find and open the file.
"""
from __future__ import annotations

import math
from typing import Any

#: The (B/Bmsy, F/Fmsy) pair stored for years before a stock's first assessed year.
FILL = (2.0, 0.0)

QUADRANTS = ("healthy", "overfished", "subject to overfishing",
             "overfished and subject to overfishing")

SEMANTICS_NOTE = (
    "stock_status and exploitation carry no units or labels in the file. "
    "SURIMI-CMSY's StockAssessment.py writes them as B/Bmsy and F/Fmsy, and "
    "they are reported that way here. Historical and simulated assessment "
    "years are merged, so the file cannot say which phase a year came from. "
    "There are no uncertainty bounds: give point values only. Years before a "
    "stock's first assessed year held a fill value (B/Bmsy 2.0, F/Fmsy 0.0) "
    "and are dropped."
)


def has_status(ds) -> bool:
    """Return True when the dataset has both `stock_status` and `exploitation`."""
    return "stock_status" in ds.variables and "exploitation" in ds.variables


def _label(code: str, stage: str) -> str:
    code, stage = str(code).strip(), str(stage).strip()
    return f"{code} {stage}" if stage else code


def _is_fill(b: float, f: float) -> bool:
    return b == FILL[0] and f == FILL[1]


def read(ds) -> dict[str, list[dict[str, Any]]]:
    """Return `{stock label: [{year, b_bmsy, f_fmsy}, ...]}` with the oldest year first.

    It drops the leading fill years and any year where either value is NaN, and it omits a stock
    with no year left. The label is the species code, followed by the life stage when there is one.
    """
    years = [int(y) for y in ds["year"].values]
    codes = ds["species_code"].values
    stages = (ds["species_life_stage"].values if "species_life_stage" in ds.variables
              else [""] * len(codes))
    b = ds["stock_status"].transpose("year", "species").values
    f = ds["exploitation"].transpose("year", "species").values
    out: dict[str, list[dict[str, Any]]] = {}
    for j, code in enumerate(codes):
        rows = []
        started = False
        for i, year in enumerate(years):
            bv, fv = float(b[i, j]), float(f[i, j])
            if math.isnan(bv) or math.isnan(fv):
                continue
            if not started and _is_fill(bv, fv):
                continue
            started = True
            rows.append({"year": year, "b_bmsy": bv, "f_fmsy": fv})
        if rows:
            out[_label(code, stages[j])] = rows
    return out


def kobe(b_bmsy: float, f_fmsy: float) -> str:
    """Return the Kobe quadrant: overfished when B/Bmsy is below 1, overfishing when F/Fmsy is above 1."""
    over = b_bmsy < 1.0
    fishing = f_fmsy > 1.0
    if over and fishing:
        return QUADRANTS[3]
    if over:
        return QUADRANTS[1]
    if fishing:
        return QUADRANTS[2]
    return QUADRANTS[0]


def _at(rows: list[dict[str, Any]], year: int) -> dict[str, Any] | None:
    return next((r for r in rows if r["year"] == year), None)


def overview(series: dict[str, list[dict[str, Any]]], year: int | None = None) -> dict[str, Any]:
    """Return the Kobe quadrant of every stock in `year`, or in the latest year when `year` is None."""
    if year is None:
        year = max(r["year"] for rows in series.values() for r in rows)
    stocks = []
    for label, rows in sorted(series.items()):
        r = _at(rows, year)
        if r:
            stocks.append({"stock": label, "b_bmsy": round(r["b_bmsy"], 4),
                           "f_fmsy": round(r["f_fmsy"], 4),
                           "kobe": kobe(r["b_bmsy"], r["f_fmsy"])})
    counts = {q: sum(s["kobe"] == q for s in stocks) for q in QUADRANTS}
    return {"year": year, "n_stocks": len(stocks), "counts": counts, "stocks": stocks}


def resolve(series: dict[str, list[dict[str, Any]]], stock: str) -> str | None:
    """Return the series label for `stock`, matching the label first and then a unique species code."""
    want = stock.strip().lower()
    for label in series:
        if label.lower() == want:
            return label
    hits = [label for label in series if label.split(" ")[0].lower() == want]
    return hits[0] if len(hits) == 1 else None
