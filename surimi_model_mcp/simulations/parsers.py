"""Make a run legible without downloading it.

`describe` picks one small parser per model, `_cmsy` for `surimi-cmsy` and `_market` for `surimi-market`.
A model without a parser gets `{}`, so its runs are still listed.

`fetch_text` has the same shape as `store.get_text` and returns `(content, truncated)`. The parsers never
take a value from truncated text, because a read cut inside a field can still be valid CSV with a wrong
value, such as "1986" read as "19".

- `_market` reports the row count of a truncated config file as None and lists the file name under
  `truncated`.
- `_cmsy` refuses a truncated `id_file.csv` and returns a `parse_error`.

`_cmsy` also refuses to guess which file to read. A run can hold several keys that end in `id_file.csv`,
for example `Out_<region><date>_id_file.csv` and a bare `id_file.csv`. The parser relies on the `Out_`
prefix and reads the run only when exactly one candidate starts with it.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

FetchText = Callable[[str], tuple[str, bool]]


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _basename(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def _select_cmsy_id_file(keys: list[str]) -> tuple[str | None, dict[str, Any] | None]:
    """Return the key of the run's own `id_file.csv`, or a refusal when the choice is ambiguous.

    The result is `(key, None)` for one pick, `(None, {})` when no file matches, and
    `(None, {"parse_error": ...})` when the `Out_` prefix does not leave exactly one candidate.
    """
    candidates = [k for k in keys if _basename(k).endswith("id_file.csv")]
    if not candidates:
        return None, {}

    out_candidates = [k for k in candidates if _basename(k).startswith("Out_")]
    if len(out_candidates) == 1:
        return out_candidates[0], None

    names = sorted(_basename(k) for k in candidates)
    return None, {
        "parse_error": (
            "cannot tell which id_file.csv is this run's own output: "
            f"{len(out_candidates)} of {len(candidates)} candidate(s) start "
            f'with "Out_" (need exactly 1) - candidates: {", ".join(names)}'
        ),
    }


def _cmsy(keys: list[str], fetch: FetchText) -> dict[str, Any]:
    idk, refusal = _select_cmsy_id_file(keys)
    if refusal is not None:
        return refusal
    if idk is None:
        return {}
    name = _basename(idk)
    text, truncated = fetch(idk)
    if truncated:
        # A cut inside a field can still parse as valid CSV, so no row is read.
        logger.warning(
            "%s was truncated by the fetch cap; refusing to parse a row "
            "from it", idk,
        )
        return {
            "parse_error": (
                f"{name} was truncated at the read cap; refusing to "
                "report values that may be cut mid-field"
            ),
            "truncated": [name],
        }
    rows = _rows(text)
    if not rows:
        return {"parse_error": f"{name} has no data rows"}
    r = rows[0]

    def num(field: str) -> int | None:
        try:
            return int(float(r.get(field, "")))
        except (TypeError, ValueError):
            return None

    out = {
        "stock": r.get("Stock"),
        "species": r.get("SciName"),
        "common_name": r.get("Name"),
        "region": r.get("Region"),
        "start_year": num("start.yr"),
        "end_year": num("end.yr"),
        "biomass_type": r.get("btype"),
    }
    if not out["stock"]:
        return {"parse_error": f"{name} missing a Stock column"}
    return out


#: `local_prices.csv` has the columns tick, market, species, price and catch. Tick 0 is the
#: model's initialisation step, and each row carries a `cold_start` flag that marks it.
_COLD_START_TICK = 0


def _parse_local_prices(text: str, truncated: bool) -> list[dict[str, Any]]:
    """Return the rows of `local_prices.csv` in file order, with the incomplete last line dropped.

    A truncated read cuts only the last line, so every row kept here was read whole. Rows without an
    integer `tick` are skipped.
    """
    if truncated and text and not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0] + "\n"
    rows = []
    for r in _rows(text):
        try:
            tick = int(r["tick"])
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({
            "tick": tick,
            "market": r.get("market"),
            "species": r.get("species"),
            "price": r.get("price"),
            "catch": r.get("catch"),
            "cold_start": tick == _COLD_START_TICK,
        })
    return rows


def _market(keys: list[str], fetch: FetchText) -> dict[str, Any]:
    cfg = [k for k in keys if "/config/" in k and k.endswith(".csv")]
    local_prices_key = next(
        (k for k in keys if _basename(k) == "local_prices.csv"), None)
    if not cfg and local_prices_key is None:
        return {}
    rows: dict[str, int | None] = {}
    truncated_files: list[str] = []
    for k in cfg:
        name = k.rsplit("/", 1)[-1]
        try:
            text, truncated = fetch(k)
            if truncated:
                # The text is only a prefix of the file, so its rows are not counted.
                logger.warning(
                    "%s was truncated by the fetch cap; not reporting a row "
                    "count derived from a partial read", k,
                )
                rows[name] = None
                truncated_files.append(name)
            else:
                rows[name] = len(_rows(text))
        except Exception as exc:
            logger.warning("could not parse %s: %s: %s", k, type(exc).__name__, exc)
            rows[name] = None
    out: dict[str, Any] = {
        "config_files": sorted(r.rsplit("/", 1)[-1] for r in cfg),
        "rows": rows,
    }
    if truncated_files:
        out["truncated"] = sorted(truncated_files)
    if local_prices_key is not None:
        try:
            text, ltruncated = fetch(local_prices_key)
            price_rows = _parse_local_prices(text, ltruncated)
            out["local_prices"] = {
                "rows": price_rows,
                "rows_read": len(price_rows),
                "truncated": ltruncated,
                "note": (
                    "tick 0 is the market model's initialisation step, a "
                    "cold-start artefact, not an observed price/quantity pair"
                ),
            }
        except Exception as exc:
            logger.warning(
                "could not parse %s: %s: %s",
                local_prices_key, type(exc).__name__, exc,
            )
    return out


PARSERS: dict[str, Callable[[list[str], FetchText], dict[str, Any]]] = {
    "surimi-cmsy": _cmsy,
    "surimi-market": _market,
}


def describe(model: str, keys: list[str], fetch_text: FetchText) -> dict[str, Any]:
    """Return the metadata that the parser for `model` reads from the run's `keys`, or `{}` without one.

    An exception in a parser becomes a `parse_error` entry and is not raised.
    """
    fn = PARSERS.get(model)
    if not fn:
        return {}
    try:
        return fn(keys, fetch_text)
    except Exception as exc:
        return {"parse_error": f"{type(exc).__name__}: {exc}"}
