import logging

from surimi_model_mcp.simulations import parsers

# The CMSY++ run-output id file: Out_<region><date>_id_file.csv in the real
# bucket. This is the file _cmsy is meant to read.
CMSY_ID = (
    '"Group","Region","Subregion","Name","SciName","Stock","start.yr","end.yr","btype"\n'
    'NA,"Southeast Shelf","Florida Keys / East Florida","Hogfish",'
    '"Lachnolaimus maximus","Hogfish - Florida Keys / East Florida",1986,2012,"CPUE"\n'
)

# The CMSY++ *input* stock database: bare id_file.csv in the real bucket.
# Unrelated schema, unrelated stock (many stocks; this is just its first
# row). Every real CMSY run carries both files side by side.
STOCK_DB = (
    "Continent,Region,Subregion,Stock,Group,Name,ScientificName,SpecCode,Source,MinOfYear\n"
    'North America,"Gulf of Maine","Georges Bank",'
    '"Acadian redfish - Gulf of Maine / Georges Bank",NA,"Acadian redfish",'
    '"Sebastes fasciatus",123,"NOAA",1913\n'
)


def _fetch_by_out_prefix(key: str) -> tuple[str, bool]:
    """Stands in for a real bucket: the Out_-prefixed key is the run output,
    any other id_file.csv-suffixed key is the unrelated stock database."""
    name = key.rsplit("/", 1)[-1]
    return (CMSY_ID, False) if name.startswith("Out_") else (STOCK_DB, False)


def test_cmsy_metadata_comes_from_id_file():
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_January082026_id_file.csv",
            "surimi-cmsy/Simulations/abc/x_KOBE.jpg"]
    md = parsers.describe("surimi-cmsy", keys, lambda k: (CMSY_ID, False))
    assert md["stock"] == "Hogfish - Florida Keys / East Florida"
    assert md["species"] == "Lachnolaimus maximus"
    assert md["start_year"] == 1986
    assert md["end_year"] == 2012
    assert md["biomass_type"] == "CPUE"


def test_cmsy_metadata_is_honest_about_demo_content():
    """Surfacing the stock name is the point: every current CMSY run is the
    CMSY++ demo stock, and the chat must not imply otherwise."""
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_January082026_id_file.csv"]
    md = parsers.describe("surimi-cmsy", keys, lambda k: (CMSY_ID, False))
    assert "Florida" in md["stock"]


def test_cmsy_picks_the_out_prefixed_run_output_over_the_bare_stock_database():
    """Every real CMSY run carries both files. The bare id_file.csv is the
    CMSY++ input stock database (unrelated schema, unrelated stock) - if it
    were selected instead, the chat would silently report the wrong species
    for the run. Both have a Stock column, so this can only be caught by
    naming, not by schema."""
    keys = [
        "surimi-cmsy/Simulations/abc/Out_South_Med_id_file.csv",
        "surimi-cmsy/Simulations/abc/id_file.csv",
    ]
    md = parsers.describe("surimi-cmsy", keys, _fetch_by_out_prefix)
    assert md["stock"] == "Hogfish - Florida Keys / East Florida"
    assert md["species"] == "Lachnolaimus maximus"


def test_cmsy_picks_the_out_prefixed_file_even_when_listed_after_the_bare_one():
    """Proves the pick is not decided by list order. A naive first-match
    (`next(k for k in keys if k.endswith("id_file.csv"))`) only ever worked
    by the accident of store.find's ASCII sort putting 'O' (79) before 'i'
    (105); here the bare file is listed first, and the Out_ file must still
    win."""
    keys = [
        "surimi-cmsy/Simulations/abc/id_file.csv",
        "surimi-cmsy/Simulations/abc/Out_South_Med_id_file.csv",
    ]
    md = parsers.describe("surimi-cmsy", keys, _fetch_by_out_prefix)
    assert md["stock"] == "Hogfish - Florida Keys / East Florida"


def test_cmsy_ambiguous_id_files_refuses_and_names_the_candidates():
    """Two Out_-prefixed candidates (or, symmetrically, none) is exactly as
    unresolvable as the run-output/stock-database case - refuse rather than
    guess, and name what was seen so a human can act on it. Reporting the
    wrong stock is worse than reporting none."""
    keys = [
        "surimi-cmsy/Simulations/abc/Out_North_id_file.csv",
        "surimi-cmsy/Simulations/abc/Out_South_id_file.csv",
    ]
    md = parsers.describe("surimi-cmsy", keys, _fetch_by_out_prefix)
    assert md.get("parse_error")
    assert "Out_North_id_file.csv" in md["parse_error"]
    assert "Out_South_id_file.csv" in md["parse_error"]
    assert "stock" not in md


def test_market_metadata_counts_config_rows():
    csv = "species_code,life_stage,species_name\nANE,adult,Engraulis\nHKE,NA,Merluccius\n"
    keys = ["surimi-market/simulations/xyz/config/species.csv"]
    md = parsers.describe("surimi-market", keys, lambda k: (csv, False))
    assert md["config_files"] == ["species.csv"]
    assert md["rows"]["species.csv"] == 2


def test_market_metadata_never_reports_a_row_count_from_truncated_text():
    """prices.csv in the real bucket is 46,436 rows - far past store.get_text's
    65536-byte cap - so a truncated fetch must not silently yield a plausible
    but wrong row count. It must report None and say why."""
    csv = "a,b,c\n1,2,3\n4,5,6\n"  # looks like a complete, tiny, valid csv
    keys = ["surimi-market/simulations/xyz/config/prices.csv"]
    md = parsers.describe("surimi-market", keys, lambda k: (csv, True))
    assert md["rows"]["prices.csv"] is None
    assert "prices.csv" in md.get("truncated", [])


def test_cmsy_truncated_id_file_refuses_to_parse_a_row():
    """A read cut mid-field must not yield a plausible-looking wrong number:
    a truncated "1986" reads back as "19", which passes as a real year if
    nothing checks the sibling truncated flag. Refuse to parse entirely and
    say why, the same guarantee _market gives its row counts."""
    cut_mid_year = (
        '"Group","Region","Subregion","Name","SciName","Stock","start.yr","end.yr","btype"\n'
        'NA,"Southeast Shelf","Florida Keys / East Florida","Hogfish",'
        '"Lachnolaimus maximus","Hogfish - Florida Keys / East Florida",19'
    )
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_id_file.csv"]
    md = parsers.describe("surimi-cmsy", keys, lambda k: (cut_mid_year, True))
    assert "start_year" not in md
    assert "end_year" not in md
    assert "stock" not in md
    assert md.get("parse_error")
    assert md["truncated"] == ["Out_South_Med_id_file.csv"]


LOCAL_PRICES = (
    "tick,market,species,price,catch\n"
    "0,ES-CAT,ANE,3.5,120.0\n"
    "0,ES-CAT,HKE,6.2,40.0\n"
    "0,ES-MUR,ANE,3.4,80.0\n"
    "1,ES-CAT,ANE,3.6,130.0\n"
)


def test_market_metadata_reads_local_prices_as_a_price_beside_catch_series():
    """Q-E3-4: correlating catch/price needs local_prices.csv itself (price
    beside quantity, per market and species) - not just the /config/ file
    row counts _market already returns."""
    keys = [
        "surimi-market/simulations/xyz/config/species.csv",
        "surimi-market/simulations/xyz/local_prices.csv",
    ]

    def fetch(key):
        if key.endswith("species.csv"):
            return "species_code,life_stage\nANE,adult\n", False
        return LOCAL_PRICES, False

    md = parsers.describe("surimi-market", keys, fetch)
    rows = md["local_prices"]["rows"]
    assert {"tick": 0, "market": "ES-CAT", "species": "ANE",
            "price": "3.5", "catch": "120.0", "cold_start": True} in rows
    assert {"tick": 1, "market": "ES-CAT", "species": "ANE",
            "price": "3.6", "catch": "130.0", "cold_start": False} in rows
    assert md["local_prices"]["rows_read"] == 4
    assert "cold-start" in md["local_prices"]["note"].lower()
    assert "tick 0" in md["local_prices"]["note"]


def test_market_metadata_local_prices_never_reports_a_row_from_a_partial_cut_line():
    """A truncated fetch cuts mid-row - that partial row is dropped, not
    parsed into a plausible-looking-but-wrong value, exactly like _market's
    existing row-count truncation guarantee."""
    cut = LOCAL_PRICES + "1,ES-CAT,HK"  # trailing partial row, no newline
    keys = ["surimi-market/simulations/xyz/local_prices.csv"]
    md = parsers.describe("surimi-market", keys, lambda k: (cut, True))
    rows = md["local_prices"]["rows"]
    assert len(rows) == 4          # the partial 5th row is dropped
    assert md["local_prices"]["truncated"] is True


def test_market_metadata_logs_and_nulls_a_row_count_when_fetch_raises(caplog):
    """A wrong format-string arity or similar bug in the except branch would
    otherwise break silently - the row count still needs to null out, and
    the reason still needs to be visible in the log."""
    def boom(key):
        raise ValueError("connection reset")

    keys = ["surimi-market/simulations/xyz/config/species.csv"]
    with caplog.at_level(logging.WARNING):
        md = parsers.describe("surimi-market", keys, boom)

    assert md["rows"]["species.csv"] is None
    assert "species.csv" in caplog.text
    assert "connection reset" in caplog.text


def test_unknown_model_yields_empty_metadata_without_raising():
    keys = ["poseidon/simulations/xyz/out.parquet", "poseidon/simulations/xyz/log.txt"]
    md = parsers.describe("poseidon", keys, lambda k: ("", False))
    assert md == {}


def test_parser_failure_never_breaks_the_listing():
    """A malformed id_file must not take down describe_simulation."""
    keys = ["surimi-cmsy/Simulations/abc/Out_South_Med_id_file.csv"]
    md = parsers.describe("surimi-cmsy", keys, lambda k: ("not,a,valid\ncsv", False))
    assert md.get("parse_error")
