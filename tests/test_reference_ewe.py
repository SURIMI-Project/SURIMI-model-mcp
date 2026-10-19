"""reference/ewe.py: the regex-based .eiixml table/group parser. Network-free
- store.find/get_text are monkeypatched. The fixture below reproduces the
shape found in the live GSA0607EwENBS.eiixml (verified 2026-08-07): a
declared Num that does NOT match the actual <Row> count, and a large table
(EcopathDietComp, real file: 13,924 declared rows) given ZERO <Row> content
here - proving the parser never requires materialising it.
"""
import pytest

from surimi_model_mcp.reference import ewe
from surimi_model_mcp.simulations import store

MODEL_KEY = "ecopath/TestModel.eiixml"
EIIXML_CONTENT = (
    '<?xml version="1.0"?>\n'
    '<EwEModel Name="TestModel" DBVersion="6.700018" EwEVersion="6.7.0.19431">\n'
    '  <Table Name="Auxillary" Columns="DBID:System.Int64,Remark:System.String" Num="5">\n'
    '    <Row>1,"hello"</Row>\n'
    '  </Table>\n'
    # A large table declared with a big row count but NO <Row> content at
    # all (self-closing, exactly like the live file's empty tables) - if
    # the parser ever tried to materialise this it would find nothing to
    # read anyway, proving the design never depends on it being there.
    '  <Table Name="EcopathDietComp" Columns="PredID:System.Int64,PreyID:System.Int64" Num="13924" />\n'
    # declared Num (2596 in the live file; 2596 here too, for parity) does
    # NOT match the number of <Row> elements actually present (2) - the
    # same gap found in the live bucket.
    '  <Table Name="EcopathGroup" '
    'Columns="GroupID:System.Int64,GroupName:System.String,PoolColor:System.String" Num="2596">\n'
    "    <Row>1,By-catch,00000000</Row>\n"
    '    <Row>2,"Phyto, plankton",00112233</Row>\n'
    "  </Table>\n"
    "</EwEModel>\n"
)


@pytest.fixture(autouse=True)
def _isolated_ewe_cache():
    ewe.reset_cache()
    yield
    ewe.reset_cache()


def _install_find(monkeypatch, objects=None, failures=None):
    objects = objects if objects is not None else [
        {"key": MODEL_KEY, "size": len(EIIXML_CONTENT), "last_modified": "2026-01-01T00:00:00"},
        {"key": "ecopath/ASFIS_sp_2024.csv", "size": 100,
         "last_modified": "2026-01-01T00:00:00"},
    ]
    monkeypatch.setattr(store, "find", lambda prefix: (objects, failures or []))


def _install_get_text(monkeypatch, text=EIIXML_CONTENT, truncated=False, calls=None):
    calls = calls if calls is not None else {"n": 0}

    def get_text(key, max_bytes=65536):
        calls["n"] += 1
        assert key == MODEL_KEY
        return text, truncated

    monkeypatch.setattr(store, "get_text", get_text)
    return calls


def test_list_models_filters_to_eiixml_only(monkeypatch):
    _install_find(monkeypatch)
    models, err = ewe.list_models()
    assert err is None
    assert [m["name"] for m in models] == ["TestModel"]
    assert models[0]["file"] == "TestModel.eiixml"


def test_describe_model_returns_table_inventory_without_materialising_large_table(monkeypatch):
    _install_find(monkeypatch)
    _install_get_text(monkeypatch)
    parsed, err = ewe.describe_model("TestModel")
    assert err is None
    assert parsed["model_name"] == "TestModel"
    assert parsed["db_version"] == "6.700018"
    assert parsed["ewe_version"] == "6.7.0.19431"
    assert parsed["table_count"] == 3
    by_name = {t["name"]: t for t in parsed["tables"]}
    assert by_name["EcopathDietComp"]["row_count"] == 13924   # declared, never scanned
    assert by_name["Auxillary"]["row_count"] == 5


def test_describe_model_is_case_insensitive_and_matches_filename_too(monkeypatch):
    _install_find(monkeypatch)
    _install_get_text(monkeypatch)
    parsed, err = ewe.describe_model("testmodel")
    assert err is None and parsed["model_name"] == "TestModel"
    parsed2, err2 = ewe.describe_model("TestModel.eiixml")
    assert err2 is None and parsed2["model_name"] == "TestModel"


def test_functional_group_names_are_read_and_declared_mismatch_is_noted(monkeypatch):
    _install_find(monkeypatch)
    _install_get_text(monkeypatch)
    parsed, err = ewe.describe_model("TestModel")
    assert err is None
    fg = parsed["functional_groups"]
    assert fg["declared_row_count"] == 2596
    assert fg["names_found"] == 2
    # The embedded comma inside a quoted GroupName must survive intact -
    # proves Row content is CSV-parsed, not naively split on ",".
    assert fg["names"] == ["By-catch", "Phyto, plankton"]
    assert "note" in fg
    assert "2596" in fg["note"] and "2" in fg["note"]


def test_unknown_model_is_a_clean_miss_naming_whats_available(monkeypatch):
    _install_find(monkeypatch)
    parsed, err = ewe.describe_model("NoSuchModel")
    assert parsed is None
    assert "TestModel" in err


def test_cache_serves_second_describe_without_refetch(monkeypatch):
    _install_find(monkeypatch)
    calls = _install_get_text(monkeypatch)
    ewe.describe_model("TestModel")
    assert calls["n"] == 1
    ewe.describe_model("TestModel")
    assert calls["n"] == 1


def test_list_models_cache_avoids_a_second_find_call(monkeypatch):
    find_calls = {"n": 0}

    def find(prefix):
        find_calls["n"] += 1
        return [{"key": MODEL_KEY, "size": 10, "last_modified": None}], []

    monkeypatch.setattr(store, "find", find)
    ewe.list_models()
    assert find_calls["n"] == 1
    ewe.list_models()
    assert find_calls["n"] == 1


def test_truncated_read_is_flagged_not_hidden(monkeypatch):
    _install_find(monkeypatch)
    _install_get_text(monkeypatch, truncated=True)
    parsed, err = ewe.describe_model("TestModel")
    assert err is None
    assert parsed["truncated"] is True
    assert "truncated_note" in parsed


def test_transient_store_error_from_find_propagates_uncached(monkeypatch):
    state = {"fail": True}
    calls = {"n": 0}

    def find(prefix):
        calls["n"] += 1
        if state["fail"]:
            raise store.SimStoreError("boom")
        return [{"key": MODEL_KEY, "size": 10, "last_modified": None}], []

    monkeypatch.setattr(store, "find", find)

    with pytest.raises(store.SimStoreError):
        ewe.list_models()
    assert calls["n"] == 1

    state["fail"] = False
    models, err = ewe.list_models()
    assert err is None and models
    assert calls["n"] == 2   # retried - not cached
