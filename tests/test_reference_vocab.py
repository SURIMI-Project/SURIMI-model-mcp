"""reference/vocab.py: loading, indexing, and caching the five CSV
vocabularies. Network-free throughout - store.get_text is monkeypatched with
small fixture CSVs shaped exactly like the live headers (verified against
the real s3://project-surimi files on 2026-08-07)."""
import pytest

from surimi_model_mcp.reference import vocab
from surimi_model_mcp.simulations import store

ASFIS_CSV = (
    "ISSCAAP_Group,Taxonomic_Code,Alpha3_Code,Scientific_Name,English_name,"
    "French_name,Spanish_name,Arabic_name,Chinese_name,Russian_name,Author,"
    "Family,Order or higher taxa,FishStat_Data\n"
    "34,185001102202,ANK,Lophius budegassa,Blackbellied angler,"
    "Baudroie rousse,Rape negro,,,,Spinola 1807,LOPHIIDAE,LOPHIIFORMES,YES\n"
    "13,121003106603,ESP,Setipinna phasa,Gangetic hairfin anchovy,,,,,,"
    "(Hamilton 1822),ENGRAULIDAE,CLUPEIFORMES,NO\n"
)
# Real ASFIS_sp_2024.csv carries a UTF-8 BOM + trailing space on the first
# header ("﻿ISSCAAP_Group "), which store.get_text's plain "utf-8"
# decode does not strip.
ASFIS_CSV_WITH_BOM = "﻿" + ASFIS_CSV.replace("ISSCAAP_Group,", "ISSCAAP_Group ,", 1)

ISSCFG_CSV = (
    "FISHERY_CODE,FISHERY_NAME,FISHERY_CATEGORY_NAME,FISHERY_TYPE_NAME,"
    "FISHERY_TYPE_DESCRIPTION,GEAR_CODE,GEAR_NAME,GEAR_GROUP_CODE,GEAR_GROUP_NAME\n"
    '"DL.FR[IS]TR","Medium-scale drifting longlines fishery","Longline fisheries",'
    '"Medium-scale (NJAs and/or ABNJ)","targets tropical tunas","DL",'
    '"Drifting longlines","09","Hooks and lines"\n'
    '"PS.ESP.A","Purse seine A","Surrounding nets","Large-scale","desc A","PS",'
    '"Purse seine","01","Surrounding nets"\n'
    '"PS.ESP.B","Purse seine B","Surrounding nets","Large-scale","desc B","PS",'
    '"Purse seine","01","Surrounding nets"\n'
)

WORMS_CSV = (
    "scientificName,kingdom,phylum,class,order,family,genus,taxonRank,FAO_Code\n"
    "Lophius budegassa,Animalia,Chordata,Teleostei,Lophiiformes,Lophiidae,"
    "Lophius,Species,ANK\n"
)

FISHBASE_CSV = (
    "SpecCode,Genus,Species,FBname\n"
    "5094,Lophius,budegassa,Blackbellied angler\n"
)

ISO3166_CSV = (
    "name,alpha-2,alpha-3,region,sub-region\n"
    "Spain,ES,ESP,Europe,Southern Europe\n"
)

_CONTENT = {
    vocab.ASFIS_KEY: ASFIS_CSV,
    vocab.ISSCFG_KEY: ISSCFG_CSV,
    vocab.WORMS_KEY: WORMS_CSV,
    vocab.FISHBASE_KEY: FISHBASE_CSV,
    vocab.ISO3166_KEY: ISO3166_CSV,
}


@pytest.fixture(autouse=True)
def _isolated_vocab_cache():
    """vocab._cache/_errors are life-of-process module caches - without this
    a later test's "no refetch" assertion would pass for the wrong reason."""
    vocab.reset_cache()
    yield
    vocab.reset_cache()


def _install_fake_get_text(monkeypatch, content=None, calls=None):
    content = content or _CONTENT
    calls = calls if calls is not None else {"n": 0}

    def get_text(key, max_bytes=65536):
        calls["n"] += 1
        return content[key], False

    monkeypatch.setattr(store, "get_text", get_text)
    return calls


def test_asfis_hit_by_code(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.asfis()
    assert err is None
    row = idx["by_code"]["ANK"]
    assert row["Scientific_Name"] == "Lophius budegassa"
    assert row["Family"] == "LOPHIIDAE"


def test_asfis_strips_bom_and_header_whitespace(monkeypatch):
    _install_fake_get_text(monkeypatch, content={**_CONTENT, vocab.ASFIS_KEY: ASFIS_CSV_WITH_BOM})
    idx, err = vocab.asfis()
    assert err is None
    assert "ISSCAAP_Group" in idx["rows"][0]
    assert idx["by_code"]["ANK"]["Scientific_Name"] == "Lophius budegassa"


def test_isscfg_hit_by_fishery_code(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.isscfg()
    assert err is None
    row = idx["by_fishery_code"]["DL.FR[IS]TR"]
    assert row["FISHERY_NAME"] == "Medium-scale drifting longlines fishery"


def test_isscfg_gear_code_can_have_multiple_fisheries(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.isscfg()
    assert err is None
    assert len(idx["by_gear_code"]["PS"]) == 2


def test_isscfg_unknown_code_is_a_clean_miss(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.isscfg()
    assert err is None
    assert "DTS" not in idx["by_fishery_code"]
    assert "DTS" not in idx["by_gear_code"]


def test_worms_hit_by_fao_code(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.worms()
    assert err is None
    assert idx["by_fao_code"]["ANK"]["scientificName"] == "Lophius budegassa"


def test_fishbase_hit_by_binomial(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.fishbase()
    assert err is None
    assert idx["by_binomial"]["lophius budegassa"]["FBname"] == "Blackbellied angler"


def test_iso3166_hit_returns_spain_europe_southern_europe(monkeypatch):
    _install_fake_get_text(monkeypatch)
    idx, err = vocab.iso3166()
    assert err is None
    row = idx["by_code"]["ESP"]
    assert row["name"] == "Spain"
    assert row["region"] == "Europe"
    assert row["sub-region"] == "Southern Europe"


def test_cache_serves_second_call_without_refetch(monkeypatch):
    calls = _install_fake_get_text(monkeypatch)
    vocab.asfis()
    assert calls["n"] == 1
    vocab.asfis()
    assert calls["n"] == 1   # cached: no re-fetch


def test_a_load_failure_degrades_only_that_source(monkeypatch):
    """A malformed WoRMS file must not stop ASFIS/ISSCFG/FishBase/ISO3166
    from loading - each source is cached and can fail independently."""
    broken = {**_CONTENT, vocab.WORMS_KEY: "not,the,right,header\n1,2,3,4\n"}
    _install_fake_get_text(monkeypatch, content=broken)

    widx, werr = vocab.worms()
    assert widx is None
    assert werr is not None and "WoRMS" in werr

    aidx, aerr = vocab.asfis()
    assert aerr is None
    assert aidx["by_code"]["ANK"]["Scientific_Name"] == "Lophius budegassa"


def test_deterministic_load_failure_is_cached_and_not_refetched(monkeypatch):
    broken = {**_CONTENT, vocab.WORMS_KEY: "not,the,right,header\n1,2,3,4\n"}
    calls = _install_fake_get_text(monkeypatch, content=broken)
    vocab.worms()
    assert calls["n"] == 1
    idx, err = vocab.worms()
    assert idx is None and err is not None
    assert calls["n"] == 1   # cached: the deterministic failure is not retried


def test_transient_store_error_is_not_cached_and_is_retried(monkeypatch):
    state = {"fail": True}
    calls = {"n": 0}

    def get_text(key, max_bytes=65536):
        calls["n"] += 1
        if state["fail"]:
            raise store.SimStoreError("boom")
        return _CONTENT[key], False

    monkeypatch.setattr(store, "get_text", get_text)

    with pytest.raises(store.SimStoreError):
        vocab.asfis()
    assert calls["n"] == 1

    state["fail"] = False
    idx, err = vocab.asfis()
    assert err is None
    assert idx["by_code"]["ANK"]["Scientific_Name"] == "Lophius budegassa"
    assert calls["n"] == 2   # retried - the transient failure was not cached


def test_truncated_read_refuses_to_build_a_partial_index(monkeypatch):
    def get_text(key, max_bytes=65536):
        return _CONTENT[vocab.ASFIS_KEY][:20], True   # truncated

    monkeypatch.setattr(store, "get_text", get_text)
    idx, err = vocab.asfis()
    assert idx is None
    assert "truncated" in err.lower()


def test_vocabulary_keys_are_the_ones_in_the_bucket():
    """The five reference files moved from surimi-ecopath/config/ to ecopath/ in the
    naming migration; the old path answers Forbidden on MinIO, which looked like an
    expired token and cost the chat rounds on every species lookup."""
    for key in (vocab.ASFIS_KEY, vocab.ISSCFG_KEY, vocab.WORMS_KEY,
                vocab.FISHBASE_KEY, vocab.ISO3166_KEY):
        assert key.startswith("ecopath/"), key
        assert "surimi-ecopath" not in key, key
    assert vocab.ASFIS_KEY == "ecopath/ASFIS_sp_2024.csv"
