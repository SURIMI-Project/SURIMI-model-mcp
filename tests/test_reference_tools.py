"""reference_tools.py: the four MCP-facing reference-layer tools. Network-free
- store.get_text/find are monkeypatched with small fixture files shaped like
the live vocabularies (verified against s3://project-surimi on 2026-08-07).
"""
import pytest

from surimi_model_mcp.tools import lookups as rt
from surimi_model_mcp.reference import contract, ewe, vocab
from surimi_model_mcp.simulations import store

ASFIS_CSV = (
    "ISSCAAP_Group,Taxonomic_Code,Alpha3_Code,Scientific_Name,English_name,"
    "French_name,Spanish_name,Arabic_name,Chinese_name,Russian_name,Author,"
    "Family,Order or higher taxa,FishStat_Data\n"
    "34,185001102202,ANK,Lophius budegassa,Blackbellied angler,"
    "Baudroie rousse,Rape negro,,,,Spinola 1807,LOPHIIDAE,LOPHIIFORMES,YES\n"
    # A real, live collision (verified 2026-08-07): ESP is BOTH a valid FAO
    # species code (Gangetic hairfin anchovy) and Spain's ISO 3166 code.
    "13,121003106603,ESP,Setipinna phasa,Gangetic hairfin anchovy,,,,,,"
    "(Hamilton 1822),ENGRAULIDAE,CLUPEIFORMES,NO\n"
    "34,185001102203,ANH,Lophius americanus,American angler,,,,,,"
    "Valenciennes 1837,LOPHIIDAE,LOPHIIFORMES,YES\n"
    # The live ranking trap (verified against the deployed server 2026-08-15):
    # "hake" is a substring of kus-CHAKE-witschi, and this row sorts BEFORE the
    # real hakes in the source file, so first-N-in-file-order handed back a
    # freshwater Aral Sea cyprinid as the top hit for "hake" while European
    # hake never appeared at all. Order here mirrors the live file on purpose.
    "13,,OEK,Capoetobrama kuschakewitschi,Sharpray,,,,,,"
    "(Kessler 1872),CYPRINIDAE,CYPRINIFORMES,NO\n"
    # Sorts FIRST alphabetically among the anglers but carries no FAO capture
    # statistics - so it must rank AFTER the two FishStat=YES anglers. Without
    # the FishStat tie-break this row would lead, which is how the tie-break
    # gets tested rather than assumed.
    "34,,ANX,Lophius litulon,Abyssal angler,,,,,,"
    "(Jordan 1902),LOPHIIDAE,LOPHIIFORMES,NO\n"
    "32,,HKR,Urophycis chuss,Red hake,,,,,,"
    "(Walbaum 1792),PHYCIDAE,GADIFORMES,YES\n"
    "32,,HKE,Merluccius merluccius,European hake,Merlu europeen,Merluza europea,"
    ",,,(Linnaeus 1758),MERLUCCIIDAE,GADIFORMES,YES\n"
)

ISSCFG_CSV = (
    "FISHERY_CODE,FISHERY_NAME,FISHERY_CATEGORY_NAME,FISHERY_TYPE_NAME,"
    "FISHERY_TYPE_DESCRIPTION,GEAR_CODE,GEAR_NAME,GEAR_GROUP_CODE,GEAR_GROUP_NAME\n"
    '"DL.FR[IS]TR","Medium-scale drifting longlines fishery","Longline fisheries",'
    '"Medium-scale (NJAs and/or ABNJ)","targets tropical tunas","DL",'
    '"Drifting longlines","09","Hooks and lines"\n'
    '"PS.ESP.A","Purse seine A","Surrounding nets","Large-scale","desc A","PS",'
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

CONTRACT_YAML = (
    "simulation:\n"
    "  case_study_name: Northwestern Mediterranean Sea\n"
    "  start_date_time: 2013-01-01T00:00:00\n"
    "  maximum_end_date_time: 2050-01-01T00:00:00\n"
    "  time_step: P1M\n"
    "  geography:\n"
    "    crs:\n      authority: EPSG\n      code: '4326'\n"
    "    xres: 0.0833\n    yres: 0.0833\n    ncol: 109\n    nrow: 82\n"
    "    xmin: -1.0\n    xmax: 8.0\n    ymin: 37.0\n    ymax: 43.8\n"
    "standards:\n"
    "  species_code: FAO 3-alpha\n  gear_code: ISSCFG\n"
    "  country_code: ISO 3166-1 alpha-3\n"
    "  measurements:\n    system: SI\n    units:\n    - quantity: mass\n      unit: kg\n"
    "items:\n"
    "  markets:\n  - market_code: ESALC\n"
    "  species:\n  - species_code: ANK\n"
    "  fleet_segments:\n  - gear_code: PS\n    country_code: ESP\n"
)

EWE_KEY = "ecopath/TestModel.eiixml"
EIIXML_CONTENT = (
    '<?xml version="1.0"?>\n'
    '<EwEModel Name="TestModel" DBVersion="6.700018" EwEVersion="6.7.0.19431">\n'
    '  <Table Name="EcopathGroup" '
    'Columns="GroupID:System.Int64,GroupName:System.String" Num="88">\n'
    "    <Row>1,By-catch</Row>\n"
    "  </Table>\n"
    "</EwEModel>\n"
)

_VOCAB_CONTENT = {
    vocab.ASFIS_KEY: ASFIS_CSV, vocab.ISSCFG_KEY: ISSCFG_CSV,
    vocab.WORMS_KEY: WORMS_CSV, vocab.FISHBASE_KEY: FISHBASE_CSV,
    vocab.ISO3166_KEY: ISO3166_CSV, contract.KEY: CONTRACT_YAML,
    EWE_KEY: EIIXML_CONTENT,
}


@pytest.fixture(autouse=True)
def _isolated_reference_caches():
    vocab.reset_cache()
    contract.reset_cache()
    ewe.reset_cache()
    yield
    vocab.reset_cache()
    contract.reset_cache()
    ewe.reset_cache()


@pytest.fixture(autouse=True)
def _fake_store(monkeypatch):
    def get_text(key, max_bytes=65536):
        return _VOCAB_CONTENT[key], False

    def find(prefix):
        return [{"key": EWE_KEY, "size": len(EIIXML_CONTENT), "last_modified": None}], []

    monkeypatch.setattr(store, "get_text", get_text)
    monkeypatch.setattr(store, "find", find)


def test_tools_registry_is_the_four_names():
    assert rt.TOOLS == [
        "model_lookup_code", "model_search_taxon",
        "model_get_contract", "model_describe_ecosystem_model",
    ]
    for name in rt.TOOLS:
        assert callable(getattr(rt, name))


# ── model_lookup_code ──

def test_lookup_code_ank_returns_species_with_family_and_order():
    out = rt.model_lookup_code("ANK")
    assert out["found"] is True
    assert out["kind"] == "species"
    assert out["scientific_name"] == "Lophius budegassa"
    assert out["family"] == "LOPHIIDAE"
    assert out["order"] == "LOPHIIFORMES"


def test_lookup_code_ank_cross_references_worms_and_fishbase():
    out = rt.model_lookup_code("ANK")
    refs = out["cross_references"]
    assert refs["worms"]["taxon_rank"] == "Species"
    assert refs["fishbase"]["common_name"] == "Blackbellied angler"


def test_lookup_code_dl_fr_is_tr_resolves_against_isscfg():
    out = rt.model_lookup_code("DL.FR[IS]TR", kind="gear")
    assert out["found"] is True
    assert out["kind"] == "gear"
    assert out["fishery_name"] == "Medium-scale drifting longlines fishery"


def test_lookup_code_dts_misses_cleanly_within_gear(monkeypatch):
    """DTS is absent from this ISSCFG fixture (as it is from the live
    ISSCFG_2024.csv - verified 2026-08-07: neither FISHERY_CODE nor
    GEAR_CODE). Scoped to kind="gear" specifically: DTS is a REAL, live FAO
    species code (Pteroctopus schmidti / Dana octopus) - a genuine
    cross-domain collision found during live verification - so an
    unscoped/auto-detect lookup would legitimately resolve it as a species,
    not miss. See test_lookup_code_cross_domain_collision below."""
    out = rt.model_lookup_code("DTS", kind="gear")
    assert out["found"] is False
    assert "DTS" not in str(out.get("suggestions", []))  # no invented name


def test_lookup_code_esp_country_returns_spain_europe_southern_europe():
    out = rt.model_lookup_code("ESP", kind="country")
    assert out["found"] is True
    assert out["name"] == "Spain"
    assert out["region"] == "Europe"
    assert out["sub_region"] == "Southern Europe"


def test_lookup_code_cross_domain_collision_reports_both_not_one():
    """ESP is a real, live collision (verified 2026-08-07): both Spain (ISO
    3166) and Setipinna phasa / Gangetic hairfin anchovy (FAO/ASFIS). Auto-
    detect (kind=None) must report BOTH, never silently pick one."""
    out = rt.model_lookup_code("ESP")
    assert out["found"] is True
    assert out["ambiguous"] is True
    kinds = {m["kind"] for m in out["matches"]}
    assert kinds == {"species", "country"}
    country = next(m for m in out["matches"] if m["kind"] == "country")
    species = next(m for m in out["matches"] if m["kind"] == "species")
    assert country["name"] == "Spain"
    assert species["scientific_name"] == "Setipinna phasa"


def test_lookup_code_unknown_code_returns_no_invented_name():
    out = rt.model_lookup_code("ZZZ999")
    assert out["found"] is False
    assert "error" not in out
    assert isinstance(out["suggestions"], list)
    assert "ZZZ999" not in out["suggestions"]


def test_lookup_code_suggests_only_real_near_matches():
    out = rt.model_lookup_code("ANI", kind="species")   # 1 char off ANK/ANH
    assert out["found"] is False
    # Derived from the fixture, not hardcoded: the claim under test is "every
    # suggestion is a REAL code", which must keep holding as rows are added.
    real_codes = {line.split(",")[2] for line in ASFIS_CSV.strip().splitlines()[1:]}
    for s in out["suggestions"]:
        assert s in real_codes, f"{s} is not a real code in the fixture"


def test_lookup_code_invalid_kind_is_a_clean_error():
    out = rt.model_lookup_code("ANK", kind="not-a-kind")
    assert out["found"] is False
    assert "error" in out


def test_lookup_code_empty_string_is_a_clean_error():
    out = rt.model_lookup_code("   ")
    assert out["found"] is False
    assert "error" in out


def test_a_vocabulary_load_failure_degrades_only_that_cross_reference(monkeypatch):
    """WoRMS fails to load; the ASFIS-sourced species match must still come
    back (family/order/scientific_name), with only the WoRMS slice of
    cross_references marked degraded - not the whole tool call."""
    def get_text(key, max_bytes=65536):
        if key == vocab.WORMS_KEY:
            raise store.SimStoreError("boom")
        return _VOCAB_CONTENT[key], False

    monkeypatch.setattr(store, "get_text", get_text)
    out = rt.model_lookup_code("ANK")
    assert out["found"] is True
    assert out["scientific_name"] == "Lophius budegassa"
    assert out["cross_references"]["worms"] is None
    assert out["cross_references"]["degraded"]
    assert out["cross_references"]["fishbase"]["common_name"] == "Blackbellied angler"

    # A DIFFERENT tool (independent load path) must be completely unaffected.
    contract_out = rt.model_get_contract()
    assert "error" not in contract_out
    assert contract_out["case_study_name"] == "Northwestern Mediterranean Sea"


def test_a_species_load_failure_degrades_the_whole_species_domain(monkeypatch):
    def get_text(key, max_bytes=65536):
        if key == vocab.ASFIS_KEY:
            raise store.SimStoreError("boom")
        return _VOCAB_CONTENT[key], False

    monkeypatch.setattr(store, "get_text", get_text)
    out = rt.model_lookup_code("ANK", kind="species")
    assert out["found"] is False
    assert out["degraded"]

    # An unrelated tool (EwE models) is unaffected.
    models_out = rt.model_describe_ecosystem_model()
    assert "error" not in models_out
    assert models_out["count"] == 1


# ── model_search_taxon ──

def test_search_taxon_finds_hits_across_sources_with_source_labels():
    out = rt.model_search_taxon("budegassa")
    sources = {r["source"] for r in out["results"]}
    assert sources == {"asfis", "worms", "fishbase"}
    assert out["count"] == len(out["results"])


def test_search_taxon_respects_limit():
    out = rt.model_search_taxon("a", limit=1)
    assert len(out["results"]) == 1


def test_search_taxon_empty_name_is_a_clean_error():
    out = rt.model_search_taxon("   ")
    assert out["results"] == []
    assert "error" in out


def test_search_taxon_ranks_whole_word_hits_above_mid_word_accidents():
    """A name that merely CONTAINS the query inside a longer word must not
    outrank a name where the query is a word. Regression: "hake" returned
    Capoetobrama kuschakewitschi (Sharpray) at rank 1 because it sorted first
    in the source file."""
    out = rt.model_search_taxon("hake", limit=5)
    names = [r.get("english_name") or r.get("common_name") or "" for r in out["results"]]
    assert names[0] != "Sharpray", f"mid-word accident still ranked first: {names}"
    real_hakes = [n for n in names if n.lower().endswith("hake")]
    assert real_hakes, f"no genuine hake in the results at all: {names}"
    assert names.index(real_hakes[0]) < (names.index("Sharpray") if "Sharpray" in names else 99)


def test_search_taxon_reports_that_more_matches_exist_beyond_the_limit():
    """21 species are genuinely named "... hake" in ASFIS, so no ranking can
    know which one a caller means. What the tool CAN do is admit the list was
    cut, so the caller narrows instead of assuming the top 5 is all of it."""
    out = rt.model_search_taxon("a", limit=1)
    assert out["count"] == 1
    assert out.get("total_matches", 0) > 1


def test_search_taxon_prefers_species_fao_keeps_statistics_for():
    """Equal-quality name matches are tie-broken by ASFIS's own FishStat_Data
    flag before the neutral alphabetical fallback, so commercially recorded
    species come first. ANX ("Abyssal angler", FishStat=NO) sorts first
    alphabetically and must still land last."""
    out = rt.model_search_taxon("angler", limit=5)
    codes = [r["code"] for r in out["results"] if r["source"] == "asfis"]
    assert codes == ["ANH", "ANK", "ANX"], f"FishStat tie-break not applied: {codes}"


def test_search_taxon_exact_name_match_wins():
    """An exact whole-name match outranks every partial hit."""
    out = rt.model_search_taxon("European hake", limit=5)
    assert out["results"], "exact name found nothing"
    assert out["results"][0]["code"] == "HKE"


def test_search_taxon_mid_word_match_is_still_findable():
    """Ranking demotes mid-word hits; it must not drop them. Searching the
    accident's own fragment still finds it."""
    out = rt.model_search_taxon("kuschake", limit=5)
    names = [r.get("english_name") or r.get("common_name") or "" for r in out["results"]]
    assert "Sharpray" in names, f"demoted match disappeared entirely: {names}"


# ── model_get_contract ──

def test_get_contract_returns_grid_period_and_standards():
    out = rt.model_get_contract()
    assert out["case_study_name"] == "Northwestern Mediterranean Sea"
    assert out["start_date_time"] == "2013-01-01T00:00:00"
    assert out["maximum_end_date_time"] == "2050-01-01T00:00:00"
    assert out["grid"]["crs"] == "EPSG:4326"
    assert out["standards"]["species_code"] == "FAO 3-alpha"


def test_get_contract_names_the_path_it_reads_and_the_other_case_studies():
    """The contract moved on 2026-09-15 from surimi-controller/config/ to a folder
    per case study. The note says which one this tool reads and that the bucket
    holds others, so a caller does not take northwestern_med for the only case."""
    out = rt.model_get_contract()
    note = out["divergence_note"]
    assert "controller/northwestern_med/northwestern_med_contract.yaml" in note
    assert "aegean_sea" in note and "anchovy_bay" in note
    assert "surimi-controller" in note      # names the path that no longer exists


def test_get_contract_transient_failure_is_a_degraded_response_not_a_crash(monkeypatch):
    def get_text(key, max_bytes=65536):
        raise store.SimStoreError("boom")

    monkeypatch.setattr(store, "get_text", get_text)
    out = rt.model_get_contract()
    assert "error" in out


# ── model_describe_ecosystem_model ──

def test_describe_ecosystem_model_lists_available_models_with_no_argument():
    out = rt.model_describe_ecosystem_model()
    assert out["count"] == 1
    assert out["models"][0]["name"] == "TestModel"


def test_describe_ecosystem_model_named_returns_table_inventory_and_groups():
    out = rt.model_describe_ecosystem_model("TestModel")
    assert out["model_name"] == "TestModel"
    assert out["table_count"] == 1
    assert out["functional_groups"]["names"] == ["By-catch"]


def test_describe_ecosystem_model_unknown_name_is_a_clean_miss():
    out = rt.model_describe_ecosystem_model("NoSuchModel")
    assert "error" in out
