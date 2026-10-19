"""reference/contract.py: parsing the case-study contract YAML and carrying
the case-study folder layout. Network-free - store.get_text is
monkeypatched with a fixture trimmed from the real
western_med_contract.yaml shape (verified 2026-08-07)."""
import pytest

from surimi_model_mcp.reference import contract
from surimi_model_mcp.simulations import store

CONTRACT_YAML = """\
simulation:
  case_study_name: Northwestern Mediterranean Sea
  start_date_time: 2013-01-01T00:00:00
  maximum_end_date_time: 2050-01-01T00:00:00
  time_step: P1M
  geography:
    raster_cell_origin: Centroid
    crs:
      name: WGS 84
      authority: EPSG
      code: '4326'
    xres: 0.0833
    yres: 0.0833
    ncol: 109
    nrow: 82
    xmin: -1.0185
    xmax: 8.0612002
    ymin: 37.0139008
    ymax: 43.8445009
standards:
  date_and_time: ISO 8601
  species_code: FAO 3-alpha
  gear_code: ISSCFG
  life_stage: dwc:lifeStage
  market_code: UN/LOCODE
  currency: ISO 4217
  country_code: ISO 3166-1 alpha-3
  measurements:
    system: SI
    units:
    - quantity: mass
      unit: kg
items:
  markets:
  - market_code: ESALC
  - market_code: ESBCN
  currencies:
  - currency_code: EUR
  species:
  - species_code: ALB
  - species_code: ANE
    life_stage: juvenile
  fleet_segments:
  - gear_code: EwE:ART
    country_code: ESP
    model: EwE
"""

BAD_YAML = "simulation: [unterminated"


@pytest.fixture(autouse=True)
def _isolated_contract_cache():
    contract.reset_cache()
    yield
    contract.reset_cache()


def _install(monkeypatch, text=CONTRACT_YAML, truncated=False, calls=None):
    calls = calls if calls is not None else {"n": 0}

    def get_text(key, max_bytes=65536):
        calls["n"] += 1
        assert key == contract.KEY
        return text, truncated

    monkeypatch.setattr(store, "get_text", get_text)
    return calls


def test_parses_case_study_period_grid_and_standards(monkeypatch):
    _install(monkeypatch)
    parsed, err = contract.get()
    assert err is None
    assert parsed["case_study_name"] == "Northwestern Mediterranean Sea"
    assert parsed["start_date_time"] == "2013-01-01T00:00:00"
    assert parsed["maximum_end_date_time"] == "2050-01-01T00:00:00"
    assert parsed["time_step"] == "P1M"
    assert parsed["grid"]["crs"] == "EPSG:4326"
    assert parsed["grid"]["ncol"] == 109
    assert parsed["grid"]["nrow"] == 82
    assert parsed["grid"]["bbox"]["xmin"] == -1.0185
    assert parsed["standards"]["species_code"] == "FAO 3-alpha"
    assert parsed["standards"]["gear_code"] == "ISSCFG"
    assert parsed["standards"]["country_code"] == "ISO 3166-1 alpha-3"
    assert parsed["standards"]["measurement_system"] == "SI"
    assert parsed["standards"]["measurement_units"] == ["kg"]
    assert parsed["markets"] == ["ESALC", "ESBCN"]
    assert parsed["species_count"] == 2
    assert parsed["fleet_segment_count"] == 1
    assert parsed["fleet_segments"] == [
        {"gear_code": "EwE:ART", "country_code": "ESP", "model": "EwE"}
    ]


def test_fleet_segments_carry_a_model_field_per_segment(monkeypatch):
    """Q-E3-2: 'can I turn POSEIDON off and only fish with Ecopath' needs the
    per-fleet `model` assignment (western_med_contract.yaml's items.fleet_segments[].model,
    e.g. 7 fleets -> EwE, 2 -> POSEIDON) to be reachable through model_get_contract -
    fleet_segment_count alone (a bare int) cannot answer which fleet is on which model."""
    text = CONTRACT_YAML.replace(
        "  fleet_segments:\n  - gear_code: EwE:ART\n    country_code: ESP\n    model: EwE\n",
        "  fleet_segments:\n"
        "  - gear_code: EwE:ART\n    country_code: ESP\n    model: EwE\n"
        "  - gear_code: PS\n    country_code: ESP\n    model: POSEIDON\n",
    )
    _install(monkeypatch, text=text)
    parsed, err = contract.get()
    assert err is None
    assert parsed["fleet_segment_count"] == 2
    models_seen = [fs["model"] for fs in parsed["fleet_segments"]]
    assert models_seen == ["EwE", "POSEIDON"]


def test_cache_serves_second_call_without_refetch(monkeypatch):
    calls = _install(monkeypatch)
    contract.get()
    assert calls["n"] == 1
    contract.get()
    assert calls["n"] == 1


def test_missing_required_key_is_a_deterministic_error(monkeypatch):
    _install(monkeypatch, text="simulation:\n  time_step: P1M\n")
    parsed, err = contract.get()
    assert parsed is None
    assert err is not None


def test_invalid_yaml_is_a_deterministic_error(monkeypatch):
    _install(monkeypatch, text=BAD_YAML)
    parsed, err = contract.get()
    assert parsed is None
    assert err is not None


def test_truncated_read_is_refused(monkeypatch):
    _install(monkeypatch, truncated=True)
    parsed, err = contract.get()
    assert parsed is None
    assert "truncated" in err.lower()


def test_transient_store_error_is_not_cached_and_is_retried(monkeypatch):
    state = {"fail": True}
    calls = {"n": 0}

    def get_text(key, max_bytes=65536):
        calls["n"] += 1
        if state["fail"]:
            raise store.SimStoreError("boom")
        return CONTRACT_YAML, False

    monkeypatch.setattr(store, "get_text", get_text)

    with pytest.raises(store.SimStoreError):
        contract.get()
    assert calls["n"] == 1

    state["fail"] = False
    parsed, err = contract.get()
    assert err is None
    assert parsed["case_study_name"] == "Northwestern Mediterranean Sea"
    assert calls["n"] == 2


def test_contract_key_is_the_northwestern_med_folder():
    assert contract.KEY == "controller/northwestern_med/northwestern_med_contract.yaml"
