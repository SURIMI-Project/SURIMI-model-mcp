"""The reference files are copied onto the volume at launch.

The server reads through a local cache, so once a file is cached an expired STS
token no longer breaks it. But a file is cached only after its first successful
read, and the species-code, ecosystem-model and contract files were read from
paths that no longer existed until 1.2.10, so none had ever been cached. A
restart after the token expired left them unreadable. They are small (about
14 MB), so launch fetches them all, the same way it already fetches the NetCDF
catalogue and the simulation artifacts.
"""
from surimi_model_mcp.reference import vocab, warm


def _objects(*keys):
    return ([{"key": k, "size": 1} for k in keys], [])


def test_keys_cover_the_five_vocabularies_the_models_and_the_contracts():
    listing = {
        "ecopath": _objects("ecopath/ASFIS_sp_2024.csv", "ecopath/northwestern_med/northwestern_med.eiixml",
                            "ecopath/anchovy_bay/anchovy_bay.eiixml", "ecopath/ISSCFG_2024.pdf",
                            "ecopath/northwestern_med/EwE_SURIMI.semantics"),
        "controller": _objects("controller/northwestern_med/northwestern_med_contract.yaml",
                               "controller/aegean_sea/aegean_sea_contract.yaml",
                               "controller/notes.txt"),
    }
    keys = warm.reference_keys(find=lambda prefix: listing[prefix])
    for k in (vocab.ASFIS_KEY, vocab.ISSCFG_KEY, vocab.WORMS_KEY, vocab.FISHBASE_KEY, vocab.ISO3166_KEY):
        assert k in keys, k
    assert "ecopath/northwestern_med/northwestern_med.eiixml" in keys
    assert "ecopath/anchovy_bay/anchovy_bay.eiixml" in keys
    assert "controller/northwestern_med/northwestern_med_contract.yaml" in keys
    assert "controller/aegean_sea/aegean_sea_contract.yaml" in keys
    # only what a tool can ask for: no PDF, no semantics file, no notes
    assert "ecopath/ISSCFG_2024.pdf" not in keys
    assert "controller/notes.txt" not in keys


def test_keys_have_no_duplicates():
    listing = {"ecopath": _objects(vocab.ASFIS_KEY, "ecopath/m/m.eiixml"), "controller": _objects()}
    keys = warm.reference_keys(find=lambda prefix: listing[prefix])
    assert len(keys) == len(set(keys))


def test_a_failed_listing_still_warms_the_five_fixed_vocabularies():
    """Listing needs a valid token too; the five vocabulary keys do not depend on it."""
    def boom(prefix):
        raise RuntimeError("expired token")
    keys = warm.reference_keys(find=boom)
    assert vocab.ASFIS_KEY in keys and vocab.ISO3166_KEY in keys


def test_warm_hands_the_keys_to_the_cache_and_reports_the_result():
    seen = []
    def fake_prefetch(keys):
        seen.extend(keys)
        return {"cached": len(seen), "skipped": 0, "errors": []}
    listing = {"ecopath": _objects("ecopath/m/m.eiixml"), "controller": _objects()}
    res = warm.warm(find=lambda p: listing[p], prefetch=fake_prefetch)
    assert "ecopath/m/m.eiixml" in seen and vocab.ASFIS_KEY in seen
    assert res["errors"] == []
