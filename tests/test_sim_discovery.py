import pytest

from surimi_model_mcp.simulations import discovery


@pytest.fixture(autouse=True)
def _isolated_cache():
    """The module-level scan cache must not leak state between tests, or
    pass/fail order becomes accidental rather than guaranteed."""
    discovery._reset_cache()


def test_uuid_directory_segment_identifies_a_run():
    got = discovery.parse_run(
        "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv")
    assert got == ("surimi-cmsy", "0880dde9-b43c-4888-af62-34783a6b6239", "id_file.csv")


def test_container_capitalisation_is_irrelevant():
    """CMSY writes 'Simulations', market writes 'simulations'. Both are runs."""
    got = discovery.parse_run(
        "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/config/ports.csv")
    assert got == ("surimi-market", "86da58d2-eded-47a1-8360-6a31ee2e9bb3",
                   "config/ports.csv")


def test_uuid_filename_is_not_a_run():
    """output_creator/<scenario>/<uuid>/<uuid>.nc has the uuid TWICE. Only the
    directory occurrence starts a run; the file must not create a second."""
    key = ("output_creator/aegean_sea/25ea0758-4976-461e-bd38-9a60305164cb/"
           "25ea0758-4976-461e-bd38-9a60305164cb.nc")
    model, run, rel = discovery.parse_run(key)
    assert model == "output_creator"
    assert run == "25ea0758-4976-461e-bd38-9a60305164cb"
    assert rel == "25ea0758-4976-461e-bd38-9a60305164cb.nc"


def test_key_without_uuid_is_not_a_run():
    assert discovery.parse_run("surimi-market/config/ports.csv") is None
    assert discovery.parse_run("netcdf/ar6_ocean_mask_0.25deg.nc") is None


def test_deny_prefixes_lists_the_non_model_areas():
    """Static membership only - the actual skip behaviour during a scan is
    covered by test_scan_groups_artifacts_into_runs."""
    for p in ("mcp_data/", "surimi_data/", "surimi_docs/", "hive/",
              "processed/", "raw/", "surimi_data_dummy/"):
        assert p in discovery.DENY_PREFIXES


def test_scan_groups_artifacts_into_runs(monkeypatch):
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/x_KOBE.jpg",
         "size": 20, "last_modified": "2026-01-07T12:54:00"},
        {"key": "mcp_data/tables/huge.csv", "size": 99, "last_modified": None},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["stale"] is False
    assert len(out["simulations"]) == 1
    sim = out["simulations"][0]
    assert sim["simulation_id"] == "0880dde9-b43c-4888-af62-34783a6b6239"
    assert sim["n_artifacts"] == 2
    assert sim["kinds"] == {"csv": 1, "jpg": 1}
    assert sim["last_modified"] == "2026-01-07T12:54:00"   # newest wins


def test_scan_serves_last_good_listing_when_storage_fails(monkeypatch):
    """STS tokens expire ~24h. A stale answer that says so beats an error."""
    good = [{"key": "m/simulations/0880dde9-b43c-4888-af62-34783a6b6239/a.csv",
             "size": 1, "last_modified": None}]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (good, []))
    first = discovery.scan(force=True)
    assert first["stale"] is False

    def boom(prefix=""):
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    second = discovery.scan(force=True)
    assert second["stale"] is True
    assert second["simulations"] == first["simulations"]
    assert "as_of" in second


def test_scan_payloads_are_isolated_copies_not_shared_cache_objects(monkeypatch):
    """A caller mutating a returned payload (enriching or sorting a
    simulation's nested kinds/keys) must never corrupt
    what the cache hands back to the next caller - on the cache-hit path or
    the stale-serving path. `x == x` on the same object proves nothing here;
    this mutates a NESTED field and re-fetches to check for leakage."""
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))

    first = discovery.scan(force=True)
    first["simulations"][0]["kinds"]["csv"] = 999
    first["simulations"][0]["keys"].append("planted/forgery.csv")

    cache_hit = discovery.scan()   # within CACHE_TTL_S, force=False
    assert cache_hit["simulations"][0]["kinds"]["csv"] == 1
    assert "planted/forgery.csv" not in cache_hit["simulations"][0]["keys"]

    def boom(prefix=""):
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    stale = discovery.scan(force=True)
    assert stale["stale"] is True
    assert stale["simulations"][0]["kinds"]["csv"] == 1
    assert "planted/forgery.csv" not in stale["simulations"][0]["keys"]


def test_scan_skips_zero_byte_directory_marker_keys(monkeypatch):
    """S3 consoles / some clients create a zero-byte object for the
    directory itself (key ends in "/"). parse_run gives it relative_path
    "" - it names no file and must not be counted as an artifact."""
    fake = [
        {"key": "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/",
         "size": 0, "last_modified": "2026-01-07T12:00:00"},
        {"key": "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/config/ports.csv",
         "size": 5, "last_modified": "2026-01-07T12:01:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert len(out["simulations"]) == 1
    sim = out["simulations"][0]
    assert sim["n_artifacts"] == 1
    assert sim["kinds"] == {"csv": 1}
    assert all(k for k in sim["keys"])   # no empty-string / marker key present
    assert sim["keys"] == [
        "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/config/ports.csv"
    ]


def test_scan_field_lists_only_models_with_at_least_one_run(monkeypatch):
    """`scanned` is sorted, de-duplicated model names - but only for models
    that actually contributed a run. A model name appearing on a key that
    doesn't parse as a run (no UUID directory) must not appear."""
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/a.csv",
         "size": 1, "last_modified": "2026-01-01T00:00:00"},
        {"key": "surimi-cmsy/Simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/b.csv",
         "size": 1, "last_modified": "2026-01-02T00:00:00"},
        {"key": "surimi-market/config/ports.csv",   # no UUID directory: not a run
         "size": 1, "last_modified": "2026-01-03T00:00:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["scanned"] == ["surimi-cmsy"]


def test_scan_raises_when_failing_with_no_cached_listing(monkeypatch):
    discovery._reset_cache()

    def boom(prefix=""):
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    with pytest.raises(discovery.store.SimStoreError):
        discovery.scan(force=True)


def test_scan_records_per_artifact_size(monkeypatch):
    """The run tools need per-artifact size to decide inline-fetch vs
    link without an extra round trip. store.find already returns "size" per
    object; scan() must carry it through, keyed by the artifact's own key."""
    id_key = "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv"
    jpg_key = "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/x_KOBE.jpg"
    fake = [
        {"key": id_key, "size": 812, "last_modified": "2026-01-07T12:53:00"},
        {"key": jpg_key, "size": 204800, "last_modified": "2026-01-07T12:54:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    sim = out["simulations"][0]
    assert sim["sizes"] == {id_key: 812, jpg_key: 204800}
    # "keys" must keep working for existing consumers.
    assert sim["keys"] == [id_key, jpg_key]


def test_scan_skips_nested_directory_marker_keys(monkeypatch):
    """`scan()` already skips a run-ROOT marker (rel == ""). A marker for a
    subdirectory INSIDE a run (e.g. market's config/) still yields a
    non-empty rel ("config/") and must not be counted as a phantom artifact
    with kind "none" - market runs really do have config/ subdirectories."""
    fake = [
        {"key": "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/config/",
         "size": 0, "last_modified": "2026-01-07T12:00:00"},
        {"key": "surimi-market/simulations/86da58d2-eded-47a1-8360-6a31ee2e9bb3/config/ports.csv",
         "size": 5, "last_modified": "2026-01-07T12:01:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    sim = out["simulations"][0]
    assert sim["n_artifacts"] == 1
    assert sim["kinds"] == {"csv": 1}
    assert "none" not in sim["kinds"]


def test_scan_discovers_a_never_before_seen_model_prefix(monkeypatch):
    """Generic discovery, tested end-to-end rather than riding on the
    CMSY case: a run directory under a model prefix that has NEVER existed
    in this codebase (no parser, no fixture, no prior test) is discovered
    with no code change. This is what the seven not-yet-publishing ensemble
    models depend on."""
    run_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    fake = [
        {"key": f"poseidon/Runs/{run_id}/output.nc",
         "size": 42, "last_modified": "2026-02-01T00:00:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["scanned"] == ["poseidon"]
    assert len(out["simulations"]) == 1
    sim = out["simulations"][0]
    assert sim["model"] == "poseidon"
    assert sim["simulation_id"] == run_id


def test_scan_reports_partial_when_store_find_drops_objects(monkeypatch):
    """store.find() now returns (objects, failures). When failures is
    non-empty, THIS scan itself is short - even though it succeeded and is
    not "stale" - and callers need a flag to say so."""
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
    ]
    failures = ["surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/x_KOBE.jpg: stat failed"]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, failures))
    out = discovery.scan(force=True)
    assert out["stale"] is False
    assert out["partial"] is True
    assert "partial_note" in out and out["partial_note"]


def test_scan_partial_is_false_and_note_absent_on_a_clean_listing(monkeypatch):
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["partial"] is False
    assert "partial_note" not in out


def test_scan_partial_survives_a_cache_hit(monkeypatch):
    """A flag that silently drops on a
    cache hit is worse than no flag. Deepcopy already isolates the whole
    payload, but this proves "partial" specifically rides along, not just
    the fields earlier tests happened to check."""
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
    ]
    failures = ["surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/x_KOBE.jpg: stat failed"]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, failures))
    first = discovery.scan(force=True)
    assert first["partial"] is True

    cache_hit = discovery.scan()   # within CACHE_TTL_S, force=False
    assert cache_hit["partial"] is True
    assert cache_hit["partial_note"] == first["partial_note"]


def test_scan_partial_survives_the_stale_serving_path(monkeypatch):
    """A degraded (last-good) listing that was itself partial when it was
    captured must say both things at once: stale (this answer is old) AND
    partial (it was short even when it was captured)."""
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 10, "last_modified": "2026-01-07T12:53:00"},
    ]
    failures = ["surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/x_KOBE.jpg: stat failed"]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, failures))
    first = discovery.scan(force=True)
    assert first["partial"] is True

    def boom(prefix=""):
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    stale = discovery.scan(force=True)
    assert stale["stale"] is True
    assert stale["partial"] is True
    assert stale["partial_note"] == first["partial_note"]


def test_scan_counts_run_like_keys_dropped_by_a_denied_prefix(monkeypatch):
    """"raw/" and "processed/" are ordinary English words, not namespaced to
    this project - a future model's directory could collide with one and be
    silently INVISIBLE (not even degraded) rather than discovered. Prove the
    generic-discovery bet is monitored, not blind: a run-shaped key under a
    denied prefix must still be counted and noted, even though it stays
    excluded from `simulations`/`scanned`."""
    run_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    fake = [
        {"key": f"raw/mice/{run_id}/out.nc",
         "size": 10, "last_modified": "2026-02-01T00:00:00"},
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 5, "last_modified": "2026-01-07T12:53:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["denied_run_like_keys"] == 1
    assert "denied_note" in out and "1" in out["denied_note"]
    # Still excluded - the fix is visibility into the count, not un-denying.
    assert "mice" not in out["scanned"]
    assert len(out["simulations"]) == 1


def test_scan_denied_run_like_keys_is_zero_and_note_absent_on_a_clean_listing(monkeypatch):
    fake = [
        {"key": "surimi-cmsy/Simulations/0880dde9-b43c-4888-af62-34783a6b6239/id_file.csv",
         "size": 5, "last_modified": "2026-01-07T12:53:00"},
    ]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (fake, []))
    out = discovery.scan(force=True)
    assert out["denied_run_like_keys"] == 0
    assert "denied_note" not in out


def test_scan_backs_off_after_a_failure_instead_of_retrying_every_call(monkeypatch):
    """The retry-storm shape only appears once an outage has outlasted
    CACHE_TTL_S: before that, calls are already served from the good-listing
    cache without touching storage. Simulate that precondition directly
    (age _cache["at"]) rather than sleeping 300+ real seconds, then prove
    that after one failed attempt, a later NON-forced call within the
    backoff window does not call store.find() again - it re-serves the
    already-known-stale answer instead."""
    good = [{"key": "m/simulations/0880dde9-b43c-4888-af62-34783a6b6239/a.csv",
             "size": 1, "last_modified": None}]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (good, []))
    first = discovery.scan(force=True)
    assert first["stale"] is False

    # The good listing has aged past CACHE_TTL_S: a non-forced call would
    # now fall through the TTL cache-hit check and attempt storage again -
    # the precondition for the retry-storm bug this fix closes.
    discovery._cache["at"] -= discovery.CACHE_TTL_S + 1

    calls = {"n": 0}

    def boom(prefix=""):
        calls["n"] += 1
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    second = discovery.scan()   # not forced: TTL cache expired, so this really attempts
    assert second["stale"] is True
    assert calls["n"] == 1

    third = discovery.scan()   # not forced, within the backoff window
    assert third["stale"] is True
    assert calls["n"] == 1   # no new LIST attempt - served from backoff, not a retry


def test_scan_retries_again_once_the_backoff_window_elapses(monkeypatch):
    """The backoff throttles retries, it does not disable them forever -
    once the window has passed, the next call must attempt store.find()
    again so a real recovery is noticed."""
    good = [{"key": "m/simulations/0880dde9-b43c-4888-af62-34783a6b6239/a.csv",
             "size": 1, "last_modified": None}]
    monkeypatch.setattr(discovery.store, "find", lambda prefix="": (good, []))
    discovery.scan(force=True)
    discovery._cache["at"] -= discovery.CACHE_TTL_S + 1

    calls = {"n": 0}

    def boom(prefix=""):
        calls["n"] += 1
        raise discovery.store.SimStoreError("token expired")

    monkeypatch.setattr(discovery.store, "find", boom)
    discovery.scan()   # not forced: TTL cache expired, real attempt #1
    assert calls["n"] == 1

    # Simulate the backoff window having elapsed too, without a real sleep.
    discovery._cache["failed_at"] -= discovery._FAILURE_BACKOFF_S + 1

    discovery.scan()   # not forced; both TTL and backoff windows have passed
    assert calls["n"] == 2


# ---------- the run listing must outlive the process ----------
# _cache is in-memory, so a restart loses the last good listing entirely and
# model_list_simulations goes from "stale but useful" to dark. Observed
# 2026-09-17: the STS token expired and the catalogue could not be listed.

def test_scan_persists_its_payload(tmp_path, monkeypatch):
    from surimi_model_mcp.simulations import discovery as sd
    monkeypatch.setattr(sd, "_STATE_DIR", str(tmp_path))
    sd._save_payload({"simulations": [{"model": "m", "simulation_id": "x"}],
                      "as_of": "2026-09-17T00:00:00Z"})
    got = sd._load_payload()
    assert got is not None
    assert got["simulations"][0]["simulation_id"] == "x"


def test_load_payload_absent_returns_none(tmp_path, monkeypatch):
    from surimi_model_mcp.simulations import discovery as sd
    monkeypatch.setattr(sd, "_STATE_DIR", str(tmp_path))
    assert sd._load_payload() is None
