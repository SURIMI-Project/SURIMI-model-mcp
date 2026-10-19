"""A run whose id sits in the FILE name, not a directory.

The fisheries-authority model writes its harvest-control-rule runs flat:
fisheries_authority/output/<uuid>_{mse-assessment,tac,catch-monthly,
biomass-monthly}.csv. Discovery only accepted a UUID directory, so all 35
WMed runs uploaded on 2026-09-15 were invisible to every tool, and a
reviewer's question about the red zone under the harvest rules was answered
as if no run existed.
"""
from surimi_model_mcp.simulations import discovery

U = "170c1b82-1111-4222-8333-444455556666"


def test_a_uuid_prefixed_file_belongs_to_that_run():
    assert discovery.parse_run(f"fisheries_authority/output/{U}_mse-assessment.csv") == (
        "fisheries_authority", U, f"{U}_mse-assessment.csv")


def test_the_four_files_of_one_run_share_its_id():
    ids = {discovery.parse_run(f"fisheries_authority/output/{U}_{n}.csv")[1]
           for n in ("mse-assessment", "tac", "catch-monthly", "biomass-monthly")}
    assert ids == {U}


def test_a_uuid_directory_still_wins():
    key = f"output_creator/experiments/{U}/{U}.nc"
    assert discovery.parse_run(key) == ("output_creator", U, f"{U}.nc")


def test_a_file_without_a_uuid_prefix_is_not_a_run():
    assert discovery.parse_run("cmsy/northwestern_med/id_file.csv") is None
    assert discovery.parse_run("fisheries_authority/northwestern_med/quotashare.csv") is None


def test_a_uuid_merely_inside_a_file_name_is_not_a_run():
    assert discovery.parse_run(f"market/notes_{U}.csv") is None


def test_flat_runs_are_grouped_by_the_scan(monkeypatch):
    keys = [f"fisheries_authority/output/{U}_{n}.csv"
            for n in ("mse-assessment", "tac", "catch-monthly", "biomass-monthly")]
    monkeypatch.setattr(discovery.store, "find", lambda prefix: (
        [{"key": k, "size": 10, "last_modified": "2026-09-15T10:00:00"} for k in keys], []))
    discovery._reset_cache()
    out = discovery.scan(force=True)
    discovery._reset_cache()
    runs = [s for s in out["simulations"] if s["simulation_id"] == U]
    assert len(runs) == 1
    assert runs[0]["model"] == "fisheries_authority"
    assert runs[0]["n_artifacts"] == 4
    assert runs[0]["kinds"] == {"csv": 4}
