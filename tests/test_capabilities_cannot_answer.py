"""What this server declares it cannot answer, and where those answers live.

Separate from test_capabilities_classify.py, which tests the dataclass. This
tests the content: the `cannot_answer` entries that `sources.py` registers, and the claims
they make about the filesystem.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from surimi_model_mcp import capabilities as capabilities
from surimi_model_mcp import server as server  # noqa: F401  -- import registers the capabilities


MONOREPO = Path(__file__).resolve().parents[2]

# The folders some declared paths live in that a checkout of the monorepo may not have: the data
# lake and the NetCDF sample are gitignored, CMSY++ is a separate clone. A path under one of them is
# checked only where that folder is present and not empty, so CI (code without data) can run the
# suite. The folders are named here, never derived from the path; any other path is always checked.
_DATA_ROOTS = (
    "references/SURIMI-CMSY/",
    "references/SURIMI-DATA-LAKE/",
    "surimi-mcp/data/netcdf-samples/",
)


def _data_present(lives_in: str) -> bool:
    root = next((r for r in _DATA_ROOTS if lives_in.startswith(r)), None)
    return root is None or ((MONOREPO / root).is_dir() and any((MONOREPO / root).iterdir()))


def _entries():
    return [(c.source_id, e)
            for c in capabilities.all_capabilities()
            for e in c.cannot_answer]


def test_the_two_data_sources_declare_what_they_cannot_answer():
    """A server that lists only its tools reads as if it can answer anything
    inside its domain. Both data sources have to say otherwise."""
    declared = {sid for sid, _ in _entries()}
    assert "ecoocean-netcdf" in declared
    assert "surimi-simulations" in declared


def test_every_declared_path_resolves_in_the_monorepo():
    """The assertion names the tree it asserts about.

    Two of these paths (surimi-mcp/, references/) exist only in the monorepo,
    which holds both servers and the data lake. In a standalone clone of this
    repo they are absent and correct, so asserting there would fail a true
    value. Tested as declared, never as a derived parent: dirname("a/b/") is
    "a", which resolves for the wrong reason and hides a broken path.
    """
    if not (MONOREPO / "surimi-mcp").is_dir():
        pytest.skip("not the monorepo tree; these paths live only there")
    for source_id, e in _entries():
        if not e.lives_in or e.lives_in.startswith("s3://"):
            continue
        if not _data_present(e.lives_in):
            continue        # its data folder is not checked out here
        assert (MONOREPO / e.lives_in).exists(), f"{source_id}: {e.lives_in}"


def test_an_absent_answer_names_nowhere():
    """no_mechanism and a genuinely uncollected series must not invent a place.

    The field exists so the model can say where an answer lives. An entry that
    points somewhere plausible but wrong is worse than one that points nowhere,
    because the model will quote it.
    """
    for source_id, e in _entries():
        if e.why == "no_mechanism":
            assert e.lives_in == "", f"{source_id}: {e.lives_in}"


def test_every_detail_carries_a_number():
    """An abstention saying a file exists is weak; one saying how big it is and
    what is in it is checkable. Convention shared with the data server."""
    for source_id, e in _entries():
        assert any(ch.isdigit() for ch in e.detail), source_id


def test_the_gridded_record_declares_that_it_stops_before_the_contract_starts():
    """The single fact most likely to be misread off this server: the served
    file's data ends 2009-12 and every SURIMI contract begins 2013-01, so no
    data-bearing step falls inside the case-study period at all."""
    texts = [e.detail for sid, e in _entries() if sid == "ecoocean-netcdf"]
    joined = " ".join(texts)
    assert "2009" in joined and "2013" in joined


def test_describe_surfaces_the_entries_to_a_client():
    out = capabilities.describe(["model_list_files"])
    blobs = [c for c in out["data_sources"] if c["source_id"] == "ecoocean-netcdf"]
    assert blobs, out["data_sources"]
    assert blobs[0]["cannot_answer"], "describe() dropped the field"


def test_no_entry_claims_that_nothing_has_been_run():
    """A run landed on 2026-09-15 and the first version of this field said none
    had. It was written from a catalogue instead of from the server, which is
    the one mistake this field exists to stop the MODEL making."""
    joined = " ".join(e.detail for _, e in _entries()).lower()
    for false_claim in ("no surimi experiment has been run",
                        "nothing has been run",
                        "no run has been produced"):
        assert false_claim not in joined, false_claim


def test_the_projection_gap_is_about_the_horizon_not_the_absence_of_output():
    """The run is a hindcast: 2013-01 to 2017-01. What is missing is the
    future, not the output."""
    sims = [e for sid, e in _entries() if sid == "surimi-simulations"]
    assert sims
    detail = " ".join(e.detail for e in sims)
    assert "2017-01" in detail
    assert "hindcast" in detail
