"""A findable file still has to say whose it is.

1.2.1 made model_list_files carry each data file's variables, which fixed four
refusals. It also made the external EcoOcean/FishMIP file easy to reach without
saying it is external, and the next run used it to answer "is small-demersal
biomass actually declining in the Mediterranean" as though it were SURIMI's own
result. That is the fabrication this marker exists to stop: the answer may be
right about the file and wrong about the project.

A file a SURIMI run wrote carries simulation_id. Everything else under the
reference prefix is external or reference data, and the listing now says so.
"""
from surimi_model_mcp.tools import gridded as nc_tools


def test_a_run_output_file_is_marked_surimi():
    listing = {"files": [{"uri": "s3://b/output_creator/experiments/x/x.nc",
                          "kind": "data", "simulation_id": "9eeadb70", "model": "output_creator"}]}
    out = nc_tools.mark_origin(listing)
    assert out["files"][0]["origin"] == "surimi-run"


def test_a_file_with_no_simulation_id_is_marked_external():
    listing = {"files": [{"uri": "s3://b/netcdf/ecoocean_bd30cm.nc", "kind": "data"}]}
    out = nc_tools.mark_origin(listing)
    assert out["files"][0]["origin"] == "external-or-reference"


def test_an_external_entry_says_not_to_present_it_as_surimi_output():
    listing = {"files": [{"uri": "s3://b/netcdf/ecoocean_bd30cm.nc", "kind": "data"}]}
    out = nc_tools.mark_origin(listing)
    note = out["files"][0]["origin_note"].lower()
    assert "not surimi model output" in note


def test_a_mask_is_left_alone():
    listing = {"files": [{"uri": "s3://b/netcdf/ar6.nc", "kind": "mask"}]}
    out = nc_tools.mark_origin(listing)
    assert "origin" not in out["files"][0]


def test_the_listing_is_not_mutated():
    listing = {"files": [{"uri": "s3://b/x.nc", "kind": "data"}]}
    nc_tools.mark_origin(listing)
    assert "origin" not in listing["files"][0]


def test_a_preset_origin_is_not_overwritten():
    """_reference_sample_files() (the bundled data/netcdf-samples/ scan)
    already knows a file it finds is real SURIMI output and presets
    origin/origin_note itself, since that directory is trustworthy in a way
    the simulation_id-or-external default cannot tell from the entry alone.
    mark_origin must leave that preset alone rather than recompute it."""
    listing = {"files": [{
        "uri": "/some/local/dir/westmed-2013-2017-biomass-catch-sales.nc",
        "kind": "data", "origin": "surimi-sample", "origin_note": "custom"}]}
    out = nc_tools.mark_origin(listing)
    assert out["files"][0]["origin"] == "surimi-sample"
    assert out["files"][0]["origin_note"] == "custom"
