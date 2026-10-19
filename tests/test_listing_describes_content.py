"""A file listing that names only files gets the right file discarded.

In the expert baseline the assistant refused four questions the tools can
answer, each time writing that the only NetCDF it could see was the
output_creator experiment. The catalogue also held
`ecoocean_..._bd30cm_global_monthly_1961_2015.nc`, whose one variable carries
`long_name: "Biomass Density of Small Demersals <30cm"` and which model_trend
answers "how has small-demersal biomass changed" from in a single call.

The listing gave a URI, a size and a kind. Nothing in it said what the file
HOLDS, so the model dismissed it as external and refused. Carrying each data
file's variables into the listing is what makes it choosable.
"""
import pytest

from surimi_model_mcp.tools import gridded as nc_tools


@pytest.fixture(autouse=True)
def _cached_by_default(monkeypatch):
    """Most tests here exercise the enrichment path, which only runs on a file
    already on disk. The uncached case has its own test and overrides this."""
    monkeypatch.setattr(nc_tools.ncio, "read_freshness",
                        lambda uri: {"cached": True})


def test_a_data_entry_carries_its_variables(monkeypatch):
    listing = {"prefix": "s3://b/netcdf/", "count": 1, "files": [
        {"uri": "s3://b/netcdf/eco_bd30cm.nc", "size": 10, "kind": "data"}]}
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables", lambda p: [
        {"name": "bd30cm", "units": "g m-2",
         "long_name": "Biomass Density of Small Demersals <30cm"}])
    out = nc_tools.describe_listing_contents(listing)
    v = out["files"][0]["variables"]
    assert v == [{"name": "bd30cm", "units": "g m-2",
                  "long_name": "Biomass Density of Small Demersals <30cm"}]


def test_a_mask_entry_is_left_alone(monkeypatch):
    """Masks hold regions, not measured variables. Opening them buys nothing."""
    called = []
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables",
                        lambda p: called.append(p) or [])
    listing = {"files": [{"uri": "s3://b/m.nc", "size": 1, "kind": "mask"}]}
    out = nc_tools.describe_listing_contents(listing)
    assert called == []
    assert "variables" not in out["files"][0]


def test_an_unreadable_file_is_noted_not_fatal(monkeypatch):
    """One bad object must not take down the whole catalogue."""
    def boom(p):
        raise OSError("truncated")
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables", boom)
    listing = {"files": [{"uri": "s3://b/x.nc", "size": 1, "kind": "data"}]}
    out = nc_tools.describe_listing_contents(listing)
    assert out["files"][0]["variables_error"]
    assert "variables" not in out["files"][0]


def test_the_listing_is_not_mutated(monkeypatch):
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables",
                        lambda p: [{"name": "v", "units": "m", "long_name": "V"}])
    listing = {"files": [{"uri": "s3://b/x.nc", "size": 1, "kind": "data"}]}
    nc_tools.describe_listing_contents(listing)
    assert "variables" not in listing["files"][0]


def test_only_the_three_useful_fields_are_carried(monkeypatch):
    """A full describe_file per entry would flood the reply it is meant to help."""
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables", lambda p: [
        {"name": "v", "units": "m", "long_name": "V", "shape": [648, 720, 1440],
         "dims": ["time", "lat", "lon"], "dtype": "float32"}])
    listing = {"files": [{"uri": "s3://b/x.nc", "size": 1, "kind": "data"}]}
    out = nc_tools.describe_listing_contents(listing)
    assert set(out["files"][0]["variables"][0]) == {"name", "units", "long_name"}


def test_an_uncached_file_is_never_downloaded_to_list_it(monkeypatch):
    """model_list_files is the FIRST call an assistant makes.

    The s3 cache is write-once and fetches the WHOLE object, and the EcoOcean
    file is about 2.7 GB. Opening an uncached file here would turn a catalogue
    listing into a multi-gigabyte download before the model sees a single name.
    Enrich what is already on disk and say so for the rest.
    """
    opened = []
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables",
                        lambda p: opened.append(p) or [])
    monkeypatch.setattr(nc_tools.ncio, "read_freshness",
                        lambda uri: {"cached": False})
    listing = {"files": [{"uri": "s3://b/huge.nc", "size": 2_700_000_000,
                          "kind": "data"}]}
    out = nc_tools.describe_listing_contents(listing)
    assert opened == []
    assert "variables" not in out["files"][0]
    assert "not cached" in out["files"][0]["variables_note"]


def test_a_cached_file_is_still_enriched(monkeypatch):
    monkeypatch.setattr(nc_tools.nc_inspect, "list_variables",
                        lambda p: [{"name": "bd30cm", "units": "g m-2",
                                    "long_name": "Small Demersals"}])
    monkeypatch.setattr(nc_tools.ncio, "read_freshness",
                        lambda uri: {"cached": True})
    listing = {"files": [{"uri": "s3://b/eco.nc", "size": 1, "kind": "data"}]}
    out = nc_tools.describe_listing_contents(listing)
    assert out["files"][0]["variables"][0]["name"] == "bd30cm"
