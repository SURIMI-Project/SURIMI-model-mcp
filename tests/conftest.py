"""Pytest fixtures shared across the SURIMI model server tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def tiny_nc(tmp_path: Path) -> str:
    """A tiny CF-compliant 3D netcdf file (time, lat, lon).

    4 timesteps, 3x4 grid (lat x lon).
    Variable `biomass` with known values so tests can assert exact aggregates.
    """
    ntime, nlat, nlon = 4, 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    # Days since epoch for 4 monthly steps in 2010
    time_idx = np.array([0, 31, 59, 90], dtype="float64")

    # Make biomass[t, lat, lon] = t*100 + lat_idx*10 + lon_idx
    biomass = np.zeros((ntime, nlat, nlon), dtype="float32")
    for t in range(ntime):
        for i in range(nlat):
            for j in range(nlon):
                biomass[t, i, j] = t * 100 + i * 10 + j

    ds = xr.Dataset(
        data_vars={
            "biomass": (("time", "lat", "lon"), biomass, {
                "units": "kg m-2",
                "long_name": "Biomass density",
            }),
        },
        coords={
            "time": ("time", time_idx, {"units": "days since 2010-01-01"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
            "lon": ("lon", lon, {"units": "degrees_east"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "Synthetic test dataset",
        },
    )

    path = tmp_path / "tiny.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_no_cf(tmp_path: Path) -> str:
    """Variant with no Conventions attribute, to test CF-compliance fallback."""
    arr = np.arange(2 * 2 * 2, dtype="float32").reshape(2, 2, 2)
    ds = xr.Dataset(
        data_vars={"x": (("time", "lat", "lon"), arr)},
        coords={"time": [0, 1], "lat": [-10.0, 10.0], "lon": [0.0, 90.0]},
    )
    path = tmp_path / "nocf.nc"
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def tiny_nc_multiyear(tmp_path: Path) -> str:
    """6 monthly steps spanning 2009-2010 so year-filtering is observable.

    biomass[t, lat, lon] = (t+1) * 100  (uniform per timestep)
    timesteps: 3 in 2009 (Jan/Feb/Mar) then 3 in 2010 (Jan/Feb/Mar).
    So year=2009 mean = mean(100,200,300)=200; year=2010 = mean(400,500,600)=500.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    # days since 2009-01-01 : 2009 Jan/Feb/Mar then 2010 Jan/Feb/Mar
    time_idx = np.array([0, 31, 59, 365, 396, 424], dtype="float64")
    biomass = np.zeros((6, nlat, nlon), dtype="float32")
    for t in range(6):
        biomass[t, :, :] = (t + 1) * 100.0

    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), biomass, {"units": "kg"})},
        coords={
            "time": ("time", time_idx, {"units": "days since 2009-01-01"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "multiyear.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_noleap_multiyear(tmp_path: Path) -> str:
    """Like tiny_nc_multiyear but with a NON-standard calendar + 1661 epoch.

    Mirrors the real EcoOcean file: units "days since 1661-01-01" + calendar
    "noleap" (365_day). decode_cf_datetime returns cftime objects for this
    calendar, which the old year-extraction mis-decoded. Time written raw
    (decode_times=False on read), 3 steps in 2009 then 3 in 2010, biomass
    uniform per step so year=2009 mean=200, year=2010 mean=500.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    # noleap: 365 days/year. days from 1661-01-01 to 2009-01-01 = (2009-1661)*365
    base = (2009 - 1661) * 365
    time_idx = np.array(
        [base + 0, base + 31, base + 59, base + 365, base + 396, base + 424],
        dtype="float64",
    )
    biomass = np.zeros((6, nlat, nlon), dtype="float32")
    for t in range(6):
        biomass[t, :, :] = (t + 1) * 100.0

    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), biomass, {"units": "kg"})},
        coords={
            "time": ("time", time_idx,
                     {"units": "days since 1661-01-01", "calendar": "noleap"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "noleap_multiyear.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_gap(tmp_path: Path) -> str:
    """Like tiny_nc but timestep t=1 is all-NaN, so a region series has a gap.

    Real EcoOcean series have missing months; nc_trend used to crash on them
    ("cannot convert float NaN to integer" in the Mann-Kendall loop). This
    reproduces a single missing step: East region points = [12.5, NaN, 212.5,
    312.5] -> 3 valid, monotonically increasing.
    """
    ntime, nlat, nlon = 4, 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    time_idx = np.array([0, 31, 59, 90], dtype="float64")
    biomass = np.zeros((ntime, nlat, nlon), dtype="float32")
    for t in range(ntime):
        for i in range(nlat):
            for j in range(nlon):
                biomass[t, i, j] = t * 100 + i * 10 + j
    biomass[1, :, :] = np.nan  # whole second timestep missing

    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), biomass, {"units": "kg m-2"})},
        coords={
            "time": ("time", time_idx, {"units": "days since 2010-01-01"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "gap.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_region_mask(tmp_path: Path) -> str:
    """A region mask matching the tiny_nc grid: 2 regions splitting the longitude.

    Region 1: lon < 0 (i.e. lon indices 0, 1)
    Region 2: lon >= 0 (lon indices 2, 3)
    Mask is a netcdf file with a `region` variable: (n_regions, lat, lon) of weights.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")

    # Region 1: West (lon < 0), Region 2: East (lon >= 0)
    mask = np.zeros((2, nlat, nlon), dtype="float32")
    for j, lon_val in enumerate(lon):
        if lon_val < 0:
            mask[0, :, j] = 1.0
        else:
            mask[1, :, j] = 1.0

    ds = xr.Dataset(
        data_vars={
            "region": (("region_id", "lat", "lon"), mask),
        },
        coords={
            "region_id": ("region_id", [1, 2]),
            "region_name": ("region_id", np.array(["West", "East"], dtype="<U16")),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    path = tmp_path / "mask.nc"
    ds.to_netcdf(path)
    return str(path)


@pytest.fixture
def tiny_nc_empty_tail(tmp_path: Path) -> str:
    """3 years where the LAST year is entirely NaN, the shape of the real file.

    The live EcoOcean file is named ..._1961_2015.nc but 2010-2014 are 100% NaN,
    so "most recent year in the file" lands on empty data. Here: 2008 and 2009
    carry values, 2010 is all-NaN. A correct year-default picks 2009.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    # 2 monthly steps per year for 2008, 2009, 2010
    time_idx = np.array([0, 31, 366, 397, 731, 762], dtype="float64")
    biomass = np.zeros((6, nlat, nlon), dtype="float32")
    biomass[0:2, :, :] = 100.0   # 2008
    biomass[2:4, :, :] = 200.0   # 2009  <- most recent year WITH data
    biomass[4:6, :, :] = np.nan  # 2010  <- empty tail

    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), biomass, {"units": "kg"})},
        coords={
            "time": ("time", time_idx, {"units": "days since 2008-01-01"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "empty_tail.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture(autouse=True)
def _isolate_reference_samples(tmp_path, monkeypatch):
    """Point nc_tools' bundled-reference-sample dir at an empty path by
    default, so the real files under data/netcdf-samples/ (the westmed
    sample + its mask, added for Q-E6-5) don't change model_list_files'
    count/notes for every test that isn't specifically about them. Tests
    that DO want the merge (tests/test_reference_samples.py) monkeypatch
    this explicitly afterwards, which overrides this default within the
    same test.
    """
    # Env var, not setattr: several discovery tests `importlib.reload(nc_tools)`
    # to re-read SURIMI_NETCDF_PREFIX, which would silently undo a setattr by
    # recomputing the real default path on reload.
    monkeypatch.setenv("SURIMI_NETCDF_REFERENCE_DIR",
                        str(tmp_path / "no-reference-samples-here"))
    from surimi_model_mcp.tools import gridded as nc_tools
    monkeypatch.setattr(nc_tools, "_REFERENCE_SAMPLES_DIR",
                         str(tmp_path / "no-reference-samples-here"))


@pytest.fixture(autouse=True)
def _isolate_sim_cache(tmp_path, monkeypatch):
    """Give every test its own simulation cache directory.

    simulations/cache.py defaults to /tmp/surimi-simcache, which is process- and
    run-global: without this, a key cached by one test is served to the next,
    and a test that monkeypatches _fs to assert a read failure silently gets a
    cache hit instead. Both happened the first time the cache was wired in.
    """
    from surimi_model_mcp.simulations import cache
    monkeypatch.setattr(cache, "_CACHE_DIR", str(tmp_path / "simcache"))


@pytest.fixture
def tiny_nc_varying_nan(tmp_path: Path) -> str:
    """A file whose missing pattern is NOT constant along time.

    Mirrors the shape of the real gridded file: a persistent land mask plus a
    tail of steps that carry no data at all. 20 monthly steps on a 4x5 grid.

      - lat row 0 is NaN at every step          -> land mask, 1/4 of each slice
      - steps 16..19 are entirely NaN           -> empty tail, 4/20 of the axis

    True nan_fraction is therefore 0.25 + 0.75 * 0.2 = 0.40, and no single
    slice of the cube equals that: an all-data step reads 0.25 and an empty
    step reads 1.0. A sampler that lands on the last step overstates; one that
    misses the tail understates.
    """
    ntime, nlat, nlon = 20, 4, 5
    arr = np.ones((ntime, nlat, nlon), dtype="float32")
    arr[:, 0, :] = np.nan          # land mask, every step
    arr[16:, :, :] = np.nan        # empty tail

    ds = xr.Dataset(
        data_vars={"biomass": (("time", "lat", "lon"), arr, {"units": "kg m-2"})},
        coords={
            "time": ("time", np.arange(ntime, dtype="float64") * 30.0,
                     {"units": "days since 2000-01-01"}),
            "lat": ("lat", np.linspace(-30.0, 30.0, nlat).astype("float32")),
            "lon": ("lon", np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "varying_nan.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_species(tmp_path: Path) -> str:
    """A 4D file (time, species, lat, lon) on the tiny_nc grid.

    The real EcoOcean catch/biomass variables carry a functional-group axis, so
    the leading dim is not always time and the extra axis is not spatial. Two
    species with values an order of magnitude apart, so an accidental average
    ACROSS species is visible in the number rather than plausible:

      catch[t, 0, :, :] = 100 * (t + 1)   species "cod"
      catch[t, 1, :, :] = 900 * (t + 1)   species "herring"

    3 steps in 2009 then 3 in 2010, same time axis as tiny_nc_multiyear, so
    per-year means are cod 2009=200 / 2010=500 and herring 2009=1800 /
    2010=4500. The cross-species mean is 1000 for 2009 and 2500 for 2010, which
    matches no species and is what the silent broadcast used to return.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    time_idx = np.array([0, 31, 59, 365, 396, 424], dtype="float64")

    catch = np.zeros((6, 2, nlat, nlon), dtype="float32")
    for t in range(6):
        catch[t, 0, :, :] = 100.0 * (t + 1)
        catch[t, 1, :, :] = 900.0 * (t + 1)

    ds = xr.Dataset(
        data_vars={
            "catch": (("time", "species", "lat", "lon"), catch, {"units": "t"}),
        },
        coords={
            "time": ("time", time_idx, {"units": "days since 2009-01-01"}),
            "species": ("species", np.array(["cod", "herring"], dtype="<U16")),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "species.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_species_leading(tmp_path: Path) -> str:
    """Same data as tiny_nc_species but shaped (species, time, lat, lon).

    Dimension order is a file-authoring choice, not a guarantee: `dims[0]` on
    this variable is `species` (size 2), so the old year filter indexed the
    SPECIES axis with time indices and returned cod/herring as if they were
    2009/2010.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    time_idx = np.array([0, 31, 59, 365, 396, 424], dtype="float64")

    catch = np.zeros((2, 6, nlat, nlon), dtype="float32")
    for t in range(6):
        catch[0, t, :, :] = 100.0 * (t + 1)
        catch[1, t, :, :] = 900.0 * (t + 1)

    ds = xr.Dataset(
        data_vars={
            "catch": (("species", "time", "lat", "lon"), catch, {"units": "t"}),
        },
        coords={
            "time": ("time", time_idx, {"units": "days since 2009-01-01"}),
            "species": ("species", np.array(["cod", "herring"], dtype="<U16")),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "species_leading.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_coded_species(tmp_path: Path) -> str:
    """A species axis carrying CODES as a variable, not as a coordinate.

    This is the shape SURIMI's own output contract mandates and the shape the
    western-Med model sample actually has: `species` has no coordinate at all,
    and `species_code` / `species_life_stage` are 1-D variables along it. In
    that file 94 species entries carry 86 distinct codes, so a code is not a
    key: ANE appears three times, once per life stage.

    Four entries, mirroring that:
      idx 0  HKE ''          catch = 100 * (t + 1)
      idx 1  ANE 'adult'     catch = 200 * (t + 1)
      idx 2  ANE 'juvenile'  catch = 300 * (t + 1)
      idx 3  PIL ''          catch = 400 * (t + 1)
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    time_idx = np.array([0, 31, 59, 365, 396, 424], dtype="float64")

    catch = np.zeros((6, 4, nlat, nlon), dtype="float32")
    for t in range(6):
        for s in range(4):
            catch[t, s, :, :] = 100.0 * (s + 1) * (t + 1)

    ds = xr.Dataset(
        data_vars={
            "catch": (("time", "species", "lat", "lon"), catch, {"units": "t"}),
            "species_code": ("species", np.array(["HKE", "ANE", "ANE", "PIL"])),
            "species_life_stage": (
                "species", np.array(["", "adult", "juvenile", ""])),
        },
        coords={
            "time": ("time", time_idx, {"units": "days since 2009-01-01"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "coded_species.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)


@pytest.fixture
def tiny_nc_two_labels(tmp_path: Path) -> str:
    """A fleet axis described by TWO code variables and named by neither.

    The western-Med sample's fleet dim carries fleet_gear_code and
    fleet_country_code. Neither is `fleet_code`, so picking one would be a
    guess, and the positions stay integers.
    """
    nlat, nlon = 3, 4
    lat = np.linspace(-30.0, 30.0, nlat).astype("float32")
    lon = np.linspace(-180.0, 180.0, nlon, endpoint=False).astype("float32")
    catch = np.ones((2, 2, nlat, nlon), dtype="float32")

    ds = xr.Dataset(
        data_vars={
            "catch": (("time", "fleet", "lat", "lon"), catch, {"units": "t"}),
            "fleet_gear_code": ("fleet", np.array(["OTB", "PS"])),
            "fleet_country_code": ("fleet", np.array(["ESP", "FRA"])),
        },
        coords={
            "time": ("time", np.array([0.0, 31.0]), {"units": "days since 2009-01-01"}),
            "lat": ("lat", lat), "lon": ("lon", lon),
        },
        attrs={"Conventions": "CF-1.8"},
    )
    path = tmp_path / "two_labels.nc"
    ds.to_netcdf(path, format="NETCDF4")
    return str(path)
