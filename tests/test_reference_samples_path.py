"""The reference sample directory defaults to data/netcdf-samples/ at the repository root (2026-10-13).

The default is built from the module's own location. Moving the module into a package without changing
that would point it one folder too deep, and model_list_files would quietly stop naming the sample.
conftest.py replaces the directory for every other test, so this one reads the default itself.
"""
import importlib
from pathlib import Path

from surimi_model_mcp.tools import gridded as nc_tools

ROOT = Path(__file__).resolve().parents[1]


def test_the_default_is_the_repository_data_folder(monkeypatch):
    monkeypatch.delenv("SURIMI_NETCDF_REFERENCE_DIR", raising=False)
    module = importlib.reload(nc_tools)
    try:
        assert Path(module._REFERENCE_SAMPLES_DIR) == ROOT / "data" / "netcdf-samples"
        assert any(Path(module._REFERENCE_SAMPLES_DIR).glob("*.nc")), "the sample files are missing"
    finally:
        importlib.reload(nc_tools)
