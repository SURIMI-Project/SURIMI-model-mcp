"""All server code lives in the package (2026-10-13)."""
import os

from surimi_model_mcp.paths import APP_ROOT


def test_no_python_module_sits_at_the_repository_root():
    stray = sorted(f for f in os.listdir(APP_ROOT) if f.endswith(".py"))
    assert stray == [], f"move these into surimi_model_mcp/: {stray}"


def test_every_package_folder_is_a_package():
    for dirpath, dirnames, filenames in os.walk(APP_ROOT / "surimi_model_mcp"):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        if any(f.endswith(".py") for f in filenames):
            assert "__init__.py" in filenames, f"{dirpath} has modules but no __init__.py"
