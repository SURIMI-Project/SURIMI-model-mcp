"""The image carries the whole server package and starts it as a package (2026-10-13).

The Dockerfile used to copy an explicit list of root modules, and a module missing from that list would
have shipped as a container that crash-loops with ModuleNotFoundError. Since the code moved into
surimi_model_mcp/ the image copies the package directory, so a new module is in the image by construction.
"""
from surimi_model_mcp.paths import APP_ROOT

DOCKERFILE = (APP_ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_the_image_copies_the_whole_package():
    assert "COPY surimi_model_mcp/ surimi_model_mcp/" in DOCKERFILE


def test_the_image_starts_the_server_as_a_package():
    assert 'CMD ["python", "-m", "surimi_model_mcp"]' in DOCKERFILE


def test_the_image_copies_no_root_module():
    copies = [l for l in DOCKERFILE.splitlines() if l.startswith("COPY ")]
    assert copies == ["COPY surimi_model_mcp/ surimi_model_mcp/"], copies
