"""Every import in the server resolves, the ones inside functions included (2026-10-13).

In SURIMI-data-mcp's package restructure a function kept importing a helper from the module it had left.
The import ran only when the function was called, a broad except caught the error, and the species lookup
answered "not found" on the live server while every test passed. This walks every import statement in
the server's source, at any depth, and resolves it.
"""
import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"tests", "scripts", ".venv", "benchmark", ".git", ".superpowers"}


def _sources() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if not set(p.relative_to(ROOT).parts) & SKIP)


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                package = _module_name(path) if path.name == "__init__.py" else _module_name(path).rpartition(".")[0]
                module = importlib.util.resolve_name("." * node.level + module, package)
            for alias in node.names:
                yield node.lineno, module, alias.name


CASES = [(str(p.relative_to(ROOT)), line, module, name) for p in _sources() for line, module, name in _imports(p)]


@pytest.mark.parametrize("src,line,module,name", CASES, ids=[f"{c[0]}:{c[1]}:{c[3] or c[2]}" for c in CASES])
def test_the_import_resolves(src, line, module, name):
    loaded = importlib.import_module(module)
    if name and name != "*":
        found = hasattr(loaded, name) or importlib.util.find_spec(f"{module}.{name}") is not None
        assert found, f"{src}:{line}: from {module} import {name}"


def test_the_walk_sees_the_imports_inside_functions():
    # simulations/cache.py imports store inside three functions; if the walk missed nested imports
    # this test file would be checking nothing that matters.
    nested = [c for c in CASES if c[0].endswith("cache.py") and c[2].endswith("simulations") and c[3] == "store"]
    assert len(nested) >= 3, nested
