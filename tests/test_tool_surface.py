"""The tool surface is the server's public contract (2026-10-13).

Every client sees each tool's name, description and input schema. The package restructure must leave
all three unchanged, so this compares them with the pinned snapshot in `tests/fixtures/tool_surface.json`.
Regenerate only on purpose:  PYTHONPATH=. python tests/test_tool_surface.py --write
"""
import asyncio
import json
import sys
from pathlib import Path

from surimi_model_mcp import server as server

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tool_surface.json"


def surface() -> list[dict]:
    tools = asyncio.run(server.mcp.list_tools())
    rows = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
    # Through JSON once, so tuples and lists compare the same way the fixture stores them.
    return json.loads(json.dumps(sorted(rows, key=lambda r: r["name"])))


def test_the_same_tools_are_served():
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert [t["name"] for t in surface()] == [t["name"] for t in expected]


def test_each_tool_keeps_its_description_and_schema():
    expected = {t["name"]: t for t in json.loads(FIXTURE.read_text(encoding="utf-8"))}
    for tool in surface():
        assert tool == expected[tool["name"]], f"{tool['name']} changed"


def test_the_server_serves_26_tools():
    assert len(surface()) == 26


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.parent.mkdir(exist_ok=True)
    FIXTURE.write_text(json.dumps(surface(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(f"wrote {len(surface())} tools to {FIXTURE}")
