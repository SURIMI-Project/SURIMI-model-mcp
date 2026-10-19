"""Binary artifacts are served by this server, not by a presigned MinIO URL.

`fs.sign()` HMACs a remote URL with the S3 credentials, so it is the one read
path a local cache cannot rescue - it died with the STS token while every other
read kept working. Serving the bytes ourselves removes that asymmetry.

This is not a new exposure: text artifacts were already returned inline and the
ingress is unauthenticated. Path safety comes from the allowlist, not from
sanitising the filename - a name that is not an artifact of that run is refused
before anything touches the filesystem.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.simulations import cache


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from surimi_model_mcp.app_factory import build_artifact_routes
    from starlette.applications import Starlette
    monkeypatch.setattr(cache, "_CACHE_DIR", str(tmp_path / "c"))
    return TestClient(Starlette(routes=build_artifact_routes()))


def _run(keys):
    """_find_run returns (run, degradation) - both, always."""
    return ({"simulation_id": "sim-1", "keys": keys, "model": "surimi-cmsy"},
            {"stale": False, "note": None, "partial": False,
             "partial_note": None})


def test_serves_bytes_for_a_real_artifact(client, monkeypatch):
    key = "surimi-cmsy/Simulations/sim-1/kobe.png"
    monkeypatch.setattr(sim_tools, "_find_run", lambda sid: _run([key]))
    monkeypatch.setattr(cache, "fetch", lambda k: b"\x89PNG\r\n\x1a\n binary")
    r = client.get("/artifact/sim-1/kobe.png")
    assert r.status_code == 200
    assert r.content == b"\x89PNG\r\n\x1a\n binary"
    assert r.headers["content-type"].startswith("image/png")


def test_filename_not_in_the_run_is_refused(client, monkeypatch):
    monkeypatch.setattr(sim_tools, "_find_run",
                        lambda sid: _run(["surimi-cmsy/Simulations/sim-1/a.csv"]))
    called = []
    monkeypatch.setattr(cache, "fetch", lambda k: called.append(k) or b"")
    r = client.get("/artifact/sim-1/secrets.env")
    assert r.status_code == 404
    assert called == [], "must refuse before any fetch"


def test_traversal_filename_is_refused_by_the_allowlist(client, monkeypatch):
    monkeypatch.setattr(sim_tools, "_find_run",
                        lambda sid: _run(["surimi-cmsy/Simulations/sim-1/a.csv"]))
    called = []
    monkeypatch.setattr(cache, "fetch", lambda k: called.append(k) or b"")
    for attempt in ("../../etc/passwd", "..%2f..%2fetc%2fpasswd", "a.csv/../../x"):
        assert client.get(f"/artifact/sim-1/{attempt}").status_code in (404, 400)
    assert called == []


def test_unknown_simulation_is_refused(client, monkeypatch):
    monkeypatch.setattr(sim_tools, "_find_run",
                        lambda sid: (None, {"stale": False, "note": None,
                                            "partial": False, "partial_note": None}))
    assert client.get("/artifact/nope/x.csv").status_code == 404


def test_artifact_tool_returns_a_local_url_not_a_presigned_one(monkeypatch):
    key = "surimi-cmsy/Simulations/sim-1/kobe.png"
    monkeypatch.setattr(sim_tools, "_find_run", lambda sid: _run([key]))
    monkeypatch.setattr(sim_tools.store, "presign",
                        lambda *a, **k: pytest.fail("presign must not be called"))
    out = sim_tools.model_get_artifact("sim-1", "kobe.png")
    assert out["kind"] == "link"
    assert "/artifact/sim-1/kobe.png" in out["url"]
    assert "minio" not in out["url"]
    # The short-lived-link caveat is gone because the link is no longer short-lived.
    assert "expires_in_s" not in out and "expiry_note" not in out
