"""Local prefetch cache for simulation artifacts.

The point of this cache is surviving credential expiry, so the tests that
matter most are the ones that revoke S3 *after* warm-up and assert reads keep
working. See docs 2026-08-10 spec: a dead STS token previously made 8 models
look like they had failed a benchmark when the server simply could not read.
"""
from __future__ import annotations

import pytest

from surimi_model_mcp.simulations import cache


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CACHE_DIR", str(tmp_path / "simcache"))
    return tmp_path / "simcache"


class _FS:
    """Minimal read-only stand-in for the store's s3fs wrapper."""

    def __init__(self, objects: dict[str, bytes], fail: set[str] | None = None):
        self.objects = objects
        self.fail = fail or set()
        self.reads: list[str] = []

    def cat_file(self, path, start=None, end=None):
        key = path.split("/", 1)[1]
        self.reads.append(key)
        if key in self.fail:
            raise OSError("The Access Key Id you provided does not exist in "
                          "our records.")
        b = self.objects[key]
        return b[start:end] if (start is not None or end is not None) else b


def test_cache_stores_and_returns_bytes(cache_dir, monkeypatch):
    fs = _FS({"surimi-cmsy/Simulations/a/out.csv": b"col1,col2\n1,2\n"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    got = cache.fetch("surimi-cmsy/Simulations/a/out.csv")
    assert got == b"col1,col2\n1,2\n"
    assert cache.is_cached("surimi-cmsy/Simulations/a/out.csv")


def test_second_read_does_not_touch_s3(cache_dir, monkeypatch):
    fs = _FS({"k/x.csv": b"data"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    cache.fetch("k/x.csv")
    cache.fetch("k/x.csv")
    assert fs.reads == ["k/x.csv"], "cached read must not go back to S3"


def test_read_survives_credentials_dying_after_warmup(cache_dir, monkeypatch):
    """The whole reason this cache exists."""
    fs = _FS({"k/x.csv": b"important numbers"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    cache.prefetch(["k"], list_keys=lambda p: ["k/x.csv"])
    # Credentials die.
    dead = _FS({"k/x.csv": b"important numbers"}, fail={"k/x.csv"})
    monkeypatch.setattr(cache, "_fs", lambda: dead)
    assert cache.fetch("k/x.csv") == b"important numbers"
    assert dead.reads == [], "must be served from cache, never attempted live"


def test_uncached_key_after_expiry_raises_named_error(cache_dir, monkeypatch):
    dead = _FS({"k/new.csv": b"x"}, fail={"k/new.csv"})
    monkeypatch.setattr(cache, "_fs", lambda: dead)
    with pytest.raises(Exception) as ei:
        cache.fetch("k/new.csv")
    assert "credential" in str(ei.value).lower()


def test_prefetch_records_per_object_failure_without_raising(cache_dir, monkeypatch):
    fs = _FS({"k/a": b"1", "k/b": b"2"}, fail={"k/b"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    res = cache.prefetch(["k"], list_keys=lambda p: ["k/a", "k/b"])
    assert res["cached"] == 1
    assert len(res["errors"]) == 1 and "k/b" in res["errors"][0]["key"]
    assert cache.is_cached("k/a") and not cache.is_cached("k/b")


def test_partial_download_never_becomes_a_cache_hit(cache_dir, monkeypatch):
    """A killed pod must not leave a truncated file that reads as complete."""
    class _Boom(_FS):
        def cat_file(self, path, start=None, end=None):
            raise KeyboardInterrupt("pod killed mid-download")
    monkeypatch.setattr(cache, "_fs", lambda: _Boom({}))
    with pytest.raises(KeyboardInterrupt):
        cache.fetch("k/half.csv")
    assert not cache.is_cached("k/half.csv")
    assert list(cache_dir.glob("*.tmp*")) == [], "no temp file left behind"


def test_keys_with_spaces_and_unicode_are_cacheable(cache_dir, monkeypatch):
    # Real key from the bucket; content-addressing must make the charset moot.
    key = ("surimi-cmsy/Simulations/a/"
           "Hogfish - Florida Keys - East Florida2026-01-08-0903.RnW")
    fs = _FS({key: b"content"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    assert cache.fetch(key) == b"content"


def test_traversal_style_key_cannot_escape_the_cache_dir(cache_dir, monkeypatch):
    fs = _FS({"../../etc/passwd": b"nope"})
    monkeypatch.setattr(cache, "_fs", lambda: fs)
    cache.fetch("../../etc/passwd")
    written = list(cache_dir.iterdir())
    assert len(written) == 1
    assert written[0].parent == cache_dir, "content-addressing must contain it"
