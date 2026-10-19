"""The read-only guarantee is the point of this module, so it is tested first."""
import datetime
import logging
import pathlib

import pytest

from surimi_model_mcp.tools import runs as sim_tools
from surimi_model_mcp.simulations import store

# Two vocabularies, because either could be reached from this package:
# boto3 verbs (in case anything ever reaches past s3fs) and the actual s3fs
# write surface that _fs()/_read_only() wraps.
MUTATORS = (
    # boto3 vocabulary
    "put_object", "delete_object", "delete_objects", "copy_object",
    "put_bucket", "create_bucket", "upload_file", "upload_fileobj",
    # s3fs vocabulary - the real write methods S3FileSystem exposes
    ".put_file(", ".rm_file(", ".rmdir(", ".makedirs(", ".copy(",
    ".merge(", ".mv(", ".touch(", ".pipe_file(", ".write_bytes(", ".rm(",
)


def test_store_exposes_only_read_verbs():
    public = {n for n in dir(store) if not n.startswith("_")}
    assert {"find", "get_text", "presign"} <= public
    assert not {"put", "delete", "write", "upload", "remove"} & public


def _guarded_files() -> list[pathlib.Path]:
    """Every file the read-only guarantee must hold for: the whole
    `simulations` package, plus `tools/runs.py` (imported here as
    `sim_tools`), the module that calls `store` for the run tools.

    Uses `sim_tools.__file__` (the real, absolute path of the already-
    imported module), not a bare relative `pathlib.Path("sim_tools.py")`.
    The relative form resolves against the process CWD, not the repo root -
    run from anywhere else (e.g. `/tmp`) it points at a file that does not
    exist, and a `continue`-on-missing guard let that pass silently instead
    of failing. Reproduced: a poisoned `runs.py` containing
    `store._fs().rm_file('x')` still made both guard tests pass when pytest
    was invoked from outside the repo root.
    """
    pkg = pathlib.Path(store.__file__).parent
    return list(pkg.rglob("*.py")) + [pathlib.Path(sim_tools.__file__)]


#: `os.`-qualified calls are the local filesystem, not S3. simulations/cache.py
#: legitimately creates and replaces files in its on-disk cache directory, and
#: `os.makedirs(` would otherwise trip the `.makedirs(` needle purely on the
#: dot. Stripping these before the scan keeps the guard aimed at what it is for
#: - a write verb invoked on an s3fs filesystem object - instead of forcing a
#: contrived idiom on local writes. `fs.makedirs(` is still caught; see
#: test_write_verb_guard_still_catches_a_filesystem_mutator.
_LOCAL_FS_CALLS = ("os.makedirs(", "os.rm(", "os.rmdir(", "os.replace(")


def _strip_local_fs(src: str) -> str:
    for call in _LOCAL_FS_CALLS:
        src = src.replace(call, "")
    return src


def test_package_contains_no_mutating_s3_call():
    """A grep-level guarantee: no write verb appears anywhere in the package."""
    offenders = []
    for f in _guarded_files():
        src = _strip_local_fs(f.read_text())
        for m in MUTATORS:
            if m in src:
                offenders.append(f"{f}: {m}")
    assert offenders == [], offenders


def test_write_verb_guard_still_catches_a_filesystem_mutator():
    """The local-fs exemption must not blunt the guard it is carved out of."""
    src = _strip_local_fs("fs = _fs()\nfs.makedirs('project-surimi/x')\n")
    assert ".makedirs(" in src, "an s3fs write verb must still be visible"
    for needle in (".rm(", ".put_file(", ".mv("):
        assert needle in _strip_local_fs(f"fs{needle}'x')")


def test_boto3_is_not_imported():
    """The image ships s3fs, not boto3. Importing boto3 would work locally and
    fail in the container - fail here instead."""
    for f in _guarded_files():
        assert "import boto3" not in f.read_text(), f


def test_write_verb_guard_covers_sim_tools_regardless_of_process_cwd(tmp_path, monkeypatch):
    """Reproduces the reported gap directly: chdir somewhere with no
    `sim_tools.py` at all (as a relative pathlib.Path("sim_tools.py")
    would see from anywhere but the repo root) and confirm the guarded file
    list still contains the real, absolute path of `tools/runs.py`."""
    monkeypatch.chdir(tmp_path)
    guarded = _guarded_files()
    assert pathlib.Path(sim_tools.__file__) in guarded
    assert all(f.is_absolute() for f in guarded)


class _FakeUnderlyingFS:
    """Stands in for a raw s3fs.S3FileSystem: exposes both read and write
    verbs, so wrapping it in _read_only() is the only thing stopping a caller
    from reaching the write ones."""

    def find(self, *a, **k):
        return []

    def info(self, *a, **k):
        return {}

    def cat_file(self, *a, **k):
        return b""

    def sign(self, *a, **k):
        return "https://example.invalid/signed"

    def put_file(self, *a, **k):
        raise AssertionError("put_file must never be reachable")

    def rmdir(self, *a, **k):
        raise AssertionError("rmdir must never be reachable")

    def write_bytes(self, *a, **k):
        raise AssertionError("write_bytes must never be reachable")


def test_read_only_wrapper_refuses_write_verbs():
    """The architectural guard: even though the underlying object has these
    methods, _read_only() must not expose them."""
    wrapped = store._read_only(_FakeUnderlyingFS())
    for name in ("put_file", "rmdir", "write_bytes"):
        with pytest.raises(AttributeError):
            getattr(wrapped, name)


def test_read_only_wrapper_allows_read_verbs():
    wrapped = store._read_only(_FakeUnderlyingFS())
    for name in ("find", "info", "cat_file", "sign"):
        assert callable(getattr(wrapped, name))


def test_read_only_wrapper_does_not_leak_the_underlying_filesystem():
    """A named-verb blocklist is not enough if the wrapped object is still
    reachable through some other attribute (e.g. `._fs`). The filesystem must
    be closed over, not stored, so there is no detour back to it."""
    wrapped = store._read_only(_FakeUnderlyingFS())
    for name in ("_fs", "_wrapped", "fs"):
        with pytest.raises(AttributeError):
            getattr(wrapped, name)
    assert vars(wrapped) == {}


class _FakeS3FS:
    """A minimal, in-memory stand-in for s3fs.S3FileSystem covering just the
    four calls store.py makes, used to contract-test find/get_text/presign
    without network access or credentials."""

    def __init__(self, listing, contents, fail_paths=()):
        self._listing = listing          # {full_path: {"size": int, "LastModified": dt}}
        self._contents = contents         # {full_path: bytes}
        self.sign_calls = []
        # Paths that `find()` reports as present but `info()` refuses to
        # stat - the per-object failure store.find() must surface, not
        # silently drop.
        self._fail_paths = list(fail_paths)

    def find(self, path):
        names = [p for p in self._listing if p.startswith(path)]
        names += [p for p in self._fail_paths if p.startswith(path)]
        return names

    def info(self, path):
        if path in self._fail_paths:
            raise OSError(f"could not stat {path}")
        return self._listing[path]

    def cat_file(self, path, start=0, end=None):
        data = self._contents[path]
        return data[start:end] if end is not None else data[start:]

    def sign(self, path, expiration=900):
        self.sign_calls.append((path, expiration))
        return f"https://example.invalid/{path}?exp={expiration}"


def _install_fake_fs(monkeypatch, fake):
    # Route through _read_only() too, so these contract tests also assert the
    # store module never calls anything outside the allowed verbs.
    monkeypatch.setattr(store, "_fs", lambda: store._read_only(fake))


def test_find_returns_bucket_relative_keys_with_exact_shape(monkeypatch):
    lm = datetime.datetime(2026, 1, 15, 12, 0, 0)
    fake = _FakeS3FS(
        listing={
            f"{store.BUCKET}/runs/2026/results.json": {"size": 42, "LastModified": lm},
            f"{store.BUCKET}/runs/2026/summary.txt": {"size": 7, "LastModified": lm},
        },
        contents={},
    )
    _install_fake_fs(monkeypatch, fake)

    result, failures = store.find("runs/2026")

    assert failures == []
    assert len(result) == 2
    for row in result:
        assert set(row.keys()) == {"key", "size", "last_modified"}
        assert not row["key"].startswith(store.BUCKET)
    keys = {row["key"] for row in result}
    assert keys == {"runs/2026/results.json", "runs/2026/summary.txt"}
    sizes = {row["key"]: row["size"] for row in result}
    assert sizes["runs/2026/results.json"] == 42
    assert all(row["last_modified"] == lm.isoformat() for row in result)


def test_find_reports_failures_without_dropping_the_objects_that_resolved(monkeypatch):
    """A per-object fs.info() failure must not silently shrink the returned
    list - the caller (discovery.scan) needs to know the listing may be
    short, not just see fewer rows than actually exist."""
    lm = datetime.datetime(2026, 1, 15, 12, 0, 0)
    good_path = f"{store.BUCKET}/runs/2026/results.json"
    bad_path = f"{store.BUCKET}/runs/2026/broken.csv"
    fake = _FakeS3FS(
        listing={good_path: {"size": 42, "LastModified": lm}},
        contents={},
        fail_paths=[bad_path],
    )
    _install_fake_fs(monkeypatch, fake)

    objects, failures = store.find("runs/2026")

    assert len(objects) == 1
    assert objects[0]["key"] == "runs/2026/results.json"
    assert len(failures) == 1
    assert "broken.csv" in failures[0]
    assert "could not stat" in failures[0]


def test_find_still_logs_a_warning_when_objects_fail_to_stat(monkeypatch, caplog):
    """The pre-existing log line stays useful - `failures` is an addition,
    not a replacement."""
    lm = datetime.datetime(2026, 1, 15, 12, 0, 0)
    good_path = f"{store.BUCKET}/runs/2026/results.json"
    bad_path = f"{store.BUCKET}/runs/2026/broken.csv"
    fake = _FakeS3FS(
        listing={good_path: {"size": 42, "LastModified": lm}},
        contents={},
        fail_paths=[bad_path],
    )
    _install_fake_fs(monkeypatch, fake)

    with caplog.at_level(logging.WARNING):
        store.find("runs/2026")

    assert any("broken.csv" in r.getMessage() for r in caplog.records)


def test_get_text_reports_truncated_when_over_cap(monkeypatch):
    key = "runs/2026/log.txt"
    long_content = b"x" * 100
    fake = _FakeS3FS(
        listing={},
        contents={f"{store.BUCKET}/{key}": long_content},
    )
    _install_fake_fs(monkeypatch, fake)

    text, truncated = store.get_text(key, max_bytes=10)

    assert truncated is True
    assert len(text) == 10


def test_get_text_reports_not_truncated_under_cap(monkeypatch):
    key = "runs/2026/log.txt"
    short_content = b"hello"
    fake = _FakeS3FS(
        listing={},
        contents={f"{store.BUCKET}/{key}": short_content},
    )
    _install_fake_fs(monkeypatch, fake)

    text, truncated = store.get_text(key, max_bytes=65536)

    assert truncated is False
    assert text == "hello"


def test_presign_passes_expiration_through_to_sign(monkeypatch):
    key = "runs/2026/artifact.nc"
    fake = _FakeS3FS(listing={}, contents={})
    _install_fake_fs(monkeypatch, fake)

    url = store.presign(key, expires_in=1800)

    assert fake.sign_calls == [(f"{store.BUCKET}/{key}", 1800)]
    assert "exp=1800" in url


def test_client_pins_sigv4_because_minio_rejects_sigv2_presigned_urls(monkeypatch):
    """botocore defaults to SigV2 for presigning through s3fs; MinIO 403s
    every SigV2-signed URL. Losing config_kwargs here silently breaks every
    presigned link (reports, plots, NetCDF) in production while every other
    read (find/get_text) keeps working, since only presign() is signature-
    version sensitive. No network/credentials: the S3FileSystem constructor
    itself is faked so nothing real is contacted."""
    import s3fs

    captured = {}

    class _FakeS3FileSystem:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(s3fs, "S3FileSystem", _FakeS3FileSystem)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example.invalid")

    store._fs()

    assert captured.get("config_kwargs", {}).get("signature_version") == "s3v4"


def test_fs_wraps_a_client_construction_failure_in_simstoreerror(monkeypatch):
    """store's whole contract - and sim_tools._summarize_run's
    transient-vs-deterministic split downstream of it - depends on every
    storage failure surfacing as SimStoreError specifically. Before this
    fix, s3fs.S3FileSystem(...) was called UNGUARDED inside _fs(): a
    failure during client construction (the realistic trigger being an STS
    token expiring mid-credential-resolution, exactly the kind of thing
    likeliest right after a cold pod restart) would raise whatever
    exception type s3fs/botocore happened to produce, not SimStoreError -
    and reach parsers.describe's blanket except as an ordinary,
    indistinguishable-from-genuine parse_error."""
    import s3fs

    class _FakeS3FileSystem:
        def __init__(self, **kwargs):
            raise RuntimeError("STS token expired mid-construction")

    monkeypatch.setattr(s3fs, "S3FileSystem", _FakeS3FileSystem)
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.example.invalid")

    with pytest.raises(store.SimStoreError):
        store._fs()


def test_find_get_text_presign_all_surface_a_construction_failure_as_simstoreerror(monkeypatch):
    """The _fs() guard above is one line of defence; find/get_text/presign
    now call _fs() INSIDE their own try too, so the contract holds even if
    a future _fs() edit reopens the leak - reproduced directly here rather
    than trusting the single _fs()-level test to cover all three callers."""
    def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(store, "_fs", boom)

    with pytest.raises(store.SimStoreError):
        store.find("runs/2026")
    with pytest.raises(store.SimStoreError):
        store.get_text("runs/2026/log.txt")
    with pytest.raises(store.SimStoreError):
        store.presign("runs/2026/artifact.nc")


# ---------------------------------------------------------------------------
# An expired/rotated S3 credential must be named as such
# ---------------------------------------------------------------------------
# On EDITO the server runs on an Onyxia STS token that lives ~24h. When it
# lapses, MinIO answers reads with "The Access Key Id you provided does not
# exist in our records" (a rotated/invalid key) or an ExpiredToken fault.
# Wrapped as a bare "could not read <key>: <raw boto text>", that is
# indistinguishable to a caller from the file being absent -- so an LLM
# reports "the data isn't available" and a benchmark scores it as the MODEL
# failing. Observed 2026-08-10: 56 of 104 stock-assessment turns across 8
# models were graded FAIL for correctly refusing to invent numbers the server
# could not fetch. The message must say the credentials are the problem and
# what fixes it, so an operator and a model both read it correctly.

def test_expired_credential_error_is_named(monkeypatch):
    class _Boom:
        def cat_file(self, *a, **k):
            raise OSError("The Access Key Id you provided does not exist in "
                          "our records.")
    monkeypatch.setattr(store, "_fs", lambda: _Boom())
    with pytest.raises(store.SimStoreError) as ei:
        store.get_text("surimi-cmsy/Simulations/x/out.csv")
    msg = str(ei.value).lower()
    assert "credential" in msg
    assert "relaunch" in msg          # tells the operator the remedy
    assert "not that the file is missing" in msg


def test_expired_token_fault_is_named(monkeypatch):
    class _Boom:
        def cat_file(self, *a, **k):
            raise OSError("An error occurred (ExpiredToken): The provided "
                          "token has expired.")
    monkeypatch.setattr(store, "_fs", lambda: _Boom())
    with pytest.raises(store.SimStoreError) as ei:
        store.get_text("k")
    assert "credential" in str(ei.value).lower()


def test_ordinary_read_failure_is_not_relabelled(monkeypatch):
    # A genuine missing-object error must NOT be dressed up as a credential
    # problem -- that would send an operator chasing a token that is fine.
    class _Boom:
        def cat_file(self, *a, **k):
            raise FileNotFoundError("The specified key does not exist.")
    monkeypatch.setattr(store, "_fs", lambda: _Boom())
    with pytest.raises(store.SimStoreError) as ei:
        store.get_text("k")
    assert "credential" not in str(ei.value).lower()
