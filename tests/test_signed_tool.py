"""Tests for the with_receipt decorator that wraps any tool function
to automatically produce HMAC-signed receipts alongside the output."""
from __future__ import annotations

import os

import pytest

from surimi_model_mcp.provenance.receipts import verify_receipt
from surimi_model_mcp.provenance.signed_tool import with_receipt, ToolError


SECRET = b"unit-test-secret"


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", SECRET.decode())


# ---------- basic wrapping ----------

def test_wraps_function_returns_value_and_receipt():
    @with_receipt("test_tool")
    def add(a: int, b: int) -> int:
        return a + b

    out = add(2, 3)
    assert "value" in out
    assert "receipt" in out
    assert out["value"] == 5


def test_receipt_is_verifiable_with_same_secret():
    @with_receipt("test_tool")
    def add(a: int, b: int) -> int:
        return a + b

    out = add(2, 3)
    result = verify_receipt(out["receipt"], secret=SECRET)
    assert result["verified"] is True


def test_receipt_records_tool_id():
    @with_receipt("my_special_tool")
    def f() -> str:
        return "hello"

    out = f()
    assert out["receipt"]["tool_id"] == "my_special_tool"


def test_receipt_records_input_params():
    @with_receipt("test_tool")
    def f(x: int, y: int = 10) -> int:
        return x + y

    out = f(5, y=20)
    assert out["receipt"]["input_params"] == {"x": 5, "y": 20}


def test_receipt_records_output_value():
    @with_receipt("test_tool")
    def f(a: int) -> dict:
        return {"doubled": a * 2}

    out = f(7)
    assert out["receipt"]["output_value"] == {"doubled": 14}


# ---------- determinism ----------

def test_different_inputs_produce_different_signatures():
    @with_receipt("test_tool")
    def f(a: int) -> int:
        return a * 2

    s1 = f(1)["receipt"]["signature"]
    s2 = f(2)["receipt"]["signature"]
    assert s1 != s2


def test_same_inputs_at_different_times_produce_different_signatures():
    """Timestamp uniqueness should make signatures differ between calls."""
    import time

    @with_receipt("test_tool")
    def f(a: int) -> int:
        return a * 2

    s1 = f(1)["receipt"]["signature"]
    time.sleep(1.1)
    s2 = f(1)["receipt"]["signature"]
    assert s1 != s2


# ---------- error handling ----------

def test_propagates_function_errors():
    @with_receipt("test_tool")
    def f() -> int:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        f()


def test_raises_tool_error_when_secret_missing(monkeypatch):
    monkeypatch.delenv("SURIMI_RECEIPT_SECRET", raising=False)

    @with_receipt("test_tool")
    def f() -> int:
        return 1

    with pytest.raises(ToolError):
        f()


# ---------- provenance ----------

def test_extracts_provenance_from_return_dict():
    """If the wrapped function returns a dict with a 'provenance' key,
    the decorator should hoist it into the receipt."""
    @with_receipt("test_tool")
    def f() -> dict:
        return {"result": 42, "provenance": {"source": "test"}}

    out = f()
    assert out["receipt"]["provenance"] == {"source": "test"}


def test_provenance_in_receipt_affects_signature():
    @with_receipt("test_tool")
    def f(p: str) -> dict:
        return {"result": 1, "provenance": {"x": p}}

    s1 = f("a")["receipt"]["signature"]
    s2 = f("b")["receipt"]["signature"]
    assert s1 != s2


# ---------- read freshness in provenance ----------

def test_provenance_carries_read_freshness_for_a_file_param(monkeypatch, tiny_nc):
    """A signature over a read with no read time launders a stale read: the
    receipt asserts what the data said without saying WHEN it was read."""
    @with_receipt("test_tool")
    def f(file: str) -> dict:
        return {"result": 1, "provenance": {"file_sha256": "abc"}}

    out = f(tiny_nc)
    prov = out["receipt"]["provenance"]
    assert prov["file_sha256"] == "abc"
    assert prov["read_as_of"].endswith("Z")
    assert prov["read_stale"] is False


def test_provenance_marks_a_stale_cache_read(monkeypatch, tiny_nc, tmp_path):
    import os as _os
    import shutil
    import time as _time
    import types

    from surimi_model_mcp.netcdf import io as ncio

    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://minio.test")
    monkeypatch.setattr(ncio, "_CACHE_DIR", str(tmp_path / "cache"))

    class _FakeFS:
        def __init__(self, **kwargs):
            pass

        def get_file(self, uri, local):
            shutil.copy(tiny_nc, local)

    monkeypatch.setitem(
        __import__("sys").modules, "s3fs",
        types.SimpleNamespace(S3FileSystem=_FakeFS),
    )

    uri = "s3://b/stale.nc"
    ncio._localize(uri)
    local = ncio._cache_path(uri)
    aged = _time.time() - (ncio.CACHE_STALE_AFTER_S + 60)
    _os.utime(local, (aged, aged))

    @with_receipt("test_tool")
    def f(file: str) -> dict:
        return {"result": 1, "provenance": {"file_sha256": "abc"}}

    prov = f(uri)["receipt"]["provenance"]
    assert prov["read_stale"] is True
    assert uri in prov["read_note"]


def test_read_freshness_covers_every_uri_param(monkeypatch, tiny_nc, tiny_region_mask):
    """Both `file` and `mask_file` feed the number, so both must be dated."""
    @with_receipt("test_tool")
    def f(file: str, mask_file: str) -> dict:
        return {"result": 1, "provenance": {"file_sha256": "a"}}

    prov = f(tiny_nc, tiny_region_mask)["receipt"]["provenance"]
    assert set(prov["read_sources"]) == {tiny_nc, tiny_region_mask}


def test_freshness_is_not_added_when_no_uri_params():
    """A tool that reads no resource gets no read_* keys, so a verifier can
    never mistake silence for a dated read."""
    @with_receipt("test_tool")
    def f(a: int) -> dict:
        return {"result": a, "provenance": {"x": 1}}

    prov = f(1)["receipt"]["provenance"]
    assert "read_as_of" not in prov
    assert "read_stale" not in prov


def test_freshness_does_not_break_a_tool_without_provenance():
    """Adding freshness must not invent a provenance block where the tool
    returned none: an unsigned-provenance receipt stays exactly as before."""
    @with_receipt("test_tool")
    def f(file: str) -> int:
        return 3

    out = f("/nonexistent/x.nc")
    assert out["value"] == 3
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_receipt_with_freshness_still_verifies(tiny_nc):
    @with_receipt("test_tool")
    def f(file: str) -> dict:
        return {"result": 1, "provenance": {"file_sha256": "abc"}}

    out = f(tiny_nc)
    assert verify_receipt(out["receipt"], secret=SECRET)["verified"] is True


def test_read_freshness_covers_the_data_file_param(tmp_path, monkeypatch):
    """model_validate_alignment names its grid parameter `data_file`.

    _URI_PARAMS listed only ("file", "mask_file"), so that tool's receipt
    dated the mask and said nothing about the grid it was checked against.
    A receipt that dates half its inputs understates the age of the answer.
    """
    from surimi_model_mcp.provenance import signed_tool as signed_tool
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", "test-secret")
    grid = tmp_path / "grid.nc"
    grid.write_bytes(b"x")
    out = signed_tool._read_freshness({"data_file": str(grid)})
    assert out is not None
    assert out["read_sources"] == [str(grid)]
