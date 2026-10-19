"""Tests for the HMAC-signed tool receipt module.

Receipt pattern follows arxiv 2603.10060: every analytical tool returns
{tool_id, input_params, output_value, timestamp, signature}. The signature
is HMAC-SHA256 over a canonical serialization of the other fields, keyed
by a server-side secret. Downstream verify() recomputes and compares.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from decimal import Decimal

import pytest

from surimi_model_mcp.provenance.receipts import (
    ReceiptError,
    _canonical_bytes,
    issue_receipt,
    verify_receipt,
)


SECRET = b"test-secret-do-not-use-in-prod"


# ---------- Issue ----------

def test_issue_receipt_returns_dict_with_all_fields():
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc", "year": 2010, "n": 3},
        output_value=[{"region": "Humboldt", "value": 18.4}],
        secret=SECRET,
    )
    assert set(r.keys()) >= {
        "tool_id", "input_params", "output_value",
        "timestamp", "signature", "version",
    }
    assert r["tool_id"] == "model_top_regions"
    assert r["input_params"] == {"file": "s3://b/x.nc", "year": 2010, "n": 3}
    assert r["output_value"] == [{"region": "Humboldt", "value": 18.4}]


def test_issue_receipt_timestamp_is_iso8601_utc():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    assert r["timestamp"].endswith("Z")
    assert "T" in r["timestamp"]


def test_issue_receipt_signature_is_hex_string_of_expected_length():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    assert isinstance(r["signature"], str)
    # SHA-256 -> 64 hex chars
    assert len(r["signature"]) == 64
    assert all(c in "0123456789abcdef" for c in r["signature"])


def test_issue_receipt_includes_version_for_forward_compat():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    assert r["version"] == 1


def test_issue_receipt_signatures_differ_for_different_outputs():
    r1 = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    r2 = issue_receipt(tool_id="x", input_params={}, output_value=2, secret=SECRET)
    assert r1["signature"] != r2["signature"]


def test_issue_receipt_canonical_serialization_invariant_to_key_order():
    """Same data with different key order must produce same signature."""
    r1 = issue_receipt(
        tool_id="x",
        input_params={"a": 1, "b": 2},
        output_value={"x": 1, "y": 2},
        secret=SECRET,
        timestamp="2026-05-26T12:00:00Z",
    )
    r2 = issue_receipt(
        tool_id="x",
        input_params={"b": 2, "a": 1},
        output_value={"y": 2, "x": 1},
        secret=SECRET,
        timestamp="2026-05-26T12:00:00Z",
    )
    assert r1["signature"] == r2["signature"]


def test_issue_receipt_includes_provenance_extras_when_provided():
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc"},
        output_value=[],
        secret=SECRET,
        provenance={"file_sha256": "abc123", "mask_sha256": "def456"},
    )
    assert r["provenance"] == {"file_sha256": "abc123", "mask_sha256": "def456"}


def test_provenance_changes_signature():
    """Two receipts with same payload but different provenance must sign differently."""
    base = dict(tool_id="x", input_params={}, output_value=1, secret=SECRET,
                timestamp="2026-05-26T12:00:00Z")
    r1 = issue_receipt(**base, provenance={"file_sha256": "a"})
    r2 = issue_receipt(**base, provenance={"file_sha256": "b"})
    assert r1["signature"] != r2["signature"]


# ---------- Verify ----------

def test_verify_receipt_accepts_valid_receipt():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is True
    assert result["reason"] == "ok"


def test_verify_receipt_rejects_tampered_output():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    r["output_value"] = 999  # tamper
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False
    assert result["reason"] == "signature_mismatch"


def test_verify_receipt_rejects_tampered_input_params():
    r = issue_receipt(tool_id="x", input_params={"a": 1}, output_value=1, secret=SECRET)
    r["input_params"] = {"a": 2}
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False


def test_verify_receipt_rejects_tampered_tool_id():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    r["tool_id"] = "y"
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False


def test_verify_receipt_rejects_tampered_provenance():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET,
                      provenance={"file_sha256": "a"})
    r["provenance"]["file_sha256"] = "b"
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False


def test_verify_receipt_rejects_wrong_secret():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    result = verify_receipt(r, secret=b"different-secret")
    assert result["verified"] is False
    assert result["reason"] == "signature_mismatch"


def test_verify_receipt_rejects_missing_signature():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    del r["signature"]
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False
    assert result["reason"] == "missing_field"


def test_verify_receipt_rejects_unknown_version():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    r["version"] = 999
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False
    assert result["reason"] == "unsupported_version"


def test_verify_receipt_constant_time_compare_used():
    """Smoke test: verify() should not raise on different-length signatures."""
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    r["signature"] = "short"
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is False


# ---------- Errors ----------

def test_issue_receipt_raises_on_empty_secret():
    with pytest.raises(ReceiptError):
        issue_receipt(tool_id="x", input_params={}, output_value=1, secret=b"")


def test_issue_receipt_raises_on_non_jsonable_output():
    with pytest.raises(ReceiptError):
        issue_receipt(
            tool_id="x", input_params={},
            output_value={"obj": object()},
            secret=SECRET,
        )


def test_verify_receipt_raises_on_empty_secret():
    r = issue_receipt(tool_id="x", input_params={}, output_value=1, secret=SECRET)
    with pytest.raises(ReceiptError):
        verify_receipt(r, secret=b"")


# ---------- DB-native types (Decimal, datetime) ----------

def test_issue_receipt_handles_decimal_output():
    """Trino/Postgres return Decimal for numeric columns; must serialize."""
    from decimal import Decimal
    r = issue_receipt(
        tool_id="query_data_signed",
        input_params={"sql": "SELECT SUM(x)"},
        output_value={"rows": [{"total": Decimal("42.50")}]},
        secret=SECRET,
    )
    # Decimal becomes its string form (deterministic, exact)
    assert r["output_value"]["rows"][0]["total"] == "42.50"
    assert verify_receipt(r, secret=SECRET)["verified"] is True


def test_issue_receipt_handles_datetime_output():
    import datetime
    r = issue_receipt(
        tool_id="query_data_signed",
        input_params={},
        output_value={"ts": datetime.datetime(2026, 5, 28, 7, 12, 0)},
        secret=SECRET,
    )
    assert r["output_value"]["ts"].startswith("2026-05-28T07:12:00")
    assert verify_receipt(r, secret=SECRET)["verified"] is True


def test_issue_receipt_handles_date_output():
    import datetime
    r = issue_receipt(
        tool_id="x", input_params={},
        output_value={"d": datetime.date(2026, 5, 28)},
        secret=SECRET,
    )
    assert r["output_value"]["d"] == "2026-05-28"


def test_decimal_serialization_is_deterministic():
    """Same Decimal value must produce the same signature across calls."""
    from decimal import Decimal
    ts = "2026-05-28T00:00:00Z"
    r1 = issue_receipt(tool_id="x", input_params={}, output_value=Decimal("1.10"),
                       secret=SECRET, timestamp=ts)
    r2 = issue_receipt(tool_id="x", input_params={}, output_value=Decimal("1.10"),
                       secret=SECRET, timestamp=ts)
    assert r1["signature"] == r2["signature"]


def test_still_raises_on_truly_unserializable():
    """object() has no JSON form; must still raise (regression guard)."""
    with pytest.raises(ReceiptError):
        issue_receipt(tool_id="x", input_params={},
                      output_value={"o": object()}, secret=SECRET)


# ---------- Non-finite floats (NaN / +inf / -inf) ----------
#
# A model file's dead/no-data tail (e.g. this dataset's 2010-2014 gap) yields
# NaN cell values. Before this fix, NaN passed through _coerce unchanged and
# _canonical_bytes serialized it as the non-standard JSON literal `NaN`. The
# receipt was then handed back over MCP, where a standards-compliant JSON
# layer renders `NaN` as `null` -- so the client re-signs a payload with
# `null` where the server signed `NaN`, and verification fails with
# signature_mismatch on a perfectly legitimate answer. That's worse than an
# unsigned answer: it's a false tamper alarm on correct output.
#
# Fix: NaN/+inf/-inf are coerced to None (JSON `null`) at issue time, so the
# bytes that get signed are exactly what any JSON transport will carry.

def test_nan_output_signs_and_verifies_round_trip():
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc", "year": 2012, "n": 3},
        output_value=[float("nan"), float("nan"), float("nan")],
        secret=SECRET,
    )
    assert r["output_value"] == [None, None, None]
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is True
    assert result["reason"] == "ok"


def test_nan_output_survives_a_json_round_trip_before_verify():
    """Simulates the MCP transport: receipt serialized to JSON and back."""
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc", "year": 2012},
        output_value={"values": [float("nan"), 1.5, float("nan")]},
        secret=SECRET,
    )
    transported = json.loads(json.dumps(r))
    result = verify_receipt(transported, secret=SECRET)
    assert result["verified"] is True


def test_positive_infinity_output_signs_and_verifies_round_trip():
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc"},
        output_value={"value": float("inf")},
        secret=SECRET,
    )
    assert r["output_value"]["value"] is None
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is True


def test_negative_infinity_output_signs_and_verifies_round_trip():
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc"},
        output_value={"value": float("-inf")},
        secret=SECRET,
    )
    assert r["output_value"]["value"] is None
    result = verify_receipt(r, secret=SECRET)
    assert result["verified"] is True


def test_nan_in_input_params_also_coerced():
    r = issue_receipt(
        tool_id="x",
        input_params={"threshold": float("nan")},
        output_value=1,
        secret=SECRET,
    )
    assert r["input_params"]["threshold"] is None
    assert verify_receipt(r, secret=SECRET)["verified"] is True


def test_canonical_bytes_raises_receipt_error_not_value_error_on_nan():
    """The hard guard: if a non-finite value ever reaches signing directly
    (bypassing _coerce), it must fail loudly with ReceiptError, never mint
    an unverifiable receipt, and never leak a raw ValueError."""
    with pytest.raises(ReceiptError):
        _canonical_bytes({"x": float("nan")})


def test_canonical_bytes_raises_receipt_error_not_value_error_on_infinity():
    with pytest.raises(ReceiptError):
        _canonical_bytes({"x": float("inf")})
    with pytest.raises(ReceiptError):
        _canonical_bytes({"x": float("-inf")})


# ---------- Regression guard: finite-payload signatures must NOT change ----------
#
# Every receipt already issued for a NaN-free result must still verify
# byte-identically after this fix. This pins a realistic nested payload
# (dict, list, Decimal, datetime, unicode) against its signature computed
# with the pre-fix code, so any accidental change to canonical serialization
# for finite values is caught immediately.

def test_finite_payload_signature_is_unchanged_by_the_nan_fix():
    ts = "2026-08-06T00:00:00Z"
    payload = {
        "regions": [
            {"name": "Møre og Romsdal", "value": Decimal("3.14159")},
            {"name": "Humboldt", "value": Decimal("2.71828")},
        ],
        "count": 2,
        "generated_at": datetime.datetime(2026, 8, 6, 12, 30, 0),
        "note": "café ☂ — mixed units",
        "flags": {"partial": False, "sources": ["s3", "trino"]},
    }
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc", "year": 2009, "n": 3},
        output_value=payload,
        secret=b"pinned-secret-for-regression-guard-do-not-rotate",
        timestamp=ts,
        provenance={"file_sha256": "deadbeefcafe", "mask_sha256": "feedface1234"},
    )
    assert r["signature"] == (
        "30f5aca7566c3ccdb0e26ad93c0120ff44aaf41644e621ebf5368cbec9562a7a"
    )
    assert verify_receipt(
        r, secret=b"pinned-secret-for-regression-guard-do-not-rotate"
    )["verified"] is True
