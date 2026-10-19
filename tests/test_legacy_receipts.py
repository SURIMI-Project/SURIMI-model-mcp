"""A receipt issued BEFORE the nc_* -> model_* rename must still verify.

The signature covers the stored payload (which still says nc_top_regions),
not the current tool name, so verification is unaffected by the rename.
"""
import os

from surimi_model_mcp.tools import gridded as nc_tools
from surimi_model_mcp.provenance.receipts import issue_receipt, verify_receipt

SECRET = b"legacy-secret"

# A receipt exactly as the pre-rename server would have issued it.
LEGACY = issue_receipt(
    tool_id="nc_top_regions",
    input_params={"file": "s3://b/x.nc", "var": "biomass", "n": 2, "agg": "mean"},
    output_value=[{"region": "Mediterranean", "value": 1.38}],
    secret=SECRET,
    provenance={"file_sha256": "deadbeef"},
)


def test_legacy_nc_receipt_still_verifies():
    res = verify_receipt(LEGACY, secret=SECRET)
    assert res["verified"] is True
    assert res["reason"] == "ok"


def test_legacy_nc_receipt_verifies_through_the_renamed_tool(monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", SECRET.decode())
    res = nc_tools.model_verify(LEGACY)
    assert res["verified"] is True


def test_tampered_legacy_receipt_still_fails():
    tampered = {**LEGACY, "output_value": [{"region": "Mediterranean", "value": 999.9}]}
    res = verify_receipt(tampered, secret=SECRET)
    assert res["verified"] is False
    assert res["reason"] == "signature_mismatch"


# A receipt issued BEFORE read-freshness was recorded: its provenance has
# file_sha256 and nothing else. Every new key must be optional, or every
# number signed before this change stops verifying.
PRE_FRESHNESS = issue_receipt(
    tool_id="model_top_regions",
    input_params={"file": "s3://b/x.nc", "var": "biomass", "n": 2, "agg": "mean"},
    output_value=[{"region": "Mediterranean", "value": 1.38}],
    secret=SECRET,
    provenance={"file_sha256": "deadbeef", "mask_sha256": "cafe", "var": "biomass"},
)


def test_receipt_without_read_freshness_still_verifies():
    res = verify_receipt(PRE_FRESHNESS, secret=SECRET)
    assert res["verified"] is True
    assert "read_as_of" not in PRE_FRESHNESS["provenance"]


def test_receipt_without_read_freshness_verifies_through_the_tool(monkeypatch):
    monkeypatch.setenv("SURIMI_RECEIPT_SECRET", SECRET.decode())
    assert nc_tools.model_verify(PRE_FRESHNESS)["verified"] is True


def test_adding_read_freshness_to_an_old_receipt_breaks_it():
    """The keys are signed, not decorative: a verifier cannot backfill them."""
    forged = {
        **PRE_FRESHNESS,
        "provenance": {**PRE_FRESHNESS["provenance"], "read_stale": False},
    }
    assert verify_receipt(forged, secret=SECRET)["verified"] is False
