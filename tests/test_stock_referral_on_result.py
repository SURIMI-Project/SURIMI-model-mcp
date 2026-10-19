"""Experiment 1d: the referral has to ride on the tool RESULT.

1c put "Northeast Atlantic stock status lives on the data server" in a
CannotAnswer declaration. Muse never read it: for a stock question it calls
model_list_simulations, gets a listing of one unrelated run, and stops. The
project's own rule, learned the hard way, is that a capability declaration is
not a guard -- a warning reaches the model only when it rides on the result of
the call it actually makes.

These assert on the note and on where it is attached, without reaching S3: the
listing itself needs a live bucket, and a test that silently skips without one
would let the regression back in.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surimi_model_mcp.tools import runs as sim_tools


def test_the_note_names_the_dataset_and_the_server():
    n = sim_tools._STOCK_STATUS_NOTE
    assert "ices-stock-assessment-latest" in n
    assert "data server" in n.lower()


def test_the_note_sends_the_mediterranean_to_the_run_not_to_ices():
    """Until 2026-09-25 the note said a Mediterranean stock was unanswerable
    and that stock status 'is not in these runs'. Run 9eeadb70 carries B/Bmsy
    and F/Fmsy for 47 WMed stocks; the note now points there, and still
    forbids answering a Mediterranean stock from ICES."""
    n = sim_tools._STOCK_STATUS_NOTE
    assert "model_run_stock_status" in n
    assert "Mediterranean" in n
    assert "not in these runs" not in n
    assert "must not be answered from ICES" in n


def test_the_note_is_attached_to_the_listing_result_unconditionally():
    """Attached to `result` right before it is returned, so an empty listing
    carries it too. The refusal that stopped the search happened on a listing
    with one unrelated run in it."""
    src = open(sim_tools.__file__, encoding="utf-8").read()
    i = src.index("def model_list_simulations")
    body = src[i:src.index("\ndef ", i + 10)]
    assert 'result["stock_status_note"] = _STOCK_STATUS_NOTE' in body
    assert body.index('result["stock_status_note"]') < body.rindex("return result")
    assert "if " not in body.split('result["stock_status_note"]')[0].splitlines()[-1]
