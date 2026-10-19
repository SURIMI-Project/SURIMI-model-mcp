"""Experiment 1c: point stock questions at the server that now holds the data.

The model server declares stock status unanswerable and says nothing about
where it IS answerable. Muse asked it about Northeast Atlantic stocks, was told
no, and stopped -- it never reached the data server's ICES tables, registered
in experiment 1. A CannotAnswer that names no alternative ends the search.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from surimi_model_mcp import sources


def _declarations() -> str:
    return " ".join(open(sources.__file__, encoding="utf-8").read().split())


def test_the_stock_status_refusal_names_the_data_server():
    txt = _declarations()
    i = txt.index("how UNCERTAIN a stock's status is")
    block = txt[i:i + 2600]
    assert "ices-stock-assessment-latest" in block, "must name the dataset by id"
    assert "data server" in block.lower()


def test_it_says_what_that_table_covers_and_what_it_does_not():
    txt = _declarations()
    i = txt.index("how UNCERTAIN a stock's status is")
    block = txt[i:i + 2600]
    assert "Area 27" in block or "Northeast Atlantic" in block
    assert "GSA06" in block or "Mediterranean" in block


def test_the_referral_does_not_promise_a_mediterranean_answer():
    # SURIMI's own case study is GSA06 and ICES does not cover it. Referring a
    # Mediterranean stock question to ICES would trade a refusal for a wrong
    # answer, which the objective penalises five times harder.
    txt = _declarations()
    i = txt.index("how UNCERTAIN a stock's status is")
    block = txt[i:i + 2600]
    assert "not" in block.lower()
