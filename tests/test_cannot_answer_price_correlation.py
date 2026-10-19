"""The market price law has no discard term, and the model prescribed one.

Asked how to correlate catches AND DISCARDS to prices from the market model,
the assistant returned a confident four-step recipe naming discard columns to
regress against price. The price law is
`P_new = P_last * (Q_last / Q_new) ^ epsilon` at
references/SURIMI-market/DOCUMENTATION.md:275, where Q is landed catch
quantity. Discards never enter it, so the correlation it prescribed cannot be
computed from the model at all.
"""
from surimi_model_mcp import server as server  # noqa: F401

from surimi_model_mcp import capabilities as capabilities


def _joined():
    out = []
    for c in capabilities.all_capabilities():
        for ca in (c.cannot_answer or ()):
            out.append((ca.question_class + " " + (ca.detail or "")).lower())
    return " ".join(out)


def test_the_price_law_is_declared_to_have_no_discard_term():
    j = _joined()
    assert "discard" in j
    assert "price" in j


def test_the_law_itself_is_quoted_so_the_answer_is_checkable():
    assert "q_last" in _joined()
