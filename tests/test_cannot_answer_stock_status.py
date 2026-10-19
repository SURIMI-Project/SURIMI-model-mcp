"""Naming the tempting substitute is what stops the substitution.

Three expert questions became fabrications the same way. Asked for a stock's
current status, its trend, and its uncertainty, the assistant answered from the
served empirical biomass survey: tonnages for status, a five-species sum for
trend, and a survey coefficient of variation for uncertainty ("plus or minus
7-9 per cent"). Every figure was real and receipt-signed, and none of them is a
stock assessment. B/Bmsy, F/Fmsy and the Kobe posterior live in CMSY++ output
that no tool reaches.

The server declared it cannot say what stock status WILL BE in coming years. It
said nothing about the current assessment, so nothing contradicted the
substitution at the point the model made it.
"""
from surimi_model_mcp import server as server  # noqa: F401  -- registers capabilities on import

from surimi_model_mcp import capabilities as capabilities


def _classes():
    out = []
    for c in capabilities.all_capabilities():
        for ca in (c.cannot_answer or ()):
            out.append((ca.question_class + " " + (ca.detail or "")).lower())
    return out


def test_the_current_stock_assessment_is_declared_unreachable():
    joined = " ".join(_classes())
    assert "b/bmsy" in joined or "bbmsy" in joined
    assert "f/fmsy" in joined or "ffmsy" in joined


def test_it_names_survey_biomass_as_the_wrong_substitute():
    """The whole point: the model reached for this and nothing said no."""
    joined = " ".join(_classes())
    assert "survey" in joined and "not a stock assessment" in joined


def test_it_names_the_coefficient_of_variation_substitution():
    joined = " ".join(_classes())
    assert "coefficient of variation" in joined or "cv" in joined


def test_the_future_scenario_entry_still_exists():
    """The new entry is about the CURRENT assessment and must not replace the
    one about coming years."""
    joined = " ".join(_classes())
    assert "coming" in joined


# --- 2026-09-25: the declarations caught up with the bucket ------------------

def _entry(fragment):
    for c in capabilities.all_capabilities():
        for ca in (c.cannot_answer or ()):
            text = (ca.question_class + " " + (ca.detail or "")).lower()
            if fragment in text:
                return text
    raise AssertionError(f"no cannot_answer entry mentions {fragment!r}")


def test_wmed_status_is_declared_served_and_only_its_uncertainty_is_not():
    """Run 9eeadb70 carries B/Bmsy and F/Fmsy for 47 WMed stocks, served by
    model_run_stock_status. What is still missing is uncertainty: no
    CMSY_output table with posterior bounds exists in the bucket."""
    t = _entry("model_run_stock_status")
    assert "47" in t
    assert "uncertain" in t and "bounds" in t
    assert "zero surimi-cmsy runs" not in " ".join(_classes())


def test_the_coming_years_entry_names_the_harvest_rule_projections():
    """Five fisheries_authority harvest-control-rule runs project to 2026, so
    'the only run is a hindcast' is no longer true."""
    t = _entry("coming")
    assert "harvest-control-rule" in t and "2026" in t
    assert "units" in t
