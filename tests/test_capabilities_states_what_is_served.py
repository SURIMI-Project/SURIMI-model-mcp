"""The capability block said what cannot be answered and not what can.

`cannot_answer` correctly refuses the SURIMI case-study period, because the
EcoOcean record stops at 2009-12. In the expert baseline the model generalised
that refusal to questions the file DOES answer: "how has small-demersal biomass
changed in the Mediterranean over the available time period" was refused, and
model_trend answers it from this very file in one call, decreasing at
-0.0038 g m-2 per year with p = 4.3e-73 over 588 valid steps.

Naming the window that HAS data, beside the window that does not, is what stops
a correct refusal from swallowing an answerable question.
"""
from surimi_model_mcp import server as server  # noqa: F401  -- registers the capabilities on import

from surimi_model_mcp import capabilities as capabilities


def _ecoocean():
    return next(c for c in capabilities.all_capabilities()
                if c.source_id == "ecoocean-netcdf")


def test_coverage_names_the_window_that_holds_data():
    cov = _ecoocean().coverage
    assert cov["served_period"] == "1961-01 to 2009-12"


def test_coverage_says_the_tail_is_empty():
    """648 steps on the axis, 588 with data. A caller reading only the axis
    range asks for a year that returns nulls and reads it as a broken tool."""
    cov = _ecoocean().coverage
    assert cov["axis_period"] == "1961-01 to 2014-12"
    assert "588" in str(cov["steps_with_data"])


def test_coverage_says_what_it_can_be_asked():
    """Stated positively, so a refusal about 2013 onward does not generalise."""
    cov = _ecoocean().coverage
    ans = cov["answerable_today"].lower()
    assert "small demersal" in ans or "bd30cm" in ans
    assert "trend" in ans


def _surimi_reference():
    return next(c for c in capabilities.all_capabilities()
                if c.source_id == "surimi-reference")


def test_coverage_states_annual_only_retrieval_beside_the_monthly_step():
    """Q-E3-9: 'what is the time scale' needs the true simulation step -
    model_get_contract's time_step (P1M, monthly) - kept distinct from the
    analytics tools (model_time_series et al.), which only ever hand back
    annual rollups regardless of that underlying step. Conflating the two
    would let the model claim monthly retrieval is available when it is not."""
    cov = _surimi_reference().coverage
    note = cov["retrieval_granularity"].lower()
    assert "annual" in note
    assert "model_time_series" in note
    assert "monthly" in note or "time_step" in note
