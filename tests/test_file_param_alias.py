"""One file argument, two spellings, because the surface was split in half.

The six inspection tools took `path` and the five analytic tools took `file`.
An assistant that called model_describe_file(path=X) and then model_trend with
the same keyword got a validation error, and the expert baseline shows exactly
that shape: it inspected a file, failed to analyse it, and refused a question
the tools can answer. model_trend answers Q-E5-1 in one call when the keyword
is right.

Accepting both spellings everywhere is additive: no existing call breaks.
"""
import pytest

from surimi_model_mcp.tools import gridded as nc_tools


INSPECTION = [
    nc_tools.model_describe_file,
    nc_tools.model_list_variables,
    nc_tools.model_get_time_range,
    nc_tools.model_get_spatial_bounds,
    nc_tools.model_check_cf_compliance,
    nc_tools.model_get_coverage_summary,
]


@pytest.mark.parametrize("fn", INSPECTION, ids=lambda f: f.__name__)
def test_an_inspection_tool_accepts_the_analytic_spelling(fn, monkeypatch):
    """`file=` must work where only `path=` did."""
    seen = {}
    monkeypatch.setattr(nc_tools.nc_inspect, fn.__name__[len("model_"):],
                        lambda p: seen.setdefault("p", p) or {"ok": True},
                        raising=False)
    fn(file="s3://b/x.nc")
    assert seen["p"] == "s3://b/x.nc"


@pytest.mark.parametrize("fn", INSPECTION, ids=lambda f: f.__name__)
def test_the_original_spelling_still_works(fn, monkeypatch):
    seen = {}
    monkeypatch.setattr(nc_tools.nc_inspect, fn.__name__[len("model_"):],
                        lambda p: seen.setdefault("p", p) or {"ok": True},
                        raising=False)
    fn("s3://b/x.nc")
    assert seen["p"] == "s3://b/x.nc"


@pytest.mark.parametrize("fn", INSPECTION, ids=lambda f: f.__name__)
def test_neither_spelling_is_a_named_error(fn):
    with pytest.raises(ValueError, match="path"):
        fn()


@pytest.mark.parametrize("fn", INSPECTION, ids=lambda f: f.__name__)
def test_two_different_values_is_a_named_error(fn):
    with pytest.raises(ValueError, match="both"):
        fn(path="a.nc", file="b.nc")
