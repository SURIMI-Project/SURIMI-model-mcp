"""Registry of the data sources this server serves, with a check that the registry and FastMCP list the same tools.

`sources.py` registers the declarations, and `describe` builds the answer of `describe_model_capabilities`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


#: The three reasons an answer can be missing, checked in `CannotAnswer`.
#: `not_served` means the data exists but no tool reaches it, `no_mechanism`
#: means no engine computes it, and `not_collected` means nobody has gathered
#: the data. SURIMI-data-mcp uses the same three values.
_WHY = ("not_served", "no_mechanism", "not_collected")


@dataclass(frozen=True)
class CannotAnswer:
    """One class of question this server cannot answer, with the reason and where the answer lives.

    `why` must be one of `_WHY`, and the constructor raises `ValueError` otherwise. `lives_in` names the place that holds the answer and stays empty when the answer exists nowhere.
    """

    question_class: str
    why: str
    detail: str
    lives_in: str = ""

    def __post_init__(self) -> None:
        if self.why not in _WHY:
            raise ValueError(
                f"why must be one of {_WHY}, got {self.why!r}")


@dataclass(frozen=True)
class DataSourceCapability:
    """One data source: its identity, coverage, the tools that serve it and the questions it cannot answer."""

    source_id: str
    title: str
    domain: str
    description: str
    coverage: dict[str, Any]
    tools: tuple[str, ...]
    cannot_answer: tuple[CannotAnswer, ...] = ()


_REGISTRY: list[DataSourceCapability] = []


def register(spec: DataSourceCapability) -> None:
    """Add `spec` to the registry, replacing any entry with the same `source_id`."""
    global _REGISTRY
    _REGISTRY = [c for c in _REGISTRY if c.source_id != spec.source_id]
    _REGISTRY.append(spec)


def all_capabilities() -> list[DataSourceCapability]:
    """Return a copy of the registered sources."""
    return list(_REGISTRY)


def claimed_tools() -> set[str]:
    """Return the names of all tools that a registered source lists."""
    return {t for cap in _REGISTRY for t in cap.tools}


def describe(tool_names: list[str]) -> dict[str, Any]:
    """Return the registered sources, a tool inventory grouped by layer and by domain, and a usage hint for the LLM."""
    return {
        "data_sources": [asdict(c) for c in _REGISTRY],
        "tool_inventory": {
            "total": len(tool_names),
            "layers": _bucket_by_layer(tool_names),
            "by_domain": _bucket_by_domain(tool_names),
        },
        "usage_hint": (
            "Two independent data paths on this server. Gridded model/ocean "
            "data: call model_list_files to find a NetCDF data file and its "
            "region mask, model_list_variables (or model_describe_file) to "
            "pick a variable, model_list_regions(mask_file) for exact region "
            "names, then an analytic tool - model_top_regions (top-N "
            "regions), model_time_series (per-year rollup), "
            "model_compare_periods, or model_trend. Results are "
            "pre-aggregated and each carries a signed calculation record "
            "(receipt); re-check any number with model_verify. Simulation runs: call "
            "model_list_simulations first for any question about a model "
            "RUN or experiment, model_describe_simulation(simulation_id) for "
            "its artifacts and parsed metadata, and "
            "model_get_artifact(simulation_id, filename) to fetch one - "
            "inline text, or a download link on this server's /artifact "
            "route for other files. Tabular "
            "fisheries/economics data (not gridded, not a run) lives in the "
            "separate surimi-mcp. Reference layer: model_lookup_code(code) "
            "resolves an FAO species / ISSCFG gear / ISO country code to its "
            "meaning (never guess one), model_search_taxon(name) is the "
            "reverse lookup, model_get_contract() answers what SURIMI can "
            "model where/when/in what units, and "
            "model_describe_ecosystem_model() lists/describes the EwE "
            "models. All four are external standards or configuration, not "
            "SURIMI output."
        ),
    }


def drift_check(registered_tool_names: set[str]) -> dict[str, list[str]]:
    """Compare the tools registered on FastMCP with the tools the sources list.

    The result names the tools missing from the registry and the tools missing from FastMCP.
    """
    claimed = claimed_tools()
    return {
        "missing_from_registry": sorted(registered_tool_names - claimed),
        "missing_from_fastmcp": sorted(claimed - registered_tool_names),
    }


def _classify(name: str) -> str:
    if name == "describe_model_capabilities":
        return "meta"
    if name.startswith("list_"):
        return "lookup"
    if name.startswith("query_") or name.startswith("describe_"):
        return "fetcher"
    if name.startswith("search_") or name == "explain_indicator":
        return "search"
    return "recipe"


def _bucket_by_layer(tool_names: list[str]) -> dict[str, list[str]]:
    layers: dict[str, list[str]] = {
        "meta": [], "lookup": [], "fetcher": [], "search": [], "recipe": [],
    }
    for name in tool_names:
        layers.setdefault(_classify(name), []).append(name)
    return layers


def _bucket_by_domain(tool_names: list[str]) -> dict[str, list[str]]:
    by_domain: dict[str, list[str]] = {}
    for cap in _REGISTRY:
        for t in cap.tools:
            by_domain.setdefault(cap.domain, []).append(t)
    unclaimed = [t for t in tool_names if t not in claimed_tools()]
    if unclaimed:
        by_domain["_unclaimed"] = unclaimed
    return by_domain
