"""This server must not refuse what the data server now answers.

Found 2026-09-21 alongside the same staleness on the data server. The
`ecoocean-netcdf` refusal covers "biomass or abundance of one named species,
which species make up a group, or which species declined most". The middle
clause is no longer true of the pair: the data server exposes
`data_list_functional_groups`, which returns the matched group and its species
with scientific name, FAO code and whether each is fished.

Both halves of the original claim are worth keeping apart. bd30cm really has
no species axis, so per-species abundance genuinely cannot come from this
server. Group MEMBERSHIP never needed an abundance grid: it is a vocabulary
lookup, and it lives one server away.

This is the exp1d lesson again. A refusal that names no reachable alternative
ends the model's search, and the fix that worked there was putting the
cross-server pointer where the model actually reads it.
"""
from __future__ import annotations

from surimi_model_mcp import capabilities as capabilities
from surimi_model_mcp import server as server  # noqa: F401  -- import registers the capabilities


def _classes() -> str:
    return " ".join(e.question_class for c in capabilities.all_capabilities()
                    for e in c.cannot_answer)


def _details() -> str:
    return " ".join(e.detail or "" for c in capabilities.all_capabilities()
                    for e in c.cannot_answer)


def test_group_membership_is_not_refused():
    assert "which species make up a group" not in _classes()


def test_the_refusal_names_the_tool_on_the_other_server():
    assert "data_list_functional_groups" in _details()


def test_per_species_abundance_is_still_refused():
    """bd30cm has no species axis; that half of the claim is unchanged."""
    c = _classes()
    assert "biomass or abundance of one named species" in c
