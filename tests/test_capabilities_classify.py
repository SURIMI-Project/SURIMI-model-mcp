from surimi_model_mcp import capabilities as capabilities


def test_meta_tool_classifies_as_meta():
    """After the rename the meta tool must still bucket as 'meta'.

    Without updating the literal it falls through to the startswith("describe_")
    branch and is silently misclassified as 'fetcher'.
    """
    out = capabilities.describe(["describe_model_capabilities"])
    layers = out["tool_inventory"]["layers"]
    assert "describe_model_capabilities" in layers.get("meta", [])


def test_old_meta_name_is_gone():
    out = capabilities.describe(["describe_model_capabilities"])
    layers = out["tool_inventory"]["layers"]
    assert "describe_capabilities" not in str(layers)


def test_usage_hint_names_the_simulation_tools_and_drops_the_gridded_only_claim():
    """This branch made the meta tool's orientation prose false: it still
    said "gridded-only" and never named model_list_simulations /
    model_describe_simulation / model_get_artifact, so a model asking "what
    CMSY runs exist?" that reads the hint first would decline or redirect.
    drift_check only compares tool tuples and cannot catch stale prose."""
    hint = capabilities.describe(["describe_model_capabilities"])["usage_hint"]
    assert "gridded-only" not in hint
    for name in ("model_list_simulations", "model_describe_simulation",
                 "model_get_artifact"):
        assert name in hint


def test_usage_hint_has_no_stale_nc_prefixed_tool_names():
    """Leftover from the nc_ -> model_ rename: the hint named nine tools
    (nc_list_files, nc_list_variables, ...) that do not exist under those
    names anymore."""
    hint = capabilities.describe(["describe_model_capabilities"])["usage_hint"]
    assert "nc_" not in hint


# ---------- cannot_answer: absence as a quotable fact ----------
# Mirrors the shape defined on the data server (surimi_data_mcp/capabilities.py,
# commit dc84a81e). The dataclass must match across both servers so a reader
# comparing two capability replies does not have to reconcile two vocabularies;
# the guidance prose around it is correctly server-specific and is not mirrored.

def test_cannot_answer_defaults_to_empty():
    """Every existing registration must be untouched by the new field."""
    from surimi_model_mcp import capabilities as capabilities
    c = capabilities.DataSourceCapability(
        source_id="x", title="X", domain="d", description="",
        coverage={}, tools=("t",),
    )
    assert c.cannot_answer == ()


def test_cannot_answer_validates_why():
    """`why` is validated, not merely documented.

    The three values are the whole point: not_served is plumbing, no_mechanism
    needs an engine nobody wrote, not_collected needs data nobody gathered.
    Only that distinction tells a reader whether waiting helps. Left as free
    text they collapse into prose.
    """
    from surimi_model_mcp import capabilities as capabilities
    capabilities.CannotAnswer(
        question_class="q", why="not_served", detail="d", lives_in="somewhere")
    for bad in ("unserved", "NOT_SERVED", "", "no_data"):
        try:
            capabilities.CannotAnswer(question_class="q", why=bad, detail="d")
        except ValueError:
            continue
        raise AssertionError(f"why={bad!r} should have been rejected")


def test_lives_in_defaults_empty_and_is_not_guessed():
    """A no_mechanism gap has nowhere to point, so the default is empty."""
    from surimi_model_mcp import capabilities as capabilities
    c = capabilities.CannotAnswer(
        question_class="q", why="no_mechanism", detail="no engine exists")
    assert c.lives_in == ""


def test_describe_surfaces_cannot_answer_with_no_new_tool():
    """asdict already walks the nested dataclass, so the existing
    describe_model_capabilities carries it with no tool or client change."""
    from surimi_model_mcp import capabilities as capabilities
    capabilities.register(capabilities.DataSourceCapability(
        source_id="probe-cannot-answer", title="T", domain="d",
        description="", coverage={}, tools=("model_list_files",),
        cannot_answer=(capabilities.CannotAnswer(
            question_class="what will catches be in 2050",
            why="no_mechanism",
            detail="no projection engine is exposed by this server"),),
    ))
    try:
        out = capabilities.describe(["model_list_files"])
        blob = str(out)
        assert "no projection engine" in blob
        assert "no_mechanism" in blob
    finally:
        capabilities._REGISTRY = [
            c for c in capabilities._REGISTRY
            if c.source_id != "probe-cannot-answer"]
