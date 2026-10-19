"""Registers the capability declarations behind `describe_model_capabilities`.

Each declaration names its tools and, for questions the server cannot answer, says where the answer does live. Importing this module registers the declarations.
"""
from surimi_model_mcp import capabilities
from surimi_model_mcp.tools import lookups as _ref
from surimi_model_mcp.tools import runs as _sim

capabilities.register(capabilities.DataSourceCapability(
    source_id="ecoocean-netcdf",
    title="EcoOcean gridded ocean data (NetCDF)",
    domain="ocean",
    description=(
        "Gridded ocean-model output (e.g. biomass, catch) on a lat/lon/time "
        "grid, with named region masks. Analytical tools aggregate server-side "
        "and return compact, pivotable results (top-N regions, per-year cube "
        "slices, trends) -- not raw grid cells. Every numeric result carries a "
        "signed calculation record (receipt)."
    ),
    # `served_period` and `answerable_today` name the window that has data,
    # beside the `cannot_answer` entry for the window that has none.
    coverage={
        "format": "NetCDF/HDF5 on S3",
        "grids": "global 0.25-deg class",
        "axis_period": "1961-01 to 2014-12",
        "served_period": "1961-01 to 2009-12",
        "steps_with_data": "588 of 648 monthly steps; the last 60 are null "
                           "in every cell",
        "answerable_today": (
            "Small-demersal biomass density (bd30cm, g m-2) anywhere the "
            "0.25-degree AR6 mask names a region, for 1961-2009: a trend with "
            "a significance test (model_trend), a per-year series "
            "(model_time_series), a regional ranking (model_top_regions) or a "
            "two-period comparison (model_compare_periods). Judge a file by "
            "the variables in the listing, not by its filename."
        ),
    },
    tools=(
        "model_list_files", "model_list_regions", "model_describe_file",
        "model_list_variables", "model_get_time_range", "model_get_spatial_bounds",
        "model_check_cf_compliance", "model_get_coverage_summary",
        "model_validate_alignment",
        "model_top_regions", "model_time_series", "model_compare_periods",
        "model_trend", "model_verify",
    ),
    # What this gridded file cannot answer, and where the answer does live.
    # The question classes use the experts' wording, because the LLM quotes
    # this text back.
    cannot_answer=(
        capabilities.CannotAnswer(
            question_class=(
                "anything about the SURIMI case-study period, 2013 onward: "
                "recent biomass, catch, discards, prices, or how a fleet did "
                "in a given year"
            ),
            why="not_served",
            detail=(
                "The served file is an external EcoOcean run. Its axis carries "
                "648 monthly steps from 1961-01 to 2014-12, but only 588 hold "
                "data: the last 60 steps, 2010-01 through 2014-12, are null in "
                "every cell. So the record stops at 2009-12 while "
                "northwestern_med_contract.yaml declares 2013-01-01 to "
                "2050-01-01 at monthly resolution. The two do not overlap by "
                "a single step, which is why a question about a recent year "
                "returns nulls that read as a broken tool. Two files DO cover "
                "that period. One is the output of the SURIMI run of "
                "2026-09-15. The other is a 53,181,510-byte sample, "
                "westmed-2013-2017-biomass-catch-sales.nc, bundled with this "
                "server (data/netcdf-samples/) and listed in the default "
                "model_list_files listing, as origin 'surimi-sample' since it "
                "carries no simulation_id. The sample has 49 monthly steps "
                "from 2013-01 to 2017-01 on a 109x82 western-Med grid. "
                "SURIMI_LOCAL_NETCDF_SAMPLES additionally merges in an "
                "operator-supplied local directory when set, for an "
                "alternative or updated copy, marked 'external-or-reference' "
                "since that path is not verified. The served AR6 mask is "
                "0.25-degree global and the sample grid is 1/12 degree, so "
                "alignment refuses that pair rather than answering over the "
                "wrong cells. The bundled mask westmed_full_domain_mask.nc "
                "sits on the sample's own grid and holds one region, "
                "westmed_full_domain, that covers every cell. With it the "
                "regional tools aggregate the sample over the whole western "
                "Mediterranean but not over a sub-area, and this repository "
                "holds no GFCM sub-area mask. The per-species structure is "
                "readable via model_list_variables/model_describe_file, and "
                "select={\"species\": \"<code>\"} on "
                "model_top_regions/model_time_series/model_compare_periods "
                "isolates one species of the sample. A code that appears for "
                "both life stages is refused, and the error names the "
                "positions to select instead."
            ),
            lives_in="surimi-mcp/data/netcdf-samples/westmed-2013-2017-biomass-catch-sales.nc",
        ),
        capabilities.CannotAnswer(
            question_class=(
                "biomass or abundance of one named species, or which species "
                "declined most"
            ),
            why="not_served",
            detail=(
                "Which species make up a functional group IS answerable, on "
                "the DATA server: call data_list_functional_groups there and it "
                "returns the matched group and its species with scientific "
                "name, FAO code and whether each is fished. It needs no "
                "abundance grid, so do not refuse it here -- name that tool. "
                "Per-species ABUNDANCE is the part this server cannot reach. "
                "The one served variable is bd30cm, a SIZE CLASS (demersal "
                "under 30 cm) on (time, lat, lon) with no species axis at all, "
                "so a per-species number cannot be computed from it however the "
                "tools are called. Observed species data exists: MEDITS survey "
                "CPUE holds 950,530 rows across 1,416 species from 1994 to "
                "2021. The western-Med model sample adds a species axis of 94 "
                "entries with 86 distinct species codes, where a code can "
                "appear once per life stage (adult or juvenile)."
            ),
            lives_in=(
                "references/SURIMI-DATA-LAKE/2_Abundance/"
                "2.1_MEDITS CPUE ESTIMATED FAO 37/medits_cpue_all_species_wide.csv"
            ),
        ),
        capabilities.CannotAnswer(
            question_class=(
                "anything about one Mediterranean sub-basin, such as the "
                "Aegean, the Adriatic, or a single GFCM geographical sub-area"
            ),
            why="not_served",
            detail=(
                "The served mask is ar6_ocean_mask_0.25deg.nc and its 15 "
                "regions are AR6 ocean basins. The finest Mediterranean region "
                "on it is the whole Mediterranean, so an Aegean question is "
                "answered over the entire sea with no error raised. Aegean "
                "observations do exist: MEDITS carries between 141 and 154 "
                "hauls a year there in the recent period, with data in "
                "1994-2001, 2003-2006, 2008, 2013, 2014, 2016 and 2018-2021. "
                "Closing this on the grid needs a GFCM sub-area mask, which no "
                "SURIMI repository holds."
            ),
            lives_in=(
                "references/SURIMI-DATA-LAKE/2_Abundance/"
                "2.1_MEDITS CPUE ESTIMATED FAO 37/medits_cpue_all_species_wide.csv"
            ),
        ),
        capabilities.CannotAnswer(
            question_class=(
                "comparing a biomass trend against an environmental driver such "
                "as sea temperature or primary productivity"
            ),
            why="not_collected",
            detail=(
                "No gridded temperature or productivity field exists in any "
                "SURIMI source. The one served variable is bd30cm and nothing "
                "else is on the grid. The nearest material is not a driver "
                "series and must not be presented as one: the FishBase and "
                "SeaLifeBase traits table gives min, max and mean temperature "
                "in degrees C per species over 1,229 species, which describes "
                "where a species lives rather than what the sea did in a given "
                "month."
            ),
        ),
        capabilities.CannotAnswer(
            question_class=(
                "whether a decline was CAUSED by climate change, fishing "
                "pressure or anything else, and what would have happened under "
                "a different policy"
            ),
            why="no_mechanism",
            detail=(
                "Attribution and counterfactuals need an engine nobody in "
                "SURIMI wrote. The four analytical tools here (model_top_regions, "
                "model_time_series, model_compare_periods and model_trend) "
                "rank regions, build series, compare periods and test "
                "trends, which describes what happened and cannot "
                "separate one cause from another. The observed half of such a "
                "question is often answerable and should be given: the trend "
                "in bd30cm over 1961-2009 is one of them. The causal half is "
                "not, and no amount of querying reaches it."
            ),
        ),
    ),
))

capabilities.register(capabilities.DataSourceCapability(
    source_id="surimi-simulations",
    title="SURIMI model simulation runs",
    domain="model-output",
    description=(
        "Runs produced by the SURIMI model ensemble, discovered live from "
        "object storage. Each run is identified by a simulation id and carries "
        "the artifacts its model wrote (reports, plots, inputs, configuration). "
        "Discovery is generic, so a model appears here as soon as it writes its "
        "first run."
    ),
    coverage={"discovery": "live, cached 300s", "access": "read-only"},
    tools=tuple(_sim.TOOLS),
    cannot_answer=(
        capabilities.CannotAnswer(
            question_class=(
                "how catches or DISCARDS drive prices in the market model, or "
                "correlating a discard quantity against a price"
            ),
            why="no_mechanism",
            detail=(
                "The market price law is "
                "`P_new = P_last * (Q_last / Q_new) ^ epsilon` at "
                "references/SURIMI-market/DOCUMENTATION.md:275, where Q is "
                "LANDED catch quantity in kg. There is no discard term in it, "
                "so a discard-to-price correlation cannot be computed from "
                "this model however the data is arranged. Price responds to 1 "
                "quantity and nothing else. Do not prescribe a recipe that "
                "regresses a discard column against price: say the mechanism "
                "does not exist. The catch half is a real relationship and can "
                "be described from the law above."
            ),
            # Empty because the answer exists nowhere. The detail cites the law.
            lives_in="",
        ),
        capabilities.CannotAnswer(
            question_class=(
                "how UNCERTAIN a stock's status is (bounds on B/Bmsy or "
                "F/Fmsy), or a CMSY++ reference point such as Bmsy, Fmsy or MSY"
            ),
            why="not_collected",
            detail=(
                "The CURRENT status and the trend of a Western Mediterranean "
                "stock ARE served: model_run_stock_status reads B/Bmsy and "
                "F/Fmsy for 47 WMed stocks, up to 2013, from the output_creator "
                "run's NetCDF, with each stock's Kobe position. Those are point "
                "values with no uncertainty bounds. The bounds, and the Bmsy, "
                "Fmsy and MSY reference points, exist only in a CMSY++ result "
                "table (CMSY_output.csv / CMSY_output_historical.csv), which "
                "model_get_stock_status/model_get_stock_trajectory read with "
                "their 2.5th/97.5th posterior-percentile bounds; no such table "
                "is in the bucket (cmsy/northwestern_med holds inputs only), so "
                "say the uncertainty is not available; never invent a range. "
                "For a NORTHEAST ATLANTIC stock the answer is on the sibling DATA "
                "server, not here: its ices-stock-assessment-latest dataset "
                "(ICES Stock Database, FAO Area 27, 538 stocks) carries "
                "stock_size with stock_size_description, the blim, bpa, fmsy "
                "and msy_b_trigger reference points, and low_stock_size / "
                "high_stock_size as the published confidence interval. Say so "
                "and point there instead of stopping. It does NOT cover the "
                "Mediterranean, so a Mediterranean stock must not be answered "
                "from ICES. "
                "DO NOT ANSWER FROM THE SURVEY BIOMASS TABLE. "
                "`surimi-biomass-wmed-empirical` on the data server holds "
                "observed survey biomass as `biomass_kt` for 5 species. It is a "
                "measurement of how much fish was seen, not a stock "
                "assessment, and it carries no reference point, so it cannot "
                "say whether a stock is overfished, improving or healthy. "
                "Summing it across species does not make a stock trajectory. "
                "Its `cv_uncertainty` column is the coefficient of variation "
                "of that survey estimate, NOT the uncertainty of a stock "
                "status: quoting it as plus or minus a few per cent on the "
                "status is wrong. Use model_run_stock_status for status, and "
                "name the survey table only as observed biomass if the user "
                "wants it."
            ),
            lives_in="references/SURIMI-CMSY/R_files/AA_CMSY++.R",
        ),
        capabilities.CannotAnswer(
            question_class=(
                "what catches, profits or stock status will be in the COMING "
                "years, under a quota, a closure or any other future scenario"
            ),
            why="not_collected",
            detail=(
                "Two kinds of SURIMI run exist, both written 2026-09-15. The "
                "fisheries_authority model's 35 harvest-control-rule runs "
                "(four CSVs each: mse-assessment, tac, catch-monthly, "
                "biomass-monthly) cover juvenile hake and adult sardine, five "
                "of them projecting to 2026, and bluefin tuna and adult red "
                "mullet to 2017-2018; they carry Blim, Bbase, Fopt, quota, TAC "
                "by fleet and catch. Their files state no units, and Blim sits "
                "on a different scale from Biomass, so say that before "
                "comparing them or calling a red zone. No run reaches 2050 or "
                "carries a scenario label. The output_creator run is a "
                "hindcast: 49 monthly steps from 2013-01 to 2017-01. It carries "
                "gross_catch, live_discards and dead_discards on (time, "
                "species, fleet, lat, lon), biomass on (time, species, lat, "
                "lon), price and sales, plus stock_status and exploitation on "
                "(year, species) for 1950-2013 with 3,008 of 6,144 cells "
                "populated. Read it before saying anything is missing. Two "
                "contracts disagree on the intended horizon, 2024 against "
                "2050, and neither has been reached."
            ),
        ),
    ),
))

capabilities.register(capabilities.DataSourceCapability(
    source_id="surimi-reference",
    title="Consortium reference layer (vocabularies, contract, EwE models)",
    domain="reference",
    description=(
        "External standards SURIMI codes its data against (FAO/ASFIS species "
        "codes, ISSCFG gear/fishery codes, WoRMS taxonomy, FishBase common "
        "names, ISO 3166 countries) plus the case-study contract (area, "
        "period, standards) and EwE ecosystem-model structure. None of this "
        "is SURIMI simulation output - the vocabularies are external "
        "standards the project consumes, the contract and EwE models are "
        "configuration, not results."
    ),
    coverage={
        "discovery": "static reference files, cached after first load",
        "access": "read-only",
        # The simulation's own step is `time_step` in `model_get_contract`,
        # which is separate from the annual rollups of the analytics tools.
        "retrieval_granularity": (
            "model_get_contract's time_step field states the SIMULATION's "
            "own step directly (currently P1M, monthly). The analytics "
            "tools return annual rollups: model_time_series gives one value "
            "per year in by_year and model_top_regions covers one calendar "
            "year, whatever the step of the grid or the contract. "
            "model_time_series returns the monthly steps themselves only "
            "when points_for_years names a window of at most 120 steps, "
            "model_compare_periods takes step indices, and model_trend "
            "reads the whole per-step series."
        ),
    },
    tools=tuple(_ref.TOOLS),
))

capabilities.register(capabilities.DataSourceCapability(
    source_id="meta",
    title="Self-description",
    domain="meta",
    description="Capability introspection for client discovery.",
    tools=("describe_model_capabilities",),
    coverage={},
))
