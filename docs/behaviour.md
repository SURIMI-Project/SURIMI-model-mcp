# How the model server behaves in detail

This page is for developers and for anyone writing a client for SURIMI-model-mcp. It lists the guarantees and edge cases that the [README](../README.md) leaves out. The design is in [model_mcp_architecture.md](../model_mcp_architecture.md).

## Storage is read-only

The tools that list and read runs reach object storage through `surimi_model_mcp/simulations/store.py`. That module allows four operations of the wrapped `s3fs.S3FileSystem`, which are `find`, `info`, `cat_file` and `sign`. The filesystem is held in a closure, so a caller cannot reach the unrestricted object through an attribute, and `__getattr__` refuses any other name.

`tests/test_sim_store.py` also scans the `simulations` package and `tools/runs.py` for boto3 and s3fs write calls. `model_run_stock_status` opens the run's NetCDF file through `surimi_model_mcp/netcdf/io.py` instead. That module downloads the file from the bucket into a local cache and never writes to the bucket.

## Flags for a degraded listing

Every response of `model_list_simulations`, `model_describe_simulation`, `model_get_artifact`, `model_get_stock_status`, `model_get_stock_trajectory` and `model_run_stock_status` carries two pairs of flags, whether it succeeds or returns an error.

- `stale` and `note`: `stale: true` means storage was unreachable at the scan (the EDITO storage token expires about 24 hours after launch) and the server answered from the last good listing. `note` gives the reason. `n_artifacts`, `last_modified` and the artifact `size` then come from that last listing.
- `partial` and `partial_note`: `partial: true` means a fresh scan could not read some objects and left them out. Counts and manifests may be too low, and a "no simulation with id ..." error in this state does not prove that the run is missing.

A client should show these flags to users, for example "23 runs (storage was partly unreachable, the list may be incomplete)".

## Summaries in `model_list_simulations`

Each run in the listing has a `summary`, one line built from the same parser that `model_describe_simulation` uses. For a CMSY++ run it gives the stock, the year range and the biomass type, for example `"Hogfish - Florida Keys / East Florida, 1986-2012, CPUE"`. The run also has a `stock` field, and a client should read `stock` to answer "which stock". For a market run the summary names the configuration files with their row counts and reports how many rows of `local_prices.csv` were read.

- `summary: null` means unknown. The run's model has no parser, the parser found nothing, or it refused a truncated or ambiguous `id_file.csv`. It is never a guess from a file name.
- A summary costs at least one storage read per run and is cached per `simulation_id` until the process restarts. A storage error while fetching is not cached. That run gets `summary: null` for this call, is counted in `summaries_failed` with `summaries_failed_note`, and is tried again on the next call.
- One call summarises at most the 50 newest runs. The others have no `summary` key at all, which differs from `summary: null`, and `summaries_omitted` with `summaries_omitted_note` gives their number.
- The first call after a deploy or restart fetches the summaries through up to 8 concurrent workers, and later calls read the cache.

## Market runs: what the metadata covers

`model_describe_simulation` counts the rows of every CSV file under a market run's `config/` folder and parses `local_prices.csv`. Any other file appears in the run's `artifacts` and can be fetched with `model_get_artifact`, but `metadata` does not describe it. The `artifacts` list holds all of a run's files.

## Reads capped at 64 KiB

A text read stops at 64 KiB, and two parts of the server treat the cap differently.

- The metadata parsers behind `model_describe_simulation` never take a value from a truncated read, because a cut field can still parse as a wrong value (`"1986"` cut to `"19"`). The value is reported as absent and the file is named in the parser's `truncated` list. `local_prices.csv` is the exception, because the parser keeps its complete rows and drops only the cut last line.
- `model_get_artifact` returns what it read, with `truncated: true` and a `truncation_note` that tells the caller to quote the content and never count or total it. The 46,436-row `prices.csv` of a market run comes back as about its first 3,761 rows.

## Two artifacts with the same file name

`model_get_artifact` finds a file by its base name within a run. When two keys in the run share a base name, for example `summary.csv` in two subdirectories, it refuses and names the full key of every candidate. A call cannot yet select one by its full key, while `model_describe_simulation` still lists both files under `artifacts`.
