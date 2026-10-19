FROM python:3.12-slim

LABEL org.opencontainers.image.title="surimi-model-mcp"
LABEL org.opencontainers.image.description="SURIMI model MCP server. Gridded ocean-data analytics (biomass/catch on lat/lon/time grids + region masks), live-discovered simulation runs, and the SURIMI reference layer (vocabularies, case-study contract, EwE model structure), read from S3/MinIO. Exposes 26 tools: gridded discovery/analytics (model_list_files, model_list_regions, model_describe_file, model_list_variables, model_get_time_range, model_get_spatial_bounds, model_check_cf_compliance, model_get_coverage_summary, model_validate_alignment, model_top_regions, model_time_series, model_compare_periods, model_trend, model_verify), simulation runs and stock status (model_list_simulations, model_describe_simulation, model_get_artifact, model_get_stock_status, model_get_stock_trajectory, model_explain_assessment_field, model_run_stock_status), and reference layer (model_lookup_code, model_search_taxon, model_get_contract, model_describe_ecosystem_model), plus describe_model_capabilities, over streamable-http (/mcp) and SSE (/sse). Analytical outputs are pre-aggregated cube slices carrying HMAC-signed receipts (arxiv 2603.10060). DB-free: no postgres, no CSV bundle."
LABEL org.opencontainers.image.source="https://github.com/SURIMI-Project/SURIMI-model-mcp"
LABEL org.opencontainers.image.url="https://www.surimi-project.eu/"
LABEL org.opencontainers.image.licenses="EUPL-1.2"
LABEL org.opencontainers.image.vendor="SURIMI Project"

WORKDIR /app

# The h5py and netCDF4 wheels bundle their own HDF5 and netCDF libraries, so the image installs no apt packages for them.
# pyyaml parses the case-study contract in surimi_model_mcp/reference/contract.py. Without it, model_get_contract returns an error and the other tools keep working.
RUN pip install --no-cache-dir \
        fastmcp==3.3.1 uvicorn starlette \
        xarray netcdf4 h5py h5netcdf numpy s3fs pyyaml

COPY surimi_model_mcp/ surimi_model_mcp/

EXPOSE 26611

CMD ["python", "-m", "surimi_model_mcp"]
