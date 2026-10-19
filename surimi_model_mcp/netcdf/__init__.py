"""NetCDF code behind the gridded tools of the SURIMI model server.

Modules:
- `io`: opens local and `s3://` files, caches S3 objects on disk, hashes files and lists objects.
- `discovery`: lists the NetCDF files under a prefix and classifies each as data, mask or other.
- `inspect`: reports a file's structure, variables, time range, spatial bounds, CF compliance and coverage.
- `analytics`: ranks regions, builds time series, compares periods and estimates trends.
- `regions`: loads a region mask and checks that it sits on the data file's grid.
"""
