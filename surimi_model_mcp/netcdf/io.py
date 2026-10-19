"""Opens NetCDF files from local paths and `s3://` URIs, caches S3 objects on disk and hashes them.

`inspect`, `analytics` and `regions` call this module, so none of them handles the local and S3 cases itself. S3 access uses `s3fs` with the endpoint in `AWS_ENDPOINT_URL` and the credentials in the other `AWS_*` variables.

The S3 cache is written once per URI and never checked against the remote object. `read_freshness` therefore dates every read and flags an old copy, and that date goes into the signed calculation record (the `receipt`).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import xarray as xr


class NetCDFIOError(Exception):
    """Raised on IO failure (missing file, missing S3 config, open error)."""


_S3_PREFIX = "s3://"
_FILE_PREFIX = "file://"
_CACHE_DIR = os.environ.get("SURIMI_NC_CACHE", "/tmp/surimi-s3cache")

#: Age in seconds (24 hours) from which a read off a cached S3 copy is reported as stale.
#: The value matches the roughly 24 hour lifetime of the STS token, after which the server
#: can no longer fetch the object again.
CACHE_STALE_AFTER_S = 24 * 3600


def is_s3_uri(uri: str) -> bool:
    """Return True when `uri` starts with `s3://`."""
    return uri.startswith(_S3_PREFIX)


def _cache_path(uri: str) -> str:
    """Return the local cache path for an `s3://` URI, named by the first 16 hex characters of the SHA-256 of the URI.

    The same URI always maps to the same file, so the cache cannot notice a change to the remote object. `read_freshness` uses the path to date a read.
    """
    key = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"{key}.nc")


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def read_freshness(uri: str) -> dict[str, Any]:
    """Report when the bytes behind `uri` were last fetched and whether that copy is old.

    `as_of` is the modification time of the cache file, which is the download time and not the time of the call. A local path is read from disk on every call, so it is never stale, and a path with no file behind it returns `as_of` None and `cached` False. The fields `as_of`, `stale` and `note` follow `tools/runs.py`.

    Returns `{as_of, stale, note, age_s, cached}`.
    """
    local = _cache_path(uri) if is_s3_uri(uri) else _strip_file_scheme(uri)
    try:
        mtime = os.path.getmtime(local)
    except OSError:
        # No file on disk, so there is no read to date.
        return {"as_of": None, "stale": False, "note": None,
                "age_s": None, "cached": False}

    age = max(0.0, _now() - mtime)
    # Only a cached S3 object can go stale. A local file is the source itself.
    stale = bool(is_s3_uri(uri) and age >= CACHE_STALE_AFTER_S)
    note = None
    if stale:
        note = (
            f"served from a local cache copy of {uri} fetched "
            f"{int(age // 3600)}h ago; the remote object was not re-checked "
            f"and may have changed since"
        )
    return {"as_of": _iso(mtime), "stale": stale, "note": note,
            "age_s": age, "cached": True}


def _now() -> float:
    import time

    return time.time()


def _localize(uri: str) -> str:
    """Return a local filesystem path for `uri`.

    An `s3://` object is downloaded once to the cache file named by `_cache_path`, and later calls reuse it. HDF5 reads need random access, which works on a local file and not on a streamed S3 handle. The cache is never invalidated, and `read_freshness` records the age of the copy instead.
    """
    if not is_s3_uri(uri):
        return _strip_file_scheme(uri)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    local = _cache_path(uri)
    if os.path.exists(local):
        return local
    # The lock lets concurrent first reads of one URI share a single download.
    # The existence check is repeated inside it because a thread that waited
    # may find the file already there.
    with _download_lock(local):
        if os.path.exists(local):
            return local
        fs = _s3_filesystem()
        fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".part")
        os.close(fd)
        try:
            fs.get_file(uri, tmp)        # streams to disk in chunks, bounded RAM
            os.replace(tmp, local)       # atomic; avoids half-downloaded cache hits
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise NetCDFIOError(f"could not download {uri}: {exc}") from exc
    return local


#: One lock for every NetCDF and HDF5 operation in the process. The C libraries
#: are not thread-safe and FastMCP runs sync tools in a thread pool. The lock is
#: reentrant, so a serialised function can call another one.
NC_LOCK = threading.RLock()


def serialized(fn):
    """Wrap `fn` so that it runs while holding `NC_LOCK`.

    The wrapper keeps the signature of `fn`, which FastMCP reads to build the tool schema.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with NC_LOCK:
            return fn(*args, **kwargs)

    wrapper._serialized_by_nc_lock = True
    return wrapper


_download_locks: dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _download_lock(local: str) -> threading.Lock:
    with _download_locks_guard:
        return _download_locks.setdefault(local, threading.Lock())


def _strip_file_scheme(uri: str) -> str:
    if uri.startswith(_FILE_PREFIX):
        return uri[len(_FILE_PREFIX):]
    return uri


def _s3_filesystem():
    """Build an `s3fs` filesystem from the `AWS_*` variables, with `AWS_ENDPOINT_URL` as the endpoint.

    Raises `NetCDFIOError` when `AWS_ENDPOINT_URL` is not set or `s3fs` is not installed.
    """
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise NetCDFIOError(
            "AWS_ENDPOINT_URL not set; cannot reach S3/MinIO for an s3:// URI"
        )
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover - import guard
        raise NetCDFIOError("s3fs not installed; cannot read s3:// URIs") from exc

    return s3fs.S3FileSystem(
        key=os.environ.get("AWS_ACCESS_KEY_ID"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        token=os.environ.get("AWS_SESSION_TOKEN"),
        client_kwargs={"endpoint_url": endpoint},
    )


def open_dataset(uri: str, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF dataset with `xarray` from a local path or `s3://` URI, passing `kwargs` through.

    An S3 object is downloaded to the local cache first (see `_localize`). Raises `NetCDFIOError` when the file is missing, is not a file or cannot be opened.
    """
    path = Path(_localize(uri))
    if not path.exists():
        raise NetCDFIOError(f"file not found: {uri}")
    if not path.is_file():
        raise NetCDFIOError(f"not a file: {uri}")
    try:
        return xr.open_dataset(path, **kwargs)
    except Exception as exc:
        raise NetCDFIOError(f"could not open {uri}: {exc}") from exc


def resource_sha256(uri: str) -> str:
    """Return the SHA-256 of the bytes behind `uri`, which the analytics record as `file_sha256` and `mask_sha256`.

    The file is read in 1 MiB chunks, after an S3 object has been downloaded to the cache. Raises `NetCDFIOError` when the file is missing.
    """
    path = Path(_localize(uri))
    if not path.exists():
        raise NetCDFIOError(f"file not found: {uri}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exists(uri: str) -> bool:
    """Return True when `uri` exists, and False for an S3 URI when S3 is not configured."""
    if is_s3_uri(uri):
        try:
            return bool(_s3_filesystem().exists(uri))
        except NetCDFIOError:
            return False
    return Path(_strip_file_scheme(uri)).exists()


def list_objects(prefix: str) -> list[dict[str, Any]]:
    """List the files directly under `prefix`, an `s3://` prefix or a local directory.

    Returns `[{uri, size}]` for files only, without directories. Raises `NetCDFIOError` when the listing fails or a local prefix is not a directory.
    """
    if is_s3_uri(prefix):
        fs = _s3_filesystem()
        path = prefix[len(_S3_PREFIX):].rstrip("/")
        try:
            entries = fs.ls(path, detail=True)
        except Exception as exc:
            raise NetCDFIOError(f"could not list {prefix}: {exc}") from exc
        out: list[dict[str, Any]] = []
        for e in entries:
            name = e.get("name") or e.get("Key")
            if not name:
                continue
            if e.get("type") == "directory":
                continue
            size = int(e.get("size") or e.get("Size") or 0)
            out.append({"uri": _S3_PREFIX + name, "size": size})
        return out

    p = Path(_strip_file_scheme(prefix))
    if not p.is_dir():
        raise NetCDFIOError(f"not a directory: {prefix}")
    return [
        {"uri": str(f), "size": f.stat().st_size}
        for f in sorted(p.iterdir())
        if f.is_file()
    ]


#: Lower-case substrings of an error message that mean the credentials failed.
#: A caller must not read such an error as an empty prefix or a missing key.
_AUTH_ERROR_MARKERS = (
    "access key id you provided does not exist",
    "expiredtoken",
    "token has expired",
    "invalidaccesskeyid",
    "signaturedoesnotmatch",
    "accessdenied",
    "403",
)


def is_auth_error(message: str) -> bool:
    """Return True when `message` contains one of `_AUTH_ERROR_MARKERS`, which means the credentials failed."""
    low = str(message).lower()
    return any(m in low for m in _AUTH_ERROR_MARKERS)


def _listing_path(prefix: str) -> str:
    """Where the cached listing for `prefix` lives, beside the cached files."""
    key = hashlib.sha256(prefix.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_CACHE_DIR, f"listing-{key}.json")


def save_listing(prefix: str, files: list[dict[str, Any]]) -> None:
    """Write the listing of `prefix` and the time it was saved to a JSON file in the cache directory.

    `load_listing` reads it back, so `model_list_files` can still name the cached files after the credentials expire. The file is written under a temporary name and then renamed.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    payload = {
        "prefix": prefix,
        "as_of": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    tmp = _listing_path(prefix) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, _listing_path(prefix))


def load_listing(prefix: str) -> dict[str, Any] | None:
    """Return the listing saved for `prefix` as `{prefix, as_of, files}`, or None when none was saved or it cannot be read."""
    try:
        with open(_listing_path(prefix), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def prefetch(prefix: str) -> dict[str, Any]:
    """Download every `.nc` object under the `s3://` `prefix` into the local cache and save the listing.

    The server calls it at startup, so later reads come from the cache and keep working after the credentials expire. A failed download is recorded in `errors` and does not stop the others, and a prefix that is not `s3://` returns at once.

    Returns `{prefix, localized, skipped, errors}`, where `errors` is a list of `{uri, error}`. A local prefix adds `local`, and a listing that fails adds `auth_error`.
    """
    if not is_s3_uri(prefix):
        return {"prefix": prefix, "localized": 0, "skipped": 0,
                "errors": [], "local": True}
    try:
        objs = list_objects(prefix)
    except NetCDFIOError as exc:
        return {"prefix": prefix, "localized": 0, "skipped": 0,
                "errors": [{"uri": prefix, "error": str(exc)}],
                "auth_error": is_auth_error(str(exc))}
    # Save the listing now, so it can still be served after the credentials expire.
    save_listing(prefix, [{"uri": o["uri"], "size": o.get("size")} for o in objs])

    localized = skipped = 0
    errors: list[dict[str, str]] = []
    for o in objs:
        uri = o["uri"]
        if not uri.lower().endswith(".nc"):
            skipped += 1
            continue
        try:
            _localize(uri)
            localized += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort
            errors.append({"uri": uri, "error": str(exc)})
    return {"prefix": prefix, "localized": localized, "skipped": skipped,
            "errors": errors}
