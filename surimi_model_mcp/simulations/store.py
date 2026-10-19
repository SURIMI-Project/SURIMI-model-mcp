"""Read-only S3 access for simulation artifacts.

The bucket is shared with other SURIMI models and other teams, so the server never writes to it.
`_fs()` returns the wrapper from `_read_only()` and never the s3fs file system itself. The wrapper
allows only `find`, `info`, `cat_file` and `sign`, and it keeps the file system in a closure, so
a caller has no attribute that leads back to it.

The module uses s3fs, as `netcdf/io.py` does, and `sign` produces the presigned URLs for binary
artifacts. `_fs()` pins SigV4 signing because MinIO rejects SigV2 presigned URLs.
"""
from __future__ import annotations

import logging
import os
from typing import Any

BUCKET = os.environ.get("SURIMI_SIM_BUCKET", "project-surimi")

logger = logging.getLogger(__name__)


class SimStoreError(Exception):
    """Raised when object storage is unreachable or misconfigured."""


def _read_only(fs):
    """Return a wrapper that exposes only `find`, `info`, `cat_file` and `sign` of `fs`.

    Any other attribute raises `AttributeError`.
    """
    allowed = frozenset({"find", "info", "cat_file", "sign"})

    class _ReadOnlyFS:
        def __getattr__(self, name):
            if name not in allowed:
                raise AttributeError(
                    f"{name!r} is not available: surimi-model-mcp is read-only "
                    "and must never modify the shared bucket"
                )
            return getattr(fs, name)

    return _ReadOnlyFS()


def _fs():
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise SimStoreError(
            "AWS_ENDPOINT_URL not set; cannot reach S3/MinIO for simulations"
        )
    try:
        import s3fs
    except ImportError as exc:                       # pragma: no cover
        raise SimStoreError(f"s3fs unavailable: {exc}") from exc
    try:
        fs = s3fs.S3FileSystem(
            anon=False,
            client_kwargs={"endpoint_url": endpoint},
            # MinIO rejects SigV2 presigned URLs with 403, so pin SigV4.
            config_kwargs={"signature_version": "s3v4"},
        )
    except Exception as exc:
        # Client construction can fail on an expiring token, a malformed endpoint or a botocore error.
        # Callers expect every storage failure to be a SimStoreError, so wrap it.
        raise SimStoreError(f"could not construct S3 client: {exc}") from exc
    return _read_only(fs)


#: MinIO and boto messages that point to the credentials and not to a missing object.
#: `_describe` matches them case-insensitively.
_CREDENTIAL_FAULTS = (
    "access key id you provided does not exist",
    "expiredtoken",
    "the provided token has expired",
    "invalidaccesskeyid",
    "signaturedoesnotmatch",
    "tokenrefreshrequired",
    "credentials",
)


def _describe(action: str, target: str, exc: Exception) -> SimStoreError:
    """Return a `SimStoreError` for a failed `action` on `target`, with credential faults named as such.

    On EDITO the STS token lapses after about a day, and the raw boto message can read like a missing
    file. The returned message says that the server credentials are invalid or expired and that an
    operator must refresh them.
    """
    raw = str(exc)
    if any(f in raw.lower() for f in _CREDENTIAL_FAULTS):
        return SimStoreError(
            f"could not {action} {target}: the server's S3 credentials are "
            f"invalid or expired - this is a server configuration problem, "
            f"not that the file is missing. An operator must refresh the "
            f"credentials (relaunch the service on EDITO to inject a fresh "
            f"STS token). Underlying error: {raw}"
        )
    return SimStoreError(f"could not {action} {target}: {raw}")


def find(prefix: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Return `(objects, failures)` for every object under the bucket-relative `prefix`.

    Each object is a dictionary with `key`, `size` and `last_modified`. An object whose `info` call fails
    is left out of `objects` and described in `failures`, so `discovery.scan` can report a partial listing.
    """
    path = f"{BUCKET}/{prefix.strip('/')}" if prefix.strip("/") else BUCKET
    try:
        # `_fs()` is called inside the try so that any failure from it becomes a SimStoreError.
        fs = _fs()
        names = fs.find(path)
    except SimStoreError:
        raise
    except Exception as exc:
        raise _describe("list", path, exc) from exc
    out = []
    failures: list[str] = []
    for n in names:
        try:
            info = fs.info(n)
        except Exception as exc:
            failures.append(f"{n}: {exc}")
            continue
        lm = info.get("LastModified")
        out.append({
            "key": n.split("/", 1)[1] if "/" in n else n,
            "size": int(info.get("size") or 0),
            "last_modified": lm.isoformat() if hasattr(lm, "isoformat") else None,
        })
    if failures:
        logger.warning(
            "find(%r): %d of %d objects could not be stat'd and were omitted "
            "from the result: %s",
            prefix, len(failures), len(names), "; ".join(failures[:5]),
        )
    return out, failures


def get_bytes(key: str) -> bytes:
    """Return the whole object as bytes through `cache.fetch`, which reads S3 only on a cache miss."""
    from surimi_model_mcp.simulations import cache
    return cache.fetch(key)


def get_text(key: str, max_bytes: int = 65536) -> tuple[str, bool]:
    """Return `(content, truncated)` for a text object, cut to `max_bytes` bytes.

    The cache stores whole objects, so the cap is applied after the read and not as a ranged read.
    """
    try:
        raw = get_bytes(key)
    except SimStoreError:
        raise
    except Exception as exc:
        raise _describe("read", key, exc) from exc
    truncated = len(raw) > max_bytes
    return raw[:max_bytes].decode("utf-8", "replace"), truncated


def presign(key: str, expires_in: int = 900) -> str:
    """Return a presigned GET URL for `key` that is valid for `expires_in` seconds.

    The URL stops working when the STS token expires, which can be sooner than `expires_in`.
    """
    try:
        fs = _fs()
        return fs.sign(f"{BUCKET}/{key}", expiration=expires_in)
    except SimStoreError:
        raise
    except Exception as exc:
        raise _describe("sign", key, exc) from exc
