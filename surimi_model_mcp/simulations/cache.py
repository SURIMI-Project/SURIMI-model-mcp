"""On-disk cache for simulation artifacts, so reads keep working after the storage token expires.

On EDITO the server runs on an Onyxia STS token that lapses about 24 hours after launch. Once it
lapses, uncached reads fail with a storage error. `prefetch_keys` fills the cache at startup while
the token is still valid.

Each file is stored under the SHA-256 digest of its bucket key, in the directory named by
`SURIMI_SIM_CACHE` (default `/tmp/surimi-simcache`). Hashing keeps keys with spaces, parentheses
or unicode out of the file system, and a key can never name a path outside the cache directory.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

_CACHE_DIR = os.environ.get("SURIMI_SIM_CACHE", "/tmp/surimi-simcache")


def _fs():
    # Indirection so tests can substitute a filesystem without touching S3.
    from surimi_model_mcp.simulations import store
    return store._fs()


def _path(key: str) -> str:
    return os.path.join(_CACHE_DIR, hashlib.sha256(key.encode()).hexdigest())


def is_cached(key: str) -> bool:
    """Return True when the object for `key` is already in the cache directory."""
    return os.path.exists(_path(key))


def fetch(key: str) -> bytes:
    """Return the bytes of `key` from the cache, or read them from S3 and cache them on a miss.

    A miss is normal, because the warm-up runs in a background thread and a run added after launch
    was never in it. A failed S3 read raises `store.SimStoreError`.
    """
    local = _path(key)
    if os.path.exists(local):
        with open(local, "rb") as fh:
            return fh.read()

    from surimi_model_mcp.simulations import store
    try:
        raw = _fs().cat_file(f"{store.BUCKET}/{key}")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise store._describe("read", key, exc) from exc

    os.makedirs(_CACHE_DIR, exist_ok=True)
    # Write to a temp file and rename it into place with os.replace, so a pod killed
    # mid-write leaves no truncated file that a later read takes for a cache hit.
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        os.replace(tmp, local)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return raw


def prefetch_keys(keys: Iterable[str]) -> dict[str, Any]:
    """Cache each key in `keys` that is not cached yet and return `cached`, `skipped` and `errors`.

    The server startup calls this with the keys that discovery found. A failure on one key is
    recorded in `errors` and does not stop the others.
    """
    cached = skipped = 0
    errors: list[dict[str, str]] = []
    for key in keys:
        if is_cached(key):
            skipped += 1
            continue
        try:
            fetch(key)
            cached += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                       # noqa: BLE001
            errors.append({"key": key, "error": str(exc)})
    if errors:
        logger.warning("sim prefetch: %d object(s) could not be cached: %s",
                       len(errors), "; ".join(e["key"] for e in errors[:5]))
    return {"cached": cached, "skipped": skipped, "errors": errors}


def prefetch(prefixes: Iterable[str],
             list_keys: Callable[[str], list[str]] | None = None) -> dict[str, Any]:
    """Cache every object under each prefix in `prefixes` and return `cached`, `skipped` and `errors`.

    `list_keys` lists the keys under a prefix and defaults to `store.find`. A failed listing or a
    failed read is recorded in `errors` and is never raised, so one bad object does not leave the
    rest uncached.
    """
    if list_keys is None:
        from surimi_model_mcp.simulations import store

        def list_keys(prefix: str) -> list[str]:      # noqa: F811
            objs, _failures = store.find(prefix)
            return [o["key"] for o in objs]

    cached = skipped = 0
    errors: list[dict[str, str]] = []
    for prefix in prefixes:
        try:
            keys = list_keys(prefix)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:                       # noqa: BLE001
            errors.append({"key": prefix, "error": f"list failed: {exc}"})
            continue
        for key in keys:
            if is_cached(key):
                skipped += 1
                continue
            try:
                fetch(key)
                cached += 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:                   # noqa: BLE001
                errors.append({"key": key, "error": str(exc)})
    if errors:
        logger.warning("sim prefetch: %d object(s) could not be cached: %s",
                       len(errors), "; ".join(e["key"] for e in errors[:5]))
    return {"cached": cached, "skipped": skipped, "errors": errors}
