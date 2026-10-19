"""Copies the reference files into the local cache once at launch.

The server reads S3 objects through `simulations.cache`, which fills a file on its first successful read. This module reads the vocabularies, ecosystem models and case-study contracts at launch, so they stay readable after the STS token expires.

Objects uploaded to the bucket after launch are not seen until the next launch with a valid token.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from surimi_model_mcp.reference import vocab
from surimi_model_mcp.simulations import cache, store

logger = logging.getLogger(__name__)

#: The five vocabulary files have fixed keys, so they are still cached when the
#: listing in `reference_keys` fails.
_FIXED = (vocab.ASFIS_KEY, vocab.ISSCFG_KEY, vocab.WORMS_KEY,
          vocab.FISHBASE_KEY, vocab.ISO3166_KEY)


def reference_keys(find: Callable[[str], tuple[list[dict[str, Any]], list[str]]] = store.find) -> list[str]:
    """Return the keys to cache, in a stable order and without duplicates.

    The server finds the ecosystem models (`ecopath/<case study>/*.eiixml`) and the case-study contracts (`controller/<case study>/*_contract.yaml`) by listing, and skips other files under those prefixes. A prefix whose listing fails is logged and left out.
    """
    keys = list(_FIXED)
    for prefix, wanted in (("ecopath", lambda k: k.lower().endswith(".eiixml")),
                           ("controller", lambda k: k.endswith("_contract.yaml"))):
        try:
            objects, _failures = find(prefix)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("reference warm: could not list %s/: %s", prefix, exc)
            continue
        keys.extend(o["key"] for o in objects if wanted(o["key"]))
    return list(dict.fromkeys(keys))


def warm(find: Callable = store.find,
         prefetch: Callable[[Iterable[str]], dict[str, Any]] = cache.prefetch_keys) -> dict[str, Any]:
    """Fetch every key from `reference_keys` into the cache and return the result of `cache.prefetch_keys`."""
    return prefetch(reference_keys(find))
