"""Issues and verifies HMAC-signed receipts, the calculation records of the analytical tools.

A receipt holds the tool name, the inputs, the output and optional provenance, signed with an HMAC-SHA256 keyed by a server-side secret. `verify_receipt` signs the same payload again and compares the two signatures in constant time. Without the secret, nobody can produce a valid signature.

Usage:
    r = issue_receipt(
        tool_id="model_top_regions",
        input_params={"file": "s3://b/x.nc", "year": 2010, "n": 3},
        output_value=[{"region": "Humboldt", "value": 18.4}],
        secret=os.environ["SURIMI_RECEIPT_SECRET"].encode(),
        provenance={"file_sha256": "...", "mask_sha256": "..."},
    )
    # r is JSON-serializable and is returned beside the tool output.

    res = verify_receipt(r, secret=...)
    if res["verified"]: ...
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import math
from decimal import Decimal
from typing import Any


SCHEMA_VERSION = 1


class ReceiptError(ValueError):
    """Raised when a receipt cannot be built or checked: an empty secret or a value that cannot be signed."""


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _coerce(obj: Any) -> Any:
    """Convert a value into a JSON-native form that signs the same way every time.

    - `Decimal` becomes a string.
    - `datetime` and `date` become ISO 8601 strings.
    - NaN and infinite floats become `None`, because JSON has no literal for them and a transport that rewrites them to `null` would break the signature.
    - `dict`, `list` and `tuple` are converted recursively.
    - `str`, `int`, `bool`, `None` and finite floats are returned unchanged.

    Any other type raises `ReceiptError`. `issue_receipt` calls this, so the stored receipt is JSON-serializable and not only the signed bytes.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _coerce(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce(v) for v in obj]
    raise ReceiptError(
        f"value of type {type(obj).__name__} is not serializable for a receipt"
    )


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize a payload to the JSON bytes that are signed.

    The JSON has sorted keys, no whitespace between separators and no ASCII escaping. `allow_nan=False` makes a non-finite float that bypassed `_coerce` raise `ReceiptError` and not enter a signature.
    """
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ReceiptError(
            f"payload contains a non-finite value and cannot be signed: {exc}"
        ) from exc


def _sign(payload: dict[str, Any], secret: bytes) -> str:
    canonical = _canonical_bytes(payload)
    return hmac.new(secret, canonical, hashlib.sha256).hexdigest()


def issue_receipt(
    tool_id: str,
    input_params: dict[str, Any],
    output_value: Any,
    secret: bytes,
    *,
    timestamp: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a signed receipt for one tool call, with `provenance` included only when it is given.

    The `timestamp` defaults to the current UTC time, and an empty `secret` raises `ReceiptError`.
    """
    if not secret:
        raise ReceiptError("secret must be non-empty bytes")

    body: dict[str, Any] = {
        "version": SCHEMA_VERSION,
        "tool_id": tool_id,
        "input_params": _coerce(input_params),
        "output_value": _coerce(output_value),
        "timestamp": timestamp or _now_iso(),
    }
    if provenance is not None:
        body["provenance"] = _coerce(provenance)

    body["signature"] = _sign(body, secret)
    return body


def verify_receipt(receipt: dict[str, Any], secret: bytes) -> dict[str, Any]:
    """Check a receipt's signature against `secret` and return `{verified, reason}`.

    A missing field, an unsupported version or a signature mismatch gives `verified` False with a `reason`. Only an empty secret raises `ReceiptError`.
    """
    if not secret:
        raise ReceiptError("secret must be non-empty bytes")

    required = {"version", "tool_id", "input_params", "output_value", "timestamp", "signature"}
    missing = required - set(receipt.keys())
    if missing:
        return {"verified": False, "reason": "missing_field", "missing": sorted(missing)}

    if receipt["version"] != SCHEMA_VERSION:
        return {
            "verified": False,
            "reason": "unsupported_version",
            "got": receipt["version"],
            "supported": SCHEMA_VERSION,
        }

    claimed = receipt["signature"]
    body = {k: v for k, v in receipt.items() if k != "signature"}
    expected = _sign(body, secret)

    if not hmac.compare_digest(claimed, expected):
        return {"verified": False, "reason": "signature_mismatch"}

    return {"verified": True, "reason": "ok"}
