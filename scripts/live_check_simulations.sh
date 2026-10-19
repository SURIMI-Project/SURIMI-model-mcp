#!/usr/bin/env bash
# This script checks the simulation tools against the real EDITO MinIO bucket, where the unit tests use a fake filesystem.
# It covers run discovery, inline text and artifact links, and it only lists and reads.
# It needs the four AWS_* environment variables from Onyxia (My Account, S3 credentials), and the tokens last about 24 hours.
set -euo pipefail
cd "$(dirname "$0")/.."

missing=()
for var in AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_ENDPOINT_URL; do
  if [ -z "${!var:-}" ]; then
    missing+=("$var")
  fi
done

if [ "${#missing[@]}" -gt 0 ]; then
  {
    echo "SKIPPED: missing required env var(s): ${missing[*]}"
    echo
    echo "This script needs live STS credentials for EDITO MinIO. Get them from:"
    echo "  Onyxia -> My Account -> S3 credentials"
    echo "and export all four (the endpoint shown there, e.g. https://minio.dive.edito.eu):"
    echo "  export AWS_ACCESS_KEY_ID=..."
    echo "  export AWS_SECRET_ACCESS_KEY=..."
    echo "  export AWS_SESSION_TOKEN=..."
    echo "  export AWS_ENDPOINT_URL=..."
    echo
    echo "Tokens last ~24h from issue; re-fetch from Onyxia once they expire."
  } >&2
  exit 1
fi

PYTHONPATH=. .venv/bin/python - <<'PY'
"""Check tools/runs.py against the real bucket, read-only.

The unit tests use a fake filesystem, which cannot show these three faults:

- Two files end in "id_file.csv" in a CMSY run: the run's output
  (Out_<region><date>_id_file.csv) and the CMSY++ input stock database
  (id_file.csv). model_get_artifact matches file names exactly, so the script
  takes the real name from model_describe_simulation's file list.
- A binary file comes back as a link to the server's /artifact route. Only a
  real HTTP fetch shows that the link works.
- The number of runs must stay at or above a floor, and the stale and partial
  flags must be present in the listing.
"""
import urllib.request

from surimi_model_mcp.tools import runs as sim_tools

out = sim_tools.model_list_simulations()
count = out["count"]
print(
    f"runs discovered: {count}  models: {out['scanned']}  "
    f"stale={out['stale']}  partial={out['partial']}"
)
# The check uses a floor and not an exact count, because other teams add runs to the bucket.
# On 2026-08-06 it held 21 surimi-cmsy runs and 2 surimi-market runs.
assert count >= 23, (
    f"expected at least 23 runs in the live bucket (floor as of 2026-08-06: "
    f"21 CMSY + 2 market), got {count} - investigate before assuming this "
    "script is wrong"
)

# After a healthy scan, both flags are False.
assert out["stale"] is False, (
    f"expected a fresh listing (stale=False) against live storage, got "
    f"stale={out['stale']!r} note={out.get('note')!r}"
)
assert out["partial"] is False, (
    f"expected a complete listing (partial=False), got "
    f"partial={out['partial']!r} partial_note={out.get('partial_note')!r}"
)

cmsy = [s for s in out["simulations"] if s["model"] == "surimi-cmsy"]
assert cmsy, "expected CMSY runs"
run = cmsy[0]["simulation_id"]

d = sim_tools.model_describe_simulation(run)
print(f"run {run}: {d['n_artifacts']} artifacts")
print(f"  metadata: {d['metadata']}")
assert d["stale"] is False and d["partial"] is False, (
    f"expected model_describe_simulation to report stale=False/partial=False "
    f"on a healthy run too, got stale={d['stale']!r} partial={d['partial']!r}"
)
stock = d["metadata"].get("stock")
assert stock, "CMSY metadata should name a stock"
assert "hogfish" in stock.lower(), (
    f"expected the CMSY++ demo stock (Hogfish - Florida Keys / East "
    f"Florida), got stock={stock!r}"
)

# The script takes the run's own id file from the artifact list, and it expects exactly one name that starts with `Out_`.
out_id_files = [
    a["name"] for a in d["artifacts"]
    if a["name"].endswith("id_file.csv") and a["name"].startswith("Out_")
]
assert len(out_id_files) == 1, (
    "expected exactly one Out_-prefixed id file naming this run's own "
    f"output, found {out_id_files!r} among "
    f"{[a['name'] for a in d['artifacts']]!r}"
)
id_filename = out_id_files[0]

txt = sim_tools.model_get_artifact(run, id_filename)
assert txt["kind"] == "text" and txt["content"], (
    f"{id_filename} should come back inline as text, got {txt!r}"
)
print(f"  {id_filename}: {len(txt['content'])} chars inline")

# The script fetches the artifact link and requires HTTP 200.
# The CMSY .jpg names contain spaces and hyphens, for example "Hogfish - Florida Keys - East Florida_AN.jpg".
jpgs = [a["name"] for a in d["artifacts"] if a["name"].lower().endswith(".jpg")]
assert jpgs, f"expected at least one .jpg artifact in CMSY run {run}"
jpg = jpgs[0]
link = sim_tools.model_get_artifact(run, jpg)
assert link["kind"] == "link" and link["url"].startswith("http"), (
    f"expected a presigned URL for {jpg!r}, got {link!r}"
)
code = urllib.request.urlopen(link["url"], timeout=30).status
print(f"  {jpg}: presigned URL -> HTTP {code}")
assert code == 200, (
    f"presigned URL for {jpg!r} returned HTTP {code}, expected 200 - check "
    "simulations/store.py's signature_version='s3v4' pin (MinIO 403s the "
    "SigV2 URLs s3fs/botocore default to for presigning)"
)

# The tool must refuse a file name that points outside the run.
escape = sim_tools.model_get_artifact(run, "../../mcp_data/tables/secret.csv")
assert "error" in escape, (
    f"expected model_get_artifact to refuse a path-escape filename, got "
    f"{escape!r}"
)
print(f"  path-escape refused: {escape['error']}")

print("LIVE CHECK OK")
PY
