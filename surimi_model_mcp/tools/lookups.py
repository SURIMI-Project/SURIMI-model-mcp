"""The reference tools: look up FAO, ISSCFG, WoRMS, FishBase and ISO codes, read the case-study contract and describe EwE models.

None of this is SURIMI simulation output. The vocabularies are external standards
(FAO species codes, ISSCFG gear codes, WoRMS taxonomy, FishBase common names and
ISO 3166 countries, read through `reference/vocab.py`). The case-study contract
(`reference/contract.py`) and the EwE model structure (`reference/ewe.py`) are SURIMI's
own configuration. Like the tools in `tools/runs.py`, these tools return no signature.
`register_all` in `tools/__init__.py` registers every name in `TOOLS` with `mcp.tool()`.
"""
from __future__ import annotations

import difflib
from typing import Any

from surimi_model_mcp.reference import contract, ewe, vocab
from surimi_model_mcp.simulations import store

_VALID_KINDS = ("species", "gear", "country")


def _match_species(raw: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """(match, degraded_note, candidates_for_suggestion_when_no_match)."""
    try:
        idx, err = vocab.asfis()
    except store.SimStoreError as exc:
        return None, f"ASFIS species codes unavailable this call: {exc}", []
    if err:
        return None, f"ASFIS species codes unavailable: {err}", []
    row = idx["by_code"].get(raw) or idx["by_code"].get(raw.upper())
    if not row:
        return None, None, sorted(idx["by_code"])
    return {
        "kind": "species",
        "code": row.get("Alpha3_Code"),
        "scientific_name": row.get("Scientific_Name"),
        "english_name": row.get("English_name"),
        "family": row.get("Family"),
        "order": row.get("Order or higher taxa"),
        "isscaap_group": (row.get("ISSCAAP_Group") or "").strip() or None,
        "names": {
            "fr": row.get("French_name") or None,
            "es": row.get("Spanish_name") or None,
            "ar": row.get("Arabic_name") or None,
            "zh": row.get("Chinese_name") or None,
            "ru": row.get("Russian_name") or None,
        },
        "cross_references": _species_cross_references(row),
    }, None, []


def _species_cross_references(asfis_row: dict[str, str]) -> dict[str, Any]:
    """WoRMS (joined on FAO_Code) and FishBase (joined on Genus+Species) for
    one ASFIS row - each independently degradable, so a WoRMS problem never
    hides a FishBase hit or the primary ASFIS match itself."""
    code = asfis_row.get("Alpha3_Code", "")
    sci = (asfis_row.get("Scientific_Name") or "").strip()
    refs: dict[str, Any] = {"worms": None, "fishbase": None}
    notes: list[str] = []

    try:
        widx, werr = vocab.worms()
    except store.SimStoreError as exc:
        widx, werr = None, None
        notes.append(f"WoRMS cross-reference unavailable this call: {exc}")
    if werr:
        notes.append(f"WoRMS cross-reference unavailable: {werr}")
    elif widx:
        wrow = widx["by_fao_code"].get(code)
        if wrow:
            refs["worms"] = {
                "scientific_name": wrow.get("scientificName"),
                "kingdom": wrow.get("kingdom"), "phylum": wrow.get("phylum"),
                "class": wrow.get("class"), "order": wrow.get("order"),
                "family": wrow.get("family"), "genus": wrow.get("genus"),
                "taxon_rank": wrow.get("taxonRank"),
            }

    try:
        fidx, ferr = vocab.fishbase()
    except store.SimStoreError as exc:
        fidx, ferr = None, None
        notes.append(f"FishBase cross-reference unavailable this call: {exc}")
    if ferr:
        notes.append(f"FishBase cross-reference unavailable: {ferr}")
    elif fidx and sci:
        frow = fidx["by_binomial"].get(sci.lower())
        if frow:
            refs["fishbase"] = {
                "spec_code": frow.get("SpecCode"),
                "common_name": frow.get("FBname"),
            }

    if notes:
        refs["degraded"] = notes
    return refs


def _match_country(raw: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    try:
        idx, err = vocab.iso3166()
    except store.SimStoreError as exc:
        return None, f"ISO 3166 countries unavailable this call: {exc}", []
    if err:
        return None, f"ISO 3166 countries unavailable: {err}", []
    row = idx["by_code"].get(raw) or idx["by_code"].get(raw.upper())
    if not row:
        return None, None, sorted(idx["by_code"])
    return {
        "kind": "country",
        "code": row.get("alpha-3"), "alpha_2": row.get("alpha-2"),
        "name": row.get("name"), "region": row.get("region"),
        "sub_region": row.get("sub-region"),
    }, None, []


def _match_gear(raw: str) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """ISSCFG has two separate real code axes sharing the same file: the
    compound FISHERY_CODE (e.g. "DL.FR[IS]TR", spec's example) which
    identifies one specific fishery with its category/type/description, and
    the shorter GEAR_CODE (e.g. "PS") which many fisheries share. Tried in
    that order; a GEAR_CODE hit says up front how many fisheries share it
    rather than silently picking one."""
    try:
        idx, err = vocab.isscfg()
    except store.SimStoreError as exc:
        return None, f"ISSCFG gear/fishery codes unavailable this call: {exc}", []
    if err:
        return None, f"ISSCFG gear/fishery codes unavailable: {err}", []

    row = idx["by_fishery_code"].get(raw) or idx["by_fishery_code"].get(raw.upper())
    if row:
        return {
            "kind": "gear", "code": row.get("FISHERY_CODE"), "match_field": "fishery_code",
            "fishery_name": row.get("FISHERY_NAME"),
            "fishery_category": row.get("FISHERY_CATEGORY_NAME"),
            "fishery_type": row.get("FISHERY_TYPE_NAME"),
            "fishery_type_description": row.get("FISHERY_TYPE_DESCRIPTION"),
            "gear_code": row.get("GEAR_CODE"), "gear_name": row.get("GEAR_NAME"),
        }, None, []

    gear_rows = idx["by_gear_code"].get(raw) or idx["by_gear_code"].get(raw.upper())
    if gear_rows:
        r0 = gear_rows[0]
        return {
            "kind": "gear", "code": r0.get("GEAR_CODE"), "match_field": "gear_code",
            "gear_name": r0.get("GEAR_NAME"),
            "gear_group_code": r0.get("GEAR_GROUP_CODE"), "gear_group_name": r0.get("GEAR_GROUP_NAME"),
            "matched_fishery_count": len(gear_rows),
            "note": (
                f"{len(gear_rows)} ISSCFG fishery entries share gear_code "
                f"{r0.get('GEAR_CODE')!r} - this is the gear itself, not one "
                "specific fishery; look up a FISHERY_CODE for category/type "
                "detail on a single fishery"
            ),
        }, None, []

    candidates = sorted(set(idx["by_fishery_code"]) | set(idx["by_gear_code"]))
    return None, None, candidates


_MATCHERS = {"species": _match_species, "gear": _match_gear, "country": _match_country}


def model_lookup_code(code: str, kind: str | None = None) -> dict[str, Any]:
    """Resolve a reference-vocabulary code to its meaning. Read this FIRST for
    "what species is ANK" / "what does gear code X mean" / "what country is
    ESP" - never guess a code's meaning from memory, this tool is the source
    of truth for what the FAO/ISSCFG/ISO standards actually say.

    `kind` is one of "species" (FAO 3-alpha, cross-referenced to WoRMS
    taxonomy and FishBase common name when they join), "gear" (ISSCFG
    fishery/gear code), or "country" (ISO 3166-1 alpha-3) - or omitted to
    search all three. Omitting `kind` does NOT guess a domain from the
    code's shape; it checks all three vocabularies and reports every REAL
    hit. This matters because the vocabularies genuinely collide: FAO
    species codes and ISO country codes share the same 3-letter alphabet, so
    e.g. "ESP" is both Spain (ISO 3166) and Setipinna phasa / Gangetic
    hairfin anchovy (FAO/ASFIS) - a real collision, not a bug. When more than
    one domain matches, `ambiguous: true` and `matches` lists all of them;
    read `ambiguity_note` and do not present either as THE answer without
    picking a domain via `kind`. When exactly one domain matches (the common
    case), its fields are merged directly into the top-level result.

    An unknown code returns `found: false` with `suggestions` - real entries
    from the searched vocabulary/vocabularies that are textually close, from
    difflib against the actual code list, never an invented name. An empty
    `suggestions` list means nothing close was found either; still not a
    guess.

    `degraded` (only present when non-empty) lists which underlying source(s)
    could not be loaded this call - a network blip or expiring STS token,
    not proof the code does not exist; a real answer from an unaffected
    source can still appear alongside a `degraded` entry for one that
    failed.

    These are external standards (FAO/ISSCAAP, ISSCFG, WoRMS, FishBase, ISO
    3166) the SURIMI project consumes for coding its own data - not SURIMI
    output.
    """
    raw = (code or "").strip()
    if not raw:
        return {"code": code, "kind": kind, "found": False,
                "error": "code must be a non-empty string"}
    if kind is not None and kind not in _VALID_KINDS:
        return {"code": raw, "kind": kind, "found": False,
                "error": f"kind must be one of {list(_VALID_KINDS)} or omitted, got {kind!r}"}

    domains = [kind] if kind else list(_VALID_KINDS)
    matches: list[dict[str, Any]] = []
    degraded: list[str] = []
    candidate_pool: list[str] = []
    for d in domains:
        m, note, candidates = _MATCHERS[d](raw)
        if m:
            matches.append(m)
        if note:
            degraded.append(note)
        candidate_pool.extend(candidates)

    result: dict[str, Any] = {"code": raw, "kind": kind}
    if degraded:
        result["degraded"] = degraded

    if len(matches) == 1:
        result["found"] = True
        result["ambiguous"] = False
        result.update(matches[0])
    elif len(matches) > 1:
        result["found"] = True
        result["ambiguous"] = True
        result["matches"] = matches
        result["ambiguity_note"] = (
            f"{raw!r} is a real, distinct entry in {len(matches)} different "
            "reference vocabularies - this happens (see this tool's "
            "docstring); pass kind= explicitly to pick one instead of "
            "treating either match as THE answer"
        )
    else:
        result["found"] = False
        suggestions = sorted(difflib.get_close_matches(raw.upper(), candidate_pool, n=3, cutoff=0.6))
        result["suggestions"] = suggestions
        where = f"among {kind} codes" if kind else "in the species, gear, or country vocabularies"
        result["message"] = (
            f"no entry for {raw!r} {where}"
            + (f"; closest real entries: {', '.join(suggestions)}" if suggestions
               else " - no close match either, this is a clean miss, not an invented name")
        )
    return result


def _words(text: str) -> list[str]:
    """Lowercase word tokens, split on anything non-alphanumeric. Used for
    word-boundary scoring; deliberately not a regex (a vocabulary is data, and
    hand-rolled patterns over species names are how "hake" matched
    kus-chake-witschi in the first place)."""
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


#: Relevance tiers for a vocabulary name against a query. A higher score ranks
#: first, so a match inside a longer word never outranks a match on a whole word.
_EXACT_NAME = 100      # the whole name IS the query
_PHRASE_AT_START = 80  # query is the opening word(s): "European hake" <- "european"
_PHRASE_ANYWHERE = 60  # query is a whole word further in: "Red hake" <- "hake"
_WORD_PREFIX = 40      # query starts a word: "anchovy" <- "anchov"
_WITHIN_WORD = 10      # query buried inside a word: "kuschakewitschi" <- "hake"


def _match_score(query: str, text: str) -> int:
    """How well `text` matches `query`; 0 means no match at all.

    Substring containment still decides whether something matches, so nothing
    that could be found before disappears. The score only decides where it ranks.
    """
    if not text:
        return 0
    hay = text.lower().strip()
    if query not in hay:
        return 0
    if hay == query:
        return _EXACT_NAME

    q_words, h_words = _words(query), _words(hay)
    if q_words:
        n = len(q_words)
        for i in range(len(h_words) - n + 1):
            if h_words[i:i + n] == q_words:
                # Position counts only for a multi-word query, where the caller
                # gave a phrase order. For one word, "Red hake" and "hake weakfish"
                # match "hake" equally well, so the tie-breaks decide the order.
                if n == 1:
                    return _PHRASE_ANYWHERE
                return _PHRASE_AT_START if i == 0 else _PHRASE_ANYWHERE
        if n == 1:
            for word in h_words:
                if word.startswith(q_words[0]):
                    return _WORD_PREFIX
    return _WITHIN_WORD


def model_search_taxon(name: str, limit: int = 10) -> dict[str, Any]:
    """Reverse lookup: a species name or fragment to codes, searched
    case-insensitively across ASFIS (scientific + English name), WoRMS
    (scientific name), and FishBase (common name + binomial). Use this when
    you have a name, not a code - e.g. "what's the FAO code for anchovy".

    Each result carries `source` ("asfis" | "worms" | "fishbase") so a
    caller can tell an FAO alpha-3 code apart from a FishBase common-name
    hit rather than assuming every result is the same kind of thing.

    All three sources are searched in full and results are RANKED BY MATCH
    QUALITY, then capped at `limit` total (not `limit` per source). A name
    that contains the query as a word ("Red hake" for "hake") always outranks
    one that merely contains it inside a longer word ("Capoetobrama
    kuschakewitschi"); an exact whole-name match outranks both. Equal matches
    are ordered by whether FAO keeps capture statistics for the species, then
    source, then alphabetically.

    `total_matches` appears when the cap hid results. Treat a large value as
    "the query is ambiguous, not answered": 21 ASFIS species are named
    "... hake", so ranking cannot know which one is meant - narrow the query
    (a binomial, or the full common name) rather than reporting the first hit
    as THE answer.

    `degraded` (only present when non-empty) names any source that could not
    be searched this call; results from unaffected sources still come back.
    """
    query = (name or "").strip().lower()
    limit = max(1, int(limit)) if limit else 10
    if not query:
        return {"name": name, "results": [], "count": 0,
                "error": "name must be a non-empty string"}

    # Every source is scanned in full and all matches are scored before the
    # result is cut to `limit`.
    #
    # Sort key, in order: relevance, whether FAO keeps capture statistics for the
    # species (the ASFIS `FishStat_Data` flag), source (ASFIS, then WoRMS, then
    # FishBase), the name alphabetically, and read order. The last two carry no
    # information about which species the caller means. `total_matches` tells the
    # caller when to narrow the query.
    scored: list[tuple[tuple[int, int, int, str, int], dict[str, Any]]] = []
    degraded: list[str] = []
    seq = 0

    def _add(score: int, source_rank: int, label: str, row: dict[str, Any],
             in_fishstat: bool = False) -> None:
        nonlocal seq
        scored.append(((-score, 0 if in_fishstat else 1, source_rank,
                        label.lower(), seq), row))
        seq += 1

    try:
        idx, err = vocab.asfis()
    except store.SimStoreError as exc:
        idx, err = None, str(exc)
    if err:
        degraded.append(f"ASFIS species codes unavailable: {err}")
    elif idx:
        for r in idx["rows"]:
            sci, eng = r.get("Scientific_Name") or "", r.get("English_name") or ""
            score = max(_match_score(query, sci), _match_score(query, eng))
            if score:
                _add(score, 0, eng or sci,
                     {"source": "asfis", "code": r.get("Alpha3_Code"),
                      "scientific_name": sci, "english_name": eng},
                     in_fishstat=(r.get("FishStat_Data") or "").strip().upper() == "YES")

    try:
        idx, err = vocab.worms()
    except store.SimStoreError as exc:
        idx, err = None, str(exc)
    if err:
        degraded.append(f"WoRMS taxonomy unavailable: {err}")
    elif idx:
        for r in idx["rows"]:
            sci = r.get("scientificName") or ""
            score = _match_score(query, sci)
            if score:
                _add(score, 1, sci,
                     {"source": "worms", "code": r.get("FAO_Code") or None,
                      "scientific_name": sci, "taxon_rank": r.get("taxonRank")})

    try:
        idx, err = vocab.fishbase()
    except store.SimStoreError as exc:
        idx, err = None, str(exc)
    if err:
        degraded.append(f"FishBase species occurrence summary unavailable: {err}")
    elif idx:
        for r in idx["rows"]:
            fb = r.get("FBname") or ""
            binomial = f"{r.get('Genus', '')} {r.get('Species', '')}".strip()
            score = max(_match_score(query, fb), _match_score(query, binomial))
            if score:
                _add(score, 2, fb or binomial,
                     {"source": "fishbase", "code": r.get("SpecCode"),
                      "scientific_name": binomial, "common_name": fb})

    scored.sort(key=lambda pair: pair[0])
    total = len(scored)
    results = [row for _, row in scored[:limit]]

    out: dict[str, Any] = {"name": name, "results": results, "count": len(results)}
    if total > len(results):
        out["total_matches"] = total
    if degraded:
        out["degraded"] = degraded
    return out


def model_get_contract() -> dict[str, Any]:
    """The case-study contract: what SURIMI can model, over what area,
    period, in what units, coded how. Parses
    controller/northwestern_med/northwestern_med_contract.yaml: case study name,
    start/end date, time step, CRS + grid (resolution, dimensions, bbox),
    the standards block (which code standard each field uses - species,
    gear, life stage, market, currency, country, measurement system/units),
    market codes, currency codes, and counts of the species/fleet-segment
    entries the contract lists. `fleet_segments` carries each segment's own
    `model` assignment ("EwE" or "POSEIDON") - a per-fleet LABEL only: the
    controller calls both services unconditionally for every fleet every
    step, so this field says which model a fleet is ASSIGNED to for
    after-the-fact splitting, not a switch that turns either model off.

    `divergence_note` is present on every response and must not be dropped
    when relaying this: it says that this is the northwestern_med contract,
    that the bucket holds others (aegean_sea, anchovy_bay) which this tool
    does not read, and that the older western_med copy no longer exists.

    This is model configuration, not a simulation result.
    """
    try:
        parsed, err = contract.get()
    except store.SimStoreError as exc:
        return {"error": f"case-study contract unavailable this call: {exc}"}
    if err:
        return {"error": f"case-study contract unavailable: {err}"}
    return {**parsed, "divergence_note": contract.DIVERGENCE_NOTE}


def model_describe_ecosystem_model(name: str | None = None) -> dict[str, Any]:
    """Structure of an EwE (Ecopath with Ecosim) model export (.eiixml):
    model name, EwE/DB version, the full table inventory (name + declared
    row/column count for every table in the file), and the functional-group
    NAMES from the EcopathGroup table. Metadata and group names ONLY - never
    the diet-composition matrix or Ecosim scenario tables, which this tool
    does not read at all.

    Call with no argument first to see which models exist (`models`, live
    from object storage - never a hardcoded list, so a newly-uploaded model
    appears without a server update). Call again with a `name` (matches
    either the model's internal name or its filename, case-insensitively)
    for the full table inventory of one model.

    `functional_groups.declared_row_count` is the EcopathGroup table's Num
    attribute as the file states it; `functional_groups.names_found`/`names`
    are what this tool actually read from the file's `<Row>` elements. These
    two numbers can differ - see `functional_groups.note` when they do, and
    report `names_found`, not `declared_row_count`, as how many functional
    groups are actually nameable here.

    This is SURIMI ensemble model CONFIGURATION, not a simulation result -
    running or solving the model is out of scope for this tool.
    """
    if name is None or not str(name).strip():
        try:
            models, err = ewe.list_models()
        except store.SimStoreError as exc:
            return {"error": f"EwE model listing unavailable this call: {exc}"}
        if err:
            return {"error": f"EwE model listing unavailable: {err}"}
        return {
            "models": [{"name": m["name"], "file": m["file"], "size_bytes": m["size_bytes"]}
                       for m in models],
            "count": len(models),
        }

    try:
        parsed, err = ewe.describe_model(str(name))
    except store.SimStoreError as exc:
        return {"name": name, "error": f"EwE model {name!r} unavailable this call: {exc}"}
    if err:
        return {"name": name, "error": err}
    return parsed


TOOLS = [
    "model_lookup_code",
    "model_search_taxon",
    "model_get_contract",
    "model_describe_ecosystem_model",
]
