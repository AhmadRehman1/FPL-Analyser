"""M1b: evidence integration & reliability weighting.

Turns potentially-conflicting evidence_claims into one blended view per
(subject_entity, claim_type), per M0's conflict-resolution spec:
"Each active claim's weight_i = source_reliability_score_i x confidence_i x decay_i,
normalized across active claims. Numeric payloads blend as a weighted average;
categorical payloads (e.g. injury status) blend as a weighted probability distribution
over categories rather than forcing a single winner."

M1b adds one thing to the weight formula: a fact_type_multiplier boost for FACT-tagged
rows from high-tier sources. FACT-tagged evidence is never promoted into fact_reconciled
regardless -- that boundary lives entirely upstream of this module, in M0's schema.
"""

import json
from datetime import date, datetime

import duckdb
import pandas as pd

from . import decay as decay_mod
from . import params as params_mod
from . import snapshot as snapshot_mod

# M1b names "high-tier sources" for the FACT multiplier without pinning an exact tier
# cutoff. Interpreted here as the top two of the four source_tier_weights tiers (official,
# journalist) -- an implementation judgment call with the same invented-default status as
# every other unpinned constant in this project (tier weights, lambda, xi, rho...), flagged
# for the same M7 recalibration pass.
HIGH_TIER_SOURCE_TYPES = {"official", "journalist"}


def _to_date(x) -> date | None:
    # A real, live-CI-observed gap in the prior check: pandas' own null-date sentinel,
    # pd.NaT, is neither `None` nor a `float` (it's its own NaTType), so the old
    # `isinstance(x, float) and pd.isna(x)` guard never caught it -- it fell through every
    # branch to pd.Timestamp(x).date(), which for NaT input just returns NaT right back
    # (not a real date, and not caught as falsy either), silently poisoning decay()'s date
    # subtraction downstream with a TypeError instead of being treated as "no observed date."
    # pd.isna() alone correctly recognizes None/NaN/NaT (it's only unsafe on array-likes,
    # never the case for a single claim's scalar value here), so check it first and directly.
    if x is None:
        return None
    if pd.isna(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


def effective_weight(
    con: duckdb.DuckDBPyConnection,
    claim_row: dict,
    asof: datetime,
    decay_params_version: int,
    fact_multiplier_params_version: int,
) -> float:
    # claim_row may come from a pandas DataFrame (snapshot.get_claims_asof), where a SQL
    # NULL in a numeric column becomes NaN, not Python None -- `is not None` doesn't catch
    # it, and an unnoticed NaN silently poisons every downstream sum/clip. pd.isna() catches
    # both representations, whichever caller (dict literal or DataFrame row) this came from.
    reliability = claim_row["source_reliability_score"]
    reliability = 0.0 if pd.isna(reliability) else reliability
    confidence = claim_row["confidence"]
    confidence = 1.0 if pd.isna(confidence) else confidence
    observed = _to_date(claim_row["observed_date"])
    if observed is not None:
        d = decay_mod.decay_for_claim_type(con, claim_row["claim_type"], observed, asof.date(), decay_params_version)
    else:
        d = 1.0
    w = reliability * confidence * d

    if claim_row["information_type"] == "FACT":
        row = con.execute("SELECT source_type FROM sources WHERE source_id = ?", [claim_row["source_id"]]).fetchone()
        if row and row[0] in HIGH_TIER_SOURCE_TYPES:
            try:
                mult, _ = params_mod.resolve_param(
                    con, "fact_type_multiplier_params", "multiplier", fact_multiplier_params_version
                )
                w *= mult
            except params_mod.ParamNotFoundError:
                pass  # not configured for this version -- no boost, not a crash
    return w


def _active_claims(con, subject_entity_type, subject_entity_id, claim_type, asof):
    df = snapshot_mod.get_claims_asof(
        con, asof, subject_entity_type=subject_entity_type,
        subject_entity_id=subject_entity_id, claim_type=claim_type,
    )
    return df.to_dict("records")


def blend_numeric(
    con: duckdb.DuckDBPyConnection, subject_entity_type: str, subject_entity_id: str,
    claim_type: str, asof: datetime, decay_params_version: int, fact_multiplier_params_version: int,
) -> float | None:
    """Weighted average of claim_value_numeric across active claims. None if there's no
    numeric evidence at all (not 0.0 -- absence of evidence isn't evidence of absence)."""
    claims = _active_claims(con, subject_entity_type, subject_entity_id, claim_type, asof)
    total_w, total_wv = 0.0, 0.0
    for c in claims:
        if c["claim_value_numeric"] is None or pd.isna(c["claim_value_numeric"]):
            continue
        w = effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
        total_w += w
        total_wv += w * c["claim_value_numeric"]
    if total_w == 0:
        return None
    return total_wv / total_w


def aggregate_evidence_weight(
    con: duckdb.DuckDBPyConnection, subject_entity_type: str, subject_entity_id: str,
    claim_types: list[str], asof: datetime, decay_params_version: int, fact_multiplier_params_version: int,
) -> float:
    """Priority 2 addition: the sum of effective_weight() across every active claim of the
    given claim_types for this subject -- a coarse "how much genuine, reliability/decay-
    weighted evidence exists" signal, not a sentiment-direction-aware one.

    community_sentiment/analyst_debate/youtube_evidence rows never carry a
    claim_value_numeric (checked: every ingest_workbook.py call site for these three types
    passes claim_value_numeric=None, their substance lives entirely in the free-text
    claim_value JSON), so blend_numeric() above always returns None for them -- there is no
    real per-player "consensus score" anywhere in this project to blend. Full NLP extraction
    of what these claims actually SAY (positive/negative, how strongly) is explicitly out of
    scope for the MVP consensus-divergence check this feeds (consensus_check.py) -- this
    returns 0.0 (not None) when there's no evidence at all, since "no evidence found" is a
    real, meaningful value here (unlike blend_numeric's None, which distinguishes "no
    evidence" from "evidence exists but isn't numeric" -- that distinction doesn't apply to a
    sum, where an empty sum is legitimately 0)."""
    claims = []
    for claim_type in claim_types:
        claims.extend(_active_claims(con, subject_entity_type, subject_entity_id, claim_type, asof))
    return sum(effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version) for c in claims)


def blend_categorical(
    con: duckdb.DuckDBPyConnection, subject_entity_type: str, subject_entity_id: str,
    claim_type: str, category_key: str, asof: datetime,
    decay_params_version: int, fact_multiplier_params_version: int,
) -> dict[str, float]:
    """Weighted probability distribution over categories found at claim_value[category_key]
    (e.g. category_key='category' for injury_status). {} if there's no categorical
    evidence at all."""
    claims = _active_claims(con, subject_entity_type, subject_entity_id, claim_type, asof)
    weights: dict[str, float] = {}
    for c in claims:
        if not c["claim_value"]:
            continue
        payload = json.loads(c["claim_value"])
        category = payload.get(category_key)
        if category is None:
            continue
        w = effective_weight(con, c, asof, decay_params_version, fact_multiplier_params_version)
        weights[category] = weights.get(category, 0.0) + w
    total = sum(weights.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in weights.items()}
