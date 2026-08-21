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

============================================================
B9: bias-guardrail review (explicit review pass, not new code -- results only, cross-checked
against the actual code, not assumed).
============================================================

(1) Does consensus/high-ownership get an implicit boost via volume-of-coverage?
Checked ingest_workbook.py's source-reliability formula: base_reliability_score =
tier_weight * log_scaled(citation_count) -- citation_count is confirmed PER-SOURCE (an
outlet's own citation count, e.g. how often "BBC Sport" itself is referenced), never
per-player. A source being widely cited is a legitimate credibility signal (log-scaled, so
diminishing returns, not linear), not a proxy for any specific player's popularity.
community_sentiment/analyst_debate/youtube_evidence -- the claim_types this concern names
most directly -- are ingested but confirmed to have ZERO live consumers anywhere in
minutes_model.py or expected_points.py (grepped both before writing this): the question is
currently moot, not merely "not found to be a problem."

A real, adjacent effect DOES exist and is worth naming even though it isn't the mechanism
above: minutes_model.compute_logit_adjustment() SUMS magnitude*effective_weight() across
every active claim of a shift claim_type for a player (not a weighted average) -- a popular
player who genuinely attracts more independent claims (more journalists covering them) will
accumulate a larger total adjustment than an identically-situated player with fewer sources
writing about them, purely from claim COUNT, capped only by minutes_adjustment_params' global
cap. This is a real, structural "more coverage -> more pull" property of the existing SUM
design (predates this review), distinct from and NOT the citation_count/reliability-score
mechanism above. A2's role-shift multiplier and A3's set-piece uplift were checked against
the same concern when built: both use a weighted AVERAGE/side-selection (blend_categorical,
or picking whichever side's total effective_weight is larger) rather than a sum, so
additional corroborating claims increase confidence in the SAME conclusion, not the
magnitude of the adjustment itself -- deliberately not replicating the sum pattern.

(2) Does anything overweight the most recent 1-2 gameweeks without explicit justification?
The only recency-weighting mechanisms found (grepped both modules for gameweek-ordered
queries and decay usage): M1's Dixon-Coles xi (model_decay_params, ~385-day half-life,
Dixon & Coles 1997's own published decay) and M2's minutes_model_decay_params xi (~200-day
half-life, explicitly reasoned in scripts/run_ingestion.py's own seeding comment: "rotation
patterns track manager tenure/squad changes more than pure calendar time"). Both are
versioned, both carry an explicit justification comment at their seed site. A6's new
claim_type_decay_params half-lives (7-75 days depending on claim_type) are the only other
recency-weighting in the codebase and are likewise individually justified per claim_type in
decay.py's own seed_v1_params(). expected_points.py's one gameweek-ordered query
(_player_rate_pool's "ORDER BY gw DESC LIMIT 1") reads the latest CUMULATIVE stats row per
season, not a specific recent gameweek's performance -- fact_player_season_stats is a
running total, so this is "read the most complete available snapshot," not a recency
weighting at all. No unexplained short-window (1-2 gameweek) bias found anywhere.
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
    if x is None or (isinstance(x, float) and pd.isna(x)):
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
