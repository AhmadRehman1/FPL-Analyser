"""Pinned exponential evidence-decay formula (M0 spec).

decay = 1.0 when a claim_type has no claim_type_decay_params row yet -- unset never
silently zeroes evidence out.
"""

from datetime import date

from . import params as params_mod


def decay(observed_date: date, asof_date: date, half_life_days: float) -> float:
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    days = (asof_date - observed_date).days
    return 0.5 ** (days / half_life_days)


def decay_for_claim_type(con, claim_type: str, observed_date: date, asof_date: date, params_version: int) -> float:
    try:
        value_numeric, _ = params_mod.resolve_param(
            con, "claim_type_decay_params", "decay_half_life_days", params_version,
            dimensions={"claim_type": claim_type},
        )
    except params_mod.ParamNotFoundError:
        return 1.0
    if value_numeric is None:
        return 1.0
    return decay(observed_date, asof_date, value_numeric)


# ============================================================
# Priority 7c: claim_type_decay_params has never actually been populated anywhere in this
# project (grepped every ingest/seed call site, confirmed zero write_param calls for this
# family before this) -- decay_for_claim_type's own try/except ParamNotFoundError -> 1.0
# fallback means every claim type has been decaying at a flat 1.0 (no decay at all) since M0,
# regardless of which decay_params_version real callers pass. This is the first real
# implementation, not a "tightening" of windows that turn out to have never existed.
#
# Every value below is an invented v1 default (no claim-type-specific decay literature to
# cite, same status as every other unpinned constant in this project), flagged for M7
# recalibration once real evidence-outcome data exists to fit against. The ordering itself is
# the deliberate part: a starting-role read (predicted_xi) turns over faster than a set-piece
# rota (set_piece_order_override), which turns over faster than a manager's season-long
# tactical tendency (manager_tendency), which should barely decay across a normal gameweek gap.
#
# predicted_xi half-life history: this was 1.5 days originally, on the mental model of a
# "same-day team-news leak worthless by kickoff." That is not what a predicted_xi claim is in
# this project -- they come from the *weekly* manual evidence pull (data/external/evidence_pull_
# sources/ + the workbook, ingested by run_ingestion.py's [evidence_claims]/[research_pull]
# steps) and predict a player's SEASON starting ROLE ("nailed starter", "rotation risk"), with
# observed_date routinely 1-4 weeks stale because the pull doesn't re-touch every player every
# week. A 1.5-day half-life drove every such claim to ~1e-3..1e-6 weight before
# minutes_model.compute_logit_adjustment() ever saw it, silently zeroing the entire predicted_xi
# channel (100+ claims) that evidence-pull-v2 exists to feed -- e.g. a confirmed nailed striker
# whose own start history is depressed by a past injury or transfer holdout got no correction at
# all. 21.0 (= transfer_likelihood: both track "the current state of a situation that evolves
# over weeks, not hours") keeps ~80% weight after one pull cycle, ~50% after three. A genuinely
# same-day claim still gets full weight regardless -- the half-life governs only how a claim
# AGES, not its initial strength.
# ============================================================

CLAIM_TYPE_DECAY_HALF_LIFE_DAYS_V1 = {
    "predicted_xi": 21.0,
    "injury_status": 3.0,
    "set_piece_order_override": 45.0,
    "community_sentiment": 10.0,
    "analyst_debate": 10.0,
    "youtube_evidence": 10.0,
    "transfer_likelihood": 21.0,
    "manager_tendency": 120.0,
}


def seed_v1_params(con) -> None:
    for claim_type, half_life_days in CLAIM_TYPE_DECAY_HALF_LIFE_DAYS_V1.items():
        params_mod.write_param(
            con, "claim_type_decay_params", 1, "2026-08-10", "decay_half_life_days",
            value_numeric=half_life_days, dimensions={"claim_type": claim_type},
        )
