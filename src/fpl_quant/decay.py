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
# A6: claim_type_decay_params was confirmed NEVER seeded anywhere in this project -- grepped
# every module before writing this (minutes_model.py, expected_points.py, and every test file)
# -- meaning decay_for_claim_type() above has been silently returning 1.0 (no decay at all) for
# every real caller since evidence_blend.effective_weight() was first written, including in the
# live pipeline (scripts/run_ingestion.py already resolves decay_params_version=1 for
# minutes_model.run()). A month-old injury report has been carrying EXACTLY the same weight as
# one from this morning. Real gap, not a hypothetical one -- fixed here with actual, claim-
# type-specific half-lives rather than one flat number, since these evidence types genuinely
# decay at different real-world rates:
#
#   predicted_xi (lineup predictions): the fastest-moving evidence in the whole project -- a
#     provisional XI guess from 3 weeks ago is close to worthless by matchday, team news
#     changes week to week. Short half-life.
#   injury_status: injuries resolve or worsen on a similar timescale (days, not weeks) -- an
#     "Out" claim from two weeks ago says much less than one from yesterday. Short half-life,
#     slightly longer than predicted_xi's (a confirmed long-term injury genuinely does stay
#     relevant longer than a specific matchday lineup guess).
#   transfer_likelihood: transfer-window rumors move on a moderate timescale -- faster than a
#     season-long pattern, slower than day-to-day team news. Medium half-life.
#   manager_tendency / set_piece_order_override: both describe a relatively STABLE, slow-
#     changing real-world fact (a manager's rotation pattern, a designated penalty taker) that
#     typically persists for months once established. Long half-life.
#
# Every value is an invented v1 default (no real claim-outcome-vs-age data was analyzed to fit
# these from), same status as every other unpinned constant in this project, flagged for M7
# recalibration once real data exists to fit them against.
# ============================================================

def seed_v1_params(con) -> None:
    half_lives = {
        "predicted_xi": 7.0,
        "injury_status": 10.0,
        "transfer_likelihood": 21.0,
        "manager_tendency": 75.0,
        "set_piece_order_override": 75.0,
    }
    for claim_type, half_life_days in half_lives.items():
        params_mod.write_param(
            con, "claim_type_decay_params", 1, "2026-08-10", "decay_half_life_days",
            value_numeric=half_life_days, dimensions={"claim_type": claim_type},
        )
