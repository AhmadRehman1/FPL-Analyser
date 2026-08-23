"""Roadmap Feature 5: price-change prediction, surfaced as a transfer-timing hint --
strictly informational, never folded into the EP ranking or the optimizer objective.

Reuses transfer_planner.price_change_risk_by_player() (Priority 4) unchanged rather than
re-deriving the threshold logic here: FPL's exact proprietary price-change formula (how
large a net-transfer swing at a given ownership level actually triggers a real change) is
not public, so that function compares this-gameweek-only net transfer activity
(transfers_in_event - transfers_out_event, NOT season-cumulative) against explicit,
versioned rise_threshold/fall_threshold floats -- a real, disclosed approximation, not a
claim to replicate FPL's undisclosed algorithm exactly.

Methodology, stated plainly (not a black box): a player's forecast direction is "rise" when
today's net transfer count clears rise_threshold, "fall" when it clears fall_threshold in
the other direction, "stable" otherwise. confidence is a simple, disclosed ratio of how far
net transfers have cleared the relevant threshold (net / threshold, clipped to [0, 1]) --
NOT a calibrated probability from any backtested model; there is no historical FPL
price-change ground truth ingested anywhere in this project to calibrate against. Every
real FPL price change is exactly one £0.1m step, so delta_pence is always +-10 (0 when
stable) -- never a magnitude this project has no way to actually predict.

This module NEVER appears in expected_points/uncertainty/squad_optimizer's own ranking
inputs -- see transfer_planner.evaluate_transfers()'s own docstring guarantee (verified by
test_evaluate_transfers_attaches_momentum_without_changing_the_ranking and its price-change-
risk sibling) that attaching this signal never changes which transfer wins.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb

from . import transfer_planner as tp

DELTA_PENCE_PER_STEP = 10


@dataclass(frozen=True)
class PriceForecast:
    player_uid: str
    direction: str  # "rise" | "fall" | "stable"
    delta_pence: int
    estimated_date: str | None
    confidence: float


def forecast_price_changes(
    con: duckdb.DuckDBPyConnection, *, target_season: str, as_of_gameweek: int,
    rise_threshold: float, fall_threshold: float, data_asof: date,
) -> list[PriceForecast]:
    """One PriceForecast per player with real transfers_in_event/transfers_out_event data at
    (target_season, as_of_gameweek) -- a player with no such data (direction=None from
    price_change_risk_by_player(), see its own docstring on why that's distinguished from a
    real "stable") is omitted entirely, never defaulted to a fabricated "stable" forecast.
    """
    risk_by_player = tp.price_change_risk_by_player(con, target_season, as_of_gameweek, rise_threshold, fall_threshold)
    estimated_date = (data_asof + timedelta(days=1)).isoformat()

    out = []
    for player_uid, risk in risk_by_player.items():
        direction = risk["price_change_direction"]
        if direction is None:
            continue
        net = risk["net_transfers_event"]
        if direction == "rise":
            delta_pence = DELTA_PENCE_PER_STEP
            confidence = min(1.0, abs(net) / rise_threshold) if rise_threshold else 0.0
            est_date = estimated_date
        elif direction == "fall":
            delta_pence = -DELTA_PENCE_PER_STEP
            confidence = min(1.0, abs(net) / fall_threshold) if fall_threshold else 0.0
            est_date = estimated_date
        else:
            direction = "stable"
            delta_pence = 0
            confidence = 0.0
            est_date = None
        out.append(PriceForecast(
            player_uid=player_uid, direction=direction, delta_pence=delta_pence,
            estimated_date=est_date, confidence=confidence,
        ))
    return out
