"""Roadmap Feature 4: a standalone, interactive "what-if" layer generalizing Feature 3's own
injury-toggle sensitivity mechanism into multiple scenario kinds. Built on
decision_engine.recommend_best_move() (called twice -- once unperturbed, once against a
temp-shadowed input) rather than a second, parallel decision mechanism.

Every scenario shadows real input tables via a connection-scoped TEMP TABLE (the same
technique backtest.asof_scope() and decision_engine._injury_sensitivity() already use) --
mutates a COPY of model inputs, never param_versions or any other committed table, and the
shadow is always dropped before this function returns, success or failure. A caller re-runs
recommend_best_move() itself afterward for the REAL, unperturbed recommendation; provenance
from a scenario call must never be mistaken for a real model run.

dgw_swing is validated (team_uids/add_gws structure) but not yet applied -- expected_points.
run() itself has a documented DGW-out-of-scope boundary (see its own module docstring); a
scenario that tried to model a double gameweek would silently misrepresent that gap rather
than genuinely simulate it. apply_scenario() raises NotImplementedError for this kind,
loudly, rather than pretending to support it.
"""

from dataclasses import dataclass

import duckdb

from . import decision_engine as de
from .errors import InvalidScenarioError

SCENARIO_KINDS = ("injury", "lineup_change", "price_change", "dgw_swing")


@dataclass(frozen=True)
class Scenario:
    kind: str
    player_uid: str | None = None
    team_uids: list[str] | None = None
    out_for_gws: int | None = None
    starting: bool | None = None
    delta: float | None = None
    add_gws: list[int] | None = None


@dataclass(frozen=True)
class ScenarioResult:
    baseline_decision: de.Decision
    perturbed_decision: de.Decision
    delta_ep: float
    flipped: bool


def validate_scenario(scenario: Scenario, con: duckdb.DuckDBPyConnection) -> None:
    """Raises InvalidScenarioError on an unknown kind, an unknown player_uid/team_uid, or an
    impossible perturbation. Never touches the DB beyond read-only existence checks."""
    if scenario.kind not in SCENARIO_KINDS:
        raise InvalidScenarioError(f"unknown scenario kind: {scenario.kind!r} (must be one of {SCENARIO_KINDS})")

    if scenario.kind in ("injury", "lineup_change", "price_change"):
        if not scenario.player_uid:
            raise InvalidScenarioError(f"{scenario.kind} scenario needs player_uid")
        if con.execute("SELECT 1 FROM dim_player WHERE player_uid = ?", [scenario.player_uid]).fetchone() is None:
            raise InvalidScenarioError(f"unknown player_uid: {scenario.player_uid!r}")

    if scenario.kind == "injury" and scenario.out_for_gws is not None and scenario.out_for_gws < 1:
        raise InvalidScenarioError(f"injury scenario out_for_gws must be >= 1, got {scenario.out_for_gws}")

    if scenario.kind == "lineup_change" and scenario.starting is None:
        raise InvalidScenarioError("lineup_change scenario needs starting (bool)")

    if scenario.kind == "price_change":
        if scenario.delta is None:
            raise InvalidScenarioError("price_change scenario needs delta")
        current = con.execute(
            "SELECT now_cost FROM fact_player_season_stats WHERE player_uid = ? AND now_cost IS NOT NULL "
            "QUALIFY row_number() OVER (PARTITION BY player_uid ORDER BY gw DESC) = 1",
            [scenario.player_uid],
        ).fetchone()
        if current is not None and current[0] is not None and current[0] + scenario.delta <= 0:
            raise InvalidScenarioError(
                f"price_change delta {scenario.delta} would take {scenario.player_uid}'s price "
                f"({current[0]}) to <= 0 -- impossible"
            )

    if scenario.kind == "dgw_swing":
        if not scenario.team_uids:
            raise InvalidScenarioError("dgw_swing scenario needs team_uids")
        for team_uid in scenario.team_uids:
            if con.execute("SELECT 1 FROM dim_team WHERE team_uid = ?", [team_uid]).fetchone() is None:
                raise InvalidScenarioError(f"unknown team_uid: {team_uid!r}")
        if not scenario.add_gws:
            raise InvalidScenarioError("dgw_swing scenario needs add_gws")


def _minutes_shadow(con: duckdb.DuckDBPyConnection, player_uid: str, p0: float, p1: float, p2: float) -> None:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE minutes_model_outputs AS "
        "SELECT * REPLACE (? AS p_0min, ? AS p_1_59min, ? AS p_60plus_min) FROM main.minutes_model_outputs WHERE player_uid = ? "
        "UNION ALL SELECT * FROM main.minutes_model_outputs WHERE player_uid != ?",
        [p0, p1, p2, player_uid, player_uid],
    )


def _price_shadow(con: duckdb.DuckDBPyConnection, player_uid: str, delta: float) -> None:
    con.execute(
        "CREATE OR REPLACE TEMP TABLE fact_player_season_stats AS "
        "SELECT * REPLACE (now_cost + ? AS now_cost) FROM main.fact_player_season_stats WHERE player_uid = ? "
        "UNION ALL SELECT * FROM main.fact_player_season_stats WHERE player_uid != ?",
        [delta, player_uid, player_uid],
    )


def apply_scenario(con: duckdb.DuckDBPyConnection, base_state: dict, scenario: Scenario) -> ScenarioResult:
    """base_state: the exact kwargs dict decision_engine.recommend_best_move() itself takes
    (entry_id, calibration_asof_date, target_season, target_gameweek, input_state_version,
    ts_model_version, mm_model_version, and every *_params_version) -- one real, already-
    calibrated model run's worth of context, not re-derived here. Always calls
    recommend_best_move() with include_sensitivity=False for both legs (a scenario re-solve
    computing its OWN sensitivity toggle would be a second, nested what-if -- out of scope
    for one apply_scenario() call).
    """
    validate_scenario(scenario, con)

    baseline = de.recommend_best_move(con, **base_state, include_sensitivity=False)

    if scenario.kind == "dgw_swing":
        raise NotImplementedError(
            "dgw_swing is validated but not yet applied -- expected_points.run() has a "
            "documented DGW-out-of-scope boundary this scenario kind would need to work "
            "around first, not silently paper over."
        )

    shadow_table = "minutes_model_outputs" if scenario.kind in ("injury", "lineup_change") else "fact_player_season_stats"
    try:
        if scenario.kind == "injury":
            _minutes_shadow(con, scenario.player_uid, 1.0, 0.0, 0.0)
        elif scenario.kind == "lineup_change":
            if scenario.starting:
                _minutes_shadow(con, scenario.player_uid, 0.0, 0.0, 1.0)
            else:
                _minutes_shadow(con, scenario.player_uid, 1.0, 0.0, 0.0)
        elif scenario.kind == "price_change":
            _price_shadow(con, scenario.player_uid, scenario.delta)

        perturbed = de.recommend_best_move(con, **base_state, include_sensitivity=False)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {shadow_table}")

    return ScenarioResult(
        baseline_decision=baseline, perturbed_decision=perturbed,
        delta_ep=perturbed.ep_lift - baseline.ep_lift, flipped=perturbed.action != baseline.action,
    )
