"""Roadmap Feature 3 -- "Explain My Move," the single highest-leverage feature this roadmap
names. Given the manager's real squad + chips + budget, outputs ONE recommended action plus
why, the downside, "what would change my mind," and the historical track record of this
input pattern -- not a ranked player list.

Built entirely on EXISTING M8 machinery: transfer_planner.run() (the same real planning
invocation scripts/run_transfer_planner_for_real_squad.py already produces -- transfer
evaluation plus all four chip evaluations) and backtest._decide_gameweek_action() (the exact
decision rule the walk-forward backtest itself already validated against real data). One
decision engine, not two independently-invented ones: a live "Explain My Move" call and a
backtest walk-forward step now make the accept/hold call identically.

Sensitivity (the "what would change my mind" field) toggles the manager's highest-EP current
XI player to a real "ruled out" state via a connection-scoped TEMP TABLE shadow of
minutes_model_outputs -- the SAME technique backtest.asof_scope() already uses (DuckDB
resolves an unqualified table name against temp before main, so ep.run()/uncertainty.run()/
transfer_planner.run() all see the perturbed input with zero code changes elsewhere) -- then
genuinely re-solves and reports whether the recommendation flips. This is the single,
highest-value toggle for a v1 ship; Feature 4 (scenario.py) generalizes the same underlying
shadow-and-resolve technique into a standalone, multi-scenario-kind interactive layer on top
of this module's own recommend_best_move(), not a competing mechanism.
"""

import json
from dataclasses import dataclass
from datetime import date

import duckdb

from . import backtest as bt
from . import transfer_planner as tp

# Below this net-value gap between the #1 and #2 ranked transfer options, the engine is not
# confidently better on #1 -- surfaced via runner_up rather than silently picking one.
RUNNER_UP_DELTA_THRESHOLD = 0.5

# A pattern (this decision's own action-kind) seen fewer than this many times in the
# supplied historical actions log is too rare to score honestly -- the InsufficientHistory
# sentinel (optimal_in_n_of_71=None) fires instead of a fabricated rate.
INSUFFICIENT_HISTORY_SAMPLE_THRESHOLD = 5


@dataclass(frozen=True)
class Swap:
    out_player_uid: str
    in_player_uid: str
    delta_ep: float
    reason: str


@dataclass(frozen=True)
class Sensitivity:
    if_condition: str
    then_action: str
    delta_ep: float


@dataclass(frozen=True)
class TrackRecord:
    pattern: str
    optimal_in_n_of_71: int | None
    sample_size: int


@dataclass(frozen=True)
class Provenance:
    model_version: str
    data_asof: str


@dataclass(frozen=True)
class Decision:
    action: str
    swaps: list[Swap]
    ep_lift: float
    downside_ci: tuple[float, float]
    sensitivity: list[Sensitivity]
    track_record: TrackRecord
    provenance: Provenance
    runner_up: "Decision | None"


def _plan_action_and_swaps(con: duckdb.DuckDBPyConnection, plan_run_id: int, accept_transfer_rank: int | None, accept_chip: str | None):
    if accept_chip:
        row = con.execute(
            "SELECT score_or_gain FROM chip_evaluations WHERE run_id = ? AND chip_type = ?", [plan_run_id, accept_chip]
        ).fetchone()
        gain = row[0] if row and row[0] is not None else 0.0
        return accept_chip, [], gain
    if accept_transfer_rank:
        row = con.execute(
            "SELECT player_out, player_in, horizon_value_gain, net_value FROM transfer_recommendations "
            "WHERE run_id = ? AND rank = ?",
            [plan_run_id, accept_transfer_rank],
        ).fetchone()
        player_out, player_in, horizon_value_gain, net_value = row
        swaps = [Swap(out_player_uid=player_out, in_player_uid=player_in, delta_ep=horizon_value_gain, reason="model-ranked transfer")]
        return f"transfer_in:{player_out}->{player_in}", swaps, net_value
    return "roll", [], 0.0


def _squad_downside_ci(con: duckdb.DuckDBPyConnection, ep_mv: int, un_mv: int, xi_uids: set[str], captain_uid: str | None) -> tuple[float, float]:
    """Sums each XI player's own Cornish-Fisher quantile_05/quantile_95 for the target
    gameweek (captain doubled, matching real FPL scoring) -- a disclosed independence
    approximation (ignores cross-player covariance, the same simplification F1's own
    per-player bands already carry), not a joint Monte Carlo draw of the whole squad."""
    if not xi_uids:
        return (0.0, 0.0)
    placeholders = ",".join(["?"] * len(xi_uids))
    rows = con.execute(
        f"SELECT u.player_uid, u.quantile_05, u.quantile_95 FROM uncertainty_outputs u "
        f"JOIN ep_outputs o ON o.model_version = ? AND o.player_uid = u.player_uid AND o.fixture_match_id = u.fixture_match_id "
        f"WHERE u.model_version = ? AND u.player_uid IN ({placeholders})",
        [ep_mv, un_mv, *xi_uids],
    ).fetchall()
    low = sum((q05 * 2 if uid == captain_uid else q05) for uid, q05, q95 in rows)
    high = sum((q95 * 2 if uid == captain_uid else q95) for uid, q05, q95 in rows)
    return (low, high)


def _track_record(action: str, historical_actions: list[dict] | None) -> TrackRecord:
    action_kind = action.split(":")[0]
    pattern = f"action_kind:{action_kind}"
    if not historical_actions or len(historical_actions) < INSUFFICIENT_HISTORY_SAMPLE_THRESHOLD:
        return TrackRecord(pattern=pattern, optimal_in_n_of_71=None, sample_size=len(historical_actions or []))
    matches = sum(
        1 for a in historical_actions
        if (a.get("accepted_chip") == action_kind)
        or (action_kind == "transfer_in" and a.get("accepted_transfer_rank") is not None)
        or (action_kind == "roll" and a.get("accepted_chip") is None and a.get("accepted_transfer_rank") is None)
    )
    return TrackRecord(pattern=pattern, optimal_in_n_of_71=matches, sample_size=len(historical_actions))


def _injury_sensitivity(
    con: duckdb.DuckDBPyConnection, run_kwargs: dict, *, target_ep_mv: int | None, xi_uids: set[str],
    baseline_action: str, baseline_ep_lift: float,
) -> list[Sensitivity]:
    if not xi_uids or target_ep_mv is None:
        return []
    placeholders = ",".join(["?"] * len(xi_uids))
    star = con.execute(
        f"SELECT player_uid FROM ep_outputs WHERE model_version = ? AND player_uid IN ({placeholders}) "
        f"ORDER BY ep_total DESC LIMIT 1",
        [target_ep_mv, *xi_uids],
    ).fetchone()
    if star is None:
        return []
    star_uid = star[0]

    con.execute(
        "CREATE OR REPLACE TEMP TABLE minutes_model_outputs AS "
        "SELECT * REPLACE (1.0 AS p_0min, 0.0 AS p_1_59min, 0.0 AS p_60plus_min) FROM main.minutes_model_outputs WHERE player_uid = ? "
        "UNION ALL SELECT * FROM main.minutes_model_outputs WHERE player_uid != ?",
        [star_uid, star_uid],
    )
    try:
        perturbed = recommend_best_move(con, **run_kwargs, include_sensitivity=False, _is_runner_up_call=True)
    finally:
        con.execute("DROP TABLE IF EXISTS minutes_model_outputs")

    if perturbed.action == baseline_action:
        return []
    return [Sensitivity(
        if_condition=f"{star_uid} ruled out", then_action=perturbed.action,
        delta_ep=perturbed.ep_lift - baseline_ep_lift,
    )]


def recommend_best_move(
    con: duckdb.DuckDBPyConnection,
    entry_id: int,
    calibration_asof_date: date,
    target_season: str,
    target_gameweek: int,
    input_state_version: int,
    ts_model_version: int,
    mm_model_version: int,
    horizon_params_version: int,
    scoring_params_version: int,
    bps_params_version: int,
    tau_params_version: int,
    rho_residual_params_version: int,
    corr_params_version: int,
    transfer_cost_params_version: int,
    lambda_params_version: int,
    guardrail_params_version: int,
    wildcard_threshold_params_version: int,
    free_hit_threshold_params_version: int,
    kappa_tc_params_version: int,
    *,
    rank_posture: str = "neutral",
    historical_actions: list[dict] | None = None,
    include_sensitivity: bool = True,
    _is_runner_up_call: bool = False,
) -> Decision:
    """rank_posture is accepted for interface stability with the roadmap's own suggested
    contract but not yet wired into transfer_planner.run() (which has no rank-relative-
    variance posture lever of its own outside squad_optimizer's EO/field-covariance terms,
    themselves gated behind separate opt-in params this function doesn't thread through) --
    left as a documented no-op rather than silently ignored without a trace.

    historical_actions: an already-computed backtest.run_season_simulation()['actions'] log
    (or an equivalent list of {"accepted_transfer_rank", "accepted_chip"} dicts) to score
    track_record against. None (the default) always yields the InsufficientHistory sentinel
    -- this function never re-runs a 71-gameweek backtest itself just to answer one live
    recommendation.
    """
    plan_run_id = tp.run(
        con, calibration_asof_date, target_season, target_gameweek, input_state_version,
        ts_model_version, mm_model_version, horizon_params_version, scoring_params_version,
        bps_params_version, tau_params_version, rho_residual_params_version, corr_params_version,
        transfer_cost_params_version, lambda_params_version, guardrail_params_version,
        wildcard_threshold_params_version, free_hit_threshold_params_version, kappa_tc_params_version,
    )

    state_row = con.execute(
        "SELECT chips_used_set1, chips_used_set2 FROM manager_state_versions WHERE state_version = ?",
        [input_state_version],
    ).fetchone()
    chips_used_set1 = set(json.loads(state_row[0])) if state_row and state_row[0] else set()
    chips_used_set2 = set(json.loads(state_row[1])) if state_row and state_row[1] else set()

    accept_transfer_rank, accept_chip = bt._decide_gameweek_action(
        con, plan_run_id, chips_used_set1, chips_used_set2, target_gameweek, accept_transfer_if_net_value_above=0.0,
    )
    action, swaps, ep_lift = _plan_action_and_swaps(con, plan_run_id, accept_transfer_rank, accept_chip)

    ep_mv_json, un_mv_json = con.execute(
        "SELECT ep_model_versions, uncertainty_model_versions FROM transfer_plan_runs WHERE run_id = ?", [plan_run_id]
    ).fetchone()
    target_ep_mv = json.loads(ep_mv_json).get(str(target_gameweek))
    target_un_mv = json.loads(un_mv_json).get(str(target_gameweek))

    current_holdings = tp._read_holdings(con, input_state_version)
    resulting_uids = {h["player_uid"] for h in current_holdings if h["in_xi"]}
    captain_uid = next((h["player_uid"] for h in current_holdings if h["is_captain"]), None)
    if swaps:
        out_uid, in_uid = swaps[0].out_player_uid, swaps[0].in_player_uid
        resulting_uids = (resulting_uids - {out_uid}) | {in_uid}
        if captain_uid == out_uid:
            captain_uid = in_uid

    downside_ci = (
        _squad_downside_ci(con, target_ep_mv, target_un_mv, resulting_uids, captain_uid)
        if target_ep_mv is not None else (0.0, 0.0)
    )

    run_kwargs = dict(
        entry_id=entry_id, calibration_asof_date=calibration_asof_date, target_season=target_season,
        target_gameweek=target_gameweek, input_state_version=input_state_version,
        ts_model_version=ts_model_version, mm_model_version=mm_model_version,
        horizon_params_version=horizon_params_version, scoring_params_version=scoring_params_version,
        bps_params_version=bps_params_version, tau_params_version=tau_params_version,
        rho_residual_params_version=rho_residual_params_version, corr_params_version=corr_params_version,
        transfer_cost_params_version=transfer_cost_params_version, lambda_params_version=lambda_params_version,
        guardrail_params_version=guardrail_params_version, wildcard_threshold_params_version=wildcard_threshold_params_version,
        free_hit_threshold_params_version=free_hit_threshold_params_version, kappa_tc_params_version=kappa_tc_params_version,
        rank_posture=rank_posture, historical_actions=historical_actions,
    )

    sensitivity: list[Sensitivity] = []
    if include_sensitivity and not _is_runner_up_call:
        current_xi_uids = {h["player_uid"] for h in current_holdings if h["in_xi"]}
        sensitivity = _injury_sensitivity(
            con, run_kwargs, target_ep_mv=target_ep_mv, xi_uids=current_xi_uids,
            baseline_action=action, baseline_ep_lift=ep_lift,
        )

    track_record = _track_record(action, historical_actions)
    provenance = Provenance(
        model_version=f"ep_v{target_ep_mv}/un_v{target_un_mv}/plan_run{plan_run_id}",
        data_asof=calibration_asof_date.isoformat(),
    )

    runner_up = None
    if accept_transfer_rank is not None and not _is_runner_up_call:
        top_row = con.execute(
            "SELECT net_value FROM transfer_recommendations WHERE run_id = ? AND rank = ?", [plan_run_id, accept_transfer_rank]
        ).fetchone()
        second_row = con.execute(
            "SELECT net_value FROM transfer_recommendations WHERE run_id = ? AND rank = ?", [plan_run_id, accept_transfer_rank + 1]
        ).fetchone()
        if top_row is not None and second_row is not None and (top_row[0] - second_row[0]) < RUNNER_UP_DELTA_THRESHOLD:
            ru_action, ru_swaps, ru_ep_lift = _plan_action_and_swaps(con, plan_run_id, accept_transfer_rank + 1, None)
            runner_up = Decision(
                action=ru_action, swaps=ru_swaps, ep_lift=ru_ep_lift, downside_ci=(0.0, 0.0), sensitivity=[],
                track_record=_track_record(ru_action, historical_actions), provenance=provenance, runner_up=None,
            )

    return Decision(
        action=action, swaps=swaps, ep_lift=ep_lift, downside_ci=downside_ci,
        sensitivity=sensitivity, track_record=track_record, provenance=provenance, runner_up=runner_up,
    )
