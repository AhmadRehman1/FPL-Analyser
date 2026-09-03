import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_transfer_planner_for_real_squad import (  # noqa: E402
    _analytic_gameweek_ep, _build_chip_preview_squad, _gameweek_ep_model_version,
    _order_chip_evaluations, _resolve_decision_log_row, attach_recommendation_breakdowns,
    reconcile_chips_with_timing_sweep,
)


def _sweep_report():
    return {
        "comparison": {
            "eval_end_gameweek": 19, "swept_best_gameweek": 12, "swept_best_points": 660.0,
            "greedy_gameweek": 3, "greedy_total_points": 642.0,
            "swept_table": [{"gameweek": g, "total_projected_points": p}
                            for g, p in [(9, 656), (10, 659), (11, 658), (12, 660)]],
        },
        "sensitivity": {"wildcard_week_by_bundle": {"active": 12}, "wildcard_stable": True},
        "free_hit_scan": [
            {"gameweek": 3, "free_hit_gain": 17.0, "clears_threshold": True},
            {"gameweek": 8, "free_hit_gain": 22.0, "clears_threshold": True},
        ],
        "bench_boost_window": [
            {"gameweek": 12, "recommended_combo": False},
            {"gameweek": 13, "recommended_combo": True},
        ],
    }


def _chips_all_recommended():
    return [
        {"chip_type": "wildcard", "recommended": True, "score": 30.0},
        {"chip_type": "free_hit", "recommended": True, "score": 17.0},
        {"chip_type": "bench_boost", "recommended": True, "score": 5.0},
        {"chip_type": "triple_captain", "recommended": True, "score": 4.0},
    ]


def test_timing_sweep_downgrades_chips_that_are_not_at_their_best_week():
    chips = reconcile_chips_with_timing_sweep(_chips_all_recommended(), plan_for_gameweek=3, sweep_report=_sweep_report())
    by = {c["chip_type"]: c for c in chips}
    assert by["wildcard"]["recommended"] is False and by["wildcard"]["timing"]["best_gameweek"] == 12
    assert "GW12" in by["wildcard"]["detail_timing"]
    assert by["free_hit"]["recommended"] is False and by["free_hit"]["timing"]["best_gameweek"] == 8
    assert by["bench_boost"]["recommended"] is False  # best BB week is 13, not 3
    assert by["triple_captain"]["recommended"] is True  # never touched


def test_timing_sweep_keeps_the_chip_recommended_in_its_best_week():
    chips = reconcile_chips_with_timing_sweep(_chips_all_recommended(), plan_for_gameweek=12, sweep_report=_sweep_report())
    by = {c["chip_type"]: c for c in chips}
    assert by["wildcard"]["recommended"] is True and by["wildcard"]["timing"]["is_best_week_now"] is True


def test_timing_sweep_never_promotes_a_chip_the_planner_declined():
    chips = _chips_all_recommended()
    chips[0]["recommended"] = False  # planner said no to wildcard
    out = reconcile_chips_with_timing_sweep(chips, plan_for_gameweek=12, sweep_report=_sweep_report())
    assert out[0]["recommended"] is False  # still no -- sweep only ever downgrades


def test_missing_or_non_covering_sweep_leaves_flags_untouched():
    for report in (None, {"comparison": {"eval_end_gameweek": 8}}):
        chips = reconcile_chips_with_timing_sweep(_chips_all_recommended(), plan_for_gameweek=12, sweep_report=report)
        assert all(c["recommended"] for c in chips)
        assert all(c["timing"]["available"] is False for c in chips)


def test_order_chip_evaluations_matches_chip_priority_even_when_db_order_disagrees():
    # This is the exact real-world shape that motivated the fix: chip_evaluations rows come
    # back wildcard/free_hit/triple_captain/bench_boost (insertion order in transfer_planner.py),
    # the reverse of backtest.py's own CHIP_PRIORITY (bench_boost before triple_captain).
    chips_out = [
        {"chip_type": "wildcard", "recommended": False, "score": 1.0},
        {"chip_type": "free_hit", "recommended": False, "score": 2.0},
        {"chip_type": "triple_captain", "recommended": True, "score": 3.7},
        {"chip_type": "bench_boost", "recommended": True, "score": 9.0},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert [c["chip_type"] for c in ordered] == ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    # the dashboard's own chips.find(c => c.recommended) now correctly lands on bench_boost,
    # not triple_captain, when both clear their threshold in the same week.
    assert next(c for c in ordered if c["recommended"])["chip_type"] == "bench_boost"


def test_order_chip_evaluations_preserves_all_rows():
    chips_out = [
        {"chip_type": "bench_boost", "recommended": False, "score": None},
        {"chip_type": "wildcard", "recommended": False, "score": None},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert {c["chip_type"] for c in ordered} == {"bench_boost", "wildcard"}
    assert len(ordered) == 2


def test_order_chip_evaluations_unknown_chip_type_sorts_last():
    chips_out = [
        {"chip_type": "some_future_chip", "recommended": False, "score": None},
        {"chip_type": "wildcard", "recommended": False, "score": None},
    ]
    ordered = _order_chip_evaluations(chips_out)
    assert [c["chip_type"] for c in ordered] == ["wildcard", "some_future_chip"]


def test_build_chip_preview_squad_resolves_names_and_clubs():
    # Real shape: transfer_planner.read_fresh_chip_squad()'s own return rows.
    preview_rows = [
        {"player_uid": "uid-1", "in_xi": True, "is_captain": True, "is_vice": False},
        {"player_uid": "uid-2", "in_xi": False, "is_captain": False, "is_vice": False},
    ]
    name_by_uid = {"uid-1": "Erling Haaland", "uid-2": "Bernd Leno"}
    team_by_player = {"uid-1": "team-mci", "uid-2": "team-ful"}
    team_names = {"team-mci": "Man City", "team-ful": "Fulham"}

    out = _build_chip_preview_squad(preview_rows, name_by_uid, team_by_player, team_names)

    assert out == [
        {"player_name": "Erling Haaland", "club": "Man City", "in_xi": True, "is_captain": True, "is_vice": False},
        {"player_name": "Bernd Leno", "club": "Fulham", "in_xi": False, "is_captain": False, "is_vice": False},
    ]


def test_build_chip_preview_squad_falls_back_to_uid_when_name_unresolved():
    # A player_uid missing from dim_player (shouldn't happen against real data, but the
    # dashboard should still show all 15/11 names rather than silently dropping one) falls
    # back to the raw uid rather than crashing the whole export.
    preview_rows = [{"player_uid": "uid-missing", "in_xi": True, "is_captain": False, "is_vice": True}]
    out = _build_chip_preview_squad(preview_rows, {}, {}, {})
    assert out == [{"player_name": "uid-missing", "club": None, "in_xi": True, "is_captain": False, "is_vice": True}]


def test_build_chip_preview_squad_empty_input():
    assert _build_chip_preview_squad([], {}, {}, {}) == []


def test_resolve_decision_log_row_hold_is_a_real_logged_row_not_an_absence():
    # Plan Track C's own edge case: a "hold" recommendation must still produce a real row, not
    # be skipped, so "hold was correct" is measurable later just like "transfer was correct".
    hold_rec = ("hold", 12.5, 14.0)
    row = _resolve_decision_log_row(hold_rec, recs_out=[{"player_out": "A", "player_in": "B"}], chips_out=[], captain_recommendation=None)
    assert row["recommended_action"] == "hold"
    # Even though a transfer_recommendations row exists, it's not what was recommended this week.
    assert row["recommended_transfer_out"] is None
    assert row["recommended_transfer_in"] is None
    assert row["recommended_chip"] is None
    assert row["recommended_captain"] is None


def test_resolve_decision_log_row_transfer_now_uses_top_ranked_recommendation():
    hold_rec = ("transfer_now", 20.0, 14.0)
    recs_out = [
        {"player_out": "Player Out 1", "player_in": "Player In 1"},
        {"player_out": "Player Out 2", "player_in": "Player In 2"},
    ]
    row = _resolve_decision_log_row(hold_rec, recs_out, chips_out=[], captain_recommendation=None)
    assert row["recommended_action"] == "transfer_now"
    assert row["recommended_transfer_out"] == "Player Out 1"
    assert row["recommended_transfer_in"] == "Player In 1"


def test_resolve_decision_log_row_no_hold_rec_falls_back_to_no_action_available():
    row = _resolve_decision_log_row(None, recs_out=[], chips_out=[], captain_recommendation=None)
    assert row["recommended_action"] == "no_action_available"
    assert row["recommended_transfer_out"] is None
    assert row["recommended_transfer_in"] is None


def test_resolve_decision_log_row_picks_the_recommended_chip():
    chips_out = [
        {"chip_type": "wildcard", "recommended": False, "score": 1.0},
        {"chip_type": "bench_boost", "recommended": True, "score": 9.0},
    ]
    row = _resolve_decision_log_row(("hold", 1.0, 2.0), recs_out=[], chips_out=chips_out, captain_recommendation=None)
    assert row["recommended_chip"] == "bench_boost"


def test_resolve_decision_log_row_no_chip_recommended_is_none():
    chips_out = [{"chip_type": "wildcard", "recommended": False, "score": 1.0}]
    row = _resolve_decision_log_row(("hold", 1.0, 2.0), recs_out=[], chips_out=chips_out, captain_recommendation=None)
    assert row["recommended_chip"] is None


def test_resolve_decision_log_row_must_be_called_with_priority_ordered_chips():
    # Real, observed scenario this file's own header comment already documents: chip_evaluations
    # rows come back wildcard/free_hit/triple_captain/bench_boost (DB insertion order), the
    # reverse of CHIP_PRIORITY's bench_boost-before-triple_captain -- and both can legitimately
    # clear their own recommendation threshold in the same gameweek. The dashboard snapshot
    # correctly resolves this via _order_chip_evaluations(); the decision log must use the exact
    # same resolution, or the two outputs of one run would disagree about what was recommended.
    raw_db_order = [
        {"chip_type": "wildcard", "recommended": False, "score": 1.0},
        {"chip_type": "free_hit", "recommended": False, "score": 2.0},
        {"chip_type": "triple_captain", "recommended": True, "score": 3.7},
        {"chip_type": "bench_boost", "recommended": True, "score": 9.0},
    ]
    unordered_row = _resolve_decision_log_row(("hold", 1.0, 2.0), recs_out=[], chips_out=raw_db_order, captain_recommendation=None)
    ordered_row = _resolve_decision_log_row(
        ("hold", 1.0, 2.0), recs_out=[], chips_out=_order_chip_evaluations(raw_db_order), captain_recommendation=None,
    )
    # Calling with the raw, unordered rows silently picks the wrong winner -- this assertion
    # documents *why* main() must always order chips_out first, not that it's desirable.
    assert unordered_row["recommended_chip"] == "triple_captain"
    assert ordered_row["recommended_chip"] == "bench_boost"


def test_resolve_decision_log_row_captain_change_is_logged():
    captain_recommendation = {"recommended_name": "Erling Haaland", "matches_current": False}
    row = _resolve_decision_log_row(("hold", 1.0, 2.0), recs_out=[], chips_out=[], captain_recommendation=captain_recommendation)
    assert row["recommended_captain"] == "Erling Haaland"


def test_resolve_decision_log_row_captain_matching_current_is_not_logged_as_a_change():
    # No actual deviation to "follow" when the model's pick is already the manager's captain.
    captain_recommendation = {"recommended_name": "Erling Haaland", "matches_current": True}
    row = _resolve_decision_log_row(("hold", 1.0, 2.0), recs_out=[], chips_out=[], captain_recommendation=captain_recommendation)
    assert row["recommended_captain"] is None


# ============================================================
# attach_recommendation_breakdowns (Gap 5) -- transfer_plan_runs.ep_model_versions is a JSON
# OBJECT keyed by str(gameweek), NOT a list. The first cut of this did json.loads(...)[0] and
# the scheduled pipeline crashed with KeyError: 0 on every real run for ~12h. These tests pin
# the real shape and the best-effort fallback.
# ============================================================

def _seed_breakdown_scenario(con, *, gw=3, players=("player_cap", "player_in", "player_out")):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('t_h','H'),('t_a','A') ON CONFLICT DO NOTHING")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1','2026-2027',?,'t_h','t_a',FALSE,"
        "'Premier League','2026-08-24', current_timestamp)", [gw],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 't_h')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_mv = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10','2026-2027', ?, ?, 1, 1, 1) RETURNING model_version", [ts_mv, mm_mv],
    ).fetchone()[0]
    un_mv = con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1) "
        "RETURNING model_version", [ep_mv, mm_mv, ts_mv],
    ).fetchone()[0]
    for i, uid in enumerate(players):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING",
                    [uid, uid.replace("player_", "").title()])
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, 'm1', 0.9, ?, 0.3, 0.1, -0.1, 0.0, 0.5, 0, 0, -0.02, 0, ?, 20.0)",
            [ep_mv, uid, 2.0 + i, 3.68 + i],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, var_goals, "
            "var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, var_total, skew, "
            "excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 10.0, 0.4, 0.2, 0.5, 2.0, 6.0, 12.0)",
            [un_mv, uid],
        )
    state_version = con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available) "
        "VALUES ('2026-2027', ?, 1) RETURNING state_version", [gw],
    ).fetchone()[0]
    return ep_mv, un_mv, state_version


def _make_plan_run(con, state_version, ep_mv, un_mv, *, gw=3, mv_json=None):
    ep_json = mv_json if mv_json is not None else json.dumps({str(gw): ep_mv})
    un_json = mv_json if mv_json is not None else json.dumps({str(gw): un_mv})
    return con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, input_state_version, "
        "horizon_params_version, transfer_cost_params_version, ep_model_versions, uncertainty_model_versions) "
        "VALUES ('2026-08-24','2026-2027', ?, ?, 1, 1, ?, ?) RETURNING run_id",
        [gw, state_version, ep_json, un_json],
    ).fetchone()[0]


def test_attach_breakdowns_reads_the_json_object_shape_and_fills_captain_and_transfers(con):
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con)
    run_id = _make_plan_run(con, sv, ep_mv, un_mv, gw=3)
    explain = {}
    recs = [(1, "player_out", "player_in", 5.0, 0.0, 5.0)]
    out = attach_recommendation_breakdowns(
        con, explain, run_id, 3,
        tc_detail={"captain_candidate": "player_flat", "all_candidates": [
            {"player_uid": "player_cap", "mean_total": 8.0, "var_total": 20.0},
            {"player_uid": "player_flat", "mean_total": 5.0, "var_total": 1.0},
        ]},
        actual_captain_uid="player_out",
        recs=recs, name_by_uid={"player_cap": "Cap Player"},
    )
    # weekly captain follows E[points] (player_cap, 8.0), not the risk-adjusted candidate
    assert out["captain_breakdown"]["recommended"]["name"] == "Cap Player"
    assert out["captain_breakdown"]["recommended"]["ep"]["categories"]["goals"] == 2.0
    assert out["captain_breakdown"]["recommended"]["risk"]["ceiling"] == 12.0
    assert out["captain_breakdown"]["current"]["player_uid"] == "player_out"
    assert len(out["transfer_breakdowns"]) == 1
    assert out["transfer_breakdowns"][0]["player_in"]["name"] == "In"       # dim_player fallback
    assert out["transfer_breakdowns"][0]["player_out"]["ep"]["total"] is not None


def test_attach_breakdowns_omits_rather_than_crashes_when_target_gameweek_missing(con):
    # The exact regression: ep_model_versions '{}' -> no key for GW3 -> must NOT raise.
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con)
    run_id = _make_plan_run(con, sv, ep_mv, un_mv, gw=3, mv_json="{}")
    explain = {"top_transfers": []}
    out = attach_recommendation_breakdowns(
        con, explain, run_id, 3,
        tc_detail={"captain_candidate": "player_cap", "all_candidates": [{"player_uid": "player_cap", "mean_total": 6.0, "var_total": 1.0}]},
        actual_captain_uid=None, recs=[(1, "player_out", "player_in", 5.0, 0.0, 5.0)], name_by_uid={},
    )
    assert "captain_breakdown" not in out
    assert out["transfer_breakdowns"] == []


def test_attach_breakdowns_no_captain_candidate_still_does_transfers(con):
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con)
    run_id = _make_plan_run(con, sv, ep_mv, un_mv, gw=3)
    out = attach_recommendation_breakdowns(
        con, {}, run_id, 3, tc_detail=None, actual_captain_uid=None,
        recs=[(1, "player_out", "player_in", 5.0, 0.0, 5.0), (2, "player_cap", "player_in", 1.0, 0.0, 1.0)],
        name_by_uid={},
    )
    assert "captain_breakdown" not in out
    assert len(out["transfer_breakdowns"]) == 2


def test_attach_breakdowns_explains_the_passed_in_captain_not_the_mc_mean_pick(con):
    # rec_cap_uid is build_captain_recommendation()'s analytic-EP pick -- the breakdown must
    # explain THAT player, even when the MC-mean argmax would name someone else.
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con)
    run_id = _make_plan_run(con, sv, ep_mv, un_mv, gw=3)
    out = attach_recommendation_breakdowns(
        con, {}, run_id, 3,
        tc_detail={"all_candidates": [
            {"player_uid": "player_cap", "mean_total": 9.0, "var_total": 1.0},
            {"player_uid": "player_in", "mean_total": 4.0, "var_total": 1.0},
        ]},
        actual_captain_uid="player_out",
        recs=[(1, "player_out", "player_in", 5.0, 0.0, 5.0)],
        name_by_uid={"player_cap": "Cap", "player_in": "In Player"},
        rec_cap_uid="player_in",
    )
    assert out["captain_breakdown"]["recommended"]["player_uid"] == "player_in"


# ============================================================
# _gameweek_ep_model_version + _analytic_gameweek_ep -- the analytic E[points] the weekly
# captain directive ranks by (M6's MC mean_total compresses a big favourite; see
# reporting.build_captain_recommendation()).
# ============================================================

def test_gameweek_ep_model_version_reads_the_json_object_shape(con):
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con, gw=3)
    run_id = _make_plan_run(con, sv, ep_mv, un_mv, gw=3)
    assert _gameweek_ep_model_version(con, run_id, 3) == ep_mv
    assert _gameweek_ep_model_version(con, run_id, 9) is None       # gameweek absent from the object
    assert _gameweek_ep_model_version(con, 999_999, 3) is None      # no such run
    empty_run = _make_plan_run(con, sv, ep_mv, un_mv, gw=3, mv_json="{}")
    assert _gameweek_ep_model_version(con, empty_run, 3) is None    # the KeyError:0 regression shape


def test_analytic_gameweek_ep_sums_ep_total_over_a_players_fixtures(con):
    ep_mv, un_mv, sv = _seed_breakdown_scenario(con, gw=3)  # player_cap ep_total 3.68, player_in 4.68
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m2','2026-2027',3,'t_a','t_h',FALSE,"
        "'Premier League','2026-08-25', current_timestamp)"
    )
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
        "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
        "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, 'player_cap', 'm2', 0.9, 1.0, 0.3, 0.1, "
        "-0.1, 0.0, 0.5, 0, 0, -0.02, 0, 2.5, 20.0)", [ep_mv],
    )
    out = _analytic_gameweek_ep(con, ep_mv, ["player_cap", "player_in", "missing_uid"])
    assert round(out["player_cap"], 2) == 6.18   # 3.68 + 2.5, double gameweek
    assert round(out["player_in"], 2) == 4.68
    assert "missing_uid" not in out
    assert _analytic_gameweek_ep(con, ep_mv, []) == {}
