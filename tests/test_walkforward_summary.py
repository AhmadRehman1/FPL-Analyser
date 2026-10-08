"""scripts/walkforward_summary.py runs against the real schema."""

import importlib.util
from pathlib import Path

import duckdb

from fpl_quant import db

_SPEC = importlib.util.spec_from_file_location(
    "walkforward_summary", Path(__file__).resolve().parents[1] / "scripts" / "walkforward_summary.py"
)
wfs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wfs)


def test_summarize_empty_run():
    con = duckdb.connect(":memory:")
    db.apply_schema(con)
    out = wfs.summarize(con, 1)
    assert out["headline"]["beats_crowd_points_delta"] is None
    assert set(out["price_band"]) == {"<5.0", "5.0-7.0", "7.0-9.0", "9.0+"}
    assert out["captain"] == {"n": 0}
    assert out["captain_counterfactuals"] == {"n": 0}
    assert out["per_gameweek"] == []


def test_captain_rule_points_scores_each_rule_on_the_same_xi():
    # (ep, var, p95, realized): the steady 6.0 player vs the boom-or-bust 5.5 one
    gw1 = [(6.0, 4.0, 9.0, 2.0), (5.5, 25.0, 14.0, 15.0), (3.0, 1.0, 5.0, 4.0)]
    gw2 = [(6.0, 4.0, 9.0, 8.0), (5.5, 25.0, 14.0, 1.0), (3.0, 1.0, 5.0, 3.0)]
    out = wfs.captain_rule_points([gw1, gw2])
    assert out["n"] == 2
    assert out["points_per_gw"]["top_ep"] == 5.0           # (2 + 8) / 2
    assert out["points_per_gw"]["top_p95"] == 8.0          # (15 + 1) / 2
    assert out["points_per_gw"]["ep_plus_half_sd"] == 8.0  # 5.5 + 2.5 beats 6.0 + 1.0
    assert out["vs_top_ep"]["top_p95"]["mean"] == 3.0
    assert out["hindsight_points_per_gw"] == 11.5          # (15 + 8) / 2


def test_captain_rule_points_skips_gameweeks_with_missing_data():
    assert wfs.captain_rule_points([[(None, 1.0, 2.0, 3.0)]]) == {"n": 0}


def _seed_two_season_run(con):
    """One step per season: a 2024-25 cold step with a 10.0m starter and a 4.5m non-starter,
    a 2025-26 mature step with the same pair; one beats_real metric per step."""
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('a', 'A'), ('b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('star', 'S', 'Midfielder'), ('kid', 'K', 'Midfielder')")
    for season, gw, tier, beats_real, star_prior in (("2024-2025", 6, "cold", -1.0, 0.9), ("2025-2026", 20, "mature", 2.0, 0.95)):
        mv = con.execute(
            "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
            "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
            "VALUES ('2026-01-01', ?, 1, 1, 1, 1, '[]') RETURNING model_version", [season],
        ).fetchone()[0]
        for uid, prior, cost in (("star", star_prior, 10.0), ("kid", 0.1, 4.5)):
            con.execute(
                "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
                "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
                "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
                "VALUES (?, ?, 'Midfielder', ?, 0.0, ?, 0.0, ?, 0.1, 0.1, 0.1, 0.8, 0)",
                [mv, uid, prior, prior, prior],
            )
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
                "VALUES (?, ?, ?, ?, current_timestamp)", [uid, season, gw, cost],
            )
        match_id = f"m_{season}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, competition, "
            "kickoff_time, _ingested_at) VALUES (?, ?, ?, 'a', 'b', TRUE, 'Premier League', '2026-01-01', current_timestamp)",
            [match_id, season, gw],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, minutes_played, "
            "_ingested_at) VALUES ('star', ?, ?, 0, 90, 90, current_timestamp)", [match_id, season],
        )
        con.execute(
            "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, mm_model_version) "
            "VALUES (?, ?, ?, ?, '2026-01-01', ?)", [run_id, season, gw, tier, mv],
        )
        con.execute(
            "INSERT INTO backtest_metrics (backtest_run_id, season, gameweek, tier, metric_name, metric_value) "
            "VALUES (?, ?, ?, ?, 'beats_real_avg_points_delta', ?)", [run_id, season, gw, tier, beats_real],
        )
    return run_id


def test_summarize_reports_each_season_and_the_minutes_prior_by_price_band():
    con = duckdb.connect(":memory:")
    db.apply_schema(con)
    run_id = _seed_two_season_run(con)
    out = wfs.summarize(con, run_id)
    assert out["headline"]["beats_real_avg_points_delta"] == 0.5
    assert out["headline_by_season"]["2024-2025"]["beats_real_avg_points_delta"] == -1.0
    assert out["headline_by_season"]["2025-2026"]["beats_real_avg_points_delta"] == 2.0
    prior = out["minutes_prior_by_price_band"]
    assert prior["2024-2025"]["cold"]["9.0+"] == {"p_start_prior": 0.9, "p_start_final": 0.9, "started_share": 1.0, "n": 1}
    assert prior["2025-2026"]["mature"]["<5.0"]["started_share"] == 0.0


def test_promoted_club_cuts_read_the_segment_metrics_by_season():
    con = duckdb.connect(":memory:")
    db.apply_schema(con)
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    for gw, value in ((5, -3.0), (6, -2.0)):
        for name, metric_value in (
            ("match_score_log_lik_mean", value), ("match_score_log_lik_mean:promoted_match", value - 1.0),
            ("ep_total_calibration_mae:vs_promoted_team", 2.5), ("ep_total_calibration_mean_resid:promoted_team", -0.5),
        ):
            con.execute(
                "INSERT INTO backtest_metrics (backtest_run_id, season, gameweek, tier, metric_name, metric_value) "
                "VALUES (?, '2025-2026', ?, 'mature', ?, ?)", [run_id, gw, name, metric_value],
            )
    out = wfs.summarize(con, run_id)
    assert out["headline"]["match_score_log_lik_mean"] == -2.5
    assert out["promoted_clubs"] == {"2025-2026": {
        "promoted_match_score_log_lik_mean": -3.5,
        "promoted_team": {"ep_total_calibration_mean_resid": -0.5},
        "vs_promoted_team": {"ep_total_calibration_mae": 2.5},
    }}


def test_breakout_cuts_measure_the_group_at_each_step(monkeypatch):
    """The breakout block (docs/plans/2026-10_breakout_players.md, R2): realized minus predicted
    for the group at each step; promoted-club players apart; a season with no earlier season null."""
    from test_breakout import SEASON, _deadline_after, _seed

    from fpl_quant import breakout

    con = duckdb.connect(":memory:")
    db.apply_schema(con)
    _seed(con)
    monkeypatch.setattr(breakout, "build_club_spells", lambda con, seasons: None)  # _seed made the spells
    monkeypatch.setattr(wfs.backtest, "gameweek_deadline", lambda con, season, gw: _deadline_after(gw - 1))
    ts_mv = con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2025-09-01', 0.2, 1, 1, 'team_a') RETURNING model_version"
    ).fetchone()[0]
    mm_mv = con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2025-09-01', ?, 1, 1, 1, 1, '[]') RETURNING model_version", [SEASON],
    ).fetchone()[0]
    run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    for gw in (5, 6):  # GW5: only 4 club matches before it (all started); GW6: 5
        ep_mv = con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2025-09-01', ?, ?, ?, 1, 1, 1) RETURNING model_version", [SEASON, ts_mv, mm_mv],
        ).fetchone()[0]
        con.execute(
            "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, ep_model_version) "
            "VALUES (?, ?, ?, 'mature', '2025-09-01', ?)", [run_id, SEASON, gw, ep_mv],
        )
        fixture = "ab_5" if gw == 5 else "ab_4"  # any real match id; ep_outputs only needs one
        for uid, predicted, realized in (("new", 2.0, 5), ("backup", 3.0, 4), ("regular", 4.0, 4), ("promoted", 1.0, 6)):
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
                "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
                "ep_total, expected_bps) VALUES (?, ?, ?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?, 0)",
                [ep_mv, uid, fixture, predicted],
            )
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, _ingested_at) "
                "VALUES (?, ?, ?, ?, current_timestamp)", [uid, SEASON, gw, realized],
            )

    out = wfs.breakout_cuts(con, run_id)[SEASON]

    # both steps: new (+3) and backup (+1) are breakout; regular never; promoted apart (+5)
    assert out["breakout"] == {"n_player_steps": 4, "mean_resid": 2.0, "mae": 2.0}
    assert out["breakout_promoted"] == {"n_player_steps": 2, "mean_resid": 5.0, "mae": 5.0}
    assert out["breakout_widened"]["n_player_steps"] == 4
    assert {s["player"] for s in out["sample"]} == {"new", "backup"}
