from datetime import datetime

import pytest

from fpl_quant import adversarial_check as ac
from fpl_quant import squad_optimizer as so


def _seed_pool_and_squad(con, target_season="2026-2027", target_gameweek=2, bench_p_start=0.8, complete=True):
    so.seed_v1_params(con)
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    for club in clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [club, club])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, ?, 'clubA', 'clubB', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)", [target_season, target_gameweek],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'clubA')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [target_season],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_mv = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1) RETURNING model_version", [target_season, ts_mv, mm_mv],
    ).fetchone()[0]
    un_mv = con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1) "
        "RETURNING model_version", [ep_mv, mm_mv, ts_mv],
    ).fetchone()[0]

    players = []
    for i in range(2):
        players.append((f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6]))
    for i in range(5):
        players.append((f"def{i}", "Defender", 2.5 + i * 0.3, 4.0 + i * 0.5, clubs[i % 6]))
    for i in range(5):
        players.append((f"mid{i}", "Midfielder", 3.0 + i * 0.4, 5.0 + i * 0.5, clubs[i % 6]))
    for i in range(3):
        players.append((f"fwd{i}", "Forward", 3.5 + i * 0.5, 6.0 + i * 0.5, clubs[i % 6]))

    for uid, position, mu, price, club in players:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), club, target_season, uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp)", [uid, target_season, price],
        )
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, mu],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)", [un_mv, uid],
        )
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
            "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, ?, 0.7, 1.0, 0.7, 0.0, 0.8, 0.0, 0.1, 0.1, 0.8, 20)",
            [mm_mv, uid, position],
        )

    xi_uids = [uid for uid, *_ in players[:11]] if complete else [uid for uid, *_ in players[:10]]
    bench_uids = [uid for uid, *_ in players if uid not in xi_uids]

    run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES ('2026-08-10', ?, ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 9.0) RETURNING run_id",
        [target_season, target_gameweek, ep_mv, un_mv],
    ).fetchone()[0]
    for uid, position, *_rest in players:
        in_squad = complete or (uid != players[-1][0])
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, ?, ?, ?, FALSE)",
            [run_id, uid, in_squad, uid in xi_uids, uid == "mid0"],
        )
    if bench_p_start != 0.8:
        for uid in bench_uids:
            con.execute(
                "UPDATE minutes_model_outputs SET p_start_final = ? WHERE model_version = ? AND player_uid = ?",
                [bench_p_start, mm_mv, uid],
            )
    return run_id


def test_adversarial_review_returns_every_check_regardless_of_pass_fail(con):
    """Per spec: findings are surfaced regardless of whether the squad passed -- a clean,
    fully-legal squad must still get a full report with every check explicitly marked
    triggered=False, not an empty list."""
    run_id = _seed_pool_and_squad(con)
    findings = ac.adversarial_review(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        consensus_price_band=0.5, consensus_divergence_ratio_threshold=0.2, bench_p_start_threshold=0.25,
    )
    checks = {f["check"] for f in findings}
    assert checks == {"squad_completeness", "budget_legality", "consensus_defying_captain", "concentration_risk", "weak_bench"}
    by_check = {f["check"]: f for f in findings}
    assert by_check["squad_completeness"]["triggered"] is False
    assert by_check["budget_legality"]["triggered"] is False
    assert by_check["weak_bench"]["triggered"] is False  # bench_p_start=0.8, well above 0.25


def test_adversarial_review_catches_incomplete_squad(con):
    run_id = _seed_pool_and_squad(con, complete=False)
    findings = ac.adversarial_review(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        consensus_price_band=0.5, consensus_divergence_ratio_threshold=0.2, bench_p_start_threshold=0.25,
    )
    by_check = {f["check"]: f for f in findings}
    assert by_check["squad_completeness"]["triggered"] is True


def test_adversarial_review_catches_weak_bench(con):
    run_id = _seed_pool_and_squad(con, bench_p_start=0.1)
    findings = ac.adversarial_review(
        con, run_id, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
        consensus_price_band=0.5, consensus_divergence_ratio_threshold=0.2, bench_p_start_threshold=0.25,
    )
    by_check = {f["check"]: f for f in findings}
    assert by_check["weak_bench"]["triggered"] is True


def test_adversarial_review_raises_on_unknown_run(con):
    with pytest.raises(ValueError):
        ac.adversarial_review(
            con, 999, datetime(2026, 8, 10), decay_params_version=1, fact_multiplier_params_version=1,
            consensus_price_band=0.5, consensus_divergence_ratio_threshold=0.2, bench_p_start_threshold=0.25,
        )
