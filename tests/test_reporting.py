from datetime import date, datetime

import pytest

from fpl_quant import params
from fpl_quant import reporting


def _seed_raw_teams_csv(con, season, rows):
    """squad_optimizer.explain_run()'s club audit resolves player->team via
    reconcile._season_root_table(), which reads fact_raw's teams.csv table -- a real
    dependency, not an artifact to route around in tests (same pattern
    test_minutes_model.py already established). rows: list of (code, name)."""
    table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
    for code, name in rows:
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?)', [code, name])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', ?, ?)",
        [table, season, f"fakehash_{table}", len(rows)],
    )


def _seed_full_squad_scenario(con, captain_position="Defender"):
    """Two players (p1 captain, p2), one fixture, full M1-M6 chain + squad_optimizer_runs --
    the minimal real shape every M9 adapter reads through. Mirrors the seeding pattern
    established in test_transfer_planner.py (no cross-test-file import precedent in this
    project), adapted for a 2-player squad with real ep_outputs/uncertainty_outputs/
    monte_carlo data on both."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', ?)", [captain_position])
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Player Two', 'Forward')")
    _seed_raw_teams_csv(con, "2026-2027", [("1", "A"), ("2", "B")])
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', '2026-2027', 'team_a', 't')")
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('B', '2026-2027', 'team_b', 't')")
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player One', 'player one', '1', '2026-2027', 'p1')"
    )
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player Two', 'player two', '2', '2026-2027', 'p2')"
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2026-2027', 1, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-21', current_timestamp)"
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    for uid, position in (("p1", captain_position), ("p2", "Forward")):
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
            "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, ?, 0.7, 0.7, 0.0, 0.0, 0.7, 0.0, 0.3, 0.2, 0.5, 0)",
            [mm_mv, uid, position],
        )
    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', '2026-2027', ?, ?, 1, 1, 1)", [ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1)",
        [ep_mv, mm_mv, ts_mv],
    )
    un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]

    for uid, total in (("p1", 5.0), ("p2", 4.0)):
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 1.5,1.0,0.5,1.0,0,0.5,0.3,0,0,0,0, ?, 20.0)",
            [ep_mv, uid, total],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 2.0, 0,0, 1.0, 3.0, 7.0, 9.0)",
            [un_mv, uid],
        )

    con.execute(
        "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES ('2026-08-10', '2026-2027', 1, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 9.0) RETURNING run_id",
        [ep_mv, un_mv],
    )
    run_id = con.execute("SELECT max(run_id) FROM squad_optimizer_runs").fetchone()[0]
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p1', TRUE, TRUE, TRUE, FALSE)", [run_id],
    )
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p2', TRUE, TRUE, FALSE, TRUE)", [run_id],
    )
    params.write_param(con, "squad_optimizer_guardrail_params", 1, "2026-08-10", "xi_club_concentration_cap", value_numeric=3)

    con.execute(
        "INSERT INTO monte_carlo_run_versions (model_version, calibration_asof_date, squad_optimizer_run_id, "
        "ep_model_version, minutes_model_version, team_strength_model_version, uncertainty_model_version, "
        "rho_residual_params_version, z_fixture_lambda_representative, z_fixture_variance, n_antithetic_pairs, "
        "query_id, seed) VALUES (nextval('seq_monte_carlo_model_version'), '2026-08-10', ?, ?, ?, ?, ?, 1, 0.1, 0.1, 100, "
        "'test', 1) RETURNING model_version",
        [run_id, ep_mv, mm_mv, ts_mv, un_mv],
    )
    mc_mv = con.execute("SELECT max(model_version) FROM monte_carlo_run_versions").fetchone()[0]
    for uid, mean in (("p1", 5.2), ("p2", 4.1)):
        con.execute(
            "INSERT INTO monte_carlo_player_summary (model_version, player_uid, mean_total, var_total, "
            "quantile_05, quantile_25, quantile_75, quantile_95, min_total, max_total) "
            "VALUES (?, ?, ?, 2.0, 1.0, 3.0, 7.0, 9.0, 0.0, 15.0)", [mc_mv, uid, mean],
        )
    return run_id, ep_mv, un_mv, mc_mv


# ============================================================
# compute_automated_flags
# ============================================================

def test_compute_automated_flags_all_pass_for_a_clean_defender_captain(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    flags = reporting.compute_automated_flags(con, run_id)
    by_name = {f["name"]: f for f in flags}
    assert by_name["divergence_check"]["passed"] is True
    assert by_name["captained_goalkeeper"]["passed"] is True
    assert by_name["club_concentration"]["passed"] is True  # only 1 player per club, well under cap=3


def test_compute_automated_flags_catches_captained_goalkeeper(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Goalkeeper")
    flags = reporting.compute_automated_flags(con, run_id)
    by_name = {f["name"]: f for f in flags}
    assert by_name["captained_goalkeeper"]["passed"] is False


def test_compute_automated_flags_catches_club_at_cap(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con)
    params.write_param(con, "squad_optimizer_guardrail_params", 2, "2026-08-10", "xi_club_concentration_cap", value_numeric=1)
    con.execute("UPDATE squad_optimizer_runs SET guardrail_params_version = 2 WHERE run_id = ?", [run_id])
    flags = reporting.compute_automated_flags(con, run_id)
    by_name = {f["name"]: f for f in flags}
    # both players are on different clubs (team_a/team_b) -- each club has exactly 1 player,
    # which now equals the (lowered) cap of 1, so both clubs register as "at the cap"
    assert by_name["club_concentration"]["passed"] is False


# ============================================================
# build_report / render_report_text
# ============================================================

def test_build_report_headline_and_sections(con):
    run_id, ep_mv, un_mv, mc_mv = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id, active_param_versions={"squad_optimizer_guardrail_params": 1})

    headline = report["headline"]
    assert len(headline["squad"]) == 2
    assert headline["captain"]["player_uid"] == "p1"
    # p1 (captain, doubled) + p2, both in_xi: 5.0*2 + 4.0 = 14.0
    assert headline["total_projected_ep"] == pytest.approx(14.0)

    assert set(report["category_breakdown"].keys()) == {"p1", "p2"}
    assert report["category_breakdown"]["p1"]["total"] == pytest.approx(5.0)
    assert set(report["risk"]["analytic"].keys()) == {"p1", "p2"}
    assert set(report["risk"]["empirical"].keys()) == {"p1", "p2"}
    assert report["risk"]["empirical"]["p1"]["mean"] == pytest.approx(5.2)
    assert report["guardrail_audit"]["divergence_check_passed"] is True
    assert report["backtest_summary"] is None  # not requested
    assert report["transfer_chip_rationale"] is None  # not requested
    assert len(report["parameter_transparency"]) == 1
    assert report["human_prompt"] == reporting.HUMAN_PROMPT
    # automated_flags and human_prompt are genuinely separate keys, per the spec's own
    # explicit reasoning that automated checks must never read as self-certification
    assert "automated_flags" in report and "human_prompt" in report
    assert report["automated_flags"] != report["human_prompt"]


def test_render_report_text_includes_every_top_level_section(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id)
    text = reporting.render_report_text(report)
    assert "Player One" in text
    assert "(C)" in text  # captain marker
    assert "Automated flags" in text
    assert reporting.HUMAN_PROMPT in text


def test_build_report_raises_on_unknown_run_id(con):
    with pytest.raises(ValueError):
        reporting.build_report(con, 999)


# ============================================================
# Priority 1 -- captain_risk_eo section
# ============================================================

def test_build_report_captain_risk_eo_section_when_requested(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con, captain_position="Defender")
    for uid, price, ownership in (("p1", 5.0, 60.0), ("p2", 6.0, 10.0)):
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, "
            "_ingested_at) VALUES (?, '2026-2027', 1, ?, ?, current_timestamp)", [uid, price, ownership],
        )
    params.write_param(con, "ownership_params", 1, "2026-08-10", "captaincy_concentration", value_numeric=0.3)

    report = reporting.build_report(con, run_id, ownership_params_version=1)
    cr = report["captain_risk_eo"]
    assert cr is not None
    assert cr["captain_uid"] == "p1"
    assert cr["captain_eo"] is not None
    assert cr["posture_label"] in ("template", "differential", "unknown")


def test_build_report_captain_risk_eo_section_absent_by_default(con):
    """Absence is recorded plainly (a None section), never silently dropped -- same
    convention already established for transfer_chip_rationale/backtest_summary above."""
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id)
    assert report["captain_risk_eo"] is None
    assert "captain_risk_eo" in report


def test_render_report_text_includes_captain_risk_eo_when_present(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    for uid, price, ownership in (("p1", 5.0, 60.0), ("p2", 6.0, 10.0)):
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, "
            "_ingested_at) VALUES (?, '2026-2027', 1, ?, ?, current_timestamp)", [uid, price, ownership],
        )
    params.write_param(con, "ownership_params", 1, "2026-08-10", "captaincy_concentration", value_numeric=0.3)
    report = reporting.build_report(con, run_id, ownership_params_version=1)
    text = reporting.render_report_text(report)
    assert "Captain rank-risk" in text
