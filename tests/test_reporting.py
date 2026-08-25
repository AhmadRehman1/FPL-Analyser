from datetime import datetime

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


def test_compute_automated_flags_sanity_checks_are_opt_in(con):
    """Without sanity_check_params_version, the two new Priority 2 flags must not appear at
    all -- backward compatible for every existing caller that doesn't pass it."""
    run_id, *_ = _seed_full_squad_scenario(con)
    flags = reporting.compute_automated_flags(con, run_id)
    names = {f["name"] for f in flags}
    assert "nailed_attacking_return" not in names
    assert "rotation_risk_def_mid" not in names


def test_compute_automated_flags_catches_no_nailed_attacking_return(con):
    """_seed_full_squad_scenario's p2 (Forward) has p_start_final=0.7, below the spec's own
    0.75 threshold -- with only one attacking-position player in the XI, the flag must fire."""
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "nailed_p_start_threshold", value_numeric=0.75)
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "rotation_risk_p_start_threshold", value_numeric=0.55)
    flags = reporting.compute_automated_flags(con, run_id, sanity_check_params_version=1)
    by_name = {f["name"]: f for f in flags}
    assert by_name["nailed_attacking_return"]["passed"] is False


def test_compute_automated_flags_passes_when_a_mid_fwd_is_clearly_nailed(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con, captain_position="Defender")
    mm_mv = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [un_mv]
    ).fetchone()[0]
    con.execute("UPDATE minutes_model_outputs SET p_start_final = 0.95 WHERE model_version = ? AND player_uid = 'p2'", [mm_mv])
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "nailed_p_start_threshold", value_numeric=0.75)
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "rotation_risk_p_start_threshold", value_numeric=0.55)
    flags = reporting.compute_automated_flags(con, run_id, sanity_check_params_version=1)
    by_name = {f["name"]: f for f in flags}
    assert by_name["nailed_attacking_return"]["passed"] is True


def test_compute_automated_flags_catches_rotation_risk_when_all_def_mid_below_threshold(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con, captain_position="Defender")
    mm_mv = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [un_mv]
    ).fetchone()[0]
    con.execute("UPDATE minutes_model_outputs SET p_start_final = 0.3 WHERE model_version = ? AND player_uid = 'p1'", [mm_mv])
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "nailed_p_start_threshold", value_numeric=0.75)
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "rotation_risk_p_start_threshold", value_numeric=0.55)
    flags = reporting.compute_automated_flags(con, run_id, sanity_check_params_version=1)
    by_name = {f["name"]: f for f in flags}
    assert by_name["rotation_risk_def_mid"]["passed"] is False


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


def test_build_report_z_fixture_correlation_dilution_absent_without_empirical_pairs(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id, active_param_versions={"squad_optimizer_guardrail_params": 1})
    assert report["z_fixture_correlation_dilution"] is None


def test_build_report_z_fixture_correlation_dilution_reports_the_real_spread(con):
    """Phase B hardening (see monte_carlo.z_fixture_correlation_distribution's own docstring):
    a single representative-lambda calibration doesn't imply a single correlation across every
    pair, so this must report the actual min/median/max spread, not just a mean."""
    run_id, ep_mv, un_mv, mc_mv = _seed_full_squad_scenario(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p3', 'Player Three', 'Forward')")
    con.execute(
        "INSERT INTO monte_carlo_player_summary (model_version, player_uid, mean_total, var_total, "
        "quantile_05, quantile_25, quantile_75, quantile_95, min_total, max_total) "
        "VALUES (?, 'p3', 4.0, 8.0, 1.0, 3.0, 7.0, 9.0, 0.0, 15.0)", [mc_mv],
    )
    # p1/p2 var_total=2.0 each (see _seed_full_squad_scenario); p3 var_total=8.0 above.
    # p1/p2 (opponents in fixture m1): corr = 1.0/sqrt(2*2) = 0.5
    # p1/p3 (teammate, hypothetical): corr = 1.0/sqrt(2*8) = 0.25
    # p2/p3 (teammate, hypothetical): corr = 3.0/sqrt(2*8) = 0.75
    for a, b, relationship, cov in (("p1", "p2", "opponent", 1.0), ("p1", "p3", "teammate", 1.0), ("p2", "p3", "teammate", 3.0)):
        con.execute(
            "INSERT INTO monte_carlo_empirical_covariance "
            "(model_version, player_uid_a, player_uid_b, relationship, empirical_covariance, m4_covariance) "
            "VALUES (?, ?, ?, ?, ?, 0.15)",
            [mc_mv, a, b, relationship, cov],
        )

    report = reporting.build_report(con, run_id, active_param_versions={"squad_optimizer_guardrail_params": 1})
    dilution = report["z_fixture_correlation_dilution"]
    assert dilution["n_pairs"] == 3
    assert dilution["min"] == pytest.approx(0.25)
    assert dilution["median"] == pytest.approx(0.5)
    assert dilution["max"] == pytest.approx(0.75)
    assert dilution["mean"] == pytest.approx(0.5)


def test_render_report_text_includes_every_top_level_section(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id)
    text = reporting.render_report_text(report)
    assert "Player One" in text
    assert "(C)" in text  # captain marker
    assert "Automated flags" in text
    assert reporting.HUMAN_PROMPT in text
    assert "90% range" in text
    assert "Player confidence scores" in text
    assert "confidence=0.00" in text  # both seeded players have weight_own=0.0


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


# ============================================================
# Priority 2 -- consensus_divergence / adversarial_review sections
# ============================================================

def test_build_report_consensus_and_adversarial_sections_absent_by_default(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id)
    assert report["consensus_divergence"] is None
    assert report["adversarial_review"] is None


def test_build_report_consensus_and_adversarial_sections_when_requested(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con, captain_position="Defender")
    for uid, price in (("p1", 5.0), ("p2", 6.0)):
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, '2026-2027', 1, ?, current_timestamp)", [uid, price],
        )
    reporting.seed_v1_params(con)
    params.write_param(con, "bench_quality_params", 1, "2026-08-10", "min_bench_p_start_probability", value_numeric=0.25)

    report = reporting.build_report(
        con, run_id,
        consensus_check_params_version=1, evidence_decay_params_version=1,
        evidence_fact_multiplier_params_version=1, bench_quality_params_version=1,
        report_asof=datetime(2026, 8, 10),
    )
    assert report["consensus_divergence"] is not None  # a real list (possibly empty), not None
    assert isinstance(report["consensus_divergence"], list)
    assert report["adversarial_review"] is not None
    checks = {f["check"] for f in report["adversarial_review"]}
    assert "budget_legality" in checks and "squad_completeness" in checks


# ============================================================
# Priority 6 -- _squad_ep_range / player_confidence_score (pure logic)
# ============================================================

def test_squad_ep_range_computes_normal_approximation_with_captain_doubling():
    risk_analytic = {"p1": {"var_total": 2.0}, "p2": {"var_total": 2.0}}
    r = reporting._squad_ep_range(risk_analytic, {"p1", "p2"}, "p1", {}, total_ep=14.0)
    # total_var = (1+3)*2.0 [captain p1] + (1+0)*2.0 [p2] = 10.0 -> std_dev = sqrt(10)
    import math
    expected_std = math.sqrt(10.0)
    assert r["std_dev"] == pytest.approx(expected_std)
    assert r["floor"] == pytest.approx(14.0 - 1.645 * expected_std)
    assert r["ceiling"] == pytest.approx(14.0 + 1.645 * expected_std)


def test_squad_ep_range_includes_cross_covariance_term():
    risk_analytic = {"p1": {"var_total": 2.0}, "p2": {"var_total": 2.0}}
    sigma_pairs = {("p1", "p2"): 0.5}
    r_with_cov = reporting._squad_ep_range(risk_analytic, {"p1", "p2"}, None, sigma_pairs, total_ep=14.0)
    r_without_cov = reporting._squad_ep_range(risk_analytic, {"p1", "p2"}, None, {}, total_ep=14.0)
    assert r_with_cov["std_dev"] > r_without_cov["std_dev"]


def test_squad_ep_range_missing_variance_data_returns_none_range():
    risk_analytic = {"p1": {"var_total": 2.0}}  # p2 missing entirely
    r = reporting._squad_ep_range(risk_analytic, {"p1", "p2"}, None, {}, total_ep=14.0)
    assert r["floor"] is None and r["ceiling"] is None and r["std_dev"] is None
    assert "insufficient" in r["caveat"]


def test_squad_ep_range_empty_xi_returns_none_range():
    r = reporting._squad_ep_range({}, set(), None, {}, total_ep=0.0)
    assert r["floor"] is None
    assert "insufficient" in r["caveat"]


def test_squad_ep_range_refuses_negative_variance():
    risk_analytic = {"p1": {"var_total": 1.0}, "p2": {"var_total": 1.0}}
    # a wildly negative covariance can drive the aggregate variance below zero -- must be
    # refused rather than silently sqrt()'d or reported nonsensically.
    sigma_pairs = {("p1", "p2"): -100.0}
    r = reporting._squad_ep_range(risk_analytic, {"p1", "p2"}, None, sigma_pairs, total_ep=10.0)
    assert r["floor"] is None and r["ceiling"] is None
    assert "negative variance" in r["caveat"]


def test_player_confidence_score_both_components():
    result = reporting.player_confidence_score(weight_own=0.8, evidence_weight=2.5, evidence_weight_normalization=5.0)
    assert result["confidence_score"] == pytest.approx((0.8 + 0.5) / 2)
    assert result["normalized_evidence_weight"] == pytest.approx(0.5)


def test_player_confidence_score_weight_own_only():
    result = reporting.player_confidence_score(weight_own=0.6, evidence_weight=None, evidence_weight_normalization=5.0)
    assert result["confidence_score"] == pytest.approx(0.6)
    assert result["normalized_evidence_weight"] is None


def test_player_confidence_score_neither_component_available():
    result = reporting.player_confidence_score(weight_own=None, evidence_weight=None, evidence_weight_normalization=5.0)
    assert result["confidence_score"] is None  # absence of data, not "zero confidence"


def test_player_confidence_score_clamps_evidence_weight_above_normalization():
    result = reporting.player_confidence_score(weight_own=None, evidence_weight=50.0, evidence_weight_normalization=5.0)
    assert result["normalized_evidence_weight"] == pytest.approx(1.0)
    assert result["confidence_score"] == pytest.approx(1.0)


# ============================================================
# Priority 6 -- build_report wiring: total_projected_ep_range, category_breakdown
# floor/ceiling, confidence_scores
# ============================================================

def test_build_report_includes_squad_ep_range_and_breakdown_floor_ceiling(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)

    r = report["headline"]["total_projected_ep_range"]
    assert r["floor"] is not None and r["ceiling"] is not None
    assert r["floor"] < report["headline"]["total_projected_ep"] < r["ceiling"]

    assert report["category_breakdown"]["p1"]["floor"] == pytest.approx(1.0)
    assert report["category_breakdown"]["p1"]["ceiling"] == pytest.approx(9.0)


def test_build_report_confidence_scores_present_with_weight_own_only_by_default(con):
    """confidence_scores is always computed (not opt-in); without the evidence-weight params
    only the weight_own component is available."""
    run_id, *_ = _seed_full_squad_scenario(con)
    report = reporting.build_report(con, run_id)
    scores = report["confidence_scores"]
    assert set(scores.keys()) == {"p1", "p2"}
    for uid in ("p1", "p2"):
        assert scores[uid]["weight_own"] == pytest.approx(0.0)  # seeded weight_own=0.0
        assert scores[uid]["evidence_weight"] is None
        assert scores[uid]["confidence_score"] == pytest.approx(0.0)


def test_build_report_confidence_scores_include_evidence_weight_when_params_given(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    reporting.seed_v1_params(con)
    report = reporting.build_report(
        con, run_id,
        consensus_check_params_version=1, evidence_decay_params_version=1,
        evidence_fact_multiplier_params_version=1, confidence_score_params_version=1,
        report_asof=datetime(2026, 8, 10),
    )
    scores = report["confidence_scores"]
    for uid in ("p1", "p2"):
        # no evidence claims seeded -> aggregate_evidence_weight sums to 0.0, a real value
        assert scores[uid]["evidence_weight"] == pytest.approx(0.0)
        assert scores[uid]["normalized_evidence_weight"] == pytest.approx(0.0)


# ============================================================
# Priority 7a -- understat_signal section
# ============================================================

def test_build_report_understat_signal_empty_when_no_understat_data(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    assert report["understat_signal"] == {}


def test_build_report_understat_signal_present_when_understat_data_exists(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    con.execute(
        "INSERT INTO fact_understat_player_season (player_uid, season, understat_player_id, "
        "source_player_name, games, minutes, goals, assists, xg, npxg, xa, xgchain, xgbuildup, "
        "shots, key_passes, _ingested_at) VALUES "
        "('p1', '2026-2027', '99', 'Player One', 5, 450, 3, 1, 3.5, 3.0, 1.2, 4.0, 2.0, 15, 8, current_timestamp)"
    )
    report = reporting.build_report(con, run_id)
    assert "p1" in report["understat_signal"]
    assert report["understat_signal"]["p1"]["understat_xg_per_90"] == pytest.approx(3.5 / 450 * 90)
    assert "p2" not in report["understat_signal"]

    text = reporting.render_report_text(report)
    assert "Understat xG second opinion" in text
    assert "Player One" in text


# ============================================================
# Priority 8c -- week-over-week diff report
# ============================================================

def test_snapshot_for_diff_extracts_the_small_subset(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    snap = reporting.snapshot_for_diff(report)
    assert snap["target_season"] == "2026-2027"
    assert snap["target_gameweek"] == 1
    assert {p["player_uid"] for p in snap["squad"]} == {"p1", "p2"}
    assert snap["captain_uid"] == "p1"
    assert snap["total_projected_ep"] == pytest.approx(report["headline"]["total_projected_ep"])
    assert snap["doubtful_flags"] == {}  # sanity_check_params_version not requested
    assert snap["weight_own_by_uid"] == {"p1": pytest.approx(0.0), "p2": pytest.approx(0.0)}


def test_snapshot_for_diff_includes_weight_own_when_nonzero(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    con.execute("UPDATE minutes_model_outputs SET weight_own = 0.6")
    report = reporting.build_report(con, run_id)
    snap = reporting.snapshot_for_diff(report)
    assert snap["weight_own_by_uid"] == {"p1": pytest.approx(0.6), "p2": pytest.approx(0.6)}


def test_save_and_load_report_snapshot_round_trips(con, tmp_path):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    saved_path = reporting.save_report_snapshot(report, tmp_path)
    assert saved_path.name == "2026-2027_gw1.json"

    loaded = reporting.load_report_snapshot("2026-2027", 1, tmp_path)
    assert loaded == reporting.snapshot_for_diff(report)


def test_load_report_snapshot_none_when_missing(tmp_path):
    assert reporting.load_report_snapshot("2026-2027", 1, tmp_path) is None


# ============================================================
# build_captain_recommendation
# ============================================================

def _tc_detail(*candidates, recommended_uid=None):
    """candidates: (player_uid, mean_total) pairs. recommended_uid defaults to the highest
    mean_total, matching evaluate_triple_captain()'s own tc_score-sorted best[0]."""
    all_candidates = [{"player_uid": uid, "mean_total": mt, "var_total": 1.0, "tc_score": mt} for uid, mt in candidates]
    best_uid = recommended_uid or max(candidates, key=lambda c: c[1])[0]
    return {"recommended": True, "captain_candidate": best_uid, "all_candidates": all_candidates}


def test_build_captain_recommendation_flags_a_better_option():
    detail = _tc_detail(("p_haaland", 9.2), ("p_salah", 6.5))
    names = {"p_haaland": "Erling Haaland", "p_salah": "Mohamed Salah"}
    rec = reporting.build_captain_recommendation(detail, "p_salah", names)
    assert rec["recommended_name"] == "Erling Haaland"
    assert rec["current_name"] == "Mohamed Salah"
    assert rec["matches_current"] is False
    assert rec["potential_gain"] == pytest.approx(2.7)


def test_build_captain_recommendation_matches_current_no_gain_claimed():
    detail = _tc_detail(("p_haaland", 9.2), ("p_salah", 6.5))
    names = {"p_haaland": "Erling Haaland", "p_salah": "Mohamed Salah"}
    rec = reporting.build_captain_recommendation(detail, "p_haaland", names)
    assert rec["matches_current"] is True
    assert rec["potential_gain"] == 0.0


def test_build_captain_recommendation_none_when_not_recommended():
    assert reporting.build_captain_recommendation({"recommended": False}, "p_salah", {}) is None
    assert reporting.build_captain_recommendation(None, "p_salah", {}) is None


def test_build_captain_recommendation_handles_unresolved_current_captain():
    detail = _tc_detail(("p_haaland", 9.2), ("p_salah", 6.5))
    rec = reporting.build_captain_recommendation(detail, None, {"p_haaland": "Erling Haaland"})
    assert rec["current_name"] is None
    assert rec["current_expected_points"] is None
    assert rec["potential_gain"] == 0.0  # nothing real to compare against -- not a fabricated gain


# ============================================================
# build_track_record_summary
# ============================================================

def _seed_backtest_run(con, *, warm_up_gameweeks=3, steps, metrics):
    """steps: list of (season, gameweek, tier). metrics: list of (season, gameweek, tier,
    metric_name, metric_value), matching backtest_metrics' real composite key."""
    backtest_run_id = con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks, notes) VALUES (?, 'test') RETURNING backtest_run_id",
        [warm_up_gameweeks],
    ).fetchone()[0]
    for season, gw, tier in steps:
        con.execute(
            "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, divergence_check_passed) "
            "VALUES (?, ?, ?, ?, '2026-08-10', TRUE)",
            [backtest_run_id, season, gw, tier],
        )
    for season, gw, tier, metric_name, metric_value in metrics:
        con.execute(
            "INSERT INTO backtest_metrics (backtest_run_id, season, gameweek, tier, metric_name, metric_value) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [backtest_run_id, season, gw, tier, metric_name, metric_value],
        )
    return backtest_run_id


def test_build_track_record_summary_none_when_no_backtest_run_yet():
    # con is never touched when backtest_run_id is None -- passing None for it here is the point.
    summary = reporting.build_track_record_summary(None, {"parameter_transparency": []}, None)
    assert summary == {
        "backtest_run_id": None, "n_gameweek_steps": None, "seasons_covered": [], "metrics": [],
        "parameters_total": 0, "parameters_backtested": 0, "parameters_still_invented": 0,
    }


def test_build_track_record_summary_real_steps_and_metrics(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    params.write_param(con, "squad_optimizer_guardrail_params", 1, "2026-08-10", "xi_club_concentration_cap", value_numeric=3)
    report = reporting.build_report(con, run_id, active_param_versions={"squad_optimizer_guardrail_params": 1})

    backtest_run_id = _seed_backtest_run(
        con,
        steps=[("2025-2026", 10, "warm"), ("2025-2026", 11, "warm"), ("2026-2027", 1, "cold")],
        metrics=[
            ("2025-2026", 10, "warm", "brier_appearance", 0.18),
            ("2025-2026", 11, "warm", "brier_appearance", 0.22),
            ("2026-2027", 1, "cold", "brier_appearance", 0.20),
        ],
    )

    summary = reporting.build_track_record_summary(con, report, backtest_run_id)
    assert summary["backtest_run_id"] == backtest_run_id
    assert summary["n_gameweek_steps"] == 3
    assert summary["seasons_covered"] == ["2025-2026", "2026-2027"]
    assert summary["metrics"] == [{"metric_name": "brier_appearance", "mean_value": pytest.approx(0.2), "n_observations": 3}]
    assert summary["parameters_total"] == 1
    assert summary["parameters_backtested"] == 0  # no recalibration_proposals row for this family yet
    assert summary["parameters_still_invented"] == 1


def test_build_track_record_summary_flags_backtested_params(con):
    run_id, *_ = _seed_full_squad_scenario(con)
    params.write_param(con, "squad_optimizer_guardrail_params", 1, "2026-08-10", "xi_club_concentration_cap", value_numeric=3)
    backtest_run_id = _seed_backtest_run(con, steps=[("2025-2026", 10, "warm")], metrics=[])
    con.execute(
        "INSERT INTO recalibration_proposals (backtest_run_id, param_family, param_key, new_params_version, "
        "metric_name, metric_before, metric_after) VALUES (?, 'squad_optimizer_guardrail_params', 'xi_club_concentration_cap', "
        "2, 'brier_appearance', 0.3, 0.2)",
        [backtest_run_id],
    )
    # transparency_panel() reads recalibration_proposals live, so build_report must run AFTER
    # the proposal above exists for backtested_via_m7 to see it.
    report = reporting.build_report(con, run_id, active_param_versions={"squad_optimizer_guardrail_params": 1})

    summary = reporting.build_track_record_summary(con, report, backtest_run_id)
    assert summary["parameters_backtested"] == 1
    assert summary["parameters_still_invented"] == 0


def test_diff_reports_no_previous_snapshot(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    diff = reporting.diff_reports(None, report)
    assert diff["has_previous"] is False
    assert diff["squad_changes"] is None


def test_diff_reports_detects_squad_and_captain_changes(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report_prev = reporting.build_report(con, run_id)
    previous_snapshot = reporting.snapshot_for_diff(report_prev)

    # p2 transferred out, p3 (new signing) transferred in, captaincy moves to p2
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p3', 'Player Three', 'Forward')")
    con.execute("UPDATE squad_optimizer_selections SET in_squad = FALSE, in_xi = FALSE WHERE run_id = ? AND player_uid = 'p2'", [run_id])
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p3', TRUE, TRUE, FALSE, FALSE)", [run_id],
    )
    con.execute("UPDATE squad_optimizer_selections SET is_captain = FALSE WHERE run_id = ? AND player_uid = 'p1'", [run_id])
    con.execute("UPDATE squad_optimizer_selections SET is_captain = TRUE WHERE run_id = ? AND player_uid = 'p2'", [run_id])
    # p2 needs its own category_breakdown/risk rows to still appear in the new report despite
    # no longer being "in_squad" -- simplest is to just leave it out of the new report's squad
    # read entirely, which is exactly what in_squad=FALSE already achieves.

    report_cur = reporting.build_report(con, run_id)
    diff = reporting.diff_reports(previous_snapshot, report_cur)

    assert diff["has_previous"] is True
    assert diff["squad_changes"]["out"] == ["Player Two"]
    assert diff["captain_changed"] is True
    assert diff["previous_captain"] == "Player One"
    assert diff["current_captain"] is None or diff["current_captain"] != "Player One"


def test_diff_reports_flags_newly_doubtful_starters(con):
    run_id, ep_mv, un_mv, _mc_mv = _seed_full_squad_scenario(con, captain_position="Defender")
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "nailed_p_start_threshold", value_numeric=0.75)
    params.write_param(con, "sanity_check_params", 1, "2026-08-10", "rotation_risk_p_start_threshold", value_numeric=0.55)
    mm_mv = con.execute(
        "SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [un_mv]
    ).fetchone()[0]

    # previous week: p2 (Forward) is clearly nailed (0.95) -> nailed_attacking_return passes
    con.execute("UPDATE minutes_model_outputs SET p_start_final = 0.95 WHERE model_version = ? AND player_uid = 'p2'", [mm_mv])
    report_prev = reporting.build_report(con, run_id, sanity_check_params_version=1)
    previous_snapshot = reporting.snapshot_for_diff(report_prev)
    assert previous_snapshot["doubtful_flags"]["nailed_attacking_return"] is True

    # this week: p2's p_start_final drops back below threshold -- newly doubtful
    con.execute("UPDATE minutes_model_outputs SET p_start_final = 0.7 WHERE model_version = ? AND player_uid = 'p2'", [mm_mv])
    report_cur = reporting.build_report(con, run_id, sanity_check_params_version=1)
    diff = reporting.diff_reports(previous_snapshot, report_cur)
    assert "nailed_attacking_return" in diff["newly_doubtful_flags"]


def test_render_diff_text_no_previous(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    diff = reporting.diff_reports(None, report)
    text = reporting.render_diff_text(diff)
    assert "no prior gameweek snapshot" in text


def test_render_diff_text_with_previous(con):
    run_id, *_ = _seed_full_squad_scenario(con, captain_position="Defender")
    report = reporting.build_report(con, run_id)
    previous_snapshot = reporting.snapshot_for_diff(report)
    diff = reporting.diff_reports(previous_snapshot, report)
    text = reporting.render_diff_text(diff)
    assert "unchanged" in text  # nothing changed between identical snapshots


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


# ============================================================
# player_uid -> canonical_name resolution for PWA-facing scripts (F2/F3/F9's shared writer-
# side name resolution -- decision_engine.py/squad_grade.py/elite_tracking.py all key their
# output by the DB's own internal player_uid, meaningless to the PWA on its own).
# ============================================================

def test_resolve_player_names_batches_and_skips_unknown_uids(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Bruno Fernandes', 'Midfielder')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Erling Haaland', 'Forward')")
    names = reporting.resolve_player_names(con, {"p1", "p2", "not_a_real_uid"})
    assert names == {"p1": "Bruno Fernandes", "p2": "Erling Haaland"}


def test_resolve_player_names_empty_input_returns_empty_without_a_query(con):
    assert reporting.resolve_player_names(con, set()) == {}


def test_uids_in_action_extracts_from_a_transfer_and_is_empty_otherwise():
    assert reporting.uids_in_action("transfer_in:p1->p2") == {"p1", "p2"}
    assert reporting.uids_in_action("roll") == set()
    assert reporting.uids_in_action(None) == set()


def test_humanize_action_rewrites_a_transfer_and_falls_back_for_unknown_uids():
    names = {"p1": "Bruno Fernandes", "p2": "Erling Haaland"}
    assert reporting.humanize_action("transfer_in:p1->p2", names) == "Bruno Fernandes -> Erling Haaland"
    assert reporting.humanize_action("transfer_in:p1->unknown_uid", names) == "Bruno Fernandes -> unknown_uid"


def test_humanize_action_passes_through_non_transfer_actions_and_none():
    names = {"p1": "Bruno Fernandes"}
    assert reporting.humanize_action("roll", names) == "roll"
    assert reporting.humanize_action("wildcard", names) == "wildcard"
    assert reporting.humanize_action(None, names) is None


def test_humanize_condition_rewrites_ruled_out_and_passes_through_other_text():
    names = {"p1": "Bruno Fernandes"}
    assert reporting.humanize_condition("p1 ruled out", names) == "Bruno Fernandes ruled out"
    assert reporting.humanize_condition("something else entirely", names) == "something else entirely"
    assert reporting.humanize_condition(None, names) is None


def _fake_decision_payload(action="transfer_in:p1->p2", runner_up=None, sensitivity=None):
    return {
        "action": action,
        "swaps": [{"out_player_uid": "p1", "in_player_uid": "p2", "delta_ep": 1.0, "reason": "test"}] if action.startswith("transfer_in:") else [],
        "sensitivity": sensitivity or [],
        "runner_up": runner_up,
    }


def test_uids_referenced_in_decision_payload_covers_swaps_action_and_sensitivity():
    payload = _fake_decision_payload(
        sensitivity=[{"if_condition": "p3 ruled out", "then_action": "transfer_in:p1->p4", "delta_ep": 0.5}],
        runner_up=_fake_decision_payload(action="transfer_in:p1->p5"),
    )
    uids = reporting.uids_referenced_in_decision_payload(payload)
    assert uids == {"p1", "p2", "p3", "p4", "p5"}


def test_uids_referenced_in_decision_payload_empty_for_a_roll():
    assert reporting.uids_referenced_in_decision_payload(_fake_decision_payload(action="roll")) == set()


def test_humanize_decision_payload_adds_display_fields_without_mutating_raw_ones():
    names = {"p1": "Bruno Fernandes", "p2": "Erling Haaland", "p3": "Third Player"}
    payload = _fake_decision_payload(
        sensitivity=[{"if_condition": "p3 ruled out", "then_action": "transfer_in:p2->p1", "delta_ep": 0.5}],
        runner_up=_fake_decision_payload(action="transfer_in:p2->p1"),
    )
    out = reporting.humanize_decision_payload(payload, names)

    # raw fields untouched
    assert out["action"] == "transfer_in:p1->p2"
    assert out["swaps"][0]["out_player_uid"] == "p1"

    # display fields added
    assert out["action_display"] == "Bruno Fernandes -> Erling Haaland"
    assert out["swaps"][0]["out_name"] == "Bruno Fernandes"
    assert out["swaps"][0]["in_name"] == "Erling Haaland"
    assert out["sensitivity"][0]["if_condition_display"] == "Third Player ruled out"
    assert out["sensitivity"][0]["then_action_display"] == "Erling Haaland -> Bruno Fernandes"
    assert out["runner_up"]["action_display"] == "Erling Haaland -> Bruno Fernandes"
    # _fake_decision_payload's swap is always hardcoded p1->p2 regardless of the action string
    # passed (only the action itself varies here) -- out_player_uid is still "p1" ("Bruno
    # Fernandes"), independently of runner_up's own action string being reversed.
    assert out["runner_up"]["swaps"][0]["out_name"] == "Bruno Fernandes"

    # original dict passed in is never mutated
    assert "action_display" not in payload


def test_humanize_decision_payload_handles_a_roll_with_no_swaps():
    payload = _fake_decision_payload(action="roll")
    out = reporting.humanize_decision_payload(payload, {})
    assert out["action_display"] == "roll"
    assert out["swaps"] == []


def test_humanize_swap_list_adds_names_without_mutating_input():
    swaps = [{"out_player_uid": "p1", "in_player_uid": "p2", "delta_ep": 1.0, "reason": "test"}]
    names = {"p1": "Bruno Fernandes", "p2": "Erling Haaland"}
    out = reporting.humanize_swap_list(swaps, names)
    assert out[0]["out_name"] == "Bruno Fernandes"
    assert out[0]["in_name"] == "Erling Haaland"
    assert "out_name" not in swaps[0]


# Priority 8d: the public Track Record page's transparency log -- list_report_snapshots(),
# load_latest_provenance(), and build_transparency_log(). All pure file reads over already-
# committed JSON, so these tests stage files in a tmp_path dir rather than touching the DB.

def _write_snapshot(history_dir, season, gw, captain="Marcos Senesi Bar\u00f3n", ep=45.5, in_xi=11, bench=4):
    import json as _json
    history_dir.mkdir(parents=True, exist_ok=True)
    squad = [{"player_uid": f"p{i}", "name": f"P{i}", "in_xi": i < in_xi, "is_captain": False} for i in range(in_xi + bench)]
    squad[0]["is_captain"] = True
    payload = {
        "target_season": season, "target_gameweek": gw, "squad": squad,
        "captain_uid": "p0", "captain_name": captain,
        "total_projected_ep": ep, "doubtful_flags": {}, "weight_own_by_uid": {},
    }
    (history_dir / f"{season}_gw{gw}.json").write_text(_json.dumps(payload))


def test_list_report_snapshots_returns_empty_when_dir_missing(tmp_path):
    assert reporting.list_report_snapshots(tmp_path / "does_not_exist") == []


def test_list_report_snapshots_sorts_newest_gameweek_first_and_skips_non_snapshots(tmp_path):
    _write_snapshot(tmp_path, "2026-2027", 1, ep=45.5)
    _write_snapshot(tmp_path, "2026-2027", 14, ep=45.0)
    _write_snapshot(tmp_path, "2026-2027", 2, ep=46.0)
    # An unrelated JSON file that matches the *_gw*.json glob but not the season_gwN.json convention
    (tmp_path / "not_a_snapshot_gwX.json").write_text("{}")
    rows = reporting.list_report_snapshots(tmp_path)
    gameweeks = [r["gameweek"] for r in rows]
    assert gameweeks == [14, 2, 1]  # numeric GW order, not string order
    assert all(r["season"] == "2026-2027" for r in rows)
    assert rows[0]["captain_name"] == "Marcos Senesi Bar\u00f3n"
    assert rows[0]["squad_size"] == 15
    assert rows[0]["in_xi_count"] == 11


def test_list_report_snapshots_skips_corrupt_json(tmp_path):
    _write_snapshot(tmp_path, "2026-2027", 1)
    (tmp_path / "2026-2027_gw2.json").write_text("{not valid json")
    rows = reporting.list_report_snapshots(tmp_path)
    assert [r["gameweek"] for r in rows] == [1]


def test_load_latest_provenance_none_when_missing(tmp_path):
    assert reporting.load_latest_provenance(tmp_path) is None


def test_load_latest_provenance_picks_newest_dated_file(tmp_path):
    import json as _json
    (tmp_path / "provenance_2026-08-23.json").write_text(_json.dumps({"data_asof": "2026-08-23"}))
    (tmp_path / "provenance_2026-08-24.json").write_text(_json.dumps({"data_asof": "2026-08-24"}))
    prov = reporting.load_latest_provenance(tmp_path)
    assert prov["data_asof"] == "2026-08-24"


def test_build_transparency_log_assembles_all_sections_honestly(tmp_path):
    _write_snapshot(tmp_path, "2026-2027", 1)
    _write_snapshot(tmp_path, "2026-2027", 14)
    import json as _json
    (tmp_path / "provenance_2026-08-24.json").write_text(_json.dumps({"data_asof": "2026-08-24"}))
    track_record = {
        "n_gameweek_steps": None, "seasons_covered": [], "metrics": [],
        "parameters_total": 71, "parameters_backtested": 0,
    }
    diff = {"has_previous": False, "current_gameweek": 14, "squad_changes": None}
    log = reporting.build_transparency_log(track_record, tmp_path, diff)
    # Backtest status is passed through verbatim, not re-derived or invented.
    assert log["backtest"]["n_gameweek_steps"] is None
    assert log["backtest"]["parameters_backtested"] == 0
    # Snapshots + provenance + diff are all the real committed artifacts.
    assert [r["gameweek"] for r in log["snapshots"]] == [14, 1]
    assert log["provenance"]["data_asof"] == "2026-08-24"
    assert log["latest_diff"] is diff


def test_build_transparency_log_stays_honest_when_nothing_exists_yet(tmp_path):
    log = reporting.build_transparency_log({}, tmp_path, None)
    assert log["backtest"]["n_gameweek_steps"] is None
    assert log["snapshots"] == []
    assert log["latest_diff"] is None
    assert log["provenance"] is None
