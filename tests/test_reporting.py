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
