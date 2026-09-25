"""Phase 1A (price-band calibration) / Phase 1B (selection-residual "optimiser curse")
diagnostics. See src/fpl_quant/calibration_diagnostics.py's module docstring for why this
recomputes from raw rows rather than trusting backtest_metrics' pre-aggregated segment means.
"""

from datetime import datetime

import pytest

from fpl_quant import backtest as bt
from fpl_quant import calibration_diagnostics as cd


# ============================================================
# Pure statistics helpers
# ============================================================

def test_mean_mae_rmse_median_hand_computed():
    xs = [1.0, -2.0, 3.0, -3.0]
    assert cd._mean(xs) == pytest.approx(-0.25)
    assert cd._mae(xs) == pytest.approx(2.25)
    assert cd._rmse(xs) == pytest.approx((sum(x * x for x in xs) / 4) ** 0.5)
    assert cd._median(xs) == pytest.approx((-2.0 + 1.0) / 2)  # sorted: -3,-2,1,3 -> mid two


def test_summary_stats_are_none_not_fabricated_for_empty_slice():
    summary = cd.summarize_residuals([])
    assert summary["n"] == 0
    assert summary["mean_resid"] is None
    assert summary["mae"] is None
    assert summary["mean_resid_ci95_gw_clustered"] is None


def test_cluster_bootstrap_ci_returns_none_below_two_clusters():
    rows = [{"season": "2025-2026", "gameweek": 10, "residual": 1.0}]
    assert cd.cluster_bootstrap_ci(rows, lambda r: r["residual"], lambda r: (r["season"], r["gameweek"])) is None


def test_cluster_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    rows = [
        {"season": "2025-2026", "gameweek": gw, "residual": resid}
        for gw, resid in ((10, 1.0), (10, 1.5), (11, -2.0), (12, 0.5), (12, 0.7))
    ]
    ci_a = cd.cluster_bootstrap_ci(rows, lambda r: r["residual"], lambda r: (r["season"], r["gameweek"]), n_boot=500, seed=42)
    ci_b = cd.cluster_bootstrap_ci(rows, lambda r: r["residual"], lambda r: (r["season"], r["gameweek"]), n_boot=500, seed=42)
    assert ci_a == ci_b
    lo, hi = ci_a
    assert lo <= hi


def test_cluster_bootstrap_ci_brackets_the_true_mean_on_low_variance_clusters():
    # Every cluster's values sit tightly around 2.0 -- the resample distribution of the mean
    # should too, so a 95% interval built from it should contain 2.0.
    rows = [
        {"season": "s", "gameweek": gw, "residual": resid}
        for gw in range(1, 11) for resid in (1.9, 2.0, 2.1)
    ]
    lo, hi = cd.cluster_bootstrap_ci(rows, lambda r: r["residual"], lambda r: (r["season"], r["gameweek"]), n_boot=1000, seed=7)
    assert lo <= 2.0 <= hi


# ============================================================
# Price-band boundary (delegation + NaN-safety)
# ============================================================

def test_price_band_boundaries_are_half_open_on_the_lower_edge():
    assert bt._price_band(4.999) == "<5.0"
    assert bt._price_band(5.0) == "5.0-7.0"
    assert bt._price_band(6.999) == "5.0-7.0"
    assert bt._price_band(7.0) == "7.0-9.0"
    assert bt._price_band(8.999) == "7.0-9.0"
    assert bt._price_band(9.0) == "9.0+"


def test_price_band_unknown_for_none_and_nan():
    assert bt._price_band(None) == "unknown"
    assert bt._price_band(float("nan")) == "unknown"


# ============================================================
# compute_curse: pure sign-convention core, deterministic tiny fixture (mission-required test)
# ============================================================

def test_curse_positive_when_selection_underperforms_matched_peers():
    # Selected players: residuals -2.0 (band A) and -1.0 (band B). Band pool means: A=0.0, B=1.0
    # (i.e. an average player in these bands did better than the ones actually picked).
    result = cd.compute_curse([-2.0, -1.0], ["A", "B"], {"A": 0.0, "B": 1.0})
    assert result["sel_resid"] == pytest.approx(-1.5)
    assert result["band_resid"] == pytest.approx(0.5)
    assert result["curse"] == pytest.approx(2.0)
    assert result["curse"] > 0


def test_curse_negative_when_selection_outperforms_matched_peers():
    result = cd.compute_curse([3.0, 4.0], ["A", "B"], {"A": 1.0, "B": 2.0})
    assert result["curse"] == pytest.approx(-2.0)
    assert result["curse"] < 0


def test_curse_zero_when_selection_exactly_matches_band_means():
    result = cd.compute_curse([0.5, 1.5], ["A", "B"], {"A": 0.5, "B": 1.5})
    assert result["curse"] == pytest.approx(0.0)


def test_curse_sum_scale_is_curse_times_n():
    result = cd.compute_curse([-2.0, -1.0, -3.0], ["A", "A", "B"], {"A": 0.0, "B": 1.0})
    assert result["curse_sum_scale"] == pytest.approx(result["curse"] * 3)


def test_curse_raises_on_a_band_missing_from_the_pool_rather_than_defaulting_silently():
    with pytest.raises(KeyError):
        cd.compute_curse([-2.0], ["unseen_band"], {"A": 0.0})


def test_curse_handles_empty_selection_without_fabricating_a_value():
    result = cd.compute_curse([], [], {})
    assert result["n"] == 0
    assert result["curse"] is None


# ============================================================
# DB-backed fixture: two scored gameweek-steps, four players each, real squad_optimizer
# selections -- exercises fetch_calibration_rows / price_band_calibration_report /
# selection_curse_report end to end against real schema/db.py-applied tables.
# ============================================================

def _seed_two_gameweek_scenario(con):
    """p1/p2 share price band <5.0 and position Defender; p3/p4 share 9.0+ and Forward.
    XI each week = {p1, p3} (one per band); p2/p4 are squad-but-bench. ep_total is held at 2.0
    for every player-gameweek so residual = event_points - 2.0, chosen by hand below so the
    price-band-matched XI curse is +2.0 in GW10 and -2.0 in GW11 (worked in the PR description /
    test docstring below) -- a deliberately mixed-sign two-gameweek scenario, not a single lucky
    draw, per the mission's own "don't claim curse~0 from one sample" instruction."""
    for uid, name in (("team_a", "A"), ("team_b", "B")):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
    for uid, name, position in (
        ("p1", "P1", "Defender"), ("p2", "P2", "Defender"),
        ("p3", "P3", "Forward"), ("p4", "P4", "Forward"),
    ):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, name, position])

    now = datetime.now()
    bt.ep.seed_v1_params(con)
    backtest_run_id = con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id"
    ).fetchone()[0]

    ts_mv = con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2025-11-01', 0.2, 1, 1, 'team_a') RETURNING model_version"
    ).fetchone()[0]
    mm_mv = con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2025-11-01', '2025-2026', 1, 1, 1, 1, '[]') RETURNING model_version"
    ).fetchone()[0]

    # (residual per player) -2,+2 (band<5.0 mean 0.0) / -1,+3 (band9+ mean 1.0) -> XI{p1,p3}
    # sel_resid=-1.5, band_resid=0.5, curse=+2.0
    # +3,-1 (band<5.0 mean 1.0) / +4,0 (band9+ mean 2.0) -> XI{p1,p3}
    # sel_resid=3.5, band_resid=1.5, curse=-2.0
    per_gw = {
        10: {"p1": -2.0, "p2": 2.0, "p3": -1.0, "p4": 3.0},
        11: {"p1": 3.0, "p2": -1.0, "p3": 4.0, "p4": 0.0},
    }
    price_of = {"p1": 4.5, "p2": 4.0, "p3": 12.0, "p4": 11.0}

    for gw, resid_of in per_gw.items():
        match_id = f"m{gw}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
            "home_score, away_score, finished, competition, _ingested_at) VALUES "
            "(?, '2025-2026', ?, ?, 'team_a', 'team_b', 1, 1, TRUE, 'Premier League', ?)",
            [match_id, gw, datetime(2025, 11, min(gw, 28), 15, 0), now],
        )
        for uid in resid_of:
            con.execute(
                "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
                "minutes_played, goals, assists, team_goals_conceded, _ingested_at) "
                "VALUES (?, ?, '2025-2026', 0, 90, 90, 0, 0, 1, ?)",
                [uid, match_id, now],
            )

        ep_mv = con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2025-11-01', '2025-2026', ?, ?, 1, 1, 1) RETURNING model_version",
            [ts_mv, mm_mv],
        ).fetchone()[0]
        for uid, resid in resid_of.items():
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
                "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
                "ep_total, expected_bps) VALUES (?, ?, ?, 1.0, 0.5, 0.2, 0, 0, 0, 0.3, 0, 0, 0, 0, 2.0, 20.0)",
                [ep_mv, uid, match_id],
            )
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, now_cost, _ingested_at) "
                "VALUES (?, '2025-2026', ?, ?, ?, current_timestamp)",
                [uid, gw, 2.0 + resid, price_of[uid]],
            )

        un_mv = con.execute(
            "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
            "team_strength_model_version, rho_residual_params_version) VALUES ('2025-11-01', ?, ?, ?, 1) "
            "RETURNING model_version",
            [ep_mv, mm_mv, ts_mv],
        ).fetchone()[0]
        so_run_id = con.execute(
            "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, "
            "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
            "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
            "VALUES ('2025-11-01', '2025-2026', ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 10.0) RETURNING run_id",
            [gw, ep_mv, un_mv],
        ).fetchone()[0]
        for uid, in_xi in (("p1", True), ("p2", False), ("p3", True), ("p4", False)):
            con.execute(
                "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
                "VALUES (?, ?, TRUE, ?, FALSE, FALSE)", [so_run_id, uid, in_xi],
            )

        con.execute(
            "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, "
            "ep_model_version, so_run_id, divergence_check_passed) "
            "VALUES (?, '2025-2026', ?, 'mature', ?, ?, ?, TRUE)",
            [backtest_run_id, gw, now, ep_mv, so_run_id],
        )
    return backtest_run_id


def test_fetch_calibration_rows_computes_correct_price_bands_and_residuals(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    rows, missing = cd.fetch_calibration_rows(con, backtest_run_id, tier="mature")
    assert len(rows) == 8  # 4 players x 2 gameweeks
    assert missing == {"missing_realized_or_predicted": 0, "missing_price": 0, "missing_position": 0}
    by_key = {(r["player_uid"], r["gameweek"]): r for r in rows}
    assert by_key[("p1", 10)]["price_band"] == "<5.0"
    assert by_key[("p1", 10)]["residual"] == pytest.approx(-2.0)
    assert by_key[("p3", 11)]["price_band"] == "9.0+"
    assert by_key[("p3", 11)]["residual"] == pytest.approx(4.0)


def test_fetch_calibration_rows_component_residuals_sum_to_total(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    rows, _ = cd.fetch_calibration_rows(con, backtest_run_id, tier="mature")
    for r in rows:
        comp_sum = sum(realized - predicted for realized, predicted in r["components"].values())
        assert comp_sum == pytest.approx(r["residual"], abs=1e-9)


def test_price_band_calibration_report_matches_hand_computed_band_means(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    report = cd.price_band_calibration_report(con, backtest_run_id, tier="mature", n_boot=200, seed=0)
    assert report["overall"]["n"] == 8
    # <5.0 band residuals across both GWs: p1,p2 in GW10 (-2,+2), GW11 (+3,-1) -> mean = 0.5
    assert report["by_price_band"]["<5.0"]["mean_resid"] == pytest.approx((-2 + 2 + 3 - 1) / 4)
    assert report["by_price_band"]["9.0+"]["mean_resid"] == pytest.approx((-1 + 3 + 4 + 0) / 4)
    assert report["by_position"]["Defender"]["n"] == 4
    assert report["by_position"]["Forward"]["n"] == 4
    assert "Defender|<5.0" in report["by_position_price_band"]
    assert report["missing_data_counts"]["missing_realized_or_predicted"] == 0
    assert report["git_sha"] is None or isinstance(report["git_sha"], str)
    assert report["seed"] == 0 and report["backtest_run_id"] == backtest_run_id


def test_selection_curse_for_step_matches_hand_computation_gw10(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    rows, _ = cd.fetch_calibration_rows(con, backtest_run_id, tier="mature")
    gw10_rows = [r for r in rows if r["gameweek"] == 10]
    so_run_id = con.execute(
        "SELECT so_run_id FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND gameweek = 10", [backtest_run_id]
    ).fetchone()[0]
    result = cd.selection_curse_for_step(con, so_run_id, gw10_rows, matched="price_band", selection="xi")
    assert result["sel_resid"] == pytest.approx(-1.5)
    assert result["band_resid"] == pytest.approx(0.5)
    assert result["curse"] == pytest.approx(2.0)
    assert result["missing_selected_players"] == 0


def test_selection_curse_for_step_matches_hand_computation_gw11_opposite_sign(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    rows, _ = cd.fetch_calibration_rows(con, backtest_run_id, tier="mature")
    gw11_rows = [r for r in rows if r["gameweek"] == 11]
    so_run_id = con.execute(
        "SELECT so_run_id FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND gameweek = 11", [backtest_run_id]
    ).fetchone()[0]
    result = cd.selection_curse_for_step(con, so_run_id, gw11_rows, matched="price_band", selection="xi")
    assert result["curse"] == pytest.approx(-2.0)


def test_selection_curse_report_covers_all_four_variants_and_both_gameweeks(con):
    backtest_run_id = _seed_two_gameweek_scenario(con)
    report = cd.selection_curse_report(con, backtest_run_id, tier="mature", n_boot=200, seed=0)
    assert report["n_steps_with_selection"] == 2
    assert set(report["variants"]) == {
        "price_band__xi", "price_band__squad", "position_price_band__xi", "position_price_band__squad",
    }
    xi_variant = report["variants"]["price_band__xi"]
    assert xi_variant["n_gameweeks"] == 2
    assert xi_variant["mean_curse"] == pytest.approx(0.0)  # +2.0 and -2.0 average out
    assert xi_variant["frac_positive"] == pytest.approx(0.5)
    # squad-scale (all 4 players, unmatched by XI-only) is a real, differently-scoped number --
    # just check it runs and returns 2 gameweeks, not a specific value.
    assert report["variants"]["price_band__squad"]["n_gameweeks"] == 2


def test_selection_curse_report_returns_no_selection_steps_when_none_exist(con):
    backtest_run_id = con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id"
    ).fetchone()[0]
    report = cd.selection_curse_report(con, backtest_run_id, tier="mature")
    assert report["n_steps_with_selection"] == 0
    for variant in report["variants"].values():
        assert variant["n_gameweeks"] == 0
        assert variant["mean_curse"] is None
