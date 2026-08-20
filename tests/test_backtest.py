import math
from datetime import date, datetime

import pandas as pd
import pytest

from fpl_quant import backtest as bt
from fpl_quant import minutes_model


# ============================================================
# scoring helpers (pure functions)
# ============================================================

def test_log_score_bernoulli_matches_log_of_assigned_probability():
    assert bt.log_score_bernoulli(0.8, True) == pytest.approx(math.log(0.8))
    assert bt.log_score_bernoulli(0.8, False) == pytest.approx(math.log(0.2))


def test_log_score_bernoulli_rewards_confident_correct_over_unsure():
    confident_right = bt.log_score_bernoulli(0.95, True)
    unsure_right = bt.log_score_bernoulli(0.55, True)
    assert confident_right > unsure_right


def test_log_score_bernoulli_punishes_confident_wrong_hardest():
    confident_wrong = bt.log_score_bernoulli(0.95, False)
    unsure_wrong = bt.log_score_bernoulli(0.55, False)
    assert confident_wrong < unsure_wrong


def test_brier_bernoulli_zero_for_perfect_prediction():
    assert bt.brier_bernoulli(1.0, True) == pytest.approx(0.0)
    assert bt.brier_bernoulli(0.0, False) == pytest.approx(0.0)


def test_brier_bernoulli_one_for_maximally_wrong_prediction():
    assert bt.brier_bernoulli(1.0, False) == pytest.approx(1.0)
    assert bt.brier_bernoulli(0.0, True) == pytest.approx(1.0)


def test_log_score_categorical_matches_observed_class_probability():
    probs = {"0": 0.2, "1_59": 0.3, "60plus": 0.5}
    assert bt.log_score_categorical(probs, "60plus") == pytest.approx(math.log(0.5))


def test_brier_categorical_zero_for_perfect_one_hot_prediction():
    probs = {"0": 0.0, "1_59": 0.0, "60plus": 1.0}
    assert bt.brier_categorical(probs, "60plus") == pytest.approx(0.0)


def test_brier_categorical_matches_hand_computed_value():
    probs = {"0": 0.2, "1_59": 0.3, "60plus": 0.5}
    # observed "0": (0.2-1)^2 + (0.3-0)^2 + (0.5-0)^2
    expected = (0.2 - 1) ** 2 + 0.3 ** 2 + 0.5 ** 2
    assert bt.brier_categorical(probs, "0") == pytest.approx(expected)


def test_log_score_poisson_matches_scipy_poisson_logpmf():
    from scipy.stats import poisson as sp_poisson
    assert bt.log_score_poisson(1.3, 2) == pytest.approx(sp_poisson.logpmf(2, 1.3))


def test_log_score_poisson_prefers_correct_lambda_to_wrong_one():
    # observing 3 goals should score higher under a lambda=3-ish prediction than lambda=0.1
    assert bt.log_score_poisson(3.0, 3) > bt.log_score_poisson(0.1, 3)


def test_minutes_state_buckets():
    assert bt._minutes_state(0) == "0"
    assert bt._minutes_state(45) == "1_59"
    assert bt._minutes_state(59) == "1_59"
    assert bt._minutes_state(60) == "60plus"
    assert bt._minutes_state(90) == "60plus"


# ============================================================
# tier_for
# ============================================================

def test_tier_for_2024_25_boundaries():
    assert bt.tier_for("2024-2025", 1) == "cold"
    assert bt.tier_for("2024-2025", 9) == "cold"
    assert bt.tier_for("2024-2025", 10) == "warm"
    assert bt.tier_for("2024-2025", 38) == "warm"


def test_tier_for_2025_26_boundaries():
    assert bt.tier_for("2025-2026", 1) == "warm"
    assert bt.tier_for("2025-2026", 15) == "warm"
    assert bt.tier_for("2025-2026", 16) == "mature"
    assert bt.tier_for("2025-2026", 38) == "mature"


def test_tier_for_rejects_unknown_season():
    with pytest.raises(ValueError):
        bt.tier_for("2026-2027", 1)


# ============================================================
# fixtures: a two-team, two-gameweek league across two seasons
# ============================================================

def _seed_two_gameweek_league(con):
    """team_a vs team_b, GW10 (2025-11-01) and GW20 (2026-01-03) of 2025-2026, plus one
    2024-2025 match for cross-season pass-through checks. Real dates lifted from the actual
    ingested DB (not invented) so this mirrors the real leak scenario."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Midfielder')")

    def _match(match_id, season, gw, kickoff, finished=True):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, "
            "away_team_uid, home_score, away_score, finished, competition, _ingested_at) "
            "VALUES (?, ?, ?, ?, 'team_a', 'team_b', 2, 1, ?, 'Premier League', ?)",
            [match_id, season, gw, kickoff, finished, datetime.now()],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, "
            "finish_min, minutes_played, goals, _ingested_at) VALUES ('p1', ?, ?, 0, 90, 90, 1, ?)",
            [match_id, season, datetime.now()],
        )

    _match("prior24", "2024-2025", 30, datetime(2025, 3, 1, 15, 0))
    _match("gw10m", "2025-2026", 10, datetime(2025, 11, 1, 15, 0))
    _match("gw20m", "2025-2026", 20, datetime(2026, 1, 3, 17, 30))

    for season, gw in (("2024-2025", 30), ("2025-2026", 5), ("2025-2026", 10), ("2025-2026", 20)):
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, _ingested_at) "
            "VALUES ('p1', ?, ?, 90, ?)",
            [season, gw, datetime.now()],
        )


# ============================================================
# gameweek_deadline
# ============================================================

def test_gameweek_deadline_returns_earliest_kickoff(con):
    _seed_two_gameweek_league(con)
    assert bt.gameweek_deadline(con, "2025-2026", 10) == datetime(2025, 11, 1, 15, 0)


def test_gameweek_deadline_none_when_no_fixtures(con):
    _seed_two_gameweek_league(con)
    assert bt.gameweek_deadline(con, "2025-2026", 99) is None


# ============================================================
# asof_scope -- the load-bearing leak-prevention test
# ============================================================

def test_asof_scope_hides_future_match_and_reveals_it_after_exit(con):
    _seed_two_gameweek_league(con)

    with bt.asof_scope(con, "2025-2026", 10):
        visible_ids = {r[0] for r in con.execute("SELECT match_id FROM fact_match").fetchall()}
        assert "gw20m" not in visible_ids  # a different, future gameweek: fully invisible, schedule included
        assert "prior24" in visible_ids

    visible_ids_after = {r[0] for r in con.execute("SELECT match_id FROM fact_match").fetchall()}
    assert visible_ids_after == {"prior24", "gw10m", "gw20m"}


def test_asof_scope_exposes_target_gameweeks_schedule_with_result_hidden(con):
    """The gameweek being predicted needs its own fixture list visible (that's what
    expected_points.run()/monte_carlo.run() predict against) -- only the *result* is
    unknowable in advance, not the schedule. A strict kickoff_time cutoff would hide GW10's own
    match entirely (its kickoff_time is exactly the deadline), which is wrong: real fixture
    lists are announced well before deadline."""
    _seed_two_gameweek_league(con)
    with bt.asof_scope(con, "2025-2026", 10):
        row = con.execute(
            "SELECT home_team_uid, away_team_uid, home_score, away_score, finished FROM fact_match WHERE match_id = 'gw10m'"
        ).fetchone()
        assert row is not None, "GW10's own schedule row must be visible"
        assert row[0] == "team_a" and row[1] == "team_b"
        assert row[2] is None and row[3] is None, "GW10's own result must be hidden"
        assert row[4] is False


def test_asof_scope_hides_future_player_match_stats(con):
    _seed_two_gameweek_league(con)
    with bt.asof_scope(con, "2025-2026", 10):
        visible = {r[0] for r in con.execute("SELECT match_id FROM fact_player_match_stats").fetchall()}
        assert "gw20m" not in visible
    visible_after = {r[0] for r in con.execute("SELECT match_id FROM fact_player_match_stats").fetchall()}
    assert "gw20m" in visible_after


def test_asof_scope_truncates_in_progress_season_stats_by_gw_not_date(con):
    """fact_player_season_stats has no date column at all -- its own gw column is the asof
    signal (see module docstring). GW20's row must be hidden even though it has no kickoff_time
    of its own to filter on."""
    _seed_two_gameweek_league(con)
    with bt.asof_scope(con, "2025-2026", 10):
        rows = con.execute(
            "SELECT gw FROM fact_player_season_stats WHERE season = '2025-2026' ORDER BY gw"
        ).fetchall()
        assert [r[0] for r in rows] == [5]  # gw10's and gw20's own rows both excluded (gw < 10 only)


def test_asof_scope_passes_prior_completed_season_through_whole(con):
    _seed_two_gameweek_league(con)
    with bt.asof_scope(con, "2025-2026", 10):
        rows = con.execute("SELECT gw FROM fact_player_season_stats WHERE season = '2024-2025'").fetchall()
        assert [r[0] for r in rows] == [30]  # unaffected by the 2025-2026 gw cutoff


def test_asof_scope_drops_temp_tables_on_exception():
    """finally-block cleanup must run even if the caller's code inside the scope raises --
    otherwise a crashed walk-forward step would leave a stale shadow poisoning the next one."""
    import duckdb as ddb
    from fpl_quant import db as db_mod

    con = ddb.connect(":memory:")
    db_mod.apply_schema(con)
    _seed_two_gameweek_league(con)

    with pytest.raises(RuntimeError):
        with bt.asof_scope(con, "2025-2026", 10):
            raise RuntimeError("simulated failure mid-step")

    visible_ids = {r[0] for r in con.execute("SELECT match_id FROM fact_match").fetchall()}
    assert visible_ids == {"prior24", "gw10m", "gw20m"}
    con.close()


# ============================================================
# has_fittable_history -- the cold-start guard
# ============================================================

def test_has_fittable_history_false_when_gameweek_has_no_fixtures(con):
    _seed_two_gameweek_league(con)
    assert bt.has_fittable_history(con, "2025-2026", 99) is False


def test_has_fittable_history_false_when_no_prior_matches(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, "
        "away_team_uid, home_score, away_score, finished, competition, _ingested_at) "
        "VALUES ('firstever', '2024-2025', 1, '2024-08-10 15:00:00', 'team_a', 'team_b', 1, 0, TRUE, 'Premier League', ?)",
        [datetime.now()],
    )
    assert bt.has_fittable_history(con, "2024-2025", 1) is False


def test_has_fittable_history_true_once_prior_matches_exist(con):
    _seed_two_gameweek_league(con)
    assert bt.has_fittable_history(con, "2025-2026", 20) is True  # gw10m is finished and predates gw20's deadline


# ============================================================
# propose_recalibration -- writes a version, never activates it
# ============================================================

def _seed_backtest_run(con, notes="test run"):
    return con.execute(
        "INSERT INTO backtest_runs (warm_up_gameweeks, notes) VALUES (0, ?) RETURNING backtest_run_id", [notes]
    ).fetchone()[0]


def test_propose_recalibration_writes_a_new_param_version(con):
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    backtest_run_id = _seed_backtest_run(con)

    proposal_id = bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=1,
    )
    assert proposal_id is not None

    row = con.execute(
        "SELECT status, old_value, new_value, old_params_version, new_params_version FROM recalibration_proposals WHERE proposal_id = ?",
        [proposal_id],
    ).fetchone()
    assert row[0] == "pending"
    assert row[1] == pytest.approx(0.0018)
    assert row[2] == pytest.approx(0.003)
    assert row[4] == row[3] + 1


def test_propose_recalibration_does_not_change_what_old_version_resolves_to(con):
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    backtest_run_id = _seed_backtest_run(con)
    bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=1,
    )
    # the live path (whatever resolve_param call a caller was already making at v1) sees no change
    v1, _ = params.resolve_param(con, "model_decay_params", "xi", 1)
    assert v1 == pytest.approx(0.0018)


def test_propose_recalibration_handles_missing_old_version_gracefully(con):
    backtest_run_id = _seed_backtest_run(con)
    proposal_id = bt.propose_recalibration(
        con, backtest_run_id, "risk_aversion_params", "lambda_value", 0.20,
        metric_name="realized_sharpe", metric_before=0.0, metric_after=1.2, old_params_version=None,
    )
    row = con.execute("SELECT old_value, old_params_version FROM recalibration_proposals WHERE proposal_id = ?", [proposal_id]).fetchone()
    assert row == (None, None)


# ============================================================
# refit_xi_rho -- profile-likelihood grid search
# ============================================================

def _seed_round_robin_matches(con, uids, results):
    """results: list of (home, away, home_goals, away_goals). Mirrors the pattern
    test_team_strength.py uses for fit_dixon_coles(), inlined here (no cross-test-file
    import precedent exists in this project's test suite)."""
    for i, (h, a, hg, ag) in enumerate(results):
        con.execute(
            "INSERT INTO fact_match (match_id, season, home_team_uid, away_team_uid, home_score, "
            "away_score, kickoff_time, finished, competition, _ingested_at) "
            "VALUES (?, '2024-2025', ?, ?, ?, ?, ?, TRUE, 'Premier League', current_timestamp)",
            [f"m{i}", uids[h], uids[a], hg, ag, datetime(2025, 1, 1) + pd.Timedelta(days=i)],
        )


def test_refit_xi_rho_picks_a_grid_point_with_finite_likelihood(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_c', 'C')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_d', 'D')")
    uids = {"A": "team_a", "B": "team_b", "C": "team_c", "D": "team_d"}
    results = [
        ("A", "B", 3, 0), ("B", "A", 0, 2), ("A", "C", 4, 1), ("C", "A", 0, 3),
        ("A", "D", 5, 0), ("D", "A", 0, 4), ("B", "C", 1, 1), ("C", "B", 1, 1),
        ("B", "D", 2, 0), ("D", "B", 0, 2), ("C", "D", 2, 0), ("D", "C", 0, 2),
    ]
    _seed_round_robin_matches(con, uids, results)

    result = bt.refit_xi_rho(con, fit_seasons=("2024-2025",), xi_grid=(0.001, 0.003), rho_grid=(-0.10, -0.15))
    assert result["xi"] in (0.001, 0.003)
    assert result["rho"] in (-0.10, -0.15)
    assert math.isfinite(result["neg_log_likelihood"])


def test_refit_xi_rho_raises_on_no_matches(con):
    with pytest.raises(ValueError):
        bt.refit_xi_rho(con, fit_seasons=("2024-2025",))


# ============================================================
# refit_rho_residual -- moment-matching against realized covariance
# ============================================================

def _seed_realized_goals_assists(con, backtest_run_id, tier, values_by_gw_and_player):
    for (season, gw), player_values in values_by_gw_and_player.items():
        for player_uid, value in player_values.items():
            con.execute(
                "INSERT INTO backtest_metrics (backtest_run_id, season, gameweek, tier, metric_name, metric_value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [backtest_run_id, season, gw, tier, f"realized_goals_assists:{player_uid}", value],
            )


def test_refit_rho_residual_recovers_a_sane_rho_from_correlated_synthetic_data(con):
    backtest_run_id = _seed_backtest_run(con)
    rng_pairs = {}
    import random

    random.seed(0)
    values = {}
    for gw in range(1, 21):
        shared_bump = random.choice([0, 1, 2])  # shared "tempo" factor -- induces real positive covariance
        values[("2025-2026", gw)] = {
            "p_a": shared_bump + random.choice([0, 0, 1]),
            "p_b": shared_bump + random.choice([0, 0, 1]),
        }
    _seed_realized_goals_assists(con, backtest_run_id, "warm", values)

    result = bt.refit_rho_residual(con, backtest_run_id, tiers=("warm",), min_shared_gameweeks=5)
    assert 0.0 <= result["rho_residual"] <= 0.99
    assert result["empirical_cov"] > 0  # the synthetic shared_bump construction guarantees positive covariance
    assert result["n_pairs"] == 1


def test_refit_rho_residual_excludes_pairs_below_min_shared_gameweeks(con):
    backtest_run_id = _seed_backtest_run(con)
    values = {("2025-2026", 1): {"p_a": 1, "p_b": 1}, ("2025-2026", 2): {"p_a": 0, "p_b": 2}}
    _seed_realized_goals_assists(con, backtest_run_id, "warm", values)
    with pytest.raises(ValueError):
        bt.refit_rho_residual(con, backtest_run_id, tiers=("warm",), min_shared_gameweeks=5)


def test_refit_rho_residual_raises_when_no_data_recorded(con):
    backtest_run_id = _seed_backtest_run(con)
    with pytest.raises(ValueError):
        bt.refit_rho_residual(con, backtest_run_id)


# ============================================================
# refit_minutes_and_evidence_params -- block coordinate descent
# ============================================================

def _seed_minutes_recalibration_scenario_at(con, season, target_gw, prior_gw, id_prefix):
    """One target gameweek with a prior gameweek's real history for p1 (a nailed starter) so
    minutes_model.run() has something non-degenerate to fit, plus minimal ep_outputs/
    team_strength/ep_model_versions rows to satisfy FKs -- the smallest scenario that exercises
    the real asof_scope + minutes_model.run() + log-score path end to end. Parametrized by
    (season, gameweek, id_prefix) so a single test can seed two disjoint steps (e.g. one per
    season) to exercise refit_minutes_and_evidence_params()'s select/holdout split for real,
    not just against a single step reused as both."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A') ON CONFLICT DO NOTHING")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B') ON CONFLICT DO NOTHING")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Midfielder') ON CONFLICT DO NOTHING")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Player Two', 'Midfielder') ON CONFLICT DO NOTHING")
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', ?, 'team_a', 't')", [season])
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player One', 'player one', '1', ?, 'p1')", [season],
    )
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES ('Player Two', 'player two', '1', ?, 'p2')", [season],
    )
    raw_table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', 'fakehash', 1) ON CONFLICT DO NOTHING", [raw_table, season],
    )
    con.execute(f'CREATE TABLE IF NOT EXISTS "{raw_table}" (code VARCHAR, name VARCHAR)')
    con.execute(f'INSERT INTO "{raw_table}" VALUES (\'1\', \'A\')')

    now = datetime.now()
    prior_kickoff = datetime(2025, 10, 1) if season == "2025-2026" else datetime(2024, 10, 1)
    target_kickoff = datetime(2025, 10, 8) if season == "2025-2026" else datetime(2024, 10, 8)
    prior_id, target_id = f"{id_prefix}_prior", f"{id_prefix}_target"
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES (?, ?, ?, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [prior_id, season, prior_gw, prior_kickoff, now],
    )
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, minutes_played, _ingested_at) "
        "VALUES ('p1', ?, ?, 0, 90, 90, ?)", [prior_id, season, now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES (?, ?, ?, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [target_id, season, target_gw, target_kickoff, now],
    )
    con.execute(
        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, minutes_played, _ingested_at) "
        "VALUES ('p1', ?, ?, 0, 90, 90, ?)", [target_id, season, now],
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES (?, 0.2, 1, 1, 'team_a')", [target_kickoff.date()],
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]

    bt.params_mod.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    bt.params_mod.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    bt.params_mod.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)

    mm_mv = minutes_model.run(
        con, target_kickoff.date(), season,
        decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1, fact_multiplier_params_version=1,
        lookback_seasons=(season,),
    )

    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES (?, ?, ?, ?, 1, 1, 1)", [target_kickoff.date(), season, ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    for player_uid in ("p1", "p2"):
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, ?, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 5.0)",
            [ep_mv, player_uid, target_id],
        )
    return ep_mv


def _seed_minutes_recalibration_scenario(con):
    return _seed_minutes_recalibration_scenario_at(con, "2025-2026", 5, 4, "s1")


def test_refit_minutes_and_evidence_params_never_makes_the_score_worse(con):
    ep_mv = _seed_minutes_recalibration_scenario(con)
    eval_steps = [("2025-2026", 5)]
    ep_model_version_by_step = {("2025-2026", 5): ep_mv}
    base_versions = {
        "decay_params_version": 1, "adjustment_params_version": 1,
        "shrinkage_params_version": 1, "fact_multiplier_params_version": 1,
    }
    param_grids = [{
        "param_family": "minutes_model_shrinkage_params", "param_key": "competitive_matches_threshold",
        "dimensions": None, "candidates": [5, 10, 20], "version_field": "shrinkage_params_version",
    }]

    result = bt.refit_minutes_and_evidence_params(
        con, eval_steps, ep_model_version_by_step, base_versions, param_grids, n_rounds=1,
    )
    assert math.isfinite(result["log_score"])
    assert result["log_score"] >= result["history"][0]["log_score"]  # coordinate descent only ever accepts improvements
    assert "holdout_log_score_before" not in result  # holdout not requested -- old behavior, no new keys


def test_refit_minutes_and_evidence_params_holdout_is_scored_on_disjoint_steps_not_the_select_set(con):
    """Regression test for a real overfitting-risk fix: refit_minutes_and_evidence_params()'s
    coordinate descent selects the best candidate against eval_steps, so reporting that same
    eval_steps score as evidence of improvement is optimistic by construction. holdout_steps
    must be scored independently -- this seeds two genuinely different (season, gameweek)
    steps and confirms the holdout score is actually computed against the holdout step, not
    silently recomputed against (or copied from) the select step's own in-sample score."""
    ep_mv_select = _seed_minutes_recalibration_scenario_at(con, "2025-2026", 5, 4, "select")
    ep_mv_holdout = _seed_minutes_recalibration_scenario_at(con, "2024-2025", 5, 4, "holdout")

    select_steps = [("2025-2026", 5)]
    holdout_steps = [("2024-2025", 5)]
    ep_model_version_by_step = {("2025-2026", 5): ep_mv_select, ("2024-2025", 5): ep_mv_holdout}
    base_versions = {
        "decay_params_version": 1, "adjustment_params_version": 1,
        "shrinkage_params_version": 1, "fact_multiplier_params_version": 1,
    }
    param_grids = [{
        "param_family": "minutes_model_shrinkage_params", "param_key": "competitive_matches_threshold",
        "dimensions": None, "candidates": [5, 10, 20], "version_field": "shrinkage_params_version",
    }]

    result = bt.refit_minutes_and_evidence_params(
        con, select_steps, ep_model_version_by_step, base_versions, param_grids, n_rounds=1,
        holdout_steps=holdout_steps,
    )
    assert result["n_holdout_steps"] == 1
    assert math.isfinite(result["holdout_log_score_before"])
    assert math.isfinite(result["holdout_log_score_after"])

    # cross-check: computing the holdout score directly against the holdout step alone
    # must match what refit_minutes_and_evidence_params() reported -- proves it's genuinely
    # scored against holdout_steps, not the select set.
    direct_holdout_score = bt._minutes_log_score_for_step(
        con, "2024-2025", 5, ep_mv_holdout, 1, 1, 1, 1,
    )
    assert result["holdout_log_score_before"] == pytest.approx(direct_holdout_score)


def test_write_family_version_with_override_copies_other_keys_unchanged(con):
    from fpl_quant import params

    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude", value_numeric=-4.0, dimensions={"claim_type": "injury_status", "category": "Out"})

    bt._write_family_version_with_override(
        con, "minutes_adjustment_params", 1, 2, "2026-08-11",
        "cap", {"scope": "global"}, 8.0,
    )
    new_cap, _ = params.resolve_param(con, "minutes_adjustment_params", "cap", 2, dimensions={"scope": "global"})
    unchanged_magnitude, _ = params.resolve_param(
        con, "minutes_adjustment_params", "magnitude", 2, dimensions={"claim_type": "injury_status", "category": "Out"}
    )
    assert new_cap == pytest.approx(8.0)
    assert unchanged_magnitude == pytest.approx(-4.0)


# ============================================================
# _realized_xi_points -- real FPL scoring: XI only, captain doubled
# ============================================================

def _seed_event_points(con, season, gw, points_by_player):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A') ON CONFLICT DO NOTHING")
    for player_uid, points in points_by_player.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder') ON CONFLICT DO NOTHING", [player_uid, player_uid])
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, _ingested_at) VALUES (?, ?, ?, ?, current_timestamp)",
            [player_uid, season, gw, points],
        )


def test_realized_xi_points_sums_only_the_xi_not_the_bench(con):
    _seed_event_points(con, "2025-2026", 5, {"p1": 10, "p2": 5, "bench1": 100})
    total = bt._realized_xi_points(con, "2025-2026", 5, frozenset({"p1", "p2"}), captain_uid=None)
    assert total == pytest.approx(15.0)


def test_realized_xi_points_doubles_the_captain(con):
    _seed_event_points(con, "2025-2026", 5, {"p1": 10, "p2": 5})
    total = bt._realized_xi_points(con, "2025-2026", 5, frozenset({"p1", "p2"}), captain_uid="p1")
    assert total == pytest.approx(10 * 2 + 5)


def test_realized_xi_points_treats_missing_row_as_zero(con):
    _seed_event_points(con, "2025-2026", 5, {"p1": 10})
    total = bt._realized_xi_points(con, "2025-2026", 5, frozenset({"p1", "p_unscored"}), captain_uid=None)
    assert total == pytest.approx(10.0)


# ============================================================
# has_double_gameweek -- real historical DGW guard
# ============================================================

def test_has_double_gameweek_false_for_a_normal_gameweek(con):
    _seed_two_gameweek_league(con)
    assert bt.has_double_gameweek(con, "2025-2026", 10) is False


def test_has_double_gameweek_true_when_a_team_appears_twice(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_c', 'C')")
    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('m1', '2024-2025', 25, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [datetime(2025, 2, 15), now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('m2', '2024-2025', 25, ?, 'team_a', 'team_c', TRUE, 'Premier League', ?)",
        [datetime(2025, 2, 19), now],
    )
    assert bt.has_double_gameweek(con, "2024-2025", 25) is True  # team_a plays twice


def test_run_step_selection_excludes_double_gameweeks(con):
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_c', 'C')")
    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('m1', '2024-2025', 25, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [datetime(2025, 2, 15), now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('m2', '2024-2025', 25, ?, 'team_a', 'team_c', TRUE, 'Premier League', ?)",
        [datetime(2025, 2, 19), now],
    )
    steps = [
        (s, gw) for s, gw in bt.ALL_SEASON_GAMEWEEKS if s == "2024-2025" and gw == 25
        if not bt.has_double_gameweek(con, s, gw)
    ]
    assert steps == []


# ============================================================
# refit_kappa_tc -- out-of-sample TC captain-choice grid search
# ============================================================

def _seed_kappa_tc_step(con, backtest_run_id, season, gameweek, tier, players):
    """players: [(player_uid, mean_total, var_total, realized_points), ...], all in the XI.
    Builds a real squad_optimizer_runs + monte_carlo_run_versions + monte_carlo_player_summary
    chain (mirrors test_transfer_planner.py's _seed_minimal_squad_optimizer_run /
    _seed_mc_run_and_summary pattern -- no cross-test-file import precedent in this suite) and
    links it into backtest_gameweek_steps the way a real run_gameweek_step() call would, plus
    seeds fact_player_season_stats so _realized_xi_points() has real outcomes to read."""
    for uid, *_ in players:
        con.execute(
            "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder') ON CONFLICT DO NOTHING",
            [uid, uid],
        )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [season],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1)", [season, ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1)",
        [ep_mv, mm_mv, ts_mv],
    )
    un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]
    so_run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (run_id, calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES (nextval('seq_squad_optimizer_run'), '2026-08-10', ?, ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 10.0) "
        "RETURNING run_id",
        [season, gameweek, ep_mv, un_mv],
    ).fetchone()[0]
    mc_mv = con.execute("SELECT nextval('seq_monte_carlo_model_version')").fetchone()[0]
    con.execute(
        "INSERT INTO monte_carlo_run_versions (model_version, calibration_asof_date, squad_optimizer_run_id, "
        "ep_model_version, minutes_model_version, team_strength_model_version, uncertainty_model_version, "
        "rho_residual_params_version, z_fixture_lambda_representative, z_fixture_variance, n_antithetic_pairs, "
        "query_id, seed) VALUES (?, '2026-08-10', ?, ?, ?, ?, ?, 1, 0.1, 0.1, 100, 'test', 1)",
        [mc_mv, so_run_id, ep_mv, mm_mv, ts_mv, un_mv],
    )
    for uid, mean_total, var_total, realized_points in players:
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, TRUE, TRUE, FALSE, FALSE)", [so_run_id, uid],
        )
        con.execute(
            "INSERT INTO monte_carlo_player_summary (model_version, player_uid, mean_total, var_total, "
            "quantile_05, quantile_25, quantile_75, quantile_95, min_total, max_total) "
            "VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0)", [mc_mv, uid, mean_total, var_total],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, event_points, _ingested_at) "
            "VALUES (?, ?, ?, ?, current_timestamp)", [uid, season, gameweek, realized_points],
        )
    con.execute(
        "INSERT INTO backtest_gameweek_steps (backtest_run_id, season, gameweek, tier, data_asof, so_run_id, "
        "mc_model_version, divergence_check_passed) VALUES (?, ?, ?, ?, current_timestamp, ?, ?, TRUE)",
        [backtest_run_id, season, gameweek, tier, so_run_id, mc_mv],
    )


def test_refit_kappa_tc_prefers_the_kappa_with_better_realized_outcomes(con):
    backtest_run_id = _seed_backtest_run(con)
    # high_var has the higher mean but a much wider spread; low_var has a lower mean but is
    # tight. At kappa=0 the argmax always captains high_var; high_var's *realized* points are
    # deliberately mediocre every step, while low_var's are consistently strong -- kappa=0.5
    # flips the argmax to low_var and should score a much higher realized_sharpe.
    for season, gw, hv_realized, lv_realized in [
        ("2025-2026", 10, 2, 6),
        ("2025-2026", 11, 3, 9),
        ("2025-2026", 12, 1, 7),
    ]:
        _seed_kappa_tc_step(con, backtest_run_id, season, gw, "warm", [
            ("high_var", 10.0, 100.0, hv_realized),   # sd=10 -> kappa=0.5 score = 10-5=5
            ("low_var", 8.0, 4.0, lv_realized),        # sd=2  -> kappa=0.5 score = 8-1=7
        ])
    eval_steps = [("2025-2026", 10), ("2025-2026", 11), ("2025-2026", 12)]
    mc_by_step = bt._model_version_map(con, backtest_run_id, "mc_model_version")
    xi_by_step = bt._xi_uids_by_step(con, backtest_run_id)

    result = bt.refit_kappa_tc(con, eval_steps, mc_by_step, xi_by_step, kappa_tc_grid=(0.0, 0.5))

    assert result["grid"][0.0]["n_gameweeks"] == 3
    assert result["grid"][0.5]["n_gameweeks"] == 3
    # kappa=0 always captains high_var (2*2, 2*3, 2*1 = 4, 6, 2)
    assert result["grid"][0.0]["mean_points"] == pytest.approx((4 + 6 + 2) / 3)
    # kappa=0.5 always captains low_var (2*6, 2*9, 2*7 = 12, 18, 14)
    assert result["grid"][0.5]["mean_points"] == pytest.approx((12 + 18 + 14) / 3)
    assert result["grid"][0.5]["realized_sharpe"] > result["grid"][0.0]["realized_sharpe"]
    assert result["best_kappa_tc"] == 0.5


def test_refit_kappa_tc_skips_steps_missing_mc_or_xi_data(con):
    backtest_run_id = _seed_backtest_run(con)
    _seed_kappa_tc_step(con, backtest_run_id, "2025-2026", 10, "warm", [
        ("p1", 10.0, 1.0, 5), ("p2", 8.0, 1.0, 5),
    ])
    eval_steps = [("2025-2026", 10), ("2025-2026", 99)]  # gw99 was never seeded
    mc_by_step = bt._model_version_map(con, backtest_run_id, "mc_model_version")
    xi_by_step = bt._xi_uids_by_step(con, backtest_run_id)

    result = bt.refit_kappa_tc(con, eval_steps, mc_by_step, xi_by_step, kappa_tc_grid=(0.0,))
    assert result["grid"][0.0]["n_gameweeks"] == 1  # only the real, seeded step counted


def test_xi_uids_by_step_excludes_bench_players(con):
    backtest_run_id = _seed_backtest_run(con)
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('bench1', 'B1', 'Defender')"
    )
    _seed_kappa_tc_step(con, backtest_run_id, "2025-2026", 10, "warm", [
        ("p1", 10.0, 1.0, 5), ("p2", 8.0, 1.0, 5),
    ])
    so_run_id = con.execute(
        "SELECT so_run_id FROM backtest_gameweek_steps WHERE backtest_run_id = ? AND gameweek = 10", [backtest_run_id]
    ).fetchone()[0]
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'bench1', TRUE, FALSE, FALSE, FALSE)", [so_run_id],
    )

    xi_by_step = bt._xi_uids_by_step(con, backtest_run_id)
    assert xi_by_step[("2025-2026", 10)] == {"p1", "p2"}
