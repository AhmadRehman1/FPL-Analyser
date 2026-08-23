import json
import math
from datetime import date, datetime

import pandas as pd
import pytest

from fpl_quant import backtest as bt
from fpl_quant import minutes_model
from fpl_quant import transfer_planner as tp


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


def test_asof_scope_schedule_horizon_widens_schedule_visibility_not_results(con):
    """Regression test for the new schedule_horizon_gameweeks param (needed for M8's
    compute_horizon_ep(), which plans several gameweeks ahead in one call): with the default
    (1), GW20's schedule stays fully invisible from a GW10 asof_scope, exactly like before this
    param existed. With schedule_horizon_gameweeks=11 (covers GW10..GW20 inclusive), GW20's
    schedule becomes visible -- but its RESULT must still be hidden, and its player-level match
    stats must stay fully invisible regardless (the widened window only ever touches fact_match's
    schedule columns, never fact_player_match_stats -- a real look-ahead leak would be letting a
    future gameweek's actual outcome or player stats through, not its announced fixture list)."""
    _seed_two_gameweek_league(con)
    with bt.asof_scope(con, "2025-2026", 10):  # default schedule_horizon_gameweeks=1
        row = con.execute("SELECT match_id FROM fact_match WHERE match_id = 'gw20m'").fetchone()
        assert row is None, "GW20's schedule must stay invisible at the default horizon"

    with bt.asof_scope(con, "2025-2026", 10, schedule_horizon_gameweeks=11):
        row = con.execute(
            "SELECT home_team_uid, away_team_uid, home_score, away_score, finished FROM fact_match WHERE match_id = 'gw20m'"
        ).fetchone()
        assert row is not None, "GW20's schedule must become visible within the widened horizon"
        assert row[0] == "team_a" and row[1] == "team_b"
        assert row[2] is None and row[3] is None, "GW20's result must still be hidden"
        assert row[4] is False
        stats_visible = {r[0] for r in con.execute("SELECT match_id FROM fact_player_match_stats").fetchall()}
        assert "gw20m" not in stats_visible, "player-level stats for a future gameweek must never leak, schedule or not"


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
# Priority 9b -- segment classification (_previous_season / _is_newly_promoted_team /
# _is_new_signing) and score_gameweek(compute_segments=True) wiring
# ============================================================

def test_previous_season_returns_the_prior_loaded_season(con):
    _seed_two_gameweek_league(con)  # matches in both 2024-2025 and 2025-2026
    assert bt._previous_season(con, "2025-2026") == "2024-2025"


def test_previous_season_none_for_the_earliest_loaded_season(con):
    _seed_two_gameweek_league(con)
    assert bt._previous_season(con, "2024-2025") is None


def test_previous_season_none_for_a_season_with_no_data_at_all(con):
    _seed_two_gameweek_league(con)
    assert bt._previous_season(con, "2026-2027") is None


def _seed_promotion_scenario(con):
    """team_a/team_b have real fixtures in BOTH 2024-2025 and 2025-2026 (not promoted);
    team_c's only fixture is in 2025-2026 (promoted)."""
    for uid, name in (("team_a", "A"), ("team_b", "B"), ("team_c", "C")):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('prior', '2024-2025', 30, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [datetime(2025, 3, 1, 15, 0), now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('target', '2025-2026', 10, ?, 'team_a', 'team_c', TRUE, 'Premier League', ?)",
        [datetime(2025, 11, 1, 15, 0), now],
    )


def test_is_newly_promoted_team_false_for_a_team_that_played_the_prior_season(con):
    _seed_promotion_scenario(con)
    assert bt._is_newly_promoted_team(con, "team_a", "2025-2026") is False


def test_is_newly_promoted_team_true_for_a_team_absent_from_the_prior_season(con):
    _seed_promotion_scenario(con)
    assert bt._is_newly_promoted_team(con, "team_c", "2025-2026") is True


def test_is_newly_promoted_team_none_when_theres_no_prior_season_to_compare(con):
    _seed_promotion_scenario(con)
    assert bt._is_newly_promoted_team(con, "team_a", "2024-2025") is None


def _seed_signing_scenario(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'Player One', 'Midfielder')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p2', 'Player Two', 'Forward')")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p3', 'Player Three', 'Forward')")
    # p1: team_code '1' both seasons -- not a new signing
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P1', 'p1', '1', '2024-2025', 'p1')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P1', 'p1', '1', '2025-2026', 'p1')")
    # p2: team_code '1' in 2024-2025, '3' in 2025-2026 -- a real transfer, new signing
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P2', 'p2', '1', '2024-2025', 'p2')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P2', 'p2', '3', '2025-2026', 'p2')")
    # p3: two distinct team_codes WITHIN 2025-2026 itself (a genuine mid-season transfer) -- ambiguous
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P3a', 'p3a', '1', '2025-2026', 'p3')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P3b', 'p3b', '2', '2025-2026', 'p3')")
    # fact_match rows purely so _previous_season resolves 2025-2026 -> 2024-2025
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('prior', '2024-2025', 30, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [datetime(2025, 3, 1, 15, 0), now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('target', '2025-2026', 10, ?, 'team_a', 'team_b', TRUE, 'Premier League', ?)",
        [datetime(2025, 11, 1, 15, 0), now],
    )


def test_is_new_signing_false_when_team_code_unchanged(con):
    _seed_signing_scenario(con)
    assert bt._is_new_signing(con, "p1", "2025-2026") is False


def test_is_new_signing_true_when_team_code_changed(con):
    _seed_signing_scenario(con)
    assert bt._is_new_signing(con, "p2", "2025-2026") is True


def test_is_new_signing_none_when_ambiguous_within_the_season(con):
    _seed_signing_scenario(con)
    assert bt._is_new_signing(con, "p3", "2025-2026") is None


def test_is_new_signing_none_when_no_prior_season_to_compare(con):
    _seed_signing_scenario(con)
    assert bt._is_new_signing(con, "p1", "2024-2025") is None


# ============================================================
# score_gameweek(compute_segments=True) -- end-to-end wiring
# ============================================================

def _seed_score_gameweek_segment_scenario(con):
    """p1/team_a: existed both seasons, no transfer -- no segments. p2/team_c: team_c is
    brand new to 2025-2026 AND p2 transferred in from team_a -- both promoted_team and
    new_signing. p3/team_a: confirmed primary penalty taker -- set_piece_taker. Full real
    score_gameweek() dependency chain (team_strength_snapshots, ep_outputs, minutes_model_
    outputs, fact_player_match_stats) hand-seeded, same pattern as this file's other
    manually-seeded scenarios."""
    for uid, name in (("team_a", "A"), ("team_b", "B"), ("team_c", "C")):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
    for uid, name, position in (("p1", "Player One", "Forward"), ("p2", "Player Two", "Forward"), ("p3", "Player Three", "Forward")):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, name, position])

    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "home_score, away_score, finished, competition, _ingested_at) VALUES "
        "('prior', '2024-2025', 30, ?, 'team_a', 'team_b', 1, 0, TRUE, 'Premier League', ?)",
        [datetime(2025, 3, 1, 15, 0), now],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "home_score, away_score, finished, competition, _ingested_at) VALUES "
        "('target', '2025-2026', 10, ?, 'team_a', 'team_c', 2, 1, TRUE, 'Premier League', ?)",
        [datetime(2025, 11, 1, 15, 0), now],
    )
    for uid, goals, assists in (("p1", 0, 0), ("p2", 1, 0), ("p3", 1, 0)):
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, goals, assists, team_goals_conceded, _ingested_at) "
            "VALUES (?, 'target', '2025-2026', 0, 90, 90, ?, ?, 1, ?)",
            [uid, goals, assists, now],
        )

    # player_alias: p1 stays on team_a ('1') both seasons; p2 moves from team_a ('1') to
    # team_c ('3'); p3 is on team_a ('1') in the target season only (no prior-season claim
    # needed for the set-piece segment).
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P1', 'p1', '1', '2024-2025', 'p1')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P1', 'p1', '1', '2025-2026', 'p1')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P2', 'p2', '1', '2024-2025', 'p2')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P2', 'p2', '3', '2025-2026', 'p2')")
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P3', 'p3', '1', '2025-2026', 'p3')")

    # team_alias + raw teams.csv for the target season, needed by monte_carlo._team_of_for_fixture()
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', '2025-2026', 'team_a', 't')")
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('C', '2025-2026', 'team_c', 't')")
    con.execute('CREATE TABLE "raw_2025_2026_teams" (code VARCHAR, name VARCHAR)')
    con.execute("INSERT INTO \"raw_2025_2026_teams\" VALUES ('1', 'A'), ('3', 'C')")
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES ('raw_2025_2026_teams', '2025-2026', 'teams.csv', 'fakehash', 2)"
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2025-11-01', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for uid in ("team_a", "team_b", "team_c"):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, 0.1, 0.1, 2, 1.0)",
            [ts_mv, uid],
        )

    con.execute(
        "INSERT INTO minutes_model_versions (model_version, calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES (1, '2025-11-01', '2025-2026', 1, 1, 1, 1, '[]')"
    )
    for uid in ("p1", "p2", "p3"):
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
            "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (1, ?, 'Forward', 0.9, 0.9, 1.0, 0.0, 0.9, 0.0, 0.05, 0.05, 0.9, 20)",
            [uid],
        )

    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2025-11-01', '2025-2026', ?, 1, 1, 1, 1)", [ts_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    for uid in ("p1", "p2", "p3"):
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, 'target', 1.0, 0.5, 0.2, 0, 0, 0, 0.3, 0, 0, 0, 0, 2.0, 20.0)",
            [ep_mv, uid],
        )

    # p3: confirmed primary penalty taker, asof-visible before the target gameweek's deadline
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) VALUES ('src1', 'Test Source', 'official', 0.9)")
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "claim_value, information_type, source_id, source_reliability_score, confidence, "
        "observed_date, ingested_date, tab_origin, row_origin) "
        "VALUES ('claim_p3', 'player', 'p3', 'set_piece_order_override', ?, 'FACT', 'src1', 0.9, 0.9, "
        "'2025-08-01', '2025-08-01 00:00:00', 'research_pull:SetPieceTakers', 1)",
        [json.dumps({"club": "A", "duty": "Penalties", "order": "primary"})],
    )
    bt.ep.seed_v1_params(con)  # base_scoring_matrix, set_piece_evidence_params (1.15), etc.

    backtest_run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    return backtest_run_id, ep_mv, ts_mv


def test_score_gameweek_records_segment_suffixed_metrics_when_opted_in(con):
    backtest_run_id, ep_mv, ts_mv = _seed_score_gameweek_segment_scenario(con)
    bt.score_gameweek(
        con, backtest_run_id, "2025-2026", 10, ep_mv, 1, ts_mv, 1,
        compute_segments=True, set_piece_params_version=1,
    )
    rows = dict(con.execute(
        "SELECT metric_name, metric_value FROM backtest_metrics WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchall())
    assert "log_score_goals_mean:promoted_team" in rows  # p2, on team_c
    assert "log_score_goals_mean:new_signing" in rows    # p2, transferred in
    assert "log_score_goals_mean:set_piece_taker" in rows  # p3, confirmed penalty taker
    # aggregate (unsuffixed) metrics still get recorded exactly as before -- segments are additive
    assert "log_score_goals_mean" in rows


def test_score_gameweek_records_no_segment_metrics_when_not_opted_in(con):
    backtest_run_id, ep_mv, ts_mv = _seed_score_gameweek_segment_scenario(con)
    bt.score_gameweek(con, backtest_run_id, "2025-2026", 10, ep_mv, 1, ts_mv, 1)
    names = {r[0] for r in con.execute(
        "SELECT metric_name FROM backtest_metrics WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchall()}
    assert not any(":" in n and not n.startswith("realized_") for n in names)


# ============================================================
# Priority 9c -- "beats the ownership-weighted average manager"
# ============================================================

def _seed_beats_crowd_scenario(con):
    """p1 (mu=6.0, 50% owned, scored 10, captained + in XI), p2 (mu=4.0, 20% owned, scored
    5, in XI), p3 (mu=3.0, 80% owned despite the lower projection -- a real differential-vs-
    template shape, scored 2, NOT in the model's squad but still part of the real field).
    Minimal FK chain for ep_outputs (team_strength/minutes model version rows) plus a real
    squad_optimizer_runs/selections row so so_run_id-gated code paths have something to read."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
    for uid, name in (("p1", "Player One"), ("p2", "Player Two"), ("p3", "Player Three")):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward')", [uid, name])

    now = datetime.now()
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2025-11-01', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2025-11-01', '2025-2026', 1, 1, 1, 1, '[]')"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2025-11-01', '2025-2026', ?, ?, 1, 1, 1)", [ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]

    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES "
        "('m1', '2025-2026', 10, ?, 'team_a', 'team_b', FALSE, 'Premier League', ?)",
        [datetime(2025, 11, 1, 15, 0), now],
    )

    for uid, ep_total, sbp, points in (("p1", 6.0, 50.0, 10), ("p2", 4.0, 20.0, 5), ("p3", 3.0, 80.0, 2)):
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, 'm1', 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?, 20.0)",
            [ep_mv, uid, ep_total],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, selected_by_percent, event_points, _ingested_at) "
            "VALUES (?, '2025-2026', 10, ?, ?, ?)",
            [uid, sbp, points, now],
        )

    con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2025-11-01', ?, ?, ?, 1)",
        [ep_mv, mm_mv, ts_mv],
    )
    un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES ('2025-11-01', '2025-2026', 10, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 10.0) RETURNING run_id",
        [ep_mv, un_mv],
    )
    so_run_id = con.execute("SELECT max(run_id) FROM squad_optimizer_runs").fetchone()[0]
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p1', TRUE, TRUE, TRUE, FALSE)", [so_run_id],
    )
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p2', TRUE, TRUE, FALSE, TRUE)", [so_run_id],
    )

    bt.params_mod.write_param(con, "ownership_params", 1, "2026-08-10", "captaincy_concentration", value_numeric=0.3)
    bt.ep.seed_v1_params(con)  # base_scoring_matrix etc., needed by score_gameweek's M3 loop
    return ep_mv, mm_mv, ts_mv, so_run_id


def test_avg_manager_benchmark_points_matches_direct_eo_computation(con):
    ep_mv, *_ = _seed_beats_crowd_scenario(con)
    result = bt._avg_manager_benchmark_points(con, "2025-2026", 10, ep_mv, ownership_params_version=1)

    # Recompute independently via the same real ownership.py primitives (already unit-tested
    # in test_ownership.py) rather than duplicating the EO formula by hand here -- this test's
    # job is to verify the query/wiring, not re-derive EO math.
    from fpl_quant import ownership as ownership_mod
    candidates = [
        {"player_uid": "p1", "position": "Forward", "mu": 6.0, "selected_by_percent": 50.0},
        {"player_uid": "p2", "position": "Forward", "mu": 4.0, "selected_by_percent": 20.0},
        {"player_uid": "p3", "position": "Forward", "mu": 3.0, "selected_by_percent": 80.0},
    ]
    eo_by_uid = ownership_mod.compute_eo_for_pool(candidates, captaincy_concentration=0.3)
    expected = sum((eo_by_uid[uid] / 100.0) * pts for uid, pts in (("p1", 10), ("p2", 5), ("p3", 2)))
    assert result == pytest.approx(expected)


def test_avg_manager_benchmark_points_none_when_no_ownership_data(con):
    ep_mv, *_ = _seed_beats_crowd_scenario(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = NULL WHERE season = '2025-2026' AND gw = 10")
    assert bt._avg_manager_benchmark_points(con, "2025-2026", 10, ep_mv, ownership_params_version=1) is None


def test_score_gameweek_records_beats_crowd_metrics_when_opted_in(con):
    ep_mv, mm_mv, ts_mv, so_run_id = _seed_beats_crowd_scenario(con)
    backtest_run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.score_gameweek(
        con, backtest_run_id, "2025-2026", 10, ep_mv, mm_mv, ts_mv, 1, so_run_id=so_run_id,
        ownership_params_version=1,
    )
    rows = dict(con.execute(
        "SELECT metric_name, metric_value FROM backtest_metrics WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchall())
    # p1 captained (10*2) + p2 (5) = 25
    assert rows["model_squad_realized_points"] == pytest.approx(25.0)
    assert "avg_manager_benchmark_points" in rows
    assert rows["beats_crowd_points_delta"] == pytest.approx(rows["model_squad_realized_points"] - rows["avg_manager_benchmark_points"])


def test_score_gameweek_skips_beats_crowd_metrics_when_not_opted_in(con):
    ep_mv, mm_mv, ts_mv, so_run_id = _seed_beats_crowd_scenario(con)
    backtest_run_id = con.execute("INSERT INTO backtest_runs (warm_up_gameweeks) VALUES (0) RETURNING backtest_run_id").fetchone()[0]
    bt.score_gameweek(con, backtest_run_id, "2025-2026", 10, ep_mv, mm_mv, ts_mv, 1, so_run_id=so_run_id)
    names = {r[0] for r in con.execute(
        "SELECT metric_name FROM backtest_metrics WHERE backtest_run_id = ?", [backtest_run_id]
    ).fetchall()}
    assert "beats_crowd_points_delta" not in names
    assert "model_squad_realized_points" not in names


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
# write_recalibration_seed_file / load_confirmed_recalibration_seeds -- Phase B1 hardening
# (durable, git-committed copy of a recalibration run, independent of the DuckDB file itself)
# ============================================================

def test_write_recalibration_seed_file_captures_proposal_fields(con, tmp_path):
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    backtest_run_id = _seed_backtest_run(con)
    bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=1,
    )

    out_path = bt.write_recalibration_seed_file(con, backtest_run_id, tmp_path)
    assert out_path == tmp_path / f"seeds_{backtest_run_id}.json"
    payload = json.loads(out_path.read_text())
    assert payload["backtest_run_id"] == backtest_run_id
    [proposal] = payload["proposals"]
    assert proposal["param_family"] == "model_decay_params"
    assert proposal["param_key"] == "xi"
    assert proposal["new_value"] == pytest.approx(0.003)
    assert proposal["status"] == "pending"


def test_load_confirmed_recalibration_seeds_excludes_pending_and_rejected(con, tmp_path):
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    backtest_run_id = _seed_backtest_run(con)
    confirmed_id = bt.propose_recalibration(
        con, backtest_run_id, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=1,
    )
    rejected_id = bt.propose_recalibration(
        con, backtest_run_id, "risk_aversion_params", "lambda_value", 0.5,
        metric_name="realized_sharpe", metric_before=0.0, metric_after=-1.0, old_params_version=None,
    )
    con.execute("UPDATE recalibration_proposals SET status = 'confirmed' WHERE proposal_id = ?", [confirmed_id])
    con.execute("UPDATE recalibration_proposals SET status = 'rejected' WHERE proposal_id = ?", [rejected_id])
    bt.write_recalibration_seed_file(con, backtest_run_id, tmp_path)

    seeds = bt.load_confirmed_recalibration_seeds(tmp_path)
    assert len(seeds) == 1
    assert seeds[0]["param_family"] == "model_decay_params"
    assert seeds[0]["status"] == "confirmed"


def test_load_confirmed_recalibration_seeds_reads_across_multiple_seed_files(con, tmp_path):
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    run_a = _seed_backtest_run(con, notes="run a")
    run_b = _seed_backtest_run(con, notes="run b")
    p1 = bt.propose_recalibration(
        con, run_a, "model_decay_params", "xi", 0.003,
        metric_name="neg_log_likelihood", metric_before=100.0, metric_after=95.0, old_params_version=1,
    )
    p2 = bt.propose_recalibration(
        con, run_b, "correlation_params", "rho_residual", 0.0,
        metric_name="rho_hat", metric_before=0.15, metric_after=0.0, old_params_version=None,
    )
    con.execute("UPDATE recalibration_proposals SET status = 'confirmed' WHERE proposal_id IN (?, ?)", [p1, p2])
    bt.write_recalibration_seed_file(con, run_a, tmp_path)
    bt.write_recalibration_seed_file(con, run_b, tmp_path)

    seeds = bt.load_confirmed_recalibration_seeds(tmp_path)
    families = {s["param_family"] for s in seeds}
    assert families == {"model_decay_params", "correlation_params"}


def test_load_confirmed_recalibration_seeds_empty_when_dir_missing(tmp_path):
    assert bt.load_confirmed_recalibration_seeds(tmp_path / "does_not_exist") == []


def test_recalibrate_seed_dir_none_is_a_no_op(con, monkeypatch):
    """Frozen-contract default: an existing caller that never passes seed_dir gets byte-
    identical behavior to before this parameter existed -- write_recalibration_seed_file()
    must never even be called, not just "called with no visible effect"."""
    from fpl_quant import params

    params.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    backtest_run_id = _seed_backtest_run(con)

    def _fail(*args, **kwargs):
        raise AssertionError("write_recalibration_seed_file must not be called when seed_dir is None")

    monkeypatch.setattr(bt, "write_recalibration_seed_file", _fail)
    bt.recalibrate(
        con, backtest_run_id,
        current_xi_version=1, current_rho_version=1, current_rho_residual_version=1,
        current_minutes_versions={}, current_lambda_version=1, guardrail_cap=3.0,
        minutes_param_grids=[], refit_xi_rho_flag=False, refit_rho_residual_flag=False,
        refit_minutes_flag=False, refit_lambda_flag=False,
    )


def test_recalibrate_writes_seed_file_when_seed_dir_given(con, tmp_path):
    from fpl_quant import params

    params.write_param(con, "correlation_params", 1, "2026-08-10", "rho_residual", value_numeric=0.15)
    backtest_run_id = _seed_backtest_run(con)
    bt.recalibrate(
        con, backtest_run_id,
        current_xi_version=1, current_rho_version=1, current_rho_residual_version=1,
        current_minutes_versions={}, current_lambda_version=1, guardrail_cap=3.0,
        minutes_param_grids=[], refit_xi_rho_flag=False, refit_rho_residual_flag=False,
        refit_minutes_flag=False, refit_lambda_flag=False, seed_dir=tmp_path,
    )
    out_path = tmp_path / f"seeds_{backtest_run_id}.json"
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["proposals"] == []  # every refit_*_flag disabled -- nothing to propose, still written


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
# season_cumulative_metrics -- season-long trajectory scoring, hand-computed values
# ============================================================

def test_season_cumulative_metrics_matches_hand_computed_sharpe_and_drawdown():
    """weekly_points=[50,60,40,70,30]: mean=50, population std=sqrt(200)~=14.142,
    sharpe=50/14.142~=3.5355. Cumulative surplus over the mean is [0,10,0,20,0]; the running
    peak is [0,10,10,20,20], so the underwater series is [0,0,10,0,20] -- deepest fall of 20,
    at gameweek 3 (a real -20 dip straight after the trajectory's own new high of +20)."""
    result = bt.season_cumulative_metrics([50, 60, 40, 70, 30])
    assert result["total_points"] == pytest.approx(250.0)
    assert result["mean_points"] == pytest.approx(50.0)
    assert result["n_gameweeks"] == 5
    assert result["realized_sharpe"] == pytest.approx(50.0 / math.sqrt(200.0))
    assert result["max_drawdown"] == pytest.approx(20.0)


def test_season_cumulative_metrics_constant_trajectory_has_zero_drawdown_and_no_sharpe_signal():
    """A perfectly flat trajectory has no variability to speak of: std=0 (no meaningful
    risk-adjusted signal, matching refit_lambda()'s own std<=0 -> -inf convention) and
    max_drawdown=0 (cumulative surplus is identically zero throughout, so it never falls
    below its own running peak)."""
    result = bt.season_cumulative_metrics([50.0] * 6)
    assert result["max_drawdown"] == pytest.approx(0.0)
    assert result["realized_sharpe"] == float("-inf")


def test_season_cumulative_metrics_distinguishes_a_deep_cold_streak_from_a_smooth_season():
    """The real point this metric exists to make: two trajectories with the identical total
    and identical variance can still have very different drawdowns depending on WHERE in the
    season the bad weeks land -- a real risk dimension Sharpe's single whole-season standard
    deviation can't see, and season_cumulative_metrics() must actually distinguish them, not
    just report the same number for both."""
    lumpy = [80.0, 80.0, 80.0, 20.0, 20.0, 20.0]       # one long cold streak at the end
    alternating = [80.0, 20.0, 80.0, 20.0, 80.0, 20.0]  # same total/mean/variance, spread out
    lumpy_result = bt.season_cumulative_metrics(lumpy)
    alternating_result = bt.season_cumulative_metrics(alternating)

    assert lumpy_result["total_points"] == pytest.approx(alternating_result["total_points"])
    assert lumpy_result["realized_sharpe"] == pytest.approx(alternating_result["realized_sharpe"])
    assert lumpy_result["max_drawdown"] > alternating_result["max_drawdown"]


def test_season_cumulative_metrics_empty_trajectory_returns_safe_defaults():
    result = bt.season_cumulative_metrics([])
    assert result == {
        "total_points": 0.0, "mean_points": 0.0, "n_gameweeks": 0,
        "realized_sharpe": float("-inf"), "max_drawdown": 0.0,
    }


# ============================================================
# run_season_simulation -- a real, evolving M8 manager, not a fresh M5 solve every step.
#
# Full end-to-end (real team_strength.calibrate() -> minutes_model.run() -> ep.run() ->
# uncertainty.run() -> squad_optimizer.run() -> transfer_planner.run()/apply_recommendation()
# every gameweek), so this needs a real, non-degenerate 6-club/18-player synthetic league, not
# a lightweight fixture -- the same category of investment M7/M8's own README documents needing
# "two scratch dry runs against a full copy of the real DB" for, just at a scale a unit test can
# carry. Every club fields a full round of fixtures every gameweek (so every player has one),
# and players are split nailed-starter/fringe with real differentiated historical output (not
# flat), which is what lets squad_optimizer.run()'s own divergence check pass for real rather
# than degenerately. Confirmed stable across 5 different random seeds before being written down
# here, not a single lucky run.
# ============================================================

def _seed_season_simulation_league(con, seasons=("2024-2025", "2025-2026"), n_gameweeks=5, seed=7):
    """6 clubs, a full 3-match round every gameweek (every club has a fixture every week),
    18 players (2 GK/6 DEF/6 MID/4 FWD) split into real nailed-starter vs fringe-player
    historical output so squad_optimizer's own candidate pool and divergence check have real,
    non-degenerate signal to work with -- not test_squad_optimizer.py's in-memory
    _synthetic_pool() (that bypasses the real ep.run()/uncertainty.run() pipeline entirely;
    run_season_simulation() exercises it for real, every gameweek)."""
    import random as _random
    rng = _random.Random(seed)
    now = datetime.now()
    clubs = ["A", "B", "C", "D", "E", "F"]
    uids = {}
    for name in clubs:
        uid = f"team_{name.lower()}"
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
        uids[name] = uid

    # uncertainty.run()'s cross-player-covariance roster lookup is hardcoded to
    # season_priority[0] (default "2026-2027") regardless of which season is actually being
    # processed -- a pre-existing quirk, out of scope to fix here, worked around the same way
    # the real ingested DB happens to satisfy it naturally (2026-2027 also carries
    # player_alias/team_alias rows for the same real players as prior seasons).
    for season in (*seasons, "2026-2027"):
        table = f"raw_{season.replace('-', '_')}_teams"
        con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR, elo VARCHAR)')
        for i, name in enumerate(clubs):
            con.execute(f'INSERT INTO "{table}" VALUES (?, ?, ?)', [str(i + 1), name, str(1700 + i * 30)])
            con.execute(
                "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, ?, ?, 't')",
                [name, season, uids[name]],
            )
        con.execute(
            "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
            "VALUES (?, ?, 'teams.csv', ?, ?)", [table, season, f"hash_{season}", len(clubs)],
        )

    rounds = [[("A", "B"), ("C", "D"), ("E", "F")], [("B", "C"), ("D", "E"), ("F", "A")], [("A", "D"), ("B", "E"), ("C", "F")]]
    match_ids_by_season: dict[str, dict[int, list]] = {}
    for si, season in enumerate(seasons):
        gw_dates = {gw: date(2024 + si, 8, 1) + (gw - 1) * pd.Timedelta(days=7) for gw in range(1, n_gameweeks + 3)}
        match_ids_by_season[season] = {}
        for gw, kickoff in gw_dates.items():
            fixtures = rounds[(gw - 1) % len(rounds)]
            match_ids_by_season[season][gw] = []
            for i, (h, a) in enumerate(fixtures):
                hg, ag = rng.choice([(2, 1), (1, 0), (0, 0), (3, 1), (1, 1), (2, 0)])
                mid = f"m_{season}_{gw}_{i}"
                con.execute(
                    "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, home_score, "
                    "away_score, finished, competition, kickoff_time, _ingested_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, TRUE, 'Premier League', ?, ?)",
                    [mid, season, gw, uids[h], uids[a], hg, ag, kickoff, now],
                )
                match_ids_by_season[season][gw].append((mid, h, a))

    players = []
    pid = 0
    for pos, n, n_nailed in (("Goalkeeper", 2, 1), ("Defender", 6, 4), ("Midfielder", 6, 4), ("Forward", 4, 3)):
        for i in range(n):
            pid += 1
            players.append((f"p{pid}", pos, clubs[i % 6], i < n_nailed))

    for uid, pos, club, nailed in players:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, pos])
        code = str(clubs.index(club) + 1)
        for season in (*seasons, "2026-2027"):
            con.execute(
                "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
                "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), code, season, uid],
            )
        price = 4.0 + (3.0 if nailed else 0.0) + (1.5 if pos == "Forward" else 0.0)
        for season in seasons:
            for gw, fixtures in match_ids_by_season[season].items():
                con.execute(
                    "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, minutes, "
                    "goals_scored, assists, event_points, _ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [uid, season, gw, price, 90 if nailed else 0,
                     1 if (nailed and rng.random() < 0.3) else 0, 1 if (nailed and rng.random() < 0.2) else 0,
                     rng.randint(4, 10) if nailed else rng.randint(0, 1), now],
                )
                this_match = next(((mid, h, a) for mid, h, a in fixtures if club in (h, a)), None)
                if this_match:
                    mid, _h, _a = this_match
                    con.execute(
                        "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
                        "minutes_played, goals, assists, team_goals_conceded, _ingested_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [uid, mid, season, 0 if nailed else 200, 90 if nailed else 0, 90 if nailed else 0,
                         1 if (nailed and rng.random() < 0.3) else 0, 1 if (nailed and rng.random() < 0.2) else 0, 1, now],
                    )

    # full v1 param seed, mirroring scripts/run_ingestion.py's own real sequence
    from fpl_quant import expected_points as ep_mod
    from fpl_quant import squad_optimizer as so_mod
    from fpl_quant import transfer_planner as tp_mod
    from fpl_quant import uncertainty as un_mod

    params_mod = bt.params_mod
    params_mod.write_param(con, "fact_type_multiplier_params", 1, "2026-08-10", "multiplier", value_numeric=1.2)
    params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "xi", value_numeric=0.0018)
    params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    for claim_type, magnitude in {"predicted_xi": 0.8, "manager_tendency": 1.0, "transfer_likelihood": -2.0}.items():
        params_mod.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude",
                                value_numeric=magnitude, dimensions={"claim_type": claim_type})
    for category, magnitude in {"Out": -4.0, "Doubt": -1.5, "Fit": 0.0}.items():
        params_mod.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "magnitude", value_numeric=magnitude,
                                dimensions={"claim_type": "injury_status", "category": category})
    params_mod.write_param(con, "minutes_adjustment_params", 1, "2026-08-10", "cap", value_numeric=6.0, dimensions={"scope": "global"})
    params_mod.write_param(con, "minutes_model_decay_params", 1, "2026-08-10", "xi", value_numeric=math.log(2) / 200)
    params_mod.write_param(con, "minutes_model_shrinkage_params", 1, "2026-08-10", "competitive_matches_threshold", value_numeric=10)
    ep_mod.seed_v1_params(con)
    un_mod.seed_v1_params(con)
    so_mod.seed_v1_params(con)
    tp_mod.seed_v1_params(con)


_SEASON_SIM_VERSIONS = dict(
    xi_params_version=1, rho_params_version=1,
    decay_params_version=1, adjustment_params_version=1, shrinkage_params_version=1, fact_multiplier_params_version=1,
    scoring_params_version=1, bps_params_version=1, tau_params_version=1,
    rho_residual_params_version=1, corr_params_version=1,
    lambda_params_version=1, guardrail_params_version=1,
    horizon_params_version=1, transfer_cost_params_version=1,
    wildcard_threshold_params_version=1, free_hit_threshold_params_version=1, kappa_tc_params_version=1,
)


def test_run_season_simulation_walks_forward_with_real_decisions(con):
    _seed_season_simulation_league(con)
    result = bt.run_season_simulation(
        con, "2025-2026", start_gameweek=2, end_gameweek=4, n_antithetic_pairs=200, **_SEASON_SIM_VERSIONS,
    )

    assert result["gameweeks"] == [2, 3, 4]
    assert len(result["weekly_points"]) == 3
    assert all(p >= 0.0 for p in result["weekly_points"])
    assert result["skipped_dgw_gameweeks"] == []

    # the squad must have genuinely evolved through real M8 decisions, not sat frozen --
    # confirms this is really calling transfer_planner.run()/apply_recommendation() each week,
    # not silently reusing the bootstrap squad throughout.
    assert len(result["actions"]) == 2  # one real transfer_planner.run()-informed decision per gameweek after bootstrap (GW3, GW4)

    final_holdings = tp._read_holdings(con, result["final_state_version"])
    assert len(final_holdings) == 15

    metrics = bt.season_cumulative_metrics(result["weekly_points"])
    assert math.isfinite(metrics["total_points"])
    assert metrics["max_drawdown"] >= 0.0


def test_run_season_simulation_raises_on_unfittable_start_gameweek(con):
    """No prior finished match exists anywhere -- the real cold-start guard, same one
    has_fittable_history() already protects run_gameweek_step() with."""
    with pytest.raises(ValueError):
        bt.run_season_simulation(con, "2025-2026", start_gameweek=1, end_gameweek=2, **_SEASON_SIM_VERSIONS)


# ============================================================
# bank-tracking look-ahead: _compute_bank_for_squad()'s `ORDER BY gw DESC` price lookup has no
# ceiling of its own -- correct for a real live run (no future gameweeks exist to leak from),
# a real leak inside run_season_simulation() unless the call composes with asof_scope()'s own
# shadow. Tested directly against the underlying mechanism (bootstrap_from_squad_optimizer_run
# inside vs. outside asof_scope), not the full season-sim fixture -- the existing
# _seed_season_simulation_league() gives every player one FLAT price across all gameweeks by
# construction, so it can't distinguish a leak from a non-leak; this needs a genuine multi-gw
# price difference for one player, which is cheaper to build standalone.
# ============================================================

def _seed_bootstrap_squad_with_price_history(con, season, start_gameweek, later_gameweek, price_at_start, price_at_later):
    """One real squad_optimizer_runs + squad_optimizer_selections row (one held player, p1)
    plus TWO fact_player_season_stats price snapshots for that player: one at gw < start_gameweek
    (asof-safe -- knowable at start_gameweek's own deadline) and one at later_gameweek (only
    knowable after it) -- deliberately a HIGHER price later, the real shape of the leak: bank
    would be UNDERSTATED (a real transfer that should be legal looking illegal) if the earlier,
    lower price were used, but OVERSTATED (an illegal transfer looking affordable) if the later,
    higher price leaked through -- the actually dangerous direction, per _compute_bank_for_squad()'s
    own docstring on why it conservatively returns 0.0 rather than guess."""
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'P1', 'Midfielder')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A'), ('team_b', 'B')")
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, ?, 'team_a', 'team_b', FALSE, "
        "'Premier League', ?, current_timestamp)",
        [season, start_gameweek, date(2026, 8, 1) + (start_gameweek - 1) * pd.Timedelta(days=7)],
    )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('p1', ?, ?, ?, current_timestamp)", [season, start_gameweek - 1, price_at_start],
    )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('p1', ?, ?, ?, current_timestamp)", [season, later_gameweek, price_at_later],
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
        [season, start_gameweek, ep_mv, un_mv],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
        "VALUES (?, 'p1', TRUE, TRUE, TRUE, FALSE)", [so_run_id],
    )
    return so_run_id


def test_bootstrap_bank_composes_with_asof_scope_never_leaking_a_later_price(con):
    so_run_id = _seed_bootstrap_squad_with_price_history(
        con, "2025-2026", start_gameweek=5, later_gameweek=10, price_at_start=5.0, price_at_later=9.0,
    )
    with bt.asof_scope(con, "2025-2026", 5):
        shadowed_state_version = tp.bootstrap_from_squad_optimizer_run(con, so_run_id)
    shadowed_bank = con.execute(
        "SELECT bank FROM manager_state_versions WHERE state_version = ?", [shadowed_state_version]
    ).fetchone()[0]
    # priced off gw4 (< start_gameweek=5), never gw10's 9.0 -- the fix this test guards.
    assert shadowed_bank == pytest.approx(tp.squad_optimizer.BUDGET - 5.0)

    # Contrast, in the same test: called with NO asof_scope at all (the pre-fix call site in
    # run_season_simulation()), _compute_bank_for_squad()'s own `ORDER BY gw DESC` genuinely
    # does pick up the later, real price -- proving the asof_scope wrapping above is actually
    # what makes the difference, not a coincidence of this fixture's data.
    unshadowed_state_version = tp.bootstrap_from_squad_optimizer_run(con, so_run_id)
    unshadowed_bank = con.execute(
        "SELECT bank FROM manager_state_versions WHERE state_version = ?", [unshadowed_state_version]
    ).fetchone()[0]
    assert unshadowed_bank == pytest.approx(tp.squad_optimizer.BUDGET - 9.0)


def test_run_season_simulation_calls_compute_bank_only_while_the_asof_shadow_is_active(con, monkeypatch):
    """Ties the fix directly to run_season_simulation()'s own call sites, not just the
    underlying asof_scope()/bootstrap_from_squad_optimizer_run() mechanism in isolation (see
    the price-value test above, which wraps the call in its own `with` block and so can't by
    itself catch a regression if run_season_simulation() moved the real call back outside its
    with-block). Spies on _compute_bank_for_squad() and records, at every real call during a
    real run, whether fact_player_season_stats currently resolves to asof_scope()'s shadowed
    TEMP TABLE rather than main.* directly. The bootstrap call always fires at least once;
    apply_recommendation()'s Wildcard-accept call is opportunistic (only fires if the sim's own
    decision rule accepts one that week) and is covered whenever it happens to fire."""
    _seed_season_simulation_league(con)
    seen_shadowed = []
    real_fn = tp._compute_bank_for_squad

    def _spy(con_arg, *args, **kwargs):
        is_shadowed = con_arg.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = 'fact_player_season_stats' AND temporary"
        ).fetchone()[0] > 0
        seen_shadowed.append(is_shadowed)
        return real_fn(con_arg, *args, **kwargs)

    monkeypatch.setattr(tp, "_compute_bank_for_squad", _spy)
    bt.run_season_simulation(con, "2025-2026", start_gameweek=2, end_gameweek=4, n_antithetic_pairs=200, **_SEASON_SIM_VERSIONS)

    assert seen_shadowed  # at least the bootstrap call fired
    assert all(seen_shadowed)  # every real call happened while the shadow was active -- none leaked


def test_report_season_simulation_sensitivity_writes_versions_without_touching_the_live_pin(con):
    """Regression test for the read-only contract: running a lambda/cap sensitivity sweep must
    never mutate what version=1 (the live pin) itself resolves to -- each grid candidate gets
    its own freshly written param_versions row, same discipline as
    report_concentration_sensitivity()."""
    _seed_season_simulation_league(con)
    live_lambda_before, _ = bt.params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", 1)
    live_cap_before, _ = bt.params_mod.resolve_param(con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", 1)

    # lambda=0.0 (and, empirically on this small synthetic pool, 0.05) is excluded from the
    # grid: run_season_simulation() always goes through the real, divergence-checked
    # squad_optimizer.run() (not solve() directly, unlike refit_lambda()), and that check always
    # compares its trial lambda against a lambda=0 baseline -- trialing lambda=0.0 itself would
    # compare that baseline against itself and always "fail" by construction, and a lambda close
    # enough to 0 can genuinely fail to move this pool's optimal XI/captain at all, a real
    # (if synthetic-data-specific) finding, not a test bug -- confirmed empirically (0.05 fails,
    # 0.10-0.50 all pass reliably) before picking this grid, not guessed.
    result = bt.report_season_simulation_sensitivity(
        con, "2025-2026", 2, 3, _SEASON_SIM_VERSIONS,
        lambda_grid=(0.10, 0.15), guardrail_cap_grid=(2,),
    )

    assert set(result["lambda"].keys()) == {0.10, 0.15}
    assert set(result["guardrail_cap"].keys()) == {2}
    for grid_result in {**result["lambda"], **result["guardrail_cap"]}.values():
        assert math.isfinite(grid_result["total_points"])
        assert grid_result["max_drawdown"] >= 0.0

    live_lambda_after, _ = bt.params_mod.resolve_param(con, "risk_aversion_params", "lambda_value", 1)
    live_cap_after, _ = bt.params_mod.resolve_param(con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", 1)
    assert live_lambda_after == live_lambda_before
    assert live_cap_after == live_cap_before


# ============================================================
# _decide_gameweek_action -- the harness's own decision rule, tested in isolation against a
# lightweight chip_evaluations/transfer_recommendations fixture (no real M1-M6 solve needed --
# this is pure decision logic over already-computed recommendations).
# ============================================================

def _seed_plan_run_with_recommendations(con, recommended_chips=(), top_transfer_net_value=None, target_gameweek=3, detail_by_chip=None):
    detail_by_chip = detail_by_chip or {}
    state_version = con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available, "
        "chips_used_set1, chips_used_set2, derived_from_state_version) "
        "VALUES ('2026-2027', ?, 1, '[]', '[]', NULL) RETURNING state_version", [target_gameweek],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, "
        "input_state_version, horizon_params_version, transfer_cost_params_version, ep_model_versions, "
        "uncertainty_model_versions) VALUES ('2026-08-17', '2026-2027', ?, ?, 1, 1, '{}', '{}') RETURNING run_id",
        [target_gameweek, state_version],
    )
    run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]
    for chip_type in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
        con.execute(
            "INSERT INTO chip_evaluations (run_id, chip_type, recommended, score_or_gain, detail, gw19_urgent_flag) "
            "VALUES (?, ?, ?, 1.0, ?, FALSE)",
            [run_id, chip_type, chip_type in recommended_chips, json.dumps(detail_by_chip.get(chip_type, {}))],
        )
    if top_transfer_net_value is not None:
        con.execute(
            "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('out1', 'Out', 'Midfielder'), "
            "('in1', 'In', 'Midfielder') ON CONFLICT DO NOTHING"
        )
        con.execute(
            "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, price_out, price_in, "
            "horizon_value_gain, transfer_cost, net_value) VALUES (?, 1, 'out1', 'in1', 5.0, 5.0, ?, 0.0, ?)",
            [run_id, top_transfer_net_value, top_transfer_net_value],
        )
    return run_id


def test_decide_gameweek_action_accepts_the_transfer_when_no_chip_is_recommended(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=(), top_transfer_net_value=4.0)
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert rank == 1
    assert chip is None


def test_decide_gameweek_action_declines_a_transfer_below_threshold(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=(), top_transfer_net_value=-1.0)
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert rank is None
    assert chip is None


def test_decide_gameweek_action_prefers_a_recommended_chip_over_a_good_transfer(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=("bench_boost",), top_transfer_net_value=10.0)
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert rank is None
    assert chip == "bench_boost"


def test_decide_gameweek_action_follows_chip_priority_order(con):
    """wildcard > free_hit > bench_boost > triple_captain, per CHIP_PRIORITY."""
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=("triple_captain", "bench_boost", "free_hit"))
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip == "free_hit"


def test_decide_gameweek_action_skips_a_chip_already_used_this_set(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=("wildcard", "bench_boost"))
    rank, chip = bt._decide_gameweek_action(
        con, run_id, chips_used_set1={"wildcard"}, chips_used_set2=set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0,
    )
    assert chip == "bench_boost"


def test_decide_gameweek_action_checks_the_correct_chip_set_for_the_gameweek(con):
    """A chip used in set 1 (pre-GW19) must not block re-recommending it in set 2 (GW19+) --
    the two sets are independent allocations, per the real FPL rule."""
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=("wildcard",))
    con.execute("UPDATE transfer_plan_runs SET target_gameweek = 25 WHERE run_id = ?", [run_id])
    rank, chip = bt._decide_gameweek_action(
        con, run_id, chips_used_set1={"wildcard"}, chips_used_set2=set(), target_gameweek=25, accept_transfer_if_net_value_above=0.0,
    )
    assert chip == "wildcard"


def test_decide_gameweek_action_no_recommendations_at_all_does_nothing(con):
    run_id = _seed_plan_run_with_recommendations(con, recommended_chips=())
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert rank is None and chip is None


# ============================================================
# chip timing: play-now-vs-hold, using only the model's own already-visible forward EP
# ============================================================

def test_is_best_gameweek_in_visible_horizon_true_when_target_is_the_min():
    assert bt._is_best_gameweek_in_visible_horizon({3: 10.0, 4: 20.0, 5: 15.0}, target_gameweek=3, prefer="min")


def test_is_best_gameweek_in_visible_horizon_false_when_a_later_week_is_lower():
    # gameweek 4's projected value (5.0) is lower than gameweek 3's (10.0) -- a genuinely
    # worse-looking future week is real evidence to hold a rebuild chip, not play it now.
    assert not bt._is_best_gameweek_in_visible_horizon({3: 10.0, 4: 5.0, 5: 15.0}, target_gameweek=3, prefer="min")


def test_is_best_gameweek_in_visible_horizon_true_when_target_is_the_max():
    assert bt._is_best_gameweek_in_visible_horizon({3: 20.0, 4: 10.0, 5: 15.0}, target_gameweek=3, prefer="max")


def test_is_best_gameweek_in_visible_horizon_false_when_a_later_week_is_higher():
    assert not bt._is_best_gameweek_in_visible_horizon({3: 10.0, 4: 20.0, 5: 15.0}, target_gameweek=3, prefer="max")


def test_is_best_gameweek_in_visible_horizon_defers_to_true_on_missing_data():
    # Empty per_gw (a chip evaluator with no comparison data) or target_gameweek simply not
    # present -- can't assess timing, defer to the existing threshold-only check rather than
    # silently suppressing an otherwise-real recommendation.
    assert bt._is_best_gameweek_in_visible_horizon({}, target_gameweek=3, prefer="min")
    assert bt._is_best_gameweek_in_visible_horizon({4: 5.0, 5: 15.0}, target_gameweek=3, prefer="min")


def test_is_best_gameweek_in_visible_horizon_handles_stringified_json_keys():
    # chip_evaluations.detail round-trips through json.dumps/json.loads -- dict keys come back
    # as strings ("3", not 3). The real bug this guards: comparing int target_gameweek against
    # string dict keys would never match, silently defeating the whole comparison.
    assert bt._is_best_gameweek_in_visible_horizon({"3": 10.0, "4": 20.0}, target_gameweek=3, prefer="min")
    assert not bt._is_best_gameweek_in_visible_horizon({"3": 10.0, "4": 5.0}, target_gameweek=3, prefer="min")


def test_decide_gameweek_action_holds_a_recommended_chip_when_a_later_week_looks_better(con):
    # Wildcard clears its threshold (recommended=True) but the current squad's own visible
    # trajectory says gameweek 5 is a genuinely worse week than gameweek 3 -- the real
    # situation Wildcard exists to fix is worse LATER, not now, so this should hold, not play.
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("wildcard",), target_gameweek=3,
        detail_by_chip={"wildcard": {"current_squad_value_per_gw": {3: 20.0, 4: 15.0, 5: 5.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip is None  # held, not played
    assert rank is None  # no transfer seeded either in this fixture


def test_decide_gameweek_action_plays_a_recommended_chip_when_now_is_the_best_visible_week(con):
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("wildcard",), target_gameweek=3,
        detail_by_chip={"wildcard": {"current_squad_value_per_gw": {3: 5.0, 4: 15.0, 5: 20.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip == "wildcard"


def test_decide_gameweek_action_held_chip_falls_through_to_next_priority_chip(con):
    # wildcard is recommended but timing says hold; free_hit is recommended AND timing says
    # now is genuinely its best visible week -- the loop must keep trying, not stop at the
    # first (held) candidate.
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("wildcard", "free_hit"), target_gameweek=3,
        detail_by_chip={
            "wildcard": {"current_squad_value_per_gw": {3: 20.0, 4: 5.0}},
            "free_hit": {"current_xi_value_per_gw": {3: 5.0, 4: 20.0}},
        },
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip == "free_hit"


def test_decide_gameweek_action_held_chip_falls_through_to_a_transfer(con):
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("wildcard",), top_transfer_net_value=4.0, target_gameweek=3,
        detail_by_chip={"wildcard": {"current_squad_value_per_gw": {3: 20.0, 4: 5.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip is None
    assert rank == 1


def test_decide_gameweek_action_bench_boost_timing_uses_all_gameweeks_field(con):
    # bench_boost's own evaluate_bench_boost() already carries a real per-gameweek breakdown
    # (all_gameweeks) -- this is the pre-existing bug the reviewer flagged: recommended was
    # always True whenever there's a bench, so bench_boost got taken on the very first
    # eligible week regardless of whether a later week actually maximizes bench EP.
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("bench_boost",), target_gameweek=3,
        detail_by_chip={"bench_boost": {"all_gameweeks": {3: 5.0, 4: 20.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip is None  # gw4 looks better than gw3 -- hold


def test_decide_gameweek_action_triple_captain_timing_uses_captain_value_per_gw_field(con):
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("triple_captain",), target_gameweek=3,
        detail_by_chip={"triple_captain": {"captain_value_per_gw": {3: 12.0, 4: 4.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=3, accept_transfer_if_net_value_above=0.0)
    assert chip == "triple_captain"  # gw3 is already the best visible week -- play now


def test_decide_gameweek_action_timing_gate_is_skipped_for_set2_gameweeks(con):
    # Chip timing is explicitly scoped to set-1 (GW1-18) per the task's own framing -- set-2
    # gameweeks (GW19+) keep the original threshold-only check, even when a per_gw trajectory
    # that would otherwise say "hold" is present in the detail payload.
    run_id = _seed_plan_run_with_recommendations(
        con, recommended_chips=("wildcard",), target_gameweek=25,
        detail_by_chip={"wildcard": {"current_squad_value_per_gw": {25: 20.0, 26: 5.0}}},
    )
    rank, chip = bt._decide_gameweek_action(con, run_id, set(), set(), target_gameweek=25, accept_transfer_if_net_value_above=0.0)
    assert chip == "wildcard"  # would be held under set-1 rules, but set-2 ignores timing


# ============================================================
# _realized_xi_points -- captain_multiplier extension (Triple Captain support)
# ============================================================

def test_realized_xi_points_captain_multiplier_defaults_to_double(con):
    _seed_event_points(con, "2025-2026", 5, {"p1": 10})
    total = bt._realized_xi_points(con, "2025-2026", 5, frozenset({"p1"}), captain_uid="p1")
    assert total == pytest.approx(20.0)


def test_realized_xi_points_captain_multiplier_supports_triple_captain(con):
    _seed_event_points(con, "2025-2026", 5, {"p1": 10, "p2": 5})
    total = bt._realized_xi_points(con, "2025-2026", 5, frozenset({"p1", "p2"}), captain_uid="p1", captain_multiplier=3)
    assert total == pytest.approx(10 * 3 + 5)


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
