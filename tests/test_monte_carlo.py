from datetime import date

import numpy as np
import pytest

from fpl_quant import monte_carlo as mc


# ============================================================
# deterministic seeding
# ============================================================

def test_deterministic_seed_reproducible():
    a = mc.deterministic_seed(1, date(2026, 8, 10), "squad_run_1_gw1")
    b = mc.deterministic_seed(1, date(2026, 8, 10), "squad_run_1_gw1")
    assert a == b


def test_deterministic_seed_varies_with_each_input():
    base = mc.deterministic_seed(1, date(2026, 8, 10), "squad_run_1_gw1")
    assert mc.deterministic_seed(2, date(2026, 8, 10), "squad_run_1_gw1") != base
    assert mc.deterministic_seed(1, date(2026, 8, 11), "squad_run_1_gw1") != base
    assert mc.deterministic_seed(1, date(2026, 8, 10), "squad_run_2_gw1") != base


def test_deterministic_seed_fits_default_rng_range():
    seed = mc.deterministic_seed(999, date(2026, 8, 10), "squad_run_999_gw38")
    np.random.default_rng(seed)  # must not raise


# ============================================================
# Z_fixture calibration
# ============================================================

def test_z_fixture_variance_zero_at_zero_rho():
    assert mc.z_fixture_variance(0.0, 0.5) == 0.0


def test_z_fixture_variance_zero_at_zero_lambda():
    assert mc.z_fixture_variance(0.15, 0.0) == 0.0


def test_z_fixture_variance_rejects_rho_at_or_above_one():
    with pytest.raises(ValueError):
        mc.z_fixture_variance(1.0, 0.5)


def test_z_fixture_variance_matches_closed_form():
    rho, lam = 0.15, 0.6
    sigma_z_sq = mc.z_fixture_variance(rho, lam)
    # invert the derivation: rho should be recoverable from sigma_z_sq and lam
    recovered_rho = sigma_z_sq * lam**2 / (lam + sigma_z_sq * lam**2)
    assert recovered_rho == pytest.approx(rho)


def test_z_fixture_variance_reproduces_target_correlation_empirically():
    """The actual generative-mechanism claim under test: two Poisson counts sharing a drawn
    Z_fixture ~ Gamma(mean=1, var=sigma_z_sq) should empirically correlate at ~rho_residual
    when lambda_i=lambda_j=lambda_representative -- the exact scenario the formula is
    calibrated against."""
    rho_target, lam = 0.15, 0.5
    sigma_z_sq = mc.z_fixture_variance(rho_target, lam)

    rng = np.random.default_rng(12345)
    n = 200_000
    u_z = rng.random(n)
    z = mc.sample_z_fixture(sigma_z_sq, u_z)
    x = rng.poisson(z * lam)
    y = rng.poisson(z * lam)

    empirical_rho = np.corrcoef(x, y)[0, 1]
    assert empirical_rho == pytest.approx(rho_target, abs=0.02)


# ============================================================
# bivariate Poisson score grid
# ============================================================

def test_bivariate_poisson_grid_sums_to_one():
    grid = mc.bivariate_poisson_grid(1.4, 1.1, -0.13)
    assert grid.sum() == pytest.approx(1.0)


def test_bivariate_poisson_grid_matches_tau_at_low_scores():
    from fpl_quant import team_strength as ts

    lam_home, lam_away, rho = 1.4, 1.1, -0.13
    grid = mc.bivariate_poisson_grid(lam_home, lam_away, rho, max_goals=10)
    from scipy.stats import poisson as sp_poisson

    for hx, ay in ((0, 0), (0, 1), (1, 0), (1, 1)):
        expected_unnormalized = sp_poisson.pmf(hx, lam_home) * sp_poisson.pmf(ay, lam_away) * ts.tau(hx, ay, lam_home, lam_away, rho)
        # grid is renormalized, so check the *ratio* to an unadjusted cell instead of the raw value
        ratio_grid = grid[hx, ay] / grid[5, 5]
        unadjusted_5_5 = sp_poisson.pmf(5, lam_home) * sp_poisson.pmf(5, lam_away)
        ratio_expected = expected_unnormalized / unadjusted_5_5
        assert ratio_grid == pytest.approx(ratio_expected, rel=1e-6)


def test_bivariate_poisson_grid_negative_rho_inflates_low_score_draws():
    """Dixon-Coles rho<0 (the real fitted PL value) inflates 0-0/1-1 and deflates 1-0/0-1
    relative to the independent Poisson product -- tau(0,0)=1-lam_h*lam_a*rho > 1 and
    tau(1,1)=1-rho > 1 when rho<0, while tau(1,0)=1+lam_a*rho and tau(0,1)=1+lam_h*rho < 1.
    This is the real, checkable direction of Dixon & Coles' 1997 finding (more actual 0-0/
    1-1 draws than an independent-Poisson model predicts), not the opposite."""
    lam_home, lam_away, rho = 1.3, 1.0, -0.13
    grid = mc.bivariate_poisson_grid(lam_home, lam_away, rho)
    from scipy.stats import poisson as sp_poisson

    independent_grid = np.outer(sp_poisson.pmf(np.arange(11), lam_home), sp_poisson.pmf(np.arange(11), lam_away))
    independent_grid /= independent_grid.sum()
    assert grid[0, 0] > independent_grid[0, 0]
    assert grid[1, 1] > independent_grid[1, 1]
    assert grid[1, 0] < independent_grid[1, 0]
    assert grid[0, 1] < independent_grid[0, 1]


# ============================================================
# sample_from_grid
# ============================================================

def test_sample_from_grid_recovers_empirical_frequencies():
    grid = np.array([[0.1, 0.4], [0.4, 0.1]])
    rng = np.random.default_rng(7)
    u = rng.random(200_000)
    home, away = mc.sample_from_grid(grid, u)
    freq_00 = np.mean((home == 0) & (away == 0))
    freq_01 = np.mean((home == 0) & (away == 1))
    assert freq_00 == pytest.approx(0.1, abs=0.01)
    assert freq_01 == pytest.approx(0.4, abs=0.01)


def test_sample_from_grid_antithetic_pair_covers_distribution():
    """u and 1-u should together still recover the grid's shape (sanity check that the
    antithetic-pairing mechanism doesn't bias the marginal distribution)."""
    grid = np.array([[0.1, 0.4], [0.4, 0.1]])
    rng = np.random.default_rng(3)
    u = rng.random(50_000)
    u_full = np.concatenate([u, 1.0 - u])
    home, away = mc.sample_from_grid(grid, u_full)
    freq_11 = np.mean((home == 1) & (away == 1))
    assert freq_11 == pytest.approx(0.1, abs=0.01)


# ============================================================
# sample_poisson_vec
# ============================================================

def test_sample_poisson_vec_matches_poisson_moments():
    lam = np.full(300_000, 2.3)
    rng = np.random.default_rng(1)
    u = rng.random(300_000)
    counts = mc.sample_poisson_vec(lam, u)
    assert counts.mean() == pytest.approx(2.3, abs=0.02)
    assert counts.var() == pytest.approx(2.3, abs=0.05)


def test_sample_poisson_vec_handles_zero_lambda():
    lam = np.zeros(100)
    u = np.full(100, 0.5)
    counts = mc.sample_poisson_vec(lam, u)
    assert np.all(counts == 0)


def test_sample_poisson_vec_varies_per_element_lambda():
    lam = np.array([0.0, 5.0])
    u = np.array([0.5, 0.999999])
    counts = mc.sample_poisson_vec(lam, u)
    assert counts[0] == 0
    assert counts[1] > 0


# ============================================================
# sample_minutes_state_vec
# ============================================================

def test_sample_minutes_state_vec_recovers_proportions():
    rng = np.random.default_rng(42)
    u = rng.random(200_000)
    p0, p1, p2 = 0.2, 0.3, 0.5
    states = mc.sample_minutes_state_vec(p0, p1, p2, u)
    assert np.mean(states == "0") == pytest.approx(p0, abs=0.01)
    assert np.mean(states == "1_59") == pytest.approx(p1, abs=0.01)
    assert np.mean(states == "60plus") == pytest.approx(p2, abs=0.01)


# ============================================================
# sample_plackett_luce_ranks_vec
# ============================================================

def test_plackett_luce_ranks_vec_each_realization_has_at_most_one_rank1():
    rng = np.random.default_rng(5)
    n = 5000
    strengths = {
        "a": np.full(n, 3.0), "b": np.full(n, 1.0), "c": np.full(n, 1.0), "d": np.full(n, 1.0),
    }
    ranks = mc.sample_plackett_luce_ranks_vec(strengths, rng.random(n), rng.random(n), rng.random(n))
    rank1_count = sum((ranks[p] == 1).astype(int) for p in strengths)
    assert np.all(rank1_count <= 1)


def test_plackett_luce_ranks_vec_no_player_gets_two_ranks_same_realization():
    rng = np.random.default_rng(6)
    n = 5000
    strengths = {p: rng.random(n) + 0.1 for p in ("a", "b", "c", "d", "e")}
    ranks = mc.sample_plackett_luce_ranks_vec(strengths, rng.random(n), rng.random(n), rng.random(n))
    for realization in range(200):  # spot-check a subset
        assigned = [r for p in strengths if (r := ranks[p][realization]) != 0]
        assert len(assigned) == len(set(assigned))  # ranks 1/2/3 each used at most once


def test_plackett_luce_ranks_vec_dominant_strength_wins_rank1_most_often():
    rng = np.random.default_rng(8)
    n = 20_000
    strengths = {"strong": np.full(n, 100.0), "weak1": np.full(n, 1.0), "weak2": np.full(n, 1.0)}
    ranks = mc.sample_plackett_luce_ranks_vec(strengths, rng.random(n), rng.random(n), rng.random(n))
    assert np.mean(ranks["strong"] == 1) > 0.9


def test_plackett_luce_ranks_vec_matches_analytic_rank1_probability():
    """Rank-1 probability should match the closed-form Plackett-Luce marginal
    (strength_i / total_strength), same formula M3's plackett_luce_rank_distribution() uses."""
    rng = np.random.default_rng(9)
    n = 100_000
    s = {"a": np.full(n, 2.0), "b": np.full(n, 3.0), "c": np.full(n, 5.0)}
    ranks = mc.sample_plackett_luce_ranks_vec(s, rng.random(n), rng.random(n), rng.random(n))
    total = 2.0 + 3.0 + 5.0
    for player, strength in (("a", 2.0), ("b", 3.0), ("c", 5.0)):
        assert np.mean(ranks[player] == 1) == pytest.approx(strength / total, abs=0.01)


# ============================================================
# lambda_representative (real-DB-shaped, minimal)
# ============================================================

def test_compute_lambda_representative_empty_pool_returns_fallback(con):
    from fpl_quant import expected_points as ep

    ep.seed_v1_params(con)
    lam = mc.compute_lambda_representative(con, [], ep_model_version=1, scoring_params_version=1)
    assert lam == pytest.approx(0.1)


# ============================================================
# _assemble_points -- saves must floor like every other FPL points-per-N rule
# ============================================================

def test_assemble_points_saves_floor_not_continuous_division(con):
    """Regression test for a real bug: pts_saves used continuous division (saves_count /
    saves_per_pt) instead of the real integer-floor rule FPL actually uses (1 point per 3
    saves) -- goals_conceded_floor two lines above the fixed line already gets this right
    (own_goals_against // 2), saves didn't. saves_per_point is seeded at 3.0 by
    expected_points.seed_v1_params(); with that divisor, saves_count=2 must award 0 points
    (not 0.667), and saves_count=4 must award 1 (not 1.333)."""
    from fpl_quant import expected_points as ep

    ep.seed_v1_params(con)
    state = np.array(["60plus"] * 6)
    draws = {
        "state": state,
        "goals": np.zeros(6, dtype=int), "assists": np.zeros(6, dtype=int),
        "clean_sheet": np.zeros(6, dtype=bool), "goals_conceded_floor": np.zeros(6, dtype=int),
        "defcon_hit": np.zeros(6, dtype=bool),
        "saves_count": np.array([0, 2, 3, 4, 5, 6]),
        "rank": np.zeros(6, dtype=int),
    }
    result = mc._assemble_points(con, "Goalkeeper", draws, scoring_params_version=1)
    np.testing.assert_array_equal(result["pts_saves"], [0.0, 0.0, 1.0, 1.0, 1.0, 2.0])


def test_assemble_points_saves_zero_for_non_goalkeeper(con):
    from fpl_quant import expected_points as ep

    ep.seed_v1_params(con)
    state = np.array(["60plus"] * 3)
    draws = {
        "state": state,
        "goals": np.zeros(3, dtype=int), "assists": np.zeros(3, dtype=int),
        "clean_sheet": np.zeros(3, dtype=bool), "goals_conceded_floor": np.zeros(3, dtype=int),
        "defcon_hit": np.zeros(3, dtype=bool),
        "saves_count": np.array([1, 2, 3]),  # a Defender/Midfielder/Forward never has saves in real FPL
        "rank": np.zeros(3, dtype=int),
    }
    result = mc._assemble_points(con, "Defender", draws, scoring_params_version=1)
    np.testing.assert_array_equal(result["pts_saves"], [0.0, 0.0, 0.0])


# ============================================================
# compute_field_covariance -- the real proof the naive EO%-only tracking-error term couldn't
# produce: two players at the IDENTICAL EO get genuinely different field-covariance once one
# of them shares a fixture (and therefore a real z_fixture/scoreline draw) with other
# high-EO players and the other doesn't.
# ============================================================

def _seed_field_covariance_league(con, target_season="2026-2027", target_gameweek=1):
    from fpl_quant import expected_points as ep
    from fpl_quant import uncertainty as un

    ep.seed_v1_params(con)
    un.seed_v1_params(con)
    from fpl_quant import params as params_mod
    params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    clubs = ["clubA", "clubB", "clubC", "clubD"]
    for club in clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [club, club])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid) VALUES (?, ?, ?)", [club, target_season, club])
    # _team_of_for_fixture() (used by simulate_fixture() to resolve clean-sheet side/team
    # correlation) needs a real raw teams table + player_alias, the same raw-ingestion-log
    # convention every other test seeding a raw season-root table already follows.
    raw_teams_table = f"raw_{target_season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{raw_teams_table}" (code VARCHAR, name VARCHAR)')
    for club in clubs:
        con.execute(f'INSERT INTO "{raw_teams_table}" VALUES (?, ?)', [club, club])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', 'x', ?)", [raw_teams_table, target_season, len(clubs)],
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES "
        "('m1', ?, ?, 'clubA', 'clubB', FALSE, 'Premier League', '2026-08-24', current_timestamp), "
        "('m2', ?, ?, 'clubC', 'clubD', FALSE, 'Premier League', '2026-08-24', current_timestamp)",
        [target_season, target_gameweek, target_season, target_gameweek],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'clubA')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    # clubA/clubC (and clubB/clubD) share identical strength -- the two fixtures are
    # symmetric in every way EXCEPT ownership structure, isolating that as the only thing
    # that can move field-covariance apart below.
    for club, attack, defence in (
        ("clubA", 0.3, 0.0), ("clubB", -0.1, 0.1), ("clubC", 0.3, 0.0), ("clubD", -0.1, 0.1),
    ):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, club, attack, defence],
        )
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [target_season],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1)", [target_season, ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]

    # p_crowded (clubA, fixture m1) shares m1 -- and its shared z_fixture/scoreline draw --
    # with two OTHER high-EO teammates. p_isolated (clubC, fixture m2) is the only
    # meaningfully-owned player in its own fixture. Both get the SAME EO (0.20).
    # p_opponent/p_low1/p_low2 all get EO=0.0 -- the ONLY structural difference left between
    # the two (otherwise-symmetric) fixtures is p_crowded's high-EO SAME-TEAM teammates,
    # isolating the z_fixture-shared-tempo correlation channel specifically (teammates only
    # ever correlate positively through it -- unlike opponents, which also carry a negative
    # clean-sheet-complementarity channel that would confound the direction being tested here).
    roster = [
        ("p_crowded", "clubA", "m1", 0.20), ("p_teammate1", "clubA", "m1", 0.30), ("p_teammate2", "clubA", "m1", 0.25),
        ("p_opponent", "clubB", "m1", 0.0),
        ("p_isolated", "clubC", "m2", 0.20), ("p_low1", "clubD", "m2", 0.0), ("p_low2", "clubD", "m2", 0.0),
    ]
    # expected_bps deliberately LOW and equal for the teammates/opponents/low-EO fillers (only
    # p_crowded/p_isolated -- the two players actually being compared -- get a real, higher
    # expected_bps): the Plackett-Luce bonus mechanism is a genuine zero-sum competition for
    # the SAME 3 bonus slots among everyone in a match, which creates a real NEGATIVE
    # correlation component between same-match participants of comparable strength -- found by
    # this test's own first (failing) run, where identical expected_bps for every participant
    # let that zero-sum competition swamp the positive z_fixture-driven goals/assists channel
    # this test actually means to isolate. Keeping the other participants clearly weaker avoids
    # that confound rather than asserting a wrong direction against it.
    bps_by_uid = {"p_crowded": 18.0, "p_isolated": 18.0}
    eo_by_uid = {}
    for uid, club, match_id, eo_percent in roster:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), club, target_season, uid],
        )
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_own, "
            "p_start_historical_position_avg, weight_own, p_start_historical_final, logit_adjustment_total, "
            "p_start_final, p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, "
            "competitive_matches_last_2_seasons) VALUES (?, ?, 'Midfielder', 0.8, 0.5, 1.0, 0.8, 0.0, 0.8, "
            "0.1, 0.1, 0.2, 0.7, 50)", [mm_mv, uid],
        )
        expected_bps = bps_by_uid.get(uid, 4.0)
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 1.5,0.5,0.3,0.2,0,0.1,0.3,0,0,0,0, 2.9, ?)",
            [ep_mv, uid, match_id, expected_bps],
        )
        # simulate_fixture()'s per-realization goals/assists draw (the ONLY channel real
        # z_fixture-shared-tempo correlation actually flows through) reads
        # expected_goals_per_90/expected_assists_per_90 off fact_player_season_stats, not off
        # ep_outputs -- omitting this (this test's own first two failing attempts did) leaves
        # goals/assists at a permanent 0 variance, so the ONLY same-match correlation left
        # standing is the Plackett-Luce bonus mechanism's zero-sum ranking competition
        # (negative for every match participant, teammates included), which is real but not
        # what this test means to isolate.
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, minutes, expected_goals, "
            "expected_assists, _ingested_at) VALUES (?, ?, 1, 900, 8.0, 4.0, current_timestamp)",
            [uid, target_season],
        )
        eo_by_uid[uid] = eo_percent * 100.0  # effective_ownership() divides by 100 -- store as a percent like real selected_by_percent

    return {"ts_mv": ts_mv, "mm_mv": mm_mv, "ep_mv": ep_mv, "eo_by_uid": eo_by_uid, "target_season": target_season, "target_gameweek": target_gameweek}


def test_field_covariance_differs_for_equal_eo_players_by_team_correlation_exposure(con):
    """The real proof the naive EO%-only version couldn't produce: p_crowded and p_isolated
    have the IDENTICAL EO (0.20) but genuinely different field-covariance, because p_crowded
    shares a real, simulated fixture (and its z_fixture/scoreline draw) with two other
    high-EO players while p_isolated's fixture has almost no field weight in it."""
    from fpl_quant import squad_optimizer as so

    seeded = _seed_field_covariance_league(con)
    eo_fraction_by_uid = {uid: pct / 100.0 for uid, pct in seeded["eo_by_uid"].items()}

    field_cov = mc.compute_field_covariance(
        con, date(2026, 8, 21), seeded["target_season"], seeded["target_gameweek"],
        seeded["ep_mv"], seeded["mm_mv"], seeded["ts_mv"],
        scoring_params_version=1, tau_params_version=1, rho_residual_params_version=1,
        eo_by_uid=eo_fraction_by_uid, n_antithetic_pairs=4000,
    )
    assert "p_crowded" in field_cov and "p_isolated" in field_cov
    assert field_cov["p_crowded"] > field_cov["p_isolated"], (
        f"equal-EO players must NOT get equal field-covariance -- p_crowded (shares a fixture "
        f"with other high-EO players) should exceed p_isolated: {field_cov}"
    )


def test_field_covariance_empty_eo_returns_empty_dict(con):
    seeded = _seed_field_covariance_league(con)
    result = mc.compute_field_covariance(
        con, date(2026, 8, 21), seeded["target_season"], seeded["target_gameweek"],
        seeded["ep_mv"], seeded["mm_mv"], seeded["ts_mv"],
        scoring_params_version=1, tau_params_version=1, rho_residual_params_version=1,
        eo_by_uid={},
    )
    assert result == {}
