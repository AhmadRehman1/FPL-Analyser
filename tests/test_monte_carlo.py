from datetime import date, datetime

import numpy as np
import pytest

from fpl_quant import expected_points as ep_mod
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
# sample_binomial_vec / _multinomial_allocate_team_goals (Review B4)
# ============================================================

def test_sample_binomial_vec_matches_binomial_moments():
    n = np.full(300_000, 8.0)
    p = np.full(300_000, 0.35)
    rng = np.random.default_rng(1)
    u = rng.random(300_000)
    counts = mc.sample_binomial_vec(n, p, u, max_k=8)
    assert counts.mean() == pytest.approx(8 * 0.35, abs=0.02)
    assert counts.var() == pytest.approx(8 * 0.35 * 0.65, abs=0.05)


def test_sample_binomial_vec_never_exceeds_n():
    rng = np.random.default_rng(3)
    n = rng.integers(0, 6, size=50_000).astype(float)
    p = np.full(50_000, 0.9)
    u = rng.random(50_000)
    counts = mc.sample_binomial_vec(n, p, u, max_k=5)
    assert np.all(counts <= n)
    assert np.all(counts >= 0)


def test_sample_binomial_vec_handles_zero_n():
    n = np.zeros(100)
    p = np.full(100, 0.5)
    u = np.full(100, 0.9)
    counts = mc.sample_binomial_vec(n, p, u, max_k=5)
    assert np.all(counts == 0)


def test_multinomial_allocate_team_goals_conserves_the_team_total():
    rng = np.random.default_rng(11)
    team_goals = rng.integers(0, 5, size=20_000).astype(float)
    lambdas = {
        "a": np.full(20_000, 1.5),
        "b": np.full(20_000, 0.3),
        "c": np.full(20_000, 0.8),
    }

    def u_pair_fn():
        return rng.random(20_000)

    allocated = mc._multinomial_allocate_team_goals(team_goals, lambdas, u_pair_fn)
    total = sum(allocated.values())
    assert np.array_equal(total, team_goals.astype(np.int64))
    for arr in allocated.values():
        assert np.all(arr >= 0)


def test_multinomial_allocate_team_goals_favors_the_higher_lambda_player():
    rng = np.random.default_rng(12)
    n_real = 50_000
    team_goals = np.full(n_real, 3.0)  # fixed team total -- isolates the allocation split
    lambdas = {"high": np.full(n_real, 3.0), "low": np.full(n_real, 0.3)}

    def u_pair_fn():
        return rng.random(n_real)

    allocated = mc._multinomial_allocate_team_goals(team_goals, lambdas, u_pair_fn)
    assert allocated["high"].mean() > allocated["low"].mean()
    # exact expectation: E[goals_i] = team_goals * lambda_i / sum(lambda)
    assert allocated["high"].mean() == pytest.approx(3.0 * 3.0 / 3.3, abs=0.05)
    assert allocated["low"].mean() == pytest.approx(3.0 * 0.3 / 3.3, abs=0.02)


def test_multinomial_allocate_team_goals_last_player_absorbs_zero_lambda_remainder():
    """Disclosed edge case (see _multinomial_allocate_team_goals()'s own docstring): when
    every remaining player's lambda is 0 but goals remain (own goals -- outside this
    project's modeled scope), the last player in sorted order absorbs the remainder rather
    than losing it -- conservation always holds, even in this degenerate case."""
    team_goals = np.array([2.0, 0.0, 5.0])
    lambdas = {"z1": np.zeros(3), "z2": np.zeros(3)}
    allocated = mc._multinomial_allocate_team_goals(team_goals, lambdas, lambda: np.zeros(3))
    assert np.array_equal(allocated["z1"], np.zeros(3))  # not last (sorted order) -> gets 0
    assert np.array_equal(allocated["z2"], team_goals.astype(np.int64))  # last -> absorbs all


def test_multinomial_allocate_team_goals_empty_pool_returns_empty():
    assert mc._multinomial_allocate_team_goals(np.array([1.0, 2.0]), {}, lambda: np.array([])) == {}


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
# simulate_fixture() -- Review B4 integration test: summed squad goals must equal the
# fixture's own already-drawn scoreline once every fixture participant is a squad player.
# ============================================================

def _seed_simulate_fixture_scenario(con, monkeypatch, rates_by_uid):
    """team_a (home) vs team_b (away), 2026-2027, one real match. Every roster player passed
    in rates_by_uid is a SQUAD player and the only participant on their side -- a controlled
    fixture with no non-squad participants, so the coherence invariant (summed squad goals ==
    drawn team goal total) is directly checkable with nothing else absorbing part of the
    scoreline. player_rates_shrunk/_defensive_action_rates_per_90 are monkeypatched (real
    per-90 rate history isn't what this test is verifying) to return rates_by_uid's own
    expected_goals_per_90 per player, 0 for every other rate."""
    from fpl_quant import params

    for uid, name in (("team_a", "A"), ("team_b", "B")):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
    for uid, position in rates_by_uid.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position["position"]])

    now = datetime.now()
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, kickoff_time, home_team_uid, away_team_uid, "
        "finished, competition, _ingested_at) VALUES ('m1', '2026-2027', 5, ?, 'team_a', 'team_b', FALSE, 'Premier League', ?)",
        [datetime(2026, 10, 1, 15, 0), now],
    )

    # player_alias/team_alias/raw teams.csv chain needed by monte_carlo._team_of_for_fixture().
    for uid, r in rates_by_uid.items():
        team_code = "1" if uid in ("p1", "p2") else "2"
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, '2026-2027', ?)", [uid, uid, team_code, uid],
        )
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', '2026-2027', 'team_a', 't')")
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('B', '2026-2027', 'team_b', 't')")
    con.execute('CREATE TABLE "raw_2026_2027_teams" (code VARCHAR, name VARCHAR)')
    con.execute("INSERT INTO \"raw_2026_2027_teams\" VALUES ('1', 'A'), ('2', 'B')")
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES ('raw_2026_2027_teams', '2026-2027', 'teams.csv', 'fakehash', 2)"
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-09-01', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for uid, attack, defence in (("team_a", 0.5, 0.0), ("team_b", 0.0, 0.0)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, uid, attack, defence],
        )
    params.write_param(con, "model_decay_params", 1, "2026-09-01", "rho", value_numeric=-0.13)

    con.execute(
        "INSERT INTO minutes_model_versions (model_version, calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES (1, '2026-09-01', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    for uid, r in rates_by_uid.items():
        # Deterministic minutes: everyone plays the full 90 every realization, isolating the
        # goals-allocation logic from minutes-state randomness.
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
            "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (1, ?, ?, 0.95, 0.95, 1.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0, 20)",
            [uid, r["position"]],
        )

    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-09-01', '2026-2027', ?, 1, 1, 1, 1)", [ts_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    for uid, r in rates_by_uid.items():
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, 'm1', 1.0, 0.3, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 1.6, 15.0)",
            [ep_mv, uid],
        )
    ep_mod.seed_v1_params(con)

    fixed_rates = {
        uid: {"expected_goals_per_90": r["expected_goals_per_90"], "expected_assists_per_90": 0.2, "saves_per_90": 0.0}
        for uid, r in rates_by_uid.items()
    }
    monkeypatch.setattr(ep_mod, "player_rates_shrunk", lambda con, player_uid, position, season_priority: fixed_rates[player_uid])
    monkeypatch.setattr(ep_mod, "_defensive_action_rates_per_90", lambda con, player_uid, position, season_priority: {"cbi_per_90": 0.0, "recoveries_per_90": 0.0})

    return ts_mv, ep_mv


def test_simulate_fixture_squad_goals_conserve_the_drawn_scoreline(con, monkeypatch):
    rates_by_uid = {
        "p1": {"position": "Forward", "expected_goals_per_90": 0.6},
        "p2": {"position": "Midfielder", "expected_goals_per_90": 0.2},
        "p4": {"position": "Forward", "expected_goals_per_90": 0.5},
    }
    ts_mv, ep_mv = _seed_simulate_fixture_scenario(con, monkeypatch, rates_by_uid)
    squad_uids = {"p1", "p2", "p4"}

    result = mc.simulate_fixture(
        con, "m1", "team_a", "team_b", "2026-2027", ["2026-2027"], squad_uids,
        ep_mv, 1, ts_mv, scoring_params_version=1, tau_val=1.0, sigma_z_sq=0.0,
        mean_minutes={"mean_1_59": 30.0, "mean_60plus": 90.0},
        rng=np.random.default_rng(99), n_pairs=3000,
    )

    assert set(result) == squad_uids
    home_goals_sum = result["p1"]["goals"] + result["p2"]["goals"]
    # Independently reconstruct the drawn home scoreline via the exact same lambdas
    # simulate_fixture() itself resolves, rather than reaching into its internals.
    lam_home, lam_away, _ = ep_mod._fixture_lambdas(con, "team_a", "m1", ts_mv)
    assert home_goals_sum.sum() > 0  # not a degenerate all-zero fixture
    assert np.all(result["p4"]["goals"] >= 0)

    # The real invariant: every one of the 2*n_pairs realizations' summed squad goals must
    # equal the actual drawn team total for that realization. Recompute the draw ourselves
    # from a fixed seed to compare against, since simulate_fixture() doesn't expose home_goals/
    # away_goals directly -- reproduce with the identical rng/seed/n_pairs so the SAME draw
    # sequence for the scoreline is generated (it's drawn first inside simulate_fixture()).
    rng = np.random.default_rng(99)
    grid = mc.bivariate_poisson_grid(lam_home, lam_away, -0.13)
    u = rng.random(3000)
    u_pair = np.concatenate([u, 1.0 - u])
    home_goals_expected, away_goals_expected = mc.sample_from_grid(grid, u_pair)

    np.testing.assert_array_equal(home_goals_sum, home_goals_expected)
    np.testing.assert_array_equal(result["p4"]["goals"], away_goals_expected)


# ============================================================
# report_covariance_dilution() -- Review B5
# ============================================================

def _seed_covariance_pairs(con, pairs, summary_var):
    """pairs: [(uid_a, uid_b, empirical_covariance, relationship), ...].
    summary_var: {uid: var_total} -- monte_carlo_player_summary rows for every uid involved.
    Builds the real FK chain monte_carlo_run_versions needs (mirrors test_backtest.py's
    _seed_kappa_tc_step -- no cross-test-file import precedent in this suite) since DuckDB
    enforces it. Returns the seeded mc_model_version."""
    for uid in summary_var:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Forward') ON CONFLICT DO NOTHING", [uid, uid])

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')",
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
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
    so_run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (run_id, calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES (nextval('seq_squad_optimizer_run'), '2026-08-10', '2026-2027', 1, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 10.0) "
        "RETURNING run_id",
        [ep_mv, un_mv],
    ).fetchone()[0]
    mc_mv = con.execute("SELECT nextval('seq_monte_carlo_model_version')").fetchone()[0]
    con.execute(
        "INSERT INTO monte_carlo_run_versions (model_version, calibration_asof_date, squad_optimizer_run_id, "
        "ep_model_version, minutes_model_version, team_strength_model_version, uncertainty_model_version, "
        "rho_residual_params_version, z_fixture_lambda_representative, z_fixture_variance, n_antithetic_pairs, "
        "query_id, seed) VALUES (?, '2026-08-10', ?, ?, ?, ?, ?, 1, 0.1, 0.1, 100, 'test', 1)",
        [mc_mv, so_run_id, ep_mv, mm_mv, ts_mv, un_mv],
    )

    for uid, var_total in summary_var.items():
        con.execute(
            "INSERT INTO monte_carlo_player_summary (model_version, player_uid, mean_total, var_total, "
            "quantile_05, quantile_25, quantile_75, quantile_95, min_total, max_total) "
            "VALUES (?, ?, 4.0, ?, 0.0, 2.0, 6.0, 10.0, 0.0, 15.0)",
            [mc_mv, uid, var_total],
        )
    for uid_a, uid_b, cov, relationship in pairs:
        con.execute(
            "INSERT INTO monte_carlo_empirical_covariance "
            "(model_version, player_uid_a, player_uid_b, relationship, empirical_covariance) VALUES (?, ?, ?, ?, ?)",
            [mc_mv, uid_a, uid_b, relationship, cov],
        )
    return mc_mv


def test_report_covariance_dilution_returns_none_with_no_teammate_pairs(con):
    mc_mv = _seed_covariance_pairs(con, [("a", "b", 0.5, "opponent")], summary_var={"a": 4.0, "b": 4.0})
    assert mc.report_covariance_dilution(con, mc_mv) is None


def test_report_covariance_dilution_excludes_non_teammate_pairs(con):
    mc_mv = _seed_covariance_pairs(
        con,
        [("a", "b", 2.0, "teammate"), ("a", "c", 100.0, "opponent"), ("b", "c", 100.0, "independent")],
        summary_var={"a": 4.0, "b": 4.0, "c": 4.0},
    )
    result = mc.report_covariance_dilution(con, mc_mv)
    assert result["n_pairs"] == 1
    assert result["mean"] == pytest.approx(2.0 / 4.0)  # cov/sqrt(4*4)


def test_report_covariance_dilution_reports_distribution_not_just_mean(con):
    # 4 teammate pairs at var=4.0 each (sqrt(var_a*var_b)=4.0) -> correlations 0.05/0.10/0.20/0.40
    pairs = [("a", "b", 0.2, "teammate"), ("a", "c", 0.4, "teammate"), ("a", "d", 0.8, "teammate"), ("a", "e", 1.6, "teammate")]
    mc_mv = _seed_covariance_pairs(con, pairs, summary_var={u: 4.0 for u in "abcde"})

    result = mc.report_covariance_dilution(con, mc_mv)
    assert result["n_pairs"] == 4
    expected = [0.05, 0.10, 0.20, 0.40]
    assert result["mean"] == pytest.approx(sum(expected) / len(expected))
    assert result["min"] == pytest.approx(min(expected))
    assert result["max"] == pytest.approx(max(expected))
    assert result["min"] <= result["p25"] <= result["median"] <= result["p75"] <= result["max"]
    # the distribution field is genuinely non-empty (not degenerate to a single repeated value)
    assert result["min"] != result["max"]


def test_report_covariance_dilution_skips_zero_variance_pairs(con):
    mc_mv = _seed_covariance_pairs(con, [("a", "b", 0.5, "teammate")], summary_var={"a": 0.0, "b": 4.0})
    assert mc.report_covariance_dilution(con, mc_mv) is None
