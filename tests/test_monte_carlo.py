from datetime import date

import numpy as np
import pytest

from fpl_quant import expected_points as ep
from fpl_quant import monte_carlo as mc
from fpl_quant import params as params_mod


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
# sample_binomial_vec -- multinomial goal-allocation building block (Phase B hardening)
# ============================================================

def test_sample_binomial_vec_matches_binomial_moments():
    n = np.full(300_000, 4)
    p = np.full(300_000, 0.3)
    rng = np.random.default_rng(1)
    u = rng.random(300_000)
    counts = mc.sample_binomial_vec(n, p, u)
    assert counts.mean() == pytest.approx(4 * 0.3, abs=0.02)
    assert counts.var() == pytest.approx(4 * 0.3 * 0.7, abs=0.02)


def test_sample_binomial_vec_never_exceeds_n():
    rng = np.random.default_rng(2)
    n = rng.integers(0, 6, size=50_000)
    p = rng.random(50_000)
    u = rng.random(50_000)
    counts = mc.sample_binomial_vec(n, p, u)
    assert np.all(counts <= n)
    assert np.all(counts >= 0)


def test_sample_binomial_vec_handles_zero_n():
    n = np.zeros(100, dtype=int)
    p = np.full(100, 0.5)
    u = np.full(100, 0.5)
    counts = mc.sample_binomial_vec(n, p, u)
    assert np.all(counts == 0)


def test_sample_binomial_vec_zero_p_never_scores():
    n = np.full(1000, 3)
    p = np.zeros(1000)
    u = np.linspace(0, 1, 1000, endpoint=False)
    counts = mc.sample_binomial_vec(n, p, u)
    assert np.all(counts == 0)


def test_sample_binomial_vec_p_one_always_gets_all_of_n():
    n = np.array([0, 1, 2, 5])
    p = np.full(4, 1.0)
    u = np.full(4, 0.5)
    counts = mc.sample_binomial_vec(n, p, u)
    np.testing.assert_array_equal(counts, n)


def test_sequential_binomial_decomposition_reproduces_multinomial_allocation():
    """This is exactly the construction simulate_fixture's goal-allocation section uses: draw
    player 1's share via Binomial(N, p1), player 2's share via Binomial(N-X1, p2/(1-p1)), and
    whatever's left is implicitly the untracked "rest of the team" -- the standard sequential-
    binomial decomposition of a multinomial. Verifies both the hard constraint (X1+X2 <= N
    always) and that each player's long-run share converges to their multinomial proportion."""
    rng = np.random.default_rng(7)
    n_draws = 200_000
    N = np.full(n_draws, 5)
    p1_target, p2_target = 0.4, 0.25  # remainder (0.35) is the untracked "rest of team"

    x1 = mc.sample_binomial_vec(N, np.full(n_draws, p1_target), rng.random(n_draws))
    remaining_n = N - x1
    p2_conditional = np.full(n_draws, p2_target / (1 - p1_target))
    x2 = mc.sample_binomial_vec(remaining_n, p2_conditional, rng.random(n_draws))

    assert np.all(x1 + x2 <= N)
    assert x1.mean() == pytest.approx(5 * p1_target, abs=0.03)
    assert x2.mean() == pytest.approx(5 * p2_target, abs=0.03)


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
# z_fixture_correlation_distribution -- Phase B rate-heterogeneity disclosure
# ============================================================
# Full-chain (teammate/opponent pair + non-trivial correlation spread) coverage lives in
# test_reporting.py's test_z_fixture_correlation_dilution_* -- it already has
# _seed_full_squad_scenario() building the entire M1-M6 FK chain this function's query joins
# through (monte_carlo_run_versions -> squad_optimizer_runs/ep_model_versions/etc), so those
# tests extend that fixture rather than duplicating the ~60-line chain-builder here for no-DB-
# state coverage that doesn't need it.

def test_z_fixture_correlation_distribution_none_when_model_version_has_no_rows(con):
    assert mc.z_fixture_correlation_distribution(con, 999) is None


# ============================================================
# M3 <-> M6 fixture-strength parity (#131/#134 propagation gap).
#
# M3's analytic engine scales e_goals/e_assists by _fixture_attack_multiplier and a
# defender's DefCon / a keeper's saves by _fixture_defensive_multiplier; simulate_fixture()
# historically left all four at the flat season rate. For a strong favourite that compressed
# a premium attacker's simulated mean ~1pt below M3's analytic ep_total, and left a besieged
# defender's / keeper's DefCon / saves below M3's. These tests reproduce that pre-fix
# direction (fixture_params_version=None == the old flat-rate behaviour) and pin the fix
# (default == M3 parity). They FAIL on pre-fix monte_carlo.py (which never scales).
# ============================================================

def _seed_raw_teams_csv(con, season, rows):
    table = f"raw_{season.replace('-', '_')}_teams"
    con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
    for code, name in rows:
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?)', [code, name])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', ?, ?)",
        [table, season, f"hash_{table}", len(rows)],
    )


def _seed_asymmetric_fixture(con):
    """A single fixture 'm1' in which 'fav' (home) is a big favourite over 'dog' (away),
    with a full ~11-a-side roster so each side's per-player attacking rates roughly sum to
    that side's team lambda (the realistic regime the goal-allocation split assumes). Squad =
    {att (fav Forward), dfn (dog Defender), gkp (dog Goalkeeper)}. Returns the model versions
    plus the squad uid set.
    """
    SEASON, GW = "2026-2027", 1
    ep.seed_v1_params(con)
    params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)

    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('fav', 'Favourite'), ('dog', 'Underdog')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('hist_h', 'HistH'), ('hist_a', 'HistA')")
    _seed_raw_teams_csv(con, SEASON, [("1", "Favourite"), ("2", "Underdog")])
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('Favourite', ?, 'fav', 't')", [SEASON])
    con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('Underdog', ?, 'dog', 't')", [SEASON])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, ?, 'fav', 'dog', FALSE, "
        "'Premier League', '2026-08-21', current_timestamp)", [SEASON, GW],
    )

    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.25, 1, 1, 'fav')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    # final_defence: HIGHER = better defence. fav attacks well and defends well; dog is leaky
    # with a poor attack -> fav a clear favourite at home.
    for tu, atk, dfc in (("fav", 0.45, 0.35), ("dog", -0.35, -0.45)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)", [ts_mv, tu, atk, dfc],
        )

    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', ?, 1, 1, 1, 1, '[]')", [SEASON],
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]

    # roster: att + 10 non-squad fav mids; dfn + gkp + 9 non-squad dog mids. Nailed starters.
    fav_players = [("att", "Forward", 0.70, 0.30)] + [(f"favm{i}", "Midfielder", 0.13, 0.10) for i in range(10)]
    dog_players = [("dfn", "Defender", 0.02, 0.03), ("gkp", "Goalkeeper", 0.0, 0.0)] + \
                  [(f"dogm{i}", "Midfielder", 0.11, 0.08) for i in range(9)]
    for code, players in (("1", fav_players), ("2", dog_players)):
        for uid, pos, xg90, xa90 in players:
            con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, pos])
            con.execute(
                "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
                "VALUES (?, ?, ?, ?, ?)", [uid, uid, code, SEASON, uid],
            )
            # a season-total row (minutes + expected_goals branch of _player_rate_pool)
            con.execute(
                "INSERT INTO fact_player_season_stats (player_uid, season, gw, expected_goals, expected_assists, "
                "saves_per_90, minutes, _ingested_at) VALUES (?, '2025-2026', 38, ?, ?, ?, 3000, current_timestamp)",
                [uid, xg90 / 90.0 * 3000, xa90 / 90.0 * 3000, 3.4 if pos == "Goalkeeper" else None],
            )
            con.execute(
                "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
                "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
                "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
                "VALUES (?, ?, ?, 0.85, 0.85, 1.0, 0.0, 0.85, 0.0, 0.05, 0.10, 0.85, 30)",
                [mm_mv, uid, pos],
            )

    # dfn's defensive-action history: ~11 CBIT / 90 (comfortably threshold-sensitive once the
    # besieged-defender multiplier lifts it). 30 full matches -> per-90 == per-match, and dfn
    # is the only Defender so the position-average shrinkage anchor == dfn's own rate.
    for i in range(30):
        mid = f"hist_dfn_{i}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2025-2026', ?, 'hist_h', 'hist_a', TRUE, "
            "'Premier League', '2026-01-01', current_timestamp)", [mid, i + 1],
        )
        con.execute(
            "INSERT INTO fact_player_match_stats (player_uid, match_id, season, start_min, finish_min, "
            "minutes_played, tackles, clearances, interceptions, blocks, recoveries, _ingested_at) "
            "VALUES ('dfn', ?, '2025-2026', 0, 90, 90, 11, 0, 0, 0, 0, current_timestamp)", [mid],
        )

    ep_mv = ep.run(
        con, calibration_asof_date=date(2026, 8, 10), target_season=SEASON, target_gameweek=GW,
        ts_model_version=ts_mv, mm_model_version=mm_mv, scoring_params_version=1, bps_params_version=1,
        tau_params_version=1, lookback_seasons=("2026-2027", "2025-2026"),
    )
    return ts_mv, mm_mv, ep_mv, {"att", "dfn", "gkp"}


_DEFAULT = object()


def _simulate(con, ts_mv, mm_mv, ep_mv, squad, *, fixture_params_version=_DEFAULT, n_pairs=6000, seed=99):
    """fixture_params_version=_DEFAULT calls simulate_fixture() with NO override -- i.e. its
    production default path. That is deliberate: it makes the post-fix assertions below test
    the real default behaviour, so they fail as plain assertions (not TypeErrors) when run
    against a monte_carlo.py that never applies the scaling. Pass an explicit value (e.g.
    None, the flat-rate opt-out) only where the override itself is under test."""
    tau_val, _ = params_mod.resolve_param(con, "bps_dispersion_params", "tau", 1)
    mean_minutes = ep._mean_minutes_by_bucket(con)
    rng = np.random.default_rng(seed)
    kwargs = {} if fixture_params_version is _DEFAULT else {"fixture_params_version": fixture_params_version}
    return mc.simulate_fixture(
        con, "m1", "fav", "dog", "2026-2027", ["2026-2027", "2025-2026"], squad,
        ep_mv, mm_mv, ts_mv, 1, tau_val, 0.0, mean_minutes, rng, n_pairs, **kwargs,
    )


def _m3_components(con, ep_mv):
    rows = con.execute(
        "SELECT player_uid, ep_goals, ep_assists, ep_defcon, ep_saves, ep_total FROM ep_outputs WHERE model_version = ?",
        [ep_mv],
    ).fetchall()
    return {r[0]: dict(ep_goals=r[1], ep_assists=r[2], ep_defcon=r[3], ep_saves=r[4], ep_total=r[5]) for r in rows}


def test_m6_fixture_strength_multipliers_are_material_in_this_scenario(con):
    ts_mv, _mm, _ep, _sq = _seed_asymmetric_fixture(con)
    atk = ep._fixture_attack_multiplier(con, "fav", "m1", "2026-2027", ts_mv, 1)
    sv = ep._fixture_defensive_multiplier(con, "dog", "m1", ts_mv, 1, "save_sensitivity")
    dc = ep._fixture_defensive_multiplier(con, "dog", "m1", ts_mv, 1, "defcon_sensitivity")
    assert atk > 1.25          # fav at home vs a leaky defence
    assert sv > 1.25           # dog keeper faces a barrage
    assert 1.0 < dc < sv       # damped defcon channel, same direction


def test_m6_premium_attacker_mean_matches_m3_after_fixture_scaling(con):
    ts_mv, mm_mv, ep_mv, squad = _seed_asymmetric_fixture(con)
    m3 = _m3_components(con, ep_mv)
    goal_pts, assist_pts = 4.0, 3.0
    m3_attack = m3["att"]["ep_goals"] + m3["att"]["ep_assists"]

    # default (production) path: on pre-fix monte_carlo.py this is the flat season rate, so
    # these assertions fail as plain assertions there.
    post = _simulate(con, ts_mv, mm_mv, ep_mv, squad)
    post_attack = post["att"]["goals"].mean() * goal_pts + post["att"]["assists"].mean() * assist_pts
    post_pts = mc._assemble_points(con, "Forward", post["att"], 1)["total_points"].mean()
    assert post_attack >= m3_attack * 0.85                    # M6 tracks M3's analytic attacking EP
    assert post_pts >= m3["att"]["ep_total"] - 0.75           # ... and at the total-EP level

    # flat-rate opt-out reproduces the documented pre-fix compression (~1pt on this favourite)
    pre = _simulate(con, ts_mv, mm_mv, ep_mv, squad, fixture_params_version=None)
    pre_attack = pre["att"]["goals"].mean() * goal_pts + pre["att"]["assists"].mean() * assist_pts
    pre_pts = mc._assemble_points(con, "Forward", pre["att"], 1)["total_points"].mean()
    assert pre_attack < m3_attack * 0.82
    assert pre_pts < m3["att"]["ep_total"] - 0.75
    assert post_attack > pre_attack + 0.3


def test_m6_besieged_defender_defcon_and_keeper_saves_match_m3_after_fixture_scaling(con):
    ts_mv, mm_mv, ep_mv, squad = _seed_asymmetric_fixture(con)
    m3 = _m3_components(con, ep_mv)
    defcon_pts = ep._sm(con, "defcon_points", 1)
    saves_per_pt, _ = params_mod.resolve_param(con, "base_scoring_matrix", "saves_per_point", 1)
    m3_defcon = m3["dfn"]["ep_defcon"]
    m3_saves_count = m3["gkp"]["ep_saves"] * saves_per_pt      # ep_saves == e_saves / saves_per_point

    # default (production) path -- flat rate on pre-fix monte_carlo.py.
    post = _simulate(con, ts_mv, mm_mv, ep_mv, squad)
    post_defcon = post["dfn"]["defcon_hit"].mean() * defcon_pts
    post_saves = post["gkp"]["saves_count"].mean()
    assert m3_defcon > 0.4 and m3_saves_count > 1.0           # the scenario actually exercises both
    # M3 scales the DefCon rate by _fixture_defensive_multiplier("defcon_sensitivity") and the
    # saves rate by ("save_sensitivity"); post-fix M6 tracks M3 on both.
    assert post_defcon >= m3_defcon * 0.80
    assert post_saves == pytest.approx(m3_saves_count, rel=0.12)

    # flat-rate opt-out: both sit below M3, the pre-fix direction.
    pre = _simulate(con, ts_mv, mm_mv, ep_mv, squad, fixture_params_version=None)
    pre_defcon = pre["dfn"]["defcon_hit"].mean() * defcon_pts
    pre_saves = pre["gkp"]["saves_count"].mean()
    assert pre_defcon < m3_defcon * 0.85
    assert pre_saves < m3_saves_count * 0.85
    assert post_defcon > pre_defcon + 0.05
    assert post_saves > pre_saves + 0.3


def test_m6_fixture_scaling_preserves_core_invariants(con):
    ts_mv, mm_mv, ep_mv, squad = _seed_asymmetric_fixture(con)

    a = _simulate(con, ts_mv, mm_mv, ep_mv, squad, seed=7)
    b = _simulate(con, ts_mv, mm_mv, ep_mv, squad, seed=7)
    for uid in squad:
        for cat in a[uid]:
            np.testing.assert_array_equal(a[uid][cat], b[uid][cat])  # deterministic seed -> identical output

    for uid, position in (("att", "Forward"), ("dfn", "Defender"), ("gkp", "Goalkeeper")):
        pts = mc._assemble_points(con, position, a[uid], 1)
        for cat, arr in pts.items():
            if cat == "minutes_state":
                continue
            v = float(np.var(arr))
            assert np.isfinite(v) and v >= 0.0

        total = pts["total_points"]
        q = np.quantile(total, [0.05, 0.25, 0.5, 0.75, 0.95])
        assert np.all(np.diff(q) >= 0.0)  # monotone quantiles

        state = a[uid]["state"]
        p0 = np.mean(state == "0")
        p1 = np.mean(state == "1_59")
        p2 = np.mean(state == "60plus")
        assert p0 + p1 + p2 == pytest.approx(1.0)
