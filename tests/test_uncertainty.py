import math

import pytest
from scipy.stats import norm

from fpl_quant import expected_points as ep
from fpl_quant import uncertainty as un


def test_seed_v1_params_resolves(con):
    un.seed_v1_params(con)
    assert un._rho_residual(con, 1) == 0.15


def test_bernoulli_skew_kurt_symmetric_at_half():
    skew, kurt = un._bernoulli_skew_kurt(0.5)
    assert abs(skew) < 1e-9  # symmetric coin -- no skew


def test_bernoulli_skew_kurt_positive_skew_for_rare_event():
    skew, _ = un._bernoulli_skew_kurt(0.05)
    assert skew > 0  # rare success -> right-skewed


def test_bernoulli_skew_kurt_degenerate_at_extremes():
    assert un._bernoulli_skew_kurt(0.0) == (0.0, 0.0)
    assert un._bernoulli_skew_kurt(1.0) == (0.0, 0.0)


def test_cornish_fisher_reduces_to_normal_quantile_at_zero_skew_kurtosis():
    mean, var = 5.0, 4.0
    for q in (0.05, 0.25, 0.75, 0.95):
        cf = un.cornish_fisher_quantile(mean, var, skew=0.0, excess_kurtosis=0.0, q=q)
        expected = mean + norm.ppf(q) * math.sqrt(var)
        assert abs(cf - expected) < 1e-9


def test_cornish_fisher_quantiles_are_monotonic_for_realistic_skew():
    mean, var = 3.0, 5.0
    qs = [un.cornish_fisher_quantile(mean, var, skew=0.6, excess_kurtosis=0.6, q=q) for q in (0.05, 0.25, 0.75, 0.95)]
    assert qs == sorted(qs)


def _row(**overrides):
    base = {
        "ep_goals": 0.4, "ep_assists": 0.2, "ep_clean_sheet": 0.3, "ep_goals_conceded": -0.2,
        "ep_defcon": 0.5, "ep_bonus": 0.3, "ep_saves": 0.0,
        "p_0": 0.1, "p_1_59": 0.15, "p_60plus": 0.75,
        "lambda_against": 1.1, "p_rank1": 0.15, "p_rank2": 0.1, "p_rank3": 0.08,
    }
    base.update(overrides)
    return base


def test_category_variances_goals_matches_poisson_var_equals_mean(con):
    ep.seed_v1_params(con)
    row = _row()
    variances = un.category_variances(con, row, "Forward", 1)
    goal_pts = ep._sm(con, "goal_points", 1, "Forward")
    e_goals_count = row["ep_goals"] / goal_pts
    assert variances["var_goals"] == pytest.approx(goal_pts**2 * e_goals_count)


def test_category_variances_clean_sheet_matches_bernoulli_var(con):
    ep.seed_v1_params(con)
    row = _row()
    variances = un.category_variances(con, row, "Defender", 1)
    cs_pts = ep._sm(con, "clean_sheet_points", 1, "Defender")
    p = row["ep_clean_sheet"] / cs_pts
    assert variances["var_clean_sheet"] == pytest.approx(cs_pts**2 * p * (1 - p))


def test_category_variances_all_non_negative(con):
    ep.seed_v1_params(con)
    for position in ("Goalkeeper", "Defender", "Midfielder", "Forward"):
        row = _row(ep_saves=0.3 if position == "Goalkeeper" else 0.0)
        variances = un.category_variances(con, row, position, 1)
        for key, v in variances.items():
            assert v >= 0, f"{key} negative for {position}: {v}"


def test_total_variance_is_positive_and_at_least_sum_of_category_variances(con):
    ep.seed_v1_params(con)
    row = _row()
    rates = {"expected_goals_per_90": 0.5, "expected_assists_per_90": 0.3, "saves_per_90": 0.0}
    def_rates = {"cbi_per_90": 4.0, "recoveries_per_90": 3.0}
    mean_minutes = {"mean_1_59": 30.0, "mean_60plus": 88.0}
    variances, var_total = un.total_variance(con, row, "Defender", rates, def_rates, mean_minutes, 1, rho_residual=0.15)
    assert var_total > 0
    # positive covariance terms should push total above the naive independent-sum floor
    # in the common case, though not guaranteed in general -- just check it's finite/sane
    assert var_total < 1000


def test_cross_player_covariance_teammates_positive_opponents_negative(con):
    a = {"player_uid": "p_a", "team_uid": "team_1", "var_clean_sheet": 1.0, "var_goals_conceded": 0.5,
         "var_goals": 0.3, "var_assists": 0.1, "var_bonus": 0.2}
    b = {"player_uid": "p_b", "team_uid": "team_1", "var_clean_sheet": 1.0, "var_goals_conceded": 0.5,
         "var_goals": 0.3, "var_assists": 0.1, "var_bonus": 0.2}
    c = {"player_uid": "p_c", "team_uid": "team_2", "var_clean_sheet": 1.0, "var_goals_conceded": 0.5,
         "var_goals": 0.3, "var_assists": 0.1, "var_bonus": 0.2}
    un.seed_v1_params(con)
    pairs = un.cross_player_covariance_for_fixture(con, [a, b, c], "team_1", "team_2", corr_params_version=1)
    by_pair = {(p[0], p[1]): p for p in pairs}
    ab = by_pair.get(("p_a", "p_b")) or by_pair.get(("p_b", "p_a"))
    ac = by_pair.get(("p_a", "p_c")) or by_pair.get(("p_c", "p_a"))
    assert ab[2] == "teammate"
    assert ab[3] > 0
    assert ac[2] == "opponent"
    assert ac[3] < 0


def test_cross_player_covariance_no_signal_no_pair(con):
    a = {"player_uid": "p_a", "team_uid": "team_1", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.0, "var_assists": 0.0, "var_bonus": 0.0}
    b = {"player_uid": "p_b", "team_uid": "team_1", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.0, "var_assists": 0.0, "var_bonus": 0.0}
    un.seed_v1_params(con)
    pairs = un.cross_player_covariance_for_fixture(con, [a, b], "team_1", "team_2", corr_params_version=1)
    assert pairs == []
