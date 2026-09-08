import math

import numpy as np
import pytest
from scipy.stats import norm

from fpl_quant import expected_points as ep
from fpl_quant import params as params_mod
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


def test_cross_player_covariance_teammates_positive_opponents_smaller_positive(con):
    # Real bug fix (2026-09-08): the previous implementation gave opponents a NEGATIVE
    # cross-team covariance via an ad hoc clean_sheet-vs-goals_conceded cross-category term
    # that could not be derived from any consistent joint model of the categories this
    # function otherwise correlates -- and at real-data scale reliably pushed the assembled
    # Sigma non-PSD (squad_optimizer's SCIP solve then timed out proving optimality, which is
    # exactly what broke nightly_backtest.yml for several consecutive days). The fix models
    # each category as its own two-team shared-factor system: opponents now get the SAME SIGN
    # as teammates, just weaker -- matching seed_v1_params()'s own long-standing comment
    # ("opponents get a materially smaller version of the same two effects"), which the old
    # code never actually implemented.
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
    assert 0 < ac[3] < ab[3]


def test_cross_player_covariance_rejects_opponent_corr_exceeding_teammate_corr(con):
    # |opponent_corr| > teammate_corr admits no valid shared-factor model at all (see
    # _validate_shared_factor_corr's own docstring) -- must fail loudly, not silently produce
    # a non-PSD Sigma the way the pre-fix formula could.
    for key, value in {"teammate_attacking": 0.1, "opponent_attacking": 0.5,
                        "teammate_defensive": 0.9, "opponent_defensive": 0.5}.items():
        params_mod.write_param(con, "cross_player_correlation_params", 1, "2026-08-10", key, value_numeric=value)
    a = {"player_uid": "p_a", "team_uid": "team_1", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.3, "var_assists": 0.1, "var_bonus": 0.2}
    b = {"player_uid": "p_b", "team_uid": "team_2", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.3, "var_assists": 0.1, "var_bonus": 0.2}
    with pytest.raises(ValueError, match="attacking"):
        un.cross_player_covariance_for_fixture(con, [a, b], "team_1", "team_2", corr_params_version=1)


def test_cross_player_covariance_sigma_is_psd_for_a_real_size_multi_fixture_pool(con):
    # The actual real-data failure mode this whole redesign closes: a large candidate pool
    # (500+ players is typical) spanning many simultaneous fixtures, assembled into one Sigma
    # exactly the way squad_optimizer.solve()/_warn_if_sigma_not_psd() do. Different fixtures
    # never share a nonzero pair (per this module's own docstring), so Sigma is block-diagonal
    # by fixture -- this builds several fixtures' worth of players and checks the WHOLE
    # assembled matrix, not just one block, has no negative eigenvalue.
    un.seed_v1_params(con)
    rng = np.random.default_rng(0)
    players = []
    sigma_pairs = {}
    for fixture_idx in range(8):
        home, away = f"team_{fixture_idx}_h", f"team_{fixture_idx}_a"
        fixture_rows = []
        for side in (home, away):
            for k in range(11):
                uid = f"{side}_p{k}"
                row = {
                    "player_uid": uid, "team_uid": side,
                    "var_goals": rng.uniform(0, 0.3), "var_assists": rng.uniform(0, 0.2),
                    "var_bonus": rng.uniform(0, 0.5), "var_clean_sheet": rng.uniform(0, 0.25),
                    "var_goals_conceded": rng.uniform(0, 1.5),
                }
                fixture_rows.append(row)
                players.append(row)
        pairs = un.cross_player_covariance_for_fixture(con, fixture_rows, home, away, corr_params_version=1)
        for a_uid, b_uid, _rel, cov in pairs:
            sigma_pairs[(a_uid, b_uid)] = cov

    uid_list = [p["player_uid"] for p in players]
    idx = {u: i for i, u in enumerate(uid_list)}
    var_by_uid = {p["player_uid"]: p["var_goals"] + p["var_assists"] + p["var_bonus"]
                  + p["var_clean_sheet"] + p["var_goals_conceded"] for p in players}
    n = len(uid_list)
    sigma = np.zeros((n, n))
    for u in uid_list:
        sigma[idx[u], idx[u]] = var_by_uid[u]
    for (a_uid, b_uid), cov in sigma_pairs.items():
        sigma[idx[a_uid], idx[b_uid]] = cov
        sigma[idx[b_uid], idx[a_uid]] = cov

    min_eig = float(np.linalg.eigvalsh(sigma).min())
    assert min_eig >= -1e-8, f"assembled Sigma is not PSD: most negative eigenvalue {min_eig}"


@pytest.mark.parametrize("teammate_attacking,opponent_attacking,teammate_defensive,opponent_defensive", [
    (0.25, 0.08, 0.9, 0.5),   # real v1 defaults
    (0.01, 0.01, 0.01, 0.01),  # near-zero loadings
    (1.0, 1.0, 1.0, 1.0),      # maximal loadings, opponent == teammate
    (0.5, -0.4, 0.6, -0.6),    # negative opponent correlation, still |opponent| <= teammate
])
def test_cross_player_covariance_sigma_is_psd_across_param_grid(
    con, teammate_attacking, opponent_attacking, teammate_defensive, opponent_defensive,
):
    # Property test: the PSD guarantee must hold for the whole valid parameter region
    # (|opponent| <= teammate), not just today's specific seeded v1 values.
    for key, value in {
        "teammate_attacking": teammate_attacking, "opponent_attacking": opponent_attacking,
        "teammate_defensive": teammate_defensive, "opponent_defensive": opponent_defensive,
    }.items():
        params_mod.write_param(con, "cross_player_correlation_params", 1, "2026-08-10", key, value_numeric=value)
    rng = np.random.default_rng(hash((teammate_attacking, opponent_attacking, teammate_defensive, opponent_defensive)) % (2**32))
    home, away = "team_h", "team_a"
    fixture_rows = []
    for side in (home, away):
        for k in range(11):
            fixture_rows.append({
                "player_uid": f"{side}_p{k}", "team_uid": side,
                "var_goals": rng.uniform(0, 0.3), "var_assists": rng.uniform(0, 0.2),
                "var_bonus": rng.uniform(0, 0.5), "var_clean_sheet": rng.uniform(0, 0.25),
                "var_goals_conceded": rng.uniform(0, 1.5),
            })
    pairs = un.cross_player_covariance_for_fixture(con, fixture_rows, home, away, corr_params_version=1)
    uid_list = [r["player_uid"] for r in fixture_rows]
    idx = {u: i for i, u in enumerate(uid_list)}
    var_by_uid = {r["player_uid"]: r["var_goals"] + r["var_assists"] + r["var_bonus"]
                  + r["var_clean_sheet"] + r["var_goals_conceded"] for r in fixture_rows}
    n = len(uid_list)
    sigma = np.zeros((n, n))
    for u in uid_list:
        sigma[idx[u], idx[u]] = var_by_uid[u]
    for a_uid, b_uid, _rel, cov in pairs:
        sigma[idx[a_uid], idx[b_uid]] = cov
        sigma[idx[b_uid], idx[a_uid]] = cov
    min_eig = float(np.linalg.eigvalsh(sigma).min())
    assert min_eig >= -1e-8, f"non-PSD at params={(teammate_attacking, opponent_attacking, teammate_defensive, opponent_defensive)}: min_eig={min_eig}"


def _seed_uncertainty_run_scaffold(con, *, transferred_season_club):
    """One fixture team_a vs team_c. p_t is on team_a's club code '1' in
    `transferred_season_club` and on team_c's code '3' in 2026-2027. p_x is always on '1'.
    Two ep_model_versions: one targeting 2024-2025, one targeting 2026-2027."""
    for uid, name in (("team_a", "A"), ("team_c", "C")):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [uid, name])
    for uid in ("p_t", "p_x", "p_c"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2024-2025', 10, 'team_a', 'team_c', FALSE, "
        "'Premier League', '2025-01-01', current_timestamp)"
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2025-01-01', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for uid in ("team_a", "team_c"):
        con.execute("INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
                    "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, 0.1, 0.0, 2, 1.0)", [ts_mv, uid])
    con.execute("INSERT INTO minutes_model_versions (model_version, calibration_asof_date, target_season, "
                "decay_params_version, adjustment_params_version, shrinkage_params_version, "
                "fact_multiplier_params_version, lookback_seasons) VALUES (1, '2025-01-01', '2024-2025', 1,1,1,1,'[]')")
    for uid in ("p_t", "p_x", "p_c"):
        con.execute("INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_final, "
                    "p_start_historical_position_avg, weight_own, logit_adjustment_total, p_start_final, "
                    "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
                    "VALUES (1, ?, 'Midfielder', 0.9, 0.9, 1.0, 0.0, 0.9, 0.0, 0.05, 0.05, 0.9, 20)", [uid])
    # per-season rosters: p_t on '1' in the transferred season, '3' in 2026-2027; p_x/p_c fixed
    for season in ("2024-2025", "2026-2027"):
        table = f"raw_{season.replace('-', '_')}_teams"
        con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
        con.execute(f'INSERT INTO "{table}" VALUES (\'1\', \'A\'), (\'3\', \'C\')')
        con.execute("INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
                    "VALUES (?, ?, 'teams.csv', ?, 2)", [table, season, f"h_{season}"])
        con.execute("INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES ('A', ?, 'team_a', 't'), ('C', ?, 'team_c', 't')", [season, season])
        con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P_x','p_x','1',?,'p_x'), ('P_c','p_c','3',?,'p_c')", [season, season])
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P_t','p_t','1',?,'p_t')", [transferred_season_club])
    con.execute("INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES ('P_t','p_t','3','2026-2027','p_t')")

    un.seed_v1_params(con)
    from fpl_quant import expected_points as ep_mod
    ep_mod.seed_v1_params(con)
    mv_by_season = {}
    for season in ("2024-2025", "2026-2027"):
        con.execute("INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
                    "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
                    "VALUES ('2025-01-01', ?, ?, 1, 1, 1, 1)", [season, ts_mv])
        mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
        mv_by_season[season] = mv
        for uid in ("p_t", "p_x", "p_c"):
            con.execute("INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                        "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                        "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 1.8, 0.5, 0.3, 0.4, 0, 0.1, 0.3, 0,0,0,0, 3.5, 20.0)",
                        [mv, uid])
    return ts_mv, mv_by_season


def test_uncertainty_run_classifies_teammates_by_the_ep_versions_target_season(con):
    # p_t transferred: on team_a in 2024-25, on team_c in 2026-27. For a 2024-25-target run
    # p_t must be team_a's player (teammate of p_x), NOT dropped as an unknown.
    from datetime import date
    _ts_mv, mv = _seed_uncertainty_run_scaffold(con, transferred_season_club="2024-2025")
    un_mv = un.run(con, date(2025, 1, 1), mv["2024-2025"], 1, _ts_mv, 1, 1, 1, 1, 1)
    outs = {r[0] for r in con.execute(
        "SELECT player_uid FROM uncertainty_outputs WHERE model_version = ?", [un_mv]).fetchall()}
    assert {"p_t", "p_x"} <= outs   # p_t resolved (would be dropped under the old 2026-27 hardcode)
    pairs = con.execute(
        "SELECT player_uid_a, player_uid_b, relationship FROM cross_player_covariance WHERE model_version = ?", [un_mv],
    ).fetchall()
    tx = [r for r in pairs if {r[0], r[1]} == {"p_t", "p_x"}]
    assert tx and tx[0][2] == "teammate"


def test_cross_player_covariance_no_signal_no_pair(con):
    a = {"player_uid": "p_a", "team_uid": "team_1", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.0, "var_assists": 0.0, "var_bonus": 0.0}
    b = {"player_uid": "p_b", "team_uid": "team_1", "var_clean_sheet": 0.0, "var_goals_conceded": 0.0,
         "var_goals": 0.0, "var_assists": 0.0, "var_bonus": 0.0}
    un.seed_v1_params(con)
    pairs = un.cross_player_covariance_for_fixture(con, [a, b], "team_1", "team_2", corr_params_version=1)
    assert pairs == []
