"""M9's per-player explain adapters are what Gap 5's in-app "Explain this" sheet renders --
run_transfer_planner_for_real_squad.py calls expected_points.explain_player_ep() /
uncertainty.explain_player_risk() for the captain pick and the top transfer(s) and drops the
result into real_squad_<id>.json's `explain` block, and index.html's epBreakdownBlock() /
riskRangeBlock() read specific keys off it. These are contract tests for that shape (nothing
covered explain_player_ep/explain_player_risk directly before) so a schema rename can't
silently blank the sheet.
"""

import pytest

from fpl_quant import expected_points as ep_mod
from fpl_quant import uncertainty as un_mod


def _seed_one_player_fixture(con, *, uid="player_x", ep_categories=None, q=(1.2, 3.1, 7.4, 13.6)):
    ep_categories = ep_categories or dict(
        ep_appearance=0.9, ep_goals=3.0, ep_assists=0.4, ep_clean_sheet=0.3, ep_goals_conceded=-0.05,
        ep_defcon=0.0, ep_bonus=0.6, ep_saves=0.0, ep_penalty_save=0.0, ep_cards=-0.02, ep_own_goal=0.0,
    )
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('t_home','Home'),('t_away','Away') ON CONFLICT DO NOTHING")
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, 'X', 'Forward') ON CONFLICT DO NOTHING", [uid])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1','2026-2027',3,'t_home','t_away',FALSE,"
        "'Premier League','2026-08-24', current_timestamp)"
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 't_home')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_mv = con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10','2026-2027', ?, ?, 1, 1, 1) RETURNING model_version", [ts_mv, mm_mv],
    ).fetchone()[0]
    un_mv = con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1) "
        "RETURNING model_version", [ep_mv, mm_mv, ts_mv],
    ).fetchone()[0]
    total = sum(ep_categories.values())
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
        "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
        "ep_total, expected_bps) VALUES (?, ?, 'm1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 22.0)",
        [ep_mv, uid, ep_categories["ep_appearance"], ep_categories["ep_goals"], ep_categories["ep_assists"],
         ep_categories["ep_clean_sheet"], ep_categories["ep_goals_conceded"], ep_categories["ep_defcon"],
         ep_categories["ep_bonus"], ep_categories["ep_saves"], ep_categories["ep_penalty_save"],
         ep_categories["ep_cards"], ep_categories["ep_own_goal"], total],
    )
    q05, q25, q75, q95 = q
    con.execute(
        "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, var_goals, "
        "var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, var_total, skew, "
        "excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
        "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 12.0, 0.5, 0.3, ?, ?, ?, ?)",
        [un_mv, uid, q05, q25, q75, q95],
    )
    return ep_mv, un_mv


def test_explain_player_ep_shape_matches_the_frontend_contract(con):
    ep_mv, _ = _seed_one_player_fixture(con)
    out = ep_mod.explain_player_ep(con, ep_mv, "player_x")
    assert out is not None
    # epBreakdownBlock() iterates out["categories"] and prints out["total"].
    assert set(out) >= {"player_uid", "categories", "total"}
    assert set(out["categories"]) == {
        "appearance", "goals", "assists", "clean_sheet", "goals_conceded",
        "defcon", "bonus", "saves", "penalty_save", "cards", "own_goal",
    }
    assert out["categories"]["goals"] == pytest.approx(3.0)
    assert out["total"] == pytest.approx(sum(out["categories"].values()), abs=1e-6)


def test_explain_player_ep_none_for_a_blank_gameweek(con):
    ep_mv, _ = _seed_one_player_fixture(con)
    assert ep_mod.explain_player_ep(con, ep_mv, "player_not_in_this_run") is None


def test_explain_player_risk_shape_matches_the_frontend_contract(con):
    _, un_mv = _seed_one_player_fixture(con, q=(0.5, 2.0, 6.0, 11.0))
    out = un_mod.explain_player_risk(con, un_mv, "player_x")
    assert out is not None
    # riskRangeBlock() reads floor / ceiling / q25 / q75.
    assert set(out) >= {"floor", "q25", "q75", "ceiling"}
    assert (out["floor"], out["q25"], out["q75"], out["ceiling"]) == (0.5, 2.0, 6.0, 11.0)


def test_explain_player_risk_none_for_a_blank_gameweek(con):
    _, un_mv = _seed_one_player_fixture(con)
    assert un_mod.explain_player_risk(con, un_mv, "nobody") is None
