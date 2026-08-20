from datetime import date, datetime

import pytest

from fpl_quant import params
from fpl_quant import transfer_planner as tp


# ============================================================
# seed_v1_params
# ============================================================

def test_seed_v1_params_matches_verified_and_spec_pinned_values(con):
    tp.seed_v1_params(con)
    h, _ = params.resolve_param(con, "planning_horizon_params", "horizon_gameweeks", 1)
    assert h == 5
    cost, _ = params.resolve_param(con, "transfer_cost_params", "points_per_hit", 1)
    assert cost == 4  # verified via live web search this session, not assumed from convention
    kappa, _ = params.resolve_param(con, "tc_risk_aversion_params", "kappa_tc", 1)
    assert kappa == pytest.approx(0.15)


# ============================================================
# bootstrap_from_squad_optimizer_run + _write_manager_snapshot_as_optimizer_run
# ============================================================

def _seed_minimal_squad_optimizer_run(con, target_season="2026-2027", target_gameweek=1, players=None):
    """A real squad_optimizer_runs + squad_optimizer_selections pair -- the exact shape
    bootstrap_from_squad_optimizer_run() and the shadow mechanism both depend on."""
    players = players or [("p1", True, True, False), ("p2", True, False, True), ("p3", False, False, False)]
    for uid, *_ in players:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
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
    con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1)",
        [ep_mv, mm_mv, ts_mv],
    )
    un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]
    # nextval(), not the passed-in run_id -- squad_optimizer.run() always sequence-generates
    # real run_ids; a literal here would desync the sequence and collide with a later real
    # nextval() call (exactly the bug this replaced: _write_manager_snapshot_as_optimizer_run's
    # own nextval() call returned 1, which this fixture had already claimed by literal).
    run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (run_id, calibration_asof_date, target_season, target_gameweek, "
        "ep_model_version, uncertainty_model_version, lambda_params_version, lambda_value, "
        "guardrail_params_version, divergence_check_passed, solver_status, objective_value) "
        "VALUES (nextval('seq_squad_optimizer_run'), '2026-08-10', ?, ?, ?, ?, 1, 0.15, 1, TRUE, 'optimal', 10.0) "
        "RETURNING run_id",
        [target_season, target_gameweek, ep_mv, un_mv],
    ).fetchone()[0]
    for uid, in_xi, is_captain, is_vice in players:
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, TRUE, ?, ?, ?)", [run_id, uid, in_xi, is_captain, is_vice],
        )
    return run_id, ep_mv, un_mv


def test_bootstrap_from_squad_optimizer_run_creates_matching_state(con):
    run_id, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id)

    state = con.execute(
        "SELECT season, as_of_gameweek, free_transfers_available, chips_used_set1, chips_used_set2 "
        "FROM manager_state_versions WHERE state_version = ?", [state_version],
    ).fetchone()
    assert state == ("2026-2027", 1, 1, "[]", "[]")

    holdings = tp._read_holdings(con, state_version)
    assert {h["player_uid"] for h in holdings} == {"p1", "p2", "p3"}
    p1 = next(h for h in holdings if h["player_uid"] == "p1")
    assert p1["in_xi"] is True and p1["is_captain"] is True


def test_bootstrap_raises_on_unknown_run_id(con):
    with pytest.raises(ValueError):
        tp.bootstrap_from_squad_optimizer_run(con, 999)


def test_bootstrap_computes_bank_from_leftover_budget_when_all_prices_known(con):
    """Regression test for the transfer planner's disclosed no-bank-tracking gap: a manager
    whose actual starting squad came in under M5's BUDGET (100.0) should start with that
    leftover as real spendable bank, not silently discard it."""
    run_id, _, _ = _seed_minimal_squad_optimizer_run(con)
    for uid, price in (("p1", 10.0), ("p2", 20.0), ("p3", 65.0)):  # sums to 95.0, 5.0 leftover
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, '2026-2027', 1, ?, current_timestamp)", [uid, price],
        )
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id)
    bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [state_version]).fetchone()[0]
    assert bank == pytest.approx(5.0)


def test_bootstrap_defaults_bank_to_zero_when_a_held_players_price_is_unknown(con):
    """Conservative fallback: overstating bank would let evaluate_transfers() legalize a
    transfer the manager can't actually afford, so a squad with any unresolvable price starts
    at bank=0.0 rather than guessing."""
    run_id, _, _ = _seed_minimal_squad_optimizer_run(con)
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 10.0, current_timestamp)"
    )  # p2, p3 left unpriced
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id)
    bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [state_version]).fetchone()[0]
    assert bank == 0.0


def test_write_manager_snapshot_creates_a_real_flagged_run(con):
    run_id, ep_mv, un_mv = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id)

    snapshot_run_id = tp._write_manager_snapshot_as_optimizer_run(con, state_version, date(2026, 8, 17), "2026-2027", 2, ep_mv, un_mv)
    assert snapshot_run_id != run_id  # a genuine new row, real sequence-generated id

    is_snapshot, solver_status = con.execute(
        "SELECT is_manager_snapshot, solver_status FROM squad_optimizer_runs WHERE run_id = ?", [snapshot_run_id]
    ).fetchone()
    assert is_snapshot is True
    assert solver_status == "manager_snapshot"  # clearly distinguishable from a real solve

    players = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ?", [snapshot_run_id]
        ).fetchall()
    }
    assert players == {"p1", "p2", "p3"}

    # the real solve is untouched and still flagged as such
    real_is_snapshot = con.execute("SELECT is_manager_snapshot FROM squad_optimizer_runs WHERE run_id = ?", [run_id]).fetchone()[0]
    assert real_is_snapshot is False


# ============================================================
# evaluate_transfers -- pure logic against a synthetic horizon
# ============================================================

def _seed_ep_and_holdings_for_transfers(con, target_season="2026-2027"):
    """Two horizon gameweeks' worth of candidate pools, seeded directly via ep_outputs/
    uncertainty_outputs/fact_player_season_stats/player_alias so _horizon_ep_by_player() can
    read them through squad_optimizer.fetch_candidate_pool() unmodified."""
    teams = ["team_a", "team_b"]
    for t in teams:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [t, t])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, 2, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)", [target_season],
    )
    players = {
        "p_out": ("Midfielder", "1", 8.0, 5.0),   # position, team_code, price, ep
        "p_in_same_pos_cheaper": ("Midfielder", "1", 4.0, 9.0),   # legal: same position, cheaper, big upgrade
        "p_in_wrong_pos": ("Forward", "1", 4.0, 12.0),            # illegal: wrong position
        "p_in_too_expensive": ("Midfielder", "1", 9.0, 20.0),     # illegal: costs more than p_out
    }
    for uid, (position, team_code, price, ep_val) in players.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), team_code, target_season, uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp)", [uid, target_season, price],
        )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
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
    con.execute(
        "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
        "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1)",
        [ep_mv, mm_mv, ts_mv],
    )
    un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]
    for uid, (_position, _team_code, _price, ep_val) in players.items():
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, ep_val],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)", [un_mv, uid],
        )
    return {"p_out": {"player_uid": "p_out", "in_xi": True, "is_captain": False, "is_vice": False}}, {2: (ep_mv, un_mv)}


def test_evaluate_transfers_ranks_by_net_value_and_enforces_position_and_budget(con):
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]

    results = tp.evaluate_transfers(con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4)
    in_uids = {r["player_in"] for r in results}
    assert "p_in_same_pos_cheaper" in in_uids
    assert "p_in_wrong_pos" not in in_uids   # illegal position swap excluded
    assert "p_in_too_expensive" not in in_uids  # illegal budget-violating swap excluded

    top = results[0]
    assert top["player_in"] == "p_in_same_pos_cheaper"
    assert top["horizon_value_gain"] == pytest.approx(9.0 - 5.0)
    assert top["transfer_cost"] == 0.0  # within free allocation
    assert top["net_value"] == pytest.approx(4.0)


def test_evaluate_transfers_applies_points_hit_when_no_free_transfer(con):
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]

    results = tp.evaluate_transfers(con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=0, points_per_hit=4)
    top = results[0]
    assert top["transfer_cost"] == 4.0
    assert top["net_value"] == pytest.approx(4.0 - 4.0)


def test_evaluate_transfers_sufficient_bank_legalizes_an_otherwise_too_expensive_transfer(con):
    """Regression test for the transfer planner's real bug: with bank=0.0 (the old, only
    behavior), p_in_too_expensive (price 9.0 vs p_out's 8.0) is illegal -- no single outgoing
    player is pricey enough on its own, exactly the mechanism that made the planner unable to
    ever recommend upgrading to a genuinely premium player. Enough banked cash must legalize
    it; a bank that still falls short of the gap must not."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]  # price 8.0

    # p_in_too_expensive costs 9.0 -- illegal at bank=0.0 (price_in > price_out), legal once
    # bank covers the 1.0 gap.
    zero_bank_results = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4, bank=0.0,
    )
    assert "p_in_too_expensive" not in {r["player_in"] for r in zero_bank_results}

    insufficient_bank_results = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4, bank=0.5,
    )
    assert "p_in_too_expensive" not in {r["player_in"] for r in insufficient_bank_results}

    funded_results = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4, bank=1.0,
    )
    in_uids = {r["player_in"] for r in funded_results}
    assert "p_in_too_expensive" in in_uids
    rec = next(r for r in funded_results if r["player_in"] == "p_in_too_expensive")
    assert rec["price_out"] == pytest.approx(8.0)
    assert rec["price_in"] == pytest.approx(9.0)


# ============================================================
# evaluate_triple_captain
# ============================================================

def _seed_mc_run_and_summary(con, rows):
    """rows: [(player_uid, mean_total, var_total), ...]. evaluate_triple_captain() only ever
    reads monte_carlo_player_summary, but that table's model_version FKs into a real
    monte_carlo_run_versions row, which itself FKs the full M1-M5 chain -- reuses
    _seed_minimal_squad_optimizer_run()'s chain rather than duplicating it."""
    run_id, ep_mv, un_mv = _seed_minimal_squad_optimizer_run(con)
    ts_mv = con.execute("SELECT team_strength_model_version FROM uncertainty_model_versions WHERE model_version = ?", [un_mv]).fetchone()[0]
    mm_mv = con.execute("SELECT minutes_model_version FROM uncertainty_model_versions WHERE model_version = ?", [un_mv]).fetchone()[0]
    model_version = con.execute("SELECT nextval('seq_monte_carlo_model_version')").fetchone()[0]
    con.execute(
        "INSERT INTO monte_carlo_run_versions (model_version, calibration_asof_date, squad_optimizer_run_id, "
        "ep_model_version, minutes_model_version, team_strength_model_version, uncertainty_model_version, "
        "rho_residual_params_version, z_fixture_lambda_representative, z_fixture_variance, n_antithetic_pairs, "
        "query_id, seed) VALUES (?, '2026-08-10', ?, ?, ?, ?, ?, 1, 0.1, 0.1, 100, 'test', 1)",
        [model_version, run_id, ep_mv, mm_mv, ts_mv, un_mv],
    )
    for player_uid, mean_total, var_total in rows:
        con.execute(
            "INSERT INTO monte_carlo_player_summary (model_version, player_uid, mean_total, var_total, "
            "quantile_05, quantile_25, quantile_75, quantile_95, min_total, max_total) "
            "VALUES (?, ?, ?, ?, 0, 0, 0, 0, 0, 0)", [model_version, player_uid, mean_total, var_total],
        )
    return model_version


def test_evaluate_triple_captain_picks_highest_risk_adjusted_score(con):
    tp.seed_v1_params(con)
    for uid in ("high_mean_high_var", "mod_mean_low_var", "bench_player"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    model_version = _seed_mc_run_and_summary(con, [
        ("high_mean_high_var", 10.0, 25.0),   # 10 - 0.15*5 = 9.25
        ("mod_mean_low_var", 8.0, 4.0),        # 8 - 0.15*2 = 7.7
        ("bench_player", 20.0, 1.0),           # excluded -- not in xi_uids
    ])
    result = tp.evaluate_triple_captain(con, model_version, xi_uids={"high_mean_high_var", "mod_mean_low_var"}, kappa_tc_params_version=1)
    assert result["recommended"] is True
    assert result["captain_candidate"] == "high_mean_high_var"
    assert result["tc_score"] == pytest.approx(10.0 - 0.15 * 5.0)


def test_evaluate_triple_captain_no_data_returns_not_recommended(con):
    tp.seed_v1_params(con)
    result = tp.evaluate_triple_captain(con, 999, xi_uids={"anyone"}, kappa_tc_params_version=1)
    assert result["recommended"] is False


# ============================================================
# evaluate_bench_boost
# ============================================================

def test_evaluate_bench_boost_picks_gameweek_with_highest_bench_ep(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('bench1', 'B1', 'Defender')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_b', 'B')")
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
    for gw, ep_val in ((2, 3.0), (3, 7.0)):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, '2026-2027', ?, 'team_a', 'team_b', FALSE, "
            "'Premier League', '2026-08-24', current_timestamp)", [f"m{gw}", gw],
        )
        con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2026-08-10', '2026-2027', ?, ?, 1, 1, 1)", [ts_mv, mm_mv],
        )
        ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, 'bench1', ?, 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, f"m{gw}", ep_val],
        )
        if gw == 2:
            first_mv = ep_mv
        else:
            second_mv = ep_mv

    result = tp.evaluate_bench_boost(con, {2: (first_mv, 0), 3: (second_mv, 0)}, squad_uids={"bench1", "xi1"}, xi_uids={"xi1"})
    assert result["recommended"] is True
    assert result["target_gameweek"] == 3
    assert result["bench_ep_sum"] == pytest.approx(7.0)


def test_evaluate_bench_boost_no_bench_players(con):
    result = tp.evaluate_bench_boost(con, {2: (1, 1)}, squad_uids={"p1"}, xi_uids={"p1"})
    assert result["recommended"] is False


# ============================================================
# check_gw19_deadline
# ============================================================

def test_check_gw19_deadline_not_urgent_when_far_away():
    result = tp.check_gw19_deadline(target_gameweek=5, chips_used_set1=[])
    assert result["urgent"] is False
    assert result["forfeited_now"] is False
    assert result["gameweeks_until_gw19"] == 14


def test_check_gw19_deadline_urgent_inside_warning_window():
    result = tp.check_gw19_deadline(target_gameweek=17, chips_used_set1=["wildcard"])
    assert result["urgent"] is True
    assert set(result["unused_set1_chips"]) == {"free_hit", "triple_captain", "bench_boost"}


def test_check_gw19_deadline_not_urgent_if_all_chips_already_used():
    result = tp.check_gw19_deadline(target_gameweek=18, chips_used_set1=list(tp.ALL_CHIP_TYPES))
    assert result["urgent"] is False


def test_check_gw19_deadline_forfeited_once_gw19_arrives_with_unused_chips():
    result = tp.check_gw19_deadline(target_gameweek=19, chips_used_set1=["wildcard"])
    assert result["forfeited_now"] is True


# ============================================================
# apply_recommendation
# ============================================================

def _seed_transfer_plan_run_for_apply(con, state_version, target_gameweek=2, transfer_cost=0.0, price_out=0.0, price_in=0.0):
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p_new', 'New', 'Midfielder') "
        "ON CONFLICT DO NOTHING"
    )
    con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, "
        "input_state_version, horizon_params_version, transfer_cost_params_version, ep_model_versions, "
        "uncertainty_model_versions) VALUES ('2026-08-17', '2026-2027', ?, ?, 1, 1, '{}', '{}') RETURNING run_id",
        [target_gameweek, state_version],
    )
    run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]
    con.execute(
        "INSERT INTO transfer_recommendations (run_id, rank, player_out, player_in, price_out, price_in, "
        "horizon_value_gain, transfer_cost, net_value) VALUES (?, 1, 'p1', 'p_new', ?, ?, 4.0, ?, ?)",
        [run_id, price_out, price_in, transfer_cost, 4.0 - transfer_cost],
    )
    return run_id


def test_apply_recommendation_accepting_a_transfer_updates_holdings_and_free_transfers(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, transfer_cost=0.0)
    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=1)

    holdings = tp._read_holdings(con, new_state_version)
    uids = {h["player_uid"] for h in holdings}
    assert "p1" not in uids
    assert "p_new" in uids

    row = con.execute(
        "SELECT free_transfers_available, as_of_gameweek FROM manager_state_versions WHERE state_version = ?", [new_state_version]
    ).fetchone()
    assert row == (1, 3)  # was 1 FT, used it (cost 0.0), but a new FT is granted this gameweek too -> back to 1

    linked = con.execute("SELECT produced_by_run_id FROM manager_state_versions WHERE state_version = ?", [new_state_version]).fetchone()[0]
    assert linked == run_id


def test_apply_recommendation_declining_transfer_banks_free_transfer(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None)
    free_transfers = con.execute(
        "SELECT free_transfers_available FROM manager_state_versions WHERE state_version = ?", [new_state_version]
    ).fetchone()[0]
    assert free_transfers == 2  # 1 banked (not used) + 1 new allocation, capped at 5

    holdings = tp._read_holdings(con, new_state_version)
    assert {h["player_uid"] for h in holdings} == {"p1", "p2", "p3"}  # unchanged


def test_apply_recommendation_paid_transfer_does_not_consume_free_transfer(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, transfer_cost=4.0)
    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=1)
    free_transfers = con.execute(
        "SELECT free_transfers_available FROM manager_state_versions WHERE state_version = ?", [new_state_version]
    ).fetchone()[0]
    assert free_transfers == 2  # paid hit doesn't consume the FT, and a new one is still granted this gameweek


def test_apply_recommendation_spending_bank_on_an_upgrade_reduces_it(con):
    """The other half of the bank fix: accepting a transfer that draws on saved-up cash
    (price_in > price_out) must actually spend that cash from bank, not leave it untouched --
    otherwise the same bank could be "spent" again on a later transfer that shouldn't be legal."""
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    con.execute("UPDATE manager_state_versions SET bank = 10.0 WHERE state_version = ?", [state_version])

    run_id = _seed_transfer_plan_run_for_apply(con, state_version, transfer_cost=0.0, price_out=7.5, price_in=15.5)
    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=1)

    bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [new_state_version]).fetchone()[0]
    assert bank == pytest.approx(10.0 + 7.5 - 15.5)  # 2.0 left after funding the upgrade


def test_apply_recommendation_selling_down_grows_bank(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, transfer_cost=0.0, price_out=8.0, price_in=4.0)
    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=1)

    bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [new_state_version]).fetchone()[0]
    assert bank == pytest.approx(4.0)  # started at 0.0 (default), +8.0 sold -4.0 bought


def test_apply_recommendation_declining_transfer_leaves_bank_unchanged(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    con.execute("UPDATE manager_state_versions SET bank = 3.0 WHERE state_version = ?", [state_version])
    run_id = _seed_transfer_plan_run_for_apply(con, state_version)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None)
    bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [new_state_version]).fetchone()[0]
    assert bank == pytest.approx(3.0)


def test_apply_recommendation_accepting_a_chip_records_it_in_the_right_set(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, target_gameweek=10)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None, accept_chip="wildcard")
    chips_used_set1 = con.execute(
        "SELECT chips_used_set1 FROM manager_state_versions WHERE state_version = ?", [new_state_version]
    ).fetchone()[0]
    assert "wildcard" in chips_used_set1
