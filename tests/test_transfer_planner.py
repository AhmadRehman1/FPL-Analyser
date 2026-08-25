import json
from datetime import date

import pytest

from fpl_quant import fixture_swing
from fpl_quant import params
from fpl_quant import squad_optimizer as so
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


# ============================================================
# bootstrap_from_real_squad -- an arbitrary real squad, not a real M5 solve
# ============================================================

def _seed_model_version_chain(con, target_season="2026-2027"):
    """Just the ep_model_versions/uncertainty_model_versions FK chain
    bootstrap_from_real_squad()'s squad_optimizer_runs insert needs -- no players/selections,
    unlike _seed_minimal_squad_optimizer_run() above."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'A')")
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
    return ep_mv, un_mv


def _seed_resolvable_player(con, name, normalized, player_uid, season="2026-2027"):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [player_uid, name])
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
        "VALUES (?, ?, '1', ?, ?)", [name, normalized, season, player_uid],
    )


def test_bootstrap_from_real_squad_resolves_names_and_creates_matching_state(con):
    ep_mv, un_mv = _seed_model_version_chain(con)
    _seed_resolvable_player(con, "Bruno Fernandes", "bruno fernandes", "p_bruno")
    _seed_resolvable_player(con, "Erling Haaland", "erling haaland", "p_haaland")
    squad = [
        {"player_name": "Bruno Fernandes", "in_xi": True, "is_captain": True, "is_vice": False},
        {"player_name": "Erling Haaland", "in_xi": True, "is_captain": False, "is_vice": True},
    ]
    state_version = tp.bootstrap_from_real_squad(con, date(2026, 8, 22), "2026-2027", 2, ep_mv, un_mv, squad)

    state = con.execute(
        "SELECT season, as_of_gameweek, free_transfers_available FROM manager_state_versions WHERE state_version = ?",
        [state_version],
    ).fetchone()
    assert state == ("2026-2027", 2, 1)

    holdings = tp._read_holdings(con, state_version)
    assert {h["player_uid"] for h in holdings} == {"p_bruno", "p_haaland"}
    bruno = next(h for h in holdings if h["player_uid"] == "p_bruno")
    assert bruno["is_captain"] is True

    # the underlying squad_optimizer_runs row must be flagged is_manager_snapshot, same
    # discipline as the squad_optimizer-run-sourced bootstrap path
    run_row = con.execute(
        "SELECT so.is_manager_snapshot FROM squad_optimizer_selections sel "
        "JOIN squad_optimizer_runs so ON so.run_id = sel.run_id "
        "WHERE sel.player_uid = 'p_bruno'"
    ).fetchone()
    assert run_row[0] is True


def test_bootstrap_from_real_squad_raises_on_unresolvable_name(con):
    ep_mv, un_mv = _seed_model_version_chain(con)
    squad = [{"player_name": "Not A Real Player", "in_xi": True, "is_captain": False, "is_vice": False}]
    with pytest.raises(ValueError):
        tp.bootstrap_from_real_squad(con, date(2026, 8, 22), "2026-2027", 2, ep_mv, un_mv, squad)
    # nothing partially committed for the unresolvable squad
    assert con.execute("SELECT count(*) FROM squad_optimizer_selections").fetchone()[0] == 0


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
# evaluate_wildcard / evaluate_free_hit -- both call a real squad_optimizer.run() internally,
# so both need a real, DB-seeded 15+-player candidate pool (an in-memory pool like
# test_squad_optimizer.py's own _synthetic_pool() isn't enough here -- fetch_candidate_pool()
# reads real tables). Neither function had any test coverage before this -- closing that gap
# while fixing the free_hit_gain_threshold_params bug, not just patching the symptom.
# ============================================================

def _seed_real_squad_optimizer_candidate_pool(con, target_season="2026-2027", target_gameweek=2, extra_gameweek_mu_overrides=None):
    """Same 2 GK/6 DEF/6 MID/4 FWD/6-club shape as test_squad_optimizer.py's _synthetic_pool()
    (budget-feasible, club-cap-feasible), but written into real tables so squad_optimizer.run()
    -- called for real inside evaluate_wildcard()/evaluate_free_hit() -- can solve against it.

    extra_gameweek_mu_overrides: optional {gameweek: flat_mu} -- seeds one additional real
    ep_model_version/uncertainty_model_version pair (own fixture, own ep_outputs/
    uncertainty_outputs) per extra gameweek, giving every one of the 11 held players
    (holdings, below) that SAME flat mu at that gameweek -- different from target_gameweek's
    own fixture mu. A single-gameweek horizon_ep_versions can't exercise the per-gw trajectory
    fields evaluate_wildcard()/evaluate_free_hit()/evaluate_triple_captain() now carry
    (current_squad_value_per_gw etc.); this buys a genuine, real second data point without
    needing a full second candidate pool (only the held players need rows -- the field only
    ever sums over currently-held/XI players, never the whole pool)."""
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    for i, club in enumerate(clubs):
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [club, club])
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', ?, ?, 'clubA', 'clubB', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)", [target_season, target_gameweek],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'clubA')"
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

    players = []
    for i in range(2):
        players.append((f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6]))
    for i in range(6):
        players.append((f"def{i}", "Defender", 2.5 + i * 0.3, 4.0 + i * 0.5, clubs[i % 6]))
    for i in range(6):
        players.append((f"mid{i}", "Midfielder", 3.0 + i * 0.4, 5.0 + i * 0.5, clubs[i % 6]))
    for i in range(4):
        players.append((f"fwd{i}", "Forward", 3.5 + i * 0.5, 6.0 + i * 0.5, clubs[i % 6]))

    for uid, position, mu, price, club in players:
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), club, target_season, uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp)", [uid, target_season, price],
        )
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, mu],
        )
        # differentiated variance (proportional to mu, like M4's real output) -- a flat,
        # identical variance for every player is exactly the degenerate case
        # squad_optimizer.run()'s own divergence check is designed to reject (see
        # test_squad_optimizer.py's test_divergence_check_fails_when_variance_is_a_stub_zero).
        var = 1.0 + mu * 3.0
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, ?, 0,0,0,0,0,0)", [un_mv, uid, var],
        )

    # real positive covariance among the highest-mu players (like M4's real teammate/opponent
    # structure) -- needed alongside the differentiated variance above so lambda actually has
    # a genuine diversification incentive, matching test_squad_optimizer.py's own recipe for a
    # real (non-degenerate) divergence-check pass.
    high_mu_uids = [uid for uid, _pos, mu, _price, _club in players if mu >= 4.0]
    for i in range(len(high_mu_uids)):
        for j in range(i + 1, len(high_mu_uids)):
            con.execute(
                "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, "
                "fixture_match_id, relationship, covariance) VALUES (?, ?, ?, 'm1', 'teammate', 6.0)",
                [un_mv, *sorted([high_mu_uids[i], high_mu_uids[j]])],
            )

    holdings = [{"player_uid": uid, "in_xi": True, "is_captain": False, "is_vice": False} for uid, *_ in players[:11]]
    held_uids = [h["player_uid"] for h in holdings]

    horizon_ep_versions = {target_gameweek: (ep_mv, un_mv)}
    for gw, flat_mu in (extra_gameweek_mu_overrides or {}).items():
        mu_overrides = {uid: flat_mu for uid in held_uids}
        match_id = f"m_{gw}"
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'clubA', 'clubB', FALSE, "
            "'Premier League', '2026-08-24', current_timestamp)", [match_id, target_season, gw],
        )
        con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1)", [target_season, ts_mv, mm_mv],
        )
        gw_ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
        con.execute(
            "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
            "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1)",
            [gw_ep_mv, mm_mv, ts_mv],
        )
        gw_un_mv = con.execute("SELECT max(model_version) FROM uncertainty_model_versions").fetchone()[0]
        for uid, mu in mu_overrides.items():
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
                [gw_ep_mv, uid, match_id, mu],
            )
            con.execute(
                "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
                "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
                "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
                "VALUES (?, ?, ?, 0,0,0,0,0,0,0,0, ?, 0,0,0,0,0,0)", [gw_un_mv, uid, match_id, 1.0 + mu * 3.0],
            )
        horizon_ep_versions[gw] = (gw_ep_mv, gw_un_mv)

    return horizon_ep_versions, holdings


def test_evaluate_free_hit_uses_its_own_threshold_family_not_wildcards(con):
    """Regression test for the real bug: evaluate_free_hit() used to resolve its threshold
    against wildcard_gain_threshold_params. Seeding ONLY free_hit_gain_threshold_params (not
    wildcard's) and confirming evaluate_free_hit() still resolves cleanly proves it now reads
    its own family -- it would raise ParamNotFoundError against the old, wrong family."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "free_hit_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1.5)

    result = tp.evaluate_free_hit(
        con, date(2026, 8, 24), "2026-2027", 2, holdings, horizon_ep_versions,
        lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    assert "gain" in result
    assert isinstance(result["recommended"], bool)


def test_evaluate_free_hit_raises_if_only_wildcards_family_is_seeded(con):
    """Confirms the fix is real, not just non-crashing by coincidence: with ONLY the old
    wildcard_gain_threshold_params seeded (matching pre-fix behavior) and free_hit's own family
    absent, evaluate_free_hit() must fail to resolve -- proving it no longer falls back to
    wildcard's family under the hood."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)

    with pytest.raises(tp.params_mod.ParamNotFoundError):
        tp.evaluate_free_hit(
            con, date(2026, 8, 24), "2026-2027", 2, holdings, horizon_ep_versions,
            lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
        )


def test_evaluate_free_hit_current_xi_value_per_gw_is_a_real_per_gameweek_sum(con):
    """Same regression coverage as evaluate_wildcard()'s own version above, for Free Hit's
    XI-only (not whole-squad) trajectory field."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(
        con, target_gameweek=2, extra_gameweek_mu_overrides={3: 1.0},
    )
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "free_hit_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1.5)

    result = tp.evaluate_free_hit(
        con, date(2026, 8, 24), "2026-2027", 2, holdings, horizon_ep_versions,
        lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    assert set(result["current_xi_value_per_gw"]) == {2, 3}
    assert result["current_xi_value_per_gw"][3] == pytest.approx(11.0)  # 11 XI players x flat mu=1.0
    assert result["current_xi_value_per_gw"][2] == pytest.approx(result["current_gw_value"])


def test_evaluate_wildcard_still_uses_its_own_family(con):
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)

    result = tp.evaluate_wildcard(
        con, date(2026, 8, 24), "2026-2027", 2, current_squad_horizon_value=10.0, best_transfer_net_value=0.0,
        horizon_ep_versions=horizon_ep_versions, lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    assert "gain" in result
    assert isinstance(result["recommended"], bool)


def test_evaluate_wildcard_current_squad_value_per_gw_is_a_real_per_gameweek_sum(con):
    """Regression test for the chip-timing work: current_squad_value_per_gw must be the
    CURRENT holdings' own mu summed per horizon gameweek -- a real, zero-extra-solve number
    the season simulation's decision rule uses to tell whether now or a later visible week is
    genuinely worse for the squad it already owns, not the fresh (post-rebuild) squad's value."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(
        con, target_gameweek=2, extra_gameweek_mu_overrides={3: 1.0},
    )
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)
    result = tp.evaluate_wildcard(
        con, date(2026, 8, 24), "2026-2027", 2, current_squad_horizon_value=10.0, best_transfer_net_value=0.0,
        horizon_ep_versions=horizon_ep_versions, lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
        current_holdings=holdings,
    )
    assert set(result["current_squad_value_per_gw"]) == {2, 3}
    # gw3's mu was overridden to a flat 1.0 per held player -- 11 held players -> 11.0 exactly.
    assert result["current_squad_value_per_gw"][3] == pytest.approx(11.0)
    # gw2 is the real (non-overridden) mu the base fixture seeded -- just confirm it's the sum
    # over the held players specifically, not the whole ~24-player pool.
    assert result["current_squad_value_per_gw"][2] > 0.0


def _seed_real_wildcard_chip_evaluation(con, old_state_version, target_gameweek=2):
    """Real evaluate_wildcard() call (a genuine squad_optimizer.run() solve) plus the same
    transfer_plan_runs/chip_evaluations rows run() itself would write -- the exact shape
    apply_recommendation()'s Wildcard-accept path reads via read_fresh_chip_squad()."""
    horizon_ep_versions, _holdings = _seed_real_squad_optimizer_candidate_pool(con, target_gameweek=target_gameweek)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)
    wildcard_result = tp.evaluate_wildcard(
        con, date(2026, 8, 24), "2026-2027", target_gameweek, current_squad_horizon_value=0.0, best_transfer_net_value=0.0,
        horizon_ep_versions=horizon_ep_versions, lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    con.execute(
        "INSERT INTO transfer_plan_runs (calibration_asof_date, target_season, target_gameweek, "
        "input_state_version, horizon_params_version, transfer_cost_params_version, ep_model_versions, "
        "uncertainty_model_versions) VALUES ('2026-08-17', '2026-2027', ?, ?, 1, 1, '{}', '{}') RETURNING run_id",
        [target_gameweek, old_state_version],
    )
    run_id = con.execute("SELECT max(run_id) FROM transfer_plan_runs").fetchone()[0]
    con.execute(
        "INSERT INTO chip_evaluations (run_id, chip_type, recommended, score_or_gain, detail, gw19_urgent_flag) "
        "VALUES (?, 'wildcard', TRUE, ?, ?, FALSE)",
        [run_id, wildcard_result["gain"], json.dumps(wildcard_result, default=str)],
    )
    fresh_squad_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [wildcard_result["fresh_run_id"]]
        ).fetchall()
    }
    return run_id, fresh_squad_uids


def test_apply_recommendation_wildcard_rebuilds_holdings_from_the_fresh_squad(con):
    """Regression test for the real bug: accepting a Wildcard used to be a complete no-op on
    the squad (only the chip got marked used). The new holdings must be exactly the fresh M5
    squad Wildcard evaluated, not the old pre-Wildcard squad."""
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)  # old squad: p1, p2, p3
    old_state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id, fresh_squad_uids = _seed_real_wildcard_chip_evaluation(con, old_state_version)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None, accept_chip="wildcard")
    new_holdings_uids = {h["player_uid"] for h in tp._read_holdings(con, new_state_version)}

    assert new_holdings_uids == fresh_squad_uids
    assert new_holdings_uids != {"p1", "p2", "p3"}  # not the stale pre-Wildcard squad
    assert len(new_holdings_uids) == 15


def test_apply_recommendation_wildcard_recomputes_bank_from_the_fresh_squad(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    old_state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    con.execute("UPDATE manager_state_versions SET bank = 25.0 WHERE state_version = ?", [old_state_version])
    run_id, fresh_squad_uids = _seed_real_wildcard_chip_evaluation(con, old_state_version)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None, accept_chip="wildcard")
    new_bank = con.execute("SELECT bank FROM manager_state_versions WHERE state_version = ?", [new_state_version]).fetchone()[0]

    prices = dict(con.execute(
        "SELECT player_uid, now_cost FROM fact_player_season_stats WHERE season = '2026-2027' AND now_cost IS NOT NULL"
    ).fetchall())
    expected_bank = so.BUDGET - sum(prices[uid] for uid in fresh_squad_uids)
    assert new_bank == pytest.approx(expected_bank)
    assert new_bank != pytest.approx(25.0)  # the stale pre-Wildcard bank must not carry over


def test_apply_recommendation_rejects_wildcard_combined_with_a_transfer(con):
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    old_state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id, _fresh_squad_uids = _seed_real_wildcard_chip_evaluation(con, old_state_version)

    with pytest.raises(ValueError):
        tp.apply_recommendation(con, run_id, accept_transfer_rank=1, accept_chip="wildcard")


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


# ============================================================
# price_momentum_by_player / evaluate_transfers momentum keys -- a real, secondary signal,
# never folded into net_value or the ranking sort (research thread item #2).
# ============================================================

def test_price_momentum_by_player_computes_delta_over_the_lookback_window(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Midfielder')")
    for gw, price, ownership in ((1, 7.0, 10.0), (2, 7.5, 12.0), (3, 8.0, 15.0)):
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
            "VALUES ('p1', '2026-2027', ?, ?, ?, current_timestamp)", [gw, price, ownership],
        )
    momentum = tp.price_momentum_by_player(con, "2026-2027", as_of_gameweek=3, lookback_gameweeks=3)
    assert momentum["p1"]["price_delta"] == pytest.approx(8.0 - 7.0)
    assert momentum["p1"]["ownership_delta"] == pytest.approx(15.0 - 10.0)


def test_price_momentum_by_player_none_when_no_history_that_far_back(con):
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p1', 'p1', 'Midfielder')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
        "VALUES ('p1', '2026-2027', 1, 7.0, 10.0, current_timestamp)"
    )
    momentum = tp.price_momentum_by_player(con, "2026-2027", as_of_gameweek=1, lookback_gameweeks=3)
    assert momentum["p1"]["price_delta"] is None
    assert momentum["p1"]["ownership_delta"] is None


def test_evaluate_transfers_attaches_momentum_without_changing_the_ranking(con):
    """target_gameweek opts the momentum keys in; the ranking (net_value, sort order) must be
    byte-for-byte identical to the target_gameweek=None case -- momentum is informational
    only, never a ranking input."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]
    # a real, distinct earlier price/ownership snapshot for p_in_same_pos_cheaper, so its
    # momentum delta is genuinely non-trivial, not coincidentally zero.
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
        "VALUES ('p_in_same_pos_cheaper', '2026-2027', 2, 4.5, 20.0, current_timestamp)"
    )

    without = tp.evaluate_transfers(con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4)
    with_momentum = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4, target_gameweek=2,
    )

    assert [r["net_value"] for r in without] == [r["net_value"] for r in with_momentum]
    assert [r["player_in"] for r in without] == [r["player_in"] for r in with_momentum]
    assert "price_momentum_in" not in without[0]

    top = with_momentum[0]
    assert top["player_in"] == "p_in_same_pos_cheaper"
    assert top["price_momentum_in"] == pytest.approx(4.5 - 4.0)  # gw2 price (4.5) minus gw1's seeded price (4.0)
    # the base fixture never seeded selected_by_percent at gw1 for this player -- missing
    # history means None, not a fabricated 0.0 baseline.
    assert top["ownership_momentum_in"] is None


# ============================================================
# evaluate_transfers fixture-swing keys -- same informational-only convention as momentum
# above (never folded into net_value or the ranking sort). See fixture_swing.py.
# ============================================================

def _seed_raw_teams_csv(con, season, rows):
    """rows: list of (code, name). team_uid_by_player() resolves player->team via
    reconcile._season_root_table, which reads fact_raw's teams.csv table -- a real
    dependency, matching test_minutes_model.py's own helper of the same shape."""
    table = f"raw_{season.replace('-', '_')}_teams_tp"
    con.execute(f'CREATE TABLE "{table}" (code VARCHAR, name VARCHAR)')
    for code, name in rows:
        con.execute(f'INSERT INTO "{table}" VALUES (?, ?)', [code, name])
    con.execute(
        "INSERT INTO fact_raw_ingestion_log (raw_table_name, season, source_relpath, source_file_hash, row_count) "
        "VALUES (?, ?, 'teams.csv', ?, ?)",
        [table, season, f"fakehash_{table}", len(rows)],
    )


def test_evaluate_transfers_attaches_fixture_swing_without_changing_the_ranking(con):
    """ts_model_version (together with target_gameweek) opts the fixture_swing keys in; the
    ranking (net_value, sort order) must be byte-for-byte identical to the case without it --
    same informational-only convention price/ownership momentum already established."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]

    _seed_raw_teams_csv(con, "2026-2027", [("1", "TeamOne")])
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) "
        "VALUES ('TeamOne', '2026-2027', 'team_a', 't')"
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, attack, defence in (("team_a", 0.3, 0.0), ("team_b", -0.1, 0.1)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    # m1 (gw2) already seeded by _seed_ep_and_holdings_for_transfers -- one more fixture so the
    # rolling window has more than a single data point to average over.
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m2', '2026-2027', 3, 'team_a', 'team_b', FALSE, "
        "'Premier League', '2026-08-31', current_timestamp)"
    )

    without = tp.evaluate_transfers(con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4)
    with_swing = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4,
        target_gameweek=2, ts_model_version=ts_mv,
    )

    assert [r["net_value"] for r in without] == [r["net_value"] for r in with_swing]
    assert [r["player_in"] for r in without] == [r["player_in"] for r in with_swing]
    assert "fixture_swing_in" not in without[0]

    top = with_swing[0]
    assert top["player_in"] == "p_in_same_pos_cheaper"
    # every synthetic player in this fixture is on the same team_code -- fixture_swing_in/out
    # must both equal that one team's own rolling_swing_score(), computed independently here
    # to prove evaluate_transfers() is really reading fixture_swing's output, not inventing one.
    expected = fixture_swing.rolling_swing_score(con, "team_a", "2026-2027", as_of_gameweek=2, ts_model_version=ts_mv)
    assert expected.swing_score is not None
    assert top["fixture_swing_in"] == pytest.approx(expected.swing_score)
    assert top["fixture_swing_out"] == pytest.approx(expected.swing_score)


def test_evaluate_transfers_fixture_swing_none_without_ts_model_version(con):
    """target_gameweek alone (no ts_model_version) must not add the fixture_swing keys --
    they're gated on both, since a swing score is meaningless without a team-strength model."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]

    results = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4,
        target_gameweek=2,
    )
    assert "fixture_swing_in" not in results[0]


# ============================================================
# Priority 5 -- crowded_chip_week_score / chip-combo sequencing
# ============================================================

def _seed_crowded_week_scenario(con, target_season="2026-2027", favored_team_ownership=80.0, other_team_ownership=5.0):
    """Two teams, real team_strength_snapshots (team_a strong attack/weak defence vs.
    team_b's own), a fixture at gw2 and a further one at gw3 (so rolling_swing_score() has a
    real short/long window to average over), and real selected_by_percent so the
    ownership-weighting in crowded_chip_week_score() has genuine data to work with -- the
    fixture the roadmap itself names: a heavily-owned team's own favorable run should read as
    a broadly template-friendly (crowded) week."""
    con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES ('team_a', 'TeamA'), ('team_b', 'TeamB')")
    _seed_raw_teams_csv(con, target_season, [("1", "TeamA"), ("2", "TeamB")])
    con.execute(
        "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES "
        "('TeamA', ?, 'team_a', 't'), ('TeamB', ?, 'team_b', 't')", [target_season, target_season],
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'team_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    for team_uid, attack, defence in (("team_a", 0.5, -0.3), ("team_b", -0.3, 0.3)):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, ?, ?, 2, 1.0)",
            [ts_mv, team_uid, attack, defence],
        )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES "
        "('m2', ?, 2, 'team_a', 'team_b', FALSE, 'Premier League', '2026-08-24', current_timestamp), "
        "('m3', ?, 3, 'team_a', 'team_b', FALSE, 'Premier League', '2026-08-31', current_timestamp)",
        [target_season, target_season],
    )
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('p_a', 'p_a', 'Midfielder'), ('p_b', 'p_b', 'Midfielder')")
    con.execute(
        "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) VALUES "
        "('p_a', 'p_a', '1', ?, 'p_a'), ('p_b', 'p_b', '2', ?, 'p_b')", [target_season, target_season],
    )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
        "VALUES ('p_a', ?, 1, 5.0, ?, current_timestamp), ('p_b', ?, 1, 5.0, ?, current_timestamp)",
        [target_season, favored_team_ownership, target_season, other_team_ownership],
    )
    return ts_mv


def test_crowded_chip_week_score_reflects_ownership_weighted_swing(con):
    ts_mv = _seed_crowded_week_scenario(con, favored_team_ownership=80.0, other_team_ownership=5.0)
    result = tp.crowded_chip_week_score(con, "2026-2027", 2, ts_mv)
    assert result["crowded_score"] is not None

    # confirm it's a genuine ownership-weighted average, not coincidentally equal to a flat
    # average, by recomputing it independently against the same real swing scores.
    swing_a = fixture_swing.rolling_swing_score(con, "team_a", "2026-2027", as_of_gameweek=2, ts_model_version=ts_mv)
    swing_b = fixture_swing.rolling_swing_score(con, "team_b", "2026-2027", as_of_gameweek=2, ts_model_version=ts_mv)
    expected = (80.0 * swing_a.swing_score + 5.0 * swing_b.swing_score) / (80.0 + 5.0)
    assert result["crowded_score"] == pytest.approx(expected)


def test_crowded_chip_week_score_none_when_no_ownership_data(con):
    ts_mv = _seed_crowded_week_scenario(con)
    con.execute("UPDATE fact_player_season_stats SET selected_by_percent = NULL")
    result = tp.crowded_chip_week_score(con, "2026-2027", 2, ts_mv)
    assert result["crowded_score"] is None
    assert "caveat" in result


def test_evaluate_bench_boost_attaches_crowded_chip_week_when_opted_in(con):
    ts_mv = _seed_crowded_week_scenario(con)
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]') RETURNING model_version"
    )
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
        "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
        "VALUES ('2026-08-10', '2026-2027', ?, ?, 1, 1, 1) RETURNING model_version", [ts_mv, mm_mv],
    )
    ep_mv = con.execute("SELECT max(model_version) FROM ep_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
        "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
        "ep_total, expected_bps) VALUES (?, 'p_a', 'm2', 0,0,0,0,0,0,0,0,0,0,0, 3.0, 5.0)", [ep_mv],
    )
    horizon_ep_versions = {2: (ep_mv, 1)}

    without = tp.evaluate_bench_boost(con, horizon_ep_versions, squad_uids={"p_a", "p_b"}, xi_uids={"p_b"})
    assert "crowded_chip_week" not in without

    with_crowded = tp.evaluate_bench_boost(
        con, horizon_ep_versions, squad_uids={"p_a", "p_b"}, xi_uids={"p_b"},
        target_season="2026-2027", ts_model_version=ts_mv,
    )
    assert with_crowded["target_gameweek"] == without["target_gameweek"]  # never changes WHICH gw is picked
    assert with_crowded["bench_ep_sum"] == without["bench_ep_sum"]
    assert "crowded_chip_week" in with_crowded
    assert with_crowded["crowded_chip_week"]["crowded_score"] is not None


def test_evaluate_wildcard_bench_boost_combo_computes_synergy(con):
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=8.0)

    wildcard_result = tp.evaluate_wildcard(
        con, date(2026, 8, 24), "2026-2027", 2, current_squad_horizon_value=10.0, best_transfer_net_value=0.0,
        horizon_ep_versions=horizon_ep_versions, lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    squad_uids = {h["player_uid"] for h in holdings}
    xi_uids = {h["player_uid"] for h in holdings if h["in_xi"]}
    bench_boost_result = tp.evaluate_bench_boost(con, horizon_ep_versions, squad_uids, xi_uids)

    combo = tp.evaluate_wildcard_bench_boost_combo(con, wildcard_result, bench_boost_result, "2026-2027", 2, horizon_ep_versions)
    assert isinstance(combo["recommended_combo"], bool)
    # combo_value/naive_independent_sum are both real, single-gameweek totals at
    # target_gameweek (XI+bench under the combo vs. XI alone + the current squad's own bench
    # value) -- never mixed with wildcard_gain_alone's own multi-gameweek proxy (see the real
    # double-counting bug this replaced, in the function's own docstring/comments).
    assert combo["combo_value"] == pytest.approx(combo["fresh_squad_xi_ep_at_target_gw"] + combo["fresh_squad_bench_ep_at_target_gw"])
    assert combo["naive_independent_sum"] == pytest.approx(combo["fresh_squad_xi_ep_at_target_gw"] + combo["bench_boost_value_alone"])
    assert combo["synergy_gain"] == pytest.approx(combo["combo_value"] - combo["naive_independent_sum"])
    assert combo["synergy_gain"] == pytest.approx(combo["fresh_squad_bench_ep_at_target_gw"] - combo["bench_boost_value_alone"])
    # the fresh (post-Wildcard) squad's own bench/XI EP at target_gameweek must be real,
    # non-negative numbers derived from a real solved squad -- not placeholders.
    assert combo["fresh_squad_bench_ep_at_target_gw"] >= 0.0
    assert combo["fresh_squad_xi_ep_at_target_gw"] >= 0.0


def test_evaluate_wildcard_bench_boost_combo_never_recommended_when_synergy_is_real_but_wildcard_itself_is_not_worth_it(con):
    """Regression test for the real bug: combo_value used to mix wildcard_gain_alone (a
    multi-gameweek, whole-squad proxy) with fresh_bench_ep_at_target_gw (a real subset already
    counted inside it), so recommended_combo fired almost unconditionally -- any positive
    fresh bench EP alone was enough, even when Wildcard itself was a bad idea on its own
    merits. Confirmed against the pre-fix code first: with wildcard_result["recommended"]=False
    but genuine bench synergy (fresh bench EP well above the tiny bb_alone_value below), the
    OLD formula (combo_value = wc_alone_gain + fresh_bench_ep_at_target_gw, compared against
    max(wc_alone_gain, bb_alone_value)) would have recommended the combo anyway -- adding a
    non-negative fresh_bench_ep_at_target_gw to wc_alone_gain can never make combo_value SMALLER
    than wc_alone_gain itself, regardless of whether Wildcard was ever worth doing. The fix
    requires Wildcard to clear its own bar too."""
    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    # A deliberately huge threshold this fixture's real gain can never clear (verified: this
    # fixture's real gain is 46.3) -- isolates "Wildcard doesn't clear its own bar" as a
    # controlled precondition, not an accident of the fixture's own numbers.
    tp.params_mod.write_param(con, "wildcard_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1_000_000.0)

    wildcard_result = tp.evaluate_wildcard(
        con, date(2026, 8, 24), "2026-2027", 2, current_squad_horizon_value=10.0, best_transfer_net_value=0.0,
        horizon_ep_versions=horizon_ep_versions, lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    assert wildcard_result["recommended"] is False

    # A deliberately tiny bb_alone_value (well below the fresh squad's real bench EP) so
    # synergy_gain is genuinely positive -- isolates "Wildcard itself isn't worth it" as the
    # ONLY reason recommended_combo should be False here.
    bench_boost_result = {"recommended": True, "bench_ep_sum": 0.01}

    combo = tp.evaluate_wildcard_bench_boost_combo(con, wildcard_result, bench_boost_result, "2026-2027", 2, horizon_ep_versions)
    assert combo["synergy_gain"] > 0  # the real bench-synergy signal is genuinely positive
    assert combo["recommended_combo"] is False  # but Wildcard alone doesn't clear its own bar


def test_evaluate_wildcard_bench_boost_combo_no_action_when_wildcard_infeasible(con):
    result = tp.evaluate_wildcard_bench_boost_combo(
        con, {"recommended": False, "reason": "no fixtures"}, {"recommended": False}, "2026-2027", 2, {},
    )
    assert result["recommended_combo"] is False
    assert "reason" in result


def test_evaluate_free_hit_triple_captain_combo_uses_a_real_simulation(con):
    from fpl_quant import expected_points as ep_mod
    from fpl_quant import uncertainty as un_mod

    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    # monte_carlo.run()'s own team resolution (_team_of_for_fixture) needs the SAME raw-
    # teams.csv -> team_alias -> dim_team chain squad_optimizer.explain_run()'s club audit
    # and fixture_swing's team_uid_by_player() both already use --
    # _seed_real_squad_optimizer_candidate_pool doesn't provide it (evaluate_wildcard()/
    # evaluate_free_hit() never needed it, only fetch_candidate_pool() does), so a REAL
    # simulation test has to add it explicitly or every squad player's team_uid resolves to
    # None and gets silently skipped (see simulate_fixture()'s own "can't resolve team side,
    # skip rather than guess" comment).
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    _seed_raw_teams_csv(con, "2026-2027", [(c, c) for c in clubs])
    for c in clubs:
        con.execute(
            "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, '2026-2027', ?, 't')",
            [c, c],
        )
    so.seed_v1_params(con)
    ts_mv_for_snapshots = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    # expected_points._fixture_lambdas() (called inside monte_carlo.simulate_fixture()) needs
    # a real team_strength_snapshots row for BOTH sides of fixture 'm1' (clubA vs clubB, per
    # _seed_real_squad_optimizer_candidate_pool's own single fixture) -- not seeded by that
    # helper either, since evaluate_wildcard()/evaluate_free_hit() never simulate a fixture
    # directly.
    for c in ("clubA", "clubB"):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, 0.1, 0.0, 2, 1.0)",
            [ts_mv_for_snapshots, c],
        )
    # monte_carlo.simulate_fixture() resolves model_decay_params.rho for the Dixon-Coles
    # bivariate Poisson grid -- team_strength.py has no seed_v1_params of its own (confirmed by
    # grep); every other test needing this writes it directly.
    tp.params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    ep_mod.seed_v1_params(con)
    un_mod.seed_v1_params(con)
    tp.params_mod.write_param(con, "free_hit_gain_threshold_params", 1, "2026-08-12", "min_horizon_gain", value_numeric=1.5)
    tp.params_mod.write_param(con, "tc_risk_aversion_params", 1, "2026-08-12", "kappa_tc", value_numeric=0.15)
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]
    ep_mv, un_mv = horizon_ep_versions[2]

    # monte_carlo.simulate_fixture()'s own roster query INNER JOINs ep_outputs against
    # minutes_model_outputs (p_0min/p_1_59min/p_60plus_min) -- _seed_real_squad_optimizer_
    # candidate_pool doesn't seed this either (same reason as the team-alias chain above), so
    # without it the roster join comes back empty and nothing gets simulated at all.
    all_uids = [r[0] for r in con.execute("SELECT DISTINCT player_uid FROM ep_outputs WHERE model_version = ?", [ep_mv]).fetchall()]
    for uid in all_uids:
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
            "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, 'Midfielder', 0.8, 1.0, 0.8, 0.0, 0.8, 0.0, 0.1, 0.1, 0.8, 20)",
            [mm_mv, uid],
        )

    free_hit_result = tp.evaluate_free_hit(
        con, date(2026, 8, 24), "2026-2027", 2, holdings, horizon_ep_versions,
        lambda_params_version=1, guardrail_params_version=1, threshold_params_version=1,
    )
    current_tc_result = {"recommended": False}  # no current-squad simulation needed for this test

    combo = tp.evaluate_free_hit_triple_captain_combo(
        con, date(2026, 8, 24), free_hit_result, current_tc_result, ep_mv, mm_mv, ts_mv, un_mv,
        scoring_params_version=1, tau_params_version=1, rho_residual_params_version=1, kappa_tc_params_version=1,
        n_antithetic_pairs=200,
    )
    assert combo["combo_tc_score"] is not None
    assert combo["combo_captain_candidate"] is not None
    assert combo["recommended_combo"] is True  # beats current_tc_result's None (no current TC option available)
    # a real monte_carlo_run_versions row must actually exist -- proves this ran a genuine
    # simulation, not a stub.
    mc_row = con.execute(
        "SELECT count(*) FROM monte_carlo_run_versions WHERE model_version = ?", [combo["combo_mc_model_version"]]
    ).fetchone()[0]
    assert mc_row == 1


def test_evaluate_free_hit_triple_captain_combo_no_action_when_free_hit_infeasible(con):
    result = tp.evaluate_free_hit_triple_captain_combo(
        con, date(2026, 8, 24), {"recommended": False}, {"recommended": False}, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    assert result["recommended_combo"] is False
    assert "reason" in result


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
# Priority 4 -- price_change_risk_by_player / transfer_timing_advice
# ============================================================

def test_price_change_risk_by_player_flags_rise_and_fall(con):
    con.execute(
        "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('riser', 'riser', 'Midfielder'), "
        "('faller', 'faller', 'Midfielder'), ('steady', 'steady', 'Midfielder')"
    )
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, transfers_in_event, "
        "transfers_out_event, _ingested_at) VALUES "
        "('riser', '2026-2027', 2, 5.0, 80000, 1000, current_timestamp), "
        "('faller', '2026-2027', 2, 5.0, 1000, 80000, current_timestamp), "
        "('steady', '2026-2027', 2, 5.0, 5000, 4800, current_timestamp)"
    )
    result = tp.price_change_risk_by_player(con, "2026-2027", 2, rise_threshold=50_000.0, fall_threshold=50_000.0)
    assert result["riser"]["price_change_direction"] == "rise"
    assert result["riser"]["net_transfers_event"] == pytest.approx(79000.0)
    assert result["faller"]["price_change_direction"] == "fall"
    assert result["steady"]["price_change_direction"] == "none"


def test_price_change_risk_by_player_none_when_no_transfer_data(con):
    """Missing transfers_in_event/transfers_out_event (e.g. a source season that predates
    these columns, or simply no row at all for this gameweek) must report direction=None, not
    silently coerced to "none" (a real, known "below threshold" result) or a fabricated 0."""
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('unknown', 'unknown', 'Midfielder')")
    con.execute(
        "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
        "VALUES ('unknown', '2026-2027', 2, 5.0, current_timestamp)"
    )
    result = tp.price_change_risk_by_player(con, "2026-2027", 2, rise_threshold=50_000.0, fall_threshold=50_000.0)
    assert result["unknown"]["price_change_direction"] is None
    assert result["unknown"]["net_transfers_event"] is None


def test_transfer_timing_advice_urgent_incoming_rise():
    advice = tp.transfer_timing_advice({"price_change_direction": "rise"}, {"price_change_direction": "none"})
    assert advice is not None
    assert "today, not tomorrow" in advice


def test_transfer_timing_advice_urgent_outgoing_fall():
    advice = tp.transfer_timing_advice({"price_change_direction": "none"}, {"price_change_direction": "fall"})
    assert advice is not None
    assert "today, not tomorrow" in advice


def test_transfer_timing_advice_none_when_nothing_urgent():
    assert tp.transfer_timing_advice({"price_change_direction": "none"}, {"price_change_direction": "none"}) is None
    assert tp.transfer_timing_advice(None, None) is None


def test_evaluate_transfers_attaches_price_change_risk_without_changing_ranking(con):
    """Same informational-only convention as momentum/swing above -- the ranking (net_value,
    sort order) must be identical whether or not the price-change keys are attached."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]
    con.execute(
        "UPDATE fact_player_season_stats SET transfers_in_event = 90000, transfers_out_event = 500 "
        "WHERE player_uid = 'p_in_same_pos_cheaper'"
    )

    without = tp.evaluate_transfers(con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4)
    with_risk = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4,
        target_gameweek=1, price_change_rise_threshold=50_000.0, price_change_fall_threshold=50_000.0,
    )

    assert [r["net_value"] for r in without] == [r["net_value"] for r in with_risk]
    assert [r["player_in"] for r in without] == [r["player_in"] for r in with_risk]
    assert "price_change_risk_in" not in without[0]

    top = next(r for r in with_risk if r["player_in"] == "p_in_same_pos_cheaper")
    assert top["price_change_risk_in"]["price_change_direction"] == "rise"
    assert top["timing_advice"] is not None
    assert "today, not tomorrow" in top["timing_advice"]


def test_evaluate_transfers_price_change_risk_requires_both_thresholds(con):
    """Partial opt-in (only one threshold given) must behave exactly like not opting in at
    all -- no silent guessing of the missing threshold."""
    holdings_dict, horizon_ep_versions = _seed_ep_and_holdings_for_transfers(con)
    current_holdings = [holdings_dict["p_out"]]
    results = tp.evaluate_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4,
        target_gameweek=1, price_change_rise_threshold=50_000.0,  # fall_threshold left None
    )
    assert "price_change_risk_in" not in results[0]


# ============================================================
# Priority 3 -- evaluate_multi_transfers / evaluate_hold_recommendation
# ============================================================

def _seed_multi_gw_pool(con, mid_a_ep, def_a_ep, target_season="2026-2027"):
    """Two held players (out_mid, out_def) and a real incoming candidate pool across two
    gameweeks (2 and 3), each gameweek its own real ep_model_version/uncertainty_model_version
    pair so _horizon_ep_by_player() reads through squad_optimizer.fetch_candidate_pool()
    exactly as the real pipeline would. mid_a_ep/def_a_ep: {2: ep, 3: ep} per-gameweek
    projections for the two "good" incoming candidates -- the caller controls whether the
    real gain is front-loaded (favors transferring now) or back-loaded (favors holding for a
    next-week double move), the scenario evaluate_hold_recommendation() itself is about."""
    teams = ["team_a", "team_b"]
    for t in teams:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [t, t])
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

    # uid -> (position, price, {gw: ep})
    players = {
        "out_mid": ("Midfielder", 5.0, {2: 4.0, 3: 4.0}),
        "out_def": ("Defender", 5.0, {2: 4.0, 3: 4.0}),
        "in_mid_a": ("Midfielder", 5.0, mid_a_ep),
        "in_def_a": ("Defender", 5.0, def_a_ep),
        "in_mid_weak": ("Midfielder", 5.0, {2: 1.0, 3: 1.0}),  # always worse -- must never win
    }
    for uid, (position, price, _by_gw) in players.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, '1', ?, ?)", [uid, uid.lower(), target_season, uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp)", [uid, target_season, price],
        )

    horizon_ep_versions = {}
    for gw in (2, 3):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'team_a', 'team_b', FALSE, "
            "'Premier League', '2026-08-24', current_timestamp)", [f"m{gw}", target_season, gw],
        )
        ep_mv = con.execute(
            "INSERT INTO ep_model_versions (calibration_asof_date, target_season, team_strength_model_version, "
            "minutes_model_version, scoring_matrix_params_version, bps_params_version, bps_tau_params_version) "
            "VALUES ('2026-08-10', ?, ?, ?, 1, 1, 1) RETURNING model_version", [target_season, ts_mv, mm_mv],
        ).fetchone()[0]
        un_mv = con.execute(
            "INSERT INTO uncertainty_model_versions (calibration_asof_date, ep_model_version, minutes_model_version, "
            "team_strength_model_version, rho_residual_params_version) VALUES ('2026-08-10', ?, ?, ?, 1) "
            "RETURNING model_version", [ep_mv, mm_mv, ts_mv],
        ).fetchone()[0]
        for uid, (_pos, _price, by_gw) in players.items():
            ep_val = by_gw[gw]
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
                [ep_mv, uid, f"m{gw}", ep_val],
            )
            con.execute(
                "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
                "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
                "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
                "VALUES (?, ?, ?, 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)", [un_mv, uid, f"m{gw}"],
            )
        horizon_ep_versions[gw] = (ep_mv, un_mv)

    current_holdings = [
        {"player_uid": "out_mid", "in_xi": True, "is_captain": False, "is_vice": False},
        {"player_uid": "out_def", "in_xi": True, "is_captain": False, "is_vice": False},
    ]
    return current_holdings, horizon_ep_versions


def test_evaluate_multi_transfers_finds_a_legal_2_for_2_combo(con):
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 7.0, 3: 7.0}, def_a_ep={2: 7.0, 3: 7.0},
    )
    results = tp.evaluate_multi_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=2, points_per_hit=4,
    )
    assert results
    top = results[0]
    assert top["players_out"] == sorted(["out_mid", "out_def"])
    assert top["players_in"] == sorted(["in_mid_a", "in_def_a"])
    # (7+7 - 4-4) for each of the 2 gameweeks, both players: (14-8) + (14-8) = 12
    assert top["horizon_value_gain"] == pytest.approx(12.0)
    assert top["transfer_cost"] == 0.0  # 2 free transfers available, no hit
    assert top["net_value"] == pytest.approx(12.0)


def test_evaluate_multi_transfers_applies_a_hit_when_short_on_free_transfers(con):
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 7.0, 3: 7.0}, def_a_ep={2: 7.0, 3: 7.0},
    )
    results = tp.evaluate_multi_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1, points_per_hit=4,
    )
    top = results[0]
    assert top["transfer_cost"] == pytest.approx(4.0)  # 1 short of the 2 transfers needed -> 1 hit
    assert top["net_value"] == pytest.approx(12.0 - 4.0)


def test_evaluate_multi_transfers_never_mixes_position_multisets(con):
    """Selling 1 MID + 1 DEF must only ever be replaced by 1 MID + 1 DEF -- squad position
    quotas are fixed, so a combo that unbalances them is illegal regardless of net value."""
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 7.0, 3: 7.0}, def_a_ep={2: 7.0, 3: 7.0},
    )
    results = tp.evaluate_multi_transfers(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=2, points_per_hit=4,
    )
    for r in results:
        in_positions = sorted(
            "Midfielder" if uid in ("in_mid_a", "in_mid_weak", "out_mid") else "Defender" for uid in r["players_in"]
        )
        assert in_positions == ["Defender", "Midfielder"]


def test_evaluate_hold_recommendation_prefers_transfer_now_on_a_tie(con):
    """Flat, front-loaded EP (equal both gameweeks): transferring now captures gw2 AND gw3's
    worth of gain; holding only ever captures gw3's worth, whichever branch of the hold
    option is taken -- so 'transfer_now' and 'hold' land on the identical net_value here, and
    a tie must resolve toward acting now, per spec."""
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 7.0, 3: 7.0}, def_a_ep={2: 7.0, 3: 7.0},
    )
    result = tp.evaluate_hold_recommendation(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1,
        points_per_hit=4, target_gameweek=2,
    )
    assert result["recommended_action"] == "transfer_now"
    assert result["transfer_now_value"] == pytest.approx(result["hold_value"])


def test_evaluate_hold_recommendation_prefers_hold_for_a_backloaded_double_move(con):
    """Real motivating case: the good incoming pair's value is almost entirely NEXT week
    (back-loaded), and only 1 free transfer is available now -- capturing BOTH players
    requires holding this week to reach 2 free transfers next week and take the 2-for-2 combo,
    which beats anything a single transfer now (or next week) could achieve alone."""
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 2.0, 3: 15.0}, def_a_ep={2: 2.0, 3: 15.0},
    )
    result = tp.evaluate_hold_recommendation(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1,
        points_per_hit=4, target_gameweek=2,
    )
    assert result["recommended_action"] == "hold"
    assert result["hold_value"] > result["transfer_now_value"]
    assert result["best_hold_multi_next_week"] is not None
    assert result["best_hold_multi_next_week"]["players_in"] == sorted(["in_mid_a", "in_def_a"])
    # the hold branch's winning option must be the MULTI combo, not the single-transfer one --
    # proves evaluate_hold_recommendation() actually reaches for the 2-for-2 case, not just a
    # same-shape single swap that happens to also be available next week.
    assert result["hold_value"] == pytest.approx(result["best_hold_multi_next_week"]["net_value"])


def _seed_run_ready_state(con):
    """A real, run()-able manager_state_versions row over
    _seed_real_squad_optimizer_candidate_pool's own candidate pool -- everything
    evaluate_wildcard()/evaluate_free_hit() need (real squad_optimizer.run() solves) is
    already proven working against this fixture by the tests above; the ONE piece it does
    NOT provide is the raw teams.csv/team_alias/roster chain expected_points.run() itself
    needs to derive EP from scratch (compute_horizon_ep()'s own job) -- monkeypatched away in
    the two tests below rather than built from scratch here, since _seed_real_squad_optimizer_
    candidate_pool already returns the EXACT {gw: (ep_mv, un_mv)} shape compute_horizon_ep()
    itself would produce, over already-real ep_outputs/uncertainty_outputs data."""
    from fpl_quant import expected_points as ep_mod
    from fpl_quant import uncertainty as un_mod

    horizon_ep_versions, holdings = _seed_real_squad_optimizer_candidate_pool(con)
    so.seed_v1_params(con)
    tp.seed_v1_params(con)
    ep_mod.seed_v1_params(con)
    un_mod.seed_v1_params(con)
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    mm_mv = con.execute("SELECT max(model_version) FROM minutes_model_versions").fetchone()[0]

    # bank=5.0: the fixture's held players (players[:11], per _seed_real_squad_optimizer_
    # candidate_pool's own docstring) are exactly the LOWEST-priced player at each held
    # position -- every remaining, unheld same-position candidate costs MORE, so a real
    # 2-for-2 upgrade combo needs real bank to be affordable at all (a small, deliberately
    # nonzero float here, not a workaround for a bug: evaluate_multi_transfers()'s own price
    # constraint is correct, this just gives the wiring test a genuine affordable combo to find).
    state_version = con.execute(
        "INSERT INTO manager_state_versions (season, as_of_gameweek, free_transfers_available, "
        "chips_used_set1, chips_used_set2, bank) VALUES ('2026-2027', 2, 1, '[]', '[]', 5.0) "
        "RETURNING state_version"
    ).fetchone()[0]
    for h in holdings:
        con.execute(
            "INSERT INTO manager_squad_holdings (state_version, player_uid, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, ?, ?, ?)", [state_version, h["player_uid"], h["in_xi"], h["is_captain"], h["is_vice"]],
        )
    return state_version, ts_mv, mm_mv, horizon_ep_versions


def test_run_wires_multi_transfer_and_hold_when_opted_in(con, monkeypatch):
    """multi_transfer_pool_limit_per_position defaults to None (skip both -- see run()'s own
    docstring on why: a real, bounded combinatorial search is genuinely expensive relative to
    everything else run() does, and every existing caller -- a season simulation calling
    run() many times over in particular -- must keep paying only what it paid before this
    feature existed unless it opts in). Passing an int computes and persists both, and
    explain_plan() must surface them."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        multi_transfer_pool_limit_per_position=10,
    )

    multi_rows = con.execute("SELECT count(*) FROM multi_transfer_recommendations WHERE run_id = ?", [run_id]).fetchone()[0]
    hold_row = con.execute("SELECT recommended_action FROM hold_recommendations WHERE run_id = ?", [run_id]).fetchone()
    assert multi_rows > 0
    assert hold_row is not None
    assert hold_row[0] in ("hold", "transfer_now", "no_action_available")

    plan = tp.explain_plan(con, run_id)
    assert plan["top_multi_transfers"]
    assert plan["hold_recommendation"] is not None
    assert plan["hold_recommendation"]["recommended_action"] == hold_row[0]


def test_run_skips_multi_transfer_and_hold_by_default(con, monkeypatch):
    """Backward compatibility: the original run() call shape (no keyword arg at all) must
    leave both new tables empty, not silently compute and discard the results."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    multi_rows = con.execute("SELECT count(*) FROM multi_transfer_recommendations WHERE run_id = ?", [run_id]).fetchone()[0]
    hold_row = con.execute("SELECT 1 FROM hold_recommendations WHERE run_id = ?", [run_id]).fetchone()
    assert multi_rows == 0
    assert hold_row is None

    plan = tp.explain_plan(con, run_id)
    assert plan["top_multi_transfers"] == []
    assert plan["hold_recommendation"] is None


def test_run_wires_chip_combos_when_opted_in(con, monkeypatch):
    """evaluate_chip_combos defaults to False (see run()'s own docstring -- the Free-Hit x
    Triple-Captain combo needs a genuinely new Monte Carlo simulation, the same cost class as
    multi_transfer_pool_limit_per_position). True computes and persists both combo
    evaluations, and explain_plan() must surface them."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    # evaluate_free_hit_triple_captain_combo()'s own real Monte Carlo simulation needs the
    # team-resolution and roster chains _seed_real_squad_optimizer_candidate_pool doesn't
    # provide (see test_evaluate_free_hit_triple_captain_combo_uses_a_real_simulation's own
    # comments above for exactly why each piece here is needed).
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    _seed_raw_teams_csv(con, "2026-2027", [(c, c) for c in clubs])
    for c in clubs:
        con.execute(
            "INSERT INTO team_alias (alias_name, season, team_uid, alias_source) VALUES (?, '2026-2027', ?, 't')",
            [c, c],
        )
    for c in ("clubA", "clubB"):
        con.execute(
            "INSERT INTO team_strength_snapshots (model_version, team_uid, final_attack, final_defence, "
            "seasons_of_topflight_data, weight_own_data) VALUES (?, ?, 0.1, 0.0, 2, 1.0)",
            [ts_mv, c],
        )
    tp.params_mod.write_param(con, "model_decay_params", 1, "2026-08-10", "rho", value_numeric=-0.13)
    ep_mv, _un_mv = horizon_ep_versions[2]
    all_uids = [r[0] for r in con.execute("SELECT DISTINCT player_uid FROM ep_outputs WHERE model_version = ?", [ep_mv]).fetchall()]
    for uid in all_uids:
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
            "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, 'Midfielder', 0.8, 1.0, 0.8, 0.0, 0.8, 0.0, 0.1, 0.1, 0.8, 20)",
            [mm_mv, uid],
        )

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        evaluate_chip_combos=True,
    )

    combo_rows = con.execute(
        "SELECT combo_type, recommended_combo FROM chip_combo_evaluations WHERE run_id = ? ORDER BY combo_type", [run_id]
    ).fetchall()
    assert {r[0] for r in combo_rows} == {"wildcard_bench_boost", "free_hit_triple_captain"}

    plan = tp.explain_plan(con, run_id)
    assert plan["chip_combos"] is not None
    assert set(plan["chip_combos"]) == {"wildcard_bench_boost", "free_hit_triple_captain"}
    assert plan["chip_combos"]["wildcard_bench_boost"]["recommended_combo"] == dict(combo_rows)["wildcard_bench_boost"]


def test_run_skips_chip_combos_by_default(con, monkeypatch):
    """Backward compatibility: the original run() call shape must leave chip_combo_evaluations
    empty, and explain_plan()'s chip_combos section None."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    combo_rows = con.execute("SELECT count(*) FROM chip_combo_evaluations WHERE run_id = ?", [run_id]).fetchone()[0]
    assert combo_rows == 0

    plan = tp.explain_plan(con, run_id)
    assert plan["chip_combos"] is None


def test_evaluate_hold_recommendation_no_action_available_when_no_legal_candidate_exists(con):
    """No candidate anywhere in the pool is affordable (every incoming player costs more than
    what's held, with no bank) -- both evaluate_transfers() and evaluate_multi_transfers()
    legitimately return empty, so there's no real 'now' or 'hold' value to compare at all."""
    current_holdings, horizon_ep_versions = _seed_multi_gw_pool(
        con, mid_a_ep={2: 7.0, 3: 7.0}, def_a_ep={2: 7.0, 3: 7.0},
    )
    con.execute("UPDATE fact_player_season_stats SET now_cost = 50.0 WHERE player_uid != 'out_mid' AND player_uid != 'out_def'")
    result = tp.evaluate_hold_recommendation(
        con, current_holdings, "2026-2027", horizon_ep_versions, free_transfers_available=1,
        points_per_hit=4, target_gameweek=2, bank=0.0,
    )
    assert result["recommended_action"] == "no_action_available"
    assert result["transfer_now_value"] is None
    assert result["hold_value"] is None


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


def test_evaluate_triple_captain_captain_value_per_gw_reads_the_winning_candidates_own_trajectory(con):
    """Regression test for the chip-timing work: captain_value_per_gw must be the WINNING
    candidate's own mu trajectory (an EP proxy, not a fresh Monte Carlo re-simulation at every
    horizon gameweek -- see evaluate_triple_captain()'s own docstring for why), read straight
    off horizon_ep_map, not the runner-up's or some squad-wide aggregate."""
    tp.seed_v1_params(con)
    for uid in ("high_mean_high_var", "mod_mean_low_var"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    model_version = _seed_mc_run_and_summary(con, [
        ("high_mean_high_var", 10.0, 25.0),
        ("mod_mean_low_var", 8.0, 4.0),
    ])
    horizon_ep_map = {
        "high_mean_high_var": {"per_gw": {2: 9.0, 3: 4.0}},
        "mod_mean_low_var": {"per_gw": {2: 1.0, 3: 20.0}},
    }
    result = tp.evaluate_triple_captain(
        con, model_version, xi_uids={"high_mean_high_var", "mod_mean_low_var"}, kappa_tc_params_version=1,
        horizon_ep_map=horizon_ep_map,
    )
    assert result["captain_candidate"] == "high_mean_high_var"
    assert result["captain_value_per_gw"] == {2: 9.0, 3: 4.0}  # NOT mod_mean_low_var's trajectory


def test_evaluate_triple_captain_captain_value_per_gw_defaults_empty_without_horizon_ep_map(con):
    tp.seed_v1_params(con)
    model_version = _seed_mc_run_and_summary(con, [("p1", 10.0, 4.0)])  # p1 is seeded by the helper itself
    result = tp.evaluate_triple_captain(con, model_version, xi_uids={"p1"}, kappa_tc_params_version=1)
    assert result["captain_value_per_gw"] == {}


# ============================================================
# vice_captain_fallback_adjustment -- real gap fixed: nowhere in this project (squad_optimizer's
# MIQP objective, monte_carlo.py, evaluate_triple_captain()) ever credited a captain choice with
# the real armband-transfers-to-vice-on-DNP safety net. See the function's own docstring.
# ============================================================

def _seed_minutes_output(con, mm_mv, player_uid, p_0min):
    con.execute(
        "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
        "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
        "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
        "VALUES (?, ?, 'Midfielder', 0.8, 1.0, 0.8, 0.0, 0.8, 0.0, ?, ?, ?, 20)",
        [mm_mv, player_uid, p_0min, (1 - p_0min) * 0.2, (1 - p_0min) * 0.8],
    )


def test_vice_captain_fallback_adjustment_is_p0min_times_vice_mean_total(con):
    tp.seed_v1_params(con)
    for uid in ("captain_a", "vice_a"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    mc_mv = _seed_mc_run_and_summary(con, [("captain_a", 10.0, 4.0), ("vice_a", 5.0, 1.0)])
    mm_mv = con.execute("SELECT minutes_model_version FROM monte_carlo_run_versions WHERE model_version = ?", [mc_mv]).fetchone()[0]
    _seed_minutes_output(con, mm_mv, "captain_a", p_0min=0.1)

    out = tp.vice_captain_fallback_adjustment(con, mm_mv, mc_mv, vice_uid="vice_a", candidate_uids={"captain_a"})
    assert out == pytest.approx({"captain_a": 0.1 * 5.0})


def test_vice_captain_fallback_adjustment_returns_empty_with_no_vice(con):
    tp.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('captain_a', 'captain_a', 'Midfielder')")
    mc_mv = _seed_mc_run_and_summary(con, [("captain_a", 10.0, 4.0)])
    mm_mv = con.execute("SELECT minutes_model_version FROM monte_carlo_run_versions WHERE model_version = ?", [mc_mv]).fetchone()[0]
    assert tp.vice_captain_fallback_adjustment(con, mm_mv, mc_mv, vice_uid=None, candidate_uids={"captain_a"}) == {}


def test_vice_captain_fallback_adjustment_returns_empty_when_vice_never_simulated(con):
    tp.seed_v1_params(con)
    con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES ('captain_a', 'captain_a', 'Midfielder')")
    mc_mv = _seed_mc_run_and_summary(con, [("captain_a", 10.0, 4.0)])
    mm_mv = con.execute("SELECT minutes_model_version FROM monte_carlo_run_versions WHERE model_version = ?", [mc_mv]).fetchone()[0]
    _seed_minutes_output(con, mm_mv, "captain_a", p_0min=0.1)
    # "unsimulated_vice" has no monte_carlo_player_summary row at all -- an absent correction,
    # never a fabricated 0.0 (see the function's own docstring).
    out = tp.vice_captain_fallback_adjustment(con, mm_mv, mc_mv, vice_uid="unsimulated_vice", candidate_uids={"captain_a"})
    assert out == {}


def test_vice_captain_fallback_adjustment_omits_candidates_with_no_minutes_data(con):
    tp.seed_v1_params(con)
    for uid in ("captain_a", "captain_b", "vice_a"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
    mc_mv = _seed_mc_run_and_summary(con, [("captain_a", 10.0, 4.0), ("captain_b", 9.0, 4.0), ("vice_a", 5.0, 1.0)])
    mm_mv = con.execute("SELECT minutes_model_version FROM monte_carlo_run_versions WHERE model_version = ?", [mc_mv]).fetchone()[0]
    _seed_minutes_output(con, mm_mv, "captain_a", p_0min=0.2)
    # captain_b deliberately has no minutes_model_outputs row -- must be omitted, not defaulted.
    out = tp.vice_captain_fallback_adjustment(con, mm_mv, mc_mv, vice_uid="vice_a", candidate_uids={"captain_a", "captain_b"})
    assert set(out) == {"captain_a"}
    assert out["captain_a"] == pytest.approx(0.2 * 5.0)


def test_run_attaches_vice_fallback_adjusted_candidates_without_changing_the_recommendation(con, monkeypatch):
    """Wiring regression: run()'s own tc_result (persisted as chip_evaluations.detail for
    chip_type='triple_captain') must gain the new fields additively -- the original
    recommended/captain_candidate/tc_score stay exactly what evaluate_triple_captain() alone
    would produce, while vice_fallback_adjusted_candidates carries the real armband-correction
    re-ranking as a SEPARATE, clearly labeled field. evaluate_triple_captain() and
    ensure_squad_simulation() are monkeypatched here -- their own real behavior is already
    covered by their own dedicated tests; this test isolates the NEW composition logic run()
    itself now does on top of their output, the same way test_run_wires_multi_transfer_and_
    hold_when_opted_in() isolates its own new logic by monkeypatching compute_horizon_ep()."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(tp, "ensure_squad_simulation", lambda *a, **k: 999)

    held_uids = [h["player_uid"] for h in tp._read_holdings(con, state_version) if h["in_xi"]]
    captain_uid, vice_uid = held_uids[0], held_uids[1]
    con.execute("UPDATE manager_squad_holdings SET is_vice = TRUE WHERE state_version = ? AND player_uid = ?", [state_version, vice_uid])
    _seed_minutes_output(con, mm_mv, captain_uid, p_0min=0.2)

    fake_tc_result = {
        "recommended": True, "captain_candidate": captain_uid, "tc_score": 9.0,
        "all_candidates": [{"player_uid": captain_uid, "tc_score": 9.0, "mean_total": 10.0, "var_total": 4.0}],
        "captain_value_per_gw": {},
    }
    monkeypatch.setattr(tp, "evaluate_triple_captain", lambda *a, **k: dict(fake_tc_result))
    monkeypatch.setattr(tp, "vice_captain_fallback_adjustment", lambda con, mm_mv, mc_mv, vice_uid, candidate_uids: {captain_uid: 1.5})

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    detail_json = con.execute(
        "SELECT detail FROM chip_evaluations WHERE run_id = ? AND chip_type = 'triple_captain'", [run_id],
    ).fetchone()[0]
    tc_result = json.loads(detail_json)

    # The original evaluate_triple_captain() fields are untouched.
    assert tc_result["captain_candidate"] == captain_uid
    assert tc_result["tc_score"] == pytest.approx(9.0)
    # The new, additive fields carry the real armband correction.
    assert tc_result["vice_captain_uid"] == vice_uid
    assert tc_result["vice_fallback_adjusted_candidates"] == [
        {"player_uid": captain_uid, "tc_score": 9.0, "mean_total": 10.0, "var_total": 4.0,
         "vice_fallback_correction": 1.5, "tc_score_vice_adjusted": pytest.approx(10.5)},
    ]


def test_run_leaves_vice_fallback_fields_none_when_no_correction_is_computable(con, monkeypatch):
    """No designated vice-captain (or the vice was never simulated) -- vice_captain_fallback_
    adjustment() returns {} (see its own docstring: absence, never a fabricated 0.0), and run()
    must honestly propagate that as vice_fallback_adjusted_candidates=None, not silently
    default every candidate's correction to 0.0."""
    state_version, ts_mv, mm_mv, horizon_ep_versions = _seed_run_ready_state(con)
    monkeypatch.setattr(tp, "compute_horizon_ep", lambda *a, **k: horizon_ep_versions)
    monkeypatch.setattr(tp, "ensure_squad_simulation", lambda *a, **k: 999)

    held_uids = [h["player_uid"] for h in tp._read_holdings(con, state_version) if h["in_xi"]]
    captain_uid = held_uids[0]
    fake_tc_result = {
        "recommended": True, "captain_candidate": captain_uid, "tc_score": 9.0,
        "all_candidates": [{"player_uid": captain_uid, "tc_score": 9.0, "mean_total": 10.0, "var_total": 4.0}],
        "captain_value_per_gw": {},
    }
    monkeypatch.setattr(tp, "evaluate_triple_captain", lambda *a, **k: dict(fake_tc_result))
    monkeypatch.setattr(tp, "vice_captain_fallback_adjustment", lambda *a, **k: {})  # no vice designated in this fixture

    run_id = tp.run(
        con, date(2026, 8, 24), "2026-2027", 2, state_version, ts_mv, mm_mv, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    )
    detail_json = con.execute(
        "SELECT detail FROM chip_evaluations WHERE run_id = ? AND chip_type = 'triple_captain'", [run_id],
    ).fetchone()[0]
    tc_result = json.loads(detail_json)
    assert tc_result["vice_captain_uid"] is None
    assert tc_result["vice_fallback_adjusted_candidates"] is None


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


def test_check_gw19_deadline_urgent_and_forfeited_never_overlap():
    """Regression test for a real bug: urgent's inclusive `0 <=` lower bound meant GW19 itself
    (gameweeks_remaining == 0) was flagged both urgent ("hurry, use it now") and forfeited_now
    ("already gone") simultaneously -- a self-contradictory pair written into
    chip_evaluations.gw19_urgent_flag. Checks every gameweek in the warning window plus the
    deadline itself."""
    for gw in range(15, 21):
        result = tp.check_gw19_deadline(target_gameweek=gw, chips_used_set1=[])
        assert not (result["urgent"] and result["forfeited_now"]), f"gw={gw}: both flags true"
    at_deadline = tp.check_gw19_deadline(target_gameweek=19, chips_used_set1=[])
    assert at_deadline["urgent"] is False
    assert at_deadline["forfeited_now"] is True


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
    # bench_boost, not wildcard: this test is about generic chips_used_set bookkeeping, not
    # Wildcard's own real squad-rebuild behavior (see the dedicated wildcard tests below,
    # which need a real chip_evaluations + fresh squad_optimizer_selections fixture).
    run_id2, _, _ = _seed_minimal_squad_optimizer_run(con)
    state_version = tp.bootstrap_from_squad_optimizer_run(con, run_id2)
    run_id = _seed_transfer_plan_run_for_apply(con, state_version, target_gameweek=10)

    new_state_version = tp.apply_recommendation(con, run_id, accept_transfer_rank=None, accept_chip="bench_boost")
    chips_used_set1 = con.execute(
        "SELECT chips_used_set1 FROM manager_state_versions WHERE state_version = ?", [new_state_version]
    ).fetchone()[0]
    assert "bench_boost" in chips_used_set1
