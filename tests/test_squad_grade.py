from datetime import date

import pytest

from fpl_quant import squad_grade as sg
from fpl_quant import squad_optimizer as so_mod


def _pool():
    """2 GK/6 DEF/6 MID/4 FWD across 6 clubs -- same proven shape as squad_optimizer's own
    test_squad_optimizer.py::_synthetic_pool() (real slack above the 15/11 minimums so the
    real SCIP solve has actual choices to make, >=5 clubs so the <=3-per-club cap stays
    feasible). ep increases with i within each position so there's a real, predictable
    "best 5 defenders"/"best 5 midfielders"/etc. for the optimal solve to pick."""
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    pool = []

    def add(pid, pos, ep, price, club):
        pool.append({"player_uid": pid, "position": pos, "ep": ep, "price": price, "club": club})

    for i in range(2):
        add(f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6])
    for i in range(6):
        add(f"def{i}", "Defender", 2.0 + i * 0.5, 4.0 + i * 0.3, clubs[i % 6])
    for i in range(6):
        add(f"mid{i}", "Midfielder", 2.5 + i * 0.5, 5.0 + i * 0.3, clubs[i % 6])
    for i in range(4):
        add(f"fwd{i}", "Forward", 3.0 + i * 0.5, 6.0 + i * 0.3, clubs[i % 6])
    return pool


def _seed_squad_grade_scenario(con, target_season="2026-2027", target_gameweek=2):
    so_mod.seed_v1_params(con)
    pool = _pool()
    clubs = sorted({c["club"] for c in pool})
    for club in clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?) ON CONFLICT DO NOTHING", [club, club])
    for c in pool:
        con.execute(
            "INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            [c["player_uid"], c["player_uid"], c["position"]],
        )
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [c["player_uid"], c["player_uid"].lower(), c["club"], target_season, c["player_uid"]],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, _ingested_at) "
            "VALUES (?, ?, 1, ?, current_timestamp) ON CONFLICT DO NOTHING", [c["player_uid"], target_season, c["price"]],
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
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, ?, ?, FALSE, 'Premier League', "
        "'2026-08-24', current_timestamp)",
        [f"m{target_gameweek}", target_season, target_gameweek, clubs[0], clubs[-1]],
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
    for c in pool:
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, c["player_uid"], f"m{target_gameweek}", c["ep"]],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, ?, 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)",
            [un_mv, c["player_uid"], f"m{target_gameweek}"],
        )
    # Real covariance between the two best midfielders (the pair the lambda=0 EP-only solve
    # would otherwise stack together) -- without this, every candidate shares the same flat
    # var_total=1.0 with zero cross-covariance, so the quadratic risk term has nothing to
    # differentiate on and lambda=0 vs lambda=0.15 pick the identical squad, tripping M5's own
    # divergence-check hard-stop (same reason test_squad_optimizer.py's own fixtures always
    # seed a real sigma_pairs entry, not just flat variances).
    con.execute(
        "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, fixture_match_id, "
        "relationship, covariance) VALUES (?, 'mid4', 'mid5', ?, 'teammate', 3.0)", [un_mv, f"m{target_gameweek}"],
    )
    return {"gw": {target_gameweek: (ep_mv, un_mv)}, "pool": pool}


def _weakest_legal_holdings(pool):
    """A legal (2/5/5/3, budget-respecting, <=3-per-club) but deliberately SUBOPTIMAL squad:
    picks the lowest-ep 5 defenders/5 midfielders/3 forwards/2 GKs instead of the best ones
    -- guarantees a real, positive points_gap for the optimal solve to close."""
    by_pos = {}
    for c in pool:
        by_pos.setdefault(c["position"], []).append(c)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda c: c["ep"])  # weakest first
    quota = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
    holdings = []
    for pos, n in quota.items():
        for c in by_pos[pos][:n]:
            holdings.append({"player_uid": c["player_uid"], "in_xi": True, "is_captain": False, "is_vice": False})
    return holdings


def test_grade_squad_produces_a_nonnegative_gap_and_a_letter_grade(con):
    scenario = _seed_squad_grade_scenario(con)
    holdings = _weakest_legal_holdings(scenario["pool"])

    grade = sg.grade_squad(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, current_holdings=holdings, horizon_ep_versions=scenario["gw"],
        lambda_params_version=1, guardrail_params_version=1,
    )
    assert grade.points_gap >= 0
    assert grade.grade in ("A", "B", "C", "D")
    assert grade.optimal_ep >= grade.user_squad_ep
    assert grade.provenance.data_asof == "2026-08-24"


def test_grade_squad_is_deterministic_across_repeated_calls(con):
    scenario = _seed_squad_grade_scenario(con)
    holdings = _weakest_legal_holdings(scenario["pool"])

    def _run():
        g = sg.grade_squad(
            con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
            target_gameweek=2, current_holdings=holdings, horizon_ep_versions=scenario["gw"],
            lambda_params_version=1, guardrail_params_version=1,
        )
        return (g.grade, g.points_gap, g.optimal_ep, g.user_squad_ep, [(s.out_player_uid, s.in_player_uid) for s in g.top_swaps])

    assert _run() == _run()


def test_grade_squad_top_swap_actually_closes_the_gap_when_applied(con):
    scenario = _seed_squad_grade_scenario(con)
    holdings = _weakest_legal_holdings(scenario["pool"])

    grade = sg.grade_squad(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, current_holdings=holdings, horizon_ep_versions=scenario["gw"],
        lambda_params_version=1, guardrail_params_version=1,
    )
    assert grade.top_swaps, "the deliberately-weakest legal squad must have at least one real improving swap"
    top = grade.top_swaps[0]

    swapped_holdings = [
        h for h in holdings if h["player_uid"] != top.out_player_uid
    ] + [{"player_uid": top.in_player_uid, "in_xi": True, "is_captain": False, "is_vice": False}]

    regraded = sg.grade_squad(
        con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
        target_gameweek=2, current_holdings=swapped_holdings, horizon_ep_versions=scenario["gw"],
        lambda_params_version=1, guardrail_params_version=1,
    )
    assert regraded.points_gap < grade.points_gap
    assert regraded.user_squad_ep == pytest.approx(grade.user_squad_ep + top.delta_ep)


def test_grade_squad_raises_on_missing_gameweek(con):
    scenario = _seed_squad_grade_scenario(con)
    holdings = _weakest_legal_holdings(scenario["pool"])
    with pytest.raises(ValueError, match="no fixtures"):
        sg.grade_squad(
            con, entry_id=123, calibration_asof_date=date(2026, 8, 24), target_season="2026-2027",
            target_gameweek=99, current_holdings=holdings, horizon_ep_versions=scenario["gw"],
            lambda_params_version=1, guardrail_params_version=1,
        )


def test_letter_grade_bands():
    assert sg._letter_grade(0.0) == "A"
    assert sg._letter_grade(1.9) == "A"
    assert sg._letter_grade(2.0) == "B"
    assert sg._letter_grade(4.9) == "B"
    assert sg._letter_grade(5.0) == "C"
    assert sg._letter_grade(9.9) == "C"
    assert sg._letter_grade(10.0) == "D"
    assert sg._letter_grade(50.0) == "D"


def test_find_top_swaps_only_matches_same_position():
    horizon_ep_map = {
        "out_def": {"total_ep": 2.0, "position": "Defender", "price": 4.0},
        "in_fwd": {"total_ep": 10.0, "position": "Forward", "price": 6.0},  # higher ep, WRONG position
        "in_def": {"total_ep": 5.0, "position": "Defender", "price": 4.5},
    }
    swaps = sg._find_top_swaps({"out_def"}, {"in_fwd", "in_def"}, horizon_ep_map, n=3)
    assert len(swaps) == 1
    assert swaps[0].in_player_uid == "in_def"


def test_find_top_swaps_never_reuses_a_player_across_multiple_swaps():
    horizon_ep_map = {
        "out1": {"total_ep": 1.0, "position": "Defender", "price": 4.0},
        "out2": {"total_ep": 1.0, "position": "Defender", "price": 4.0},
        "in1": {"total_ep": 9.0, "position": "Defender", "price": 4.0},  # best -- both outs would want it
    }
    swaps = sg._find_top_swaps({"out1", "out2"}, {"in1"}, horizon_ep_map, n=3)
    assert len(swaps) == 1  # only one swap possible -- in1 can't be used twice
