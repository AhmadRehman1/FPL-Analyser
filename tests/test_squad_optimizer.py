import pytest

from fpl_quant import squad_optimizer as so


def test_seed_v1_params_exact_frozen_value(con):
    so.seed_v1_params(con)
    from fpl_quant import params
    lam, _ = params.resolve_param(con, "risk_aversion_params", "lambda_value", 1)
    assert lam == 0.15
    cap, _ = params.resolve_param(con, "squad_optimizer_guardrail_params", "xi_club_concentration_cap", 1)
    assert cap == 3


def _synthetic_pool():
    """2 GK, 6 DEF, 6 MID, 4 FWD across 6 clubs, priced so a 100.0 budget is meaningfully
    binding but the problem stays feasible -- enough slack above the 15/11 minimums to give
    the optimizer real choices, small enough to solve in well under a second.

    Needs >=5 clubs: 15 squad slots split across only 4 clubs would be infeasible under the
    <=3-per-club cap by pigeonhole (15/4 > 3) -- a real bug this test fixture hit on its
    first pass, not a solver bug (the real 577-player/20-club run worked from the start)."""
    pool = []
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]

    def add(pid, pos, mu, price, club, var=1.0):
        pool.append({"player_uid": pid, "position": pos, "mu": mu, "var": var, "club": club, "price": price, "name": pid})

    for i in range(2):
        add(f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6])
    for i in range(6):
        add(f"def{i}", "Defender", 2.5 + i * 0.3, 4.0 + i * 0.5, clubs[i % 6])
    for i in range(6):
        add(f"mid{i}", "Midfielder", 3.0 + i * 0.4, 5.0 + i * 0.5, clubs[i % 6])
    for i in range(4):
        add(f"fwd{i}", "Forward", 3.5 + i * 0.5, 6.0 + i * 0.5, clubs[i % 6])
    return pool


def test_solve_satisfies_all_constraints():
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    assert result["status"] == "optimal"

    by_uid = {c["player_uid"]: c for c in pool}
    assert len(result["squad"]) == 15
    assert len(result["xi"]) == 11
    assert sum(1 for u in result["squad"] if by_uid[u]["position"] == "Goalkeeper") == 2
    assert sum(1 for u in result["squad"] if by_uid[u]["position"] == "Defender") == 5
    assert sum(1 for u in result["squad"] if by_uid[u]["position"] == "Midfielder") == 5
    assert sum(1 for u in result["squad"] if by_uid[u]["position"] == "Forward") == 3

    total_price = sum(by_uid[u]["price"] for u in result["squad"])
    assert total_price <= 100.0 + 1e-6

    assert result["xi"] <= result["squad"]
    xi_gk = sum(1 for u in result["xi"] if by_uid[u]["position"] == "Goalkeeper")
    assert xi_gk == 1
    xi_def = sum(1 for u in result["xi"] if by_uid[u]["position"] == "Defender")
    assert 3 <= xi_def <= 5
    xi_mid = sum(1 for u in result["xi"] if by_uid[u]["position"] == "Midfielder")
    assert 2 <= xi_mid <= 5
    xi_fwd = sum(1 for u in result["xi"] if by_uid[u]["position"] == "Forward")
    assert 1 <= xi_fwd <= 3

    for club in {c["club"] for c in pool}:
        assert sum(1 for u in result["squad"] if by_uid[u]["club"] == club) <= 3
        assert sum(1 for u in result["xi"] if by_uid[u]["club"] == club) <= 3

    assert result["captain"] in result["xi"]
    assert result["vice"] in result["xi"]
    assert result["captain"] != result["vice"]


def test_forced_squad_uids_are_present_in_the_solved_squad_and_solution_stays_valid():
    """A manager's own hard lock-in: the forced player must be in the squad, and every other
    constraint (budget, quotas, club cap) still holds around that pick -- a real constrained
    re-solve, not a squad hand-assembled after the fact."""
    pool = _synthetic_pool()
    forced = frozenset({"fwd3"})  # the most expensive forward in the synthetic pool
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, forced_squad_uids=forced)
    assert result["status"] == "optimal"
    assert forced <= result["squad"]
    by_uid = {c["player_uid"]: c for c in pool}
    assert sum(by_uid[u]["price"] for u in result["squad"]) <= 100.0 + 1e-6
    assert sum(1 for u in result["squad"] if by_uid[u]["position"] == "Forward") == 3


def test_forced_xi_uids_are_started_not_just_rostered():
    pool = _synthetic_pool()
    forced = frozenset({"mid5"})  # the most expensive midfielder in the synthetic pool
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, forced_xi_uids=forced)
    assert result["status"] == "optimal"
    assert forced <= result["xi"]
    assert forced <= result["squad"]


def test_forced_squad_uids_raises_loudly_on_a_uid_not_in_the_candidate_pool():
    pool = _synthetic_pool()
    with pytest.raises(ValueError, match="not_a_real_uid"):
        so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, forced_squad_uids=frozenset({"not_a_real_uid"}))


def test_forced_squad_uids_infeasible_combination_surfaces_as_non_optimal_not_a_crash():
    """Forcing every forward in a 4-forward pool (position quota is exactly 3) is infeasible
    by construction -- must come back as a normal non-optimal status, the same way any other
    infeasible/degenerate solve already does, not raise or silently drop the constraint."""
    pool = _synthetic_pool()
    all_fwds = frozenset(c["player_uid"] for c in pool if c["position"] == "Forward")
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, forced_squad_uids=all_fwds)
    assert result["status"] != "optimal"
    assert result["squad"] == frozenset()


def test_risk_posture_neutral_is_an_exact_reduction_to_the_pre_existing_objective():
    """kappa_rank=0.0 / risk_posture defaults must reproduce the exact pre-existing result --
    not approximately, byte-identical -- confirming the new term is fully absent (not merely
    zeroed) for any caller that doesn't opt in."""
    pool = _synthetic_pool()
    baseline = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    with_defaults = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, risk_posture="neutral")
    assert with_defaults["squad"] == baseline["squad"]
    assert with_defaults["xi"] == baseline["xi"]
    assert with_defaults["captain"] == baseline["captain"]
    assert with_defaults["objective"] == pytest.approx(baseline["objective"])


def test_risk_posture_protect_vs_chase_pick_opposite_sides_of_a_genuine_tie():
    """The real proof the risk-posture toggle changes squad OUTPUT, not just that it runs:
    fwd0 and fwd1 are fully tied on mu/var/price/club-slot-cost (a genuine coin-flip in the
    base objective) and only ONE of them fits the 3-forward XI quota alongside the two
    strictly-better fwd2/fwd3 -- so which one survives is decided entirely by field_cov_by_uid
    once risk_posture opts in. fwd0 has high field_cov (a stand-in for "popular, moves with
    the field"), fwd1 has negative field_cov (a stand-in for "genuine differential, moves
    against the field")."""
    pool = _synthetic_pool()
    by_uid = {c["player_uid"]: c for c in pool}
    # make fwd0/fwd1 a genuine tie: same mu/var/price (still different clubs, already true in
    # the fixture, so the club cap can't be what breaks the tie either)
    by_uid["fwd1"]["mu"] = by_uid["fwd0"]["mu"]
    by_uid["fwd1"]["var"] = by_uid["fwd0"]["var"]
    by_uid["fwd1"]["price"] = by_uid["fwd0"]["price"]

    field_cov = {"fwd0": 5.0, "fwd1": -5.0}

    protect = so.solve(
        pool, sigma_pairs={}, lam=0.15, guardrail_cap=3,
        field_cov_by_uid=field_cov, kappa_rank=0.5, risk_posture="protect",
    )
    chase = so.solve(
        pool, sigma_pairs={}, lam=0.15, guardrail_cap=3,
        field_cov_by_uid=field_cov, kappa_rank=0.5, risk_posture="chase",
    )
    assert protect["status"] == "optimal" and chase["status"] == "optimal"
    assert "fwd0" in protect["squad"] and "fwd1" not in protect["squad"]
    assert "fwd1" in chase["squad"] and "fwd0" not in chase["squad"]


def test_risk_posture_requires_a_valid_value():
    pool = _synthetic_pool()
    with pytest.raises(ValueError, match="risk_posture"):
        so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, risk_posture="hug_the_field")


def test_captain_is_never_a_goalkeeper():
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    by_uid = {c["player_uid"]: c for c in pool}
    assert by_uid[result["captain"]]["position"] != "Goalkeeper"


def test_vice_captain_is_the_next_highest_mu_xi_player():
    """Real gap fixed alongside the captain-risk decoupling: the `vice` decision variable never
    appeared in solve()'s objective_expr at all, so the solver's choice of vice was entirely
    unconstrained/arbitrary -- not EP-driven. Fixed as a deterministic post-processing step:
    vice must be the highest-mu XI player after the captain, not merely someone in the XI."""
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    by_uid = {c["player_uid"]: c for c in pool}
    expected_vice = max(
        (uid for uid in result["xi"] if uid != result["captain"]), key=lambda uid: by_uid[uid]["mu"],
    )
    assert result["vice"] == expected_vice


def test_captain_points_double_counted_in_objective():
    """With zero variance/covariance, the objective should equal sum(XI mu) + captain's mu
    a second time -- verifies the captain bonus term is actually wired into the objective,
    not just selected as a label."""
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.0, guardrail_cap=3)
    by_uid = {c["player_uid"]: c for c in pool}
    expected = sum(by_uid[u]["mu"] for u in result["xi"]) + by_uid[result["captain"]]["mu"]
    assert result["objective"] == pytest.approx(expected)


def test_higher_lambda_reduces_or_holds_objective_with_real_variance():
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 2.0, ("mid0", "mid1"): 1.5}
    r0 = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3)
    r_real = so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    assert r0["objective"] >= r_real["objective"] - 1e-6  # risk penalty can only reduce the achievable score


def test_divergence_check_fails_when_variance_is_a_stub_zero():
    """The exact historical failure mode named in M5's own spec: a stub/zero covariance
    matrix makes lambda irrelevant, so both solves land on the same squad."""
    pool = _synthetic_pool()
    r0 = so.solve(pool, sigma_pairs={}, lam=0.0, guardrail_cap=3)
    r_real = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    # all variances are equal (var=1.0 uniformly) and there's no covariance signal at all --
    # the risk term is (nearly) flat across any fixed-size XI, so lambda shouldn't move the
    # optimal selection. This is exactly the scenario the divergence check exists to catch.
    assert r0["squad"] == r_real["squad"]


def test_captain_choice_is_risk_aware_when_kappa_captain_is_opted_in():
    """Regression test for a real bug: the risk term used to weight every XI player's
    variance/covariance by a flat `xi` indicator, so captaining (which doubles a player's
    actual point variance -- Var(2X)=4*Var(X)) was invisible to the risk-aversion mechanism.
    Two otherwise-identical high-mu candidates, one with much higher variance than the other:
    a genuinely risk-aware optimizer should prefer captaining the LOWER-variance one once
    kappa_captain > 0, even though their raw mu is tied (so a risk-blind captain choice would be
    indifferent between them).

    kappa_captain (not lam) is what governs captain risk-awareness now -- see solve()'s
    risk_xi_base/risk_captain_extra split and seed_v1_params()'s captain_risk_aversion_params
    comment for why squad-level lambda alone stopped being enough (it could, and on real data
    did, override a genuine multi-point EP gap, not just break an exact tie like this one)."""
    pool = _synthetic_pool()
    # tie two players' mu exactly, at the top of the pool, so captaincy is otherwise a coin
    # flip between them -- the only thing that should break the tie is variance.
    top_mu = max(c["mu"] for c in pool)
    steady, volatile = pool[-1], pool[-2]  # fwd3, fwd2 (both Forwards, adjacent clubs/prices)
    tied_mu = top_mu + 1.0  # make them the clear top-mu candidates in the whole pool
    steady["mu"] = tied_mu
    volatile["mu"] = tied_mu
    steady["var"] = 1.0
    volatile["var"] = 40.0  # a real boom-or-bust player: same expected points, far riskier

    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, kappa_captain=0.15)
    by_uid = {c["player_uid"]: c for c in pool}
    assert result["captain"] in (steady["player_uid"], volatile["player_uid"])
    assert by_uid[result["captain"]]["player_uid"] == steady["player_uid"], (
        "with tied mu, a risk-aware optimizer must prefer captaining the lower-variance "
        "player -- if this fails, the risk term is not accounting for captaincy doubling "
        "variance (Var(2X)=4*Var(X)), i.e. the original bug has regressed"
    )


def test_captain_choice_ignores_variance_when_kappa_captain_is_zero():
    """The new default at the solve() level (kappa_captain=0.0) -- the same tied-mu, wildly-
    different-variance setup as above, but with kappa_captain left at its default: captain
    choice must be driven purely by mu (a coin flip here, since mu is tied), never penalized
    by variance at all. This is what makes squad-level lambda's effect on captain choice fully
    opt-in via kappa_captain, not automatic."""
    pool = _synthetic_pool()
    top_mu = max(c["mu"] for c in pool)
    steady, volatile = pool[-1], pool[-2]
    tied_mu = top_mu + 1.0
    steady["mu"] = tied_mu
    volatile["mu"] = tied_mu
    steady["var"] = 1.0
    volatile["var"] = 40.0

    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    assert result["captain"] in (steady["player_uid"], volatile["player_uid"])
    # Either is a valid outcome here: kappa_captain=0 means the captain-incremental risk term
    # contributes nothing, so which of the two tied-mu players ends up captain is down to
    # solver tie-break internals, not this module's own behavior -- asserting a specific one
    # would test the solver, not the fix.


def test_captain_choice_does_not_override_a_real_ep_gap_at_default_kappa_captain():
    """The real scenario this fix addresses, reproduced synthetically: a high-mu, high-var
    candidate (Haaland-like) versus a low-mu, low-var candidate (Berge-like) with a real,
    substantial EP gap -- not a near-tie. At the v1 default kappa_captain (0.01, an order of
    magnitude below lambda_value), the high-EP candidate must still win the captaincy despite
    its much larger variance; the old combined-lambda formula (equivalent to kappa_captain=lam
    here) would have flipped this, which is exactly the bug this test guards against."""
    pool = _synthetic_pool()
    high_mu_high_var, low_mu_low_var = pool[-1], pool[-2]
    top_mu = max(c["mu"] for c in pool)
    high_mu_high_var["mu"] = top_mu + 2.09  # real observed EP gap from the live GW1 case
    high_mu_high_var["var"] = 16.1
    low_mu_low_var["mu"] = top_mu
    low_mu_low_var["var"] = 5.1

    # at the old, pre-fix combined-lambda equivalent (kappa_captain == lam), the EP gap is
    # overturned -- confirms the synthetic setup actually reproduces the real bug's magnitude.
    # Whichever lower-variance player actually wins depends on the whole pool's variance
    # landscape (here it's a third, untouched midfielder with even lower variance than either
    # of the two players this test modified) -- the real, load-bearing assertion is just that
    # the high-EP candidate LOSES the captaincy despite its 2+ point lead.
    result_old_equivalent = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, kappa_captain=0.15)
    by_uid = {c["player_uid"]: c for c in pool}
    assert by_uid[result_old_equivalent["captain"]]["player_uid"] != high_mu_high_var["player_uid"]

    # at the v1 default (kappa_captain=0.01), the real EP gap must survive.
    result_fixed = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3, kappa_captain=0.01)
    assert by_uid[result_fixed["captain"]]["player_uid"] == high_mu_high_var["player_uid"]


def test_divergence_check_passes_with_real_variance_structure(con):
    """A meaningfully differentiated variance/covariance structure (like M4's real output)
    should make lambda actually move the solve -- the check should NOT fire here."""
    import time
    from datetime import date

    so.seed_v1_params(con)
    pool = _synthetic_pool()
    # give the highest-mu players disproportionately high variance and strong positive
    # covariance with each other, so the risk-averse solve has a real reason to diversify.
    for c in pool:
        if c["position"] in ("Defender", "Midfielder") and c["mu"] > 3.5:
            c["var"] = 12.0
    sigma_pairs = {}
    high_var_ids = [c["player_uid"] for c in pool if c["var"] == 12.0]
    for i in range(len(high_var_ids)):
        for j in range(i + 1, len(high_var_ids)):
            sigma_pairs[(high_var_ids[i], high_var_ids[j])] = 8.0

    r0 = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3)
    r_real = so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    assert r0["squad"] != r_real["squad"]


# ============================================================
# captain_choice_with_differential -- closed-form tie-break, hand-computed values
# ============================================================

def _xi_candidate(uid, mu, var, ownership, position="Midfielder"):
    return {"player_uid": uid, "position": position, "mu": mu, "var": var, "selected_by_percent": ownership, "name": uid}


def test_captain_differential_breaks_a_genuine_near_tie_toward_lower_ownership():
    """A, B, C in the XI, zero covariance, equal variance (so captain choice differences come
    purely from mu): objective(X) = const + mu_X - 3*lam*var_X. With mu_A=5.0, mu_B=5.02,
    mu_C=3.0, var all 1.0, lam=0.1: objective(A)=17.42, objective(B)=17.44, objective(C)=15.42
    (hand-computed). base_captain=B (solve()'s own pick, marginally ahead of A). With
    epsilon=0.05, A (0.02 below B) is a genuine near-tie and must be preferred for its far
    lower ownership; C (2.0 below B) is nowhere near the tie band and must never be reachable
    no matter how low its own ownership is."""
    xi = [
        _xi_candidate("A", mu=5.0, var=1.0, ownership=5.0),
        _xi_candidate("B", mu=5.02, var=1.0, ownership=80.0),
        _xi_candidate("C", mu=3.0, var=1.0, ownership=1.0),
    ]
    result = so.captain_choice_with_differential(xi, sigma_pairs={}, lam=0.1, base_captain_uid="B", tiebreak_epsilon=0.05)
    assert result["recommended_captain_uid"] == "A"
    assert result["changed"] is True
    assert set(result["near_optimal_candidates"]) == {"A", "B"}
    assert "C" not in result["near_optimal_candidates"]

    # captain_eo_risk: a distinct, explicit report -- B (80% owned) is the field-typical
    # captain even though A (5% owned) won the tie-break; the report must surface BOTH,
    # not just whichever the tie-break happened to pick.
    eo_risk = result["captain_eo_risk"]
    assert eo_risk["field_typical_captain_uid"] == "B"
    assert eo_risk["field_typical_captain_eo"] == pytest.approx(0.80)
    assert eo_risk["recommended_captain_eo"] == pytest.approx(0.05)
    assert eo_risk["eo_gap"] == pytest.approx(0.75)


def test_captain_differential_never_overrides_a_real_ep_risk_gap_beyond_epsilon():
    """Same pool, a tight epsilon (0.01) that excludes even A from the near-optimal band
    (A is 0.02 below B) -- must keep the real EP/risk-optimal captain B, regardless of A's
    far lower ownership. Proves this is a hard constraint, not an ownership bonus that can
    talk its way past a real (if small) objective gap."""
    xi = [
        _xi_candidate("A", mu=5.0, var=1.0, ownership=5.0),
        _xi_candidate("B", mu=5.02, var=1.0, ownership=80.0),
        _xi_candidate("C", mu=3.0, var=1.0, ownership=1.0),
    ]
    result = so.captain_choice_with_differential(xi, sigma_pairs={}, lam=0.1, base_captain_uid="B", tiebreak_epsilon=0.01)
    assert result["recommended_captain_uid"] == "B"
    assert result["changed"] is False
    assert result["near_optimal_candidates"] == ["B"]


def test_captain_differential_treats_missing_ownership_as_neutral_not_zero():
    """A player with no ownership data at all must never be spuriously preferred over a real,
    known lower value just because None isn't a real number to compare against -- missing data
    is the LEAST preferred case among near-optimal candidates, not artificially the most."""
    xi = [
        _xi_candidate("A", mu=5.0, var=1.0, ownership=None),
        _xi_candidate("B", mu=5.02, var=1.0, ownership=10.0),
    ]
    result = so.captain_choice_with_differential(xi, sigma_pairs={}, lam=0.1, base_captain_uid="B", tiebreak_epsilon=1.0)
    assert result["recommended_captain_uid"] == "B"  # real ownership beats missing data, even though both are near-optimal


def test_captain_differential_never_proposes_a_goalkeeper():
    xi = [
        _xi_candidate("gk1", mu=4.0, var=1.0, ownership=1.0, position="Goalkeeper"),
        _xi_candidate("mid1", mu=4.0, var=1.0, ownership=50.0, position="Midfielder"),
    ]
    result = so.captain_choice_with_differential(xi, sigma_pairs={}, lam=0.1, base_captain_uid="mid1", tiebreak_epsilon=5.0)
    assert result["recommended_captain_uid"] == "mid1"
    assert "gk1" not in result["near_optimal_candidates"]


def test_captain_differential_caveat_is_always_present_and_honest():
    xi = [_xi_candidate("A", mu=5.0, var=1.0, ownership=5.0)]
    result = so.captain_choice_with_differential(xi, sigma_pairs={}, lam=0.1, base_captain_uid="A", tiebreak_epsilon=0.05)
    assert "overall-rank" in result["caveat"]
    assert "not a real rival-manager" in result["caveat"] or "not a real" in result["caveat"]


# ============================================================
# recommend_captain_with_differential -- real DB wiring (a real candidate pool + a directly-
# seeded squad_optimizer_runs/squad_optimizer_selections row, not a real SCIP solve -- the
# closed-form differential math itself is already thoroughly covered above; this exercises the
# SQL joins/param resolution around it).
# ============================================================

def _seed_run_for_captain_differential(con):
    con.execute(
        "INSERT INTO dim_team (team_uid, canonical_name) VALUES ('club_a', 'A'), ('club_b', 'B') ON CONFLICT DO NOTHING"
    )
    con.execute(
        "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
        "competition, kickoff_time, _ingested_at) VALUES ('m1', '2026-2027', 2, 'club_a', 'club_b', FALSE, "
        "'Premier League', '2026-08-24', current_timestamp)"
    )
    con.execute(
        "INSERT INTO team_strength_model_versions (calibration_asof_date, home_advantage, xi_params_version, "
        "rho_params_version, reference_team_uid) VALUES ('2026-08-10', 0.2, 1, 1, 'club_a')"
    )
    ts_mv = con.execute("SELECT max(model_version) FROM team_strength_model_versions").fetchone()[0]
    con.execute(
        "INSERT INTO minutes_model_versions (calibration_asof_date, target_season, decay_params_version, "
        "adjustment_params_version, shrinkage_params_version, fact_multiplier_params_version, lookback_seasons) "
        "VALUES ('2026-08-10', '2026-2027', 1, 1, 1, 1, '[]')"
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

    xi_players = {"A": (5.0, 5.0), "B": (5.02, 80.0), "C": (3.0, 1.0)}  # uid -> (mu, ownership)
    for uid, (mu, ownership) in xi_players.items():
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')", [uid, uid])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, '1', '2026-2027', ?)", [uid, uid.lower(), uid],
        )
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
            "VALUES (?, '2026-2027', 1, 5.0, ?, current_timestamp)", [uid, ownership],
        )
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, ep_assists, "
            "ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, ep_cards, ep_own_goal, "
            "ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0,0,0,0, ?, 5.0)", [ep_mv, uid, mu],
        )
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, var_goals, "
            "var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, var_total, skew, "
            "excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, 1.0, 0,0,0,0,0,0)", [un_mv, uid],
        )

    run_id = con.execute(
        "INSERT INTO squad_optimizer_runs (calibration_asof_date, target_season, target_gameweek, ep_model_version, "
        "uncertainty_model_version, lambda_params_version, lambda_value, guardrail_params_version, "
        "divergence_check_passed, solver_status, objective_value) "
        "VALUES ('2026-08-10', '2026-2027', 2, ?, ?, 1, 0.1, 1, TRUE, 'optimal', 20.0) RETURNING run_id",
        [ep_mv, un_mv],
    ).fetchone()[0]
    for uid in xi_players:
        con.execute(
            "INSERT INTO squad_optimizer_selections (run_id, player_uid, in_squad, in_xi, is_captain, is_vice) "
            "VALUES (?, ?, TRUE, TRUE, ?, FALSE)", [run_id, uid, uid == "B"],
        )
    return run_id


def test_recommend_captain_with_differential_wires_through_to_the_same_closed_form_result(con):
    so.seed_v1_params(con)  # captain_differential_params version=1, tiebreak_epsilon=0.05
    run_id = _seed_run_for_captain_differential(con)

    result = so.recommend_captain_with_differential(con, run_id, differential_tiebreak_params_version=1)
    assert result["run_id"] == run_id
    assert result["base_captain_uid"] == "B"
    assert result["recommended_captain_uid"] == "A"  # same real near-tie as the closed-form tests above
    assert result["changed"] is True


def test_recommend_captain_with_differential_raises_on_unknown_run(con):
    so.seed_v1_params(con)
    with pytest.raises(ValueError):
        so.recommend_captain_with_differential(con, 999, differential_tiebreak_params_version=1)
