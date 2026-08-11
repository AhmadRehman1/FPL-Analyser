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


def test_captain_is_never_a_goalkeeper():
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    by_uid = {c["player_uid"]: c for c in pool}
    assert by_uid[result["captain"]]["position"] != "Goalkeeper"


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
