"""Phase 1C: captain objective regression tests.

2026-09 audit (docs/reports/2026-09_calibration_captain_headline_audit.md, Audit B) found no committed
test that squad_optimizer._captain_objective_component() -- the closed-form captain-marginal
decomposition solve() itself relies on via captain_choice_with_differential() -- actually agrees
with the real SCIP-solved objective it claims to reproduce. These tests close that gap: the
analytical decomposition is checked against (1) a hand-computed toy fixture and (2) the real
solve() objective evaluator on a synthetic pool with nontrivial cross-covariance, and asserted to
fail loudly (a wrong lambda genuinely produces a mismatch) rather than passing vacuously.
"""

import pytest

from fpl_quant import squad_optimizer as so


# ============================================================
# 1. Hand-computed toy fixture -- analytical decomposition only, no solver involved.
# ============================================================

def test_captain_objective_component_hand_computed_no_covariance():
    # 2-player fixed XI, zero covariance: objective = sum(mu) - lam*sum((1+3*is_captain)*var).
    xi_uids = ["a", "b"]
    mu = {"a": 5.0, "b": 4.0}
    var = {"a": 2.0, "b": 1.0}
    lam = 0.15
    obj_a = so._captain_objective_component(xi_uids, var, mu, {}, "a", lam)
    # linear = 5+4 + 5 (captain a doubles a's mu) = 14; risk = (1+3)*2.0 + 1*1.0 = 9.0
    assert obj_a == pytest.approx(14.0 - lam * 9.0)
    obj_b = so._captain_objective_component(xi_uids, var, mu, {}, "b", lam)
    # linear = 9 + 4 = 13; risk = 1*2.0 + (1+3)*1.0 = 6.0
    assert obj_b == pytest.approx(13.0 - lam * 6.0)


def test_captain_objective_component_hand_computed_with_cross_covariance():
    # Same as Audit B's own toy hand-check: captaining the player MORE correlated with the rest
    # of a fixed XI costs strictly more (all else equal), via the cross-covariance term alone.
    xi_uids = ["a", "b", "c"]
    mu = {"a": 6.0, "b": 6.0, "c": 5.0}
    var = {"a": 10.0, "b": 10.0, "c": 8.0}
    cov = {("a", "c"): 4.0, ("b", "c"): 0.0, ("a", "b"): 0.0}
    lam = 0.15
    obj_a = so._captain_objective_component(xi_uids, var, mu, cov, "a", lam)
    obj_b = so._captain_objective_component(xi_uids, var, mu, cov, "b", lam)
    # a and b are otherwise symmetric (same mu, same var); the only difference is a's real
    # correlation with c. Captaining a doubles the a-c cross term (extra 2*lam*cov(a,c)=1.2)
    # relative to captaining b, which has zero cross-covariance with anything.
    assert (obj_a - obj_b) == pytest.approx(-1.2)


def test_captain_objective_component_zero_lambda_reduces_to_pure_ep():
    xi_uids = ["a", "b"]
    mu = {"a": 5.0, "b": 9.0}
    var = {"a": 100.0, "b": 0.01}  # variance must not matter at lam=0
    obj_a = so._captain_objective_component(xi_uids, var, mu, {}, "a", 0.0)
    obj_b = so._captain_objective_component(xi_uids, var, mu, {}, "b", 0.0)
    assert obj_a == pytest.approx(5.0 + 9.0 + 5.0)  # sum(xi mu) + a's own mu (captain double)
    assert obj_b == pytest.approx(5.0 + 9.0 + 9.0)
    assert obj_b > obj_a  # the higher-EP player must win captaincy when risk is switched off


def test_captain_objective_component_ignores_non_xi_entries_in_mu_and_var_dicts():
    """Regression test for a real bug this diagnostic found: linear_ep used to sum
    mu_by_uid.values() directly, so passing a mu_by_uid/var_by_uid that covers MORE than the
    fixed XI (e.g. a full candidate pool, exactly what a diagnostic script naturally has lying
    around) silently inflated the objective by every non-XI player's mu. Fixed to sum over
    xi_uids explicitly, matching how the risk term already behaved."""
    xi_uids = ["a", "b"]
    mu_xi_only = {"a": 5.0, "b": 9.0}
    mu_with_bench = {"a": 5.0, "b": 9.0, "bench1": 100.0, "bench2": -50.0}
    var = {"a": 1.0, "b": 1.0, "bench1": 1.0, "bench2": 1.0}
    result_scoped = so._captain_objective_component(xi_uids, var, mu_xi_only, {}, "a", 0.15)
    result_with_bench = so._captain_objective_component(xi_uids, var, mu_with_bench, {}, "a", 0.15)
    assert result_scoped == pytest.approx(result_with_bench)


# ============================================================
# 2. Real solve() objective evaluator agreement, on a synthetic pool with real cross-covariance.
# ============================================================

def _pool_with_covariance_structure():
    """Same shape/club-diversity discipline as test_squad_optimizer.py's _synthetic_pool
    (>=5 clubs needed for the 15-slot/3-per-club cap to stay feasible), kept local to this file
    rather than cross-imported so this diagnostic doesn't depend on another test module's
    private fixture staying stable. Positive covariance between adjacent same-position players
    (same club rotation -> correlated fixtures in the real pipeline) gives the cross-covariance
    term in _captain_objective_component something real to disagree about if it's ever wrong."""
    pool = []
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]

    def add(pid, pos, mu, price, club, var=1.0):
        pool.append({"player_uid": pid, "position": pos, "mu": mu, "var": var, "club": club, "price": price, "name": pid})

    for i in range(2):
        add(f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6])
    for i in range(6):
        add(f"def{i}", "Defender", 2.5 + i * 0.3, 4.0 + i * 0.5, clubs[i % 6], var=1.5)
    for i in range(6):
        add(f"mid{i}", "Midfielder", 3.0 + i * 0.4, 5.0 + i * 0.5, clubs[i % 6], var=1.5)
    for i in range(4):
        add(f"fwd{i}", "Forward", 3.5 + i * 0.5, 6.0 + i * 0.5, clubs[i % 6], var=2.0)

    sigma_pairs = {}
    for prefix, n in (("def", 6), ("mid", 6), ("fwd", 4)):
        for i in range(n - 1):
            sigma_pairs[(f"{prefix}{i}", f"{prefix}{i + 1}")] = 0.4
    return pool, sigma_pairs


def test_captain_objective_component_matches_real_solve_objective():
    pool, sigma_pairs = _pool_with_covariance_structure()
    lam = 0.15
    result = so.solve(pool, sigma_pairs=sigma_pairs, lam=lam, guardrail_cap=3)
    assert result["status"] == "optimal"

    xi_uids = result["xi"]
    xi_set = set(xi_uids)
    # Scoped to the XI only, matching squad_optimizer.captain_choice_with_differential()'s own
    # calling convention (its xi_candidates are already the solved XI's candidate dicts, not
    # the full pool) -- see test_captain_objective_component_ignores_non_xi_entries_in_mu_and_
    # var_dicts above for what happens if a caller doesn't scope these (now handled safely
    # either way, but this test should exercise the documented convention specifically).
    var_by_uid = {c["player_uid"]: c["var"] for c in pool if c["player_uid"] in xi_set}
    mu_by_uid = {c["player_uid"]: c["mu"] for c in pool if c["player_uid"] in xi_set}
    cov_by_xi_pair = {(a, b): cov for (a, b), cov in sigma_pairs.items() if a in xi_set and b in xi_set}

    reconstructed = so._captain_objective_component(
        xi_uids, var_by_uid, mu_by_uid, cov_by_xi_pair, result["captain"], lam,
    )
    assert reconstructed == pytest.approx(result["objective"], abs=1e-4), (
        "the closed-form captain-marginal decomposition must reproduce the real SCIP-solved "
        "objective for the solver's own chosen (XI, captain) -- a mismatch here means "
        "_captain_objective_component (and therefore captain_choice_with_differential's "
        "near-optimal-set logic) no longer agrees with what solve() actually optimizes"
    )
    assert len(cov_by_xi_pair) > 0, "fixture sanity check: the cross-covariance term must actually be exercised"


def test_captain_objective_component_reproduces_the_solvers_own_captain_choice():
    pool, sigma_pairs = _pool_with_covariance_structure()
    lam = 0.15
    result = so.solve(pool, sigma_pairs=sigma_pairs, lam=lam, guardrail_cap=3)
    assert result["status"] == "optimal"

    xi_uids = result["xi"]
    var_by_uid = {c["player_uid"]: c["var"] for c in pool}
    mu_by_uid = {c["player_uid"]: c["mu"] for c in pool}
    xi_set = set(xi_uids)
    cov_by_xi_pair = {(a, b): cov for (a, b), cov in sigma_pairs.items() if a in xi_set and b in xi_set}
    non_gk_xi = [u for u in xi_uids if dict((c["player_uid"], c["position"]) for c in pool)[u] != "Goalkeeper"]

    scores = {
        uid: so._captain_objective_component(xi_uids, var_by_uid, mu_by_uid, cov_by_xi_pair, uid, lam)
        for uid in non_gk_xi
    }
    analytical_captain = max(scores, key=scores.get)
    assert analytical_captain == result["captain"], (
        f"argmax of the closed-form decomposition ({analytical_captain}) must match the real "
        f"solver's chosen captain ({result['captain']}) for the SAME fixed XI"
    )


def test_captain_objective_component_disagrees_loudly_under_a_wrong_lambda():
    """A meta-test proving the agreement check above has power: deliberately reconstructing
    with the WRONG lambda must NOT match the real objective (unless the risk term is exactly
    zero, which this fixture's nonzero variance/covariance rules out) -- otherwise the equality
    assertion in the test above would pass vacuously regardless of correctness."""
    pool, sigma_pairs = _pool_with_covariance_structure()
    true_lam, wrong_lam = 0.15, 0.5
    result = so.solve(pool, sigma_pairs=sigma_pairs, lam=true_lam, guardrail_cap=3)
    assert result["status"] == "optimal"

    xi_uids = result["xi"]
    var_by_uid = {c["player_uid"]: c["var"] for c in pool}
    mu_by_uid = {c["player_uid"]: c["mu"] for c in pool}
    xi_set = set(xi_uids)
    cov_by_xi_pair = {(a, b): cov for (a, b), cov in sigma_pairs.items() if a in xi_set and b in xi_set}

    wrong = so._captain_objective_component(xi_uids, var_by_uid, mu_by_uid, cov_by_xi_pair, result["captain"], wrong_lam)
    assert wrong != pytest.approx(result["objective"], abs=1e-4)
