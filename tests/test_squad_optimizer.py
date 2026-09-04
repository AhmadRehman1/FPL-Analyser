from datetime import date

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


def test_warn_if_sigma_not_psd_fires_on_invalid_correlation(capsys):
    """def0/def1 both have var=1.0 (see _synthetic_pool), so a covariance of 2.0 implies a
    correlation of 2.0 -- mathematically impossible, i.e. Sigma is not PSD. Warn-only: this
    must not raise, since callers may legitimately pass an approximate/stale Sigma."""
    pool = _synthetic_pool()
    so._warn_if_sigma_not_psd(pool, {("def0", "def1"): 2.0})
    out = capsys.readouterr().out
    assert "::warning::squad_optimizer.solve" in out
    assert "not PSD" in out


def test_warn_if_sigma_not_psd_silent_on_valid_covariance(capsys):
    pool = _synthetic_pool()
    so._warn_if_sigma_not_psd(pool, {("def0", "def1"): 0.3})
    assert capsys.readouterr().out == ""


def test_solve_warns_when_sigma_pairs_not_psd(capsys):
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 2.0, ("mid0", "mid1"): 1.5}
    so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    out = capsys.readouterr().out
    assert "::warning::squad_optimizer.solve" in out


def test_solve_returns_identical_output_across_repeated_calls():
    """Priority 0 regression test: a real squad recommendation once flip-flopped between
    formations/captains across repeated runs against IDENTICAL underlying data. solve() must
    be a pure function of its (candidates, sigma_pairs, lam, guardrail_cap) inputs -- calling
    it 10 times with the exact same arguments must return the exact same squad/xi/captain/
    vice/objective every time, not merely "an" optimal solution each time."""
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 2.0, ("mid0", "mid1"): 1.5}
    results = [so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3) for _ in range(10)]
    first = results[0]
    for r in results[1:]:
        assert r["squad"] == first["squad"]
        assert r["xi"] == first["xi"]
        assert r["captain"] == first["captain"]
        assert r["vice"] == first["vice"]
        assert r["objective"] == pytest.approx(first["objective"])


def test_solve_is_invariant_to_candidate_list_ordering():
    """The actual root cause of the flip-flopping bug (see squad_optimizer.solve()'s own
    docstring comment): the candidate pool arrived from an ORDER-BY-less SQL join, whose row
    order is not guaranteed stable across runs -- and whenever the true optimum admits more
    than one exactly-tied solution (this fixture's sigma_pairs is deliberately chosen so a
    genuine tie exists, mirroring the real incident), which tied optimum SCIP's branch-and-
    bound returns depended on that arrival order. Shuffling the candidate list (simulating a
    different SQL/thread-scheduling order on a re-run against the same data) must not change
    the result once solve() is a pure function of its logical inputs."""
    import random

    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 2.0, ("mid0", "mid1"): 1.5}
    baseline = so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)

    rng = random.Random(1234)
    for _ in range(10):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        result = so.solve(shuffled, sigma_pairs, lam=0.15, guardrail_cap=3)
        assert result["xi"] == baseline["xi"]
        assert result["captain"] == baseline["captain"]
        assert result["vice"] == baseline["vice"]
        assert result["objective"] == pytest.approx(baseline["objective"])


def test_mip_gap_is_zero_on_a_proven_optimal_solve():
    """Priority 2 solve-quality transparency: a synthetic pool this small should always
    reach a PROVEN optimum well within the time limit, so mip_gap should be exactly 0.0 --
    not merely status == 'optimal' with the gap left unreported."""
    pool = _synthetic_pool()
    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    assert result["status"] == "optimal"
    assert result["mip_gap"] == pytest.approx(0.0)


# ============================================================
# Priority 1 -- risk-posture (EO) term
# ============================================================

def _posture_pool():
    """Same base pool as _synthetic_pool(), but with fwd2 and fwd3's mu tied at the top of
    the whole pool -- a genuine EP tie between two forwards, so any XI/squad preference
    between them can only come from the EO term, not from a real EP edge."""
    pool = _synthetic_pool()
    by_uid = {c["player_uid"]: c for c in pool}
    top_mu = max(c["mu"] for c in pool)
    tied_mu = top_mu + 1.0
    by_uid["fwd2"]["mu"] = tied_mu
    by_uid["fwd3"]["mu"] = tied_mu
    eo_by_uid = {c["player_uid"]: 10.0 for c in pool}
    eo_by_uid["fwd2"] = 90.0  # high-EO "template" pick
    eo_by_uid["fwd3"] = 5.0   # low-EO differential
    return pool, eo_by_uid


def test_protect_posture_favors_the_higher_eo_tied_ep_candidate():
    pool, eo_by_uid = _posture_pool()
    result = so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="protect", eo_weight_kappa=0.5)
    assert "fwd2" in result["squad"]  # the high-EO tied-mu player is kept


def test_chase_posture_favors_the_lower_eo_tied_ep_candidate():
    """Same pool, same tied EP, opposite posture: chase must make the OPPOSITE choice from
    protect -- proving the toggle actually changes squad output in the expected direction
    for both modes, not just that solve() runs under either setting."""
    pool, eo_by_uid = _posture_pool()
    result = so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="chase", eo_weight_kappa=0.5)
    assert "fwd2" not in result["squad"]  # the high-EO tied-mu player is dropped for a lower-EO alternative


def test_posture_toggle_changes_squad_output_between_modes():
    """Direct proof the toggle changes OUTPUT, not just that both modes run without error."""
    pool, eo_by_uid = _posture_pool()
    r_protect = so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="protect", eo_weight_kappa=0.5)
    r_chase = so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="chase", eo_weight_kappa=0.5)
    assert r_protect["squad"] != r_chase["squad"]


def test_eo_term_is_a_no_op_by_default():
    """kappa=0.0 (the default) must reproduce the exact same result as not passing EO
    arguments at all -- every existing caller's behavior must be byte-identical."""
    pool, eo_by_uid = _posture_pool()
    with_defaults = so.solve(pool, {}, lam=0.0, guardrail_cap=3)
    with_explicit_zero_kappa = so.solve(
        pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="protect", eo_weight_kappa=0.0
    )
    assert with_defaults["squad"] == with_explicit_zero_kappa["squad"]
    assert with_defaults["objective"] == pytest.approx(with_explicit_zero_kappa["objective"])


def test_eo_weight_kappa_requires_posture():
    pool, eo_by_uid = _posture_pool()
    with pytest.raises(ValueError):
        so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture=None, eo_weight_kappa=0.5)


def test_invalid_posture_raises():
    pool, eo_by_uid = _posture_pool()
    with pytest.raises(ValueError):
        so.solve(pool, {}, lam=0.0, guardrail_cap=3, eo_by_uid=eo_by_uid, posture="bogus", eo_weight_kappa=0.5)


# ============================================================
# Priority 1 -- field-covariance term (linear coefficient, same posture-sign convention)
# ============================================================

def test_field_cov_term_prefers_higher_covariance_candidate_under_protect():
    """Same tied-mu setup, but the signal is Cov(player, field) rather than EO -- protect
    should keep the player with higher field-covariance (moves with the field, protects
    rank), the same direction EO already demonstrated above."""
    pool = _synthetic_pool()
    by_uid = {c["player_uid"]: c for c in pool}
    top_mu = max(c["mu"] for c in pool)
    tied_mu = top_mu + 1.0
    by_uid["fwd2"]["mu"] = tied_mu
    by_uid["fwd3"]["mu"] = tied_mu
    field_cov_by_uid = {c["player_uid"]: 0.0 for c in pool}
    field_cov_by_uid["fwd2"] = 50.0
    field_cov_by_uid["fwd3"] = 1.0

    result = so.solve(
        pool, {}, lam=0.0, guardrail_cap=3, posture="protect",
        field_cov_by_uid=field_cov_by_uid, field_cov_kappa=0.1,
    )
    assert "fwd2" in result["squad"]


def test_field_cov_kappa_requires_posture():
    pool = _synthetic_pool()
    with pytest.raises(ValueError):
        so.solve(pool, {}, lam=0.0, guardrail_cap=3, field_cov_by_uid={}, field_cov_kappa=0.1, posture=None)


# ============================================================
# Priority 2 -- bench-quality floor
# ============================================================

def test_bench_quality_floor_excludes_low_p_start_from_bench():
    """A candidate below the threshold must never appear on the bench (in squad, not in xi)
    -- either not selected at all, or selected AND started."""
    pool = _synthetic_pool()
    # def5 is the cheapest/lowest-mu defender, the most likely bench filler under a plain
    # EP-maximizing solve -- flag it as a rotation risk and confirm it never lands on the bench.
    p_start_by_uid = {c["player_uid"]: 0.9 for c in pool}
    p_start_by_uid["def5"] = 0.1  # far below any reasonable threshold

    result = so.solve(
        pool, {}, lam=0.0, guardrail_cap=3,
        p_start_by_uid=p_start_by_uid, min_bench_p_start_probability=0.25,
    )
    if "def5" in result["squad"]:
        assert "def5" in result["xi"], "def5 is below the bench-quality floor -- it must never be benched"


def test_bench_quality_floor_is_a_no_op_when_unset():
    pool = _synthetic_pool()
    with_defaults = so.solve(pool, {}, lam=0.0, guardrail_cap=3)
    with_none_threshold = so.solve(pool, {}, lam=0.0, guardrail_cap=3, p_start_by_uid={"def5": 0.01}, min_bench_p_start_probability=None)
    assert with_defaults["squad"] == with_none_threshold["squad"]


def test_bench_quality_floor_never_gates_missing_p_start_data():
    """A candidate with no p_start data at all (None) must never be excluded from the bench
    purely for lacking coverage -- absence of minutes-model data isn't evidence of rotation
    risk."""
    pool = _synthetic_pool()
    result_without_floor = so.solve(pool, {}, lam=0.0, guardrail_cap=3)
    result_with_floor = so.solve(
        pool, {}, lam=0.0, guardrail_cap=3,
        p_start_by_uid={}, min_bench_p_start_probability=0.25,  # no p_start data for anyone
    )
    assert result_without_floor["squad"] == result_with_floor["squad"]


# ============================================================
# Priority 2 -- team-concentration risk
# ============================================================

def test_concentration_kappa_default_is_a_no_op():
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 5.0}  # def0/def1 share clubA -- same-club pair
    without = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3)
    with_zero = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3, concentration_kappa=0.0)
    assert without["squad"] == with_zero["squad"]
    assert without["objective"] == pytest.approx(with_zero["objective"])


def test_concentration_kappa_penalizes_same_club_stacking():
    """def0 and def1 (clubA) have a strong positive same-club covariance; a high enough
    concentration_kappa must make the solver strictly worse off for holding both of them
    together vs. lam=0's plain EP-maximizing baseline -- proving the penalty is actually
    wired into the objective, not just accepted as a parameter."""
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 5.0}
    baseline = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3)
    penalized = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3, concentration_kappa=1.0)
    both_in_baseline = "def0" in baseline["squad"] and "def1" in baseline["squad"]
    both_in_penalized = "def0" in penalized["squad"] and "def1" in penalized["squad"]
    if both_in_baseline:
        # either the penalty was strong enough to break up the same-club stack, or (if it's
        # still worth it on raw EP) the reported objective must be strictly lower than an
        # unpenalized solve holding the identical pair -- one of the two must be true.
        assert (not both_in_penalized) or (penalized["objective"] < baseline["objective"] - 1e-6)


def test_higher_lambda_reduces_or_holds_objective_with_real_variance():
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 2.0, ("mid0", "mid1"): 1.5}
    r0 = so.solve(pool, sigma_pairs, lam=0.0, guardrail_cap=3)
    r_real = so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    assert r0["objective"] >= r_real["objective"] - 1e-6  # risk penalty can only reduce the achievable score


def test_psd_warning_fires_on_non_psd_sigma(capsys):
    """def0/def1 both have var=1.0 (the _synthetic_pool default); a cov of 5.0 between them
    violates Cauchy-Schwarz (cov^2=25 > var_a*var_b=1), so the restricted 2x2 submatrix
    [[1,5],[5,1]] has eigenvalues 1+/-5 -- genuinely not PSD, not just numerically borderline."""
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 5.0}
    so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "not PSD" in out


def test_psd_warning_does_not_fire_on_psd_sigma(capsys):
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 0.3, ("mid0", "mid1"): 0.2}  # well within Cauchy-Schwarz for var=1.0
    so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    out = capsys.readouterr().out
    assert "::warning::" not in out


def test_psd_warning_does_not_raise_or_change_solve_behavior():
    """Warn-only: a non-PSD Sigma must not raise, and must not make solve() itself fail --
    this module's own test fixtures deliberately use non-PSD toy covariances (see the test
    above), so a raise here would break the established fixture pattern."""
    pool = _synthetic_pool()
    sigma_pairs = {("def0", "def1"): 5.0}
    result = so.solve(pool, sigma_pairs, lam=0.15, guardrail_cap=3)
    assert result["status"] == "optimal"


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


def test_captain_choice_is_risk_aware_not_just_risk_aware_for_squad_membership():
    """Regression test for a real bug: the risk term used to weight every XI player's
    variance/covariance by a flat `xi` indicator, so captaining (which doubles a player's
    actual point variance -- Var(2X)=4*Var(X)) was invisible to the risk-aversion mechanism.
    Two otherwise-identical high-mu candidates, one with much higher variance than the other:
    a genuinely risk-aware optimizer should prefer captaining the LOWER-variance one once
    lambda > 0, even though their raw mu is tied (so a risk-blind captain choice would be
    indifferent between them)."""
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

    result = so.solve(pool, sigma_pairs={}, lam=0.15, guardrail_cap=3)
    by_uid = {c["player_uid"]: c for c in pool}
    assert result["captain"] in (steady["player_uid"], volatile["player_uid"])
    assert by_uid[result["captain"]]["player_uid"] == steady["player_uid"], (
        "with tied mu, a risk-aware optimizer must prefer captaining the lower-variance "
        "player -- if this fails, the risk term is not accounting for captaincy doubling "
        "variance (Var(2X)=4*Var(X)), i.e. the original bug has regressed"
    )


def test_divergence_check_passes_with_real_variance_structure(con):
    """A meaningfully differentiated variance/covariance structure (like M4's real output)
    should make lambda actually move the solve -- the check should NOT fire here."""

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


# ============================================================
# run() -- Priority 1/2 wiring end-to-end (real DB, real fetch_candidate_pool() joins, not
# solve()'s own in-memory pool -- same "2 GK/6 DEF/6 MID/4 FWD/6-club" budget/club-cap-feasible
# shape test_transfer_planner.py's own _seed_real_squad_optimizer_candidate_pool() uses, kept
# local here rather than cross-imported since every other test file in this project builds its
# own seed helper rather than sharing one across files.
# ============================================================

def _seed_run_candidate_pool(con, target_season="2026-2027", target_gameweek=2):
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    for club in clubs:
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

    players = []
    for i in range(2):
        players.append((f"gk{i}", "Goalkeeper", 3.0 + i * 0.5, 4.5 + i, clubs[i % 6]))
    for i in range(6):
        players.append((f"def{i}", "Defender", 2.5 + i * 0.3, 4.0 + i * 0.5, clubs[i % 6]))
    for i in range(6):
        players.append((f"mid{i}", "Midfielder", 3.0 + i * 0.4, 5.0 + i * 0.5, clubs[i % 6]))
    for i in range(4):
        players.append((f"fwd{i}", "Forward", 3.5 + i * 0.5, 6.0 + i * 0.5, clubs[i % 6]))

    for i, (uid, position, mu, price, club) in enumerate(players):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, ?)", [uid, uid, position])
        con.execute(
            "INSERT INTO player_alias (alias_name, normalized_alias_name, team_code, season, player_uid) "
            "VALUES (?, ?, ?, ?, ?)", [uid, uid.lower(), club, target_season, uid],
        )
        # spread ownership across the pool so EO/posture and field-covariance have real,
        # differentiated data to work with -- not the same value for every player.
        selected_by_percent = 5.0 + (i * 4.0) % 60.0
        con.execute(
            "INSERT INTO fact_player_season_stats (player_uid, season, gw, now_cost, selected_by_percent, _ingested_at) "
            "VALUES (?, ?, 1, ?, ?, current_timestamp)", [uid, target_season, price, selected_by_percent],
        )
        con.execute(
            "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
            "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
            "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, 'm1', 0,0,0,0.02,0,0,0,0,0,0,0, ?, 5.0)",
            [ep_mv, uid, mu],
        )
        var = 1.0 + mu * 3.0
        con.execute(
            "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
            "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
            "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
            "VALUES (?, ?, 'm1', 0,0,0,0,0,0,0,0, ?, 0,0,0,0,0,0)", [un_mv, uid, var],
        )
        con.execute(
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
            "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, ?, 0.7, 1.0, 0.7, 0.0, 0.7, 0.0, 0.15, 0.05, 0.8, 20)",
            [mm_mv, uid, position],
        )

    # real, differentiated positive covariance among the highest-mu players (same recipe
    # test_transfer_planner.py's own helper and this file's test_divergence_check_passes_
    # with_real_variance_structure both already use) -- needed for the lambda=0-vs-lambda
    # divergence check to genuinely pass, not just for realism.
    high_mu_uids = [uid for uid, _pos, mu, _price, _club in players if mu >= 4.0]
    for i in range(len(high_mu_uids)):
        for j in range(i + 1, len(high_mu_uids)):
            con.execute(
                "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, "
                "fixture_match_id, relationship, covariance) VALUES (?, ?, ?, 'm1', 'teammate', 6.0)",
                [un_mv, *sorted([high_mu_uids[i], high_mu_uids[j]])],
            )

    return ep_mv, un_mv


def test_run_with_priority1_features_enabled_end_to_end(con):
    """Full run() call with ownership/risk-posture/field-covariance/bench-quality/
    concentration-risk ALL opted in at once -- proves the real DB wiring (param resolution,
    fetch_candidate_pool's new p_start_final join, field_covariance's real fact_match/
    ep_outputs queries) works end to end, not just solve()'s own in-memory-pool unit tests."""
    from fpl_quant import expected_points as ep_mod
    from fpl_quant import uncertainty as un_mod

    ep_mv, un_mv = _seed_run_candidate_pool(con)
    so.seed_v1_params(con)
    ep_mod.seed_v1_params(con)
    un_mod.seed_v1_params(con)

    run_id = so.run(
        con, date(2026, 8, 10), "2026-2027", 2, ep_mv, un_mv,
        lambda_params_version=1, guardrail_params_version=1,
        ownership_params_version=1, risk_posture_params_version=1,
        field_covariance_params_version=1, bench_quality_params_version=1,
        concentration_risk_params_version=1,
    )
    assert isinstance(run_id, int)

    row = con.execute(
        "SELECT ownership_params_version, risk_posture_params_version, field_covariance_params_version, "
        "bench_quality_params_version, concentration_risk_params_version, mip_gap, solver_status "
        "FROM squad_optimizer_runs WHERE run_id = ?", [run_id],
    ).fetchone()
    assert row == (1, 1, 1, 1, 1, pytest.approx(0.0), "optimal")

    audit = so.explain_run(con, run_id)
    assert audit["solve_proved_optimal"] is True
    assert audit["mip_gap"] == pytest.approx(0.0)


def test_run_rejects_ownership_without_risk_posture(con):
    ep_mv, un_mv = _seed_run_candidate_pool(con)
    so.seed_v1_params(con)
    with pytest.raises(ValueError):
        so.run(
            con, date(2026, 8, 10), "2026-2027", 2, ep_mv, un_mv,
            lambda_params_version=1, guardrail_params_version=1,
            ownership_params_version=1, risk_posture_params_version=None,
        )


def test_run_rejects_field_covariance_without_ownership(con):
    ep_mv, un_mv = _seed_run_candidate_pool(con)
    so.seed_v1_params(con)
    with pytest.raises(ValueError):
        so.run(
            con, date(2026, 8, 10), "2026-2027", 2, ep_mv, un_mv,
            lambda_params_version=1, guardrail_params_version=1,
            field_covariance_params_version=1,
        )


def test_run_without_any_priority1_2_features_is_unchanged(con):
    """The original eight-positional-argument call shape must still work exactly as before --
    backward compatibility for every existing caller (scripts, other tests)."""
    ep_mv, un_mv = _seed_run_candidate_pool(con)
    so.seed_v1_params(con)
    run_id = so.run(con, date(2026, 8, 10), "2026-2027", 2, ep_mv, un_mv, 1, 1)
    assert isinstance(run_id, int)
    row = con.execute(
        "SELECT ownership_params_version, risk_posture_params_version, field_covariance_params_version, "
        "bench_quality_params_version, concentration_risk_params_version "
        "FROM squad_optimizer_runs WHERE run_id = ?", [run_id],
    ).fetchone()
    assert row == (None, None, None, None, None)


# ============================================================
# fetch_horizon_candidate_pool() / fetch_horizon_sigma_pairs() / run(horizon_ep_versions=...)
# -- real bug fix: evaluate_wildcard() used to call run() with just target_gameweek's own
# (ep_mv, un_mv), the exact same single-gameweek shape evaluate_free_hit() correctly uses.
# Wildcard locks a squad in for the WHOLE horizon, not just the gameweek it's played, so
# scoring the rebuild on one week's numbers alone is blind to fixture-swing trades -- a
# genuinely strong all-round player having one weak week (this fixture's own "differential"
# forward, deliberately built the same shape as a real premium striker whose GW4 fixture is
# tougher than the rest of his horizon) gets passed over for players who happen to have a good
# single target-gameweek, exactly the "no full sweep" failure mode reported against the app.
# ============================================================

def _seed_two_gameweek_candidate_pool(con, target_season="2026-2027"):
    """Same shape as _seed_run_candidate_pool() above (2 GK/6 DEF/6 MID/4 FWD across 6 clubs,
    budget/club-cap-feasible), but seeded at TWO gameweeks (2 and 3) sharing one team-strength/
    minutes model (both are gameweek-agnostic snapshots -- see fetch_candidate_pool's own
    docstring) with their own ep_model_versions/uncertainty_model_versions/ep_outputs/
    uncertainty_outputs rows per gameweek. "fwd3" is the deliberate differential: a genuinely
    strong forward whose gw2 mu alone is the WORST of the five forwards (so a single-gameweek
    solve at gw2 never picks him) but whose gw2+gw3 SUM is the best (so a horizon-aware solve
    should). Returns {2: (ep_mv, un_mv), 3: (ep_mv, un_mv)}."""
    clubs = ["clubA", "clubB", "clubC", "clubD", "clubE", "clubF"]
    # Forwards get their own dedicated clubs (not reused by any GK/DEF/MID) so the <=3-per-club
    # guardrail can never coincidentally force a non-differential forward out for club-cap
    # reasons -- a real false-positive this test hit on its first pass, not a solver bug: with
    # forwards sharing clubs A-F, a tied-mu forward could get excluded purely because its club
    # was already at its cap from an unrelated DEF/MID pick, which would make "fwd3 excluded at
    # single-gw" true for the wrong reason (forced by club cap, not by its worse mu).
    forward_clubs = ["clubG", "clubH", "clubI", "clubJ", "clubK"]
    for club in clubs + forward_clubs:
        con.execute("INSERT INTO dim_team (team_uid, canonical_name) VALUES (?, ?)", [club, club])
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

    players = []
    for i in range(2):
        players.append((f"gk{i}", "Goalkeeper", 4.5 + i, clubs[i % 6]))
    for i in range(6):
        players.append((f"def{i}", "Defender", 4.0 + i * 0.5, clubs[i % 6]))
    for i in range(6):
        players.append((f"mid{i}", "Midfielder", 5.0 + i * 0.5, clubs[i % 6]))
    for i in range(5):
        players.append((f"fwd{i}", "Forward", 6.0 + i * 0.5, forward_clubs[i]))

    # gw2, gw3 mu per player -- every non-differential player scores a flat 3.0/gw (so their
    # gw2 and gw2+gw3-sum rankings agree); "fwd3" is deliberately last on gw2 alone (0.5, below
    # every other forward's 3.0) but way out ahead on the gw2+gw3 sum (0.5 + 20.0 = 20.5).
    mu_by_gw = {2: {}, 3: {}}
    for uid, _pos, _price, _club in players:
        mu_by_gw[2][uid] = 3.0
        mu_by_gw[3][uid] = 3.0
    mu_by_gw[2]["fwd3"] = 0.5
    mu_by_gw[3]["fwd3"] = 20.0

    for uid, position, price, club in players:
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
            "INSERT INTO minutes_model_outputs (model_version, player_uid, position, p_start_historical_position_avg, "
            "weight_own, p_start_historical_final, logit_adjustment_total, p_start_final, "
            "p_used_as_sub_given_not_started, p_0min, p_1_59min, p_60plus_min, competitive_matches_last_2_seasons) "
            "VALUES (?, ?, ?, 0.7, 1.0, 0.7, 0.0, 0.7, 0.0, 0.15, 0.05, 0.8, 20)",
            [mm_mv, uid, position],
        )

    versions: dict[int, tuple[int, int]] = {}
    for gw, match_id in ((2, "m2"), (3, "m3")):
        con.execute(
            "INSERT INTO fact_match (match_id, season, gameweek, home_team_uid, away_team_uid, finished, "
            "competition, kickoff_time, _ingested_at) VALUES (?, ?, ?, 'clubA', 'clubB', FALSE, "
            "'Premier League', '2026-08-24', current_timestamp)", [match_id, target_season, gw],
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
        for uid, _position, _price, _club in players:
            mu = mu_by_gw[gw][uid]
            con.execute(
                "INSERT INTO ep_outputs (model_version, player_uid, fixture_match_id, ep_appearance, ep_goals, "
                "ep_assists, ep_clean_sheet, ep_goals_conceded, ep_defcon, ep_bonus, ep_saves, ep_penalty_save, "
                "ep_cards, ep_own_goal, ep_total, expected_bps) VALUES (?, ?, ?, 0,0,0,0.02,0,0,0,0,0,0,0, ?, 5.0)",
                [ep_mv, uid, match_id, mu],
            )
            var = 1.0 + mu * 0.5
            con.execute(
                "INSERT INTO uncertainty_outputs (model_version, player_uid, fixture_match_id, var_appearance, "
                "var_goals, var_assists, var_clean_sheet, var_goals_conceded, var_defcon, var_bonus, var_saves, "
                "var_total, skew, excess_kurtosis, quantile_05, quantile_25, quantile_75, quantile_95) "
                "VALUES (?, ?, ?, 0,0,0,0,0,0,0,0, ?, 0,0,0,0,0,0)", [un_mv, uid, match_id, var],
            )
        # real, differentiated positive covariance among a small, fixed clique of non-forward
        # anchor players -- same recipe/scale _seed_run_candidate_pool() above uses (a handful
        # of correlated players, not every candidate), needed so run()'s lambda=0-vs-lambda
        # divergence check genuinely passes. Deliberately every player here has a flat 3.0 mu
        # (see mu_by_gw above), so unlike _seed_run_candidate_pool()'s own mu>=4.0 threshold, a
        # mu-based threshold here would sweep in nearly the WHOLE pool (a real bug this test
        # hit on its first pass: an 18-player fully-connected same-covariance clique is not
        # PSD, which corrupts the risk term instead of just exercising it) -- a small fixed
        # anchor set avoids that, and staying off the forwards entirely keeps the risk term
        # from ever confounding the fwd3 in/out assertions this fixture exists to isolate.
        anchor_uids = ["gk0", "gk1", "def0", "def1", "mid0"]
        for i in range(len(anchor_uids)):
            for j in range(i + 1, len(anchor_uids)):
                con.execute(
                    "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, "
                    "fixture_match_id, relationship, covariance) VALUES (?, ?, ?, ?, 'teammate', 0.6)",
                    [un_mv, *sorted([anchor_uids[i], anchor_uids[j]]), match_id],
                )
        versions[gw] = (ep_mv, un_mv)
    return versions


def test_fetch_horizon_candidate_pool_sums_mu_and_var_across_gameweeks(con):
    versions = _seed_two_gameweek_candidate_pool(con)
    pool = so.fetch_horizon_candidate_pool(con, versions, "2026-2027")
    by_uid = {c["player_uid"]: c for c in pool}

    # flat 3.0/gw non-differential player -> summed mu 6.0, summed var (1+3*0.5)*2 = 5.0
    assert by_uid["mid0"]["mu"] == pytest.approx(6.0)
    assert by_uid["mid0"]["var"] == pytest.approx(5.0)

    # differential forward: 0.5 (gw2) + 20.0 (gw3) = 20.5, worst-at-gw2 but best overall
    assert by_uid["fwd3"]["mu"] == pytest.approx(20.5)
    assert by_uid["fwd3"]["var"] == pytest.approx((1 + 0.5 * 0.5) + (1 + 20.0 * 0.5))

    # price/position/club are gameweek-agnostic -- taken from the base gameweek, not re-summed
    assert by_uid["fwd3"]["price"] == pytest.approx(7.5)
    assert by_uid["fwd3"]["position"] == "Forward"


def test_fetch_horizon_candidate_pool_rejects_empty_horizon(con):
    with pytest.raises(ValueError):
        so.fetch_horizon_candidate_pool(con, {}, "2026-2027")


def test_fetch_horizon_sigma_pairs_sums_covariance_across_gameweeks(con):
    """fwd0/fwd1 carry no auto-seeded covariance from _seed_two_gameweek_candidate_pool()'s own
    anchor clique (forwards are deliberately excluded from it -- see that fixture's own
    docstring), so this pair is a clean slate to insert against and assert an exact sum on."""
    versions = _seed_two_gameweek_candidate_pool(con)
    for gw, match_id, cov in ((2, "m2", 1.5), (3, "m3", 2.5)):
        _ep_mv, un_mv = versions[gw]
        con.execute(
            "INSERT INTO cross_player_covariance (model_version, player_uid_a, player_uid_b, "
            "fixture_match_id, relationship, covariance) VALUES (?, 'fwd0', 'fwd1', ?, 'teammate', ?)",
            [un_mv, match_id, cov],
        )
    pairs = so.fetch_horizon_sigma_pairs(con, versions, {"fwd0", "fwd1"})
    assert pairs[("fwd0", "fwd1")] == pytest.approx(4.0)


def test_horizon_aware_solve_picks_a_player_a_single_gameweek_solve_would_miss(con):
    """The direct demonstration of the bug this fixes: solving on gw2 alone never picks
    "fwd3" (its gw2 mu, 0.5, is the worst of the four forwards) -- solving on the gw2+gw3
    horizon does (its summed mu, 20.5, is far and away the best), exactly the failure mode
    a real Wildcard rebuild hits when it only ever looked at the target gameweek."""
    versions = _seed_two_gameweek_candidate_pool(con)
    ep_mv2, un_mv2 = versions[2]

    single_gw_pool = so.fetch_candidate_pool(con, ep_mv2, un_mv2, "2026-2027")
    single_gw_result = so.solve(single_gw_pool, {}, lam=0.0, guardrail_cap=3)
    assert single_gw_result["status"] == "optimal"
    assert "fwd3" not in single_gw_result["squad"]

    horizon_pool = so.fetch_horizon_candidate_pool(con, versions, "2026-2027")
    horizon_result = so.solve(horizon_pool, {}, lam=0.0, guardrail_cap=3)
    assert horizon_result["status"] == "optimal"
    assert "fwd3" in horizon_result["squad"]


def test_run_with_horizon_ep_versions_picks_the_horizon_winner(con):
    """End-to-end run() wiring: passing horizon_ep_versions=... (evaluate_wildcard()'s own new
    call shape) makes the persisted squad_optimizer_selections reflect the horizon-summed
    winner, not the single-target-gameweek one -- the same real DB round-trip
    test_run_with_priority1_features_enabled_end_to_end already covers for the Priority 1/2
    kwargs, applied to this one."""
    versions = _seed_two_gameweek_candidate_pool(con)
    so.seed_v1_params(con)
    ep_mv2, un_mv2 = versions[2]

    run_id_single = so.run(con, date(2026, 8, 10), "2026-2027", 2, ep_mv2, un_mv2, 1, 1)
    single_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [run_id_single],
        ).fetchall()
    }
    assert "fwd3" not in single_uids

    run_id_horizon = so.run(
        con, date(2026, 8, 10), "2026-2027", 2, ep_mv2, un_mv2, 1, 1, horizon_ep_versions=versions,
    )
    horizon_uids = {
        r[0] for r in con.execute(
            "SELECT player_uid FROM squad_optimizer_selections WHERE run_id = ? AND in_squad", [run_id_horizon],
        ).fetchall()
    }
    assert "fwd3" in horizon_uids
