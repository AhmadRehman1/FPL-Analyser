"""chip_timing_analysis -- the per-team Wildcard/Bench-Boost/Free-Hit timing orchestration.

The heavy forward walk lives in forward_season_sim and is tested there. These tests cover the
NEW logic this module adds: the sweep-and-compare over arm results, the robustness-check
perturbation loop, and the guards for an already-spent chip / missing real squad data.
"""

from __future__ import annotations

import pytest

from fpl_quant import chip_timing_analysis as cta


# ------------------------------------------------------------------
# helpers: hand-built forward-sim arm dicts (ForwardSimResult.to_dict() shape)
# ------------------------------------------------------------------

def _gw_row(gw, *, pts=50.0, wc_gain=None, wc_reco=False, action="hold", fh_gain=None, fh_reco=False, chips=()):
    return {
        "gameweek": gw, "projected_points": pts, "band_low": pts - 10, "band_high": pts + 10,
        "action": action, "action_detail": "",
        "wildcard_gain": wc_gain, "wildcard_recommended": wc_reco,
        "current_squad_horizon_value": None,
        "chips_used": list(chips),
        "free_hit_gain": fh_gain, "free_hit_recommended": fh_reco,
    }


def _arm(mode, *, total, rows, band=None):
    lo, hi = band or (total - 50, total + 50)
    return {
        "entry_label": "T", "season": "2026-2027", "start_gameweek": 3, "end_gameweek": 20,
        "mode": mode, "total_projected_points": total, "total_band": [lo, hi],
        "wildcard_recommendation": None, "gameweeks": rows,
    }


def _hold_arm(rows=None):
    rows = rows or [_gw_row(gw, wc_gain=float(gw), wc_reco=gw >= 10) for gw in range(3, 21)]
    d = _arm("hold_wildcard", total=900.0, rows=rows)
    d["wildcard_recommendation"] = {"gameweek": 14, "projected_gain": 14.0}
    return cta.WildcardArm.from_forward_sim_dict(d)


def _forced_arm(gw, total):
    rows = [_gw_row(g, action="wildcard" if g == gw else "hold") for g in range(3, 21)]
    return cta.WildcardArm.from_forward_sim_dict(_arm(f"force_wildcard_gw{gw}", total=total, rows=rows))


def _model_choice_arm(played_at, total):
    rows = [_gw_row(g, action="wildcard" if g == played_at else "hold") for g in range(3, 21)]
    return cta.WildcardArm.from_forward_sim_dict(_arm("model_choice", total=total, rows=rows))


# ------------------------------------------------------------------
# Step 1 -- sweep-and-compare
# ------------------------------------------------------------------

def test_arm_from_dict_parses_mode_and_forced_gameweek():
    a = _forced_arm(11, 950.0)
    assert a.forced_gameweek == 11
    assert a.wildcard_played_at == 11
    assert a.mode == "force_wildcard_gw11"

    h = _hold_arm()
    assert h.forced_gameweek is None
    assert h.wildcard_played_at is None


def test_compare_picks_the_highest_total_forced_arm_as_swept_best():
    hold = _hold_arm()
    forced = [_forced_arm(9, 905.0), _forced_arm(12, 980.0), _forced_arm(15, 940.0), _forced_arm(19, 890.0)]
    model = _model_choice_arm(9, 910.0)
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=model, forced_arms=forced,
    )
    assert cmp.swept_best_gameweek == 12
    assert cmp.swept_best_points == 980.0
    assert cmp.greedy_gameweek == 9
    assert cmp.greedy_swept_agree is False
    # the swept table carries the delta vs the hold baseline for every forced week
    assert {r["gameweek"] for r in cmp.swept_table} == {9, 12, 15, 19}
    assert next(r["delta_vs_hold"] for r in cmp.swept_table if r["gameweek"] == 12) == pytest.approx(80.0)


def test_compare_says_hold_when_no_forced_week_beats_the_hold_baseline():
    hold = _hold_arm()  # total 900
    forced = [_forced_arm(9, 880.0), _forced_arm(12, 895.0), _forced_arm(15, 870.0)]
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=_model_choice_arm(0, 900.0), forced_arms=forced,
    )
    assert cmp.swept_best_gameweek is None
    assert "hold the chip" in cmp.disagreement_note


def test_compare_flags_greedy_vs_swept_agreement():
    hold = _hold_arm()
    forced = [_forced_arm(9, 905.0), _forced_arm(12, 980.0)]
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=_model_choice_arm(12, 975.0), forced_arms=forced,
    )
    assert cmp.greedy_swept_agree is True
    assert "agree on GW12" in cmp.disagreement_note


def test_compare_handles_missing_model_choice_arm():
    hold = _hold_arm()
    forced = [_forced_arm(10, 950.0)]
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=None, forced_arms=forced,
    )
    assert cmp.greedy_gameweek is None
    assert cmp.greedy_swept_agree is None
    assert cmp.swept_best_gameweek == 10


def test_compare_tolerates_a_partial_sweep():
    # a matrix job dropped out -- only some forced gameweeks present
    hold = _hold_arm()
    forced = [_forced_arm(9, 905.0), _forced_arm(17, 970.0)]
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=_model_choice_arm(9, 910.0), forced_arms=forced,
    )
    assert cmp.sweep_gameweeks == [9, 17]
    assert cmp.swept_best_gameweek == 17


def test_comparison_to_dict_is_json_serialisable():
    import json
    hold = _hold_arm()
    cmp = cta.compare_wildcard_timing(
        entry_label="T", start_gameweek=3, eval_end_gameweek=20,
        hold_arm=hold, model_choice_arm=_model_choice_arm(11, 960.0), forced_arms=[_forced_arm(11, 960.0)],
    )
    json.dumps(cmp.to_dict())


# ------------------------------------------------------------------
# Step 3 -- Free Hit scan (read off the hold arm)
# ------------------------------------------------------------------

def test_free_hit_scan_flags_only_gameweeks_that_clear_the_threshold():
    rows = [_gw_row(3, fh_gain=0.4), _gw_row(4, fh_gain=2.1, fh_reco=True), _gw_row(5, fh_gain=None)]
    hold = cta.WildcardArm.from_forward_sim_dict(_arm("hold_wildcard", total=150.0, rows=rows))
    scan = cta.free_hit_scan_from_hold_arm(hold, threshold_min_horizon_gain=1.5)
    assert [r["clears_threshold"] for r in scan] == [False, True, False]
    assert scan[1]["recommended"] is True


# ------------------------------------------------------------------
# Step 4 -- robustness perturbation loop
# ------------------------------------------------------------------

def test_plan_perturbations_is_one_axis_at_a_time_and_dedupes():
    ps = cta.plan_perturbations(base_lambda_value=0.15, base_rho_residual_params_version=2)
    labels = [p.label for p in ps]
    assert labels[0] == "base"
    # the base lambda (0.15) and base rho version (2) don't get re-emitted as their own arms
    assert sum(1 for p in ps if p.lambda_value == 0.15 and p.rho_residual_params_version == 2 and p.ep_jitter_sigmas == 0.0) == 1
    # every perturbation moves exactly one axis off the base
    for p in ps[1:]:
        moved = [
            p.lambda_value != 0.15,
            p.rho_residual_params_version != 2,
            p.ep_jitter_sigmas != 0.0,
        ]
        assert sum(moved) == 1


def test_classify_core_fragile_splits_on_presence_in_every_solve():
    squads = {
        "base": ["a", "b", "c", "d"],
        "lambda=0.3": ["a", "b", "c", "e"],
        "rho_v1": ["a", "b", "c", "d"],
    }
    out = cta.classify_core_fragile(squads)
    assert out["core_players"] == ["a", "b", "c"]
    assert out["fragile_players"] == ["d", "e"]
    assert out["n_solves"] == 3
    assert out["appearances"]["d"] == 2 and out["appearances"]["e"] == 1


def test_classify_core_fragile_rejects_empty_input():
    with pytest.raises(ValueError):
        cta.classify_core_fragile({})


def test_classify_core_fragile_verdict_is_fragile_above_two_swing_players():
    squads = {
        "s1": ["a", "b", "c", "d", "e"],
        "s2": ["a", "f", "g", "h", "i"],
    }
    out = cta.classify_core_fragile(squads)
    assert out["verdict"] == "fragile"


def test_robustness_check_runs_one_solve_per_perturbation(monkeypatch):
    """robustness_check must call solve() once per perturbation and hand the results to
    classify_core_fragile -- verified without a real MIQP solve."""
    calls = []

    def fake_fetch_pool(con, ep_mv, un_mv, season):
        return [{"player_uid": f"p{i}", "position": "Midfielder", "mu": 5.0, "var": 4.0,
                 "club": "X", "price": 5.0} for i in range(20)]

    def fake_sigma(con, un_mv, uids):
        return {}

    def fake_solve(candidates, sigma_pairs, lam, guardrail_cap):
        calls.append(lam)
        # lambda 0.3 swaps one player, everything else identical
        base = [c["player_uid"] for c in candidates[:15]]
        if lam == 0.30:
            base = base[:-1] + ["p19"]
        return {"squad": base, "xi": base[:11]}

    monkeypatch.setattr(cta.squad_optimizer, "fetch_candidate_pool", fake_fetch_pool)
    monkeypatch.setattr(cta.squad_optimizer, "fetch_sigma_pairs", fake_sigma)
    monkeypatch.setattr(cta.squad_optimizer, "solve", fake_solve)
    monkeypatch.setattr(cta.params_mod, "resolve_param", lambda *a, **k: (3.0, None))

    perts = [
        cta.Perturbation("base", 0.15, 2, 0.0),
        cta.Perturbation("lambda=0.3", 0.30, 2, 0.0),
    ]
    rep = cta.robustness_check(
        con=None, entry_label="T", calibration_asof_date=None, target_season="2026-2027",
        target_gameweek=12, ep_model_version=1, uncertainty_model_version=1,
        ts_model_version=1, mm_model_version=1, scoring_params_version=1, bps_params_version=1,
        tau_params_version=1, corr_params_version=1, guardrail_params_version=1,
        base_lambda_value=0.15, base_rho_residual_params_version=2, perturbations=perts,
    )
    assert calls == [0.15, 0.30]
    assert rep.summary["fragile_players"] == ["p14", "p19"]
    assert rep.summary["core_count"] == 14


# ------------------------------------------------------------------
# Step 7 guards -- already-spent chip / missing squad data
# ------------------------------------------------------------------

def test_assert_wildcard_available_raises_when_set1_wildcard_already_used():
    with pytest.raises(cta.ChipAlreadyUsedError):
        cta.assert_wildcard_available(chips_used_set1=["wildcard"], chips_used_set2=[], sweep_gameweeks=[9, 12, 15])


def test_assert_wildcard_available_allows_a_set2_only_sweep_after_a_set1_wildcard():
    # set-1 wildcard spent, but the sweep is entirely in set 2 (GW19+) -- a different chip
    cta.assert_wildcard_available(chips_used_set1=["wildcard"], chips_used_set2=[], sweep_gameweeks=[20, 24])


def test_assert_wildcard_available_passes_for_a_clean_chip_state():
    cta.assert_wildcard_available(chips_used_set1=[], chips_used_set2=[], sweep_gameweeks=[9, 12, 15, 19])


def test_build_bootstrap_squad_fails_loudly_on_missing_picks():
    with pytest.raises(cta.MissingSquadDataError):
        cta.build_bootstrap_squad(entry_id=999, picks=None, element_names={})
    with pytest.raises(cta.MissingSquadDataError):
        cta.build_bootstrap_squad(entry_id=999, picks=[], element_names={})


def test_build_bootstrap_squad_maps_picks_to_the_forward_sim_shape():
    picks = [{"element": 1, "position": p, "is_captain": p == 1, "is_vice_captain": p == 2}
             for p in range(1, 16)]
    names = {i: f"Player {i}" for i in range(1, 16)}
    squad = cta.build_bootstrap_squad(entry_id=1, picks=picks, element_names=names)
    assert len(squad) == 15
    assert squad[0] == {"player_name": "Player 1", "in_xi": True, "is_captain": True, "is_vice": False}
    assert squad[11]["in_xi"] is False  # position 12 -> bench


# ------------------------------------------------------------------
# Step 0.2 -- evidence freshness for the held squad
# ------------------------------------------------------------------

def _seed_claim(con, uid, claim_type, observed_date):
    con.execute(
        "INSERT INTO evidence_claims (claim_id, subject_entity_type, subject_entity_id, claim_type, "
        "source_id, source_reliability_score, confidence, observed_date, ingested_date) "
        "VALUES (?, 'player', ?, ?, 'test-source', 0.9, 0.9, ?, ?)",
        [f"{uid}-{claim_type}", uid, claim_type, observed_date, observed_date],
    )


def test_evidence_freshness_flags_marks_stale_and_missing(con):
    from datetime import date
    con.execute("INSERT INTO sources (source_id, source_name, source_type, base_reliability_score) "
                "VALUES ('test-source', 'Test', 'official', 0.9)")
    for uid in ("p_fresh", "p_stale", "p_none"):
        con.execute("INSERT INTO dim_player (player_uid, canonical_name, position) VALUES (?, ?, 'Midfielder')",
                    [uid, uid])
    _seed_claim(con, "p_fresh", "injury_status", date(2026, 8, 28))
    _seed_claim(con, "p_stale", "injury_status", date(2026, 7, 20))
    _seed_claim(con, "p_none", "preseason_involvement", date(2026, 8, 29))  # not an availability type

    flags = cta.evidence_freshness_flags(
        con, held_player_uids=["p_fresh", "p_stale", "p_none"], as_of_date=date(2026, 8, 31), stale_days=14,
    )
    by_uid = {f["player_uid"]: f for f in flags}
    assert "p_fresh" not in by_uid
    assert by_uid["p_stale"]["status"] == "stale" and by_uid["p_stale"]["age_days"] == 42
    assert by_uid["p_none"]["status"] == "no_availability_claims"


# ------------------------------------------------------------------
# integration: forward_season_sim capture + wildcard_followups over the synthetic league
# ------------------------------------------------------------------

def test_wildcard_context_and_followups_over_the_synthetic_league(con):
    from fpl_quant import forward_season_sim as fss
    from tests.test_backtest import _seed_season_simulation_league
    from tests.test_forward_season_sim import _bootstrap_squad

    _seed_season_simulation_league(con)
    result = fss.run_forward_season_sim(
        con, entry_label="itest", target_season="2025-2026",
        start_gameweek=2, end_gameweek=4, bootstrap_squad=_bootstrap_squad(con),
        active_versions={}, force_wildcard_at=3,
    )
    assert result.wildcard_context is not None
    ctx = result.wildcard_context
    assert ctx["gameweek"] == 3
    assert ctx["fresh_run_id"] is not None
    assert len(ctx["holdings_before_uids"]) == 15

    followups = cta.wildcard_followups(
        con, entry_label="itest", target_season="2025-2026", wildcard_context=ctx,
        robustness_perturbations=[cta.Perturbation("base", 0.15, 1, 0.0), cta.Perturbation("lambda=0.3", 0.30, 1, 0.0)],
    )
    assert followups["wildcard_gameweek"] == 3
    assert followups["bench_boost_window"][0]["gameweek"] == 3
    assert followups["robustness"]["summary"]["n_solves"] == 2
